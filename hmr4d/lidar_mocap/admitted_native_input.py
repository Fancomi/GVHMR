"""Prepare a review-only full-native run from an independently accepted camera.

The original accepted freeze remains the parent authority. This adapter narrows
only purpose metadata; it does not reconstruct geometry, clocks or identities.
All native-rally, source, window and body-plan builders remain the original ones.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import time

from hmr4d import PROJ_ROOT
from . import calibration as calibration_module
from .calibration import CalibrationSnapshot, _bounded, _canonical, _checked_blob, _sha
from .new_source_repair import prepare_new_sources, write_frozen
from .pilot import (
    binding, check_binding, require_review_group, validate_camera_review,
    validate_full_native_spec, validate_native_rally,
)
from .relocations import load_manifest_relocations
from .source_index import SourceIndexedSession, _bound_file


_PURPOSE_FIELDS = frozenset({
    "review_only", "review_source_ids", "declared_calibration_source_ids",
    "production_admitted", "training_published",
})
_LINEAGE_FIELD = "admitted_review_derivation"
_DERIVED_FIELDS = _PURPOSE_FIELDS | {_LINEAGE_FIELD, "snapshot_digest"}
_PROOF_SCHEMA = "badminton-1022-strict-accepted-freeze-proof-v2"


def _read(path):
    return json.loads(Path(path).read_bytes())


def _same(left, right):
    """Original canonical encoding preserves bool/int and int/float differences."""
    return _canonical(left) == _canonical(right)


def _bound_json(entry):
    check_binding(entry)
    payload = Path(entry["path"]).read_bytes()
    if "bytes" in entry and len(payload) != entry["bytes"]:
        raise ValueError("Bound metadata byte count changed")
    return json.loads(payload)


def _accepted_context(parent_snapshot_json, manifest_path, accepted_freeze_proof):
    parent_path, manifest_path = Path(parent_snapshot_json).resolve(), Path(manifest_path).resolve()
    parent = CalibrationSnapshot.load(parent_path)
    spec = _read(manifest_path)
    scope = validate_full_native_spec(spec)
    validate_camera_review(spec)
    proof = _bound_json(accepted_freeze_proof)
    expected_parent, expected_manifest = binding(parent_path), binding(manifest_path)
    for key, expected in (("accepted_snapshot", expected_parent), ("manifest", expected_manifest)):
        actual = proof.get(key, {})
        if (not isinstance(actual.get("path"), str)
                or Path(actual["path"]).resolve() != Path(expected["path"])
                or actual.get("sha256") != expected["sha256"]):
            raise ValueError("Strict accepted freeze proof binds another " + key)
    if (proof.get("schema") != _PROOF_SCHEMA
            or proof.get("status") != "strict_accepted_freeze_completed_review_prepare_not_invoked"
            or type(proof.get("actual_exit_code")) is not int or proof["actual_exit_code"] != 0
            or any(proof.get(key) is not False for key in
                   ("model_loaded", "gpu_started", "body_labels_production_admitted", "training_published"))):
        raise ValueError("Require the completed strict accepted CPU freeze proof without body/training admission")
    provenance = parent.data.get("provenance", {})
    if (parent.data.get("review_only", False) is not False
            or any(key in parent.data for key in
                   ("review_source_ids", "declared_calibration_source_ids", _LINEAGE_FIELD))
            or parent.data.get("production_admitted") is not False
            or parent.data.get("training_published", False) is not False
            or provenance.get("camera_calibration_admitted") is not True
            or provenance.get("body_labels_production_admitted") is not False
            or provenance.get("production_admitted") is not False
            or set(scope["sources"]) != set(parent.data["sessions"])
            or not _same(parent.data["configuration_files_sha256"], spec["bundle_sha256"])
            or not _same(proof.get("accepted_bundle_sha256"), spec["bundle_sha256"])):
        raise ValueError("Parent must be this exact accepted camera/source bundle, with body/training pending")
    required_counts = dict(source_rows=sum(row["frame_count"] for row in parent.data["sessions"].values()),
                           native_rallies=len(spec["clips"]), selected_scans=scope["selected_scans"],
                           selected_seconds=scope["selected_seconds"])
    for key, expected in required_counts.items():
        if not _same(proof.get(key), expected):
            raise ValueError("Strict accepted freeze proof changed complete native scope: " + key)
    declared = provenance.get("valid_source_ids")
    if (not isinstance(declared, list) or not declared or len(set(declared)) != len(declared)
            or not set(scope["sources"]) <= set(declared)):
        raise ValueError("Accepted parent has no matching declared camera source scope")
    lineage = dict(
        schema="badminton-admitted-review-calibration-derivation-v1",
        parent_accepted_snapshot=expected_parent,
        full_native_manifest=expected_manifest,
        review_authorization=copy.deepcopy(spec["review_authorization"]),
        accepted_freeze_proof=copy.deepcopy(accepted_freeze_proof),
    )
    return parent, spec, scope, lineage


def _expected_review_data(parent, scope, lineage):
    data = copy.deepcopy(parent.data)
    data.pop("snapshot_digest", None)
    data.update(review_only=True, review_source_ids=scope["sources"],
                declared_calibration_source_ids=copy.deepcopy(parent.data["provenance"]["valid_source_ids"]),
                production_admitted=False, training_published=False)
    data[_LINEAGE_FIELD] = lineage
    data["snapshot_digest"] = _sha(_canonical(data))
    return data


def verify_review_snapshot(parent_snapshot_json, child_snapshot_json, manifest_path, *, accepted_freeze_proof):
    """Recheck full typed metadata, immutable authorities and actual artifact bytes."""
    parent, spec, scope, lineage = _accepted_context(parent_snapshot_json, manifest_path, accepted_freeze_proof)
    child = CalibrationSnapshot.load(child_snapshot_json)
    expected = _expected_review_data(parent, scope, lineage)
    if not _same(child.data, expected):
        raise ValueError("Derived review snapshot differs from its exact permitted purpose metadata")
    before = {key: value for key, value in parent.data.items() if key not in _DERIVED_FIELDS}
    after = {key: value for key, value in child.data.items() if key not in _DERIVED_FIELDS}
    if not _same(before, after):
        raise ValueError("Derived review snapshot changed protected geometry, session or clock metadata")
    for artifact in parent.data["artifacts"].values():
        original = _checked_blob(_bounded(parent.snapshot_json.parent, artifact["path"]), artifact["sha256"])
        derived = _checked_blob(_bounded(child.snapshot_json.parent, artifact["path"]), artifact["sha256"])
        if original != derived:
            raise ValueError("Derived review artifact bytes changed")
    return dict(
        schema="badminton-admitted-review-snapshot-proof-v1", status="review_snapshot_verified_metadata_only",
        parent_accepted_snapshot=binding(parent.snapshot_json), derived_review_snapshot=binding(child.snapshot_json),
        manifest=binding(manifest_path), accepted_freeze_proof=copy.deepcopy(accepted_freeze_proof),
        protected_typed_metadata_identical=True, artifact_bytes_identical=True,
        changed_fields=sorted(_PURPOSE_FIELDS | {_LINEAGE_FIELD, "snapshot_digest"}),
        sources=scope["sources"], native_rallies=len(spec["clips"]), selected_scans=scope["selected_scans"],
        selected_seconds=scope["selected_seconds"], camera_calibration_admitted=True,
        body_labels_production_admitted=False, training_published=False, model_loaded=False, gpu_started=False,
    )


def derive_review_snapshot(parent_snapshot_json, manifest_path, output_dir, *, accepted_freeze_proof):
    """Create one new metadata-only review snapshot; never rewrite its accepted parent."""
    parent, _, scope, lineage = _accepted_context(parent_snapshot_json, manifest_path, accepted_freeze_proof)
    destination = Path(output_dir).resolve()
    if (not destination.is_relative_to(calibration_module._WORKSPACE_ROOT.resolve())
            or destination.exists() or destination.is_relative_to(parent.snapshot_json.parent)
            or parent.snapshot_json.parent.is_relative_to(destination)):
        raise ValueError("Use a new independent review calibration directory inside GVHMR")
    data = _expected_review_data(parent, scope, lineage)
    # Original geometry validator runs before any output directory is created.
    CalibrationSnapshot(data, destination / "calibration_snapshot.json")
    artifacts = {}
    for artifact in parent.data["artifacts"].values():
        artifacts[artifact["path"]] = _checked_blob(
            _bounded(parent.snapshot_json.parent, artifact["path"]), artifact["sha256"])
    for name, payload in artifacts.items():
        path = _bounded(destination, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)
    snapshot_path = destination / "calibration_snapshot.json"
    with snapshot_path.open("xb") as handle:
        handle.write((json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    verify_review_snapshot(parent.snapshot_json, snapshot_path, manifest_path,
                           accepted_freeze_proof=accepted_freeze_proof)
    return CalibrationSnapshot.load(snapshot_path)


def inspect_admitted_native(manifest_path, parent_snapshot_json, *, accepted_freeze_proof):
    """Read only: validate the accepted parent and original complete native scope."""
    parent, spec, scope, _ = _accepted_context(parent_snapshot_json, manifest_path, accepted_freeze_proof)
    return dict(status="ready_for_separate_cpu_review_prepare", manifest=binding(manifest_path),
                parent_accepted_snapshot=binding(parent.snapshot_json), selection=scope,
                windows=len(spec["clips"]), camera_calibration_admitted=True,
                body_labels_production_admitted=False, training_published=False, gpu_started=False)


def prepare_admitted_full_native(manifest_path, output_root, parent_snapshot_json, *, accepted_freeze_proof, threads=1):
    """Thin pilot metadata assembly followed by the unchanged source/body builders.

    No teacher, pose, fitting or GPU stage can be selected through this API.
    The accepted-base freeze is an explicit previously completed input, avoiding
    another full-media freeze while retaining its immutable actual proof.
    """
    if type(threads) is not int or threads < 1:
        raise ValueError("CPU preparation requires a positive thread count")
    started = time.perf_counter()
    root = Path(output_root).resolve()
    if not root.is_relative_to(PROJ_ROOT / "outputs") or root == PROJ_ROOT / "outputs":
        raise ValueError("Use a separate explicit GVHMR outputs root")
    parent, spec, scope, _ = _accepted_context(parent_snapshot_json, manifest_path, accepted_freeze_proof)
    if parent.snapshot_json.parent != root / "accepted-base":
        raise ValueError("Use this run's separately frozen immutable accepted-base")
    if any((root / name).exists() for name in ("source", "repair", "pilot_input.json", "pilot_progress.json")):
        raise ValueError("Preserve existing review preparation; use an independent versioned root")
    original = binding(manifest_path)
    relocations = load_manifest_relocations(spec)
    data_root = Path(spec["data_root"]).resolve()
    for name, expected in spec["bundle_sha256"].items():
        _checked_blob(Path(spec["calibration_dir"]) / name, expected)
    registry = dict(schema="badminton-rally-run-source-registry-v1", review_only=True, training_published=False,
                    pilot_manifest=original, data_root=str(data_root), sources={},
                    execution_scope=spec["execution_scope"], coverage_audit=spec["coverage_audit"])
    if spec.get("excluded_source_indexes"):
        registry.update(excluded_source_indexes=spec["excluded_source_indexes"], no_fit_sources=spec["no_fit_sources"])
    if relocations is not None:
        registry["data_relocations"] = dict(relocations.binding)
    sessions = {}
    for sid, entry in spec["source_indexes"].items():
        path = _bound_file(data_root, entry, relocations=relocations)
        session = SourceIndexedSession.load_source_index(
            path, data_root, expected_sha256=entry["sha256"], require_admitted=True, relocations=relocations)
        index = session.source_index
        if (index.get("source_kind") != "derived_native_roi_fixed24"
                or index.get("crop_binding", {}).get("bbox_sha256") != spec["bundle_sha256"]["bbox.json"]
                or not _same(index.get("accepted_bundle_sha256"), spec["bundle_sha256"])
                or index.get("body_labels_production_admitted") is not False
                or index.get("production_admitted") is not False):
            raise ValueError("Require this exact newly admitted camera crop, with bodies pending")
        sessions[sid] = session
        registry["sources"][sid] = dict(
            baseline=False, source_index=binding(path), source_manifest_sha256=index["source_manifest"]["sha256"],
            common_clock_sha256=index["common_clock"]["sha256"], frame_map_sha256=index["frame_map"]["sha256"])
    rows = []
    for number, clip in enumerate(spec["clips"]):
        sid = clip["source_id"]
        validate_native_rally(clip, registry["sources"][sid]["source_manifest_sha256"])
        pairs = sessions[sid].pairs()
        chosen = []
        for ordinal in clip["eligible_paired_png_indices"]:
            if ordinal >= len(pairs):
                raise ValueError("Native scan ordinal is outside the actual new source")
            pair = pairs[ordinal]
            if pair["status"] != "paired" or not clip["begin_common_us"] <= pair["scan_mid_us"] < clip["end_common_us"]:
                raise ValueError("Native scan lacks both real views or leaves its original rally interval")
            chosen.append(pair)
        row = dict(source_id=sid, source_rally_index=number, review_index=number,
                   original_rally_index=clip["rally_index"], begin_common_us=clip["begin_common_us"],
                   end_common_us=clip["end_common_us"], first_scan_mid_us=chosen[0]["scan_mid_us"],
                   last_scan_mid_us=chosen[-1]["scan_mid_us"], source_begin_frame=chosen[0]["lidar_index"],
                   source_frame_count=len(chosen), eligible_paired_png_indices=[p["lidar_index"] for p in chosen],
                   review_only=True, evidence_files=clip["evidence_files"])
        if "segment_index" in clip:
            row["segment_index"] = clip["segment_index"]
        rows.append(row)
    source = root / "source"
    frozen = derive_review_snapshot(parent.snapshot_json, manifest_path, source / "calibration",
                                    accepted_freeze_proof=accepted_freeze_proof)
    proof = verify_review_snapshot(parent.snapshot_json, frozen.snapshot_json, manifest_path,
                                   accepted_freeze_proof=accepted_freeze_proof)
    ranges = source / "pilot_ranges.jsonl"
    with ranges.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    sources = {sid: dict(rallies=sum(row["source_id"] == sid for row in rows),
                         paired_unique_png_frames=sum(len(row["eligible_paired_png_indices"])
                                                      for row in rows if row["source_id"] == sid))
               for sid in scope["sources"]}
    intake = dict(schema="badminton-review-rally-intake-v1", review_only=True, training_published=False,
                  pilot_manifest=original, prepared_ranges=binding(ranges), sources=sources, rallies=len(rows),
                  effective_window_minutes=scope["selected_seconds"] / 60)
    if relocations is not None:
        intake["data_relocations"] = dict(relocations.binding)
    intake_path = source / "pilot_intake.json"
    write_frozen(intake_path, intake)
    registry["intake"] = binding(intake_path)
    write_frozen(source / "source_registry.json", registry)
    splits = require_review_group(dict(review=scope["sources"]), registry["sources"])
    write_frozen(source / "session_splits.json", splits)
    state = dict(schema=spec["schema"], review_only=True, training_published=False, stage="prepared",
                 camera_label=spec["camera_label"], manifest=original, source_root=str(source),
                 calibration_digest=frozen.data["snapshot_digest"], selection=scope,
                 prepare_seconds=time.perf_counter() - started,
                 admitted_review_proof=binding(accepted_freeze_proof["path"]))
    if relocations is not None:
        state["data_relocations"] = dict(relocations.binding)
    write_frozen(root / "pilot_input.json", dict(state, prepare_seconds=None, stage="frozen_input"))
    write_frozen(root / "admitted_review_snapshot_proof.json", proof)
    plan = prepare_new_sources(source, root / "repair", relocations=relocations)
    if len(plan["jobs"]) != len(rows):
        raise ValueError("Original source builder did not preserve every complete native rally window")
    from .repair_execution import prepare_full_native_body_plan
    body = prepare_full_native_body_plan(root, threads)
    state.update(windows=len(plan["jobs"]), body_prepare_receipt=binding(root / "body_prepare_cpu_receipt.json"),
                 body_plan=body["repair_plan"])
    for sid in scope["sources"]:
        state[sid] = dict(state="prepared_review_only", baseline=False)
    write_frozen(root / "pilot_progress.json", state)
    verify_review_snapshot(parent.snapshot_json, frozen.snapshot_json, manifest_path,
                           accepted_freeze_proof=accepted_freeze_proof)
    check_binding(original)
    return state
