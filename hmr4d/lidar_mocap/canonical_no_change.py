"""Verify an explicitly reviewed identity view without rewriting any source row.

The original full-body continuity audit stays attached to its original body
manifest.  The bilateral publication is proved to be an exact numeric/identity
subset of that body; removing rows cannot introduce an adjacent-scan edge.
This module publishes metadata evidence only, never labels, training or cuts.
"""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re

import joblib
import torch

from hmr4d import PROJ_ROOT
from . import incremental_release
from .dual_view import _verify_export_data, file_sha
from .expanded_dataset import CONTRACT, _validate_new_completion, sample_key, source_roles
from .export import FIELDS
from .rally_pipeline import read_rows
from .repair_execution import write_progress_json as write_json

PUBLICATION = "badminton-canonical-bilateral-publication-v2"
PROOF = "badminton-canonical-no-change-verification-v2"
REVIEW = "badminton-reviewed-canonical-no-change-v2"
MODE = "reviewed_no_changes"
MARKER_FIELDS = {
    "mode",
    "source_publication",
    "reviewed_continuity",
    "verification",
    "numeric_geometry_unchanged",
    "identity_keys_unchanged",
}
REVIEW_FIELDS = {
    "schema",
    "review_status",
    "reviewed_by",
    "reviewed_utc",
    "source_publication",
    "source_completion",
    "continuity_audit",
    "cases",
    "identity_changes",
    "training_authorized",
    "physical_identity_ground_truth",
}
CASE_FIELDS = {"case", "event", "verdict", "evidence", "image"}
PROOF_FIELDS = {
    "schema",
    "state",
    "producer",
    "verifiers",
    "release_policy",
    "source_publication",
    "source_completion",
    "continuity_audit",
    "reviewed_continuity",
    "native_verification",
    "aggregate_native_contract",
    "full_accepted",
    "accepted",
    "sessions",
    "identity_changes",
    "numeric_values_unchanged",
    "admission_membership_unchanged",
    "identity_keys_unchanged",
    "split_only_no_unproven_join",
    "unresolved_jump_candidates",
    "mutual_swap_candidates",
    "training_authorized",
    "physical_identity_ground_truth",
}
SESSION_FIELDS = {
    "full_accepted",
    "accepted",
    "role",
    "source_tensor",
    "source_metadata",
    "body_tensor",
    "body_metadata",
    "numeric_values_bit_exact",
    "identity_keys_unchanged",
    "exact_body_subset",
}
VERIFIERS = {
    "dual_view": "hmr4d/lidar_mocap/dual_view.py",
    "expanded_dataset": "hmr4d/lidar_mocap/expanded_dataset.py",
    "export": "hmr4d/lidar_mocap/export.py",
    "incremental_release": "hmr4d/lidar_mocap/incremental_release.py",
    "continuity_audit": "tools/audit_badminton_fresh_continuity.py",
    "native_aggregate": "tools/seal_badminton_repaired_release.py",
}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _equal(a, b):
    return _encode(a) == _encode(b)


def _integer(value, name):
    _need(type(value) is int and value >= 0, "Require an exact nonnegative integer: " + name)
    return value


def _path(value):
    _need(isinstance(value, (str, Path)) and bool(str(value)), "Require an absolute file path")
    path = Path(value)
    _need(path.is_absolute() and ".." not in path.parts, "Require an absolute path without traversal")
    resolved = path.resolve()
    _need(
        os.path.normcase(str(path)) == os.path.normcase(str(resolved)),
        "No-change evidence cannot use a symbolic link, junction or path alias",
    )
    return resolved


class _Frozen:
    def __init__(self):
        self.items = {}

    def add(self, item):
        incremental_release.require_binding(item)
        path = _path(item["path"])
        _need(path.is_file(), "No-change evidence must be an existing regular file")
        key = os.path.normcase(str(path))
        _need(key not in self.items or _equal(self.items[key], item), "Conflicting immutable file bindings")
        _need(file_sha(path) == item["sha256"], "Frozen no-change dependency changed: " + str(path))
        self.items[key] = deepcopy(item)
        return deepcopy(item)

    def bind(self, path):
        path = _path(path)
        return self.add(dict(path=str(path), sha256=file_sha(path)))

    def read(self, item):
        item = self.add(item)
        payload = Path(item["path"]).read_bytes()
        _need(hashlib.sha256(payload).hexdigest() == item["sha256"], "JSON changed while it was read")
        value = json.loads(payload.decode("utf-8-sig"))
        _need(isinstance(value, dict), "No-change metadata must be a JSON object")
        _encode(value)
        return value

    def recheck(self):
        for item in list(self.items.values()):
            self.add(item)


def _policy(path, sha256, frozen):
    selected = dict(path=str(_path(path)), sha256=sha256)
    frozen.add(selected)
    policy = incremental_release.load_release_policy(selected["path"], selected["sha256"])
    _need(_equal(policy.binding, selected), "Loader selected a different release policy")
    for item in policy.dependencies:
        frozen.add(item)
    return policy


def _entry(entry, frozen):
    _integer(entry["accepted"], "artifact accepted")
    return tuple(
        frozen.add(dict(path=entry[key], sha256=entry[sha]))
        for key, sha in (
            ("path", "sha256"),
            ("metadata", "metadata_sha256"),
        )
    )


def _body_entry(entry, dependencies, expected_tensor, frozen):
    """Use the original full continuity audit's two immutable export bindings."""
    _integer(entry["accepted"], "artifact accepted")
    _need(isinstance(dependencies, list) and len(dependencies) == 2, "Require exactly two original body dependencies")
    result = []
    for (key, sha, expected), item in zip(
        (
            ("path", "sha256", expected_tensor),
            ("metadata", "metadata_sha256", expected_tensor.with_suffix(".metadata.jsonl")),
        ),
        dependencies,
    ):
        incremental_release.require_binding(item)
        _need(
            _path(entry[key]) == _path(item["path"]) == expected,
            "Original body dependency must name the exact canonical tensor or metadata path",
        )
        _need(
            sha not in entry or entry[sha] == item["sha256"],
            "Optional full-body export hash conflicts with original audit",
        )
        result.append(frozen.add(item))
    return tuple(result)


def _body_sample_key(row, sid):
    """Read the original motion identity without adding bilateral metadata fields."""
    motion = row.get("motion_id")
    pieces = motion.split(":") if isinstance(motion, str) else []
    _need(
        len(pieces) == 4
        and pieces[0] == sid
        and row.get("session_id") == sid
        and bool(pieces[1])
        and re.fullmatch(r"[0-9]+", pieces[2]) is not None,
        "Original body motion identity has invalid source/window/track fields",
    )
    track = int(pieces[2])
    optional_track = row.get("track_id", track)
    _need(
        ("window_id" not in row or row["window_id"] == pieces[1])
        and type(optional_track) in (int, str)
        and str(optional_track).isdigit()
        and int(optional_track) == track,
        "Optional original body identity fields conflict with motion_id",
    )
    return sid, pieces[1], track, _integer(row.get("lidar_index"), "full-body scan")


def _review(review_binding, source_binding, frozen):
    review = frozen.read(review_binding)
    _need(
        set(review) == REVIEW_FIELDS
        and review.get("schema") == REVIEW
        and review.get("review_status") == "root_reviewed",
        "Require the closed root-reviewed no-change schema",
    )
    _need(
        type(review["identity_changes"]) is int
        and review["identity_changes"] == 0
        and review["training_authorized"] is False
        and review["physical_identity_ground_truth"] is False,
        "No-change review cannot authorize identity changes, training or identity ground truth",
    )
    _need(isinstance(review["reviewed_by"], str) and bool(review["reviewed_by"].strip()), "Missing root reviewer")
    _need(isinstance(review["reviewed_utc"], str), "Missing actual review time")
    stamp = datetime.fromisoformat(review["reviewed_utc"].replace("Z", "+00:00"))
    _need(stamp.tzinfo is not None, "Review time must include a timezone")
    _need(_equal(review["source_publication"], source_binding), "Root reviewed a different bilateral publication")
    audit = frozen.read(review["continuity_audit"])
    _need(
        audit.get("schema") == "badminton-fresh-continuity-audit-v1"
        and audit.get("state") == "complete"
        and audit.get("physical_identity_ground_truth") is False
        and isinstance(audit.get("pelvis_jump_candidates"), list)
        and audit.get("mutual_swap_candidates") == []
        and isinstance(review["cases"], list),
        "Require a completed original full continuity audit without mutual swaps",
    )
    candidates, seen = audit["pelvis_jump_candidates"], set()
    _need(
        audit.get("reviewed_boundaries") is None
        and _equal(audit.get("unresolved_jump_candidates"), candidates)
        and audit.get("screening_passed_no_flags") is (not candidates)
        and audit.get("reviewed_continuity_passed") is (not candidates),
        "Original full continuity audit flags must retain the original raw-audit meaning",
    )
    for case in review["cases"]:
        _need(isinstance(case, dict) and set(case) == CASE_FIELDS, "No-change review case is not closed")
        index = _integer(case["case"], "review case")
        _need(index < len(candidates) and index not in seen, "Unknown or duplicate continuity review case")
        seen.add(index)
        _need(
            _equal(case["event"], candidates[index]) and case["verdict"] == "same_person",
            "Every original continuity candidate needs a bound same-person review",
        )
        witness = frozen.read(case["evidence"])
        frozen.add(case["image"])
        _need(
            witness.get("case") == index
            and type(witness.get("case")) is int
            and _equal(witness.get("event"), case["event"])
            and _equal(witness.get("audit"), review["continuity_audit"])
            and _equal(witness.get("image"), case["image"]),
            "Native witness, image, candidate and original audit disagree",
        )
        records = witness.get("records", [])
        _need(
            isinstance(records, list)
            and len(records) == 4
            and {(r.get("side"), r.get("camera")) for r in records}
            == {(side, camera) for side in ("before", "after") for camera in ("cam1", "cam2")},
            "Require original before/after witnesses from both cameras",
        )
        for record in records:
            pose = frozen.read(record["pose"])
            frozen.add(record["fit"])
            _need(
                pose.get("raw_bgr_sha256") == record.get("raw_bgr_sha256")
                and isinstance(record.get("raw_bgr_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", record["raw_bgr_sha256"]) is not None,
                "Native witness pixel provenance changed",
            )
    _need(seen == set(range(len(candidates))), "Root review must cover every original continuity flag")
    return review, audit


def _lineage(source, body, body_binding, review, policy, frozen):
    labels = Path(body_binding["path"]).parent.parent
    repair = labels.parent
    _need(
        _path(review["source_publication"]["path"])
        == repair.with_name(repair.name + "_paired") / "datasets/dataset_manifest.json",
        "Bilateral source must be the actual completed paired publication",
    )
    _need(labels.name == "final_labels", "Require the actual full final-label body manifest")
    registry_path = repair.parent / "source/source_registry.json"
    registry_binding = frozen.bind(registry_path)
    registry = frozen.read(registry_binding)
    source_input = frozen.read(frozen.bind(repair / "new_source_input.json"))
    _need(
        source_input.get("schema") == "badminton-new-source-repair-input-v1"
        and source_input.get("review_only") is not True
        and registry.get("review_only") is not True
        and _equal(source_input.get("release_policy"), policy.binding)
        and _equal(registry.get("release_policy"), policy.binding)
        and _equal(source_input.get("session_splits"), policy.roles)
        and _equal(source_input.get("required_nonempty_splits"), policy.required)
        and _path(source_input.get("source_root", "")) == registry_path.parent
        and source_input.get("source_registry_sha256") == registry_binding["sha256"]
        and set(registry.get("sources", {})) == set(policy.scope["review_sources"])
        and set(source["exports"]) == set(body["exports"]) == set(registry["sources"]),
        "Actual source input/registry/roles differ from the explicit reviewed release policy",
    )
    for sid, entry in registry["sources"].items():
        _need(
            _equal(entry.get("source_index"), policy.scope["review_sources"][sid]["admitted_source_index"]),
            "Actual source admission differs from the reviewed policy scope",
        )
    calibration = frozen.read(frozen.bind(labels / "calibration/calibration_snapshot.json"))
    _need(
        isinstance(source.get("calibration_digest"), str)
        and re.fullmatch(r"[0-9a-f]{64}", source["calibration_digest"]) is not None
        and source["calibration_digest"] == body.get("calibration_digest") == calibration.get("snapshot_digest"),
        "Full body and bilateral publication use different cameras",
    )
    completion = frozen.read(review["source_completion"])
    _need(
        _path(review["source_completion"]["path"]) == repair / "new_source_completion_receipt.json"
        and completion.get("state") == "labels_published_viewer_prepared"
        and completion.get("dataset_manifest") == review["source_publication"]["path"]
        and type(completion.get("accepted")) is int
        and completion["accepted"] == source["accepted"]
        and isinstance(completion.get("evidence"), dict),
        "Require actual complete native/body/bilateral source completion for this publication",
    )
    for path, sha in completion["evidence"].items():
        frozen.add(dict(path=path, sha256=sha))
    for item in _validate_new_completion(repair, completion, review["source_publication"], source):
        frozen.add(item)
    return repair, completion


def _compute(source_binding, review_binding, policy, frozen):
    source = frozen.read(source_binding)
    _need(
        source.get("schema") == "badminton-ptv3-dualcam-dataset-v1"
        and source.get("status") == "complete"
        and source.get("qualification_policy") == "bilateral_detection"
        and all(source.get(name) == value for name, value in CONTRACT.items())
        and source.get("review_only") is not True
        and "canonical_identity_revision" not in source
        and ("release_policy" not in source or _equal(source["release_policy"], policy.binding)),
        "Require the actual complete original bilateral publication, never another identity revision",
    )
    _integer(source.get("accepted"), "bilateral accepted")
    for name, sha_name in (
        ("audit", "audit_sha256"),
        ("quality_summary", "quality_summary_sha256"),
        ("full_observation_manifest", "full_observation_manifest_sha256"),
    ):
        if source.get(name) is not None:
            frozen.add(dict(path=source[name], sha256=source.get(sha_name)))
    roles = source_roles(source, required_nonempty=policy.required)
    _need(
        _equal({role: entry["sessions"] for role, entry in source["suggested_session_splits"].items()}, policy.roles),
        "Bilateral whole-source roles differ from the explicit release policy",
    )
    review, audit = _review(review_binding, source_binding, frozen)
    body_binding = dict(path=source["source_dataset_manifest"], sha256=source["source_dataset_manifest_sha256"])
    body = frozen.read(body_binding)
    _need(body.get("review_only") is not True, "A review-only body cannot become a canonical publication")
    _need(_equal(audit.get("body_manifest"), body_binding), "Original full audit names a different body manifest")
    repair, completion = _lineage(source, body, body_binding, review, policy, frozen)
    native = frozen.read(audit["native_verification"])
    aggregate = frozen.read(audit["aggregate_native_contract"])
    full_count = _integer(body.get("accepted"), "full body accepted")
    _need(
        _path(audit["native_verification"]["path"]) == repair / "final_labels/native_verification.json"
        and native.get("state") == "complete"
        and native.get("passed") is True
        and type(native.get("accepted_checked")) is int
        and native["accepted_checked"] == full_count
        and native.get("calibration_digest") == source["calibration_digest"]
        and aggregate.get("passed") is True
        and type(aggregate.get("checked")) is int
        and aggregate["checked"] == full_count
        and aggregate.get("all_point_and_label_tensors_equal_native_verified_windows") is True
        and _path(aggregate.get("native_verification", "")) == _path(audit["native_verification"]["path"])
        and aggregate.get("native_verification_sha256") == audit["native_verification"]["sha256"]
        and completion["evidence"].get(audit["native_verification"]["path"]) == audit["native_verification"]["sha256"],
        "Full native/aggregate/source-completion evidence is incomplete or belongs to a different body",
    )
    counts = audit.get("counts", {})
    _need(
        type(counts.get("person_frames")) is int
        and counts["person_frames"] == full_count
        and type(counts.get("pelvis_jump_candidates")) is int
        and counts["pelvis_jump_candidates"] == len(audit["pelvis_jump_candidates"])
        and type(counts.get("mutual_swap_candidates")) is int
        and counts["mutual_swap_candidates"] == 0
        and set(audit.get("sources", {})) == set(body["exports"]),
        "Original continuity audit did not cover the complete native body scope",
    )
    producer = frozen.bind(Path(__file__).resolve())
    verifiers = {name: frozen.bind(PROJ_ROOT / name_path) for name, name_path in VERIFIERS.items()}
    audit_producer = frozen.add(audit["producer"])
    _need(audit_producer["sha256"] == verifiers["continuity_audit"]["sha256"], "Unknown continuity audit producer")
    sessions, total = {}, 0
    for sid, entry in source["exports"].items():
        tensor, metadata = _entry(entry, frozen)
        expected_tensor = repair.with_name(repair.name + "_paired") / "datasets" / (sid + ".pth")
        _need(
            _path(tensor["path"]) == expected_tensor
            and _path(metadata["path"]) == expected_tensor.with_suffix(".metadata.jsonl"),
            "Bilateral artifact paths cannot be substituted with aliases or foreign publications",
        )
        current_audit = audit["sources"][sid]
        original_tensor, original_metadata = _body_entry(
            body["exports"][sid],
            current_audit.get("dependencies"),
            repair / "final_labels/datasets" / (sid + ".pth"),
            frozen,
        )
        body_count = _integer(body["exports"][sid]["accepted"], "source full count")
        # Keep the original aggregate verifier's window byte predicates live.
        # No native/model pass or aggregate JSON rewrite is started here.
        for window in body["selected_windows"][sid]:
            directory = repair / "final_labels/sessions" / sid / "windows" / window
            report_binding = dict(
                path=str(directory / "fit_report.json"),
                sha256=native["windows"][sid + ":" + window]["body"]["report_sha256"],
            )
            report = frozen.read(report_binding)
            for name in ("ptv3.pth", "ptv3.metadata.jsonl"):
                frozen.add(dict(path=str(directory / name), sha256=report["artifacts_sha256"][name]))
        _need(
            type(current_audit.get("person_frames")) is int
            and current_audit["person_frames"] == body_count
            and _equal(current_audit.get("dependencies"), [original_tensor, original_metadata])
            and type(aggregate.get("sessions", {}).get(sid)) is int
            and aggregate["sessions"][sid] == body_count,
            "Original native continuity source dependencies or counts changed",
        )
        data, original = joblib.load(tensor["path"]), joblib.load(original_tensor["path"])
        rows, original_rows = read_rows(metadata["path"]), read_rows(original_metadata["path"])
        _need(
            len(original_rows) == body_count
            and len(rows) == entry["accepted"]
            and original["motion_id"] == [row["motion_id"] for row in original_rows]
            and data["motion_id"] == [row["motion_id"] for row in rows],
            "Body or bilateral tensor/metadata count and identity order changed",
        )
        scan_times = {}
        for row in original_rows:
            pieces = row["motion_id"].split(":")
            scan = _integer(row.get("lidar_index"), "full-body scan")
            stamp = _integer(row.get("scan_mid_us"), "full-body scan time")
            _need(len(pieces) == 4 and pieces[0] == sid, "Full-body identity belongs to another source")
            key = pieces[1], scan
            _need(key not in scan_times or scan_times[key] == stamp, "People on one native scan have different times")
            scan_times[key] = stamp
        _need(
            len({_body_sample_key(row, sid) for row in original_rows}) == len(original_rows),
            "Duplicate original full-body person-scan",
        )
        _need(len({sample_key(row) for row in rows}) == len(rows), "Duplicate bilateral person-scan")
        selected = []
        for row in rows:
            index = _integer(row.get("source_tensor_index"), "original body row index")
            _need(index < len(original_rows), "Bilateral row is outside its actual full body")
            previous = original_rows[index]
            pieces = row["motion_id"].split(":")
            _need(
                row.get("session_id") == previous.get("session_id") == sid
                and all(
                    _equal(row.get(key), previous.get(key))
                    for key in (
                        "motion_id",
                        "lidar_index",
                        "scan_mid_us",
                        "calibration_digest",
                    )
                )
                and row.get("calibration_digest") == source["calibration_digest"]
                and len(pieces) == 4
                and type(row.get("track_id")) in (int, str)
                and str(row["track_id"]) == pieces[2]
                and ("scene_id" not in row or row["scene_id"] == sid)
                and ("window_id" not in row or row["window_id"] == pieces[1])
                and ("segment_id" not in row or str(row["segment_id"]) == pieces[3])
                and ("person_id" not in row or row["person_id"] == row["motion_id"])
                and ("frame_index" not in row or _equal(row["frame_index"], row["lidar_index"]))
                and (
                    "source_tensor_path" not in row
                    or _path(row["source_tensor_path"]) == _path(original_tensor["path"])
                )
                and "canonical_identity" not in row,
                "Bilateral row is not the unchanged identity/scan/time subset of this full body",
            )
            added = {
                "scene_id",
                "person_id",
                "person_identity_scope",
                "window_id",
                "track_id",
                "segment_id",
                "frame_index",
                "source_tensor_path",
                "source_tensor_index",
                "dualcam",
            }
            _need(
                all(_equal(row.get(name), value) for name, value in previous.items() if name not in added),
                "Bilateral subset changed original native/teacher/QA metadata",
            )
            selected.append(index)
        _need(len(selected) == len(set(selected)), "Duplicate full-body subset row")
        indices = torch.tensor(selected, dtype=torch.long)
        for field, shape in FIELDS.items():
            _need(
                tuple(original[field].shape) == (body_count, *shape)
                and tuple(data[field].shape) == (len(rows), *shape)
                and original[field].dtype == data[field].dtype == torch.float32
                and original[field].device.type == data[field].device.type == "cpu"
                and bool(torch.isfinite(original[field]).all())
                and bool(torch.isfinite(data[field]).all())
                and torch.equal(data[field], original[field][indices]),
                "Bilateral subset changed original finite float32 numeric values: " + field,
            )
        sessions[sid] = dict(
            full_accepted=body_count,
            accepted=len(rows),
            role=roles[sid],
            source_tensor=tensor,
            source_metadata=metadata,
            body_tensor=original_tensor,
            body_metadata=original_metadata,
            numeric_values_bit_exact=True,
            identity_keys_unchanged=True,
            exact_body_subset=True,
        )
        total += body_count
    _need(total == full_count, "Full-body per-source counts differ from the original audit")
    for entry in source["suggested_session_splits"].values():
        _entry(entry, frozen)
        _need(
            Path(entry["metadata"]).read_bytes()
            == b"".join(Path(source["exports"][sid]["metadata"]).read_bytes() for sid in entry["sessions"]),
            "Bilateral split metadata differs from exact ordered source bytes",
        )
        if len(entry["sessions"]) == 1:
            part = source["exports"][entry["sessions"][0]]
            _need(
                all(_equal(entry[name], part[name]) for name in ("path", "metadata", "sha256", "metadata_sha256")),
                "Single-source split cannot substitute an alias or another artifact",
            )
        elif not entry["sessions"]:
            empty = joblib.load(entry["path"])
            _need(
                empty.get("motion_id") == []
                and all(
                    tuple(empty[name].shape) == (0, *shape) and empty[name].dtype == torch.float32
                    for name, shape in FIELDS.items()
                ),
                "An explicitly empty role contains foreign numeric rows",
            )
    original_verification = _verify_export_data(source)
    _need(original_verification["accepted"] == source["accepted"], "Original bilateral verifier count differs")
    proof = dict(
        schema=PROOF,
        state="complete",
        producer=producer,
        verifiers=verifiers,
        release_policy=policy.binding,
        source_publication=source_binding,
        source_completion=review["source_completion"],
        continuity_audit=review["continuity_audit"],
        reviewed_continuity=review_binding,
        native_verification=audit["native_verification"],
        aggregate_native_contract=audit["aggregate_native_contract"],
        full_accepted=full_count,
        accepted=source["accepted"],
        sessions=sessions,
        identity_changes=0,
        numeric_values_unchanged=True,
        admission_membership_unchanged=True,
        identity_keys_unchanged=True,
        split_only_no_unproven_join=True,
        unresolved_jump_candidates=[],
        mutual_swap_candidates=[],
        training_authorized=False,
        physical_identity_ground_truth=False,
    )
    policy.recheck()
    frozen.recheck()
    return source, proof


def publish_no_change_view(source_path, review_path, output_root, *, release_policy, release_policy_sha256):
    """Write only a fresh proof and manifest, after real unchanged-subset validation."""
    frozen = _Frozen()
    policy = _policy(release_policy, release_policy_sha256, frozen)
    source_binding, review_binding = frozen.bind(source_path), frozen.bind(review_path)
    output = _path(output_root)
    _need(output.is_relative_to(Path(policy.data["gvhmr_root"]) / "outputs"), "Use the explicit policy outputs root")
    _need(not output.exists() or not any(output.iterdir()), "No-change publication requires a fresh empty destination")
    source, proof = _compute(source_binding, review_binding, policy, frozen)
    repair = Path(source["source_dataset_manifest"]).parent.parent.parent
    _need(output.is_relative_to(repair), "Canonical no-change evidence must remain inside its actual repair run")
    _need(
        not any(Path(item["path"]).is_relative_to(output) for item in frozen.items.values()),
        "No-change output cannot contain any frozen source evidence",
    )
    output.mkdir(parents=True, exist_ok=True)
    proof_path = output / "canonical_verification.json"
    manifest = deepcopy(source)
    manifest.update(schema=PUBLICATION, release_policy=deepcopy(policy.binding))
    write_json(proof_path, proof)
    manifest["canonical_identity_revision"] = dict(
        mode=MODE,
        source_publication=source_binding,
        reviewed_continuity=review_binding,
        verification=frozen.bind(proof_path),
        numeric_geometry_unchanged=True,
        identity_keys_unchanged=True,
    )
    policy.recheck()
    frozen.recheck()
    write_json(output / "datasets/dataset_manifest.json", manifest)
    return manifest


def verify_no_change_publication(manifest, *, release_policy, release_policy_sha256):
    """Recompute the proof from actual source bytes; booleans alone never suffice."""
    frozen = _Frozen()
    policy = _policy(release_policy, release_policy_sha256, frozen)
    _need(isinstance(manifest, dict) and manifest.get("schema") == PUBLICATION, "Require an explicit v2 publication")
    marker = manifest.get("canonical_identity_revision")
    _need(
        isinstance(marker, dict)
        and set(marker) == MARKER_FIELDS
        and marker.get("mode") == MODE
        and marker.get("numeric_geometry_unchanged") is True
        and marker.get("identity_keys_unchanged") is True
        and _equal(manifest.get("release_policy"), policy.binding),
        "Require the closed explicit no-change publication marker and policy",
    )
    proof = frozen.read(marker["verification"])
    _need(set(proof) == PROOF_FIELDS and proof.get("schema") == PROOF, "No-change proof is not closed")
    source, expected_proof = _compute(marker["source_publication"], marker["reviewed_continuity"], policy, frozen)
    expected = dict(source, schema=PUBLICATION, release_policy=policy.binding, canonical_identity_revision=marker)
    _need(_equal(manifest, expected), "No-change publication changed original fields, paths, roles or artifacts")
    _need(_equal(proof, expected_proof), "No-change proof does not reproduce actual source-bound verification")
    frozen.recheck()
    return proof


def validate_no_change_acceptance(manifest, continuity_binding, browser, *, release_policy, release_policy_sha256):
    """Check the no-change seam after the shared real native-browser coverage gate.

    ``continuity_binding`` is the original full audit, not a fabricated audit of
    this wrapper.  Existing pixel/sequence/archive and shared viewer checks must
    still run in the caller.  This helper also rejects missing actual clip proof.
    """
    proof = verify_no_change_publication(
        manifest,
        release_policy=release_policy,
        release_policy_sha256=release_policy_sha256,
    )
    frozen = _Frozen()
    frozen.add(continuity_binding)
    _need(_equal(continuity_binding, proof["continuity_audit"]), "Seal selected another original full audit")
    _need(isinstance(browser, dict), "Require actual browser acceptance metadata")
    canonical = frozen.read(browser.get("canonical_publication"))
    _need(_equal(canonical, manifest), "Browser served a different canonical publication")
    _need(
        browser.get("passed") is True
        and browser.get("page_errors") == []
        and browser.get("native_boundary_regressions") == []
        and _equal(browser.get("canonical_no_change_proof"), manifest["canonical_identity_revision"]["verification"])
        and browser.get("canonical_mode") == MODE
        and _equal(browser.get("source_completion"), proof["source_completion"]),
        "No-change browser acceptance requires this actual proof without fabricated boundary regressions",
    )
    coverage = browser.get("frozen_source_coverage", {})
    active = sorted(sid for sid, item in proof["sessions"].items() if item["accepted"] > 0)
    empty = sorted(sid for sid, item in proof["sessions"].items() if item["accepted"] == 0)
    results = browser.get("results")
    _need(
        isinstance(results, list)
        and type(browser.get("clips")) is int
        and browser["clips"] == len(results)
        and type(browser.get("played_sources")) is int
        and browser["played_sources"] == len(active)
        and type(browser.get("declared_source_count")) is int
        and browser["declared_source_count"] == len(proof["sessions"])
        and coverage.get("active_sources") == active
        and coverage.get("empty_sources") == empty
        and type(browser.get("minimum_clip_seconds")) in (int, float)
        and browser["minimum_clip_seconds"] >= 5,
        "Browser did not cover the actual frozen no-change source scope",
    )
    rallies = {sid: set() for sid in active}
    for clip in results:
        sid = clip.get("source_id")
        _need(sid in rallies and clip.get("passed") is True, "Unknown or failed native browser clip")
        _need(
            type(clip.get("rally_index")) is int
            and clip["rally_index"] not in rallies[sid]
            and type(clip.get("unique_committed_scans")) is int
            and clip["unique_committed_scans"] >= 20
            and all(
                type(clip.get(key)) in (int, float) and math.isfinite(clip[key])
                for key in (
                    "actual_begin",
                    "actual_end",
                )
            )
            and clip["actual_end"] - clip["actual_begin"] >= 4.9,
            "Browser clip lacks distinct real rally/time/scan evidence",
        )
        frozen.add(clip["screenshot"])
        rallies[sid].add(clip["rally_index"])
    _need(all(len(values) >= 2 for values in rallies.values()), "Every source needs two actual native playback clips")
    for name in ("producer", "clip_plan", "viewer_manifest"):
        frozen.add(browser[name])
    frozen.recheck()
    return proof
