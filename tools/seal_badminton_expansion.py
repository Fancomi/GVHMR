"""Seal accepted expanded labels and archive one immutable canonical release.

Original old/new publications remain evidence. Formal role paths point directly
to the independent user data archive; no consumer manifest appears before the
archive and all native/body/identity/browser/pixel gates have passed.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat

import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.expanded_dataset import binding, check_binding
from hmr4d.lidar_mocap.rally_pipeline import read_json
from hmr4d.lidar_mocap.repair_execution import write_progress_json as write_json

# The consumer loads this tool by file spec without adding tools/ to sys.path.
_viewer_spec = importlib.util.spec_from_file_location(
    "_badminton_seal_viewer_contracts", Path(__file__).with_name("badminton_viewer_contracts.py")
)
_viewer_contracts = importlib.util.module_from_spec(_viewer_spec)
_viewer_spec.loader.exec_module(_viewer_contracts)
validate_browser_source_acceptance = _viewer_contracts.validate_browser_source_acceptance


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def future_binding(path, payload):
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(payload).hexdigest())


def _release_context(assembly, *, release_policy=None, release_policy_sha256=None):
    """Resolve only an explicitly selected v2 policy; legacy keeps its own route."""
    if (release_policy is None) != (release_policy_sha256 is None):
        raise ValueError("Release policy path and SHA256 must be supplied together")
    contract = None
    if assembly.get("consumer_contract"):
        check_binding(assembly["consumer_contract"])
        contract = read_json(assembly["consumer_contract"]["path"])
    if not contract or contract.get("schema") != "expanded-source-increment-assembly-input/v2":
        if release_policy is not None or "release_policy" in assembly:
            raise ValueError("Legacy sealing cannot override its release policy")
        return None, contract
    if release_policy is None:
        raise ValueError("V2 sealing requires an explicit release policy path and SHA256")
    from hmr4d.lidar_mocap.incremental_release import load_release_policy, validate_contract

    policy = load_release_policy(release_policy, release_policy_sha256)
    validate_contract(contract, policy)
    if (
        assembly.get("release_policy") != policy.binding
        or assembly.get("training_published") is not False
        or assembly["source_publications"]["old"] != policy.parent["bindings"]["dataset_manifest"]
        or assembly["source_publications"]["new"] != contract["bindings"]["producer_joint_manifest"]
        or type(assembly.get("old_accepted")) is not int
        or assembly["old_accepted"] != policy.parent["total"]
        or type(assembly.get("new_accepted")) is not int
        or assembly["new_accepted"] != contract["new_final_admitted_counts"]["total"]
        or type(assembly.get("accepted")) is not int
        or assembly["accepted"] != assembly["old_accepted"] + assembly["new_accepted"]
    ):
        raise ValueError("V2 assembly differs from its explicit policy, parent or current publication")
    for role in ("train", "val", "test"):
        entry = assembly["suggested_session_splits"][role]
        if (
            type(entry.get("accepted")) is not int
            or entry["accepted"] != policy.parent["counts"][role] + contract["new_final_admitted_counts"][role]
            or entry["sessions"] != policy.parent["roles"][role] + policy.roles[role]
        ):
            raise ValueError("V2 assembly changes a complete union source role or count")
    policy.recheck()
    return policy, contract


def _validate_incremental_browser(browser, publication, policy, mode):
    if (
        browser.get("schema") != "badminton-fresh-browser-acceptance-v1"
        or browser.get("release_policy") != policy.binding
        or browser.get("canonical_mode") != mode
    ):
        raise ValueError("V2 browser acceptance uses another explicit policy or canonical mode")
    check_binding(browser["clip_plan"])
    check_binding(browser["viewer_manifest"])
    plan = read_json(browser["clip_plan"]["path"])
    viewer = read_json(browser["viewer_manifest"]["path"])
    if (
        plan.get("schema") != "badminton-fresh-browser-clips-v1"
        or plan.get("release_policy") != policy.binding
        or plan.get("canonical_mode") != mode
        or plan.get("canonical_publication") != publication
        or plan.get("source_completion") != browser.get("source_completion")
        or plan.get("frozen_source_coverage") != browser.get("frozen_source_coverage")
        or not isinstance(plan.get("clips"), list)
        or len(plan["clips"]) != len(browser["results"])
    ):
        raise ValueError("V2 browser plan differs from its actual receipt, source or explicit policy")
    for planned, result in zip(plan["clips"], browser["results"]):
        if (
            not isinstance(planned, dict)
            or type(planned.get("duration_seconds")) not in (int, float)
            or planned["duration_seconds"] != 5
            or type(planned.get("rally_index")) is not int
            or type(planned.get("begin")) not in (int, float)
            or not math.isfinite(planned["begin"])
            or not all(result.get(name) == value for name, value in planned.items())
        ):
            raise ValueError("V2 browser result differs from its frozen genuine five-second clip plan")
        rallies = viewer.get("sources", {}).get(planned.get("source_id"), {}).get("rallies", [])
        matching = [rally for rally in rallies if rally.get("index") == planned["rally_index"]]
        if (
            len(matching) != 1
            or not matching[0]["begin"] <= planned["begin"]
            or not planned["begin"] + 5 <= matching[0]["end"]
            or matching[0]["end"] - matching[0]["begin"] < 5.5
        ):
            raise ValueError("V2 browser clip is outside its actual frozen native rally")
        if not all(
            type(result.get(name)) in (int, float) and math.isfinite(result[name])
            for name in ("actual_begin", "actual_end")
        ):
            raise ValueError("V2 browser playback requires finite actual source times")
        samples = result.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("V2 browser playback lacks actual native mesh/projection samples")
        for sample in samples:
            if (
                not isinstance(sample, dict)
                or type(sample.get("scan")) is not int
                or type(sample.get("sceneScan")) is not int
                or sample["scan"] != sample.get("sceneScan")
                or sample.get("webglError")
                or not isinstance(sample.get("athleteIds"), list)
                or type(sample.get("meshCount")) is not int
                or sample["meshCount"] != len(sample["athleteIds"])
                or type(sample.get("vertexCount")) is not int
                or sample["vertexCount"] != sample["meshCount"] * 6890
                or not isinstance(sample.get("projections"), list)
                or len(sample["projections"]) != 2
                or not all(
                    isinstance(p, dict)
                    and type(p.get("scan")) is int
                    and p["scan"] == sample["scan"]
                    and not p.get("error")
                    for p in sample["projections"]
                )
            ):
                raise ValueError("V2 browser native cloud, full meshes and both projections are not synchronized")
        scans = {sample["scan"] for sample in samples}
        labelled = {sample["scan"] for sample in samples if sample["meshCount"] > 0}
        if (
            len(scans) < 20
            or len(labelled) < 20
            or type(result.get("unique_committed_scans")) is not int
            or result["unique_committed_scans"] != len(scans)
            or type(result.get("labelled_scans")) is not int
            or result["labelled_scans"] != len(labelled)
        ):
            raise ValueError("V2 browser clip lacks twenty actual committed labelled native scans")
        for name in ("initial_visible_rgb", "final_visible_rgb"):
            views = result.get(name)
            if (
                not isinstance(views, list)
                or len(views) != 2
                or not all(isinstance(view, dict) for view in views)
                or {view.get("camera") for view in views} != {"cam1", "cam2"}
                or not all(
                    view.get("visible") is True
                    and type(view.get("different_rgba_components")) is int
                    and view["different_rgba_components"] == 0
                    for view in views
                )
            ):
                raise ValueError("V2 browser endpoints lack both exact PTS-bound visible RGB checks")
        check_binding(result["screenshot"])
    if mode == "reviewed_no_changes" and plan.get("canonical_no_change_proof") != browser.get(
        "canonical_no_change_proof"
    ):
        raise ValueError("V2 browser plan names another no-change verification")


def validate_acceptance(
    assembly_path, browser_path, continuity_path, pixels_path, *, release_policy=None, release_policy_sha256=None
):
    frozen = {
        name: binding(path)
        for name, path in dict(
            assembly=assembly_path, browser=browser_path, continuity=continuity_path, pixels=pixels_path
        ).items()
    }
    assembly, browser, continuity, pixels = [
        read_json(frozen[name]["path"]) for name in ("assembly", "browser", "continuity", "pixels")
    ]
    if assembly.get("schema") != "badminton-expanded-assembly-v1" or assembly.get("training_published"):
        raise ValueError("Seal an explicit unpublished exact expansion assembly")
    if not all(
        assembly.get(name) is True
        for name in (
            "all_old_rows_retained",
            "all_tensors_equal_ordered_source_concatenation",
            "all_metadata_bytes_equal_ordered_source_concatenation",
        )
    ):
        raise ValueError("Exact original sample retention is not proved")
    policy, contract = _release_context(
        assembly, release_policy=release_policy, release_policy_sha256=release_policy_sha256
    )
    for item in assembly["source_publications"].values():
        check_binding(item)
    new = read_json(assembly["source_publications"]["new"]["path"])
    no_changes = policy is not None and contract["canonical_mode"] == "reviewed_no_changes"
    if new.get("schema") == "badminton-canonical-bilateral-publication-v2" and not no_changes:
        raise ValueError("A no-change publication requires its explicit reviewed v2 contract mode")
    if (
        policy is not None
        and not no_changes
        and (
            contract["canonical_mode"] != "reviewed_cuts"
            or new.get("schema") != "badminton-canonical-bilateral-publication-v1"
            or new.get("release_policy") != policy.binding
        )
    ):
        raise ValueError("V2 cut mode requires the actual reviewed cut publication")
    validate_browser_source_acceptance(browser, assembly["source_publications"]["new"])
    if browser.get("canonical_publication") != assembly["source_publications"]["new"]:
        raise ValueError("Browser acceptance uses another canonical publication")
    if policy is not None:
        _validate_incremental_browser(
            browser, assembly["source_publications"]["new"], policy, contract["canonical_mode"]
        )
    if no_changes:
        from hmr4d.lidar_mocap.canonical_no_change import validate_no_change_acceptance

        validate_no_change_acceptance(
            new,
            frozen["continuity"],
            browser,
            release_policy=release_policy,
            release_policy_sha256=release_policy_sha256,
        )
    else:
        if not browser.get("native_boundary_regressions") or not all(
            c["passed"] for c in browser["native_boundary_regressions"]
        ):
            raise ValueError("Native identity boundary browser regressions are missing")
        if (
            continuity.get("state") != "complete"
            or continuity.get("reviewed_continuity_passed") is not True
            or continuity.get("unresolved_jump_candidates")
            or continuity.get("mutual_swap_candidates")
        ):
            raise ValueError("Post-revision native continuity review remains unresolved")
        if continuity["body_manifest"] != assembly["source_publications"]["new"]:
            raise ValueError("Continuity audit uses another canonical publication")
    if (
        pixels.get("status") != "complete"
        or pixels.get("all_person_admissions_unchanged") is not True
        or pixels.get("admission_changes")
    ):
        raise ValueError("Every affected native scan must preserve full-person bilateral admission")
    if pixels["assembly"] != frozen["assembly"]:
        raise ValueError("Native pixels were audited against another expansion")
    check_binding(pixels["initial_pixel_audit"])
    first = read_json(pixels["initial_pixel_audit"]["path"])
    if (
        first["checked_camera_frames"] != first["expected_camera_frames"]
        or pixels["checked_camera_frames"] != first["checked_camera_frames"]
    ):
        raise ValueError("Held-out native image coverage is incomplete")
    if pixels["metadata"] != binding(assembly["suggested_session_splits"]["test"]["metadata"]):
        raise ValueError("Native pixel supplement names another test metadata file")
    required = {(c["scene_id"], c["camera"], c["native"]["record_index"]) for c in first["unknown_differences"]}
    required.update((c["scene_id"], c["camera"], c["record_index"]) for c in first["corrections"])
    resolved = {(c["scene_id"], c["camera"], c["record_index"]) for c in pixels["corrections"]}
    if not required <= resolved:
        raise ValueError("Native pixel differences are not fully closed")
    for item in pixels["source_videos"].values():
        check_binding(item)
    checked = set()
    for correction in pixels["corrections"]:
        check_binding(correction["original_report"])
        evidence = correction["evidence"]
        check_binding(evidence)
        proof = read_json(evidence["path"])
        camera = correction["camera"]
        before, after = proof["original_views"][camera], proof["verified_views"][camera]
        capture = correction["verified_pixel_capture"]
        if (
            before["native_bgr_image_sha256"] != correction["previous_raw_bgr_sha256"]
            or after["native_bgr_image_sha256"] != correction["verified_raw_bgr_sha256"]
            or capture["raw_bgr_sha256"] != correction["verified_raw_bgr_sha256"]
        ):
            raise ValueError("Pixel correction hashes do not match actual before/after evidence")
        for field in ("record_index", "media_pts_us", "common_time_us", "source_frame_no"):
            if (
                capture["native"][field] != correction[field]
                or after["camera_native_source"][field] != correction[field]
            ):
                raise ValueError("Pixel correction refers to another native frame")
        if proof.get("all_person_admissions_unchanged") is not True or not all(p["unchanged"] for p in proof["people"]):
            raise ValueError("A full-person native recheck changed admission")
        if proof["metadata"] != pixels["metadata"] or proof["assembly"] != frozen["assembly"]:
            raise ValueError("Pixel admission evidence uses another publication")
        checked.add(evidence["path"])
    for name in ("producer", "clip_plan", "source_completion"):
        check_binding(browser[name])
    for item in frozen.values():
        check_binding(item)
    if policy is not None:
        policy.recheck()
    return frozen, assembly, pixels


class Archive:
    def __init__(self, root, release_id):
        self.base = Path(root).resolve()
        self.parent = self.base / "releases"
        if not re.fullmatch(r"[A-Za-z0-9_-]+", release_id):
            raise ValueError("Use a plain release identifier")
        self.final = (self.parent / release_id).resolve()
        self.stage = (self.parent / (release_id + ".partial")).resolve()
        if (
            self.stage.parent != self.parent
            or self.final.parent != self.parent
            or self.stage.exists()
            or self.final.exists()
        ):
            raise ValueError("A new archive must have an absent, bounded staging/final destination")
        self.stage.mkdir(parents=True)
        self.artifacts = []

    def destination(self, relative):
        path = (self.stage / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(self.stage):
            raise ValueError("Archive path escapes staging")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError("Archive artifact already exists: " + relative)
        return path

    def ref(self, relative, path):
        value = dict(path=relative, bytes=path.stat().st_size, sha256=file_sha(path))
        self.artifacts.append(value)
        return value

    def copy(self, source, relative, expected=None):
        source = Path(source).resolve()
        before = file_sha(source)
        if expected and before != expected:
            raise ValueError("Frozen archive input changed: " + str(source))
        target = self.destination(relative)
        shutil.copy2(source, target)
        if os.path.samefile(source, target) or file_sha(target) != before or file_sha(source) != before:
            raise ValueError("Independent archive copy verification failed: " + relative)
        return self.ref(relative, target)

    def write(self, relative, value):
        target = self.destination(relative)
        target.write_bytes(value if isinstance(value, bytes) else encode(value))
        return self.ref(relative, target)

    def future(self, ref):
        return dict(path=str(self.final / ref["path"]), sha256=ref["sha256"])


def _archive_release_provenance(archive, contract, policy):
    if policy is None:
        parent_receipt = read_json(contract["bindings"]["parent_receipt"]["path"])
        bridge = parent_receipt["model_bridge"]
        archive.copy(bridge["path"], "provenance/model_bridge.json", bridge["sha256"])
        return
    policy.recheck()
    for name, item in dict(
        release_policy=policy.binding,
        model_bridge=policy.parent["model_bridge"],
        reviewed_scope=policy.data["reviewed_scope"],
    ).items():
        archive.copy(item["path"], "provenance/" + name + ".json", item["sha256"])
    for name, item in policy.parent["bindings"].items():
        archive.copy(item["path"], "provenance/parent_release/" + name + ".json", item["sha256"])
    policy.recheck()


def _archive_canonical_evidence(archive, new, continuity, policy, contract):
    marker = new["canonical_identity_revision"]
    if policy is None or contract["canonical_mode"] == "reviewed_cuts":
        review = marker["reviewed_boundaries"]
        archive.copy(review["path"], "provenance/reviewed_track_boundaries.json", review["sha256"])
        return
    proof = read_json(marker["verification"]["path"])
    for name, item in dict(
        canonical_no_change_verification=marker["verification"],
        reviewed_continuity=marker["reviewed_continuity"],
        full_continuity_audit=proof["continuity_audit"],
        full_body_manifest=continuity["body_manifest"],
        native_verification=proof["native_verification"],
        original_bilateral_publication=proof["source_publication"],
        source_completion=proof["source_completion"],
    ).items():
        archive.copy(item["path"], "provenance/" + name + ".json", item["sha256"])


def seal(args):
    selected_policy = dict(
        release_policy=getattr(args, "release_policy", None),
        release_policy_sha256=getattr(args, "release_policy_sha256", None),
    )
    frozen, assembly, pixels = validate_acceptance(
        args.assembly, args.browser, args.continuity, args.pixels, **selected_policy
    )
    policy, contract = _release_context(assembly, **selected_policy)
    if policy is not None and Path(args.data_root).resolve() != Path(policy.data["canonical_data_root"]).resolve():
        raise ValueError("V2 archive must use the exact canonical data root from its explicit policy")
    output = Path(args.assembly).resolve().parent
    formal_dataset = output / "datasets/dataset_manifest.json"
    formal_training = output / "training_manifest.json"
    if formal_dataset.exists() or formal_training.exists() or (output / "release_receipt.json").exists():
        raise ValueError("A formal publication already exists; never rewrite it")
    sequence_binding = binding(args.sequence_report)
    sequences = read_json(args.sequence_report)
    if policy is not None and (
        sequences.get("release_policy") != policy.binding
        or sequences.get("assembly") != frozen["assembly"]
        or sequences.get("frozen_contract") != assembly["consumer_contract"]
    ):
        raise ValueError("V2 rollout sequences require this exact assembly input and release policy")
    archive = Archive(args.data_root, args.release_id)
    catalog_path = archive.base / "dataset_catalog.json"
    catalog = read_json(catalog_path)
    catalog_frozen = binding(catalog_path)
    if any(r["release_id"] == args.release_id for r in catalog["releases"]):
        raise ValueError("Release is already cataloged")
    splits, archive_splits, archive_sequences = {}, {}, {}
    for role, entry in assembly["suggested_session_splits"].items():
        tensor = archive.copy(entry["path"], role + "/data.pth", entry["sha256"])
        metadata = archive.copy(entry["metadata"], role + "/metadata.jsonl", entry["metadata_sha256"])
        splits[role] = dict(
            entry, path=str(archive.final / tensor["path"]), metadata=str(archive.final / metadata["path"])
        )
        active = [sid for sid in entry["sessions"] if assembly["source_registry"][sid]["accepted"] > 0]
        archive_splits[role] = dict(
            frames=entry["accepted"],
            source_sessions=active,
            declared_source_sessions=entry["sessions"],
            tensor=tensor,
            metadata=metadata,
        )
        if role in ("train", "val"):
            sequence = sequences["splits"][role]
            if (
                sequence["source_tensor_sha256"] != entry["sha256"]
                or sequence["source_metadata_sha256"] != entry["metadata_sha256"]
            ):
                raise ValueError("Real rollout sequences use another tensor/metadata order")
            index = archive.copy(
                sequence["windows_path"], "sequences/rollout_" + role + "_windows.npy", sequence["windows_sha256"]
            )
            archive_sequences[role] = dict(
                windows=sequence["windows"], frames_per_window=12, stride=sequence["window_stride"], indices=index
            )
        print(dict(archived_role=role, accepted=entry["accepted"]), flush=True)
        write_json(
            output / "publication_progress.json",
            dict(stage="copying_verified_archive", completed_roles=list(splits), formal_published=False),
        )
    archive.copy(sequence_binding["path"], "provenance/rollout_windows.json", sequence_binding["sha256"])
    contract = read_json(assembly["consumer_contract"]["path"])
    check_binding(assembly["consumer_contract"])
    archive.copy(
        assembly["consumer_contract"]["path"],
        "provenance/consumer_contract.json",
        assembly["consumer_contract"]["sha256"],
    )
    _archive_release_provenance(archive, contract, policy)
    for name, item in frozen.items():
        archive.copy(item["path"], "provenance/" + name + ".json", item["sha256"])
        value = read_json(item["path"])
        if value.get("producer"):
            archive.copy(value["producer"]["path"], "provenance/" + name + "_producer.py", value["producer"]["sha256"])
    for name, item in assembly["source_publications"].items():
        archive.copy(item["path"], "provenance/" + name + "_source_publication.json", item["sha256"])
    new = read_json(assembly["source_publications"]["new"]["path"])
    continuity = read_json(args.continuity)
    _archive_canonical_evidence(archive, new, continuity, policy, contract)
    native_contract = continuity["aggregate_native_contract"]
    archive.copy(native_contract["path"], "provenance/native_aggregate_contract.json", native_contract["sha256"])
    producer = archive.copy(__file__, "provenance/seal_producer.py")
    calibrations = {}
    for sid, source in assembly["source_registry"].items():
        digest = source["calibration_digest"]
        if digest not in calibrations:
            path = Path(source["calibration_snapshot"])
            cal = read_json(path)
            prefix = "calibration/" + digest[:16] + "/"
            ref = archive.copy(path, prefix + "calibration_snapshot.json", source["calibration_snapshot_sha256"])
            for item in cal["artifacts"].values():
                archive.copy(path.parent / item["path"], prefix + item["path"], item["sha256"])
            calibrations[digest] = ref
    captured = {}
    for record in pixels["captured_views"]:
        key = record["scene_id"], record["camera"], record["record_index"]
        capture = {k: deepcopy(v) for k, v in record.items() if k not in ("scene_id", "camera", "record_index")}
        relative = f"provenance/native_pixels/{key[0]}/{key[1]}/{key[2]:07d}.png"
        ref = archive.copy(capture["image"]["path"], relative, capture["image"]["sha256"])
        capture["image"] = archive.future(ref)
        captured[key] = capture
    common = dict(
        assembly,
        status="complete",
        training_published=True,
        suggested_session_splits=splits,
        acceptance=frozen,
        canonical_archive=str(archive.final),
        native_pixel_corrections=str(archive.final / "provenance/native_pixel_corrections.json"),
        sealed_utc=datetime.now(timezone.utc).isoformat(),
    )
    dataset = dict(common, schema="badminton-expanded-dataset-v1")
    dataset_bytes = encode(dataset)
    dataset_binding = future_binding(formal_dataset, dataset_bytes)
    training = dict(
        common,
        schema="badminton-expanded-training-v1",
        source_manifest=dataset_binding["path"],
        source_manifest_sha256=dataset_binding["sha256"],
    )
    training_bytes = encode(training)
    training_binding = future_binding(formal_training, training_bytes)
    final_metadata = dict(path=splits["test"]["metadata"], sha256=splits["test"]["metadata_sha256"])
    evidence = {}
    corrections = []
    for original in pixels["corrections"]:
        correction = deepcopy(original)
        old = correction["evidence"]
        if old["path"] not in evidence:
            proof = read_json(old["path"])
            proof.update(
                metadata=final_metadata,
                training_manifest=training_binding,
                source_recheck=old,
                metadata_byte_copy_bridge=dict(original=pixels["metadata"], archived=final_metadata, bit_exact=True),
            )
            sid = proof["scene_id"]
            scan = proof["lidar_index"]
            proof["captured_pixels"] = {
                cam: captured[sid, cam, v["native"]["record_index"]] for cam, v in proof["captured_pixels"].items()
            }
            ref = archive.write(f"provenance/pixel_evidence/{sid}_{scan:06d}.json", proof)
            evidence[old["path"]] = archive.future(ref)
        correction["evidence"] = evidence[old["path"]]
        key = correction["scene_id"], correction["camera"], correction["record_index"]
        correction["verified_pixel_capture"] = captured[key]
        corrections.append(correction)
    sidecar = dict(
        schema="native-pixel-corrections/v1",
        status="complete",
        created_utc=datetime.now(timezone.utc).isoformat(),
        metadata=final_metadata,
        training_manifest=training_binding,
        full_native_pixel_audit=frozen["pixels"],
        all_person_admissions_unchanged=True,
        corrections=corrections,
        native_pixel_captures=[
            dict(scene_id=sid, camera=cam, record_index=index, verified_pixel_capture=capture)
            for (sid, cam, index), capture in captured.items()
        ],
        rechecked_person_frames=pixels["rechecked_person_frames"],
        reason=pixels["capture_policy"],
    )
    archive.write("provenance/native_pixel_corrections.json", sidecar)
    archive.write("provenance/producer_dataset_manifest.json", dataset_bytes)
    archive.write("provenance/producer_training_manifest.json", training_bytes)
    release = dict(
        schema="lidar2smpl-release-v1",
        release_id=args.release_id,
        status="complete",
        created_utc=datetime.now(timezone.utc).isoformat(),
        description=(
            "Complete old qualified pool plus the admitted native-reviewed sources with conservative identity cuts"
        ),
        total_person_frames=assembly["accepted"],
        splits=archive_splits,
        rollout_sequences=archive_sequences,
        serialization="joblib (torch tensors inside); use joblib.load, not torch.load",
        coordinate_space="lidar_viewer_xyz_m",
        units="metres",
        body_model="neutral SMPL, 24 joints",
        rotation_parameters="axis-angle radians",
        translation="shaped pelvis position in viewer metres",
        point_cloud="256 actual native LiDAR returns per person-frame",
        artifacts=archive.artifacts,
        independence="Independent byte-verified copies; no hardlinks or symlinks",
        scope="Training tensors, metadata, sequence indexes and audit evidence; raw LiDAR/video and model weights remain external",
        old_qualified_frames=assembly["old_accepted"],
        new_qualified_frames=assembly["new_accepted"],
        source_registry=assembly["source_registry"],
        source_calibrations=calibrations,
        producer=producer,
        provenance_paths="Source paths inside metadata remain historical audit references; load tensors from relative split paths",
    )
    (archive.stage / "manifest.json").write_bytes(encode(release))
    verifier_path = archive.base / "tools/verify_dataset.py"
    spec = importlib.util.spec_from_file_location("canonical_archive_verifier", verifier_path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verified = verifier.verify(archive.stage)
    if not verified["passed"]:
        raise ValueError("Canonical archive verification failed")
    (archive.stage / "verification.json").write_bytes(encode(verified))
    for item in frozen.values():
        check_binding(item)
    check_binding(sequence_binding)
    check_binding(catalog_frozen)
    if policy is not None:
        policy.recheck()
        check_binding(assembly["consumer_contract"])
        if contract["canonical_mode"] == "reviewed_no_changes":
            from hmr4d.lidar_mocap.canonical_no_change import validate_no_change_acceptance

            validate_no_change_acceptance(new, frozen["continuity"], read_json(args.browser), **selected_policy)
    # This is our own new staging tree, resolved inside the explicitly named data root.
    if (
        archive.stage.resolve().parent != archive.parent
        or archive.final.resolve().parent != archive.parent
        or archive.final.exists()
        or archive.stage.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise ValueError("Refuse an unexpected or linked archive rename target")
    archive.stage.rename(archive.final)
    archive_manifest = binding(archive.final / "manifest.json")
    catalog["releases"].append(
        dict(
            release_id=args.release_id,
            manifest="releases/" + args.release_id + "/manifest.json",
            manifest_sha256=archive_manifest["sha256"],
            total_person_frames=assembly["accepted"],
            splits={r: e["accepted"] for r, e in splits.items()},
        )
    )
    write_json(catalog_path, catalog)
    receipt = dict(
        schema="badminton-expanded-release-receipt-v1",
        status="complete",
        accepted=assembly["accepted"],
        old_accepted=assembly["old_accepted"],
        new_accepted=assembly["new_accepted"],
        splits={r: e["accepted"] for r, e in splits.items()},
        dataset_manifest=dataset_binding,
        training_manifest=training_binding,
        archive_manifest=archive_manifest,
        archive_verification=binding(archive.final / "verification.json"),
        native_pixel_corrections=binding(archive.final / "provenance/native_pixel_corrections.json"),
        acceptance=frozen,
        source_numeric_values_unchanged=True,
        all_old_rows_retained=True,
        gpu_lease_released=False,
    )
    # Publish support first, then the two formal entries. Consumers still honor the GPU lease.
    for source, target in (
        (archive.final / "provenance/native_pixel_corrections.json", output / "native_pixel_corrections.json"),
        (archive.final / "provenance/producer_dataset_manifest.json", formal_dataset),
        (archive.final / "provenance/producer_training_manifest.json", formal_training),
    ):
        if target.exists():
            raise ValueError("Never replace an existing formal publication")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".publishing")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    check_binding(dataset_binding)
    check_binding(training_binding)
    write_json(output / "release_receipt.json", receipt)
    write_json(
        output / "publication_progress.json",
        dict(
            stage="formal_published",
            accepted=assembly["accepted"],
            archive=str(archive.final),
            gpu_lease_released=False,
        ),
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assembly", "browser", "continuity", "pixels", "sequence-report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("D:/PROJECT/LIDAR/datas/lidar2smpl_badminton"),
        help="Canonical release archive root (default: %(default)s)",
    )
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--release-policy", type=Path, help="Explicit frozen v2 release policy")
    parser.add_argument("--release-policy-sha256", help="Exact SHA256 of the explicit v2 policy")
    args = parser.parse_args()
    if (args.release_policy is None) != (args.release_policy_sha256 is None):
        parser.error("--release-policy and --release-policy-sha256 must be supplied together")
    if not args.assembly.resolve().is_relative_to(PROJ_ROOT / "outputs"):
        parser.error("Select an explicit project assembly output")
    torch.set_num_threads(1)
    result = seal(args)
    print(json.dumps(dict(status=result["status"], accepted=result["accepted"], archive=result["archive_manifest"])))


if __name__ == "__main__":
    main()
