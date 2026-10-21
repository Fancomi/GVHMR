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


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def future_binding(path, payload):
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(payload).hexdigest())


def validate_acceptance(assembly_path, browser_path, continuity_path, pixels_path):
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
    for item in assembly["source_publications"].values():
        check_binding(item)
    if (
        browser.get("passed") is not True
        or browser.get("clips") != 10
        or browser.get("minimum_clip_seconds", 0) < 5
        or browser.get("page_errors")
    ):
        raise ValueError("Ten real continuous native browser clips must pass")
    if browser.get("canonical_publication") != assembly["source_publications"]["new"]:
        raise ValueError("Browser acceptance uses another canonical publication")
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


def seal(args):
    frozen, assembly, pixels = validate_acceptance(args.assembly, args.browser, args.continuity, args.pixels)
    output = Path(args.assembly).resolve().parent
    formal_dataset = output / "datasets/dataset_manifest.json"
    formal_training = output / "training_manifest.json"
    if formal_dataset.exists() or formal_training.exists() or (output / "release_receipt.json").exists():
        raise ValueError("A formal publication already exists; never rewrite it")
    sequence_binding = binding(args.sequence_report)
    sequences = read_json(args.sequence_report)
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
    parent_receipt = read_json(contract["bindings"]["parent_receipt"]["path"])
    bridge = parent_receipt["model_bridge"]
    archive.copy(bridge["path"], "provenance/model_bridge.json", bridge["sha256"])
    for name, item in frozen.items():
        archive.copy(item["path"], "provenance/" + name + ".json", item["sha256"])
        value = read_json(item["path"])
        if value.get("producer"):
            archive.copy(value["producer"]["path"], "provenance/" + name + "_producer.py", value["producer"]["sha256"])
    for name, item in assembly["source_publications"].items():
        archive.copy(item["path"], "provenance/" + name + "_source_publication.json", item["sha256"])
    new = read_json(assembly["source_publications"]["new"]["path"])
    review = new["canonical_identity_revision"]["reviewed_boundaries"]
    archive.copy(review["path"], "provenance/reviewed_track_boundaries.json", review["sha256"])
    native_contract = read_json(args.continuity)["aggregate_native_contract"]
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
        description="Complete old qualified pool plus native-reviewed 1015 data with conservative canonical identity cuts",
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
    args = parser.parse_args()
    if not args.assembly.resolve().is_relative_to(PROJ_ROOT / "outputs"):
        parser.error("Select an explicit project assembly output")
    torch.set_num_threads(1)
    result = seal(args)
    print(json.dumps(dict(status=result["status"], accepted=result["accepted"], archive=result["archive_manifest"])))


if __name__ == "__main__":
    main()
