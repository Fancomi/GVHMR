"""Publish a verified reference-only union of old full data and four new sources.

No training tensors, original metadata, calibration or source manifests change.
"""

import argparse
from pathlib import Path
import json

import joblib
import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.export import FIELDS
from hmr4d.lidar_mocap.pipeline import write_json

SPLITS = dict(
    train=["20261012_154726", "20261012_163452", "20261009_202713", "20261010_192753", "20261012_160054"],
    val=["20261012_161544"],
    test=["20261010_190709"],
)
CONTRACT = ("target_body_model", "gender", "shape_basis", "coordinate_space", "point_count", "transl_convention")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_part(manifest, sid):
    entry = manifest["exports"][sid]
    for field, digest in (("path", "sha256"), ("metadata", "metadata_sha256")):
        if file_sha(entry[field]) != entry[digest]:
            raise ValueError(f"Published source changed: {sid} {field}")
    data = joblib.load(entry["path"])
    count = entry["accepted"]
    for name, shape in FIELDS.items():
        value = data[name]
        if value.shape != (count, *shape) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Invalid source training field: {sid} {name}")
    keys = []
    with Path(entry["metadata"]).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if row["session_id"] != sid or row["motion_id"] != data["motion_id"][index]:
                raise ValueError(f"Person/scene/order mismatch: {sid} row {index}")
            keys.append((sid, int(row["lidar_index"]), row["motion_id"]))
    if len(keys) != count or len(set(keys)) != count or len(data["motion_id"]) != count:
        raise ValueError(f"Duplicate or missing person-scan rows: {sid}")
    return data, keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-manifest",
        type=Path,
        default=PROJ_ROOT / "outputs/badminton_dualcam_paired_full_v2/datasets/dataset_manifest.json",
    )
    parser.add_argument(
        "--rally-manifest",
        type=Path,
        default=PROJ_ROOT / "outputs/badminton_rallies_v1_paired/datasets/dataset_manifest.json",
    )
    parser.add_argument("--output-root", type=Path, default=PROJ_ROOT / "outputs/badminton_full_plus_new_v1")
    args = parser.parse_args()
    torch.set_num_threads(4)
    old_path, rally_path, output = (
        args.old_manifest.resolve(),
        args.rally_manifest.resolve(),
        args.output_root.resolve(),
    )
    if not output.is_relative_to(PROJ_ROOT / "outputs") or any(
        output == p.parent or output.is_relative_to(p.parent) for p in (old_path, rally_path)
    ):
        raise ValueError("Reference union needs a separate output directory")
    published = output / "dataset_manifest.json"
    if published.exists():
        raise ValueError("Reference union already published; retain it or choose a new output directory")
    old, rally = read(old_path), read(rally_path)
    identities = [dict(path=str(p), sha256=file_sha(p)) for p in (old_path, rally_path)]
    if old.get("status") != "complete" or rally.get("status") != "complete":
        raise ValueError("Both immutable source publications must be complete")
    if old["calibration_digest"] != rally["calibration_digest"] or any(old[k] != rally[k] for k in CONTRACT):
        raise ValueError("Source coordinate/SMPL/calibration contracts differ")
    old_ids, rally_ids = set(old["exports"]), set(rally["exports"])
    assigned = [sid for group in SPLITS.values() for sid in group]
    if len(assigned) != len(set(assigned)) or set(assigned) != old_ids | rally_ids:
        raise ValueError("Split sessions overlap or do not cover the exact source union")
    exports, checks, overlap = {}, {}, {}
    all_keys = set()
    for sid in sorted(set(assigned)):
        origin, origin_path = (old, old_path) if sid in old_ids else (rally, rally_path)
        data, keys = load_part(origin, sid)
        if all_keys.intersection(keys):
            raise ValueError("Duplicate selected person scans across source parts")
        all_keys.update(keys)
        entry = origin["exports"][sid]
        exports[sid] = {
            **entry,
            "path": str(Path(entry["path"]).resolve()),
            "metadata": str(Path(entry["metadata"]).resolve()),
            "source_manifest": str(origin_path),
            "origin": "old_full" if sid in old_ids else "new_four_rally_sources",
        }
        checks[sid] = dict(
            accepted=len(keys),
            unique_person_scan_keys=True,
            tensor_metadata_order=True,
            finite_float32_contract=True,
            referenced_original_tensors=True,
        )
        if sid in old_ids & rally_ids:
            subset, subset_keys = load_part(rally, sid)
            index = {key: i for i, key in enumerate(keys)}
            if not set(subset_keys).issubset(index):
                raise ValueError(f"Rally overlap is not a subset of original full data: {sid}")
            take = torch.tensor([index[key] for key in subset_keys], dtype=torch.long)
            for name in FIELDS:
                if not torch.equal(data[name].index_select(0, take), subset[name]):
                    raise ValueError(f"Overlapping rally tensor differs from old full data: {sid} {name}")
            overlap[sid] = dict(
                kept_old_full=len(keys),
                omitted_rally_duplicates=len(subset_keys),
                every_rally_key_in_old_full=True,
                all_five_tensor_fields_exact=True,
                omitted_tensor_path=rally["exports"][sid]["path"],
                omitted_tensor_sha256=rally["exports"][sid]["sha256"],
                omitted_metadata_path=rally["exports"][sid]["metadata"],
                omitted_metadata_sha256=rally["exports"][sid]["metadata_sha256"],
            )
            del subset, take
        del data
        print(f"Verified {sid}: {len(keys)} original person frames", flush=True)
    for item in identities:
        if file_sha(item["path"]) != item["sha256"]:
            raise ValueError("Source manifest changed while verifying union")
    proof = dict(
        schema="badminton-reference-union-verification-v1",
        passed=True,
        sources=checks,
        overlap=overlap,
        selected_unique_person_frames=len(all_keys),
        old_full_person_frames=sum(exports[sid]["accepted"] for sid in old_ids),
        new_person_frames=sum(exports[sid]["accepted"] for sid in rally_ids - old_ids),
        omitted_duplicate_rally_person_frames=sum(r["omitted_rally_duplicates"] for r in overlap.values()),
        split_sessions_disjoint=True,
        duplicate_selected_person_scans=0,
    )
    write_json(output / "overlap_verification.json", proof)
    manifest = dict(
        schema="badminton-ptv3-reference-union-v1",
        status="complete",
        scope="old_full_three_plus_new_four",
        accepted=len(all_keys),
        exports=exports,
        source_manifests=identities,
        suggested_session_splits={
            name: dict(
                sessions=sids,
                accepted=sum(exports[sid]["accepted"] for sid in sids),
                parts=[
                    dict(
                        session_id=sid,
                        path=exports[sid]["path"],
                        sha256=exports[sid]["sha256"],
                        metadata=exports[sid]["metadata"],
                        metadata_sha256=exports[sid]["metadata_sha256"],
                        accepted=exports[sid]["accepted"],
                    )
                    for sid in sids
                ],
            )
            for name, sids in SPLITS.items()
        },
        calibration_digest=old["calibration_digest"],
        **{key: old[key] for key in CONTRACT},
        original_training_values_unchanged=True,
        reference_only=True,
        overlap_policy="Use old full publications for every overlapping session; exclude all three rally versions",
        verification=str(output / "overlap_verification.json"),
        verification_sha256=file_sha(output / "overlap_verification.json"),
        requested_training_epochs=100,
        protocol="Consumer frozen teacher/cache protocol; this CLI performs zero training",
        split_limit="Session-disjoint only; cross-session physical person identities are unverified",
        test_limit="New 190709 session excluded from train/val; replaces the previously trained-on 154726 test session",
        metric_limit="Fitted pseudo-label targets, not independent anatomical ground truth",
    )
    write_json(published, manifest)
    receipt = dict(
        schema="badminton-reference-union-receipt-v1",
        accepted=manifest["accepted"],
        dataset_manifest=dict(path=str(published), sha256=file_sha(published)),
        verification=dict(path=str(output / "overlap_verification.json"), sha256=manifest["verification_sha256"]),
        splits={
            name: dict(sessions=s["sessions"], accepted=s["accepted"])
            for name, s in manifest["suggested_session_splits"].items()
        },
        copied_or_reencoded_tensor_files=0,
    )
    write_json(output / "receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
