"""Seal repaired-label lineage and a unique fixed-size training view with real scan sequences."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.dual_view import file_sha, verify_export
from hmr4d.lidar_mocap.export import FIELDS, export_joblib
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sample_key(row):
    return row["session_id"] + "|" + row["motion_id"] + "|" + str(row["lidar_index"])


def real_sequences(rows):
    tracks = defaultdict(list)
    for i, row in enumerate(rows):
        tracks[row["motion_id"]].append((int(row["lidar_index"]), i, row["scan_mid_us"]))
    sequences = []
    for motion, frames in sorted(tracks.items()):
        current = []
        for item in sorted(frames):
            if current and (item[0] != current[-1][0] + 1 or not 0 < item[2] - current[-1][2] <= 150000):
                sequences.append(dict(motion_id=motion, indices=[v[1] for v in current], scans=[v[0] for v in current]))
                current = []
            current.append(item)
        if current:
            sequences.append(dict(motion_id=motion, indices=[v[1] for v in current], scans=[v[0] for v in current]))
    return sequences


def verify_repaired_aggregate(repair, source, output_root):
    """Join every aggregate row to the window whose native contract was verified."""
    contracts_path = repair / "final_labels/native_verification.json"
    contracts = read_json(contracts_path)
    reports, counts = {}, {}
    for sid, entry in source["exports"].items():
        aggregate = joblib.load(entry["path"])
        rows = read_rows(entry["metadata"])
        lookup = {sample_key(row): i for i, row in enumerate(rows)}
        if len(lookup) != len(rows):
            raise ValueError("Duplicate source aggregate person-scan")
        found = set()
        for window in source["selected_windows"][sid]:
            directory = repair / "final_labels/sessions" / sid / "windows" / window
            key = sid + ":" + window
            report_path = directory / "fit_report.json"
            if file_sha(report_path) != contracts["windows"][key]["body"]["report_sha256"]:
                raise ValueError("Native-verified window fit report changed")
            report = read_json(report_path)
            reports[key] = report
            for name in ("ptv3.pth", "ptv3.metadata.jsonl"):
                if file_sha(directory / name) != report["artifacts_sha256"][name]:
                    raise ValueError("Native-verified window tensor/metadata changed")
            data = joblib.load(directory / "ptv3.pth")
            local = read_rows(directory / "ptv3.metadata.jsonl")
            indices = [lookup[sample_key(row)] for row in local]
            if found.intersection(indices):
                raise ValueError("Window overlap in source aggregate")
            found.update(indices)
            for field in FIELDS:
                if not torch.equal(aggregate[field][indices], data[field]):
                    raise ValueError("Source aggregate differs from native-verified window: " + field)
            for i, row in zip(indices, local):
                for field in ("source_segment_indices", "fit_path", "scan_mid_us", "cameras", "teacher_sampling"):
                    if rows[i][field] != row[field]:
                        raise ValueError("Aggregate source membership/identity metadata changed")
        if found != set(range(len(rows))):
            raise ValueError("Some aggregate samples have no native-verified source window")
        counts[sid] = len(rows)
        print(json.dumps(dict(native_aggregate_exact=sid, samples=len(rows))), flush=True)
    path = output_root / "aggregate_source_contract.json"
    write_json(
        path,
        dict(
            passed=True,
            checked=sum(counts.values()),
            sessions=counts,
            native_verification=str(contracts_path),
            native_verification_sha256=file_sha(contracts_path),
            all_point_and_label_tensors_equal_native_verified_windows=True,
        ),
    )
    return path, reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2_paired")
    parser.add_argument("--repair-root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2")
    parser.add_argument("--train-count", type=int, default=62414)
    args = parser.parse_args()
    if args.train_count < 1:
        parser.error("--train-count must be positive")
    torch.set_num_threads(4)
    root, repair = args.root.resolve(), args.repair_root.resolve()
    output = root / "training_manifest.json"
    if output.exists():
        raise ValueError("Training release is immutable; choose another publication root")
    paired_path = root / "datasets/dataset_manifest.json"
    paired = read_json(paired_path)
    verified = verify_export(root)
    complete = read_json(repair / "completion_receipt.json")
    e2e = read_json(repair / "end_to_end_receipt.json")
    if file_sha(repair / "completion_receipt.json") != e2e["sha256"] or complete["state"] != "complete":
        raise ValueError("Source repaired run is not sealed")
    source = read_json(paired["source_dataset_manifest"])
    if file_sha(paired["source_dataset_manifest"]) != paired["source_dataset_manifest_sha256"]:
        raise ValueError("Source repaired dataset changed")
    original = read_json(repair / "final_labels/repair_plan.json")
    if file_sha(original["source_union"]) != complete["source_union_sha256"]:
        raise ValueError("Original union changed")
    aggregate_contract, fit_reports = verify_repaired_aggregate(repair, source, root)
    observer = read_json(root / "observations/manifest.json")
    producer_sources = {}
    for relative, expected in (
        ("hmr4d/lidar_mocap/repaired_pairing.py", observer["identity"]["reassociation_sha256"]),
        ("tools/publish_badminton_repaired_pairs.py", observer["identity"]["observer_sha256"]),
        ("tools/seal_badminton_repaired_release.py", file_sha(__file__)),
    ):
        current = PROJ_ROOT / relative
        if file_sha(current) != expected:
            raise ValueError("Executed producer source changed before archival")
        snapshot = root / "provenance/source" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists() and file_sha(snapshot) != expected:
            raise ValueError("Producer source archive changed")
        if not snapshot.exists():
            snapshot.write_bytes(current.read_bytes())
        producer_sources[relative] = dict(path=str(snapshot), sha256=expected)
    train = paired["suggested_session_splits"]["train"]
    if train["accepted"] < args.train_count:
        write_json(
            root / "insufficient_training_count.json",
            dict(
                state="insufficient_unique_admitted_samples",
                requested=args.train_count,
                available=train["accepted"],
                split_counts={k: v["accepted"] for k, v in paired["suggested_session_splits"].items()},
            ),
        )
        raise ValueError("Fewer uniquely admitted training samples than requested; no padding or duplication")
    data = joblib.load(train["path"])
    rows = read_rows(train["metadata"])
    keys = [sample_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate admitted training person-scan")
    chosen = rows[: args.train_count]
    samples = [
        dict({name: data[name][i].numpy() for name in FIELDS}, motion_id=data["motion_id"][i], metadata=row)
        for i, row in enumerate(chosen)
    ]
    subset = export_joblib(samples, root / "datasets" / f"train{args.train_count}.pth")
    subset.update(
        sessions=train["sessions"], sha256=file_sha(subset["path"]), metadata_sha256=file_sha(subset["metadata"])
    )
    check = joblib.load(subset["path"])
    if check["motion_id"] != data["motion_id"][: args.train_count] or any(
        not torch.equal(check[name], data[name][: args.train_count]) for name in FIELDS
    ):
        raise ValueError("Fixed training subset changed actual admitted values")
    del samples, data, check, rows
    splits = dict(paired["suggested_session_splits"], train=subset)
    sequences = {}
    split_membership = {}
    for split, entry in splits.items():
        split_rows = chosen if split == "train" else read_rows(entry["metadata"])
        keys = [sample_key(row) for row in split_rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate real person-scan in split")
        for key in keys:
            if key in split_membership:
                raise ValueError("Person-scan leakage between splits")
            split_membership[key] = split
        chunks = real_sequences(split_rows)
        path = root / f"{split}_sequences.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(json.dumps(chunk) + "\n")
        assert sorted(i for chunk in chunks for i in chunk["indices"]) == list(range(len(split_rows)))
        sequences[split] = dict(
            path=str(path),
            sha256=file_sha(path),
            sequences=len(chunks),
            samples=len(split_rows),
            maximum_length=max(len(c["indices"]) for c in chunks),
        )
    del chosen
    mapping_path = root / "identity_point_label_mapping.jsonl"
    tracks = {}
    identities = {}
    leg_cases = []
    identity_changes = defaultdict(int)
    with mapping_path.open("w", encoding="utf-8") as handle:
        for sid, entry in paired["exports"].items():
            data = joblib.load(entry["path"])
            for index, row in enumerate(read_rows(entry["metadata"])):
                track_path = Path(row["fit_path"]).parent.parent / "teacher" / row["teacher"]
                key = str(track_path)
                if key not in tracks:
                    track = torch.load(track_path, map_location="cpu", weights_only=False)
                    ids = track.get("original_track_id")
                    if ids is None:
                        teacher = read_json(track_path.parent / "manifest.json")
                        info = next(item for item in teacher["tracks"] if item["path"] == track_path.name)
                        originals = info.get("source_tracklets", [int(track["track_id"])])
                        if len(originals) != 1:
                            raise ValueError("Merged teacher context lacks per-observation original identity")
                        ids = torch.full_like(track["record_index"], int(originals[0]))
                    if len(ids) != len(track["record_index"]):
                        raise ValueError("Teacher original identities and native observations differ in length")
                    tracks[key] = dict(zip(map(int, track["record_index"]), map(int, ids)))
                    identities[key] = dict(path=key, sha256=file_sha(track_path))
                    del track
                native = row["teacher_sampling"]["record_index"]
                old_id = tracks[key][native]
                fit_shas = {view["observation"]["fit_sha256"] for view in row["dualcam"]["views"].values()}
                if len(fit_shas) != 1:
                    raise ValueError("Two cameras disagree about the repaired mesh")
                window_key = sid + ":" + row["window_id"]
                fit_name = "fits/" + Path(row["fit_path"]).name
                declared = fit_reports[window_key]["artifacts_sha256"]
                expected_fit_sha = declared.get(fit_name, declared.get(fit_name.replace("/", "\\")))
                if next(iter(fit_shas)) != expected_fit_sha:
                    raise ValueError("Admitted fit differs from the mesh whose native contract was verified")
                label = hashlib.sha256(
                    b"".join(
                        data[name][index].numpy().tobytes() for name in ("root_pose", "body_pose", "shape", "transl")
                    )
                ).hexdigest()
                record = dict(
                    sample_key=sample_key(row),
                    session_id=sid,
                    publication_tensor_index=index,
                    motion_id=row["motion_id"],
                    lidar_index=row["lidar_index"],
                    canonical_track_id=int(row["track_id"]),
                    segment_id=int(row["segment_id"]),
                    original_detector_track_id_at_teacher_observation=old_id,
                    teacher_record_index=native,
                    teacher_artifact=identities[key],
                    fit_path=row["fit_path"],
                    fit_sha256=next(iter(fit_shas)),
                    source_tensor_path=row["source_tensor_path"],
                    source_tensor_index=row["source_tensor_index"],
                    original_point_members_sha256=hashlib.sha256(
                        np.asarray(row["source_segment_indices"], dtype="<i8").tobytes()
                    ).hexdigest(),
                    pc_sha256=hashlib.sha256(data["pc"][index].numpy().tobytes()).hexdigest(),
                    native_smpl_parameters_sha256=label,
                    post_leg_refinement=row.get("post_leg_refinement"),
                )
                split = split_membership.get(record["sample_key"])
                if old_id != record["canonical_track_id"]:
                    identity_changes[split or "outside_fixed_training_view"] += 1
                if row.get("post_leg_refinement"):
                    leg_cases.append(
                        dict(
                            sample_key=record["sample_key"],
                            split=split,
                            publication_tensor_index=index,
                            fit_sha256=record["fit_sha256"],
                        )
                    )
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(dict(lineage_source=sid, samples=entry["accepted"])), flush=True)
    manifest = dict(
        schema="badminton-repaired-fixed-training-v1",
        status="complete",
        created_utc=datetime.now(timezone.utc).isoformat(),
        accepted=sum(v["accepted"] for v in splits.values()),
        suggested_session_splits=splits,
        calibration_digest=paired["calibration_digest"],
        source_manifest=str(paired_path),
        source_manifest_sha256=file_sha(paired_path),
        source_completion_receipt=str(repair / "completion_receipt.json"),
        source_completion_sha256=e2e["sha256"],
        source_native_verification=str(repair / "final_labels/native_verification.json"),
        source_native_verification_sha256=file_sha(repair / "final_labels/native_verification.json"),
        aggregate_source_contract=str(aggregate_contract),
        aggregate_source_contract_sha256=file_sha(aggregate_contract),
        identity_decisions=str(repair / "identity_evidence/identity_decisions.json"),
        identity_decisions_sha256=file_sha(repair / "identity_evidence/identity_decisions.json"),
        identity_point_label_mapping=str(mapping_path),
        identity_point_label_mapping_sha256=file_sha(mapping_path),
        original_union=str(original["source_union"]),
        original_union_sha256=complete["source_union_sha256"],
        target_body_model="neutral_smpl_24_joints",
        gender="neutral",
        point_count=256,
        coordinate_space="lidar_viewer_xyz_m",
        transl_convention="shaped_pelvis_position",
        qualification_policy="bilateral_detection",
        sequences=sequences,
        producer_source_snapshots=producer_sources,
        admitted_leg_refinement_cases=leg_cases,
        reassigned_identity_samples=dict(identity_changes),
        selection=dict(
            rule="Deterministic prefix of the admitted train concatenation in original session order; no duplicated rows or missing-scan stitching",
            full_admitted_train=train["accepted"],
            selected_train=args.train_count,
        ),
        source_run=str(repair / "final_labels"),
        calibration_snapshot=str(repair / "final_labels/calibration/calibration_snapshot.json"),
        calibration_snapshot_sha256=file_sha(repair / "final_labels/calibration/calibration_snapshot.json"),
        source_registry=str(repair / "final_labels/source_registry.json"),
        source_registry_sha256=file_sha(repair / "final_labels/source_registry.json"),
    )
    write_json(output, manifest)
    receipt = dict(
        state="complete",
        training_manifest=str(output),
        training_manifest_sha256=file_sha(output),
        full_paired_manifest=str(paired_path),
        full_paired_manifest_sha256=file_sha(paired_path),
        counts={key: value["accepted"] for key, value in splits.items()},
        full_paired_count=verified["accepted"],
        full_admitted_train=train["accepted"],
        unique_training_samples=args.train_count,
        source_refined_leg_frames=complete["native_leg_refinement"]["updated_frames"],
        admitted_leg_refinement_cases=leg_cases,
        reassigned_identity_samples=dict(identity_changes),
        source_point_and_native_smpl_contract="Every source accepted label verified; delivery is an exact tensor subset",
        no_original_delivery_changes=True,
        no_duplicate_padding=True,
        no_missing_scan_stitching=True,
    )
    write_json(root / "release_receipt.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
