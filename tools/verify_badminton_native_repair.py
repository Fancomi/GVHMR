"""Verify native observations, all accepted point members and rebuilt SMPL per window."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import joblib
import numpy as np
import smplx
import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import load_run_session, read_json, read_rows
from hmr4d.lidar_mocap.teacher import load_teacher_track
from hmr4d.utils.body_model.contracts import forward_smpl_at_pelvis


def verify_teacher(source, target, decision, mapping):
    before = read_json(source / "teacher/manifest.json")
    after = read_json(target / "teacher/manifest.json")
    for key in ["frame_rows", "camera", "video"]:
        if before["provenance"][key] != after["provenance"][key]:
            raise ValueError("Teacher native source changed")
    rows = before["provenance"]["frame_rows"]
    positions = {r["record_index"]: i for i, r in enumerate(rows)}
    detections = torch.load(source / "teacher/detections.pt", map_location="cpu", weights_only=False)["detections"]
    raw = defaultdict(set)
    for pos, det, old, new in mapping:
        raw[int(pos), np.asarray(detections[pos][det]["bbx_xyxy"], np.float32).tobytes()].add(int(old))
    old_pixels = {}
    counts = Counter(teacher_segments=len(after["tracks"]))
    for entry in before["tracks"]:
        track = load_teacher_track(source / "teacher" / entry["path"])
        for record, box, kp in zip(track["record_index"], track["bbx_xyxy"], track["kp2d"]):
            old_pixels[int(record), np.asarray(box, np.float32).tobytes()] = np.asarray(kp)
    seen = set()
    aliases = defaultdict(set)
    for entry in after["tracks"]:
        track = load_teacher_track(target / "teacher" / entry["path"])
        if track["provenance_hash"] != after["provenance_hash"]:
            raise ValueError("Teacher provenance changed")
        counts["reused_temporal_contexts" if track.get("temporal_reuse_from") else "reinferred_temporal_contexts"] += 1
        for record, common, pts, box, kp, original in zip(
            track["record_index"],
            track["common_time_us"],
            track["media_pts_us"],
            track["bbx_xyxy"],
            track["kp2d"],
            track["original_track_id"],
        ):
            record = int(record)
            pos = positions[record]
            original = int(original)
            key = record, np.asarray(box, np.float32).tobytes()
            if original not in raw[pos, key[1]]:
                raise ValueError("Crop is not from its recorded original detector identity")
            if int(common) != rows[pos]["common_time_us"] or int(pts) != rows[pos]["media_pts_us"]:
                raise ValueError("Native timestamp changed")
            identity = entry["track_id"]
            aliases[original].add(identity)
            if (identity, pos) in seen:
                raise ValueError("Two observations of one person at one native timestamp")
            seen.add((identity, pos))
            for span in (decision or {}).get("drop_spans", []):
                if original == span["track_id"] and span["begin"] <= pos <= span["end"]:
                    raise ValueError("Contaminated crop retained")
            if key in old_pixels:
                if not np.array_equal(np.asarray(kp), old_pixels[key]):
                    raise ValueError("Frozen 2D observation changed")
                counts["exact_native_keypoint_reuses"] += 1
            else:
                counts["new_retained_real_crops"] += 1
            counts["observed_person_frames"] += 1
    for a, b in (decision or {}).get("cannot_link", []):
        if aliases[a] & aliases[b]:
            raise ValueError("Two witnessed people share one repaired identity")
    if decision:
        counts["mixed_observations_removed"] = len(after["identity_repair"]["removed_observations"])
        counts["duplicate_observations_removed"] = len(after["identity_repair"]["duplicate_observations"])
    return dict(counts)


def verify_body(window, session, model, calibration_digest):
    report = read_json(window / "fit_report.json")
    if any(file_sha(window / p) != sha for p, sha in report["artifacts_sha256"].items()):
        raise ValueError("Completed fit artifact changed")
    data = joblib.load(window / "ptv3.pth")
    rows = read_rows(window / "ptv3.metadata.jsonl")
    if len(rows) != len(data["pc"]) or len(rows) != report["accepted"]:
        raise ValueError("Export sample count mismatch")
    by_scan = defaultdict(list)
    maximums = Counter()
    pose_files = {}
    for i, row in enumerate(rows):
        if row["calibration_digest"] != calibration_digest:
            raise ValueError("Camera digest changed")
        by_scan[row["lidar_index"]].append((i, row))
        for info in (row.get("native_pose_evidence") or {}).get("files", {}).values():
            pose_files[info["path"]] = info["sha256"]
    if any(file_sha(Path(p)) != sha for p, sha in pose_files.items()):
        raise ValueError("Native pose evidence changed")
    pairs = session.pairs()
    for scan, entries in by_scan.items():
        cloud = session.read_cloud(pairs[scan])
        for i, row in entries:
            if row["scan_mid_us"] != pairs[scan]["scan_mid_us"] or row["cameras"] != pairs[scan]["cameras"]:
                raise ValueError("Export scan pairing changed")
            error = float(
                np.max(np.abs(np.asarray(data["pc"][i]) - cloud[np.asarray(row["source_segment_indices"], np.int64)]))
            )
            maximums["source_point_error_m"] = max(maximums["source_point_error_m"], error)
    for start in range(0, len(rows), 32):
        end = min(start + 32, len(rows))
        batch = lambda key: torch.stack([torch.as_tensor(v) for v in data[key][start:end]])
        with torch.no_grad():
            vertices, joints = forward_smpl_at_pelvis(
                model, batch("root_pose"), batch("body_pose"), batch("shape"), batch("transl")
            )
        maximums["pelvis_error_m"] = max(
            maximums["pelvis_error_m"], float((joints[:, 0] - batch("transl").reshape(-1, 3)).abs().max())
        )
        for offset, row in enumerate(rows[start:end]):
            with np.load(row["fit_path"]) as fit:
                maximums["rebuilt_mesh_error_m"] = max(
                    maximums["rebuilt_mesh_error_m"], float(np.max(np.abs(vertices[offset].numpy() - fit["vertices"])))
                )
    if (
        maximums["source_point_error_m"] > 1e-6
        or maximums["pelvis_error_m"] > 1e-6
        or maximums["rebuilt_mesh_error_m"] > 1e-4
    ):
        raise ValueError(dict(maximums))
    return dict(
        accepted_checked=len(rows),
        native_scans=len(by_scan),
        native_pose_files=len(pose_files),
        maximums=dict(maximums),
        report_sha256=file_sha(window / "fit_report.json"),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2")
    p.add_argument("--wait", action="store_true")
    args = p.parse_args()
    root = args.root
    labels = root / "final_labels"
    torch.set_num_threads(2)
    plan = read_json(labels / "repair_plan.json")
    decisions = read_json(root / "identity_evidence/identity_decisions.json")
    if file_sha(plan["source_union"]) != plan["source_union_sha256"]:
        raise ValueError("Original published union changed")
    replay = read_json(PROJ_ROOT / "outputs/badminton_body_repair_v1/tracking_final/tracking_replay.json")
    maps = {(w["source_id"], w["window_id"]): w["mapping"] for w in replay["windows"]}
    cal = CalibrationSnapshot.load(labels / "calibration/calibration_snapshot.json")
    model = smplx.create(PROJ_ROOT / "inputs/checkpoints/body_models", model_type="smpl", gender="neutral").eval()
    out = labels / "native_verification.json"
    result = read_json(out) if out.exists() else dict(windows={}, passed=False, state="partial")
    for key in list(result["windows"]):
        sid, window = key.split(":")
        report = labels / "sessions" / sid / "windows" / window / "fit_report.json"
        if file_sha(report) != result["windows"][key]["body"]["report_sha256"]:
            result["windows"].pop(key)
    if len(result["windows"]) < len(plan["jobs"]):
        result.update(passed=False, state="partial")
    sessions = {}
    started = time.perf_counter()
    while len(result["windows"]) < len(plan["jobs"]):
        changed = False
        for job in plan["jobs"]:
            sid = job["source_id"]
            source = Path(job["source_window"])
            key = sid + ":" + source.name
            if key in result["windows"]:
                continue
            window = labels / "sessions" / sid / "windows" / source.name
            if not (window / "fit_report.json").exists():
                continue
            if sid not in sessions:
                sessions[sid] = load_run_session(labels, cal, sid)
            teacher = verify_teacher(
                source, window, decisions["windows"].get(key), np.load(maps[sid, source.name])["observations"]
            )
            body = verify_body(window, sessions[sid], model, cal.data["snapshot_digest"])
            result["windows"][key] = dict(teacher=teacher, body=body)
            changed = True
            result.update(
                verified_windows=len(result["windows"]),
                total_windows=len(plan["jobs"]),
                accepted_checked=sum(w["body"]["accepted_checked"] for w in result["windows"].values()),
                source_union_sha256=file_sha(plan["source_union"]),
                calibration_digest=cal.data["snapshot_digest"],
            )
            write_json(out, result)
            print(
                dict(
                    verified_windows=len(result["windows"]),
                    accepted_checked=result["accepted_checked"],
                    wall_seconds=time.perf_counter() - started,
                ),
                flush=True,
            )
        if not args.wait:
            break
        if not changed:
            time.sleep(5)
    result.update(
        passed=len(result["windows"]) == len(plan["jobs"]),
        state="complete" if len(result["windows"]) == len(plan["jobs"]) else "partial",
        limits=[
            "All accepted point members and SMPL reconstruction are checked. Native identity evidence is reviewed local continuity, not athlete ground truth."
        ],
    )
    write_json(out, result)


if __name__ == "__main__":
    main()
