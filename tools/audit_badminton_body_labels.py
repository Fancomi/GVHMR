"""Read-only CPU audit of track/color continuity and lower-limb label support.

Automatic flags are review candidates, not anatomical ground-truth diagnoses.
Published tensors and fitted source files are never rewritten.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.teacher import load_teacher_track, sample_track


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def key(row):
    return row["session_id"], int(row["lidar_index"]), row["motion_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=PROJ_ROOT / "outputs/badminton_rallies_v1")
    parser.add_argument(
        "--union", type=Path, default=PROJ_ROOT / "outputs/badminton_full_plus_new_v1/dataset_manifest.json"
    )
    parser.add_argument("--output-root", type=Path, default=PROJ_ROOT / "outputs/badminton_label_audit_v2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-report", type=Path, help="Reuse hash-verified metrics from the same immutable union")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = args.run_root.resolve()
    output = args.output_root.resolve()
    if not output.is_relative_to((PROJ_ROOT / "outputs").resolve()):
        raise ValueError("Audit output must remain inside workspace outputs")
    output.mkdir(parents=True, exist_ok=True)
    union = read(args.union)
    cached = None
    cache_evidence = None
    if args.reuse_report:
        if args.reuse_report.resolve().parent == output:
            raise ValueError("Reuse requires a different output directory to preserve cache evidence")
        previous_report = read(args.reuse_report)
        if previous_report["source_union_sha256"] != file_sha(args.union):
            raise ValueError("Cached audit uses another source union")
        if file_sha(previous_report["person_frame_records"]) != previous_report["person_frame_records_sha256"]:
            raise ValueError("Cached person-frame metrics changed")
        cached = {}
        with Path(previous_report["person_frame_records"]).open(encoding="utf-8") as handle:
            for line in handle:
                r = json.loads(line)
                cached[(r["source_id"], r["scan"], r["motion_id"])] = r
        cache_evidence = dict(
            report=str(args.reuse_report.resolve()),
            report_sha256=file_sha(args.reuse_report),
            metrics_sha256=previous_report["person_frame_records_sha256"],
        )
    source = read(root / "datasets/dataset_manifest.json")
    registry = read(root / "source_registry.json")
    selected_windows = {}
    for published in union["source_manifests"]:
        publication = read(published["path"])
        body = read(publication["source_dataset_manifest"])
        for sid, entry in union["exports"].items():
            if Path(entry["source_manifest"]).resolve() == Path(published["path"]).resolve():
                selected_windows[sid] = set(body["selected_windows"][sid])
    cal = CalibrationSnapshot.load(root / "calibration/calibration_snapshot.json")
    reg = torch.load(
        PROJ_ROOT / "hmr4d/utils/body_model/smpl_coco17_J_regressor.pt", map_location="cpu", weights_only=True
    ).numpy()[[13, 14, 15, 16]]
    admitted = set()
    for sid, entry in union["exports"].items():
        if file_sha(entry["metadata"]) != entry["metadata_sha256"]:
            raise ValueError("Published metadata changed")
        with Path(entry["metadata"]).open(encoding="utf-8") as handle:
            admitted.update(key(json.loads(line)) for line in handle)
    if len(admitted) != union["accepted"]:
        raise ValueError("Union contains duplicate or missing person scans")
    by_source = {}
    all_color = []
    all_frag = []
    all_swaps = []
    total = Counter()
    started = time.perf_counter()
    per_frame_path = output / "person_frames.jsonl"
    with per_frame_path.open("w", encoding="utf-8") as journal, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for sid in sorted(registry["sources"]):
            directory = Path(source["window_roots"][sid]) / "sessions" / sid / "windows"
            counts = Counter()
            flags = Counter()
            accepted_flags = Counter()
            color_events = []
            fragments = []
            swaps = []
            for quality in sorted(directory.glob("*/quality.jsonl")):
                if quality.parent.name not in selected_windows[sid]:
                    continue
                if not (quality.parent / "fit_report.json").exists():
                    continue
                rows = [json.loads(line) for line in quality.read_text(encoding="utf-8").splitlines()]
                frames = defaultdict(list)
                tracks = defaultdict(list)
                for row in rows:
                    frames[row["lidar_index"]].append(row)
                    tracks[row["motion_id"]].append(row)
                previous = {}
                last_time = None
                for scan, actors in sorted(frames.items()):
                    current = {r["motion_id"]: i % 4 for i, r in enumerate(actors)}
                    if last_time is not None and actors[0]["scan_mid_us"] - last_time <= 150000:
                        changes = [
                            dict(motion_id=k, before=previous[k], after=v, delivered=(sid, scan, k) in admitted)
                            for k, v in current.items()
                            if k in previous and previous[k] != v
                        ]
                        if changes:
                            color_events.append(dict(source_id=sid, scan=scan, changes=changes))
                    previous = current
                    last_time = actors[0]["scan_mid_us"]
                lightweight = []
                for motion, group in tracks.items():
                    group.sort(key=lambda r: r["scan_mid_us"])
                    if cached is not None:
                        lightweight.extend((row, None, None) for row in group)
                        continue
                    teacher = load_teacher_track(quality.parent / "teacher" / group[0]["teacher"])
                    sampled = sample_track(teacher, [r["scan_mid_us"] for r in group])
                    for index, row in enumerate(group):
                        lightweight.append(
                            (row, sampled["kp2d"][index], sampled["smpl_params_incam"]["body_pose"][index])
                        )

                def inspect(job):
                    row, kp, tp = job
                    if cached is not None:
                        result = dict(cached[key(row)])
                        if (
                            Path(result["fit_path"]).resolve() != Path(row["fit_path"]).resolve()
                            or result["body_qa_accepted"] != row["accepted"]
                            or result["delivered"] != (key(row) in admitted)
                        ):
                            raise ValueError("Cached metric source/qualification changed")
                        # SMPL and SMPL-X use different rest bases. Preserve the
                        # raw parameter diagnostic without treating it as a pose flag.
                        if "foot_rotation_from_teacher_deg" in result:
                            result["cross_model_local_parameter_delta_deg"] = result.pop(
                                "foot_rotation_from_teacher_deg"
                            )
                        result["flags"] = [
                            flag for flag in result["flags"] if flag != "ankle_or_foot_rotation_over_45deg_from_teacher"
                        ]
                        return result
                    with np.load(row["fit_path"]) as fit:
                        joints = fit["joints"].copy()
                        vertices = fit["vertices"]
                        body = fit["body_pose"].reshape(23, 3)
                        cloud = fit["source_cloud"]
                        uv, _ = cal.project(reg @ vertices, "cam2", distort=False)
                        near = np.linalg.norm(uv - kp[[13, 14, 15, 16], :2], axis=1)
                        distance, which = cKDTree(cloud).query(joints[[7, 8, 10, 11]], k=1)
                        support = cKDTree(cloud).query_ball_point(joints[[10, 11]], r=0.12, return_length=True)
                    delta = (
                        (
                            Rotation.from_rotvec(body[[6, 7, 9, 10]])
                            * Rotation.from_rotvec(tp.reshape(21, 3)[[6, 7, 9, 10]]).inv()
                        ).magnitude()
                        * 180
                        / np.pi
                    )
                    toe = joints[[10, 11]] - joints[[7, 8]]
                    yaw = np.degrees(np.arctan2(toe[:, 1], toe[:, 0]))
                    spread = abs(float((yaw[0] - yaw[1] + 180) % 360 - 180))
                    reasons = []
                    if ((near[2:] > 20) & (kp[[15, 16], 2] >= 0.5)).any():
                        reasons.append("confident_ankle_reprojection_over_20px")
                    if min(support) == 0:
                        reasons.append("one_or_both_toe_joints_have_no_returns_within_12cm")
                    if spread > 100:
                        reasons.append("left_right_toe_heading_separation_over_100deg")
                    return dict(
                        source_id=sid,
                        scan=row["lidar_index"],
                        time_us=row["scan_mid_us"],
                        motion_id=row["motion_id"],
                        delivered=key(row) in admitted,
                        body_qa_accepted=row["accepted"],
                        pelvis=joints[0].tolist(),
                        ankles=joints[[7, 8]].tolist(),
                        ankle_error_px=near[2:].tolist(),
                        knee_error_px=near[:2].tolist(),
                        ankle_confidence=kp[[15, 16], 2].tolist(),
                        toe_support_12cm=support.tolist(),
                        foot_nearest_return_m=distance.tolist(),
                        cross_model_local_parameter_delta_deg=delta.tolist(),
                        toe_heading_separation_deg=spread,
                        flags=reasons,
                        fit_path=row["fit_path"],
                        surface_mean_mm=row["qa"]["final_p2surface_mean_m"] * 1000,
                    )

                records = list(pool.map(inspect, lightweight))
                for r in records:
                    journal.write(json.dumps(r, allow_nan=False) + "\n")
                    flags.update(r["flags"])
                    counts["fitted_person_frames"] += 1
                    counts["delivered_person_frames"] += r["delivered"]
                    counts["body_qa_rejected_person_frames"] += not r["body_qa_accepted"]
                    if r["delivered"]:
                        accepted_flags.update(r["flags"])
                per_motion = defaultdict(list)
                per_scan = defaultdict(list)
                for r in records:
                    per_motion[r["motion_id"]].append(r)
                    per_scan[r["scan"]].append(r)
                starts = [sorted(v, key=lambda r: r["time_us"])[0] for v in per_motion.values()]
                ends = [sorted(v, key=lambda r: r["time_us"])[-1] for v in per_motion.values()]
                for end in ends:
                    candidates = []
                    for begin in starts:
                        gap = (begin["time_us"] - end["time_us"]) / 1e6
                        distance = float(np.linalg.norm(np.array(begin["pelvis"]) - end["pelvis"]))
                        if end["motion_id"] != begin["motion_id"] and 0 < gap <= 0.25 and distance < 0.5:
                            candidates.append((distance, begin, gap))
                    if candidates:
                        distance, begin, gap = min(candidates, key=lambda x: x[0])
                        fragments.append(
                            dict(
                                source_id=sid,
                                before_scan=end["scan"],
                                after_scan=begin["scan"],
                                before_motion_id=end["motion_id"],
                                after_motion_id=begin["motion_id"],
                                gap_seconds=gap,
                                pelvis_distance_m=distance,
                                delivered_before=end["delivered"],
                                delivered_after=begin["delivered"],
                            )
                        )
                before = None
                for scan, people in sorted(per_scan.items()):
                    if before is not None:
                        old_scan, old = before
                        if people[0]["time_us"] - old[0]["time_us"] <= 150000:
                            old_map = {p["motion_id"]: p for p in old}
                            same = [p for p in people if p["motion_id"] in old_map]
                            for ia, a in enumerate(same):
                                for b in same[ia + 1 :]:
                                    pa, pb = np.array(a["pelvis"]), np.array(b["pelvis"])
                                    oa = np.array(old_map[a["motion_id"]]["pelvis"])
                                    ob = np.array(old_map[b["motion_id"]]["pelvis"])
                                    own = np.linalg.norm(pa - oa) + np.linalg.norm(pb - ob)
                                    cross = np.linalg.norm(pa - ob) + np.linalg.norm(pb - oa)
                                    if own > 1.5 and cross < 0.5 and own - cross > 1:
                                        swaps.append(
                                            dict(
                                                source_id=sid,
                                                before_scan=old_scan,
                                                after_scan=scan,
                                                motion_ids=[a["motion_id"], b["motion_id"]],
                                                own_motion_m=float(own),
                                                cross_motion_m=float(cross),
                                            )
                                        )
                    before = (scan, people)
                journal.flush()
                print(
                    json.dumps(
                        dict(
                            source=sid,
                            window=quality.parent.name,
                            scanned=counts["fitted_person_frames"],
                            delivered=counts["delivered_person_frames"],
                        )
                    ),
                    flush=True,
                )
            counts["viewer_recolor_events"] = len(color_events)
            counts["viewer_affected_color_assignments"] = sum(len(e["changes"]) for e in color_events)
            counts["viewer_recolor_events_involving_delivered"] = sum(
                any(change["delivered"] for change in event["changes"]) for event in color_events
            )
            counts["viewer_delivered_color_assignments"] = sum(
                change["delivered"] for event in color_events for change in event["changes"]
            )
            counts["track_fragment_candidates"] = len(fragments)
            counts["track_fragment_candidates_both_delivered"] = sum(
                event["delivered_before"] and event["delivered_after"] for event in fragments
            )
            counts["track_fragment_candidates_either_delivered"] = sum(
                event["delivered_before"] or event["delivered_after"] for event in fragments
            )
            counts["mutual_body_swap_candidates"] = len(swaps)
            by_source[sid] = dict(
                counts=dict(counts), review_flags=dict(flags), delivered_review_flags=dict(accepted_flags)
            )
            all_color.extend(color_events)
            all_frag.extend(fragments)
            all_swaps.extend(swaps)
            total.update(counts)
            write_json(
                output / "progress.json", dict(state="running", completed_sources=list(by_source), counts=dict(total))
            )
    if total["delivered_person_frames"] != union["accepted"]:
        raise ValueError("Audit did not cover every delivered union label")
    report = dict(
        schema="badminton-body-label-audit-v1",
        state="complete",
        source_union=str(args.union.resolve()),
        source_union_sha256=file_sha(args.union),
        scope="Selected fitted windows of the two immutable source publications; all delivered union labels covered",
        cached_metric_evidence=cache_evidence,
        counts=dict(total),
        sources=by_source,
        viewer_recolor_events=all_color,
        track_fragment_candidates=all_frag,
        mutual_body_swap_candidates=all_swaps,
        person_frame_records=str(per_frame_path),
        person_frame_records_sha256=file_sha(per_frame_path),
        wall_seconds=time.perf_counter() - started,
        limits=[
            "Color reassignments are exact viewer defects. Fragment and swap thresholds are screening candidates, not verified identity truth.",
            "Ankle reprojection uses the same participating 2D detector, not independent anatomy. Foot support flags mark unobserved regions, not proven bad poses.",
            "Raw SMPL/SMPL-X local parameter differences use different rest bases; they do not measure geometric foot rotation and are not pose quality flags.",
            "No calibration, teacher, fitted body or published training value was changed.",
        ],
    )
    write_json(output / "report.json", report)
    print(json.dumps(dict(state="complete", counts=dict(total), wall_seconds=report["wall_seconds"])), flush=True)


if __name__ == "__main__":
    main()
