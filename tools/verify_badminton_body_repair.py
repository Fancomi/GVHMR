"""Independently verify every repaired teacher observation against frozen detections.

Uses the completed CPU replay's IDs as a regression reference, not identity truth.
Also proves that previously available native 2D observations are reused exactly.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json
from hmr4d.lidar_mocap.teacher import load_teacher_track


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v1/labels")
    parser.add_argument(
        "--replay",
        type=Path,
        default=PROJ_ROOT / "outputs/badminton_body_repair_v1/tracking_final/tracking_replay.json",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    plan = read_json(args.root / "repair_plan.json")
    receipt = read_json(args.root / "repair_teacher_receipt.json")
    replay = read_json(args.replay)
    if receipt["state"] != "complete" or receipt["windows"] != len(plan["jobs"]):
        raise ValueError("Teacher rebuild is incomplete")
    if (
        file_sha(plan["source_union"]) != replay["source_union_sha256"]
        or plan["source_union_sha256"] != replay["source_union_sha256"]
    ):
        raise ValueError("Frozen original union or replay binding changed")
    windows = {(w["source_id"], w["window_id"]): w for w in replay["windows"]}
    total, by_source = Counter(), {}
    for job in plan["jobs"]:
        sid, source = job["source_id"], Path(job["source_window"])
        target = args.root / "sessions" / sid / "windows" / source.name / "teacher"
        before, after = read_json(source / "teacher/manifest.json"), read_json(target / "manifest.json")
        if (
            before["provenance"]["camera"] != after["provenance"]["camera"]
            or before["provenance"]["frame_rows"] != after["provenance"]["frame_rows"]
        ):
            raise ValueError("Camera/native source rows changed")
        frozen = before["provenance"]["frame_rows"]
        positions = {row["record_index"]: i for i, row in enumerate(frozen)}
        detections = torch.load(source / "teacher/detections.pt", map_location="cpu", weights_only=False)["detections"]
        mapping = np.load(windows[sid, source.name]["mapping"])["observations"]
        expected = {}
        for pos, det, _, track in mapping:
            key = int(pos), np.asarray(detections[pos][det]["bbx_xyxy"], dtype=np.float32).tobytes()
            expected.setdefault(key, set()).add(int(track))
        old_pixels = {}
        for info in before["tracks"]:
            value = load_teacher_track(source / "teacher" / info["path"])
            for record, box, kp in zip(value["record_index"], value["bbx_xyxy"], value["kp2d"]):
                old_pixels[int(record), np.asarray(box, dtype=np.float32).tobytes()] = np.asarray(kp)
        counts = Counter(windows=1, teacher_segments=len(after["tracks"]))
        for info in after["tracks"]:
            value = load_teacher_track(target / info["path"])
            if value["provenance_hash"] != after["provenance_hash"] or value["track_id"] != info["track_id"]:
                raise ValueError("Saved track provenance/ID differs from manifest")
            if len(value["record_index"]) != info["frame_count"]:
                raise ValueError("Saved temporal sequence length differs")
            for record, common, pts, box, kp in zip(
                value["record_index"], value["common_time_us"], value["media_pts_us"], value["bbx_xyxy"], value["kp2d"]
            ):
                record = int(record)
                position = positions[record]
                if int(common) != frozen[position]["common_time_us"] or int(pts) != frozen[position]["media_pts_us"]:
                    raise ValueError("Saved observation no longer binds original PTS/common time")
                raw_box = np.asarray(box, dtype=np.float32).tobytes()
                if info["track_id"] not in expected[position, raw_box]:
                    raise ValueError("Saved observed bbox/ID differs from independently replayed detections")
                pixel_key = record, raw_box
                if pixel_key in old_pixels:
                    if not np.array_equal(np.asarray(kp), old_pixels[pixel_key]):
                        raise ValueError("A frozen native 2D observation changed during cache reuse")
                    counts["exact_reused_keypoint_observations"] += 1
                else:
                    counts["newly_retained_observed_crops"] += 1
                counts["observed_person_frames"] += 1
            counts[
                (
                    "exact_reused_temporal_contexts"
                    if value.get("temporal_reuse_from")
                    else "newly_inferred_temporal_contexts"
                )
            ] += 1
        total.update(counts)
        by_source.setdefault(sid, Counter()).update(counts)
        print(f"[verified] {sid}/{source.name}", flush=True)
    if total["observed_person_frames"] != replay["counts"]["new_retained_observations"]:
        raise ValueError("Full retained-observation coverage differs from replay")
    report = dict(
        state="complete",
        passed=True,
        source_union_sha256=plan["source_union_sha256"],
        counts=dict(total),
        sources={k: dict(v) for k, v in by_source.items()},
        replay_sha256=file_sha(args.replay),
        teacher_receipt_sha256=file_sha(args.root / "repair_teacher_receipt.json"),
        training_published=False,
        limits=[
            "Detection/PTS/K/cached-keypoint and replay-ID agreement are checked; real-person identity truth remains unverified.",
            "Full teacher completion is separate from fitting and publishing new body labels.",
        ],
    )
    write_json(args.root / "teacher_verification.json", report)
    print(dict(passed=True, counts=dict(total)), flush=True)


if __name__ == "__main__":
    main()
