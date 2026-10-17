"""Pair independently observed native poses using frozen camera epipolar geometry."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.multiview import associate_poses
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import load_run_session, read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2")
    p.add_argument("--partial", action="store_true")
    args = p.parse_args()
    plan = read_json(args.root / "case_plan.json")
    cal = CalibrationSnapshot.load(PROJ_ROOT / "outputs/badminton_rallies_v1/calibration/calibration_snapshot.json")
    counts = Counter()
    for sid, scans in plan["selected_scans"].items():
        session = load_run_session(PROJ_ROOT / "outputs/badminton_rallies_v1", cal, sid)
        pairs = session.pairs()
        directory = args.root / "view_associations" / sid
        directory.mkdir(parents=True, exist_ok=True)
        for scan in scans:
            pair = pairs[scan]
            paths = [
                args.root / "native_poses" / sid / cam / f"{pair['cameras'][cam]['record_index']:07d}.json"
                for cam in ("cam1", "cam2")
            ]
            if not all(path.exists() for path in paths):
                if args.partial:
                    counts["missing"] += 1
                    continue
                raise ValueError(f"Missing native pose for {sid}/{scan}")
            a, b = map(read_json, paths)
            poses1 = np.asarray(a["keypoints"], dtype=float).reshape(-1, 17, 3)
            poses2 = np.asarray(b["keypoints"], dtype=float).reshape(-1, 17, 3)
            result = associate_poses(poses1, poses2, cal)
            write_json(
                directory / f"{scan:06d}.json",
                dict(
                    source_id=sid,
                    scan=scan,
                    scan_mid_us=pair["scan_mid_us"],
                    native_files=list(map(str, paths)),
                    pairs=result,
                ),
            )
            counts["scans"] += 1
            counts["pairs"] += len(result)
            counts["accepted_pairs"] += sum(r["accepted"] for r in result)
        print(dict(source=sid, counts=dict(counts)), flush=True)
    write_json(
        args.root / "view_association_summary.json",
        dict(
            counts=dict(counts),
            partial=args.partial,
            note="Fixed-camera native pose correspondence; not anatomical ground truth",
        ),
    )


if __name__ == "__main__":
    main()
