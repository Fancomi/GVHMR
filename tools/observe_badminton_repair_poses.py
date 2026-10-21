"""Observe both native cameras independently for repair/identity adjudication.

Only source native frames are decoded. No mesh, camera or point cloud is moved.
The cache includes all in-court detections, not crops chosen by fitted-joint error.
"""

import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.repair_execution import write_progress_json as write_json
from hmr4d.lidar_mocap.rally_pipeline import load_run_session, read_json
from hmr4d.lidar_mocap.relocations import load_pilot_run_relocations
from hmr4d.lidar_mocap.rgb_observations import RGBKeypointChecker
from hmr4d.lidar_mocap.segmentation import court_polygon


def groups(indices, maximum_gap=180):
    result = []
    for index in sorted(indices):
        if not result or index - result[-1][-1] > maximum_gap:
            result.append([])
        result[-1].append(index)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2")
    p.add_argument("--source-root", type=Path, default=PROJ_ROOT / "outputs/badminton_rallies_v1")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--source", action="append")
    p.add_argument("--precision", choices=("float32", "float16"), default="float16")
    p.add_argument("--reuse-poses-root", type=Path)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.threads < 1:
        p.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    cal = CalibrationSnapshot.load(args.source_root / "calibration/calibration_snapshot.json")
    relocations = load_pilot_run_relocations(args.source_root, cal.data)
    plan = read_json(args.root / "case_plan.json")
    if file_sha(plan["source_union"]) != plan["source_union_sha256"]:
        raise ValueError("Frozen union changed")
    checker = RGBKeypointChecker(cal, camera_name="cam1", device=args.device, threads=args.threads, pose_batch_size=4)
    config = dict(
        calibration_digest=cal.data["snapshot_digest"],
        models=checker.models,
        precision=args.precision,
        detector_imgsz=checker.detector_imgsz,
        scope="all_in_court_native_detections",
    )
    config_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    unresolved = read_json(plan["unresolved_native_endpoints"])
    total, reused = 0, 0
    start = time.perf_counter()
    for sid, scans in plan["selected_scans"].items():
        if args.source and sid not in args.source:
            continue
        session = load_run_session(args.source_root, cal, sid, relocations=relocations)
        pairs = session.pairs()
        native = {cam: set(pairs[i]["cameras"][cam]["record_index"] for i in scans) for cam in ("cam1", "cam2")}
        rows1, rows2 = session.camera_rows("cam1"), session.camera_rows("cam2")
        times1 = np.asarray([r["common_time_us"] for r in rows1], dtype=np.int64)
        for case in unresolved:
            if case["event"]["source_id"] != sid:
                continue
            original = read_json(Path(case["source_window"]) / "teacher/manifest.json")["provenance"]["frame_rows"]
            positions = case.get("context_positions")
            if positions is None:
                positions = [
                    pos + d for pos in case["positions"] for d in (-6, -3, 0, 3, 6) if 0 <= pos + d < len(original)
                ]
            extra = [original[pos]["record_index"] for pos in positions]
            indices2 = {e["native_record_index"] for e in case["event"]["replay_endpoints"]} | set(extra)
            for index2 in indices2:
                target = rows2[index2]["common_time_us"]
                q = np.searchsorted(times1, target)
                options = [i for i in (q - 1, q) if 0 <= i < len(times1)]
                index1 = min(options, key=lambda i: abs(times1[i] - target))
                if abs(times1[index1] - target) > 8500:
                    raise ValueError("No native second view within declared clock tolerance")
                native["cam2"].add(index2)
                native["cam1"].add(index1)
        for cam in ("cam1", "cam2"):
            checker.select_camera(cam)
            cached = {}
            for directory in (
                PROJ_ROOT / "outputs/badminton_dualcam_paired_full_v2/observations",
                PROJ_ROOT / "outputs/badminton_rallies_v1_paired/observations",
            ):
                report_path = directory / f"{sid}_{cam}.json"
                if not report_path.exists():
                    continue
                report = read_json(report_path)
                if report["calibration_digest"] != cal.data["snapshot_digest"]:
                    raise ValueError("Independent detection cache uses another calibration")
                with Path(report["frames_jsonl"]).open(encoding="utf-8") as handle:
                    for line in handle:
                        record = json.loads(line)
                        source = record["camera_native_source"]
                        index = source["record_index"]
                        if index in native[cam]:
                            cached[index] = (
                                record["detector_boxes_xyxy"],
                                record["detector_confidences"],
                                record["native_bgr_image_sha256"],
                                str(report_path),
                            )
            directory = args.root / "native_poses" / sid / cam
            directory.mkdir(parents=True, exist_ok=True)
            rows = session.camera_rows(cam)
            missing = []
            for index in sorted(native[cam]):
                path = directory / f"{index:07d}.json"
                if args.reuse_poses_root and not path.exists():
                    candidate = args.reuse_poses_root / "native_poses" / sid / cam / path.name
                    if candidate.exists():
                        previous = read_json(candidate)
                        if previous["config_digest"] != config_digest or previous["native"] != rows[index]:
                            raise ValueError("Explicitly reused pose has another native binding")
                        shutil.copyfile(candidate, path)
                if path.exists():
                    old = read_json(path)
                    if old["config_digest"] != config_digest or old["native"] != rows[index]:
                        raise ValueError("Pose observation cache settings/native source changed")
                    reused += 1
                else:
                    missing.append(index)
            polygon = np.asarray(court_polygon(cal, cam), np.float32)
            for subset in groups(missing):
                for index, bgr in session.iter_video_frames(cam, subset):
                    rgb = cv2.remap(bgr[:, :, ::-1], checker.map_x, checker.map_y, cv2.INTER_LINEAR)
                    raw_sha = hashlib.sha256(bgr.tobytes()).hexdigest()
                    cache_mismatch = None
                    if index in cached and raw_sha != cached[index][2]:
                        cache_mismatch = dict(
                            expected_raw_bgr_sha256=cached[index][2],
                            actual_raw_bgr_sha256=raw_sha,
                            report=cached[index][3],
                        )
                    if index in cached and cache_mismatch is None:
                        boxes, confidences, expected_sha, evidence = cached[index]
                        boxes, confidences = np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(
                            confidences, np.float32
                        )
                    else:
                        result = checker.yolo.predict(
                            rgb[:, :, ::-1].copy(),
                            classes=[0],
                            conf=0.35,
                            device=checker.inference_device,
                            imgsz=checker.detector_imgsz,
                            verbose=False,
                        )[0]
                        boxes, confidences = result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
                        evidence = "new_independent_native_YOLO_detection"
                    keep = [
                        j
                        for j, b in enumerate(boxes)
                        if b[3] - b[1] >= 35
                        and cv2.pointPolygonTest(polygon, (float((b[0] + b[2]) / 2), float(b[3])), True) >= -40
                    ]
                    boxes, confidences = boxes[keep], confidences[keep]
                    amp = (
                        torch.autocast("cuda", dtype=torch.float16)
                        if args.device == "cuda" and args.precision == "float16"
                        else nullcontext()
                    )
                    with amp:
                        keypoints = checker.pose_keypoints(rgb, boxes)
                    record = dict(
                        schema="badminton-native-pose-v1",
                        config_digest=config_digest,
                        source_id=sid,
                        camera=cam,
                        native=rows[index],
                        boxes=boxes.tolist(),
                        detection_confidence=confidences.tolist(),
                        keypoints=keypoints.tolist(),
                        raw_bgr_sha256=raw_sha,
                        detection_evidence=evidence,
                        legacy_detection_cache_rejected=cache_mismatch,
                        coordinate_space="opencv_undistorted_K_preserved",
                        source_video=str(session.video_paths[cam]),
                    )
                    write_json(directory / f"{index:07d}.json", record)
                    total += 1
                    if total % 25 == 0:
                        write_json(
                            args.root / "pose_progress.json",
                            dict(
                                state="running",
                                source_id=sid,
                                camera=cam,
                                computed=total,
                                reused=reused,
                                wall_seconds=time.perf_counter() - start,
                            ),
                        )
                        print(dict(source=sid, camera=cam, computed=total, reused=reused), flush=True)
            print(dict(source=sid, camera=cam, complete=len(native[cam])), flush=True)
    write_json(
        args.root / "pose_receipt.json",
        dict(
            state="complete" if not args.source or set(args.source) >= set(plan["selected_scans"]) else "partial",
            config=config,
            config_digest=config_digest,
            computed=total,
            reused=reused,
            plan_sha256=file_sha(args.root / "case_plan.json"),
            wall_seconds=time.perf_counter() - start,
            source_union_sha256=plan["source_union_sha256"],
        ),
    )


if __name__ == "__main__":
    main()
