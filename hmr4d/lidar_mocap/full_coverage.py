"""Nominal coverage and the existing bilateral-presence gate on native evidence."""

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .calibration import CalibrationSnapshot
from .dual_view import DualViewGate, detection_reasons
from .pilot import binding
from .rally_pipeline import load_run_session, read_json, read_rows
from .relocations import load_pilot_run_relocations
from .repair_execution import write_progress_json
from .rgb_observations import associate_boxes, clip_box
from .teacher import load_teacher_track, sample_track


def coverage_counts(fitted_per_scan, accepted_per_scan, expected_people=2):
    fitted = np.asarray(fitted_per_scan, dtype=np.int64)
    accepted = np.asarray(accepted_per_scan, dtype=np.int64)
    if (expected_people < 1 or fitted.shape != accepted.shape or fitted.ndim != 1
            or np.any(accepted < 0) or np.any(accepted > fitted)):
        raise ValueError("Accepted/fitted counts must address the same actual selected scans")
    expected = len(fitted) * expected_people
    return dict(selected_scans=len(fitted), nominal_people_per_scan=expected_people,
        nominal_person_scan_cells=expected, fitted=int(fitted.sum()), accepted=int(accepted.sum()),
        qa_rejected=int((fitted - accepted).sum()), net_unfilled_before_qa=expected - int(fitted.sum()),
        gross_unfilled_before_qa=int(np.maximum(expected_people - fitted, 0).sum()),
        additional_fitted_over_nominal=int(np.maximum(fitted - expected_people, 0).sum()),
        net_unfilled_after_qa=expected - int(accepted.sum()),
        gross_unfilled_after_qa=int(np.maximum(expected_people - accepted, 0).sum()),
        additional_accepted_over_nominal=int(np.maximum(accepted - expected_people, 0).sum()),
        fitted_people_histogram={str(k): int(v) for k, v in sorted(Counter(map(int, fitted)).items())})


def audit_full_coverage(root, expected_people=2):
    import torch

    torch.set_num_threads(2)
    if expected_people < 1:
        raise ValueError("The explicit nominal people count must be positive")
    started = time.perf_counter()
    root = Path(root).resolve()
    labels, repair = root / "repair/final_labels", root / "repair"
    plan = read_json(labels / "repair_plan.json")
    native_path = labels / "native_verification.json"
    native = read_json(native_path)
    if native.get("passed") is not True or native.get("verified_windows") != len(plan["jobs"]):
        raise ValueError("Coverage qualification follows complete native member/SMPL verification")
    dataset_path = labels / "datasets/dataset_manifest.json"
    dataset = read_json(dataset_path)
    cal = CalibrationSnapshot.load(labels / "calibration/calibration_snapshot.json")
    relocations = load_pilot_run_relocations(labels, cal.data)
    pose_receipt_path = repair / "pose_receipt.json"
    pose_receipt = read_json(pose_receipt_path)
    if (dataset.get("training_published") is not False or pose_receipt.get("state") != "complete"
            or pose_receipt["config"]["calibration_digest"] != cal.data["snapshot_digest"]):
        raise ValueError("The native evidence and review-only dataset must use the same frozen cameras")
    sessions, pose_cache, native_files, report_files = {}, {}, {}, {}
    fitted_counts, accepted_counts = [], []
    missing, qa_rejected, bilateral_rows = [], [], []
    missing_stage_counts, bilateral_reasons = Counter(), Counter()
    seen, qualified = set(), 0
    progress_path = root / "coverage_bilateral_progress.json"

    def observation(sid, cam, native_pair):
        path = repair / "native_poses" / sid / cam / f"{native_pair['record_index']:07d}.json"
        key = str(path)
        if key not in pose_cache:
            blob = path.read_bytes()
            record = json.loads(blob)
            if record["config_digest"] != pose_receipt["config_digest"] or record["source_id"] != sid or record["camera"] != cam:
                raise ValueError("Native presence evidence changed its source/camera/configuration")
            if record.get("coordinate_space") != "opencv_undistorted_K_preserved":
                raise ValueError("Native presence boxes must use the adopted undistorted camera pixels")
            if len(record.get("raw_bgr_sha256", "")) != 64:
                raise ValueError("Native detections lack their actual decoded pixel digest")
            pose_cache[key] = record
            native_files[key] = hashlib.sha256(blob).hexdigest()
        record = pose_cache[key]
        if any(record["native"].get(name) != native_pair.get(name)
               for name in ("record_index", "common_time_us", "media_pts_us", "source_frame_no")):
            raise ValueError("Presence evidence is not the actual paired RGB record/PTS/common time")
        return record, key

    for number, job in enumerate(plan["jobs"]):
        sid, source_window = job["source_id"], Path(job["source_window"])
        output = labels / "sessions" / sid / "windows" / source_window.name
        config, report = read_json(output / "window.json"), read_json(output / "fit_report.json")
        report_files[str(output / "fit_report.json")] = binding(output / "fit_report.json")["sha256"]
        rows = read_rows(output / "quality.jsonl")
        by_scan, exclusions = defaultdict(list), defaultdict(list)
        for row in rows:
            by_scan[row["lidar_index"]].append(row)
            if not row["accepted"]:
                qa_rejected.append(dict(source_id=sid, window=source_window.name,
                    lidar_index=row["lidar_index"], motion_id=row["motion_id"], reasons=row["rejection_reasons"]))
        for row in report["segmentation_exclusions"]:
            exclusions[row["lidar_index"]].append(row)
        if sid not in sessions:
            session = load_run_session(labels, cal, sid, relocations=relocations)
            sessions[sid] = {pair["lidar_index"]: pair for pair in session.pairs()}
        pairs = [sessions[sid][index] for index in config["source_pairs"]]
        times = [pair["scan_mid_us"] for pair in pairs]
        teacher_manifest = read_json(output / "teacher/manifest.json")
        teachers = [sample_track(load_teacher_track(output / "teacher" / item["path"]), times)
                    for item in teacher_manifest["tracks"]]
        for position, pair in enumerate(pairs):
            scan = pair["lidar_index"]
            if (sid, scan) in seen:
                raise ValueError("The nominal coverage ledger contains an overlapping original scan")
            seen.add((sid, scan))
            current = by_scan[scan]
            accepted = [row for row in current if row["accepted"]]
            fitted_counts.append(len(current))
            accepted_counts.append(len(accepted))
            records = {cam: observation(sid, cam, pair["cameras"][cam]) for cam in ("cam1", "cam2")}
            deficit = max(0, expected_people - len(current))
            if deficit:
                available = sum(bool(sample["valid"][position]) for sample in teachers)
                before_pose = min(deficit, max(0, expected_people - available))
                stages = dict(no_usable_continuous_primary_teacher=before_pose,
                              native_pose_or_point_segmentation_exclusion=deficit - before_pose)
                missing_stage_counts.update(stages)
                primary_record = records["cam2"][0]
                missing.append(dict(source_id=sid, window=source_window.name, lidar_index=scan,
                    scan_mid_us=pair["scan_mid_us"], nominal_missing_cells=deficit,
                    fitted_cells=len(current), valid_primary_teacher_contexts=available,
                    actual_primary_native_detections=len(primary_record["boxes"]), stage_counts=stages,
                    recorded_exclusions=exclusions[scan], native_cameras=pair["cameras"],
                    url=f"?source={sid}&scan={scan}"))
            if not accepted:
                continue
            geometry = []
            for row in accepted:
                with np.load(row["fit_path"], allow_pickle=False) as fit:
                    geometry.append(fit["vertices"].astype(np.float64))
            views = [dict() for _ in accepted]
            for cam in ("cam1", "cam2"):
                record, path = records[cam]
                boxes = np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 4)
                confidence = np.asarray(record["detection_confidence"], dtype=np.float32).reshape(-1)
                if len(boxes) != len(confidence):
                    raise ValueError("Native presence boxes and detector confidences differ")
                projected = []
                for vertices in geometry:
                    uv, depth = cal.project(vertices, cam, distort=False)
                    projected.append(clip_box(uv, depth, *cal.camera(cam)["image_size"]))
                visible = [i for i, box in enumerate(projected) if box is not None]
                matches, _ = associate_boxes([projected[i] for i in visible], boxes)
                matches = {visible[i]: match for i, match in matches.items()}
                for i in range(len(accepted)):
                    person = {}
                    if i in matches:
                        match = matches[i]
                        index = int(match["detection_index"])
                        person = dict(detector_index=index, independent_detection_bbox_xyxy=boxes[index].tolist(),
                            independent_detection_confidence=float(confidence[index]),
                            association_bbox_iou=match["bbox_iou"], association_margin=match["association_margin"])
                    reasons = detection_reasons(person)
                    views[i][cam] = dict(passed=not reasons, reasons=reasons, observation=person,
                        evidence_file=path, native=pair["cameras"][cam], raw_bgr_sha256=record["raw_bgr_sha256"])
                    bilateral_reasons.update(cam + ":" + reason for reason in reasons)
            for row, view in zip(accepted, views):
                passed = all(value["passed"] for value in view.values())
                qualified += int(passed)
                bilateral_rows.append(dict(source_id=sid, window=source_window.name, lidar_index=scan,
                    motion_id=row["motion_id"], fit_path=row["fit_path"], passed=passed, views=view))
        write_progress_json(progress_path, dict(state="running", completed_windows=number + 1,
            windows=len(plan["jobs"]), tested_qa_cells=len(bilateral_rows), bilateral_rule_pass=qualified,
            training_published=False))
    counts = coverage_counts(fitted_counts, accepted_counts, expected_people)
    if counts["accepted"] != dataset["accepted"] or len(bilateral_rows) != dataset["accepted"]:
        raise ValueError("Coverage, qualification and the completed native dataset counts differ")
    details_path = root / "coverage_bilateral_cells.json"
    write_progress_json(details_path, dict(nominal_missing_scan_cases=missing, qa_rejections=qa_rejected,
                                         bilateral_cells=bilateral_rows))
    result = dict(schema="badminton-full-native-coverage-bilateral-v1", counts=counts,
        missing_nominal_cells_by_stage=dict(missing_stage_counts),
        qa_rejection_reasons=dict(Counter(reason for row in qa_rejected for reason in row["reasons"])),
        bilateral_detection_rule=dict(gate=asdict(DualViewGate()), tested=counts["accepted"],
            passed=qualified, excluded=counts["accepted"] - qualified, exclusion_reasons=dict(bilateral_reasons)),
        evidence=dict(dataset=binding(dataset_path), native_verification=binding(native_path),
            native_pose_receipt=binding(pose_receipt_path), calibration=binding(labels / "calibration/calibration_snapshot.json"),
            source_fit_reports_sha256=report_files, native_pose_files_sha256=native_files, cells=binding(details_path)),
        limitations=["Two people per scan is an explicit nominal assumption, not independently verified physical identity ground truth",
                     "Bilateral qualification reuses actual native detections, exact record/PTS/common times and the existing presence rule",
                     "Native RGB pixels were not decoded again by this CPU accounting step; no formal training export/publication is claimed"],
        report_seconds=time.perf_counter() - started, training_published=False)
    write_progress_json(root / "coverage_bilateral_audit.json", result)
    write_progress_json(progress_path, dict(state="complete", counts=counts, bilateral_rule_pass=qualified,
                                         report=binding(root / "coverage_bilateral_audit.json"), training_published=False))
    return result
