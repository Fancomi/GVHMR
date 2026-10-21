"""Review-only fit quality on native observations, with measured stage wall times."""

from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch

from hmr4d import PROJ_ROOT
from .calibration import CalibrationSnapshot
from .dual_view import file_sha
from .pipeline import write_json
from .rally_pipeline import read_json, read_rows


def statistics(values):
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    return (
        dict(
            count=len(data),
            mean=float(data.mean()),
            median=float(np.median(data)),
            p90=float(np.percentile(data, 90)),
            max=float(data.max()),
        )
        if len(data)
        else dict(count=0)
    )


def report_pilot(root):
    started = time.perf_counter()
    root = Path(root).resolve()
    if not root.is_relative_to(PROJ_ROOT / "outputs"):
        raise ValueError("Use the explicitly prepared workspace pilot")
    labels = root / "repair/final_labels"
    dataset_path = labels / "datasets/dataset_manifest.json"
    dataset = read_json(dataset_path)
    native = read_json(labels / "native_verification.json")
    if dataset.get("review_only") is not True or native.get("passed") is not True:
        raise ValueError("Pilot report requires review-only bodies and completed native verification")
    cal = CalibrationSnapshot.load(labels / "calibration/calibration_snapshot.json")
    if cal.data.get("review_only") is not True or dataset["calibration_digest"] != cal.data["snapshot_digest"]:
        raise ValueError("Pilot report camera/body scope differs")
    plan = read_json(labels / "repair_plan.json")
    regressor = torch.load(
        PROJ_ROOT / "hmr4d/utils/body_model/smpl_coco17_J_regressor.pt", map_location="cpu", weights_only=False
    ).numpy()
    rows, source_files, pose_cache = [], {}, {}
    for job in plan["jobs"]:
        window = labels / "sessions" / job["source_id"] / "windows" / Path(job["source_window"]).name
        fit_path = window / "fit_report.json"
        fit = read_json(fit_path)
        quality_path = window / "quality.jsonl"
        if fit["artifacts_sha256"]["quality.jsonl"] != file_sha(quality_path):
            raise ValueError("Pilot fit quality changed after completion")
        source_files[str(fit_path)] = file_sha(fit_path)
        for row in read_rows(quality_path):
            raw_errors = {}
            pose = row.get("native_pose_evidence") or {}
            with np.load(row["fit_path"]) as fitted:
                joints = regressor @ fitted["vertices"]
            for cam in ("cam1", "cam2"):
                info, detection = pose.get("files", {}).get(cam), pose.get(cam + "_detection")
                if info is None or detection is None:
                    continue
                path = info["path"]
                if path not in pose_cache:
                    if file_sha(path) != info["sha256"]:
                        raise ValueError("Raw native keypoint evidence changed")
                    pose_cache[path] = read_json(path)
                kp = np.asarray(pose_cache[path]["keypoints"][detection], float)
                uv, depth = cal.project(joints, cam, distort=False)
                valid = (kp[:, 2] >= 0.5) & (depth > 0) & np.isfinite(kp).all(axis=1)
                distance = np.linalg.norm(uv - kp[:, :2], axis=1)
                ankles = valid & (np.arange(len(kp)) >= 15)
                raw_errors[cam] = dict(
                    joints=int(valid.sum()),
                    ankles=int(ankles.sum()),
                    mean_px=float(distance[valid].mean()) if valid.any() else None,
                    ankle_mean_px=float(distance[ankles].mean()) if ankles.any() else None,
                )
            rows.append(
                dict(
                    accepted=row["accepted"],
                    qa=row["qa"],
                    raw_camera=raw_errors,
                    rejection_reasons=row["rejection_reasons"],
                )
            )

    def cohort(selected):
        keys = (
            "teacher_surface_mean_m",
            "aligned_surface_mean_m",
            "surface_mean_m",
            "surface_p90_m",
            "max_ground_penetration_m",
        )
        return dict(
            person_scans=len(selected),
            surface_and_ground_mm={
                key: statistics([r["qa"][key] * 1000 for r in selected if key in r["qa"]]) for key in keys
            },
            raw_native_camera_px={
                cam: {
                    metric: statistics(
                        [
                            r["raw_camera"][cam][metric]
                            for r in selected
                            if r["raw_camera"].get(cam, {}).get(metric) is not None
                        ]
                    )
                    for metric in ("mean_px", "ankle_mean_px")
                }
                for cam in ("cam1", "cam2")
            },
            filtered_confidence_weighted_camera_px={
                cam: statistics(
                    [
                        r["qa"]["reprojection_by_camera_px"][cam]
                        for r in selected
                        if r["qa"].get("reprojection_by_camera_px", {}).get(cam) is not None
                    ]
                )
                for cam in ("cam1", "cam2")
            },
        )

    accepted = [row for row in rows if row["accepted"]]
    if len(accepted) != dataset["accepted"]:
        raise ValueError("Pilot report and aggregate counts differ")
    timings = read_json(root / "stage_timings.json")
    result = dict(
        schema="badminton-review-pilot-quality-v1",
        review_only=True,
        training_published=False,
        calibration_digest=cal.data["snapshot_digest"],
        dataset=dict(path=str(dataset_path), sha256=file_sha(dataset_path)),
        selection=read_json(root / "pilot_input.json")["selection"],
        cohorts=dict(all_fitted=cohort(rows), accepted=cohort(accepted)),
        rejections=dict(Counter(reason for row in rows for reason in row["rejection_reasons"])),
        source_fit_reports_sha256=source_files,
        measured_stage_timings=timings,
        metric_scope=dict(
            surface="One-sided observed points to full SMPL triangles; equal person-scan weighting. QA uses at most 4096 returns per person-scan.",
            raw_camera="Original independent native COCO17 observations with confidence >=0.5; no ray rejection or joint-label permutation applied in this metric.",
            filtered_camera="The fit's confidence-weighted native observations after the existing contradictory-ray filter.",
            accuracy_limit="Reprojection/observed surface residuals, not independent anatomical ground truth; mask IoU is not measured here.",
        ),
        report_seconds=time.perf_counter() - started,
    )
    write_json(root / "quality_report.json", result)
    return result
