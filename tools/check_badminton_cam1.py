"""CPU-only RGB keypoint observations on paired Cam1 or Cam2 review frames.

This never runs HMR2/GVHMR or optimizes the fitted body. YOLO person detections
are associated one-to-one with the fixed fitted meshes using bbox IoU, then
ViTPose observes the original native video after calibration-based undistortion.
These are detector consistency measurements, not human-labelled ground truth.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

# Set before importing torch or its dependencies. This process must not create
# a CUDA context while the production pairing worker is using the GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from hmr4d import PROJ_ROOT as PROJECT_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot


from hmr4d.lidar_mocap.rgb_observations import (
    JOINT_NAMES, RGBKeypointChecker as Cam1Check, add_raw_pixel_observations,
    sha256_file, write_json, statistics,
)


def summarize(report):
    records = [athlete for frame in report["frames"] for athlete in frame["athletes"]]
    measured = [record for record in records if record["status"] == "measured"]
    metrics = [joint for athlete in measured for joint in athlete["joint_metrics"] if joint["eligible"]]
    summary = {"checked_frames": len(report["frames"]), "checked_clips": len({f["clip_id"] for f in report["frames"]}),
               "athlete_frame_candidates": len(records), "measured_athlete_frames": len(measured),
               "excluded_athlete_frames": len(records) - len(measured),
               "eligible_joints_in_measured_frames": len(metrics),
               "coverage_of_candidate_17_joints": len(metrics) / max(len(records) * 17, 1),
               "coverage_of_measured_17_joints": len(metrics) / max(len(measured) * 17, 1),
               "pixel_error": statistics([m["pixel_error"] for m in metrics]),
               "focal_normalized_error": statistics([m["focal_normalized_error"] for m in metrics]),
               "image_diagonal_normalized_error": statistics([m["image_diagonal_normalized_error"] for m in metrics]),
               "bbox_height_normalized_error": statistics([m["bbox_height_normalized_error"] for m in metrics]),
               "exclusion_reasons": {reason: sum(reason in r["rejection_reasons"] for r in records)
                                     for reason in sorted({reason for r in records for reason in r["rejection_reasons"]})},
               "inference_seconds": sum(f["timing_seconds"]["total"] for f in report["frames"])}
    return summary


def write_reports(path, report):
    add_raw_pixel_observations(report)
    report["summary"] = summarize(report)
    report["per_clip"] = {clip_id: summarize({"frames": [frame for frame in report["frames"] if frame["clip_id"] == clip_id]})
                          for clip_id in sorted({frame["clip_id"] for frame in report["frames"]})}
    write_json(path, report)
    summary = report["summary"]
    error = summary["pixel_error"]
    camera_label = report.get("camera_name", "cam1").capitalize()
    format_number = lambda value: "无有效测量" if value is None else f"{value:.3f}"
    text = f"""# {camera_label} 原始 RGB 关键点一致性检查

检查 {summary['checked_clips']} 个片段、{summary['checked_frames']} 个真实视频帧；{summary['measured_athlete_frames']}/{summary['athlete_frame_candidates']} 个人物帧可量化，排除 {summary['excluded_athlete_frames']} 个人物帧。

{camera_label} 像素误差 mean/median/P90：{format_number(error['mean'])}/{format_number(error['median'])}/{format_number(error['p90'])} px；合格关节 {summary['eligible_joints_in_measured_frames']} 个，占全部候选17关节的 {summary['coverage_of_candidate_17_joints']:.1%}。

使用原始 {camera_label} 视频按记录的 native index/PTS 解码，再按冻结 K/d 去畸变。CPU YOLO 检测框与固定 SMPL 投影框唯一关联，CPU ViTPose 观测关键点；SMPL COCO17 mesh regressor 以同一 K、distort=False 投影。低置信度、图像截断、其他拟合人物遮挡及身份关联歧义均记录排除原因。没有将新 RGB 观测回馈拟合。

这些误差是对 {camera_label} RGB 检测器的一致性，不是人工人体 GT 精度或严格无偏验证。关联和遮挡筛选使用既有拟合体，标定可能已利用其他 Cam1 RGB mask；自遮挡、网线和其他未建模遮挡未认证。原生时钟曝光精度与逐点扫描补偿未验证。Cam2 现有关键点误差是在去畸变像素空间参与优化后的残差，新增 Cam2 RGB 检测仅供轨迹和时间诊断，也不是 held-out 验证。两相机焦距不同，另存焦距、图像对角线和独立检测框高度归一化误差。

所有模型在 CPU 上执行；CUDA context initialized=false。详情、逐关节覆盖、源时间和模型 SHA256 见 [{path.name}]({path.name})。
"""
    text += ("\n| 片段 | 帧数 | 可量化人物帧/候选 | 合格关节覆盖 | mean px | median px | P90 px | mean 焦距归一化 | mean 图像对角线比例 | mean 人物框高度比例 |\n"
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for clip_id, value in report["per_clip"].items():
        focal = value['focal_normalized_error']['mean']
        image_scale = value['image_diagonal_normalized_error']['mean']
        body_scale = value['bbox_height_normalized_error']['mean']
        focal_text = "无测量" if focal is None else f"{focal:.6f}"
        image_text = "无测量" if image_scale is None else f"{image_scale:.2%}"
        body_text = "无测量" if body_scale is None else f"{body_scale:.2%}"
        text += (f"| {clip_id} | {value['checked_frames']} | "
                 f"{value['measured_athlete_frames']}/{value['athlete_frame_candidates']} | "
                 f"{value['coverage_of_candidate_17_joints']:.1%} | "
                 f"{format_number(value['pixel_error']['mean'])} | "
                 f"{format_number(value['pixel_error']['median'])} | {format_number(value['pixel_error']['p90'])} | "
                 f"{focal_text} | {image_text} | {body_text} |\n")
    path.with_suffix(".md").write_text(text, encoding="utf-8")


def verify_current_selection(manifest_path, report):
    """Allow render counters to change, but reject changed source inputs."""
    from hmr4d.lidar_mocap.sequence_review import review_selection_digest

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest["calibration_digest"] != report["calibration_digest"]:
        raise ValueError("Review calibration changed during the CPU check")
    clips = {clip["id"]: clip for clip in manifest["clips"]}
    for observed in report["frames"]:
        clip = clips[observed["clip_id"]]
        source = next(frame for frame in clip["frames"] if frame["lidar_index"] == observed["lidar_index"])
        if (clip["session_id"] != observed["session_id"] or source["scan_mid_us"] != observed["scan_mid_us"]
                or source["cameras"] != observed["paired_cameras"]):
            raise ValueError("Review camera/source timestamps changed during the CPU check")
        expected = {athlete["motion_id"]: athlete for athlete in source["athletes"]}
        if set(expected) != {athlete["motion_id"] for athlete in observed["athletes"]}:
            raise ValueError("Review athlete selection changed during the CPU check")
        for athlete in observed["athletes"]:
            previous = expected[athlete["motion_id"]]
            path = Path(previous["fit_path"])
            if not path.is_absolute():
                path = Path(manifest["output_root"]) / path
            if (path.resolve() != Path(athlete["fit_path"]).resolve()
                    or previous["accepted"] != athlete["fit_accepted"]
                    or sha256_file(path) != athlete["fit_sha256"]):
                raise ValueError("Review fitted geometry changed during the CPU check")
    digest = review_selection_digest(manifest)
    if manifest.get("selection_sha256") != digest:
        raise ValueError("Review manifest stable selection digest is invalid")
    initial = report.get("selection_sha256_at_start")
    if initial is not None and initial != digest:
        raise ValueError("Review complete source selection changed during the CPU check")
    # Earlier diagnostic reports used a local fingerprint with QA fields. Use
    # the renderer's public source-selection contract for the final binding.
    report.pop("sequence_selection_sha256_at_completion", None)
    report.update(selection_sha256=digest, current_selection_sha256=digest,
                  selection_digest_matches_manifest=True,
                  source_selection_matches_current_manifest=True,
                  manifest_sha256_at_completion=sha256_file(manifest_path),
                  manifest_render_status_at_completion=manifest["state"],
                  manifest_presentation_fields_are_mutable=True,
                  manifest_sha256_role="Whole-file hash at check start; not asserted equal to later rendered manifest")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/badminton_corner_fix_v1")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--clip", action="append", default=[], help="Review clip ID; repeat to select multiple clips")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--stride", type=int, help="Observe every Nth original LiDAR scan in each selected clip")
    sampling.add_argument("--all-frames", action="store_true", help="Observe every original LiDAR scan")
    parser.add_argument("--camera", choices=("cam1", "cam2"), default="cam1")
    parser.add_argument("--report-path", type=Path, help="JSON report path inside output-root; companion Markdown is written")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--detector-imgsz", type=int, default=960)
    parser.add_argument("--flip-test", action="store_true")
    parser.add_argument("--max-total-seconds", type=float, default=900.)
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    output = args.output_root.resolve()
    if not output.is_relative_to(PROJECT_ROOT / "outputs") or args.threads < 1:
        raise ValueError("Check output must stay in project outputs and CPU threads must be positive")
    if args.stride is not None and args.stride < 1:
        raise ValueError("--stride must be a positive integer")
    manifest_path = (args.manifest or output / "sequence_review/manifest.json").resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from hmr4d.lidar_mocap.sequence_review import review_selection_digest
    selection_sha256 = review_selection_digest(manifest)
    if manifest.get("selection_sha256") != selection_sha256:
        raise ValueError("Review manifest stable source selection digest is invalid")
    calibration = CalibrationSnapshot.load(manifest["calibration_snapshot"])
    if manifest["calibration_digest"] != calibration.data["snapshot_digest"]:
        raise ValueError("Sequence manifest and frozen calibration digest differ")
    unknown_clips = set(args.clip) - {clip["id"] for clip in manifest["clips"]}
    if unknown_clips:
        raise ValueError(f"Unknown review clip IDs: {sorted(unknown_clips)}")
    selected_clips = [clip for clip in manifest["clips"] if not args.clip or clip["id"] in args.clip]
    selected = []
    for clip in selected_clips:
        if args.stride is not None or args.all_frames:
            indices = list(range(0, len(clip["frames"]), args.stride or 1))
        else:
            indices = sorted({0, len(clip["frames"]) // 2, len(clip["frames"]) - 1})
        for index in indices:
            selected.append(({**clip, "output_root": manifest["output_root"]}, clip["frames"][index]))
    custom_selection = bool(args.clip or args.stride is not None or args.all_frames or args.camera != "cam1")
    if args.report_path is not None:
        destination = args.report_path.resolve()
    elif custom_selection:
        tag = "_".join(args.clip) if args.clip else "all_clips"
        cadence = f"stride{args.stride}" if args.stride else ("all_frames" if args.all_frames else "anchors")
        destination = output / "calibration_refinement" / f"rgb_keypoints_{args.camera}_{tag}_{cadence}.json"
    else:
        destination = output / "sequence_review" / "cam1_check.json"
    destination = destination.resolve()
    if not destination.is_relative_to(output) or destination.suffix != ".json":
        raise ValueError("--report-path must be a JSON report inside --output-root")
    protected = output / "sequence_review/cam1_check.json"
    if custom_selection and destination == protected.resolve():
        raise ValueError("Custom camera/clip sampling cannot overwrite the original Cam1 check")
    started = time.perf_counter()
    checker = Cam1Check(calibration, args.threads, args.detector_imgsz, args.flip_test, camera_name=args.camera)
    loading_seconds = time.perf_counter() - started
    report = {"schema": ("badminton-rgb-keypoint-observations-v2" if custom_selection
                         else "badminton-cam1-independent-rgb-consistency-v1"),
              "created_utc": datetime.now(timezone.utc).isoformat(), "manifest": str(manifest_path),
              "checker_sha256": sha256_file(__file__), "completed": False,
              "manifest_sha256": sha256_file(manifest_path),
              "selection_sha256_at_start": selection_sha256,
              "calibration_snapshot": manifest["calibration_snapshot"], "calibration_digest": manifest["calibration_digest"],
              "camera_name": args.camera, "camera": checker.camera, "models": checker.models, "device": "cpu", "cpu_threads": args.threads,
              "detector_imgsz": args.detector_imgsz, "flip_test": args.flip_test,
              "confidence_threshold": .5, "minimum_eligible_joints": 6,
              "projection_space": f"undistorted_{args.camera}_K_preserved_distortion_zero",
              "not_ground_truth": True, "not_fed_back_to_fitting": True,
              "metric_independence_role": ("Cam1 RGB not fed to body fitting; bbox association and camera calibration are not independent"
                                           if args.camera == "cam1" else "Cam2 source view already used by body fitting; new RGB trajectory observations are diagnostic"),
              "association_selection_bias": "Unique IoU association and occlusion filter depend on fixed fitted meshes; not strictly unbiased",
              "calibration_independent_of_cam1_rgb": False,
              "unverified_visibility": ["self_occlusion", "net_wires", "unmodelled_people_or_objects"],
              "cam2_metric_role": "undistorted_pixel_residual_used_in_optimization_not_heldout_validation",
              "focal_normalization_definition": "norm([delta_u/fx,delta_v/fy])",
              "image_normalization_definition": "pixel_error/image_diagonal_pixels",
              "body_normalization_definition": "pixel_error/independent_YOLO_bbox_height_pixels",
              "original_exposure_accuracy_verified": False,
              "requested_clips": args.clip, "stride": args.stride, "all_frames": args.all_frames,
              "selection": (f"every_{args.stride}_native_lidar_scan_per_clip" if args.stride else
                            "all_native_lidar_frames_per_clip" if args.all_frames else "start_middle_end_per_clip"),
              "temporal_sampling_role": "Real camera common timestamps; any subsequent trajectory interpolation is diagnostic, not measured exposure truth",
              "planned_frames": len(selected),
              "model_loading_and_hash_seconds": loading_seconds, "frames": []}
    for position, (clip, frame) in enumerate(selected):
        record = checker.check_frame(clip, frame)
        report["frames"].append(record)
        write_reports(destination, report)
        print(json.dumps({"checked": len(report['frames']), "planned": len(selected), "clip": clip["id"],
                          "frame_seconds": record["timing_seconds"],
                          "measured_athlete_frames": report["summary"]["measured_athlete_frames"],
                          "eligible_joint_coverage": report["summary"]["coverage_of_candidate_17_joints"],
                          "mean_px": report["summary"]["pixel_error"]["mean"]}, ensure_ascii=False), flush=True)
        # Evaluate a real frame before changing coverage. Do not spend more than
        # the specified total budget merely to keep the denser optional audit.
        if position == 0 and not (args.stride or args.all_frames):
            estimate = loading_seconds + record["timing_seconds"]["total"] * len(selected)
            report["estimated_all_selected_seconds_after_first_frame"] = estimate
            if estimate > args.max_total_seconds:
                middle = []
                for candidate in selected_clips:
                    middle.append(({**candidate, "output_root": manifest["output_root"]},
                                   candidate["frames"][len(candidate["frames"]) // 2]))
                already = {(clip["id"], frame["lidar_index"])}
                selected[:] = selected[:1] + [item for item in middle if (item[0]["id"], item[1]["lidar_index"]) not in already]
                report["selection"] = "middle_per_clip_plus_initial_runtime_probe"
                report["coverage_reduction_reason"] = "Actual first-frame CPU time predicts more than 15 minutes for 30 selected frames"
                report["planned_frames"] = len(selected)
    verify_current_selection(manifest_path, report)
    report["completed"] = True
    report["total_seconds"] = time.perf_counter() - started
    write_reports(destination, report)
    print(json.dumps({"report": str(destination), "completed": True, "summary": report["summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
