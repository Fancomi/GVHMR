"""Read-only CPU audit of completed badminton windows; write JSON and Markdown.

Run with ``uv run --no-sync python tools/report_badminton_quality.py
--output-root outputs/badminton_corner_fix_v1 --output-dir outputs/.../reports``.
No body models, video inference or CUDA operations are used. A progress watermark
freezes the completed-job scope while a longer pipeline job is still running.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_utc(value):
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("Job timestamps must include a UTC offset")
    return moment.astimezone(timezone.utc)


def completed_windows(root, pairing, progress, started_utc=None, validated_windows=None):
    """Prefer regular job windows, including a short terminal window, over smoke.

    Configured window length defines a regular full-run window; actual duration
    alone would wrongly drop a short terminal window or count a pre-job smoke.
    Only reports committed at the captured progress timestamp are selected.
    """
    root = Path(root)
    target_seconds = pairing["settings"]["window_seconds"]
    watermark = parse_utc(progress["updated_utc"]).timestamp()
    started = parse_utc(started_utc).timestamp() if started_utc is not None else None
    candidates = defaultdict(list)
    excluded = Counter()
    for report_path in sorted((root / "sessions").glob("*/windows/*/fit_report.json")):
        config = read_json(report_path.parent / "window.json")
        report = read_json(report_path)
        key = (config['session_id'], report_path.parent.name)
        if validated_windows is not None and key not in validated_windows:
            excluded['not_selected_for_export'] += 1
            continue
        if config["calibration_digest"] != pairing["calibration_digest"]:
            excluded["different_calibration"] += 1
            continue
        if config["settings"].get("window_seconds") != target_seconds:
            excluded["smoke_or_different_window_settings"] += 1
            continue
        if config["settings"] != pairing["settings"]:
            excluded["different_job_settings"] += 1
            continue
        modified = report_path.stat().st_mtime
        if started is not None and modified < started and validated_windows is None:
            excluded["completed_before_current_job"] += 1
            continue
        if modified > watermark:
            excluded["completed_after_progress_watermark"] += 1
            continue
        left, right = config["left_us"], config["right_us"]
        if right <= left:
            raise ValueError(f"Invalid window duration: {report_path}")
        candidates[config["session_id"]].append((right - left, config, report, report_path.parent))
    selected = []
    for sid, entries in sorted(candidates.items()):
        chosen = []
        for _, config, report, directory in sorted(entries, key=lambda entry: entry[0], reverse=True):
            if any(old["left_us"] <= config["left_us"] and old["right_us"] >= config["right_us"]
                   for old in chosen):
                excluded["contained_overlapping_window"] += 1
                continue
            if any(max(old["left_us"], config["left_us"]) < min(old["right_us"], config["right_us"])
                   for old in chosen):
                raise ValueError(f"Partially overlapping completed windows: {sid}")
            chosen.append(config)
            selected.append((config, report, directory))
    selected.sort(key=lambda entry: (entry[0]["session_id"], entry[0]["left_us"]))
    if validated_windows is not None:
        selected_keys = {(config['session_id'], directory.name) for config, _, directory in selected}
        if selected_keys != set(validated_windows):
            raise ValueError('Exported windows are missing, mismatched or beyond the completion watermark')
    return selected, dict(excluded)


def metric(rows, key):
    values = [row["qa"].get(key) for row in rows]
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not values:
        return {"sample_count": 0, "missing_count": len(rows), "mean": None,
                "median": None, "min": None, "max": None}
    return {"sample_count": len(values), "missing_count": len(rows) - len(values),
            "mean": statistics.fmean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def quality_group(rows):
    # Each row is one athlete at one scan. These are equally weighted frame
    # statistics, not a pooled percentile over all returns in the recording.
    stage_keys = {
        "teacher_mapped_before_lidar_translation": ("init_p2surface_mean_m", "init_p2surface_p90_m"),
        "smpl_topology_converted_before_lidar_translation": ("converted_surface_mean_m", "converted_p2surface_p90_m"),
        "cloud_center_aligned_before_optimization": ("aligned_surface_mean_m", "aligned_p2surface_p90_m"),
        "final_smpl_all_triangles": ("final_p2surface_mean_m", "final_p2surface_p90_m"),
    }
    return {
        "person_scan_count": len(rows),
        "surface_stages": {
            name: {"frame_mean_m": metric(rows, mean_key), "frame_p90_m": metric(rows, p90_key)}
            for name, (mean_key, p90_key) in stage_keys.items()
        },
        "primary_cam2_reprojection_px": metric(rows, "reprojection_mean_px"),
        "pose_change_from_converted_teacher_rotation_matrix_rms": metric(rows, "pose_change_from_teacher_matrix_rms"),
        "pelvis_change_from_converted_teacher_m": metric(rows, "pelvis_shift_from_teacher_m"),
        "ground_penetration_m": metric(rows, "floor_penetration_m"),
        "shape_max_abs": metric(rows, "shape_max_abs"),
        "shape_at_bound_count": sum(bool(row["qa"].get("shape_at_bound")) for row in rows),
        "cloud_return_count": {"mean": statistics.fmean([row["source_point_count"] for row in rows]) if rows else None,
                               "min": min((row["source_point_count"] for row in rows), default=None),
                               "max": max((row["source_point_count"] for row in rows), default=None)},
    }


def shape_audit(rows):
    tracks = defaultdict(list)
    failures = []
    for row in rows:
        try:
            with np.load(row["fit_path"]) as fit:
                shape = fit["shape"].copy()
            if shape.shape != (10,) or not np.isfinite(shape).all():
                raise ValueError("Expected ten finite neutral-SMPL shape coefficients")
            tracks[row["motion_id"]].append(shape)
        except (OSError, KeyError, ValueError) as exc:
            failures.append({"fit_path": row.get("fit_path"), "error": str(exc)})
    spreads = {track: float(np.ptp(np.stack(values), axis=0).max()) for track, values in tracks.items()}
    return {"audited_samples": sum(map(len, tracks.values())), "track_window_count": len(tracks),
            "max_within_track_window_beta_spread": max(spreads.values(), default=None),
            "track_windows_with_beta_spread_over_1e_6": sum(spread > 1e-6 for spread in spreads.values()),
            "failures": failures,
            "scope": "Same tracked segment/window only; person identity and shape consistency across windows are not established"}


def read_cam1_check(output_root, explicit_path, calibration_digest):
    path = Path(explicit_path) if explicit_path is not None else Path(output_root) / "sequence_review/cam1_check.json"
    if not path.exists():
        if explicit_path is not None:
            raise ValueError(f"Requested Cam1 report does not exist: {path}")
        return None
    payload = path.read_bytes()
    source = json.loads(payload.decode("utf-8"))
    if source.get("calibration_digest") != calibration_digest:
        raise ValueError("Cam1 check and surface audit use different frozen calibrations")
    provenance_keys = ("schema", "created_utc", "checker_sha256", "manifest", "manifest_sha256",
                       "calibration_digest", "projection_space", "not_ground_truth",
                       "not_fed_back_to_fitting", "association_selection_bias",
                       "calibration_independent_of_cam1_rgb", "unverified_visibility",
                       "confidence_threshold", "minimum_eligible_joints", "selection",
                       "original_exposure_accuracy_verified", "focal_normalization_definition",
                       "image_normalization_definition", "body_normalization_definition")
    return {
        "source_report": str(path.resolve()), "source_sha256": hashlib.sha256(payload).hexdigest(),
        "completed": source.get("completed", False), "summary": source.get("summary"),
        "per_clip": source.get("per_clip"),
        "provenance": {key: source.get(key) for key in provenance_keys},
        "scope": "Selected start/middle/end native Cam1 frames from the ten 5-second review clips; separate from all completed full-job surface/throughput statistics",
        "statistic_definition": "Cam1 mean/median/P90 pool qualified detector joints in selected frames; not mean frame-P90 and not 3D joint accuracy",
        "bias_limits": [
            "Independent Cam1 RGB detections were not fed back to fitting; detector observations are not manual ground truth",
            "Identity association and occlusion filtering depend on the fitted meshes and can bias the selected residuals",
            "Frozen calibration may already use other Cam1 RGB masks; camera calibration is not independent of all Cam1 imagery",
            "Cam1 and Cam2 use different focal lengths; raw pixel residuals cannot be compared directly",
            "Errors use undistorted Cam1 pixel coordinates with the original K; focal/image-diagonal/detection-bbox normalization is also recorded",
        ],
    }


def build_report(output_root, cam1_report=None):
    from hmr4d.lidar_mocap.export import acceptance_reasons

    root = Path(output_root).resolve()
    pairing = read_json(root / "pairing_report.json")
    status = read_json(root / "job_status.json")
    progress = read_json(root / "progress.json")
    dataset_path = root / "datasets/dataset_manifest.json"
    dataset = read_json(dataset_path) if dataset_path.exists() else None
    validated_windows = None
    if status['state'] == 'complete':
        if not dataset or dataset.get('calibration_digest') != pairing['calibration_digest']:
            raise ValueError('A completed job requires an export with the same frozen calibration')
        validated_windows = {(sid, window) for sid, windows in dataset['selected_windows'].items() for window in windows}
        if len(validated_windows) != status['completed_windows']:
            raise ValueError('Completed job and exported window counts disagree')
    windows, exclusions = completed_windows(root, pairing, progress, status["started_utc"], validated_windows)
    if not windows:
        raise ValueError("No completed full-run windows at the captured progress watermark")
    rows, passed, window_reports = [], [], []
    scan_totals = Counter()
    segmentation = Counter()
    rejection = Counter()
    consistency = []
    cached_fitted = cached_passed = cached_seconds = cached_windows = 0
    sessions = defaultdict(lambda: {"completed_windows": 0, "source_seconds": 0., "scans": 0,
                                   "fitted": 0, "accepted": 0})
    for config, report, directory in windows:
        quality = [json.loads(line) for line in (directory / "quality.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(quality) != report["fitted"]:
            raise ValueError(f"Incomplete quality records in {directory}")
        accepted = [row for row in quality if row.get("accepted", False)
                    and not acceptance_reasons(row["qa"], row["source_point_count"])]
        if len(accepted) != report["accepted"]:
            consistency.append({"window": str(directory), "stored_accepted": report["accepted"],
                                "current_policy_accepted": len(accepted)})
        rows.extend(quality)
        passed.extend(accepted)
        for row in quality:
            rejection.update(row.get("rejection_reasons", []))
        source_scans = set(config["source_pairs"])
        fitted_scans = {row["lidar_index"] for row in quality}
        accepted_scans = {row["lidar_index"] for row in accepted}
        segment_failures = report.get("segmentation_exclusions", [])
        segmentation.update(row["reason"] for row in segment_failures)
        excluded_scans = {row["lidar_index"] for row in segment_failures}
        scan_totals.update(paired_source_scans=len(source_scans), scans_with_fit=len(fitted_scans),
                           scans_with_accepted_fit=len(accepted_scans), scans_without_fit=len(source_scans - fitted_scans),
                           scans_without_fit_or_recorded_segmentation_failure=len(source_scans - fitted_scans - excluded_scans),
                           person_scan_segmentation_failures=len(segment_failures))
        duration = (config["right_us"] - config["left_us"]) / 1e6
        summary = {"session": config["session_id"], "window": config["window_id"],
                   "source_seconds": duration, "source_scans": len(source_scans),
                   "fitted": len(quality), "accepted": len(accepted),
                   "fit_algorithm_digest": report.get("fit_algorithm_digest"),
                   "settings": report.get("settings"), "fitting_config": report.get("fitting_config")}
        summary['reused_validated_cache'] = (directory/'fit_report.json').stat().st_mtime < parse_utc(status['started_utc']).timestamp()
        if summary['reused_validated_cache']:
            cached_windows += 1
            cached_fitted += len(quality)
            cached_passed += len(accepted)
            cached_seconds += duration
        window_reports.append(summary)
        session = sessions[config["session_id"]]
        session["completed_windows"] += 1
        session["source_seconds"] += duration
        session["scans"] += len(source_scans)
        session["fitted"] += len(quality)
        session["accepted"] += len(accepted)

    elapsed = (parse_utc(progress["updated_utc"]) - parse_utc(status["started_utc"])).total_seconds()
    if elapsed <= 0:
        raise ValueError("Completion watermark precedes job start")
    seconds = sum(window["source_seconds"] for window in window_reports)
    if validated_windows is not None and (consistency or len(passed) != dataset['accepted']):
        raise ValueError('Complete quality totals do not match the validated exported dataset')
    cam1_check = read_cam1_check(root, cam1_report, pairing["calibration_digest"])
    return {
        "schema": "badminton-quality-efficiency-report-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(), "output_root": str(root),
        "calibration_digest": pairing["calibration_digest"],
        "job_state_at_capture": status["state"], "progress_watermark": progress,
        "scope": ("All windows selected by the completed validated export, including reused caches; smoke runs excluded"
                  if validated_windows is not None else
                  "Only completed regular full-run windows at the captured progress timestamp; smoke runs and active/incomplete windows excluded"),
        "completed_windows": window_reports, "excluded_window_counts": exclusions,
        "sessions": dict(sessions),
        "sample_counts": {"all_fitted": len(rows), "passed": len(passed), "rejected": len(rows) - len(passed),
                          "acceptance_fraction": len(passed) / len(rows) if rows else None},
        "scan_and_segmentation_counts": dict(scan_totals), "segmentation_failure_reasons": dict(segmentation),
        "fit_rejection_reasons": dict(rejection), "acceptance_consistency_issues": consistency,
        "quality": {"all_fitted": quality_group(rows), "passed": quality_group(passed)},
        "shape_audit": shape_audit(rows),
        "efficiency": {"started_utc": status["started_utc"], "completed_watermark_utc": progress["updated_utc"],
                       "wall_seconds": elapsed, "completed_source_seconds": seconds-cached_seconds,
                       "source_seconds_per_wall_second": (seconds-cached_seconds) / elapsed,
                       "wall_seconds_per_source_second": elapsed / (seconds-cached_seconds) if seconds>cached_seconds else None,
                       "fitted_person_scans_per_wall_second": (len(rows)-cached_fitted) / elapsed,
                       "passed_person_scans_per_wall_second": (len(passed)-cached_passed) / elapsed,
                       "validated_cache_windows": cached_windows, "validated_cache_fitted": cached_fitted,
                       "validated_cache_passed": cached_passed, "validated_cache_source_seconds": cached_seconds,
                       "total_export_source_seconds": seconds,
                       "scope": "Pairing-run wall time through the progress watermark includes teacher, fitting and source I/O; reused pre-job fit caches are included in quality but excluded from throughput numerators. Teacher feature caches may also be reused. Later dataset export, verification and final continuous reviews are excluded; this is not uncached model inference throughput"},
        "cam1_rgb_check": cam1_check,
        "current_aggregate_manifest": {"path": str(dataset_path), "accepted": dataset.get("accepted"),
                                       "selected_windows": dataset.get("selected_windows")}
                                      if dataset else None,
        "measurement_limits": [
            "Surface values equally average each athlete/scan statistic; mean frame-p90 is not a pooled recording-wide 90th percentile",
            "Teacher initialization is mapped SMPL-X geometry before LiDAR translation; converted and centre-aligned initialization p90 may be absent in existing caches",
            "cam2 2D reprojection is an optimization constraint, not an independent validation set; Cam1 detector consistency, when provided, covers only selected review frames",
            "Cam1 detector pixel residuals are not 3D joint accuracy; association/occlusion selection can depend on fitted meshes and calibration may use other Cam1 masks",
            "Cam1/Cam2 pixel coordinates are undistorted with their own K; different focal lengths prevent direct raw-pixel comparison",
            "Surface residuals measure observed returns only; unobserved body parts, pose and shape have no independent 3D ground truth or MPJPE validation",
            "Pose-change rotation-matrix RMS is dimensionless, not degrees or pose accuracy",
            "Track/window shape sharing is audited; cross-window person identity/shape consistency is not established",
            "Device-reported timestamps are paired; physical exposure accuracy and per-scan motion deskew are unverified",
        ],
    }


def markdown_report(report):
    counts = report["sample_counts"]
    speed = report["efficiency"]
    lines = [
        f"截至 {speed['completed_watermark_utc']}，已完成 {len(report['completed_windows'])} 个正式窗口、"
        f"{speed.get('total_export_source_seconds',speed['completed_source_seconds']):.1f} 秒源数据：拟合 {counts['all_fitted']} 条，"
        f"通过 {counts['passed']} 条，拒绝 {counts['rejected']} 条（通过率 {counts['acceptance_fraction']:.1%}）。",
        "", "当前正在计算的窗口及旧 smoke 片段未计入。", "",
        f"从 job_status.started_utc 到 progress.updated_utc 的实际壁钟为 {speed['wall_seconds']/60:.2f} 分钟。"
        f"每秒壁钟完成 {speed['source_seconds_per_wall_second']:.3f} 秒源数据，"
        f"吞吐 {speed['fitted_person_scans_per_wall_second']:.2f} 条拟合/秒、"
        f"{speed['passed_person_scans_per_wall_second']:.2f} 条通过/秒；含 teacher、拟合与 I/O，后续导出和连续预览不在此壁钟内。",
        f"质量统计含 {speed.get('validated_cache_windows',0)} 个已验证的历史缓存窗口"
        f"（{speed.get('validated_cache_passed',0)} 条通过样本）；吞吐分子排除这些窗口。视频特征也可能复用缓存，因此不是无缓存推理速度。", "",
        "| 点到全三角面距离 | 全部拟合 mean / 平均每帧P90（mm） | 通过样本 mean / 平均每帧P90（mm） |",
        "|---|---:|---:|",
    ]
    labels = {
        "teacher_mapped_before_lidar_translation": "视频初始化，未经点云平移",
        "smpl_topology_converted_before_lidar_translation": "SMPL拓扑转换后，未经点云平移",
        "cloud_center_aligned_before_optimization": "按点云中心对齐后，尚未拟合",
        "final_smpl_all_triangles": "最终SMPL，仅可见点单向距离",
    }
    def pair(group, stage):
        value = report["quality"][group]["surface_stages"][stage]
        mean = value["frame_mean_m"]["mean"]
        p90 = value["frame_p90_m"]["mean"]
        left = f"{mean*1000:.2f}" if mean is not None else "未记录"
        right = f"{p90*1000:.2f}" if p90 is not None else "未记录"
        return f"{left} / {right}"
    for name, label in labels.items():
        lines.append(f"| {label} | {pair('all_fitted', name)} | {pair('passed', name)} |")
    passed = report["quality"]["passed"]
    projection = passed["primary_cam2_reprojection_px"]["mean"]
    pose = passed["pose_change_from_converted_teacher_rotation_matrix_rms"]["mean"]
    audit = report["shape_audit"]
    lines.extend([
        "", "mean 是每个人/帧平均距离的等权平均；表中 P90 是每帧 P90 的平均，并非所有点合并后的 P90。"
        "中心对齐及转换后的初始化 P90 若旧缓存未记录，此处明确留空。", "",
        f"通过样本的 cam2 重投影均值为 {projection:.2f}px；cam2 是拟合约束，不能作为独立验证精度。"
        if projection is not None else "通过样本 cam2 重投影未记录。",
        f"相对转换后视频先验的 body rotation-matrix RMS 改动均值为 {pose:.4f}（无量纲，不代表角度或真实姿态误差）。"
        if pose is not None else "姿态改动统计未记录。",
        f"已读取 {audit['audited_samples']} 份 beta；{audit['track_window_count']} 条窗口轨迹的最大形态系数跨度"
        f"为 {audit['max_within_track_window_beta_spread']:.6g}，{audit['track_windows_with_beta_spread_over_1e_6']} 条存在 >1e-6 漂移。"
        if audit["max_within_track_window_beta_spread"] is not None else "没有可审计的形态文件。",
        "形态只在同一窗口轨迹内共享；跨窗口身份和形态一致性尚未建立。", "",
        f"配对扫描 {report['scan_and_segmentation_counts'].get('paired_source_scans',0)} 帧，"
        f"无任何拟合的扫描 {report['scan_and_segmentation_counts'].get('scans_without_fit',0)} 帧；"
        f"已记录人物/扫描分割失败 {report['scan_and_segmentation_counts'].get('person_scan_segmentation_failures',0)} 次。",
        f"拟合拒绝原因：{json.dumps(report['fit_rejection_reasons'],ensure_ascii=False)}。",
        f"分割失败原因：{json.dumps(report['segmentation_failure_reasons'],ensure_ascii=False)}。", "",
        "点云残差和 cam2 重投影均来自拟合约束；没有独立 3D 姿态真值，遮挡部位质量及 MPJPE 尚无法量化。",
    ])
    cam1 = report.get("cam1_rgb_check")
    if cam1 and cam1.get("completed") and cam1.get("summary"):
        summary = cam1["summary"]
        pixel = summary["pixel_error"]
        focal = summary["focal_normalized_error"]["mean"]
        lines.extend([
            "",
            f"另对 {summary['checked_clips']} 个连续 5 秒 review 片段各取开始/中间/结束帧，"
            f"检查 {summary['checked_frames']} 个原生 Cam1 帧；"
            f"{summary['measured_athlete_frames']}/{summary['athlete_frame_candidates']} 个人物帧可量化，"
            f"合格关节 {summary['eligible_joints_in_measured_frames']} 个，"
            f"候选 COCO17 关节覆盖率 {summary['coverage_of_candidate_17_joints']:.1%}。"
            "这项检查范围与上述已完成 full-job 总体点云统计分开。",
            f"选定帧的 Cam1 检测关节残差 mean/median/P90 为 "
            f"{pixel['mean']:.3f}/{pixel['median']:.3f}/{pixel['p90']:.3f}px；"
            f"按焦距归一化后的 mean 为 {focal:.6f}。"
            "此 P90 汇总合格关节，和上表平均每帧 P90 的统计口径不同。",
            "检测使用去畸变 Cam1 图像并保留原 K，RGB 检测没有回馈拟合。"
            "它检验与检测器的一致性，不是人工 GT 或 3D 关节精度；"
            "身份关联及遮挡筛选依赖拟合网格，标定也可能用过其他 Cam1 RGB mask，不能称严格无偏验证。"
            "两相机焦距不同，不能直接比较 Cam1 与 Cam2 的原始像素残差。",
            f"逐帧/逐关节依据及偏差说明见 [{Path(cam1['source_report']).name}]({Path(cam1['source_report']).as_posix()})。",
        ])
    elif cam1:
        lines.extend(["", "已读取 Cam1 检查报告，但检查尚未完成；本报告不发布其临时精度结果。"])
    else:
        lines.extend(["", "本次报告未加载独立 Cam1 RGB 检查结果。"])
    if report["acceptance_consistency_issues"]:
        lines.extend(["", f"发现接受数量与当前质量门禁不一致：{report['acceptance_consistency_issues']}"])
    if audit["failures"]:
        lines.extend(["", f"形态读取失败 {len(audit['failures'])} 份，详情见 JSON。"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/badminton_corner_fix_v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cam1-report", type=Path,
                        help="Optional Cam1 RGB check JSON; defaults to output-root/sequence_review/cam1_check.json if present")
    args = parser.parse_args()
    report = build_report(args.output_root, args.cam1_report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "quality_efficiency.json"
    markdown_path = args.output_dir / "quality_efficiency.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path.resolve()), "markdown": str(markdown_path.resolve()),
                      "counts": report["sample_counts"], "efficiency": report["efficiency"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
