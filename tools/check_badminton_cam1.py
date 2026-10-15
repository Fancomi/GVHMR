"""CPU-only RGB keypoint observations on paired Cam1 or Cam2 review frames.

This never runs HMR2/GVHMR or optimizes the fitted body. YOLO person detections
are associated one-to-one with the fixed fitted meshes using bbox IoU, then
ViTPose observes the original native video after calibration-based undistortion.
These are detector consistency measurements, not human-labelled ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

# Set before importing torch or its dependencies. This process must not create
# a CUDA context while the production pairing worker is using the GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from hmr4d import PROJ_ROOT as PROJECT_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.sessions import SessionIndex


JOINT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
               "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
               "left_wrist", "right_wrist", "left_hip", "right_hip",
               "left_knee", "right_knee", "left_ankle", "right_ankle"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {"count": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9))}


def bbox_iou(left, right):
    lower = np.maximum(left[:2], right[:2])
    upper = np.minimum(left[2:], right[2:])
    intersection = float(np.prod(np.maximum(upper - lower, 0)))
    denominator = float(np.prod(left[2:] - left[:2]) + np.prod(right[2:] - right[:2]) - intersection)
    return intersection / max(denominator, 1e-9)


def clip_box(uv, depth, width, height):
    valid = np.isfinite(uv).all(1) & (depth > 0)
    if valid.sum() < 10:
        return None
    low, high = uv[valid].min(0), uv[valid].max(0)
    low = np.maximum(low, [0, 0])
    high = np.minimum(high, [width - 1, height - 1])
    if np.any(high - low < 20):
        return None
    return np.concatenate([low, high]).astype(np.float32)


def associate_boxes(mesh_boxes, detection_boxes, minimum_iou=.1, minimum_margin=.08):
    """Association uses fixed boxes, never chooses a match by keypoint error."""
    if not mesh_boxes or not len(detection_boxes):
        return {}, {}
    ious = np.array([[bbox_iou(a, b) for b in detection_boxes] for a in mesh_boxes])
    rows, columns = linear_sum_assignment(1 - ious)
    matches, rejected = {}, {}
    for row, column in zip(rows, columns):
        score = float(ious[row, column])
        alternatives = np.concatenate([np.delete(ious[row], column), np.delete(ious[:, column], row)])
        margin = float(score - alternatives.max()) if len(alternatives) else 1.0
        if score < minimum_iou:
            rejected[int(row)] = "no_independent_detection_with_sufficient_bbox_iou"
        elif margin < minimum_margin:
            rejected[int(row)] = "ambiguous_person_association"
        else:
            matches[int(row)] = {"detection_index": int(column), "bbox_iou": score,
                                 "association_margin": margin}
    return matches, rejected


def normalized_crop(rgb, bbox):
    import torch

    center = (bbox[:2] + bbox[2:]) / 2
    width, height = bbox[2:] - bbox[:2]
    size = max(float(height), float(width) * 4 / 3) * 1.2
    half = size / 2
    source = np.stack([center - half, center + [half, -half], center]).astype(np.float32)
    destination = np.array([[0, 0], [255, 0], [127.5, 127.5]], dtype=np.float32)
    transform = cv2.getAffineTransform(source, destination)
    crop = cv2.warpAffine(rgb, transform, (256, 256), flags=cv2.INTER_LINEAR)
    image = torch.from_numpy(crop.copy()).float() / 255
    image = (image - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
    return image.permute(2, 0, 1), np.array([*center, size], dtype=np.float32)


def to_raw_brown_pixels(keypoints, camera):
    """Map fixed-baseline rectified observations back to the original image.

    This changes only pixel coordinates; timestamp/confidence remain measured
    at the same native frame. Candidate K/d refinements should compare their
    raw Brown projection with these fixed original-image coordinates.
    """
    points = np.asarray(keypoints, dtype=np.float64)
    K = np.asarray(camera["K"], dtype=np.float64)
    rays = np.column_stack([points[:, :2], np.ones(len(points))]) @ np.linalg.inv(K).T
    rays /= rays[:, 2:3]
    x, y = rays[:, 0], rays[:, 1]
    k1, k2, p1, p2, k3 = camera["distortion_opencv"]
    r2 = x * x + y * y
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    pixels = np.column_stack([xd, yd, np.ones(len(points))]) @ K.T
    result = points.copy()
    result[:, :2] = pixels[:, :2] / pixels[:, 2:3]
    return result.tolist()


def add_raw_pixel_observations(report):
    camera = report["camera"]
    for frame in report["frames"]:
        for observed in frame.get("detector_observations", []):
            if observed.get("coco17_keypoints") is not None:
                observed["coco17_keypoints_raw_distorted"] = to_raw_brown_pixels(observed["coco17_keypoints"], camera)
        for athlete in frame["athletes"]:
            if athlete.get("all_coco17_observations") is not None:
                athlete["all_coco17_observations_raw_distorted"] = to_raw_brown_pixels(athlete["all_coco17_observations"], camera)
    report["raw_pixel_mapping"] = "Undistorted detector coordinates mapped with the frozen baseline K/Brown d; same native observation timestamp and confidence"


class Cam1Check:
    def __init__(self, calibration, threads=8, detector_imgsz=960, flip_test=False, camera_name="cam1"):
        import torch
        from ultralytics import YOLO
        from hmr4d.utils.preproc.vitpose_pytorch import build_model

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(2)
        self.calibration = calibration
        self.camera_name = camera_name
        self.camera = calibration.camera(camera_name)
        self.detector_imgsz = detector_imgsz
        self.flip_test = flip_test
        self.confidence_threshold = .5
        self.minimum_joints = 6
        self.sessions = {}
        self.yolo = YOLO(PROJECT_ROOT / "inputs/checkpoints/yolo/yolov8x.pt")
        self.pose = build_model("ViTPose_huge_coco_256x192", checkpoint=None).cpu().eval()
        weights = PROJECT_ROOT / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
        state = torch.load(weights, map_location="cpu")["state_dict"]
        self.pose.load_state_dict(state)
        del state
        self.regressor = torch.load(PROJECT_ROOT / "hmr4d/utils/body_model/smpl_coco17_J_regressor.pt",
                                    map_location="cpu").float().numpy()
        K = np.asarray(self.camera["K"], dtype=np.float64)
        distortion = np.asarray(self.camera["distortion_opencv"], dtype=np.float64)
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            K, distortion, None, K, tuple(self.camera["image_size"]), cv2.CV_32FC1)
        # Static asset hashes make this detector check reproducible.
        self.models = {"vitpose": {"path": str(weights), "sha256": sha256_file(weights)},
                       "yolo": {"path": str(PROJECT_ROOT / "inputs/checkpoints/yolo/yolov8x.pt"),
                                "sha256": sha256_file(PROJECT_ROOT / "inputs/checkpoints/yolo/yolov8x.pt")},
                       "coco17_regressor": {"path": str(PROJECT_ROOT / "hmr4d/utils/body_model/smpl_coco17_J_regressor.pt"),
                                            "sha256": sha256_file(PROJECT_ROOT / "hmr4d/utils/body_model/smpl_coco17_J_regressor.pt")},
                       "smpl_triangle_topology": {"path": str(PROJECT_ROOT / "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"),
                                                 "sha256": sha256_file(PROJECT_ROOT / "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl")}}
        self.faces = None
        # SMPL is used only to read its constant face topology; no body forward,
        # fitting, HMR2 or GVHMR inference occurs in this check.
        import smplx
        topology = smplx.create(PROJECT_ROOT / "inputs/checkpoints/body_models", model_type="smpl",
                                gender="neutral", num_betas=10)
        self.faces = np.asarray(topology.faces, dtype=np.int64)
        del topology
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU check unexpectedly created a CUDA context")

    def session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionIndex.load(Path(self.calibration.data["sessions_root"]) / session_id)
        return self.sessions[session_id]

    def pose_keypoints(self, rgb, boxes):
        import torch
        from hmr4d.utils.kpts.kp2d_utils import keypoints_from_heatmaps
        from hmr4d.utils.geo.flip_utils import flip_heatmap_coco17

        if not len(boxes):
            return np.empty((0, 17, 3), dtype=np.float32)
        images, xys = zip(*(normalized_crop(rgb, box) for box in boxes))
        images = torch.stack(images)[:, :, :, 32:224]
        xys = np.stack(xys)
        with torch.inference_mode():
            parts = []
            for start in range(0, len(images), 2):
                batch = images[start:start + 2]
                if self.flip_test:
                    heatmap, mirrored = self.pose(torch.cat([batch, batch.flip(3)])).chunk(2)
                    heatmap = (heatmap + flip_heatmap_coco17(mirrored)) / 2
                else:
                    heatmap = self.pose(batch)
                parts.append(heatmap.float().numpy())
        heatmap = np.concatenate(parts)
        center = xys[:, :2]
        scale = np.stack([xys[:, 2] * .75, xys[:, 2]], axis=1) / 200
        predictions, confidence = keypoints_from_heatmaps(heatmap, center, scale, use_udp=True)
        return np.concatenate([predictions, confidence], axis=-1).astype(np.float32)

    def _other_mesh_occludes(self, joint_uv, joint_depth, meshes_camera, own_index):
        """Conservative foreign-athlete mesh ray test; no GT visibility claim."""
        K = np.asarray(self.camera["K"], dtype=np.float64)
        pixels = np.column_stack([joint_uv, np.ones(len(joint_uv))])
        rays = pixels @ np.linalg.inv(K).T
        blocked = np.zeros(len(rays), dtype=bool)
        for index, vertices in enumerate(meshes_camera):
            if index == own_index:
                continue
            triangles = vertices[self.faces]
            a = triangles[:, 0]
            edge1, edge2 = triangles[:, 1] - a, triangles[:, 2] - a
            for joint, ray in enumerate(rays):
                if not np.isfinite(ray).all() or joint_depth[joint] <= 0:
                    continue
                h = np.cross(ray[None], edge2)
                determinant = np.sum(edge1 * h, axis=1)
                valid = np.abs(determinant) > 1e-9
                inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=valid)
                s = -a
                u = inverse * np.sum(s * h, axis=1)
                q = np.cross(s, edge1)
                v = inverse * np.sum(ray[None] * q, axis=1)
                depth = inverse * np.sum(edge2 * q, axis=1)
                hit = valid & (u >= 0) & (v >= 0) & (u + v <= 1) & (depth > .001)
                blocked[joint] |= bool(np.any(hit & (depth < joint_depth[joint] - .10)))
        return blocked

    def check_frame(self, clip, frame):
        started = time.perf_counter()
        session = self.session(clip["session_id"])
        camera_row = frame["cameras"][self.camera_name]
        native = session._camera_rows[self.camera_name][camera_row["record_index"]]
        for key in ("record_index", "media_pts_us", "common_time_us"):
            if int(camera_row[key]) != native[key]:
                raise ValueError(f"Sequence manifest has mismatched {self.camera_name} {key}")
        if int(camera_row["common_time_us"]) - int(frame["scan_mid_us"]) != int(camera_row["time_error_us"]):
            raise ValueError("Sequence camera time error does not equal source time minus scan midpoint")
        # Read unmodified native video, never the rendered review JPEG.
        raw_bgr = session.read_image(self.camera_name, camera_row["record_index"])
        rgb = cv2.remap(raw_bgr[:, :, ::-1], self.map_x, self.map_y, cv2.INTER_LINEAR)
        width, height = self.camera["image_size"]
        prepared = []
        for athlete in frame["athletes"]:
            path = Path(athlete["fit_path"])
            if not path.is_absolute():
                path = Path(clip["output_root"]) / path
            with np.load(path, allow_pickle=False) as fit:
                vertices = fit["vertices"].astype(np.float64)
            if vertices.shape != (6890, 3) or not np.isfinite(vertices).all():
                raise ValueError(f"Expected finite fitted neutral SMPL mesh: {path}")
            vertex_uv, depth = self.calibration.project(vertices, self.camera_name, distort=False)
            joints = self.regressor @ vertices
            joint_uv, joint_depth = self.calibration.project(joints, self.camera_name, distort=False)
            prepared.append({"athlete": athlete, "fit_path": str(path), "fit_sha256": sha256_file(path),
                             "bbox": clip_box(vertex_uv, depth, width, height),
                             "joints_uv": joint_uv, "joints_depth": joint_depth, "joints_viewer": joints,
                             "vertices_camera": self.calibration.viewer_to_camera(vertices, self.camera_name)})
        visible = [index for index, value in enumerate(prepared) if value["bbox"] is not None]
        before_detection = time.perf_counter()
        detected = self.yolo.predict(rgb[:, :, ::-1].copy(), classes=[0], conf=.35,
                                     device="cpu", imgsz=self.detector_imgsz, verbose=False)[0]
        boxes = detected.boxes.xyxy.cpu().numpy().astype(np.float32)
        confidences = detected.boxes.conf.cpu().numpy().astype(np.float32)
        detection_seconds = time.perf_counter() - before_detection
        matches, ambiguous = associate_boxes([prepared[index]["bbox"] for index in visible], boxes)
        # Preserve candidate RGB observations even if one-to-one association is
        # ambiguous. Those raw measurements remain diagnostic data; ambiguity
        # still excludes them from the reported aggregate fit consistency.
        candidates = {index: [j for j, box in enumerate(boxes)
                              if bbox_iou(prepared[index]["bbox"], box) >= .1] for index in visible}
        detection_indices = sorted({j for indices in candidates.values() for j in indices})
        before_pose = time.perf_counter()
        observations = self.pose_keypoints(rgb, boxes[detection_indices])
        pose_seconds = time.perf_counter() - before_pose
        observations = {index: observation for index, observation in zip(detection_indices, observations)}
        matches = {visible[index]: value for index, value in matches.items()}
        ambiguous = {visible[index]: value for index, value in ambiguous.items()}
        records = []
        meshes_camera = [item["vertices_camera"] for item in prepared]
        for index, item in enumerate(prepared):
            athlete = item["athlete"]
            result = {"motion_id": athlete["motion_id"], "fit_path": item["fit_path"],
                      "fit_sha256": item["fit_sha256"], "fit_accepted": athlete.get("accepted"),
                      "fixed_fit_projected_bbox_xyxy": None if item["bbox"] is None else item["bbox"].tolist(),
                      "fixed_smpl_coco17_viewer_m": item["joints_viewer"].tolist(),
                      "fixed_smpl_geometry_time_us": int(frame["scan_mid_us"]),
                      "all_coco17_fixed_fit_projection": item["joints_uv"].tolist(),
                      "all_coco17_observations": None, "detector_index": None,
                      "association_candidate_detection_indices": candidates.get(index, []),
                      "eligible_mask": [False] * 17,
                      "cam2_optimization_residual_px": athlete.get("qa", {}).get("reprojection_by_camera_px", {}).get("cam2"),
                      "eligible_joint_count": 0, "in_view_joint_count": 0, "joint_metrics": [],
                      "eligible_joint_coverage_of_17": 0., "rejection_reasons": []}
            if item["bbox"] is None:
                result.update(status="excluded", rejection_reasons=["fitted_body_outside_selected_camera_or_insufficient_image_extent"])
                records.append(result)
                continue
            if index not in matches:
                result.update(status="excluded", rejection_reasons=[ambiguous.get(index, "no_unique_independent_rgb_detection")])
                records.append(result)
                continue
            match = matches[index]
            detected_index = match["detection_index"]
            observation = observations[detected_index]
            projected = item["joints_uv"]
            finite = np.isfinite(projected).all(1) & np.isfinite(observation).all(1)
            in_view = finite & (item["joints_depth"] > 0)
            in_view &= (projected >= [5, 5]).all(1) & (projected <= [width - 6, height - 6]).all(1)
            observed_in_view = (observation[:, :2] >= [5, 5]).all(1)
            observed_in_view &= (observation[:, :2] <= [width - 6, height - 6]).all(1)
            occluded = self._other_mesh_occludes(projected, item["joints_depth"], meshes_camera, index)
            eligible = in_view & observed_in_view & (observation[:, 2] >= self.confidence_threshold) & ~occluded
            errors = np.linalg.norm(projected - observation[:, :2], axis=1)
            residual = projected - observation[:, :2]
            focal = np.array([self.camera["K"][0][0], self.camera["K"][1][1]])
            normalized_focal = np.linalg.norm(residual / focal, axis=1)
            image_normalized = errors / np.hypot(width, height)
            body_scale = max(float(boxes[detected_index, 3] - boxes[detected_index, 1]), 1.)
            body_normalized = errors / body_scale
            result.update(independent_detection_bbox_xyxy=boxes[detected_index].tolist(),
                          detector_index=int(detected_index), eligible_mask=eligible.tolist(),
                          independent_detection_confidence=float(confidences[detected_index]),
                          association_bbox_iou=match["bbox_iou"], association_margin=match["association_margin"],
                          crop_source="independent_cpu_YOLO_detection_after_unique_fixed_fit_bbox_association",
                          eligible_joint_count=int(eligible.sum()), in_view_joint_count=int(in_view.sum()),
                          other_athlete_occluded_joint_count=int((occluded & in_view).sum()),
                          low_confidence_joint_count=int(((observation[:, 2] < .5) & in_view).sum()),
                          eligible_joint_coverage_of_17=float(eligible.mean()),
                          pixel_error=statistics(errors[eligible]), focal_normalized_error=statistics(normalized_focal[eligible]),
                          image_diagonal_normalized_error=statistics(image_normalized[eligible]),
                          independent_bbox_height_normalized_error=statistics(body_normalized[eligible]),
                          all_coco17_observations=observation.tolist(), all_coco17_fixed_fit_projection=projected.tolist())
            for joint in range(17):
                reasons = []
                if not in_view[joint] or not observed_in_view[joint]:
                    reasons.append("outside_image_or_truncated")
                if observation[joint, 2] < .5:
                    reasons.append("confidence_below_0.5")
                if occluded[joint]:
                    reasons.append("behind_other_fitted_athlete_mesh")
                result["joint_metrics"].append({"joint": JOINT_NAMES[joint], "joint_index": joint,
                    "eligible": bool(eligible[joint]), "confidence": float(observation[joint, 2]),
                    "pixel_error": float(errors[joint]) if finite[joint] else None,
                    "focal_normalized_error": float(normalized_focal[joint]) if finite[joint] else None,
                    "image_diagonal_normalized_error": float(image_normalized[joint]) if finite[joint] else None,
                    "bbox_height_normalized_error": float(body_normalized[joint]) if finite[joint] else None,
                    "excluded_reasons": reasons})
            if int(eligible.sum()) < self.minimum_joints:
                result.update(status="excluded", rejection_reasons=["fewer_than_6_eligible_joints"])
            else:
                result.update(status="measured")
            records.append(result)
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU independent keypoint check unexpectedly created a CUDA context")
        record = {"clip_id": clip["id"], "session_id": clip["session_id"], "scan_mid_us": frame["scan_mid_us"],
                "lidar_index": frame["lidar_index"], "offset_s": frame.get("offset_s"),
                "camera_name": self.camera_name, "raw_camera_video": str(session.video_paths[self.camera_name]),
                "camera_native_source": native, "observation_common_time_us": native["common_time_us"],
                "fixed_geometry_common_time_us": int(frame["scan_mid_us"]), "geometry_interpolated": False,
                "paired_cameras": frame["cameras"], "image_source": "original_native_video_not_review_overlay",
                "native_bgr_image_sha256": hashlib.sha256(raw_bgr.tobytes()).hexdigest(),
                "undistorted_rgb_image_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "image_space": f"undistorted_{self.camera_name}_pixels_K_preserved", "image_original_distortion": self.camera["distortion_opencv"],
                "image_check_distortion": [0.] * 5, "athletes": records,
                "detector_boxes_xyxy": boxes.tolist(), "detector_confidences": confidences.tolist(),
                "detector_observations": [{"detector_index": j, "bbox_xyxy": boxes[j].tolist(),
                    "detection_confidence": float(confidences[j]),
                    "coco17_keypoints": observations[j].tolist() if j in observations else None,
                    "pose_inferred": j in observations,
                    "pose_selection": "fixed_mesh_bbox_iou_ge_0.1_candidate" if j in observations else "outside_fitted_person_candidates"}
                    for j in range(len(boxes))],
                "timing_seconds": {"detector": detection_seconds, "pose": pose_seconds,
                                   "total": time.perf_counter() - started}, "cuda_context_initialized": False}
        if self.camera_name == "cam1":
            record.update(raw_cam1_video=record["raw_camera_video"], cam1_native_source=native)
        return record


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
