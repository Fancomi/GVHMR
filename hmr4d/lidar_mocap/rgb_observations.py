"""Independent native RGB observations for fixed-SMPL body quality checks.

CPU is the default; explicitly requested CUDA accelerates detection/pose only.
Models, cropping, pixel spaces and association are shared with the production
RGB consistency CLI. This module never optimizes cameras, bodies or point clouds.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from hmr4d import PROJ_ROOT as PROJECT_ROOT
from .sessions import SessionIndex

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


class RGBKeypointChecker:
    def __init__(self, calibration, threads=8, detector_imgsz=960, flip_test=False, camera_name="cam1",
                 device="cpu", pose_batch_size=2, observe_pose=True):
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        cv2.setNumThreads(2)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda") or pose_batch_size < 1:
            raise ValueError("RGB checker requires CPU/CUDA and positive pose batch size")
        self.pose_batch_size = pose_batch_size
        self.observe_pose = observe_pose
        self.inference_device = "cpu" if self.device.type == "cpu" else (self.device.index or 0)
        self.calibration = calibration
        self.camera_name = camera_name
        self.camera = calibration.camera(camera_name)
        self.detector_imgsz = detector_imgsz
        self.flip_test = flip_test
        self.confidence_threshold = .5
        self.minimum_joints = 6
        self.sessions = {}
        self.yolo = YOLO(PROJECT_ROOT / "inputs/checkpoints/yolo/yolov8x.pt")
        weights = PROJECT_ROOT / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
        if observe_pose:
            from hmr4d.utils.preproc.vitpose_pytorch import build_model
            self.pose = build_model("ViTPose_huge_coco_256x192", checkpoint=None).to(self.device).eval()
            state = torch.load(weights, map_location="cpu")["state_dict"]
            self.pose.load_state_dict(state)
            del state
        self.regressor = torch.load(PROJECT_ROOT / "hmr4d/utils/body_model/smpl_coco17_J_regressor.pt",
                                    map_location="cpu").float().numpy()
        self.select_camera(camera_name)
        # Static asset hashes make this detector check reproducible.
        self.models = {"yolo": {"path": str(PROJECT_ROOT / "inputs/checkpoints/yolo/yolov8x.pt"),
                                "sha256": sha256_file(PROJECT_ROOT / "inputs/checkpoints/yolo/yolov8x.pt")},
                       "coco17_regressor": {"path": str(PROJECT_ROOT / "hmr4d/utils/body_model/smpl_coco17_J_regressor.pt"),
                                            "sha256": sha256_file(PROJECT_ROOT / "hmr4d/utils/body_model/smpl_coco17_J_regressor.pt")}}
        if observe_pose:
            topology_path = PROJECT_ROOT / "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"
            self.models.update(vitpose=dict(path=str(weights),sha256=sha256_file(weights)),
                               smpl_triangle_topology=dict(path=str(topology_path),sha256=sha256_file(topology_path)))
        self.faces = None
        # SMPL is used only to read its constant face topology; no body forward,
        # fitting, HMR2 or GVHMR inference occurs in this check.
        if observe_pose:
            import smplx
            topology = smplx.create(PROJECT_ROOT / "inputs/checkpoints/body_models", model_type="smpl",
                                    gender="neutral", num_betas=10)
            self.faces = np.asarray(topology.faces, dtype=np.int64)
            del topology
        if self.device.type == "cpu" and torch.cuda.is_initialized():
            raise RuntimeError("CPU check unexpectedly created a CUDA context")

    def select_camera(self, camera_name):
        self.camera_name = camera_name
        self.camera = self.calibration.camera(camera_name)
        K = np.asarray(self.camera["K"], dtype=np.float64)
        distortion = np.asarray(self.camera["distortion_opencv"], dtype=np.float64)
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            K, distortion, None, K, tuple(self.camera["image_size"]), cv2.CV_32FC1)

    def session(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionIndex.load(Path(self.calibration.data["sessions_root"]) / session_id,
                                                          self.calibration.frame_map_path(session_id))
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
            for start in range(0, len(images), self.pose_batch_size):
                batch = images[start:start + self.pose_batch_size].to(self.device)
                if self.flip_test:
                    heatmap, mirrored = self.pose(torch.cat([batch, batch.flip(3)])).chunk(2)
                    heatmap = (heatmap + flip_heatmap_coco17(mirrored)) / 2
                else:
                    heatmap = self.pose(batch)
                parts.append(heatmap.float().cpu().numpy())
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
            # Exact broad phase for meshes entirely in front of the camera:
            # their projected triangles cannot extend outside the vertex box.
            # Most court athletes do not overlap in the image; avoid thousands
            # of triangle tests for rays that provably cannot hit this person.
            possible = np.ones(len(rays), dtype=bool)
            if (vertices[:, 2] > 0).all():
                homogeneous = vertices @ K.T
                projected = homogeneous[:, :2] / homogeneous[:, 2:3]
                possible = ((joint_uv >= projected.min(0)-1e-6).all(1)
                            & (joint_uv <= projected.max(0)+1e-6).all(1))
                if not possible.any():
                    continue
            triangles = vertices[self.faces]
            a = triangles[:, 0]
            edge1, edge2 = triangles[:, 1] - a, triangles[:, 2] - a
            for joint, ray in enumerate(rays):
                if not possible[joint] or not np.isfinite(ray).all() or joint_depth[joint] <= 0:
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

    def check_frame(self, clip, frame, raw_bgr=None, cached_detections=None):
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
        if raw_bgr is None:
            raw_bgr = session.read_image(self.camera_name, camera_row["record_index"])
        if raw_bgr.shape != (self.camera["image_size"][1], self.camera["image_size"][0], 3):
            raise ValueError("Native image dimensions differ from the frozen camera")
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
        if cached_detections is None:
            detected = self.yolo.predict(rgb[:, :, ::-1].copy(), classes=[0], conf=.35,
                                         device=self.inference_device, imgsz=self.detector_imgsz, verbose=False)[0]
            boxes = detected.boxes.xyxy.cpu().numpy().astype(np.float32)
            confidences = detected.boxes.conf.cpu().numpy().astype(np.float32)
        else:
            boxes = np.asarray([d['bbx_xyxy'] for d in cached_detections],dtype=np.float32).reshape(-1,4)
            confidences = np.asarray([d['detection_confidence'] for d in cached_detections],dtype=np.float32)
            if not np.isfinite(boxes).all() or not np.isfinite(confidences).all() or not (boxes[:,2:]>boxes[:,:2]).all():
                raise ValueError('Invalid cached native detections')
        detection_seconds = time.perf_counter() - before_detection
        matches, ambiguous = associate_boxes([prepared[index]["bbox"] for index in visible], boxes)
        # Preserve candidate RGB observations even if one-to-one association is
        # ambiguous. Those raw measurements remain diagnostic data; ambiguity
        # still excludes them from the reported aggregate fit consistency.
        candidates = {index: [j for j, box in enumerate(boxes)
                              if bbox_iou(prepared[index]["bbox"], box) >= .1] for index in visible}
        detection_indices = sorted({j for indices in candidates.values() for j in indices})
        before_pose = time.perf_counter()
        observations = self.pose_keypoints(rgb, boxes[detection_indices]) if self.observe_pose else []
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
            if not self.observe_pose:
                result.update(status='detected', detector_index=int(detected_index),
                              independent_detection_bbox_xyxy=boxes[detected_index].tolist(),
                              independent_detection_confidence=float(confidences[detected_index]),
                              association_bbox_iou=match['bbox_iou'], association_margin=match['association_margin'],
                              pose_observation_status='not_requested_person_detection_presence_rule')
                records.append(result)
                continue
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
                          crop_source="independent_YOLO_detection_after_unique_fixed_fit_bbox_association",
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
        if self.device.type == "cpu" and torch.cuda.is_initialized():
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
                                   "total": time.perf_counter() - started}, "cuda_context_initialized": torch.cuda.is_initialized(),
                "inference_device": str(self.device)}
        record['inference_scope'] = 'person_detection_and_pose_diagnostics' if self.observe_pose else 'person_detection_and_association_only'
        record['person_detection_source'] = 'existing_native_teacher_detection_cache' if cached_detections is not None else 'new_native_YOLO_detection'
        if self.camera_name == "cam1":
            record.update(raw_cam1_video=record["raw_camera_video"], cam1_native_source=native)
        return record


