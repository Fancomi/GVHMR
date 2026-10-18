"""Reassociate immutable native detections to repaired bodies, never old identities."""

from pathlib import Path
import time

import numpy as np

from .dual_view import file_sha
from .rgb_observations import associate_boxes, clip_box


def reassociate_native_frame(frame, cached, camera, calibration, evidence):
    started = time.perf_counter()
    if cached["camera_name"] != camera or cached["lidar_index"] != frame["lidar_index"]:
        raise ValueError("Cached detection belongs to another camera/scan")
    if cached["scan_mid_us"] != frame["scan_mid_us"] or cached["paired_cameras"] != frame["cameras"]:
        raise ValueError("Cached detection has different native frame/time addressing")
    native = cached["camera_native_source"]
    for key in ("record_index", "media_pts_us", "common_time_us"):
        if native[key] != frame["cameras"][camera][key]:
            raise ValueError("Cached detection native PTS/index changed")
    for name in ("native_bgr_image_sha256", "undistorted_rgb_image_sha256"):
        if len(cached.get(name, "")) != 64:
            raise ValueError("Cached detection has no native pixel digest")
    boxes = np.asarray(cached["detector_boxes_xyxy"], dtype=np.float32).reshape(-1, 4)
    confidence = np.asarray(cached["detector_confidences"], dtype=np.float32)
    if (
        len(boxes) != len(confidence)
        or not np.isfinite(boxes).all()
        or not np.isfinite(confidence).all()
        or not (boxes[:, 2:] > boxes[:, :2]).all()
    ):
        raise ValueError("Invalid immutable native detections")
    width, height = calibration.camera(camera)["image_size"]
    prepared = []
    for athlete in frame["athletes"]:
        path = Path(athlete["fit_path"]).resolve()
        with np.load(path, allow_pickle=False) as fit:
            vertices = fit["vertices"].astype(np.float64)
        if vertices.shape != (6890, 3) or not np.isfinite(vertices).all():
            raise ValueError("Expected a finite native neutral-SMPL mesh")
        uv, depth = calibration.project(vertices, camera, distort=False)
        box = clip_box(uv, depth, width, height)
        prepared.append(dict(athlete=athlete, path=path, sha=file_sha(path), box=box))
    visible = [i for i, item in enumerate(prepared) if item["box"] is not None]
    matches, rejected = associate_boxes([prepared[i]["box"] for i in visible], boxes)
    matches = {visible[i]: value for i, value in matches.items()}
    rejected = {visible[i]: value for i, value in rejected.items()}
    athletes = []
    for i, item in enumerate(prepared):
        source = item["athlete"]
        person = dict(
            motion_id=source["motion_id"],
            fit_path=str(item["path"]),
            fit_sha256=item["sha"],
            fit_accepted=source["accepted"],
            fixed_smpl_geometry_time_us=frame["scan_mid_us"],
            fixed_fit_projected_bbox_xyxy=None if item["box"] is None else item["box"].tolist(),
            detector_index=None,
            status="excluded",
            rejection_reasons=[],
            pose_observation_status="not_requested_person_detection_presence_rule",
        )
        if i in matches:
            match = matches[i]
            index = match["detection_index"]
            person.update(
                status="detected",
                detector_index=index,
                independent_detection_bbox_xyxy=boxes[index].tolist(),
                independent_detection_confidence=float(confidence[index]),
                association_bbox_iou=match["bbox_iou"],
                association_margin=match["association_margin"],
            )
        else:
            person["rejection_reasons"] = [rejected.get(i, "no_unique_independent_rgb_detection")]
        athletes.append(person)
    record = dict(cached)
    record.update(
        clip_id="repaired_all_native_scans",
        athletes=athletes,
        fixed_geometry_common_time_us=frame["scan_mid_us"],
        geometry_interpolated=False,
        inference_scope="person_detection_and_association_only",
        person_detection_source="immutable_native_detections_reassociated_to_repaired_SMPL",
        reused_native_evidence=evidence,
        cuda_context_initialized=False,
        inference_device="cpu",
        timing_seconds=dict(detector=0.0, pose=0.0, total=time.perf_counter() - started),
    )
    return record
