"""Exclude contradictory joint rays without changing measured coordinates/cameras."""

import hashlib
from pathlib import Path

import numpy as np

from .multiview import associate_poses, epipolar_errors, fundamental_matrix
from .pose_provider import NativePoseProvider
from .rally_pipeline import read_json
from .dual_view import file_sha


class RayFilteredPoseProvider(NativePoseProvider):
    def __init__(self, root, calibration):
        super().__init__(root, calibration)
        self.F = fundamental_matrix(calibration)
        self.policy_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def bind(self, session, pair, tracks):
        result, excluded = super().bind(session, pair, tracks)
        if pair["lidar_index"] in self.selected.get(session.session_id, set()) and tracks:
            index = pair["cameras"]["cam1"]["record_index"]
            path = self.root / "native_poses" / session.session_id / "cam1" / f"{index:07d}.json"
            first = read_json(path)
            direct = {
                r["cam2_index"]: r
                for r in associate_poses(first["keypoints"], np.stack([t["kp2d"] for t in tracks]), self.calibration)
                if r["accepted"]
            }
            occupied = {t["pose_evidence"]["cam1_detection"] for t in result if "cam1" in t.get("views", {})}
            current = {t["track_id"]: t for t in result}
            for i, original in enumerate(tracks):
                if i not in direct:
                    continue
                value = current.get(original["track_id"])
                if value is not None and "cam1" in value.get("views", {}):
                    continue
                partner = direct[i]["cam1_index"]
                if partner in occupied:
                    continue
                value = dict(original)
                value["views"] = {
                    "cam2": dict(bbox=original["bbox"], kp2d=np.asarray(original["kp2d"], np.float32).copy()),
                    "cam1": dict(
                        bbox=np.asarray(first["boxes"][partner], np.float32),
                        kp2d=np.asarray(first["keypoints"][partner], np.float32),
                    ),
                }
                value["pose_evidence"] = dict(
                    files={"cam1": dict(path=str(path), sha256=file_sha(path))},
                    cam1_detection=partner,
                    primary_source="original_timestamp_checked_teacher_observation",
                    secondary_status="unique_native_ray_correspondence_from_teacher_pose",
                    median_epipolar_px=direct[i]["median_epipolar_px"],
                    margin_px=direct[i]["margin_px"],
                )
                current[original["track_id"]] = value
                occupied.add(partner)
            result = [current[t["track_id"]] for t in tracks if t["track_id"] in current]
            excluded = [e for e in excluded if e["track_id"] not in current]
        for track in result:
            if "cam1" not in track.get("views", {}):
                continue
            first = track["views"]["cam1"]["kp2d"]
            second = track["views"]["cam2"]["kp2d"]
            valid = (first[:, 2] >= 0.35) & (second[:, 2] >= 0.35)
            errors = epipolar_errors(first, second, self.F)
            body = np.arange(5, 17)[valid[5:17]]
            threshold = max(6.0, min(10.0, 2.5 * float(np.median(errors[body])))) if len(body) else 6.0
            rejected = valid & (errors > threshold)
            first[rejected, 2] = 0
            second[rejected, 2] = 0
            track["pose_evidence"]["inconsistent_joint_rays"] = np.flatnonzero(rejected).tolist()
            track["pose_evidence"]["joint_ray_error_px"] = errors.tolist()
            track["pose_evidence"]["joint_ray_threshold_px"] = threshold
            track["pose_evidence"]["joint_ray_policy_sha256"] = self.policy_sha256
        return result, excluded
