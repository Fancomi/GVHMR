"""Bind fixed-native two-camera measurements to each fitted source scan."""

from pathlib import Path

import numpy as np

from .dual_view import file_sha
from .multiview import associate_poses, match_teacher_observations
from .rally_pipeline import read_json


class NativePoseProvider:
    def __init__(self, root, calibration):
        self.root = Path(root)
        self.calibration = calibration
        self.plan = read_json(self.root / "case_plan.json")
        self.selected = {sid: set(indices) for sid, indices in self.plan["selected_scans"].items()}
        receipt = read_json(self.root / "pose_receipt.json")
        if (
            receipt["state"] != "complete"
            or receipt["config"]["calibration_digest"] != calibration.data["snapshot_digest"]
        ):
            raise ValueError("Native poses are incomplete or use another camera snapshot")
        self.digest = file_sha(self.root / "pose_receipt.json")
        self.config_digest = receipt["config_digest"]

    def bind(self, session, pair, tracks):
        if pair["lidar_index"] not in self.selected.get(session.session_id, set()):
            return tracks, []
        records = {}
        paths = {}
        for cam in ("cam1", "cam2"):
            index = pair["cameras"][cam]["record_index"]
            path = self.root / "native_poses" / session.session_id / cam / f"{index:07d}.json"
            record = read_json(path)
            if record["config_digest"] != self.config_digest or record["native"] != session.camera_rows(cam)[index]:
                raise ValueError("Native pose timestamp/pixel configuration changed")
            records[cam] = record
            paths[cam] = dict(path=str(path), sha256=file_sha(path))
        a, b = records["cam1"], records["cam2"]
        matched = match_teacher_observations(tracks, b["keypoints"])
        partners = {
            r["cam2_index"]: r
            for r in associate_poses(a["keypoints"], b["keypoints"], self.calibration)
            if r["accepted"]
        }
        result, excluded = [], []
        for index, track in enumerate(tracks):
            if index not in matched:
                excluded.append(dict(track_id=track["track_id"], reason="no_unique_exact_native_primary_pose"))
                continue
            j = matched[index]
            value = dict(
                track, bbox=np.asarray(b["boxes"][j], np.float32), kp2d=np.asarray(b["keypoints"][j], np.float32)
            )
            value["views"] = {"cam2": dict(bbox=value["bbox"], kp2d=value["kp2d"])}
            value["pose_evidence"] = dict(
                files=paths, cam2_detection=j, secondary_status="no_unique_native_ray_correspondence"
            )
            if j in partners:
                link = partners[j]
                i = link["cam1_index"]
                value["views"]["cam1"] = dict(
                    bbox=np.asarray(a["boxes"][i], np.float32), kp2d=np.asarray(a["keypoints"][i], np.float32)
                )
                value["pose_evidence"].update(
                    cam1_detection=i,
                    secondary_status="unique_native_ray_correspondence",
                    median_epipolar_px=link["median_epipolar_px"],
                    margin_px=link["margin_px"],
                )
            result.append(value)
        return result, excluded
