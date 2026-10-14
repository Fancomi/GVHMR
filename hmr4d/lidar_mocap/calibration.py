"""Freeze the explicitly adopted badminton calibration without changing sources.

Points use row vectors in this API. Fixed24 storage is raw LiDAR metres; viewer
points have exactly one application of the source's rounded ``lidar_rotation``.
Camera coordinates follow OpenCV (right, down, forward). The snapshot records
the development calibration's physical limits as well as its exact matrices.
"""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_ACTIVE_RELATIVE = Path(".local/evidence/BADMINTON-CALIB-1012/multiframe/active-selection.json")
_SCHEMA = "gvhmr-adopted-lidar-calibration-v1"


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _json(blob):
    return json.loads(blob.decode("utf-8-sig"))


def _canonical(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _bounded(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or any(".creating" in part.lower() for part in path.parts):
        raise ValueError(f"Calibration dependency escapes its root or is a creating candidate: {name}")
    return path


def _checked_blob(path, expected=None):
    blob = Path(path).read_bytes()
    if expected is not None and _sha(blob) != expected.lower():
        raise ValueError(f"Calibration source SHA256 mismatch: {path}")
    return blob


def _matrix(value, name, shape):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} requires finite shape {shape}")
    return result


def _rotation(value, name, tolerance=1e-5):
    result = _matrix(value, name, (3, 3))
    if np.max(np.abs(result.T @ result - np.eye(3))) > tolerance or abs(np.linalg.det(result) - 1) > tolerance:
        raise ValueError(f"{name} is not a proper rotation within the recorded rounding tolerance")
    return result


def _proper_rotation(matrix):
    left, _, right = np.linalg.svd(matrix)
    sign = np.eye(3)
    sign[2, 2] = np.linalg.det(left @ right)
    return left @ sign @ right


class CalibrationSnapshot:
    """Immutable-by-convention, self-contained calibration and frame-map bundle.

    ``freeze`` checks current source hashes before publishing. ``load`` checks
    only the frozen local bundle, so a later adoption does not alter an old run.
    Existing snapshots may be reused by ``freeze`` only when their adopted
    pointer and source manifest hashes still agree with current sources.
    """

    def __init__(self, data, snapshot_json):
        self.data = data
        self.snapshot_json = Path(snapshot_json).resolve()
        self._validate_geometry()

    @classmethod
    def freeze(cls, lidar_project_root, sessions_root, output_dir):
        source_root = Path(lidar_project_root).resolve()
        sessions_root = Path(sessions_root).resolve()
        output_dir = Path(output_dir).resolve()
        if not output_dir.is_relative_to(_WORKSPACE_ROOT.resolve()):
            raise ValueError("Calibration output_dir must remain inside the GVHMR workspace")
        if output_dir.is_relative_to(source_root) or output_dir.is_relative_to(sessions_root):
            raise ValueError("Calibration snapshots cannot be written into either source root")
        snapshot_path = output_dir / "calibration_snapshot.json"
        active_path = _bounded(source_root, _ACTIVE_RELATIVE)
        active_blob = _checked_blob(active_path)
        active = _json(active_blob)
        if active.get("status") != "adopted_for_current_local_calibration":
            raise ValueError("The active pointer does not identify an adopted calibration")
        if active.get("variant") != "device_mid":
            raise ValueError("Only the explicitly adopted device midpoint timing policy is supported")
        base = active_path.parent
        artifacts = {}
        blobs = {}

        def retain(path, expected=None):
            path = Path(path).resolve()
            if not path.is_relative_to(base):
                raise ValueError(f"Calibration evidence escapes active-selection root: {path}")
            blob = _checked_blob(path, expected)
            relative = Path("evidence") / path.relative_to(base)
            key = relative.as_posix()
            artifacts[key] = {"path": key, "source_path": str(path), "sha256": _sha(blob)}
            blobs[key] = blob
            return _json(blob)

        retain(active_path, _sha(active_blob))
        linked = {}
        for key in ("calibration", "timing_policy", "frame_index", "bbox", "geometry_verification", "bbox_verification"):
            if key not in active or key + "_sha256" not in active:
                raise ValueError(f"The adopted pointer lacks a hashed dependency: {key}")
            linked[key] = retain(_bounded(base, active[key]), active[key + "_sha256"])
        calibration = linked["calibration"]
        policy = linked["timing_policy"]
        index = linked["frame_index"]
        if calibration.get("production_admitted", False) != active.get("production_admitted", False):
            raise ValueError("Active pointer and calibration disagree on production admission")
        if not linked["geometry_verification"].get("adopted_for_current_local_calibration"):
            raise ValueError("Geometry verification does not attest adoption")
        if not linked["bbox_verification"].get("adopted"):
            raise ValueError("BBox verification does not attest adoption")
        if linked["geometry_verification"].get("calibration_sha256") != active["calibration_sha256"]:
            raise ValueError("Geometry verification does not match the selected calibration")
        bbox_verification = linked["bbox_verification"]
        if bbox_verification.get("bbox_sha256") != active["bbox_sha256"]:
            raise ValueError("BBox verification does not match the selected bbox")
        if bbox_verification.get("selected_calibration_sha256") != active["calibration_sha256"]:
            raise ValueError("BBox verification is bound to another camera calibration")
        if index.get("policy_sha256") != active["timing_policy_sha256"]:
            raise ValueError("Frame index is bound to another timing policy")
        if calibration.get("timing_policy") != policy:
            raise ValueError("Camera calibration and adopted timing policy disagree")
        if (policy.get("variant") != "device_mid" or policy.get("lidar_reference") != "scan_midpoint"
                or policy.get("extra_offset_us") != {"cam1": 0, "cam2": 0}
                or policy.get("timestamp_unit") != "integer_microseconds"):
            raise ValueError("Timing policy differs from the adopted common device midpoint contract")

        # The reference plane is frozen by the adopted bbox stage's delivery
        # manifest; no newly modified or mtime-selected fit result is accepted.
        bbox_dir = _bounded(base, active["bbox"]).parent
        delivery = retain(bbox_dir / "delivery-freeze.json")
        delivery_files = {entry["path"]: entry["sha256"] for entry in delivery["files"]}
        if delivery_files.get("bbox.json") != active["bbox_sha256"]:
            raise ValueError("BBox delivery freeze does not match the active selection")
        if "fit-results.json" not in delivery_files:
            raise ValueError("BBox delivery freeze lacks a reference-plane result")
        fit = retain(bbox_dir / "fit-results.json", delivery_files["fit-results.json"])
        if fit.get("camera_calibration_sha256") != active["calibration_sha256"]:
            raise ValueError("Reference-plane result is bound to another camera calibration")
        if fit.get("selected") != linked["bbox"]:
            raise ValueError("Reference-plane result and adopted bbox disagree")

        sync_dir = _bounded(base, active["timing_policy"]).parent
        provenance = retain(sync_dir / "provenance.json")
        sync_freeze = retain(sync_dir / "delivery-freeze.json")
        provenance_path = sync_dir / "provenance.json"
        frozen_provenance = next((r["sha256"] for r in sync_freeze["files"]
                                  if Path(r["path"]).name == "provenance.json"), None)
        if frozen_provenance is None or _sha(provenance_path.read_bytes()) != frozen_provenance:
            raise ValueError("Timing provenance differs from its delivered freeze")
        if provenance.get("frame_maps") != index.get("frame_maps"):
            raise ValueError("Timing provenance and selected frame-map index disagree")
        source_audits = {row["source_id"]: row for row in provenance["source_clock_audits"]}
        valid_ids = set(calibration["valid_source_ids"])
        map_entries = index["frame_maps"]
        if {entry["source_id"] for entry in map_entries} != valid_ids or set(source_audits) != valid_ids:
            raise ValueError("Camera, timing and frame-map source identities disagree")
        sessions = {}
        common_q = None
        for entry in map_entries:
            sid = entry["source_id"]
            if Path(sid).name != sid or sid in sessions:
                raise ValueError("Invalid or repeated calibration source identity")
            manifest_path = _bounded(sessions_root, Path(sid) / "output-lidar" / f"lidar_{sid}" / "manifest.json")
            audit = source_audits[sid]
            manifest_blob = _checked_blob(manifest_path, audit["manifest_sha256"])
            manifest = _json(manifest_blob)
            if manifest.get("coordinate_space") != "lidar_sensor_xyz":
                raise ValueError("Source manifest does not declare raw LiDAR storage")
            q = _rotation(np.asarray(manifest["lidar_rotation"]).reshape(3, 3), "source raw-to-viewer Q")
            if common_q is None:
                common_q = q
            elif not np.array_equal(common_q, q):
                raise ValueError("Adopted sources have different raw-to-viewer matrices")
            if int(entry["rows"]) != int(manifest["frame_count"]):
                raise ValueError("Frozen frame map and source frame count disagree")
            map_path = _bounded(sync_dir, entry["path"])
            map_blob = _checked_blob(map_path, entry["sha256"])
            rows = [json.loads(line) for line in map_blob.splitlines() if line.strip()]
            if len(rows) != entry["rows"] or [r["lidar_frame"] for r in rows] != list(range(entry["rows"])):
                raise ValueError("Frozen frame map must cover every LiDAR frame exactly once")
            for row in rows:
                frame = row["lidar_frame"]
                capture = manifest["frame_capture_timestamps"][frame]
                first, last = capture["device_first_packet_us"], capture["device_last_packet_us"]
                if row["device_first_us"] != first or row["device_last_us"] != last or row["scan_mid_us"] != (first + last) // 2:
                    raise ValueError("Frozen frame map differs from source device scan evidence")
                for name in ("cam1", "cam2"):
                    match = row["cameras"][name]
                    fields = ("record_index", "media_pts_us", "video_common_us", "time_error_us")
                    if match["status"] == "matched":
                        if any(not isinstance(match.get(k), int) for k in fields):
                            raise ValueError("Matched frame requires integer media and clock identity")
                        if match["video_common_us"] - row["scan_mid_us"] != match["time_error_us"]:
                            raise ValueError("Frozen camera time error is inconsistent")
                        if not 0 <= match["record_index"] < audit["tracks"][name]["rows"] or match["media_pts_us"] < 0:
                            raise ValueError("Frozen camera record lies outside its source")
                    elif any(match.get(k) is not None for k in fields):
                        raise ValueError("Unmatched frames must not reuse media endpoints")
            map_relative = f"frame_maps/{sid}.jsonl"
            artifacts[map_relative] = {"path": map_relative, "source_path": str(map_path), "sha256": _sha(map_blob)}
            blobs[map_relative] = map_blob
            sessions[sid] = dict(manifest_path=str(manifest_path), manifest_sha256=_sha(manifest_blob),
                                 frame_count=int(manifest["frame_count"]), frame_map=map_relative,
                                 frame_map_sha256=_sha(map_blob), source_clock_audit=copy.deepcopy(audit))
        fit_q = _matrix(fit["source_rotation_raw_to_viewer"], "bbox source Q", (3, 3))
        if not np.array_equal(fit_q, common_q):
            raise ValueError("Reference plane and actual source Q disagree")
        plane = copy.deepcopy(fit["reference_plane"])
        # Independently recompute the plane instead of trusting an unbound box
        # top surface. Rounded Q requires the exact inverse for plane normals.
        court = calibration["court_registration"]
        court_r = _rotation(court["R_raw_lidar_to_court"], "raw-to-court rotation")
        court_t = _matrix(court["t_raw_lidar_to_court_m"], "raw-to-court translation", (3,))
        court_to_viewer = common_q @ court_r.T
        origin = -court_to_viewer @ court_t
        normal = np.linalg.inv(court_to_viewer).T @ np.array([0., 0., 1.])
        normal /= np.linalg.norm(normal)
        if (not np.allclose(origin, plane["origin_viewer_m"], atol=1e-8, rtol=0)
                or not np.allclose(normal, plane["normal_viewer"], atol=1e-8, rtol=0)):
            raise ValueError("Reference plane is inconsistent with the frozen court registration")
        cameras = copy.deepcopy(calibration["cameras"])
        for camera in cameras.values():
            camera["image_size"] = calibration["image_size"]
            camera["projection_model"] = "opencv_pinhole_brown5"
        data = dict(schema=_SCHEMA, frozen_utc=datetime.now(timezone.utc).isoformat(),
                    active_selection_sha256=_sha(active_blob), active_selection=active,
                    lidar_project_root=str(source_root), sessions_root=str(sessions_root),
                    cameras=cameras, Q_raw_to_viewer=common_q.tolist(), reference_plane=plane,
                    bbox=linked["bbox"], court_registration=court, timing_policy=policy,
                    sessions=sessions, artifacts=artifacts,
                    coordinate_convention=calibration["coordinate_convention"],
                    production_admitted=bool(calibration.get("production_admitted", False)),
                    physical_limits=dict(physical_exposure_accuracy_verified=policy.get("physical_exposure_accuracy_verified", False),
                                         per_point_scan_times_available=policy.get("per_point_scan_times_available", False),
                                         calibration_assumptions=calibration.get("assumptions", []),
                                         reference_plane_limit=plane.get("source"),
                                         bbox_limits=fit.get("limitations", [])),
                    Q_rounding=dict(max_orthogonality_error=float(np.max(np.abs(common_q.T @ common_q - np.eye(3)))),
                                    determinant=float(np.linalg.det(common_q)),
                                    point_inverse="exact matrix inverse",
                                    orientation_inverse="nearest proper SO(3) of Q @ R_raw.T"))
        # Fail source races before writing any output. Source files are never
        # changed, even when a frozen destination already exists.
        for key, artifact in artifacts.items():
            _checked_blob(artifact["source_path"], artifact["sha256"])
        for session in sessions.values():
            _checked_blob(session["manifest_path"], session["manifest_sha256"])
        if snapshot_path.exists():
            old = cls.load(snapshot_path)
            if old.data["active_selection_sha256"] != data["active_selection_sha256"] or old.data["sessions"] != sessions:
                raise ValueError("Existing snapshot belongs to another adoption; load it explicitly or use a new output_dir")
            return old
        data["snapshot_digest"] = _sha(_canonical(data))
        cls(data, snapshot_path)  # Geometry validation precedes output creation.
        for key, blob in blobs.items():
            destination = _bounded(output_dir, key)
            if destination.exists() and _sha(destination.read_bytes()) != _sha(blob):
                raise ValueError(f"Frozen destination already contains different evidence: {destination}")
        for key, blob in blobs.items():
            destination = _bounded(output_dir, key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                with destination.open("xb") as handle:
                    handle.write(blob)
        with snapshot_path.open("xb") as handle:
            handle.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n")
        return cls.load(snapshot_path)

    @classmethod
    def load(cls, snapshot_json):
        snapshot_json = Path(snapshot_json).resolve()
        data = _json(snapshot_json.read_bytes())
        if data.get("schema") != _SCHEMA:
            raise ValueError("Unsupported calibration snapshot schema")
        payload = copy.deepcopy(data)
        expected = payload.pop("snapshot_digest", None)
        if expected != _sha(_canonical(payload)):
            raise ValueError("Calibration snapshot metadata digest mismatch")
        for artifact in data["artifacts"].values():
            _checked_blob(_bounded(snapshot_json.parent, artifact["path"]), artifact["sha256"])
        return cls(data, snapshot_json)

    def _validate_geometry(self):
        q = _rotation(self.data["Q_raw_to_viewer"], "snapshot Q")
        plane = self.data["reference_plane"]
        _matrix(plane["origin_viewer_m"], "reference plane origin", (3,))
        normal = _matrix(plane["normal_viewer"], "reference plane normal", (3,))
        if abs(np.linalg.norm(normal) - 1) > 1e-6 or normal[2] <= 0:
            raise ValueError("Reference plane requires an upward unit normal")
        for name, camera in self.data["cameras"].items():
            raw_r = _rotation(camera["R_raw_lidar_to_camera"], f"{name} raw rotation")
            _matrix(camera["t_m"], f"{name} translation", (3,))
            k = _matrix(camera["K"], f"{name} K", (3, 3))
            _matrix(camera["distortion_opencv"], f"{name} Brown distortion", (5,))
            if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.array_equal(k[2], [0, 0, 1]):
                raise ValueError("Camera K requires positive focal lengths and pinhole final row")
            if len(camera["image_size"]) != 2 or min(camera["image_size"]) <= 0:
                raise ValueError("Camera image_size must be positive width/height")
            if camera.get("projection_model") != "opencv_pinhole_brown5":
                raise ValueError("Unsupported camera projection model")
            if "R_viewer_to_camera" in camera and not np.allclose(camera["R_viewer_to_camera"], raw_r @ np.linalg.inv(q), atol=2e-10, rtol=0):
                raise ValueError("Canonical raw and viewer camera matrices disagree")

    @property
    def Q(self):
        return np.asarray(self.data["Q_raw_to_viewer"], dtype=np.float64).copy()

    @property
    def reference_plane(self):
        return copy.deepcopy(self.data["reference_plane"])

    def camera(self, name):
        return copy.deepcopy(self.data["cameras"][name])

    def _camera(self, camera):
        return self.data["cameras"][camera] if isinstance(camera, str) else camera

    def frame_map_path(self, session_id):
        return _bounded(self.snapshot_json.parent, self.data["sessions"][session_id]["frame_map"])

    def viewer_to_camera(self, points, camera):
        camera = self._camera(camera)
        if hasattr(points, "detach"):
            import torch
            transform = np.asarray(camera["R_raw_lidar_to_camera"]) @ np.linalg.inv(self.Q)
            matrix = torch.as_tensor(transform, dtype=points.dtype, device=points.device)
            offset = torch.as_tensor(camera["t_m"], dtype=points.dtype, device=points.device)
            return points @ matrix.T + offset
        points = np.asarray(points, dtype=np.float64)
        return points @ np.linalg.inv(self.Q).T @ np.asarray(camera["R_raw_lidar_to_camera"]).T + camera["t_m"]

    def camera_to_viewer(self, points, camera):
        camera = self._camera(camera)
        if hasattr(points, "detach"):
            import torch
            transform = self.Q @ np.asarray(camera["R_raw_lidar_to_camera"]).T
            matrix = torch.as_tensor(transform, dtype=points.dtype, device=points.device)
            offset = torch.as_tensor(camera["t_m"], dtype=points.dtype, device=points.device)
            return (points - offset) @ matrix.T
        # The raw camera matrix is a full-precision SO(3); Q is rounded.
        return (np.asarray(points, dtype=np.float64) - camera["t_m"]) @ np.asarray(camera["R_raw_lidar_to_camera"]) @ self.Q.T

    def camera_rotation_to_viewer(self, camera):
        """Proper column-vector rotation, for SMPL root/body orientations."""
        camera = self._camera(camera)
        return _proper_rotation(self.Q @ np.asarray(camera["R_raw_lidar_to_camera"]).T)

    def project(self, points_viewer, camera, distort=True):
        camera_dict = self._camera(camera)
        xyz = self.viewer_to_camera(points_viewer, camera_dict)
        depth = xyz[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = xyz[..., 0] / depth, xyz[..., 1] / depth
            k1, k2, p1, p2, k3 = camera_dict["distortion_opencv"] if distort else (0, 0, 0, 0, 0)
            r2 = x * x + y * y
            radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
            xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            homogeneous = np.stack((xd, yd, np.ones_like(xd)), axis=-1) @ np.asarray(camera_dict["K"]).T
            uv = homogeneous[..., :2] / homogeneous[..., 2:]
        return uv, depth

    def project_torch(self, points_viewer, camera, distort=True):
        """Differentiable Brown projection on the input tensor's dtype/device.

        Depth is returned separately for masking. At zero depth the denominator
        is bounded to avoid infinities during optimization; callers must reject
        nonpositive depths when assigning correspondences.
        """
        import torch
        camera = self._camera(camera)
        def tensor(value):
            return torch.as_tensor(value, dtype=points_viewer.dtype, device=points_viewer.device)
        transform = np.asarray(camera["R_raw_lidar_to_camera"]) @ np.linalg.inv(self.Q)
        xyz = points_viewer @ tensor(transform).T + tensor(camera["t_m"])
        depth = xyz[..., 2]
        eps = torch.finfo(points_viewer.dtype).eps
        denominator = torch.where(depth.abs() > eps, depth, torch.where(depth >= 0, eps, -eps))
        x, y = xyz[..., 0] / denominator, xyz[..., 1] / denominator
        k1, k2, p1, p2, k3 = tensor(camera["distortion_opencv"] if distort else [0] * 5).unbind()
        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        homogeneous = torch.stack((xd, yd, torch.ones_like(xd)), dim=-1) @ tensor(camera["K"]).T
        return homogeneous[..., :2] / homogeneous[..., 2:], depth
