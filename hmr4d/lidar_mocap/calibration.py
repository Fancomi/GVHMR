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

from hmr4d import PROJ_ROOT
from .sessions import SessionIndex


_WORKSPACE_ROOT = PROJ_ROOT
_CONFIG_RELATIVE = Path("config/calibration/badminton-20261012")
_CONFIG_FILES = ("calibration.json", "sources.json", "sync-policy.json", "bbox.json", "roi-config.txt")
_CANONICAL_SCHEMA = "gvhmr-canonical-lidar-calibration-v2"
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


def bbox_rotation_matrix(rotation_deg):
    """Canonical BBox-local to viewer rotation, including roll and pitch."""
    x, y, z = np.deg2rad(_matrix(rotation_deg, "bbox rotation_deg", (3,)))
    cx, cy, cz = np.cos([x, y, z])
    sx, sy, sz = np.sin([x, y, z])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx



def _same_inputs(old, current):
    """Reusing an immutable legacy bundle requires the same physical inputs."""
    if old.get("sessions_root") != current["sessions_root"]:
        return False
    if old.get("schema") == _CANONICAL_SCHEMA:
        if old.get("configuration_sha256") != current["configuration_sha256"]:
            return False
    elif old.get("active_selection", {}).get("calibration_sha256") != current["provenance"]["accepted_source_calibration_sha256"]:
        return False
    for key in ("Q_raw_to_viewer", "bbox", "timing_policy", "production_admitted"):
        if old.get(key) != current[key]:
            return False
    for key in ("R_raw_lidar_to_court", "t_raw_lidar_to_court_m"):
        if old["court_registration"].get(key) != current["court_registration"][key]:
            return False
    for key in ("origin_viewer_m", "normal_viewer", "court_to_raw_R", "court_origin_raw_m",
                "ground_bbox_centre_normal_offset_m", "ground_bbox_upper_surface_margin_m"):
        value = current["reference_plane"].get(key)
        previous = old["reference_plane"].get(key)
        if value is not None and (previous is None or not np.allclose(previous, value, atol=1e-8, rtol=0)):
            return False
    for name in ("cam1", "cam2"):
        for key in ("K", "distortion_opencv", "R_raw_lidar_to_camera", "R_viewer_to_camera", "t_m", "image_size", "projection_model"):
            if old["cameras"][name].get(key) != current["cameras"][name].get(key):
                return False
    if set(old["sessions"]) != set(current["sessions"]):
        return False
    for sid, session in current["sessions"].items():
        previous = old["sessions"][sid]
        for key in ("manifest_sha256", "frame_count", "frame_map_sha256"):
            if previous.get(key) != session[key]:
                return False
        before, after = previous["source_clock_audit"], session["source_clock_audit"]
        if before.get("common_index_sha256") != after["common_index_sha256"]:
            return False
        for name in ("cam1", "cam2"):
            for key in ("rows", "canonical_sha256"):
                if before["tracks"][name].get(key) != after["tracks"][name][key]:
                    return False
    return True


class CalibrationSnapshot:
    """Immutable-by-convention, self-contained calibration and frame-map bundle.

    ``freeze`` checks current source hashes before publishing. ``load`` checks
    only the frozen local bundle, so a later adoption does not alter an old run.
    ``freeze`` reads the declared canonical result plus original recordings,
    never another project's experimental worktree. Historical bundles remain
    loadable and are reused only after numerical and source equivalence checks.
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
        config_dir = _bounded(source_root, _CONFIG_RELATIVE)
        inputs = {_bounded(config_dir, name): _checked_blob(_bounded(config_dir, name)) for name in _CONFIG_FILES}
        blobs = {f"evidence/{name}": inputs[_bounded(config_dir, name)] for name in _CONFIG_FILES}
        hashes = {name: _sha(blobs[f"evidence/{name}"]) for name in _CONFIG_FILES}
        artifacts = {f"evidence/{name}": dict(path=f"evidence/{name}", source_path=str(config_dir / name),
                                             sha256=hashes[name]) for name in _CONFIG_FILES}
        calibration = _json(blobs["evidence/calibration.json"])
        sources = _json(blobs["evidence/sources.json"])
        policy = _json(blobs["evidence/sync-policy.json"])
        bbox = _json(blobs["evidence/bbox.json"])
        if (calibration.get("schema") != "badminton-calibration-result-v1"
                or sources.get("schema") != "badminton-calibration-sources-v1"):
            raise ValueError("Unsupported canonical calibration/sources schema")
        if set(calibration.get("cameras", {})) != {"cam1", "cam2"}:
            raise ValueError("Canonical calibration requires both complete cameras: cam1 and cam2")
        result_files = sources.get("result_files")
        if result_files is not None and result_files != {name: hashes[name] for name in _CONFIG_FILES if name != "sources.json"}:
            raise ValueError("Canonical result file SHA256 mismatch")
        for key, actual in (("accepted_bbox_sha256", hashes["bbox.json"]),
                            ("sync_policy_sha256", hashes["sync-policy.json"])):
            if sources.get(key) != actual:
                raise ValueError(f"Canonical source SHA256 mismatch: {key}")
        # This is original calibration provenance, not the portable JSON's SHA.
        provenance_sha = sources.get("accepted_source_calibration_sha256", "")
        if (not isinstance(provenance_sha, str) or len(provenance_sha) != 64
                or any(c not in "0123456789abcdef" for c in provenance_sha)):
            raise ValueError("Canonical sources lacks the original accepted calibration SHA256")
        if calibration.get("production_admitted") is not sources.get("production_admitted"):
            raise ValueError("Canonical sources and calibration disagree on production admission")
        if not isinstance(calibration.get("production_admitted"), bool):
            raise ValueError("Canonical calibration requires explicit production_admitted")
        expected_policy = dict(schema="selected-device-common-sync-v1", variant="device_mid",
                               lidar_reference="scan_midpoint", timestamp_unit="integer_microseconds",
                               clock_axis="device_reported_timestamps_in_host_seeded_PHC_epoch",
                               target_formula="(device_first_packet_us + device_last_packet_us) // 2",
                               video_selection="nearest_ZCAM_common_NTP_timestamp", tie_break="earlier_record_index",
                               extra_offset_us={"cam1": 0, "cam2": 0},
                               media_binding="canonical_media_pts_us_for_selected_record_index",
                               missing_time_behavior="unmatched; never reuse an endpoint frame or infer a receive-clock fallback")
        if calibration.get("timing_policy") != policy or any(policy.get(k) != v for k, v in expected_policy.items()):
            raise ValueError("Timing policy differs from the canonical common device midpoint contract")
        q = _rotation(np.asarray(calibration["lidar_rotation"]).reshape(3, 3), "canonical source raw-to-viewer M")
        roi = dict(line.split("=", 1) for line in blobs["evidence/roi-config.txt"].decode("utf-8-sig").splitlines()
                   if line.strip() and not line.lstrip().startswith("#"))
        roi_q = _matrix([float(x) for x in roi.get("lidar_rotation", "").split(",")], "ROI source M", (9,)).reshape(3, 3)
        if not np.array_equal(roi_q, q) or roi.get("bbox_json") != "bbox.json":
            raise ValueError("Canonical ROI, bbox path and source M disagree")
        ids = calibration.get("valid_source_ids", [])
        entries = sources.get("frame_maps", [])
        if (not ids or len(set(ids)) != len(ids) or set(ids) != set(sources.get("valid_source_ids", []))
                or len(entries) != len(ids) or {e["source_id"] for e in entries} != set(ids)):
            raise ValueError("Canonical calibration and frame-map source identities disagree")
        sessions = {}
        for entry in entries:
            sid = entry["source_id"]
            if not isinstance(sid, str) or Path(sid).name != sid or sid in (".", ".."):
                raise ValueError("Invalid canonical source identity")
            session_dir = _bounded(sessions_root, sid)
            manifest_path = _bounded(session_dir, Path("output-lidar") / f"lidar_{sid}" / "manifest.json")
            clock_path = _bounded(session_dir, Path("meta") / sid / f"{sid}_common_clock.json")
            sidecars = {name: _bounded(session_dir, Path("output-zcam") / f"zcam_{sid}" / f"{name}.frames.jsonl")
                        for name in ("cam1", "cam2")}
            raw_paths = [manifest_path, clock_path, *sidecars.values()]
            for path in raw_paths:
                inputs[path] = _checked_blob(path)
            session = SessionIndex.load(session_dir)
            if not np.array_equal(session.Q, q):
                raise ValueError("Canonical calibration and actual source M disagree")
            rows = []
            for pair in session.pairs():
                if pair["scan_mid_us"] is None:
                    raise ValueError("Source scan lacks recorded device-clock timing")
                cameras = {}
                for name, camera in pair["cameras"].items():
                    cameras[name] = dict(status=camera["status"], record_index=camera["record_index"],
                                         media_pts_us=camera["media_pts_us"], video_common_us=camera["common_time_us"],
                                         time_error_us=camera["time_error_us"])
                rows.append(dict(lidar_frame=pair["lidar_index"], device_first_us=pair["scan_start_us"],
                                 device_last_us=pair["scan_end_us"], scan_mid_us=pair["scan_mid_us"], cameras=cameras))
            map_blob = b"".join(json.dumps(row, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n" for row in rows)
            if len(rows) != entry.get("rows") or _sha(map_blob) != entry.get("sha256"):
                raise ValueError(f"Rebuilt frame-map source rows/SHA256 mismatch: {sid}")
            relative = f"frame_maps/{sid}.jsonl"
            blobs[relative] = map_blob
            artifacts[relative] = dict(path=relative, sha256=_sha(map_blob),
                                       source="Rebuilt from declared original session clock and native video sidecars")
            manifest_sha = _sha(inputs[manifest_path])
            audit = dict(source_id=sid, manifest_sha256=manifest_sha, common_index_sha256=_sha(inputs[clock_path]),
                         tracks={name: dict(rows=len(session.camera_rows(name)), canonical_sha256=_sha(inputs[path]))
                                 for name, path in sidecars.items()})
            sessions[sid] = dict(manifest_path=str(manifest_path), manifest_sha256=manifest_sha,
                                 frame_count=len(rows), frame_map=relative, frame_map_sha256=_sha(map_blob),
                                 source_clock_audit=audit,
                                 source_evidence_sha256={str(path): _sha(inputs[path]) for path in raw_paths})
        plane = copy.deepcopy(calibration["court_reference_plane"])
        court = copy.deepcopy(calibration["court_registration"])
        court_r = _rotation(court["R_raw_lidar_to_court"], "raw-to-court rotation")
        court_t = _matrix(court["t_raw_lidar_to_court_m"], "raw-to-court translation", (3,))
        court_to_viewer = q @ court_r.T
        origin = -court_to_viewer @ court_t
        normal = np.linalg.inv(court_to_viewer).T @ np.array([0., 0., 1.])
        normal /= np.linalg.norm(normal)
        if (not np.allclose(origin, plane["origin_viewer_m"], atol=1e-8, rtol=0)
                or not np.allclose(normal, plane["normal_viewer"], atol=1e-8, rtol=0)):
            raise ValueError("Reference plane is inconsistent with the frozen court registration")
        ground = next((box for box in bbox.get("boxes", []) if box.get("role") == "ground"), None)
        if ground is None:
            raise ValueError("Canonical shared-court bbox lacks an explicit ground role")
        ground_rotation = bbox_rotation_matrix(ground["rotation_deg"])
        centre_offset = float((_matrix(ground["center"], "ground bbox centre", (3,)) - origin) @ normal)
        size = _matrix(ground["size"], "ground bbox size", (3,))
        upper_margin = centre_offset + float(size[2]) / 2
        if (np.any(size <= 0) or not np.allclose(ground_rotation[:, 2], normal, atol=1e-7, rtol=0)
                or not np.isclose(centre_offset, plane["ground_bbox_centre_normal_offset_m"], atol=1e-8, rtol=0)
                or not np.isclose(upper_margin, plane["ground_bbox_upper_surface_margin_m"], atol=1e-8, rtol=0)):
            raise ValueError("SharedCourtPlane and full ground bbox pose/margins disagree")
        bbox_geometry = dict(rotation_convention="Rz @ Ry @ Rx", ground_rotation_local_to_viewer=ground_rotation.tolist(),
                             ground_centre_normal_offset_m=centre_offset, ground_upper_surface_margin_m=upper_margin)
        plane.setdefault("source", plane.get("note", "Frozen raw-to-court registration, court Z=0"))
        cameras = copy.deepcopy(calibration["cameras"])
        for camera in cameras.values():
            camera["image_size"] = calibration["image_size"]
            camera["projection_model"] = "opencv_pinhole_brown5"
        data = dict(schema=_CANONICAL_SCHEMA, frozen_utc=datetime.now(timezone.utc).isoformat(),
                    configuration_sha256=_sha(_canonical(hashes)), configuration_files_sha256=hashes,
                    provenance=sources, lidar_project_root=str(source_root), sessions_root=str(sessions_root),
                    cameras=cameras, Q_raw_to_viewer=q.tolist(), reference_plane=plane, bbox=bbox,
                    court_registration=court, court=calibration.get("court", {}), timing_policy=policy,
                    geometry_variant="canonical_shared_court_plane", bbox_rotation_convention="Rz @ Ry @ Rx",
                    bbox_geometry=bbox_geometry, sessions=sessions, artifacts=artifacts,
                    coordinate_convention=calibration["coordinate_convention"],
                    production_admitted=calibration["production_admitted"],
                    physical_limits=dict(physical_exposure_accuracy_verified=policy.get("physical_exposure_accuracy_verified", False),
                                         per_point_scan_times_available=policy.get("per_point_scan_times_available", False),
                                         calibration_assumptions=calibration.get("assumptions", []),
                                         reference_plane_limit=plane["source"],
                                         bbox_limits=[calibration.get("court", {}).get("dimension_basis", "Not independently surveyed")]),
                    Q_rounding=dict(max_orthogonality_error=float(np.max(np.abs(q.T @ q - np.eye(3)))),
                                    determinant=float(np.linalg.det(q)), point_inverse="exact matrix inverse",
                                    orientation_inverse="nearest proper SO(3) of Q @ R_raw.T"))
        snapshot_path = output_dir / "calibration_snapshot.json"
        cls(data, snapshot_path)  # Geometry validation precedes output creation.
        for path, blob in inputs.items():
            _checked_blob(path, _sha(blob))  # Reject source races, including sidecars.
        if snapshot_path.exists():
            old = cls.load(snapshot_path)
            if not _same_inputs(old.data, data):
                raise ValueError("Existing snapshot belongs to another adoption; load it explicitly or use a new output_dir")
            return old  # Preserve historical bytes and digest, including v1 bundles.
        data["snapshot_digest"] = _sha(_canonical(data))
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
        if data.get("schema") not in (_SCHEMA, _CANONICAL_SCHEMA):
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
        if set(self.data["cameras"]) != {"cam1", "cam2"}:
            raise ValueError("Calibration requires both complete cameras: cam1 and cam2")
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
