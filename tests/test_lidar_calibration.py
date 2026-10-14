"""Adoption, source identity and projection contract checks (no GPU required)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hmr4d.lidar_mocap import calibration as module
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return module._sha(path.read_bytes())


class SnapshotFixture:
    """Small, genuinely source-bound fixture matching the adopted file schema."""

    def __init__(self, root):
        self.workspace = root / "gvhmr"
        self.workspace.mkdir()
        self.project = root / "lidar_project"
        self.sessions = root / "sessions"
        self.output = self.workspace / "test-snapshot"
        self.base = self.project / module._ACTIVE_RELATIVE.parent
        self.active_path = self.base / "active-selection.json"
        self.sid = "20261012_154726"
        angle = .17
        self.q = np.round(np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                                    [-np.sin(angle), 0, np.cos(angle)]]), 7)
        theta = .3
        raw_r = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        policy = dict(variant="device_mid", lidar_reference="scan_midpoint",
                      extra_offset_us=dict(cam1=0, cam2=0), timestamp_unit="integer_microseconds",
                      physical_exposure_accuracy_verified=False, per_point_scan_times_available=False)
        policy_hash = write_json(self.base / "selected-sync-v1/sync-policy.json", policy)
        camera = dict(K=[[1000., 0, 960.], [0, 1100., 540.], [0, 0, 1]],
                      distortion_opencv=[-.03, .01, .002, -.003, -.001],
                      R_raw_lidar_to_camera=raw_r.tolist(),
                      R_viewer_to_camera=(raw_r @ np.linalg.inv(self.q)).tolist(), t_m=[.4, -.2, 1.3])
        court = dict(R_raw_lidar_to_court=np.eye(3).tolist(), t_raw_lidar_to_court_m=[0, 0, 2.8])
        config = dict(image_size=[1920, 1080], coordinate_convention="p_camera = R @ p_raw + t; metres",
                      cameras=dict(cam1=camera, cam2=copy.deepcopy(camera)), valid_source_ids=[self.sid],
                      timing_policy=policy, court_registration=court, production_admitted=False,
                      assumptions=["Not a surveyed physical calibration"])
        calib_hash = write_json(self.base / "cam1-mask-selected-v1/calibration.json", config)
        geometry_hash = write_json(self.base / "cam1-mask-selected-v1/verification.json",
                                   dict(adopted_for_current_local_calibration=True, calibration_sha256=calib_hash))
        captures = [dict(device_first_packet_us=10000, device_last_packet_us=20000),
                    dict(device_first_packet_us=110000, device_last_packet_us=120000)]
        manifest = dict(frame_count=2, coordinate_space="lidar_sensor_xyz",
                        lidar_rotation=self.q.reshape(-1).tolist(), frame_capture_timestamps=captures)
        self.manifest_path = self.sessions / self.sid / "output-lidar" / f"lidar_{self.sid}" / "manifest.json"
        manifest_hash = write_json(self.manifest_path, manifest)
        rows = []
        for index, capture in enumerate(captures):
            first, last = capture.values()
            target = (first + last) // 2
            match = dict(status="matched", record_index=index, media_pts_us=index * 100000,
                         video_common_us=target + 100, time_error_us=100)
            rows.append(dict(lidar_frame=index, device_first_us=first, device_last_us=last,
                             scan_mid_us=target, cameras=dict(cam1=match, cam2=copy.deepcopy(match))))
        map_path = self.base / f"selected-sync-v1/frame-map/{self.sid}.jsonl"
        map_path.parent.mkdir(parents=True)
        map_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        self.map_path = map_path
        entry = dict(source_id=self.sid, path=f"frame-map/{self.sid}.jsonl", rows=2,
                     sha256=module._sha(map_path.read_bytes()))
        index_hash = write_json(self.base / "selected-sync-v1/selected-frames.json",
                                dict(policy_sha256=policy_hash, frame_maps=[entry]))
        provenance_hash = write_json(self.base / "selected-sync-v1/provenance.json",
                                      dict(frame_maps=[entry], source_clock_audits=[dict(source_id=self.sid,
                                           manifest_sha256=manifest_hash, tracks=dict(cam1=dict(rows=2), cam2=dict(rows=2)))]))
        write_json(self.base / "selected-sync-v1/delivery-freeze.json",
                   dict(files=[dict(path=str(self.base / "selected-sync-v1/provenance.json"), sha256=provenance_hash)]))
        bbox = dict(version=1, boxes=[dict(center=[0, 0, -2.8], size=[6.1, 13.4, .27], rotation_deg=[0, 0, 3])])
        bbox_hash = write_json(self.base / "cam2-bbox-v1/bbox.json", bbox)
        bbox_verify_hash = write_json(self.base / "cam2-bbox-v1/verification.json",
                                      dict(adopted=True, bbox_sha256=bbox_hash, selected_calibration_sha256=calib_hash))
        origin = -self.q @ np.array([0., 0., 2.8])
        normal = np.linalg.inv(self.q).T @ np.array([0., 0., 1.])
        normal /= np.linalg.norm(normal)
        fit = dict(camera_calibration_sha256=calib_hash, selected=bbox,
                   source_rotation_raw_to_viewer=self.q.tolist(),
                   reference_plane=dict(origin_viewer_m=origin.tolist(), normal_viewer=normal.tolist(),
                                        source="Frozen court Z=0, not Ignore top"), limitations=["Development only"])
        fit_hash = write_json(self.base / "cam2-bbox-v1/fit-results.json", fit)
        write_json(self.base / "cam2-bbox-v1/delivery-freeze.json",
                   dict(files=[dict(path="bbox.json", sha256=bbox_hash), dict(path="fit-results.json", sha256=fit_hash)]))
        self.active = dict(status="adopted_for_current_local_calibration", variant="device_mid",
                           calibration="cam1-mask-selected-v1/calibration.json", calibration_sha256=calib_hash,
                           geometry_verification="cam1-mask-selected-v1/verification.json", geometry_verification_sha256=geometry_hash,
                           timing_policy="selected-sync-v1/sync-policy.json", timing_policy_sha256=policy_hash,
                           frame_index="selected-sync-v1/selected-frames.json", frame_index_sha256=index_hash,
                           bbox="cam2-bbox-v1/bbox.json", bbox_sha256=bbox_hash,
                           bbox_verification="cam2-bbox-v1/verification.json", bbox_verification_sha256=bbox_verify_hash,
                           production_admitted=False)
        write_json(self.active_path, self.active)

    def freeze(self):
        return CalibrationSnapshot.freeze(self.project, self.sessions, self.output)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gvhmr-calibration-test-")
        self.fixture = SnapshotFixture(Path(self.temp.name))
        self.workspace_patch = patch.object(module, "_WORKSPACE_ROOT", self.fixture.workspace)
        self.workspace_patch.start()

    def tearDown(self):
        self.workspace_patch.stop()
        self.temp.cleanup()

    def test_exact_rounded_q_roundtrip_and_orientation(self):
        snapshot = self.fixture.freeze()
        camera_points = np.array([[1, -.4, 2], [-.5, .1, 5], [3, 4, 30.]])
        viewer = snapshot.camera_to_viewer(camera_points, "cam1")
        np.testing.assert_allclose(snapshot.viewer_to_camera(viewer, "cam1"), camera_points, atol=1e-13)
        camera = snapshot.camera("cam1")
        wrong = viewer @ snapshot.Q @ np.asarray(camera["R_raw_lidar_to_camera"]).T + camera["t_m"]
        self.assertGreater(np.max(np.abs(wrong - camera_points)), 1e-8)
        orientation = snapshot.camera_rotation_to_viewer("cam1")
        np.testing.assert_allclose(orientation.T @ orientation, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(orientation), 1, places=14)
        self.assertGreater(snapshot.data["Q_rounding"]["max_orthogonality_error"], 0)

    def test_brown_radial_and_tangential_distortion(self):
        snapshot = self.fixture.freeze()
        viewer = snapshot.camera_to_viewer([[1., -.4, 2.]], "cam1")
        uv, depth = snapshot.project(viewer, "cam1")
        x, y = .5, -.2
        r2 = x*x + y*y
        k1, k2, p1, p2, k3 = snapshot.camera("cam1")["distortion_opencv"]
        radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
        xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
        yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
        np.testing.assert_allclose(uv[0], [1000*xd + 960, 1100*yd + 540], atol=1e-10)
        np.testing.assert_allclose(depth, [2.])
        plain, _ = snapshot.project(viewer, "cam1", distort=False)
        np.testing.assert_allclose(plain[0], [1460, 320], atol=1e-10)
        self.assertGreater(np.max(np.abs(uv - plain)), 1)

    def test_torch_projection_gradients_and_transform(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is unavailable")
        snapshot = self.fixture.freeze()
        camera = torch.tensor([[1., -.4, 2.], [-.2, .3, 4.]], dtype=torch.float64, requires_grad=True)
        viewer = snapshot.camera_to_viewer(camera, "cam1")
        torch.testing.assert_close(snapshot.viewer_to_camera(viewer, "cam1"), camera, atol=1e-13, rtol=0)
        for distort in (True, False):
            uv, depth = snapshot.project_torch(viewer, "cam1", distort=distort)
            expected_uv, expected_depth = snapshot.project(viewer.detach().numpy(), "cam1", distort=distort)
            np.testing.assert_allclose(uv.detach().numpy(), expected_uv, atol=1e-10)
            np.testing.assert_allclose(depth.detach().numpy(), expected_depth, atol=1e-13)
        uv.sum().backward()
        self.assertTrue(torch.isfinite(camera.grad).all().item())
        self.assertGreater(camera.grad.abs().sum().item(), 0)

    def test_snapshot_is_independent_of_later_pointer_changes(self):
        snapshot = self.fixture.freeze()
        first_digest = snapshot.data["snapshot_digest"]
        self.assertFalse(snapshot.data["production_admitted"])
        self.assertEqual(snapshot.data["timing_policy"]["variant"], "device_mid")
        self.assertTrue(snapshot.frame_map_path(self.fixture.sid).is_file())
        self.assertEqual(self.fixture.freeze().data["snapshot_digest"], first_digest)
        changed = copy.deepcopy(self.fixture.active)
        changed["selected_utc"] = "later-adoption"
        write_json(self.fixture.active_path, changed)
        loaded = CalibrationSnapshot.load(snapshot.snapshot_json)
        self.assertEqual(loaded.data["snapshot_digest"], first_digest)
        with self.assertRaisesRegex(ValueError, "another adoption"):
            self.fixture.freeze()

    def test_source_manifest_tampering_is_rejected(self):
        self.fixture.manifest_path.write_bytes(self.fixture.manifest_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.fixture.freeze()
        self.assertFalse(self.fixture.output.exists())

    def test_frozen_frame_map_tampering_is_rejected(self):
        snapshot = self.fixture.freeze()
        snapshot.frame_map_path(self.fixture.sid).write_bytes(b"{}\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            CalibrationSnapshot.load(snapshot.snapshot_json)

    def test_candidate_and_creating_pointer_rejected(self):
        active = copy.deepcopy(self.fixture.active)
        active["status"] = "candidate"
        write_json(self.fixture.active_path, active)
        with self.assertRaisesRegex(ValueError, "adopted"):
            self.fixture.freeze()
        active["status"] = "adopted_for_current_local_calibration"
        active["calibration"] = "calibration.json.creating"
        write_json(self.fixture.active_path, active)
        with self.assertRaisesRegex(ValueError, "creating"):
            self.fixture.freeze()

    def test_output_cannot_escape_workspace(self):
        with self.assertRaisesRegex(ValueError, "inside the GVHMR"):
            CalibrationSnapshot.freeze(self.fixture.project, self.fixture.sessions, Path(self.temp.name) / "elsewhere")

    def test_reference_plane_is_bound_to_court_registration(self):
        fit_path = self.fixture.base / "cam2-bbox-v1/fit-results.json"
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        fit["reference_plane"]["origin_viewer_m"][2] += .2
        new_hash = write_json(fit_path, fit)
        delivery_path = fit_path.parent / "delivery-freeze.json"
        delivery = json.loads(delivery_path.read_text(encoding="utf-8"))
        delivery["files"][1]["sha256"] = new_hash
        write_json(delivery_path, delivery)
        with self.assertRaisesRegex(ValueError, "court registration"):
            self.fixture.freeze()
        self.assertFalse(self.fixture.output.exists())


if __name__ == "__main__":
    unittest.main()
