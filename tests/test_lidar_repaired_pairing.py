from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hmr4d.lidar_mocap.repaired_pairing import reassociate_native_frame


class Camera:
    def camera(self, name):
        return dict(image_size=[100, 100])

    def project(self, vertices, name, distort=False):
        return vertices[:, :2], vertices[:, 2]


class RepairedPairingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "fit.npz"
        vertices = np.tile([[10, 10, 1], [50, 50, 1]], (3445, 1))
        np.savez_compressed(self.path, vertices=vertices)
        native = dict(record_index=7, media_pts_us=100, common_time_us=1000)
        self.frame = dict(
            lidar_index=1,
            scan_mid_us=1000,
            cameras=dict(cam1=native),
            athletes=[dict(motion_id="new:12:0", fit_path=str(self.path), accepted=True)],
        )
        self.cached = dict(
            camera_name="cam1",
            lidar_index=1,
            scan_mid_us=1000,
            paired_cameras=dict(cam1=native),
            camera_native_source=native,
            native_bgr_image_sha256="a" * 64,
            undistorted_rgb_image_sha256="b" * 64,
            detector_boxes_xyxy=[[10, 10, 50, 50]],
            detector_confidences=[0.9],
            athletes=[dict(motion_id="old:99:0")],
        )

    def test_native_detections_get_new_identity_and_fit_binding(self):
        result = reassociate_native_frame(self.frame, self.cached, "cam1", Camera(), dict(report="sealed"))
        person = result["athletes"][0]
        self.assertEqual(person["motion_id"], "new:12:0")
        self.assertEqual(person["status"], "detected")
        self.assertEqual(len(person["fit_sha256"]), 64)
        self.assertEqual(result["native_bgr_image_sha256"], "a" * 64)
        self.assertEqual(self.cached["athletes"][0]["motion_id"], "old:99:0")

    def test_rejected_other_body_still_prevents_ambiguous_association(self):
        self.frame["athletes"].append(dict(motion_id="other:2:0", fit_path=str(self.path), accepted=False))
        result = reassociate_native_frame(self.frame, self.cached, "cam1", Camera(), {})
        self.assertTrue(all(p["status"] == "excluded" for p in result["athletes"]))

    def test_changed_native_pts_cannot_reuse_detections(self):
        changed = deepcopy(self.cached)
        changed["camera_native_source"] = dict(changed["camera_native_source"], media_pts_us=101)
        with self.assertRaisesRegex(ValueError, "PTS/index"):
            reassociate_native_frame(self.frame, changed, "cam1", Camera(), {})

    def test_missing_pixel_provenance_cannot_reuse_detections(self):
        del self.cached["native_bgr_image_sha256"]
        with self.assertRaisesRegex(ValueError, "pixel digest"):
            reassociate_native_frame(self.frame, self.cached, "cam1", Camera(), {})
