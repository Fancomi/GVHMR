import unittest

import numpy as np
import torch

from hmr4d.lidar_mocap.fitting import _projection_loss
from hmr4d.lidar_mocap.segmentation import segment_people


class ImageCalibration:
    def project_torch(self, points, camera, distort=False):
        return points[..., :2], points[..., 2]


class DualLossTests(unittest.TestCase):
    def test_missing_secondary_frames_do_not_reduce_primary_pose_force(self):
        vertices = torch.zeros(2, 17, 3)
        vertices[:, :, 0] = 10
        vertices[:, :, 2] = 1
        primary = torch.zeros(2, 17, 3)
        primary[:, :, 2] = 1
        secondary = torch.zeros_like(primary)
        args = (vertices, torch.eye(17), ImageCalibration())
        single, _ = _projection_loss(*args, primary, "cam2", "undistorted")
        absent, _ = _projection_loss(*args, dict(cam2=primary, cam1=secondary), "cam2", "undistorted")
        self.assertAlmostEqual(single.item(), absent.item(), places=10)
        secondary[0, :, 0] = 10
        secondary[0, :, 2] = 1
        partial, _ = _projection_loss(*args, dict(cam2=primary, cam1=secondary), "cam2", "undistorted")
        self.assertAlmostEqual(partial.item(), single.item() * 0.75, places=10)


class RayCalibration:
    reference_plane = dict(origin_viewer_m=[0, 0, 0], normal_viewer=[0, 1, 0])

    def project(self, points, camera, distort=False):
        xyz = np.asarray(points).copy()
        if camera == "cam1":
            xyz[:, 0] -= 2
        return np.column_stack([1000 + 500 * xyz[:, 0] / xyz[:, 2], 500 - 500 * xyz[:, 1] / xyz[:, 2]]), xyz[:, 2]


class DualSegmentationTests(unittest.TestCase):
    def test_secondary_camera_excludes_another_person_along_primary_ray(self):
        rng = np.random.default_rng(3)
        person = np.column_stack(
            [rng.uniform(-0.07, 0.07, 240), np.linspace(0.08, 1.65, 240), 5 + rng.uniform(-0.03, 0.03, 240)]
        )
        other = person.copy()
        other[:, 2] += 3
        cloud = np.concatenate([person, other]).astype(np.float32)

        def kp(u):
            p = np.tile([u, 435.0, 0.9], (17, 1))
            p[5:7, 1] = 350
            p[11:13, 1] = 420
            p[13:15, 1] = 460
            p[15:17, 1] = 492
            return p

        primary = dict(track_id="person", bbox=np.array([986, 327, 1014, 500]), kp2d=kp(1000))
        before = segment_people(cloud, [primary], RayCalibration())["person"]
        track = dict(primary, views=dict(cam2=primary, cam1=dict(bbox=np.array([788, 327, 812, 500]), kp2d=kp(800))))
        after = segment_people(cloud, [track], RayCalibration())["person"]
        self.assertEqual(before["status"], "segmented")
        self.assertTrue((before["indices"] >= len(person)).any())
        self.assertEqual(after["status"], "segmented")
        self.assertTrue((after["indices"] < len(person)).all())
        np.testing.assert_array_equal(after["points"], cloud[after["indices"]])


if __name__ == "__main__":
    unittest.main()
