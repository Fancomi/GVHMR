import unittest
import numpy as np

from hmr4d.lidar_mocap.leg_refinement import leg_ray_hypotheses, unordered_leg_errors
from test_lidar_multiview import Cameras as BaseCameras, projected


class Cameras(BaseCameras):
    def project(self, xyz, cam, distort=False):
        return projected(xyz, cam, self)[:, :2], self.viewer_to_camera(xyz, cam)[:, 2]


class LegCorrespondenceTests(unittest.TestCase):
    def test_opposite_view_leg_label_swap_has_a_geometric_witness(self):
        cal = Cameras()
        xyz = np.zeros((17, 3))
        xyz[:, 2] = 6
        xyz[13:17, 1] = [-0.4, 0.5, -0.2, 1.1]
        a, b = projected(xyz, "cam1", cal), projected(xyz, "cam2", cal)
        b[[13, 14, 15, 16]] = b[[14, 13, 16, 15]]
        before = b.copy()
        result = leg_ray_hypotheses(a, b, cal)
        self.assertIsNotNone(result)
        self.assertLess(max(result["swapped_epipolar_px"]), 1e-8)
        np.testing.assert_array_equal(before, b)
        error = unordered_leg_errors(xyz, dict(cam1=a, cam2=b), cal, np.eye(17))
        self.assertLess(max(error.values()), 1e-8)

    def test_collinear_standing_legs_do_not_force_a_swap(self):
        cal = Cameras()
        xyz = np.zeros((17, 3))
        xyz[:, 2] = 6
        a, b = projected(xyz, "cam1", cal), projected(xyz, "cam2", cal)
        self.assertIsNone(leg_ray_hypotheses(a, b, cal))
