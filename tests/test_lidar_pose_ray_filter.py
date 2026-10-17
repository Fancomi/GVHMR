import unittest
from unittest.mock import patch

import numpy as np

from hmr4d.lidar_mocap.pose_ray_filter import RayFilteredPoseProvider


class JointRayTests(unittest.TestCase):
    def test_contradictory_ankle_is_excluded_without_changing_coordinates(self):
        provider = RayFilteredPoseProvider.__new__(RayFilteredPoseProvider)
        provider.selected = {}
        provider.policy_sha256 = "policy"
        provider.F = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], float)
        a = np.ones((17, 3), np.float32)
        a[:, :2] = np.arange(34).reshape(17, 2)
        b = a.copy()
        b[15, 1] += 20
        track = dict(views=dict(cam1=dict(kp2d=a.copy()), cam2=dict(kp2d=b.copy())), pose_evidence={})
        with patch("hmr4d.lidar_mocap.pose_ray_filter.NativePoseProvider.bind", return_value=([track], [])):
            result, _ = provider.bind(type("Session", (), dict(session_id="s"))(), dict(lidar_index=0), [])
        assert result[0]["views"]["cam1"]["kp2d"][15, 2] == 0
        assert result[0]["views"]["cam2"]["kp2d"][15, 2] == 0
        assert result[0]["views"]["cam2"]["kp2d"][14, 2] == 1
        np.testing.assert_array_equal(result[0]["views"]["cam1"]["kp2d"][:, :2], a[:, :2])
        np.testing.assert_array_equal(result[0]["views"]["cam2"]["kp2d"][:, :2], b[:, :2])
