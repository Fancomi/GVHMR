import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from hmr4d.lidar_mocap.leg_selection import select_correction_sequence


def state(degrees, gain):
    matrix = Rotation.from_rotvec([np.deg2rad(degrees), 0, 0]).as_matrix()
    return dict(delta_rotations=np.broadcast_to(matrix, (8, 3, 3)), gain_px=gain)


class CorrectionSequenceTests(unittest.TestCase):
    def test_large_isolated_improvement_cannot_create_a_jump(self):
        frames = [
            dict(time_us=i * 100000, states=[state(0, 0)] + ([state(40, 100), state(20, 20)] if i == 1 else []))
            for i in range(3)
        ]
        self.assertEqual(select_correction_sequence(frames), [0, 2, 0])

    def test_supported_contiguous_correction_can_rise_and_fall(self):
        frames = [
            dict(time_us=i * 100000, states=[state(0, 0), state(degrees, 20)])
            for i, degrees in enumerate([0, 20, 40, 20, 0])
        ]
        self.assertEqual(select_correction_sequence(frames), [1, 1, 1, 1, 1])

    def test_missing_time_is_not_interpolated(self):
        frames = [dict(time_us=0, states=[state(0, 0)]), dict(time_us=900000, states=[state(0, 0), state(40, 100)])]
        self.assertEqual(select_correction_sequence(frames), [0, 0])
