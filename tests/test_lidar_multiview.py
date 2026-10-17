import unittest
import numpy as np

from hmr4d.lidar_mocap.multiview import (
    associate_poses,
    epipolar_errors,
    fundamental_matrix,
    projection_matrix,
    triangulate,
)


class Cameras:
    def camera(self, name):
        return dict(
            K=[[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]],
            R_viewer_to_camera=np.eye(3).tolist(),
            t_m=[0.0, 0, 0] if name == "cam1" else [-2.0, 0, 0],
        )

    def viewer_to_camera(self, xyz, camera):
        return np.asarray(xyz) + np.asarray(self.camera(camera)["t_m"])


def projected(xyz, camera, calibration):
    P = projection_matrix(calibration, camera)
    h = np.column_stack([xyz, np.ones(len(xyz))]) @ P.T
    return np.column_stack([h[:, :2] / h[:, 2:3], np.ones(len(xyz))])


class NativeViewTests(unittest.TestCase):
    def test_same_body_rays_recover_world_geometry_and_permutation(self):
        cal = Cameras()
        rng = np.random.default_rng(5)
        a = rng.normal(size=(17, 3)) * 0.2 + np.array([0, 0, 6])
        b = a + np.array([1, 1.3, 1])
        first = np.stack([projected(a, "cam1", cal), projected(b, "cam1", cal)])
        second = np.stack([projected(b, "cam2", cal), projected(a, "cam2", cal)])
        F = fundamental_matrix(cal)
        self.assertLess(epipolar_errors(first[0], second[1], F).max(), 1e-9)
        recovered = triangulate(first[0], second[1], projection_matrix(cal, "cam1"), projection_matrix(cal, "cam2"))
        np.testing.assert_allclose(recovered, a, atol=1e-10)
        matches = associate_poses(first, second, cal)
        self.assertEqual([(m["cam1_index"], m["cam2_index"]) for m in matches if m["accepted"]], [(0, 1), (1, 0)])

    def test_indistinguishable_native_rays_are_not_forced_into_identity(self):
        cal = Cameras()
        xyz = np.column_stack([np.linspace(-0.1, 0.1, 17), np.linspace(0, 1, 17), np.full(17, 6.0)])
        a = projected(xyz, "cam1", cal)
        b = projected(xyz, "cam2", cal)
        matches = associate_poses(np.stack([a, a]), np.stack([b, b]), cal)
        self.assertTrue(matches and all(not m["accepted"] for m in matches))


if __name__ == "__main__":
    unittest.main()
