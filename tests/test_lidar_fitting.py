"""Geometry-contract tests; run with python -m unittest discover -s tests."""

import unittest

import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.fitting import (
    FittingConfig, SMPLFitter, point_to_surface_squared, robust_surface_loss,
)


class SurfaceDistanceTests(unittest.TestCase):
    def test_triangle_interior_distance_and_gradient(self):
        vertices = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]], requires_grad=True)
        cloud = torch.tensor([[0.25, 0.25, 0.3]])
        distances = point_to_surface_squared(vertices, torch.tensor([[0, 1, 2]]), [cloud])
        # The closest point lies inside the face, not at a mesh vertex.
        self.assertAlmostEqual(distances[0].item(), 0.09, places=6)
        loss = robust_surface_loss(distances)
        loss.backward()
        self.assertTrue(torch.isfinite(vertices.grad).all())
        self.assertLess(vertices.grad[..., 2].sum().item(), 0)
        self.assertAlmostEqual(vertices.grad[..., :2].abs().sum().item(), 0, places=6)

    def test_unobserved_mesh_surface_receives_no_cloud_force(self):
        vertices = torch.tensor([[
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
            [0., 0., 5.], [1., 0., 5.], [0., 1., 5.],
        ]], requires_grad=True)
        distances = point_to_surface_squared(
            vertices, torch.tensor([[0, 1, 2], [3, 4, 5]]), [torch.tensor([[0.2, 0.3, 0.1]])],
        )
        robust_surface_loss(distances).backward()
        self.assertGreater(vertices.grad[0, :3].norm().item(), 0)
        self.assertEqual(vertices.grad[0, 3:].abs().sum().item(), 0)

    def test_frames_cannot_match_each_others_surfaces(self):
        triangle = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        vertices = torch.stack([triangle, triangle + torch.tensor([0., 0., 2.])])
        clouds = [torch.tensor([[0.2, 0.2, 0.1]]), torch.tensor([[0.2, 0.2, 0.2]])]
        distances = point_to_surface_squared(vertices, torch.tensor([[0, 1, 2]]), clouds)
        self.assertAlmostEqual(distances[0].item(), 0.01, places=6)
        self.assertAlmostEqual(distances[1].item(), 3.24, places=5)

    def test_small_body_triangles_are_surfaces(self):
        # A 1cm² face would be discarded by PyTorch3D's default area threshold.
        vertices = torch.tensor([[[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]]])
        distances = point_to_surface_squared(
            vertices, torch.tensor([[0, 1, 2]]), [torch.tensor([[.003, .003, .02]])],
        )
        self.assertAlmostEqual(distances[0].item(), .0004, delta=1e-7)

    def test_empty_cloud_is_not_a_valid_constraint(self):
        vertices = torch.zeros(1, 3, 3)
        with self.assertRaisesRegex(ValueError, "Empty clouds"):
            point_to_surface_squared(vertices, torch.tensor([[0, 1, 2]]), [torch.empty(0, 3)])


@unittest.skipUnless(
    (PROJ_ROOT / "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl").exists()
    and (PROJ_ROOT / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz").exists(),
    "Licensed body model assets are not installed",
)
class PelvisConventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_count = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.fitter = SMPLFitter(device="cpu", config=FittingConfig(conversion_steps=0))

    @classmethod
    def tearDownClass(cls):
        del cls.fitter
        torch.set_num_threads(cls.thread_count)

    def test_pelvis_position_and_native_translation_equivalence(self):
        root = torch.tensor([[0.1, -.2, .3]])
        body = torch.zeros(1, 69)
        shape = torch.tensor([[.5, -.2, .1, 0., 0., 0., 0., 0., 0., 0.]])
        pelvis = torch.tensor([[12., -.5, -1.9]])
        vertices, joints = self.fitter.forward_pelvis(root, body, shape, pelvis)
        self.assertTrue(torch.allclose(joints[:, 0], pelvis, atol=1e-6))
        neutral = self.fitter.student(
            betas=shape, global_orient=root, body_pose=body, transl=torch.zeros(1, 3),
        )
        native_offset = pelvis - neutral.joints[:, 0]
        native = self.fitter.student(
            betas=shape, global_orient=root, body_pose=body, transl=native_offset,
        )
        self.assertLess((vertices - native.vertices).abs().max().item(), 2e-6)
        self.assertGreater((native_offset - pelvis).abs().max().item(), .01)

    def test_cloud_loss_can_change_pelvis_through_native_smpl(self):
        pelvis = torch.tensor([[0., 0., 0.]], requires_grad=True)
        vertices, _ = self.fitter.forward_pelvis(
            torch.zeros(1, 3), torch.zeros(1, 69), torch.zeros(1, 10), pelvis,
        )
        # Surface observations translated by 8cm provide an actual geometric
        # gradient through the body model, not a detached diagnostic loss.
        surface_indices = vertices.detach()[0, :, 0].topk(8).indices
        points = vertices.detach()[0, surface_indices]
        points = points + torch.tensor([.08, 0., 0.])
        distances = point_to_surface_squared(vertices, self.fitter.faces, [points])
        # Every observation is beyond the body's maximum x: no triangular
        # face can be closer than that coordinate bound. This catches the
        # fixed-epsilon barycentric bias of metre-scale PyTorch3D faces.
        lower_bound = points[:, 0] - vertices.detach()[0, :, 0].max()
        self.assertTrue(torch.all(distances[0].sqrt() >= lower_bound - 1e-5))
        robust_surface_loss(distances).backward()
        self.assertTrue(torch.isfinite(pelvis.grad).all())
        self.assertLess(pelvis.grad[0, 0].item(), 0)

    def test_optimizer_recovers_shifted_visible_cloud_and_shared_shape(self):
        class IdentityCalibration:
            reference_plane = {"origin_viewer_m": [0., 0., -10.], "normal_viewer": [0., 0., 1.]}

            def camera_to_viewer(self, points, camera):
                return points

            def camera_rotation_to_viewer(self, camera):
                return torch.eye(3)

        fitter = SMPLFitter(
            device="cpu", conversion_steps=20, topology_vertices=512,
            point_limit=64, qa_point_limit=64, align_translation=False,
        )
        teacher = {
            "body_pose": torch.zeros(2, 63), "betas": torch.zeros(2, 10),
            "global_orient": torch.zeros(2, 3),
            "transl": torch.tensor([[0., 0., 4.], [.03, 0., 4.]]),
        }
        calibration = IdentityCalibration()
        target = fitter.teacher_vertices(teacher, calibration)
        indices = torch.linspace(0, 6889, 64).long()
        # Synthetic surface points exercise optimizer behavior only; these are
        # never exported as LiDAR recordings or training data.
        clouds = [vertices[indices] + torch.tensor([.18, 0., 0.]) for vertices in target]
        result = fitter.fit_batch(teacher, clouds, calibration, iterations=20)
        self.assertTrue(torch.equal(result["shape"][0], result["shape"][1]))
        self.assertTrue(torch.allclose(result["joints"][:, 0], result["transl"], atol=1e-6))
        self.assertEqual(tuple(result["body_pose"].shape), (2, 23, 3))
        for quality in result["qa"]:
            self.assertTrue(quality["finite"])
            self.assertLess(quality["final_p2surface_mean_m"], quality["init_p2surface_mean_m"] * .5)
            self.assertLess(quality["final_p2surface_p90_m"], quality["init_p2surface_p90_m"])


if __name__ == "__main__":
    unittest.main()
