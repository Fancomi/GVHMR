"""Geometry-contract tests; run with python -m unittest discover -s tests."""

import unittest

import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.fitting import (
    FittingConfig, SMPLFitter, candidate_point_to_surface_squared,
    motion_reference_loss, point_to_surface_squared, robust_surface_loss,
    rotation_motion_loss, vertex_face_adjacency,
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


class CandidateSurfaceTests(unittest.TestCase):
    def test_triangle_interior_uses_surface_and_preserves_gradient(self):
        vertices = torch.tensor([[[0., 0., 0.], [.01, 0., 0.], [0., .01, 0.]]], requires_grad=True)
        faces = torch.tensor([[0, 1, 2]])
        clouds = [torch.tensor([[.003, .003, .02]])]
        distances = candidate_point_to_surface_squared(vertices, faces, clouds, neighbors=1)
        self.assertAlmostEqual(distances[0].item(), .0004, delta=1e-8)
        robust_surface_loss(distances).backward()
        self.assertTrue(torch.isfinite(vertices.grad).all())
        self.assertLess(vertices.grad[..., 2].sum().item(), 0)
        self.assertAlmostEqual(vertices.grad[..., :2].abs().sum().item(), 0, places=6)

    def test_different_frames_and_ragged_clouds_cannot_cross_match(self):
        triangle = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        vertices = torch.stack([triangle, triangle + torch.tensor([0., 0., 2.])])
        faces = torch.tensor([[0, 1, 2]])
        clouds = [torch.tensor([[.2, .2, .1]]), torch.tensor([[.2, .2, .2], [.3, .2, .3]])]
        fast = candidate_point_to_surface_squared(vertices, faces, clouds)
        full = point_to_surface_squared(vertices, faces, clouds)
        self.assertEqual([len(value) for value in fast], [1, 2])
        for candidate, reference in zip(fast, full):
            self.assertTrue(torch.allclose(candidate, reference, atol=1e-6, rtol=1e-6))

    def test_missing_global_face_overestimates_instead_of_underestimating(self):
        # The large triangle has the globally closest surface, but its vertices
        # are distant. K=1 intentionally selects only the small raised triangle.
        vertices = torch.tensor([[
            [-10., -10., 0.], [10., -10., 0.], [0., 10., 0.],
            [-.1, -.1, .1], [.1, -.1, .1], [0., .1, .1],
        ]])
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
        cloud = [torch.tensor([[0., 0., 0.]])]
        fast = candidate_point_to_surface_squared(vertices, faces, cloud, neighbors=1)[0]
        full = point_to_surface_squared(vertices, faces, cloud)[0]
        self.assertAlmostEqual(fast.item(), .01, places=6)
        self.assertAlmostEqual(full.item(), 0, places=6)
        self.assertGreater(fast.item(), full.item())

    def test_degenerate_face_reduces_to_finite_segments(self):
        vertices = torch.tensor([[[0., 0., 0.], [.01, 0., 0.], [.02, 0., 0.]]], requires_grad=True)
        faces = torch.tensor([[0, 1, 2]])
        fast = candidate_point_to_surface_squared(vertices, faces, [torch.tensor([[.01, .03, 0.]])])
        self.assertAlmostEqual(fast[0].item(), .0009, places=7)
        fast[0].sum().backward()
        self.assertTrue(torch.isfinite(vertices.grad).all())

    def test_default_optimizer_retains_full_face_backend(self):
        self.assertEqual(FittingConfig().optimizer_surface, "full_faces")
        with self.assertRaises(ValueError):
            FittingConfig(optimizer_surface="nearest_vertex")


class ObservedMotionTests(unittest.TestCase):
    def test_real_fast_translation_is_preserved_while_correction_jitter_has_gradient(self):
        # This is a fast genuine 80cm movement, not a zero-velocity prior.
        observed = torch.tensor([[0., 0., 0.], [.8, 0., .3], [1.4, 0., .5]])
        times = torch.tensor([1791794092000000, 1791794092100000, 1791794092200000])
        fitted = (observed + torch.tensor([.2, 0., -.1])).requires_grad_()
        self.assertAlmostEqual(motion_reference_loss(fitted, observed, times).item(), 0., places=12)
        jitter = fitted.detach().clone()
        jitter[1, 0] += .25
        jitter.requires_grad_()
        loss = motion_reference_loss(jitter, observed, times)
        loss.backward()
        self.assertGreater(loss.item(), 0.)
        self.assertGreater(jitter.grad[1, 0].item(), 0.)

    def test_real_time_scaling_and_missing_scan_boundary(self):
        observed = torch.zeros(2, 3)
        fitted = torch.tensor([[0., 0., 0.], [.2, 0., 0.]])
        fast = motion_reference_loss(fitted, observed, [1_000_000, 1_100_000])
        slow = motion_reference_loss(fitted, observed, [1_000_000, 1_200_000])
        self.assertAlmostEqual(fast.item(), slow.item() * 4., places=7)
        separated = motion_reference_loss(fitted, observed, [1_000_000, 1_400_000])
        self.assertEqual(separated.item(), 0.)

    def test_relative_rotations_preserve_quick_root_and_racket_motion(self):
        from pytorch3d.transforms import axis_angle_to_matrix

        reference = axis_angle_to_matrix(torch.tensor([[0., 0., 0.], [0., 0., 1.2], [0., 0., 1.8]]))
        correction = axis_angle_to_matrix(torch.tensor([.1, -.2, 0.]))
        times = [1_000_000, 1_100_000, 1_200_000]
        # World root increment: a constant right correction is preserved.
        self.assertLess(rotation_motion_loss(reference @ correction, reference, times).item(), 1e-12)
        # Local articulated joint increment: a constant left correction is preserved.
        self.assertLess(rotation_motion_loss(correction @ reference, reference, times, local=True).item(), 1e-12)

    def test_nonincreasing_physical_times_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly increase"):
            motion_reference_loss(torch.zeros(2, 3), torch.zeros(2, 3), [100, 100])


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

    def test_candidate_faces_against_full_smpl_surface(self):
        generator = torch.Generator().manual_seed(719)
        root = torch.tensor([[.1, -.2, .3]])
        body = torch.randn(1, 69, generator=generator) * .2
        shape = torch.randn(1, 10, generator=generator) * .3
        vertices, _ = self.fitter.forward_pelvis(root, body, shape, torch.tensor([[12., .8, -1.9]]))
        vertices = vertices.detach()
        face_ids = torch.randint(len(self.fitter.faces), (128,), generator=generator)
        triangles = vertices[0, self.fitter.faces[face_ids]]
        weights = torch.rand(128, 3, generator=generator)
        weights = weights / weights.sum(-1, keepdim=True)
        surface = (triangles * weights[..., None]).sum(-2)
        shifted = surface + torch.randn(128, 3, generator=generator) * .04
        scattered = vertices.mean(1)[0] + torch.randn(128, 3, generator=generator) * .8
        clouds = [torch.cat([shifted, scattered])]
        full = point_to_surface_squared(vertices, self.fitter.faces, clouds)[0].sqrt()
        fast = candidate_point_to_surface_squared(
            vertices, self.fitter.faces, clouds, self.fitter.face_adjacency,
        )[0].sqrt()
        # The candidate surface is a subset, so its distance is an upper bound.
        self.assertTrue(torch.all(fast >= full - 1e-5))
        # Around a realistically shifted body surface, eight vertex neighbours
        # recover the actual closest face to millimetre precision.
        self.assertLess((fast[:128] - full[:128]).abs().max().item(), .001)

    def test_adjacency_contains_only_incident_faces(self):
        adjacency = vertex_face_adjacency(self.fitter.faces, 6890)
        for vertex in (0, 100, 3000, 6889):
            face_ids = adjacency[vertex]
            face_ids = face_ids[face_ids >= 0]
            self.assertGreater(len(face_ids), 0)
            self.assertTrue((self.fitter.faces[face_ids] == vertex).any(-1).all())

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
            optimizer_surface="candidate_faces",
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
        self.assertEqual(result["optimizer_surface"], "candidate_faces")
        for quality in result["qa"]:
            self.assertTrue(quality["finite"])
            self.assertLess(quality["final_p2surface_mean_m"], quality["init_p2surface_mean_m"] * .5)
            self.assertLess(quality["final_p2surface_p90_m"], quality["init_p2surface_p90_m"])

    def test_batch_state_preserves_observed_jump_and_resets_after_missing_interval(self):
        class IdentityCalibration:
            reference_plane = {"origin_viewer_m": [0., 0., -10.], "normal_viewer": [0., 0., 1.]}

            def camera_to_viewer(self, points, camera):
                return points

            def camera_rotation_to_viewer(self, camera):
                return torch.eye(3)

        calibration = IdentityCalibration()
        teacher = {"body_pose": torch.zeros(1, 63), "betas": torch.zeros(1, 10),
                   "global_orient": torch.zeros(1, 3), "transl": torch.tensor([[0., 0., 4.]])}
        target = self.fitter.teacher_vertices(teacher, calibration)
        cloud = target[0, torch.linspace(0, 6889, 16).long()]
        initial = self.fitter.fit_batch(teacher, [cloud], calibration, iterations=0, timestamps_us=[1_000_000])
        previous = {key: initial[key][0] for key in ('shape', 'transl', 'root_pose', 'body_pose')}
        previous.update({key: initial['continuity'][key][0] for key in
                         ('scan_time_us', 'aligned_transl', 'teacher_root_pose', 'teacher_body_pose')})
        previous.update(shape_fixed=True)
        previous['transl'] = previous['transl'] + torch.tensor([.07, 0., 0.])
        moved_teacher = dict(teacher, transl=teacher['transl'] + torch.tensor([.6, 0., .2]))
        moved_cloud = cloud + torch.tensor([.6, 0., .2])
        continued = self.fitter.fit_batch(moved_teacher, [moved_cloud], calibration, previous=previous,
                                         iterations=0, timestamps_us=[1_100_000])
        self.assertTrue(continued['continuity']['previous_boundary_connected'])
        self.assertTrue(torch.allclose(continued['transl'][0] - previous['transl'],
                                       torch.tensor([.6, 0., .2]), atol=1e-5))
        disconnected = self.fitter.fit_batch(moved_teacher, [moved_cloud], calibration, previous=previous,
                                            iterations=0, timestamps_us=[1_400_000])
        self.assertFalse(disconnected['continuity']['previous_boundary_connected'])
        self.assertTrue(torch.allclose(disconnected['transl'][0],
                                       disconnected['continuity']['mesh_median_aligned_transl'][0], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
