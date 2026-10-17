"""Guarded lower-limb refinement for independently observed left/right ambiguity.

Root, pelvis, shape and upper-body parameters stay fixed. Both correspondence
hypotheses use the same recorded pixels and point members; no camera is fitted.
"""

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix

from .fitting import _projection_loss, point_to_surface_squared
from .multiview import epipolar_errors, fundamental_matrix, projection_matrix, triangulate
from hmr4d.utils.body_model.contracts import forward_smpl_at_pelvis

LEG = [13, 14, 15, 16]
SWAPPED = [14, 13, 16, 15]
BODY_JOINTS = [0, 1, 3, 4, 6, 7, 9, 10]  # body-pose indices, excluding the root.


def leg_ray_hypotheses(first, second, calibration):
    a, b = np.asarray(first), np.asarray(second)
    valid = (a[LEG, 2] >= 0.5) & (b[LEG, 2] >= 0.5)
    if not valid.all():
        return None
    F = fundamental_matrix(calibration)
    original = epipolar_errors(a, b, F)[LEG]
    swapped = a.copy()
    swapped[LEG] = a[SWAPPED]
    corrected = epipolar_errors(swapped, b, F)[LEG]
    if np.median(original) < 10 or np.median(original) - np.median(corrected) < 7 or corrected.max() > 6:
        return None
    xyz = triangulate(swapped, b, projection_matrix(calibration, "cam1"), projection_matrix(calibration, "cam2"))[LEG]
    if not np.isfinite(xyz).all() or any(
        (calibration.viewer_to_camera(xyz, cam)[:, 2] <= 0).any() for cam in ["cam1", "cam2"]
    ):
        return None
    return dict(original_epipolar_px=original.tolist(), swapped_epipolar_px=corrected.tolist())


def unordered_leg_errors(vertices, observations, calibration, regressor):
    xyz = np.asarray(regressor) @ np.asarray(vertices)
    result = {}
    for cam, kp in observations.items():
        kp = np.asarray(kp)
        uv, depth = calibration.project(xyz, cam, distort=False)
        candidates = []
        for indices in [LEG, SWAPPED]:
            confidence = kp[indices, 2]
            valid = (confidence >= 0.5) & (depth[LEG] > 0)
            error = np.linalg.norm(uv[LEG] - kp[indices, :2], axis=1)
            candidates.append(float(np.average(error[valid], weights=confidence[valid])) if valid.any() else None)
        result[cam] = min(c for c in candidates if c is not None) if any(c is not None for c in candidates) else None
    return result


def refine_leg_candidates(sample, observations, calibration, model, regressor, *, iterations=100, device="cuda"):
    """Two native-label hypotheses, bounded at 40 degrees from the stored pose."""
    device = torch.device(device)
    as_tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    reference = as_tensor(sample["body_pose"]).reshape(1, 23, 3).repeat(2, 1, 1)
    root = as_tensor(sample["root_pose"]).reshape(1, 3).repeat(2, 1)
    shape = as_tensor(sample["shape"]).reshape(1, 10).repeat(2, 1)
    pelvis = as_tensor(sample["transl"]).reshape(1, 3).repeat(2, 1)
    delta = torch.zeros((2, len(BODY_JOINTS), 3), device=device, requires_grad=True)
    faces = torch.as_tensor(model.faces, device=device, dtype=torch.long)
    reg = as_tensor(regressor)
    cloud = as_tensor(sample["source_cloud"])
    selected = cloud[torch.linspace(0, len(cloud) - 1, min(512, len(cloud)), device=device).long()]
    views = {cam: [] for cam in observations}
    targets = []
    F = fundamental_matrix(calibration)
    for which in range(2):
        a, b = [np.asarray(observations[cam]).copy() for cam in ["cam1", "cam2"]]
        points = a if which == 0 else b
        points[LEG] = points[SWAPPED]
        valid = (a[:, 2] >= 0.35) & (b[:, 2] >= 0.35)
        errors = epipolar_errors(a, b, F)
        threshold = max(6.0, min(10.0, 2.5 * float(np.median(errors[5:17][valid[5:17]]))))
        invalid = valid & (errors > threshold)
        a[invalid, 2] = b[invalid, 2] = 0
        views["cam1"].append(a)
        views["cam2"].append(b)
        targets.append(
            triangulate(a, b, projection_matrix(calibration, "cam1"), projection_matrix(calibration, "cam2"))[LEG]
        )
    views = {cam: as_tensor(np.stack(v)) for cam, v in views.items()}
    targets = as_tensor(np.stack(targets))
    ref_matrix = axis_angle_to_matrix(reference)
    optimizer = torch.optim.Adam([delta], lr=0.015)
    plane_origin = as_tensor(calibration.reference_plane["origin_viewer_m"])
    plane_normal = as_tensor(calibration.reference_plane["normal_viewer"])
    for step in range(iterations):
        body = reference.clone()
        body[:, BODY_JOINTS] += delta
        vertices, joints = forward_smpl_at_pelvis(model, root, body, shape, pelvis)
        distances = point_to_surface_squared(vertices, faces, [selected, selected])
        surface = torch.stack(
            [torch.where(d < 0.08**2, 0.5 * d, 0.08 * (d.sqrt() - 0.04)).mean() for d in distances]
        ).mean()
        projection, _ = _projection_loss(vertices, reg, calibration, views, "cam2", "undistorted", 2.0)
        matrices = axis_angle_to_matrix(body)
        prior = (
            0.002 * (matrices - ref_matrix).square().mean()
            + 0.03 * (matrices[:, [6, 7, 9, 10]] - ref_matrix[:, [6, 7, 9, 10]]).square().mean()
        )
        xyz = torch.einsum("jv,tvc->tjc", reg, vertices)[:, LEG]
        joint_error = (xyz - targets).square().sum(-1).clamp_max(0.15**2).mean()
        ground = torch.relu(-((vertices - plane_origin) @ plane_normal) - 0.02).square().mean()
        loss = surface + 3 * projection + prior + 0.05 * joint_error + 0.5 * ground
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            length = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            delta.mul_(torch.minimum(torch.ones_like(length), np.deg2rad(40) / length))
    with torch.no_grad():
        full_delta = delta.detach().clone()
        delta = torch.cat([full_delta, full_delta * 0.5], dim=0)
        body = reference.repeat(2, 1, 1).clone()
        body[:, BODY_JOINTS] += delta
        vertices, joints = forward_smpl_at_pelvis(
            model, root.repeat(2, 1), body, shape.repeat(2, 1), pelvis.repeat(2, 1)
        )
        actual_cloud = cloud[torch.linspace(0, len(cloud) - 1, min(4096, len(cloud)), device=device).long()]
        distances = point_to_surface_squared(vertices, faces, [actual_cloud] * 4)
        baseline = point_to_surface_squared(as_tensor(sample["vertices"]).reshape(1, 6890, 3), faces, [actual_cloud])[
            0
        ].sqrt()
        before_errors = unordered_leg_errors(sample["vertices"], observations, calibration, regressor)
        before_penetration = float(torch.relu(-((as_tensor(sample["vertices"]) - plane_origin) @ plane_normal)).max())
        results = []
        for which in range(4):
            geometry = vertices[which].cpu().numpy()
            errors = unordered_leg_errors(geometry, observations, calibration, regressor)
            d = distances[which].sqrt()
            surface_mean = float(d.mean())
            surface_p90 = float(torch.quantile(d, 0.9))
            penetration = float(torch.relu(-((vertices[which] - plane_origin) @ plane_normal)).max())
            improvement = np.mean([before_errors[c] - errors[c] for c in observations])
            accepted = (
                surface_mean <= float(baseline.mean()) + 0.001
                and surface_p90 <= float(torch.quantile(baseline, 0.9)) + 0.002
                and improvement >= max(1.0, 0.10 * np.mean(list(before_errors.values())))
                and all(errors[c] <= before_errors[c] + 1 for c in observations)
                and penetration <= before_penetration + 0.01
            )
            results.append(
                dict(
                    camera_permuted=["cam1", "cam2"][which % 2],
                    correction_fraction=1.0 if which < 2 else 0.5,
                    accepted=bool(accepted),
                    unordered_leg_before_px=before_errors,
                    unordered_leg_after_px=errors,
                    mean_leg_improvement_px=float(improvement),
                    surface_mean_m=surface_mean,
                    surface_p90_m=surface_p90,
                    baseline_surface_mean_m=float(baseline.mean()),
                    baseline_surface_p90_m=float(torch.quantile(baseline, 0.9)),
                    max_penetration_m=penetration,
                    vertices=geometry,
                    joints=joints[which].cpu().numpy(),
                    body_pose=body[which].cpu().numpy(),
                    max_delta_angle_deg=float(delta[which].norm(dim=-1).max().cpu() * 180 / np.pi),
                )
            )
    return results
