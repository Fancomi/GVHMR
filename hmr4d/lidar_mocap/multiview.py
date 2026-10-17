"""Fixed-camera native COCO correspondence and triangulation; no camera refinement."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def projection_matrix(calibration, camera):
    info = calibration.camera(camera)
    K = np.asarray(info["K"], dtype=np.float64)
    R = np.asarray(info["R_viewer_to_camera"], dtype=np.float64)
    t = np.asarray(info["t_m"], dtype=np.float64).reshape(3, 1)
    return K @ np.concatenate([R, t], axis=1)


def fundamental_matrix(calibration, first="cam1", second="cam2"):
    a, b = calibration.camera(first), calibration.camera(second)
    ra, rb = np.asarray(a["R_viewer_to_camera"]), np.asarray(b["R_viewer_to_camera"])
    ta, tb = np.asarray(a["t_m"]).reshape(3), np.asarray(b["t_m"]).reshape(3)
    R = rb @ ra.T
    t = tb - R @ ta
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.linalg.inv(np.asarray(b["K"])).T @ tx @ R @ np.linalg.inv(np.asarray(a["K"]))
    return F / np.linalg.norm(F)


def epipolar_errors(first, second, F):
    a = np.column_stack([np.asarray(first)[:, :2], np.ones(len(first))])
    b = np.column_stack([np.asarray(second)[:, :2], np.ones(len(second))])
    lb, la = a @ F.T, b @ F
    residual = np.abs(np.sum(b * lb, axis=1))
    return (
        0.5
        * residual
        * (
            1 / np.maximum(np.linalg.norm(lb[:, :2], axis=1), 1e-12)
            + 1 / np.maximum(np.linalg.norm(la[:, :2], axis=1), 1e-12)
        )
    )


def triangulate(first, second, p_first, p_second):
    first, second = np.asarray(first), np.asarray(second)
    points = []
    for a, b in zip(first, second):
        A = np.stack(
            [
                a[0] * p_first[2] - p_first[0],
                a[1] * p_first[2] - p_first[1],
                b[0] * p_second[2] - p_second[0],
                b[1] * p_second[2] - p_second[1],
            ]
        )
        _, _, v = np.linalg.svd(A)
        h = v[-1]
        points.append(h[:3] / h[3] if abs(h[3]) > 1e-12 else np.full(3, np.nan))
    return np.asarray(points)


def associate_poses(first, second, calibration, maximum_epipolar_px=6.0, minimum_margin_px=1.0):
    """Return unique native detections paired by measured joint rays.

    Geometric ambiguity remains explicit. No nearest fitted-body residual is
    used to select a partner, and no absent keypoint is invented.
    """
    first, second = np.asarray(first), np.asarray(second)
    cost = np.full((len(first), len(second)), 1e6)
    diagnostics = {}
    F = fundamental_matrix(calibration)
    selected = np.arange(5, 17)
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            valid = (a[:, 2] >= 0.35) & (b[:, 2] >= 0.35)
            body = selected[valid[selected]]
            if len(body) < 6 or valid[[5, 6, 11, 12]].sum() < 3:
                continue
            errors = epipolar_errors(a, b, F)[body]
            median, p90 = float(np.median(errors)), float(np.quantile(errors, 0.9))
            cost[i, j] = median + 0.15 * p90
            diagnostics[i, j] = dict(median_epipolar_px=median, p90_epipolar_px=p90, valid_joints=len(body))
    result = []
    if cost.size:
        rr, cc = linear_sum_assignment(cost)
        for i, j in zip(rr, cc):
            metrics = diagnostics.get((i, j))
            if metrics is None:
                continue
            alternatives = np.concatenate([np.delete(cost[i], j), np.delete(cost[:, j], i)])
            margin = float(alternatives.min() - cost[i, j]) if len(alternatives) else None
            accepted = metrics["median_epipolar_px"] <= maximum_epipolar_px and (
                margin is None or margin >= minimum_margin_px
            )
            valid = (first[i, :, 2] >= 0.35) & (second[j, :, 2] >= 0.35)
            xyz = triangulate(
                first[i], second[j], projection_matrix(calibration, "cam1"), projection_matrix(calibration, "cam2")
            )
            depth1 = calibration.viewer_to_camera(xyz, "cam1")[:, 2]
            depth2 = calibration.viewer_to_camera(xyz, "cam2")[:, 2]
            visible = np.isfinite(xyz).all(1) & (depth1 > 0) & (depth2 > 0) & valid
            accepted = bool(accepted and visible[[5, 6, 11, 12]].sum() >= 3)
            result.append(
                dict(
                    cam1_index=int(i),
                    cam2_index=int(j),
                    accepted=accepted,
                    margin_px=margin,
                    **metrics,
                    triangulated_joints=np.where(visible[:, None], xyz, 0).tolist(),
                    triangulated_valid=visible.tolist(),
                    rejection_reason=None if accepted else "ambiguous_or_inconsistent_native_rays",
                )
            )
    return result


def match_teacher_observations(tracks, observed):
    """Match native primary-camera poses without relying on monocular depth."""
    observed = np.asarray(observed, dtype=np.float64).reshape(-1, 17, 3)
    cost = np.full((len(tracks), len(observed)), 1e6)
    for i, track in enumerate(tracks):
        a = np.asarray(track["kp2d"])
        height = max(float(track["bbox"][3] - track["bbox"][1]), 35.0)
        for j, b in enumerate(observed):
            valid = (a[:, 2] >= 0.35) & (b[:, 2] >= 0.35)
            body = np.arange(5, 17)[valid[5:17]]
            if len(body) < 5:
                continue
            cost[i, j] = float(np.median(np.linalg.norm(a[body, :2] - b[body, :2], axis=1))) / height
    result = {}
    if cost.size:
        rr, cc = linear_sum_assignment(cost)
        for i, j in zip(rr, cc):
            alternatives = np.concatenate([np.delete(cost[i], j), np.delete(cost[:, j], i)])
            margin = float(alternatives.min() - cost[i, j]) if len(alternatives) else 1e6
            if cost[i, j] <= 0.2 and margin >= 0.035:
                result[int(i)] = int(j)
    return result
