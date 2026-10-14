"""Convert the video teacher to neutral SMPL and fit *visible* LiDAR surfaces.

All output geometry is in the calibrated viewer coordinate system, in metres.
``transl`` is the pelvis position expected by the downstream PySMPL model, not
the additive translation parameter of the smplx package.  A call must contain
one person track; its ten shape coefficients are shared across the window.

The LiDAR term is deliberately point-to-triangle only.  Missing returns on an
occluded limb are not evidence that the limb should move onto another surface.
The video pose, calibrated reprojection and physical ground plane constrain
those parts of the body.  This module reports measurements; the pipeline owns
the acceptance policy for training labels.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.ops import knn_points
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from hmr4d import PROJ_ROOT
from hmr4d.utils.body_model.contracts import forward_smpl_at_pelvis, map_smplx_vertices


@dataclass(frozen=True)
class FittingConfig:
    conversion_steps: int = 60
    conversion_lr: float = 0.02
    learning_rate: float = 0.015
    point_limit: int = 512
    qa_point_limit: int = 4096
    topology_vertices: int = 1723
    cloud_huber_m: float = 0.08
    teacher_geometry_weight: float = 0.08
    pose_prior_weight: float = 0.002
    shape_prior_weight: float = 0.00002
    root_prior_weight: float = 0.002
    reprojection_weight: float = 0.3
    floor_weight: float = 0.5
    temporal_weight: float = 0.02
    ground_tolerance_m: float = 0.02
    max_beta: float = 4.0
    align_translation: bool = True
    optimizer_surface: str = "full_faces"
    candidate_neighbors: int = 8

    def __post_init__(self):
        if min(self.point_limit, self.qa_point_limit, self.topology_vertices) < 1:
            raise ValueError("Sampling limits must be positive")
        if min(self.learning_rate, self.conversion_lr, self.cloud_huber_m, self.max_beta) <= 0:
            raise ValueError("Optimizer rates, robust distance and shape bound must be positive")
        if self.optimizer_surface not in {"full_faces", "candidate_faces"}:
            raise ValueError("optimizer_surface must be 'full_faces' or 'candidate_faces'")
        if self.candidate_neighbors < 1:
            raise ValueError("candidate_neighbors must be positive")


def point_to_surface_squared(
    vertices: torch.Tensor, faces: torch.Tensor, clouds: Sequence[torch.Tensor]
) -> list[torch.Tensor]:
    """Exact differentiable one-sided squared point-to-triangle distances.

    The PyTorch3D default triangle-area threshold is inappropriate for a body
    mesh in metres: it would classify many valid SMPL faces as line segments.
    A threshold of 1e-12 m² preserves the actual triangular surfaces.  The
    library also contains a fixed 1e-8 barycentric denominator epsilon: using
    metres directly can classify points outside small hand faces as inside.
    Locally centred millimetres avoid that bias, and the squared distances
    are converted back to m². Each frame is packed separately, so nearby
    people/frames cannot cross-match.
    """
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape (T, V, 3)")
    if len(clouds) != len(vertices):
        raise ValueError("Each mesh must have exactly one cloud")
    if not clouds:
        return []
    counts = [len(cloud) for cloud in clouds]
    if any(count == 0 for count in counts):
        raise ValueError("Empty clouds cannot constrain a fitted label")
    local_origins = vertices.detach().mean(dim=1)
    scale = 1000.0
    points = torch.cat([
        (cloud.to(device=vertices.device, dtype=vertices.dtype) - origin) * scale
        for cloud, origin in zip(clouds, local_origins)
    ])
    counts_tensor = torch.tensor(counts, device=vertices.device, dtype=torch.long)
    point_starts = torch.cat([counts_tensor.new_zeros(1), counts_tensor.cumsum(0)[:-1]])
    faces = faces.to(device=vertices.device, dtype=torch.long)
    local_vertices = (vertices - local_origins[:, None]) * scale
    triangles = local_vertices[:, faces].reshape(-1, 3, 3).contiguous()
    triangle_starts = torch.arange(len(vertices), device=vertices.device) * len(faces)
    squared = point_face_distance(
        points.contiguous(), point_starts, triangles, triangle_starts,
        max(counts), 1e-12 * scale * scale,
    ) / (scale * scale)
    return list(squared.split(counts))


def vertex_face_adjacency(faces: torch.Tensor, vertex_count: int) -> torch.Tensor:
    """Build a fixed mesh-topology lookup, padding missing incidences with -1."""
    cpu_faces = faces.detach().to(device="cpu", dtype=torch.long)
    if cpu_faces.ndim != 2 or cpu_faces.shape[1] != 3 or len(cpu_faces) == 0:
        raise ValueError("faces must be a nonempty (F,3) array")
    if vertex_count < 1 or int(cpu_faces.min()) < 0 or int(cpu_faces.max()) >= vertex_count:
        raise ValueError("Faces reference invalid vertices")
    incident = [[] for _ in range(vertex_count)]
    for face_id, triangle in enumerate(cpu_faces.tolist()):
        for vertex_id in set(triangle):
            incident[vertex_id].append(face_id)
    degree = max(map(len, incident))
    lookup = torch.full((vertex_count, degree), -1, dtype=torch.long)
    for vertex_id, face_ids in enumerate(incident):
        if face_ids:
            lookup[vertex_id, :len(face_ids)] = torch.tensor(face_ids, dtype=torch.long)
    return lookup.to(faces.device)


def _paired_triangle_squared(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Distances to explicit candidate triangles, in locally centred mm².

    points has shape (...,3); triangles has shape (...,M,3,3). For an interior
    projection use the plane distance, otherwise use the closest of all three
    finite edge segments. Degenerate faces reduce to their edges/vertices.
    Gradients flow to the selected triangle vertices, including edge cases.
    """
    a, b, c = triangles.unbind(dim=-2)
    point = points.unsqueeze(-2)
    ab, ac = b - a, c - a
    ap = point - a
    d00 = ab.square().sum(-1)
    d01 = (ab * ac).sum(-1)
    d11 = ac.square().sum(-1)
    d20 = (ap * ab).sum(-1)
    d21 = (ap * ac).sum(-1)
    normal = torch.cross(ab, ac, dim=-1)
    normal_squared = normal.square().sum(-1)
    # The cross-product expression avoids cancellation of nearly equal dot
    # products. Millimetre units keep the epsilon below meaningful face sizes.
    denominator = normal_squared.clamp_min(1e-12)
    u = (d11 * d20 - d01 * d21) / denominator
    v = (d00 * d21 - d01 * d20) / denominator
    inside = (u >= 0) & (v >= 0) & (u + v <= 1) & (normal_squared > 1e-12)
    plane_squared = (ap * normal).sum(-1).square() / denominator

    edge_squared = []
    for start, end in ((a, b), (a, c), (b, c)):
        edge = end - start
        relative = point - start
        fraction = ((relative * edge).sum(-1) / edge.square().sum(-1).clamp_min(1e-12)).clamp(0, 1)
        residual = relative - fraction.unsqueeze(-1) * edge
        edge_squared.append(residual.square().sum(-1))
    boundary_squared = torch.stack(edge_squared, dim=-1).min(dim=-1).values
    return torch.where(inside, plane_squared, boundary_squared)


def candidate_point_to_surface_squared(
    vertices: torch.Tensor, faces: torch.Tensor, clouds: Sequence[torch.Tensor],
    adjacency: Optional[torch.Tensor] = None, neighbors: int = 8,
) -> list[torch.Tensor]:
    """Fast one-sided distance over faces incident to the K nearest vertices.

    Each optimization call refreshes KNN from the current mesh. The resulting
    candidate surface distance is exact *within that candidate set*, which
    may omit the globally closest triangle. It is an optimization term only;
    acceptance metrics always use ``point_to_surface_squared`` over all faces.
    KNN indices are discrete and detached; triangle vertices retain gradients.
    """
    if vertices.ndim != 3 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape (T,V,3)")
    if len(clouds) != len(vertices):
        raise ValueError("Each mesh must have exactly one cloud")
    if not clouds:
        return []
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    counts = [len(cloud) for cloud in clouds]
    if any(count == 0 for count in counts):
        raise ValueError("Empty clouds cannot constrain a fitted label")
    faces = faces.to(device=vertices.device, dtype=torch.long)
    if adjacency is None:
        adjacency = vertex_face_adjacency(faces, vertices.shape[1])
    else:
        adjacency = adjacency.to(device=vertices.device, dtype=torch.long)
    if adjacency.ndim != 2 or len(adjacency) != vertices.shape[1]:
        raise ValueError("Invalid vertex-face adjacency")

    scale = 1000.0
    local_origins = vertices.detach().mean(dim=1)
    local_vertices = (vertices - local_origins[:, None]) * scale
    local_points = torch.stack([
        torch.nn.functional.pad(
            (cloud.to(device=vertices.device, dtype=vertices.dtype) - origin) * scale,
            (0, 0, 0, max(counts) - len(cloud)),
        )
        for cloud, origin in zip(clouds, local_origins)
    ])
    lengths = torch.tensor(counts, device=vertices.device, dtype=torch.long)
    with torch.no_grad():
        nearest = knn_points(
            local_points.detach(), local_vertices.detach(), lengths1=lengths,
            K=min(neighbors, vertices.shape[1]), return_nn=False,
        ).idx
        candidate_faces = adjacency[nearest].flatten(start_dim=-2)
        candidate_valid = candidate_faces >= 0
        candidate_vertex_ids = faces[candidate_faces.clamp_min(0)]
    frame_ids = torch.arange(len(vertices), device=vertices.device)[:, None, None, None]
    triangles = local_vertices[frame_ids, candidate_vertex_ids]
    squared = _paired_triangle_squared(local_points, triangles)
    squared = squared.masked_fill(~candidate_valid, torch.inf).min(dim=-1).values / (scale * scale)
    return [distance[:count] for distance, count in zip(squared, counts)]


def robust_surface_loss(distances_squared: Sequence[torch.Tensor], delta: float = 0.08):
    """Per-frame Huber loss in squared-metre units, then equal frame weighting."""
    if not distances_squared:
        raise ValueError("No observed surfaces")
    if delta <= 0:
        raise ValueError("Huber distance must be positive")
    losses = []
    for squared in distances_squared:
        distance = torch.sqrt(squared.clamp_min(1e-12))
        losses.append(torch.where(
            distance <= delta, 0.5 * squared,
            delta * (distance - 0.5 * delta),
        ).mean())
    return torch.stack(losses).mean()


def _sample_cloud(cloud: torch.Tensor, limit: int) -> torch.Tensor:
    if limit < 1:
        raise ValueError("Point limit must be positive")
    if len(cloud) <= limit:
        return cloud
    indices = torch.linspace(0, len(cloud) - 1, limit, device=cloud.device).long()
    return cloud[indices]


def _as_float(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _plane_tensors(calibration, device):
    plane = getattr(calibration, "reference_plane", None)
    if callable(plane):
        plane = plane()
    if plane is None:
        raise ValueError("Fitting requires the adopted physical reference plane")
    if isinstance(plane, Mapping):
        origin = plane.get("origin_viewer_m", plane.get("origin"))
        normal = plane.get("normal_viewer", plane.get("normal"))
    else:
        origin, normal = plane
    origin = _as_float(origin, device).reshape(3)
    normal = _as_float(normal, device).reshape(3)
    if not torch.isfinite(origin).all() or not torch.isfinite(normal).all():
        raise ValueError("Invalid ground plane")
    if normal.norm() < 0.9:
        raise ValueError("Ground normal is not a unit direction")
    normal = normal / normal.norm()
    return origin, normal


def _camera_to_viewer(calibration, points, camera):
    # The teacher target is a constant; the calibration may provide NumPy or
    # torch transforms.  Optimized vertices never pass through this fallback.
    try:
        result = calibration.camera_to_viewer(points, camera)
    except (TypeError, ValueError):
        result = calibration.camera_to_viewer(points.detach().cpu().numpy(), camera)
    return _as_float(result, points.device)


def _project(calibration, points, camera, space):
    if space not in {"undistorted", "distorted"}:
        raise ValueError("keypoints_space must be 'undistorted' or 'distorted'")
    try:
        result = calibration.project_torch(points, camera, distort=space == "distorted")
    except TypeError:
        if space == "undistorted":
            # Do not silently compare a Brown-distorted projection with teacher
            # keypoints detected on the undistorted source frames.
            if not hasattr(calibration, "viewer_to_camera"):
                raise TypeError("Calibration must support undistorted projection")
            cam = calibration.viewer_to_camera(points, camera)
            camera_info = calibration.cameras[camera]
            intrinsics = _as_float(camera_info["K"], points.device)
            homogeneous = cam @ intrinsics.T
            return homogeneous[..., :2] / homogeneous[..., 2:].clamp_min(1e-5), cam[..., 2]
        result = calibration.project_torch(points, camera)
    if isinstance(result, tuple):
        return result[0], result[1]
    if isinstance(result, Mapping):
        return result["uv"], result.get("depth")
    return result, None


def _projection_loss(vertices, coco_regressor, calibration, keypoints, camera, space):
    joints = torch.einsum("jv,tvc->tjc", coco_regressor, vertices)
    cameras = keypoints if isinstance(keypoints, Mapping) else {camera: keypoints}
    losses = []
    measurements = {}
    for name, observations in cameras.items():
        observations = _as_float(observations, vertices.device)
        if observations.shape != (len(vertices), 17, 3):
            raise ValueError("COCO keypoints must have shape (T,17,3), including confidence")
        projection, depth = _project(calibration, joints, name, space)
        confidence = torch.nan_to_num(observations[..., 2]).clamp(0, 1)
        finite = torch.isfinite(observations[..., :2]).all(-1) & torch.isfinite(projection).all(-1)
        if depth is not None:
            finite = finite & (depth > 0)
        confidence = confidence * finite
        error = (projection - torch.nan_to_num(observations[..., :2])).norm(dim=-1)
        error = torch.where(finite, error, torch.zeros_like(error))
        # 50 pixels marks the quadratic/linear boundary; divide by image-scale
        # squared so this term can share metre-based fitting weights.
        huber = torch.where(error < 50, 0.5 * error.square(), 50 * (error - 25)) / 1e6
        losses.append((huber * confidence).sum() / confidence.sum().clamp_min(1))
        frame_error = (error * confidence).sum(-1) / confidence.sum(-1).clamp_min(1)
        measurements[name] = (frame_error, (confidence > 0.25).sum(-1))
    if not losses:
        return vertices.sum() * 0, measurements
    return torch.stack(losses).mean(), measurements


class SMPLFitter:
    """One reusable neutral SMPL model and calibrated video-to-cloud optimizer."""

    def __init__(
        self, device="cuda", config: Optional[FittingConfig] = None,
        model_root: Optional[Path] = None, **config_overrides,
    ):
        import smplx
        if config is not None and config_overrides:
            raise ValueError("Pass either a FittingConfig or named configuration values")
        self.config = config or FittingConfig(**config_overrides)
        self.device = torch.device(device)
        model_root = Path(model_root or PROJ_ROOT / "inputs/checkpoints/body_models")
        self.student = smplx.create(
            model_root, model_type="smpl", gender="neutral", num_betas=10,
            create_body_pose=False, create_betas=False, create_global_orient=False,
            create_transl=False,
        ).eval().to(self.device).requires_grad_(False)
        # Use the exact GVHMR hand-mean/body-model configuration when producing
        # teacher vertices.  Shape coefficients belong to this model only.
        from hmr4d.utils.body_model.body_model_smplx import BodyModelSMPLX
        self.teacher = BodyModelSMPLX(
            model_path=model_root, model_type="smplx", gender="neutral",
            num_pca_comps=12, flat_hand_mean=False,
        ).eval().to(self.device).requires_grad_(False)
        self.mapping = torch.load(
            PROJ_ROOT / "hmr4d/utils/body_model/smplx2smpl_sparse.pt",
            map_location=self.device, weights_only=True,
        )
        self.coco_regressor = torch.load(
            PROJ_ROOT / "hmr4d/utils/body_model/smpl_coco17_J_regressor.pt",
            map_location=self.device, weights_only=True,
        )
        if self.mapping.shape != (6890, 10475):
            raise ValueError("Unexpected SMPL-X to SMPL topology mapping")
        self.faces = torch.as_tensor(self.student.faces, device=self.device, dtype=torch.long)
        self.face_adjacency = vertex_face_adjacency(self.faces, len(self.student.v_template))
        count = min(self.config.topology_vertices, len(self.student.v_template))
        self.topology_indices = torch.linspace(0, len(self.student.v_template) - 1, count,
                                              device=self.device).long()

    def forward_pelvis(self, root, body, shape, pelvis):
        """SMPL geometry using the downstream pelvis translation convention."""
        return forward_smpl_at_pelvis(self.student, root, body, shape, pelvis)

    @torch.no_grad()
    def teacher_vertices(self, teacher_params, calibration, camera="cam2"):
        params = {key: _as_float(teacher_params[key], self.device).reshape(-1, dimension)
                  for key, dimension in (("body_pose", 63), ("betas", 10),
                                         ("global_orient", 3), ("transl", 3))}
        source = self.teacher(**params)
        mapped_camera = map_smplx_vertices(source.vertices, self.mapping)
        return _camera_to_viewer(calibration, mapped_camera, camera)

    def fit_batch(
        self, teacher_params: Mapping[str, Any], clouds: Sequence[Any], calibration,
        camera="cam2", keypoints=None, previous: Optional[Mapping] = None,
        iterations=80, keypoints_space="undistorted",
    ) -> dict:
        """Fit one person's contiguous window and return CPU tensors plus QA.

        ``previous`` may supply the last frame's root_pose/body_pose/transl and
        shape as a boundary prior.  Setting ``shape_fixed=True`` fixes that
        track's previously estimated shape.  No cloud is manufactured or
        duplicated by this function, and no frame is implicitly accepted.
        """
        cfg = self.config
        if iterations < 0 or cfg.conversion_steps < 0:
            raise ValueError("Optimization step counts must be nonnegative")
        params = {key: _as_float(teacher_params[key], self.device).reshape(-1, dimension)
                  for key, dimension in (("body_pose", 63), ("betas", 10),
                                         ("global_orient", 3), ("transl", 3))}
        count = len(params["body_pose"])
        if count < 1 or any(len(value) != count for value in params.values()) or len(clouds) != count:
            raise ValueError("Teacher and cloud frame counts differ or are empty")
        if any(not torch.isfinite(value).all() for value in params.values()):
            raise ValueError("Nonfinite video teacher parameters")
        observed = []
        for cloud in clouds:
            points = _as_float(cloud, self.device).reshape(-1, 3)
            points = points[torch.isfinite(points).all(-1)]
            if len(points) == 0:
                raise ValueError("An empty/nonfinite cloud must be rejected before fitting")
            observed.append(points)
        optimization_clouds = [_sample_cloud(points, cfg.point_limit) for points in observed]
        qa_clouds = [_sample_cloud(points, cfg.qa_point_limit) for points in observed]
        plane_origin, plane_normal = _plane_tensors(calibration, self.device)

        with torch.no_grad():
            target = self.teacher_vertices(params, calibration, camera)
            viewer_rotation = _as_float(calibration.camera_rotation_to_viewer(camera), self.device)
            seed_root = matrix_to_axis_angle(viewer_rotation @ axis_angle_to_matrix(params["global_orient"]))
            seed_body = torch.cat([params["body_pose"], params["body_pose"].new_zeros((count, 6))], -1)
            seed_shape = params["betas"].mean(0, keepdim=True).clamp(-cfg.max_beta, cfg.max_beta)
            seed_pelvis = torch.einsum("v,tvc->tc", self.student.J_regressor[0], target)
            if previous is not None and previous.get("shape") is not None:
                seed_shape = _as_float(previous["shape"], self.device).reshape(-1, 10).mean(0, keepdim=True)
                seed_shape = seed_shape.clamp(-cfg.max_beta, cfg.max_beta)
            shape_fixed = bool(previous is not None and previous.get("shape_fixed", False))
            copied_vertices, _ = self.forward_pelvis(seed_root, seed_body, seed_shape, seed_pelvis)
            copied_topology_error = (copied_vertices - target).norm(dim=-1).mean(-1)

        root = seed_root.detach().clone().requires_grad_()
        body = seed_body.detach().clone().requires_grad_()
        shape = seed_shape.detach().clone().requires_grad_(not shape_fixed)
        pelvis = seed_pelvis.detach().clone().requires_grad_()
        parameters = [root, body, pelvis] + ([] if shape_fixed else [shape])
        optimizer = torch.optim.Adam(parameters, lr=cfg.conversion_lr)
        topology_target = target[:, self.topology_indices]
        for _ in range(cfg.conversion_steps):
            vertices, _ = self.forward_pelvis(root, body, shape, pelvis)
            loss = (vertices[:, self.topology_indices] - topology_target).square().mean()
            loss = loss + shape.square().mean() * 1e-7
            self._step(loss, optimizer, parameters, shape)

        with torch.no_grad():
            converted_vertices, converted_joints = self.forward_pelvis(root, body, shape, pelvis)
            converted_topology_error = (converted_vertices - target).norm(dim=-1).mean(-1)
            teacher_distances = point_to_surface_squared(target, self.faces, qa_clouds)
            converted_distances = point_to_surface_squared(converted_vertices, self.faces, qa_clouds)
            prior_body = body.detach().clone()
            prior_root = root.detach().clone()
            prior_shape = shape.detach().clone()
            relative_target = converted_vertices.detach() - pelvis.detach()[:, None]
            teacher_pelvis = pelvis.detach().clone()
            # Monocular depth is a weak initialization.  A robust cloud-centre
            # shift puts the observed surface within the optimizer's range;
            # fitting and reprojection, not this shift, determine the label.
            if cfg.align_translation:
                shifts = torch.stack([cloud.median(0).values - verts.median(0).values
                                      for cloud, verts in zip(observed, converted_vertices)])
                pelvis.add_(shifts)
            aligned_pelvis = pelvis.detach().clone()
            aligned_vertices, _ = self.forward_pelvis(root, body, shape, pelvis)
            aligned_distances = point_to_surface_squared(aligned_vertices, self.faces, qa_clouds)

        optimizer = torch.optim.Adam(parameters, lr=cfg.learning_rate)
        prior_body_matrix = axis_angle_to_matrix(prior_body.reshape(count, 23, 3))
        prior_root_matrix = axis_angle_to_matrix(prior_root)
        loss_history = []
        for step in range(iterations):
            vertices, _ = self.forward_pelvis(root, body, shape, pelvis)
            if cfg.optimizer_surface == "candidate_faces":
                optimization_distances = candidate_point_to_surface_squared(
                    vertices, self.faces, optimization_clouds, self.face_adjacency,
                    neighbors=cfg.candidate_neighbors,
                )
            else:
                optimization_distances = point_to_surface_squared(vertices, self.faces, optimization_clouds)
            loss = robust_surface_loss(
                optimization_distances, cfg.cloud_huber_m,
            )
            relative_vertices = vertices - pelvis[:, None]
            loss = loss + cfg.teacher_geometry_weight * (
                relative_vertices[:, self.topology_indices] - relative_target[:, self.topology_indices]
            ).square().mean()
            body_matrix = axis_angle_to_matrix(body.reshape(count, 23, 3))
            root_matrix = axis_angle_to_matrix(root)
            loss = loss + cfg.pose_prior_weight * (body_matrix - prior_body_matrix).square().mean()
            loss = loss + cfg.root_prior_weight * (root_matrix - prior_root_matrix).square().mean()
            loss = loss + cfg.shape_prior_weight * (shape - prior_shape).square().mean()
            heights = (vertices - plane_origin) @ plane_normal
            loss = loss + cfg.floor_weight * torch.relu(-heights - cfg.ground_tolerance_m).square().mean()
            if keypoints is not None:
                reprojection, _ = _projection_loss(vertices, self.coco_regressor, calibration,
                                                    keypoints, camera, keypoints_space)
                loss = loss + cfg.reprojection_weight * reprojection
            if count > 1:
                # Preserve observed video articulation changes, rather than
                # flattening the fast racket action into a static pose.
                body_delta = body_matrix[1:] - body_matrix[:-1]
                prior_delta = prior_body_matrix[1:] - prior_body_matrix[:-1]
                loss = loss + cfg.temporal_weight * (body_delta - prior_delta).square().mean()
                pelvis_delta = pelvis[1:] - pelvis[:-1]
                expected_delta = aligned_pelvis[1:] - aligned_pelvis[:-1]
                loss = loss + cfg.temporal_weight * 0.25 * (pelvis_delta - expected_delta).square().mean()
            if previous is not None and previous.get("transl") is not None:
                last_pelvis = _as_float(previous["transl"], self.device).reshape(-1, 3)[-1]
                # Boundary continuity is a weak regularizer; a jumping/running
                # athlete remains free to move between windows.
                loss = loss + cfg.temporal_weight * 0.05 * (pelvis[0] - last_pelvis).square().mean()
            if previous is not None and previous.get("body_pose") is not None:
                last_body = _as_float(previous["body_pose"], self.device).reshape(-1, 23, 3)[-1]
                last_body_matrix = axis_angle_to_matrix(last_body)
                loss = loss + cfg.temporal_weight * 0.05 * (body_matrix[0] - last_body_matrix).square().mean()
            if previous is not None and previous.get("root_pose") is not None:
                last_root = _as_float(previous["root_pose"], self.device).reshape(-1, 3)[-1]
                last_root_matrix = axis_angle_to_matrix(last_root)
                loss = loss + cfg.temporal_weight * 0.05 * (root_matrix[0] - last_root_matrix).square().mean()
            self._step(loss, optimizer, parameters, shape)
            if step == 0 or step == iterations - 1 or (step + 1) % 10 == 0:
                loss_history.append({"step": step + 1, "loss": float(loss.detach().cpu())})

        with torch.no_grad():
            vertices, joints = self.forward_pelvis(root, body, shape, pelvis)
            final_distances = point_to_surface_squared(vertices, self.faces, qa_clouds)
            heights = (vertices - plane_origin) @ plane_normal
            if keypoints is not None:
                _, reprojection_measurements = _projection_loss(vertices, self.coco_regressor, calibration,
                                                                keypoints, camera, keypoints_space)
            else:
                reprojection_measurements = {}
            finite = torch.isfinite(vertices).flatten(1).all(-1) & torch.isfinite(body).all(-1)
            qa = []
            for index in range(count):
                measured = final_distances[index].clamp_min(0).sqrt()
                teacher_measured = teacher_distances[index].clamp_min(0).sqrt()
                converted_measured = converted_distances[index].clamp_min(0).sqrt()
                aligned_measured = aligned_distances[index].clamp_min(0).sqrt()
                mean = float(measured.mean().cpu())
                initial_mean = float(teacher_measured.mean().cpu())
                valid_reprojections = [value[0][index] for value in reprojection_measurements.values()
                                       if int(value[1][index].cpu()) > 0]
                row = {
                    "finite": bool(finite[index].cpu()),
                    "cloud_points": len(observed[index]), "qa_points": len(qa_clouds[index]),
                    "surface_metric": "one_sided_observed_point_to_smpl_triangle_metres",
                    "optimizer_surface": cfg.optimizer_surface,
                    "optimizer_candidate_neighbors": cfg.candidate_neighbors if cfg.optimizer_surface == "candidate_faces" else None,
                    "surface_mean_m": mean, "surface_median_m": float(measured.median().cpu()),
                    "surface_p90_m": float(torch.quantile(measured, 0.9).cpu()),
                    "surface_p95_m": float(torch.quantile(measured, 0.95).cpu()),
                    "surface_inlier_fraction_10cm": float((measured <= 0.1).float().mean().cpu()),
                    "teacher_surface_mean_m": initial_mean,
                    "converted_surface_mean_m": float(converted_measured.mean().cpu()),
                    "aligned_surface_mean_m": float(aligned_measured.mean().cpu()),
                    "surface_improvement_m": initial_mean - mean,
                    "final_p2surface_mean_m": mean,
                    "final_p2surface_p90_m": float(torch.quantile(measured, 0.9).cpu()),
                    "init_p2surface_mean_m": initial_mean,
                    "init_p2surface_p90_m": float(torch.quantile(teacher_measured, 0.9).cpu()),
                    "topology_copy_mean_m": float(copied_topology_error[index].cpu()),
                    "topology_converted_mean_m": float(converted_topology_error[index].cpu()),
                    "max_ground_penetration_m": float(torch.relu(-heights[index]).max().cpu()),
                    "floor_penetration_m": float(torch.relu(-heights[index]).max().cpu()),
                    "ground_penetrating_vertex_fraction": float((heights[index] < -cfg.ground_tolerance_m).float().mean().cpu()),
                    "shape_max_abs": float(shape.abs().max().cpu()),
                    "shape_at_bound": bool((shape.abs() >= cfg.max_beta - 1e-3).any().cpu()),
                    "pelvis_shift_from_teacher_m": float((pelvis[index] - teacher_pelvis[index]).norm().cpu()),
                    "pose_change_from_teacher_matrix_rms": float((axis_angle_to_matrix(body[index].reshape(23, 3)) - prior_body_matrix[index]).square().mean().sqrt().cpu()),
                    "keypoints_space": keypoints_space if keypoints is not None else None,
                    "reprojection_mean_px": (
                        float(torch.stack(valid_reprojections).mean().cpu())
                        if valid_reprojections else None
                    ),
                    "reprojection_by_camera_px": {name: (
                                                      float(value[0][index].cpu())
                                                      if int(value[1][index].cpu()) > 0 else None
                                                   )
                                                   for name, value in reprojection_measurements.items()},
                    "reprojection_valid_joints": {name: int(value[1][index].cpu())
                                                    for name, value in reprojection_measurements.items()},
                }
                qa.append(row)

            cpu = lambda tensor: tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
            return {
                "root_pose": cpu(root[:, None]), "body_pose": cpu(body.reshape(count, 23, 3)),
                "shape": cpu(shape.expand(count, -1)), "transl": cpu(pelvis),
                "vertices": cpu(vertices), "joints": cpu(joints),
                "coco17": cpu(torch.einsum("jv,tvc->tjc", self.coco_regressor, vertices)),
                "qa": qa, "loss_history": loss_history,
                "teacher_init": {
                    "target_vertices": cpu(target), "vertices": cpu(converted_vertices),
                    "joints": cpu(converted_joints), "transl": cpu(teacher_pelvis),
                    "root_pose": cpu(prior_root[:, None]), "body_pose": cpu(prior_body.reshape(count, 23, 3)),
                    "shape": cpu(prior_shape.expand(count, -1)),
                },
                "coordinate_space": "viewer_m", "translation_convention": "pelvis_position",
                "shape_shared_within_window": True, "shape_fixed": shape_fixed,
                "optimizer_surface": cfg.optimizer_surface,
            }

    def _step(self, loss, optimizer, parameters, shape):
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite SMPL fitting loss; reject this window")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in parameters):
            raise RuntimeError("Nonfinite SMPL fitting gradient; reject this window")
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        with torch.no_grad():
            shape.clamp_(-self.config.max_beta, self.config.max_beta)
