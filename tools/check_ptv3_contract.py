"""Numerically check SMPL-X -> SMPL fitting and the sibling data contract.

Uses three generated SMPL-X poses, not video predictions. This checks the
interface and geometry conventions; it does not measure pseudo-label quality.
Run with: uv run --no-sync python tools/check_ptv3_contract.py
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import torch
import smplx
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ptv3-root', type=Path, default=Path('../rollout_lidar_mocap_badminton'))
    parser.add_argument('--steps', type=int, default=180)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    downstream = args.ptv3_root.resolve()
    sys.path.insert(0, str(downstream))
    sys.path.insert(0, str(downstream / 'third_party/vis'))
    from vis_tools import PySMPL
    from src.data.single_frame import MocapFrameDataset
    from src.models.frame_model_backbone import SharedFrameSMPLHeads
    from src.lib.utils import pose_to_pose6d, pose6d_to_pose
    from hmr4d.utils.smplx_utils import make_smplx

    torch.manual_seed(2026)
    device = 'cuda'
    n = 3
    teacher = make_smplx('supermotion').to(device)
    student = smplx.create(root / 'inputs/checkpoints/body_models', model_type='smpl',
                           gender='neutral', batch_size=n).to(device)
    mapping = torch.load(root / 'hmr4d/utils/body_model/smplx2smpl_sparse.pt').to(device)
    regressor = torch.load(root / 'hmr4d/utils/body_model/smpl_neutral_J_regressor.pt').to(device)
    betas = torch.tensor([[0.5, -0.25, 0.2, 0., 0., 0., 0., 0., 0., 0.]], device=device)
    body = torch.randn(n, 63, device=device) * 0.12
    body[:, 45:48] += torch.tensor([0., 0., 0.5], device=device)
    orient = torch.tensor([[0., 0., 0.], [0., 0.3, 0.], [0.1, -0.3, 0.]], device=device)
    offset = torch.tensor([[0., 0., 4.], [0.1, 0., 4.], [0.2, 0., 4.]], device=device)
    with torch.no_grad():
        source = teacher(betas=betas.expand(n, -1), body_pose=body, global_orient=orient, transl=offset)
        target = torch.stack([mapping @ verts for verts in source.vertices]).detach()
    fitted_shape = betas.clone().requires_grad_()
    fitted_body = torch.cat([body, torch.zeros(n, 6, device=device)], dim=-1).requires_grad_()
    fitted_root = orient.clone().requires_grad_()
    fitted_offset = offset.clone().requires_grad_()

    def forward():
        return student(betas=fitted_shape.expand(n, -1), body_pose=fitted_body,
                       global_orient=fitted_root, transl=fitted_offset)

    initial = forward()
    initial_error_mm = (initial.vertices.detach() - target).norm(dim=-1).mean().item() * 1000
    optimizer = torch.optim.Adam([fitted_shape, fitted_body, fitted_root, fitted_offset], lr=0.025)
    for _ in range(args.steps):
        prediction = forward()
        loss = (prediction.vertices - target).square().mean() + fitted_shape.square().mean() * 1e-7
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        fitted = forward()
        fitted_error_mm = (fitted.vertices - target).norm(dim=-1).mean().item() * 1000
        pelvis = fitted.joints[:, 0]
        pose = torch.cat([fitted_root[:, None], fitted_body.reshape(n, 23, 3)], dim=1)
        shape = fitted_shape.expand(n, -1)
        pysmpl = PySMPL().to(device)
        same_frame = pysmpl(shape.contiguous(), pose, pelvis).vertices
        forward_error_m = (same_frame - fitted.vertices).abs().max().item()
        rotation = torch.tensor([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]], device=device)
        pose_target = pose.clone()
        pose_target[:, 0] = matrix_to_axis_angle(rotation @ axis_angle_to_matrix(pose[:, 0]))
        pelvis_target = pelvis @ rotation.T
        target_output = pysmpl(shape.contiguous(), pose_target, pelvis_target)
        transform_error_m = (target_output.vertices - same_frame @ rotation.T).abs().max().item()
        # Sparse mapping and the regressor provide SMPL-topology joint targets.
        assert mapping.shape == (6890, 10475)
        assert regressor.shape == (24, 6890)
        assert torch.isfinite(target_output.vertices).all()
        assert forward_error_m < 1e-4, forward_error_m
        assert transform_error_m < 1e-4, transform_error_m
        assert fitted_error_mm < initial_error_mm, (initial_error_mm, fitted_error_mm)

        # These vertex subsets are for a loader smoke check, not simulated LiDAR.
        pc = target_output.vertices[:, torch.linspace(0, 6889, 256, device=device).long()]
        output_dir = root / 'outputs/setup'
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path = output_dir / 'ptv3_contract_smoke.pth'
        joblib.dump({'root_pose': pose_target[:, :1].cpu(), 'body_pose': pose_target[:, 1:].cpu(),
                     'transl': pelvis_target.cpu(), 'shape': shape.cpu(), 'pc': pc.cpu(),
                     'motion_id': ['contract_smoke'] * n}, data_path)
        dataset = MocapFrameDataset(str(data_path))
        sample = dataset[0]
        assert sample['pose'].shape == (24, 3)
        assert sample['pc'].shape == (256, 3)
        round_trip = pose6d_to_pose(pose_to_pose6d(pose_target))
        rotation_error = (axis_angle_to_matrix(round_trip) - axis_angle_to_matrix(pose_target)).abs().max().item()
        assert rotation_error < 1e-5, rotation_error
        heads = SharedFrameSMPLHeads().to(device).eval()
        mean = pc.mean(dim=1, keepdim=True)
        head_output = heads(torch.randn(n, 256, 320, device=device), pc - mean, mean)
        assert head_output['pred_pose'].shape == (n, 24, 6)
        assert head_output['pred_shape'].shape == (n, 10)
        assert head_output['pred_kp3d'].shape == (n, 24, 3)
        assert all(torch.isfinite(v).all() for v in head_output.values())

    report = {
        'scope': 'Three generated SMPL-X poses; interface smoke check, not video label validation',
        'smplx_to_smpl_matrix': list(mapping.shape),
        'smpl_joint_regressor': list(regressor.shape),
        'direct_parameter_copy_mean_vertex_error_mm': initial_error_mm,
        'fitted_smpl_mean_vertex_error_mm': fitted_error_mm,
        'pysmpl_forward_max_error_m': forward_error_m,
        'target_coordinate_transform_max_error_m': transform_error_m,
        'downstream_loader': 'passed',
        'downstream_pose6d_round_trip_max_matrix_error': rotation_error,
        'shared_ptv3_head': {k: list(v.shape) for k, v in head_output.items()},
    }
    (output_dir / 'ptv3_contract_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
