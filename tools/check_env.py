"""Check the local runtime without requiring inference checkpoints."""

import argparse
import importlib
import json
import sys
import subprocess
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-assets', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda}
    for name in ('torchvision', 'pytorch3d', 'lightning', 'hydra', 'cv2', 'av', 'smplx', 'chumpy', 'pycolmap'):
        importlib.import_module(name)
    from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
    from hmr4d.utils.vis.renderer import Renderer
    from hmr4d.configs import register_store_gvhmr
    register_store_gvhmr()
    from hydra import compose, initialize_config_module
    with initialize_config_module(version_base='1.3', config_module='hmr4d.configs'):
        compose(config_name='demo', overrides=['video_name=check_env'])

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    report['gpu'] = torch.cuda.get_device_name()
    from pytorch3d.ops import knn_points
    from pytorch3d.transforms import axis_angle_to_matrix
    from pytorch3d.renderer import rasterize_meshes
    from pytorch3d.structures import Meshes
    points = torch.randn(1, 16, 3, device='cuda', requires_grad=True)
    distances = knn_points(points, points.detach() + 0.1, K=1).dists
    distances.sum().backward()
    assert torch.isfinite(points.grad).all()
    assert points.grad.abs().sum() > 0
    assert torch.allclose(axis_angle_to_matrix(torch.zeros(1, 3, device='cuda')), torch.eye(3, device='cuda')[None])
    verts = torch.tensor([[[-0.5, -0.5, 1.], [0., 0.5, 1.], [0.5, -0.5, 1.]]], device='cuda')
    faces = torch.tensor([[[0, 1, 2]]], device='cuda')
    pix_to_face, _, _, _ = rasterize_meshes(Meshes(verts=verts, faces=faces), image_size=16)
    assert (pix_to_face >= 0).any(), 'CUDA rasterization returned no triangle'
    report['cuda_knn_backward'] = 'passed'
    report['cuda_rasterization'] = 'passed'
    import imageio_ffmpeg
    report['ffmpeg'] = subprocess.check_output([imageio_ffmpeg.get_ffmpeg_exe(), '-version'], text=True).splitlines()[0]
    paths = [
        'inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz',
        'inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl',
        'inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt',
        'inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt',
        'inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth',
        'inputs/checkpoints/yolo/yolov8x.pt',
    ]
    report['missing_assets'] = [p for p in paths if not (root / p).is_file()]
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.require_assets and report['missing_assets']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
