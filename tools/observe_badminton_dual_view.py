"""Observe all accepted source labels in both native cameras; resumable 2D QC."""
import argparse
import json
from pathlib import Path

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.dual_view_observer import observe_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=PROJ_ROOT/'outputs/badminton_corner_fix_v1_continuity')
    parser.add_argument('--output-root', type=Path, default=PROJ_ROOT/'outputs/badminton_dualcam_paired_full_v1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--pose-batch-size', type=int, default=4)
    parser.add_argument('--detector-imgsz', type=int, default=960)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--cache-report',dest='cache_reports',type=Path,action='append',default=[],help='Reuse verified native observations, including the complete prefix of an interrupted journal')
    parser.add_argument('--max-new-frames', type=int, help='Bounded runtime probe; rerun without this option to resume all frames')
    args = parser.parse_args()
    if (args.output_root/'datasets/dataset_manifest.json').exists():
        parser.error('Published datasets are immutable; use a separate output folder')
    if args.max_new_frames is not None and args.max_new_frames < 1:
        parser.error('--max-new-frames must be positive')
    result = observe_all(**vars(args))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
