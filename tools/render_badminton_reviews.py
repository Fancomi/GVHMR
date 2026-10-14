"""Render continuous SMPL/LiDAR and paired-camera clips without using CUDA."""
import argparse
import json
from pathlib import Path
import torch
from hmr4d.lidar_mocap.sequence_review import render_sequences


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=Path('outputs/badminton_corner_fix_v1'))
    parser.add_argument('--review-dir',type=Path)
    parser.add_argument('--clips',type=int,default=10)
    parser.add_argument('--seconds',type=float,default=5)
    parser.add_argument('--image-width',type=int,default=1280)
    parser.add_argument('--html-only',action='store_true')
    args=parser.parse_args()
    if min(args.clips,args.seconds,args.image_width)<=0:
        parser.error('Clips, duration and image width must be positive')
    torch.set_num_threads(4)
    review=args.review_dir or args.output_root/'sequence_review'
    result=render_sequences(args.output_root,review,count=args.clips,seconds=args.seconds,
                            image_width=args.image_width,html_only=args.html_only)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
