"""Preview fitted SMPL and original LiDAR/video, or verify existing previews.

Input: a completed run with its frozen calibration, fits and declared original
recordings. Output: offline HTML, exact native geometry and camera overlays in
--review-dir. Requires installed neutral SMPL assets; inference/CUDA is not used.
For immutable historical previews pass --verify-only --verification-report
with a new report destination. Missing sources/calibration mismatches fail.
"""
import argparse
import json
from pathlib import Path
import torch
from hmr4d.lidar_mocap.sequence_review import render_sequences,verify_native_review


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=Path('outputs/badminton_corner_fix_v1'))
    parser.add_argument('--review-dir',type=Path)
    parser.add_argument('--clips',type=int,default=10)
    parser.add_argument('--seconds',type=float,default=5)
    parser.add_argument('--image-width',type=int,default=1280)
    parser.add_argument('--html-only',action='store_true')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--verification-report',type=Path)
    parser.add_argument('--required-case',action='append',default=[],metavar='SESSION:SCAN',
                        help='Include a fixed diagnostic interval, replacing a slot from the same session')
    args=parser.parse_args()
    if min(args.clips,args.seconds,args.image_width)<=0:
        parser.error('Clips, duration and image width must be positive')
    torch.set_num_threads(4)
    review=args.review_dir or args.output_root/'sequence_review'
    if args.verify_only:
        print(json.dumps(verify_native_review(args.output_root,review,args.verification_report),indent=2))
        return
    result=render_sequences(args.output_root,review,count=args.clips,seconds=args.seconds,
                            image_width=args.image_width,html_only=args.html_only,
                            required_cases=[(value.rsplit(':',1)[0],int(value.rsplit(':',1)[1])) for value in args.required_case])
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
