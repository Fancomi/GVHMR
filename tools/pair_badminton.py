"""Stage adapter for the badminton pipeline; full jobs use run_badminton_pipeline.py.

Input: declared canonical LiDAR calibration and original session recordings.
Output: immutable snapshot, tracks/fits and PTv3 single-frame pairs in output-root.
Use local uv with body-model assets (CUDA required for teacher/fitting stages).
Changed calibration/source identity needs a new output root; see job status/errors.
"""
import argparse
import json
import torch
from hmr4d.lidar_mocap.pipeline import BadmintonPipeline


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','teacher','fit','run','export'))
    parser.add_argument('--sessions-root',default='D:/PROJECT/LIDAR/datas/badminton/sessions')
    parser.add_argument('--lidar-project-root',default='D:/PROJECT/LIDAR/lidar_project')
    parser.add_argument('--output-root',default='outputs/badminton')
    parser.add_argument('--session',action='append',dest='session_ids')
    parser.add_argument('--camera',choices=('cam1','cam2'),default='cam2')
    parser.add_argument('--start-seconds',type=float,default=0)
    parser.add_argument('--duration-seconds',type=float)
    parser.add_argument('--window-seconds',type=float,default=60)
    parser.add_argument('--teacher-stride',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--fit-batch-size',type=int,default=128)
    parser.add_argument('--iterations',type=int,default=80)
    parser.add_argument('--preproc-precision',choices=('float32','float16'),default='float32')
    parser.add_argument('--roi-margin-px',type=float,default=40)
    parser.add_argument('--no-resume',action='store_true')
    args = parser.parse_args()
    if args.teacher_stride != 2:
        parser.error('These 59.94fps recordings require stride=2 for the GVHMR ~30fps prior')
    if min(args.batch_size,args.fit_batch_size,args.iterations,args.window_seconds)<=0 or args.start_seconds<0:
        parser.error('Batch sizes, iterations and windows must be positive; start must be nonnegative')
    if args.duration_seconds is not None and args.duration_seconds<=0:
        parser.error('duration-seconds must be positive')
    pipeline = BadmintonPipeline(args.sessions_root,args.lidar_project_root,args.output_root,camera=args.camera,
                                 teacher_stride=args.teacher_stride,batch_size=args.batch_size,
                                 fit_batch_size=args.fit_batch_size,iterations=args.iterations,
                                 window_seconds=args.window_seconds,preproc_precision=args.preproc_precision,
                                 roi_margin_px=args.roi_margin_px)
    if args.stage=='prepare':
        report = pipeline.prepare()
    elif args.stage=='export':
        report = pipeline.aggregate(args.session_ids)
    else:
        report = pipeline.run(args.session_ids,start_seconds=args.start_seconds,duration_seconds=args.duration_seconds,
                              stage=args.stage,resume=not args.no_resume)
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
