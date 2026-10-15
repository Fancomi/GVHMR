"""Export existing SMPL/LiDAR labels with corresponding detections in both views.

The default training rule excludes a person-frame when either native camera
lacks its uniquely associated detection. Pose/limb/visibility residuals remain
diagnostic. Optional full_body policy additionally requires reviewed native
crops. Outputs are separate and immutable; source fits/cameras are unchanged.
"""
import argparse
import json
from pathlib import Path
import time

from hmr4d.lidar_mocap.dual_view import DualViewGate, audit_cached, render_candidate_crops, export_reviewed, verify_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('audit', 'export','verify','finalize'))
    parser.add_argument('--source-root', type=Path, default=Path('outputs/badminton_corner_fix_v1_continuity'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs/badminton_dualcam_paired_pilot_v1'))
    parser.add_argument('--evidence', type=Path, action='append')
    parser.add_argument('--review', type=Path)
    parser.add_argument('--policy', choices=('bilateral_detection','full_body'), default='bilateral_detection')
    parser.add_argument('--full-observations', type=Path, help='Completed full native observer manifest; its six journals are audited on disk')
    parser.add_argument('--wait',action='store_true',help='For finalize only: wait for the running native observer, then audit/export/verify automatically')
    args = parser.parse_args()
    if args.stage=='verify':
        print(json.dumps(verify_export(args.output_root),indent=2));return
    if args.stage=='finalize':
        if args.full_observations is None or args.policy!='bilateral_detection':
            parser.error('finalize requires --full-observations and bilateral_detection policy')
        if (args.output_root/'datasets/dataset_manifest.json').exists():
            print(json.dumps(verify_export(args.output_root),indent=2));return
        while not args.full_observations.exists() or not json.loads(args.full_observations.read_text(encoding='utf-8'))['completed']:
            if not args.wait:raise ValueError('Full native observer has not completed')
            print(json.dumps(dict(state='waiting_for_native_observer',path=str(args.full_observations))),flush=True)
            time.sleep(30)
    if args.stage in ('audit','finalize'):
        reports = args.evidence or [args.source_root/'sequence_review'/name for name in
                                    ('cam1_check.json','cam2_check.json','cam1_case_check.json','cam2_case_check.json')]
        if args.full_observations:
            observations=json.loads(args.full_observations.read_text(encoding='utf-8'))
            if not observations['completed']:raise ValueError('Full observer has not completed')
            reports=observations['reports']
        result=audit_cached(args.source_root,args.output_root,reports,DualViewGate(policy=args.policy),disk_evidence=bool(args.full_observations))
        if not args.full_observations:
            result = render_candidate_crops(args.source_root,args.output_root)
    if args.stage in ('export','finalize'):
        review=args.review or (args.output_root/'native_visual_review.jsonl' if args.policy=='full_body' else None)
        result = export_reviewed(args.source_root,args.output_root,review,detection_only=args.policy=='bilateral_detection',full_observation_manifest=args.full_observations)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
