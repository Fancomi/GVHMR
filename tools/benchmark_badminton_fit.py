"""Compare fitting losses on the same real athlete returns and video prior."""
import json
import argparse
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.fitting import SMPLFitter
from hmr4d.lidar_mocap.teacher import load_teacher_track,sample_track


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scans',type=int,default=8)
    parser.add_argument('--window',default='000000000000_000006000000')
    parser.add_argument('--motion-index',type=int,default=0)
    parser.add_argument('--surface',choices=('both','full_faces','candidate_faces'),default='both')
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path('outputs/badminton')
    window = output/'sessions/20261012_154726/windows'/args.window
    all_rows = [json.loads(line) for line in (window/'quality.jsonl').read_text(encoding='utf-8').splitlines()]
    names = list(dict.fromkeys(r['motion_id'] for r in all_rows))
    rows = [r for r in all_rows if r['motion_id']==names[args.motion_index]][:args.scans]
    calibration = CalibrationSnapshot.load(output/'calibration/calibration_snapshot.json')
    track = load_teacher_track(window/'teacher'/rows[0]['teacher'])
    sampled = sample_track(track,[r['scan_mid_us'] for r in rows])
    clouds = []
    for row in rows:
        with np.load(row['fit_path']) as fit:
            clouds.append(fit['source_cloud'].copy())
    fitter = SMPLFitter(device='cuda')
    report = {}
    results = {}
    for surface in (('full_faces','candidate_faces') if args.surface=='both' else (args.surface,)):
        fitter.config = replace(fitter.config,optimizer_surface=surface)
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fitter.fit_batch(sampled['smpl_params_incam'],clouds,calibration,
                                  keypoints=sampled['kp2d'],iterations=80)
        torch.cuda.synchronize()
        duration = time.perf_counter()-start
        report[surface] = {'seconds':duration,'mean_surface_mm':np.mean([q['final_p2surface_mean_m'] for q in result['qa']])*1000,
                           'mean_p90_mm':np.mean([q['final_p2surface_p90_m'] for q in result['qa']])*1000,
                           'reprojection_px':np.mean([q['reprojection_mean_px'] for q in result['qa']]),
                           'finite':all(q['finite'] for q in result['qa'])}
        results[surface] = result
        print(surface,report[surface],flush=True)
    report['scope'] = f'Same {len(rows)} real person/scan pairs, 60 conversion + 80 fitting steps; final QA always all triangles'
    if args.surface=='both':
        report['speed_ratio'] = report['full_faces']['seconds']/report['candidate_faces']['seconds']
        report['mean_surface_delta_mm'] = report['candidate_faces']['mean_surface_mm']-report['full_faces']['mean_surface_mm']
        if abs(report['mean_surface_delta_mm'])>5 or not report['candidate_faces']['finite']:
            raise RuntimeError('Candidate optimization needs further quality review')
    (output/f'fit_benchmark_{len(rows)}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
