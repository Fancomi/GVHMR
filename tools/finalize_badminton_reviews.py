"""Finish all-person reviews after the calibrated full dataset job succeeds.

Runs separately from the GPU supervisor so an in-flight fitting process never
needs to reload modified code. Status remains waiting/running until every
render, native-source check and camera-mask review actually succeeds.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    temporary=path.with_suffix(path.suffix+'.writing')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def verify_native_review(output, review):
    import numpy as np
    import smplx
    import torch
    from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
    from hmr4d.lidar_mocap.sessions import SessionIndex
    from hmr4d.utils.body_model.contracts import forward_smpl_at_pelvis

    torch.set_num_threads(4)
    calibration=CalibrationSnapshot.load(output/'calibration/calibration_snapshot.json')
    manifest=read_json(review/'manifest.json')
    model=smplx.create(Path(__file__).resolve().parents[1]/'inputs/checkpoints/body_models',model_type='smpl',gender='neutral')
    sessions={};checked=0;mesh_error=0.;source_error=0.
    for clip in manifest['clips']:
        sid=clip['session_id']
        if sid not in sessions:
            session=SessionIndex.load(Path(calibration.data['sessions_root'])/sid,calibration.frame_map_path(sid))
            sessions[sid]=(session,session.pairs())
        session,pairs=sessions[sid]
        raw=(review/clip['id']/'data.js').read_text('utf-8')
        payload=json.loads(raw[len('window.REVIEW_CLIP_READY('):-3])
        for i in sorted({0,len(clip['frames'])//2,len(clip['frames'])-1}):
            frame=clip['frames'][i];stored=payload['frames'][i]
            if stored['lidar_index']!=frame['lidar_index'] or stored['scan_mid_us']!=frame['scan_mid_us']:
                raise ValueError('Rendered source clock differs from the selection')
            cloud=session.read_cloud(pairs[frame['lidar_index']])
            unpack=lambda value:np.frombuffer(base64.b64decode(value),dtype='<f4').reshape(-1,3)
            np.testing.assert_array_equal(unpack(stored['cloud']),cloud)
            by_id={a['motion_id']:a for a in stored['athletes']}
            for athlete in frame['athletes']:
                path=Path(athlete['fit_path']).resolve()
                if not path.is_relative_to(output/'sessions'):
                    raise ValueError('Final review contains a fit from another output root')
                with np.load(path) as fit:
                    source=cloud[fit['source_indices']]
                    np.testing.assert_array_equal(source,fit['source_cloud'])
                    source_error=max(source_error,float(np.abs(source-fit['source_cloud']).max()))
                    geometry=by_id[athlete['motion_id']]
                    for key in ('vertices','joints'):
                        np.testing.assert_array_equal(unpack(geometry[key]),fit[key])
                    np.testing.assert_array_equal(unpack(geometry['cloud']),source)
                    tensor=lambda key:torch.from_numpy(fit[key].copy())[None]
                    with torch.no_grad():
                        vertices,joints=forward_smpl_at_pelvis(model,tensor('root_pose'),tensor('body_pose'),tensor('shape'),tensor('transl'))
                    error=max(float((vertices[0]-torch.from_numpy(fit['vertices'])).abs().max()),
                              float((joints[0]-torch.from_numpy(fit['joints'])).abs().max()))
                    if error>3e-6:raise ValueError('Stored review mesh is not its native neutral SMPL')
                    mesh_error=max(mesh_error,error);checked+=1
    result={'passed':True,'clips':len(manifest['clips']),'native_person_frames_checked':checked,
            'source_cloud_max_error_m':source_error,'native_smpl_max_error_m':mesh_error,
            'all_fit_paths_inside_current_output':True,'geometry_interpolated':False}
    write_json(review/'geometry_verification.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--court-evidence',type=Path,default=Path('outputs/badminton_corner_fix_v1/calibration_refinement/court_evaluation.json'))
    parser.add_argument('--baseline-root',type=Path,default=Path('outputs/badminton_corner_fix_v1'))
    parser.add_argument('--wait-for-job',action='store_true')
    parser.add_argument('--max-wait-hours',type=float,default=12.)
    parser.add_argument('--verify-only',action='store_true',help='Verify an already rendered review without waiting or inference')
    parser.add_argument('--review-dir',type=Path,help='Override the review folder inside the current output root')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];os.chdir(root)
    output=args.output_root.resolve();review=(args.review_dir or output/'sequence_review').resolve()
    if not output.is_relative_to(root/'outputs'):raise ValueError('Review output must stay under project outputs')
    if not review.is_relative_to(output):raise ValueError('Review must stay inside the selected output root')
    review.mkdir(parents=True,exist_ok=True)
    if args.verify_only:
        print(json.dumps(verify_native_review(output,review)));return
    status_path=output/'review_job_status.json'
    previous=read_json(status_path) if status_path.exists() else {}
    if previous.get('state') in {'running','waiting_for_full_job'} and previous.get('pid')!=os.getpid():
        pid=previous['pid']
        if os.name=='nt':
            found=subprocess.run(['powershell','-NoProfile','-Command',f"Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}' | Select-Object -ExpandProperty CommandLine"],capture_output=True,text=True,check=True)
            if 'tools/finalize_badminton_reviews.py' in found.stdout:
                raise RuntimeError('The review finalizer is already running')
    status={'schema':'badminton-all-person-review-job-v1','state':'waiting_for_full_job','pid':os.getpid(),
            'started_utc':datetime.now(timezone.utc).isoformat(),'output_root':str(output)}
    write_json(status_path,status)
    python=str(root/'.venv/Scripts/python.exe');node='C:/Program Files/nodejs/node.exe'
    def run(phase,command):
        status.update(state='running',phase=phase);write_json(status_path,status)
        with (review/(phase+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        deadline=time.monotonic()+args.max_wait_hours*3600
        while True:
            job=read_json(output/'job_status.json')
            if job['state']=='complete':break
            if job['state']!='running':raise RuntimeError(f"Full job cannot be finalized: {job['state']} {job.get('error','')}")
            if not args.wait_for_job:raise RuntimeError('Full job is still running; use --wait-for-job')
            if time.monotonic()>deadline:raise TimeoutError('Waiting for full fitting exceeded the configured timeout')
            status.update(full_job_state=job['state'],phase=job.get('current_phase'),last_checked_utc=datetime.now(timezone.utc).isoformat())
            if (output/'progress.json').exists():status['fitting_progress']=read_json(output/'progress.json')
            write_json(status_path,status);time.sleep(20)
        if job.get('completed_windows')!=40:raise RuntimeError('Expected all 40 calibrated source windows')
        from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
        frozen=CalibrationSnapshot.load(output/'calibration/calibration_snapshot.json')
        def recheck_adoption():
            CalibrationSnapshot.freeze(Path('D:/PROJECT/LIDAR/lidar_project'),Path(frozen.data['sessions_root']),output/'calibration')
        recheck_adoption()
        dataset=read_json(output/'datasets/dataset_manifest.json');verification=read_json(output/'datasets/verification.json')
        if verification['calibration_digest']!=job['calibration_digest']:raise RuntimeError('Dataset verification belongs to another calibration')
        status.update(completed_windows=job['completed_windows'],accepted=dataset['accepted']);write_json(status_path,status)
        render=[python,'tools/render_badminton_reviews.py','--output-root',str(output),'--required-case','20261012_163452:9']
        run('rendering',render)
        status['geometry_verification']=verify_native_review(output,review);write_json(status_path,status)
        for camera in ('cam1','cam2'):
            run(camera+'_keypoints',[python,'tools/check_badminton_cam1.py','--output-root',str(output),'--camera',camera,
                '--report-path',str(review/(camera+'_check.json')),'--max-total-seconds','1800'])
            run(camera+'_case_keypoints',[python,'tools/check_badminton_cam1.py','--output-root',str(output),
                '--camera',camera,'--clip','clip10','--all-frames','--report-path',str(review/(camera+'_case_check.json')),
                '--max-total-seconds','2400'])
        run('case_diagnosis',[python,'tools/diagnose_badminton_all_tracks.py','--output-root',str(args.baseline_root.resolve()),
            '--production-root',str(output),'--production-window','000000000000_000060000000',
            '--output-dir',str(review/'case_diagnosis')])
        run('quality_report',[python,'tools/report_badminton_quality.py','--output-root',str(output),'--output-dir',str(review)])
        run('mask_alignment',[python,'tools/evaluate_badminton_review_masks.py','--output-root',str(output),
            '--manifest',str(review/'manifest.json'),'--court-evidence',str(args.court_evidence.resolve()),
            '--evaluation-dir',str(review/'mask_alignment'),'--device','cuda:0'])
        manifest=read_json(review/'manifest.json')
        manifest['finalized_reviews']={'mask_alignment':'mask_alignment/index.html','case_diagnosis':'case_diagnosis/diagnosis.md',
                                      'cam1_case_check':'cam1_case_check.md','cam2_case_check':'cam2_case_check.md'}
        write_json(review/'manifest.json',manifest)
        run('html_refresh',render+['--html-only'])
        run('browser_review',[node,'tools/check_badminton_reviews.mjs','--review-dir',str(review),'--cleanup-profile'])
        run('browser_masks',[node,'tools/check_badminton_reviews.mjs','--review-dir',str(review/'mask_alignment'),'--gallery','--cleanup-profile'])
        recheck_adoption()
        status.update(state='complete',phase='complete',finished_utc=datetime.now(timezone.utc).isoformat(),
            review=str(review/'index.html'),mask_review=str(review/'mask_alignment/index.html'))
        write_json(status_path,status)
        print(json.dumps(status,ensure_ascii=False))
    except Exception as error:
        status.update(state='failed',error=str(error),finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(status_path,status);raise


if __name__=='__main__':main()
