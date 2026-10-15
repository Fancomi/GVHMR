"""Badminton production pipeline: original LiDAR/video -> neutral SMPL pairs.

Inputs: --sessions-root recordings and --lidar-project-root's canonical
config/calibration/badminton-20261012 five-file result. Outputs: a new run folder
with an immutable calibration bundle, fits, datasets and verification status.
Run through project-local uv; fitting needs installed CUDA/body-model assets.
--prepare-only checks calibration, source clocks and pairing without inference.
On failure read job_status.json, resolve its error and rerun this command;
validated caches resume. Changed physical inputs require a new output root.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.pipeline import BadmintonPipeline,write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sessions-root',type=Path,default=Path('D:/PROJECT/LIDAR/datas/badminton/sessions'))
    parser.add_argument('--lidar-project-root',type=Path,default=Path('D:/PROJECT/LIDAR/lidar_project'))
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--output-root',type=Path,default=Path('outputs/badminton'))
    args = parser.parse_args()
    root = PROJ_ROOT
    os.chdir(root)
    output = args.output_root.resolve()
    if not output.is_relative_to(root):
        raise ValueError('Output must remain inside the GVHMR workspace')
    output.mkdir(parents=True,exist_ok=True)
    status_path = output/'job_status.json'
    status = {'schema':'badminton-full-job-v1','state':'running','supervisor_pid':os.getpid(),
              'started_utc':datetime.now(timezone.utc).isoformat(),'output_root':str(output),
              'current_phase':'preparing'}
    write_json(status_path,status)
    python = sys.executable
    inputs = ['--sessions-root',str(args.sessions_root.resolve()),'--lidar-project-root',str(args.lidar_project_root.resolve())]
    try:
        pipeline = BadmintonPipeline(args.sessions_root,args.lidar_project_root,output,
                                     preproc_precision='float16',roi_margin_px=20)
        pipeline.prepare()  # Reject changed inputs; never rewrite a frozen snapshot.
        status.update(calibration_digest=pipeline.calibration.data['snapshot_digest'],
                      configuration_sha256=pipeline.calibration.data.get('configuration_sha256'),
                      geometry_variant=pipeline.calibration.data.get('geometry_variant'))
        write_json(status_path,status)
        if args.prepare_only:
            status.update(state='prepared',current_phase='prepared',finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(status_path,status)
            return
        code = None
        for attempt in range(3):
            if code==0:
                break
            status.update(current_phase='fitting',attempt=attempt+1)
            write_json(status_path,status)
            suffix = '' if attempt==0 else f'_resume_{attempt+1}'
            with (output/f'full_run{suffix}.log').open('w',encoding='utf-8') as log, (output/f'full_run{suffix}.err.log').open('w',encoding='utf-8') as errors:
                result = subprocess.run([python,'-u','tools/pair_badminton.py','run','--preproc-precision','float16',
                                         '--roi-margin-px','20','--output-root',str(output),*inputs],stdout=log,stderr=errors)
                code = result.returncode
        if code:
            raise RuntimeError(f'Pairing process exited with {code}; matching caches remain resumable')
        pipeline = BadmintonPipeline(args.sessions_root,args.lidar_project_root,output,
                                     preproc_precision='float16',roi_margin_px=20)
        calibration = pipeline._load()
        expected = []
        for sid in sorted(calibration.data['sessions']):
            session = pipeline._session(sid)
            expected += [(sid,w[0]) for w in pipeline._windows(session)]
        run_report = json.loads((output/'last_run_report.json').read_text(encoding='utf-8'))
        actual = {(r['session_id'],r['window']) for r in run_report.get('windows',[])}
        if run_report.get('stage')!='run' or actual!=set(expected):
            raise RuntimeError('The full run has not completed every required window')
        CalibrationSnapshot.freeze(pipeline.lidar_project_root,pipeline.sessions_root,output/'calibration')
        status.update(current_phase='exporting',completed_windows=len(actual))
        write_json(status_path,status)
        with (output/'dataset_export.log').open('w',encoding='utf-8') as log:
            subprocess.run([python,'tools/pair_badminton.py','export','--output-root',str(output),*inputs],stdout=log,stderr=subprocess.STDOUT,check=True)
        exported = json.loads((output/'datasets/dataset_manifest.json').read_text(encoding='utf-8'))
        status.update(current_phase='verifying',accepted=exported['accepted'])
        write_json(status_path,status)
        with (output/'dataset_verification.log').open('w',encoding='utf-8') as log:
            subprocess.run([python,'tools/verify_badminton_dataset.py','--output-root',str(output)],stdout=log,stderr=subprocess.STDOUT,check=True)
        CalibrationSnapshot.freeze(pipeline.lidar_project_root,pipeline.sessions_root,output/'calibration')
        status.update(state='complete',current_phase='complete',finished_utc=datetime.now(timezone.utc).isoformat(),
                      dataset_manifest=str(output/'datasets/dataset_manifest.json'),
                      verification=str(output/'datasets/verification.json'))
        write_json(status_path,status)
    except Exception as error:
        status.update(state='failed',error=str(error),finished_utc=datetime.now(timezone.utc).isoformat(),
                      recovery='Resolve the reported cause and rerun this command; matching completed caches are validated and reused')
        write_json(status_path,status)
        raise


if __name__=='__main__':
    main()
