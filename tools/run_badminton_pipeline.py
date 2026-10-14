"""Run, export and verify all badminton sessions with a durable local status file.

Optional --adopt-pid waits for a verified already running Windows pairing CLI.
It never attaches to arbitrary processes or interprets existence as success.
"""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.pipeline import BadmintonPipeline,write_json


def wait_for_existing(pid,root,output):
    if os.name!='nt':
        raise ValueError('--adopt-pid requires Windows process identity verification')
    command = f"Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}' | Select-Object CommandLine | ConvertTo-Json -Compress"
    found = subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
    identity = json.loads(found.stdout or 'null')
    line = identity.get('CommandLine','') if identity else ''
    expected_output = str(output.relative_to(root)).replace('\\','/')
    output_matches = ('--output-root' not in line and expected_output=='outputs/badminton') or expected_output in line.replace('\\','/')
    if ('tools/pair_badminton.py run' not in line or '--preproc-precision float16' not in line
            or '--roi-margin-px 20' not in line or not output_matches
            or '--session' in line or '--duration-seconds' in line):
        raise ValueError('PID does not identify the expected full calibrated pairing job')
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=(wintypes.HANDLE,wintypes.DWORD)
    kernel.GetExitCodeProcess.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD))
    kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x100000|0x1000,False,pid)
    if not handle:
        raise OSError(ctypes.get_last_error(),'Cannot verify/wait for pairing process')
    try:
        while kernel.WaitForSingleObject(handle,5000)==0x102:
            pass
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):
            raise OSError(ctypes.get_last_error(),'Cannot read pairing exit status')
        return code.value
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adopt-pid',type=int)
    parser.add_argument('--output-root',type=Path,default=Path('outputs/badminton'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    output = args.output_root.resolve()
    if not output.is_relative_to(root):
        raise ValueError('Output must remain inside the GVHMR workspace')
    output.mkdir(parents=True,exist_ok=True)
    status_path = output/'job_status.json'
    status = {'schema':'badminton-full-job-v1','state':'running','supervisor_pid':os.getpid(),
              'started_utc':datetime.now(timezone.utc).isoformat(),'output_root':str(output),
              'current_phase':'fitting','adopted_pairing_pid':args.adopt_pid}
    write_json(status_path,status)
    python = str(root/'.venv/Scripts/python.exe') if os.name=='nt' else sys.executable
    try:
        pipeline = BadmintonPipeline('D:/PROJECT/LIDAR/datas/badminton/sessions',
                                     'D:/PROJECT/LIDAR/lidar_project',output,
                                     preproc_precision='float16',roi_margin_px=20)
        pipeline.prepare()  # Also rejects resuming a superseded adopted snapshot.
        status.update(calibration_digest=pipeline.calibration.data['snapshot_digest'],
                      active_selection_sha256=pipeline.calibration.data['active_selection_sha256'],
                      geometry_variant=pipeline.calibration.data.get('geometry_variant'))
        write_json(status_path,status)
        code = None
        if args.adopt_pid:
            code = wait_for_existing(args.adopt_pid,root,output)
        for attempt in range(3):
            if code==0:
                break
            status.update(current_phase='fitting',attempt=attempt+1)
            write_json(status_path,status)
            suffix = '' if attempt==0 and not args.adopt_pid else f'_resume_{attempt+1}'
            with (output/f'full_run{suffix}.log').open('w',encoding='utf-8') as log, (output/f'full_run{suffix}.err.log').open('w',encoding='utf-8') as errors:
                result = subprocess.run([python,'-u','tools/pair_badminton.py','run','--preproc-precision','float16',
                                         '--roi-margin-px','20','--output-root',str(output)],stdout=log,stderr=errors)
                code = result.returncode
        if code:
            raise RuntimeError(f'Pairing process exited with {code}; matching caches remain resumable')
        pipeline = BadmintonPipeline('D:/PROJECT/LIDAR/datas/badminton/sessions',
                                     'D:/PROJECT/LIDAR/lidar_project',output,
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
            subprocess.run([python,'tools/pair_badminton.py','export','--output-root',str(output)],stdout=log,stderr=subprocess.STDOUT,check=True)
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
