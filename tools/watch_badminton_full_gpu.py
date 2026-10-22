"""Consume a real continuous grant, start 1019 B02, and enforce its total deadline."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.gpu_handoff import read_continuous_grant, make_local_stage_lease, stop_owned_cuda_trees
from hmr4d.lidar_mocap.pilot import binding
from hmr4d.lidar_mocap.repair_execution import write_progress_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--grant", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cuda-visible-devices", default="0")
    args = parser.parse_args()
    request_path, grant_path, status_path = [path.resolve() for path in (args.request, args.grant, args.status)]
    if args.threads < 1 or not status_path.is_relative_to(PROJ_ROOT / "outputs"):
        parser.error("Use a local outputs status and positive CPU threads")
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity([0, 1])
    request = json.loads(request_path.read_bytes())
    root = Path(request["scope"][0]["root"]).resolve()
    progress = root / "full_teacher_progress.batch02.json"
    release = root / "teacher_batch_02_release_receipt.json"
    previous = json.loads(status_path.read_bytes()) if status_path.exists() else {}
    if previous.get("pid") and previous["pid"] != os.getpid():
        try:
            old = psutil.Process(previous["pid"])
            if old.create_time() == previous.get("create_time"):
                raise RuntimeError("Another continuous GPU receiver is already running")
        except psutil.NoSuchProcess:
            pass
    state = dict(previous, state="waiting_for_real_continuous_grant", pid=os.getpid(),
        create_time=process.create_time(), request=binding(request_path), grant_path=str(grant_path),
        started_utc=datetime.now(timezone.utc).isoformat(), cpu_only_receiver=True,
        gpu_started=previous.get("gpu_started", False), training_published=False, hidden=True,
        cpu_affinity=[0, 1], priority="BelowNormal")
    authorization, last_signature, child, observer, log_handles = None, None, None, None, []
    last_saved = 0.

    def save(force=False):
        nonlocal last_saved
        if force or time.monotonic() - last_saved >= 10:
            state["heartbeat_utc"] = datetime.now(timezone.utc).isoformat()
            write_progress_json(status_path, state)
            last_saved = time.monotonic()

    def spawn(command, log_name, *, gpu=False):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.cuda_visible_devices if gpu else "-1")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            environment[key] = str(args.threads)
        handle = (status_path.parent / log_name).open("ab")
        log_handles.append(handle)
        result = subprocess.Popen(command, cwd=PROJ_ROOT, env=environment, stdout=handle,
            stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        return result

    save(True)
    while True:
        now = datetime.now(timezone.utc)
        if grant_path.exists():
            signature = hashlib.sha256(grant_path.read_bytes()).hexdigest()
            if signature != last_signature:
                try:
                    if authorization is not None:
                        raise ValueError("The active continuous grant changed; its total deadline cannot be extended")
                    updated = read_continuous_grant(request_path, grant_path)
                except Exception as error:
                    if authorization is not None:
                        stopped = stop_owned_cuda_trees(authorization["roots"])
                        state.update(state="stopped_changed_or_revoked_grant", stopped=stopped,
                                     error_type=type(error).__name__, error=str(error))
                        save(True)
                        return
                    state.update(state="waiting_for_valid_continuous_grant", error_type=type(error).__name__, error=str(error))
                    save()
                else:
                    authorization, last_signature = updated, signature
                    state.update(grant=updated["binding"], approved_deadline_utc=updated["deadline"].isoformat())
                    if not state.get("first_teacher_started"):
                        state["state"] = "waiting_for_approved_start" if now < updated["start"] else "dispatching_first_teacher"
                    state.pop("error", None)
                    state.pop("error_type", None)
        elif authorization is not None:
            stopped = stop_owned_cuda_trees(authorization["roots"])
            state.update(state="stopped_removed_grant", stopped=stopped)
            save(True)
            return
        if authorization is not None:
            if now >= authorization["deadline"]:
                stopped = stop_owned_cuda_trees(authorization["roots"])
                state.update(state="deadline_stopped_owned_cuda_trees", stopped=stopped,
                             stopped_utc=datetime.now(timezone.utc).isoformat())
                if child is not None and child.poll() is not None and not release.exists():
                    partial = json.loads(progress.read_bytes()) if progress.exists() else {}
                    partial.update(state="failed", safe_to_release_gpu=True, exit_code=child.returncode,
                                   reason="continuous_gpu_deadline")
                    write_progress_json(progress, partial)
                save(True)
                return
            if now >= authorization["start"] and not state.get("first_teacher_started"):
                if args.lease.exists() or progress.exists() or release.exists():
                    raise RuntimeError("B02 artifacts already exist; preserve them and dispatch a separate reviewed batch")
                lease = make_local_stage_lease(authorization, root, "teacher", now=now)
                if (datetime.fromisoformat(lease["approved_deadline_utc"]) - now).total_seconds() < 600:
                    state.update(state="insufficient_remaining_time_for_teacher")
                    save(True)
                    return
                write_progress_json(args.lease, lease)
                command = [sys.executable, "-u", str(PROJ_ROOT / "tools/run_badminton_full_native.py"),
                    "--root", str(root), "--stage", "teacher", "--gpu-lease", str(args.lease),
                    "--threads", str(args.threads), "--progress-output", str(progress),
                    "--resume-progress", str(root / "full_teacher_progress.json")]
                child = spawn(command, "continuous_1019_teacher_batch02.log", gpu=True)
                state.update(state="running_1019_teacher_batch02", first_teacher_started=True, gpu_started=True,
                    teacher_cli_pid=child.pid, teacher_cli_create_time=psutil.Process(child.pid).create_time(),
                    local_lease=str(args.lease.resolve()), teacher_progress=str(progress), command=command)
                observer = spawn([sys.executable, "-u", str(PROJ_ROOT / "tools/release_badminton_full_gpu.py"),
                    "--root", str(root), "--lease", str(args.lease), "--progress", str(progress),
                    "--receipt", str(release), "--kind", "teacher"], "continuous_1019_release_batch02.log")
                state["release_observer_cli_pid"] = observer.pid
                save(True)
            if child is not None and child.poll() is not None:
                code = child.returncode
                if code and not state.get("teacher_failure_closed"):
                    stopped = stop_owned_cuda_trees([root])
                    partial = json.loads(progress.read_bytes()) if progress.exists() else {}
                    if partial.get("safe_to_release_gpu") is not True:
                        partial.update(state="failed", safe_to_release_gpu=True, exit_code=code)
                        write_progress_json(progress, partial)
                    state.update(teacher_failure_closed=True, failed_owned_processes_stopped=stopped)
                if observer is not None and observer.poll() is not None:
                    state.update(state="first_teacher_released_waiting_next_dispatch" if code == 0 and observer.returncode == 0
                                 else "first_teacher_failed_waiting_review", teacher_exit_code=code,
                                 release_observer_exit_code=observer.returncode)
                    if release.exists():
                        state["first_teacher_release_receipt"] = binding(release)
                    if progress.exists():
                        state["first_teacher_final_state"] = json.loads(progress.read_bytes()).get("state")
                    state["deadline_watchdog_active"] = True
                    save(True)
                    child, observer = None, None
        save()
        time.sleep(2)


if __name__ == "__main__":
    main()
