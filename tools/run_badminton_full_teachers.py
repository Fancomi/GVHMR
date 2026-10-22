"""Run full-native teachers serially, yielding at a completed camera-window boundary."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

import psutil

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.pilot import binding, check_binding, validate_full_native_spec
from hmr4d.lidar_mocap.rally_pipeline import read_json
from hmr4d.lidar_mocap.repair_execution import (
    require_gpu_lease, write_progress_json, load_isolated_stage_progress, run_isolated_native_worker,
    remaining_lease_seconds, begin_isolated_batch,
)


def teacher_directory(repair, job, camera):
    window = Path(job["source_window"])
    return (window / "teacher" if camera == "cam2" else
            repair / "secondary_teachers" / camera / "sessions" / job["source_id"] / "windows" / window.name / "teacher")


def check_cache(directory, entry):
    check_binding(entry["manifest"])
    if Path(entry["manifest"]["path"]).resolve() != (directory / "manifest.json").resolve():
        raise ValueError("Teacher checkpoint points to another full-run window")
    for relative, expected in entry["artifacts_sha256"].items():
        if file_sha(directory / relative) != expected:
            raise ValueError("A completed native teacher cache changed")


def cache_entry(directory):
    manifest = read_json(directory / "manifest.json")
    expected = {"manifest.json", "detections.pt", "vitpose.pt", "features.pt"}
    expected.update(track["path"] for track in manifest["tracks"])
    return dict(manifest=binding(directory / "manifest.json"),
                artifacts_sha256={name: file_sha(directory / name) for name in sorted(expected)})


def run_full_teachers(args):
    if args.threads < 1 or args.safety_seconds < 180:
        raise ValueError("Use positive thread count and at least 180 seconds of launch safety margin")
    root = args.root.resolve()
    if not root.is_relative_to(PROJ_ROOT / "outputs") or root == PROJ_ROOT / "outputs":
        raise ValueError("Use the separately prepared full-native root")
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity([0, 1])
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    frozen = read_json(root / "pilot_input.json")
    check_binding(frozen["manifest"])
    validate_full_native_spec(read_json(frozen["manifest"]["path"]))
    source, repair = root / "source", root / "repair"
    input_path = repair / "new_source_input.json"
    plan, input_sha = read_json(input_path), file_sha(input_path)
    state_path, state = load_isolated_stage_progress(root, getattr(args, "progress_output", None),
        getattr(args, "resume_progress", None), default_name="full_teacher_progress.json",
        release_pattern="teacher_batch_*_release_receipt.json", binding_key="full_teacher_progress")
    if state.get("source_input_sha256", input_sha) != input_sha:
        raise ValueError("Full-native teacher plan changed")
    if state.get("state") == "running" and state.get("supervisor_pid") != os.getpid():
        previous_pid = state.get("supervisor_pid")
        if previous_pid and psutil.pid_exists(previous_pid):
            raise ValueError("Another full-native teacher supervisor is still running")
    begin_isolated_batch(state, "new_camera_windows_this_batch")
    if (root / "teacher_reuse_receipt.json").exists():
        reused = read_json(root / "teacher_reuse_receipt.json")
        check_binding(reused["source_input"])
        for key, info in reused["windows"].items():
            current = state["windows"].setdefault(key, {})
            for camera, item in info["cameras"].items():
                current.setdefault(camera, dict(manifest=item["target"],
                    artifacts_sha256=item["copied_files_sha256"], reused=True, seconds=0))
    started = time.perf_counter()
    state.update(state="running", supervisor_pid=os.getpid(), source_input_sha256=input_sha,
                 gpu_lease=str(args.gpu_lease.resolve()), started_batch_utc=datetime.now(timezone.utc).isoformat(),
                 windows_total=len(plan["jobs"]), review_only=True, training_published=False,
                 safe_to_release_gpu=False)
    state["completed_camera_windows"] = sum(len(v) for v in state["windows"].values())
    progress = read_json(root / "pilot_progress.json")
    for key in ("failed_stage", "failed_command", "error_type", "error", "failed_utc", "exit_code"):
        progress.pop(key, None)
    progress.update(stage="teacher_running", supervisor_pid=os.getpid(),
                    completed_teacher_camera_windows=state["completed_camera_windows"])
    write_progress_json(root / "pilot_progress.json", progress)
    write_progress_json(root / "teacher_progress_pointer.json", dict(
        progress_path=str(state_path), gpu_lease=str(args.gpu_lease.resolve()), state="running"))
    logs = root / "logs/full_teachers"
    logs.mkdir(parents=True, exist_ok=True)
    new_count, exhausted = 0, False
    write_progress_json(state_path, state)
    for job in plan["jobs"]:
        sid, name = job["source_id"], Path(job["source_window"]).name
        key = sid + ":" + name
        current = state["windows"].setdefault(key, {})
        for camera in ("cam2", "cam1"):
            directory = teacher_directory(repair, job, camera)
            if camera in current:
                check_cache(directory, current[camera])
                continue
            require_gpu_lease(args.gpu_lease)
            remaining = remaining_lease_seconds(read_json(args.gpu_lease), frozen["manifest"]["sha256"])
            if remaining < args.safety_seconds:
                exhausted = True
                break
            if file_sha(input_path) != input_sha:
                raise ValueError("Full-native input changed during teacher execution")
            command = [sys.executable, "-u", str(PROJ_ROOT / "tools/prepare_badminton_new_source_repair.py"),
                "--source-root", str(source), "--root", str(repair), "--stage", "teacher",
                "--source", sid, "--window=" + name, "--camera", camera, "--threads", str(args.threads)]
            log_path = logs / f"{sid}_{name}_{camera}.log"
            execution = run_isolated_native_worker(command, cwd=PROJ_ROOT, log_path=log_path,
                state=state, state_path=state_path, active=dict(source_id=sid, window=name, camera=camera))
            if execution["exit_code"]:
                state.update(state="failed", exit_code=execution["exit_code"], safe_to_release_gpu=True)
                write_progress_json(state_path, state)
                raise RuntimeError("Native teacher failed; see " + str(log_path))
            current[camera] = dict(cache_entry(directory), reused=False, **execution)
            new_count += 1
            state["new_camera_windows_this_batch"] = new_count
            state.pop("active", None)
            state["completed_camera_windows"] = sum(len(v) for v in state["windows"].values())
            write_progress_json(state_path, state)
            print(dict(source=sid, window=name, camera=camera, complete=state["completed_camera_windows"],
                       total=2 * len(plan["jobs"]), seconds=current[camera]["seconds"]), flush=True)
        if exhausted:
            break
    complete = sum(len(v) for v in state["windows"].values()) == 2 * len(plan["jobs"])
    state.update(state="complete" if complete else "yielded_before_deadline", safe_to_release_gpu=True, exit_code=0,
        completed_camera_windows=sum(len(v) for v in state["windows"].values()),
        new_camera_windows_this_batch=new_count, batch_seconds=time.perf_counter() - started,
        completed_batch_utc=datetime.now(timezone.utc).isoformat())
    write_progress_json(state_path, state)
    if complete:
        windows = {key: {camera: value["manifest"] for camera, value in item.items()}
                   for key, item in state["windows"].items()}
        write_progress_json(root / "dual_teacher_receipt.json", dict(complete=True, windows=windows,
            source_input_sha256=input_sha, cameras=["cam1", "cam2"], training_published=False,
            no_unverified_cross_camera_track_merge=True,
            fusion_policy="Native v2 fit uses cam2 prior plus both native keypoint views and LiDAR; cam1 retained independently"))
    progress = read_json(root / "pilot_progress.json")
    progress.update(stage="teacher_complete" if complete else "teacher_safe_boundary",
                    completed_teacher_camera_windows=state["completed_camera_windows"])
    write_progress_json(root / "pilot_progress.json", progress)
    write_progress_json(root / "teacher_progress_pointer.json", dict(
        progress_path=str(state_path), progress_sha256=file_sha(state_path),
        gpu_lease=str(args.gpu_lease.resolve()), state=state["state"]))
    print(dict(state=state["state"], safe_to_release_gpu=True, complete=state["completed_camera_windows"]), flush=True)
    return dict(state=state["state"], safe_to_release_gpu=True,
                completed_camera_windows=state["completed_camera_windows"], windows_total=len(plan["jobs"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu-lease", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--safety-seconds", type=int, default=300)
    parser.add_argument("--progress-output", type=Path)
    parser.add_argument("--resume-progress", type=Path)
    run_full_teachers(parser.parse_args())


if __name__ == "__main__":
    main()
