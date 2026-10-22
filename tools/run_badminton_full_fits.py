"""Fit independent full-native windows, releasing GPU before CPU aggregation."""

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
    load_isolated_stage_progress, remaining_lease_seconds, require_gpu_lease,
    run_isolated_native_worker, write_progress_json,
)


def fit_inputs(root):
    frozen = read_json(root / "pilot_input.json")
    check_binding(frozen["manifest"])
    validate_full_native_spec(read_json(frozen["manifest"]["path"]))
    repair = root / "repair"
    plan = read_json(repair / "new_source_input.json")
    input_sha = file_sha(repair / "new_source_input.json")
    dual = read_json(root / "dual_teacher_receipt.json")
    if dual.get("complete") is not True or dual["source_input_sha256"] != input_sha:
        raise ValueError("Full fitting requires every independent native camera teacher")
    required = {j["source_id"] + ":" + Path(j["source_window"]).name for j in plan["jobs"]}
    if set(dual["windows"]) != required:
        raise ValueError("Independent teacher coverage differs from the full fit plan")
    for camera_windows in dual["windows"].values():
        for item in camera_windows.values():
            check_binding(item)
    identity = read_json(repair / "identity_evidence/identity_decisions.json")
    pose = read_json(repair / "pose_receipt.json")
    if (identity.get("review_status") != "reviewed" or identity["source_union_sha256"] != input_sha
            or not required <= set(identity["windows"]) or pose.get("state") != "complete"
            or pose["source_union_sha256"] != input_sha or identity["calibration_digest"] != plan["calibration_digest"]
            or pose["config"]["calibration_digest"] != plan["calibration_digest"]
            or pose["plan_sha256"] != file_sha(repair / "case_plan.json")):
        raise ValueError("Full fitting requires complete current poses and reviewed native identity evidence")
    if (root / "fit_reuse_audit.json").exists():
        applied = read_json(root / "fit_reuse_applied_receipt.json")
        check_binding(applied["audit"])
        check_binding(applied["completed_current_pose_receipt"])
        check_binding(applied["reviewed_current_identity"])
    return frozen, plan, input_sha


def fit_command(root, threads):
    repair = root / "repair"
    return [sys.executable, "-u", str(PROJ_ROOT / "tools/repair_badminton_body_labels.py"),
        "--union", str(repair / "new_source_input.json"), "--source-root", str(root / "source"),
        "--output-root", str(repair / "final_labels"), "--native-poses-root", str(repair),
        "--joint-ray-filter", "--identity-decisions", str(repair / "identity_evidence/identity_decisions.json"),
        "--distal-pose-weight", "0.03", "--lower-limb-reprojection-weight", "2", "--threads", str(threads)]


def validate_root(root, threads):
    root = Path(root).resolve()
    if not root.is_relative_to(PROJ_ROOT / "outputs") or root == PROJ_ROOT / "outputs" or threads < 1:
        raise ValueError("Use an independent full-native output and positive threads")
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    psutil.Process().cpu_affinity([0, 1])
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(threads)
    return root


def run_full_fits(args):
    root = validate_root(args.root, args.threads)
    if args.safety_seconds < 180:
        raise ValueError("Reserve at least 180 seconds before starting another native fit")
    frozen, plan, input_sha = fit_inputs(root)
    state_path, state = load_isolated_stage_progress(root, getattr(args, "progress_output", None),
        getattr(args, "resume_progress", None), default_name="full_fit_progress.json",
        release_pattern="fit_batch_*_release_receipt.json", binding_key="full_fit_progress")
    if state.get("source_input_sha256", input_sha) != input_sha:
        raise ValueError("Full-native fit plan changed")
    if state.get("state") == "running" and psutil.pid_exists(state.get("supervisor_pid", 0)):
        raise ValueError("Another full-native fit supervisor is running")
    require_gpu_lease(args.gpu_lease)
    remaining_lease_seconds(read_json(args.gpu_lease), frozen["manifest"]["sha256"])
    state.update(state="running", supervisor_pid=os.getpid(), source_input_sha256=input_sha,
        gpu_lease=str(args.gpu_lease.resolve()), windows_total=len(plan["jobs"]), review_only=True,
        training_published=False, safe_to_release_gpu=False, started_batch_utc=datetime.now(timezone.utc).isoformat())
    logs = root / "logs/full_fits"
    logs.mkdir(parents=True, exist_ok=True)
    command = fit_command(root, args.threads)
    started, new_windows = time.perf_counter(), 0
    cached = {}
    for job in plan["jobs"]:
        sid, name = job["source_id"], Path(job["source_window"]).name
        path = root / "repair/final_labels/sessions" / sid / "windows" / name / "fit_report.json"
        if path.exists():
            cached.setdefault(sid, []).append(name)
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    for sid, names in cached.items():
        execution = run_isolated_native_worker(command + ["--require-completed", "--device", "cpu",
            "--partition-worker", "--source", sid] + ["--window=" + name for name in names],
            cwd=PROJ_ROOT, log_path=logs / f"{sid}_completed_only.log", state=state, state_path=state_path,
            active=dict(source_id=sid, role="cpu_completed_cache_validation"), env=cpu_env)
        if execution["exit_code"]:
            state.update(state="failed", safe_to_release_gpu=True, exit_code=execution["exit_code"])
            write_progress_json(state_path, state)
            raise RuntimeError("Existing full-native fit cache changed; no fallback recomputation performed")
        for name in names:
            path = root / "repair/final_labels/sessions" / sid / "windows" / name / "fit_report.json"
            state["windows"][sid + ":" + name] = dict(report=binding(path), reused_or_completed=True)
    state.pop("active", None)
    write_progress_json(state_path, state)
    for job in plan["jobs"]:
        sid, name = job["source_id"], Path(job["source_window"]).name
        key = sid + ":" + name
        if key in state["windows"]:
            check_binding(state["windows"][key]["report"])
            continue
        require_gpu_lease(args.gpu_lease)
        if remaining_lease_seconds(read_json(args.gpu_lease), frozen["manifest"]["sha256"]) < args.safety_seconds:
            break
        if file_sha(root / "repair/new_source_input.json") != input_sha:
            raise ValueError("Full source input changed during fitting")
        execution = run_isolated_native_worker(command + ["--partition-worker", "--source", sid, "--window=" + name],
            cwd=PROJ_ROOT, log_path=logs / f"{sid}_{name}.log", state=state, state_path=state_path,
            active=dict(source_id=sid, window=name, role="native_v2_fit"))
        if execution["exit_code"]:
            state.update(state="failed", safe_to_release_gpu=True, exit_code=execution["exit_code"])
            write_progress_json(state_path, state)
            raise RuntimeError("Native full-window fitting failed; see the recorded worker log")
        report = root / "repair/final_labels/sessions" / sid / "windows" / name / "fit_report.json"
        state["windows"][key] = dict(report=binding(report), reused_or_completed=False, **execution)
        state.pop("active", None)
        state["completed_windows"] = len(state["windows"])
        write_progress_json(state_path, state)
        new_windows += 1
        print(dict(source=sid, window=name, completed=len(state["windows"]), total=len(plan["jobs"])), flush=True)
    complete = len(state["windows"]) == len(plan["jobs"])
    state.update(state="fit_windows_complete" if complete else "yielded_before_deadline", safe_to_release_gpu=True,
        completed_windows=len(state["windows"]), exit_code=0, new_windows_this_batch=new_windows,
        batch_seconds=time.perf_counter() - started, completed_batch_utc=datetime.now(timezone.utc).isoformat(),
        cpu_aggregation_pending=complete)
    write_progress_json(state_path, state)
    progress = read_json(root / "pilot_progress.json")
    progress.update(stage="fit_windows_complete_cpu_aggregation_pending" if complete else "fit_safe_boundary",
                    completed_fit_windows=len(state["windows"]))
    write_progress_json(root / "pilot_progress.json", progress)
    return dict(state=state["state"], safe_to_release_gpu=True, completed_windows=len(state["windows"]),
                cpu_aggregation_pending=complete)


def aggregate_full_fits(args):
    root = validate_root(args.root, args.threads)
    _, plan, _ = fit_inputs(root)
    state_path = root / "full_aggregate_progress.json"
    state = dict(state="running_cpu_aggregate", cpu_only=True, gpu_started=False, training_published=False)
    execution = run_isolated_native_worker(fit_command(root, args.threads) + ["--require-completed", "--device", "cpu"],
        cwd=PROJ_ROOT, log_path=root / "logs/full_aggregate_cpu.log", state=state, state_path=state_path,
        active=dict(role="cpu_completed_only_aggregate"), env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
    state.pop("active", None)
    state.update(state="complete_cpu_aggregate" if execution["exit_code"] == 0 else "failed", **execution)
    write_progress_json(state_path, state)
    if execution["exit_code"]:
        raise RuntimeError("Completed-only full aggregate failed; it did not start new fitting")
    progress = read_json(root / "pilot_progress.json")
    progress.update(stage="fit_complete", fitted_windows=len(plan["jobs"]), training_published=False)
    write_progress_json(root / "pilot_progress.json", progress)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu-lease", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--safety-seconds", type=int, default=300)
    parser.add_argument("--progress-output", type=Path)
    parser.add_argument("--resume-progress", type=Path)
    parser.add_argument("--aggregate-only", action="store_true")
    arguments = parser.parse_args()
    if not arguments.aggregate_only and arguments.gpu_lease is None:
        parser.error("Full fitting requires a fresh held GPU lease")
    print(aggregate_full_fits(arguments) if arguments.aggregate_only else run_full_fits(arguments))
