"""Advance fresh teachers to native observations, then stop for real identity review."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json
from hmr4d.lidar_mocap.repair_execution import require_gpu_lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gpu-lease", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    try:
        execute(args)
    except Exception as error:
        root = args.root.resolve()
        state_path = root / "production_stage.json"
        if root.is_relative_to(PROJ_ROOT / "outputs") and state_path.exists():
            state = read_json(state_path)
            if state.get("schema") == "badminton-fresh-production-stage-v1":
                state.update(
                    stage="failed",
                    failed_stage=state.get("failed_stage", state.get("stage")),
                    error_type=type(error).__name__,
                    error=str(error),
                    updated_utc=datetime.now(timezone.utc).isoformat(),
                )
                write_json(state_path, state)
        raise


def execute(args):
    root, source = args.root.resolve(), args.source_root.resolve()
    plan_path = root / "new_source_input.json"
    plan = read_json(plan_path)
    if plan.get("schema") != "badminton-new-source-repair-input-v1" or Path(plan["source_root"]).resolve() != source:
        raise ValueError("Select an explicit frozen fresh-source intake")
    state = dict(
        schema="badminton-fresh-production-stage-v1",
        stage="waiting_initial_teachers",
        source_input_sha256=file_sha(plan_path),
        source_root=str(source),
        teacher_windows=len(plan["jobs"]),
        training_published=False,
    )
    state_path = root / "production_stage.json"
    write_json(state_path, state)
    while True:
        require_gpu_lease(args.gpu_lease)
        progress_path = root / "initial_teacher_progress.json"
        progress = read_json(progress_path) if progress_path.exists() else {}
        if progress.get("state") == "failed":
            raise ValueError("Initial teacher worker failed; native observation is not admitted")
        if progress.get("state") == "complete":
            if progress.get("source_input_sha256") != state["source_input_sha256"]:
                raise ValueError("Initial teachers used another source input")
            if progress.get("windows_complete") != len(plan["jobs"]) or len(progress.get("windows", {})) != len(
                plan["jobs"]
            ):
                raise ValueError("Initial teacher coverage is incomplete")
            if any(file_sha(entry["teacher_manifest"]) != entry["sha256"] for entry in progress["windows"].values()):
                raise ValueError("Completed initial teacher changed")
            break
        time.sleep(5)
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
        PYTHONFAULTHANDLER="1",
    )
    for stage, script, command in (
        (
            "native_pose_observation",
            "observe_badminton_repair_poses.py",
            ["--root", str(root), "--source-root", str(source), "--threads", str(args.threads)],
        ),
        ("pending_identity_inventory", "prepare_badminton_identity_review.py", ["--root", str(root)]),
    ):
        require_gpu_lease(args.gpu_lease)
        if file_sha(plan_path) != state["source_input_sha256"]:
            raise ValueError("Frozen source input changed")
        state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)
        log = root / (stage + ".log")
        with log.open("ab") as handle:
            result = subprocess.run(
                [sys.executable, "-u", str(PROJ_ROOT / "tools" / script), *command],
                cwd=PROJ_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            state.update(stage="failed", failed_stage=stage, log=str(log), exit_code=result.returncode)
            write_json(state_path, state)
            raise RuntimeError(f"Fresh production {stage} failed; see {log}")
    receipt = read_json(root / "pose_receipt.json")
    if receipt.get("state") != "complete" or receipt["config"]["calibration_digest"] != plan["calibration_digest"]:
        raise ValueError("Native observations are incomplete or use another calibration")
    state.update(
        stage="awaiting_identity_inventory_review",
        pose_receipt=str(root / "pose_receipt.json"),
        pose_receipt_sha256=file_sha(root / "pose_receipt.json"),
        inventory=str(root / "identity_evidence/boundary_inventory.pending.json"),
        updated_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_json(state_path, state)
    print("Fresh native observations and pending endpoints ready; identity is not approved automatically", flush=True)


if __name__ == "__main__":
    main()
