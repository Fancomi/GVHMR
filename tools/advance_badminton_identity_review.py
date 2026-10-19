"""Supervise fresh native identity evidence through geometric review and sealing."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import psutil

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.rally_pipeline import read_json
from hmr4d.lidar_mocap.repair_execution import require_gpu_lease, write_progress_json as write_json


def execute(args):
    root, source = args.root.resolve(), args.source_root.resolve()
    plan_path = root / "new_source_input.json"
    plan = read_json(plan_path)
    if plan.get("schema") != "badminton-new-source-repair-input-v1" or Path(plan["source_root"]).resolve() != source:
        raise ValueError("Select the frozen fresh-source repair and source roots")
    if not root.is_relative_to(PROJ_ROOT / "outputs"):
        raise ValueError("Production state must remain inside outputs")
    state = dict(
        schema="badminton-fresh-production-stage-v1",
        stage="identity_native_pose_observation",
        source_input_sha256=file_sha(plan_path),
        source_root=str(source),
        training_published=False,
        identity_cases=len(read_json(root / "identity_evidence/boundaries.json")),
    )
    state_path = root / "production_stage.json"

    def update(**values):
        state.update(values, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)

    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
        PYTHONFAULTHANDLER="1",
    )

    def run(stage, script, command):
        require_gpu_lease(args.gpu_lease)
        if file_sha(plan_path) != state["source_input_sha256"]:
            raise ValueError("Frozen production input changed")
        log = root / (stage + ".log")
        update(stage=stage, log=str(log))
        with log.open("ab") as handle:
            result = subprocess.run(
                [sys.executable, "-u", str(PROJ_ROOT / "tools" / script), *map(str, command)],
                cwd=PROJ_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"{stage} failed ({result.returncode}); see {log}")

    try:
        require_gpu_lease(args.gpu_lease)
        if args.wait_pose_worker_pid:
            worker = psutil.Process(args.wait_pose_worker_pid)
            arguments = worker.cmdline()
            command = " ".join(arguments)
            if "observe_badminton_repair_poses.py" not in command or "identity_evidence" not in command:
                raise ValueError("Refusing to attach to an unrelated process")
            worker_root = (Path(worker.cwd()) / arguments[arguments.index("--root") + 1]).resolve()
            if worker_root != root / "identity_evidence":
                raise ValueError("Identity worker belongs to another production run")
            update(worker_pid=worker.pid, worker_created=worker.create_time())
            while True:
                require_gpu_lease(args.gpu_lease)
                try:
                    worker.wait(timeout=5)
                    break
                except psutil.TimeoutExpired:
                    pass
            update(worker_finished=True)
        else:
            run(
                "identity_native_pose_observation",
                "observe_badminton_repair_poses.py",
                [
                    "--root",
                    root / "identity_evidence",
                    "--source-root",
                    source,
                    "--reuse-poses-root",
                    root,
                    "--threads",
                    args.threads,
                ],
            )
        receipt_path = root / "identity_evidence/pose_receipt.json"
        receipt = read_json(receipt_path)
        if (
            receipt.get("state") != "complete"
            or receipt["config"]["calibration_digest"] != plan["calibration_digest"]
            or receipt["source_union_sha256"] != state["source_input_sha256"]
        ):
            raise ValueError("Dense native identity observations are incomplete or have another source binding")
        update(identity_pose_receipt=str(receipt_path), identity_pose_receipt_sha256=file_sha(receipt_path))
        run("identity_geometric_review", "review_badminton_identity_views.py", ["--root", root])
        run("identity_geometric_seal", "build_badminton_identity_decisions.py", ["--root", root])
        decision_path = root / "identity_evidence/identity_decisions.json"
        decision = read_json(decision_path)
        update(
            stage=(
                "awaiting_identity_case_adjudication"
                if decision["review_status"] != "reviewed"
                else "identity_ready_for_body_prepare"
            ),
            identity_decisions=str(decision_path),
            identity_decisions_sha256=file_sha(decision_path),
            unresolved_cases=decision["unresolved_cases"],
        )
    except Exception as error:
        update(stage="failed", failed_stage=state["stage"], error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gpu-lease", type=Path, required=True)
    parser.add_argument(
        "--wait-pose-worker-pid", type=int, help="Attach read-only to an already running identity observer"
    )
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    execute(args)


if __name__ == "__main__":
    main()
