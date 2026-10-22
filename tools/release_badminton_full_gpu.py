"""Release this run's lease after its stage and actual owned CUDA processes exit."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import psutil

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.pilot import binding
from hmr4d.lidar_mocap.rally_pipeline import read_json
from hmr4d.lidar_mocap.repair_execution import native_gpu_command_matches, write_progress_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--kind", choices=("teacher", "fit", "poses"), required=True)
    args = parser.parse_args()
    root, progress_path, receipt_path = args.root.resolve(), args.progress.resolve(), args.receipt.resolve()
    if (not root.is_relative_to(PROJ_ROOT / "outputs") or root == PROJ_ROOT / "outputs"
            or not progress_path.is_relative_to(root) or not receipt_path.is_relative_to(root)
            or receipt_path.exists()):
        parser.error("Use this run's actual progress and a new in-run release receipt")
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    psutil.Process().cpu_affinity([0, 1])
    frozen = read_json(root / "pilot_input.json")
    started = datetime.now(timezone.utc).isoformat()
    observed, stable = set(), 0
    while True:
        lease = read_json(args.lease)
        if lease.get("owner") != "GVHMR" or lease.get("manifest_sha256") != frozen["manifest"]["sha256"]:
            raise ValueError("Release observer is not the owner of this source-bound lease")
        if lease["status"] != "held":
            raise ValueError("Another writer already changed the lease; this observer will not overwrite it")
        state = read_json(progress_path) if progress_path.exists() else {}
        alive = []
        for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
            try:
                if (process.pid != os.getpid() and (process.info["name"] or "").lower() == "python.exe"
                        and native_gpu_command_matches(process.info["cmdline"], root)):
                    identity = (process.pid, process.info["create_time"])
                    observed.add(identity)
                    alive.append(identity)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        completed = (state.get("state") in ("complete", "yielded_before_deadline", "fit_windows_complete", "failed")
                     and state.get("safe_to_release_gpu") is True)
        if args.kind == "poses":
            completed = state.get("stage") in ("poses_complete", "failed")
        stable = stable + 1 if completed and not alive else 0
        if stable >= 2:
            snapshot = receipt_path.with_suffix(".progress.json")
            if snapshot.exists():
                raise ValueError("Release progress snapshot already exists")
            snapshot.write_bytes(progress_path.read_bytes())
            progress = binding(snapshot)
            key = {"teacher": "full_teacher_progress", "fit": "full_fit_progress", "poses": "full_pose_progress"}[args.kind]
            before = binding(args.lease)
            lease.update(status="released", released_utc=datetime.now(timezone.utc).isoformat(),
                active_stage=args.kind + "_safe_boundary", release_reason="Completed stage boundary and all actual owned native CUDA executors exited; no next GPU stage started",
                all_observed_owned_processes_exited=True, **{key: progress})
            write_progress_json(args.lease, lease)
            result = dict(schema="gvhmr-full-native-gpu-release-v1", status="released", kind=args.kind,
                watcher_started_utc=started, released_utc=lease["released_utc"], deadline_utc=lease["approved_deadline_utc"],
                lease_before=before, lease_after=binding(args.lease), live_progress_path=str(progress_path),
                observed_process_identities=[dict(pid=pid, create_time=created, exited=True) for pid, created in sorted(observed)],
                **{key: progress})
            write_progress_json(receipt_path, result)
            print(json.dumps(dict(status="released", released_utc=lease["released_utc"], receipt=str(receipt_path))), flush=True)
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
