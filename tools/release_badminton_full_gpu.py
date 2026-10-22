"""Release this run's lease after its stage and actual owned CUDA processes exit."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

import psutil

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.pilot import binding
from hmr4d.lidar_mocap.rally_pipeline import read_json
from hmr4d.lidar_mocap.repair_execution import (
    native_gpu_command_matches,
    write_progress_json,
    read_live_bytes as read_state_bytes,
)


def identity_alive(identity):
    if (
        type(identity.get("pid")) is not int
        or identity["pid"] <= 0
        or type(identity.get("create_time")) not in (int, float)
        or not math.isfinite(identity["create_time"])
        or identity["create_time"] <= 0
    ):
        raise ValueError("An observed process needs its original PID and create_time")
    try:
        return psutil.Process(identity["pid"]).create_time() == identity["create_time"]
    except psutil.NoSuchProcess:
        return False


def observe_owned(root):
    """Include the original launchers, redirectors and all actual descendants."""
    observed = {}
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            if (
                process.pid != os.getpid()
                and (process.info["name"] or "").lower() == "python.exe"
                and native_gpu_command_matches(process.info["cmdline"], root)
            ):
                for owned in [process, *process.children(recursive=True)]:
                    item = dict(pid=owned.pid, create_time=owned.create_time())
                    observed[(item["pid"], item["create_time"])] = item
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--kind", choices=("teacher", "fit", "poses"), required=True)
    parser.add_argument("--seed-evidence", type=Path, help="Actual prior process observation for crash recovery")
    parser.add_argument("--seed-evidence-sha256")
    args = parser.parse_args()
    root, progress_path, receipt_path = args.root.resolve(), args.progress.resolve(), args.receipt.resolve()
    if (
        not root.is_relative_to(PROJ_ROOT / "outputs")
        or root == PROJ_ROOT / "outputs"
        or not progress_path.is_relative_to(root)
        or not receipt_path.is_relative_to(root)
        or receipt_path.exists()
    ):
        parser.error("Use this run's actual progress and a new in-run release receipt")
    if bool(args.seed_evidence) != bool(args.seed_evidence_sha256):
        parser.error("Recovery evidence requires its explicit SHA256")
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    psutil.Process().cpu_affinity([0, 1])
    frozen = read_json(root / "pilot_input.json")
    started = datetime.now(timezone.utc).isoformat()
    observed, stable = {}, 0
    journal = receipt_path.with_suffix(".observed.json")
    seed = None
    if args.seed_evidence:
        seed = binding(args.seed_evidence)
        if seed["sha256"] != args.seed_evidence_sha256:
            raise ValueError("Prior actual process evidence changed")
        prior = json.loads(read_state_bytes(args.seed_evidence))
        if (
            prior.get("root") != str(root)
            or prior.get("lease_before") != binding(args.lease)
            or not prior.get("observed_process_identities")
        ):
            raise ValueError("Recovery observations must bind this exact held lease/root")
        for item in prior["observed_process_identities"]:
            identity_alive(item)  # Validate the actual original identity before preserving it.
            observed[(item["pid"], item["create_time"])] = dict(pid=item["pid"], create_time=item["create_time"])
    if journal.exists():
        prior = json.loads(read_state_bytes(journal))
        if prior.get("root") != str(root) or prior.get("lease") != str(args.lease.resolve()):
            raise ValueError("Prior observer journal belongs to another lease/root")
        for item in prior["observed_process_identities"]:
            identity_alive(item)
            observed[(item["pid"], item["create_time"])] = item
    while True:
        observed.update(observe_owned(root))
        alive = [item for item in observed.values() if identity_alive(item)]
        write_progress_json(
            journal,
            dict(
                schema="gvhmr-full-native-process-observations-v1",
                root=str(root),
                lease=str(args.lease.resolve()),
                seed_evidence=seed,
                observed_process_identities=list(observed.values()),
                alive=alive,
                observer=dict(pid=os.getpid(), create_time=psutil.Process().create_time()),
                heartbeat_utc=datetime.now(timezone.utc).isoformat(),
            ),
        )
        lease = json.loads(read_state_bytes(args.lease))
        if lease.get("owner") != "GVHMR" or lease.get("manifest_sha256") != frozen["manifest"]["sha256"]:
            raise ValueError("Release observer is not the owner of this source-bound lease")
        if lease["status"] != "held":
            raise ValueError("Another writer already changed the lease; this observer will not overwrite it")
        payload = read_state_bytes(progress_path) if progress_path.exists() else b"{}"
        state = json.loads(payload)
        completed = (
            state.get("state") in ("complete", "yielded_before_deadline", "fit_windows_complete", "failed")
            and state.get("safe_to_release_gpu") is True
        )
        if args.kind == "poses":
            completed = state.get("stage") in ("poses_complete", "failed")
        stable = stable + 1 if completed and observed and not alive else 0
        if stable >= 2:
            snapshot = receipt_path.with_suffix(".progress.json")
            if snapshot.exists():
                raise ValueError("Release progress snapshot already exists")
            if read_state_bytes(progress_path) != payload:
                stable = 0
                continue
            snapshot.write_bytes(payload)
            progress = binding(snapshot)
            key = {"teacher": "full_teacher_progress", "fit": "full_fit_progress", "poses": "full_pose_progress"}[
                args.kind
            ]
            before = binding(args.lease)
            lease.update(
                status="released",
                released_utc=datetime.now(timezone.utc).isoformat(),
                active_stage=args.kind + "_safe_boundary",
                release_reason="Completed stage boundary and all actual owned native CUDA executors exited; no next GPU stage started",
                all_observed_owned_processes_exited=True,
                **{key: progress},
            )
            write_progress_json(args.lease, lease)
            result = dict(
                schema="gvhmr-full-native-gpu-release-v1",
                status="released",
                kind=args.kind,
                watcher_started_utc=started,
                released_utc=lease["released_utc"],
                deadline_utc=lease["approved_deadline_utc"],
                lease_before=before,
                lease_after=binding(args.lease),
                live_progress_path=str(progress_path),
                observed_process_identities=[dict(item, exited=True) for _, item in sorted(observed.items())],
                process_observations=binding(journal),
                seed_evidence=seed,
                **{key: progress},
            )
            write_progress_json(receipt_path, result)
            print(
                json.dumps(dict(status="released", released_utc=lease["released_utc"], receipt=str(receipt_path))),
                flush=True,
            )
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
