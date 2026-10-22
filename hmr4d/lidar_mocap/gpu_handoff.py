"""Explicit continuous GPU handoffs and bounded, source-bound local leases."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import psutil

from hmr4d import PROJ_ROOT
from .pilot import binding, check_binding
from .repair_execution import native_gpu_command_matches


def utc_time(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("GPU authorization timestamps require an explicit UTC offset")
    return result.astimezone(timezone.utc)


def read_continuous_grant(request_path, grant_path):
    request_path, grant_path = Path(request_path).resolve(), Path(grant_path).resolve()
    request = json.loads(request_path.read_bytes())
    grant = json.loads(grant_path.read_bytes())
    request_binding = binding(request_path)
    if (request.get("schema") != "gvhmr-continuous-production-gpu-request-v1"
            or grant.get("schema") != "gvhmr-continuous-production-gpu-grant-v1"
            or grant.get("status") != "held" or grant.get("owner") != "GVHMR"
            or grant.get("request_sha256") != request_binding["sha256"]
            or grant.get("request") != request_binding
            or grant.get("allow_local_stage_leases") is not True
            or grant.get("native_gpu_returned") is not True or grant.get("training_published") is not False):
        raise ValueError("A real source-bound continuous GPU grant is required")
    start, deadline = utc_time(grant["approved_start_utc"]), utc_time(grant["approved_deadline_utc"])
    if deadline <= start or deadline > start + timedelta(hours=request["requested_maximum_hours"]):
        raise ValueError("Continuous GPU deadline exceeds the explicitly requested window")
    expected = {str(Path(item["root"]).resolve()): item for item in request["scope"]}
    approved = grant["authorized_roots_with_actual_manifest_sha256"]
    roots = {str(Path(name).resolve()): item for name, item in approved.items()}
    if set(roots) != set(expected):
        raise ValueError("Continuous GPU authorization changed its independent output roots")
    base = (PROJ_ROOT / "outputs").resolve()
    for name, item in roots.items():
        root = Path(name)
        original = expected[name]
        expected_manifest = original.get("manifest", {}).get("path", original.get("manifest_path"))
        if (root == base or not root.is_relative_to(base)
                or Path(item["manifest_path"]).resolve() != Path(expected_manifest).resolve()):
            raise ValueError("Continuous GPU source/output binding leaves the authorized scope")
        proof = dict(path=str(Path(item["manifest_path"]).resolve()), sha256=item["manifest_sha256"])
        check_binding(proof)
        if original.get("manifest") and proof != original["manifest"]:
            raise ValueError("Continuous GPU authorization changed a prepared manifest")
    if grant.get("sports_stop_receipt") != request["sports_stop_receipt"]:
        raise ValueError("Continuous GPU grant must retain its actual Sports360 stop proof")
    check_binding(grant["sports_stop_receipt"])
    receipts = grant.get("native_exit_receipts", [])
    if (not isinstance(receipts, list) or len(receipts) < 4
            or len({(str(Path(item["path"]).resolve()), item["sha256"]) for item in receipts}) < 4):
        raise ValueError("All four native exits require their actual receipt bindings")
    for item in receipts:
        check_binding(item)
    limits = grant["maximum_stage_minutes"]
    for stage, maximum in request["requested_local_stage_lease_limits_minutes"].items():
        value = limits.get(stage)
        if type(value) is not int or not 0 < value <= maximum:
            raise ValueError("Local GPU stage duration exceeds the continuous request")
    return dict(grant=grant, binding=binding(grant_path), request=request_binding,
                roots=roots, start=start, deadline=deadline)


def make_local_stage_lease(authorization, root, stage, *, now=None):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or not authorization["start"] <= now < authorization["deadline"]:
        raise ValueError("Continuous GPU window is not currently open")
    check_binding(authorization["binding"])
    root = str(Path(root).resolve())
    if root not in authorization["roots"] or stage not in authorization["grant"]["maximum_stage_minutes"]:
        raise ValueError("The continuous grant does not authorize this root/stage")
    item = authorization["roots"][root]
    minutes = authorization["grant"]["maximum_stage_minutes"][stage]
    deadline = min(authorization["deadline"], now + timedelta(minutes=minutes))
    return dict(status="held", owner="GVHMR", root=root, stage=stage,
                manifest_sha256=item["manifest_sha256"], training_published=False,
                approved_start_utc=now.isoformat(), approved_deadline_utc=deadline.isoformat(),
                continuous_gpu_grant=authorization["binding"],
                authorization="Explicit local-stage delegation in the active continuous production grant")


def stop_owned_cuda_trees(roots):
    """Stop only known CUDA executors in these outputs, checking PID creation time."""
    roots = [Path(root).resolve() for root in roots]
    base = (PROJ_ROOT / "outputs").resolve()
    if not roots or any(root == base or not root.is_relative_to(base) for root in roots):
        raise ValueError("Deadline cleanup requires independent authorized outputs")
    targets = {}
    for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            if any(native_gpu_command_matches(process.info["cmdline"], root) for root in roots):
                for child in [*process.children(recursive=True), process]:
                    targets[child.pid] = child.create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    stopped, waiting = [], []
    for pid, created in targets.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == created:
                process.terminate()
                waiting.append(process)
                stopped.append(dict(pid=pid, create_time=created))
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(waiting, timeout=5)
    for process in alive:
        try:
            if process.create_time() == targets[process.pid]:
                process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    if alive:
        raise RuntimeError("Some owned CUDA processes did not exit at the continuous deadline")
    return stopped
