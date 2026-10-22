from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest

from hmr4d.lidar_mocap import gpu_handoff as handoff
from hmr4d.lidar_mocap.pilot import binding


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, "PROJ_ROOT", tmp_path)
    roots = [tmp_path / "outputs" / name for name in ("first", "second")]
    for root in roots:
        root.mkdir(parents=True)
    manifests = [tmp_path / f"manifest_{i}.json" for i in range(2)]
    for i, path in enumerate(manifests):
        path.write_text(json.dumps(dict(source=i)), encoding="utf-8")
    proofs = [tmp_path / f"exit_{i}.json" for i in range(5)]
    for i, path in enumerate(proofs):
        path.write_text(json.dumps(dict(actual_exit=i)), encoding="utf-8")
    request = dict(schema="gvhmr-continuous-production-gpu-request-v1", requested_maximum_hours=30,
        scope=[dict(root=str(root), manifest=binding(manifest)) for root, manifest in zip(roots, manifests)],
        sports_stop_receipt=binding(proofs[0]),
        requested_local_stage_lease_limits_minutes=dict(teacher=120, poses=240, fit=120))
    request_path, grant_path = tmp_path / "request.json", tmp_path / "grant.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    start = datetime(2026, 10, 22, 6, tzinfo=timezone.utc)
    grant = dict(schema="gvhmr-continuous-production-gpu-grant-v1", status="held", owner="GVHMR",
        request=binding(request_path), request_sha256=binding(request_path)["sha256"],
        approved_start_utc=start.isoformat(), approved_deadline_utc=(start + timedelta(hours=30)).isoformat(),
        authorized_roots_with_actual_manifest_sha256={str(root):dict(manifest_path=str(manifest),
            manifest_sha256=binding(manifest)["sha256"]) for root, manifest in zip(roots, manifests)},
        allow_local_stage_leases=True, native_gpu_returned=True, training_published=False,
        sports_stop_receipt=binding(proofs[0]), native_exit_receipts=[binding(path) for path in proofs[1:]],
        maximum_stage_minutes=dict(teacher=120, poses=240, fit=120))
    grant_path.write_text(json.dumps(grant), encoding="utf-8")
    return request_path, grant_path, roots, grant, start


def test_continuous_local_leases_remain_inside_actual_source_window(window):
    request, grant_path, roots, _, start = window
    authorization = handoff.read_continuous_grant(request, grant_path)
    lease = handoff.make_local_stage_lease(authorization, roots[0], "teacher", now=start)
    assert lease["manifest_sha256"] == binding(Path(authorization["roots"][str(roots[0])]["manifest_path"]))["sha256"]
    assert handoff.utc_time(lease["approved_deadline_utc"]) == start + timedelta(hours=2)
    deadline = authorization["deadline"]
    tail = handoff.make_local_stage_lease(authorization, roots[1], "poses", now=deadline - timedelta(minutes=15))
    assert handoff.utc_time(tail["approved_deadline_utc"]) == deadline
    for now in (start - timedelta(seconds=1), deadline, deadline + timedelta(seconds=1)):
        with pytest.raises(ValueError, match="not currently open"):
            handoff.make_local_stage_lease(authorization, roots[0], "teacher", now=now)
    with pytest.raises(ValueError, match="does not authorize"):
        handoff.make_local_stage_lease(authorization, roots[0], "unapproved", now=start)
    grant_path.write_text("changed after authorization", encoding="utf-8")
    with pytest.raises(ValueError, match="input changed"):
        handoff.make_local_stage_lease(authorization, roots[0], "teacher", now=start)


@pytest.mark.parametrize("change", ["example", "native_live", "delegation", "training", "request",
                                   "deadline", "stage", "manifest", "exit", "duplicate_exit"])
def test_incomplete_changed_and_unapproved_continuous_grants_cannot_launch(window, change):
    request, path, _, original, start = window
    grant = deepcopy(original)
    if change == "example": grant["status"] = "example_not_a_grant"
    if change == "native_live": grant["native_gpu_returned"] = False
    if change == "delegation": grant["allow_local_stage_leases"] = False
    if change == "training": grant["training_published"] = True
    if change == "request": grant["request_sha256"] = "f" * 64
    if change == "deadline": grant["approved_deadline_utc"] = (start + timedelta(hours=31)).isoformat()
    if change == "stage": grant["maximum_stage_minutes"]["teacher"] = 121
    if change == "manifest": next(iter(grant["authorized_roots_with_actual_manifest_sha256"].values()))["manifest_sha256"] = "f" * 64
    if change == "exit": Path(grant["native_exit_receipts"][0]["path"]).write_text("changed", encoding="utf-8")
    if change == "duplicate_exit": grant["native_exit_receipts"] = [grant["native_exit_receipts"][0]] * 4
    path.write_text(json.dumps(grant), encoding="utf-8")
    with pytest.raises(ValueError):
        handoff.read_continuous_grant(request, path)


def test_deadline_stops_only_actual_owned_process_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(handoff, "PROJ_ROOT", tmp_path)
    owned, foreign = [tmp_path / "outputs" / name for name in ("owned", "other")]
    for root in (owned, foreign): root.mkdir(parents=True)
    script = tmp_path / "repair_badminton_body_labels.py"
    script.write_text("import sys,time\nfrom pathlib import Path\nPath(sys.argv[-1], 'ready').write_text('ready')\ntime.sleep(60)\n", encoding="utf-8")
    children = []
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    try:
        for root in (owned, foreign):
            children.append(subprocess.Popen([sys.executable, str(script), "--root", str(root)],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"), creationflags=flags))
        deadline = time.monotonic() + 10
        while not all((root / "ready").exists() for root in (owned, foreign)):
            if time.monotonic() >= deadline: raise RuntimeError("Test CPU children did not become ready")
            time.sleep(.02)
        stopped = handoff.stop_owned_cuda_trees([owned])
        children[0].wait(timeout=5)
        assert stopped and children[1].poll() is None
    finally:
        for child in children:
            if child.poll() is None:
                try:
                    process = psutil.Process(child.pid)
                    for item in [*process.children(recursive=True), process]: item.terminate()
                except psutil.NoSuchProcess:
                    pass
                child.wait(timeout=5)
