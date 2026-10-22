"""Real CPU Windows Job faults; the active production CUDA tree is untouched."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from hmr4d.lidar_mocap import identity_continuation as continuation
from hmr4d.lidar_mocap.windows_gpu_job import WindowsGPUJob
from hmr4d.lidar_mocap import windows_gpu_job
from test_lidar_gpu_handoff import window
from test_lidar_identity_continuation import complete, put


def wait_file(path):
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError("Synthetic CPU child did not become ready")
        time.sleep(0.02)
    return json.loads(path.read_bytes())


@pytest.mark.skipif(os.name != "nt", reason="Windows private Job ownership")
@pytest.mark.parametrize(
    "fault",
    ["ctime", "state_write", "persistent_state_write", "observer_popen", "observer_identity", "observer_update"],
)
def test_actual_spawn_fault_reaps_only_its_CPU_job_subtrees(complete, monkeypatch, fault):
    # These are real processes and Job objects. Only the model command is
    # replaced by a CPU sleeper; Job.start and continuation.spawn stay real.
    for path in (
        complete.root / "gpu_leases/identity_poses_01.json",
        complete.root / "identity_poses_01_release_receipt.json",
    ):
        path.unlink()
    put(complete.request.parent / "live_progress.json", dict(state="waiting_for_actual_reviewed_identity"))
    controller = continuation.IdentityContinuation(complete.root, complete.request, complete.grant)
    script = complete.project / "observe_badminton_repair_poses.py"
    script.write_text(
        "import json,os,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "import psutil\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'\n"
        "if '--leaf' not in sys.argv:\n"
        "    info={'pid':os.getpid(),'create_time':psutil.Process().create_time()}\n"
        "    if '--foreign' not in sys.argv:\n"
        "        child=subprocess.Popen([sys.executable,__file__,'--leaf'],creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)\n"
        "        info['leaf_pid']=child.pid\n"
        "        info['leaf_create_time']=psutil.Process(child.pid).create_time()\n"
        "    Path(sys.argv[1]).write_text(json.dumps(info))\n"
        "time.sleep(90)\n",
        encoding="utf-8",
    )
    own_ready = complete.project / "own-ready.json"
    foreign_ready = complete.project / "foreign-ready.json"
    original_popen = subprocess.Popen
    original_process = psutil.Process
    original_start = WindowsGPUJob.start
    original_writer = continuation.write_progress_json
    children, foreign, leaf_ids = [], [], []
    injected = [False]

    def CPU_job_start(self, command, **kwargs):
        kwargs["env"] = dict(kwargs["env"], CUDA_VISIBLE_DEVICES="-1")
        kwargs["affinity"] = [26, 27]
        return original_start(self, command, **kwargs)

    def popen(command, **kwargs):
        is_observer = any(Path(arg).name == "release_badminton_full_gpu.py" for arg in command)
        if is_observer and fault == "observer_popen":
            raise OSError("synthetic observer Popen failure")
        child = original_popen(command, **kwargs)
        children.append((child, is_observer))
        if not is_observer:
            # Created after the controller's predecessor scan. This unrelated
            # sibling deliberately names the same authorized root and script;
            # cleanup by a broad root predicate would incorrectly kill it.
            sibling = original_popen(
                [sys.executable, str(script), str(foreign_ready), "--foreign", "--root", str(complete.root)],
                cwd=complete.project,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"),
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS,
            )
            foreign.append(sibling)
            wait_file(foreign_ready)
        return child

    def process(pid=None):
        if pid is not None and children and pid == children[-1][0].pid and not injected[0]:
            if (fault == "ctime" and not children[-1][1]) or (fault == "observer_identity" and children[-1][1]):
                injected[0] = True
                raise psutil.AccessDenied(pid)
        return original_process(pid)

    def writer(path, value, **kwargs):
        if path == controller.state_path:
            stage = value.get("stage")
            should_fail = fault in ("state_write", "persistent_state_write") and stage == "identity_native_observations"
            should_fail |= fault == "observer_update" and stage == "identity_cuda_release"
            if (should_fail and not injected[0]) or (fault == "persistent_state_write" and injected[0]):
                if not injected[0]:
                    ready = wait_file(own_ready)
                    leaf_ids.append((ready["leaf_pid"], ready["leaf_create_time"]))
                injected[0] = True
                raise PermissionError("synthetic state persistence failure")
        return original_writer(path, value, **kwargs)

    def command(root, role, threads, lease=None):
        assert role == "observe"
        return [sys.executable, str(script), str(own_ready), "--root", str(root)]

    monkeypatch.setattr(WindowsGPUJob, "start", CPU_job_start)
    monkeypatch.setattr(continuation, "worker_command", command)
    monkeypatch.setattr(continuation.subprocess, "Popen", popen)
    view = SimpleNamespace(
        Process=process,
        process_iter=psutil.process_iter,
        BELOW_NORMAL_PRIORITY_CLASS=psutil.BELOW_NORMAL_PRIORITY_CLASS,
        NoSuchProcess=psutil.NoSuchProcess,
        AccessDenied=psutil.AccessDenied,
    )
    monkeypatch.setattr(continuation, "psutil", view)
    monkeypatch.setattr(windows_gpu_job, "psutil", view)
    monkeypatch.setattr(continuation, "write_progress_json", writer)
    try:
        with pytest.raises((psutil.AccessDenied, PermissionError, OSError)):
            controller.run()
        assert children and all(child.poll() is not None for child, _ in children)
        assert foreign and foreign[0].poll() is None
        assert all(item["closed"] for item in controller.owned)
        assert all(item["child"] is None or item["child"].returncode is not None for item in controller.owned)
        if own_ready.exists():
            item = json.loads(own_ready.read_bytes())
            leaf_ids.append((item["leaf_pid"], item["leaf_create_time"]))
        for pid, created in set(leaf_ids):
            try:
                assert original_process(pid).create_time() != created
            except psutil.NoSuchProcess:
                pass
        failures = list((complete.evidence / "continuation_failures").glob("*.json"))
        assert failures
        records = [json.loads(path.read_bytes()) for path in failures]
        assert not any(record["cleanup_errors"] for record in records)
        assert all(job["actual_job_empty"] for record in records for job in record["owned_jobs"])
        assert any(record["owned_jobs"] for record in records)
        assert controller.state.get("failure_evidence")
        assert not controller.release_path.exists()
    finally:
        # Test-only cleanup uses the exact retained Popen handles; it never
        # scans or terminates a production root or an unrelated PID by name.
        for owned in controller.owned:
            if not owned["closed"]:
                owned["job"].terminate()
                owned["job"].wait_empty()
                owned["job"].close()
        for child in [*(child for child, _ in children), *foreign]:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=10)
