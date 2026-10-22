"""Real Windows process tests with CUDA disabled and no production process access."""

import json
import os
from pathlib import Path
import sys
import time
import uuid

import psutil
import pytest

from hmr4d.lidar_mocap.windows_gpu_job import WindowsGPUJob

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows Job Object integration")


def wait_for(path):
    end = time.monotonic() + 8
    while not path.exists():
        if time.monotonic() > end:
            raise RuntimeError("Own CPU test child did not start")
        time.sleep(.02)
    return json.loads(path.read_text(encoding="utf-8"))


def probe(tmp_path):
    path = tmp_path / "cpu_probe.py"
    path.write_text("import json,os,psutil,subprocess,sys,time\nfrom pathlib import Path\n"
        "target=Path(sys.argv[1]); p=psutil.Process()\n"
        "target.write_text(json.dumps(dict(pid=p.pid,create_time=p.create_time(),affinity=p.cpu_affinity(),"
        "priority=p.nice(),cuda=os.environ.get('CUDA_VISIBLE_DEVICES'))),encoding='utf-8')\n"
        "if len(sys.argv)>2:\n"
        " subprocess.Popen([sys.executable,__file__,sys.argv[2]],creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)\n"
        "time.sleep(60)\n", encoding="utf-8")
    return path


def test_suspended_owned_venv_launcher_assigns_descendants_before_resume(tmp_path):
    script = probe(tmp_path)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    executable = Path(__file__).resolve().parents[1] / ".venv/Scripts/python.exe"
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1")
    with WindowsGPUJob("Local\\GVHMRGuardTest_" + uuid.uuid4().hex) as job:
        with (tmp_path / "probe.log").open("ab") as log:
            child, launched = job.start([str(executable), "-u", str(script), str(first), str(second)],
                cwd=tmp_path, env=environment, log=log, affinity=[26, 27])
        try:
            records = [wait_for(first), wait_for(second)]
            assert launched["assigned_before_resume"]
            assert all(r["pid"] in job.pids() for r in records)
            assert all(r["affinity"] == [26, 27] and r["cuda"] == "-1" for r in records)
            assert all(r["priority"] == psutil.BELOW_NORMAL_PRIORITY_CLASS for r in records)
            with pytest.raises(OSError): WindowsGPUJob(job.name)
            with WindowsGPUJob(job.name, create=False) as watcher:
                assert set(watcher.pids()) == set(job.pids())
                watcher.terminate()
                watcher.wait_empty()
            child.wait(timeout=5)
            assert not job.pids()
        finally:
            job.terminate()
            job.wait_empty()
            child.wait(timeout=5)


def test_deadline_kills_only_private_cpu_job_even_when_controller_is_blocked(tmp_path):
    script = probe(tmp_path)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    with WindowsGPUJob("Local\\GVHMRGuardForeignTest_" + uuid.uuid4().hex) as foreign, WindowsGPUJob(
            "Local\\GVHMRGuardDeadlineTest_" + uuid.uuid4().hex) as owned:
        with (tmp_path / "jobs.log").open("ab") as log:
            other, _ = foreign.start([sys.executable, str(script), str(tmp_path / "other.json")],
                cwd=tmp_path, env=environment, log=log, affinity=[26, 27])
            child, _ = owned.start([sys.executable, str(script), str(tmp_path / "own.json")],
                cwd=tmp_path, env=environment, log=log, affinity=[26, 27])
        try:
            wait_for(tmp_path / "other.json")
            wait_for(tmp_path / "own.json")
            with owned.deadline_guard(.25) as state:
                time.sleep(.7)  # Simulate blocking I/O in the main controller.
                assert state["fired"] and state["error"] is None
                owned.wait_empty()
                child.wait(timeout=5)
                assert foreign.pids() and other.poll() is None
        finally:
            owned.terminate()
            foreign.terminate()
            owned.wait_empty()
            foreign.wait_empty()
            child.wait(timeout=5)
            other.wait(timeout=5)


def test_last_handle_close_kills_descendants_without_controller_cleanup(tmp_path):
    script = probe(tmp_path)
    job = WindowsGPUJob("Local\\GVHMRGuardCloseTest_" + uuid.uuid4().hex)
    with (tmp_path / "close.log").open("ab") as log:
        child, _ = job.start([sys.executable, str(script), str(tmp_path / "close.json")], cwd=tmp_path,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"), log=log, affinity=[26, 27])
    try:
        wait_for(tmp_path / "close.json")
        job.close()
        child.wait(timeout=5)
        assert child.returncode is not None
    finally:
        job.close()
        child.wait(timeout=5)
