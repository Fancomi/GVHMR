"""Exact stage contracts and real CPU-only Windows Job/watchdog fault tests."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

import psutil
import pytest

from hmr4d.lidar_mocap import owned_gpu_stage as owner
from hmr4d.lidar_mocap.pilot import binding
from hmr4d.lidar_mocap.windows_gpu_job import WindowsGPUJob


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return binding(path)


def wait_file(path, timeout=12):
    end = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= end:
            raise AssertionError("CPU fixture did not produce " + str(path))
        time.sleep(.03)
    return json.loads(path.read_bytes())


@pytest.fixture
def stage(tmp_path, monkeypatch):
    actual_root = owner.PROJ_ROOT
    repo = tmp_path / "synthetic_repo"
    root = repo / "outputs/native"
    monkeypatch.setattr(owner, "PROJ_ROOT", repo)
    for relative in ("tools/run_badminton_pilot.py", "tools/run_badminton_full_teachers.py",
                     "tools/run_badminton_full_fits.py", "hmr4d/lidar_mocap/windows_gpu_job.py",
                     "hmr4d/lidar_mocap/guarded_leg_execution.py", "hmr4d/lidar_mocap/guarded_leg_gpu.py",
                     "hmr4d/lidar_mocap/repair_execution.py"):
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(actual_root / relative, destination)
    scripts = repo / ".venv/Scripts"
    scripts.mkdir(parents=True)
    os.link(actual_root / ".venv/Scripts/python.exe", scripts / "python.exe")
    # The real venv launcher must find its existing pyvenv.cfg and packages.
    shutil.copyfile(actual_root / ".venv/pyvenv.cfg", repo / ".venv/pyvenv.cfg")
    manifest = put(repo / "input/manifest.json", dict(schema="synthetic-only"))
    source = put(root / "repair/new_source_input.json", dict(schema="synthetic-only"))
    pilot = put(root / "pilot_input.json", dict(schema="badminton-native-rally-fit-v1", review_only=True,
                                               training_published=False, manifest=manifest))
    sampling = put(root / "teacher_sampling_input.json", dict(synthetic=True))
    lease_path = root / "gpu_leases/teacher_cpu_fixture.json"
    descriptor_path = root / "authority/descriptor.json"
    mode = "success"

    def scripts_for(selected="success"):
        worker = repo / "tools/run_badminton_full_native.py"
        worker.write_text(
            "import json,os,psutil,subprocess,sys,time\nfrom pathlib import Path\n"
            f"root=Path({str(root)!r}); mode={selected!r}\n"
            "desc='--cpu-descendant' in sys.argv; p=psutil.Process()\n"
            "(root/('cpu_descendant.json' if desc else 'cpu_launcher.json')).write_text(json.dumps(dict("
            "pid=p.pid,create_time=p.create_time(),affinity=p.cpu_affinity(),priority=p.nice(),"
            "cuda=os.environ.get('CUDA_VISIBLE_DEVICES'))),encoding='utf-8')\n"
            "if not desc: subprocess.Popen([sys.executable,__file__,'--cpu-descendant'],"
            "creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)\n"
            "time.sleep(.5 if mode=='success' else (.3 if mode=='error' and not desc else 60))\n"
            "sys.exit(3 if mode=='error' and not desc else 0)\n", encoding="utf-8")
        entry = repo / "tools/run_badminton_owned_stage.py"
        entry.write_text(
            "import json,os,sys\nfrom pathlib import Path\n"
            f"sys.path.insert(0,{str(actual_root)!r})\n"
            "from hmr4d.lidar_mocap import owned_gpu_stage as m\n"
            f"m.PROJ_ROOT=Path({str(repo)!r})\n"
            "m.gpu_process_observation=lambda allowance:dict(blocking_compute_pids=[],synthetic_cpu_only=True)\n"
            "Base=m.WindowsGPUJob\n"
            "class CPUJob(Base):\n"
            " def start(self,command,*,cwd,env,log,affinity):\n"
            "  assert env['CUDA_VISIBLE_DEVICES']=='0' and affinity==[0,1]\n"
            "  env=dict(env,CUDA_VISIBLE_DEVICES='-1'); affinity=[26,27]\n"
            "  return super().start(command,cwd=cwd,env=env,log=log,affinity=affinity)\n"
            "m.WindowsGPUJob=CPUJob\n"
            + ("m._remaining=lambda checked:1.2 if '--watchdog' in sys.argv else 400.0\n"
               if selected == "deadline" else "")
            + "import runpy\n"
            + f"runpy.run_path({str(actual_root / 'tools/run_badminton_owned_stage.py')!r},run_name='__main__')\n",
            encoding="utf-8")

    scripts_for(mode)
    descriptor = dict(schema="gvhmr-owned-full-stage-descriptor-v1", root=str(root), stage="teacher",
        pilot_input=pilot, manifest=manifest, source_input=source, teacher_sampling_input=sampling,
        lease_path=str(lease_path), progress_output=str(root / "full_teacher_progress.cpu_fixture.json"),
        resume_progress=None, absolute_hard_deadline_utc=(datetime.now(timezone.utc) + timedelta(hours=3)).isoformat(),
        code={role: binding(path) for role, path in owner.implementation_files().items()})
    descriptor_binding = put(descriptor_path, descriptor)
    now = datetime.now(timezone.utc)
    desktop_path = root / "authority/desktop.json"
    desktop = dict(request=descriptor_binding, observed_utc=now.isoformat(),
                   valid_until_utc=(now + timedelta(hours=3)).isoformat(), synthetic_fixture=True)
    desktop_binding = put(desktop_path, desktop)
    lease = dict(schema="gvhmr-owned-full-stage-local-lease-v1", status="held", owner="GVHMR", root=str(root),
        stage="teacher", manifest_sha256=manifest["sha256"], owned_stage_descriptor=descriptor_binding,
        desktop_allowance=desktop_binding, desktop_allowance_fixture_only=False,
        approved_start_utc=(now - timedelta(seconds=1)).isoformat(),
        approved_deadline_utc=(now + timedelta(minutes=10)).isoformat(), training_published=False,
        cuda_visible_devices="0", max_concurrent_cuda_trees=1)
    lease_binding = put(lease_path, lease)
    monkeypatch.setattr(owner, "gpu_process_observation", lambda allowance: dict(blocking_compute_pids=[]))
    return SimpleNamespace(**locals())


def checked(stage):
    return owner.load_authority(stage.descriptor_path, binding(stage.descriptor_path)["sha256"],
                                stage.lease_path, binding(stage.lease_path)["sha256"])


@pytest.mark.parametrize("selected", ["teacher", "poses", "fit"])
def test_fixed_stage_mapping_preserves_original_cli_and_teacher_requires_sampling(stage, selected):
    descriptor = deepcopy(stage.descriptor)
    descriptor["stage"] = selected
    if selected != "teacher":
        descriptor["teacher_sampling_input"] = None
    if selected == "poses":
        descriptor["progress_output"] = None
    elif selected == "fit":
        descriptor["progress_output"] = str(stage.root / "full_fit_progress.cpu_fixture.json")
    proof = put(stage.descriptor_path, descriptor)
    stage.desktop["request"] = proof
    stage.lease.update(stage=selected, owned_stage_descriptor=proof,
                       desktop_allowance=put(stage.desktop_path, stage.desktop))
    put(stage.lease_path, stage.lease)
    value = checked(stage)
    command = owner.stage_command(value)
    assert command[:3] == [str(stage.repo / ".venv/Scripts/python.exe"), "-u",
                            str(stage.repo / "tools/run_badminton_full_native.py")]
    assert command[command.index("--stage") + 1] == selected
    assert command[command.index("--threads") + 1] == "1"
    assert ("--teacher-sampling-input" in command) == (selected == "teacher")
    assert ("--progress-output" in command) == (selected != "poses")
    assert "run_badminton_full_sequence.py" not in " ".join(command)


@pytest.mark.parametrize("change", ["command", "root", "manifest", "source", "sampling", "stage", "code",
                                    "lease_path", "progress_escape", "resume_escape", "poses_progress"])
def test_descriptor_rejects_arbitrary_command_and_changed_or_escaping_inputs(stage, change):
    value = deepcopy(stage.descriptor)
    if change == "command": value["command"] = ["cmd.exe"]
    if change == "root": value["root"] = str(stage.repo / "elsewhere")
    if change == "manifest": value["manifest"]["sha256"] = "0" * 64
    if change == "source": value["source_input"]["sha256"] = "0" * 64
    if change == "sampling": value["teacher_sampling_input"] = None
    if change == "stage": value["stage"] = "all"
    if change == "code": value["code"]["owner"]["sha256"] = "0" * 64
    if change == "lease_path": value["lease_path"] = str(stage.repo / "outside.json")
    if change == "progress_escape": value["progress_output"] = str(stage.repo / "outside.json")
    if change == "resume_escape": value["resume_progress"] = put(stage.repo / "outside.json", {})
    if change == "poses_progress": value.update(stage="poses", teacher_sampling_input=None)
    proof = put(stage.descriptor_path, value)
    with pytest.raises((ValueError, FileNotFoundError)):
        owner.inspect_descriptor(proof["path"], proof["sha256"])


@pytest.mark.parametrize("change", ["expired", "over_120", "future", "root", "stage", "descriptor", "manifest",
                                    "cuda", "concurrent", "fixture", "desktop_stale", "desktop_short",
                                    "desktop_request"])
def test_exact_local_lease_and_fresh_desktop_fail_closed_without_continuous_grant(stage, change):
    lease = deepcopy(stage.lease)
    now = datetime.now(timezone.utc)
    if change == "expired": lease["approved_deadline_utc"] = (now - timedelta(seconds=1)).isoformat()
    if change == "over_120":
        lease["approved_deadline_utc"] = (now + timedelta(minutes=121)).isoformat()
    if change == "future": lease["approved_start_utc"] = (now + timedelta(seconds=30)).isoformat()
    if change == "root": lease["root"] = str(stage.repo)
    if change == "stage": lease["stage"] = "fit"
    if change == "descriptor": lease["owned_stage_descriptor"]["sha256"] = "0" * 64
    if change == "manifest": lease["manifest_sha256"] = "0" * 64
    if change == "cuda": lease["cuda_visible_devices"] = "1"
    if change == "concurrent": lease["max_concurrent_cuda_trees"] = 2
    if change == "fixture": lease["desktop_allowance_fixture_only"] = True
    if change.startswith("desktop_"):
        desktop = deepcopy(stage.desktop)
        if change == "desktop_stale": desktop["observed_utc"] = (now - timedelta(minutes=16)).isoformat()
        if change == "desktop_short": desktop["valid_until_utc"] = now.isoformat()
        if change == "desktop_request": desktop["request"]["sha256"] = "0" * 64
        lease["desktop_allowance"] = put(stage.desktop_path, desktop)
    put(stage.lease_path, lease)
    with pytest.raises(ValueError): checked(stage)


def test_desktop_freshness_is_at_admission_and_valid_until_still_covers_full_stage(stage):
    desktop = deepcopy(stage.desktop)
    desktop["observed_utc"] = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    stage.lease["desktop_allowance"] = put(stage.desktop_path, desktop)
    proof = put(stage.lease_path, stage.lease)
    assert owner.load_authority(stage.descriptor_path, stage.descriptor_binding["sha256"],
                                stage.lease_path, proof["sha256"], require_fresh=False)["desktop"]
    with pytest.raises(ValueError): checked(stage)


def test_foreign_gpu_blocks_before_lock_or_job_and_is_never_stopped(stage, monkeypatch):
    monkeypatch.setattr(owner, "gpu_process_observation", lambda allowance: dict(blocking_compute_pids=[os.getpid()]))
    with pytest.raises(RuntimeError, match="Foreign CUDA"):
        owner.execute(stage.descriptor_path, stage.descriptor_binding["sha256"],
                      stage.lease_path, stage.lease_binding["sha256"])
    assert not (stage.root / "owned_gpu_stages").exists()


def launch(stage, mode):
    stage.scripts_for(mode)
    stage.descriptor["code"] = {role: binding(path) for role, path in owner.implementation_files().items()}
    descriptor = put(stage.descriptor_path, stage.descriptor)
    stage.desktop["request"] = descriptor
    stage.lease["desktop_allowance"] = put(stage.desktop_path, stage.desktop)
    stage.lease["owned_stage_descriptor"] = descriptor
    lease = put(stage.lease_path, stage.lease)
    command = [sys.executable, "-u", str(stage.repo / "tools/run_badminton_owned_stage.py"), "--execute",
               "--descriptor", descriptor["path"], "--descriptor-sha256", descriptor["sha256"],
               "--lease", lease["path"], "--lease-sha256", lease["sha256"]]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1",
                       PYTHONPATH=str(stage.actual_root) + ";" + str(stage.actual_root / ".venv/Lib/site-packages"))
    log = (stage.root / "fixture_controller.log").open("ab")
    child = subprocess.Popen(command, cwd=stage.actual_root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    log.close()
    return child


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows Job integration")
@pytest.mark.parametrize("mode", ["success", "error", "deadline", "controller_loss", "watchdog_loss"])
def test_actual_windows_watchdog_and_inherited_cpu_descendants_stop_only_the_owned_tree(stage, mode):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    sentinel_script = stage.repo / "sentinel.py"
    sentinel_script.write_text("import time;time.sleep(60)", encoding="utf-8")
    with WindowsGPUJob("Local\\OwnedStageForeignCPU_" + uuid.uuid4().hex) as foreign:
        with (stage.root / "sentinel.log").open("ab") as log:
            sentinel, _ = foreign.start([sys.executable, str(sentinel_script)], cwd=stage.actual_root,
                env=environment, log=log, affinity=[26, 27])
        controller = launch(stage, mode)
        watcher = None
        try:
            launcher = wait_file(stage.root / "cpu_launcher.json")
            descendant = wait_file(stage.root / "cpu_descendant.json")
            assert all(item["cuda"] == "-1" and item["affinity"] == [26, 27]
                       and item["priority"] == psutil.BELOW_NORMAL_PRIORITY_CLASS for item in (launcher, descendant))
            lock = wait_file(stage.root / "owned_gpu_stages/execution.lock.json")
            directory = stage.root / "owned_gpu_stages" / lock["nonce"]
            context = wait_file(directory / "context.json")
            watcher = WindowsGPUJob(context["job_name"], create=False)
            assert wait_file(directory / "launch.json")["assigned_before_resume"]
            if mode == "controller_loss":
                controller.terminate()  # Original test Popen handle; foreign sentinel is unaffected.
            if mode == "watchdog_loss":
                ready = wait_file(directory / "watchdog_ready.json")
                process = psutil.Process(ready["pid"])
                assert process.create_time() == ready["create_time"]
                process.terminate()  # Exact private fixture watchdog; its controller must reap the owned Job.
            controller.wait(timeout=15)
            if mode == "success":
                release = wait_file(directory / "release_receipt.json")
                assert release["job_empty"] and release["status"] == "released"
                assert controller.returncode == 0
                with pytest.raises(ValueError, match="already attempted"):
                    owner.execute(stage.descriptor_path, binding(stage.descriptor_path)["sha256"],
                                  stage.lease_path, binding(stage.lease_path)["sha256"])
            else:
                if mode == "controller_loss": wait_file(directory / "watchdog_failure.json")
                else: wait_file(directory / "failure.json")
                assert controller.returncode != 0
            watcher.wait_empty(timeout=5)
            ready = wait_file(directory / "watchdog_ready.json")
            end = time.monotonic() + 5
            while owner.identity_alive(ready) and time.monotonic() < end:
                time.sleep(.03)
            assert not owner.identity_alive(ready)
            assert sentinel.poll() is None and foreign.pids()
        finally:
            if controller.poll() is None: controller.terminate()
            controller.wait(timeout=5)
            if watcher is not None:
                watcher.terminate()
                watcher.wait_empty()
                watcher.close()
            foreign.terminate()
            foreign.wait_empty()
            sentinel.wait(timeout=5)
