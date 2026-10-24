"""CPU metadata and real Windows Job regressions; no model/GPU/data admission.

The subprocess fixtures inject only a fake GPU observation leaf. The real Job,
independent watchdog core, process identities, cleanup and deadlines execute.
Synthetic metadata compatibility is separate from these actual CPU lifecycles.
"""

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import psutil
import pytest

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap import incremental_guard_execution as execution
from hmr4d.lidar_mocap.guarded_leg_gpu import identity_alive
from hmr4d.lidar_mocap.pilot import binding
from hmr4d.lidar_mocap.windows_gpu_job import WindowsGPUJob

CPU = [18, 19]
WINDOWS = pytest.mark.skipif(os.name != "nt", reason="Real Windows Job Object regression")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return binding(path)


def read(path):
    return json.loads(Path(path).read_bytes())


def env():
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="-1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(PROJ_ROOT),
        PYTHONUTF8="1",
    )


def checked(root):
    return dict(
        root=root,
        deadline=datetime.now(timezone.utc) + timedelta(minutes=2),
        request={"minimum_remaining_launch_seconds": 0},
    )


@contextmanager
def cpu_process(path, code, *, arguments=()):
    """Own every fixture process before it can execute, just like a real phase."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with WindowsGPUJob("Local\\GVHMRIncrementalCPUFixture_" + uuid.uuid4().hex) as job:
        with path.open("wb") as log:
            command = [sys.executable, "-B", "-c", code, *map(str, arguments)]
            process, launched = job.start(command, cwd=PROJ_ROOT, env=env(), log=log, affinity=CPU)
            try:
                yield process, launched, job
            finally:
                job.terminate()
                job.wait_empty()
                process.wait(timeout=6)


def wait_file(path, process=None, seconds=12):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if process is not None and process.poll() is not None:
            pytest.fail("Fixture process exited before " + str(path))
        if time.monotonic() >= deadline:
            pytest.fail("Bounded wait expired for " + str(path))
        time.sleep(0.03)
    return read(path)


@pytest.fixture
def context_case(tmp_path, monkeypatch):
    root = tmp_path / "formal" / "repair"
    folder = root.parent / "gpu_guarded_leg"
    directory = folder / "attempts" / ("a" * 32)
    directory.mkdir(parents=True)
    request = put(folder / "request.json", {"fixture_only": True})
    grant = put(folder / "grant.json", {"fixture_only": True})
    descriptor = put(folder / "descriptor.json", {"fixture_only": True})
    implementation = put(folder / "implementation.json", {"fixture_only": True, "new_driver": True})
    preflight = put(directory / "preflight.json", {"fixture_only": True})
    start = datetime.now(timezone.utc)
    lease = put(
        folder / "guarded_leg_stage_lease.json",
        dict(
            status="held",
            owner="GVHMR",
            root=str(root),
            stage="guarded_leg",
            manifest_sha256="a" * 64,
            continuous_gpu_grant=grant,
            approved_start_utc=start.isoformat(),
            approved_deadline_utc=(start + timedelta(minutes=2)).isoformat(),
            authorization_profile=execution.PROFILE,
            incremental_descriptor=descriptor,
        ),
    )
    stage = checked(root)
    stage.update(
        authorization_profile=execution.PROFILE,
        request_binding=request,
        descriptor_binding=descriptor,
        grant_path=Path(grant["path"]),
        request={"minimum_remaining_launch_seconds": 0, "scope": [{"manifest": {"sha256": "a" * 64}}]},
        authorization=dict(
            binding=grant,
            start=start,
            deadline=start + timedelta(minutes=2),
            grant={"independent_guard": implementation, "maximum_stage_minutes": {"guarded_leg": 30}},
        ),
    )
    stage["deadline"] = start + timedelta(minutes=2)
    command = [
        sys.executable,
        "-u",
        str(execution.ENTRY),
        "--execute",
        "--request",
        request["path"],
        "--grant",
        grant["path"],
        "--descriptor",
        descriptor["path"],
        "--descriptor-sha256",
        descriptor["sha256"],
    ]
    own = psutil.Process()
    context = dict(
        schema=execution.CONTEXT_SCHEMA,
        authorization_profile=execution.PROFILE,
        nonce=directory.name,
        root=str(root),
        job_name="Local\\GVHMRGuardedLeg_" + directory.name,
        watchdog_job_name="Local\\GVHMRIncrementalWatchdog_" + directory.name,
        driver=dict(pid=own.pid, create_time=own.create_time(), command=command),
        request=request,
        grant=grant,
        descriptor=descriptor,
        lease=lease,
        preflight=preflight,
        implementation=implementation,
        watchdog_entry=binding(execution.ENTRY),
        immutable_bindings=[descriptor],
    )
    path = directory / "context.json"
    put(path, context)
    monkeypatch.setattr(execution, "immutable_anchors", lambda value: [descriptor])
    return SimpleNamespace(root=root, folder=folder, directory=directory, path=path, context=context, checked=stage)


def test_context_pins_new_driver_descriptor_and_real_implementation(context_case, monkeypatch):
    value = context_case
    monkeypatch.setattr(execution, "_driver_alive", lambda value: True)
    execution.validate_context(value.context, value.path, value.checked)
    assert value.context["watchdog_entry"] == binding(execution.ENTRY)
    assert "incremental" in value.context["driver"]["command"][2]


@pytest.mark.parametrize(
    "change",
    [
        "old_entry",
        "descriptor_path",
        "descriptor_sha",
        "nonce",
        "job",
        "watchdog_job",
        "profile",
        "implementation",
        "anchors",
        "request",
        "lease_descriptor",
        "lease_deadline",
        "extra_context",
    ],
)
def test_context_mutation_never_reaches_job_or_gpu(context_case, monkeypatch, change):
    value = context_case
    context = deepcopy(value.context)
    monkeypatch.setattr(execution, "_driver_alive", lambda value: True)
    if change == "old_entry":
        context["driver"]["command"][2] = str(PROJ_ROOT / "tools/run_badminton_guarded_leg.py")
    elif change == "descriptor_path":
        context["driver"]["command"][9] = str(value.folder / "alias.json")
    elif change == "descriptor_sha":
        context["driver"]["command"][11] = "f" * 64
    elif change == "nonce":
        context["nonce"] = "b" * 32
    elif change == "job":
        context["job_name"] += "foreign"
    elif change == "watchdog_job":
        context["watchdog_job_name"] += "foreign"
    elif change == "profile":
        context["authorization_profile"] = "legacy"
    elif change == "implementation":
        context["implementation"] = context["request"]
    elif change == "anchors":
        context["immutable_bindings"] = []
    elif change == "request":
        context["request"] = context["descriptor"]
    elif change in ("lease_descriptor", "lease_deadline"):
        path = Path(context["lease"]["path"])
        lease = read(path)
        if change == "lease_descriptor":
            lease["incremental_descriptor"] = context["request"]
        else:
            lease["approved_deadline_utc"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        context["lease"] = put(path, lease)
    else:
        context["allow_unbound_driver"] = True
    monkeypatch.setattr(execution, "WindowsGPUJob", lambda *a, **k: pytest.fail("Job must not be opened"))
    monkeypatch.setattr(execution, "_gpu_snapshot", lambda value: pytest.fail("GPU must not be queried"))
    with pytest.raises(ValueError):
        execution.validate_context(context, value.path, value.checked)
    assert not (value.directory / "completion.json").exists()


@pytest.mark.parametrize("change", ["old_entry", "context", "sha", "extra"])
def test_watchdog_command_pins_its_own_entry_and_exact_context(context_case, change):
    value = context_case
    digest = binding(value.path)["sha256"]
    command = [
        sys.executable,
        "-u",
        str(execution.ENTRY),
        "--watchdog",
        "--context",
        str(value.path),
        "--context-sha256",
        digest,
    ]
    assert execution._watchdog_command(command, value.path, digest)
    if change == "old_entry":
        command[2] = str(PROJ_ROOT / "tools/run_badminton_guarded_leg.py")
    elif change == "context":
        command[5] = str(value.directory / "alias.json")
    elif change == "sha":
        command[7] = "f" * 64
    else:
        command += ["--unbound", "true"]
    assert not execution._watchdog_command(command, value.path, digest)


def test_missing_authority_has_no_gpu_or_lease_side_effect(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "cpu_only", lambda: None)
    monkeypatch.setattr(execution, "_gpu_snapshot", lambda value: pytest.fail("No GPU probe before real authority"))
    with pytest.raises((FileNotFoundError, ValueError)):
        execution.execute(
            tmp_path / "not-issued.json",
            tmp_path / "grant.json",
            descriptor_path=tmp_path / "descriptor.json",
            descriptor_sha256="a" * 64,
        )
    assert not (tmp_path / "guarded_leg_stage_lease.json").exists()
    assert not (tmp_path / "execution.lock.json").exists()


def test_desktop_startup_freshness_does_not_shorten_an_issued_absolute_window(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    request = put(tmp_path / "request.json", {"fixture_only": True})
    allowance = put(
        tmp_path / "desktop.json",
        dict(
            request=request,
            observed_utc=(now - timedelta(minutes=16)).isoformat(),
            valid_until_utc=(now + timedelta(minutes=10)).isoformat(),
        ),
    )
    value = dict(
        request_binding=request,
        authorization=dict(
            deadline=now + timedelta(minutes=5),
            grant=dict(
                desktop_allowance=allowance,
                desktop_allowance_fixture_only=False,
            ),
        ),
    )
    with pytest.raises(ValueError, match="fresh"):
        execution._desktop_allowance(value)
    monkeypatch.setattr(execution, "gpu_process_observation", lambda item: dict(binding=item, fixture_only=True))
    assert execution._gpu_snapshot(value)["binding"] == allowance
    assert execution._gpu_snapshot(value)["fixture_only"] is True


def test_cli_requires_explicit_descriptor_before_allocation_or_authority(tmp_path):
    result = subprocess.run(
        [sys.executable, "-B", str(execution.ENTRY), "--preflight", "--request", "missing", "--grant", "missing"],
        cwd=tmp_path,
        env=env(),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 2
    assert "descriptor path/SHA256 pair" in result.stderr
    assert not list(tmp_path.iterdir())


@WINDOWS
def test_real_cpu_phase_is_assigned_before_resume_and_has_exact_exit(tmp_path):
    marker = tmp_path / "worker.json"
    code = "import json,os,psutil,sys;json.dump({'cuda':os.environ['CUDA_VISIBLE_DEVICES'],'affinity':psutil.Process().cpu_affinity()},open(sys.argv[1],'w'))"
    with cpu_process(tmp_path / "guard.log", "import time;time.sleep(30)") as (guard, ready, _):
        with WindowsGPUJob("Local\\GVHMRIncrementalPhase_" + uuid.uuid4().hex) as job:
            observed = {}
            launched = execution._run_owned_command(
                job,
                [sys.executable, "-B", "-c", code, str(marker)],
                checked(tmp_path),
                {
                    key: put(tmp_path / (key + ".json"), {"fixture_only": True})
                    for key in (
                        "request",
                        "grant",
                        "lease",
                        "preflight",
                        "implementation",
                        "descriptor",
                        "watchdog_entry",
                    )
                },
                tmp_path,
                guard,
                ready,
                observed,
                phase="cpu_fixture",
                env=env(),
                affinity=CPU,
            )
            assert launched["assigned_before_resume"] is True
            assert job.pids() == []
            assert not identity_alive(launched)
            assert (launched["pid"], launched["create_time"]) in observed
            assert read(marker) == {"cuda": "-1", "affinity": CPU}


@WINDOWS
@pytest.mark.parametrize("failure", ["nonzero", "guard_exit", "deadline", "persistence"])
def test_real_phase_failure_cleans_only_its_job_without_completion(tmp_path, monkeypatch, failure):
    context = {
        key: put(tmp_path / (key + ".json"), {"fixture_only": True})
        for key in ("request", "grant", "lease", "preflight", "implementation", "descriptor", "watchdog_entry")
    }
    with cpu_process(tmp_path / "foreign.log", "import time;time.sleep(30)") as (sentinel, _, _):
        guard_code = "import time;time.sleep(0.15)" if failure == "guard_exit" else "import time;time.sleep(30)"
        with cpu_process(tmp_path / "guard.log", guard_code) as (guard, ready, _):
            stage = checked(tmp_path)
            if failure == "deadline":
                stage["deadline"] = datetime.now(timezone.utc) + timedelta(seconds=10.2)
            if failure == "persistence":
                writer = execution.write_progress_json

                def fail_launch(path, value):
                    if path.name.endswith(".launch.json"):
                        raise OSError("synthetic persistence failure after real spawn")
                    return writer(path, value)

                monkeypatch.setattr(execution, "write_progress_json", fail_launch)
            code = "raise SystemExit(7)" if failure == "nonzero" else "import time;time.sleep(30)"
            with WindowsGPUJob("Local\\GVHMRIncrementalFailure_" + uuid.uuid4().hex) as job:
                with pytest.raises((RuntimeError, TimeoutError, OSError)):
                    execution._run_owned_command(
                        job,
                        [sys.executable, "-B", "-c", code],
                        stage,
                        context,
                        tmp_path,
                        guard,
                        ready,
                        {},
                        phase="cpu_fixture",
                        env=env(),
                        affinity=CPU,
                    )
                assert job.pids() == []
                assert sentinel.poll() is None
            if failure == "nonzero":
                assert read(tmp_path / "cpu_fixture.failure.json")["actual_exit_code"] == 7
            assert not (tmp_path / "completion.json").exists()


@WINDOWS
def test_real_job_timer_cleans_descendants_and_preserves_foreign_sentinel(tmp_path):
    marker = tmp_path / "descendant.json"
    code = (
        "import json,psutil,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)']);"
        "json.dump({'pid':p.pid,'create_time':psutil.Process(p.pid).create_time()},open(sys.argv[1],'w'));time.sleep(30)"
    )
    with cpu_process(tmp_path / "foreign.log", "import time;time.sleep(30)") as (sentinel, _, _):
        with WindowsGPUJob("Local\\GVHMRIncrementalTimer_" + uuid.uuid4().hex) as job:
            with (tmp_path / "owned.log").open("wb") as log:
                child, launched = job.start(
                    [sys.executable, "-B", "-c", code, str(marker)], cwd=PROJ_ROOT, env=env(), log=log, affinity=CPU
                )
                descendant = wait_file(marker, child)
                with job.deadline_guard(0.2) as timer:
                    child.wait(timeout=6)
                    job.wait_empty()
                    assert timer["fired"] is True
                assert not identity_alive(launched)
                # A terminated Windows process object can remain briefly while
                # its last inherited handle closes. Check the original identity,
                # never assume a still-present/reused PID denotes a live child.
                end = time.monotonic() + 3
                while identity_alive(descendant) and time.monotonic() < end:
                    time.sleep(0.03)
                assert not identity_alive(descendant)
                assert sentinel.poll() is None


@WINDOWS
def test_watchdog_launcher_exit_cannot_leave_its_cpu_descendant(tmp_path):
    marker = tmp_path / "orphan.json"
    code = (
        "import json,psutil,subprocess,sys;"
        "p=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)']);"
        "json.dump({'pid':p.pid,'create_time':psutil.Process(p.pid).create_time()},open(sys.argv[1],'w'))"
    )
    with cpu_process(tmp_path / "foreign.log", "import time;time.sleep(30)") as (sentinel, _, _):
        with WindowsGPUJob("Local\\GVHMRIncrementalWatchOrphan_" + uuid.uuid4().hex) as job:
            with (tmp_path / "guard.log").open("wb") as log:
                guard, _ = job.start(
                    [sys.executable, "-B", "-c", code, str(marker)], cwd=PROJ_ROOT, env=env(), log=log, affinity=CPU
                )
                original = wait_file(marker, guard)
                assert guard.wait(timeout=6) == 0
                assert job.pids()
                execution._wait_guard(guard, job=job, require_clean=False)
                assert job.pids() == []
                end = time.monotonic() + 3
                while identity_alive(original) and time.monotonic() < end:
                    time.sleep(0.03)
                assert not identity_alive(original)
                assert sentinel.poll() is None


def watchdog_fixture(tmp_path, worker_job, controller, *, foreign_pid=None):
    """A local CPU fixture, not a request/grant or a production readiness proof."""
    directory = tmp_path / "watch"
    directory.mkdir()
    context = {
        key: put(directory / (key + ".json"), {"fixture_only": True})
        for key in ("request", "grant", "lease", "preflight", "implementation", "descriptor", "watchdog_entry")
    }
    context.update(
        driver=dict(
            pid=controller.pid,
            create_time=psutil.Process(controller.pid).create_time(),
            command=psutil.Process(controller.pid).cmdline(),
        ),
        job_name=worker_job.name,
        immutable_bindings=[context["descriptor"]],
    )
    path = directory / "context.json"
    put(path, context)
    data = put(
        directory / "fixture.json",
        dict(
            context=str(path),
            deadline=(datetime.now(timezone.utc) + timedelta(minutes=2)).isoformat(),
            blocking=[] if foreign_pid is None else [foreign_pid],
        ),
    )
    code = (
        "import json,sys;from datetime import datetime;"
        "from pathlib import Path;from hmr4d.lidar_mocap import incremental_guard_execution as m;"
        "v=json.load(open(sys.argv[1]));p=Path(v['context']);c=json.loads(p.read_bytes());"
        "m._gpu_snapshot=lambda checked:{'blocking_compute_pids':v['blocking'],'fixture_only':True};"
        "m._watch_owned(c,p,{'deadline':datetime.fromisoformat(v['deadline']),"
        "'request':{'minimum_remaining_launch_seconds':0}})"
    )
    return SimpleNamespace(directory=directory, context=context, path=path, code=code, arguments=[data["path"]])


@WINDOWS
def test_real_independent_cpu_watchdog_completes_only_after_owned_job_empty(tmp_path):
    with cpu_process(tmp_path / "controller.log", "import time;time.sleep(30)") as (controller, _, _):
        with WindowsGPUJob("Local\\GVHMRIncrementalWatch_" + uuid.uuid4().hex) as worker_job:
            fixture = watchdog_fixture(tmp_path, worker_job, controller)
            with cpu_process(tmp_path / "watchdog.log", fixture.code, arguments=fixture.arguments) as (guard, _, _):
                ready = wait_file(fixture.directory / "guard_ready.json", guard)
                assert ready["actual_affinity"] == CPU
                put(
                    fixture.directory / "all_phases_complete.json",
                    dict(
                        phases={phase: {"fixture_only": True} for phase in ("propose", "select", "apply")},
                        context=binding(fixture.path),
                    ),
                )
                assert guard.wait(timeout=12) == 0
                result = read(fixture.directory / "actual_cuda_exit.json")
                assert result["job_empty"] is True and result["gpu0_compute_pids_empty"] is True
                assert result["gpu0_process_classification"]["fixture_only"] is True
                assert worker_job.pids() == []
                assert controller.poll() is None


@WINDOWS
@pytest.mark.parametrize("failure", ["foreign_owner", "controller_exit", "anchor_mutation", "live_completion"])
def test_real_independent_watchdog_failure_closes_owned_tree_only(tmp_path, failure):
    with cpu_process(tmp_path / "foreign.log", "import time;time.sleep(30)") as (sentinel, _, _):
        with cpu_process(tmp_path / "controller.log", "import time;time.sleep(30)") as (controller, _, _):
            with WindowsGPUJob("Local\\GVHMRIncrementalWatchFailure_" + uuid.uuid4().hex) as worker_job:
                with (tmp_path / "worker.log").open("wb") as log:
                    worker, launched = worker_job.start(
                        [sys.executable, "-B", "-c", "import time;time.sleep(30)"],
                        cwd=PROJ_ROOT,
                        env=env(),
                        log=log,
                        affinity=CPU,
                    )
                    fixture = watchdog_fixture(
                        tmp_path,
                        worker_job,
                        controller,
                        foreign_pid=sentinel.pid if failure == "foreign_owner" else None,
                    )
                    with cpu_process(tmp_path / "watchdog.log", fixture.code, arguments=fixture.arguments) as (
                        guard,
                        _,
                        _,
                    ):
                        wait_file(fixture.directory / "guard_ready.json", guard)
                        if failure == "controller_exit":
                            controller.terminate()
                            controller.wait(timeout=6)
                        elif failure == "anchor_mutation":
                            Path(fixture.context["descriptor"]["path"]).write_text('{"changed":true}', encoding="utf-8")
                        elif failure == "live_completion":
                            put(
                                fixture.directory / "all_phases_complete.json",
                                dict(
                                    phases={phase: {"fixture_only": True} for phase in ("propose", "select", "apply")},
                                    context=binding(fixture.path),
                                ),
                            )
                        assert guard.wait(timeout=12) != 0
                        abort = read(fixture.directory / "guard_abort.json")
                        assert abort["actual_owned_exit_verified"] is True
                        assert abort["no_foreign_process_terminated"] is True
                        assert worker_job.pids() == []
                        worker.wait(timeout=6)
                        assert not identity_alive(launched)
                        assert sentinel.poll() is None
                        assert not (fixture.directory / "actual_cuda_exit.json").exists()
                        assert not (fixture.directory / "completion.json").exists()


# Reuse the existing tiny synthetic metadata builders, never production tensors.
from test_lidar_formal_baseline import stage
from test_lidar_formal_completion import pending, finished
from test_lidar_formal_policy_compat import producer_finished, rebind


def test_new_driver_envelope_is_accepted_by_frozen_postleg_checker(producer_finished):
    from hmr4d.lidar_mocap import formal_completion_policy_compat as compat

    value = producer_finished
    descriptor = put(value.root / "synthetic_incremental_descriptor.json", {"fixture_only": True})
    value.request.update(authorization_profile=execution.PROFILE, incremental_descriptor=descriptor)
    context = read(value.context["path"])
    new_entry = value.root.parent.parent / "tools/run_badminton_incremental_guarded_leg.py"
    new_entry.parent.mkdir(parents=True, exist_ok=True)
    new_entry.write_text("# synthetic entry binding; not executed or production authority\n", encoding="utf-8")
    context.update(
        schema=execution.CONTEXT_SCHEMA,
        authorization_profile=execution.PROFILE,
        descriptor=descriptor,
        watchdog_entry=binding(new_entry),
        watchdog_job_name="Local\\GVHMRIncrementalWatchdog_" + Path(value.context["path"]).parent.name,
        driver=deepcopy(value.execution["observed_process_identities"][0]),
    )
    context["driver"]["command"] = [
        sys.executable,
        "-u",
        str(new_entry),
        "--execute",
        "--request",
        value.execution["request"]["path"],
        "--grant",
        value.grant["path"],
        "--descriptor",
        descriptor["path"],
        "--descriptor-sha256",
        descriptor["sha256"],
    ]
    put(Path(value.context["path"]), context)
    rebind(value)
    check = compat._baseline(value.root, value.baseline_binding, preleg=False)
    result = compat._execution(check, value.execution_binding)
    assert result["state"] == "complete_actual_cuda_exited"
    assert read(value.context["path"])["watchdog_entry"] == binding(new_entry)
    assert compat.frozen._execution.__module__ == "hmr4d.lidar_mocap.formal_completion"
