"""Typed incremental leg execution; the three numerical producers stay unchanged.

Only ownership and metadata orchestration live here. The independent watchdog
validates this entry and the typed descriptor, never the legacy date authorizer.
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import psutil

from hmr4d import PROJ_ROOT
from .gpu_handoff import make_local_stage_lease, utc_time
from .guarded_leg_execution import (
    _assert_clean_outputs,
    _assert_time,
    _phase_output,
    _remaining,
    _stop_driver,
    _use_lease_deadline,
    foreign_compute_pids,
    gpu_process_observation,
)
from .guarded_leg_gpu import bound_json, identity_alive, read_json
from .incremental_guard import PROFILE, immutable_anchors, load_authorization
from .pilot import binding, check_binding
from .repair_execution import write_progress_json
from .windows_gpu_job import WindowsGPUJob

ENTRY = PROJ_ROOT / "tools/run_badminton_incremental_guarded_leg.py"
CONTEXT_SCHEMA = "gvhmr-incremental-guard-context-v1"


def cpu_only():
    """Use the original guard allocation; no numeric CUDA work runs here."""
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    process = psutil.Process()
    process.cpu_affinity([26, 27])
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    try:
        process.ionice(psutil.IOPRIO_LOW)
    except (AttributeError, OSError, ValueError):
        pass


def _desktop_allowance(checked, *, startup=True):
    grant = checked["authorization"]["grant"]
    if grant.get("desktop_allowance_fixture_only") is not False:
        raise ValueError("Actual execution requires separately frozen real desktop identities")
    item = grant["desktop_allowance"]
    policy = bound_json(item)
    age = (datetime.now(timezone.utc) - utc_time(policy["observed_utc"])).total_seconds()
    if (
        policy.get("request") != checked["request_binding"]
        or age < 0
        or (startup and age > 900)
        or utc_time(policy["valid_until_utc"]) < checked["authorization"]["deadline"]
    ):
        raise ValueError("Desktop identities are not fresh and bound through this entire actual grant")
    return item


def _gpu_snapshot(checked):
    # Pass the binding explicitly: the legacy module's global allowance is never changed.
    # Freshness is checked at startup, like the original driver. The frozen
    # identity policy still has its own absolute expiry throughout a long stage.
    return gpu_process_observation(_desktop_allowance(checked, startup=False))


def _anchors(context, *, full=False):
    for key in ("request", "grant", "lease", "preflight", "implementation", "descriptor", "watchdog_entry"):
        check_binding(context[key])
    if full:
        for item in context["immutable_bindings"]:
            check_binding(item)


def _driver_command(checked, command):
    if not isinstance(command, list) or len(command) != 12:
        return False
    if [command[i] for i in (1, 3, 4, 6, 8, 10)] != [
        "-u",
        "--execute",
        "--request",
        "--grant",
        "--descriptor",
        "--descriptor-sha256",
    ]:
        return False
    expected = (
        (2, ENTRY),
        (5, Path(checked["request_binding"]["path"])),
        (7, checked["grant_path"]),
        (9, Path(checked["descriptor_binding"]["path"])),
    )
    return (
        Path(command[0]).is_absolute()
        and all(
            Path(command[index]).is_absolute() and Path(command[index]).resolve() == path.resolve()
            for index, path in expected
        )
        and command[11] == checked["descriptor_binding"]["sha256"]
    )


def _driver_alive(identity):
    if not identity_alive(identity):
        return False
    try:
        return psutil.Process(identity["pid"]).cmdline() == identity["command"]
    except psutil.NoSuchProcess:
        return False


def _watchdog_command(command, context_path, context_sha):
    return (
        isinstance(command, list)
        and len(command) == 8
        and Path(command[0]).is_absolute()
        and [command[i] for i in (1, 3, 4, 6)] == ["-u", "--watchdog", "--context", "--context-sha256"]
        and Path(command[2]).resolve() == ENTRY
        and Path(command[5]).resolve() == Path(context_path).resolve()
        and command[7] == context_sha
    )


def validate_context(context, context_path, checked):
    """Pin this real driver, descriptor, implementation and exact private Job."""
    context_path = Path(context_path).resolve()
    directory = context_path.parent
    expected_keys = {
        "schema",
        "authorization_profile",
        "nonce",
        "root",
        "job_name",
        "watchdog_job_name",
        "driver",
        "request",
        "grant",
        "descriptor",
        "lease",
        "preflight",
        "implementation",
        "watchdog_entry",
        "immutable_bindings",
    }
    nonce = context.get("nonce", "")
    if (
        set(context) != expected_keys
        or context.get("schema") != CONTEXT_SCHEMA
        or context.get("authorization_profile") != PROFILE
        or checked.get("authorization_profile") != PROFILE
        or not isinstance(nonce, str)
        or re.fullmatch(r"[0-9a-f]{32}", nonce) is None
        or context_path.name != "context.json"
        or directory.parent != Path(checked["request_binding"]["path"]).parent / "attempts"
        or directory.name != nonce
        or context.get("job_name") != "Local\\GVHMRGuardedLeg_" + nonce
        or context.get("watchdog_job_name") != "Local\\GVHMRIncrementalWatchdog_" + nonce
        or Path(context.get("root", "")).resolve() != checked["root"]
        or context.get("request") != checked["request_binding"]
        or context.get("grant") != checked["authorization"]["binding"]
        or context.get("descriptor") != checked["descriptor_binding"]
        or context.get("implementation") != checked["authorization"]["grant"]["independent_guard"]
        or context.get("watchdog_entry") != binding(ENTRY)
        or not _driver_command(checked, context.get("driver", {}).get("command"))
        or context.get("immutable_bindings") != immutable_anchors(checked)
    ):
        raise ValueError("Incremental context leaves its exact new driver, descriptor, owner or immutable inputs")
    for role, path in (
        ("preflight", directory / "preflight.json"),
        ("lease", directory.parent.parent / "guarded_leg_stage_lease.json"),
    ):
        if Path(context[role]["path"]).resolve() != path:
            raise ValueError("Incremental context contains another attempt or lease path")
    lease = bound_json(context["lease"])
    if lease.get("authorization_profile") != PROFILE or lease.get("incremental_descriptor") != context["descriptor"]:
        raise ValueError("Local lease does not bind this typed incremental descriptor")
    _use_lease_deadline(checked, lease)
    _anchors(context, full=True)
    if not _driver_alive(context["driver"]):
        raise RuntimeError("The exact incremental controller is no longer alive")


def _release_lease(context, directory, reason):
    lease = read_json(context["lease"]["path"])
    if lease.get("status") == "released":
        if lease.get("actual_cuda_exited") is not True or lease.get("attempt") != str(directory):
            raise ValueError("A different release already occupies this private lease")
        return
    check_binding(context["lease"])
    lease.update(
        status="released",
        actual_cuda_exited=True,
        reason=reason,
        released_utc=datetime.now(timezone.utc).isoformat(),
        attempt=str(directory),
    )
    write_progress_json(Path(context["lease"]["path"]), lease)


def _watch_owned(context, context_path, checked):
    """Monitor only the already validated named Job, including a dead controller."""
    directory = Path(context_path).resolve().parent
    context_sha = binding(context_path)["sha256"]
    observed = {}
    with WindowsGPUJob(context["job_name"], create=False) as job, job.deadline_guard(_remaining(checked)):
        own = psutil.Process()
        write_progress_json(
            directory / "guard_ready.json",
            dict(
                pid=own.pid,
                create_time=own.create_time(),
                command=own.cmdline(),
                context_sha256=context_sha,
                actual_affinity=own.cpu_affinity(),
                ready=True,
                authorization_profile=PROFILE,
                incremental_descriptor=context["descriptor"],
            ),
        )
        try:
            last_gpu_check = 0.0
            while True:
                _anchors(context)
                _assert_time(checked)
                if not _driver_alive(context["driver"]):
                    raise RuntimeError("The exact owned incremental controller exited")
                abort_path = directory / "abort_request.json"
                if abort_path.exists():
                    if read_json(abort_path).get("context_sha256") != context_sha:
                        raise ValueError("Abort signal belongs to another context")
                    before = job.identities()
                    job.terminate()
                    job.wait_empty()
                    write_progress_json(
                        directory / "guard_abort.json",
                        dict(
                            state="driver_abort_owned_job_exited",
                            observed_process_identities=before,
                            actual_owned_exit_verified=True,
                            no_foreign_process_terminated=True,
                        ),
                    )
                    return
                for item in job.identities():
                    observed[item["pid"], item["create_time"]] = item
                if time.monotonic() - last_gpu_check >= 1:
                    _anchors(context, full=True)
                    snapshot = _gpu_snapshot(checked)
                    foreign = foreign_compute_pids(snapshot["blocking_compute_pids"], job.pids(), observed)
                    if foreign:
                        raise RuntimeError("Foreign CUDA owner appeared; stopping only the owned Job")
                    last_gpu_check = time.monotonic()
                complete = directory / "all_phases_complete.json"
                if complete.exists():
                    material = read_json(complete)
                    if (
                        set(material.get("phases", {})) != {"propose", "select", "apply"}
                        or material.get("context") != binding(context_path)
                        or job.pids()
                    ):
                        raise RuntimeError("Three-phase completion differs from this context or live owned Job")
                    _anchors(context, full=True)
                    snapshot = _gpu_snapshot(checked)
                    if snapshot["blocking_compute_pids"]:
                        time.sleep(0.1)
                        continue
                    write_progress_json(
                        directory / "actual_cuda_exit.json",
                        dict(
                            state="all_owned_cuda_exited",
                            context_sha256=context_sha,
                            observed_process_identities=list(observed.values()),
                            job_empty=True,
                            gpu0_compute_pids_empty=True,
                            gpu0_process_classification=snapshot,
                            completed_utc=datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    return
                write_progress_json(
                    directory / "watchdog.json",
                    dict(
                        state="guarding",
                        heartbeat_utc=datetime.now(timezone.utc).isoformat(),
                        driver=context["driver"],
                        owned=job.identities(),
                        gpu_started=(directory / "propose.launch.json").exists(),
                    ),
                )
                time.sleep(0.2)
        except BaseException as error:
            before = job.identities()
            job.terminate()
            job.wait_empty()
            write_progress_json(
                directory / "guard_abort.json",
                dict(
                    state="owned_private_job_stopped",
                    error=type(error).__name__ + ": " + str(error),
                    observed_process_identities=before,
                    actual_owned_exit_verified=True,
                    no_foreign_process_terminated=True,
                    utc=datetime.now(timezone.utc).isoformat(),
                ),
            )
            try:
                _stop_driver(context["driver"])
            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                # An unverifiable/reused identity is never terminated. The
                # actual owned numeric Job has already been proved empty.
                pass
            _release_lease(context, directory, "independent_incremental_guard_abort")
            raise


def watchdog(context_path, context_sha):
    cpu_only()
    context_path = Path(context_path).resolve()
    context = bound_json(dict(path=str(context_path), sha256=context_sha))
    checked = load_authorization(
        context["request"]["path"],
        context["grant"]["path"],
        deep=False,
        descriptor_path=context["descriptor"]["path"],
        descriptor_sha256=context["descriptor"]["sha256"],
    )
    if not _watchdog_command(psutil.Process().cmdline(), context_path, context_sha):
        raise ValueError("Watchdog must execute its own exact bound incremental CLI")
    _desktop_allowance(checked)
    validate_context(context, context_path, checked)
    with WindowsGPUJob(context["watchdog_job_name"], create=False) as watchdog_job:
        if os.getpid() not in watchdog_job.pids():
            raise ValueError("The independent watchdog is outside its actual private CPU Job")
    _watch_owned(context, context_path, checked)


def _run_owned_command(job, command, checked, context, directory, guard, ready, observed, *, phase, env, affinity):
    """Run one command under the real suspended-assigned Job API."""
    if job.pids():
        raise RuntimeError("A previous phase still owns descendants")
    try:
        with (directory / (phase + ".log")).open("xb") as log:
            child, launched = job.start(command, cwd=PROJ_ROOT, env=env, log=log, affinity=affinity)
            write_progress_json(directory / (phase + ".launch.json"), launched)
            observed[launched["pid"], launched["create_time"]] = launched
            while child.poll() is None or job.pids():
                _anchors(context)
                _assert_time(checked)
                if guard.poll() is not None or not identity_alive(ready):
                    raise RuntimeError("The independent watchdog exited during the owned phase")
                for identity in job.identities():
                    observed[identity["pid"], identity["create_time"]] = identity
                time.sleep(0.2)
            job.wait_empty()
            if child.returncode:
                write_progress_json(
                    directory / (phase + ".failure.json"),
                    dict(
                        actual_exit_code=child.returncode,
                        actual_job_empty=True,
                        launcher=launched,
                    ),
                )
                raise RuntimeError(phase + " failed; preserve the actual log and partial outputs")
    except BaseException:
        job.terminate()
        job.wait_empty()
        raise
    return launched


def _run_owned_phase(job, item, checked, context, directory, guard, ready, observed):
    cuda = item["cuda"]
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0" if cuda else "-1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
    )
    return _run_owned_command(
        job,
        item["argv"],
        checked,
        context,
        directory,
        guard,
        ready,
        observed,
        phase=item["phase"],
        env=environment,
        affinity=[0, 1] if cuda else [26, 27],
    )


def _wait_guard(guard, *, job=None, require_clean=True):
    if guard is None:
        return
    try:
        guard.wait(timeout=8)
        if job is not None:
            job.wait_empty()
    except (subprocess.TimeoutExpired, RuntimeError):
        if job is not None:
            job.terminate()
            job.wait_empty()
        else:
            # This is our original Popen kernel handle, never a name/PID scan.
            guard.terminate()
        guard.wait(timeout=6)
        if require_clean:
            raise RuntimeError("Owned independent watchdog failed to exit")


def execute(request_path, grant_path, *, descriptor_path, descriptor_sha256):
    cpu_only()
    request_path, grant_path = Path(request_path).resolve(), Path(grant_path).resolve()
    checked = load_authorization(
        request_path,
        grant_path,
        descriptor_path=descriptor_path,
        descriptor_sha256=descriptor_sha256,
    )
    own = psutil.Process()
    identity = dict(pid=own.pid, create_time=own.create_time(), command=own.cmdline())
    if not _driver_command(checked, identity["command"]):
        raise ValueError("Use the explicit new incremental execute CLI and descriptor pair")
    _desktop_allowance(checked)
    _assert_time(checked, first=True)
    _assert_clean_outputs(checked["root"])
    if _gpu_snapshot(checked)["blocking_compute_pids"]:
        raise RuntimeError("GPU-0 still has a compute owner; no CUDA process launched")
    folder = request_path.parent
    lease_path = folder / "guarded_leg_stage_lease.json"
    if lease_path.exists():
        raise ValueError("Preserve the existing stage lease; no automatic relaunch")
    lock = folder / "execution.lock.json"
    with lock.open("x", encoding="utf-8") as handle:
        import json

        json.dump(dict(state="starting", driver=identity, request=binding(request_path)), handle)
    directory = folder / "attempts" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    write_progress_json(
        directory / "preflight.json",
        dict(
            state="complete",
            request=binding(request_path),
            grant=binding(grant_path),
            readiness=checked["authorization"]["grant"]["readiness"],
            implementation=checked["authorization"]["grant"]["independent_guard"],
            incremental_descriptor=checked["descriptor_binding"],
            actual_cpu_affinity=own.cpu_affinity(),
            native_and_baseline_evidence_checked=True,
        ),
    )
    lease = make_local_stage_lease(checked["authorization"], checked["root"], "guarded_leg")
    lease.update(authorization_profile=PROFILE, incremental_descriptor=checked["descriptor_binding"])
    _use_lease_deadline(checked, lease)
    _assert_time(checked, first=True)
    write_progress_json(lease_path, lease)
    write_progress_json(directory / "lease_before.json", lease)
    context = dict(
        schema=CONTEXT_SCHEMA,
        authorization_profile=PROFILE,
        nonce=directory.name,
        root=str(checked["root"]),
        job_name="Local\\GVHMRGuardedLeg_" + directory.name,
        watchdog_job_name="Local\\GVHMRIncrementalWatchdog_" + directory.name,
        driver=identity,
        request=binding(request_path),
        grant=binding(grant_path),
        descriptor=checked["descriptor_binding"],
        lease=binding(lease_path),
        preflight=binding(directory / "preflight.json"),
        implementation=checked["authorization"]["grant"]["independent_guard"],
        watchdog_entry=binding(ENTRY),
        immutable_bindings=immutable_anchors(checked),
    )
    context_path = directory / "context.json"
    write_progress_json(context_path, context)
    validate_context(context, context_path, checked)
    guard, phases, observed = None, {}, {}
    result = None
    with (
        WindowsGPUJob(context["job_name"]) as job,
        WindowsGPUJob(context["watchdog_job_name"]) as watchdog_job,
        job.deadline_guard(_remaining(checked)),
        watchdog_job.deadline_guard(_remaining(checked)),
    ):
        try:
            command = [
                sys.executable,
                "-u",
                str(ENTRY),
                "--watchdog",
                "--context",
                str(context_path),
                "--context-sha256",
                binding(context_path)["sha256"],
            ]
            with (directory / "watchdog.log").open("xb") as log:
                guard, guard_launch = watchdog_job.start(
                    command,
                    cwd=PROJ_ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"),
                    log=log,
                    affinity=[26, 27],
                )
                write_progress_json(directory / "watchdog.launch.json", guard_launch)
            ready_path = directory / "guard_ready.json"
            startup_deadline = time.monotonic() + 60
            while not ready_path.exists():
                _anchors(context)
                _assert_time(checked)
                if guard.poll() is not None or time.monotonic() >= startup_deadline:
                    raise RuntimeError("Independent incremental watchdog was not ready before CUDA launch")
                time.sleep(0.1)
            ready = read_json(ready_path)
            if (
                ready.get("context_sha256") != binding(context_path)["sha256"]
                or ready.get("ready") is not True
                or ready.get("authorization_profile") != PROFILE
                or ready.get("incremental_descriptor") != checked["descriptor_binding"]
                or not _watchdog_command(ready.get("command"), context_path, binding(context_path)["sha256"])
                or ready.get("pid") not in watchdog_job.pids()
                or ready.get("actual_affinity") != [26, 27]
                or not identity_alive(ready)
            ):
                raise RuntimeError("Independent watchdog readiness belongs to another live context")
            for item in checked["commands"]:
                phase = item["phase"]
                fresh = load_authorization(
                    request_path,
                    grant_path,
                    deep=phase == "apply",
                    descriptor_path=checked["descriptor_binding"]["path"],
                    descriptor_sha256=checked["descriptor_binding"]["sha256"],
                )
                fresh["deadline"] = min(fresh["deadline"], checked["deadline"])
                _anchors(context, full=True)
                _assert_time(fresh, first=phase == "propose")
                if guard.poll() is not None or not identity_alive(ready):
                    raise RuntimeError("Independent watchdog exited before phase launch")
                if item["cuda"] and _gpu_snapshot(fresh)["blocking_compute_pids"]:
                    raise RuntimeError("GPU-0 is not free before the next original leg phase")
                launched = _run_owned_phase(job, item, fresh, context, directory, guard, ready, observed)
                phases[phase] = dict(
                    output=_phase_output(checked["root"], phase, checked["configuration"]["planned_windows"]),
                    actual_job_empty=True,
                    launcher=launched,
                )
                write_progress_json(directory / (phase + ".exit.json"), phases[phase])
            _anchors(context, full=True)
            write_progress_json(
                directory / "all_phases_complete.json", dict(phases=phases, context=binding(context_path))
            )
            _wait_guard(guard, job=watchdog_job)
            exit_path = directory / "actual_cuda_exit.json"
            exited = read_json(exit_path) if exit_path.is_file() else {}
            if (
                guard.returncode != 0
                or exited.get("state") != "all_owned_cuda_exited"
                or exited.get("context_sha256") != binding(context_path)["sha256"]
                or exited.get("job_empty") is not True
                or exited.get("gpu0_compute_pids_empty") is not True
            ):
                raise RuntimeError("Independent guard did not prove actual owned CUDA exit")
            _anchors(context, full=True)
            result = dict(
                schema="gvhmr-guarded-leg-execution-v1",
                state="complete_actual_cuda_exited",
                request=context["request"],
                grant=context["grant"],
                phases=phases,
                actual_exit=binding(exit_path),
                observed_process_identities=list(observed.values()),
                training_published=False,
                independent_guard=binding(ready_path),
                authorization_profile=PROFILE,
                incremental_descriptor=checked["descriptor_binding"],
            )
        except BaseException as error:
            write_progress_json(directory / "abort_request.json", dict(context_sha256=binding(context_path)["sha256"]))
            job.terminate()
            job.wait_empty()
            write_progress_json(
                directory / "driver_failure.json",
                dict(
                    state="failed_owned_job_exited",
                    error=type(error).__name__ + ": " + str(error),
                    request=context["request"],
                    grant=context["grant"],
                    observed=list(observed.values()),
                    training_published=False,
                ),
            )
            raise
        finally:
            job.wait_empty()
            _wait_guard(guard, job=watchdog_job, require_clean=False)
            watchdog_job.wait_empty()
            _release_lease(context, directory, "complete" if result is not None else "owned_driver_failure")
    # Completion is published only after both the Job and watchdog actually exit,
    # and the private lease has been released successfully.
    write_progress_json(lock, dict(state="complete_actual_cuda_exited", driver=identity, attempt=str(directory)))
    write_progress_json(directory / "completion.json", result)
    return result
