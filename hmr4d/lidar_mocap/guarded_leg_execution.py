"""Independent Windows job/watchdog execution of the three fixed leg phases."""

from datetime import datetime, timedelta, timezone
import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import psutil

from hmr4d import PROJ_ROOT
from .gpu_handoff import make_local_stage_lease, utc_time
from .guarded_leg_gpu import (
    SHUTDOWN_MARGIN_SECONDS, bound_json, identity_alive, immutable_anchors, load_authorization, read_json,
)
from .pilot import binding, check_binding
from .repair_execution import write_progress_json
from .windows_gpu_job import WindowsGPUJob


DESKTOP_ALLOWANCE = None  # Each actual driver/watchdog loads the issued manifest's fresh binding.


class _UnicodeString(ctypes.Structure):
    _fields_ = [("length", wintypes.USHORT), ("maximum", wintypes.USHORT), ("buffer", ctypes.c_void_p)]


class _ProcessImage(ctypes.Structure):
    _fields_ = [("pid", wintypes.HANDLE), ("image", _UnicodeString)]


def _kernel_process_executable(pid):
    """Read the actual kernel image name when OpenProcess denies image queries."""
    if os.name != "nt":
        raise OSError("Windows kernel process image queries require Windows")
    query = ctypes.WinDLL("ntdll").NtQuerySystemInformation
    query.argtypes = [ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p]
    query.restype = ctypes.c_long
    buffer = ctypes.create_unicode_buffer(32760)
    item = _ProcessImage(pid=pid, image=_UnicodeString(
        maximum=ctypes.sizeof(buffer) - 2, buffer=ctypes.addressof(buffer)))
    status = query(88, ctypes.byref(item), ctypes.sizeof(item), None)
    if (status or not item.image.length or item.image.length % 2
            or item.image.length > item.image.maximum or item.image.buffer != ctypes.addressof(buffer)):
        raise OSError("Kernel process image query failed: " + hex(status & 0xffffffff))
    native = ctypes.wstring_at(buffer, item.image.length // 2)
    if "\0" in native or not native.casefold().startswith("\\device\\"):
        raise ValueError("Kernel process image has an unknown native path prefix")
    devices = ctypes.WinDLL("kernel32", use_last_error=True).QueryDosDeviceW
    devices.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    devices.restype = wintypes.DWORD
    choices = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        mapped = ctypes.create_unicode_buffer(32768)
        if devices(letter + ":", mapped, len(mapped)):
            prefix = mapped.value
            if prefix and native.casefold().startswith(prefix.casefold() + "\\"):
                choices.append((prefix, letter + ":"))
    if not choices:
        raise ValueError("No actual DOS device mapping exists for the kernel process image")
    prefix, drive = max(choices, key=lambda pair: len(pair[0]))
    path = drive + native[len(prefix):]
    if not Path(path).is_absolute() or ".." in Path(path).parts or ":" in path[2:]:
        raise ValueError("Kernel process image did not resolve to an ordinary absolute executable path")
    return path


def process_executable(pid, *, expected_create_time=None):
    """Return a real image path, rechecking PID identity across either query."""
    process = psutil.Process(pid)
    created = process.create_time()
    if expected_create_time is not None and created != expected_create_time:
        raise psutil.NoSuchProcess(pid, msg="Process creation time changed before its image query")
    try:
        executable = process.exe()
    except psutil.AccessDenied:
        executable = _kernel_process_executable(pid)
    if psutil.Process(pid).create_time() != created:
        raise psutil.NoSuchProcess(pid, msg="Process creation time changed during its image query")
    return executable


def cpu_only():
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    process = psutil.Process()
    process.cpu_affinity([26, 27])
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def configure_desktop_allowance(checked):
    grant = checked["authorization"]["grant"]
    implementation = bound_json(grant["independent_guard"])
    source = grant if "desktop_allowance" in grant else implementation
    allowance = source.get("desktop_allowance")
    policy = bound_json(allowance)
    if source.get("desktop_allowance_fixture_only") is not False:
        raise ValueError("A machine fixture cannot authorize actual guarded execution")
    if policy.get("request") != checked["request_binding"]:
        raise ValueError("Desktop evidence belongs to another actual guarded request")
    age = (datetime.now(timezone.utc) - utc_time(policy["observed_utc"])).total_seconds()
    if not 0 <= age <= 900:
        raise ValueError("Capture the actual desktop identities within fifteen minutes before leg startup")
    if utc_time(policy["valid_until_utc"]) < utc_time(checked["authorization"]["grant"]["approved_deadline_utc"]):
        raise ValueError("Desktop identity policy must cover the entire approved guarded-leg deadline")
    global DESKTOP_ALLOWANCE
    DESKTOP_ALLOWANCE = allowance


def gpu_process_observation(desktop_allowance=None):
    """Keep raw WDDM data, exempting only separately frozen desktop identities."""
    allowance = desktop_allowance or DESKTOP_ALLOWANCE
    if allowance is None:
        raise ValueError("An explicitly issued fresh desktop binding is required before GPU inspection")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    result = subprocess.run(["nvidia-smi", "--id=0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=2, creationflags=flags, check=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not value.isdigit() for value in lines):
        raise RuntimeError("Cannot verify actual GPU-0 compute ownership")
    raw = {int(value) for value in lines}
    policy = bound_json(allowance)
    if (policy.get("schema") != "gvhmr-guarded-leg-wddm-desktop-identities-v1"
            or policy.get("state") != "frozen_observed_desktop_only"
            or policy.get("future_or_reused_PID_exempt") is not False
            or datetime.now(timezone.utc) >= utc_time(policy["valid_until_utc"])):
        raise ValueError("A current hash-bound original-desktop policy is required")
    check_binding(policy["request"])
    entries = {}
    for item in policy["desktop_identities"]:
        if (type(item.get("pid")) is not int or item["pid"] <= 0 or item["pid"] in entries
                or type(item.get("create_time")) not in (int, float)
                or not math.isfinite(item["create_time"]) or item["create_time"] <= 0
                or not Path(item.get("executable", "")).is_absolute()
                or item.get("classification") != "root_observed_desktop_GUI"
                or Path(item["executable"]).name.lower() in ("python.exe", "pythonw.exe")):
            raise ValueError("Desktop exclusion lacks its original non-Python PID/create-time/executable")
        entries[item["pid"]] = item
    ignored = []
    for pid in raw & entries.keys():
        original = entries[pid]
        try:
            process = psutil.Process(pid)
            if (process.create_time() == original["create_time"]
                    and os.path.normcase(process_executable(pid, expected_create_time=original["create_time"]))
                    == os.path.normcase(original["executable"])):
                ignored.append(original)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, ValueError):
            pass  # Unknown/inaccessible/reused/stale identities remain blocking.
    return dict(schema="gvhmr-gpu-process-classification-v1", observed_utc=datetime.now(timezone.utc).isoformat(),
                raw_wddm_compute_pids=sorted(raw), frozen_desktop_exclusions=ignored,
                blocking_compute_pids=sorted(raw - {item["pid"] for item in ignored}),
                desktop_allowance=allowance)


def compute_pids():
    observation = gpu_process_observation()
    compute_pids.last_observation = observation
    return set(observation["blocking_compute_pids"])


def _assert_time(checked, *, first=False, now=None):
    now = now or datetime.now(timezone.utc)
    cutoff = checked["deadline"] - timedelta(seconds=SHUTDOWN_MARGIN_SECONDS)
    if now >= cutoff:
        raise TimeoutError("Guarded-leg shutdown margin/deadline reached")
    if first and (checked["deadline"] - now).total_seconds() < checked["request"]["minimum_remaining_launch_seconds"]:
        raise TimeoutError("Insufficient actual time remains before first guarded CUDA launch")


def _anchors(context):
    for key in ("request", "grant", "lease", "preflight", "implementation"):
        check_binding(context[key])


def _remaining(checked):
    return (checked["deadline"] - datetime.now(timezone.utc)).total_seconds() - SHUTDOWN_MARGIN_SECONDS


def _use_lease_deadline(checked, lease):
    grant = checked["authorization"]["grant"]
    start, deadline = utc_time(lease["approved_start_utc"]), utc_time(lease["approved_deadline_utc"])
    if (lease.get("status") != "held" or lease.get("owner") != "GVHMR"
            or lease.get("stage") != "guarded_leg" or lease.get("root") != str(checked["root"])
            or lease.get("continuous_gpu_grant") != checked["authorization"]["binding"]
            or lease.get("manifest_sha256") != checked["request"]["scope"][0]["manifest"]["sha256"]
            or not checked["authorization"]["start"] <= start < deadline <= checked["deadline"]
            or deadline > start + timedelta(minutes=grant["maximum_stage_minutes"]["guarded_leg"])):
        raise ValueError("Local leg lease leaves its actual command, manifest or approved time window")
    checked["deadline"] = min(checked["deadline"], deadline)


def validate_context(context, context_path, checked):
    directory = Path(context_path).resolve().parent
    command = context["driver"]["command"]
    readpass = checked["authorization"]["grant"].get("verified_read_pass")
    canonical = len(command) == (12 if readpass else 8) and [command[i] for i in (1, 3, 4, 6)] == [
        "-u", "--execute", "--request", "--grant"]
    if canonical:
        for index, expected in ((2, PROJ_ROOT / "tools/run_badminton_guarded_leg.py"),
                                (5, Path(checked["request_binding"]["path"])), (7, checked["grant_path"])):
            canonical = canonical and Path(command[index]).is_absolute() and Path(command[index]).resolve() == expected
    if (directory.parent != Path(context["request"]["path"]).parent / "attempts"
            or directory.name != context["nonce"] or not len(context["nonce"]) == 32
            or context["job_name"] != "Local\\GVHMRGuardedLeg_" + context["nonce"]
            or Path(context["root"]).resolve() != checked["root"]
            or not canonical
            or context["immutable_bindings"] != immutable_anchors(checked)):
        raise ValueError("Watchdog context leaves its exact owner, command or stage bindings")
    if readpass:
        from .verified_read_pass import validate_live_receipt
        validate_live_receipt(checked, context["driver"])


def foreign_compute_pids(current, members, observed):
    foreign = set(current) - set(members)
    for pid in list(foreign):
        try:
            psutil.Process(pid).create_time()
        except psutil.NoSuchProcess:
            if any(key[0] == pid for key in observed):
                foreign.remove(pid)  # A departed own PID may briefly remain in NVML output.
    return foreign


def _stop_driver(identity):
    if identity_alive(identity):
        process = psutil.Process(identity["pid"])
        if process.create_time() == identity["create_time"] and process.cmdline() == identity["command"]:
            process.terminate()


def watchdog(context_path, context_sha):
    cpu_only()
    context = bound_json(dict(path=str(Path(context_path).resolve()), sha256=context_sha))
    directory = Path(context_path).resolve().parent
    checked = load_authorization(context["request"]["path"], context["grant"]["path"], deep=False)
    configure_desktop_allowance(checked)
    validate_context(context, context_path, checked)
    _anchors(context)
    _use_lease_deadline(checked, bound_json(context["lease"]))
    if not identity_alive(context["driver"]):
        raise RuntimeError("The stage controller already exited")
    state_path = directory / "watchdog.json"
    observed = {}
    with WindowsGPUJob(context["job_name"], create=False) as job, job.deadline_guard(_remaining(checked)):
        own = psutil.Process()
        write_progress_json(directory / "guard_ready.json", dict(
            pid=own.pid, create_time=own.create_time(), command=own.cmdline(),
            context_sha256=context_sha, actual_affinity=own.cpu_affinity(), ready=True))
        try:
            last_gpu_check = 0.
            while True:
                _anchors(context)
                _assert_time(checked)
                if not identity_alive(context["driver"]):
                    raise RuntimeError("Owned stage controller exited or its PID was reused")
                abort_path = directory / "abort_request.json"
                if abort_path.exists():
                    if read_json(abort_path).get("context_sha256") != context_sha:
                        raise ValueError("Abort signal uses another context")
                    before = job.identities()
                    job.terminate()
                    job.wait_empty()
                    write_progress_json(directory / "guard_abort.json", dict(
                        state="driver_abort_owned_job_exited", observed_process_identities=before,
                        actual_owned_exit_verified=True, no_foreign_process_terminated=True))
                    return
                identities = job.identities()
                for item in identities:
                    observed[item["pid"], item["create_time"]] = item
                if time.monotonic() - last_gpu_check >= 1:
                    for item in context["immutable_bindings"]:
                        _assert_time(checked)
                        check_binding(item)
                    foreign = foreign_compute_pids(compute_pids(), job.pids(), observed)
                    if foreign:
                        raise RuntimeError("Foreign CUDA ownership appeared; stopping only this private job")
                    last_gpu_check = time.monotonic()
                if (directory / "all_phases_complete.json").exists():
                    if job.pids():
                        raise RuntimeError("Completion was declared while owned descendants remained")
                    if compute_pids():
                        time.sleep(.1)
                        continue
                    write_progress_json(directory / "actual_cuda_exit.json", dict(
                        state="all_owned_cuda_exited", context_sha256=context_sha,
                        observed_process_identities=list(observed.values()), job_empty=True,
                        gpu0_compute_pids_empty=True,
                        gpu0_process_classification=getattr(compute_pids, "last_observation", None),
                        completed_utc=datetime.now(timezone.utc).isoformat()))
                    return
                write_progress_json(state_path, dict(state="guarding", heartbeat_utc=datetime.now(timezone.utc).isoformat(),
                    driver=context["driver"], owned=identities, gpu_started=(directory / "propose.launch.json").exists()))
                time.sleep(.2)
        except BaseException as error:
            before = job.identities()
            job.terminate()
            job.wait_empty()
            write_progress_json(directory / "guard_abort.json", dict(
                state="owned_private_job_stopped", error=type(error).__name__ + ": " + str(error),
                observed_process_identities=before, actual_owned_exit_verified=True,
                no_foreign_process_terminated=True, utc=datetime.now(timezone.utc).isoformat()))
            _stop_driver(context["driver"])
            released = read_json(context["lease"]["path"])
            released.update(status="released", actual_cuda_exited=True, reason="independent_guard_abort",
                            released_utc=datetime.now(timezone.utc).isoformat(), attempt=str(directory))
            write_progress_json(Path(context["lease"]["path"]), released)
            raise


def _phase_output(root, phase, expected_windows):
    relative = {"propose": "leg_refinement_proposals/receipt.json", "select": "leg_refinement_selection.json",
                "apply": "leg_refinement_receipt.json"}[phase]
    path = root / relative
    value = read_json(path)
    plan = read_json(root / "final_labels/repair_plan.json")
    keys = {j["source_id"] + ":" + Path(j["source_window"]).name for j in plan["jobs"]}
    if value.get("state") != "complete":
        raise ValueError("The actual " + phase + " output is incomplete")
    if phase != "apply" and (set(value.get("windows", {})) != keys or len(keys) != expected_windows
                             or value.get("source_union_sha256") != plan["source_union_sha256"]):
        raise ValueError("The actual " + phase + " window coverage is incomplete")
    if phase == "propose":
        for item in value["windows"].values():
            check_binding(item)
    if phase == "select" and value.get("source_proposals_sha256") != binding(
            root / "leg_refinement_proposals/receipt.json")["sha256"]:
        raise ValueError("Selection uses another proposal receipt")
    if phase == "apply":
        selection_path = root / "leg_refinement_selection.json"
        selection = read_json(selection_path)
        if (not set(value.get("windows", {})) <= set(selection["windows"])
                or value.get("selected_frames") != selection["selected_frames"]
                or value.get("updated_frames") != selection["selected_frames"]
                or value.get("selection_sha256") != binding(selection_path)["sha256"]):
            raise ValueError("Actual applied leg updates differ from the completed selection")
        post = read_json(root / "final_labels/post_refinement.json")
        if any(post.get(key) != value.get(key) for key in ("state", "updated_frames", "selected_frames",
                                                         "selection_sha256", "windows")):
            raise ValueError("The actual post-refinement receipt is incomplete")
    return binding(path)


def _assert_clean_outputs(root):
    for relative in ("leg_refinement_proposals/receipt.json", "leg_refinement_selection.json",
                     "leg_refinement_receipt.json", "final_labels/post_refinement.json"):
        if (root / relative).exists():
            raise ValueError("Preserve previous leg outputs; this separate authorization permits one fresh attempt")


def execute(request_path, grant_path, *, pregrant_path=None, pregrant_sha256=None):
    cpu_only()
    if pregrant_path is None:
        if pregrant_sha256 is not None:
            raise ValueError("A CPU pregrant SHA requires its exact explicit bundle")
        return _execute(request_path, grant_path)
    from .verified_read_pass import VerifiedReadPass
    with VerifiedReadPass.prepare(request_path, pregrant_path, pregrant_sha256) as read_pass:
        grant_path = Path(grant_path).resolve()
        while not grant_path.exists():
            if datetime.now(timezone.utc) >= read_pass.wait_deadline:
                raise TimeoutError("No actual grant arrived before the separate CPU waiting deadline")
            time.sleep(.5)
        return _execute(request_path, grant_path, read_pass=read_pass)


def _execute(request_path, grant_path, *, read_pass=None):
    request_path, grant_path = Path(request_path).resolve(), Path(grant_path).resolve()
    checked = load_authorization(request_path, grant_path, deep=read_pass is None)
    grant = checked["authorization"]["grant"]
    if read_pass is None and grant.get("verified_read_pass"):
        raise ValueError("This grant requires the same actual live CPU pregrant driver and held read capability")
    if read_pass is not None:
        checked["readiness"] = read_pass.validate_grant(checked)
        read_pass.check_baseline(checked)
    configure_desktop_allowance(checked)
    _assert_time(checked, first=True)
    _assert_clean_outputs(checked["root"])
    if compute_pids():
        raise RuntimeError("GPU-0 still has a compute owner; no new CUDA process was launched")
    folder = request_path.parent
    lock = folder / "execution.lock.json"
    process = psutil.Process()
    identity = dict(pid=process.pid, create_time=process.create_time(), command=process.cmdline())
    with lock.open("x", encoding="utf-8") as handle:
        json.dump(dict(state="starting", driver=identity, request=binding(request_path)), handle)
    nonce = uuid.uuid4().hex
    directory = folder / "attempts" / nonce
    directory.mkdir(parents=True, exist_ok=False)
    preflight = directory / "preflight.json"
    write_progress_json(preflight, dict(state="complete", request=binding(request_path), grant=binding(grant_path),
        readiness=checked["authorization"]["grant"]["readiness"], implementation=checked["authorization"]["grant"]["independent_guard"],
        actual_cpu_affinity=process.cpu_affinity(), native_and_baseline_evidence_checked=True))
    lease_path = folder / "guarded_leg_stage_lease.json"
    if lease_path.exists():
        raise ValueError("Preserve the existing stage lease; do not relaunch or overwrite an earlier attempt")
    lease = make_local_stage_lease(checked["authorization"], checked["root"], "guarded_leg")
    _use_lease_deadline(checked, lease)
    _assert_time(checked, first=True)
    write_progress_json(lease_path, lease)
    lease_binding = binding(lease_path)
    write_progress_json(directory / "lease_before.json", lease)
    context = dict(nonce=nonce, root=str(checked["root"]), job_name="Local\\GVHMRGuardedLeg_" + nonce,
                   driver=identity, request=binding(request_path), grant=binding(grant_path),
                   lease=lease_binding, preflight=binding(preflight),
                   implementation=checked["authorization"]["grant"]["independent_guard"],
                   immutable_bindings=immutable_anchors(checked))
    context_path = directory / "context.json"
    write_progress_json(context_path, context)
    guard = None
    phases, observed = {}, {}
    with WindowsGPUJob(context["job_name"]) as job, job.deadline_guard(_remaining(checked)):
        try:
            command = [sys.executable, "-u", str(PROJ_ROOT / "tools/run_badminton_guarded_leg.py"),
                       "--watchdog", "--context", str(context_path), "--context-sha256", binding(context_path)["sha256"]]
            with (directory / "watchdog.log").open("ab") as log:
                guard = subprocess.Popen(command, cwd=PROJ_ROOT, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS, close_fds=True)
            deadline = time.monotonic() + 60
            ready_path = directory / "guard_ready.json"
            while not ready_path.exists():
                _anchors(context)
                _assert_time(checked)
                if guard.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Independent watchdog did not become ready before CUDA launch")
                time.sleep(.1)
            ready = read_json(ready_path)
            if (ready.get("context_sha256") != binding(context_path)["sha256"] or ready.get("ready") is not True
                    or not identity_alive(ready)):
                raise RuntimeError("Independent guard process/context is not actually ready")
            for item in checked["commands"]:
                phase, cuda = item["phase"], item["cuda"]
                fresh = load_authorization(request_path, grant_path, deep=phase == "apply" and read_pass is None)
                if read_pass is not None:
                    read_pass.validate_grant(fresh)
                fresh["deadline"] = min(fresh["deadline"], checked["deadline"])
                _anchors(context)
                _assert_time(fresh, first=phase == "propose")
                if guard.poll() is not None or not identity_alive(ready):
                    raise RuntimeError("Independent watchdog exited before phase launch")
                if job.pids():
                    raise RuntimeError("Previous phase descendants have not actually exited")
                if cuda and compute_pids():
                    raise RuntimeError("GPU-0 is not actually free before the next leg phase")
                if read_pass is not None and phase == "apply":
                    released = read_pass.before_apply(fresh)
                    write_progress_json(directory / "read_pass_apply_release.json", dict(
                        state="apply_invalidated", cpu_read_pass=read_pass.receipt, exact_write_paths=released))
                    _assert_time(fresh)
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0" if cuda else "-1",
                                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
                affinity = [0, 1] if cuda else [26, 27]
                with (directory / (phase + ".log")).open("ab") as log:
                    child, launched = job.start(item["argv"], cwd=PROJ_ROOT, env=environment, log=log, affinity=affinity)
                    write_progress_json(directory / (phase + ".launch.json"), launched)
                    observed[launched["pid"], launched["create_time"]] = launched
                    while child.poll() is None or job.pids():
                        _anchors(context)
                        _assert_time(checked)
                        if guard.poll() is not None or not identity_alive(ready):
                            raise RuntimeError("Independent watchdog failed during the owned phase")
                        for child_identity in job.identities():
                            observed[child_identity["pid"], child_identity["create_time"]] = child_identity
                        time.sleep(.2)
                    if child.returncode:
                        raise RuntimeError(phase + " failed; preserve the actual log and partial outputs")
                job.wait_empty()
                phases[phase] = dict(output=_phase_output(checked["root"], phase,
                    checked["configuration"]["planned_windows"]), actual_job_empty=True, launcher=launched)
                write_progress_json(directory / (phase + ".exit.json"), phases[phase])
            write_progress_json(directory / "all_phases_complete.json", dict(phases=phases, context=binding(context_path)))
            guard.wait(timeout=8)
            if guard.returncode or not (directory / "actual_cuda_exit.json").exists():
                raise RuntimeError("Independent guard did not confirm actual CUDA exit")
            result = dict(schema="gvhmr-guarded-leg-execution-v1", state="complete_actual_cuda_exited",
                request=context["request"], grant=context["grant"], phases=phases,
                actual_exit=binding(directory / "actual_cuda_exit.json"), observed_process_identities=list(observed.values()),
                training_published=False, independent_guard=binding(ready_path))
            write_progress_json(directory / "completion.json", result)
        except BaseException as error:
            job.terminate()
            job.wait_empty()
            write_progress_json(directory / "driver_failure.json", dict(
                state="failed_owned_job_exited", error=type(error).__name__ + ": " + str(error),
                request=context["request"], grant=context["grant"], observed=list(observed.values()), training_published=False))
            write_progress_json(directory / "abort_request.json", dict(context_sha256=binding(context_path)["sha256"]))
            raise
        finally:
            # Never announce release before this private kernel job is actually empty.
            job.wait_empty()
            if guard is not None and guard.poll() is None:
                guard.wait(timeout=8)
            released = dict(lease, status="released", actual_cuda_exited=True,
                            released_utc=datetime.now(timezone.utc).isoformat(), attempt=str(directory))
            write_progress_json(lease_path, released)
    write_progress_json(lock, dict(state="complete_actual_cuda_exited", driver=identity, attempt=str(directory)))
    return result
