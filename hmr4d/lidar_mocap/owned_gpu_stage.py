"""One exact full-native stage in a private Windows Job with an independent owner.

The descriptor is a source/code contract, never authority. A separately issued,
held local lease and fresh exact desktop snapshot are required for execution.
"""

from datetime import datetime, timedelta, timezone
import os
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import psutil

from hmr4d import PROJ_ROOT
from .gpu_handoff import utc_time
from .guarded_leg_gpu import bound_json, identity_alive, read_json
from .guarded_leg_execution import foreign_compute_pids, gpu_process_observation
from .pilot import binding, check_binding
from .repair_execution import write_progress_json
from .windows_gpu_job import WindowsGPUJob


MARGIN = 10


def cpu_controller():
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    process = psutil.Process()
    process.cpu_affinity([26, 27])
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)


def implementation_files():
    files = {"owner": Path(__file__).resolve()}
    for name in ("run_badminton_owned_stage.py", "run_badminton_full_native.py", "run_badminton_pilot.py",
                 "run_badminton_full_teachers.py", "run_badminton_full_fits.py"):
        files[name] = PROJ_ROOT / "tools" / name
    for name in ("windows_gpu_job.py", "guarded_leg_execution.py", "guarded_leg_gpu.py", "repair_execution.py"):
        files[name] = PROJ_ROOT / "hmr4d/lidar_mocap" / name
    return files


def _absolute(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or ":" in str(path)[2:]:
        raise ValueError("Owned stage paths require exact absolute names without traversal or alternate streams")
    return path.resolve()


def inspect_descriptor(path, sha256):
    proof = dict(path=str(_absolute(path)), sha256=sha256)
    descriptor = bound_json(proof)
    fields = {"schema", "root", "stage", "pilot_input", "manifest", "source_input", "teacher_sampling_input",
              "lease_path", "progress_output", "resume_progress", "absolute_hard_deadline_utc", "code"}
    if set(descriptor) != fields or descriptor["schema"] != "gvhmr-owned-full-stage-descriptor-v1":
        raise ValueError("Use the explicit closed owned full-stage descriptor")
    root, stage = _absolute(descriptor["root"]), descriptor["stage"]
    if root == PROJ_ROOT / "outputs" or not root.is_relative_to(PROJ_ROOT / "outputs"):
        raise ValueError("Owned stage requires a separately prepared outputs root")
    if stage not in ("teacher", "poses", "fit"):
        raise ValueError("Only fixed teacher, poses and fit stages are supported")
    for role, expected in (("pilot_input", root / "pilot_input.json"),
                           ("source_input", root / "repair/new_source_input.json")):
        if descriptor[role]["path"] != str(expected):
            raise ValueError("An owned stage input points outside its exact prepared root")
        check_binding(descriptor[role])
    pilot = bound_json(descriptor["pilot_input"])
    if (pilot.get("schema") != "badminton-native-rally-fit-v1" or pilot.get("review_only") is not True
            or pilot.get("training_published") is not False or pilot.get("manifest") != descriptor["manifest"]):
        raise ValueError("Owned stage is not the frozen full-native input")
    check_binding(descriptor["manifest"])
    expected_code = {role: binding(file) for role, file in implementation_files().items()}
    if descriptor["code"] != expected_code:
        raise ValueError("Owned stage implementation or fixed entry bytes changed")
    lease_path = _absolute(descriptor["lease_path"])
    if lease_path.parent != root / "gpu_leases" or not re.fullmatch(r"[A-Za-z0-9_-]+\.json", lease_path.name):
        raise ValueError("Use a distinct exact local lease in this root's gpu_leases directory")
    sampling = descriptor["teacher_sampling_input"]
    if stage == "teacher":
        if not sampling or sampling["path"] != str(root / "teacher_sampling_input.json"):
            raise ValueError("Teacher stage requires its explicit frozen paired sampling input")
        check_binding(sampling)
    elif sampling is not None:
        raise ValueError("Sampling input belongs only to the fixed teacher stage")
    progress = descriptor["progress_output"]
    resume = descriptor["resume_progress"]
    if stage in ("teacher", "fit"):
        progress = _absolute(progress)
        if progress.parent != root or not re.fullmatch(r"full_" + stage + r"_progress\.[A-Za-z0-9_-]+\.json",
                                                       progress.name):
            raise ValueError("Use a distinct stage-specific progress file in the exact prepared root")
        if resume is not None:
            check_binding(resume)
            if _absolute(resume["path"]).parent != root or _absolute(resume["path"]) == progress:
                raise ValueError("Resume must be a separately bound checkpoint of this same run")
    elif progress is not None or resume is not None:
        raise ValueError("The existing poses stage uses its original progress interface")
    hard = utc_time(descriptor["absolute_hard_deadline_utc"])
    return dict(descriptor=descriptor, binding=proof, root=root, stage=stage,
                lease_path=lease_path, hard_deadline=hard, code=expected_code)


def load_authority(descriptor_path, descriptor_sha256, lease_path, lease_sha256, *, require_fresh=True):
    checked = inspect_descriptor(descriptor_path, descriptor_sha256)
    lease_binding = dict(path=str(_absolute(lease_path)), sha256=lease_sha256)
    lease = bound_json(lease_binding)
    start, deadline = utc_time(lease["approved_start_utc"]), utc_time(lease["approved_deadline_utc"])
    now = datetime.now(timezone.utc)
    if (Path(lease_binding["path"]) != checked["lease_path"]
            or lease.get("schema") != "gvhmr-owned-full-stage-local-lease-v1"
            or lease.get("status") != "held" or lease.get("owner") != "GVHMR"
            or lease.get("root") != str(checked["root"]) or lease.get("stage") != checked["stage"]
            or lease.get("owned_stage_descriptor") != checked["binding"]
            or lease.get("manifest_sha256") != checked["descriptor"]["manifest"]["sha256"]
            or lease.get("training_published") is not False or lease.get("cuda_visible_devices") != "0"
            or lease.get("max_concurrent_cuda_trees") != 1
            or not start <= now < deadline <= min(start + timedelta(minutes=120), checked["hard_deadline"])):
        raise ValueError("The held source-bound local lease is outside its exact stage or fixed <=120 minute window")
    desktop = bound_json(lease["desktop_allowance"])
    age = (now - utc_time(desktop["observed_utc"])).total_seconds()
    if (lease.get("desktop_allowance_fixture_only") is not False
            or desktop.get("request") != checked["binding"] or age < 0 or (require_fresh and age > 900)
            or utc_time(desktop["valid_until_utc"]) < deadline):
        raise ValueError("Owned stage requires a fresh exact desktop capture covering the entire local lease")
    checked.update(lease=lease, lease_binding=lease_binding, deadline=deadline,
                   desktop=lease["desktop_allowance"])
    return checked


def stage_command(checked):
    descriptor = checked["descriptor"]
    command = [str(PROJ_ROOT / ".venv/Scripts/python.exe"), "-u",
               str(PROJ_ROOT / "tools/run_badminton_full_native.py"), "--root", str(checked["root"]),
               "--stage", checked["stage"], "--gpu-lease", str(checked["lease_path"]), "--threads", "1"]
    if checked["stage"] == "teacher":
        command += ["--teacher-sampling-input", descriptor["teacher_sampling_input"]["path"]]
    if checked["stage"] in ("teacher", "fit"):
        command += ["--safety-seconds", "300", "--progress-output", descriptor["progress_output"]]
        if descriptor["resume_progress"]:
            command += ["--resume-progress", descriptor["resume_progress"]["path"]]
    return command


def _remaining(checked):
    seconds = (checked["deadline"] - datetime.now(timezone.utc)).total_seconds() - MARGIN
    if seconds <= 0:
        raise TimeoutError("Owned full stage reached its absolute shutdown margin")
    return seconds


def _refresh(context):
    check_binding(context["self_binding"])
    checked = load_authority(context["descriptor"]["path"], context["descriptor"]["sha256"],
                             context["lease"]["path"], context["lease"]["sha256"], require_fresh=False)
    if checked["deadline"].isoformat() != context["deadline"]:
        raise ValueError("Owned full-stage authority changed")
    _remaining(checked)
    return checked


def _gpu(checked, job=None, observed=None):
    observation = gpu_process_observation(checked["desktop"])
    foreign = foreign_compute_pids(observation["blocking_compute_pids"], job.pids() if job else [], observed or {})
    if foreign:
        raise RuntimeError("Foreign CUDA ownership blocks this stage; only its own private Job may be stopped")
    return observation


def _context(path, sha256):
    proof = dict(path=str(_absolute(path)), sha256=sha256)
    context = bound_json(proof)
    checked = load_authority(context["descriptor"]["path"], context["descriptor"]["sha256"],
                             context["lease"]["path"], context["lease"]["sha256"])
    directory = Path(proof["path"]).parent
    nonce = context["nonce"]
    if (not re.fullmatch("[0-9a-f]{32}", nonce) or directory != checked["root"] / "owned_gpu_stages" / nonce
            or context["job_name"] != "Local\\GVHMROwnedFullStage_" + nonce
            or context["command"] != stage_command(checked)
            or context["deadline"] != checked["deadline"].isoformat()):
        raise ValueError("Watchdog context is not this exact source/stage/command/private Job")
    command = context["driver"]["command"]
    expected = ["-u", str(PROJ_ROOT / "tools/run_badminton_owned_stage.py"), "--execute",
                "--descriptor", context["descriptor"]["path"], "--descriptor-sha256", context["descriptor"]["sha256"],
                "--lease", context["lease"]["path"], "--lease-sha256", context["lease"]["sha256"]]
    if command[1:] != expected or not identity_alive(context["driver"]):
        raise ValueError("Owned stage watchdog lacks its exact live driver command and PID/create-time")
    context["self_binding"] = proof
    return context, checked, directory


def watchdog(context_path, context_sha256):
    cpu_controller()
    context, checked, directory = _context(context_path, context_sha256)
    observed = {}
    with WindowsGPUJob(context["job_name"], create=False) as job, job.deadline_guard(_remaining(checked)):
        own = psutil.Process()
        write_progress_json(directory / "watchdog_ready.json", dict(
            pid=own.pid, create_time=own.create_time(), command=own.cmdline(), ready=True,
            context_sha256=context_sha256, actual_affinity=own.cpu_affinity()))
        try:
            while True:
                checked = _refresh(context)
                if not identity_alive(context["driver"]):
                    raise RuntimeError("Owned stage controller exited or its PID was reused")
                identities = job.identities()
                for item in identities:
                    observed[item["pid"], item["create_time"]] = item
                observation = _gpu(checked, job, observed)
                if (directory / "abort.json").exists():
                    raise RuntimeError("Owned stage controller requested private Job cleanup")
                if (directory / "child_exit.json").exists() and not job.pids():
                    write_progress_json(directory / "watchdog_exit.json", dict(
                        state="owned_job_exited", job_empty=True, context_sha256=context_sha256,
                        observed_process_identities=list(observed.values()), gpu_observation=observation,
                        no_foreign_process_terminated=True))
                    return
                time.sleep(.2)
        except BaseException as error:
            job.terminate()
            job.wait_empty()
            write_progress_json(directory / "watchdog_failure.json", dict(
                state="owned_job_exited_after_guard_failure", error=repr(error), job_empty=True,
                context_sha256=context_sha256, observed_process_identities=list(observed.values()),
                no_foreign_process_terminated=True))
            raise


def execute(descriptor_path, descriptor_sha256, lease_path, lease_sha256):
    cpu_controller()
    checked = load_authority(descriptor_path, descriptor_sha256, lease_path, lease_sha256)
    if _remaining(checked) < 300:
        raise TimeoutError("Reserve at least 300 actual seconds before launching the fixed full stage")
    if checked["descriptor"]["progress_output"] and Path(checked["descriptor"]["progress_output"]).exists():
        raise ValueError("Preserve existing batch progress; use a distinct bound output/checkpoint")
    _gpu(checked)
    nonce = uuid.uuid4().hex
    folder = checked["root"] / "owned_gpu_stages"
    folder.mkdir(parents=True, exist_ok=True)
    claim = folder / "claims" / (checked["lease_path"].stem + ".json")
    if claim.exists():
        raise ValueError("This exact local lease was already attempted; preserve its evidence and use new authority")
    lock = folder / "execution.lock.json"
    own = psutil.Process()
    driver = dict(pid=own.pid, create_time=own.create_time(), command=own.cmdline())
    with lock.open("x", encoding="utf-8") as handle:
        json.dump(dict(nonce=nonce, driver=driver, descriptor=checked["binding"],
                       lease=checked["lease_binding"]), handle)
    claim.parent.mkdir(parents=True, exist_ok=True)
    with claim.open("x", encoding="utf-8") as handle:
        json.dump(dict(nonce=nonce, descriptor=checked["binding"], lease=checked["lease_binding"]), handle)
    guard = child = None
    directory = folder / nonce
    directory.mkdir(exist_ok=False)
    context = dict(nonce=nonce, driver=driver, descriptor=checked["binding"], lease=checked["lease_binding"],
                   job_name="Local\\GVHMROwnedFullStage_" + nonce, command=stage_command(checked),
                   deadline=checked["deadline"].isoformat())
    path = directory / "context.json"
    write_progress_json(path, context)
    context["self_binding"] = binding(path)
    observed = {}
    empty = False
    with WindowsGPUJob(context["job_name"]) as job, job.deadline_guard(_remaining(checked)) as timer:
        try:
            command = [sys.executable, "-u", str(PROJ_ROOT / "tools/run_badminton_owned_stage.py"),
                       "--watchdog", "--context", str(path), "--context-sha256", context["self_binding"]["sha256"]]
            with (directory / "watchdog.log").open("ab") as log:
                guard = subprocess.Popen(command, cwd=PROJ_ROOT, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS, close_fds=True)
            ready_path = directory / "watchdog_ready.json"
            limit = time.monotonic() + min(60, _remaining(checked))
            while not ready_path.exists():
                _refresh(context)
                if guard.poll() is not None or time.monotonic() >= limit:
                    raise RuntimeError("Independent owned-stage watchdog did not actually become ready")
                time.sleep(.05)
            ready = read_json(ready_path)
            if (ready.get("context_sha256") != context["self_binding"]["sha256"]
                    or ready.get("ready") is not True or not identity_alive(ready)):
                raise ValueError("Owned-stage watchdog readiness is stale or belongs to another context")
            _refresh(context)
            _gpu(checked)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            with (directory / "stage.log").open("ab") as log:
                _refresh(context)
                if timer["fired"] or timer["error"]:
                    raise TimeoutError("Owned Job deadline fired before stage launch")
                child, launch = job.start(context["command"], cwd=PROJ_ROOT, env=environment, log=log, affinity=[0, 1])
                _refresh(context)
                write_progress_json(directory / "launch.json", launch)
                while child.poll() is None or job.pids():
                    if child.poll() not in (None, 0):
                        raise RuntimeError("Fixed stage launcher failed while owned descendants remained")
                    _refresh(context)
                    if guard.poll() is not None or not identity_alive(ready):
                        raise RuntimeError("Independent owned-stage watchdog exited during the stage")
                    for item in job.identities():
                        observed[item["pid"], item["create_time"]] = item
                    time.sleep(.2)
            job.wait_empty()
            empty = True
            write_progress_json(directory / "child_exit.json", dict(exit_code=child.returncode, job_empty=True))
            guard.wait(timeout=min(8, _remaining(checked)))
            if guard.returncode or not (directory / "watchdog_exit.json").exists():
                raise RuntimeError("Independent watchdog did not confirm actual owned Job exit")
            if child.returncode:
                raise RuntimeError("Fixed full stage failed; preserve the original log and completed checkpoints")
            result = dict(schema="gvhmr-owned-full-stage-release-v1", status="released", state="stage_cli_exit_zero",
                descriptor=checked["binding"], local_lease=checked["lease_binding"], context=context["self_binding"],
                launcher=launch, job_empty=True, observed_process_identities=list(observed.values()),
                watchdog_exit=binding(directory / "watchdog_exit.json"), no_foreign_process_terminated=True,
                released_utc=datetime.now(timezone.utc).isoformat(), training_published=False)
            write_progress_json(directory / "release_receipt.json", result)
            return result
        except BaseException as error:
            job.terminate()
            job.wait_empty()
            empty = True
            write_progress_json(directory / "abort.json", dict(context_sha256=context["self_binding"]["sha256"]))
            write_progress_json(directory / "failure.json", dict(
                state="owned_job_exited_after_failure", error=repr(error), job_empty=True,
                descriptor=checked["binding"], local_lease=checked["lease_binding"],
                observed_process_identities=list(observed.values()), no_foreign_process_terminated=True))
            raise
        finally:
            if child is not None:
                child.wait(timeout=6)
            if guard is not None and guard.poll() is None:
                try:
                    guard.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    guard.terminate()  # The exact live Popen handle; never a PID/name scan.
                    guard.wait(timeout=5)
            if empty and read_json(lock).get("nonce") == nonce:
                lock.unlink()
