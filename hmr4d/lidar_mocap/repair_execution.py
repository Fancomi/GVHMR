"""Keep repair timing with the run, independently of disposable console logs."""

import json
import hashlib
from pathlib import Path
import time


def write_progress_json(path, value, *, attempts=10):
    """Retry transient Windows reader locks without changing atomic JSON content."""
    from .pipeline import write_json

    for attempt in range(attempts):
        try:
            return write_json(path, value)
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.025 * 2**attempt, 0.25))


def require_gpu_lease(path, owner="GVHMR"):
    if path is not None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("status") != "held" or data.get("owner") != owner:
            raise ValueError("The shared GPU lease is not held by " + owner)


def remaining_lease_seconds(lease, manifest_sha256, now=None):
    from datetime import datetime, timezone

    if (lease.get("status") != "held" or lease.get("owner") != "GVHMR"
            or lease.get("manifest_sha256") != manifest_sha256
            or lease.get("training_published") is not False):
        raise ValueError("Full stage requires its own held manifest-bound GPU lease")
    deadline = datetime.fromisoformat(lease["approved_deadline_utc"].replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("GPU deadline must specify UTC offset")
    return (deadline - (now or datetime.now(timezone.utc))).total_seconds()


def begin_isolated_batch(state, counter_key):
    """Keep old batch statistics with their old attempt when resuming windows."""
    from datetime import datetime, timezone

    keys = ("exit_code", "new_camera_windows_this_batch", "new_windows_this_batch", "batch_seconds",
            "completed_batch_utc", "started_batch_utc")
    previous = {key: state.pop(key) for key in keys if key in state}
    if previous:
        state.setdefault("previous_attempts", []).append(dict(previous, state=state.get("state"), active=state.get("active")))
    state.update({counter_key: 0, "batch_seconds": 0., "completed_batch_utc": None,
                  "started_batch_utc": datetime.now(timezone.utc).isoformat()})


def teacher_batch_metrics(current, baseline, *, now=None):
    """Derive current counts from frozen window sets, never old resumed counters."""
    from datetime import datetime, timezone

    def keys(value):
        return {(name, camera) for name, row in value.get("windows", {}).items()
                for camera in row if camera in ("cam1", "cam2")}

    done, before = keys(current), keys(baseline)
    if not before <= done:
        raise ValueError("Resumed teacher progress lost previously completed camera windows")
    start = datetime.fromisoformat(current["started_batch_utc"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(current["completed_batch_utc"].replace("Z", "+00:00")) if current.get("completed_batch_utc") else None
    terminal = current.get("state") in ("complete", "yielded_before_deadline", "failed")
    completed = end if terminal and end is not None and end >= start else None
    return dict(completed_camera_windows=len(done), baseline_completed_camera_windows=len(before),
                new_camera_windows_this_batch=len(done - before),
                batch_seconds=max(0., ((completed or now or datetime.now(timezone.utc)) - start).total_seconds()),
                completed_batch_utc=completed.isoformat() if completed else None,
                ignores_raw_resumed_previous_batch_statistics=True)


def full_native_fit_command(root, threads):
    import sys
    from hmr4d import PROJ_ROOT

    root = Path(root)
    repair = root / "repair"
    return [sys.executable, "-u", str(PROJ_ROOT / "tools/repair_badminton_body_labels.py"),
        "--union", str(repair / "new_source_input.json"), "--source-root", str(root / "source"),
        "--output-root", str(repair / "final_labels"), "--native-poses-root", str(repair),
        "--joint-ray-filter", "--identity-decisions", str(repair / "identity_evidence/identity_decisions.json"),
        "--distal-pose-weight", "0.03", "--lower-limb-reprojection-weight", "2", "--threads", str(threads)]


def prepare_full_native_body_plan(root, threads):
    """Finish the all-window CPU plan before any partition can acquire GPU."""
    import os
    import psutil
    from hmr4d import PROJ_ROOT
    from .dual_view import file_sha

    root = Path(root).resolve()
    if threads < 1 or not root.is_relative_to(PROJ_ROOT / "outputs") or root == PROJ_ROOT / "outputs":
        raise ValueError("Use the prepared full-native output and positive threads")
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity([0, 1])
    source_input = root / "repair/new_source_input.json"
    source_sha = file_sha(source_input)
    source = json.loads(source_input.read_bytes())
    path = root / "repair/final_labels/repair_plan.json"

    def check_plan():
        plan = json.loads(path.read_bytes())
        expected = {(item["source_id"], item["source_window"]) for item in source["jobs"]}
        actual = {(item["source_id"], item["source_window"]) for item in plan["jobs"]}
        if (file_sha(source_input) != source_sha or plan["source_union_sha256"] != source_sha
                or actual != expected or len(plan["jobs"]) != len(source["jobs"])
                or plan["fitting_config"]["distal_pose_prior_weight"] != .03
                or plan["fitting_config"]["lower_limb_reprojection_weight"] != 2):
            raise ValueError("Full native CPU body plan changed its complete window scope or fitting policy")
        return plan

    receipt_path = root / "body_prepare_cpu_receipt.json"
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_bytes())
        entry = previous["repair_plan"]
        if (previous.get("state") != "ready_cpu_only" or previous.get("exit_code") != 0
                or previous.get("gpu_started") is not False or previous.get("training_published") is not False
                or Path(entry["path"]).resolve() != path or file_sha(path) != entry["sha256"]):
            raise ValueError("Existing full body-plan receipt changed; preserve it and use an independent output")
        check_plan()
        return previous
    state = dict(state="preparing_full_body_plan_cpu", cpu_only=True, gpu_started=False, training_published=False)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[key] = str(threads)
    command = full_native_fit_command(root, threads) + ["--stage", "prepare", "--device", "cpu"]
    execution = run_isolated_native_worker(command, cwd=PROJ_ROOT, log_path=root / "body_prepare_cpu.log",
        state=state, state_path=root / "body_prepare_cpu_progress.json",
        active=dict(role="cpu_full_body_plan"), env=environment)
    if execution["exit_code"]:
        state.update(state="failed", **execution)
        state.pop("active", None)
        write_progress_json(root / "body_prepare_cpu_progress.json", state)
        raise RuntimeError("Full native CPU body-plan preparation failed; no GPU stage was started")
    try:
        plan = check_plan()
    except Exception as error:
        state.update(state="failed", error_type=type(error).__name__, error=str(error), **execution)
        state.pop("active", None)
        write_progress_json(root / "body_prepare_cpu_progress.json", state)
        raise
    receipt = dict(schema="badminton-full-body-prepare-v1", state="ready_cpu_only", cpu_only=True,
        gpu_started=False, training_published=False, **execution,
        cli_seconds=execution["seconds"], cli_pid=state.get("active", {}).get("cli_pid"),
        source_input=dict(path=str(source_input), sha256=source_sha),
        repair_plan=dict(path=str(path), sha256=file_sha(path)), windows=len(plan["jobs"]))
    write_progress_json(receipt_path, receipt)
    state.update(state="ready_cpu_only", **execution, repair_plan=receipt["repair_plan"])
    state.pop("active", None)
    write_progress_json(root / "body_prepare_cpu_progress.json", state)
    return receipt


def require_completed_fit_windows(root, jobs):
    """Fail before pose/model setup when a collection-only run has missing fits."""
    missing = []
    for sid, window in jobs:
        output = Path(root) / "sessions" / sid / "windows" / window.name
        if not (output / "fit_report.json").is_file() or not (output / "teacher/manifest.json").is_file():
            missing.append(sid + ":" + window.name)
    if missing:
        raise ValueError("Completed-only execution found unfinished native windows: " + ", ".join(missing))


def load_isolated_stage_progress(root, output, resume, *, default_name, release_pattern, binding_key):
    """A later GPU batch cannot rewrite the progress SHA sealed by an earlier one."""
    root = Path(root).resolve()
    path = Path(output or root / default_name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Stage progress must stay inside this independent full run")
    bindings = {}
    for receipt in root.glob(release_pattern):
        data = json.loads(receipt.read_bytes())
        entry = data.get(binding_key, data.get("progress", {}))
        if entry.get("path"):
            previous = Path(entry["path"]).resolve()
            bindings[previous] = entry["sha256"]
            if previous == path:
                raise ValueError("Completed batch progress is SHA-bound; choose a fresh output and resume the old one")
    previous = Path(resume).resolve() if resume is not None else None
    if previous is not None:
        if not previous.is_relative_to(root) or previous == path:
            raise ValueError("Resume from a separate prior progress file inside this full run")
        if previous in bindings and hashlib.sha256(previous.read_bytes()).hexdigest() != bindings[previous]:
            raise ValueError("Previously released batch progress changed")
    state = (json.loads(path.read_bytes()) if path.exists() else
             json.loads(previous.read_bytes()) if previous is not None else dict(windows={}))
    return path, state


def run_isolated_native_worker(command, *, cwd, log_path, state, state_path, active, env=None):
    """Record one hidden native subprocess and its actual Python descendants."""
    from datetime import datetime, timezone
    import os
    import subprocess
    import psutil

    timer = time.perf_counter()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        child = subprocess.Popen(command, cwd=cwd, env=env or os.environ.copy(), stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        state["active"] = dict(active, cli_pid=child.pid,
            started_utc=datetime.now(timezone.utc).isoformat(), log=str(log_path))
        write_progress_json(state_path, state)
        while child.poll() is None:
            try:
                descendants = psutil.Process(child.pid).children(recursive=True)
                state["active"]["python_pids"] = [p.pid for p in descendants if p.name().lower() == "python.exe"]
            except psutil.NoSuchProcess:
                pass
            if state.get("started_batch_utc"):
                state["batch_seconds"] = (datetime.now(timezone.utc) - datetime.fromisoformat(
                    state["started_batch_utc"].replace("Z", "+00:00"))).total_seconds()
            write_progress_json(state_path, state)
            time.sleep(1)
    return dict(exit_code=child.returncode, seconds=time.perf_counter() - timer,
                completed_utc=datetime.now(timezone.utc).isoformat(), log=str(log_path))


def native_gpu_command_matches(command, root):
    """Identify this run's known CUDA executors, excluding explicit CPU stages."""
    import os

    command = list(command or [])
    names = {Path(arg).name for arg in command}
    known = {"run_badminton_full_native.py", "run_badminton_pilot.py", "run_badminton_full_teachers.py",
             "run_badminton_full_fits.py", "prepare_badminton_new_source_repair.py",
             "run_badminton_initial_teachers.py", "repair_badminton_body_labels.py", "observe_badminton_repair_poses.py"}
    if not names & known:
        return False

    def option(name):
        for i, arg in enumerate(command):
            if arg == name and i + 1 < len(command):
                return command[i + 1]
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
        return None

    if option("--device") == "cpu" or "--aggregate-only" in command:
        return False
    if names & {"run_badminton_full_native.py", "run_badminton_pilot.py"} and option("--stage") not in ("teacher", "fit", "poses"):
        return False
    if "prepare_badminton_new_source_repair.py" in names and option("--stage") != "teacher":
        return False
    prefix = os.path.normcase(str(Path(root).resolve()))
    for arg in command:
        value = arg.split("=", 1)[1] if arg.startswith("--") and "=" in arg else arg
        value = os.path.normcase(os.path.normpath(value))
        if value == prefix or value.startswith(prefix + os.sep):
            return True
    return False


def record_execution(root: Path, *, baseline_finished: bool = False):
    path = root / "execution_timing.json"
    data = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {
            "schema": "badminton-repair-execution-v1",
            "started_unix_s": time.time(),
            "training_published": False,
        }
    )
    if baseline_finished:
        data.setdefault("baseline_finished_unix_s", time.time())
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def repair_efficiency(root: Path, fitted: int, selected_seconds: float):
    path = root / "execution_timing.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    started, baseline = data["started_unix_s"], data["baseline_finished_unix_s"]
    finished = time.time()
    if not 0 < started <= baseline <= finished:
        raise ValueError("Repair execution timestamps are out of order")
    elapsed = finished - started
    if elapsed <= 0:
        raise ValueError("Repair execution has no elapsed wall time")
    return dict(
        baseline_fit_wall_seconds=baseline - started,
        wall_seconds=elapsed,
        selected_source_seconds=selected_seconds,
        fitted_person_scans_per_wall_second=fitted / elapsed,
        scope=(
            "Elapsed from the full fit job launch through refinement, rendering and validation; "
            "pre-existing native/pixel caches are reused."
        ),
    )
