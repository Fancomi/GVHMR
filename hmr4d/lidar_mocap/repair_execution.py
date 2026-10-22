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
