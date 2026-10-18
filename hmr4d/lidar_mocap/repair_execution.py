"""Keep repair timing with the run, independently of disposable console logs."""

import json
from pathlib import Path
import time


def require_gpu_lease(path, owner="GVHMR"):
    if path is not None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("status") != "held" or data.get("owner") != owner:
            raise ValueError("The shared GPU lease is not held by " + owner)


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
