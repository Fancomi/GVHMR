"""Resume the sealed repair with per-window native-process isolation."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2")
    args = p.parse_args()
    root = args.root.resolve()
    labels = root / "final_labels"
    plan = read_json(labels / "repair_plan.json")
    command = [
        sys.executable,
        "-u",
        str(PROJ_ROOT / "tools/repair_badminton_body_labels.py"),
        "--stage",
        "run",
        "--output-root",
        str(labels),
        "--seed-teacher-root",
        str(PROJ_ROOT / "outputs/badminton_body_repair_v1/labels"),
        "--native-poses-root",
        str(root),
        "--joint-ray-filter",
        "--lower-limb-reprojection-weight",
        "2",
        "--identity-decisions",
        str(root / "identity_evidence/identity_decisions.json"),
    ]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", PYTHONFAULTHANDLER="1")
    for cursor, job in enumerate(plan["jobs"], 1):
        sid = job["source_id"]
        window = Path(job["source_window"]).name
        dest = labels / "sessions" / sid / "windows" / window
        if sid == "20261012_163452":
            # The bulk auxiliary worker owns only this source. Stop and confirm
            # its termination before the primary can touch any unfinished window.
            for process in psutil.process_iter(["name", "cmdline"]):
                try:
                    cmd = process.info["cmdline"] or []
                    if (
                        process.info["name"].lower() == "python.exe"
                        and "--partition-worker" in cmd
                        and "--window" not in cmd
                        and "20261012_163452" in cmd
                    ):
                        process.terminate()
                        try:
                            process.wait(timeout=20)
                        except psutil.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=20)
                except psutil.NoSuchProcess:
                    pass
        if not (dest / "fit_report.json").exists():
            success = False
            for attempt in range(1, 4):
                log = root / f"window_{sid}_{window}_attempt{attempt}.log"
                with log.open("wb") as handle:
                    result = subprocess.run(
                        command + ["--partition-worker", "--source", sid, "--window", window],
                        env=env,
                        cwd=PROJ_ROOT,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                print(dict(window=window, source=sid, attempt=attempt, exit_code=result.returncode), flush=True)
                if result.returncode == 0 and (dest / "fit_report.json").exists():
                    success = True
                    break
            if not success:
                raise RuntimeError(f"Window failed after three isolated processes: {sid}/{window}")
        write_json(
            labels / "repair_progress.json",
            dict(
                completed_windows=cursor,
                total_windows=len(plan["jobs"]),
                last_window=str(dest),
                process_isolation="one_window_per_native_process",
                training_published=False,
            ),
        )
    # One cached pass validates every receipt and constructs the full aggregate.
    result = subprocess.run(command, env=env, cwd=PROJ_ROOT)
    if result.returncode:
        raise RuntimeError(f"Final aggregate/resume validation failed: {result.returncode}")
    print("All windows and full aggregate complete", flush=True)


if __name__ == "__main__":
    main()
