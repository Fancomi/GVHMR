"""Run a frozen fresh-source teacher plan serially with native-process isolation."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.body_repair import published_windows
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json
from hmr4d.lidar_mocap.repair_execution import require_gpu_lease


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gpu-lease", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    if args.threads < 1 or args.attempts < 1:
        parser.error("Thread and attempt limits must be positive")
    root, source = args.root.resolve(), args.source_root.resolve()
    input_path = root / "new_source_input.json"
    plan = read_json(input_path)
    if plan.get("schema") != "badminton-new-source-repair-input-v1" or Path(plan["source_root"]).resolve() != source:
        raise ValueError("Select the explicit frozen fresh-source root")
    jobs = published_windows(input_path)
    input_sha = file_sha(input_path)
    logs = root / "initial_teacher/logs"
    logs.mkdir(parents=True, exist_ok=True)
    progress_path = root / "initial_teacher_progress.json"
    progress = dict(
        state="running",
        source_input=str(input_path),
        source_input_sha256=input_sha,
        calibration_digest=plan["calibration_digest"],
        started_utc=datetime.now(timezone.utc).isoformat(),
        windows_total=len(jobs),
        windows_complete=0,
        windows={},
        training_published=False,
    )
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
        PYTHONFAULTHANDLER="1",
    )
    write_json(progress_path, progress)
    for cursor, (sid, window) in enumerate(jobs, 1):
        require_gpu_lease(args.gpu_lease)
        if file_sha(input_path) != input_sha:
            raise ValueError("Fresh source input changed during production")
        command = [
            sys.executable,
            "-u",
            str(PROJ_ROOT / "tools/prepare_badminton_new_source_repair.py"),
            "--source-root",
            str(source),
            "--root",
            str(root),
            "--stage",
            "teacher",
            "--source",
            sid,
            "--window",
            window.name,
            "--threads",
            str(args.threads),
        ]
        started = time.perf_counter()
        success = False
        for attempt in range(1, args.attempts + 1):
            require_gpu_lease(args.gpu_lease)
            log = logs / f"{sid}_{window.name}_attempt{attempt}.log"
            with log.open("ab") as handle:
                result = subprocess.run(command, cwd=PROJ_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            print(dict(source=sid, window=window.name, attempt=attempt, exit_code=result.returncode), flush=True)
            manifest_path = window / "teacher/manifest.json"
            if result.returncode == 0 and manifest_path.exists():
                manifest = read_json(manifest_path)
                success = True
                break
        if not success:
            progress.update(state="failed", failed_window=str(window), last_log=str(log))
            write_json(progress_path, progress)
            raise RuntimeError(f"Fresh teacher window failed after isolated retries: {window}")
        progress["windows"][sid + ":" + window.name] = dict(
            teacher_manifest=str(manifest_path),
            sha256=file_sha(manifest_path),
            tracks=len(manifest["tracks"]),
            observed_native_frames=manifest["frame_count"],
            worker_wall_seconds=time.perf_counter() - started,
        )
        progress.update(
            windows_complete=cursor, last_window=str(window), updated_utc=datetime.now(timezone.utc).isoformat()
        )
        write_json(progress_path, progress)
        print(dict(completed=cursor, total=len(jobs), tracks=len(manifest["tracks"])), flush=True)
    progress.update(state="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(progress_path, progress)
    print("All frozen fresh-source teachers complete; no body fitting or training publication performed", flush=True)


if __name__ == "__main__":
    main()
