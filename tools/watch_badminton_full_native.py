"""Wait for a valid frozen full-native intake, then prepare it on CPU only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from hmr4d import PROJ_ROOT


def fingerprint(path):
    """Cheap dependency state; unchanged failures do not repeatedly hash videos."""
    if not path.is_file():
        return None
    blob = path.read_bytes()
    signature = [hashlib.sha256(blob).hexdigest()]
    try:
        spec = json.loads(blob)
        if spec.get("schema") == "badminton-native-no-fit-needed-v1":
            paths = {Path(spec[k]["path"]) for k in ("source_index", "coverage_audit")}
            paths.update(Path(item["path"]) for item in spec["evidence_files"])
        else:
            paths = {Path(spec[k]["path"]) for k in ("review_authorization", "coverage_audit")}
            paths.update(Path(spec["data_root"]) / item["file"] for item in spec["source_indexes"].values())
            paths.update(Path(spec["calibration_dir"]) / name for name in spec["bundle_sha256"])
            if spec.get("data_relocations") is not None:
                paths.add(Path(spec["data_root"]) / spec["data_relocations"]["file"])
            paths.update(Path(item["path"]) for clip in spec["clips"] for item in clip["evidence_files"])
        for dependency in sorted(paths):
            try:
                stat = dependency.stat()
                signature.append((str(dependency), stat.st_size, stat.st_mtime_ns))
            except FileNotFoundError:
                signature.append((str(dependency), None, None))
    except (KeyError, TypeError, json.JSONDecodeError):
        pass
    return signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = args.manifest.resolve()
    if (not root.is_relative_to(PROJ_ROOT / "outputs") or root == PROJ_ROOT / "outputs"
            or args.poll_seconds < 5 or args.threads < 1):
        parser.error("Use an independent outputs root, at least five-second polling and positive threads")
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    psutil.Process().cpu_affinity([0, 1])
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "intake_watch.json"
    state = dict(state="waiting_for_full_native_manifest", manifest_path=str(manifest),
        started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), cpu_only=True,
        gpu_started=False, training_published=False, hidden=True, cpu_affinity=[0, 1], priority="BelowNormal")

    def save():
        from hmr4d.lidar_mocap.repair_execution import write_progress_json
        write_progress_json(status_path, state)

    save()
    previous, previous_no_fit = None, None
    while True:
        suffix = ".full-native-rallies.json"
        if manifest.name.endswith(suffix):
            no_fit_path = manifest.with_name(manifest.name[:-len(suffix)] + ".no-fit-needed.json")
            if no_fit_path.is_file():
                if manifest.is_file():
                    raise ValueError("One source declares both fitted windows and no-fit-needed; reconcile the intake")
                no_fit_signature = fingerprint(no_fit_path)
                if no_fit_signature == previous_no_fit:
                    time.sleep(args.poll_seconds)
                    continue
                previous_no_fit = no_fit_signature
                try:
                    from hmr4d.lidar_mocap.pilot import binding, validate_no_fit_source
                    proof = validate_no_fit_source(binding(no_fit_path), source_id=manifest.name[:-len(suffix)])
                except Exception as error:
                    state.update(state="waiting_for_valid_no_fit_evidence", error_type=type(error).__name__, error=str(error))
                    save()
                    time.sleep(args.poll_seconds)
                    continue
                state.update(state="no_fit_needed", no_fit_receipt=binding(no_fit_path),
                    reason=proof["reason"], completed_utc=datetime.now(timezone.utc).isoformat(), gpu_started=False)
                save()
                print(json.dumps(dict(state="no_fit_needed", gpu_started=False)), flush=True)
                return
        current = fingerprint(manifest)
        if current is None or current == previous:
            time.sleep(args.poll_seconds)
            continue
        previous = current
        state.update(state="checking_frozen_native_inputs", last_checked_utc=datetime.now(timezone.utc).isoformat())
        save()
        try:
            from hmr4d.lidar_mocap.full_native import inspect_full_native_manifest
            inspected = inspect_full_native_manifest(manifest)
        except Exception as error:
            state.update(state="waiting_for_valid_full_native_inputs", error_type=type(error).__name__, error=str(error))
            save()
            continue
        frozen_path = root / "pilot_input.json"
        if frozen_path.exists():
            frozen = json.loads(frozen_path.read_bytes())
            if frozen["manifest"] != inspected["manifest"]:
                raise ValueError("This output already freezes another manifest; select an independent new root")
        state.update(state="preparing_cpu", inspection=inspected)
        state.pop("error_type", None)
        state.pop("error", None)
        command = [sys.executable, "-u", str(PROJ_ROOT / "tools/run_badminton_full_native.py"),
            "--manifest", str(manifest), "--root", str(root), "--stage", "prepare", "--threads", str(args.threads)]
        timer = time.perf_counter()
        with (root / "intake_prepare_cpu.log").open("ab") as log:
            child = subprocess.Popen(command, cwd=PROJ_ROOT, env=os.environ.copy(), stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            state["prepare_cli_pid"] = child.pid
            save()
            code = child.wait()
        state.update(state="ready_cpu_only" if code == 0 else "cpu_prepare_failed", prepare_exit_code=code,
            prepare_cli_seconds=time.perf_counter() - timer, completed_utc=datetime.now(timezone.utc).isoformat())
        save()
        if code:
            raise RuntimeError("Frozen CPU intake prepare failed; see intake_prepare_cpu.log")
        print(json.dumps(dict(state=state["state"], gpu_started=False)), flush=True)
        return


if __name__ == "__main__":
    main()
