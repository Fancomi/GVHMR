"""Finish post-fit leg repair, revalidate original members and rebuild all ten clips."""

import argparse
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

import psutil

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json


def active_baseline():
    for process in psutil.process_iter(["cmdline"]):
        cmd = process.info["cmdline"] or []
        if any(Path(arg).name in ["run_badminton_repair_windows.py", "repair_badminton_body_labels.py"] for arg in cmd):
            return True
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--observer-max-new-frames", type=int, default=500)
    args = p.parse_args()
    if args.threads < 1 or args.observer_max_new_frames < 1:
        p.error("Thread and observer limits must be positive")
    root = args.root.resolve()
    labels = root / "final_labels"
    input_plan = read_json(labels / "repair_plan.json")
    if read_json(input_plan["source_union"]).get("schema") == "badminton-new-source-repair-input-v1":
        from hmr4d.lidar_mocap.new_source_completion import finish_new_sources
        finish_new_sources(root, threads=args.threads, observer_max_new_frames=args.observer_max_new_frames)
        return
    expected = len(read_json(labels / "repair_plan.json")["jobs"])
    while True:
        paths = [
            labels / "repair_receipt.json",
            labels / "native_verification.json",
            root / "evaluation/metrics.json",
            root / "leg_refinement_proposals/receipt.json",
            root / "comparison/manifest.json",
        ]
        fit, contract, metrics, proposals, viewer = [read_json(path) if path.exists() else {} for path in paths]
        if (
            fit.get("windows") == expected
            and contract.get("passed")
            and contract.get("verified_windows") == expected
            and metrics.get("verified_windows") == expected
            and metrics.get("state") == "complete"
            and proposals.get("state") == "complete"
            and proposals.get("windows_complete") == expected
            and viewer.get("state") == "complete"
            and not active_baseline()
        ):
            break
        time.sleep(5)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", PYTHONFAULTHANDLER="1")

    def run(script, *arguments):
        log = root / (Path(script).stem + "_final.log")
        with log.open("wb") as handle:
            result = subprocess.run(
                [sys.executable, "-u", str(PROJ_ROOT / "tools" / script), "--root", str(root), *arguments],
                env=env,
                cwd=PROJ_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"{script} failed ({result.returncode}); see {log}")
        print(dict(stage=script, state="complete"), flush=True)

    archive = root / "leg_refinement_before"
    archive.mkdir(exist_ok=True)
    for source in paths[:3]:
        dest = archive / source.name
        if not dest.exists():
            shutil.copyfile(source, dest)
    if not (labels / "post_refinement.json").exists():
        run("select_badminton_leg_refinement.py")
    legs_path = root / "leg_refinement_receipt.json"
    legs = read_json(legs_path) if legs_path.exists() else {}
    if legs.get("state") != "complete":
        run("apply_badminton_leg_refinement.py", "--prepare-only")
        run("apply_badminton_leg_refinement.py")
    run("verify_badminton_native_repair.py")
    aggregate_log = root / "aggregate_contract_final.log"
    with aggregate_log.open("wb") as handle:
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                str(PROJ_ROOT / "tools/verify_badminton_dataset.py"),
                "--output-root",
                str(labels),
                "--samples-per-session",
                "64",
            ],
            env=env,
            cwd=PROJ_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"Aggregate contract verification failed: {aggregate_log}")
    run("report_badminton_native_repair.py")
    run("render_badminton_repair_comparison.py")
    run("finalize_badminton_native_repair.py")
    complete = root / "completion_receipt.json"
    if read_json(complete)["state"] != "complete":
        raise ValueError("Final validation receipt is missing")
    write_json(
        root / "end_to_end_receipt.json",
        dict(state="complete", completion_receipt=str(complete), sha256=file_sha(complete), training_published=False),
    )
    print("Repair, all-member verification, native metrics and ten continuous clips are complete", flush=True)


if __name__ == "__main__":
    main()
