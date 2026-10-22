"""Explicit bounded review-only rally pilot; no formal training publication."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.pilot import prepare_pilot, prepare_full_native_rallies, run_pilot_stage, serve_pilot
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json
from hmr4d.lidar_mocap.relocations import load_run_relocations, load_pilot_run_relocations


def main(*, full_native=False):
    p = argparse.ArgumentParser(description=(
        "Explicit full native rally execution; fitting does not publish training data." if full_native else __doc__
    ))
    p.add_argument("--manifest", type=Path)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--stage", choices=("prepare", "teacher", "poses", "identity", "fit", "verify", "viewer")
        + (("aggregate",) if full_native else ()), default="prepare"
    )
    p.add_argument("--gpu-lease", type=Path)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--port", type=int, default=8781)
    p.add_argument("--data-relocations", type=Path)
    p.add_argument("--data-relocations-sha256")
    p.add_argument("--viewer-output", type=Path)
    if full_native:
        p.add_argument("--safety-seconds", type=int, default=300)
        p.add_argument("--progress-output", type=Path)
        p.add_argument("--resume-progress", type=Path)
    a = p.parse_args()
    if a.threads < 1:
        p.error("--threads must be positive")
    if (a.data_relocations is None) != (a.data_relocations_sha256 is None):
        p.error("--data-relocations and --data-relocations-sha256 must be supplied together")
    if a.stage != "viewer" and (a.data_relocations is not None or a.viewer_output is not None):
        p.error("Runtime relocation flags apply to viewer; new source preparation binds its manifest inputs")
    try:
        if a.stage == "prepare":
            if a.manifest is None:
                p.error("prepare requires the exact new pilot --manifest")
            prepare = prepare_full_native_rallies if full_native else prepare_pilot
            result = prepare(a.manifest, a.root)
        elif a.stage == "viewer":
            if full_native and read_json(a.root / "pilot_input.json")["schema"] != "badminton-native-rally-fit-v1":
                p.error("Select the separately prepared full-native run")
            labels = a.root / "repair/final_labels"
            relocations = (
                load_pilot_run_relocations(labels)
                if a.data_relocations is None
                else load_run_relocations(labels, a.data_relocations, a.data_relocations_sha256)
            )
            serve_pilot(a.root, a.port, relocations=relocations, viewer_output=a.viewer_output)
            return
        else:
            if full_native and read_json(a.root / "pilot_input.json")["schema"] != "badminton-native-rally-fit-v1":
                p.error("Select the separately prepared full-native run")
            if full_native and a.stage == "teacher":
                if a.gpu_lease is None:
                    p.error("Full teacher execution requires its separate held GPU lease")
                from run_badminton_full_teachers import run_full_teachers
                result = run_full_teachers(a)
            elif full_native and a.stage == "fit":
                if a.gpu_lease is None:
                    p.error("Full fitting requires its separate held GPU lease")
                from run_badminton_full_fits import run_full_fits
                result = run_full_fits(a)
            elif full_native and a.stage == "aggregate":
                from run_badminton_full_fits import aggregate_full_fits
                result = aggregate_full_fits(a)
            else:
                result = run_pilot_stage(a.root, a.stage, gpu_lease=a.gpu_lease, threads=a.threads)
    except Exception as error:
        root = a.root.resolve()
        if root.is_relative_to(PROJ_ROOT / "outputs") and (root / "pilot_input.json").is_file():
            progress_path = root / "pilot_progress.json"
            progress = read_json(progress_path) if progress_path.exists() else dict(
                review_only=True, training_published=False
            )
            progress.update(
                stage="failed",
                failed_stage=a.stage,
                error_type=type(error).__name__,
                error=str(error),
                failed_utc=datetime.now(timezone.utc).isoformat(),
            )
            write_json(root / "pilot_progress.json", progress)
        raise
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
