"""Explicit bounded review-only rally pilot; no formal training publication."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.pilot import prepare_pilot, run_pilot_stage, serve_pilot
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--stage", choices=("prepare", "teacher", "poses", "identity", "fit", "verify", "viewer"), default="prepare"
    )
    p.add_argument("--gpu-lease", type=Path)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--port", type=int, default=8781)
    a = p.parse_args()
    if a.threads < 1:
        p.error("--threads must be positive")
    try:
        if a.stage == "prepare":
            if a.manifest is None:
                p.error("prepare requires the exact new pilot --manifest")
            result = prepare_pilot(a.manifest, a.root)
        elif a.stage == "viewer":
            serve_pilot(a.root, a.port)
            return
        else:
            result = run_pilot_stage(a.root, a.stage, gpu_lease=a.gpu_lease, threads=a.threads)
    except Exception as error:
        root = a.root.resolve()
        if root.is_relative_to(PROJ_ROOT / "outputs") and (root / "pilot_input.json").is_file():
            progress = read_json(root / "pilot_progress.json")
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
