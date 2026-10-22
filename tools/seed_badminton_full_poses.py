"""Reuse identical native 2D observations, retaining original scope provenance."""

import argparse
from pathlib import Path

from hmr4d.lidar_mocap.full_native import seed_full_native_poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reuse-run", type=Path, action="append", required=True)
    args = parser.parse_args()
    result = seed_full_native_poses(args.root, args.reuse_run)
    print(dict(reused_native_camera_frames=result["reused_native_camera_frames"], state=result["state"]), flush=True)


if __name__ == "__main__":
    main()
