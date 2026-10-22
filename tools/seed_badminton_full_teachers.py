"""Reuse native-verified pilot teachers after checking exact full-run inputs."""

import argparse
from pathlib import Path

from hmr4d.lidar_mocap.full_native import seed_full_teacher_caches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reuse-run", type=Path, action="append", required=True)
    args = parser.parse_args()
    result = seed_full_teacher_caches(args.root, args.reuse_run)
    print(dict(reused_windows=result["reused_windows"], camera_windows=result["reused_camera_windows"]), flush=True)


if __name__ == "__main__":
    main()
