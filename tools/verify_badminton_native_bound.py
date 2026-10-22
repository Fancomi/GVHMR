"""Run the existing native verifier with an actual CPU model/runtime witness."""

import os

os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import argparse
import json
from pathlib import Path
import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--verified-formal-cache",
        action="store_true",
        help="Recheck changed formal windows and require witnessed actual/derived cache coverage",
    )
    args = parser.parse_args()
    process = psutil.Process()
    process.cpu_affinity([26, 27])
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    if args.verified_formal_cache:
        from hmr4d.lidar_mocap.formal_native_derive import run_required_actual_native

        result = run_required_actual_native(args.root, args.threads, postleg=True)
    else:
        from hmr4d.lidar_mocap.native_witness import run_bound_native_verification

        result = run_bound_native_verification(args.root, args.threads)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
