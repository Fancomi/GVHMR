"""Freeze completed formal pre-leg evidence on the reserved CPU cores."""

import os

os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")

import argparse
import json
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Explicit formal run containing repair/")
    parser.add_argument("--numeric-import", type=Path, required=True)
    parser.add_argument("--numeric-import-sha256", required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    process = psutil.Process()
    process.cpu_affinity([26, 27])
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if process.cpu_affinity() != [26, 27] or process.nice() != psutil.BELOW_NORMAL_PRIORITY_CLASS:
        raise RuntimeError("Reserved E26/27 BelowNormal CPU allocation was not applied")
    os.environ.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    import torch

    torch.set_num_threads(args.threads)
    from hmr4d.lidar_mocap.formal_baseline import freeze_formal_baseline

    receipt = freeze_formal_baseline(args.root, args.numeric_import, args.numeric_import_sha256)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
