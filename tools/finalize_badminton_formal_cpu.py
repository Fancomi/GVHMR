"""Prepare genuine before-QA or verify completed guarded-leg output on CPU."""

import os

os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")

import argparse
import json
from pathlib import Path

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare-leg", "post-leg"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--leg-execution", type=Path)
    parser.add_argument("--leg-execution-sha256")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.stage == "post-leg" and (not args.leg_execution or not args.leg_execution_sha256):
        parser.error("post-leg requires an explicit leg execution path and SHA256")
    if args.stage == "prepare-leg" and (args.leg_execution or args.leg_execution_sha256):
        parser.error("prepare-leg cannot consume an after-leg execution")
    process = psutil.Process()
    process.cpu_affinity([26, 27])
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if process.cpu_affinity() != [26, 27] or process.nice() != psutil.BELOW_NORMAL_PRIORITY_CLASS:
        raise RuntimeError("Reserved E26/27 BelowNormal CPU allocation was not applied")
    os.environ.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    import torch

    torch.set_num_threads(args.threads)
    from hmr4d.lidar_mocap.formal_completion import prepare_formal_leg_completion, verify_formal_postleg_cpu

    baseline = dict(path=str(args.baseline.resolve()), sha256=args.baseline_sha256)
    if args.stage == "prepare-leg":
        receipt = prepare_formal_leg_completion(args.root, baseline)
    else:
        receipt = verify_formal_postleg_cpu(
            args.root,
            baseline,
            dict(path=str(args.leg_execution.resolve()), sha256=args.leg_execution_sha256),
            threads=args.threads,
        )
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
