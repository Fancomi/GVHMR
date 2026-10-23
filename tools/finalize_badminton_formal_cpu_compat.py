"""Verify postleg on CPU using explicit producer fingerprints and genuine CPU test evidence.

--metadata-only reads contracts/selected report JSON without starting native/QA.
Full execution reuses the original native, dataset and numerical QA helpers.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                  MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-sha256", required=True)
    parser.add_argument("--leg-execution", type=Path, required=True)
    parser.add_argument("--leg-execution-sha256", required=True)
    parser.add_argument("--compatibility-implementation", type=Path, required=True)
    parser.add_argument("--compatibility-implementation-sha256", required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    import psutil

    own = psutil.Process()
    own.cpu_affinity([26, 27])
    own.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    from hmr4d.lidar_mocap.formal_completion_policy_compat import (
        inspect_policy_metadata, verify_formal_postleg_cpu,
    )

    item = lambda path, sha: dict(path=str(path.resolve()), sha256=sha)
    baseline = item(args.baseline, args.baseline_sha256)
    execution = item(args.leg_execution, args.leg_execution_sha256)
    tested = item(args.compatibility_implementation, args.compatibility_implementation_sha256)
    if args.metadata_only:
        result = inspect_policy_metadata(args.root, baseline, execution, tested)
    else:
        import torch

        torch.set_num_threads(args.threads)
        result = verify_formal_postleg_cpu(args.root, baseline, execution, threads=args.threads,
                                         compatibility_test_execution=tested)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
