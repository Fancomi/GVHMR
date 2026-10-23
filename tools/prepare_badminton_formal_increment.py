"""Canonical CPU metadata entry: prepare a policy-bound formal increment, without body inference.

Inputs: explicit frozen v2 policy path/SHA and a fresh independent output root.
Output: camera/source/window/body plans and genuine input-equivalence metadata.
Missing actual parent final audit, changed accepted source or existing output is an error.
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-policy", type=Path, required=True)
    parser.add_argument("--release-policy-sha256", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(1,), default=1)
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    if os.name == "nt":
        import psutil

        process = psutil.Process()
        process.cpu_affinity([24, 25])
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        process.ionice(psutil.IOPRIO_LOW)
    from hmr4d.lidar_mocap.formal_incremental_intake import prepare_formal_increment

    result = prepare_formal_increment(args.release_policy, args.release_policy_sha256, args.root)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
