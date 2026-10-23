"""Inspect or explicitly prepare an accepted-camera, review-only native intake.

CPU only. Requires the exact prior strict accepted freeze and its SHA-bound
proof. Camera acceptance does not admit body labels or publish training data.
"""

import argparse
import json
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_key] = "1"

from hmr4d.lidar_mocap.admitted_native_input import inspect_admitted_native, prepare_admitted_full_native
from hmr4d.lidar_mocap.pilot import check_binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--accepted-parent", type=Path, required=True)
    parser.add_argument("--accepted-parent-sha256", required=True)
    parser.add_argument("--accepted-freeze-proof", type=Path, required=True)
    parser.add_argument("--accepted-freeze-proof-sha256", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--prepare", action="store_true",
                        help="Create the separately bound review snapshot and original CPU body plan")
    args = parser.parse_args()
    if args.threads != 1:
        parser.error("This accepted-source CPU adapter requires --threads 1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        parser.error("Set CUDA_VISIBLE_DEVICES=-1 for this CPU-only adapter")
    if os.name == "nt":
        import psutil
        process = psutil.Process()
        process.cpu_affinity([22, 23, 24, 25])
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        process.ionice(psutil.IOPRIO_LOW)
    parent = dict(path=str(args.accepted_parent.resolve()), sha256=args.accepted_parent_sha256)
    proof = dict(path=str(args.accepted_freeze_proof.resolve()), sha256=args.accepted_freeze_proof_sha256)
    check_binding(parent)
    check_binding(proof)
    if args.prepare:
        result = prepare_admitted_full_native(args.manifest, args.root, args.accepted_parent,
                                             accepted_freeze_proof=proof, threads=args.threads)
    else:
        result = inspect_admitted_native(args.manifest, args.accepted_parent, accepted_freeze_proof=proof)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
