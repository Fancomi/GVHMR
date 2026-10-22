"""Derive only witnessed equivalent windows; actually verify the remainder."""

import os

os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import argparse
import json
from pathlib import Path
import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--numeric-import", type=Path, required=True)
    parser.add_argument("--numeric-import-sha256", required=True)
    parser.add_argument("--parent-witness", type=Path, action="append", default=[])
    parser.add_argument("--parent-witness-sha256", action="append", default=[])
    parser.add_argument("--run-required-actual", action="store_true")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if len(args.parent_witness) != len(args.parent_witness_sha256):
        parser.error("Each original parent witness needs its explicit SHA256")
    process = psutil.Process()
    process.cpu_affinity([26, 27])
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    from hmr4d.lidar_mocap.formal_native_derive import prepare_derived_native, run_required_actual_native
    from hmr4d.lidar_mocap.formal_baseline import _bound

    witnesses = {}
    for path, sha in zip(args.parent_witness, args.parent_witness_sha256):
        item = dict(path=str(path.resolve()), sha256=sha)
        value, _ = _bound(item, {})
        parent = str(Path(value["repair_root"]).resolve().parent)
        if parent in witnesses:
            raise ValueError("A parent has two conflicting execution witnesses")
        witnesses[parent] = item
    result = prepare_derived_native(
        args.root, dict(path=str(args.numeric_import.resolve()), sha256=args.numeric_import_sha256), witnesses
    )
    if args.run_required_actual:
        native = run_required_actual_native(args.root.resolve() / "repair", args.threads)
        result.update(
            actual_native_complete=native.get("passed") is True,
            verification_coverage=native.get("verification_coverage"),
        )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
