"""Inspect or execute one exact full-native stage under an independently watched private Windows Job."""

import argparse
import os
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")

from hmr4d.lidar_mocap.owned_gpu_stage import cpu_controller, execute, inspect_descriptor, watchdog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--watchdog", action="store_true")
    parser.add_argument("--descriptor", type=Path)
    parser.add_argument("--descriptor-sha256")
    parser.add_argument("--lease", type=Path)
    parser.add_argument("--lease-sha256")
    parser.add_argument("--context", type=Path)
    parser.add_argument("--context-sha256")
    args = parser.parse_args()
    cpu_controller()
    if args.watchdog:
        if not args.context or not args.context_sha256 or args.descriptor or args.lease:
            parser.error("Watchdog requires only its exact context and SHA256")
        watchdog(args.context, args.context_sha256)
    else:
        if not args.descriptor or not args.descriptor_sha256:
            parser.error("Use an explicit descriptor and exact SHA256")
        if args.execute:
            if not args.lease or not args.lease_sha256:
                parser.error("Execution additionally requires a separately issued exact local lease and SHA256")
            print(execute(args.descriptor, args.descriptor_sha256, args.lease, args.lease_sha256)["status"])
        else:
            if args.lease or args.context:
                parser.error("Descriptor inspection does not consume a lease or Job context")
            checked = inspect_descriptor(args.descriptor, args.descriptor_sha256)
            print(dict(state="descriptor_inspected_not_authorized", stage=checked["stage"],
                       root=str(checked["root"]), gpu_started=False))


if __name__ == "__main__":
    main()
