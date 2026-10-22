"""Inspect or execute the independently authorized, private-job guarded-leg stage."""

import os

os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")

import argparse
from pathlib import Path

from hmr4d.lidar_mocap.guarded_leg_execution import cpu_only, execute, watchdog
from hmr4d.lidar_mocap.guarded_leg_gpu import inspect_request, load_authorization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-request", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--watchdog", action="store_true")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--grant", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--context-sha256")
    args = parser.parse_args()
    cpu_only()
    if args.watchdog:
        if not args.context or not args.context_sha256:
            parser.error("Independent watchdog requires an exact bound context")
        watchdog(args.context, args.context_sha256)
    else:
        if not args.request or ((args.preflight or args.execute) and not args.grant):
            parser.error("Use an explicit request and, for actual readiness/execution, the new grant")
        if args.inspect_request:
            value = inspect_request(args.request)
            print(dict(state="request_inspected_not_granted", target=str(value["root"]), gpu_started=False))
        elif args.preflight:
            load_authorization(args.request, args.grant)
            print(dict(state="actual_grant_and_readiness_verified_no_cuda_launched", gpu_started=False))
        else:
            print(execute(args.request, args.grant)["state"])


if __name__ == "__main__":
    main()
