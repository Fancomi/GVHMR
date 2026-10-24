"""Canonical typed incremental guard entry; no numerical algorithm is changed.

Inputs are explicit request/grant/descriptor bindings. Inspection and preflight
are CPU-only; execution requires actual root-issued authority and a private Job.
"""

import os

os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")

import argparse
from pathlib import Path

from hmr4d.lidar_mocap.incremental_guard import inspect_request, load_authorization
from hmr4d.lidar_mocap.incremental_guard_execution import cpu_only, execute, watchdog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect-request", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--watchdog", action="store_true")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--grant", type=Path)
    parser.add_argument("--descriptor", type=Path)
    parser.add_argument("--descriptor-sha256")
    parser.add_argument("--context", type=Path)
    parser.add_argument("--context-sha256")
    args = parser.parse_args()
    if args.watchdog:
        if (
            not args.context
            or not args.context_sha256
            or any((args.request, args.grant, args.descriptor, args.descriptor_sha256))
        ):
            parser.error("Watchdog requires only its exact context and SHA256")
        watchdog(args.context, args.context_sha256)
        return
    if args.context or args.context_sha256:
        parser.error("Context arguments are only valid for the independent watchdog")
    if not args.request or not args.descriptor or not args.descriptor_sha256:
        parser.error("Use an explicit request and descriptor path/SHA256 pair")
    if args.inspect_request and args.grant:
        parser.error("Request inspection cannot consume a GPU grant")
    if (args.preflight or args.execute) and not args.grant:
        parser.error("Actual preflight/execution requires a new explicitly issued grant")
    cpu_only()
    keywords = dict(descriptor_path=args.descriptor, descriptor_sha256=args.descriptor_sha256)
    if args.inspect_request:
        checked = inspect_request(args.request, **keywords)
        print(dict(state="incremental_request_inspected_not_granted", target=str(checked["root"]), gpu_started=False))
    elif args.preflight:
        load_authorization(args.request, args.grant, **keywords)
        print(dict(state="actual_incremental_grant_verified_no_cuda_launched", gpu_started=False))
    else:
        print(execute(args.request, args.grant, **keywords)["state"])


if __name__ == "__main__":
    main()
