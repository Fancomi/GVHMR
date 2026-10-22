"""Continue genuine native identity evidence under an existing continuous grant."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import json
from pathlib import Path

from hmr4d.lidar_mocap.identity_continuation import IdentityContinuation, freeze_completion, run_worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Prepared full-native run containing repair")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--grant", type=Path)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--verify-complete",
        action="store_true",
        help="Read and freeze actual existing identity outcomes without inference",
    )
    parser.add_argument("--worker", choices=("observe", "review", "seal"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-script", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-lease", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-lease-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.worker_script or args.verify_complete or args.request or args.grant:
            parser.error("Internal worker mode requires its exact script and no controller options")
        if bool(args.worker_lease) != bool(args.worker_lease_sha256):
            parser.error("Internal observation worker requires both lease path and SHA256")
        lease = (
            dict(path=str(args.worker_lease.resolve()), sha256=args.worker_lease_sha256) if args.worker_lease else None
        )
        run_worker(args.root, args.worker, args.worker_script, args.threads, lease)
        return
    if args.worker_script or args.worker_lease or args.worker_lease_sha256:
        parser.error("Worker scripts are accepted only by internal worker mode")
    if args.verify_complete:
        if args.request or args.grant:
            parser.error("Existing-completion verification does not acquire GPU authority")
        import psutil

        process = psutil.Process()
        process.cpu_affinity([26, 27])
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        result = freeze_completion(args.root)
    else:
        if not args.request or not args.grant:
            parser.error("Continuation requires the actual existing request and continuous grant")
        result = IdentityContinuation(args.root, args.request, args.grant, args.threads).run()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
