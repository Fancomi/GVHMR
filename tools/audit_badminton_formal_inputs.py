"""Audit actual admitted camera/source inputs against frozen completed review inputs."""

import argparse
import os
from pathlib import Path

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.formal_reuse import audit_formal_inputs
from hmr4d.lidar_mocap.repair_execution import write_progress_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, action="append", required=True)
    parser.add_argument("--formal-source-root", type=Path, required=True)
    parser.add_argument("--role-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-relocations", type=Path)
    parser.add_argument("--data-relocations-sha256")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJ_ROOT / "outputs") or output.exists():
        parser.error("Use an absent audit receipt under project outputs")
    if args.threads < 1:
        parser.error("--threads must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import torch

    torch.set_num_threads(args.threads)
    result = audit_formal_inputs(args.review_root, args.formal_source_root, args.role_contract,
        data_relocations=args.data_relocations, data_relocations_sha256=args.data_relocations_sha256)
    write_progress_json(output, result)
    print(dict(state=result["state"], sources=len(result["sources"]), training_published=False))


if __name__ == "__main__":
    main()
