"""Prepare an exact full-old-plus-new union for independent final acceptance."""

import argparse
import json
from pathlib import Path

import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.expanded_dataset import assemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--consumer-contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1 or not args.output_root.resolve().is_relative_to(PROJ_ROOT / "outputs"):
        parser.error("Use a positive thread count and a separate outputs/ destination")
    torch.set_num_threads(args.threads)
    manifest = assemble(args.old_manifest, args.new_manifest, args.consumer_contract, args.output_root)
    print(json.dumps(dict(status=manifest["status"], accepted=manifest["accepted"], training_published=False)))


if __name__ == "__main__":
    main()
