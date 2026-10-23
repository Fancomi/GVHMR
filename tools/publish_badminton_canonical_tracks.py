"""Publish a numerically unchanged bilateral view with reviewed canonical track cuts."""

import argparse
import json
from pathlib import Path

import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.canonical_tracks import publish_canonical_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--reviewed-boundaries", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--release-policy", type=Path)
    parser.add_argument("--release-policy-sha256")
    args = parser.parse_args()
    if (args.release_policy is None) != (args.release_policy_sha256 is None):
        parser.error("Supply --release-policy and --release-policy-sha256 together")
    if not args.output_root.resolve().is_relative_to(PROJ_ROOT / "outputs"):
        parser.error("Use a separate explicit outputs/ destination")
    torch.set_num_threads(1)
    manifest = publish_canonical_view(
        args.source_manifest,
        args.reviewed_boundaries,
        args.output_root,
        release_policy=args.release_policy,
        release_policy_sha256=args.release_policy_sha256,
    )
    print(
        json.dumps(
            dict(
                accepted=manifest["accepted"],
                splits={k: v["accepted"] for k, v in manifest["suggested_session_splits"].items()},
            )
        )
    )


if __name__ == "__main__":
    main()
