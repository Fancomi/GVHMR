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
    args = parser.parse_args()
    if not args.output_root.resolve().is_relative_to(PROJ_ROOT / "outputs"):
        parser.error("Use a separate explicit outputs/ destination")
    torch.set_num_threads(1)
    manifest = publish_canonical_view(args.source_manifest, args.reviewed_boundaries, args.output_root)
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
