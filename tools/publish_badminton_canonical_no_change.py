"""Publish a source-bound reviewed no-change identity view; metadata CPU only.

Inputs are an actual bilateral publication, a closed root no-change review and
an explicitly frozen incremental release policy. Outputs are one fresh proof
and publication manifest; no source label, tensor, admission or identity changes.
Pending parent release evidence, incomplete review or changed bytes fail closed.
"""

import argparse
import json
from pathlib import Path

from hmr4d.lidar_mocap.canonical_no_change import publish_no_change_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--reviewed-continuity", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--release-policy", type=Path, required=True)
    parser.add_argument("--release-policy-sha256", required=True)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    result = publish_no_change_view(
        args.source_manifest,
        args.reviewed_continuity,
        args.output_root,
        release_policy=args.release_policy,
        release_policy_sha256=args.release_policy_sha256,
    )
    print(json.dumps(dict(schema=result["schema"], accepted=result["accepted"], identity_changes=0)))


if __name__ == "__main__":
    main()
