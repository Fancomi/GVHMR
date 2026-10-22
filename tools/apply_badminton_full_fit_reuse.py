"""Import audited SMPL tensors unchanged after current pose/identity completion."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from pathlib import Path

from hmr4d.lidar_mocap.fit_reuse import apply_full_fit_reuse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = apply_full_fit_reuse(args.root)
    print(dict(reused_windows=result["reused_windows"], fitted=result["reused_fitted_person_scans"],
               accepted=result["reused_qa_accepted"], current_native_verification_pending=True), flush=True)


if __name__ == "__main__":
    main()
