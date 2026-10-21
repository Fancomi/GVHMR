"""Report native pilot fit quality and measured efficiency without a GPU."""

import argparse
from pathlib import Path

import torch

from hmr4d.lidar_mocap.pilot_report import report_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = report_pilot(args.root)
    print(
        dict(
            report=str(args.root / "quality_report.json"),
            fitted=result["cohorts"]["all_fitted"]["person_scans"],
            accepted=result["cohorts"]["accepted"]["person_scans"],
            seconds=result["report_seconds"],
        )
    )


if __name__ == "__main__":
    main()
