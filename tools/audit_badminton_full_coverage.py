"""Account for every selected scan, QA rejection and cached native bilateral gate."""

import argparse
from pathlib import Path

import psutil

from hmr4d.lidar_mocap.full_coverage import audit_full_coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-people", type=int, default=2)
    args = parser.parse_args()
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity([0, 1])
    result = audit_full_coverage(args.root, args.expected_people)
    print(dict(counts=result["counts"], bilateral=result["bilateral_detection_rule"]["passed"],
               seconds=result["report_seconds"]))


if __name__ == "__main__":
    main()
