"""Audit whether verified pilot fits can enter the new full scope unchanged."""

import argparse
from pathlib import Path

from hmr4d.lidar_mocap.fit_reuse import audit_full_fit_reuse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-code", nargs=2, action="append", default=[], metavar=("PILOT_ROOT", "GIT_REV"))
    args = parser.parse_args()
    result = audit_full_fit_reuse(args.root, {str(Path(root).resolve()): revision for root, revision in args.source_code})
    print(dict(state=result["state"], eligible_windows=result["eligible_windows"],
               fitted=result["eligible_fitted_person_scans"], accepted=result["eligible_qa_accepted"]), flush=True)


if __name__ == "__main__":
    main()
