"""Generate and serve the seven-source native dual-camera rally HTML on localhost.

Requires a prepared build root. Original MP4 files stream with HTTP byte ranges;
no copied/transcoded videos or GPU inference. Exact fitted overlays load on demand.
"""

import argparse
from pathlib import Path

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.rally_viewer import prepare_viewer, serve_viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJ_ROOT / "outputs/badminton_rallies_v1")
    parser.add_argument("--baseline-root", type=Path, default=PROJ_ROOT / "outputs/badminton_corner_fix_v1_continuity")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    data = prepare_viewer(args.output_root, args.baseline_root)
    serve_viewer(data, port=args.port)


if __name__ == "__main__":
    main()
