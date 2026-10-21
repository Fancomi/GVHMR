"""Generate and serve the seven-source native dual-camera rally HTML on localhost.

Requires a prepared build root. Original MP4 files stream with HTTP byte ranges;
no copied/transcoded videos or GPU inference. Exact fitted overlays load on demand.
"""

import argparse
from pathlib import Path

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.rally_viewer import prepare_viewer, serve_viewer
from hmr4d.lidar_mocap.relocations import load_run_relocations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJ_ROOT / "outputs/badminton_rallies_v1")
    parser.add_argument("--baseline-root", type=Path, default=PROJ_ROOT / "outputs/badminton_corner_fix_v1_continuity")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--data-relocations", type=Path)
    parser.add_argument("--data-relocations-sha256")
    parser.add_argument("--viewer-output", type=Path, help="Separate generated runtime/viewer directory")
    parser.add_argument(
        "--display-crop", type=Path, help="Explicit display-only crop sidecar, bound to frozen bbox SHA"
    )
    args = parser.parse_args()
    try:
        relocations = load_run_relocations(args.output_root, args.data_relocations, args.data_relocations_sha256)
    except ValueError as exc:
        parser.error(str(exc))
    data = prepare_viewer(
        args.output_root,
        args.baseline_root,
        display_crop=args.display_crop,
        relocations=relocations,
        viewer_output=args.viewer_output,
    )
    serve_viewer(data, port=args.port)


if __name__ == "__main__":
    main()
