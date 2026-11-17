"""CPU-only inspection of an explicit external source index; never admits fitting."""
import argparse
import json
from pathlib import Path

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.relocations import load_relocations
from hmr4d.lidar_mocap.rgb_observations import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--source-index', type=Path)
    mode.add_argument('--final-current-release', type=Path,
                      help='Explicit portable release: v1 verifies membership; v2 inspects current PNG XYZ/grid only')
    parser.add_argument('--source-id', help='Original source ID, required with --final-current-release')
    parser.add_argument('--final-current-release-sha256')
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--source-index-sha256')
    parser.add_argument('--data-relocations',type=Path)
    parser.add_argument('--data-relocations-sha256')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if not args.report.resolve().is_relative_to(PROJ_ROOT/'outputs'):
        parser.error('--report must be inside project outputs')
    if args.final_current_release is not None:
        if not args.source_id:
            parser.error('--final-current-release requires --source-id')
        if any(v is not None for v in (args.source_index_sha256, args.data_relocations,
                                       args.data_relocations_sha256)):
            parser.error('Final-current mode does not accept legacy index/relocation options')
        from hmr4d.lidar_mocap.final_current import prepare_final_current

        try:
            report = prepare_final_current(args.final_current_release, args.data_root, args.source_id,
                                           args.final_current_release_sha256)
        except (ValueError, KeyError, OSError) as exc:
            parser.error(str(exc))
        write_json(args.report, report)
        print(json.dumps(report, indent=2))
        return
    if args.source_id is not None or args.final_current_release_sha256 is not None:
        parser.error('--source-id/--final-current-release-sha256 require --final-current-release')
    from hmr4d.lidar_mocap.source_index import prepare_source_index

    try:
        relocations=load_relocations(args.data_root,args.data_relocations,args.data_relocations_sha256)
    except ValueError as exc:
        parser.error(str(exc))
    report=prepare_source_index(args.source_index,args.data_root,args.source_index_sha256,relocations=relocations)
    write_json(args.report,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
