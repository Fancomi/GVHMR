"""CPU-only inspection of an explicit external source index; never admits fitting."""
import argparse
import json
from pathlib import Path

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.relocations import load_relocations
from hmr4d.lidar_mocap.source_index import prepare_source_index
from hmr4d.lidar_mocap.rgb_observations import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-index',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--source-index-sha256')
    parser.add_argument('--data-relocations',type=Path)
    parser.add_argument('--data-relocations-sha256')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if not args.report.resolve().is_relative_to(PROJ_ROOT/'outputs'):
        parser.error('--report must be inside project outputs')
    try:
        relocations=load_relocations(args.data_root,args.data_relocations,args.data_relocations_sha256)
    except ValueError as exc:
        parser.error(str(exc))
    report=prepare_source_index(args.source_index,args.data_root,args.source_index_sha256,relocations=relocations)
    write_json(args.report,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
