"""CPU-only inspection of an explicit external source index; never admits fitting."""
import argparse
import json
from pathlib import Path

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.source_index import prepare_source_index
from hmr4d.lidar_mocap.rgb_observations import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-index',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--source-index-sha256')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if not args.report.resolve().is_relative_to(PROJ_ROOT/'outputs'):
        parser.error('--report must be inside project outputs')
    report=prepare_source_index(args.source_index,args.data_root,args.source_index_sha256)
    write_json(args.report,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
