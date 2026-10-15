"""Verify an explicit sealed rally inventory and materialize native parent ranges."""
import argparse
import json
from pathlib import Path

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.rally_inventory import verify_inventory
from hmr4d.lidar_mocap.rgb_observations import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acquisition-manifest',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--acquisition-sha256')
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();output=args.output_root.resolve()
    if not output.is_relative_to(PROJ_ROOT/'outputs'):parser.error('--output-root must be inside project outputs')
    if (output/'verification.json').exists():parser.error('Use a new output folder for a new intake verification')
    report,rows=verify_inventory(args.acquisition_manifest,args.data_root,args.acquisition_sha256)
    output.mkdir(parents=True,exist_ok=True)
    ranges_path=output/'prepared_ranges.jsonl'
    ranges_path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    report.update(data_root=str(args.data_root.resolve()),
                  prepared_ranges=dict(path=str(ranges_path),sha256=file_sha(ranges_path),records=len(rows)))
    write_json(output/'verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
