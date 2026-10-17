"""Freeze ten continuous comparisons and all known foot/track review observations."""

import argparse
from collections import defaultdict
from pathlib import Path

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.body_repair import published_windows
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2")
    p.add_argument("--union", type=Path, default=PROJ_ROOT / "outputs/badminton_full_plus_new_v1/dataset_manifest.json")
    args = p.parse_args()
    inventory, allowed = [], defaultdict(set)
    for sid, window in published_windows(args.union):
        config = read_json(window / "window.json")
        scans = sorted(config["source_pairs"])
        allowed[sid].update(scans)
        runs = []
        for scan in scans:
            if not runs or scan != runs[-1][-1] + 1:
                runs.append([])
            runs[-1].append(scan)
        inventory.append(
            dict(source_id=sid, source_window=str(window), scans=scans, runs=[r for r in runs if len(r) >= 50])
        )
    requested = [
        ("20261009_202713", 1523, True),
        ("20261009_202713", 2401, False),
        ("20261010_190709", 40, False),
        ("20261010_192753", 2940, False),
        ("20261012_154726", 3114, False),
        ("20261012_160054", 4173, False),
        ("20261012_161544", 186, False),
        ("20261012_163452", 1676, False),
        ("20261012_163452", 8280, False),
        ("20261012_163452", 703, False),
    ]
    clips, wanted = [], defaultdict(set)
    for sid, center, begins in requested:
        choices = [
            (min(abs(scan - center) for scan in run), entry, run)
            for entry in inventory
            if entry["source_id"] == sid
            for run in entry["runs"]
        ]
        _, entry, run = min(choices, key=lambda v: v[0])
        start = max(run[0], min(run[-1] - 49, center if begins else center - 25))
        scans = list(range(start, start + 50))
        if not set(scans) <= allowed[sid]:
            raise ValueError("Comparison would include unfitted/out-of-scope scans")
        clips.append(
            dict(
                id=f"clip{len(clips)+1:02d}",
                source_id=sid,
                source_window=entry["source_window"],
                lidar_indices=scans,
                requested_case_scan=center,
                seconds=5,
            )
        )
        wanted[sid].update(scans)
    audit_path = PROJ_ROOT / "outputs/badminton_label_audit_v2/report.json"
    audit = read_json(audit_path)
    for event in audit["track_fragment_candidates"]:
        sid = event["source_id"]
        for scan in (event["before_scan"], event["after_scan"]):
            wanted[sid].update(i for i in range(scan - 2, scan + 3) if i in allowed[sid])
    import json

    records = PROJ_ROOT / "outputs/badminton_label_audit_v2/person_frames.jsonl"
    with records.open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r["delivered"] and "confident_ankle_reprojection_over_20px" in r["flags"]:
                wanted[r["source_id"]].update(
                    i for i in range(r["scan"] - 1, r["scan"] + 2) if i in allowed[r["source_id"]]
                )
    plan = dict(
        schema="badminton-body-repair-cases-v2",
        source_union=str(args.union.resolve()),
        source_union_sha256=file_sha(args.union),
        original_audit_sha256=file_sha(audit_path),
        clips=clips,
        selected_scans={k: sorted(v) for k, v in wanted.items()},
        track_candidates=audit["track_fragment_candidates"],
        unresolved_native_endpoints=str(
            PROJ_ROOT / "outputs/badminton_body_repair_v1/tracking_final/review/cases.json"
        ),
    )
    write_json(args.output_root / "case_plan.json", plan)
    print(
        dict(
            clips=len(clips),
            selected_scans=sum(map(len, wanted.values())),
            by_source={k: len(v) for k, v in wanted.items()},
        )
    )


if __name__ == "__main__":
    main()
