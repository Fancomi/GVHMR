"""Prepare independent opposite-view evidence for every actual ID boundary."""

import argparse
from pathlib import Path

import numpy as np

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.body_repair import published_windows
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2")
    args = p.parse_args()
    original = read_json(args.root / "case_plan.json")
    replay = read_json(PROJ_ROOT / "outputs/badminton_body_repair_v1/tracking_final/tracking_replay.json")
    windows = {(sid, path.name): path for sid, path in published_windows(original["source_union"])}
    cases = []
    same_id = 0
    for window in replay["windows"]:
        a = np.load(window["mapping"])["observations"]
        ranges = {int(k): v[v[:, 0].argsort()] for k in np.unique(a[:, 2]) for v in [a[a[:, 2] == k]]}
        for event in window["fragment_candidates"]:
            left, right = [int(event[side + "_motion_id"].split(":")[-2]) for side in ("before", "after")]
            if left == right:
                same_id += 1
                continue
            positions = [int(ranges[left][-1, 0]), int(ranges[right][0, 0])]
            source = windows[window["source_id"], window["window_id"]]
            rows = read_json(source / "teacher/manifest.json")["provenance"]["frame_rows"]
            context = sorted({i for pos in positions for i in range(pos - 12, pos + 13) if 0 <= i < len(rows)})
            cases.append(
                dict(
                    index=len(cases),
                    source_window=str(source),
                    positions=positions,
                    context_positions=context,
                    event=event,
                )
            )
    directory = args.root / "identity_evidence"
    path = directory / "boundaries.json"
    write_json(path, cases)
    write_json(
        directory / "case_plan.json",
        dict(
            original,
            clips=[],
            selected_scans={sid: [] for sid in original["selected_scans"]},
            unresolved_native_endpoints=str(path),
            same_original_track_segments=same_id,
        ),
    )
    print(
        dict(
            actual_ID_boundaries=len(cases),
            same_ID_segments=same_id,
            native_context_samples=sum(len(c["context_positions"]) for c in cases),
        )
    )


if __name__ == "__main__":
    main()
