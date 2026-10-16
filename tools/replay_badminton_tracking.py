"""Replay both tracker versions on all frozen native detections, using CPU only.

IDs are compared through identical observed detections, never by label colors.
An automatic linked-fragment result is not verified real-person identity truth.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.body_repair import published_windows
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.teacher import associate_detections, _split_tracks


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def replay(source_id, window, fragments, output):
    manifest = read(window / "teacher/manifest.json")
    provenance = manifest["provenance"]
    rows = provenance["frame_rows"]
    options = provenance["options"]
    cache_path = window / "teacher/detections.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache["provenance_hash"] != manifest["provenance_hash"]:
        raise ValueError("Detection and teacher provenance differ")
    frames = [[dict(det, replay_detection_index=i) for i, det in enumerate(frame)] for frame in cache["detections"]]
    settings = dict(max_track_gap_frames=options.get("max_track_gap_frames", 1))
    old = associate_detections(frames, prediction_mode="corners_v1", **settings)
    new = associate_detections(frames)
    old_map = {(o["position"], o["replay_detection_index"]): k for k, values in old.items() for o in values}
    new_map = {(o["position"], o["replay_detection_index"]): k for k, values in new.items() for o in values}
    if old_map.keys() != new_map.keys():
        raise ValueError("Tracker changed the set of actual observations")
    observations = []
    for obs, old_id in old_map.items():
        observations.append([*obs, old_id, new_map[obs]])
    mapping_path = output / "track_maps" / source_id / (window.name + ".npz")
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mapping_path, observations=np.asarray(observations, dtype=np.int32))
    quality = {}
    if fragments:
        for line in (window / "quality.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            quality[row["lidar_index"], row["motion_id"]] = row
    reviewed = []
    for event in fragments:
        endpoints = []
        for side in ("before", "after"):
            motion = event[side + "_motion_id"]
            row = quality[event[side + "_scan"], motion]
            old_id = int(motion.split(":")[-2])
            choices = old[old_id]
            observed = min(choices, key=lambda o: abs(rows[o["position"]]["common_time_us"] - row["scan_mid_us"]))
            observation_key = observed["position"], observed["replay_detection_index"]
            endpoints.append(
                dict(
                    old_track_id=old_id,
                    new_track_id=new_map[observation_key],
                    native_record_index=rows[observed["position"]]["record_index"],
                    sample_time_error_us=rows[observed["position"]]["common_time_us"] - row["scan_mid_us"],
                )
            )
        reviewed.append(
            dict(
                event,
                replay_endpoints=endpoints,
                linked_by_new_tracker=endpoints[0]["new_track_id"] == endpoints[1]["new_track_id"],
            )
        )
    split_existing_links = 0
    for values in old.values():
        for a, b in zip(values, values[1:]):
            if b["position"] == a["position"] + 1:
                split_existing_links += (
                    new_map[a["position"], a["replay_detection_index"]]
                    != new_map[b["position"], b["replay_detection_index"]]
                )
    counts = Counter(
        native_frames=len(rows),
        observed_detections=len(old_map),
        old_tracks=len(old),
        new_tracks=len(new),
        old_links_split=split_existing_links,
        audited_fragment_candidates=len(reviewed),
        linked_candidates=sum(e["linked_by_new_tracker"] for e in reviewed),
        linked_both_delivered=sum(
            e["linked_by_new_tracker"] and e["delivered_before"] and e["delivered_after"] for e in reviewed
        ),
    )
    for name, tracks in (("old", old), ("new", new)):
        segments, rejected = _split_tracks(tracks, rows, options.get("min_track_frames", 8))
        counts[name + "_retained_segments"] = len(segments)
        counts[name + "_retained_observations"] = sum(len(s["observations"]) for s in segments)
        counts[name + "_ambiguous_observations"] = sum(o["identity_ambiguous"] for v in tracks.values() for o in v)
    return dict(
        source_id=source_id,
        window_id=window.name,
        counts=dict(counts),
        fragment_candidates=reviewed,
        detections_sha256=file_sha(cache_path),
        teacher_manifest_sha256=file_sha(window / "teacher/manifest.json"),
        mapping=str(mapping_path),
        mapping_columns=["position", "detection_index", "old_track_id", "new_track_id"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--union", type=Path, default=PROJ_ROOT / "outputs/badminton_full_plus_new_v1/dataset_manifest.json"
    )
    parser.add_argument("--audit", type=Path, default=PROJ_ROOT / "outputs/badminton_label_audit_v2/report.json")
    parser.add_argument("--output-root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v1")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output_root.resolve()
    if not output.is_relative_to((PROJ_ROOT / "outputs").resolve()):
        raise ValueError("Repair outputs must stay in workspace outputs")
    output.mkdir(parents=True, exist_ok=True)
    union, audit = read(args.union), read(args.audit)
    if audit["source_union_sha256"] != file_sha(args.union):
        raise ValueError("Audit and union differ")
    tasks = []
    for sid, window in published_windows(args.union):
        events = [
            e
            for e in audit["track_fragment_candidates"]
            if e["source_id"] == sid and e["before_motion_id"].split(":")[1] == window.name
        ]
        tasks.append((sid, window, events, output))
    started = time.perf_counter()
    windows, counts = [], Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(lambda job: replay(*job), tasks):
            windows.append(result)
            counts.update(result["counts"])
            write_json(output / "tracking_progress.json", dict(windows=len(windows), counts=dict(counts)))
            print(
                json.dumps(dict(source=result["source_id"], window=result["window_id"], counts=result["counts"])),
                flush=True,
            )
    if counts["audited_fragment_candidates"] != len(audit["track_fragment_candidates"]):
        raise ValueError("Fragment candidate coverage is incomplete")
    write_json(
        output / "tracking_replay.json",
        dict(
            schema="badminton-tracking-repair-replay-v1",
            state="complete",
            source_union_sha256=file_sha(args.union),
            original_audit_sha256=file_sha(args.audit),
            counts=dict(counts),
            windows=windows,
            wall_seconds=time.perf_counter() - started,
            limits=[
                "Actual observations are unchanged; native missing detections are never filled.",
                "Legacy links are a regression reference, not real-person identity truth.",
                "This replay does not rerun temporal GVHMR or rewrite fitted labels.",
            ],
        ),
    )
    print(json.dumps(dict(state="complete", counts=dict(counts))), flush=True)


if __name__ == "__main__":
    main()
