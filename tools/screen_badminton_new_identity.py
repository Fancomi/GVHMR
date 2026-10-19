"""Screen every frozen fresh-source tracklet endpoint for native identity review.

Screening proposes cases only. It neither approves identities nor changes the
established opposite-camera geometric review thresholds.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path

import numpy as np
import torch

from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json


def screen(root):
    root = Path(root).resolve()
    pending_path = root / "identity_evidence/boundary_inventory.pending.json"
    pending = read_json(pending_path)
    if pending["source_union_sha256"] != file_sha(root / "new_source_input.json"):
        raise ValueError("Endpoint inventory belongs to another frozen input")
    groups = defaultdict(list)
    for entry in pending["tracklets"]:
        groups[entry["source_window"]].append(entry)
    producer_bytes = Path(__file__).read_bytes()
    producer_sha = hashlib.sha256(producer_bytes).hexdigest()
    snapshot = root / "identity_evidence/producers" / ("screen_identity_" + producer_sha[:16] + ".py")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists() and snapshot.read_bytes() != producer_bytes:
        raise ValueError("Frozen inventory producer changed")
    if not snapshot.exists():
        snapshot.write_bytes(producer_bytes)
    evidence = {str(pending_path): file_sha(pending_path), str(snapshot): producer_sha}
    cases, windows = [], []
    for name, tracklets in groups.items():
        window = Path(name)
        manifest_path, cache_path = window / "teacher/manifest.json", window / "teacher/detections.pt"
        manifest = read_json(manifest_path)
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache["provenance_hash"] != manifest["provenance_hash"]:
            raise ValueError("Teacher native detection provenance changed")
        evidence.update({str(p): file_sha(p) for p in (manifest_path, cache_path)})
        detections = cache["detections"]
        linked = {}
        retained = set()
        for info in manifest["tracks"]:
            ids = info.get("source_tracklets", [info["track_id"]])
            retained.update(ids)
            for identity in ids:
                linked[identity] = int(info["track_id"])
        endpoints = []
        for entry in tracklets:
            values = dict(entry)
            for side in ("first", "last"):
                det = detections[entry[side + "_position"]][entry[side + "_detection_index"]]
                values[side + "_box"] = np.asarray(det["bbx_xyxy"], dtype=float)
            endpoints.append(values)
        count = 0
        for before in endpoints:
            for after in endpoints:
                a, b = before["original_track_id"], after["original_track_id"]
                gap = (after["native_first"]["common_time_us"] - before["native_last"]["common_time_us"]) / 1e6
                if a == b or not 0 < gap <= 0.35 or not ({a, b} & retained):
                    continue
                box_a, box_b = before["last_box"], after["first_box"]
                heights = [box[3] - box[1] for box in (box_a, box_b)]
                distance = float(np.linalg.norm((box_a[:2] + box_a[2:] - box_b[:2] - box_b[2:]) / 2))
                normalized = distance / max(min(heights), 35)
                already_linked = a in linked and b in linked and linked[a] == linked[b]
                if not already_linked and (normalized > 0.75 or max(heights) / min(heights) > 3):
                    continue
                cases.append(
                    dict(
                        source_id=before["source_id"],
                        window=window.name,
                        original_track_ids=[a, b],
                        gap_seconds=gap,
                        normalized_endpoint_distance=normalized,
                        already_linked_by_teacher=already_linked,
                        endpoint_boxes=[box_a.tolist(), box_b.tolist()],
                    )
                )
                count += 1
        windows.append(
            dict(
                source_id=tracklets[0]["source_id"],
                window=window.name,
                tracklets=len(tracklets),
                retained_original_ids=len(retained),
                candidates=count,
            )
        )
        print(dict(window=window.name, tracklets=len(tracklets), candidates=count), flush=True)
    result = dict(
        schema="badminton-new-source-identity-inventory-v1",
        review_status="pending",
        source_union_sha256=pending["source_union_sha256"],
        cases=cases,
        evidence_files=evidence,
        tracklets=len(pending["tracklets"]),
        windows=windows,
        source_counts=dict(Counter(case["source_id"] for case in cases)),
        policy=dict(
            maximum_gap_seconds=0.35,
            maximum_normalized_center_distance=0.75,
            maximum_height_ratio=3,
            include_all_existing_short_links=True,
            either_endpoint_requires_retained_teacher_pose=True,
        ),
        note="Complete endpoint screening; candidate scope needs review. No person identities approved.",
    )
    write_json(root / "identity_evidence/boundary_inventory.screened.json", result)
    return dict(
        windows=len(windows),
        tracklets=result["tracklets"],
        candidates=len(cases),
        source_counts=result["source_counts"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(screen(args.root), flush=True)


if __name__ == "__main__":
    main()
