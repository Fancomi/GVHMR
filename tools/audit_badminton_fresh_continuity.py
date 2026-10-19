"""Screen every native-verified fresh body for pelvis jumps and mutual ID swaps.

Reuse the existing mutual-swap criterion. Flags require native review; absence
of screening flags is not athlete ground truth or proof of physical identities.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.expanded_dataset import binding, check_binding
from hmr4d.lidar_mocap.rally_pipeline import read_json, read_rows
from hmr4d.lidar_mocap.repair_execution import write_progress_json as write_json


def audit(root, manifest_path=None, output_root=None):
    root = Path(root).resolve()
    labels = root / "final_labels"
    manifest_path = Path(manifest_path).resolve() if manifest_path else labels / "datasets/dataset_manifest.json"
    native_path = labels / "native_verification.json"
    manifest, native = read_json(manifest_path), read_json(native_path)
    if not native.get("passed") or native.get("state") != "complete":
        raise ValueError("Require all-window native verification first")
    if native["calibration_digest"] != manifest["calibration_digest"]:
        raise ValueError("Native verification and body publication use different cameras")
    output = Path(output_root).resolve() if output_root else root / "continuity_audit"
    output.mkdir(parents=True, exist_ok=True)
    review, review_binding = None, None
    if manifest.get("canonical_identity_revision"):
        from hmr4d.lidar_mocap.canonical_tracks import verify_canonical_publication

        verify_canonical_publication(manifest)
        if Path(manifest["source_dataset_manifest"]).resolve() != labels / "datasets/dataset_manifest.json":
            raise ValueError("Canonical view does not derive from this native-verified body run")
        previous = read_json(root / "continuity_audit/audit.json")
        check_binding(previous["aggregate_native_contract"])
        aggregate_path = Path(previous["aggregate_native_contract"]["path"])
        review_binding = manifest["canonical_identity_revision"]["reviewed_boundaries"]
        check_binding(review_binding)
        review = read_json(review_binding["path"])
    else:
        from seal_badminton_repaired_release import verify_repaired_aggregate

        aggregate_path, _ = verify_repaired_aggregate(root, manifest, output)
    frozen = [binding(manifest_path), binding(native_path)]
    counts, jumps, swaps, sources = defaultdict(int), [], [], {}
    for sid, entry in manifest["exports"].items():
        dependencies = [binding(entry[key]) for key in ("path", "metadata")]
        if any(
            entry.get(name) is not None and entry[name] != dependency["sha256"]
            for name, dependency in zip(("sha256", "metadata_sha256"), dependencies)
        ):
            raise ValueError("Body tensors or metadata changed")
        data, rows = joblib.load(entry["path"]), read_rows(entry["metadata"])
        pelvis = np.asarray(data["transl"], dtype=np.float32)
        if len(rows) != entry["accepted"] or pelvis.shape != (len(rows), 3) or not np.isfinite(pelvis).all():
            raise ValueError("Body tensor/metadata count or finite pelvis contract changed")
        if data["motion_id"] != [r["motion_id"] for r in rows]:
            raise ValueError("Body tensor/metadata identity order changed")
        windows = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(rows):
            window = row["motion_id"].split(":")[1]
            windows[window][row["lidar_index"]].append(
                dict(
                    motion_id=row["motion_id"],
                    scan=row["lidar_index"],
                    time_us=row["scan_mid_us"],
                    pelvis=pelvis[index],
                    metadata_index=index,
                )
            )
        local_jumps, local_swaps, edges = [], [], 0
        for window, frames in windows.items():
            before = None
            for scan, people in sorted(frames.items()):
                if before is not None:
                    old_scan, old = before
                    dt = (people[0]["time_us"] - old[0]["time_us"]) / 1e6
                    if scan == old_scan + 1 and 0 < dt <= 0.15:
                        old_map = {p["motion_id"]: p for p in old}
                        same = [p for p in people if p["motion_id"] in old_map]
                        edges += len(same)
                        for actor in same:
                            distance = float(np.linalg.norm(actor["pelvis"] - old_map[actor["motion_id"]]["pelvis"]))
                            if distance > 0.75:
                                local_jumps.append(
                                    dict(
                                        source_id=sid,
                                        window=window,
                                        before_scan=old_scan,
                                        after_scan=scan,
                                        motion_id=actor["motion_id"],
                                        distance_m=distance,
                                        elapsed_s=dt,
                                    )
                                )
                        for ia, a in enumerate(same):
                            for b in same[ia + 1 :]:
                                pa, pb = a["pelvis"], b["pelvis"]
                                oa, ob = old_map[a["motion_id"]]["pelvis"], old_map[b["motion_id"]]["pelvis"]
                                own = float(np.linalg.norm(pa - oa) + np.linalg.norm(pb - ob))
                                cross = float(np.linalg.norm(pa - ob) + np.linalg.norm(pb - oa))
                                if own > 1.5 and cross < 0.5 and own - cross > 1:
                                    local_swaps.append(
                                        dict(
                                            source_id=sid,
                                            window=window,
                                            before_scan=old_scan,
                                            after_scan=scan,
                                            motion_ids=[a["motion_id"], b["motion_id"]],
                                            own_motion_m=own,
                                            cross_motion_m=cross,
                                        )
                                    )
                before = scan, people
        counts["person_frames"] += len(rows)
        counts["adjacent_person_edges"] += edges
        counts["pelvis_jump_candidates"] += len(local_jumps)
        counts["mutual_swap_candidates"] += len(local_swaps)
        jumps.extend(local_jumps)
        swaps.extend(local_swaps)
        sources[sid] = dict(
            person_frames=len(rows),
            adjacent_person_edges=edges,
            pelvis_jump_candidates=len(local_jumps),
            mutual_swap_candidates=len(local_swaps),
            dependencies=dependencies,
        )
        print(
            json.dumps(dict(source=sid, **{k: v for k, v in sources[sid].items() if k != "dependencies"})), flush=True
        )
        write_json(
            output / "progress.json", dict(state="running", counts=dict(counts), completed_sources=list(sources))
        )
        frozen.extend(dependencies)
    if counts["person_frames"] != manifest["accepted"] or (
        review is None and counts["person_frames"] != native["accepted_checked"]
    ):
        raise ValueError("Continuity audit did not cover every native-verified body")
    for dependency in frozen:
        check_binding(dependency)
    producer = output / "producer.py"
    if producer.exists() and producer.read_bytes() != Path(__file__).read_bytes():
        raise ValueError("Continuity audit producer snapshot changed")
    shutil.copyfile(__file__, producer)
    same = (
        {
            (case["event"]["motion_id"], case["event"]["before_scan"], case["event"]["after_scan"])
            for case in review["cases"]
            if case["verdict"] == "same_person"
        }
        if review
        else set()
    )
    unresolved = [
        event for event in jumps if (event["motion_id"], event["before_scan"], event["after_scan"]) not in same
    ]
    result = dict(
        schema="badminton-fresh-continuity-audit-v1",
        state="complete",
        screening_passed_no_flags=not jumps and not swaps,
        reviewed_continuity_passed=not unresolved and not swaps,
        reviewed_boundaries=review_binding,
        unresolved_jump_candidates=unresolved,
        physical_identity_ground_truth=False,
        native_verification=frozen[1],
        aggregate_native_contract=binding(aggregate_path),
        body_manifest=frozen[0],
        producer=binding(producer),
        counts=dict(counts),
        sources=sources,
        pelvis_jump_candidates=jumps,
        mutual_swap_candidates=swaps,
        limits=[
            "Checks actual consecutive native scans inside each fitting window; gaps and different raw identities are not forcibly joined.",
            "Pelvis jump >0.75m in <=150ms is a review candidate, not a changed admission gate.",
            "Mutual-swap screening reuses own>1.5m, cross<0.5m, own-cross>1m; absence of flags does not establish athlete identity truth.",
        ],
    )
    write_json(output / "audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if not args.root.resolve().is_relative_to(PROJ_ROOT / "outputs"):
        parser.error("Select an explicit fresh production outputs root")
    torch.set_num_threads(1)
    if args.output_root and not args.output_root.resolve().is_relative_to(args.root.resolve()):
        parser.error("Derived continuity evidence must remain inside the selected repair run")
    result = audit(args.root, args.manifest, args.output_root)
    print(
        json.dumps(
            dict(
                state=result["state"],
                counts=result["counts"],
                screening_passed_no_flags=result["screening_passed_no_flags"],
            )
        )
    )


if __name__ == "__main__":
    main()
