"""Adjudicate native ID boundaries using the other camera and fixed-world joint rays."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.multiview import associate_poses
from hmr4d.lidar_mocap.repair_execution import write_progress_json as write_json
from hmr4d.lidar_mocap.rally_pipeline import load_run_session, read_json
from hmr4d.lidar_mocap.relocations import load_pilot_run_relocations
from hmr4d.lidar_mocap.teacher import load_teacher_track


def pose_distance(a, b, height):
    valid = (a[:, 2] >= 0.35) & (b[:, 2] >= 0.35)
    indices = np.array([5, 6, 11, 12])[valid[[5, 6, 11, 12]]]
    if len(indices) < 3:
        return 1e6
    return float(np.median(np.linalg.norm(a[indices, :2] - b[indices, :2], axis=1))) / max(height, 35)


def individual_match(kp, box, poses):
    distances = [pose_distance(kp, p, box[3] - box[1]) for p in poses]
    if not distances:
        return None
    order = np.argsort(distances)
    if distances[order[0]] > 0.2 or (len(order) > 1 and distances[order[1]] - distances[order[0]] < 0.035):
        return None
    return int(order[0])


def track_opposite(entries, states, timestamp, next_id):
    states = {k: s for k, s in states.items() if 0 < timestamp - s["time"] <= 350000}
    keys = sorted(states)
    cost = np.full((len(keys), len(entries)), 1e6)
    for i, k in enumerate(keys):
        old = states[k]
        dt = (timestamp - old["time"]) / 1e6
        velocity = old.get("velocity", np.zeros(3))
        if np.linalg.norm(velocity) > 8:
            velocity = np.zeros(3)
        for j, e in enumerate(entries):
            d = pose_distance(old["pose"], e["pose"], old["height"])
            if d > 1e5:
                d = np.linalg.norm((old["box"][:2] + old["box"][2:] - e["box"][:2] - e["box"][2:]) / 2) / max(
                    old["height"], 35
                )
            if e["root"] is not None and old["root"] is not None:
                world = np.linalg.norm(e["root"] - old["root"] - velocity * dt)
                cost[i, j] = 0.65 * d + 0.35 * world / 0.6
            else:
                cost[i, j] = d
    matches = {}
    if cost.size:
        rr, cc = linear_sum_assignment(cost)
        for i, j in zip(rr, cc):
            other = np.concatenate([np.delete(cost[i], j), np.delete(cost[:, j], i)])
            if cost[i, j] < 0.65 and (not len(other) or other.min() - cost[i, j] >= 0.10):
                matches[int(j)] = keys[i]
    for j, e in enumerate(entries):
        if j in matches:
            k = matches[j]
        elif len(keys) and cost[:, j].min() < 0.65:
            e["identity"] = None
            continue
        else:
            k = next_id
            next_id += 1
        old = states.get(k)
        velocity = (
            np.zeros(3)
            if old is None or old["root"] is None or e["root"] is None
            else (e["root"] - old["root"]) / ((timestamp - old["time"]) / 1e6)
        )
        states[k] = dict(
            time=timestamp, root=e["root"], pose=e["pose"], height=e["height"], velocity=velocity, box=e["box"]
        )
        e["identity"] = k
    return states, next_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2")
    p.add_argument("--partial", action="store_true")
    p.add_argument("--source", action="append", help="Review complete cases for explicit fresh sources only")
    args = p.parse_args()
    source_plan = read_json(args.root / "case_plan.json")
    source_input = read_json(source_plan["source_union"])
    fresh_sources = source_input.get("schema") == "badminton-new-source-repair-input-v1"
    source_root = Path(source_input["source_root"]) if fresh_sources else PROJ_ROOT / "outputs/badminton_rallies_v1"
    cal = CalibrationSnapshot.load(source_root / "calibration/calibration_snapshot.json")
    relocations = load_pilot_run_relocations(source_root, cal.data)
    boundary_path = args.root / "identity_evidence/boundaries.json"
    cases = read_json(boundary_path)
    if args.source:
        if not fresh_sources or not set(args.source).issubset({c["event"]["source_id"] for c in cases}):
            raise ValueError("Select existing fresh-source case IDs")
        cases = [case for case in cases if case["event"]["source_id"] in args.source]
    input_hashes = {}

    def record_input(path):
        if fresh_sources:
            name = str(Path(path).resolve())
            if name not in input_hashes:
                input_hashes[name] = file_sha(path)

    for path in (boundary_path, Path(source_plan["source_union"]), Path(__file__)):
        record_input(path)
    cached = {}
    if fresh_sources and not args.source:
        for sid in {case["event"]["source_id"] for case in cases}:
            path = args.root / "identity_evidence/source_reviews" / (sid + ".json")
            if not path.exists():
                continue
            report = read_json(path)
            expected = {case["index"]: case for case in cases if case["event"]["source_id"] == sid}
            actual = {case["index"]: case for case in report["cases"]}
            if (
                report.get("selected_sources") != [sid]
                or set(expected) != set(actual)
                or not report.get("inputs")
                or any(file_sha(name) != sha for name, sha in report["inputs"].items())
                or any(case["missing_native_context"] for case in actual.values())
            ):
                raise ValueError("Source identity review cache has incomplete or changed native evidence: " + sid)
            if any(
                any(
                    actual[i].get(key) != expected[i][key]
                    for key in ("source_window", "event", "positions", "context_positions")
                )
                for i in expected
            ):
                raise ValueError("Source identity review refers to different cases")
            cached.update(actual)
            input_hashes.update(report["inputs"])
            record_input(path)
    if not fresh_sources:
        replay = read_json(PROJ_ROOT / "outputs/badminton_body_repair_v1/tracking_final/tracking_replay.json")
        windows = {(w["source_id"], w["window_id"]): w for w in replay["windows"]}
    sessions = {}
    loaded = {}
    results = []
    counts = Counter()
    origin = np.asarray(cal.reference_plane["origin_viewer_m"])
    normal = np.asarray(cal.reference_plane["normal_viewer"])
    for case in cases:
        if case["index"] in cached:
            result = cached[case["index"]]
            results.append(result)
            counts[result["status"]] += 1
            continue
        sid = case["event"]["source_id"]
        source = Path(case["source_window"])
        key = (sid, source.name)
        if sid not in sessions:
            sessions[sid] = load_run_session(source_root, cal, sid, relocations=relocations)
        session = sessions[sid]
        if key not in loaded:
            record_input(source / "teacher/manifest.json")
            record_input(source / "teacher/detections.pt")
            manifest = read_json(source / "teacher/manifest.json")
            rows = manifest["provenance"]["frame_rows"]
            record_to_pos = {r["record_index"]: i for i, r in enumerate(rows)}
            observations = defaultdict(list)
            if fresh_sources:
                from hmr4d.lidar_mocap.new_source_repair import native_teacher_mapping

                mapping = native_teacher_mapping(source)
            else:
                mapping = np.load(windows[key]["mapping"])["observations"]
            detections = torch.load(source / "teacher/detections.pt", map_location="cpu", weights_only=False)[
                "detections"
            ]
            old_ids = {(int(r[0]), int(r[1])): int(r[2]) for r in mapping}
            kp = {}
            for info in manifest["tracks"]:
                record_input(source / "teacher" / info["path"])
                track = load_teacher_track(source / "teacher" / info["path"])
                for record, box, points in zip(track["record_index"], track["bbx_xyxy"], track["kp2d"]):
                    kp[record_to_pos[int(record)], np.asarray(box, np.float32).tobytes()] = np.asarray(points)
            for pos, frame in enumerate(detections):
                for j, d in enumerate(frame):
                    if (pos, j) not in old_ids:
                        continue
                    box = np.asarray(d["bbx_xyxy"], np.float32)
                    observations[pos].append(
                        dict(original_id=old_ids[pos, j], detection_index=j, box=box, kp=kp.get((pos, box.tobytes())))
                    )
            loaded[key] = (rows, observations)
        rows, original = loaded[key]
        times = np.asarray([r["common_time_us"] for r in session.camera_rows("cam1")], np.int64)
        states = {}
        next_id = 1
        votes = defaultdict(list)
        timeline = []
        missing = 0
        presence = []
        for pos in case["context_positions"]:
            native2 = rows[pos]
            target = native2["common_time_us"]
            q = np.searchsorted(times, target)
            candidates = [i for i in (q - 1, q) if 0 <= i < len(times)]
            index1 = int(min(candidates, key=lambda i: abs(times[i] - target)))
            paths = [
                args.root / "identity_evidence/native_poses" / sid / cam / f"{index:07d}.json"
                for cam, index in [("cam1", index1), ("cam2", native2["record_index"])]
            ]
            if not all(path.exists() for path in paths):
                missing += 1
                continue
            for path in paths:
                record_input(path)
            a, b = map(read_json, paths)
            pa = np.asarray(a["keypoints"]).reshape(-1, 17, 3)
            pb = np.asarray(b["keypoints"]).reshape(-1, 17, 3)
            entries = [
                dict(cam1=i, cam2=None, root=None, pose=points, height=box[3] - box[1], box=np.asarray(box), link=None)
                for i, (points, box) in enumerate(zip(pa, a["boxes"]))
            ]
            for pair in associate_poses(pa, pb, cal):
                if not pair["accepted"]:
                    continue
                xyz = np.asarray(pair["triangulated_joints"])
                valid = np.asarray(pair["triangulated_valid"])
                hip = [i for i in (11, 12) if valid[i]]
                if not hip:
                    continue
                root = xyz[hip].mean(0)
                height = float((root - origin) @ normal)
                if not 0.15 <= height <= 2.8:
                    continue
                index = pair["cam1_index"]
                box = a["boxes"][index]
                entries[index].update(cam2=pair["cam2_index"], root=root, link=pair)
            states, next_id = track_opposite(entries, states, a["native"]["common_time_us"], next_id)
            physical = {e["cam2"]: e for e in entries if e.get("identity") is not None and e["cam2"] is not None}
            by_camera1 = {e["cam1"]: e for e in entries if e.get("identity") is not None}
            presence.append(
                dict(
                    position=pos,
                    people={str(e["identity"]): e["root"].tolist() for e in physical.values() if e["root"] is not None},
                )
            )
            frame = []
            for old in original[pos]:
                if old["kp"] is None:
                    continue
                j = individual_match(old["kp"], old["box"], pb)
                direct = [r for r in associate_poses(pa, np.asarray(old["kp"])[None], cal) if r["accepted"]]
                if direct and direct[0]["cam1_index"] in by_camera1:
                    direct_link = direct[0]
                    e = by_camera1[direct_link["cam1_index"]]
                    xyz = np.asarray(direct_link["triangulated_joints"])
                    mask = np.asarray(direct_link["triangulated_valid"])
                    hip = [h for h in (11, 12) if mask[h]]
                    if not hip:
                        continue
                    old_root = xyz[hip].mean(0)
                    epi = direct_link["median_epipolar_px"]
                elif j in physical:
                    e = physical[j]
                    old_root = e["root"]
                    epi = e["link"]["median_epipolar_px"]
                else:
                    continue
                record = dict(
                    position=pos,
                    original_track_id=old["original_id"],
                    detection_index=old["detection_index"],
                    opposite_identity=e["identity"],
                    cam1_detection=e["cam1"],
                    cam2_detection=j,
                    native_cam1=index1,
                    native_cam2=native2["record_index"],
                    root=old_root.tolist(),
                    median_epipolar_px=epi,
                )
                record["primary_pose_distance"] = (
                    pose_distance(old["kp"], pb[j], old["box"][3] - old["box"][1]) if j is not None else None
                )
                votes[old["original_id"]].append(record)
                frame.append(record)
            timeline.extend(frame)
        summaries = []
        for side, pos in zip(("before", "after"), case["positions"]):
            tid = int(case["event"][side + "_motion_id"].split(":")[-2])
            samples = [
                r
                for r in votes[tid]
                if (pos - 12 <= r["position"] <= pos if side == "before" else pos <= r["position"] <= pos + 12)
            ]
            counter = Counter(r["opposite_identity"] for r in samples)
            best, n = counter.most_common(1)[0] if counter else (None, 0)
            ratio = n / max(len(samples), 1)
            summaries.append(
                dict(
                    original_track_id=tid,
                    identity=best,
                    samples=len(samples),
                    support=n,
                    ratio=ratio,
                    confident=n >= 4 and ratio >= 0.8,
                )
            )
        if missing and not args.partial:
            raise ValueError(f'Missing native context for identity case {case["index"]}')
        status = "insufficient_or_ambiguous_native_evidence"
        coexist = 0
        if all(s["confident"] for s in summaries):
            if summaries[0]["identity"] == summaries[1]["identity"]:
                status = "same_native_person"
            else:
                ka, kb = [str(s["identity"]) for s in summaries]
                coexist = sum(
                    ka in f["people"]
                    and kb in f["people"]
                    and np.linalg.norm(np.asarray(f["people"][ka]) - f["people"][kb]) > 0.35
                    for f in presence
                )
                if coexist >= 3:
                    status = "different_native_people"
        counts[status] += 1
        result = dict(
            case,
            status=status,
            votes=summaries,
            timeline=timeline,
            presence=presence,
            different_person_coexistence_frames=int(coexist),
            missing_native_context=missing,
        )
        results.append(result)
        write_json(
            args.root / "identity_evidence/review_progress.json", dict(counts=dict(counts), completed=len(results))
        )
        print(dict(index=case["index"], source=sid, status=status, votes=summaries), flush=True)
    output = args.root / "identity_evidence/native_identity_review.json"
    if args.source:
        if len(args.source) != 1:
            raise ValueError("One source per resumable review process")
        output = args.root / "identity_evidence/source_reviews" / (args.source[0] + ".json")
    write_json(
        output,
        dict(
            counts=dict(counts),
            cases=results,
            partial=args.partial or bool(args.source),
            selected_sources=args.source,
            inputs=input_hashes,
            limits=["Fixed native pose rays and opposite-camera continuity, not manual identity ground truth."],
        ),
    )


if __name__ == "__main__":
    main()
