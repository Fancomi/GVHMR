"""Adapt explicit fresh-source identity inventories to the established native review."""

from pathlib import Path
import hashlib
import numpy as np
from .body_repair import published_windows
from .dual_view import file_sha
from .new_source_repair import native_teacher_mapping
from .pipeline import write_json
from .rally_pipeline import read_json


def prepare_identity(root, inventory_path=None):
    root = Path(root)
    original = read_json(root / "case_plan.json")
    if file_sha(original["source_union"]) != original["source_union_sha256"]:
        raise ValueError("Fresh source input changed")
    directory = root / "identity_evidence"
    windows, endpoints = {}, []
    for sid, window in published_windows(original["source_union"]):
        mapping = native_teacher_mapping(window)
        rows = read_json(window / "teacher/manifest.json")["provenance"]["frame_rows"]
        ranges = {int(key): mapping[mapping[:, 2] == key] for key in np.unique(mapping[:, 2])}
        windows[sid + ":" + window.name] = (window, rows, ranges)
        for track, values in ranges.items():
            positions = sorted(set(map(int, values[:, 0])))
            endpoints.append(
                dict(
                    source_id=sid,
                    window=window.name,
                    original_track_id=track,
                    source_window=str(window),
                    first_position=positions[0],
                    last_position=positions[-1],
                    first_detection_index=int(values[values[:, 0].argmin(), 1]),
                    last_detection_index=int(values[values[:, 0].argmax(), 1]),
                    native_first=rows[positions[0]],
                    native_last=rows[positions[-1]],
                )
            )
    if inventory_path is None:
        path = directory / "boundary_inventory.pending.json"
        write_json(
            path,
            dict(
                schema="badminton-new-source-identity-inventory-v1",
                review_status="pending",
                source_union_sha256=original["source_union_sha256"],
                tracklets=endpoints,
                cases=[],
                evidence_files={},
                note="Observed endpoints only; no identity decisions or complete-boundary claim",
            ),
        )
        return dict(state="pending_boundary_inventory", path=str(path))
    inventory = read_json(inventory_path)
    if (
        inventory.get("schema") != "badminton-new-source-identity-inventory-v1"
        or inventory.get("review_status") != "reviewed"
        or inventory.get("source_union_sha256") != original["source_union_sha256"]
        or not inventory.get("evidence_files")
    ):
        raise ValueError("Boundary inventory scope has not been verified against native evidence")
    if any(file_sha(path) != sha for path, sha in inventory["evidence_files"].items()):
        raise ValueError("Boundary inventory evidence changed")
    cases = []
    for item in inventory["cases"]:
        key = item["source_id"] + ":" + item["window"]
        source, rows, ranges = windows[key]
        before, after = map(int, item["original_track_ids"])
        if before == after:
            raise ValueError("Same original identity is not a new boundary")
        positions = [int(ranges[before][:, 0].max()), int(ranges[after][:, 0].min())]
        context = sorted({i for pos in positions for i in range(pos - 12, pos + 13) if 0 <= i < len(rows)})
        sid = item["source_id"]
        event = dict(
            source_id=sid,
            before_motion_id=f"{sid}:{source.name}:{before}:0",
            after_motion_id=f"{sid}:{source.name}:{after}:0",
            replay_endpoints=[dict(native_record_index=rows[pos]["record_index"]) for pos in positions],
            identity_kind="detector_tracklet_descriptor_not_a_fitted_label",
        )
        cases.append(
            dict(
                index=len(cases), source_window=str(source), positions=positions, context_positions=context, event=event
            )
        )
    path = directory / "boundaries.json"
    write_json(path, cases)
    write_json(
        directory / "case_plan.json",
        dict(
            original,
            clips=[],
            selected_scans={sid: [] for sid in original["selected_scans"]},
            unresolved_native_endpoints=str(path),
            boundary_inventory=str(Path(inventory_path).resolve()),
            boundary_inventory_sha256=file_sha(inventory_path),
        ),
    )
    return dict(state="prepared", cases=len(cases))


def seal_identity(root, abstention_policy=None):
    root = Path(root)
    evidence = root / "identity_evidence"
    plan = read_json(root / "case_plan.json")
    scope = read_json(evidence / "case_plan.json")
    if file_sha(plan["source_union"]) != plan["source_union_sha256"]:
        raise ValueError("Fresh identity source input changed")
    if file_sha(scope["boundary_inventory"]) != scope["boundary_inventory_sha256"]:
        raise ValueError("Reviewed boundary inventory changed")
    review = read_json(evidence / "native_identity_review.json")
    boundaries = read_json(evidence / "boundaries.json")
    poses = read_json(evidence / "pose_receipt.json")
    if review.get("partial") or len(review["cases"]) != len(boundaries) or poses.get("state") != "complete":
        raise ValueError("Native identity review has incomplete coverage")
    if not review.get("inputs") or any(file_sha(path) != sha for path, sha in review["inputs"].items()):
        raise ValueError("Native identity review inputs are missing or changed")
    policy = None
    if abstention_policy is not None:
        policy = read_json(abstention_policy)
        if (
            policy.get("schema") != "badminton-new-source-identity-abstention-v1"
            or policy.get("review_status") != "reviewed"
            or policy.get("action") != "keep_unproven_original_tracklets_separate"
            or policy.get("source_union_sha256") != plan["source_union_sha256"]
            or policy.get("boundary_inventory_sha256") != scope["boundary_inventory_sha256"]
            or not policy.get("evidence_files")
            or any(file_sha(path) != sha for path, sha in policy["evidence_files"].items())
        ):
            raise ValueError("Conservative identity abstention policy is not bound to this reviewed evidence")
    windows = {}
    for sid, source in published_windows(plan["source_union"]):
        windows[sid + ":" + source.name] = dict(
            source_window=str(source),
            source_teacher_manifest_sha256=file_sha(source / "teacher/manifest.json"),
            links=[],
            cannot_link=[],
            drop_spans=[],
            review_cases=[],
        )
    pending = []
    for expected, case in zip(boundaries, review["cases"]):
        if any(
            case.get(key) != expected[key]
            for key in ("index", "source_window", "event", "positions", "context_positions")
        ):
            raise ValueError("Native review does not match the frozen boundary inventory")
        key = case["event"]["source_id"] + ":" + Path(case["source_window"]).name
        target = windows[key]
        ids = [vote["original_track_id"] for vote in case["votes"]]
        expected_ids = [int(case["event"][side + "_motion_id"].split(":")[-2]) for side in ("before", "after")]
        if ids != expected_ids:
            raise ValueError("Identity votes refer to different original tracklets")
        if case["status"] == "same_native_person":
            target["links"].append(
                dict(tracks=ids, case=case["index"], method="native_opposite_camera_continuity_and_epipolar_pose")
            )
        elif case["status"] == "different_native_people":
            target["cannot_link"].append(ids)
        else:
            pending.append(case["index"])
        target["review_cases"].append(case["index"])
    paths = [
        evidence / "native_identity_review.json",
        evidence / "pose_receipt.json",
        evidence / "boundaries.json",
        Path(scope["boundary_inventory"]),
    ]
    paths += [Path(path) for path in review["inputs"]]
    if policy is not None:
        endpoints_path = evidence / "boundary_inventory.pending.json"
        endpoints = read_json(endpoints_path)
        if endpoints["source_union_sha256"] != plan["source_union_sha256"]:
            raise ValueError("Raw tracklet reset inventory belongs to another source")
        for target in windows.values():
            target["reset_tracklets"] = []
        for entry in endpoints["tracklets"]:
            key = entry["source_id"] + ":" + entry["window"]
            windows[key]["reset_tracklets"].append(int(entry["original_track_id"]))
        for target in windows.values():
            target["reset_tracklets"] = sorted(set(target["reset_tracklets"]))
            parent = {key: key for key in target["reset_tracklets"]}

            def identity(key):
                while parent[key] != key:
                    key = parent[key]
                return key

            for link in target["links"]:
                a, b = map(identity, link["tracks"])
                parent[b] = a
            if any(identity(a) == identity(b) for a, b in target["cannot_link"]):
                raise ValueError("Geometric identity graph contradicts a witnessed different-person constraint")
        paths += [Path(abstention_policy), endpoints_path, *map(Path, policy["evidence_files"])]
    paths += sorted((evidence / "native_poses").glob("*/*/*.json"))
    producer_sources = {}
    frozen_paths = []
    for path in paths:
        if path.suffix == ".py":
            content = path.read_bytes()
            sha = hashlib.sha256(content).hexdigest()
            snapshot = evidence / "producers/runtime" / (path.stem + "_" + sha[:16] + ".py")
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if snapshot.exists() and snapshot.read_bytes() != content:
                raise ValueError("Frozen identity producer source changed")
            if not snapshot.exists():
                snapshot.write_bytes(content)
            producer_sources[str(path)] = dict(path=str(snapshot), sha256=sha)
            frozen_paths.append(snapshot)
        else:
            frozen_paths.append(path)
    result = dict(
        schema="badminton-native-identity-decisions-v1",
        review_status="pending" if pending and policy is None else "reviewed",
        source_union_sha256=plan["source_union_sha256"],
        calibration_digest=poses["config"]["calibration_digest"],
        windows=windows,
        unresolved_cases=pending if policy is None else [],
        abstained_relationship_cases=pending if policy is not None else [],
        abstention_policy=str(Path(abstention_policy).resolve()) if policy is not None else None,
        evidence_files={str(p): file_sha(p) for p in frozen_paths},
        producer_sources=producer_sources,
        training_published=False,
        limits=["Local detector identities; no old-scene manual overrides"],
    )
    if policy is not None:
        result["limits"].append(
            "Abstained relationships remain unknown. Their raw tracklets are kept separate; "
            "reviewed processing decisions do not assert complete physical-person identity truth."
        )
    write_json(evidence / "identity_decisions.json", result)
    return dict(
        state=result["review_status"],
        windows=len(windows),
        unresolved=len(result["unresolved_cases"]),
        abstained=len(result["abstained_relationship_cases"]),
    )


def render_endpoints(root, *, offset=0, limit=20):
    """Render bounded original endpoint pairs; draw boxes only in their declared rectified space."""
    import cv2
    import hashlib
    import html
    import torch
    from .calibration import CalibrationSnapshot
    from .rally_pipeline import load_run_session
    from .teacher import _undistorter

    root = Path(root)
    inventory_path = root / "identity_evidence/boundary_inventory.pending.json"
    inventory = read_json(inventory_path)
    plan = read_json(root / "new_source_input.json")
    if inventory["source_union_sha256"] != file_sha(root / "new_source_input.json"):
        raise ValueError("Endpoint inventory belongs to another input")
    source_root = Path(plan["source_root"])
    cal = CalibrationSnapshot.load(source_root / "calibration/calibration_snapshot.json")
    sessions, detectors = {}, {}
    entries = []
    directory = root / "identity_evidence/endpoint_previews"
    directory.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    for ordinal, entry in enumerate(inventory["tracklets"][offset : offset + limit], offset):
        sid = entry["source_id"]
        if sid not in sessions:
            sessions[sid] = load_run_session(source_root, cal, sid)
        session = sessions[sid]
        window = Path(entry["source_window"])
        if str(window) not in detectors:
            detectors[str(window)] = torch.load(
                window / "teacher/detections.pt", map_location="cpu", weights_only=False
            )["detections"]
        rows1 = [row for row in session.camera_rows("cam1") if row["common_time_us"] is not None]
        for side in ("first", "last"):
            primary = entry["native_" + side]
            secondary = min(
                rows1, key=lambda row: (abs(row["common_time_us"] - primary["common_time_us"]), row["record_index"])
            )
            if abs(secondary["common_time_us"] - primary["common_time_us"]) > 8500:
                entries.append(
                    dict(tracklet=ordinal, side=side, state="no_opposite_frame_within_existing_8_5ms_tolerance")
                )
                continue
            files, native = {}, {}
            for cam, row in (("cam1", secondary), ("cam2", primary)):
                image = session.read_image(cam, row["record_index"])
                raw_sha = hashlib.sha256(image.tobytes()).hexdigest()
                image = _undistorter(cal.camera(cam))(image)
                if cam == "cam2":
                    detection = detectors[str(window)][entry[side + "_position"]][entry[side + "_detection_index"]]
                    box = np.rint(detection["bbx_xyxy"]).astype(int)
                    cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 255, 255), 3)
                path = directory / f"tracklet{ordinal:05d}-{side}-{cam}.jpg"
                if not cv2.imwrite(str(path), image):
                    raise ValueError("Endpoint image write failed")
                files[cam] = dict(file=path.name, sha256=file_sha(path))
                native[cam] = dict(row, raw_bgr_sha256=raw_sha)
            entries.append(
                dict(
                    tracklet=ordinal,
                    side=side,
                    source_id=sid,
                    original_track_id=entry["original_track_id"],
                    native=native,
                    images=files,
                    state="rendered",
                )
            )
    receipt = dict(
        scope="native_endpoint_evidence_only_not_identity_admission",
        inventory_sha256=file_sha(inventory_path),
        calibration_digest=cal.data["snapshot_digest"],
        offset=offset,
        limit=limit,
        total_tracklets=len(inventory["tracklets"]),
        rows=entries,
        image_space="undistorted_K_preserved; yellow primary detector box; native PTS and raw SHA retained",
    )
    write_json(directory / f"endpoints-{offset}.json", receipt)
    sections = []
    for entry in entries:
        title = html.escape(f"Tracklet {entry['tracklet']} / {entry['side']} / {entry['state']}")
        images = "".join(
            f'<figure><figcaption>{cam}</figcaption><img width="900" src="{info["file"]}"></figure>'
            for cam, info in entry.get("images", {}).items()
        )
        sections.append(f"<section><h2>{title}</h2>{images}</section>")
    page = (
        '<meta charset="utf-8"><title>Native identity endpoints</title><p>Undistorted original RGB, unchanged K. Yellow: primary detector box. Pending identity evidence.</p>'
        + "".join(sections)
    )
    (directory / f"index-{offset}.html").write_text(page, encoding="utf-8")
    return dict(page=str(directory / f"index-{offset}.html"), rendered=sum(e["state"] == "rendered" for e in entries))
