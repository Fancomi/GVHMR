"""Resumable native RGB observations for every existing accepted person-frame.

One shared detector/pose model observes both cameras. Streaming native decoding
checks actual PTS; all fitted athletes (including rejected labels) participate
in association and foreign-person visibility. No body or camera is optimized.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from hmr4d import PROJ_ROOT
from .calibration import CalibrationSnapshot
from .dual_view import _bound_output, _canonical, _read, file_sha
from .rgb_observations import RGBKeypointChecker, add_raw_pixel_observations, write_json
from .rally_pipeline import window_directory, load_run_session
from .relocations import load_pilot_run_relocations


def _teacher_locations(source, dataset, sid, window):
    """Use the declared initial teacher when repaired bodies have no detector cache."""
    directory = window_directory(source, dataset, sid, window)
    config = _read(directory / "window.json")
    primary = directory / "teacher"
    if not (primary / "detections.pt").is_file():
        declared = config.get("body_repair", {}).get("source_window")
        if declared:
            initial = Path(declared).resolve()
            expected = (source.parent / "initial_teacher/sessions" / sid / "windows" / window).resolve()
            if source.name != "final_labels" or initial != expected:
                raise ValueError("Initial teacher leaves the declared repair window")
            primary = initial / "teacher"
    locations = [(config["settings"]["camera"], primary, config)]
    if source.name == "final_labels":
        for camera in ("cam1", "cam2"):
            if camera != config["settings"]["camera"]:
                secondary = (
                    source.parent / "secondary_teachers" / camera / "sessions" / sid / "windows" / window / "teacher"
                )
                locations.append((camera, secondary, config))
    return locations


def _formal_teacher_proof(source, dataset, calibration):
    """Require the existing bound physical-input mapping for formal cache reuse."""
    registry_path = source / "source_registry.json"
    if not registry_path.exists():
        return None
    registry = _read(registry_path)
    if registry.get("schema") != "badminton-formal-derived-source-registry-v1":
        return None
    plan_path = source / "repair_plan.json"
    plan = _read(plan_path)
    input_path = Path(plan["source_union"]).resolve()
    if input_path != source.parent / "new_source_input.json" or file_sha(input_path) != plan["source_union_sha256"]:
        raise ValueError("Formal teacher reuse requires its exact prepared source input")
    prepared = _read(input_path)
    proof_binding = prepared.get("formal_derivation", {}).get("input_equivalence", {})
    proof_path = Path(proof_binding.get("path", "")).resolve()
    if proof_path != source.parent.parent / "formal_input_equivalence.json" or file_sha(
        proof_path
    ) != proof_binding.get("sha256"):
        raise ValueError("Formal teacher reuse requires its bound input-equivalence proof")
    proof = _read(proof_path)
    evidence = proof.get("evidence", {})
    if (
        proof.get("schema") != "badminton-review-formal-input-equivalence-v1"
        or proof.get("state") != "complete"
        or proof.get("training_published") is not False
        or proof.get("fitted_tensors_imported") is not False
        or prepared.get("schema") != "badminton-new-source-repair-input-v1"
        or prepared.get("source_registry_sha256") != file_sha(registry_path)
        or prepared.get("formal_derivation", {}).get("camera_admission_handoff")
        != registry.get("camera_admission_handoff")
        or prepared.get("calibration_digest") != calibration.data["snapshot_digest"]
        or plan.get("calibration_digest") != calibration.data["snapshot_digest"]
        or evidence.get("formal_registry", {}).get("sha256") != file_sha(registry_path)
        or evidence.get("formal_calibration", {}).get("sha256")
        != file_sha(source / "calibration/calibration_snapshot.json")
        or set(proof.get("sources", {})) != set(registry["sources"])
        or set(dataset["exports"]) != set(registry["sources"])
    ):
        raise ValueError("Formal teacher input/camera equivalence does not bind this publication")
    for sid, entry in registry["sources"].items():
        mapped = proof["sources"][sid]
        if (
            mapped.get("original_inputs_exact") is not True
            or mapped.get("cameras_court_ground_bbox_clock_exact") is not True
            or mapped.get("admitted_index") != entry["source_index"]
        ):
            raise ValueError("Formal teacher source has no exact physical calibration/input mapping")
    return dict(proof=proof_binding, source_input=dict(path=str(input_path), sha256=plan["source_union_sha256"]))


def source_selection(source):
    dataset_path = source / "datasets/dataset_manifest.json"
    dataset = _read(dataset_path)
    snapshot = source / "calibration/calibration_snapshot.json"
    if dataset.get("review_only") or (snapshot.exists() and _read(snapshot).get("review_only")):
        raise ValueError("Review-only pilot cannot be used as a training export source")
    sources = dict(
        dataset_manifest=dict(path=str(dataset_path), sha256=file_sha(dataset_path)),
        calibration=file_sha(source / "calibration/calibration_snapshot.json"),
        files=[],
    )
    if dataset.get("source_registry"):
        path = Path(dataset["source_registry"])
        if file_sha(path) != dataset["source_registry_sha256"]:
            raise ValueError("Source registry changed")
        sources["files"].append(dict(path=str(path), sha256=file_sha(path)))
    plan_path = source / "repair_plan.json"
    if plan_path.exists():
        sources["files"].append(dict(path=str(plan_path), sha256=file_sha(plan_path)))
        plan = _read(plan_path)
        prepared_path = Path(plan["source_union"])
        sources["files"].append(dict(path=str(prepared_path), sha256=file_sha(prepared_path)))
        proof = _read(prepared_path).get("formal_derivation", {}).get("input_equivalence")
        if proof:
            sources["files"].append(dict(path=proof["path"], sha256=file_sha(proof["path"])))
    for sid, entry in dataset["exports"].items():
        for name in ("path", "metadata"):
            path = Path(entry[name])
            sources["files"].append(dict(path=str(path), sha256=file_sha(path)))
        for window in dataset["selected_windows"][sid]:
            path = window_directory(source, dataset, sid, window) / "quality.jsonl"
            sources["files"].append(dict(path=str(path), sha256=file_sha(path)))
            window_path = path.parent / "window.json"
            sources["files"].append(dict(path=str(window_path), sha256=file_sha(window_path)))
            for _, teacher_directory, _ in _teacher_locations(source, dataset, sid, window):
                if not (teacher_directory / "detections.pt").is_file():
                    continue
                for name in ("manifest.json", "detections.pt"):
                    teacher_path = teacher_directory / name
                    sources["files"].append(dict(path=str(teacher_path), sha256=file_sha(teacher_path)))
    return dataset, sources


def selected_frames(source, dataset, sid):
    accepted = {}
    with Path(dataset["exports"][sid]["metadata"]).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (int(row["lidar_index"]), row["motion_id"])
            if key in accepted:
                raise ValueError("Duplicate source accepted person-frame")
            accepted[key] = row
    frames = {}
    found = set()
    for window in dataset["selected_windows"][sid]:
        path = window_directory(source, dataset, sid, window) / "quality.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                index = int(row["lidar_index"])
                frame = frames.setdefault(
                    index, dict(lidar_index=index, scan_mid_us=row["scan_mid_us"], cameras=row["cameras"], athletes=[])
                )
                if frame["scan_mid_us"] != row["scan_mid_us"] or frame["cameras"] != row["cameras"]:
                    raise ValueError("Source fitted people disagree about native frame/time")
                key = (index, row["motion_id"])
                if key in accepted:
                    if row != accepted[key]:
                        raise ValueError("Source window and exported person metadata differ")
                    found.add(key)
                frame["athletes"].append(
                    dict(motion_id=row["motion_id"], fit_path=row["fit_path"], accepted=key in accepted, qa=row["qa"])
                )
    if found != set(accepted):
        raise ValueError("Selected fit windows do not cover every accepted source label")
    return [frame for _, frame in sorted(frames.items()) if any(p["accepted"] for p in frame["athletes"])]


def _resume_rows(path, expected, camera, decode_profile=None):
    """Recover only an interrupted final write; reject changed/duplicate records."""
    done, frame_count, person_count, seconds = set(), 0, 0, 0.0
    if not path.exists():
        return done, frame_count, person_count, seconds
    with path.open("r+b") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                handle.truncate(start)
                break
            row = json.loads(line)
            if decode_profile is not None and row.get("native_decode_profile") != decode_profile:
                raise ValueError("Resume journal has an incompatible native decode profile")
            index = row["lidar_index"]
            frame = expected.get(index)
            if index in done or frame is None or row["camera_name"] != camera:
                raise ValueError("Unexpected or duplicated observation in resume journal")
            if (
                row["scan_mid_us"] != frame["scan_mid_us"]
                or row["paired_cameras"] != frame["cameras"]
                or {p["motion_id"] for p in row["athletes"]} != {p["motion_id"] for p in frame["athletes"]}
            ):
                raise ValueError("Observation journal source changed")
            done.add(index)
            frame_count += 1
            person_count += sum(p["fit_accepted"] for p in row["athletes"])
            seconds += row["timing_seconds"]["total"]
    return done, frame_count, person_count, seconds


def _cached_frames(paths, calibration_digest, decode_profile=None):
    cache = {}
    for path in paths:
        path = Path(path).resolve()
        report = _read(path)
        report_sha = file_sha(path)
        if report["calibration_digest"] != calibration_digest:
            raise ValueError("Cached detections use a different frozen calibration")
        if "frames_jsonl" in report:
            rows_path = Path(report["frames_jsonl"])
            rows_sha = file_sha(rows_path)

            def rows():
                with rows_path.open("rb") as handle:
                    for line in handle:
                        if not line.endswith(b"\n"):
                            break
                        yield json.loads(line)

            frames = rows()
        else:
            rows_path, rows_sha, frames = path, file_sha(path), report["frames"]
        for frame in frames:
            key = (frame["session_id"], frame["camera_name"], frame["lidar_index"])
            frame["reused_native_evidence"] = dict(
                report_path=str(path),
                report_sha256=report_sha,
                frames_path=str(rows_path),
                frames_sha256=rows_sha,
                models=report["models"],
                checker_sha256=report["checker_sha256"],
            )
            if decode_profile is not None:
                frame["reused_native_evidence"]["camera"] = report.get("camera")
            cache.setdefault(key, frame)
    return cache


def _cache_matches(record, frame):
    if (
        record["scan_mid_us"] != frame["scan_mid_us"]
        or record["paired_cameras"] != frame["cameras"]
        or {p["motion_id"] for p in record["athletes"]} != {p["motion_id"] for p in frame["athletes"]}
    ):
        return False
    expected = {p["motion_id"]: p for p in frame["athletes"]}
    for person in record["athletes"]:
        source = expected[person["motion_id"]]
        if Path(person["fit_path"]).resolve() != Path(source["fit_path"]).resolve() or person["fit_sha256"] != file_sha(
            source["fit_path"]
        ):
            return False
        person["fit_accepted"] = source["accepted"]
    return True


def teacher_detections(
    source, dataset, sid, camera, calibration, session, *, formal_context=None, decode_profile=None, ignored_caches=None
):
    """Reuse only exact native-frame detections; never treat interpolation as presence."""
    import torch

    if formal_context is None:
        formal_context = _formal_teacher_proof(source, dataset, calibration)
    result = {}
    for window in dataset["selected_windows"][sid]:
        locations = [item for item in _teacher_locations(source, dataset, sid, window) if item[0] == camera]
        if not locations:
            continue
        _, teacher_directory, config = locations[0]
        path = teacher_directory / "detections.pt"
        if not path.is_file():
            continue
        manifest_path = teacher_directory / "manifest.json"
        manifest = _read(manifest_path)
        provenance = manifest["provenance"]
        if decode_profile is not None and provenance.get("native_decode_profile") != decode_profile:
            if ignored_caches is not None:
                ignored_caches.append(
                    dict(path=str(manifest_path), reason="native_decode_profile_missing_or_different")
                )
            continue
        expected_hash = hashlib.sha256(_canonical(provenance)).hexdigest()
        if (
            manifest.get("status") != "complete"
            or expected_hash != manifest["provenance_hash"]
            or config["calibration_digest"] != calibration.data["snapshot_digest"]
            or provenance["camera"] != calibration.camera(camera)
        ):
            raise ValueError("Native teacher detections have mismatched calibration/provenance")
        video = session.video_paths[camera]
        stat = video.stat()
        if provenance["video"] != dict(path=str(video.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns):
            raise ValueError("Native teacher video identity changed")
        cache = torch.load(path, map_location="cpu")
        if cache["provenance_hash"] != expected_hash or cache["schema"] != manifest["schema"]:
            raise ValueError("Teacher detection cache is not bound to its provenance manifest")
        rows = provenance["frame_rows"]
        if len(rows) != len(cache["detections"]):
            raise ValueError("Teacher detections and native frame rows differ in length")
        evidence = dict(
            cache_path=str(path.resolve()),
            cache_sha256=file_sha(path),
            manifest_path=str(manifest_path.resolve()),
            manifest_sha256=file_sha(manifest_path),
            provenance_hash=expected_hash,
        )
        if decode_profile is not None:
            evidence["native_decode_profile"] = decode_profile
        if formal_context is not None:
            evidence["formal_input_equivalence"] = formal_context
        seen = set()
        for row, detections in zip(rows, cache["detections"]):
            if (
                type(row["record_index"]) is not int
                or row["record_index"] < 0
                or row["record_index"] >= len(session._camera_rows[camera])
                or row["record_index"] in seen
            ):
                raise ValueError("Teacher detector native support frames are invalid or duplicated")
            seen.add(row["record_index"])
            native = session._camera_rows[camera][row["record_index"]]
            if any(row[k] != native[k] for k in ("record_index", "common_time_us", "media_pts_us", "source_frame_no")):
                raise ValueError("Teacher detector native index/PTS/common time does not match frozen frame map")
            result.setdefault(row["record_index"], dict(detections=detections, evidence=evidence))
    return result


def _native_decode_profile(decode_threads):
    if decode_threads is None:
        return None
    if type(decode_threads) is not int or decode_threads != 1:
        raise ValueError("Only explicit single-thread decoding is supported")
    import av

    return dict(
        schema="badminton-observer-native-decode-v1",
        backend="pyav",
        thread_count=1,
        thread_type="SLICE",
        pixel_format="rgb24",
        native_pts_tolerance_us=2,
        pyav_version=av.__version__,
        library_versions={key: list(value) for key, value in av.library_versions.items()},
    )


def _pixel_matches(record, raw_bgr, checker):
    """Legacy/profile labels never substitute for the current actual pixels."""
    raw_sha = hashlib.sha256(raw_bgr.tobytes()).hexdigest()
    values = []
    for name in ("native_bgr_image_sha256", "raw_bgr_sha256"):
        if name in record:
            values.append(record[name] == raw_sha)
    if "undistorted_rgb_image_sha256" in record:
        import cv2

        rgb = cv2.remap(raw_bgr[:, :, ::-1], checker.map_x, checker.map_y, cv2.INTER_LINEAR)
        values.append(record["undistorted_rgb_image_sha256"] == hashlib.sha256(rgb.tobytes()).hexdigest())
    return bool(values) and all(values)


def _pose_cache_context(source, calibration):
    receipt_path = source.parent / "pose_receipt.json"
    if not receipt_path.is_file():
        return None
    receipt = _read(receipt_path)
    config = receipt.get("config", {})
    plan = _read(source / "repair_plan.json")
    from .full_native import pose_config_digest

    if (
        receipt.get("state") != "complete"
        or receipt.get("source_union_sha256") != plan["source_union_sha256"]
        or config.get("calibration_digest") != calibration.data["snapshot_digest"]
        or receipt.get("config_digest") != pose_config_digest(config)
    ):
        return None
    selected = receipt.get("formal_import_inventory")
    if not selected or file_sha(selected["path"]) != selected["sha256"]:
        return None
    inventory = _read(selected["path"])
    if (
        inventory.get("schema") != "badminton-formal-native-pose-import-v1"
        or inventory.get("state") != "complete"
        or inventory.get("config") != config
    ):
        return None
    targets = {row["target"]["path"]: row["target"] for row in inventory["records"]}
    if len(targets) != len(inventory["records"]):
        raise ValueError("Duplicate native pose cache target")
    return dict(
        config=config,
        config_digest=receipt["config_digest"],
        targets=targets,
        files=[dict(path=str(receipt_path), sha256=file_sha(receipt_path)), selected],
    )


def _pose_detections(context, source, sid, camera, session, native_index, bgr, checker, misses=None):
    def miss(reason):
        if misses is not None:
            misses[reason] = misses.get(reason, 0) + 1
        return None

    if context is None:
        return miss("source_cache_context_unavailable")
    config = context["config"]
    if config.get("detector_imgsz") != checker.detector_imgsz or any(
        config.get("models", {}).get(name) != item for name, item in checker.models.items()
    ):
        return miss("model_or_detector_options_differ")
    path = source.parent / "native_poses" / sid / camera / f"{native_index:07d}.json"
    selected = context["targets"].get(str(path.resolve()))
    if not selected or not path.is_file() or file_sha(path) != selected["sha256"]:
        return miss("native_file_missing_or_binding_differ")
    record = _read(path)
    if (
        record.get("schema") != "badminton-native-pose-v1"
        or record.get("config_digest") != context["config_digest"]
        or record.get("source_id") != sid
        or record.get("camera") != camera
        or record.get("native") != session.camera_rows(camera)[native_index]
        or record.get("coordinate_space") != "opencv_undistorted_K_preserved"
        or Path(record.get("source_video", "")).resolve() != session.video_paths[camera].resolve()
    ):
        return miss("native_source_camera_time_or_config_differ")
    if not _pixel_matches(record, bgr, checker):
        return miss("native_image_sha_missing_or_differ")
    boxes, confidence = record.get("boxes"), record.get("detection_confidence")
    if not isinstance(boxes, list) or not isinstance(confidence, list) or len(boxes) != len(confidence):
        return miss("invalid_native_detection_arrays")
    return dict(
        detections=[dict(bbx_xyxy=box, detection_confidence=score) for box, score in zip(boxes, confidence)],
        evidence=dict(
            cache=selected,
            config_digest=context["config_digest"],
            cache_type="native_pose_boxes_only",
            raw_bgr_sha256=record["raw_bgr_sha256"],
            method="actual_native_pixels_model_camera_source_time_equal",
        ),
    )


def _journal_matches(record, frame, bgr, checker, session, camera):
    evidence = record.get("reused_native_evidence", {})
    return (
        record.get("inference_scope") == "person_detection_and_association_only"
        and evidence.get("camera") == checker.camera
        and all(evidence.get("models", {}).get(name) == item for name, item in checker.models.items())
        and Path(record.get("raw_camera_video", "")).resolve() == session.video_paths[camera].resolve()
        and _cache_matches(record, frame)
        and _pixel_matches(record, bgr, checker)
    )


def _camera_frames(session, camera, indices, decode_threads):
    """Observer-only opt-in; the default public native reader is untouched."""
    if decode_threads is None:
        yield from session.iter_video_frames(camera, indices)
        return
    _native_decode_profile(decode_threads)
    if not indices:
        return
    import av
    from fractions import Fraction

    source_rows = session.camera_rows(camera)
    if any(type(i) is not int or i < 0 or i >= len(source_rows) for i in indices):
        raise ValueError("Native decoder requires existing integer source indices")
    if any(right <= left for left, right in zip(indices, indices[1:])):
        raise ValueError("Native decoder indices must be strictly increasing")
    rows = [source_rows[index] for index in indices]
    with av.open(str(session.video_paths[camera])) as container:
        if not container.streams.video:
            raise ValueError("Native observer source contains no video stream")
        stream = container.streams.video[0]
        stream.thread_count = 1
        stream.thread_type = "SLICE"
        if stream.thread_count != 1 or str(stream.thread_type) != "SLICE":
            raise ValueError("Native observer decoder did not apply the explicit single-thread profile")
        if stream.time_base is None or stream.time_base <= 0:
            raise ValueError("Native observer video lacks a valid time base")
        first = Fraction(rows[0]["media_pts_us"], 1_000_000) / stream.time_base
        container.seek(int(first), stream=stream, backward=True, any_frame=False)
        wanted = 0
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Native observer frame lacks PTS")
            actual = round(frame.pts * frame.time_base * 1_000_000)
            target = rows[wanted]["media_pts_us"]
            if actual < target - 2:
                continue
            if abs(actual - target) > 2:
                raise ValueError(f"Native observer PTS mismatch: {actual} vs {target}")
            yield indices[wanted], frame.to_ndarray(format="rgb24")[:, :, ::-1].copy()
            wanted += 1
            if wanted == len(indices):
                return
        raise ValueError("Native observer source ended before the requested frame")


def observe_all(
    source_root,
    output_root,
    device="cuda:0",
    pose_batch_size=4,
    detector_imgsz=960,
    threads=8,
    max_new_frames=None,
    cache_reports=(),
    decode_threads=None,
):
    source, output = _bound_output(source_root, output_root)
    dataset, sources = source_selection(source)
    calibration = CalibrationSnapshot.load(source / "calibration/calibration_snapshot.json")
    if dataset["calibration_digest"] != calibration.data["snapshot_digest"]:
        raise ValueError("Source labels and frozen calibration differ")
    relocations = load_pilot_run_relocations(source, calibration.data)
    formal_context = _formal_teacher_proof(source, dataset, calibration)
    settings = dict(
        device=device,
        pose_batch_size=pose_batch_size,
        detector_imgsz=detector_imgsz,
        threads=threads,
        flip_test=False,
        observation_scope="person_detection_and_association_only",
    )
    decode_profile = _native_decode_profile(decode_threads)
    if decode_profile is not None:
        settings["native_decode_profile"] = decode_profile
    pose_context = _pose_cache_context(source, calibration) if decode_profile is not None else None
    cache_sources = []
    for path in cache_reports:
        report = _read(path)
        item = dict(path=str(Path(path).resolve()), sha256=file_sha(path))
        if "frames_jsonl" in report:
            item.update(frames_jsonl=report["frames_jsonl"], frames_jsonl_sha256=file_sha(report["frames_jsonl"]))
        cache_sources.append(item)
    identity = dict(
        source=sources,
        settings=settings,
        checker_sha256=file_sha(PROJ_ROOT / "hmr4d/lidar_mocap/rgb_observations.py"),
        observer_sha256=file_sha(__file__),
        calibration_digest=dataset["calibration_digest"],
        cache_sources=cache_sources,
        formal_teacher_equivalence=formal_context,
    )
    if decode_profile is not None:
        identity["native_pose_cache_sources"] = pose_context["files"] if pose_context is not None else []
    identity_sha = hashlib.sha256(_canonical(identity)).hexdigest()
    observer_path = output / "observations/manifest.json"
    if observer_path.exists() and _read(observer_path)["identity_sha256"] != identity_sha:
        raise ValueError("Resume inputs/settings/checker changed; use a new output folder")
    started = time.perf_counter()
    cache = _cached_frames(cache_reports, dataset["calibration_digest"], decode_profile)
    checker = RGBKeypointChecker(
        calibration, threads, detector_imgsz, False, device=device, pose_batch_size=pose_batch_size, observe_pose=False
    )
    state = dict(
        schema="badminton-full-native-rgb-observations-v1",
        completed=False,
        created_utc=datetime.now(timezone.utc).isoformat(),
        identity=identity,
        identity_sha256=identity_sha,
        models=checker.models,
        reports=[],
        source_accepted=dataset["accepted"],
        model_loading_seconds=time.perf_counter() - started,
    )
    if decode_profile is not None:
        state["native_decode_profile"] = decode_profile
    observer_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(observer_path, state)
    new_frames = 0
    for sid in dataset["exports"]:
        checker.sessions[sid] = load_run_session(source, calibration, sid, relocations=relocations)
        frames = selected_frames(source, dataset, sid)
        frame_lookup = {f["lidar_index"]: f for f in frames}
        for camera in ("cam1", "cam2"):
            checker.select_camera(camera)
            report_path = output / "observations" / f"{sid}_{camera}.json"
            rows_path = report_path.with_suffix(".jsonl")
            done, completed_frames, completed_people, measured_seconds = _resume_rows(
                rows_path, frame_lookup, camera, decode_profile
            )
            session = checker.session(sid)
            ignored_caches = []
            teacher_cache = teacher_detections(
                source,
                dataset,
                sid,
                camera,
                calibration,
                session,
                formal_context=formal_context,
                decode_profile=decode_profile,
                ignored_caches=ignored_caches,
            )
            video = session.video_paths[camera]
            video_stat = video.stat()
            video_identity = dict(path=str(video), size=video_stat.st_size, mtime_ns=video_stat.st_mtime_ns)
            report = dict(
                schema="badminton-native-rgb-journal-v1",
                completed=False,
                calibration_digest=dataset["calibration_digest"],
                models=checker.models,
                checker_sha256=identity["checker_sha256"],
                identity_sha256=identity_sha,
                session_id=sid,
                camera_name=camera,
                camera=checker.camera,
                frames_jsonl=str(rows_path),
                source_video=video_identity,
                planned_camera_frames=len(frames),
                planned_accepted_person_frames=dataset["exports"][sid]["accepted"],
                completed_camera_frames=completed_frames,
                observed_accepted_person_frames=completed_people,
                inference_seconds=measured_seconds,
                device=device,
            )
            if decode_profile is not None:
                report["native_decode_profile"] = decode_profile
                report["ignored_teacher_caches"] = ignored_caches
                report["pixel_verified_cached_camera_frames"] = 0
                report["native_pose_cached_camera_frames"] = 0
                report["pixel_cache_mismatch_or_missing_frames"] = 0
                report["native_pose_cache_miss_reasons"] = {}
            report["exact_native_teacher_cache_frames_available"] = sum(
                f["cameras"][camera]["record_index"] in teacher_cache for f in frames
            )
            if report_path.exists() and _read(report_path)["source_video"] != video_identity:
                raise ValueError("Native source video changed since observing began")
            state["reports"].append(str(report_path))
            reused = 0
            with rows_path.open("a", encoding="utf-8") as handle:
                for frame in frames:
                    if decode_profile is not None:
                        continue  # Cross-profile reuse requires this run's actual decoded pixels.
                    key = (sid, camera, frame["lidar_index"])
                    if frame["lidar_index"] in done or key not in cache:
                        continue
                    record = cache[key]
                    if not _cache_matches(record, frame):
                        continue
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                    done.add(frame["lidar_index"])
                    reused += 1
                    report["completed_camera_frames"] += 1
                    report["observed_accepted_person_frames"] += sum(p["fit_accepted"] for p in record["athletes"])
            report["cached_camera_frames_imported_this_run"] = reused
            grouped = defaultdict(list)
            for frame in frames:
                if frame["lidar_index"] not in done:
                    grouped[frame["cameras"][camera]["record_index"]].append(frame)
            stop = max_new_frames is not None and new_frames >= max_new_frames
            if not stop:
                with rows_path.open("a", encoding="utf-8") as handle:
                    stream = _camera_frames(session, camera, sorted(grouped), decode_threads)
                    try:
                        for native_index, bgr in stream:
                            pose_cached = None
                            if decode_profile is not None:
                                pose_cached = _pose_detections(
                                    pose_context,
                                    source,
                                    sid,
                                    camera,
                                    session,
                                    native_index,
                                    bgr,
                                    checker,
                                    report["native_pose_cache_miss_reasons"],
                                )
                            for frame in grouped[native_index]:
                                clip = dict(id="full_source_session", session_id=sid, output_root=str(source))
                                cached = pose_cached if decode_profile is not None else teacher_cache.get(native_index)
                                previous = cache.get((sid, camera, frame["lidar_index"]))
                                if (
                                    decode_profile is not None
                                    and previous is not None
                                    and _journal_matches(previous, frame, bgr, checker, session, camera)
                                ):
                                    record = deepcopy(previous)
                                    record["pixel_verified_cache_reuse"] = record["reused_native_evidence"]
                                    record["timing_seconds"] = dict(detector=0.0, pose=0.0, total=0.0)
                                    report["pixel_verified_cached_camera_frames"] += 1
                                else:
                                    record = checker.check_frame(
                                        clip,
                                        frame,
                                        raw_bgr=bgr,
                                        cached_detections=cached["detections"] if cached else None,
                                    )
                                    if decode_profile is not None:
                                        if pose_cached is not None:
                                            report["native_pose_cached_camera_frames"] += 1
                                            record["person_detection_source"] = (
                                                "pixel_verified_native_pose_detection_cache"
                                            )
                                        else:
                                            report["pixel_cache_mismatch_or_missing_frames"] += 1
                                if decode_profile is not None:
                                    record["native_decode_profile"] = decode_profile
                                if cached and "pixel_verified_cache_reuse" not in record:
                                    record["teacher_detection_evidence"] = cached["evidence"]
                                add_raw_pixel_observations(dict(camera=checker.camera, frames=[record]))
                                handle.write(json.dumps(record, allow_nan=False) + "\n")
                                handle.flush()
                                new_frames += 1
                                report["completed_camera_frames"] += 1
                                report["observed_accepted_person_frames"] += sum(
                                    p["fit_accepted"] for p in record["athletes"]
                                )
                                report["inference_seconds"] += record["timing_seconds"]["total"]
                                if new_frames % 25 == 0 or new_frames <= 3:
                                    write_json(report_path, report)
                                    write_json(observer_path, state)
                                    print(
                                        json.dumps(
                                            dict(
                                                session=sid,
                                                camera=camera,
                                                completed_camera_frames=report["completed_camera_frames"],
                                                planned_camera_frames=len(frames),
                                                new_frames=new_frames,
                                                wall_seconds=time.perf_counter() - started,
                                                frame_seconds=record["timing_seconds"],
                                            )
                                        ),
                                        flush=True,
                                    )
                                if max_new_frames is not None and new_frames >= max_new_frames:
                                    stop = True
                                    break
                            if stop:
                                break
                    finally:
                        stream.close()
            report["completed"] = report["completed_camera_frames"] == len(frames)
            if report["completed"]:
                report["frames_jsonl_sha256"] = file_sha(rows_path)
            write_json(report_path, report)
            write_json(observer_path, state)
            if stop:
                state.update(last_run_seconds=time.perf_counter() - started, last_run_new_frames=new_frames)
                write_json(observer_path, state)
                return state
    state.update(completed=True, last_run_seconds=time.perf_counter() - started, last_run_new_frames=new_frames)
    write_json(observer_path, state)
    return state
