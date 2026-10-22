"""Prepare new-source windows for the existing native body-repair v2 components."""
from pathlib import Path
import shutil

from hmr4d import PROJ_ROOT
from .body_repair import published_windows
from .calibration import CalibrationSnapshot
from .dual_view import file_sha
from .pipeline import BadmintonPipeline, write_json
from .rally_pipeline import load_run_session, read_json, read_rows, select_window_pairs
from .relocations import load_pilot_run_relocations, validate_run_relocations


def write_frozen(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Prepared repair input changed; use a new root: {path}")
    write_json(path, value)


def prepare_new_sources(source_root, root, *, relocations=None):
    source_root, root = Path(source_root).resolve(), Path(root).resolve()
    if not root.is_relative_to(PROJ_ROOT / "outputs") or root == source_root:
        raise ValueError("Use a separate GVHMR outputs root")
    cal = CalibrationSnapshot.load(source_root / "calibration/calibration_snapshot.json")
    registry = read_json(source_root / "source_registry.json")
    if relocations is None:
        relocations = load_pilot_run_relocations(source_root, cal.data)
    validate_run_relocations(registry, cal.data, relocations)
    intake = read_json(registry["intake"]["path"])
    if file_sha(registry["intake"]["path"]) != registry["intake"]["sha256"]:
        raise ValueError("Source intake changed")
    if file_sha(intake["prepared_ranges"]["path"]) != intake["prepared_ranges"]["sha256"]:
        raise ValueError("Prepared rally ranges changed")
    ranges = read_rows(intake["prepared_ranges"]["path"])
    initial = root / "initial_teacher"
    initial.mkdir(parents=True, exist_ok=True)
    if not (initial / "calibration/calibration_snapshot.json").exists():
        shutil.copytree(source_root / "calibration", initial / "calibration")
    if CalibrationSnapshot.load(initial / "calibration/calibration_snapshot.json").data["snapshot_digest"] != cal.data["snapshot_digest"]:
        raise ValueError("Prepared initial teacher uses another calibration")
    write_frozen(initial / "source_registry.json", registry)
    pipeline = BadmintonPipeline(cal.data["sessions_root"], cal.data["lidar_project_root"], initial,
                                 preproc_precision="float16", roi_margin_px=20)
    jobs, selected = [], {}
    for sid, entry in registry["sources"].items():
        if entry["baseline"]:
            raise ValueError("New-source repair preparation does not reuse old body labels")
        session = load_run_session(source_root, cal, sid, relocations=relocations)
        eligible = {i for row in ranges if row["source_id"] == sid for i in row["eligible_paired_png_indices"]}
        selected[sid] = sorted(eligible)
        windows=select_window_pairs(session.pairs(), eligible)
        if registry.get('review_only'):
            from .pilot import review_rally_windows
            windows=review_rally_windows(session.pairs(),[row for row in ranges if row['source_id']==sid])
        for name, left, right, pairs in windows:
            window = initial / "sessions" / sid / "windows" / name
            config = dict(schema="badminton-window-v1", session_id=sid, window_id=name,
                          left_us=left, right_us=right, calibration_digest=cal.data["snapshot_digest"],
                          settings=pipeline.settings, source_pairs=[p["lidar_index"] for p in pairs])
            write_frozen(window / "window.json", config)
            jobs.append(dict(source_id=sid, source_window=str(window),
                             window_json_sha256=file_sha(window / "window.json")))
    plan = dict(schema="badminton-new-source-repair-input-v1", source_root=str(source_root),
                calibration_digest=cal.data["snapshot_digest"], jobs=jobs,
                source_registry_sha256=file_sha(source_root / "source_registry.json"),
                session_splits=read_json(source_root / "session_splits.json"),
                suggested_session_splits={name: dict(sessions=ids) for name, ids in read_json(source_root / "session_splits.json").items()},
                identity_status="pending_native_detection_evidence", training_published=False)
    if relocations is not None:
        plan["data_relocations"] = dict(relocations.binding)
    if registry.get('review_only'):
        from .pilot import require_review_group
        require_review_group(plan['session_splits'],registry['sources'])
        if intake.get('review_only') is not True or cal.data.get('review_only') is not True:
            raise ValueError('Review pilot requires matching intake and camera snapshot scope')
        purpose = ('full_native_rally_review' if registry.get('execution_scope') == 'all_effective_native_rallies'
                   else 'short_native_rally_review')
        plan.update(review_only=True,purpose=purpose,required_nonempty_splits=[])
    input_path = root / "new_source_input.json"
    write_frozen(input_path, plan)
    endpoints = root / "unresolved_native_endpoints.json"
    write_frozen(endpoints, [])
    cases = dict(schema="badminton-body-repair-cases-v2", source_union=str(input_path),
                 source_union_sha256=file_sha(input_path), clips=[], selected_scans=selected,
                 unresolved_native_endpoints=str(endpoints), track_candidates=[])
    write_frozen(root / "case_plan.json", cases)
    return plan


def initial_teacher_window(root, source, window_name, camera=None):
    """Run only one explicit teacher window; creates no fitted or published labels."""
    from .teacher import run_teacher
    from .segmentation import court_polygon
    root = Path(root).resolve()
    plan = read_json(root / "new_source_input.json")
    jobs = [(sid, w) for sid, w in published_windows(root / "new_source_input.json")
            if sid == source and w.name == window_name]
    if len(jobs) != 1:
        raise ValueError("Select exactly one prepared teacher window")
    sid, window = jobs[0]
    source_root = Path(plan["source_root"])
    if file_sha(source_root / "source_registry.json") != plan["source_registry_sha256"]:
        raise ValueError("New-source registry changed")
    cal = CalibrationSnapshot.load(source_root / "calibration/calibration_snapshot.json")
    if cal.data["snapshot_digest"] != plan["calibration_digest"]:
        raise ValueError("New-source calibration changed")
    relocations = load_pilot_run_relocations(source_root, cal.data)
    if plan.get("data_relocations") != (dict(relocations.binding) if relocations is not None else None):
        raise ValueError("Teacher source plan changed its relocation binding")
    session = load_run_session(source_root, cal, sid, relocations=relocations)
    config = read_json(window / "window.json")
    settings = config["settings"]
    camera = camera or settings["camera"]
    if camera not in ('cam1','cam2'):
        raise ValueError('Select one of the two frozen native cameras')
    target = (window/'teacher' if camera==settings['camera'] else
              root/'secondary_teachers'/camera/'sessions'/sid/'windows'/window.name/'teacher')
    rows = session.teacher_rows(camera, settings["teacher_stride"], config["left_us"]-1000000,
                                config["right_us"]+1000000)
    return run_teacher(session.video_paths[camera], rows, cal.camera(camera), target,
                       batch_size=settings["batch_size"], resume=True,
                       roi_polygon=court_polygon(cal, camera), roi_margin_px=settings.get("roi_margin_px", 40),
                       preproc_precision=settings.get("preproc_precision", "float32"))


def native_teacher_mapping(window):
    """Reproduce original observed tracklet identity from frozen native detections."""
    import numpy as np
    import torch
    from .teacher import associate_detections
    manifest = read_json(window / "teacher/manifest.json")
    cache = torch.load(window / "teacher/detections.pt", map_location="cpu", weights_only=False)
    if cache["provenance_hash"] != manifest["provenance_hash"]:
        raise ValueError("Native teacher/detection provenance mismatch")
    frames = [[dict(det, replay_detection_index=i) for i, det in enumerate(frame)] for frame in cache["detections"]]
    options = manifest["provenance"]["options"]
    mode = "corners_v1" if manifest["provenance"]["algorithm_version"] == 1 else "observed_v2"
    tracks = associate_detections(frames, prediction_mode=mode, max_track_gap_frames=options["max_track_gap_frames"])
    return np.asarray([[o["position"], o["replay_detection_index"], o.get("original_track_id", key), key]
                       for key, observations in tracks.items() for o in observations], dtype=np.int64).reshape(-1, 4)
