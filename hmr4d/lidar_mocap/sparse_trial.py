"""Isolated paired-native RGB frontend trial; never writes production teachers."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from hmr4d import PROJ_ROOT
from .calibration import CalibrationSnapshot
from .pilot import binding, check_binding
from .rally_pipeline import load_run_session, read_json
from .relocations import load_pilot_run_relocations
from .repair_execution import remaining_lease_seconds, require_gpu_lease, write_progress_json
from .segmentation import court_polygon
from .teacher import _jsonable, _normalize_rows, _provenance, run_teacher, load_teacher_track, validate_provenance


def prepare_trial(root, source_id, original_rally_index=1, *, production_settings=True):
    root = Path(root).resolve()
    frozen = read_json(root / "pilot_input.json")
    check_binding(frozen["manifest"])
    source = root / "source"
    cal = CalibrationSnapshot.load(source / "calibration/calibration_snapshot.json")
    relocation = load_pilot_run_relocations(source, cal.data)
    session = load_run_session(source, cal, source_id, relocations=relocation)
    ranges = [json.loads(row) for row in (source / "pilot_ranges.jsonl").read_text(encoding="utf-8").splitlines()]
    chosen = [row for row in ranges if row["source_id"] == source_id and row["original_rally_index"] == original_rally_index]
    if len(chosen) != 1:
        raise ValueError("Select one exact native rally segment for this small trial")
    chosen = chosen[0]
    inputs = read_json(root / "repair/new_source_input.json")
    jobs = []
    for job in inputs["jobs"]:
        if job["source_id"] == source_id:
            window = read_json(Path(job["source_window"]) / "window.json")
            if window["left_us"] == chosen["begin_common_us"] and window["right_us"] == chosen["end_common_us"]:
                jobs.append((job, window))
    if len(jobs) != 1:
        raise ValueError("The original frozen teacher window must match the actual native rally")
    job, window = jobs[0]
    pairs = session.pairs()
    first_scan_mid = pairs[chosen["eligible_paired_png_indices"][0]]["scan_mid_us"]
    context = [pair for pair in pairs if window["left_us"] - 1_000_000 <= pair["scan_mid_us"] <= window["right_us"] + 1_000_000
               and all(pair["cameras"][name]["status"] == "matched" for name in ("cam1", "cam2"))]
    cameras = {}
    for name in ("cam1", "cam2"):
        rows = {row["record_index"]: row for row in session.camera_rows(name)}
        observed = [{key: rows[pair["cameras"][name]["record_index"]].get(key) for key in
                    ("record_index", "common_time_us", "media_pts_us", "source_frame_no")} for pair in context]
        if any(b["record_index"] <= a["record_index"] for a, b in zip(observed, observed[1:])):
            raise ValueError("A sparse frontend must retain unique actual native records")
        baseline = _normalize_rows(session.teacher_rows(name, window["settings"]["teacher_stride"],
                                  window["left_us"] - 1_000_000, window["right_us"] + 1_000_000))
        cameras[name] = dict(camera=_jsonable(cal.camera(name)), video=str(session.video_paths[name]),
            baseline_rows=baseline, observed_rows=observed, roi_polygon=court_polygon(cal, name),
            scan_bindings=[dict(lidar_index=p["lidar_index"], scan_mid_us=p["scan_mid_us"],
                               camera=p["cameras"][name]) for p in context])
    teacher_options = dict(roi_margin_px=window["settings"]["roi_margin_px"], detector_imgsz=1280,
                           min_track_frames=8, max_track_gap_frames=6, inference_window_frames=600)
    directory = root / "benchmarks/paired_rgb10" / (source_id + "_rally" + str(original_rally_index)
                                                   + ("_production" if production_settings else ""))
    directory.mkdir(parents=True, exist_ok=True)
    recipe = dict(schema="gvhmr-paired-rgb-frontend-trial-v1", root=str(root), source_id=source_id,
        original_rally_index=original_rally_index, manifest=frozen["manifest"],
        source_input=binding(root / "repair/new_source_input.json"), source_window=binding(Path(job["source_window"]) / "window.json"),
        calibration=binding(source / "calibration/calibration_snapshot.json"), calibration_digest=cal.data["snapshot_digest"],
        origin_scan_mid_us=first_scan_mid, original_range_begin_us=window["left_us"],
        selected_scan_indices=chosen["eligible_paired_png_indices"], cameras=cameras,
        batch_size=window["settings"]["batch_size"], preproc_precision=window["settings"].get("preproc_precision", "float32"),
        output_directory=str(directory), virtual_nodes_are_rgb_observations=False, training_published=False)
    if production_settings:
        recipe["schema"] = "gvhmr-paired-rgb-frontend-trial-v2"
        recipe["teacher_options"] = teacher_options
        snapshot = root / "teacher_batch_02_release_receipt.progress.json"
        if snapshot.exists():
            prior = read_json(snapshot)
            key = source_id + ":" + Path(job["source_window"]).name
            for name, item in cameras.items():
                prior_camera = prior.get("windows", {}).get(key, {}).get(name)
                if not prior_camera or prior_camera.get("seconds", 0) <= 0:
                    continue
                check_binding(prior_camera["manifest"])
                original = read_json(prior_camera["manifest"]["path"])
                expected_options = dict(batch_size=recipe["batch_size"], roi_polygon=item["roi_polygon"],
                                        preproc_precision=recipe["preproc_precision"], **teacher_options)
                expected, digest = _provenance(item["video"], item["baseline_rows"], item["camera"], expected_options)
                if original.get("provenance_hash") == digest and original.get("provenance") == expected:
                    item["existing_baseline"] = dict(manifest=prior_camera["manifest"],
                        artifacts_sha256=prior_camera["artifacts_sha256"], seconds=prior_camera["seconds"],
                        timing_source=binding(snapshot), source_key=key, source_camera=name,
                        expected_provenance_hash=digest)
    path = directory / "input.json"
    if path.exists() and read_json(path) != recipe:
        raise ValueError("This trial input is already frozen; use an independent namespace")
    if not path.exists(): write_progress_json(path, recipe)
    return binding(path)


def run_trial(root, recipe_path, phase, lease_path):
    import psutil

    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity([0, 1])
    root, recipe_path = Path(root).resolve(), Path(recipe_path).resolve()
    recipe = read_json(recipe_path)
    directory = Path(recipe["output_directory"]).resolve()
    if (recipe["schema"] not in ("gvhmr-paired-rgb-frontend-trial-v1", "gvhmr-paired-rgb-frontend-trial-v2")
            or Path(recipe["root"]).resolve() != root
            or not directory.is_relative_to(root / "benchmarks") or not recipe_path.is_relative_to(directory)):
        raise ValueError("A trial must remain inside its explicitly authorized independent namespace")
    for name in ("manifest", "source_input", "source_window", "calibration"):
        check_binding(recipe[name])
    require_gpu_lease(lease_path)
    lease = read_json(lease_path)
    if remaining_lease_seconds(lease, recipe["manifest"]["sha256"]) < 180:
        raise ValueError("An experiment needs at least 180 seconds before its local GPU deadline")
    if (lease.get("experiment_input") != binding(recipe_path) or Path(lease["root"]).resolve() != root
            or lease.get("stage") != "teacher"):
        raise ValueError("The experiment requires its real frozen sampling-recipe SHA")
    state = dict(state="running", phase=phase, recipe=binding(recipe_path), cameras={},
                 started_utc=datetime.now(timezone.utc).isoformat(), training_published=False)
    progress = directory / (phase + "_progress.json")
    write_progress_json(progress, state)
    for name, item in recipe["cameras"].items():
        require_gpu_lease(lease_path)
        if remaining_lease_seconds(read_json(lease_path), recipe["manifest"]["sha256"]) < 180:
            raise ValueError("Insufficient local GPU lease time for the next camera")
        state["active_camera"] = name
        write_progress_json(progress, state)
        started = time.perf_counter()
        target = directory / phase / name
        if phase == "baseline30":
            result = run_teacher(item["video"], item["baseline_rows"], item["camera"], target,
                batch_size=recipe["batch_size"], preproc_precision=recipe["preproc_precision"],
                roi_polygon=item["roi_polygon"], resume=True, **recipe.get("teacher_options", {}))
        elif phase == "paired10":
            from .sparse_teacher import run_paired_teacher
            result = run_paired_teacher(item, target, recipe=binding(recipe_path), batch_size=recipe["batch_size"],
                                        precision=recipe["preproc_precision"], origin_us=recipe["origin_scan_mid_us"],
                                        frontend_options=recipe.get("teacher_options"))
        else:
            raise ValueError("Choose the explicit baseline30 or paired10 trial")
        state["cameras"][name] = dict(seconds=time.perf_counter() - started, tracks=len(result["tracks"]),
                                       real_rgb_frames=result["frame_count"], manifest=binding(target / "manifest.json"))
        write_progress_json(progress, state)
    state.update(state="complete", completed_utc=datetime.now(timezone.utc).isoformat(), safe_to_release_gpu=True)
    state.pop("active_camera", None)
    write_progress_json(progress, state)
    return state


def import_existing_baseline(recipe_path):
    """Read exact original outputs and measured cold-worker times on CPU only."""
    from .dual_view import file_sha

    recipe_path = Path(recipe_path).resolve()
    recipe = read_json(recipe_path)
    directory = Path(recipe["output_directory"])
    if not directory.resolve().is_relative_to(Path(recipe["root"]).resolve() / "benchmarks"):
        raise ValueError("An experimental baseline must stay in its independent namespace")
    for name in ("manifest", "source_input", "source_window", "calibration"):
        check_binding(recipe[name])
    result = dict(state="complete", phase="baseline30", recipe=binding(recipe_path), cameras={},
                  timing_method="historical_actual_isolated_worker_wall_seconds_not_new_cache_load_time",
                  safe_to_release_gpu=True, no_gpu_used_for_baseline_import=True, training_published=False)
    for name, camera in recipe["cameras"].items():
        original = camera["existing_baseline"]
        for key in ("manifest", "timing_source"):
            check_binding(original[key])
        entry = read_json(original["timing_source"]["path"])["windows"][original["source_key"]][name]
        if entry["manifest"] != original["manifest"] or entry["seconds"] != original["seconds"]:
            raise ValueError("Historical baseline measurement changed")
        manifest_path = Path(original["manifest"]["path"])
        for relative, digest in original["artifacts_sha256"].items():
            if file_sha(manifest_path.parent / relative) != digest:
                raise ValueError("A reused baseline artifact changed")
        manifest = read_json(manifest_path)
        validate_provenance(manifest, original["expected_provenance_hash"])
        for track in manifest["tracks"]:
            validate_provenance(load_teacher_track(manifest_path.parent / track["path"]), manifest["provenance_hash"])
        result["cameras"][name] = dict(seconds=original["seconds"], tracks=len(manifest["tracks"]),
            real_rgb_frames=manifest["frame_count"], manifest=original["manifest"], timing_source=original["timing_source"],
            reused_exact_baseline=True)
    path = directory / "baseline30_progress.json"
    if path.exists() and read_json(path) != result:
        raise ValueError("Refusing to replace an existing baseline measurement")
    write_progress_json(path, result)
    return result
