"""Experimental real paired-RGB frontend with explicit missing temporal nodes.

This module is deliberately separate from the production 30 Hz teacher. Virtual
nodes contain interpolated conditioning features, never RGB/keypoint evidence,
and only real native camera anchors are exported as teacher observations.
"""

import gc
from pathlib import Path
import time

import numpy as np

from . import teacher as native


def normalize_sparse_rows(frame_rows):
    rows = []
    for row in frame_rows:
        item = {key: int(row[key]) for key in ("record_index", "common_time_us", "media_pts_us")}
        item["source_frame_no"] = None if row.get("source_frame_no") is None else int(row["source_frame_no"])
        rows.append(item)
    if len(rows) < 2 or any(row["record_index"] < 0 for row in rows):
        raise ValueError("At least two actual paired native RGB records are required")
    for key in ("record_index", "common_time_us", "media_pts_us"):
        if np.any(np.diff(np.asarray([row[key] for row in rows], dtype=np.int64)) <= 0):
            raise ValueError("Sparse native records, common times and PTS must be strictly increasing")
    delta = np.diff(np.asarray([row["common_time_us"] for row in rows], dtype=np.int64))
    if not 75_000 <= np.median(delta) <= 125_000 or np.any(delta > 180_000):
        raise ValueError("Split discontinuities before the approximately 10 Hz paired-RGB trial")
    return rows


def latent_grid(common_times_us, origin_us):
    """Retain exact native anchors and subdivide each real interval near 30 Hz.

    This is an explicitly nonuniform latent grid: five/six-frame native steps
    can yield 27.8/33.4 ms nodes. It is not an artificial 30 fps RGB sequence.
    Integer common-clock subtraction precedes conversion to relative seconds.
    """
    times = np.asarray(common_times_us, dtype=np.int64)
    if len(times) < 2 or np.any(np.diff(times) <= 0) or np.any(np.diff(times) > 180_000):
        raise ValueError("Latent nodes require one continuous increasing observed segment")
    relative = times - int(origin_us)
    ticks, anchors = [int(relative[0])], [0]
    for before, after in zip(relative, relative[1:]):
        count = max(1, int(round(int(after - before) * 30 / 1_000_000)))
        ticks.extend(int(before) + int(round(int(after - before) * index / count)) for index in range(1, count + 1))
        anchors.append(len(ticks) - 1)
    ticks = np.asarray(ticks, dtype=np.int64)
    anchors = np.asarray(anchors, dtype=np.int64)
    right = np.searchsorted(relative, ticks, side="left").clip(0, len(relative) - 1)
    left = (right - 1).clip(0, len(relative) - 1)
    left[relative[right] == ticks] = right[relative[right] == ticks]
    width = relative[right] - relative[left]
    alpha = np.divide(ticks - relative[left], width, out=np.zeros(len(ticks)), where=width > 0)
    observed = np.zeros(len(ticks), dtype=bool)
    observed[anchors] = True
    return dict(relative_time_us=ticks, relative_seconds=ticks.astype(np.float64) / 1_000_000,
                anchor_indices=anchors, observed_mask=observed, left=left, right=right, alpha=alpha,
                origin_common_time_us=int(origin_us), policy="piecewise_near_30hz_exact_native_anchors")


def expand_conditioning(values, grid, *, keypoints=False):
    import torch

    values = torch.as_tensor(values, dtype=torch.float32)
    shape = (len(grid["alpha"]),) + (1,) * (values.ndim - 1)
    fraction = torch.as_tensor(grid["alpha"], dtype=torch.float32).reshape(shape)
    expanded = (1 - fraction) * values[grid["left"]] + fraction * values[grid["right"]]
    expanded[grid["anchor_indices"]] = values
    if keypoints:
        if values.shape[1:] != (17, 3):
            raise ValueError("Native COCO17 keypoints with confidence are required")
        expanded[~grid["observed_mask"], :, 2] = 0
    return expanded


def duration_tracking_limits(rows, *, max_native_gap_frames=6, min_native_frames=8):
    step = float(np.median(np.diff([row["common_time_us"] for row in rows])))
    # Existing 30 Hz max_gap=6 permits at most seven positions, about 233 ms.
    return dict(max_gap_frames=max(0, int(((max_native_gap_frames + 1) * 1_000_000 / 30 + 1) // step) - 1),
                minimum_span_us=int(round((min_native_frames - 1) * 1_000_000 / 30)))


def observed_segments(tracks, rows, limits):
    candidates, rejected = native._split_tracks(tracks, rows, 2)
    segments = []
    for segment in candidates:
        obs = segment["observations"]
        span = rows[obs[-1]["position"]]["common_time_us"] - rows[obs[0]["position"]]["common_time_us"]
        if span >= limits["minimum_span_us"]:
            segments.append(segment)
        else:
            rejected.append(dict(track_id=segment["track_id"], segment_id=segment["segment_id"],
                                 frame_count=len(obs), observed_span_us=span, reason="insufficient_real_duration"))
    return segments, rejected


def sampling_tracking_limits(rows, options):
    if options.get("sampling_policy") == "paired_native_rgb10_latent_v1":
        return duration_tracking_limits(rows, max_native_gap_frames=options["max_track_gap_frames"],
                                        min_native_frames=options["min_track_frames"])
    return None


def sample_latent_track(track, target_times_us, max_time_error_us=18_000):
    """Query model states while retaining nearest *real* RGB evidence metadata."""
    state = track["inferred_temporal_state"]
    if state.get("role") != "inferred_model_state_not_rgb_observation":
        raise ValueError("A sparse teacher must explicitly distinguish inferred states from RGB observations")
    grid = track["temporal_latent_grid"]
    relative = np.asarray(grid["relative_time_us"], dtype=np.int64)
    anchors = np.asarray(grid["anchor_indices"], dtype=np.int64)
    observed = native._numpy(track["common_time_us"]).astype(np.int64)
    mask = np.asarray(grid["observed_mask"], dtype=bool)
    if (len(relative) < 2 or np.any(np.diff(relative) <= 0) or np.any(np.diff(relative) > 37_000)
            or not np.array_equal(relative[anchors], observed - int(grid["origin_common_time_us"]))
            or not np.array_equal(np.flatnonzero(mask), anchors)):
        raise ValueError("Sparse model state clock/observed-anchor binding changed")
    original = dict(track)
    original.pop("inferred_temporal_state")
    sample = native.sample_track(original, target_times_us, max_time_error_us)
    target = np.asarray(target_times_us, dtype=np.int64).reshape(-1) - int(grid["origin_common_time_us"])
    right = np.searchsorted(relative, target, side="left").clip(0, len(relative) - 1)
    left = (right - 1).clip(0, len(relative) - 1)
    left[target <= relative[0]] = right[target <= relative[0]] = 0
    left[target >= relative[-1]] = right[target >= relative[-1]] = len(relative) - 1
    exact = relative[right] == target
    left[exact] = right[exact]
    gaps = relative[right] - relative[left]
    alpha = np.divide(target - relative[left], gaps, out=np.zeros(len(target)), where=gaps > 0).clip(0, 1)
    params = {}
    for key, anchored in track["smpl_params_incam"].items():
        values = native._numpy(state["smpl_params_incam"][key]).astype(np.float32)
        if (values.shape != (len(relative), *native._numpy(anchored).shape[1:]) or not np.isfinite(values).all()
                or not np.array_equal(values[anchors], native._numpy(anchored))):
            raise ValueError("Sparse inferred parameters do not bind their actual native-anchor predictions")
        if key in ("body_pose", "global_orient"):
            joints = values.shape[-1] // 3
            params[key] = native._slerp_rotvec(values[left].reshape(len(target), joints, 3),
                values[right].reshape(len(target), joints, 3), alpha).reshape(len(target), values.shape[-1])
        else:
            fraction = alpha.reshape((len(target),) + (1,) * (values.ndim - 1))
            params[key] = ((1 - fraction) * values[left] + fraction * values[right]).astype(np.float32)
    nearest = np.searchsorted(observed, sample["metadata"]["teacher_common_time_us"])
    ambiguous = native._numpy(track.get("identity_ambiguous", np.zeros(len(observed), bool))).astype(bool)
    observed_gap = sample["metadata"]["teacher_right_time_us"] - sample["metadata"]["teacher_left_time_us"]
    observed_left = np.searchsorted(observed, sample["metadata"]["teacher_left_time_us"])
    observed_right = np.searchsorted(observed, sample["metadata"]["teacher_right_time_us"])
    sample["valid"] = ((np.abs(sample["metadata"]["model_time_error_us"]) <= min(max_time_error_us, 8_500))
                       & (gaps <= 37_000) & (observed_gap <= 180_000)
                       & ~ambiguous[observed_left] & ~ambiguous[observed_right])
    sample["smpl_params_incam"] = params
    sample["metadata"].update(parameter_sampling="SO3_inferred_temporal_states_real_RGB_evidence_within_8500us",
        latent_state_role=state["role"], latent_left_relative_time_us=relative[left],
        latent_right_relative_time_us=relative[right], latent_interpolation_alpha=alpha.astype(np.float32))
    return sample


def run_paired_teacher(item, output_dir, *, recipe, batch_size, precision, origin_us, frontend_options=None,
                       native_options=None, reuse_preprocess_dir=None, identity_decisions=None, resume=True):
    import torch

    rows = normalize_sparse_rows(item["observed_rows"])
    camera, video_path = item["camera"], Path(item["video"])
    frontend_options = frontend_options or {}
    roi_margin = frontend_options.get("roi_margin_px", 40.0)
    detector_imgsz = frontend_options.get("detector_imgsz", 1280)
    limits = duration_tracking_limits(rows, max_native_gap_frames=frontend_options.get("max_track_gap_frames", 6),
                                     min_native_frames=frontend_options.get("min_track_frames", 8))
    options = dict(sparse_trial_recipe=recipe, temporal_policy="piecewise_near_30hz_exact_native_anchors",
                   virtual_keypoint_confidence=0, virtual_nodes_are_rgb_observations=False,
                   duration_tracking=limits, batch_size=batch_size, preproc_precision=precision,
                   roi_polygon=item["roi_polygon"], roi_margin_px=roi_margin, detector_imgsz=detector_imgsz)
    if native_options is not None:
        options = native_options
    provenance, digest = native._provenance(video_path, rows, camera, options)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if resume and manifest_path.exists():
        import json
        result = json.loads(manifest_path.read_text(encoding="utf-8"))
        native.validate_provenance(result, digest)
        for entry in result["tracks"]:
            native.validate_provenance(native.load_teacher_track(output_dir / entry["path"]), digest)
        return result
    timing = {}
    reused = native._preprocess_reuse(reuse_preprocess_dir, provenance) if reuse_preprocess_dir is not None else None
    started = time.perf_counter()
    path = output_dir / "detections.pt"
    detection = native._read_stage(path, digest, resume)
    if detection is None:
        detection = dict(schema=native.SCHEMA, provenance_hash=digest,
                         detections=(reused["detections"] if reused else
                                     native._detect(video_path, rows, camera, item["roi_polygon"], roi_margin, detector_imgsz)))
        native._save_tensor(path, detection)
    timing["detector_decode_load_seconds"] = time.perf_counter() - started
    if len(detection["detections"]) != len(rows):
        raise ValueError("Sparse detection cache has different native rows")
    tracks = native.associate_detections(detection["detections"], max_track_gap_frames=limits["max_gap_frames"])
    identity_receipt = None
    if identity_decisions is not None:
        from .identity_repair import apply_identity_decisions
        tracks, identity_receipt = apply_identity_decisions(tracks, identity_decisions)
    segments, rejected = observed_segments(tracks, rows, limits)
    manifest = dict(schema=native.SCHEMA, provenance_hash=digest, provenance=provenance,
                    frame_count=len(rows), coordinate_space=native.COORDINATE_SPACE, tracks=[],
                    rejected_segments=rejected, status="complete", static_camera=True,
                    tracking_identity_verified=False, preproc_precision=precision,
                    source_fps_median=1_000_000 / float(np.median(np.diff([r["common_time_us"] for r in rows]))),
                    virtual_nodes_exported_as_observations=False, timing=timing, training_published=False)
    if identity_receipt is not None:
        manifest["identity_repair"] = identity_receipt
    if reused:
        manifest["preprocessing_reuse"] = reused["evidence"]
    if not segments:
        native._save_json(manifest_path, manifest)
        return manifest
    preprocessed = {}
    for stage in ("vitpose", "features"):
        started = time.perf_counter()
        path = output_dir / (stage + ".pt")
        cache = native._read_stage(path, digest, resume)
        if cache is None:
            if reused:
                trailing = (17, 3) if stage == "vitpose" else (1024,)
                values = [torch.full((len(segment["observations"]), *trailing), float("nan")) for segment in segments]
                missing = []
                for i, segment in enumerate(segments):
                    absent = []
                    for j, obs in enumerate(segment["observations"]):
                        value = reused["values"][stage].get(native._observation_signature(obs))
                        if value is None:
                            absent.append((j, obs))
                        else:
                            values[i][j] = value
                    if absent:
                        missing.append((i, absent))
                if missing:
                    extracted = native._extract_stage(video_path, rows, camera,
                        [dict(observations=[obs for _, obs in absent]) for _, absent in missing], batch_size, stage, precision)
                    for (i, absent), tensor in zip(missing, extracted):
                        for (j, _), value in zip(absent, tensor):
                            values[i][j] = value
            else:
                values = native._extract_stage(video_path, rows, camera, segments, batch_size, stage, precision)
            if not all(torch.isfinite(value).all() for value in values):
                raise ValueError("Incomplete native sparse crop preprocessing")
            cache = dict(schema=native.SCHEMA, provenance_hash=digest, values=values)
            native._save_tensor(path, cache)
        if len(cache["values"]) != len(segments):
            raise ValueError("Sparse frontend segment count differs from its saved cache")
        preprocessed[stage] = cache["values"]
        timing[stage + "_decode_load_seconds"] = time.perf_counter() - started
    import hydra
    from hydra import compose, initialize_config_module
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # noqa: F401

    started = time.perf_counter()
    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(config_name="demo", overrides=[f"video_name={video_path.stem}", "static_cam=True"])
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()
    K = torch.as_tensor(camera["K"], dtype=torch.float32).reshape(3, 3)
    try:
        for number, segment in enumerate(segments):
            observations = segment["observations"]
            selected = [rows[obs["position"]] for obs in observations]
            length = len(selected)
            grid = latent_grid([row["common_time_us"] for row in selected], origin_us)
            boxes = torch.as_tensor(np.stack([obs["bbx_xyxy"] for obs in observations]), dtype=torch.float32)
            bbx_xys = torch.as_tensor(np.stack([native._bbox_xys(obs["bbx_xyxy"]) for obs in observations]))
            kp2d = preprocessed["vitpose"][number]
            features = preprocessed["features"][number]
            expanded = dict(bbx_xys=expand_conditioning(bbx_xys, grid),
                            kp2d=expand_conditioning(kp2d, grid, keypoints=True),
                            f_imgseq=expand_conditioning(features, grid))
            parts = {key: [] for key in ("body_pose", "betas", "global_orient", "transl")}
            with torch.no_grad():
                for start, end, context_start, context_end in native._window_ranges(
                        len(grid["relative_time_us"]), frontend_options.get("inference_window_frames", 600)):
                    n = context_end - context_start
                    data = {key: value[context_start:context_end] for key, value in expanded.items()}
                    data.update(length=torch.tensor(n), K_fullimg=K.repeat(n, 1, 1),
                                cam_angvel=torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(n, 1))
                    prediction = model.predict(data, static_cam=True)
                    kept = slice(start - context_start, end - context_start)
                    for key in parts:
                        parts[key].append(prediction["smpl_params_incam"][key][kept].detach().cpu())
                    del prediction
            states = {key: torch.cat(value) for key, value in parts.items()}
            params = {key: value[grid["anchor_indices"]] for key, value in states.items()}
            if not all(torch.isfinite(value).all() for value in params.values()):
                raise ValueError("The sparse latent trial produced non-finite SMPL parameters")
            track_id, segment_id = segment["track_id"], segment["segment_id"]
            name = f"track_{track_id:04d}_segment_{segment_id:04d}.pt"
            result = dict(schema=native.SCHEMA, provenance_hash=digest, provenance=provenance,
                          track_id=track_id, segment_id=segment_id, coordinate_space=native.COORDINATE_SPACE,
                          smpl_model_type="smplx_neutral_body21_betas10", smpl_params_incam=params,
                          kp2d=kp2d, bbx_xyxy=boxes, bbx_xys=bbx_xys, K_fullimg=K.repeat(length, 1, 1),
                          tracking_identity_verified=False, static_camera=True, native_video_frame_rows=selected,
                          inferred_temporal_state=dict(role="inferred_model_state_not_rgb_observation", smpl_params_incam=states),
                          temporal_latent_grid=grid, distortion_original=camera.get("distortion_opencv", [0.] * 5),
                          teacher_image_distortion=[0.] * 5,
                          detection_confidence=torch.tensor([obs["detection_confidence"] for obs in observations]),
                          association_cost=torch.tensor([obs["association_cost"] for obs in observations]),
                          identity_ambiguous=torch.tensor([obs["identity_ambiguous"] for obs in observations]),
                          original_track_id=torch.tensor([obs.get("original_track_id", track_id) for obs in observations]),
                          association_recovered=torch.tensor([obs.get("association_recovered", False) for obs in observations]))
            for key in ("record_index", "common_time_us", "media_pts_us", "source_frame_no"):
                result[key] = torch.tensor([row[key] if row[key] is not None else -1 for row in selected], dtype=torch.int64)
            native._save_tensor(output_dir / name, result)
            manifest["tracks"].append(native._track_manifest_entry(segment, name, result))
            print(f"Paired teacher: track {track_id} segment {segment_id}: {length} real RGB anchors, "
                  f"{len(grid['relative_time_us'])} explicit latent nodes", flush=True)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    timing["gvhmr_load_inference_save_seconds"] = time.perf_counter() - started
    native._save_json(manifest_path, manifest)
    return manifest


def run_sparse_teacher(video_path, frame_rows, camera, output_dir, *, batch_size, resume,
                       min_track_frames, max_track_gap_frames, inference_window_frames, roi_polygon,
                       roi_margin_px, detector_imgsz, preproc_precision, reuse_preprocess_dir,
                       identity_decisions, sampling_policy, sampling_recipe, sampling_origin_common_time_us):
    from .pilot import check_binding

    if sampling_policy != "paired_native_rgb10_latent_v1" or sampling_recipe is None:
        raise ValueError("Sparse inference requires an explicit bound sampling policy/input")
    check_binding(sampling_recipe)
    if batch_size < 1 or min_track_frames < 2 or inference_window_frames < 8:
        raise ValueError("Invalid sparse teacher batch/context configuration")
    native._validate_preproc_precision(preproc_precision)
    native._undistorter(camera)
    options = dict(batch_size=batch_size, min_track_frames=min_track_frames,
        max_track_gap_frames=max_track_gap_frames, inference_window_frames=inference_window_frames,
        roi_polygon=roi_polygon, roi_margin_px=roi_margin_px, detector_imgsz=detector_imgsz,
        preproc_precision=preproc_precision, sampling_policy=sampling_policy,
        sampling_recipe=sampling_recipe, sampling_origin_common_time_us=int(sampling_origin_common_time_us))
    if identity_decisions is not None:
        from .identity_repair import decision_digest
        options["identity_decisions_sha256"] = decision_digest(identity_decisions)
    item = dict(video=str(video_path), observed_rows=frame_rows, camera=camera, roi_polygon=roi_polygon)
    return run_paired_teacher(item, output_dir, recipe=sampling_recipe, batch_size=batch_size,
        precision=preproc_precision, origin_us=int(sampling_origin_common_time_us), frontend_options=options,
        native_options=options, reuse_preprocess_dir=reuse_preprocess_dir, identity_decisions=identity_decisions, resume=resume)


def augment_with_latents(anchor_directory, output_dir):
    """Reinfer the cheap temporal model from exact cached real crops/features."""
    import json
    import torch
    import hydra
    from hydra import compose, initialize_config_module
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # noqa: F401
    from .pilot import binding

    source, output_dir = Path(anchor_directory), Path(output_dir)
    source_manifest = json.loads((source / "manifest.json").read_bytes())
    original = source_manifest["provenance"]
    features = native._read_stage(source / "features.pt", source_manifest["provenance_hash"], True)["values"]
    options = dict(original["options"], inferred_temporal_state_protocol="role_bound_near30hz_real_RGB_gate8500us",
                   source_anchor_manifest=binding(source / "manifest.json"))
    provenance, digest = native._provenance(original["video"]["path"], original["frame_rows"], original["camera"], options)
    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(config_name="demo", overrides=["video_name=sparse_backend", "static_cam=True"])
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(source_manifest, provenance=provenance, provenance_hash=digest,
                    source_anchor_manifest=binding(source / "manifest.json"), tracks=[])
    try:
        for number, entry in enumerate(source_manifest["tracks"]):
            track = native.load_teacher_track(source / entry["path"])
            native.validate_provenance(track, source_manifest["provenance_hash"])
            grid = track["temporal_latent_grid"]
            data_full = dict(kp2d=expand_conditioning(track["kp2d"], grid, keypoints=True),
                             bbx_xys=expand_conditioning(track["bbx_xys"], grid),
                             f_imgseq=expand_conditioning(features[number], grid))
            K = track["K_fullimg"][0]
            parts = {key: [] for key in track["smpl_params_incam"]}
            with torch.no_grad():
                for start, end, a, b in native._window_ranges(len(grid["relative_time_us"]), 600):
                    n = b - a
                    data = {key: value[a:b] for key, value in data_full.items()}
                    data.update(length=torch.tensor(n), K_fullimg=K.repeat(n, 1, 1),
                                cam_angvel=torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(n, 1))
                    predicted = model.predict(data, static_cam=True)
                    for key in parts:
                        parts[key].append(predicted["smpl_params_incam"][key][start-a:end-a].detach().cpu())
            states = {key: torch.cat(value) for key, value in parts.items()}
            track.update(provenance=provenance, provenance_hash=digest,
                         inferred_temporal_state=dict(role="inferred_model_state_not_rgb_observation", smpl_params_incam=states),
                         smpl_params_incam={key: value[grid["anchor_indices"]] for key, value in states.items()})
            native._save_tensor(output_dir / entry["path"], track)
            native.load_teacher_track(output_dir / entry["path"])
            manifest["tracks"].append(entry)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    native._save_json(output_dir / "manifest.json", manifest)
    return manifest
