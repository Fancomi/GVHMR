"""Calibrated, timestamp-preserving GVHMR inference for multiple athletes.

Only ``smpl_params_incam`` is used as geometry. GVHMR's gravity-aligned global
trajectory has no measured relationship to the recording's LiDAR frame. Images
are undistorted with the adopted K unchanged, so output keypoints and boxes are
in that undistorted OpenCV camera image, not the original distorted image.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation


SCHEMA = "badminton_gvhmr_teacher_v1"
ALGORITHM_VERSION = 1
COORDINATE_SPACE = "opencv_camera_undistorted"


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value.resolve())
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def _file_identity(path):
    path = Path(path).resolve()
    info = path.stat()
    return {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def _validate_preproc_precision(precision):
    if precision not in ("float32", "float16"):
        raise ValueError("preproc_precision must be 'float32' or 'float16'")
    return precision


def _normalize_rows(frame_rows):
    rows = []
    for row in frame_rows:
        normalized={key:int(row[key]) for key in ("record_index", "common_time_us", "media_pts_us")}
        # Generic native sidecars can omit the original SSP frame counter.
        # PTS/decode/common-time bindings stay mandatory; never invent a counter.
        normalized["source_frame_no"]=None if row.get("source_frame_no") is None else int(row["source_frame_no"])
        rows.append(normalized)
    if not rows:
        raise ValueError("Teacher inference requires at least one native video frame")
    for key in ("record_index", "common_time_us", "media_pts_us"):
        values = np.array([r[key] for r in rows], dtype=np.int64)
        if np.any(np.diff(values) <= 0):
            raise ValueError(f"Teacher {key} must be strictly increasing without duplicate frames")
    if any(row["record_index"] < 0 for row in rows):
        raise ValueError("record_index must be a native, zero-based video decode index")
    if len(rows) > 1:
        delta = np.diff([r["common_time_us"] for r in rows])
        # GVHMR was trained at 30 Hz. In particular, do not feed 10 Hz LiDAR
        # frames while labeling them as a 30 Hz sequence.
        median_delta = float(np.median(delta))
        if not 30_000 <= median_delta <= 37_000:
            raise ValueError("Teacher rows must retain approximately 30 Hz native video cadence")
        if np.any(delta > median_delta * 1.8):
            raise ValueError("Split video discontinuities before temporal teacher inference")
    return rows


def _provenance(video_path, rows, camera, options):
    from hmr4d import PROJ_ROOT

    options = dict(options)
    precision = _validate_preproc_precision(options.get("preproc_precision", "float32"))
    # float32 was the original and remains the default computation. Omit its
    # explicit option so previously produced float32 caches retain their hash.
    # A half-precision run always has a different, recorded fingerprint.
    if precision == "float32":
        options.pop("preproc_precision", None)
    weights = ["yolo/yolov8x.pt", "vitpose/vitpose-h-multi-coco.pth",
               "hmr2/epoch=10-step=25000.ckpt", "gvhmr/gvhmr_siga24_release.ckpt"]
    value = {
        "schema": SCHEMA,
        "algorithm_version": ALGORITHM_VERSION,
        "video": _file_identity(video_path),
        "frame_rows": rows,
        "camera": _jsonable(camera),
        "options": _jsonable(options),
        "weights": [_file_identity(PROJ_ROOT / "inputs/checkpoints" / p) for p in weights],
        "coordinate_space": COORDINATE_SPACE,
        "source_content_sha256_verified": False,
        "source_identity_method": "absolute_path_size_mtime_ns_and_all_native_frame_rows",
        "physical_exposure_synchronization_verified": False,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return value, hashlib.sha256(encoded).hexdigest()


def validate_provenance(cache, expected_hash):
    """Reject stale stage/track caches rather than silently using another run."""
    if cache.get("schema") != SCHEMA or cache.get("provenance_hash") != expected_hash:
        raise ValueError("Teacher cache provenance mismatch: source, calibration, rows, or options changed")


def _save_tensor(path, data):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    torch.save(data, temporary)
    temporary.replace(path)


def _save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(_jsonable(data), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _read_stage(path, provenance_hash, resume):
    import torch

    if resume and Path(path).exists():
        cache = torch.load(path, map_location="cpu")
        validate_provenance(cache, provenance_hash)
        return cache
    return None


def _iter_video_frames(video_path, rows) -> Iterator[tuple[int, dict, np.ndarray]]:
    """Seek and decode requested real container PTS, with no artificial FPS.

    Frame sidecar metadata binds record_index to media_pts_us. Pairing code
    verifies that binding against the recording's adopted frame map. This
    decoder independently verifies each requested container PTS (rounding
    tolerance 2 us), rather than trusting an imprecise OpenCV frame seek.
    """
    from .sessions import iter_native_video_frames
    yield from iter_native_video_frames(video_path,rows)


def _undistorter(camera):
    import cv2
    from hmr4d.utils.geo.hmr_cam import resolve_camera_intrinsics

    K = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(camera.get("distortion_opencv", [0.0] * 5), dtype=np.float64)
    size = tuple(map(int, camera["image_size"]))
    if len(size) != 2 or min(size) <= 0 or distortion.shape != (5,):
        raise ValueError("Expected camera image_size=[width,height] and Brown distortion with 5 coefficients")
    if not np.isfinite(K).all() or not np.isfinite(distortion).all():
        raise ValueError("Camera calibration contains non-finite values")
    resolve_camera_intrinsics(K,*size)
    if not np.any(distortion):
        return lambda rgb: rgb
    map_x, map_y = cv2.initUndistortRectifyMap(K, distortion, None, K, size, cv2.CV_32FC1)
    return lambda rgb: cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR)


def _box_iou(a, b):
    low = np.maximum(a[:2], b[:2])
    high = np.minimum(a[2:], b[2:])
    intersection = np.prod(np.maximum(high - low, 0.0))
    area_a = np.prod(np.maximum(a[2:] - a[:2], 0.0))
    area_b = np.prod(np.maximum(b[2:] - b[:2], 0.0))
    return float(intersection / max(area_a + area_b - intersection, 1e-12))


def _association_cost(predicted, detected):
    wh = np.maximum(predicted[2:] - predicted[:2], 1.0)
    det_wh = np.maximum(detected[2:] - detected[:2], 1.0)
    center_distance = np.linalg.norm((predicted[:2] + predicted[2:] - detected[:2] - detected[2:]) / 2)
    relative_distance = center_distance / max(float(np.linalg.norm(wh)), 1.0)
    shape_ratio = max(float(np.max(det_wh / wh)), float(np.max(wh / det_wh)))
    iou = _box_iou(predicted, detected)
    if shape_ratio > 2.8 or (iou < 0.05 and relative_distance > 0.55):
        return 1000.0
    return (1.0 - iou) + 0.35 * relative_distance


def associate_detections(detections_by_frame, *, max_track_gap_frames=1):
    """One-to-one multi-athlete tracking; never select only the longest person.

    Observed rows are the only rows returned. A later segment split keeps
    missed detections out of temporal fitting and interpolation.
    """
    if max_track_gap_frames < 0:
        raise ValueError("max_track_gap_frames cannot be negative")
    tracks, active = {}, {}
    next_id = 1
    for position, frame in enumerate(detections_by_frame):
        active = {key: value for key, value in active.items()
                  if position - value["last_position"] <= max_track_gap_frames + 1}
        detections = []
        for det in frame:
            box = np.asarray(det["bbx_xyxy"], dtype=np.float32)
            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                continue
            detections.append({**det, "bbx_xyxy": box})
        keys = sorted(active)
        costs = np.full((len(keys), len(detections)), 1000.0)
        for i, key in enumerate(keys):
            state = active[key]
            gap = position - state["last_position"]
            predicted = state["box"] + state["velocity"] * gap
            for j, det in enumerate(detections):
                costs[i, j] = _association_cost(predicted, det["bbx_xyxy"])
        matched = {}
        if costs.size:
            row_indices, col_indices = linear_sum_assignment(costs)
            for i, j in zip(row_indices, col_indices):
                if costs[i, j] < 1.25:
                    alternatives = np.delete(costs[i], j)
                    rival_tracks = np.delete(costs[:, j], i)
                    ambiguous = bool((alternatives.size and alternatives.min() - costs[i, j] < 0.12)
                                     or (rival_tracks.size and rival_tracks.min() - costs[i, j] < 0.12))
                    matched[int(j)] = (keys[i], float(costs[i, j]), ambiguous)
        for j, det in enumerate(detections):
            if j in matched:
                key, cost, ambiguous = matched[j]
                previous = active[key]
                gap = position - previous["last_position"]
                velocity = (det["bbx_xyxy"] - previous["box"]) / gap
            else:
                key, cost, ambiguous = next_id, 0.0, False
                next_id += 1
                velocity = np.zeros(4, dtype=np.float32)
                tracks[key] = []
            active[key] = {"last_position": position, "box": det["bbx_xyxy"], "velocity": velocity}
            tracks[key].append({**det, "position": position, "association_cost": cost,
                                "identity_ambiguous": ambiguous})
    return tracks


def _split_tracks(tracks, rows, min_track_frames):
    segments = []
    rejected = []
    median_delta = float(np.median(np.diff([row["common_time_us"] for row in rows]))) if len(rows) > 1 else 33_367.0
    for track_id, observations in sorted(tracks.items()):
        current, pieces = [], []
        for observation in observations:
            if current:
                before = current[-1]["position"]
                after = observation["position"]
                # Break at every missed sampled frame: no unobserved bbox or
                # person is inserted as a training teacher.
                if after != before + 1 or rows[after]["common_time_us"] - rows[before]["common_time_us"] > median_delta * 1.8:
                    pieces.append(current)
                    current = []
            current.append(observation)
        if current:
            pieces.append(current)
        for segment_id, piece in enumerate(pieces):
            if len(piece) < min_track_frames:
                rejected.append({"track_id": track_id, "segment_id": segment_id,
                                 "frame_count": len(piece), "reason": "insufficient_observed_temporal_context"})
                continue
            segments.append({"track_id": track_id, "segment_id": segment_id, "observations": piece})
    return segments, rejected


def _detect(video_path, rows, camera, roi_polygon, roi_margin_px, detector_imgsz):
    import cv2
    import torch
    from ultralytics import YOLO
    from hmr4d import PROJ_ROOT

    undistort = _undistorter(camera)
    polygon = None if roi_polygon is None else np.asarray(roi_polygon, dtype=np.float32).reshape(-1, 2)
    if polygon is not None and len(polygon) < 3:
        raise ValueError("roi_polygon needs at least three undistorted image points")
    yolo = YOLO(PROJ_ROOT / "inputs/checkpoints/yolo/yolov8x.pt")
    detections = []
    expected_size = tuple(map(int, camera["image_size"]))
    try:
        for _, row, rgb in _iter_video_frames(video_path, rows):
            if (rgb.shape[1], rgb.shape[0]) != expected_size:
                raise ValueError("Video dimensions do not match the adopted camera calibration")
            rgb = undistort(rgb)
            with torch.no_grad():
                result = yolo.predict(rgb[..., ::-1].copy(), classes=[0], conf=0.35,
                                      device=0, imgsz=detector_imgsz, verbose=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            frame = []
            for box, confidence in zip(boxes, confidences):
                foot = ((float(box[0]) + float(box[2])) / 2, float(box[3]))
                if polygon is not None and cv2.pointPolygonTest(polygon, foot, True) < -roi_margin_px:
                    continue
                frame.append({"bbx_xyxy": box.astype(np.float32), "detection_confidence": float(confidence)})
            detections.append(frame)
            if len(detections) % 120 == 0:
                print(f"Teacher YOLO: {len(detections)}/{len(rows)} native frames", flush=True)
    finally:
        del yolo
        gc.collect()
        torch.cuda.empty_cache()
    return detections


def _crop_batches(video_path, rows, camera, segments, batch_size):
    import torch
    from hmr4d.network.hmr2.utils.preproc import crop_and_resize, IMAGE_MEAN, IMAGE_STD

    undistort = _undistorter(camera)
    position_map = {}
    for segment_index, segment in enumerate(segments):
        for local_index, observation in enumerate(segment["observations"]):
            box = np.asarray(observation["bbx_xyxy"], dtype=np.float32)
            position_map.setdefault(observation["position"], []).append(
                (segment_index, local_index, _bbox_xys(box)))
    useful_positions = sorted(position_map)
    useful_rows = [rows[p] for p in useful_positions]
    images, bbx_xys, identifiers = [], [], []
    for subset_position, _, rgb in _iter_video_frames(video_path, useful_rows):
        position = useful_positions[subset_position]
        rgb = undistort(rgb)
        for segment_index, local_index, box_xys in position_map[position]:
            crop, _ = crop_and_resize(rgb, box_xys[:2], box_xys[2], dst_size=256, enlarge_ratio=1.0)
            images.append(crop)
            bbx_xys.append(box_xys)
            identifiers.append((segment_index, local_index))
            if len(images) == batch_size:
                batch = torch.from_numpy(np.stack(images)).float() / 255.0
                batch = ((batch - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)
                yield batch, torch.from_numpy(np.stack(bbx_xys)).float(), identifiers
                images, bbx_xys, identifiers = [], [], []
    if images:
        batch = torch.from_numpy(np.stack(images)).float() / 255.0
        batch = ((batch - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)
        yield batch, torch.from_numpy(np.stack(bbx_xys)).float(), identifiers


def _bbox_xys(box):
    box = np.asarray(box, dtype=np.float32)
    center = (box[:2] + box[2:]) / 2.0
    width, height = box[2:] - box[:2]
    # Both extractors consume only the middle 192 of 256 crop columns.
    # Match upstream get_bbx_xys(..., bbx_ratio=[192,256]).
    size = max(float(height), float(width) * 4 / 3) * 1.2
    return np.array([*center, size], dtype=np.float32)


@contextmanager
def _quiet_extractor_progress(extractor):
    """Suppress tiny upstream batch bars, restoring the module on every exit.

    The pipeline runs image models sequentially in a single process. Restrict
    this temporary override to the active extractor's own module; the teacher
    stage's persistent progress messages remain visible.
    """
    module = importlib.import_module(type(extractor).__module__)
    original = module.tqdm
    module.tqdm = partial(original, disable=True)
    try:
        yield
    finally:
        module.tqdm = original


def _extract_stage(video_path, rows, camera, segments, batch_size, stage_name, preproc_precision="float32"):
    import torch

    _validate_preproc_precision(preproc_precision)
    if stage_name == "vitpose":
        from hmr4d.utils.preproc.vitpose import VitPoseExtractor
        extractor = VitPoseExtractor(tqdm_leave=False, batch_size=batch_size)
        trailing_shape = (17, 3)
    elif stage_name == "features":
        from hmr4d.utils.preproc.vitfeat_extractor import Extractor
        extractor = Extractor(tqdm_leave=False, batch_size=batch_size)
        trailing_shape = (1024,)
    else:
        raise ValueError(f"Unknown feature stage {stage_name}")
    results = [torch.full((len(s["observations"]), *trailing_shape), float("nan")) for s in segments]
    processed = 0
    try:
        # Keep one autocast context for the whole stage so casts of the large
        # model weights are cached across batches, rather than repeated for
        # every four-crop forward pass.
        with _quiet_extractor_progress(extractor), torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=preproc_precision == "float16"):
            for crops, bbx_xys, identifiers in _crop_batches(video_path, rows, camera, segments, batch_size):
                if stage_name == "vitpose":
                    batch_results = extractor.extract(crops, bbx_xys)
                else:
                    batch_results = extractor.extract_video_features(crops, bbx_xys)
                # Keep caches and subsequent GVHMR inference in float32 even
                # when the two image models used half-precision arithmetic.
                batch_results = batch_results.float()
                for value, (segment_index, local_index) in zip(batch_results, identifiers):
                    results[segment_index][local_index] = value
                processed += len(identifiers)
                if processed % (batch_size * 30) == 0:
                    print(f"Teacher {stage_name}: {processed} observed athlete frames", flush=True)
        if not all(torch.isfinite(value).all().item() for value in results):
            raise ValueError(f"Incomplete or non-finite {stage_name} inference")
    finally:
        del extractor
        gc.collect()
        torch.cuda.empty_cache()
    return results


def _window_ranges(length, core_frames, context_frames=60):
    for start in range(0, length, core_frames):
        end = min(start + core_frames, length)
        yield start, end, max(0, start - context_frames), min(length, end + context_frames)


def run_teacher(video_path, frame_rows, camera, output_dir, batch_size=4, resume=True, *,
                min_track_frames=8, max_track_gap_frames=1, inference_window_frames=600,
                roi_polygon=None, roi_margin_px=40.0, detector_imgsz=1280, preproc_precision="float32"):
    """Run four GPU stages independently and save every observed athlete track.

    ``frame_rows`` must be a continuous approximately 30 Hz subset of native
    video rows. ``roi_polygon`` uses undistorted pixels. Long tracks are inferred
    in windows with 60 frames of temporal context and unique retained cores.
    ``resume=False`` explicitly rebuilds this output directory's caches.
    Optional ``preproc_precision='float16'`` autocasts only ViTPose and HMR2;
    their saved outputs and the temporal GVHMR model remain float32.
    """
    import torch

    _validate_preproc_precision(preproc_precision)
    if batch_size < 1 or min_track_frames < 2 or inference_window_frames < 8:
        raise ValueError("batch_size >= 1, min_track_frames >= 2, and inference_window_frames >= 8 required")
    video_path, output_dir = Path(video_path).resolve(), Path(output_dir).resolve()
    rows = _normalize_rows(frame_rows)
    _undistorter(camera)  # Validate camera before creating a cache.
    options = dict(batch_size=batch_size, min_track_frames=min_track_frames,
                   max_track_gap_frames=max_track_gap_frames, inference_window_frames=inference_window_frames,
                   roi_polygon=roi_polygon, roi_margin_px=roi_margin_px, detector_imgsz=detector_imgsz,
                   preproc_precision=preproc_precision)
    provenance, provenance_hash = _provenance(video_path, rows, camera, options)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_provenance(manifest, provenance_hash)
        manifest.setdefault("preproc_precision", "float32")
        for info in manifest["tracks"]:
            track_path = output_dir / info["path"]
            if not track_path.is_file():
                raise ValueError(f"Teacher manifest references missing track: {track_path}")
            validate_provenance(load_teacher_track(track_path), provenance_hash)
        return manifest
    if not torch.cuda.is_available():
        raise RuntimeError("Calibrated GVHMR teacher inference requires the configured CUDA environment")
    detection_path = output_dir / "detections.pt"
    detection_cache = _read_stage(detection_path, provenance_hash, resume)
    if detection_cache is None:
        detections = _detect(video_path, rows, camera, roi_polygon, roi_margin_px, detector_imgsz)
        detection_cache = {"schema": SCHEMA, "provenance_hash": provenance_hash, "detections": detections}
        _save_tensor(detection_path, detection_cache)
    if len(detection_cache["detections"]) != len(rows):
        raise ValueError("Detection cache frame count does not match source provenance")
    tracks = associate_detections(detection_cache["detections"], max_track_gap_frames=max_track_gap_frames)
    segments, rejected = _split_tracks(tracks, rows, min_track_frames)
    manifest = {"schema": SCHEMA, "provenance_hash": provenance_hash, "provenance": provenance,
                "frame_count": len(rows), "coordinate_space": COORDINATE_SPACE, "tracks": [],
                "rejected_segments": rejected, "status": "complete", "static_camera": True,
                "preproc_precision": preproc_precision,
                "source_fps_median": 1_000_000 / float(np.median(np.diff([r["common_time_us"] for r in rows])))
                if len(rows) > 1 else None}
    if not segments:
        manifest["status"] = "no_observed_tracks_with_sufficient_context"
        _save_json(manifest_path, manifest)
        return manifest
    preprocessed = {}
    for stage in ("vitpose", "features"):
        stage_path = output_dir / f"{stage}.pt"
        cache = _read_stage(stage_path, provenance_hash, resume)
        if cache is None:
            values = _extract_stage(video_path, rows, camera, segments, batch_size, stage, preproc_precision)
            cache = {"schema": SCHEMA, "provenance_hash": provenance_hash, "values": values,
                     "preproc_precision": preproc_precision}
            _save_tensor(stage_path, cache)
        if len(cache["values"]) != len(segments):
            raise ValueError(f"Teacher {stage} cache segment count mismatch")
        preprocessed[stage] = cache["values"]
    # Imported after the larger image models have been destroyed.
    import hydra
    from hydra import compose, initialize_config_module
    from hmr4d.configs import register_store_gvhmr
    # The demo config node registers at import time; store_gvhmr registers
    # training nodes but does not import gvhmr_pl_demo itself.
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # noqa: F401

    register_store_gvhmr()
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(config_name="demo", overrides=[f"video_name={video_path.stem}", "static_cam=True"])
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()
    K = torch.as_tensor(camera["K"], dtype=torch.float32).reshape(3, 3)
    try:
        for segment_index, segment in enumerate(segments):
            track_id, segment_id = segment["track_id"], segment["segment_id"]
            name = f"track_{track_id:04d}_segment_{segment_id:04d}.pt"
            track_path = output_dir / name
            cached_track = _read_stage(track_path, provenance_hash, resume)
            observations = segment["observations"]
            length = len(observations)
            if cached_track is None:
                bbx_xyxy = torch.as_tensor(np.stack([o["bbx_xyxy"] for o in observations]), dtype=torch.float32)
                bbx_xys = torch.as_tensor(np.stack([_bbox_xys(o["bbx_xyxy"]) for o in observations]))
                kp2d = preprocessed["vitpose"][segment_index]
                features = preprocessed["features"][segment_index]
                params_parts = {key: [] for key in ("body_pose", "betas", "global_orient", "transl")}
                with torch.no_grad():
                    for start, end, context_start, context_end in _window_ranges(length, inference_window_frames):
                        window_length = context_end - context_start
                        sl = slice(context_start, context_end)
                        data = {"length": torch.tensor(window_length), "bbx_xys": bbx_xys[sl],
                                "kp2d": kp2d[sl], "K_fullimg": K.repeat(window_length, 1, 1),
                                "cam_angvel": torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(window_length, 1),
                                "f_imgseq": features[sl]}
                        prediction = model.predict(data, static_cam=True)
                        kept = slice(start - context_start, end - context_start)
                        for key in params_parts:
                            params_parts[key].append(prediction["smpl_params_incam"][key][kept].detach().cpu())
                        del prediction
                selected_rows = [rows[o["position"]] for o in observations]
                params = {key: torch.cat(parts, dim=0) for key, parts in params_parts.items()}
                if not all(torch.isfinite(value).all().item() for value in params.values()):
                    raise ValueError("GVHMR produced non-finite camera parameters")
                cached_track = {
                    "schema": SCHEMA, "provenance_hash": provenance_hash, "provenance": provenance,
                    "preproc_precision": preproc_precision,
                    "track_id": track_id, "segment_id": segment_id, "coordinate_space": COORDINATE_SPACE,
                    "smpl_model_type": "smplx_neutral_body21_betas10", "smpl_params_incam": params,
                    "kp2d": kp2d, "bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys,
                    "K_fullimg": K.repeat(length, 1, 1),
                    "distortion_original": _jsonable(camera.get("distortion_opencv", [0.] * 5)),
                    "teacher_image_distortion": [0.] * 5,
                    "detection_confidence": torch.tensor([o["detection_confidence"] for o in observations]),
                    "association_cost": torch.tensor([o["association_cost"] for o in observations]),
                    "identity_ambiguous": torch.tensor([o["identity_ambiguous"] for o in observations]),
                    "tracking_identity_verified": False,
                    "static_camera": True, "native_video_frame_rows": selected_rows,
                    **{key: torch.tensor([r[key] if r[key] is not None else -1 for r in selected_rows], dtype=torch.int64)
                       for key in ("record_index", "common_time_us", "media_pts_us", "source_frame_no")},
                }
                _save_tensor(track_path, cached_track)
            manifest["tracks"].append({"track_id": track_id, "segment_id": segment_id, "path": name,
                                       "frame_count": length,
                                       "start_common_time_us": int(cached_track["common_time_us"][0]),
                                       "end_common_time_us": int(cached_track["common_time_us"][-1])})
            print(f"Teacher GVHMR: track {track_id}, segment {segment_id}, {length} observed frames", flush=True)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    _save_json(manifest_path, manifest)
    return manifest


def load_teacher_track(path):
    import torch

    track = torch.load(path, map_location="cpu")
    if track.get("schema") != SCHEMA or track.get("coordinate_space") != COORDINATE_SPACE:
        raise ValueError("Unsupported teacher track schema or coordinate convention")
    length = len(track["common_time_us"])
    if length < 2 or np.any(np.diff(_numpy(track["common_time_us"])) <= 0):
        raise ValueError("Teacher track must retain strictly increasing observed frame timestamps")
    shapes = {"kp2d": (length, 17, 3), "bbx_xyxy": (length, 4),
              "bbx_xys": (length, 3), "K_fullimg": (length, 3, 3)}
    param_shapes = {"body_pose": (length, 63), "global_orient": (length, 3),
                    "transl": (length, 3), "betas": (length, 10)}
    for key, shape in shapes.items():
        value = _numpy(track[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Teacher track {key} must have finite shape {shape}")
    for key, shape in param_shapes.items():
        value = _numpy(track["smpl_params_incam"][key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Teacher track SMPL-X {key} must have finite shape {shape}")
    for key in ("record_index", "media_pts_us", "source_frame_no"):
        if _numpy(track[key]).shape != (length,):
            raise ValueError(f"Teacher track native {key} length mismatch")
    return track


def _slerp_rotvec(left, right, alpha):
    left, right = np.asarray(left), np.asarray(right)
    shape = left.shape
    if not len(alpha):
        return left.astype(np.float32)
    q0 = Rotation.from_rotvec(left.reshape(-1, 3)).as_quat().reshape(*shape[:-1], 4)
    q1 = Rotation.from_rotvec(right.reshape(-1, 3)).as_quat().reshape(*shape[:-1], 4)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), 0, 1)
    angle = np.arccos(dot)
    fraction = np.asarray(alpha).reshape((len(alpha),) + (1,) * (q0.ndim - 1))
    denominator = np.sin(angle)
    close = denominator < 1e-7
    denominator = np.where(close, 1.0, denominator)
    w0 = np.sin((1 - fraction) * angle) / denominator
    w1 = np.sin(fraction * angle) / denominator
    q = np.where(close, (1 - fraction) * q0 + fraction * q1, w0 * q0 + w1 * q1)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return Rotation.from_quat(q.reshape(-1, 4)).as_rotvec().reshape(shape).astype(np.float32)


def sample_track(track, target_times_us, max_time_error_us=18_000):
    """Interpolate camera SMPL-X on SO(3); retain nearest real-frame keypoints.

    ``valid`` requires an observed teacher frame within the tolerance, no large
    bracketing gap, and an unambiguous association. ``model_time_error_us`` is
    the signed nearest *teacher* sample error, separate from native camera/LiDAR
    synchronization. Callers must enforce ``valid`` before exporting labels.
    Returned numerical arrays are NumPy float32, timestamp arrays are int64.
    """
    if max_time_error_us <= 0:
        raise ValueError("max_time_error_us must be positive")
    if track.get("coordinate_space") != COORDINATE_SPACE:
        raise ValueError("Only calibrated OpenCV camera teacher tracks can be sampled")
    times = _numpy(track["common_time_us"]).astype(np.int64)
    target = np.asarray(target_times_us, dtype=np.int64).reshape(-1)
    if times.size < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Track timestamps must contain at least two increasing real frames")
    right = np.searchsorted(times, target, side="left").clip(0, len(times) - 1)
    left = (right - 1).clip(0, len(times) - 1)
    # Outside the track, nearest sampling is permitted only inside the stated
    # tolerance. No motion extrapolation is performed.
    before = target <= times[0]
    after = target >= times[-1]
    left[before], right[before] = 0, 0
    left[after], right[after] = len(times) - 1, len(times) - 1
    exact = times[right] == target
    left[exact] = right[exact]
    left_distance = np.abs(target - times[left])
    right_distance = np.abs(times[right] - target)
    nearest = np.where(left_distance <= right_distance, left, right)  # earlier on a tie
    gaps = times[right] - times[left]
    alpha = np.divide(target - times[left], gaps, out=np.zeros(len(target), dtype=np.float64), where=gaps > 0)
    alpha = np.clip(alpha, 0, 1)
    error = times[nearest] - target
    identity = _numpy(track.get("identity_ambiguous", np.zeros(len(times), dtype=bool))).astype(bool)
    valid = (np.abs(error) <= max_time_error_us) & (gaps <= 2 * max_time_error_us)
    valid &= ~identity[left] & ~identity[right]
    params = {}
    for key, value in track["smpl_params_incam"].items():
        values = _numpy(value).astype(np.float32)
        if key in ("global_orient", "body_pose"):
            flat_dim = int(np.prod(values.shape[1:]))
            shaped_left = values[left].reshape(len(target), flat_dim // 3, 3)
            shaped_right = values[right].reshape(len(target), flat_dim // 3, 3)
            params[key] = _slerp_rotvec(shaped_left, shaped_right, alpha).reshape(len(target), flat_dim)
        else:
            fraction = alpha.reshape((len(target),) + (1,) * (values.ndim - 1))
            params[key] = ((1 - fraction) * values[left] + fraction * values[right]).astype(np.float32)
    metadata = {
        "target_common_time_us": target, "teacher_common_time_us": times[nearest],
        "model_time_error_us": error, "teacher_left_time_us": times[left], "teacher_right_time_us": times[right],
        "interpolation_alpha": alpha.astype(np.float32), "provenance_hash": track["provenance_hash"],
        "track_id": track["track_id"], "segment_id": track["segment_id"],
        "coordinate_space": COORDINATE_SPACE,
        "keypoint_sampling": "nearest_observed_undistorted_native_video_frame",
        "parameter_sampling": "SO3_slerp_pose_linear_transl_betas_no_extrapolation",
        **{key: _numpy(track[key]).astype(np.int64)[nearest]
           for key in ("record_index", "media_pts_us", "source_frame_no")},
    }
    # -1 is only the tensor storage sentinel for an absent original counter.
    # Serialized fit provenance retains null, not a fabricated frame number.
    if np.any(metadata['source_frame_no']<0):
        metadata['source_frame_no']=np.array([None if n<0 else int(n) for n in metadata['source_frame_no']],dtype=object)
    return {"smpl_params_incam": params, "kp2d": _numpy(track["kp2d"])[nearest].astype(np.float32),
            "bbx_xyxy": _numpy(track["bbx_xyxy"])[nearest].astype(np.float32),
            "bbx_xys": _numpy(track["bbx_xys"])[nearest].astype(np.float32),
            "K_fullimg": _numpy(track["K_fullimg"])[nearest].astype(np.float32),
            "valid": valid, "metadata": metadata, "meta": metadata}
