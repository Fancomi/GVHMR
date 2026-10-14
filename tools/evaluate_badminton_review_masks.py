"""Evaluate current production review geometry against native RGB instance masks.

Uses the shared alignment evaluator without camera candidates, benchmark meshes,
new fitting, or altered source returns. Cached model predictions are evidence,
not segmentation ground truth or a quality pass certification.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
EVENT_SESSION = "20261012_163452"
EVENT_SCANS = (14, 15, 23, 24, 25, 37, 38, 39, 40)
NATIVE_FIELDS = ("record_index", "common_time_us", "media_pts_us", "source_frame_no")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _bounded(path, root, description):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"{description} must remain under {root}: {path}")
    return path


def _runtime(device):
    # Hide CUDA before importing torch/Ultralytics for genuinely CPU-only runs.
    previous_cuda_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tools_dir = str(WORKSPACE / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    try:
        import evaluate_badminton_alignment as alignment
    finally:
        # The shared evaluator imports the CPU-only RGB checker for its pure
        # bbox association functions. Restore its import-time environment side
        # effect before any explicitly requested CUDA inference is initialized.
        if device != "cpu":
            if previous_cuda_visibility is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_visibility
    from hmr4d.lidar_mocap.sequence_review import review_selection_digest

    alignment.review_selection_digest = review_selection_digest
    return alignment


def _quality(window, output_root, cache):
    window = _bounded(window, output_root, "Production window")
    if window not in cache:
        records = [json.loads(line) for line in (window / "quality.jsonl").read_text(encoding="utf-8").splitlines()]
        rows = defaultdict(list)
        for row in records:
            rows[int(row["lidar_index"])].append(row)
        cache[window] = rows
    return cache[window]


def select_cases(manifest, output_root):
    """Fixed first/middle/last sampling plus the named user events; deduplicate scans."""
    selected = {}
    quality = {}
    event_clips = []

    def add(clip, frame, reason):
        identity = (clip["session_id"], int(frame["lidar_index"]))
        evidence = {k: frame[k] for k in ("scan_mid_us", "lidar_path", "cameras", "athletes")}
        if identity in selected:
            previous = selected[identity]
            old = {k: previous["frame"][k] for k in evidence}
            if old != evidence:
                raise ValueError(f"Overlapping review clips disagree about source/geometry: {identity}")
            if reason not in previous["selection_reasons"]:
                previous["selection_reasons"].append(reason)
            return
        selected[identity] = {
            "key": f"{clip['id']}_{frame['lidar_index']:06d}",
            "clip_id": clip["id"], "session_id": clip["session_id"],
            "source_window": clip["source_window"], "frame": frame,
            "selection_reasons": [reason],
        }

    clips = manifest.get("clips", [])
    if not clips:
        raise ValueError("Review manifest has no clips")
    for clip in clips:
        frames = clip["frames"]
        if not frames:
            raise ValueError(f"Review clip has no source scans: {clip['id']}")
        if any(b["scan_mid_us"] <= a["scan_mid_us"] for a, b in zip(frames, frames[1:])):
            raise ValueError("Review scan timestamps must be strictly increasing")
        for index, name in ((0, "first"), (len(frames) // 2, "middle"), (len(frames) - 1, "last")):
            add(clip, frames[index], f"{clip['id']}:{name}")
        if clip["session_id"] == EVENT_SESSION:
            event_clips.append(clip)
    missing = []
    for scan in EVENT_SCANS if event_clips else ():
        existing = next(((clip, f) for clip in event_clips for f in clip["frames"]
                         if f["lidar_index"] == scan), None)
        if existing:
            add(*existing, f"user_event:{scan}")
            continue
        found = False
        for clip in event_clips:
            rows = _quality(clip["source_window"], output_root, quality).get(scan, [])
            if not rows:
                continue
            row = rows[0]
            frame = {k: row[k] for k in ("scan_mid_us", "lidar_index", "lidar_path", "cameras")}
            frame["offset_s"] = (row["scan_mid_us"] - clip["left_us"]) / 1e6
            frame["athletes"] = [{k: r[k] for k in ("motion_id", "fit_path", "accepted", "qa", "rejection_reasons")}
                                 for r in rows]
            add(clip, frame, f"user_event:{scan}:outside_clip_same_source_window")
            found = True
            break
        if not found:
            missing.append(scan)
    if missing:
        raise ValueError(f"New production source windows lack requested user-event scans: {missing}")
    return list(selected.values()), {
        "rule": "first/middle/last of every supplied clip, plus fixed user-event scans; deduplicate(session,scan)",
        "supplied_clips": len(clips), "full_selected_cases": len(selected),
        "event_session": EVENT_SESSION, "event_scans": list(EVENT_SCANS),
        "event_session_present": bool(event_clips), "quality_cache": quality,
    }


def validate_court(court, court_path, calibration, runtime):
    """Admit only inherited static evidence with exactly the same source geometry."""
    if court.get("adopted_changed"):
        raise ValueError("Court evidence must retain the adopted baseline")
    baseline = court["camera_models"]["baseline"]
    checked = {}
    for camera in ("cam1", "cam2"):
        checked[camera] = {}
        adopted = calibration.camera(camera)
        for field in ("K", "distortion_opencv", "R_raw_lidar_to_camera", "t_m", "image_size"):
            np.testing.assert_allclose(baseline[camera][field], adopted[field], rtol=0, atol=2e-10,
                                       err_msg=f"Historical court baseline differs from {camera}.{field}")
            checked[camera][field] = True
    np.testing.assert_allclose(court["source_M_raw_to_viewer"], calibration.Q, rtol=0, atol=1e-12)
    registration = calibration.data["court_registration"]
    np.testing.assert_allclose(court["registered_Q_raw_to_court"], registration["R_raw_lidar_to_court"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(court["registered_q_raw_to_court_m"], registration["t_raw_lidar_to_court_m"], rtol=0, atol=1e-12)
    source_snapshots = [(Path(path), item) for path, item in court["source_provenance"].items()
                        if Path(path).name == "calibration_snapshot.json"]
    if len(source_snapshots) != 1:
        raise ValueError("Court evidence must identify one original frozen calibration snapshot")
    old_path, provenance = source_snapshots[0]
    if runtime.sha(old_path) != provenance["sha256"]:
        raise ValueError("Historical court snapshot SHA no longer matches its evidence")
    old = runtime.CalibrationSnapshot.load(old_path)
    if old.data["snapshot_digest"] != court["parent_calibration_digest"]:
        raise ValueError("Historical court evidence is bound to another snapshot")
    if old.data["reference_plane"] != calibration.data["reference_plane"]:
        raise ValueError("Historical court and production reference planes differ")
    for segment in court["court_line_segments"]:
        xyz = np.asarray(segment["viewer_xyz_m"], dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
            raise ValueError("Court guide must contain finite dense original-viewer points")
    inherited = dict(court)
    for field in ("camera_models", "evaluations", "projected_court_lines", "repeat_corner_evaluations"):
        inherited[field] = {"baseline": court[field]["baseline"]}
    inherited["review_binding"] = {
        "court_evidence_path": str(court_path), "court_evidence_sha256": runtime.sha(court_path),
        "historical_snapshot_digest": old.data["snapshot_digest"],
        "production_snapshot_digest": calibration.data["snapshot_digest"],
        "camera_parameter_checks": checked, "source_M_and_registered_court_Q_q_equal": True,
        "reference_plane_equal": True, "snapshot_digest_equality_required": False,
        "scope": "Static marked-court evidence reused with identical geometry and cameras; no new court observations inferred from review RGB",
        "native_overlay": "All five Brown coefficients, original SourceM inverse and original native RGB; no undistortion",
    }
    return inherited


def _source_key(sid, camera, row):
    return (sid, camera, *(row[k] for k in NATIVE_FIELDS))


def legacy_mask_index(court_path, model_info, runtime):
    """Index only explicitly adjacent existing evaluation caches, without modifying them."""
    result = defaultdict(list)
    expected = dict(model_info["options"])
    expected.pop("device", None)
    for name in ("joint_alignment", "spatial_alignment"):
        directory = court_path.parent / name
        report_path = directory / "joint_alignment.json"
        if not report_path.exists():
            continue
        previous = _json(report_path)
        model = previous.get("segmentation_model", {})
        actual = dict(model.get("options", {}))
        actual.pop("device", None)
        if model.get("sha256") != model_info["sha256"] or model.get("ultralytics") != model_info["ultralytics"] or actual != expected:
            continue
        seen = set()
        for row in previous.get("measurements", []):
            native = row.get("native_camera_row", {})
            if any(k not in native for k in NATIVE_FIELDS):
                continue
            key = _source_key(row["session_id"], row["camera"], native)
            filename = f"{row['clip_id']}_{row['lidar_index']:06d}_{row['camera']}.npz"
            cache = directory / "masks" / filename
            unique = (key, cache)
            if unique in seen or not cache.exists():
                continue
            seen.add(unique)
            result[key].append({"cache": cache, "model": model, "native_row": native,
                                "report": report_path})
    return result


class MaskObserver:
    def __init__(self, runtime, model_path, model_info, options, output, legacy):
        self.runtime, self.model_path, self.model_info = runtime, model_path, model_info
        self.options, self.output, self.legacy = options, output, legacy
        self.model = None
        self.counts = {"local_cache": 0, "legacy_cache": 0, "new_inference": 0}

    def observe(self, session, camera, row, image):
        video = session.video_paths[camera]
        stat = video.stat()
        source = {"video": str(video), "video_bytes": stat.st_size, "video_mtime_ns": stat.st_mtime_ns}
        canonical_model = {k: self.model_info[k] for k in ("sha256", "ultralytics")}
        canonical_model["options"] = {k: v for k, v in self.options.items() if k != "device"}
        # Exposure identity has no dependency on the target scan's nearest-time error.
        identity_payload = {"schema": "native-person-mask-cache-v1", "model": canonical_model,
                            **source, "native_row": {k: row[k] for k in NATIVE_FIELDS},
                            "image_shape": list(image.shape),
                            "native_bgr_image_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
        identity = _digest(identity_payload)
        cache = self.output / "masks" / f"{session.session_id}_{camera}_{row['record_index']:08d}_{identity[:16]}.npz"
        meta_path = cache.with_suffix(".json")
        metadata = {"identity": identity, "identity_payload": identity_payload,
                    "requested_device": self.options["device"], "video_content_sha256_verified": False}
        if cache.exists():
            if meta_path.exists():
                previous = _json(meta_path)
                if previous.get("identity") != identity or previous.get("cache_sha256") != self.runtime.sha(cache):
                    raise ValueError("Local mask cache metadata/hash differs")
                metadata = previous
            values = self.runtime.mask_observation(None, image, cache, identity, self.options)
            origin = "local_cache"
        else:
            values = None
            for candidate in self.legacy.get(_source_key(session.session_id, camera, row), []):
                legacy_payload = {"model": candidate["model"], **source, "native_row": candidate["native_row"]}
                legacy_identity = hashlib.sha256(json.dumps(legacy_payload, sort_keys=True).encode()).hexdigest()
                with np.load(candidate["cache"]) as saved:
                    if str(saved["identity"]) != legacy_identity:
                        continue
                values = self.runtime.mask_observation(None, image, candidate["cache"], legacy_identity, self.options)
                metadata.update(origin="legacy_cache", actual_inference_device=candidate["model"]["options"]["device"],
                                legacy_cache=str(candidate["cache"]), legacy_cache_sha256=self.runtime.sha(candidate["cache"]),
                                legacy_report=str(candidate["report"]), legacy_report_sha256=self.runtime.sha(candidate["report"]),
                                legacy_identity=legacy_identity,
                                legacy_validation="Same model/package/prediction options and exact native source row/video bytes/mtime; imported frozen predictions, not rerun on requested device")
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache, masks=values[0], boxes=values[1], confidence=values[2], identity=identity)
                break
            if values is None:
                if self.model is None:
                    self.model = self.runtime.YOLO(self.model_path)
                    if self.model.task != "segment":
                        raise ValueError("The checkpoint must provide instance segmentation, not detector boxes")
                values = self.runtime.mask_observation(self.model, image, cache, identity, self.options)
                metadata.update(origin="new_inference", actual_inference_device=self.options["device"])
            origin = metadata["origin"]
        masks, boxes, confidence = values
        if masks.ndim != 3 or masks.shape[1:] != image.shape[:2] or boxes.shape != (len(masks), 4) or confidence.shape != (len(masks),):
            raise ValueError("Native masks, boxes and confidence have incompatible shapes")
        if not np.isfinite(boxes).all() or not np.isfinite(confidence).all():
            raise ValueError("Mask detections contain nonfinite evidence")
        metadata.setdefault("origin", origin)
        metadata["cache_sha256"] = self.runtime.sha(cache)
        self.runtime.write_json(meta_path, metadata)
        self.counts[origin] += 1
        return values, {"cache_path": str(cache), "cache_sha256": metadata["cache_sha256"],
                        "identity": identity, "this_run_source": origin,
                        "actual_inference_device": metadata.get("actual_inference_device", "unknown_prior_cache"),
                        "native_bgr_image_sha256": identity_payload["native_bgr_image_sha256"],
                        "native_camera_row": row, "image_space": "original_native_distorted_Brown_pixels"}


def _decode_groups(indices, maximum_gap=600):
    groups = []
    for index in sorted(indices):
        if not groups or index - groups[-1][-1] > maximum_gap:
            groups.append([])
        groups[-1].append(index)
    return groups


def _validate_case(case, output_root, calibration, session, quality_cache, runtime):
    frame = case["frame"]
    pair = session.pairs()[int(frame["lidar_index"])]
    if pair["scan_mid_us"] != frame["scan_mid_us"] or Path(pair["lidar_path"]).resolve() != Path(frame["lidar_path"]).resolve():
        raise ValueError("Review scan identity differs from actual source")
    for camera in ("cam1", "cam2"):
        if frame["cameras"][camera] != pair["cameras"][camera] or pair["cameras"][camera]["status"] != "matched":
            raise ValueError("Review camera must be the actual nearest matched native exposure")
    rows = _quality(case["source_window"], output_root, quality_cache)[int(frame["lidar_index"])]
    expected = {q["motion_id"]: q for q in rows}
    if set(expected) != {a["motion_id"] for a in frame["athletes"]}:
        raise ValueError("Review athlete set differs from current production quality records")
    fit_provenance = []
    for athlete in frame["athletes"]:
        q = expected[athlete["motion_id"]]
        if q["calibration_digest"] != calibration.data["snapshot_digest"] or q["scan_mid_us"] != frame["scan_mid_us"]:
            raise ValueError("Production fit is bound to another calibration/time")
        path = _bounded(athlete["fit_path"], output_root, "Production fit")
        if path != Path(q["fit_path"]).resolve() or athlete["accepted"] != q["accepted"] or athlete["qa"] != q["qa"]:
            raise ValueError("Review saved geometry/quality does not match current production")
        with np.load(path) as stored:
            if stored["vertices"].shape != (6890, 3) or not np.isfinite(stored["vertices"]).all():
                raise ValueError("Production SMPL vertices must be finite neutral SMPL topology")
            if stored["source_cloud"].ndim != 2 or stored["source_cloud"].shape[1] != 3 or not np.isfinite(stored["source_cloud"]).all():
                raise ValueError("Production source cloud must contain finite actual returns")
        fit_provenance.append({"motion_id": athlete["motion_id"], "path": str(path), "sha256": runtime.sha(path),
                               "accepted": q["accepted"], "rejection_reasons": q["rejection_reasons"]})
    case["fit_provenance"] = fit_provenance


def _body_summary(cases, runtime):
    rows = [a for case in cases for a in case["frame"]["athletes"]]
    accepted = [a for a in rows if a["accepted"]]
    return {"scope": "Actual current-production quality records for the same deduplicated sampled scans; each athlete counted once across cameras",
            "fitted_person_scans": len(rows), "accepted_person_scans": len(accepted),
            "rejected_person_scans": len(rows) - len(accepted),
            "accepted_surface_mean_m": runtime.summary([a["qa"]["final_p2surface_mean_m"] for a in accepted]),
            "accepted_frame_p90_surface_m": runtime.summary([a["qa"]["final_p2surface_p90_m"] for a in accepted]),
            "accepted_cam2_fitting_residual_px": runtime.summary([a["qa"]["reprojection_by_camera_px"]["cam2"] for a in accepted]),
            "not_ground_truth": True,
            "limits": "Stored one-sided observed-return-to-mesh QA; Cam2 residual participated in fitting, not heldout accuracy"}


def run(args):
    started = time.perf_counter()
    output_root = args.output_root.resolve()
    output = _bounded(args.evaluation_dir or output_root / "sequence_review/mask_alignment", output_root, "Evaluation directory")
    _bounded(output, WORKSPACE / "outputs", "Generated evaluation")
    output.mkdir(parents=True, exist_ok=True)
    status = {"schema": "badminton-review-mask-evaluation-status-v1", "status": "running", "stage": "preflight",
              "started_utc": datetime.now(timezone.utc).isoformat(), "output_root": str(output_root),
              "device": args.device, "completed_native_images": 0}

    def progress(stage, **updates):
        status.update(stage=stage, **updates)
        (output / "evaluation_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    progress("preflight")
    try:
        runtime = _runtime(args.device)
        runtime.torch.set_num_threads(args.threads)
        runtime.cv2.setNumThreads(2)
        calibration = runtime.CalibrationSnapshot.load(output_root / "calibration/calibration_snapshot.json")
        manifest_path = (args.manifest or output_root / "sequence_review/manifest.json").resolve()
        manifest = _json(manifest_path)
        progress("preflight", manifest_sha256_at_start=runtime.sha(manifest_path))
        if Path(manifest["output_root"]).resolve() != output_root or manifest["calibration_digest"] != calibration.data["snapshot_digest"]:
            raise ValueError("Review manifest belongs to another output root or calibration")
        selection_digest = runtime.review_selection_digest(manifest)
        if manifest.get("selection_sha256") not in (None, selection_digest):
            raise ValueError("Review manifest stable selection digest differs")
        cases, selection = select_cases(manifest, output_root)
        quality_cache = selection.pop("quality_cache")
        if args.case_limit:
            cases = cases[:args.case_limit]
        selection.update(evaluated_cases=len(cases), case_limit=args.case_limit,
                         limited_smoke_or_subset=args.case_limit is not None,
                         geometry="Only actual current-production fit_path meshes; no benchmark/candidate replacement",
                         reason="Same adopted baseline, actual native dual-camera RGB, fixed sampled production geometry",
                         description_zh="当前新生产输出的真实拟合；固定原标定与原生双相机帧。绿色为模型预测 mask，黄色为原始 LiDAR 点，粉色为新拟合轮廓；不是 GT 或质量通过认证。")
        court_path = args.court_evidence.resolve()
        court = validate_court(_json(court_path), court_path, calibration, runtime)
        model_path = Path(args.model)
        if args.model in ("yolo8xseg", "yolov8x-seg"):
            model_path = WORKSPACE / "inputs/checkpoints/yolo/yolov8x-seg.pt"
        model_path = model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Local segmentation checkpoint missing; no automatic model download: {model_path}")
        device = "0" if args.device == "cuda:0" else args.device
        options = {"imgsz": args.imgsz, "confidence": args.confidence, "device": device, "retina_masks": True, "classes": [0]}
        model_info = {"path": str(model_path), "sha256": runtime.sha(model_path),
                      "ultralytics": __import__("ultralytics").__version__, "options": options}
        observer = MaskObserver(runtime, model_path, model_info, options, output,
                                legacy_mask_index(court_path, model_info, runtime))
        topology = runtime.smplx.create(WORKSPACE / "inputs/checkpoints/body_models", model_type="smpl", gender="neutral")
        faces = np.asarray(topology.faces, dtype=np.int64)
        del topology
        sessions = {}
        groups = defaultdict(lambda: defaultdict(list))
        for case in cases:
            sid = case["session_id"]
            if sid not in sessions:
                sessions[sid] = runtime.SessionIndex.load(Path(calibration.data["sessions_root"]) / sid, calibration.frame_map_path(sid))
            _validate_case(case, output_root, calibration, sessions[sid], quality_cache, runtime)
            for camera in ("cam1", "cam2"):
                groups[(sid, camera)][case["frame"]["cameras"][camera]["record_index"]].append(case)
        binding = {"schema": "production-native-mask-review-binding-v1", "selection_sha256": selection_digest,
                   "evaluated_cases": [{"key": c["key"], "session_id": c["session_id"], "scan_mid_us": c["frame"]["scan_mid_us"],
                                        "native_cameras": c["frame"]["cameras"], "fits": c["fit_provenance"]} for c in cases],
                   "segmentation_model": {k: v for k, v in model_info.items() if k != "options"},
                   "prediction_options_without_device": {k: v for k, v in options.items() if k != "device"},
                   "evaluator_sha256": runtime.sha(__file__), "shared_alignment_evaluator_sha256": runtime.sha(runtime.__file__),
                   "court_evidence_sha256": runtime.sha(court_path)}
        binding_sha = _digest(binding)
        binding_path = output / "evaluation_binding.json"
        if binding_path.exists() and _json(binding_path).get("binding_sha256") != binding_sha:
            raise ValueError("Existing evaluation directory belongs to another selection/model/geometry/code; use an isolated directory")
        runtime.write_json(binding_path, {**binding, "binding_sha256": binding_sha})
        camera_base = {camera: calibration.camera(camera) for camera in ("cam1", "cam2")}
        rows = []
        reports = {c["key"]: {"key": c["key"], "session_id": c["session_id"], "clip_id": c["clip_id"],
                             "scan_mid_us": c["frame"]["scan_mid_us"], "lidar_index": c["frame"]["lidar_index"],
                             "selection_reasons": c["selection_reasons"], "fit_provenance": c["fit_provenance"],
                             "images": {}, "association": {}, "mask_evidence": {}} for c in cases}
        progress("native_masks_and_geometry", planned_native_images=2 * len(cases), selection_sha256=selection_digest)
        for (sid, camera), by_index in groups.items():
            session = sessions[sid]
            decoded = set()
            for indices in _decode_groups(by_index):
                for index, image in session.iter_video_frames(camera, indices):
                    decoded.add(index)
                    native = by_index[index][0]["frame"]["cameras"][camera]
                    (masks, boxes, confidence), evidence = observer.observe(session, camera, native, image)
                    for case in by_index[index]:
                        result, pictures, association = runtime.evaluate_case(
                            case, camera, image, masks, boxes, confidence, {"baseline": camera_base},
                            calibration, faces, {}, output, court)
                        # The shared baseline path already loaded new production meshes.
                        # Give its gallery the existing continuous-fit label without evaluating duplicate geometry.
                        for row in result:
                            row["variant"] = "continuity_baseline"
                            row["geometry_source"] = "current_production_fit_path"
                        rows.extend(result)
                        reports[case["key"]]["images"][camera] = {"continuity_baseline": pictures["baseline"]}
                        reports[case["key"]]["association"][camera] = association
                        reports[case["key"]]["mask_evidence"][camera] = evidence
                        status["completed_native_images"] += 1
                        progress("native_masks_and_geometry", mask_sources=observer.counts.copy(), current_case=case["key"], current_camera=camera)
                        print(f"[review-mask] {case['key']} {camera}: {association['matched']}/{association['baseline_athletes']} matched ({evidence['this_run_source']})", flush=True)
            if decoded != set(by_index):
                raise ValueError("Requested native camera images were not all decoded")
        progress("report_and_gallery")
        current = _json(manifest_path)
        if runtime.review_selection_digest(current) != selection_digest:
            raise ValueError("Review source selection changed during evaluation")
        for case in cases:
            for fit in case["fit_provenance"]:
                if runtime.sha(fit["path"]) != fit["sha256"]:
                    raise ValueError("Production geometry changed during evaluation")
        if args.device == "cpu" and runtime.torch.cuda.is_initialized():
            raise RuntimeError("CPU evaluation unexpectedly initialized CUDA")
        report = {
            "schema": "badminton-production-review-native-mask-evaluation-v1", "status": "complete",
            "quality_gate": "No pass certification: report measured prediction coverage, associations and residuals",
            "calibration_digest": calibration.data["snapshot_digest"], "active_selection_changed": False,
            "manifest": str(manifest_path), "manifest_sha256_at_start": status["manifest_sha256_at_start"],
            "manifest_sha256_at_completion": runtime.sha(manifest_path), "selection_sha256": selection_digest,
            "selection_matches_current_manifest": True, "evaluation_binding_sha256": binding_sha,
            "evaluator_sha256": binding["evaluator_sha256"], "shared_evaluator_sha256": binding["shared_alignment_evaluator_sha256"],
            "selection": selection, "segmentation_model": model_info, "mask_cache_sources": observer.counts,
            "requested_device": args.device, "cuda_context_initialized": runtime.torch.cuda.is_initialized(),
            "native_camera_images": 2 * len(cases), "completed_native_camera_images": status["completed_native_images"],
            "original_image_space": "original_native_distorted_Brown_pixels", "image_interpolation": False,
            "geometry_interpolation": False, "silhouette_rule": "union_of_individually_rasterized_projected_triangles",
            "cases": [reports[c["key"]] for c in cases], "measurements": rows, "mask_alignment": runtime.aggregate(rows),
            "court_evaluation": court, "body_evaluation": _body_summary(cases, runtime),
            "fixed_rgb_checks": {"status": "not_requested", "reason": "This review evaluates native person masks, frozen court and saved production surface QA; no repeated keypoint inference"},
            "inference_and_evaluation_wall_s": time.perf_counter() - started,
            "limits": ["RGB masks are model predictions, not ground truth; retina masks upsample a lower-resolution mask basis",
                       "Associations use current production mesh boxes; ambiguity, small people, net occlusion and truncated limbs can lower coverage",
                       "One-sided mask containment scores all in-view real LiDAR returns, including potentially camera-occluded surfaces",
                       "Sparse-point mask hit rate is not dense silhouette IoU; exact and fixed3px tolerance are both retained",
                       "Court guides are frozen normative geometry; the inherited marked-court evidence is not a new clip-specific manual annotation",
                       "The reviewed and named development events are not a blind heldout dataset; physical exposure synchronization remains unverified"],
        }
        runtime.write_json(output / "joint_alignment.json", report)
        runtime.write_gallery(output, report)
        progress("complete", status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                 gallery=str(output / "index.html"), report=str(output / "joint_alignment.json"),
                 mask_sources=observer.counts.copy())
        return report
    except Exception as error:
        progress(status.get("stage", "unknown"), status="failed", error_type=type(error).__name__, error=str(error),
                 failed_utc=datetime.now(timezone.utc).isoformat(), traceback=traceback.format_exc())
        # A failed rerun must not leave an earlier complete report looking
        # successful to an automated finalizer that reads this conventional file.
        (output / "joint_alignment.json").write_text(json.dumps({
            "schema": "badminton-production-review-native-mask-evaluation-v1",
            "status": "failed", "stage": status["stage"], "error_type": type(error).__name__,
            "error": str(error), "completed_native_camera_images": status["completed_native_images"],
            "quality_gate": "No successful evaluation or quality pass claimed for this run",
        }, indent=2) + "\n", encoding="utf-8")
        (output / "index.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Mask review failed</title><h1>Evaluation failed</h1><pre>'
            + html.escape(json.dumps(status, indent=2)) + '</pre><p>No successful report is claimed for this run.</p>', encoding="utf-8")
        raise


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--court-evidence", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--model", default="yolo8xseg")
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--confidence", type=float, default=.25)
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.imgsz < 32 or args.threads < 1 or (args.case_limit is not None and args.case_limit < 1):
        parser.error("imgsz/threads/case-limit must be positive (imgsz>=32)")
    if not 0 < args.confidence < 1:
        parser.error("confidence must be between0 and1")
    return args


def main(argv=None):
    args = arguments(argv)
    try:
        result = run(args)
    except Exception as error:
        print(f"[review-mask] FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                      "native_camera_images": result["completed_native_camera_images"],
                      "mask_cache_sources": result["mask_cache_sources"], "mask_alignment": result["mask_alignment"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
