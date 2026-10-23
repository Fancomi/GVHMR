"""Local native-video rally viewer; read-only inputs and HTTP byte-range media.

Playback uses recorded PTS/common-time correspondences. Paired mode buffers
complete native RGB/SMPL/point-cloud bundles before their atomic display commit.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import base64
import copy
import hashlib
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qs, urlparse

import numpy as np
import cv2

try:
    import orjson
except ImportError:
    orjson = None

from .calibration import CalibrationSnapshot
from .pipeline import write_json
from .relocations import validate_run_relocations
from .rally_pipeline import load_run_session, read_json, read_rows
from .review import EDGES
from .preview_decoder import PreviewDecoder
from .track_colors import track_color_registry


def packed(array, dtype="<f4"):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode("ascii")


def bbox_rotation_float32(degrees):
    """Eigen AngleAxisf Z*Y*X quaternion composition, retaining Float32 math."""
    r = np.asarray(degrees, dtype=np.float32) * np.float32(np.float32(np.pi) / np.float32(180))
    sx, sy, sz = np.sin(r * np.float32(0.5))
    cx, cy, cz = np.cos(r * np.float32(0.5))
    w0 = cz * cy
    x0 = -sz * sy
    y0 = cz * sy
    z0 = sz * cy
    w = w0 * cx - x0 * sx
    x = w0 * sx + x0 * cx
    y = y0 * cx + z0 * sx
    z = z0 * cx - y0 * sx
    tx = np.float32(2) * x
    ty = np.float32(2) * y
    tz = np.float32(2) * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    return np.array(
        [
            [1 - (tyy + tzz), txy - twz, txz + twy],
            [txy + twz, 1 - (txx + tzz), tyz - twx],
            [txz - twy, tyz + twx, 1 - (txx + tyy)],
        ],
        dtype=np.float32,
    )


def badminton_detection_bbox(bbox):
    """Mirror native badmintonDetectionBboxSet: ground/net Ignore + 5m Region."""
    boxes = bbox.get("boxes", [])
    if len(boxes) == 1 and boxes[0]["type"].lower() == "region":
        return dict(boxes=boxes)
    if len(boxes) < 2:
        return dict(boxes=[])
    boxes = [dict(box, type="Ignore") for box in boxes[:2]]
    ground = boxes[0]
    rotation = bbox_rotation_float32(ground.get("rotation_deg", [0, 0, 0]))
    center = np.asarray(ground["center"], dtype=np.float32)
    size = np.asarray(ground["size"], dtype=np.float32).copy()
    offset = np.array([0, 0, size[2] * np.float32(0.5) + np.float32(2.5)], dtype=np.float32)
    center = center + rotation @ offset
    size[2] = np.float32(5)
    boxes.append(dict(ground, id="region_3", type="Region", center=center.tolist(), size=size.tolist()))
    return dict(boxes=boxes)


def crop_viewer_points(points, bbox):
    """Accepted Region union minus Ignore union, evaluated after exactly one M."""
    points = np.asarray(points, dtype=np.float32)
    regions = []
    ignored = []
    for box in bbox.get("boxes", []):
        if not box.get("enabled", True):
            continue
        kind = box["type"].lower()
        if kind == "region":
            local = (points - np.asarray(box["center"], dtype=np.float32)) @ bbox_rotation_float32(
                box.get("rotation_deg", [0, 0, 0])
            )
            inside = (np.abs(local) <= np.asarray(box["size"], dtype=np.float32) * np.float32(0.5)).all(axis=1)
            regions.append(inside)
        elif kind == "ignore":
            ignored.append(box)
        else:
            raise ValueError("Unsupported accepted bbox type: " + box["type"])
    included = np.logical_or.reduce(regions) if regions else np.zeros(len(points), dtype=bool)
    included &= np.isfinite(points).all(axis=1)
    for box in ignored:
        indices = np.flatnonzero(included)
        if not len(indices):
            break
        local = (points[indices] - np.asarray(box["center"], np.float32)) @ bbox_rotation_float32(
            box.get("rotation_deg", [0, 0, 0])
        )
        inside = (np.abs(local) <= np.asarray(box["size"], np.float32) * np.float32(0.5)).all(axis=1)
        included[indices[inside]] = False
    return included


def apply_display_net_extension(bbox, settings, base_sha256):
    """One display-only positive local-X face extension, with native rounding."""
    expected = {"schema", "sport", "scope", "base_bbox_sha256", "net_ignore_id", "net_positive_local_x_extension_m"}
    if (
        set(settings) != expected
        or settings["schema"] != "badminton-display-crop-v1"
        or settings["sport"] != "badminton"
        or settings["scope"] != "display_only"
        or settings["base_bbox_sha256"] != base_sha256
    ):
        raise ValueError("Display crop schema/scope/base bbox SHA mismatch")
    extension = settings["net_positive_local_x_extension_m"]
    if (
        isinstance(extension, bool)
        or not isinstance(extension, (int, float))
        or not np.isfinite(extension)
        or extension < 0
    ):
        raise ValueError("Display net extension must be a finite nonnegative number")
    result = copy.deepcopy(bbox)
    candidates = [
        b for b in result["boxes"][:2] if b.get("id") == settings["net_ignore_id"] and b["type"].lower() == "ignore"
    ]
    if len(candidates) != 1:
        raise ValueError("Display crop must identify one accepted net Ignore")
    net = candidates[0]
    center = np.asarray(net["center"], np.float64)
    direction = bbox_rotation_float32(net.get("rotation_deg", [0, 0, 0]))[:, 0].astype(np.float64)
    net["center"] = (center + direction * (float(extension) * 0.5)).tolist()
    net["size"][0] = float(net["size"][0]) + float(extension)
    if not np.isfinite(np.asarray(net["center"] + net["size"], np.float32)).all():
        raise ValueError("Display crop geometry exceeds finite Float32")
    return result


def load_display_bbox(calibration, path=None):
    bbox = calibration.data["bbox"]
    if path is None:
        return badminton_detection_bbox(bbox), None
    path = Path(path).resolve()
    blob = path.read_bytes()
    settings = json.loads(blob)
    base_sha = settings.get("base_bbox_sha256")
    artifacts = [
        a
        for a in calibration.data["artifacts"].values()
        if a["sha256"] == base_sha and Path(a["path"]).name == "bbox.json"
    ]
    if len(artifacts) != 1:
        raise ValueError("Display crop does not match this frozen bbox artifact")
    base_blob = (calibration.snapshot_json.parent / artifacts[0]["path"]).read_bytes()
    if hashlib.sha256(base_blob).hexdigest() != base_sha or json.loads(base_blob) != bbox:
        raise ValueError("Display crop base bbox bytes/geometry mismatch")
    adjusted = apply_display_net_extension(bbox, settings, base_sha)
    evidence = dict(
        path=str(path),
        sha256=hashlib.sha256(blob).hexdigest(),
        settings=settings,
        original_inputs_unchanged=True,
        arithmetic="Float64 original center + Float32 native rotation column * Float64 half-extension, then Float32 crop",
    )
    return badminton_detection_bbox(adjusted), evidence


def write_viewer_runtime_relocations(root, calibration, relocations, sessions, output):
    """Write separate runtime evidence; original run and camera artifacts stay frozen."""
    if relocations is None:
        return None
    root, output = Path(root), Path(output)
    registry_path = root / "source_registry.json"
    registry = read_json(registry_path)
    validate_run_relocations(registry, calibration.data, relocations)
    if read_json(calibration.snapshot_json) != calibration.data:
        raise ValueError("Calibration snapshot changed during runtime preparation")

    def binding(path):
        blob = Path(path).read_bytes()
        return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob))

    readers = {}
    for sid, session in sessions.items():
        if getattr(session, "source_index_sha256", None) is not None:
            declared = registry["sources"][sid].get("source_index", {})
            if declared.get("sha256") != session.source_index_sha256:
                raise ValueError("Frozen source registry changed during runtime preparation")
            readers[sid] = dict(
                reader="source_index", source_index_sha256=session.source_index_sha256,
                source_index_path=str(session.source_index_path),
                source_index_logical_path=str(session.source_index_logical_path),
            )
        else:
            readers[sid] = session.relocation_legacy_audit
    receipt = dict(
        schema="gvhmr-viewer-runtime-relocations-v1", data_root=str(relocations.root),
        data_relocations=dict(relocations.binding), source_registry=binding(registry_path),
        calibration_snapshot=binding(calibration.snapshot_json),
        calibration_digest=calibration.data["snapshot_digest"], readers=readers,
        original_registry_modified=False, original_calibration_modified=False, original_data_modified=False,
    )
    path = output / "runtime_relocations.json"
    write_json(path, receipt)
    return binding(path)


def load_viewer_intake(run_root):
    """Read the frozen source scope, deriving formal display counts from real ranges."""
    root = Path(run_root).resolve()
    evidence = []

    def document(path, expected=None, *, rows=False):
        path = Path(path).resolve()
        payload = path.read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        if expected is not None and sha != expected:
            raise ValueError("Frozen viewer intake/ranges SHA256 mismatch")
        evidence.append(dict(path=str(path), sha256=sha))
        text = payload.decode("utf-8-sig")
        return [json.loads(line) for line in text.splitlines() if line] if rows else json.loads(text)

    registry = document(root / "source_registry.json")
    intake = document(registry["intake"]["path"], registry["intake"]["sha256"])
    ranges = intake["prepared_ranges"]
    rallies = document(ranges["path"], ranges["sha256"], rows=True)
    declared = set(registry["sources"])
    if any(row["source_id"] not in declared for row in rallies):
        raise ValueError("Viewer ranges include an undeclared source")
    if registry.get("schema") == "badminton-formal-derived-source-registry-v1":
        if intake.get("schema") != "badminton-formal-derived-intake-v1":
            raise ValueError("Formal viewer requires its derived intake")
        selected = {sid: set() for sid in declared}
        duration_us = 0
        for row in rallies:
            indices = row["eligible_paired_png_indices"]
            if (any(type(index) is not int or index < 0 for index in indices)
                    or len(set(indices)) != len(indices)
                    or len(indices) != row["source_frame_count"]):
                raise ValueError("Formal viewer range has invalid paired scan membership")
            selected[row["source_id"]].update(indices)
            duration = row["last_scan_mid_us"] - row["first_scan_mid_us"]
            if duration < 0:
                raise ValueError("Formal viewer range ends before it begins")
            duration_us += duration
        counts = {sid: len(indices) for sid, indices in selected.items()}
        if len(rallies) != intake["rallies"] or sum(counts.values()) != intake["selected_scans"]:
            raise ValueError("Formal viewer ranges disagree with the frozen intake totals")
        minutes = duration_us / 60e6
    else:
        if set(intake["sources"]) != declared:
            raise ValueError("Viewer registry and intake source coverage differ")
        counts = {sid: item["paired_unique_png_frames"] for sid, item in intake["sources"].items()}
        minutes = intake["effective_window_minutes"]
    empty = {sid for sid, count in counts.items() if count == 0}
    ranged = {row["source_id"] for row in rallies}
    if empty & ranged or declared - empty != ranged:
        raise ValueError("Viewer empty-source declaration disagrees with actual selected ranges")
    return registry, intake, rallies, dict(
        paired_scans=counts, empty_sources=sorted(empty), rally_window_minutes=minutes, evidence=evidence,
    )


def prepare_viewer(run_root, baseline_root, display_crop=None, *, relocations=None, viewer_output=None):
    root = Path(run_root).resolve()
    baseline = Path(baseline_root).resolve()
    calibration = CalibrationSnapshot.load(root / "calibration/calibration_snapshot.json")
    viewer_bbox, display_evidence = load_display_bbox(calibration, display_crop)
    registry, intake, rallies, coverage = load_viewer_intake(root)
    validate_run_relocations(registry, calibration.data, relocations)
    sources = {}
    empty_sources = {}
    sessions = {}
    media = {}
    body_path = root / "datasets/dataset_manifest.json"
    body = read_json(body_path) if root == baseline and body_path.exists() else {}
    if body and body.get("calibration_digest") != calibration.data["snapshot_digest"]:
        raise ValueError("Viewer body labels and source calibration disagree")
    for sid, entry in registry["sources"].items():
        if sid in coverage["empty_sources"]:
            empty_sources[sid] = dict(reason="no_selected_paired_scans", paired_scans=0, fitted_person_frames=0)
            continue
        session = load_run_session(root, calibration, sid, relocations=relocations)
        sessions[sid] = session
        known = {
            cam: [r for r in session.camera_rows(cam) if r["common_time_us"] is not None] for cam in ("cam1", "cam2")
        }
        begin = max(rows[0]["common_time_us"] for rows in known.values())
        end = min(rows[-1]["common_time_us"] for rows in known.values())
        cameras = {}
        for cam, rows in known.items():
            # Sparse *observed* clock anchors only drive the browser playback
            # controller. Exact review uses the full frozen per-scan map.
            anchors = rows[::60]
            if anchors[-1] != rows[-1]:
                anchors.append(rows[-1])
            cameras[cam] = dict(
                url=f"/media/{sid}/{cam}",
                clock_anchors=[
                    [round((r["common_time_us"] - begin) / 1e6, 6), r["media_pts_us"] / 1e6] for r in anchors
                ],
            )
            media[sid, cam] = session.video_paths[cam]
        clips = [
            dict(
                index=r["source_rally_index"],
                review_index=r["review_index"],
                original_rally_index=r["original_rally_index"],
                begin=(r["first_scan_mid_us"] - begin) / 1e6,
                end=(r["last_scan_mid_us"] - begin) / 1e6,
                source_begin_frame=r["source_begin_frame"],
                source_frame_count=r["source_frame_count"],
                **({"segment_index": r["segment_index"]} if "segment_index" in r else {}),
            )
            for r in rallies
            if r["source_id"] == sid
        ]
        sources[sid] = dict(
            baseline=bool(entry["baseline"] or body.get("exports", {}).get(sid, {}).get("accepted", 0)),
            common_origin_us=begin,
            duration=(end - begin) / 1e6,
            cameras=cameras,
            rallies=sorted(clips, key=lambda c: c["index"]),
            scan_times=[round((p["scan_mid_us"] - begin) / 1e6, 6) for p in session.pairs()],
        )
    manifest = dict(
        schema="badminton-native-rally-viewer-v1",
        source_count=len(sources),
        declared_source_count=len(registry["sources"]),
        empty_sources=empty_sources,
        rally_count=len(rallies),
        rally_window_minutes=coverage["rally_window_minutes"],
        unique_paired_input_scans=sum(coverage["paired_scans"].values()),
        sources=sources,
        calibration_digest=calibration.data["snapshot_digest"],
        edges=EDGES,
        geometry="Stored fitted SMPL surface samples/joints and actual segmented LiDAR returns; no interpolation",
        sync="Recorded native PTS/common-clock anchors; exact-pair mode uses frozen per-scan camera PTS",
    )
    import smplx
    from hmr4d import PROJ_ROOT

    topology = smplx.create(PROJ_ROOT / "inputs/checkpoints/body_models", model_type="smpl", gender="neutral")
    manifest["faces"] = packed(np.asarray(topology.faces), "<u2")
    manifest["reference_plane"] = calibration.reference_plane
    manifest["point_filter"] = dict(
        rule="native badmintonDetectionBboxSet: ground-top-derived 5m Region AND NOT ground/net Ignore",
        default="roi",
        available=["roi", "all"],
        boxes=viewer_bbox["boxes"],
        display_crop=display_evidence,
    )
    output = Path(viewer_output).resolve() if viewer_output is not None else root / "viewer"
    output.mkdir(parents=True, exist_ok=True)
    runtime_binding = write_viewer_runtime_relocations(root, calibration, relocations, sessions, output)
    if runtime_binding is not None:
        manifest["runtime_relocations"] = runtime_binding
        manifest["data_relocations"] = dict(relocations.binding)
    write_json(output / "manifest.json", manifest)
    template = Path(__file__).with_name("rally_viewer.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(template, encoding="utf-8")
    (output / "viewer.js").write_bytes(Path(__file__).with_name("rally_viewer_client.js").read_bytes())
    for name in ("review.js", "review.css"):
        (output / name).write_bytes((Path(__file__).with_name("review_assets") / name).read_bytes())
    return ViewerData(root, baseline, calibration, manifest, sessions, media, viewer_bbox=viewer_bbox, viewer_output=output)


class ViewerData:
    def __init__(self, root, baseline, calibration, manifest, sessions, media, viewer_bbox=None, *, viewer_output=None):
        self.root = root
        self.baseline = baseline
        self.calibration = calibration
        self.manifest = manifest
        self.sessions = sessions
        self.media = media
        self.viewer_output = Path(viewer_output) if viewer_output is not None else (root / "viewer" if root else None)
        self._quality = {}
        self._track_colors = {}
        self._color_signatures = {}
        dataset = root / "datasets/dataset_manifest.json" if root is not None else None
        self._selected_windows = read_json(dataset).get("selected_windows", {}) if dataset and dataset.exists() else {}
        self._lock = threading.Lock()
        self.viewer_bbox = (
            viewer_bbox
            if viewer_bbox is not None
            else (badminton_detection_bbox(calibration.data["bbox"]) if calibration is not None else dict(boxes=[]))
        )
        self._native_images = OrderedDict()
        self._native_lock = threading.RLock()
        self.preview_decoder = PreviewDecoder()
        self._preview_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="native_rgb")
        self._geometry_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="native_geometry")

    def status(self):
        status_path = self.root / "build_status.json"
        generated_status = self.viewer_output / "build_status.json"
        if generated_status.exists():
            status_path = generated_status
        publication_fallback = not status_path.exists() and self.manifest.get("canonical_publication") is not None
        if publication_fallback:
            binding = self.manifest["canonical_publication"]
            blob = Path(binding["path"]).read_bytes()
            if hashlib.sha256(blob).hexdigest() != binding["sha256"]:
                raise ValueError("Published viewer status manifest SHA256 mismatch")
            publication = json.loads(blob)
            state = dict(
                state="complete", status_source="canonical_publication", paired_accepted=publication["accepted"],
                sources={
                    sid: dict(baseline=source["baseline"], state="complete", rallies=len(source["rallies"]),
                              accepted=publication["exports"][sid]["accepted"])
                    for sid, source in self.manifest["sources"].items()
                },
            )
        else:
            state = read_json(status_path)
        for sid, source in state["sources"].items():
            directory = (self.baseline if source["baseline"] else self.root) / "sessions" / sid / "windows"
            reports = list(directory.glob("*/fit_report.json"))
            source["available_fit_windows"] = len(reports)
            if not source["baseline"] or publication_fallback:
                parsed = [read_json(p) for p in reports]
                source["completed_fit_person_frames"] = sum(r["fitted"] for r in parsed)
                source["completed_fit_qa_pass"] = sum(r["accepted"] for r in parsed)
        paired = self.root.with_name(self.root.name + "_paired") / "datasets/dataset_manifest.json"
        if paired.exists() and not publication_fallback:
            delivered = read_json(paired)
            state["paired_accepted"] = delivered["accepted"]
        state["preview_decoder"] = self.preview_decoder.diagnostics()
        return state

    def quality_rows(self, sid):
        directory = (
            (self.baseline if self.manifest["sources"][sid]["baseline"] else self.root) / "sessions" / sid / "windows"
        )
        with self._lock:
            for path in directory.glob("*/quality.jsonl"):
                if sid in self._selected_windows and path.parent.name not in self._selected_windows[sid]:
                    continue
                signature = path.stat().st_mtime_ns, path.stat().st_size
                if path not in self._quality or self._quality[path][0] != signature:
                    # A window is reviewable only after its atomic report exists.
                    if not (path.parent / "fit_report.json").exists():
                        continue
                    indexed = {}
                    for row in read_rows(path):
                        indexed.setdefault(row["lidar_index"], []).append(row)
                    self._quality[path] = signature, indexed
            frames = {
                i: rows
                for path, (_, indexed) in self._quality.items()
                if path.parent.parent == directory
                for i, rows in indexed.items()
            }
            signature = tuple(
                (str(path), item[0]) for path, item in self._quality.items() if path.parent.parent == directory
            )
            if self._color_signatures.get(sid) != signature:
                self._track_colors.update(track_color_registry(frames))
                self._color_signatures[sid] = signature
            return frames

    def overlay(self, sid, relative_seconds, scope="roi"):
        source = self.manifest["sources"][sid]
        session = self.sessions[sid]
        target = round(relative_seconds * 1e6) + source["common_origin_us"]
        pairs = session.pairs()
        times = [p["scan_mid_us"] for p in pairs]
        index = min(bisect_left(times, target), len(pairs) - 1)
        if index and abs(times[index - 1] - target) <= abs(times[index] - target):
            index -= 1
        pair = pairs[index]
        rows = self.quality_rows(sid).get(pair["lidar_index"], [])
        native = {
            cam: dict(pts=p["media_pts_us"] / 1e6, record_index=p["record_index"], common_time_us=p["common_time_us"])
            for cam, p in pair["cameras"].items()
            if p["status"] == "matched"
        }
        if set(native) != {"cam1", "cam2"}:
            return dict(state="unpaired", athletes=[], native={}, lidar_index=index)
        result = dict(
            state="available" if rows else "pending_or_outside_rally",
            athletes=[],
            native=native,
            lidar_index=index,
            scan_mid_us=pair["scan_mid_us"],
            delta_ms=(pair["scan_mid_us"] - target) / 1000,
        )
        for row in rows:
            projected = self.project_fit(row["fit_path"], scope)
            result["athletes"].append(
                dict(
                    motion_id=row["motion_id"],
                    color=self._track_colors[row["motion_id"]],
                    accepted=row["accepted"],
                    mean_mm=row["qa"]["final_p2surface_mean_m"] * 1000,
                    rejection_reasons=row["rejection_reasons"],
                    cameras=projected,
                )
            )
        return result

    def geometry(self, sid, relative_seconds, scope="roi"):
        """One original scan with full stored SMPL; no interpolated geometry."""
        if scope not in ("roi", "all"):
            raise ValueError("Unknown point display scope")
        result = self.overlay(sid, relative_seconds, scope)
        if not result["native"]:
            return result
        session = self.sessions[sid]
        pair = session.pairs()[result["lidar_index"]]
        cloud = session.read_cloud(pair)
        raw_count = len(cloud)
        if scope == "roi":
            cloud = cloud[crop_viewer_points(cloud, self.viewer_bbox)]
        sampled = cloud[:: max(1, (len(cloud) + 24999) // 25000)]
        rows = self.quality_rows(sid).get(pair["lidar_index"], [])
        athletes = []
        for row in rows:
            asset = self.render_fit(row["fit_path"], scope)
            athletes.append(
                dict(
                    motion_id=row["motion_id"],
                    accepted=row["accepted"],
                    qa=row["qa"],
                    rejection_reasons=row["rejection_reasons"],
                    color=self._track_colors[row["motion_id"]],
                    vertices=asset["vertices"],
                    joints=asset["joints"],
                    cloud=asset["cloud"],
                )
            )
        result["frame"] = dict(
            session_id=sid,
            scan_mid_us=pair["scan_mid_us"],
            lidar_index=pair["lidar_index"],
            cameras=pair["cameras"],
            offset_s=(pair["scan_mid_us"] - self.manifest["sources"][sid]["common_origin_us"]) / 1e6,
            cloud=packed(sampled),
            point_count=len(cloud),
            raw_point_count=raw_count,
            point_scope=scope,
            displayed_background_points=len(sampled),
            athletes=athletes,
        )
        result["native_images"] = {
            cam: self.native_image(sid, cam, pair["cameras"][cam]["record_index"]) for cam in ("cam1", "cam2")
        }
        result["background_projection"] = {}
        for cam in ("cam1", "cam2"):
            uv, depth = self.calibration.project(sampled, cam, distort=True)
            valid = (
                (depth > 0)
                & np.isfinite(uv).all(1)
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < 1920)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < 1080)
            )
            result["background_projection"][cam] = np.round(uv[valid], 1).tolist()
        return result

    def native_image(self, sid, camera, record_index):
        key = (sid, camera, record_index)
        with self._native_lock:
            if key in self._native_images:
                self._native_images.move_to_end(key)
                return self._native_images[key]
        session = self.sessions[sid]
        if hasattr(session, "video_paths"):
            image = next(self.preview_decoder.frames(session, camera, [record_index]))[1]
        else:
            image = session.read_image(camera, record_index)
        return self._cache_native_image(sid, camera, record_index, image)

    def _cache_native_image(self, sid, camera, record_index, image):
        session = self.sessions[sid]
        native = session.camera_rows(camera)[record_index]
        # PreviewDecoder checks actual container PTS against SessionIndex rows;
        # its raw pixels are checked against the fitting/offline decoder.
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise IOError("Cannot encode verified native video frame")
        blob = encoded.tobytes()
        value = dict(
            record_index=record_index,
            media_pts_us=native["media_pts_us"],
            common_time_us=native["common_time_us"],
            width=image.shape[1],
            height=image.shape[0],
            raw_bgr_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            jpeg_sha256=hashlib.sha256(blob).hexdigest(),
            jpeg_quality=85,
            jpeg_base64=base64.b64encode(blob).decode("ascii"),
        )
        with self._native_lock:
            self._native_images[sid, camera, record_index] = value
            while len(self._native_images) > 128:
                self._native_images.popitem(last=False)
        return value

    def geometry_batch(self, sid, relative_seconds, count=12, scope="roi"):
        """Decode each camera once across consecutive native scan bundles."""
        if not 1 <= count <= 16:
            raise ValueError("Batch count must be between 1 and 16")
        started = time.perf_counter()
        session = self.sessions[sid]
        pairs = session.pairs()
        target = round(relative_seconds * 1e6) + self.manifest["sources"][sid]["common_origin_us"]
        times = [p["scan_mid_us"] for p in pairs]
        index = min(bisect_left(times, target), len(pairs) - 1)
        if index and abs(times[index - 1] - target) <= abs(times[index] - target):
            index -= 1
        selected = pairs[index : index + count]

        def fill_camera(camera):
            with self._native_lock:
                missing = sorted(
                    {
                        p["cameras"][camera]["record_index"]
                        for p in selected
                        if p["cameras"][camera]["status"] == "matched"
                        and (sid, camera, p["cameras"][camera]["record_index"]) not in self._native_images
                    }
                )
            stream = (
                self.preview_decoder.frames(session, camera, missing)
                if hasattr(session, "video_paths")
                else session.iter_video_frames(camera, missing)
            )
            for native_index, image in stream:
                self._cache_native_image(sid, camera, native_index, image)

        futures = [self._preview_pool.submit(fill_camera, camera) for camera in ("cam1", "cam2")]
        for future in futures:
            future.result()
        decoded_at = time.perf_counter()
        times = [(pair["scan_mid_us"] - self.manifest["sources"][sid]["common_origin_us"]) / 1e6 for pair in selected]
        futures = [self._geometry_pool.submit(self.geometry, sid, t, scope) for t in times]
        frames = [future.result() for future in futures]
        return dict(
            source_id=sid,
            scope=scope,
            frames=frames,
            count=len(frames),
            begin_index=index,
            timing_seconds=dict(
                native_decode=decoded_at - started,
                geometry=time.perf_counter() - decoded_at,
                total=time.perf_counter() - started,
            ),
            next_time=(
                (pairs[index + count]["scan_mid_us"] - self.manifest["sources"][sid]["common_origin_us"]) / 1e6
                if index + count < len(pairs)
                else None
            ),
        )

    def project_fit(self, path, scope="roi"):
        return self.render_fit(path, scope)["cameras"]

    @lru_cache(maxsize=128)
    def render_fit(self, path, scope="roi"):
        result = {}
        with np.load(path) as fit:
            arrays = dict(
                mesh=fit["vertices"],
                joints=fit["joints"],
                cloud=fit["source_cloud"],
            )
        if scope == "roi":
            arrays["cloud"] = arrays["cloud"][crop_viewer_points(arrays["cloud"], self.viewer_bbox)]
        asset = dict(vertices=packed(arrays["mesh"]), joints=packed(arrays["joints"]), cloud=packed(arrays["cloud"]))
        arrays["cloud"] = arrays["cloud"][:: max(1, len(arrays["cloud"]) // 900)]
        for cam in ("cam1", "cam2"):
            result[cam] = {}
            for name, points in arrays.items():
                uv, depth = self.calibration.project(points, cam, distort=True)
                valid = (depth > 0) & np.isfinite(uv).all(axis=1)
                if name == "mesh":
                    xy = np.round(uv, 1).astype(np.float32)
                    xy[~valid] = -10000
                    result[cam]["mesh_xy_f32"] = packed(xy)
                    result[cam]["mesh_valid_u8"] = packed(valid, np.uint8)
                    result[cam]["mesh_vertex_count"] = len(xy)
                    continue
                values = np.round(uv, 1).tolist()
                for i in np.flatnonzero(~valid):
                    values[int(i)] = None
                result[cam][name] = values
        asset["cameras"] = result
        return asset


def serve_viewer(data, host="127.0.0.1", port=8767):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.respond(head=True)

        def do_GET(self):
            self.respond()

        def blob(self, value, mime, head=False):
            blob = (
                value
                if isinstance(value, bytes)
                else (
                    orjson.dumps(value)
                    if orjson is not None
                    else json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
                )
            )
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(blob)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if not head:
                self.wfile.write(blob)

        def respond(self, head=False):
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                if path in ("/", "/index.html"):
                    return self.blob((data.viewer_output / "index.html").read_bytes(), "text/html; charset=utf-8", head)
                if path == "/api/manifest":
                    return self.blob(data.manifest, "application/json", head)
                if path == "/viewer.js":
                    return self.blob((data.viewer_output / "viewer.js").read_bytes(), "application/javascript", head)
                if path == '/projection.js':
                    return self.blob((Path(__file__).with_name('review_assets')/'projection.js').read_bytes(), 'application/javascript', head)
                if path == "/api/status":
                    return self.blob(data.status(), "application/json", head)
                if path == "/api/overlay":
                    query = parse_qs(parsed.query)
                    return self.blob(data.overlay(query["source"][0], float(query["t"][0])), "application/json", head)
                if path == "/api/geometry":
                    query = parse_qs(parsed.query)
                    return self.blob(
                        data.geometry(query["source"][0], float(query["t"][0]), query.get("scope", ["roi"])[0]),
                        "application/json",
                        head,
                    )
                if path == "/api/bundles":
                    query = parse_qs(parsed.query)
                    return self.blob(
                        data.geometry_batch(
                            query["source"][0],
                            float(query["t"][0]),
                            int(query.get("count", ["12"])[0]),
                            query.get("scope", ["roi"])[0],
                        ),
                        "application/json",
                        head,
                    )
                if path in ("/scene/", "/scene/index.html"):
                    m = dict(
                        data.manifest,
                        stream_mode=True,
                        clip_count=1,
                        clip_duration_s=0,
                        clips=[
                            dict(
                                id="live",
                                session_id="live",
                                duration_s=0,
                                summary=dict(scans=0, fitted_person_frames=0, accepted=0, mean_surface_mm=None),
                            )
                        ],
                    )
                    assets = Path(__file__).with_name("review_assets")
                    html = (
                        (assets / "index.html")
                        .read_text(encoding="utf-8")
                        .replace("__MANIFEST__", json.dumps(m, separators=(",", ":")))
                        .replace("<body>", '<body class="stream">')
                    )
                    html = html.replace('id="background" type="checkbox"', 'id="background" type="checkbox" checked')
                    return self.blob(html.encode("utf-8"), "text/html; charset=utf-8", head)
                if path in ("/scene/review.js", "/scene/review.css"):
                    asset = data.viewer_output / Path(path).name
                    return self.blob(
                        asset.read_bytes(), "application/javascript" if asset.suffix == ".js" else "text/css", head
                    )
                match = re.fullmatch(r"/media/([^/]+)/(cam[12])", path)
                if match:
                    return self.media(data.media[match.group(1), match.group(2)], head)
                self.send_error(404)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except (KeyError, ValueError, IndexError):
                self.send_error(400)

        def media(self, path, head):
            total = path.stat().st_size
            start = 0
            end = total - 1
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
                if not match or not any(match.groups()):
                    return self.send_error(416)
                left, right = match.groups()
                if left:
                    start = int(left)
                    end = min(int(right), total - 1) if right else total - 1
                else:
                    start = max(0, total - int(right))
                if start >= total or end < start:
                    return self.send_error(416)
            self.send_response(206 if range_header else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if range_header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
            self.end_headers()
            if head:
                return
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = handle.read(min(1 << 18, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Viewer: http://{host}:{server.server_port}/", flush=True)
    server.serve_forever()
