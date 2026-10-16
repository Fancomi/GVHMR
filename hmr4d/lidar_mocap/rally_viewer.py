"""Local native-video rally viewer; read-only inputs and HTTP byte-range media.

Playback uses recorded PTS/common-time correspondences. Exact paired-frame mode
seeks each camera to its own verified native PTS before showing fitted geometry.
"""

from __future__ import annotations

from bisect import bisect_left
import base64
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlparse

import numpy as np

from .calibration import CalibrationSnapshot
from .pipeline import write_json
from .rally_pipeline import load_run_session, read_json, read_rows
from .review import EDGES


def packed(array, dtype="<f4"):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode("ascii")


def prepare_viewer(run_root, baseline_root):
    root = Path(run_root).resolve()
    baseline = Path(baseline_root).resolve()
    calibration = CalibrationSnapshot.load(root / "calibration/calibration_snapshot.json")
    registry = read_json(root / "source_registry.json")
    intake = read_json(registry["intake"]["path"])
    rallies = read_rows(intake["prepared_ranges"]["path"])
    sources = {}
    sessions = {}
    media = {}
    for sid, entry in registry["sources"].items():
        session = load_run_session(root, calibration, sid)
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
            )
            for r in rallies
            if r["source_id"] == sid
        ]
        sources[sid] = dict(
            baseline=entry["baseline"],
            common_origin_us=begin,
            duration=(end - begin) / 1e6,
            cameras=cameras,
            rallies=sorted(clips, key=lambda c: c["index"]),
        )
    manifest = dict(
        schema="badminton-native-rally-viewer-v1",
        source_count=len(sources),
        rally_count=len(rallies),
        rally_window_minutes=intake["effective_window_minutes"],
        unique_paired_input_scans=sum(s["paired_unique_png_frames"] for s in intake["sources"].values()),
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
    output = root / "viewer"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    template = Path(__file__).with_name("rally_viewer.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(template, encoding="utf-8")
    return ViewerData(root, baseline, calibration, manifest, sessions, media)


class ViewerData:
    def __init__(self, root, baseline, calibration, manifest, sessions, media):
        self.root = root
        self.baseline = baseline
        self.calibration = calibration
        self.manifest = manifest
        self.sessions = sessions
        self.media = media
        self._quality = {}
        self._lock = threading.Lock()

    def status(self):
        state = read_json(self.root / "build_status.json")
        for sid, source in state["sources"].items():
            directory = (self.baseline if source["baseline"] else self.root) / "sessions" / sid / "windows"
            reports = list(directory.glob("*/fit_report.json"))
            source["available_fit_windows"] = len(reports)
            if not source["baseline"]:
                parsed = [read_json(p) for p in reports]
                source["completed_fit_person_frames"] = sum(r["fitted"] for r in parsed)
                source["completed_fit_qa_pass"] = sum(r["accepted"] for r in parsed)
        paired = self.root.with_name(self.root.name + "_paired") / "datasets/dataset_manifest.json"
        if paired.exists():
            delivered = read_json(paired)
            state["paired_accepted"] = delivered["accepted"]
        return state

    def quality_rows(self, sid):
        directory = (
            (self.baseline if self.manifest["sources"][sid]["baseline"] else self.root) / "sessions" / sid / "windows"
        )
        with self._lock:
            for path in directory.glob("*/quality.jsonl"):
                signature = path.stat().st_mtime_ns, path.stat().st_size
                if path not in self._quality or self._quality[path][0] != signature:
                    # A window is reviewable only after its atomic report exists.
                    if not (path.parent / "fit_report.json").exists():
                        continue
                    indexed = {}
                    for row in read_rows(path):
                        indexed.setdefault(row["lidar_index"], []).append(row)
                    self._quality[path] = signature, indexed
            return {
                i: rows
                for path, (_, indexed) in self._quality.items()
                if path.parent.parent == directory
                for i, rows in indexed.items()
            }

    def overlay(self, sid, relative_seconds):
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
            projected = self.project_fit(row["fit_path"])
            result["athletes"].append(
                dict(
                    motion_id=row["motion_id"],
                    accepted=row["accepted"],
                    mean_mm=row["qa"]["final_p2surface_mean_m"] * 1000,
                    rejection_reasons=row["rejection_reasons"],
                    cameras=projected,
                )
            )
        return result

    def geometry(self, sid, relative_seconds):
        """One original scan with full stored SMPL; no interpolated geometry."""
        result = self.overlay(sid, relative_seconds)
        if not result["native"]:
            return result
        session = self.sessions[sid]
        pair = session.pairs()[result["lidar_index"]]
        cloud = session.read_cloud(pair)
        sampled = cloud[:: max(1, (len(cloud) + 24999) // 25000)]
        rows = self.quality_rows(sid).get(pair["lidar_index"], [])
        athletes = []
        colors = [(255, 157, 48), (79, 163, 255), (128, 224, 134), (216, 132, 226)]
        for k, row in enumerate(rows):
            with np.load(row["fit_path"]) as fit:
                athletes.append(
                    dict(
                        motion_id=row["motion_id"],
                        accepted=row["accepted"],
                        qa=row["qa"],
                        rejection_reasons=row["rejection_reasons"],
                        color=colors[k % len(colors)],
                        vertices=packed(fit["vertices"]),
                        joints=packed(fit["joints"]),
                        cloud=packed(fit["source_cloud"]),
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
            displayed_background_points=len(sampled),
            athletes=athletes,
        )
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

    @lru_cache(maxsize=128)
    def project_fit(self, path):
        result = {}
        with np.load(path) as fit:
            arrays = dict(
                mesh=fit["vertices"],
                joints=fit["joints"],
                cloud=fit["source_cloud"][:: max(1, len(fit["source_cloud"]) // 900)],
            )
        for cam in ("cam1", "cam2"):
            result[cam] = {}
            for name, points in arrays.items():
                uv, depth = self.calibration.project(points, cam, distort=True)
                result[cam][name] = [
                    [round(float(x), 1), round(float(y), 1)] if d > 0 and np.isfinite([x, y]).all() else None
                    for (x, y), d in zip(uv, depth)
                ]
        return result


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
                else json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
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
                    return self.blob((data.root / "viewer/index.html").read_bytes(), "text/html; charset=utf-8", head)
                if path == "/api/manifest":
                    return self.blob(data.manifest, "application/json", head)
                if path == "/viewer.js":
                    return self.blob(
                        Path(__file__).with_name("rally_viewer_client.js").read_bytes(), "application/javascript", head
                    )
                if path == "/api/status":
                    return self.blob(data.status(), "application/json", head)
                if path == "/api/overlay":
                    query = parse_qs(parsed.query)
                    return self.blob(data.overlay(query["source"][0], float(query["t"][0])), "application/json", head)
                if path == "/api/geometry":
                    query = parse_qs(parsed.query)
                    return self.blob(data.geometry(query["source"][0], float(query["t"][0])), "application/json", head)
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
                    asset = Path(__file__).with_name("review_assets") / Path(path).name
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
