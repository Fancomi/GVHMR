"""Read-only video/LiDAR indexing on the recorded device common clock.

The PNG grid contains raw sensor coordinates. The manifest rotation is applied
only when a caller explicitly requests viewer coordinates. Neither media FPS
nor host receive timestamps are used to construct a pair.
"""

from __future__ import annotations

import bisect
import json
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np


MAX_CAMERA_ERROR_US = 8500


def _read_json(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _integer(value, name: str):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer device timestamp/index")
    return value


def _local_file(directory: Path, name: str) -> Path:
    path = (directory / name).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"Source path escapes its recording directory: {name}")
    return path


def decode_fixed24(path, point_grid=None) -> np.ndarray:
    """Decode raw XYZ in metres, retaining only cells with three valid axes.

    ``point_grid`` is a manifest mapping or ``(point_width, point_height)``.
    Without it, the width and three equal-height bands are read from the PNG.
    OpenCV reads BGR; PNG RGB bytes mean high, middle, low, respectively.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read fixed24 PNG: {path}")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("fixed24 requires an RGB8 PNG with exactly three channels")
    if point_grid is None:
        if image.shape[0] % 3:
            raise ValueError("fixed24 image height must contain three equal XYZ bands")
        width, height = image.shape[1], image.shape[0] // 3
    elif isinstance(point_grid, Mapping):
        width, height = point_grid["point_width"], point_grid["point_height"]
        if float(point_grid.get("scale", 1000)) != 1000 or float(point_grid.get("center", 256)) != 256:
            raise ValueError("This recording contract requires fixed24 scale=1000 and center=256")
    else:
        width, height = point_grid
    if (isinstance(width, bool) or isinstance(height, bool)
            or not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer))
            or width <= 0 or height <= 0 or image.shape[:2] != (height * 3, width)):
        raise ValueError("fixed24 PNG dimensions do not match the authoritative point grid")
    channels = image.astype(np.uint32)
    encoded = (channels[:, :, 2] << 16) | (channels[:, :, 1] << 8) | channels[:, :, 0]
    encoded = encoded.reshape(3, height, width)
    valid = (encoded != 0).all(axis=0)
    return encoded[:, valid].T.astype(np.float32) / np.float32(1000) - np.float32(256)


def iter_native_video_frames(video_path, rows):
    """Seek to the first real media PTS and yield ``(position, row, RGB)``.

    Source sidecars bind ``record_index`` to ``media_pts_us``. This decoder
    independently verifies each requested container timestamp within 2 us.
    Seeking starts at the preceding keyframe; no FPS arithmetic or endpoint
    substitution is used to identify an image.
    """
    import av

    rows = list(rows)
    if not rows:
        return
    for key in ("record_index", "media_pts_us"):
        values = [row.get(key) for row in rows]
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
               or value < 0 for value in values):
            raise ValueError(f"Native video {key} must contain nonnegative integer source identities")
        if any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError(f"Native video {key} must be strictly increasing")
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"Source contains no video stream: {video_path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if stream.time_base is None or stream.time_base <= 0:
            raise ValueError("Source video has no valid media time base")
        first_pts = Fraction(int(rows[0]["media_pts_us"]), 1_000_000) / stream.time_base
        container.seek(int(first_pts), stream=stream, backward=True, any_frame=False)
        wanted = 0
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Source video contains a frame without a media timestamp")
            actual_us = round(frame.pts * frame.time_base * 1_000_000)
            target_us = int(rows[wanted]["media_pts_us"])
            if actual_us < target_us - 2:
                continue
            if abs(actual_us - target_us) > 2:
                raise ValueError(f"Video PTS missing/mismatched at decode index "
                                 f"{rows[wanted]['record_index']}: {actual_us} vs {target_us} us")
            yield wanted, rows[wanted], frame.to_ndarray(format="rgb24")
            wanted += 1
            if wanted == len(rows):
                return
        raise ValueError(f"Source video ended before requested frame {rows[wanted]['record_index']}")


class SessionIndex:
    """Source frame indices plus explicit clock evidence for one recording.

    Pair records retain every source LiDAR frame. ``status == 'paired'`` means
    nonempty foreground, a scan midpoint in the common overlap, and two nearby
    video frames. It does not assert verified exposure synchronization.
    """

    @classmethod
    def load(cls, session_dir, frozen_frame_map_path=None):
        self = cls()
        self.session_dir = Path(session_dir).resolve()
        self.session_id = self.session_dir.name
        self.lidar_dir = self.session_dir / "output-lidar" / f"lidar_{self.session_id}"
        self.manifest = _read_json(self.lidar_dir / "manifest.json")
        self.video_dir = self.session_dir / "output-zcam" / f"zcam_{self.session_id}"
        self.video_paths = {name: self.video_dir / f"{name}.mp4" for name in ("cam1", "cam2")}
        self.Q = np.asarray(self.manifest["lidar_rotation"], dtype=np.float64).reshape(3, 3)
        if not np.isfinite(self.Q).all() or not np.allclose(self.Q @ self.Q.T, np.eye(3), atol=2e-4):
            raise ValueError("Manifest lidar_rotation is not a finite raw-to-viewer rotation")
        if np.linalg.det(self.Q) <= 0:
            raise ValueError("Manifest lidar_rotation must preserve handedness")
        if self.manifest.get("coordinate_encoding") != "fixed24":
            raise ValueError("Session requires the fixed24 source contract")
        if self.manifest.get("coordinate_space", "lidar_sensor_xyz") != "lidar_sensor_xyz":
            raise ValueError("Source PNG coordinates must be raw lidar_sensor_xyz")
        self.meta_dir = self.session_dir / "meta" / self.session_id
        clock_path = self.meta_dir / f"{self.session_id}_common_clock.json"
        self.common_clock = _read_json(clock_path) if clock_path.exists() else {}
        if self.common_clock and self.common_clock.get("schema") != "external_gm_common_timeline_v1":
            raise ValueError("Unsupported common-clock evidence schema")
        if self.common_clock.get("session_id", self.session_id) != self.session_id:
            raise ValueError("Common-clock session identity mismatch")
        self._tracks = {}
        for track in self.common_clock.get("tracks", []):
            name = track["device_id"]
            if name in self._tracks:
                raise ValueError(f"Duplicate common-clock track: {name}")
            self._tracks[name] = track
        self._camera_rows = {name: self._load_camera(name) for name in self.video_paths}
        self._camera_known = {}
        self._camera_times = {}
        for name, rows in self._camera_rows.items():
            known = [row for row in rows if row["common_time_us"] is not None]
            times = [row["common_time_us"] for row in known]
            if any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError(f"Non-monotonic device common clock for {name}")
            self._camera_known[name], self._camera_times[name] = known, times
        self._lidar_clock = {}
        for row in self._tracks.get("lidar1", {}).get("rows", []):
            frame_id = _integer(row.get("frame_id"), "lidar frame_id")
            if frame_id is None or frame_id in self._lidar_clock:
                raise ValueError("Missing/duplicate frame_id in the LiDAR common clock")
            self._lidar_clock[frame_id] = row
        self._frozen = self._load_frozen_map(frozen_frame_map_path) if frozen_frame_map_path else None
        self._pairs = self._build_pairs()
        return self

    def _load_camera(self, name):
        clock_rows = self._tracks.get(name, {}).get("rows", [])
        by_pts = {}
        for row in clock_rows:
            pts = _integer(row.get("media_pts_us"), f"{name} media_pts_us")
            if pts is None or pts in by_pts:
                raise ValueError(f"Missing/duplicate media PTS in {name} common clock")
            by_pts[pts] = row
        companion_path = self.video_dir / f"{name}.frames.jsonl"
        if companion_path.exists():
            source_rows = _read_jsonl(companion_path)
            source = []
            for i, row in enumerate(source_rows):
                if row.get("record_index") != i:
                    raise ValueError(f"{name} record_index must be the verified MP4 decode index")
                if row.get("synthesized") or row.get("discontinuity") or row.get("connection_segment", 0) != 0:
                    raise ValueError(f"{name} video contains synthesized frames or a clock discontinuity")
                numerator = row.get("media_time_base_num", 1)
                denominator = row.get("media_time_base_den", 1000000)
                if denominator <= 0 or (int(row["media_pts"]) * numerator * 1000000) % denominator:
                    raise ValueError(f"{name} media PTS does not have an exact microsecond representation")
                pts = int(row["media_pts"]) * numerator * 1000000 // denominator
                source.append((i, pts, row.get("ssp_frame_no")))
        else:
            raw_name = self._tracks.get(name, {}).get("raw_index", f"{self.session_id}_{name}_video_frames.jsonl")
            raw_path = _local_file(self.meta_dir, raw_name)
            if not raw_path.exists():
                if clock_rows:
                    raise ValueError(f"{name} has clock rows but lacks a source decode-index sidecar")
                return []
            source = [(i, _integer(row.get("container_pts_us"), "container_pts_us"), row.get("frame_no"))
                      for i, row in enumerate(_read_jsonl(raw_path))]
        rows, seen_pts = [], set()
        for record_index, pts, source_no in source:
            if pts is None or pts in seen_pts:
                raise ValueError(f"Missing/duplicate source PTS for {name}")
            seen_pts.add(pts)
            evidence = by_pts.get(pts, {})
            rows.append({"record_index": record_index,
                         "common_time_us": _integer(evidence.get("common_time_us"), "common_time_us"),
                         "media_pts_us": pts,
                         "source_frame_no": evidence.get("frame_no", source_no)})
        if set(by_pts) - seen_pts:
            raise ValueError(f"{name} common-clock media PTS is absent from the source MP4 index")
        if any(b["media_pts_us"] <= a["media_pts_us"] for a, b in zip(rows, rows[1:])):
            raise ValueError(f"Non-monotonic source media PTS for {name}")
        return rows

    def _load_frozen_map(self, path):
        path = Path(path)
        data = _read_jsonl(path) if path.suffix.lower() == ".jsonl" else _read_json(path)
        if isinstance(data, dict):
            data = data.get("rows", data.get("frames"))
        if not isinstance(data, list):
            raise ValueError("Frozen frame map must be an array or JSONL of source scan records")
        result = {}
        for row in data:
            index = _integer(row.get("lidar_frame"), "frozen lidar_frame")
            if index is None or index in result:
                raise ValueError("Missing/duplicate source LiDAR index in frozen frame map")
            result[index] = row
        if set(result) != set(range(self.manifest["frame_count"])):
            raise ValueError("Frozen frame map must cover every source LiDAR frame exactly once")
        return result

    def camera_rows(self, name):
        return self._camera_rows[name]

    def teacher_rows(self, camera, stride=2, start_us=None, end_us=None):
        """Select original frames; returned timestamps preserve their true cadence."""
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError("Teacher stride must be a positive integer")
        return [row for row in self.camera_rows(camera) if row["record_index"] % stride == 0
                and row["common_time_us"] is not None
                and (start_us is None or row["common_time_us"] >= start_us)
                and (end_us is None or row["common_time_us"] <= end_us)]

    def _nearest_camera(self, name, time_us):
        times = self._camera_times[name]
        empty = {"record_index": None, "source_frame_no": None, "media_pts_us": None,
                 "common_time_us": None, "time_error_us": None}
        if not times:
            return {**empty, "status": "missing_clock"}
        if time_us is None:
            return {**empty, "status": "missing_lidar_clock"}
        if time_us < times[0] or time_us > times[-1]:
            return {**empty, "status": "outside_video_time_range"}
        at = bisect.bisect_left(times, time_us)
        candidates = [i for i in (at - 1, at) if 0 <= i < len(times)]
        # Ties select the earlier exposure deterministically.
        at = min(candidates, key=lambda i: (abs(times[i] - time_us), times[i]))
        row = self._camera_known[name][at]
        error = row["common_time_us"] - time_us
        return {**row, "time_error_us": error,
                "status": "matched" if abs(error) <= MAX_CAMERA_ERROR_US else "time_error_exceeds_tolerance"}

    def _frozen_camera(self, camera, scan_index, mid):
        frozen = self._frozen[scan_index].get("cameras", {}).get(camera)
        if frozen is None:
            raise ValueError(f"Frozen frame map lacks camera {camera} at scan {scan_index}")
        nearest = self._nearest_camera(camera, mid)
        frozen_index = frozen.get("record_index")
        if frozen_index is None:
            if frozen.get("status") == "matched" or nearest["record_index"] is not None:
                raise ValueError("Frozen unmatched camera row disagrees with the common-clock nearest frame")
            for key in ("media_pts_us", "video_common_us", "time_error_us"):
                if frozen.get(key) is not None:
                    raise ValueError("Frozen unmatched camera row contains fabricated timing")
            return nearest
        index = _integer(frozen_index, "frozen record_index")
        rows = self._camera_rows[camera]
        if index < 0 or index >= len(rows):
            raise ValueError("Frozen camera decode index is outside source video rows")
        row = rows[index]
        if row["common_time_us"] is None or mid is None:
            raise ValueError("Frozen camera match lacks recorded common-clock evidence")
        expected = {"media_pts_us": row["media_pts_us"], "video_common_us": row["common_time_us"],
                    "time_error_us": row["common_time_us"] - mid}
        if any(frozen.get(key) != value for key, value in expected.items()):
            raise ValueError("Frozen camera timing disagrees with the source common clock")
        if nearest["record_index"] != index:
            raise ValueError("Frozen camera match is not the nearest recorded exposure")
        return nearest

    def _build_pairs(self):
        count = _integer(self.manifest.get("frame_count"), "frame_count")
        frames = self.manifest.get("frames", [])
        if count is None or len(frames) != count:
            raise ValueError("Source manifest frame records must cover frame_count")
        captures = self.manifest.get("frame_capture_timestamps")
        if captures is not None and len(captures) != count:
            raise ValueError("Source capture evidence length does not match frame_count")
        begin = _integer(self.common_clock.get("common_overlap_begin_us"), "common_overlap_begin_us")
        end = _integer(self.common_clock.get("common_overlap_end_us"), "common_overlap_end_us")
        if begin is not None and end is not None and end < begin:
            raise ValueError("Common overlap ends before it starts")
        pairs, seen_ids, previous_start = [], set(), None
        for i, frame in enumerate(frames):
            frame_id = _integer(frame.get("frame_id"), "manifest frame_id")
            if frame_id is None or frame_id in seen_ids:
                raise ValueError("Missing/duplicate source frame_id in manifest")
            seen_ids.add(frame_id)
            clock = self._lidar_clock.get(frame_id, {})
            start = _integer(clock.get("common_time_us"), "LiDAR scan start")
            end_scan = _integer(clock.get("end_common_time_us"), "LiDAR scan end")
            if start is not None and end_scan is not None:
                if end_scan < start or (previous_start is not None and start <= previous_start):
                    raise ValueError("LiDAR common-clock scans are reversed or non-monotonic")
                previous_start = start
                if captures is not None and (captures[i].get("device_first_packet_us") != start
                                             or captures[i].get("device_last_packet_us") != end_scan):
                    raise ValueError("Manifest LiDAR device timestamps disagree with the common clock")
                mid, duration = start + (end_scan - start) // 2, end_scan - start
            else:
                mid, duration = None, None
            if self._frozen is not None:
                frozen = self._frozen[i]
                expected = {"device_first_us": start, "device_last_us": end_scan, "scan_mid_us": mid}
                if mid is None or any(frozen.get(key) != value for key, value in expected.items()):
                    raise ValueError("Frozen LiDAR scan timing disagrees with the source common clock")
            cameras = {name: self._frozen_camera(name, i, mid) if self._frozen is not None
                       else self._nearest_camera(name, mid) for name in self.video_paths}
            reasons = []
            if mid is None:
                reasons.append("missing_lidar_clock")
            if begin is None or end is None:
                reasons.append("missing_common_overlap")
            elif mid is not None and not begin <= mid <= end:
                reasons.append("outside_common_overlap")
            kept = _integer(frame.get("kept_points"), "kept_points")
            if kept is None:
                reasons.append("missing_foreground_count")
            elif kept <= 0:
                reasons.append("empty_foreground")
            for name, camera in cameras.items():
                if camera["status"] != "matched":
                    reasons.append(f"{name}_{camera['status']}")
            if self.common_clock.get("synchronization_lost_during_capture"):
                reasons.append("synchronization_lost_during_capture")
            quality = "device_reported_common_clock_unverified_exposure"
            if not self.common_clock:
                quality = "missing_common_clock"
            elif (self.common_clock.get("physical_accuracy_verified")
                  and self.common_clock.get("exposure_offset_calibrated")):
                quality = "device_reported_common_clock_verified_exposure"
            pairs.append({"session_id": self.session_id, "lidar_index": i, "lidar_frame_id": frame_id,
                          "lidar_path": str(_local_file(self.lidar_dir, frame["file"])),
                          "scan_start_us": start, "scan_end_us": end_scan, "scan_mid_us": mid,
                          "scan_duration_us": duration, "kept_points": kept, "cameras": cameras,
                          "status": "excluded" if reasons else "paired", "reasons": reasons,
                          "clock_quality": quality})
        if set(self._lidar_clock) - seen_ids:
            raise ValueError("Common-clock LiDAR frame IDs are absent from the source manifest")
        return pairs

    def pairs(self):
        return self._pairs

    def read_cloud(self, pair, viewer=True) -> np.ndarray:
        points = decode_fixed24(pair["lidar_path"], self.manifest)
        if pair.get("kept_points") is not None and len(points) != pair["kept_points"]:
            raise ValueError("Decoded foreground count disagrees with the source manifest")
        return (points @ self.Q.T).astype(np.float32) if viewer else points

    def iter_video_frames(self, camera, indices: Iterable[int]):
        """Seek once, verify real container PTS, and yield native indices/BGR."""
        indices = list(indices)
        if not indices:
            return
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in indices):
            raise ValueError("Video frame indices must be integers")
        if indices[0] < 0 or indices[-1] >= len(self._camera_rows[camera]):
            raise IndexError("Requested decode index is outside the verified video index")
        if any(b <= a for a, b in zip(indices, indices[1:])):
            raise ValueError("Video indices must be strictly increasing and unique")
        rows = [self._camera_rows[camera][int(index)] for index in indices]
        for _, row, rgb in iter_native_video_frames(self.video_paths[camera], rows):
            yield int(row["record_index"]), rgb[:, :, ::-1].copy()

    def read_image(self, camera, record_index):
        frames = self.iter_video_frames(camera, [record_index])
        try:
            return next(frames)[1]
        finally:
            frames.close()
