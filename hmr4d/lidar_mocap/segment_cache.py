"""Reuse frozen recorded person-cloud members only for identical primary observations."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from .dual_view import file_sha
from .rally_pipeline import read_json, read_rows
from .teacher import load_teacher_track, sample_track


def observation_key(bbox, keypoints):
    return np.asarray(bbox, np.float32).tobytes() + np.asarray(keypoints, np.float32).tobytes()


class FrozenSegmentCache:
    def __init__(self, source_window, pairs, calibration_digest):
        self.source = Path(source_window)
        config = read_json(self.source / "window.json")
        if config["calibration_digest"] != calibration_digest or set(config["source_pairs"]) != {
            p["lidar_index"] for p in pairs
        }:
            raise ValueError("Frozen segmentation sources/calibration/scans differ")
        report = read_json(self.source / "fit_report.json")
        manifest = read_json(self.source / "teacher/manifest.json")
        if report["teacher_provenance_hash"] != manifest["provenance_hash"]:
            raise ValueError("Original segmentation and observations use different provenance")
        times = [p["scan_mid_us"] for p in pairs]
        self.frames = defaultdict(list)
        self.results = {}
        teacher_index = {Path(entry["path"]).name: i for i, entry in enumerate(manifest["tracks"])}
        for i, entry in enumerate(manifest["tracks"]):
            sampled = sample_track(load_teacher_track(self.source / "teacher" / entry["path"]), times)
            for k, pair in enumerate(pairs):
                if sampled["valid"][k]:
                    self.frames[pair["lidar_index"]].append(
                        (i, observation_key(sampled["bbx_xyxy"][k], sampled["kp2d"][k]))
                    )
        for row in read_rows(self.source / "quality.jsonl"):
            self.results[row["lidar_index"], teacher_index[Path(row["teacher"]).name]] = row
        self.excluded = {(r["lidar_index"], int(r["track"])): r["reason"] for r in report["segmentation_exclusions"]}
        self.evidence = dict(
            source_window=str(self.source),
            quality_sha256=file_sha(self.source / "quality.jsonl"),
            teacher_sha256=file_sha(self.source / "teacher/manifest.json"),
            policy="identical_observed_primary_geometry_and_frozen_recorded_members",
        )
        self.evidence["fit_report_sha256"] = file_sha(self.source / "fit_report.json")
        self.artifact_hashes = report.get("artifacts_sha256", {})
        self.checked = set()
        self.hits, self.misses = 0, 0

    def get(self, pair, tracks):
        expected = self.frames.get(pair["lidar_index"], [])
        if len(expected) != len(tracks) or any(t.get("views") for t in tracks):
            self.misses += 1
            return None
        available = defaultdict(list)
        for index, key in expected:
            available[key].append(index)
        mapping = []
        for t in tracks:
            key = observation_key(t["bbox"], t["kp2d"])
            if not available[key]:
                self.misses += 1
                return None
            mapping.append((t, available[key].pop(0)))
        result = {}
        for t, index in mapping:
            key = pair["lidar_index"], index
            if key in self.results:
                row = self.results[key]
                path = Path(row["fit_path"])
                relative = path.relative_to(self.source).as_posix()
                if path not in self.checked:
                    if file_sha(path) != self.artifact_hashes.get(relative):
                        raise ValueError("Frozen recorded segment artifact changed")
                    self.checked.add(path)
                with np.load(row["fit_path"]) as f:
                    points, indices = f["source_cloud"].copy(), f["source_indices"].copy()
                result[str(t["track_id"])] = dict(
                    status="segmented", points=points, indices=indices, source_fit=row["fit_path"]
                )
            elif key in self.excluded:
                result[str(t["track_id"])] = dict(
                    status=self.excluded[key], points=np.empty((0, 3), np.float32), indices=np.empty(0, np.int64)
                )
            else:
                self.misses += 1
                return None
        self.hits += 1
        return result
