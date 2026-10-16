"""Frozen image caches may survive a tracking change, never a camera/source change."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from hmr4d.lidar_mocap.teacher import ALGORITHM_VERSION, SCHEMA, _preprocess_reuse


class PixelCacheReuseTests(unittest.TestCase):
    def test_tracking_change_reuses_exact_pixels_and_rejects_other_camera_or_source(self):
        old = dict(
            algorithm_version=1,
            schema=SCHEMA,
            video=dict(path="original.mp4", size=123),
            options=dict(max_track_gap_frames=1, min_track_frames=2),
            frame_rows=[dict(common_time_us=1000000 + i * 33367, record_index=i * 2) for i in range(8)],
            camera=dict(K=[[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]),
            weights=["frozen"],
        )
        fingerprint = hashlib.sha256(json.dumps(old, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        detections = [[dict(bbx_xyxy=[0, 0, 40, 100])] for _ in range(8)]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "manifest.json").write_text(
                json.dumps(
                    dict(
                        provenance=old,
                        provenance_hash=fingerprint,
                        tracks=[dict(track_id=1, segment_id=0, frame_count=8, path="track.pt")],
                    )
                ),
                encoding="utf-8",
            )
            torch.save(dict(schema=SCHEMA, provenance_hash=fingerprint, detections=detections), path / "detections.pt")
            for stage, shape in (("vitpose", (8, 17, 3)), ("features", (8, 1024))):
                torch.save(
                    dict(schema=SCHEMA, provenance_hash=fingerprint, values=[torch.ones(shape)]), path / f"{stage}.pt"
                )
            new = copy.deepcopy(old)
            new["algorithm_version"] = ALGORITHM_VERSION
            new["options"]["max_track_gap_frames"] = 6
            result = _preprocess_reuse(path, new)
            self.assertEqual(result["detections"], detections)
            self.assertEqual(len(result["values"]["features"]), 8)
            self.assertTrue(all(torch.equal(v, torch.ones(1024)) for v in result["values"]["features"].values()))
            for field, value in (
                ("camera", dict(K=[[900, 0, 960], [0, 1000, 540], [0, 0, 1]])),
                ("video", dict(path="different.mp4", size=123)),
                ("weights", ["changed"]),
            ):
                wrong = copy.deepcopy(new)
                wrong[field] = value
                with self.assertRaisesRegex(ValueError, "identical native frames"):
                    _preprocess_reuse(path, wrong)


if __name__ == "__main__":
    unittest.main()
