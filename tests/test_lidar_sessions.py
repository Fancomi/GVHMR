import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from hmr4d.lidar_mocap.sessions import SessionIndex, decode_fixed24


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "synthetic"
        self.lidar = self.root / "output-lidar" / "lidar_synthetic"
        self.video = self.root / "output-zcam" / "zcam_synthetic"
        self.meta = self.root / "meta" / "synthetic"
        for folder in (self.lidar, self.video, self.meta):
            folder.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, relative, value):
        path = self.root / relative
        path.write_text(json.dumps(value), encoding="utf-8")

    def make_session(self, mids=(100000, 110000, 120000), times=(100000, 110000, 120000), empty=()):
        frames = [{"frame_id": i + 1, "file": f"frame_{i:06d}.png", "kept_points": 0 if i in empty else 1}
                  for i in range(len(mids))]
        captures = [{"device_first_packet_us": t - 1000, "device_last_packet_us": t + 1000} for t in mids]
        manifest = {"coordinate_encoding": "fixed24", "coordinate_space": "lidar_sensor_xyz",
                    "lidar_rotation": [0, -1, 0, 1, 0, 0, 0, 0, 1], "frame_count": len(frames),
                    "frames": frames, "frame_capture_timestamps": captures,
                    "point_width": 2, "point_height": 1, "scale": 1000, "center": 256}
        self.write("output-lidar/lidar_synthetic/manifest.json", manifest)
        tracks = [{"device_id": "lidar1", "rows": [{"frame_id": i + 1, "common_time_us": t - 1000,
                  "end_common_time_us": t + 1000} for i, t in enumerate(mids)]}]
        for camera in ("cam1", "cam2"):
            rows = [{"common_time_us": t, "media_pts_us": i * 13001, "frame_no": 601 + i}
                    for i, t in enumerate(times)]
            tracks.append({"device_id": camera, "rows": rows})
            sidecar = [{"record_index": i, "media_pts": row["media_pts_us"],
                        "media_time_base_num": 1, "media_time_base_den": 1000000,
                        "discontinuity": False, "synthesized": False, "connection_segment": 0}
                       for i, row in enumerate(rows)]
            (self.video / f"{camera}.frames.jsonl").write_text(
                "\n".join(json.dumps(row) for row in sidecar), encoding="utf-8")
        clock = {"schema": "external_gm_common_timeline_v1", "session_id": "synthetic",
                 "common_overlap_begin_us": min(times), "common_overlap_end_us": max(times),
                 "physical_accuracy_verified": False, "exposure_offset_calibrated": False,
                 "tracks": tracks}
        self.write("meta/synthetic/synthetic_common_clock.json", clock)
        return clock

    def test_rgb_fixed24_zero_axis_and_float32(self):
        encoded = np.array([[257234, 256000], [254000, 0], [256100, 256500]], dtype=np.uint32).reshape(3, 2)
        rgb = np.stack(((encoded >> 16) & 255, (encoded >> 8) & 255, encoded & 255), axis=-1).astype(np.uint8)
        path = self.lidar / "fixed.png"
        self.assertTrue(cv2.imwrite(str(path), rgb[:, :, ::-1]))
        decoded = decode_fixed24(path, (2, 1))
        expected = np.array([[257234, 254000, 256100]], dtype=np.float32) / np.float32(1000) - np.float32(256)
        np.testing.assert_array_equal(decoded, expected)
        self.assertEqual(decoded.dtype, np.float32)
        with self.assertRaises(ValueError):
            decode_fixed24(path, (1, 2))

    def test_nearest_tie_boundary_discrete_pts_and_empty(self):
        self.make_session(mids=(99000, 105000, 121000), times=(100000, 110000, 120000), empty=(1,))
        index = SessionIndex.load(self.root)
        pairs = index.pairs()
        self.assertEqual(len(pairs), 3)
        self.assertIsNone(pairs[0]["cameras"]["cam1"]["record_index"])
        self.assertIsNone(pairs[2]["cameras"]["cam1"]["record_index"])
        self.assertEqual(pairs[1]["cameras"]["cam1"]["record_index"], 0)
        self.assertEqual(pairs[1]["cameras"]["cam1"]["time_error_us"], -5000)
        self.assertIn("empty_foreground", pairs[1]["reasons"])
        self.assertEqual(index.camera_rows("cam1")[1]["media_pts_us"], 13001)
        self.assertEqual([r["record_index"] for r in index.teacher_rows("cam1")], [0, 2])

    def test_missing_clock_keeps_decode_index_and_excludes_gap(self):
        clock = self.make_session(mids=(100000, 110000, 120000), times=(100000, 110000, 120000))
        clock["tracks"][1]["rows"].pop(1)
        self.write("meta/synthetic/synthetic_common_clock.json", clock)
        index = SessionIndex.load(self.root)
        self.assertIsNone(index.camera_rows("cam1")[1]["common_time_us"])
        self.assertEqual(index.pairs()[2]["cameras"]["cam1"]["record_index"], 2)
        self.assertEqual(index.pairs()[1]["status"], "excluded")
        self.assertIn("cam1_time_error_exceeds_tolerance", index.pairs()[1]["reasons"])

    def test_missing_lidar_clock_never_uses_receive_time(self):
        clock = self.make_session()
        clock["tracks"][0]["rows"].pop(1)
        self.write("meta/synthetic/synthetic_common_clock.json", clock)
        index = SessionIndex.load(self.root)
        self.assertEqual(len(index.pairs()), 3)
        self.assertIsNone(index.pairs()[1]["scan_mid_us"])
        self.assertIn("missing_lidar_clock", index.pairs()[1]["reasons"])

    def test_raw_viewer_rotation_once_and_count_evidence(self):
        self.make_session()
        index = SessionIndex.load(self.root)
        pair = index.pairs()[0]
        encoded = np.array([[257000, 0], [258000, 0], [259000, 0]], dtype=np.uint32)
        rgb = np.stack(((encoded >> 16) & 255, (encoded >> 8) & 255, encoded & 255), axis=-1).astype(np.uint8)
        self.assertTrue(cv2.imwrite(pair["lidar_path"], rgb[:, :, ::-1]))
        np.testing.assert_array_equal(index.read_cloud(pair, viewer=False), [[1, 2, 3]])
        np.testing.assert_array_equal(index.read_cloud(pair, viewer=True), [[-2, 1, 3]])
        self.assertEqual(pair["status"], "paired")
        with self.assertRaisesRegex(ValueError, "count disagrees"):
            index.read_cloud({**pair, "kept_points": 2})

    def test_no_common_clock_retains_all_source_scans(self):
        self.make_session()
        # A source manifest still carries device and host timestamps. They must
        # not replace the absent cross-device common-clock evidence.
        self.write("meta/synthetic/synthetic_common_clock.json", {})
        index = SessionIndex.load(self.root)
        self.assertEqual(len(index.pairs()), 3)
        self.assertTrue(all(p["status"] == "excluded" for p in index.pairs()))
        self.assertTrue(all(p["scan_mid_us"] is None for p in index.pairs()))

    def test_frozen_map_verified_and_mismatch_rejected(self):
        self.make_session()
        index = SessionIndex.load(self.root)
        frozen = []
        for pair in index.pairs():
            frozen.append({"lidar_frame": pair["lidar_index"], "device_first_us": pair["scan_start_us"],
                           "device_last_us": pair["scan_end_us"], "scan_mid_us": pair["scan_mid_us"],
                           "cameras": {name: {"status": row["status"], "record_index": row["record_index"],
                                               "media_pts_us": row["media_pts_us"], "video_common_us": row["common_time_us"],
                                               "time_error_us": row["time_error_us"]}
                                       for name, row in pair["cameras"].items()}})
        path = self.root / "frozen.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in frozen), encoding="utf-8")
        self.assertEqual(SessionIndex.load(self.root, path).pairs(), index.pairs())
        frozen[1]["cameras"]["cam1"]["media_pts_us"] += 1
        path.write_text("\n".join(json.dumps(row) for row in frozen), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "timing disagrees"):
            SessionIndex.load(self.root, path)

    def test_video_indices_stream_without_seek(self):
        self.make_session()
        index = SessionIndex.load(self.root)
        class Capture:
            def __init__(self):
                self.at = -1
                self.released = False
            def isOpened(self):
                return True
            def grab(self):
                self.at += 1
                return self.at < 3
            def retrieve(self):
                return True, np.full((1, 1, 3), self.at, dtype=np.uint8)
            def release(self):
                self.released = True
        capture = Capture()
        with patch("hmr4d.lidar_mocap.sessions.cv2.VideoCapture", return_value=capture):
            result = list(index.iter_video_frames("cam1", [0, 2]))
        self.assertEqual([i for i, _ in result], [0, 2])
        self.assertEqual(int(result[1][1][0, 0, 0]), 2)
        self.assertTrue(capture.released)
        with self.assertRaises(ValueError):
            list(index.iter_video_frames("cam1", [2, 1]))


if __name__ == "__main__":
    unittest.main()
