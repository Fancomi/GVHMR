import base64
import hashlib
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from hmr4d.lidar_mocap.rally_viewer import (
    ViewerData,
    badminton_detection_bbox,
    crop_viewer_points,
    bbox_rotation_float32,
)


class NativeImageBindingTests(unittest.TestCase):
    def test_rgb_is_bound_to_real_record_and_native_pts(self):
        images = [np.full((24, 32, 3), value, dtype=np.uint8) for value in (20, 170)]
        rows = [
            dict(media_pts_us=90123, common_time_us=1700000000000010),
            dict(media_pts_us=106807, common_time_us=1700000000016694),
        ]
        session = SimpleNamespace(camera_rows=lambda cam: rows, read_image=lambda cam, i: images[i].copy())
        viewer = ViewerData(None, None, None, None, {"s": session}, {})
        first = viewer.native_image("s", "cam1", 0)
        second = viewer.native_image("s", "cam1", 1)
        self.assertEqual(second["record_index"], 1)
        self.assertEqual(second["media_pts_us"], 106807)
        self.assertEqual(second["raw_bgr_sha256"], hashlib.sha256(images[1].tobytes()).hexdigest())
        self.assertNotEqual(first["raw_bgr_sha256"], second["raw_bgr_sha256"])
        blob = base64.b64decode(second["jpeg_base64"])
        self.assertEqual(second["jpeg_sha256"], hashlib.sha256(blob).hexdigest())
        decoded = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(decoded, images[1])


class NativeRoiTests(unittest.TestCase):
    def bbox(self, rotation=(0, 0, 0)):
        return dict(
            boxes=[
                dict(type="Ignore", enabled=True, center=[0, 0, 0], size=[6, 14, 0.2], rotation_deg=list(rotation)),
                dict(type="Ignore", enabled=True, center=[0, 0, 0.9], size=[6, 0.6, 1.8], rotation_deg=[0, 0, 0]),
            ]
        )

    def test_ground_top_region_net_ignore_and_closed_boundaries(self):
        bbox = badminton_detection_bbox(self.bbox())
        points = np.array(
            [
                [2, 3, 0.1],
                [2, 3, 0.10001],
                [2, 3, 5.1],
                [2, 3, 5.1001],
                [3.001, 3, 1],
                [0, 0, 1],
                [2, 0.3, 1],
                [2, 0.301, 1],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            crop_viewer_points(points, bbox), [False, True, True, False, False, False, False, True]
        )

    def test_tilted_ground_derived_region_and_obsolete_region_ignored(self):
        original = self.bbox((12, -8, 31))
        original["boxes"].append(dict(type="Region", center=[999, 999, 999], size=[9999] * 3))
        bbox = badminton_detection_bbox(original)
        rotation = bbox_rotation_float32([12, -8, 31])
        center = np.asarray(bbox["boxes"][2]["center"], np.float32)
        expected = rotation @ np.array([0, 0, 2.6], np.float32)
        np.testing.assert_array_equal(center, expected)
        points = np.array([[2, 4, 1], [4, 4, 1], [2, 4, 6]], np.float32) @ rotation.T
        np.testing.assert_array_equal(crop_viewer_points(points, bbox), [True, False, False])

    def test_display_crop_preserves_source_array_and_original_order(self):
        points = np.array([[0, 0, 1], [2, 3, 1], [2, 3, 6], [1, 2, 2]], np.float32)
        before = points.copy()
        mask = crop_viewer_points(points, badminton_detection_bbox(self.bbox()))
        np.testing.assert_array_equal(points, before)
        np.testing.assert_array_equal(np.flatnonzero(mask), [1, 3])

    def test_batch_uses_one_native_decode_stream_per_camera(self):
        calls = []
        rows = [dict(record_index=i, media_pts_us=i * 100000, common_time_us=i * 100000) for i in range(4)]
        pairs = [
            dict(scan_mid_us=i * 100000, cameras={c: dict(status="matched", record_index=i) for c in ("cam1", "cam2")})
            for i in range(4)
        ]

        def stream(cam, indices):
            calls.append((cam, list(indices)))
            for i in indices:
                yield i, np.full((16, 16, 3), i, np.uint8)

        session = SimpleNamespace(camera_rows=lambda c: rows, pairs=lambda: pairs, iter_video_frames=stream)
        viewer = ViewerData(None, None, None, dict(sources={"s": dict(common_origin_us=0)}), {"s": session}, {})
        viewer.geometry = lambda sid, t, scope: dict(index=round(t * 10))
        result = viewer.geometry_batch("s", 0, 4)
        self.assertEqual(calls, [("cam1", [0, 1, 2, 3]), ("cam2", [0, 1, 2, 3])])
        self.assertEqual(result["count"], 4)
        viewer.geometry_batch("s", 0, 4)
        self.assertEqual(calls[-2:], [("cam1", []), ("cam2", [])])


if __name__ == "__main__":
    unittest.main()
