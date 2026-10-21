import base64
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from hmr4d.lidar_mocap.rally_viewer import (
    ViewerData,
    badminton_detection_bbox,
    crop_viewer_points,
    bbox_rotation_float32,
    apply_display_net_extension,
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


class PublishedStatusTests(unittest.TestCase):
    def test_missing_live_status_uses_bound_publication_and_actual_fit_counts_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publication_path = root / "canonical-publication.json"
            publication_path.write_text(json.dumps(dict(
                accepted=7, exports=dict(s=dict(accepted=7), empty=dict(accepted=0)),
            )), encoding="utf-8")
            manifest = dict(
                canonical_publication=dict(path=str(publication_path),
                                           sha256=hashlib.sha256(publication_path.read_bytes()).hexdigest()),
                sources=dict(s=dict(baseline=True, rallies=[{}]), empty=dict(baseline=False, rallies=[])),
            )
            report = root / "sessions/s/windows/first/fit_report.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps(dict(fitted=12, accepted=10)), encoding="utf-8")
            viewer = ViewerData(root, root, None, manifest, {}, {})
            before = {path:path.read_bytes() for path in (publication_path, report)}
            state = viewer.status()
            self.assertEqual(state["status_source"], "canonical_publication")
            self.assertEqual(state["paired_accepted"], 7)
            self.assertEqual(state["sources"]["s"]["accepted"], 7)
            self.assertEqual(state["sources"]["s"]["completed_fit_person_frames"], 12)
            self.assertEqual(state["sources"]["s"]["completed_fit_qa_pass"], 10)
            self.assertEqual(state["sources"]["empty"]["accepted"], 0)
            self.assertEqual(state["sources"]["empty"]["available_fit_windows"], 0)
            self.assertFalse((root / "build_status.json").exists())
            self.assertEqual({path:path.read_bytes() for path in before}, before)
            publication_path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                viewer.status()

    def test_existing_live_status_keeps_priority_over_publication_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live = root / "build_status.json"
            live.write_text(json.dumps(dict(state="running", sources={})), encoding="utf-8")
            viewer = ViewerData(root, root, None, dict(canonical_publication=dict(path="missing")), {}, {})
            before = live.read_bytes()
            self.assertEqual(viewer.status()["state"], "running")
            self.assertEqual(live.read_bytes(), before)

    def test_missing_status_without_a_bound_publication_is_not_invented(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            viewer = ViewerData(root, root, None, dict(sources={}), {}, {})
            with self.assertRaises(FileNotFoundError):
                viewer.status()

    def test_independent_generated_status_precedes_original_status_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "build_status.json"
            original.write_text(json.dumps(dict(state="original", sources={})), encoding="utf-8")
            output = root / "independent/viewer"
            output.mkdir(parents=True)
            (output / "build_status.json").write_text(
                json.dumps(dict(state="repair_candidate", sources={})), encoding="utf-8"
            )
            viewer = ViewerData(root, root, None, dict(sources={}), {}, {}, viewer_output=output)
            before = original.read_bytes()
            self.assertEqual(viewer.status()["state"], "repair_candidate")
            self.assertEqual(original.read_bytes(), before)
            (output / "build_status.json").unlink()
            self.assertEqual(viewer.status()["state"], "original")


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

    def test_display_net_extension_changes_only_positive_local_face(self):
        original = self.bbox()
        original["boxes"][0]["size"][0] = 10
        original["boxes"][1]["id"] = "region_2"
        settings = dict(
            schema="badminton-display-crop-v1",
            sport="badminton",
            scope="display_only",
            base_bbox_sha256="a" * 64,
            net_ignore_id="region_2",
            net_positive_local_x_extension_m=0.19,
        )
        adjusted = apply_display_net_extension(original, settings, "a" * 64)
        points = np.array([[-3.1, 0, 1], [3.1, 0, 1], [3.25, 0, 1]], np.float32)
        np.testing.assert_array_equal(
            crop_viewer_points(points, badminton_detection_bbox(original)), [True, True, True]
        )
        np.testing.assert_array_equal(
            crop_viewer_points(points, badminton_detection_bbox(adjusted)), [True, False, True]
        )
        self.assertEqual(original["boxes"][1]["center"], [0, 0, 0.9])
        self.assertEqual(adjusted["boxes"][1]["center"], [0.095, 0, 0.9])
        self.assertEqual(adjusted["boxes"][1]["size"], [6.19, 0.6, 1.8])
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            apply_display_net_extension(original, settings, "b" * 64)

    def test_display_net_uses_original_float64_center_and_native_float32_direction(self):
        bbox = self.bbox()
        bbox["boxes"][1].update(id="region_2", center=[11.8, 0.736139, -1.92984], rotation_deg=[0, 0, 3.38739])
        settings = dict(
            schema="badminton-display-crop-v1",
            sport="badminton",
            scope="display_only",
            base_bbox_sha256="a" * 64,
            net_ignore_id="region_2",
            net_positive_local_x_extension_m=0.19,
        )
        adjusted = apply_display_net_extension(bbox, settings, "a" * 64)
        np.testing.assert_array_equal(adjusted["boxes"][1]["center"], [11.894834022521973, 0.74175223357746, -1.92984])

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
