import base64
import copy
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
    apply_display_ground_extension,
    load_display_bbox,
    validate_ground_display_parent,
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
            publication_path.write_text(
                json.dumps(
                    dict(
                        accepted=7,
                        exports=dict(s=dict(accepted=7), empty=dict(accepted=0)),
                    )
                ),
                encoding="utf-8",
            )
            manifest = dict(
                canonical_publication=dict(
                    path=str(publication_path), sha256=hashlib.sha256(publication_path.read_bytes()).hexdigest()
                ),
                sources=dict(s=dict(baseline=True, rallies=[{}]), empty=dict(baseline=False, rallies=[])),
            )
            report = root / "sessions/s/windows/first/fit_report.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps(dict(fitted=12, accepted=10)), encoding="utf-8")
            viewer = ViewerData(root, root, None, manifest, {}, {})
            before = {path: path.read_bytes() for path in (publication_path, report)}
            state = viewer.status()
            self.assertEqual(state["status_source"], "canonical_publication")
            self.assertEqual(state["paired_accepted"], 7)
            self.assertEqual(state["sources"]["s"]["accepted"], 7)
            self.assertEqual(state["sources"]["s"]["completed_fit_person_frames"], 12)
            self.assertEqual(state["sources"]["s"]["completed_fit_qa_pass"], 10)
            self.assertEqual(state["sources"]["empty"]["accepted"], 0)
            self.assertEqual(state["sources"]["empty"]["available_fit_windows"], 0)
            self.assertFalse((root / "build_status.json").exists())
            self.assertEqual({path: path.read_bytes() for path in before}, before)
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


class BackgroundGroundDisplayTests(unittest.TestCase):
    sid = "20261022_161632"

    def settings(self, path):
        return dict(
            schema="badminton-background-ground-display-v1",
            sport="badminton",
            scope="background_display_only",
            source_ids=[self.sid],
            base_bbox_sha256="a" * 64,
            ground_ignore_id="ground",
            ground_positive_local_z_extension_m=0.015,
            dataset_manifest=dict(path=str(path.resolve()), sha256="b" * 64),
            preview_only=True,
            formal_crop_unchanged=True,
        )

    def bbox(self, rotation=(0, 0, 0)):
        bbox = NativeRoiTests().bbox(rotation)
        bbox["boxes"][0]["id"] = "ground"
        bbox["boxes"][1]["id"] = "net"
        return bbox

    def test_tilted_extension_retains_original_region_net_and_ground_bottom(self):
        bbox = self.bbox((12, -8, 31))
        before = copy.deepcopy(bbox)
        settings = self.settings(Path("unused"))
        changed = apply_display_ground_extension(bbox, settings, "a" * 64)
        original = badminton_detection_bbox(bbox)
        self.assertEqual(changed["boxes"][1:], original["boxes"][1:])
        direction = bbox_rotation_float32(bbox["boxes"][0]["rotation_deg"])[:, 2].astype(np.float64)
        old_ground, new_ground = original["boxes"][0], changed["boxes"][0]
        np.testing.assert_array_equal(
            new_ground["center"], np.asarray(old_ground["center"], np.float64) + direction * 0.0075
        )
        np.testing.assert_allclose(
            np.asarray(new_ground["center"]) - direction * new_ground["size"][2] / 2,
            np.asarray(old_ground["center"]) - direction * old_ground["size"][2] / 2,
            atol=1e-15,
            rtol=0,
        )
        self.assertEqual(bbox, before)

    def test_closed_settings_reject_missing_extra_and_invalid_values(self):
        original = self.settings(Path("unused"))
        invalid = []
        extra = dict(original, extra=True)
        invalid.append(extra)
        for field in original:
            value = copy.deepcopy(original)
            del value[field]
            invalid.append(value)
        for field, value in [
            ("scope", "formal"),
            ("preview_only", 1),
            ("formal_crop_unchanged", False),
            ("source_ids", [self.sid, self.sid]),
            ("source_ids", []),
            ("source_ids", [1]),
            ("dataset_manifest", dict(path="relative", sha256="b" * 64)),
            ("dataset_manifest", dict(path=str(Path("unused").resolve()), sha256="bad")),
            ("ground_ignore_id", "net"),
            ("ground_positive_local_z_extension_m", True),
            ("ground_positive_local_z_extension_m", float("nan")),
            ("ground_positive_local_z_extension_m", -0.01),
            ("base_bbox_sha256", "c" * 64),
        ]:
            invalid.append(dict(original, **{field: value}))
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                apply_display_ground_extension(self.bbox(), settings, "a" * 64)

    def test_load_sidecar_checks_frozen_bbox_bytes_and_keeps_region(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bbox = self.bbox()
            bbox_path = root / "bbox.json"
            bbox_path.write_text(json.dumps(bbox), encoding="utf-8")
            sha = hashlib.sha256(bbox_path.read_bytes()).hexdigest()
            calibration = SimpleNamespace(
                snapshot_json=root / "snapshot.json",
                data=dict(
                    bbox=bbox,
                    artifacts=dict(bbox=dict(path="bbox.json", sha256=sha)),
                ),
            )
            settings = dict(self.settings(root / "dataset.json"), base_bbox_sha256=sha)
            path = root / "display.json"
            path.write_text(json.dumps(settings), encoding="utf-8")
            changed, evidence = load_display_bbox(calibration, path)
            self.assertEqual(changed["boxes"][2], badminton_detection_bbox(bbox)["boxes"][2])
            self.assertEqual(evidence["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            bbox_path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "bytes/geometry mismatch"):
                load_display_bbox(calibration, path)

    def test_actual_parent_binding_and_review_scope_are_mandatory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "labels"
            path = root / "datasets/dataset_manifest.json"
            path.parent.mkdir(parents=True)
            body = dict(review_only=True, training_published=False, exports={self.sid: {}})
            path.write_text(json.dumps(body), encoding="utf-8")
            settings = self.settings(path)
            settings["dataset_manifest"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            evidence = dict(settings=settings)
            registry = dict(sources={self.sid: {}})
            output = root.parent / "new_viewer"
            validate_ground_display_parent(root, root, registry, body, evidence, output)
            for alteration in ["hash", "path", "sources", "review", "training", "baseline", "output"]:
                changed = copy.deepcopy(evidence)
                actual = copy.deepcopy(body)
                baseline, target = root, output
                if alteration == "hash":
                    changed["settings"]["dataset_manifest"]["sha256"] = "0" * 64
                elif alteration == "path":
                    changed["settings"]["dataset_manifest"]["path"] = str(root / "other.json")
                elif alteration == "sources":
                    changed["settings"]["source_ids"] = ["20261023_000000"]
                elif alteration == "review":
                    actual["review_only"] = False
                elif alteration == "training":
                    actual["training_published"] = True
                elif alteration == "baseline":
                    baseline = root.parent
                else:
                    target = root / "viewer"
                with self.subTest(alteration=alteration), self.assertRaises(ValueError):
                    validate_ground_display_parent(root, baseline, registry, actual, changed, target)

    def fixture(self, root, *, filtered):
        cloud = np.array([[2, 3, 0.105], [2, 3, 0.105], [2, 3, 0.11], [2, 3, 0.13], [2, 3, 0.09]], np.float32)
        bbox = self.bbox()
        settings = self.settings(root / "datasets/dataset_manifest.json")
        evidence = dict(settings=settings, sha256="c" * 64)
        manifest = dict(sources={self.sid: dict(common_origin_us=0, baseline=True)})
        if filtered:
            manifest["point_filter"] = dict(display_crop=evidence)
        calibration = SimpleNamespace(
            data=dict(bbox=bbox), project=lambda pts, cam, distort: (pts[:, :2] * 10, np.ones(len(pts)))
        )
        camera = dict(status="matched", record_index=0, media_pts_us=0, common_time_us=0)
        pair = dict(lidar_index=0, scan_mid_us=0, cameras={cam: dict(camera) for cam in ("cam1", "cam2")})
        session = SimpleNamespace(
            pairs=lambda: [pair],
            read_cloud=lambda pair: cloud.copy(),
            camera_rows=lambda cam: [camera],
            read_image=lambda cam, index: np.full((16, 16, 3), 100, np.uint8),
        )
        viewer = ViewerData(
            root,
            root,
            calibration,
            manifest,
            {self.sid: session},
            {},
            viewer_bbox=(
                apply_display_ground_extension(bbox, settings, "a" * 64) if filtered else badminton_detection_bbox(bbox)
            ),
        )
        rows = []
        for track, index, accepted in [(1, 1, True), (2, 2, False)]:
            path = root / "sessions" / self.sid / "windows/window/fits" / f"fit{track}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                np.savez(
                    path,
                    source_indices=np.array([index], np.int64),
                    source_cloud=cloud[[index]],
                    vertices=cloud[[index]],
                    joints=cloud[[index]],
                )
            row = dict(
                fit_path=str(path),
                motion_id=f"{self.sid}:window:{track}:0",
                lidar_index=0,
                accepted=accepted,
                rejection_reasons=[] if accepted else ["stored_rejected"],
                qa=dict(final_p2surface_mean_m=0.01),
            )
            rows.append(row)
            viewer._track_colors[row["motion_id"]] = "#abcdef"
        viewer.quality_rows = lambda sid: {0: rows}
        return viewer, cloud, rows

    def test_geometry_protects_exact_ordinals_including_rejected_and_preserves_person_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, cloud, _ = self.fixture(root, filtered=False)
            new, _, rows = self.fixture(root, filtered=True)
            original, changed = old.geometry(self.sid, 0), new.geometry(self.sid, 0)
            np.testing.assert_array_equal(
                np.frombuffer(base64.b64decode(changed["frame"]["cloud"]), "<f4").reshape(-1, 3), cloud[[1, 2, 3]]
            )
            self.assertEqual(changed["frame"]["background_filter"]["removed_background_points"], 1)
            self.assertEqual(changed["frame"]["background_filter"]["removed_fit_members"], 0)
            self.assertEqual(changed["frame"]["athletes"], original["frame"]["athletes"])
            self.assertEqual(changed["athletes"], original["athletes"])
            self.assertEqual(changed["native_images"], original["native_images"])
            self.assertEqual(new.geometry(self.sid, 0, "all"), old.geometry(self.sid, 0, "all"))
            for row in rows:
                self.assertEqual(new.render_fit(row["fit_path"]), old.render_fit(row["fit_path"]))
            before = new.fit_source_members.cache_info().hits
            new.geometry(self.sid, 0)
            self.assertGreater(new.fit_source_members.cache_info().hits, before)

    def test_empty_scan_has_no_protected_fits_and_keeps_only_background_above_slab(self):
        with tempfile.TemporaryDirectory() as directory:
            viewer, cloud, _ = self.fixture(Path(directory), filtered=True)
            viewer.quality_rows = lambda sid: {}
            result = viewer.geometry(self.sid, 0)
            self.assertEqual(result["frame"]["point_count"], 1)
            self.assertEqual(result["frame"]["background_filter"]["removed_background_points"], 3)
            self.assertEqual(result["frame"]["background_filter"]["protected_members"], 0)

    def test_invalid_member_indices_bytes_scan_and_source_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            viewer, cloud, rows = self.fixture(Path(directory), filtered=True)
            row = rows[0]
            good = dict(row)
            for index, points in [
                (np.array([-1], np.int64), cloud[[1]]),
                (np.array([len(cloud)], np.int64), cloud[[1]]),
                (np.array([1], np.float32), cloud[[1]]),
                (np.array([[1]], np.int64), cloud[[1]]),
                (np.array([1], np.int64), cloud[[3]]),
                (np.array([1], np.int64), cloud[[1]].astype(np.float64)),
            ]:
                np.savez(row["fit_path"], source_indices=index, source_cloud=points)
                viewer.fit_source_members.cache_clear()
                with self.subTest(indices=index), self.assertRaisesRegex(ValueError, "indices/cloud mismatch"):
                    viewer.protected_source_members(self.sid, 0, cloud, [row])
            for field, value in [
                ("lidar_index", 1),
                ("lidar_index", False),
                ("motion_id", "foreign:0"),
                ("fit_path", str(Path(directory) / "elsewhere.npz")),
            ]:
                altered = dict(good, **{field: value})
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "source/scan mismatch"):
                    viewer.protected_source_members(self.sid, 0, cloud, [altered])
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                viewer.protected_source_members("foreign", 0, cloud, [])
            Path(row["fit_path"]).unlink()
            with self.assertRaises(FileNotFoundError):
                viewer.protected_source_members(self.sid, 0, cloud, [good])


if __name__ == "__main__":
    unittest.main()
