"""CPU tests for pairing/tracking semantics; no pretrained-model invocation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from hmr4d.lidar_mocap.teacher import (
    COORDINATE_SPACE,
    SCHEMA,
    _bbox_xys,
    _iter_video_frames,
    _normalize_rows,
    _provenance,
    _preprocess_reuse,
    _split_tracks,
    _validate_preproc_precision,
    associate_detections,
    sample_track,
    validate_provenance,
)


def _rows(count=8):
    return [{"record_index": i * 2, "common_time_us": 1_000_000 + i * 33_367,
             "media_pts_us": i * 33_367, "source_frame_no": 58_741 + i * 2} for i in range(count)]


def _track():
    count = 3
    global_orient = np.zeros((count, 3), dtype=np.float32)
    global_orient[:, 2] = np.deg2rad([179, -179, -175])
    return {
        "schema": SCHEMA, "provenance_hash": "test", "track_id": 1, "segment_id": 0,
        "coordinate_space": COORDINATE_SPACE,
        "common_time_us": np.array([0, 33_334, 66_668], dtype=np.int64),
        "record_index": np.array([10, 12, 14]), "source_frame_no": np.array([1010, 1012, 1014]),
        "media_pts_us": np.array([100_000, 133_334, 166_668]),
        "smpl_params_incam": {"global_orient": global_orient,
                              "body_pose": np.tile(global_orient, (1, 21)),
                              "betas": np.zeros((count, 10), dtype=np.float32),
                              "transl": np.array([[0., 0., 10.], [1., 0., 10.], [2., 0., 10.]])},
        "kp2d": np.stack([np.full((17, 3), i) for i in range(count)]),
        "bbx_xyxy": np.tile([0., 0., 20., 40.], (count, 1)),
        "bbx_xys": np.tile([10., 20., 48.], (count, 1)),
        "K_fullimg": np.tile(np.eye(3), (count, 1, 1)),
        "identity_ambiguous": np.zeros(count, dtype=bool),
    }


class TrackingTests(unittest.TestCase):
    def test_every_person_is_preserved_and_detection_order_does_not_set_identity(self):
        frames = []
        for i in range(6):
            first = {"bbx_xyxy": [10 + 2 * i, 10, 30 + 2 * i, 70], "detection_confidence": 0.9}
            second = {"bbx_xyxy": [150 - i, 10, 175 - i, 80], "detection_confidence": 0.8}
            frames.append([first, second] if i % 2 == 0 else [second, first])
        tracks = associate_detections(frames)
        self.assertEqual(len(tracks), 2)
        self.assertEqual([len(t) for t in tracks.values()], [6, 6])
        self.assertEqual(tracks[1][-1]["bbx_xyxy"][0], 20)
        self.assertEqual(tracks[2][-1]["bbx_xyxy"][0], 145)

    def test_missing_detection_splits_temporal_context_without_filling_a_person(self):
        detection = {"bbx_xyxy": [10, 10, 30, 70], "detection_confidence": 0.9}
        frames = [[detection], [detection], [], [detection], [detection], [detection]]
        tracks = associate_detections(frames, max_track_gap_frames=1)
        self.assertEqual(len(tracks), 1)
        segments, rejected = _split_tracks(tracks, _rows(6), min_track_frames=2)
        self.assertEqual([len(s["observations"]) for s in segments], [2, 3])
        self.assertEqual(rejected, [])
        self.assertEqual(sum(len(s["observations"]) for s in segments), 5)

    def test_ambiguous_identity_is_flagged(self):
        start = [{"bbx_xyxy": [0, 0, 20, 60]}, {"bbx_xyxy": [25, 0, 45, 60]}]
        overlap = [{"bbx_xyxy": [10, 0, 30, 60]}, {"bbx_xyxy": [10, 0, 30, 60]}]
        tracks = associate_detections([start, overlap])
        self.assertTrue(any(t[-1]["identity_ambiguous"] for t in tracks.values()))

    def test_crop_has_room_for_arms_after_192_column_feature_crop(self):
        np.testing.assert_allclose(_bbox_xys([0, 0, 120, 60]), [60, 30, 192])

    def test_real_1009_narrow_detection_does_not_create_a_new_identity(self):
        boxes = [[1246.6621, 657.41785, 1328.89685, 801.42896],
                 [1280.0231, 658.14246, 1328.38391, 800.67432],
                 [1248.5032, 658.46503, 1328.68726, 801.40906]]
        frames = [[dict(bbx_xyxy=box)] for box in boxes]
        self.assertEqual(len(associate_detections(frames, prediction_mode="corners_v1")), 2)
        fixed = associate_detections(frames)
        self.assertEqual(len(fixed), 1)
        np.testing.assert_array_equal([o["bbx_xyxy"] for o in fixed[1]], np.asarray(boxes, dtype=np.float32))

    def test_real_missing_interval_does_not_force_a_track_link(self):
        det = dict(bbx_xyxy=[0, 0, 40, 100])
        tracks = associate_detections([[det]] + [[] for _ in range(8)] + [[det]])
        self.assertEqual(len(tracks), 2)

    def test_ambiguous_recovery_does_not_force_one_of_two_people(self):
        first = [dict(bbx_xyxy=[0, 0, 40, 100]), dict(bbx_xyxy=[5, 0, 45, 100])]
        later = [dict(bbx_xyxy=[2.5, 0, 42.5, 100])]
        self.assertEqual(len(associate_detections([first, [], [], later])), 3)

    def test_recovery_preserves_every_original_continuous_tracklet(self):
        boxes = [[0, 0, 82, 145], [33, 0, 82, 145], [1, 0, 82, 145], [2, 0, 82, 145]]
        frames = [[dict(bbx_xyxy=box, observation_id=i)] for i, box in enumerate(boxes)]
        before = associate_detections(frames, prediction_mode="corners_v1", max_track_gap_frames=1)
        after = associate_detections(frames)
        mapping = {o['observation_id']: k for k, group in after.items() for o in group}
        self.assertEqual(len(mapping), len(frames))
        for group in before.values():
            self.assertEqual(len({mapping[o['observation_id']] for o in group}), 1)


class SamplingTests(unittest.TestCase):
    def test_rotation_branch_crossing_uses_so3_and_keypoints_stay_observed(self):
        sampled = sample_track(_track(), [16_667])
        rotation = Rotation.from_rotvec(sampled["smpl_params_incam"]["global_orient"]).as_matrix()[0]
        np.testing.assert_allclose(rotation, Rotation.from_euler("z", 180, degrees=True).as_matrix(), atol=1e-6)
        np.testing.assert_allclose(sampled["smpl_params_incam"]["transl"], [[0.5, 0, 10]])
        np.testing.assert_array_equal(sampled["kp2d"], np.zeros((1, 17, 3)))
        self.assertTrue(sampled["valid"][0])
        self.assertEqual(sampled["metadata"]["model_time_error_us"][0], -16_667)
        self.assertEqual(sampled["metadata"]["record_index"][0], 10)

    def test_time_rejection_and_exact_sample_identity(self):
        track = _track()
        track["identity_ambiguous"][0] = True
        sampled = sample_track(track, [33_334, 200_000, -30_000])
        np.testing.assert_array_equal(sampled["valid"], [True, False, False])
        self.assertEqual(sampled["metadata"]["model_time_error_us"][1], -133_332)

    def test_no_interpolation_over_unobserved_large_gap(self):
        track = _track()
        track["common_time_us"] = np.array([0, 33_334, 166_668])
        self.assertFalse(sample_track(track, [150_000])["valid"][0])
        self.assertTrue(sample_track(track, [166_668])["valid"][0])

    def test_empty_request_is_supported(self):
        sampled = sample_track(_track(), [])
        self.assertEqual(sampled["smpl_params_incam"]["body_pose"].shape, (0, 63))
        self.assertEqual(sampled["valid"].shape, (0,))


class ProvenanceTests(unittest.TestCase):
    def test_half_precision_changes_cache_hash_and_default_preserves_existing_hash(self):
        camera = {"K": np.eye(3), "distortion_opencv": [0] * 5, "image_size": [1920, 1080]}
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "source.mp4"
            video.write_bytes(b"source identity")
            _, old_hash = _provenance(video, _rows(), camera, {"batch_size": 4})
            options = {"batch_size": 4, "preproc_precision": "float32"}
            _, explicit_default_hash = _provenance(video, _rows(), camera, options)
            half_provenance, half_hash = _provenance(video, _rows(), camera,
                                                    {"batch_size": 4, "preproc_precision": "float16"})
            self.assertEqual(old_hash, explicit_default_hash)
            self.assertEqual(options["preproc_precision"], "float32")
            self.assertNotEqual(old_hash, half_hash)
            self.assertEqual(half_provenance["options"]["preproc_precision"], "float16")
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                validate_provenance({"schema": SCHEMA, "provenance_hash": old_hash}, half_hash)

    def test_unsupported_precision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "preproc_precision"):
            _validate_preproc_precision("bfloat16")

    def test_lidar_10hz_cannot_be_called_video_30hz(self):
        rows = _rows()
        for i, row in enumerate(rows):
            row["common_time_us"] = 1_000_000 + i * 100_000
        with self.assertRaisesRegex(ValueError, "30 Hz"):
            _normalize_rows(rows)
        self.assertEqual(_normalize_rows(_rows())[1]["record_index"], 2)

    def test_changed_calibration_indices_or_source_rejects_cached_inference(self):
        camera = {"K": np.eye(3), "distortion_opencv": [0] * 5, "image_size": [1920, 1080]}
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "source.mp4"
            video.write_bytes(b"source identity")
            _, original_hash = _provenance(video, _rows(), camera, {})
            for changed_rows, changed_camera in [(_rows(), {**camera, "distortion_opencv": [-0.03, 0, 0, 0, 0]}),
                                                 ([{**r, "record_index": r["record_index"] + 1} for r in _rows()], camera)]:
                _, changed_hash = _provenance(video, changed_rows, changed_camera, {})
                with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                    validate_provenance({"schema": SCHEMA, "provenance_hash": original_hash}, changed_hash)
            video.write_bytes(b"changed source identity")
            _, changed_hash = _provenance(video, _rows(), camera, {})
            self.assertNotEqual(original_hash, changed_hash)


class ContainerTimestampTests(unittest.TestCase):
    def test_seek_reads_actual_pts_and_rejects_wrong_sidecar_pts(self):
        import av

        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "native.mp4"
            with av.open(str(video), "w") as container:
                stream = container.add_stream("mpeg4", rate=30)
                stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
                for i in range(15):
                    frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 10, dtype=np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            with av.open(str(video)) as container:
                pts = [int(round(f.pts * float(f.time_base) * 1_000_000)) for f in container.decode(video=0)]
            rows = [{"record_index": i, "media_pts_us": pts[i], "common_time_us": 1_000_000 + pts[i],
                     "source_frame_no": 100 + i} for i in [5, 7, 9]]
            decoded = list(_iter_video_frames(video, rows))
            self.assertEqual([item[1]["record_index"] for item in decoded], [5, 7, 9])
            np.testing.assert_allclose([item[2].mean() for item in decoded], [50, 70, 90], atol=5)
            rows[1]["media_pts_us"] += 100
            with self.assertRaisesRegex(ValueError, "PTS missing/mismatched"):
                list(_iter_video_frames(video, rows))


if __name__ == "__main__":
    unittest.main()
