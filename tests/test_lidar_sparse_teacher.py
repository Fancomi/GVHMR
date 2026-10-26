"""Timing, evidence and cache contracts for the paired-native RGB teacher."""

import numpy as np
import pytest

from hmr4d.lidar_mocap.sparse_teacher import (
    duration_tracking_limits, expand_conditioning, latent_grid, normalize_sparse_rows,
)
from hmr4d.lidar_mocap.teacher import _normalize_rows


def test_native_anchors_and_integer_clock_are_exact():
    origin = 1_792_494_618_360_116
    times = np.array([origin + 398, origin + 100_498, origin + 183_915, origin + 284_015], dtype=np.int64)
    grid = latent_grid(times, origin)
    assert np.array_equal(grid["relative_time_us"][grid["anchor_indices"]], times - origin)
    assert grid["observed_mask"].sum() == len(times)
    assert np.all(np.diff(grid["relative_time_us"]) > 0)
    assert np.max(np.diff(grid["relative_time_us"])) <= 33_367
    assert grid["relative_seconds"][0] == 0.000398


def test_virtual_keypoints_are_missing_not_fabricated_observations():
    import torch

    grid = latent_grid([0, 100_100, 200_200], 0)
    kp = torch.arange(3 * 17 * 3, dtype=torch.float32).reshape(3, 17, 3)
    expanded = expand_conditioning(kp, grid, keypoints=True)
    assert torch.equal(expanded[grid["anchor_indices"]], kp)
    assert not torch.any(expanded[~grid["observed_mask"], :, 2])
    features = torch.tensor([[0.], [3.], [6.]])
    expected = torch.as_tensor(grid["relative_time_us"] / 100_100 * 3, dtype=torch.float32)
    assert torch.allclose(expand_conditioning(features, grid)[:, 0], expected)


def test_sparse_rows_do_not_weaken_production_native_cadence():
    rows = [dict(record_index=377 + i * 6, media_pts_us=6_289_617 + i * 100_100,
                 common_time_us=1_792_494_618_365_522 + i * 100_100) for i in range(4)]
    sparse = normalize_sparse_rows(rows)
    assert sparse[0]["source_frame_no"] is None
    assert duration_tracking_limits(sparse) == dict(max_gap_frames=1, minimum_span_us=233_333)
    with pytest.raises(ValueError, match="30 Hz"):
        _normalize_rows(rows)
    with pytest.raises(ValueError, match="strictly increasing"):
        normalize_sparse_rows([rows[0], rows[0]])
    with pytest.raises(ValueError, match="discontinuit"):
        normalize_sparse_rows([rows[0], rows[-1]])


@pytest.mark.parametrize("times", [[1, 1], [2, 1], [0, 200_000]])
def test_latent_grid_cannot_bridge_discontinuities(times):
    with pytest.raises(ValueError):
        latent_grid(times, 0)


def _latent_track():
    from hmr4d.lidar_mocap.teacher import COORDINATE_SPACE

    times = np.array([0, 100_100, 200_200], dtype=np.int64)
    grid = latent_grid(times, 0)
    params = {"body_pose": np.zeros((7, 63), np.float32), "global_orient": np.zeros((7, 3), np.float32),
              "betas": np.zeros((7, 10), np.float32), "transl": np.zeros((7, 3), np.float32)}
    params["transl"][2, 1] = 1
    return dict(coordinate_space=COORDINATE_SPACE, common_time_us=times, record_index=np.array([10, 16, 22]),
                media_pts_us=times + 500_000, source_frame_no=np.array([11, 17, 23]),
                kp2d=np.ones((3, 17, 3), np.float32), bbx_xyxy=np.ones((3, 4), np.float32),
                bbx_xys=np.ones((3, 3), np.float32), K_fullimg=np.tile(np.eye(3), (3, 1, 1)),
                provenance_hash="test", track_id=1, segment_id=0, temporal_latent_grid=grid,
                smpl_params_incam={key: value[grid["anchor_indices"]] for key, value in params.items()},
                inferred_temporal_state=dict(role="inferred_model_state_not_rgb_observation", smpl_params_incam=params))


def test_lidar_query_uses_latent_pose_and_only_actual_native_evidence():
    from hmr4d.lidar_mocap.teacher import sample_track

    track = _latent_track()
    sampled = sample_track(track, [95_000, 50_000])
    assert sampled["valid"].tolist() == [True, False]
    assert sampled["metadata"]["record_index"].tolist() == [16, 10]
    assert sampled["metadata"]["model_time_error_us"][0] == 5_100
    assert sampled["smpl_params_incam"]["transl"][0, 1] > 0.1
    assert np.array_equal(sampled["kp2d"][0], track["kp2d"][1])
    ordinary = dict(track)
    ordinary.pop("inferred_temporal_state")
    assert not sample_track(ordinary, [95_000])["valid"][0]
    assert sample_track(track, [])["smpl_params_incam"]["body_pose"].shape == (0, 63)


def test_latent_role_anchor_and_identity_guards():
    from hmr4d.lidar_mocap.teacher import sample_track

    track = _latent_track()
    track["identity_ambiguous"] = np.array([True, False, False])
    assert not sample_track(track, [95_000])["valid"][0]
    track = _latent_track()
    track["inferred_temporal_state"]["role"] = "RGB"
    with pytest.raises(ValueError, match="distinguish"):
        sample_track(track, [95_000])
    track = _latent_track()
    track["temporal_latent_grid"]["observed_mask"][1] = True
    with pytest.raises(ValueError, match="binding"):
        sample_track(track, [95_000])


def _production_cache(tmp_path, monkeypatch):
    import hmr4d
    from hmr4d.lidar_mocap.teacher import _provenance, _save_json, SCHEMA
    from hmr4d.lidar_mocap.pilot import binding

    # Provenance stats these files; no test loads them as pretrained models.
    for relative in ("yolo/yolov8x.pt", "vitpose/vitpose-h-multi-coco.pth",
                     "hmr2/epoch=10-step=25000.ckpt", "gvhmr/gvhmr_siga24_release.ckpt"):
        identity = tmp_path / "inputs/checkpoints" / relative
        identity.parent.mkdir(parents=True, exist_ok=True)
        identity.write_bytes(b"test-only weight identity; not a loadable model")
    monkeypatch.setattr(hmr4d, "PROJ_ROOT", tmp_path)
    video = tmp_path / "native.mp4"
    video.write_bytes(b"test-only-no-video-decode")
    policy = tmp_path / "sampling.json"
    policy.write_text("{}", encoding="utf-8")
    rows = normalize_sparse_rows([dict(record_index=10+i*6, media_pts_us=i*100_100,
                                       common_time_us=1_792_494_618_000_000+i*100_100) for i in range(4)])
    camera = dict(K=[[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], image_size=[640, 480], distortion_opencv=[0]*5)
    options = dict(batch_size=4, min_track_frames=8, max_track_gap_frames=6, inference_window_frames=600,
        roi_polygon=None, roi_margin_px=20, detector_imgsz=1280, preproc_precision="float16",
        sampling_policy="paired_native_rgb10_latent_v1", sampling_recipe=binding(policy),
        sampling_origin_common_time_us=rows[0]["common_time_us"])
    provenance, digest = _provenance(video, rows, camera, options)
    directory = tmp_path / "teacher"
    directory.mkdir()
    manifest = dict(schema=SCHEMA, provenance=provenance, provenance_hash=digest, tracks=[], frame_count=len(rows))
    _save_json(directory / "manifest.json", manifest)
    return directory, manifest, video, rows, camera, options


def test_common_teacher_resumes_sparse_cache_without_dense_preprocessing(tmp_path, monkeypatch):
    from hmr4d.lidar_mocap import teacher

    directory, manifest, video, rows, camera, options = _production_cache(tmp_path, monkeypatch)
    assert teacher.run_teacher(video, rows, camera, directory, **options) == manifest
    assert manifest["provenance"]["options"] == options
    monkeypatch.setattr(teacher, "_detect", lambda *args: [[] for _ in rows])
    rebuilt = teacher.run_teacher(video, rows, camera, directory, resume=False, **options)
    assert rebuilt["provenance"] == manifest["provenance"]
    assert rebuilt["provenance_hash"] == manifest["provenance_hash"]
    assert rebuilt["tracks"] == []


def test_pixel_reuse_reproduces_duration_matched_sparse_segments(tmp_path, monkeypatch):
    import torch
    from hmr4d.lidar_mocap import teacher

    directory, manifest, _, rows, _, options = _production_cache(tmp_path, monkeypatch)
    detections = [[dict(bbx_xyxy=np.array([20, 20, 100, 200], np.float32), detection_confidence=.9)] for _ in rows]
    limits = duration_tracking_limits(rows)
    tracks = teacher.associate_detections(detections, max_track_gap_frames=limits["max_gap_frames"])
    from hmr4d.lidar_mocap.sparse_teacher import observed_segments
    segments, rejected = observed_segments(tracks, rows, limits)
    assert len(segments) == 1 and not rejected
    segment = segments[0]
    manifest["tracks"] = [dict(track_id=segment["track_id"], segment_id=segment["segment_id"],
                               frame_count=4, path="track.pt")]
    teacher._save_json(directory / "manifest.json", manifest)
    teacher._save_tensor(directory / "detections.pt", dict(schema=teacher.SCHEMA,
        provenance_hash=manifest["provenance_hash"], detections=detections))
    for stage, shape in (("vitpose", (4, 17, 3)), ("features", (4, 1024))):
        teacher._save_tensor(directory / (stage + ".pt"), dict(schema=teacher.SCHEMA,
            provenance_hash=manifest["provenance_hash"], values=[torch.ones(shape)]))
    reused = teacher._preprocess_reuse(directory, manifest["provenance"])
    assert len(reused["values"]["vitpose"]) == 4
    assert len(reused["values"]["features"]) == 4
