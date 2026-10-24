"""CPU control tests; real source RGB decode comparisons live in recovery evidence."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hmr4d.lidar_mocap import dual_view_observer as observer
from hmr4d.lidar_mocap.full_native import pose_config_digest


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return dict(path=str(path.resolve()), sha256=observer.file_sha(path))


@pytest.fixture
def av_control(monkeypatch):
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    class Frame:
        def __init__(self, pts=20, time_base=Fraction(1, 1000000)):
            self.pts, self.time_base = pts, time_base

        def to_ndarray(self, format):
            assert format == "rgb24"
            return rgb

    stream = SimpleNamespace(thread_count=0, thread_type="AUTO", time_base=Fraction(1, 1000000))

    class Container:
        streams = SimpleNamespace(video=[stream])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            state.closed = True

        def seek(self, offset, **kwargs):
            state.seek = offset, kwargs

        def decode(self, selected):
            assert selected is stream
            yield from state.frames

    state = SimpleNamespace(frames=[Frame()], stream=stream, Frame=Frame, rgb=rgb, closed=False, opens=0)

    def opened(path):
        state.opens += 1
        return Container()

    monkeypatch.setitem(
        sys.modules,
        "av",
        SimpleNamespace(open=opened, __version__="fixture", library_versions={"libavcodec": (61, 3, 100)}),
    )
    return state


def session(tmp_path):
    video = tmp_path / "native.mp4"
    video.touch()
    rows = [
        dict(record_index=i, media_pts_us=(i + 1) * 20, common_time_us=100 + i, source_frame_no=i) for i in range(3)
    ]
    return SimpleNamespace(video_paths={"cam1": video}, camera_rows=lambda camera: rows), rows


def test_default_delegates_to_unchanged_session_reader(tmp_path):
    source, _ = session(tmp_path)
    marker = object()
    source.iter_video_frames = lambda camera, indices: iter([(indices[0], marker)])
    assert list(observer._camera_frames(source, "cam1", [1], None)) == [(1, marker)]
    assert observer._native_decode_profile(None) is None


def test_explicit_safe_reader_keeps_index_seek_pts_and_bgr_bytes(tmp_path, av_control):
    source, _ = session(tmp_path)
    av_control.frames = [av_control.Frame(0), av_control.Frame(20), av_control.Frame(60)]
    frames = list(observer._camera_frames(source, "cam1", [0, 2], 1))
    assert [index for index, image in frames] == [0, 2]
    assert av_control.stream.thread_count == 1 and av_control.stream.thread_type == "SLICE"
    assert av_control.seek[0] == 20 and av_control.seek[1]["backward"] is True
    assert all(np.array_equal(image, av_control.rgb[:, :, ::-1]) for _, image in frames)
    assert av_control.closed


@pytest.mark.parametrize("value", [True, False, 0, 2, -1, 1.0])
def test_safe_profile_requires_explicit_integer_one(value):
    with pytest.raises(ValueError):
        observer._native_decode_profile(value)


@pytest.mark.parametrize("indices", [[True], [-1], [3], [1, 1], [2, 0], [0.0]])
def test_safe_reader_rejects_non_native_or_unordered_indices(tmp_path, av_control, indices):
    source, _ = session(tmp_path)
    with pytest.raises(ValueError):
        list(observer._camera_frames(source, "cam1", indices, 1))
    assert av_control.opens == 0


@pytest.mark.parametrize("kind", ["missing_pts", "missing_time_base", "wrong_pts", "ended"])
def test_safe_reader_fails_on_missing_or_wrong_native_pts(tmp_path, av_control, kind):
    source, _ = session(tmp_path)
    values = dict(
        missing_pts=[av_control.Frame(None)],
        missing_time_base=[av_control.Frame(20, None)],
        wrong_pts=[av_control.Frame(23)],
        ended=[],
    )
    av_control.frames = values[kind]
    with pytest.raises(ValueError):
        list(observer._camera_frames(source, "cam1", [0], 1))
    assert av_control.closed


@pytest.mark.parametrize("delta", [-2, 0, 2])
def test_original_two_microsecond_tolerance_is_preserved(tmp_path, av_control, delta):
    source, _ = session(tmp_path)
    av_control.frames = [av_control.Frame(20 + delta)]
    assert len(list(observer._camera_frames(source, "cam1", [0], 1))) == 1


@pytest.fixture
def native_cache(tmp_path):
    source = tmp_path / "repair/final_labels"
    sid = "20261022_161632"
    selected, rows = session(tmp_path)
    bgr = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    yy, xx = np.mgrid[:2, :3]
    models = {
        "yolo": dict(path="same-yolo", sha256="a" * 64),
        "coco17_regressor": dict(path="same-regressor", sha256="b" * 64),
    }
    checker = SimpleNamespace(
        models=models,
        detector_imgsz=960,
        map_x=xx.astype(np.float32),
        map_y=yy.astype(np.float32),
        camera=dict(physical_camera=True),
    )
    calibration = SimpleNamespace(data=dict(snapshot_digest="same-calibration"))
    config = dict(calibration_digest="same-calibration", models=models, detector_imgsz=960)
    record = dict(
        schema="badminton-native-pose-v1",
        config_digest=pose_config_digest(config),
        source_id=sid,
        camera="cam1",
        native=rows[0],
        raw_bgr_sha256=hashlib.sha256(bgr.tobytes()).hexdigest(),
        coordinate_space="opencv_undistorted_K_preserved",
        source_video=str(selected.video_paths["cam1"]),
        boxes=[[0, 0, 2, 1]],
        detection_confidence=[0.9],
    )
    path = source.parent / "native_poses" / sid / "cam1/0000000.json"
    inventory_path = tmp_path / "inventory.json"
    save(source / "repair_plan.json", dict(source_union_sha256="same-source"))

    def refresh():
        target = save(path, record)
        inventory = save(
            inventory_path,
            dict(
                schema="badminton-formal-native-pose-import-v1",
                state="complete",
                config=config,
                records=[dict(target=target)],
            ),
        )
        save(
            source.parent / "pose_receipt.json",
            dict(
                state="complete",
                source_union_sha256="same-source",
                config=config,
                config_digest=pose_config_digest(config),
                formal_import_inventory=inventory,
            ),
        )
        return observer._pose_cache_context(source, calibration)

    return SimpleNamespace(
        source=source,
        sid=sid,
        session=selected,
        checker=checker,
        bgr=bgr,
        record=record,
        path=path,
        context=refresh(),
        refresh=refresh,
        calibration=calibration,
    )


def test_legacy_native_pose_boxes_are_reused_only_after_actual_pixel_match(native_cache):
    c = native_cache
    result = observer._pose_detections(c.context, c.source, c.sid, "cam1", c.session, 0, c.bgr, c.checker)
    assert result["detections"] == [dict(bbx_xyxy=[0, 0, 2, 1], detection_confidence=0.9)]
    assert "native_decode_profile" not in c.record
    assert result["evidence"]["raw_bgr_sha256"] == c.record["raw_bgr_sha256"]


@pytest.mark.parametrize(
    "change",
    [
        "missing_sha",
        "wrong_sha",
        "wrong_time",
        "wrong_source",
        "wrong_camera",
        "wrong_config",
        "wrong_video",
        "wrong_model",
        "changed_file",
    ],
)
def test_native_cache_missing_or_nonmatching_evidence_falls_back_without_modifying_input(native_cache, change):
    c = native_cache
    if change == "missing_sha":
        c.record.pop("raw_bgr_sha256")
    elif change == "wrong_sha":
        c.record["raw_bgr_sha256"] = "0" * 64
    elif change == "wrong_time":
        c.record["native"] = dict(c.record["native"], media_pts_us=21)
    elif change == "wrong_source":
        c.record["source_id"] = "foreign"
    elif change == "wrong_camera":
        c.record["camera"] = "cam2"
    elif change == "wrong_config":
        c.record["config_digest"] = "0" * 64
    elif change == "wrong_video":
        c.record["source_video"] = str(c.path)
    elif change == "wrong_model":
        c.checker.models = dict(c.checker.models, yolo=dict(path="foreign", sha256="c" * 64))
    if change != "changed_file":
        c.context = c.refresh()
    else:
        c.path.write_bytes(c.path.read_bytes() + b"\n")
    before = c.path.read_bytes()
    misses = {}
    assert observer._pose_detections(c.context, c.source, c.sid, "cam1", c.session, 0, c.bgr, c.checker, misses) is None
    assert sum(misses.values()) == 1
    assert c.path.read_bytes() == before


def test_pixel_gate_requires_real_evidence_and_checks_every_declared_hash(native_cache):
    c = native_cache
    raw = hashlib.sha256(c.bgr.tobytes()).hexdigest()
    assert not observer._pixel_matches(dict(native_decode_profile="safe"), c.bgr, c.checker)
    assert observer._pixel_matches(dict(native_bgr_image_sha256=raw), c.bgr, c.checker)
    assert not observer._pixel_matches(
        dict(native_bgr_image_sha256=raw, undistorted_rgb_image_sha256="0" * 64), c.bgr, c.checker
    )


def test_same_safe_profile_journal_resumes_and_legacy_is_preserved_on_rejection(tmp_path, av_control):
    profile = observer._native_decode_profile(1)
    frame = dict(lidar_index=1, scan_mid_us=2, cameras={}, athletes=[])
    row = dict(
        lidar_index=1,
        scan_mid_us=2,
        paired_cameras={},
        camera_name="cam1",
        athletes=[],
        timing_seconds=dict(total=0.0),
        native_decode_profile=profile,
    )
    path = tmp_path / "journal.jsonl"
    save(path, row)
    assert observer._resume_rows(path, {1: frame}, "cam1", profile)[0] == {1}
    row.pop("native_decode_profile")
    save(path, row)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="profile"):
        observer._resume_rows(path, {1: frame}, "cam1", profile)
    assert path.read_bytes() == before


def test_cross_profile_journal_admission_uses_pixels_models_camera_time_and_fit(tmp_path, native_cache, monkeypatch):
    c = native_cache
    fit = tmp_path / "fit.npz"
    fit.write_bytes(b"not loaded fixture fit")
    frame = dict(scan_mid_us=2, cameras={}, athletes=[dict(motion_id="person", fit_path=str(fit), accepted=True)])
    record = dict(
        inference_scope="person_detection_and_association_only",
        raw_camera_video=str(c.session.video_paths["cam1"]),
        scan_mid_us=2,
        paired_cameras={},
        native_bgr_image_sha256=hashlib.sha256(c.bgr.tobytes()).hexdigest(),
        athletes=[dict(motion_id="person", fit_path=str(fit), fit_sha256=observer.file_sha(fit))],
        reused_native_evidence=dict(camera=c.checker.camera, models=c.checker.models),
    )
    assert observer._journal_matches(record, frame, c.bgr, c.checker, c.session, "cam1")
    for field, value in [("native_bgr_image_sha256", "0" * 64), ("scan_mid_us", 3), ("raw_camera_video", str(fit))]:
        wrong = deepcopy(record)
        wrong[field] = value
        assert not observer._journal_matches(wrong, frame, c.bgr, c.checker, c.session, "cam1")
