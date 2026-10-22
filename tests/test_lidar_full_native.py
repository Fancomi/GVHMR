"""Full-run reuse must retain exact local temporal inputs."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import runpy

import pytest

from hmr4d.lidar_mocap.full_native import validate_window_reuse, pose_config_digest, rebind_native_pose_record


def window():
    return dict(session_id="actual", window_id="100_200", left_us=100, right_us=200,
                settings=dict(camera="cam2", teacher_stride=2), source_pairs=[2, 3, 4], calibration_digest="old")


def test_scope_digest_can_differ_only_after_caller_checks_physical_camera_equivalence():
    before = window()
    after = dict(before, calibration_digest="full_scope")
    previous = deepcopy((before, after))
    validate_window_reuse(before, after)
    assert (before, after) == previous


@pytest.mark.parametrize("key,value", [
    ("session_id", "another"), ("window_id", "100_300"), ("left_us", 99), ("right_us", 201),
    ("settings", dict(camera="cam1", teacher_stride=2)), ("source_pairs", [2, 3]),
])
def test_changed_local_teacher_inputs_cannot_reuse_frozen_prediction(key, value):
    before = window()
    with pytest.raises(ValueError, match="exact original native"):
        validate_window_reuse(before, dict(before, **{key: value}))


def test_full_teacher_lease_uses_actual_utc_deadline_and_manifest():
    module = runpy.run_path(str(Path(__file__).parents[1] / "tools/run_badminton_full_teachers.py"))
    remaining = module["remaining_lease_seconds"]
    now = datetime(2026, 10, 22, tzinfo=timezone.utc)
    lease = dict(status="held", owner="GVHMR", manifest_sha256="actual", training_published=False,
                 approved_deadline_utc="2026-10-22T00:30:00+00:00")
    assert remaining(lease, "actual", now) == 1800
    assert remaining(dict(lease, approved_deadline_utc="2026-10-21T23:59:00Z"), "actual", now) == -60
    for changed in (dict(lease, status="released"), dict(lease, owner="another"),
                    dict(lease, manifest_sha256="other"), dict(lease, training_published=True),
                    dict(lease, approved_deadline_utc="2026-10-22T00:30:00")):
        with pytest.raises(ValueError):
            remaining(changed, "actual", now)


def test_native_pose_scope_rebinding_preserves_every_observation_and_original_evidence():
    config = dict(calibration_digest="old_scope", models=dict(pose="actual_sha"), precision="float16")
    current = dict(config, calibration_digest="full_scope")
    native = dict(record_index=7, common_time_us=1000000, media_pts_us=33333, source_frame_no=97)
    record = dict(config_digest=pose_config_digest(config), native=native, raw_bgr_sha256="pixels",
                  boxes=[[1.25, 3.5, 9, 10]], keypoints=[[[4.5, 6.25, 0.97]]])
    previous = deepcopy(record)
    result = rebind_native_pose_record(record, config, current, native, dict(path="old.json", sha256="original"))
    assert record == previous
    assert result["config_digest"] == pose_config_digest(current)
    assert result["boxes"] == record["boxes"] and result["keypoints"] == record["keypoints"]
    assert result["raw_bgr_sha256"] == record["raw_bgr_sha256"] and result["native"] == native
    assert result["reuse_native_evidence"]["source"]["sha256"] == "original"
    for changed_config, changed_native in (
        (dict(current, models=dict(pose="changed")), native),
        (dict(current, precision="float32"), native),
        (current, dict(native, common_time_us=1000001)),
    ):
        with pytest.raises(ValueError, match="Native pose reuse changed"):
            rebind_native_pose_record(record, config, changed_config, changed_native, {})
