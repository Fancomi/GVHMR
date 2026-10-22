"""Full-run reuse must retain exact local temporal inputs."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import runpy

import pytest

from hmr4d.lidar_mocap.full_native import validate_window_reuse, pose_config_digest, rebind_native_pose_record
from hmr4d.lidar_mocap.fit_reuse import (
    validate_calibration_reuse, calibration_execution_ast, same_teacher_values, rebind_fit_row,
    validate_audited_fit_execution,
)


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


def test_calibration_reuse_ignores_only_scope_metadata_and_keeps_every_numerical_input():
    previous = dict(cameras=dict(cam1=dict(K=[1, 2], R=[3])), reference_plane=dict(z=0),
                    timing_policy=dict(tolerance=18), sessions=dict(actual=dict(index="exact")),
                    review_only=True, frozen_utc="old", snapshot_digest="old_scope")
    current = deepcopy(previous)
    current.update(frozen_utc="new", snapshot_digest="full_scope", data_relocations=dict(sha256="explicit"))
    current["sessions"]["additional"] = dict(index="new")
    validate_calibration_reuse(previous, current)
    for changed in (dict(current, reference_plane=dict(z=.1)), dict(current, review_only=False),
                    dict(current, timing_policy=dict(tolerance=30)),
                    dict(current, sessions=dict(actual=dict(index="changed")))):
        with pytest.raises(ValueError):
            validate_calibration_reuse(previous, changed)


def test_calibration_execution_proof_allows_freeze_metadata_but_rejects_camera_math_changes():
    before = b'def _same_inputs(a,b): return True\nclass CalibrationSnapshot:\n def freeze(self): return 1\n def camera(self,x): return x*2\n'
    metadata_change = b'def _same_inputs(a,b): return a==b\nclass CalibrationSnapshot:\n def freeze(self): return 3\n def camera(self,x): return x*2\n'
    math_change = metadata_change.replace(b'x*2', b'x*3')
    assert calibration_execution_ast(before) == calibration_execution_ast(metadata_change)
    assert calibration_execution_ast(before) != calibration_execution_ast(math_change)


def test_fit_reuse_import_rejects_stale_compatibility_and_window_execution_proof():
    audit = dict(windows=dict(actual=dict(current_fit_algorithm_digest="current")),
                 algorithms=dict(original=dict(current_digest="current")))
    previous = deepcopy(audit)
    validate_audited_fit_execution(audit, "current")
    assert audit == previous
    with pytest.raises(ValueError, match="repeat the compatibility/input audit"):
        validate_audited_fit_execution(audit, "new-execution")
    for group, field in (("windows", "current_fit_algorithm_digest"), ("algorithms", "current_digest")):
        changed = deepcopy(audit)
        next(iter(changed[group].values()))[field] = "stale"
        with pytest.raises(ValueError, match="repeat the compatibility/input audit"):
            validate_audited_fit_execution(changed, "current")


def test_completed_only_preflight_checks_all_windows_before_any_model_setup(tmp_path):
    from hmr4d.lidar_mocap.repair_execution import require_completed_fit_windows

    jobs = [("actual", Path("source/positive_window")), ("actual", Path("source/-0001_0020"))]
    for name in ("fit_report.json", "teacher/manifest.json"):
        path = tmp_path / "sessions/actual/windows/positive_window" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unchanged")
    with pytest.raises(ValueError, match="actual:-0001_0020"):
        require_completed_fit_windows(tmp_path, jobs)
    assert (tmp_path / "sessions/actual/windows/positive_window/fit_report.json").read_bytes() == b"unchanged"
    for name in ("fit_report.json", "teacher/manifest.json"):
        path = tmp_path / "sessions/actual/windows/-0001_0020" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"done")
    require_completed_fit_windows(tmp_path, jobs)


def test_reused_teacher_requires_exact_parameter_bytes_and_preserves_native_observations():
    import numpy as np

    before = dict(provenance_hash="old", temporal_reuse_from="old.pt",
                  smpl_params_incam=dict(transl=np.array([[1, 2, 3]], np.float32)),
                  association_cost=np.array([np.nan], np.float32), record_index=[7, 9])
    after = deepcopy(before)
    after.update(provenance_hash="new", temporal_reuse_from="current.pt")
    assert same_teacher_values(before, after)
    after["smpl_params_incam"]["transl"][0, 0] += .001
    assert not same_teacher_values(before, after)
    after = deepcopy(before)
    after["smpl_params_incam"]["transl"] = after["smpl_params_incam"]["transl"].astype(np.float16)
    assert not same_teacher_values(before, after)


def test_fit_metadata_scope_rebinding_keeps_qa_rejection_and_all_numeric_metrics(tmp_path):
    path = str((tmp_path / "old_pose.json").resolve())
    poses = {path: dict(source=dict(path=path, sha256="old_pixels"),
                       target=dict(path=str(tmp_path / "new_pose.json"), sha256="rebound_metadata"))}
    row = dict(calibration_digest="old_scope", fit_path="original/fits/body.npz",
               accepted=False, rejection_reasons=["floor_penetration"], source_segment_indices=[1, 3, 8],
               qa=dict(surface_mean_m=.023, max_ground_penetration_m=.17),
               teacher_sampling=dict(provenance_hash="old", target_common_time_us=12345),
               native_pose_evidence=dict(files=dict(cam1=dict(path=path, sha256="old_pixels")),
                                         median_epipolar_px=2.3))
    before = deepcopy(row)
    result = rebind_fit_row(row, "full_scope", tmp_path / "window", poses, "current_teacher")
    assert row == before
    assert result["accepted"] is False and result["rejection_reasons"] == row["rejection_reasons"]
    assert result["qa"] == row["qa"] and result["source_segment_indices"] == row["source_segment_indices"]
    assert result["teacher_sampling"]["target_common_time_us"] == row["teacher_sampling"]["target_common_time_us"]
    assert result["native_pose_evidence"]["median_epipolar_px"] == 2.3
    assert result["native_pose_evidence"]["files"]["cam1"] == poses[path]["target"]
    with pytest.raises(ValueError, match="exact original native observation"):
        rebind_fit_row(row, "full_scope", tmp_path / "window", {}, "teacher")


def test_new_gpu_batch_cannot_overwrite_or_resume_a_changed_released_progress_file(tmp_path):
    import hashlib
    import json
    from hmr4d.lidar_mocap.repair_execution import load_isolated_stage_progress

    previous = tmp_path / "progress01.json"
    previous.write_text(json.dumps(dict(windows=dict(done=dict(sha256="actual")))) , encoding="utf-8")
    original = previous.read_bytes()
    receipt = dict(full_teacher_progress=dict(path=str(previous), sha256=hashlib.sha256(original).hexdigest()))
    (tmp_path / "teacher_batch_01_release_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    options = dict(default_name="progress01.json", release_pattern="teacher_batch_*_release_receipt.json",
                   binding_key="full_teacher_progress")
    with pytest.raises(ValueError, match="SHA-bound"):
        load_isolated_stage_progress(tmp_path, None, None, **options)
    output, state = load_isolated_stage_progress(tmp_path, tmp_path / "progress02.json", previous, **options)
    assert output.name == "progress02.json" and state["windows"]["done"]["sha256"] == "actual"
    state["windows"]["done"]["sha256"] = "new_batch"
    assert previous.read_bytes() == original and not output.exists()
    previous.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="progress changed"):
        load_isolated_stage_progress(tmp_path, tmp_path / "progress02.json", previous, **options)
    with pytest.raises(ValueError, match="inside"):
        load_isolated_stage_progress(tmp_path, tmp_path.parent / "outside.json", None, **options)


@pytest.mark.skipif(__import__("os").name != "nt", reason="Native Windows hidden subprocess contract")
def test_real_hidden_worker_records_actual_exit_codes_and_below_normal_priority(tmp_path):
    import json
    import sys
    import psutil
    from hmr4d.lidar_mocap.repair_execution import run_isolated_native_worker

    for code in (0, 7):
        state = {}
        path = tmp_path / f"worker_{code}.json"
        log = tmp_path / f"worker_{code}.log"
        script = "import json,os,psutil; print(json.dumps(dict(pid=os.getpid(),priority=psutil.Process().nice())),flush=True); raise SystemExit(" + str(code) + ")"
        result = run_isolated_native_worker([sys.executable, "-c", script], cwd=tmp_path, log_path=log,
            state=state, state_path=path, active=dict(role="cpu_worker_contract"))
        actual = json.loads(log.read_bytes())
        assert result["exit_code"] == code and actual["priority"] == psutil.BELOW_NORMAL_PRIORITY_CLASS
        assert json.loads(path.read_bytes())["active"]["role"] == "cpu_worker_contract"
        assert not psutil.pid_exists(actual["pid"])


def test_gpu_process_matcher_keeps_other_runs_and_explicit_cpu_stages_out(tmp_path):
    from hmr4d.lidar_mocap.repair_execution import native_gpu_command_matches

    root = tmp_path / "actual"
    assert native_gpu_command_matches(["python", "run_badminton_full_native.py", "--root", str(root), "--stage", "teacher"], root)
    assert native_gpu_command_matches(["python", "repair_badminton_body_labels.py", "--output-root", str(root / "repair")], root)
    assert not native_gpu_command_matches(["python", "repair_badminton_body_labels.py", "--output-root", str(root / "repair"), "--device", "cpu"], root)
    assert not native_gpu_command_matches(["python", "run_badminton_full_native.py", "--root", str(root), "--stage", "aggregate"], root)
    assert not native_gpu_command_matches(["python", "run_badminton_full_native.py", "--root", str(root) + "_other", "--stage", "fit"], root)
    assert not native_gpu_command_matches(["python", "watch_badminton_full_native.py", "--root", str(root)], root)
