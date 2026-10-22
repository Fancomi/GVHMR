"""Synthetic execution-envelope checks; the existing reconstructor is unchanged."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hmr4d.lidar_mocap import native_witness as witness


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return witness._binding(path)


@pytest.fixture
def execution(tmp_path, monkeypatch):
    project = tmp_path / "project"
    repair = project / "outputs/parent/repair"
    monkeypatch.setattr(witness, "PROJ_ROOT", project)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    model = project / "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"synthetic model bytes, not a real SMPL model")
    verifier = project / "tools/verify_badminton_native_repair.py"
    verifier.parent.mkdir(parents=True)
    verifier.write_bytes(b"# synthetic verifier entry\n")
    contracts = project / "hmr4d/utils/body_model/contracts.py"
    contracts.parent.mkdir(parents=True)
    contracts.write_bytes(b"# synthetic source code\n")

    def runtime():
        return dict(
            models=dict(neutral_smpl=witness._binding(model)),
            files={str(path): witness._sha(path) for path in (model, verifier, contracts)},
            runtime_versions=dict(python="synthetic", torch="synthetic"),
            device="cpu",
            model_type="smpl",
            gender="neutral",
        )

    monkeypatch.setattr(witness, "capture_runtime", runtime)
    source = put(repair / "new_source_input.json", dict(schema="synthetic-native-input"))
    put(repair / "identity_evidence/identity_decisions.json", dict(review_status="reviewed"))
    put(repair / "pose_receipt.json", dict(state="complete"))
    put(repair / "final_labels/calibration/calibration_snapshot.json", dict(snapshot_digest="c" * 64))
    put(repair / "final_labels/source_registry.json", dict(sources={}))
    jobs = []
    native_windows = {}
    for index in range(2):
        sid, name = "20261020_182854", f"window_{index}"
        initial = repair / "initial_teacher/sessions" / sid / "windows" / name
        target = repair / "final_labels/sessions" / sid / "windows" / name
        jobs.append(dict(source_id=sid, source_window=str(initial)))
        for base in (initial, target):
            put(base / "teacher/manifest.json", dict(tracks=[dict(path="track.pt")]))
            (base / "teacher/track.pt").write_bytes(b"synthetic native teacher numeric payload")
        (initial / "teacher/detections.pt").write_bytes(b"synthetic native detections")
        poses = {
            camera: put(repair / "native_poses" / camera / (name + ".json"), dict(native_record=index))
            for camera in ("cam1", "cam2")
        }
        cloud = repair / (name + "_cloud.png")
        cloud.write_bytes(b"synthetic encoded points")
        metadata = json.dumps(dict(native_pose_evidence=dict(files=poses), lidar_path=str(cloud))) + "\n"
        assets = {}
        for relative, content in {
            "quality.jsonl": metadata,
            "ptv3.metadata.jsonl": metadata,
            "ptv3.pth": "synthetic point tensors",
            "fits/body.npz": "synthetic SMPL tensors",
        }.items():
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            assets[relative] = witness._sha(path)
        report = put(
            target / "fit_report.json",
            dict(accepted=1, artifacts_sha256=assets, source_lidar_sha256={str(cloud): witness._sha(cloud)}),
        )
        native_windows[f"{sid}:{name}"] = dict(
            teacher=dict(observed_person_frames=1),
            body=dict(
                accepted_checked=1,
                native_scans=1,
                native_pose_files=2,
                report_sha256=report["sha256"],
                maximums=dict(source_point_error_m=0.0, pelvis_error_m=0.0, rebuilt_mesh_error_m=0.0),
            ),
        )
    put(
        repair / "final_labels/repair_plan.json",
        dict(jobs=jobs, source_union_sha256=source["sha256"], calibration_digest="c" * 64),
    )
    for key, job in zip(native_windows, jobs):
        native_windows[key]["teacher_inputs_sha256"] = witness.teacher_fingerprints(repair, job)
    native = dict(
        passed=True,
        state="complete",
        windows=native_windows,
        verified_windows=2,
        total_windows=2,
        accepted_checked=2,
        source_union_sha256=source["sha256"],
        calibration_digest="c" * 64,
    )
    calls = []

    def verify(path, run_name):
        calls.append(dict(path=path, argv=list(witness.sys.argv)))
        with open(model, "rb") as handle:
            assert handle.read().startswith(b"synthetic")
        put(repair / "final_labels/native_verification.json", native)

    monkeypatch.setattr(witness.runpy, "run_path", verify)
    return SimpleNamespace(**locals())


def test_first_actual_execution_records_the_real_opened_model_and_both_runtime_snapshots(execution):
    result = witness.run_bound_native_verification(execution.repair)
    value, native = witness.validate_native_witness(result["witness"], execution.repair)
    assert result["state"] == "actual_native_complete_witnessed"
    assert value["preexisting_checked_windows"] == 0 and value["all_windows_reconstructed_in_this_execution"]
    assert value["actual_model_file_reads"][0]["model"] == witness._binding(execution.model)
    assert value["runtime_before"] == value["runtime_after"]
    assert set(value["actual_window_keys"]) == set(native["windows"])
    assert execution.calls[0]["path"] == str(execution.verifier)
    assert execution.calls[0]["argv"][-2:] == ["--threads", "1"]
    assert value["gpu_started"] is False


def test_existing_valid_actual_witness_is_revalidated_without_repeating_reconstruction(execution):
    first = witness.run_bound_native_verification(execution.repair)
    second = witness.run_bound_native_verification(execution.repair)
    assert second["witness"] == first["witness"] and len(execution.calls) == 1


def test_cached_historical_windows_cannot_acquire_a_backfilled_model_witness(execution):
    put(execution.repair / "final_labels/native_verification.json", dict(windows={"historical": {}}))
    result = witness.run_bound_native_verification(execution.repair)
    assert result["state"] == "actual_native_complete_witness_unavailable"
    assert not result["derived_reuse_allowed"] and not result["historical_model_witness_backfilled"]
    assert not (execution.repair / "native_verification_witness.json").exists()
    assert len(execution.calls) == 1


def test_an_unobserved_model_loader_does_not_gain_a_successful_model_witness(execution, monkeypatch):
    def verify(path, run_name):
        put(execution.repair / "final_labels/native_verification.json", execution.native)

    monkeypatch.setattr(witness.runpy, "run_path", verify)
    result = witness.run_bound_native_verification(execution.repair)
    assert result["state"] == "actual_native_complete_witness_unavailable"
    assert not (execution.repair / "native_verification_witness.json").exists()


def test_failed_real_verifier_cannot_publish_a_witness(execution, monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("synthetic reconstruction failure")

    monkeypatch.setattr(witness.runpy, "run_path", failed)
    with pytest.raises(RuntimeError):
        witness.run_bound_native_verification(execution.repair)
    assert not (execution.repair / "native_verification_witness.json").exists()


@pytest.mark.parametrize("asset", ["model", "contracts"])
def test_model_or_library_change_during_execution_blocks_witness_publication(execution, monkeypatch, asset):
    original = witness.runpy.run_path

    def changed(*args, **kwargs):
        original(*args, **kwargs)
        getattr(execution, asset).write_bytes(b"changed during native verification")

    monkeypatch.setattr(witness.runpy, "run_path", changed)
    with pytest.raises(ValueError):
        witness.run_bound_native_verification(execution.repair)
    assert not (execution.repair / "native_verification_witness.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        "model",
        "contracts",
        "teacher",
        "report",
        "witness_sha",
        "missing_reads",
        "cached",
        "window_scope",
        "missing_geometry",
    ],
)
def test_changed_or_incomplete_historical_execution_chain_cannot_support_derivation(execution, change):
    result = witness.run_bound_native_verification(execution.repair)
    item = result["witness"]
    value = json.loads(Path(item["path"]).read_bytes())
    if change in ("model", "contracts"):
        getattr(execution, change).write_bytes(b"changed historical input")
    elif change == "teacher":
        next(iter(execution.native["windows"].values()))["teacher_inputs_sha256"]
        (Path(execution.jobs[0]["source_window"]) / "teacher/track.pt").write_bytes(b"changed teacher")
    elif change == "report":
        path = execution.repair / "final_labels/sessions/20261020_182854/windows/window_0/fit_report.json"
        put(path, dict(accepted=2))
    elif change == "witness_sha":
        item = dict(item, sha256="f" * 64)
    else:
        if change == "missing_reads":
            value["actual_model_file_reads"] = []
        if change == "cached":
            value["preexisting_checked_windows"] = 1
        if change == "window_scope":
            value["actual_window_keys"].pop()
        if change == "missing_geometry":
            value["inputs"].pop("calibration")
        item = put(Path(item["path"]), value)
    with pytest.raises(ValueError):
        witness.validate_native_witness(item, execution.repair)


def test_only_newly_reconstructed_windows_receive_a_subset_execution_witness(execution):
    current = deepcopy(execution.native)
    current["windows"] = {next(iter(current["windows"])): next(iter(current["windows"].values()))}
    put(execution.repair / "final_labels/native_verification.json", current)
    allowed = set(current["windows"])
    result = witness.run_bound_native_verification(execution.repair, verified_cached_keys=allowed)
    value, _ = witness.validate_native_witness(result["witness"], execution.repair, require_full=False)
    assert value["state"] == "actual_subset_complete" and value["preexisting_checked_windows"] == 1
    assert set(value["cached_window_keys"]) == allowed
    assert len(value["actual_window_keys"]) == 1 and not allowed & set(value["actual_window_keys"])
    with pytest.raises(ValueError):
        witness.validate_native_witness(result["witness"], execution.repair)


def test_an_unwitnessed_cached_window_is_really_rebuilt_when_formal_fallback_requires_it(execution):
    put(execution.repair / "final_labels/native_verification.json", execution.native)
    result = witness.run_bound_native_verification(execution.repair, verified_cached_keys=set())
    value, _ = witness.validate_native_witness(result["witness"], execution.repair)
    assert value["preexisting_checked_windows"] == 0 and len(value["actual_window_keys"]) == 2
    prior = list((execution.repair / "native_witness_attempts").glob("*/prior_native_checkpoint.json"))
    assert len(prior) == 1 and len(json.loads(prior[0].read_bytes())["windows"]) == 2
