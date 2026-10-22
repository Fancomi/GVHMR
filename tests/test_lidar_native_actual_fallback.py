"""Exercise all-actual fallback through real provenance, coverage and entry gates."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hmr4d.lidar_mocap import formal_baseline as baseline
from hmr4d.lidar_mocap import formal_native_derive as derived, native_witness as witness
from hmr4d.lidar_mocap.guarded_leg_gpu import _complete_native
from test_lidar_formal_import_audit import case as audited_case, write, rows
from test_lidar_formal_native_derive import bundle, prepare
from test_lidar_formal_baseline import stage, read, refresh_report


def simulate_all_actual(bundle, monkeypatch):
    """Replace only the numerical verifier, retaining every real wrapper check."""
    repair = bundle.case.formal / "repair"
    path = repair / "final_labels/native_verification.json"
    registry = repair / "final_labels/source_registry.json"
    registry.write_bytes((bundle.case.formal / "source/source_registry.json").read_bytes())
    calls = []

    def actual_check(entry, run_name):
        current = read(path)
        assert current["windows"] == {}, "All windows must really reach the numerical verifier"
        with open(bundle.model, "rb") as handle:
            handle.read()
        jobs = derived._scope(read(repair / "final_labels/repair_plan.json")["jobs"])
        for key, item in bundle.refreshed["windows"].items():
            parent = read(Path(item["parent_root"]) / "repair/final_labels/native_verification.json")
            fresh = deepcopy(parent["windows"][key])
            fresh.pop("verification_origin", None)
            fresh["body"]["report_sha256"] = witness._sha(Path(item["target_window"]) / "fit_report.json")
            fresh["teacher_inputs_sha256"] = witness.teacher_fingerprints(repair, jobs[key])
            current["windows"][key] = fresh
        current.update(
            passed=True,
            state="complete",
            verified_windows=3,
            total_windows=3,
            accepted_checked=3,
            source_union_sha256=bundle.result["source_union_sha256"],
            calibration_digest=bundle.result["calibration_digest"],
        )
        write(path, current)
        calls.append(set(current["windows"]))

    monkeypatch.setattr(witness.runpy, "run_path", actual_check)
    return calls


def assert_all_actual(repair, native):
    keys = set(native["windows"])
    assert native["verification_coverage"]["actual_window_keys"] == sorted(keys)
    assert native["verification_coverage"]["derived_window_keys"] == []
    assert len(native["actual_native_witnesses"]) == 1
    item = native["actual_native_witnesses"][0]
    actual, _ = witness.validate_native_witness(item, repair, require_full=False, window_keys=keys)
    assert actual["state"] == "actual_complete" and actual["preexisting_checked_windows"] == 0
    assert set(actual["actual_window_keys"]) == keys and actual["cached_window_keys"] == []
    assert all(
        value["verification_origin"] == dict(mode="actual", witness=item) for value in native["windows"].values()
    )


def test_no_parent_witness_all_actual_fallback_reaches_real_coverage_and_guard(bundle, monkeypatch):
    assert prepare(bundle, {})["state"] == "actual_native_check_required"
    calls = simulate_all_actual(bundle, monkeypatch)
    repair = bundle.case.formal / "repair"
    final = derived.run_required_actual_native(repair)
    assert calls == [set(bundle.refreshed["windows"])]
    assert_all_actual(repair, final)
    assert not derived.has_derived_native(final)
    assert derived.validate_native_derivations(repair, final)
    assert derived.refresh_native_coverage(repair) == final
    _complete_native(final, set(final["windows"]), repair=repair)
    imported = Path(bundle.import_binding["path"])
    damaged = read(imported)
    damaged["qa_values_unchanged"] = False
    write(imported, damaged)
    with pytest.raises(ValueError, match="formal numerical import"):
        _complete_native(final, set(final["windows"]), repair=repair)


def test_all_postleg_windows_really_reconstruct_and_pass_coverage_and_guard(bundle, monkeypatch):
    from hmr4d.lidar_mocap.formal_completion import _native_after

    prepare(bundle)
    for item in bundle.refreshed["windows"].values():
        path = Path(item["target_window"]) / "fit_report.json"
        report = read(path)
        report["post_leg_refinement"] = dict(selection_sha256="a" * 64, frames=1)
        write(path, report)
    calls = simulate_all_actual(bundle, monkeypatch)
    repair = bundle.case.formal / "repair"
    final = derived.run_required_actual_native(repair, postleg=True)
    assert calls == [set(bundle.refreshed["windows"])]
    assert_all_actual(repair, final)
    assert final["derived_native_proof"]  # Preserve the immutable pre-leg lineage.
    assert derived.validate_native_derivations(repair, final, postleg=True)
    assert derived.refresh_native_coverage(repair, postleg=True) == final
    assert (
        _native_after(
            dict(
                repair=repair,
                keys=set(final["windows"]),
                accepted=3,
                source_sha=final["source_union_sha256"],
                calibration=final["calibration_digest"],
                before=dict(windows={key: dict(accepted=1) for key in final["windows"]}),
            )
        )
        == final
    )
    # The pre-leg guard must still reject post-leg substitution of its baseline.
    with pytest.raises(ValueError, match="removed or substituted"):
        _complete_native(final, set(final["windows"]), repair=repair)


def test_all_actual_fallback_freezes_real_baseline_and_guard_without_mocking_validation(stage, monkeypatch):
    project = stage.root.parent.parent
    for module in (derived, witness):
        monkeypatch.setattr(module, "PROJ_ROOT", project)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    model = project / "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"synthetic neutral model used only by the numerical stand-in")
    verifier = project / "tools/verify_badminton_native_repair.py"
    verifier.parent.mkdir(parents=True)
    verifier.write_bytes(b"# synthetic numerical verifier\n")
    contracts = project / "hmr4d/utils/body_model/contracts.py"
    contracts.parent.mkdir(parents=True)
    contracts.write_bytes(b"# synthetic body contract\n")

    def runtime():
        return dict(
            models=dict(neutral_smpl=witness._binding(model)),
            files={str(path): witness._sha(path) for path in (model, verifier, contracts)},
            runtime_versions=dict(python="synthetic"),
            device="cpu",
            model_type="smpl",
            gender="neutral",
        )

    monkeypatch.setattr(derived, "capture_runtime", runtime)
    monkeypatch.setattr(witness, "capture_runtime", runtime)
    repair = stage.repair
    write(repair / "final_labels/calibration/calibration_snapshot.json", dict(snapshot_digest="c" * 64))
    write(repair / "final_labels/source_registry.json", dict(sources={}))
    jobs = baseline._scope(read(stage.plan["path"])["jobs"])
    for key, job in jobs.items():
        window = stage.current_dirs[key]
        for folder in (Path(job["source_window"]), window):
            write(folder / "teacher/manifest.json", dict(tracks=[dict(path="track.pt")]))
            (folder / "teacher/track.pt").write_bytes(b"synthetic teacher tensors")
        (Path(job["source_window"]) / "teacher/detections.pt").write_bytes(b"synthetic detections")
        pose = repair / "native_poses" / (key.replace(":", "_") + ".json")
        write(pose, dict(native_record=True))
        cloud = repair / (key.replace(":", "_") + ".png")
        cloud.write_bytes(b"synthetic original point cloud")
        report_path = window / "fit_report.json"
        report = read(report_path)
        metadata = [
            dict(row, lidar_path=str(cloud), native_pose_evidence=dict(files=dict(cam1=witness._binding(pose))))
            for row in [json.loads(line) for line in (window / "quality.jsonl").read_bytes().splitlines()]
            if row["accepted"]
        ]
        rows(window / "ptv3.metadata.jsonl", metadata)
        report["artifacts_sha256"]["ptv3.metadata.jsonl"] = witness._sha(window / "ptv3.metadata.jsonl")
        report["source_lidar_sha256"] = {str(cloud): witness._sha(cloud)}
        write(report_path, report)
        refresh_report(stage, key)
    original = read(stage.native["path"])
    for key, entry in original["windows"].items():
        entry["teacher"] = dict(observed_person_frames=1)
        entry["body"].update(
            native_scans=1,
            native_pose_files=1,
            maximums=dict(source_point_error_m=0.0, pelvis_error_m=0.0, rebuilt_mesh_error_m=0.0),
        )
        entry["teacher_inputs_sha256"] = witness.teacher_fingerprints(repair, jobs[key])
    # Historical results have no actual execution witness and cannot be cached.
    write(Path(stage.native["path"]), original)
    calls = []

    def actual_check(entry, run_name):
        assert read(stage.native["path"])["windows"] == {}
        with open(model, "rb") as handle:
            handle.read()
        write(Path(stage.native["path"]), original)
        calls.append(True)

    monkeypatch.setattr(witness.runpy, "run_path", actual_check)
    final = derived.run_required_actual_native(repair)
    assert calls == [True]
    assert_all_actual(repair, final)
    assert derived.refresh_native_coverage(repair) == final
    _complete_native(final, set(stage.keys), repair=repair)
    frozen = stage.freeze()
    assert read(frozen["baseline"]["path"])["state"] == "complete"
    assert read(frozen["snapshots"]["native"]["path"])["verification_coverage"] == final["verification_coverage"]
    assert frozen["gpu_started"] is False and not (stage.folder.parent / "guarded_leg_stage_lease.json").exists()
    final["verification_coverage"]["actual_window_keys"].pop()
    write(Path(stage.native["path"]), final)
    with pytest.raises(ValueError, match="coverage is incomplete"):
        stage.freeze()
