"""Synthetic complete equivalence chains reject missing witnesses and changed data."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hmr4d.lidar_mocap import formal_native_derive as derived, native_witness as witness
from hmr4d.lidar_mocap import formal_numeric_import as importer, formal_import_audit as audit
from test_lidar_formal_import_audit import case as audited_case, write, rows, refresh_report, asset
from test_lidar_formal_numeric_import import bundle as numeric_bundle


@pytest.fixture
def bundle(audited_case, monkeypatch):
    case = audited_case
    project = case.formal.parent.parent
    for module in (derived, witness):
        monkeypatch.setattr(module, "PROJ_ROOT", project)
    # The reused import fixture has a/b/c source IDs; source membership and
    # role/equivalence checks remain real, while the ID-spelling check is covered
    # separately by the existing strict formal-baseline tests.
    scope = lambda jobs: {job["source_id"] + ":" + Path(job["source_window"]).name: job for job in jobs}
    monkeypatch.setattr(derived, "_scope", scope)
    monkeypatch.setattr(witness, "_scope", scope)
    model = project / "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl"
    asset(model, b"synthetic original neutral model bytes")
    model_binding = witness._binding(model)
    for parent in case.parents:
        pose_path = parent / "repair/pose_receipt.json"
        pose = audit.read_json(pose_path)
        pose["config"]["models"] = dict(smpl_triangle_topology=model_binding)
        pose["config_digest"] = audit.pose_config_digest(pose["config"])
        write(pose_path, pose)
        for sid, item in case.per_source.items():
            if item["parent"] != parent:
                continue
            for path in (parent / "repair/native_poses" / sid).glob("*/*.json"):
                value = audit.read_json(path)
                value["config_digest"] = pose["config_digest"]
                write(path, value)
            quality = audit.read_rows(item["window"] / "quality.jsonl")
            for row in quality:
                for camera, binding in row["native_pose_evidence"]["files"].items():
                    row["native_pose_evidence"]["files"][camera] = audit.binding(binding["path"])
            rows(item["window"] / "quality.jsonl", quality)
            rows(item["window"] / "ptv3.metadata.jsonl", quality[:1])
            report_path = item["window"] / "fit_report.json"
            report = audit.read_json(report_path)
            report["native_pose_receipt_sha256"] = audit.file_sha(pose_path)
            report["settings"]["native_pose_receipt_sha256"] = audit.file_sha(pose_path)
            write(report_path, report)
            body = audit.read_json(item["window"] / "window.json")
            body["settings"] = report["settings"]
            write(item["window"] / "window.json", body)
            refresh_report(case, sid)
    numeric = numeric_bundle.__wrapped__(case, monkeypatch)
    # Add the actual native result fields used by the real witness validator.
    for parent in case.parents:
        path = parent / "repair/final_labels/native_verification.json"
        native = audit.read_json(path)
        for entry in native["windows"].values():
            entry["teacher"] = dict(observed_person_frames=1)
            entry["body"].update(
                native_scans=1,
                native_pose_files=2,
                maximums=dict(source_point_error_m=0.0, pelvis_error_m=0.0, rebuilt_mesh_error_m=0.0),
            )
        write(path, native)
    refreshed = audit.audit_formal_numeric_import(case.formal, case.proof, case.parents)
    proof = importer._write_once(case.formal / "import_evidence/audit_after_native_fields.json", refreshed)
    result = importer.import_formal_numeric_bodies(case.formal, proof)
    import_binding = result["receipt"]
    contracts = project / "hmr4d/utils/body_model/contracts.py"
    asset(contracts, b"synthetic frozen body contract")

    def runtime():
        paths = [model, project / "tools/verify_badminton_native_repair.py", contracts]
        return dict(
            models=dict(neutral_smpl=witness._binding(model)),
            files={str(p): witness._sha(p) for p in paths},
            runtime_versions=dict(python="synthetic", torch="synthetic"),
            device="cpu",
            model_type="smpl",
            gender="neutral",
        )

    monkeypatch.setattr(witness, "capture_runtime", runtime)
    monkeypatch.setattr(derived, "capture_runtime", runtime)
    parent_witnesses = {}
    for parent in case.parents:
        repair = parent / "repair"
        native_path = repair / "final_labels/native_verification.json"
        (repair / "final_labels/source_registry.json").write_bytes(
            (parent / "source/source_registry.json").read_bytes()
        )
        snapshot = repair / "native_witness_attempts/synthetic/native_result.json"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes(native_path.read_bytes())
        now = datetime.now(timezone.utc).isoformat()
        native = audit.read_json(native_path)
        value = dict(
            schema=witness.SCHEMA,
            state="actual_complete",
            repair_root=str(repair),
            exit_code=0,
            preexisting_checked_windows=0,
            all_windows_reconstructed_in_this_execution=True,
            actual_window_keys=sorted(native["windows"]),
            runtime_before=runtime(),
            runtime_after=runtime(),
            actual_model_file_reads=[dict(model=model_binding, opened_utc=now)],
            started_utc=now,
            completed_utc=now,
            native_result_snapshot=witness._binding(snapshot),
            inputs={
                name: witness._binding(path)
                for name, path in dict(
                    plan=repair / "final_labels/repair_plan.json",
                    source=repair / "new_source_input.json",
                    identity=repair / "identity_evidence/identity_decisions.json",
                    pose=repair / "pose_receipt.json",
                    calibration=repair / "final_labels/calibration/calibration_snapshot.json",
                    source_registry=repair / "final_labels/source_registry.json",
                ).items()
            },
            gpu_started=False,
        )
        path = repair / "native_verification_witness.json"
        write(path, value)
        parent_witnesses[str(parent)] = witness._binding(path)
    return SimpleNamespace(**locals())


def prepare(bundle, witnesses=None):
    return derived.prepare_derived_native(
        bundle.case.formal, bundle.import_binding, bundle.parent_witnesses if witnesses is None else witnesses
    )


def test_complete_witnessed_import_can_derive_all_windows_without_reconstructing(bundle):
    result = prepare(bundle)
    path = bundle.case.formal / "repair/final_labels/native_verification.json"
    native = audit.read_json(path)
    assert result["derived_windows"] == 3 and result["actual_required"] == {}
    assert native["passed"] is True and native["accepted_checked"] == 3
    assert len(derived.validate_native_derivations(bundle.case.formal / "repair", native)) >= 5
    for key, entry in native["windows"].items():
        assert entry["verification_origin"]["mode"] == "derived"
        assert entry["verification_origin"]["maximums_source"] == "actual_parent_reconstruction"
    assert audit.read_json(result["proof"]["path"])["numeric_reconstruction_recomputed"] is False


def test_missing_historical_witness_requires_real_verification_for_only_that_parent(bundle):
    witnesses = dict(bundle.parent_witnesses)
    witnesses.pop(str(bundle.case.parents[0]))
    result = prepare(bundle, witnesses)
    native = audit.read_json(bundle.case.formal / "repair/final_labels/native_verification.json")
    assert result["derived_windows"] == 2 and set(result["actual_required"]) == {"a:0_1"}
    assert native["passed"] is False and native["state"] == "partial"
    assert native["verified_windows"] == 2 and native["total_windows"] == 3


def test_no_original_model_witness_never_creates_a_passed_native_result(bundle):
    result = prepare(bundle, {})
    assert result["state"] == "actual_native_check_required"
    assert not (bundle.case.formal / "repair/final_labels/native_verification.json").exists()


@pytest.mark.parametrize("change", ["model", "library", "point", "teacher", "geometry", "pose", "quality", "identity"])
def test_changed_numeric_geometry_model_runtime_or_review_cannot_be_derived(bundle, change):
    item = next(iter(bundle.refreshed["windows"].values()))
    target = Path(item["target_window"])
    if change == "model":
        bundle.model.write_bytes(b"different neutral model")
    elif change == "library":
        bundle.contracts.write_bytes(b"changed runtime contract")
    elif change == "point":
        (target / "ptv3.pth").write_bytes(b"changed numeric point tensors")
    elif change == "teacher":
        path = target / "teacher/track.pt"
        value = torch.load(path, weights_only=False)
        value["body_pose"] += 1
        torch.save(value, path)
    elif change == "geometry":
        path = bundle.case.formal / "source/calibration/calibration_snapshot.json"
        value = audit.read_json(path)
        value["court"] = "changed"
        write(path, value)
    elif change == "pose":
        path = next((bundle.case.formal / "repair/native_poses").glob("*/*/*.json"))
        value = audit.read_json(path)
        value["keypoints"] = []
        write(path, value)
    elif change == "quality":
        value = audit.read_rows(target / "quality.jsonl")
        value[0]["qa"]["reprojection_mean_px"] += 1
        rows(target / "quality.jsonl", value)
    else:
        path = bundle.case.formal / "repair/identity_evidence/identity_decisions.json"
        value = audit.read_json(path)
        next(iter(value["windows"].values()))["links"] = [[1, 2]]
        write(path, value)
    try:
        result = prepare(bundle)
    except (ValueError, FileNotFoundError, KeyError):
        return
    native_path = bundle.case.formal / "repair/final_labels/native_verification.json"
    if native_path.exists():
        native = audit.read_json(native_path)
        assert native["passed"] is False
    else:
        assert result["state"] == "actual_native_check_required"


@pytest.mark.parametrize(
    "change", ["proof_sha", "missing_origin", "maximums", "teacher_fingerprint", "missing_proof", "missing_witness"]
)
def test_guard_validation_does_not_trust_a_derived_passed_flag(bundle, change):
    result = prepare(bundle)
    native = audit.read_json(bundle.case.formal / "repair/final_labels/native_verification.json")
    entry = next(iter(native["windows"].values()))
    if change == "proof_sha":
        native["derived_native_proof"]["sha256"] = "f" * 64
    elif change == "missing_origin":
        entry.pop("verification_origin")
    elif change == "maximums":
        entry["body"]["maximums"]["rebuilt_mesh_error_m"] = 0.1
    elif change == "teacher_fingerprint":
        entry["teacher_inputs_sha256"] = {}
    elif change == "missing_proof":
        native.pop("derived_native_proof")
    else:
        Path(next(iter(bundle.parent_witnesses.values()))["path"]).write_bytes(b"{}")
    native["passed"] = True
    with pytest.raises((ValueError, FileNotFoundError, KeyError)):
        derived.validate_native_derivations(bundle.case.formal / "repair", native)


def test_removed_derived_origin_cannot_be_reclassified_as_actual_without_a_real_postleg_change(bundle):
    prepare(bundle)
    native = audit.read_json(bundle.case.formal / "repair/final_labels/native_verification.json")
    next(iter(native["windows"].values())).pop("verification_origin")
    with pytest.raises(ValueError):
        derived.validate_native_derivations(bundle.case.formal / "repair", native, postleg=True)


def test_guard_native_gate_checks_the_chain_and_not_only_the_passed_flag(bundle):
    from hmr4d.lidar_mocap.guarded_leg_gpu import _complete_native

    prepare(bundle)
    native = audit.read_json(bundle.case.formal / "repair/final_labels/native_verification.json")
    _complete_native(native, set(native["windows"]), repair=bundle.case.formal / "repair")
    next(iter(native["windows"].values()))["body"]["maximums"]["pelvis_error_m"] = 0.1
    with pytest.raises(ValueError):
        _complete_native(native, set(native["windows"]), repair=bundle.case.formal / "repair")


@pytest.mark.parametrize("damage", [None, "missing_actual_witness", "missing_actual_model_read"])
def test_postleg_really_changed_and_reverified_window_can_replace_only_its_derived_entry(bundle, monkeypatch, damage):
    prepare(bundle)
    path = bundle.case.formal / "repair/final_labels/native_verification.json"
    native = audit.read_json(path)
    key = next(iter(native["windows"]))
    report_path = Path(bundle.refreshed["windows"][key]["target_window"]) / "fit_report.json"
    report = audit.read_json(report_path)
    report["post_leg_refinement"] = dict(selection_sha256="a" * 64, frames=1)
    write(report_path, report)
    registry = bundle.case.formal / "repair/final_labels/source_registry.json"
    registry.write_bytes((bundle.case.formal / "source/source_registry.json").read_bytes())

    def actual_check(entry, run_name):
        # Only a synthetic substitute for the real unchanged verifier in this
        # temporary fixture; it observes the actual model open and new report.
        with open(bundle.model, "rb") as handle:
            handle.read()
        current = audit.read_json(path)
        fresh = deepcopy(native["windows"][key])
        fresh.pop("verification_origin")
        fresh["body"]["report_sha256"] = audit.file_sha(report_path)
        current["windows"][key] = fresh
        current.update(passed=True, state="complete", verified_windows=3, total_windows=3, accepted_checked=3)
        write(path, current)

    monkeypatch.setattr(witness.runpy, "run_path", actual_check)
    final = derived.run_required_actual_native(bundle.case.formal / "repair", postleg=True)
    assert final["verification_coverage"]["actual_window_keys"] == [key]
    assert len(final["verification_coverage"]["derived_window_keys"]) == 2
    assert final["windows"][key]["verification_origin"]["mode"] == "actual"
    derived.validate_native_derivations(bundle.case.formal / "repair", final, postleg=True)
    if damage == "missing_actual_witness":
        final["actual_native_witnesses"] = []
    elif damage == "missing_actual_model_read":
        item = final["windows"][key]["verification_origin"]["witness"]
        value = audit.read_json(item["path"])
        value["actual_model_file_reads"] = []
        write(Path(item["path"]), value)
        changed = witness._binding(item["path"])
        final["windows"][key]["verification_origin"]["witness"] = changed
        final["actual_native_witnesses"] = [changed]
    if damage:
        with pytest.raises(ValueError):
            derived.validate_native_derivations(bundle.case.formal / "repair", final, postleg=True)
