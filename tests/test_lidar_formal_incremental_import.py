"""Exercise the unchanged numerical audit with one policy-bound synthetic parent."""

from types import SimpleNamespace

import pytest

from hmr4d.lidar_mocap import formal_import_audit as audit, incremental_release as inc
from test_lidar_formal_import_audit import case, write, run


def single_parent(case, monkeypatch):
    roles = dict(train=["a"], val=[], test=[])
    policy_path = case.formal.parent / "verified_policy.json"
    write(policy_path, dict(schema=inc.POLICY, new_source_splits=roles))
    selected = audit.binding(policy_path)
    policy = SimpleNamespace(
        binding=selected,
        roles=roles,
        required=["train"],
        recheck=lambda: None,
        scope=dict(review_sources=dict(a=dict(review_root=str(case.parents[0])))),
    )
    # Policy/parent JSON validation has separate real-byte regressions. This stand-in
    # isolates the existing full numerical/teacher/native audit from metadata setup.
    monkeypatch.setattr(
        inc, "policy_from_binding", lambda item: policy if item == selected else pytest.fail("wrong policy")
    )
    intake_path = case.formal / "source/formal_intake.json"
    intake = audit.read_json(intake_path)
    intake.update(
        source_role_contract=selected,
        release_policy=selected,
        parent_review_manifests=intake["parent_review_manifests"][:1],
    )
    write(intake_path, intake)
    registry_path = case.formal / "source/source_registry.json"
    registry = audit.read_json(registry_path)
    registry.update(
        sources=dict(a=registry["sources"]["a"]), release_policy=selected, intake=audit.binding(intake_path)
    )
    write(registry_path, registry)
    cal_path = case.formal / "source/calibration/calibration_snapshot.json"
    cal = audit.read_json(cal_path)
    cal["sessions"] = dict(a=cal["sessions"]["a"])
    write(cal_path, cal)
    write(case.formal / "repair/final_labels/calibration/calibration_snapshot.json", cal)
    write(case.formal / "source/session_splits.json", roles)
    proof_path = case.formal / "formal_input_equivalence.json"
    proof = audit.read_json(proof_path)
    proof.update(role_assignment=roles, sources=dict(a=proof["sources"]["a"]))
    proof["evidence"].update(
        role_contract=selected,
        release_policy=selected,
        formal_registry=audit.binding(registry_path),
        formal_calibration=audit.binding(cal_path),
    )
    write(proof_path, proof)
    case.proof = audit.binding(proof_path)
    input_path = case.formal / "repair/new_source_input.json"
    source = audit.read_json(input_path)
    source.update(
        jobs=[job for job in source["jobs"] if job["source_id"] == "a"],
        session_splits=roles,
        required_nonempty_splits=["train"],
        release_policy=selected,
        source_registry_sha256=audit.file_sha(registry_path),
    )
    source["formal_derivation"]["input_equivalence"] = case.proof
    write(input_path, source)
    body_path = case.formal / "repair/final_labels/repair_plan.json"
    body = audit.read_json(body_path)
    body.update(jobs=source["jobs"], source_union_sha256=audit.file_sha(input_path))
    write(body_path, body)
    return policy


def test_single_parent_runs_the_full_original_numerical_inventory(case, monkeypatch):
    single_parent(case, monkeypatch)
    result = audit.audit_formal_numeric_import(case.formal, case.proof, case.parents[:1])
    assert result["state"] == "eligible_parent_complete_not_imported"
    assert result["eligible_windows"] == result["eligible_accepted"] == 1
    assert result["eligible_fitted"] == 2 and set(result["windows"]) == {"a:0_1"}
    assert set(result["parent_evidence"]) == {str(case.parents[0])}
    assert result["teacher_files"] and result["fitted_tensors_imported"] is False


def test_single_parent_does_not_skip_original_native_completion_barrier(case, monkeypatch):
    single_parent(case, monkeypatch)
    native_path = case.parents[0] / "repair/final_labels/native_verification.json"
    native = audit.read_json(native_path)
    native["passed"] = False
    write(native_path, native)
    with pytest.raises(ValueError, match="Parent native verification"):
        audit.audit_formal_numeric_import(case.formal, case.proof, case.parents[:1])


def test_single_parent_cannot_select_a_foreign_complete_review_root(case, monkeypatch):
    single_parent(case, monkeypatch)
    with pytest.raises(ValueError, match="exact v2 policy"):
        audit.audit_formal_numeric_import(case.formal, case.proof, case.parents[1:])


def test_legacy_two_parent_path_does_not_load_an_incremental_policy(case, monkeypatch):
    monkeypatch.setattr(inc, "policy_from_binding", lambda *args: pytest.fail("Legacy cannot load a v2 policy"))
    result = run(case)
    assert result["eligible_windows"] == 3
    assert "release_policy" not in result
    assert set(result["parent_evidence"]) == set(map(str, case.parents))


def test_one_parent_without_explicit_v2_binding_remains_rejected(case):
    with pytest.raises(ValueError, match="Exactly two"):
        audit.audit_formal_numeric_import(case.formal, case.proof, case.parents[:1])
