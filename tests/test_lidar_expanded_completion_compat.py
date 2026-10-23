"""Real typed compatibility proof at the assembly bridge; synthetic CPU metadata only."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from hmr4d.lidar_mocap import expanded_dataset as expanded
from hmr4d.lidar_mocap import formal_completion as frozen
from hmr4d.lidar_mocap import formal_completion_policy_compat as compat
from test_lidar_formal_baseline import stage, put, read, bytes_file
from test_lidar_formal_completion import pending, finished
from test_lidar_formal_policy_compat import producer_finished, durable_bundle, CHECK_TESTS

LEGACY_ROLES = {"before", "native", "dataset", "post_refinement", "leg_receipt", "after", "comparison"}


def publication_bridge(value, receipt, monkeypatch):
    """Add genuine synthetic bilateral publication bindings to an original completed CPU fixture."""
    repair, labels = value.repair, value.repair / "final_labels"
    project = value.root.parent.parent
    monkeypatch.setattr(expanded, "PROJ_ROOT", project)
    registry = put(
        labels / "source_registry.json",
        dict(
            schema="badminton-formal-derived-source-registry-v1", sources={key.split(":")[0]: {} for key in value.keys}
        ),
    )
    body_path = labels / "datasets/dataset_manifest.json"
    body = read(body_path)
    body.update(calibration_digest=receipt["calibration_digest"], source_registry_sha256=registry["sha256"])
    body_binding = put(body_path, body)
    paired = repair.with_name("repair_paired")
    observations = put(
        paired / "observations/manifest.json",
        dict(
            completed=True,
            source_accepted=receipt["accepted"],
            identity=dict(calibration_digest=receipt["calibration_digest"], source=dict(dataset_manifest=body_binding)),
        ),
    )
    publication = dict(
        accepted=2,
        calibration_digest=receipt["calibration_digest"],
        exports=deepcopy(body["exports"]),
        source_dataset_manifest=str(body_path),
        source_dataset_manifest_sha256=body_binding["sha256"],
        full_observation_manifest=observations["path"],
        full_observation_manifest_sha256=observations["sha256"],
    )
    parent = put(paired / "datasets/dataset_manifest.json", publication)
    verification = put(paired / "verification.json", dict(accepted=2, sessions=deepcopy(body["exports"])))
    viewer = put(labels / "viewer/manifest.json", dict(calibration_digest=receipt["calibration_digest"]))
    allocation = dict(
        schema="gvhmr-formal-cpu-worker-v1", affinity=[26, 27], priority="BelowNormal", cuda_visible_devices="-1"
    )
    for role, worker in receipt["workers"].items():
        name = "verify_badminton_native_bound.py" if role == "native" else "verify_badminton_dataset.py"
        entry = project / "tools" / name
        bytes_file(entry, b"# Synthetic CPU worker metadata; never executed.\n")
        worker.update(entry=compat._binding(entry), allocation=deepcopy(allocation))
    receipt_path = repair / "evaluation/formal_postleg_cpu_receipt.json"
    selected = put(receipt_path, receipt)
    completion = dict(
        state="labels_published_viewer_prepared",
        training_published=False,
        bilateral_labels_published=True,
        archive_published=False,
        consumer_admitted=False,
        dataset_manifest=parent["path"],
        accepted=2,
        formal_postleg_cpu_receipt=selected,
        evidence={
            item["path"]: item["sha256"]
            for item in (parent, body_binding, observations, verification, viewer, selected)
        },
    )
    for role in ("before", "native", "leg_receipt", "after", "comparison"):
        item = receipt["evidence"][role]
        completion["evidence"][item["path"]] = item["sha256"]

    def rebind(changed=None):
        item = put(receipt_path, receipt if changed is None else changed)
        completion["formal_postleg_cpu_receipt"] = item
        completion["evidence"][item["path"]] = item["sha256"]
        put(repair / "new_source_completion_receipt.json", completion)

    def validate():
        return expanded._validate_new_completion(repair, completion, parent, publication)

    def forbidden(*args, **kwargs):
        raise AssertionError("Assembly metadata bridge must never re-run a full completion or CPU worker")

    for module in (frozen, compat):
        monkeypatch.setattr(module, "verify_formal_postleg_cpu", forbidden)
        monkeypatch.setattr(module, "_run_cpu_check", forbidden)
        monkeypatch.setattr(module, "_baseline", forbidden)
    rebind()
    return SimpleNamespace(value=value, receipt=receipt, completion=completion, rebind=rebind, validate=validate)


@pytest.fixture
def legacy_bridge(finished, monkeypatch):
    return publication_bridge(finished, finished.verify(), monkeypatch)


@pytest.fixture
def compatibility_bridge(producer_finished, monkeypatch):
    value = producer_finished
    value.tested = durable_bundle(value)
    monkeypatch.setattr(compat, "_tested_binding", CHECK_TESTS)
    receipt = value.verify_compat()
    return publication_bridge(value, receipt, monkeypatch)


def test_original_complete_seven_role_receipt_keeps_legacy_path(legacy_bridge, monkeypatch):
    value = legacy_bridge
    assert set(value.receipt["evidence"]) == LEGACY_ROLES
    monkeypatch.setattr(
        compat,
        "validate_compatibility_receipt",
        lambda *args: pytest.fail("Legacy completion must not select compatibility policy"),
    )
    assert value.validate()
    assert value.value.native_calls == ["native", "dataset"]


def test_complete_eighth_role_uses_real_source_bound_typed_validator(compatibility_bridge):
    value = compatibility_bridge
    assert set(value.receipt["evidence"]) == LEGACY_ROLES | {compat.PROOF_ROLE}
    dependencies = value.validate()
    proof = read(value.receipt["evidence"][compat.PROOF_ROLE]["path"])
    assert value.receipt["evidence"][compat.PROOF_ROLE] in dependencies
    assert proof["compatibility_implementation"] in dependencies
    assert all(item in dependencies for item in proof["producers"].values())
    assert all(item in dependencies for item in proof["changed_window_reports"].values())
    assert value.value.native_calls == ["native", "dataset"]


@pytest.mark.parametrize("change", ["flag", "code", "test", "producer", "phase", "report", "extra_field"])
def test_rehashed_proof_cannot_replace_actual_typed_material(compatibility_bridge, change):
    value = compatibility_bridge
    selected = value.receipt["evidence"][compat.PROOF_ROLE]
    proof = read(selected["path"])
    if change == "flag":
        proof["native_qa_complete"] = True
    elif change == "code":
        proof["compatibility_code"]["checker"]["sha256"] = "f" * 64
    elif change == "test":
        proof["test_execution"]["sha256"] = "f" * 64
    elif change == "producer":
        proof["producers"]["application_entry"] = proof["producers"]["refinement_policy"]
    elif change == "phase":
        proof["phase_outputs"]["apply"]["sha256"] = "f" * 64
    elif change == "report":
        next(iter(proof["changed_window_reports"].values()))["sha256"] = "f" * 64
    else:
        proof["untyped_success"] = True
    value.receipt["evidence"][compat.PROOF_ROLE] = put(Path(selected["path"]), proof)
    value.rebind()
    with pytest.raises(ValueError):
        value.validate()
    assert value.value.native_calls == ["native", "dataset"]


@pytest.mark.parametrize("change", ["missing_proof", "tamper", "wrong_path", "proof_flag", "extra_role"])
def test_invalid_proof_binding_or_extra_role_cannot_admit_completion(compatibility_bridge, change):
    value = compatibility_bridge
    selected = value.receipt["evidence"][compat.PROOF_ROLE]
    if change == "missing_proof":
        Path(selected["path"]).unlink()
    elif change == "tamper":
        Path(selected["path"]).write_text("{}")
    elif change == "wrong_path":
        value.receipt["evidence"][compat.PROOF_ROLE] = put(
            value.value.repair / "evaluation/wrong/proof.json", read(selected["path"])
        )
    elif change == "proof_flag":
        value.receipt["evidence"][compat.PROOF_ROLE] = True
    else:
        value.receipt["evidence"]["untyped_extra"] = selected
    value.rebind()
    with pytest.raises((ValueError, FileNotFoundError)):
        value.validate()
    assert value.value.native_calls == ["native", "dataset"]


def test_unknown_role_also_fails_legacy_without_loading_compatibility(legacy_bridge, monkeypatch):
    value = legacy_bridge
    value.receipt["evidence"]["untyped_extra"] = value.receipt["evidence"]["native"]
    value.rebind()
    monkeypatch.setattr(compat, "validate_compatibility_receipt", lambda *args: pytest.fail("Unexpected dispatch"))
    with pytest.raises(ValueError, match="verification evidence"):
        value.validate()


@pytest.mark.parametrize("change", ["producer_source", "compatibility_source", "test_output", "selected_report"])
def test_live_bound_files_cannot_change_behind_an_unchanged_proof(compatibility_bridge, change):
    value = compatibility_bridge
    proof = read(value.receipt["evidence"][compat.PROOF_ROLE]["path"])
    if change == "producer_source":
        item = proof["producers"]["application_entry"]
    elif change == "compatibility_source":
        item = proof["compatibility_code"]["checker"]
    elif change == "test_output":
        item = read(proof["compatibility_implementation"]["path"])["test_outputs"]["junit"]
    else:
        item = next(iter(proof["changed_window_reports"].values()))
    Path(item["path"]).write_bytes(b"changed synthetic bound bytes\n")
    with pytest.raises(ValueError):
        value.validate()
    assert value.value.native_calls == ["native", "dataset"]


def durable_files(value):
    proof = read(value.receipt["evidence"][compat.PROOF_ROLE]["path"])
    implementation = read(proof["compatibility_implementation"]["path"])
    return dict(
        code_archive=next(iter(implementation["code_archive"].values())),
        test_execution=implementation["test_execution"],
        test_outputs=implementation["test_outputs"]["stdout"],
        test_sources=implementation["test_sources"][0]["archived"],
        original_checker=implementation["original_checker"]["archived"],
    )


@pytest.mark.parametrize("role", ["code_archive", "test_execution", "test_outputs", "test_sources", "original_checker"])
def test_verified_durable_leaves_remain_bound_until_assembly_finishes(compatibility_bridge, role):
    value = compatibility_bridge
    dependencies = value.validate()
    leaves = durable_files(value)
    assert all(item in dependencies for item in leaves.values())
    Path(leaves[role]["path"]).write_bytes(b"changed durable bytes after metadata validation\n")
    with pytest.raises(ValueError, match="Frozen dependency changed"):
        for item in dependencies:
            expanded.check_binding(item)


def test_historical_temporary_test_outputs_are_not_reopened(compatibility_bridge):
    value = compatibility_bridge
    originals = [value.value.root / name for name in ("test.log", "test.xml", "test_receipt.json")]
    for path in originals:
        path.unlink()
    dependencies = value.validate()
    assert all(str(path) not in {item["path"] for item in dependencies} for path in originals)
    assert all(item in dependencies for item in durable_files(value).values())
