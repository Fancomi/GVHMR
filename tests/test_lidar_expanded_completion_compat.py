"""Real typed compatibility proof at the assembly bridge; synthetic CPU metadata only."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from hmr4d.lidar_mocap import expanded_dataset as expanded
from hmr4d.lidar_mocap import formal_completion as frozen
from hmr4d.lidar_mocap import formal_completion_policy_compat as compat
from test_lidar_formal_baseline import stage as original_stage, put, read, bytes_file
from test_lidar_formal_completion import pending, finished
from test_lidar_formal_policy_compat import producer_finished, durable_bundle, CHECK_TESTS
from test_lidar_formal_dataset_compat import resumable

LEGACY_ROLES = {"before", "native", "dataset", "post_refinement", "leg_receipt", "after", "comparison"}


@pytest.fixture
def stage(tmp_path, monkeypatch):
    """Bind the real formal source/registry chain before the original baseline is frozen."""
    value = original_stage.__wrapped__(tmp_path, monkeypatch)
    registry = put(
        value.repair / "final_labels/source_registry.json",
        dict(schema="badminton-formal-derived-source-registry-v1", sources={k.split(":")[0]: {} for k in value.keys}),
    )
    source = read(value.source["path"])
    source["source_registry_sha256"] = registry["sha256"]
    value.source = put(Path(value.source["path"]), source)
    plan = read(value.plan["path"])
    plan["source_union_sha256"] = value.source["sha256"]
    value.plan = put(Path(value.plan["path"]), plan)
    for name in ("native", "poses", "identity"):
        item = getattr(value, name)
        document = read(item["path"])
        document["source_union_sha256"] = value.source["sha256"]
        if name == "identity":
            document["evidence_files"][value.poses["path"]] = value.poses["sha256"]
        setattr(value, name, put(Path(item["path"]), document))
    value.imported["source_union_sha256"] = value.source["sha256"]
    put(value.import_path, value.imported)
    return value


def publication_bridge(value, receipt, monkeypatch, *, source_union=False):
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
    if source_union:
        body.pop("source_registry_sha256")
        body.update(schema="badminton-ptv3-dataset-v1", source_union_manifest_sha256=value.source["sha256"])
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
        if worker.get("state") == "reused_completed_native_witness":
            continue
        name = "verify_badminton_native_bound.py" if role == "native" else "verify_badminton_dataset.py"
        entry = project / "tools" / name
        if role == "dataset" and compat.PROOF_ROLE in receipt["evidence"]:
            proof = read(receipt["evidence"][compat.PROOF_ROLE]["path"])
            entry = Path(proof["compatibility_code"]["dataset_entry"]["path"])
        else:
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

    def rebind_body(document):
        body_binding = put(body_path, document)
        observed = read(observations["path"])
        observed["identity"]["source"]["dataset_manifest"] = body_binding
        observations.update(put(Path(observations["path"]), observed))
        publication.update(
            source_dataset_manifest_sha256=body_binding["sha256"],
            full_observation_manifest_sha256=observations["sha256"],
        )
        parent.update(put(Path(parent["path"]), publication))
        for item in (body_binding, observations, parent):
            completion["evidence"][item["path"]] = item["sha256"]
        rebind()

    def forbidden(*args, **kwargs):
        raise AssertionError("Assembly metadata bridge must never re-run a full completion or CPU worker")

    for module in (frozen, compat):
        monkeypatch.setattr(module, "verify_formal_postleg_cpu", forbidden)
        monkeypatch.setattr(module, "_run_cpu_check", forbidden)
        monkeypatch.setattr(module, "_baseline", forbidden)
        monkeypatch.setattr(module, "_native_after", forbidden)
        monkeypatch.setattr(module, "_dataset_after", forbidden)
    monkeypatch.setattr(compat, "_run_dataset_check", forbidden)
    rebind()
    return SimpleNamespace(
        value=value,
        receipt=receipt,
        completion=completion,
        rebind=rebind,
        validate=validate,
        body_path=body_path,
        rebind_body=rebind_body,
        registry=registry,
        publication=publication,
    )


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


@pytest.fixture
def union_bridge(finished, monkeypatch):
    return publication_bridge(finished, finished.verify(), monkeypatch, source_union=True)


@pytest.fixture
def resumed_bridge(resumable, monkeypatch):
    value = resumable
    value.tested = durable_bundle(value)
    monkeypatch.setattr(compat, "_tested_binding", CHECK_TESTS)
    receipt = value.resume_verify()
    return publication_bridge(value, receipt, monkeypatch, source_union=True)


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


def test_real_postleg_writer_uses_bound_source_union_then_registry(union_bridge):
    value = union_bridge
    body = read(value.body_path)
    assert "source_registry_sha256" not in body
    assert body["source_union_manifest_sha256"] == value.value.source["sha256"]
    assert read(value.value.source["path"])["source_registry_sha256"] == value.registry["sha256"]
    dependencies = value.validate()
    assert value.value.source in dependencies and value.registry in dependencies


@pytest.mark.parametrize("change", ["missing_union", "wrong_union", "schema", "direct_none", "direct_wrong"])
def test_rehashed_body_cannot_bypass_either_real_writer_link(union_bridge, change):
    value = union_bridge
    body = read(value.body_path)
    if change == "missing_union":
        body.pop("source_union_manifest_sha256")
    elif change == "wrong_union":
        body["source_union_manifest_sha256"] = "f" * 64
    elif change == "schema":
        body["schema"] = "unknown-body-writer"
    else:
        body["source_registry_sha256"] = None if change == "direct_none" else "f" * 64
    value.rebind_body(body)
    with pytest.raises(ValueError, match="source/calibration/window/body scope"):
        value.validate()


@pytest.mark.parametrize("bad", [None, "f" * 64])
def test_rehashed_source_input_must_bind_the_exact_existing_registry(union_bridge, bad):
    value = union_bridge
    source = read(value.value.source["path"])
    if bad is None:
        source.pop("source_registry_sha256")
    else:
        source["source_registry_sha256"] = bad
    selected = put(Path(value.value.source["path"]), source)
    plan_path = value.value.repair / "final_labels/repair_plan.json"
    plan = read(plan_path)
    plan["source_union_sha256"] = selected["sha256"]
    put(plan_path, plan)
    value.receipt["source_input"] = selected
    body = read(value.body_path)
    body["source_union_manifest_sha256"] = selected["sha256"]
    value.rebind_body(body)
    with pytest.raises(ValueError, match="source/calibration/window/body scope"):
        value.validate()


def test_true_native_resume_and_proof_bound_dataset_entry_close_all_dependencies(resumed_bridge):
    value = resumed_bridge
    worker = value.receipt["workers"]["native"]
    assert worker["state"] == "reused_completed_native_witness" and "exit_code" not in worker
    assert value.receipt["workers"]["dataset"]["entry"]["path"].endswith("verify_badminton_formal_dataset.py")
    dependencies = value.validate()
    resume = read(worker["resume"]["path"])
    previous = read(resume["previous_implementation"]["path"])
    assert all(item in dependencies for item in (worker["entry"], worker["log"], worker["resume"]))
    assert all(item in dependencies for item in worker["actual_native_witnesses"])
    assert all(item in dependencies for item in resume["evidence"].values())
    assert resume["exit_readback"] in dependencies and resume["previous_implementation"] in dependencies
    assert all(item in dependencies for item in previous["code_archive"].values())
    assert previous["test_execution"] in dependencies
    assert value.value.native_calls == ["dataset"] and value.value.refresh == [True]


@pytest.mark.parametrize("change", ["state_only", "fake_exit", "missing_proof", "wrong_witness", "missing_witness"])
def test_resume_is_never_a_state_or_fabricated_exit_shortcut(resumed_bridge, change):
    value = resumed_bridge
    worker = value.receipt["workers"]["native"]
    if change == "state_only":
        value.receipt["workers"]["native"] = dict(state="reused_completed_native_witness")
    elif change == "fake_exit":
        worker["exit_code"] = 0
    elif change == "missing_proof":
        value.receipt["evidence"].pop(compat.PROOF_ROLE)
    elif change == "wrong_witness":
        worker["actual_native_witnesses"] = []
    else:
        Path(worker["actual_native_witnesses"][0]["path"]).unlink()
    value.rebind()
    with pytest.raises((ValueError, FileNotFoundError)):
        value.validate()


@pytest.mark.parametrize("change", ["root", "baseline", "execution", "native_result", "exit_identity", "native_log"])
def test_real_native_resume_validator_rejects_rehashed_bad_predecessor_metadata(resumed_bridge, change):
    value = resumed_bridge
    worker = value.receipt["workers"]["native"]
    resume = read(worker["resume"]["path"])
    if change == "root":
        resume["root"] = str(value.value.root / "elsewhere")
    elif change in ("baseline", "execution"):
        resume["baseline" if change == "baseline" else "leg_execution"]["sha256"] = "f" * 64
    elif change == "native_result":
        resume["evidence"]["native_result"]["sha256"] = "f" * 64
    elif change == "exit_identity":
        exited = read(resume["exit_readback"]["path"])
        exited["process_identities"].pop()
        resume["exit_readback"] = put(Path(resume["exit_readback"]["path"]), exited)
    else:
        path = Path(resume["evidence"]["native_log"]["path"])
        bytes_file(path, b"wrong native allocation/result log\n")
        resume["evidence"]["native_log"] = compat._binding(path)
        worker["log"] = resume["evidence"]["native_log"]
    worker["resume"] = put(Path(worker["resume"]["path"]), resume)
    value.rebind()
    with pytest.raises(ValueError):
        value.validate()


@pytest.mark.parametrize("role", ["prior_archive", "prior_test_output", "witness"])
def test_reused_native_metadata_leaves_remain_bound_until_assembly_finishes(resumed_bridge, role):
    value = resumed_bridge
    dependencies = value.validate()
    worker = value.receipt["workers"]["native"]
    resume = read(worker["resume"]["path"])
    previous = read(resume["previous_implementation"]["path"])
    selected = (
        next(iter(previous["code_archive"].values()))
        if role == "prior_archive"
        else previous["test_outputs"]["stdout"] if role == "prior_test_output" else worker["actual_native_witnesses"][0]
    )
    assert selected in dependencies
    Path(selected["path"]).write_bytes(b"changed after verified native metadata reuse\n")
    with pytest.raises(ValueError, match="Frozen dependency changed"):
        for item in dependencies:
            expanded.check_binding(item)


@pytest.mark.parametrize(
    "change", ["no_proof", "other_entry", "wrong_sha", "nonzero", "bool_exit", "allocation", "hidden", "log"]
)
def test_dataset_adapter_requires_actual_typed_entry_and_original_cpu_constraints(compatibility_bridge, change):
    value = compatibility_bridge
    worker = value.receipt["workers"]["dataset"]
    if change == "no_proof":
        value.receipt["evidence"].pop(compat.PROOF_ROLE)
    elif change == "other_entry":
        path = value.value.root / "elsewhere/verify_badminton_formal_dataset.py"
        bytes_file(path, Path(worker["entry"]["path"]).read_bytes())
        worker["entry"] = compat._binding(path)
    elif change == "wrong_sha":
        worker["entry"]["sha256"] = "f" * 64
    elif change in ("nonzero", "bool_exit"):
        worker["exit_code"] = 1 if change == "nonzero" else False
    elif change == "allocation":
        worker["allocation"]["affinity"] = [0, 1]
    elif change == "hidden":
        worker["hidden_window"] = False
    else:
        path = value.value.root / "elsewhere/dataset.log"
        bytes_file(path, Path(worker["log"]["path"]).read_bytes())
        worker["log"] = compat._binding(path)
    value.rebind()
    with pytest.raises(ValueError):
        value.validate()
