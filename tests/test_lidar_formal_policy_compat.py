"""Source-bound producer metadata and frozen CPU orchestration; no model/production assets."""

import ast
from copy import deepcopy
import inspect
from pathlib import Path

import pytest

from hmr4d.lidar_mocap import formal_completion as frozen
from hmr4d.lidar_mocap import formal_completion_policy_compat as compat
from hmr4d.lidar_mocap import new_source_completion as consumer
from test_lidar_formal_baseline import stage, put, read, bytes_file
from test_lidar_formal_completion import pending, finished

CHECK_TESTS = compat._tested_binding


def rebind(value):
    """Keep synthetic receipts byte-bound after deliberately changing metadata."""
    proposal = put(Path(value.proposal["path"]), value.proposals)
    value.selected["source_proposals_sha256"] = proposal["sha256"]
    selection = put(Path(value.selection["path"]), value.selected)
    value.applied["selection_sha256"] = selection["sha256"]
    for key, item in value.applied["windows"].items():
        path = value.current_dirs[key] / "fit_report.json"
        report = read(path)
        report["post_leg_refinement"].update(
            selection_sha256=selection["sha256"], policy_sha256=value.applied["policy_sha256"]
        )
        item["fit_report_sha256"] = put(path, report)["sha256"]
    applied = put(Path(value.apply["path"]), value.applied)
    put(Path(value.post["path"]), dict(value.applied, source="bounded_native_leg_correspondence"))
    value.request["scope"][0]["configuration"] = put(value.config_path, value.configuration)
    request = put(Path(value.execution["request"]["path"]), value.request)
    ready = read(value.ready["path"])
    ready.update(request=request, configuration=value.request["scope"][0]["configuration"])
    grant = read(value.grant["path"])
    grant.update(request=request, request_sha256=request["sha256"], readiness=put(Path(value.ready["path"]), ready))
    grant_binding = put(Path(value.grant["path"]), grant)
    context_path = Path(value.context["path"])
    context = read(context_path)
    preflight = read(context["preflight"]["path"])
    preflight.update(request=request, grant=grant_binding, readiness=grant["readiness"])
    context.update(request=request, grant=grant_binding, preflight=put(Path(context["preflight"]["path"]), preflight))
    context_binding = put(context_path, context)
    outputs = dict(propose=proposal, select=selection, apply=applied)
    for phase, info in value.execution["phases"].items():
        info["output"] = outputs[phase]
        put(context_path.parent / (phase + ".exit.json"), info)
    put(
        context_path.parent / "all_phases_complete.json",
        dict(phases=value.execution["phases"], context=context_binding),
    )
    for role in ("actual_exit", "independent_guard"):
        item = value.execution[role]
        data = read(item["path"])
        data["context_sha256"] = context_binding["sha256"]
        value.execution[role] = put(Path(item["path"]), data)
    value.execution.update(request=request, grant=grant_binding)
    value.execution_binding = put(value.execution_path, value.execution)


@pytest.fixture
def producer_finished(finished, monkeypatch):
    value = finished
    project = value.root.parent.parent
    monkeypatch.setattr(compat, "PROJ_ROOT", project)
    policies = {}
    for role, relative in (
        ("refinement_policy", "hmr4d/lidar_mocap/leg_refinement.py"),
        ("selection_policy", "hmr4d/lidar_mocap/leg_selection.py"),
        ("application_entry", "tools/apply_badminton_leg_refinement.py"),
    ):
        path = project / relative
        bytes_file(path, ("# distinct synthetic source: " + role + "\n").encode())
        policies[role] = compat._binding(path)
    assert len({item["sha256"] for item in policies.values()}) == 3
    for role, path in compat.compatibility_files().items():
        bytes_file(path, ("# synthetic compatibility source " + role + "\n").encode())
    value.request = read(value.execution["request"]["path"])
    scope = value.request["scope"][0]
    value.config_path = Path(scope["configuration"]["path"])
    value.configuration = read(value.config_path)
    value.configuration["code_and_inputs"] = deepcopy(policies)
    scope["existing_input_and_code_bindings"] = deepcopy(policies)
    scope["commands"][2]["entry_binding"] = policies["application_entry"]
    value.proposals = read(value.proposal["path"])
    value.proposals.update(policy_sha256=policies["refinement_policy"]["sha256"], training_published=False)
    value.selected = read(value.selection["path"])
    value.selected.update(
        refinement_policy_sha256=policies["refinement_policy"]["sha256"],
        selection_policy_sha256=policies["selection_policy"]["sha256"],
        training_published=False,
    )
    value.applied = read(value.apply["path"])
    value.applied["policy_sha256"] = policies["application_entry"]["sha256"]
    value.policies = policies
    raw = put(project / "synthetic_mock_cpu_test.json", dict(synthetic_mock_only=True))
    value.tested = put(project / "synthetic_mock_implementation.json", dict(test_execution=raw))
    monkeypatch.setattr(compat, "_tested_binding", lambda item: (compat._bound(item, {})[0] and item))
    monkeypatch.setattr(compat, "_run_cpu_check", value.worker)
    monkeypatch.setattr(
        compat,
        "_run_dataset_check",
        lambda checked, directory, threads: value.worker(checked, "dataset", directory, threads),
    )
    rebind(value)
    value.verify_compat = lambda: compat.verify_formal_postleg_cpu(
        value.root, value.baseline_binding, value.execution_binding, compatibility_test_execution=value.tested
    )
    return value


def test_distinct_actual_producer_fingerprints_execute_original_cpu_pipeline_and_reuse(producer_finished):
    value = producer_finished
    receipt = value.verify_compat()
    assert value.native_calls == ["native", "dataset"]
    assert receipt["state"] == frozen.STATE and receipt["accepted"] == 2
    proof = read(receipt["evidence"][compat.PROOF_ROLE]["path"])
    assert proof["producers"] == value.policies
    assert proof["state"] == "metadata_checked_native_qa_pending" and proof["native_qa_complete"] is False
    assert proof["selected_frames"] == proof["updated_frames"] == proof["changed_windows"] == 1
    assert receipt == value.verify_compat()
    assert value.native_calls == ["native", "dataset"]
    reused, _ = consumer._reuse_formal_postleg(
        value.repair,
        read(value.repair / "final_labels/repair_plan.json"),
        value.repair / "evaluation/formal_postleg_cpu_receipt.json",
        compat._sha(value.repair / "evaluation/formal_postleg_cpu_receipt.json"),
        1,
    )
    assert reused == receipt and value.native_calls == ["native", "dataset"]


def test_metadata_inspection_does_not_start_workers_or_claim_native_completion(producer_finished):
    value = producer_finished
    material = compat.inspect_policy_metadata(value.root, value.baseline_binding, value.execution_binding, value.tested)
    assert material["native_qa_complete"] is False and material["gpu_started"] is False
    assert value.native_calls == []
    assert not (value.repair / "evaluation/formal_postleg_cpu_receipt.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        "fake_same_sha",
        "wrong_apply",
        "wrong_refinement",
        "wrong_selection",
        "wrong_source_path",
        "selected_count",
        "updated_count",
        "body_count",
        "training",
        "partial",
        "changed_source",
        "wrong_apply_entry",
        "job_busy",
    ],
)
def test_wrong_producer_count_flags_or_exit_cannot_start_native_workers(producer_finished, change):
    value = producer_finished
    if change == "fake_same_sha":
        value.proposals["policy_sha256"] = value.selected["refinement_policy_sha256"] = "b" * 64
        value.applied["policy_sha256"] = "b" * 64
    elif change == "wrong_apply":
        value.applied["policy_sha256"] = value.policies["refinement_policy"]["sha256"]
    elif change == "wrong_refinement":
        value.selected["refinement_policy_sha256"] = value.policies["application_entry"]["sha256"]
    elif change == "wrong_selection":
        value.selected["selection_policy_sha256"] = "f" * 64
    elif change in ("wrong_source_path", "wrong_apply_entry"):
        role = "refinement_policy" if change == "wrong_source_path" else "application_entry"
        value.request["scope"][0]["existing_input_and_code_bindings"][role] = value.policies["selection_policy"]
        value.configuration["code_and_inputs"][role] = value.policies["selection_policy"]
    elif change == "selected_count":
        value.selected["selected_frames"] = 2
    elif change == "updated_count":
        value.applied["updated_frames"] = 0
    elif change == "body_count":
        value.applied["body_qa_accepted"] = 3
    elif change == "training":
        value.applied["training_published"] = True
    elif change == "partial":
        value.applied["state"] = "applying"
    elif change == "changed_source":
        Path(value.policies["refinement_policy"]["path"]).write_bytes(b"changed source")
    rebind(value)
    if change == "job_busy":
        item = value.execution["actual_exit"]
        data = read(item["path"])
        data["job_empty"] = False
        value.execution["actual_exit"] = put(Path(item["path"]), data)
        value.execution_binding = put(value.execution_path, value.execution)
    with pytest.raises(ValueError):
        value.verify_compat()
    assert value.native_calls == []


@pytest.mark.parametrize("change", ["missing", "flag", "wrong_code", "wrong_test", "extra_field"])
def test_finish_requires_full_exact_typed_compatibility_proof(producer_finished, change):
    value = producer_finished
    receipt = value.verify_compat()
    receipt_path = value.repair / "evaluation/formal_postleg_cpu_receipt.json"
    if change == "missing":
        with pytest.raises(ValueError):
            compat.validate_compatibility_receipt(value.root, dict(receipt, evidence={}))
        return
    proof_item = receipt["evidence"][compat.PROOF_ROLE]
    proof = read(proof_item["path"])
    if change == "flag":
        proof["native_qa_complete"] = True
    elif change == "wrong_code":
        proof["compatibility_code"]["checker"]["sha256"] = "f" * 64
    elif change == "wrong_test":
        proof["test_execution"] = dict(path=str(value.tested["path"]), sha256="f" * 64)
    else:
        proof["untyped"] = True
    receipt["evidence"][compat.PROOF_ROLE] = put(Path(proof_item["path"]), proof)
    item = put(receipt_path, receipt)
    with pytest.raises(ValueError):
        consumer._reuse_formal_postleg(
            value.repair, read(value.repair / "final_labels/repair_plan.json"), receipt_path, item["sha256"], 1
        )
    assert value.native_calls == ["native", "dataset"]


def test_original_legacy_receipt_still_dispatches_original_checker(finished):
    receipt = finished.verify()
    path = finished.repair / "evaluation/formal_postleg_cpu_receipt.json"
    assert compat.PROOF_ROLE not in receipt["evidence"]
    reused, _ = consumer._reuse_formal_postleg(
        finished.repair, read(finished.repair / "final_labels/repair_plan.json"), path, compat._sha(path), 1
    )
    assert reused == receipt and finished.native_calls == ["native", "dataset"]


def test_execution_preconditions_ast_and_numeric_helper_objects_are_frozen():
    assert ast.dump(ast.parse(inspect.getsource(compat._execution))) == ast.dump(
        ast.parse(inspect.getsource(frozen._execution))
    )
    for name in ("_baseline", "_gone", "_artifacts", "_native_after", "_dataset_after", "_qa_scope", "_unchanged"):
        assert getattr(compat, name) is getattr(frozen, name)


def durable_bundle(value, change=None):
    stdout = value.root / "test.log"
    junit = value.root / "test.xml"
    bytes_file(stdout, b"synthetic test output\n")
    bytes_file(junit, b"<synthetic/>\n")
    proof = dict(
        schema="gvhmr-formal-policy-compat-cpu-test-v1",
        state="passed",
        exit_code=0,
        passed=1,
        failed=0,
        fixture_only=False,
        gpu_started=False,
        actual_affinity=[26, 27],
        cuda_visible_devices="-1",
        code={role: compat._binding(path) for role, path in compat.compatibility_files().items()},
        stdout=compat._binding(stdout),
        junit=compat._binding(junit),
        source_bytes_unchanged=True,
        test_files=[compat._binding(compat.compatibility_files()["focused_tests"])],
    )
    if change == "fixture":
        proof["fixture_only"] = True
    elif change == "failed":
        proof["failed"] = 1
    elif change == "wrong_code":
        proof["code"]["entry"]["sha256"] = "f" * 64
    elif change == "wrong_allocation":
        proof["actual_affinity"] = [0, 1]
    elif change == "gpu":
        proof["gpu_started"] = True
    folder = value.root / "durable"

    def copy(item, name):
        target = folder / name
        bytes_file(target, Path(item["path"]).read_bytes())
        return compat._binding(target)

    raw = put(value.root / "test_receipt.json", proof)
    original = dict(path=str(Path(frozen.__file__).resolve()), sha256=compat.FROZEN_SHA256)
    implementation = dict(
        schema="gvhmr-formal-policy-compat-implementation-v1",
        code=proof["code"],
        code_archive={
            role: copy(compat._binding(path), "code/" + role + ".py")
            for role, path in compat.compatibility_files().items()
        },
        test_execution=copy(raw, "actual_test_execution.json"),
        test_outputs={role: copy(proof[role], role + ".bin") for role in ("stdout", "junit")},
        test_sources=[dict(original=item, archived=copy(item, "tested_source.py")) for item in proof["test_files"]],
        original_checker=dict(original=original, archived=copy(original, "frozen_formal_completion.py")),
    )
    return put(folder / "implementation.json", implementation)


@pytest.mark.parametrize("change", ["fixture", "failed", "wrong_code", "wrong_allocation", "gpu"])
def test_actual_test_evidence_requires_exact_code_and_cpu_isolation(producer_finished, change):
    with pytest.raises(ValueError):
        CHECK_TESTS(durable_bundle(producer_finished, change))


def test_durable_actual_evidence_does_not_read_original_temporary_outputs(producer_finished):
    value = producer_finished
    item = durable_bundle(value)
    (value.root / "test.log").unlink()
    (value.root / "test.xml").unlink()
    (value.root / "test_receipt.json").unlink()
    assert CHECK_TESTS(item) == item
    copied = Path(read(item["path"])["test_outputs"]["stdout"]["path"])
    copied.write_bytes(b"changed permanent evidence")
    with pytest.raises(ValueError):
        CHECK_TESTS(item)
