"""Explicit postleg producer-fingerprint compatibility; frozen native implementation stays unchanged.

Only historical metadata/orchestration is copied. Native, artifacts, body QA,
dataset verification and numerical comparisons use the frozen helpers directly.
"""

import json
from pathlib import Path
import re
import uuid

from hmr4d import PROJ_ROOT
from . import formal_completion as frozen
from .formal_completion import (
    STATE, _artifacts, _baseline, _binding, _bound, _count, _dataset_after, _encode, _exact_path,
    _gone, _load, _native_after, _preflight_target, _publish, _qa_scope, _run_cpu_check, _sha, _unchanged,
)

FROZEN_SHA256 = "7a641255a3d6ee53ba9b00b5ed9e8240b65b91be9a79e03a259cfc888a253c1c"
PROOF_ROLE = "policy_fingerprint_compatibility"
PROOF_SCHEMA = "gvhmr-formal-postleg-policy-fingerprint-compatibility-v1"


def compatibility_files():
    return {role: PROJ_ROOT / relative for role, relative in dict(
        checker="hmr4d/lidar_mocap/formal_completion_policy_compat.py",
        entry="tools/finalize_badminton_formal_cpu_compat.py",
        consumer="hmr4d/lidar_mocap/new_source_completion.py",
        focused_tests="tests/test_lidar_formal_policy_compat.py",
    ).items()}


def _assert_frozen():
    if _sha(frozen.__file__) != FROZEN_SHA256:
        raise ValueError("The frozen original formal completion implementation changed")


def _tested_binding(item):
    implementation, _ = _bound(item, {})
    expected = {role: _binding(path) for role, path in compatibility_files().items()}
    if (set(implementation) != {"schema", "code", "code_archive", "test_execution", "test_outputs",
                                "test_sources", "original_checker"}
            or implementation["schema"] != "gvhmr-formal-policy-compat-implementation-v1"
            or implementation["code"] != expected or set(implementation["code_archive"]) != set(expected)):
        raise ValueError("Use the exact durable compatibility implementation and actual CPU evidence")
    folder = Path(item["path"]).resolve().parent

    def archived(proof, sha):
        if (not Path(proof["path"]).resolve().is_relative_to(folder) or proof["sha256"] != sha
                or _binding(proof["path"]) != proof):
            raise ValueError("A permanent compatibility evidence copy changed or left its bundle")

    for role, proof in implementation["code_archive"].items():
        archived(proof, expected[role]["sha256"])
    original = implementation["original_checker"]
    if original["original"] != dict(path=str(Path(frozen.__file__).resolve()), sha256=FROZEN_SHA256):
        raise ValueError("The durable proof names another original frozen checker")
    archived(original["archived"], FROZEN_SHA256)
    archived(implementation["test_execution"], implementation["test_execution"]["sha256"])
    tested, _ = _bound(implementation["test_execution"], {})
    if (tested.get("schema") != "gvhmr-formal-policy-compat-cpu-test-v1" or tested.get("state") != "passed"
            or tested.get("exit_code") != 0 or type(tested.get("passed")) is not int or tested["passed"] < 1
            or tested.get("failed") != 0 or tested.get("fixture_only") is not False
            or tested.get("gpu_started") is not False or tested.get("actual_affinity") != [26, 27]
            or tested.get("cuda_visible_devices") != "-1" or tested.get("code") != expected
            or tested.get("source_bytes_unchanged") is not True or not tested.get("test_files")):
        raise ValueError("Actual isolated CPU compatibility tests are missing or bind different code")
    if set(implementation["test_outputs"]) != {"stdout", "junit"}:
        raise ValueError("Both actual durable CPU outputs are required")
    for role in ("stdout", "junit"):
        archived(implementation["test_outputs"][role], tested[role]["sha256"])
    if [entry["original"] for entry in implementation["test_sources"]] != tested["test_files"]:
        raise ValueError("Durable test source inventory differs from the actual run")
    for entry in implementation["test_sources"]:
        archived(entry["archived"], entry["original"]["sha256"])
    return item


def _resolve_tests(root, item):
    if item is None:
        receipt = _load(Path(root).resolve() / "repair/evaluation/formal_postleg_cpu_receipt.json", {})
        proof = _bound(receipt["evidence"][PROOF_ROLE], {})[0]
        item = proof["compatibility_implementation"]
    return _tested_binding(item)


def _policies(checked, execution, outputs):
    request, _ = _bound(execution["request"], checked["observed"])
    scope = request["scope"][0]
    configuration, _ = _bound(scope["configuration"], checked["observed"])
    bindings = scope.get("existing_input_and_code_bindings", {})
    if (len(request["scope"]) != 1 or scope.get("stage") != "guarded_leg"
            or Path(scope["root"]).resolve() != checked["repair"]
            or configuration.get("target_root") != str(checked["repair"])
            or configuration.get("algorithm_changed") is not False
            or configuration.get("code_and_inputs") != bindings):
        raise ValueError("Producer policies lack their exact executed request/configuration")
    policies = {}
    for role, relative in (
        ("refinement_policy", "hmr4d/lidar_mocap/leg_refinement.py"),
        ("selection_policy", "hmr4d/lidar_mocap/leg_selection.py"),
        ("application_entry", "tools/apply_badminton_leg_refinement.py"),
    ):
        expected = _binding(PROJ_ROOT / relative)
        if bindings.get(role) != expected:
            raise ValueError("An executed producer source binding changed")
        checked["observed"][expected["path"]] = expected["sha256"]
        policies[role] = expected
    commands = {row["phase"]: row for row in scope["commands"]}
    if commands["apply"]["entry_binding"] != policies["application_entry"]:
        raise ValueError("Application fingerprint differs from the actually executed entry")
    proposals, selection, applied = [outputs[phase] for phase in ("propose", "select", "apply")]
    if (proposals.get("policy_sha256") != policies["refinement_policy"]["sha256"]
            or selection.get("refinement_policy_sha256") != policies["refinement_policy"]["sha256"]
            or selection.get("selection_policy_sha256") != policies["selection_policy"]["sha256"]
            or applied.get("policy_sha256") != policies["application_entry"]["sha256"]):
        raise ValueError("Actual leg fingerprints do not match their distinct bound producer sources")
    if any(value.get("state") != "complete" or value.get("training_published") is not False
           for value in (proposals, selection, applied)):
        raise ValueError("Complete unpublished genuine phase metadata is required")
    return policies


def _metadata_material(checked, execution_binding, test_execution):
    """Read receipts and selected report JSON only; this does not prove native/QA completion."""
    _assert_frozen()
    test_execution = _tested_binding(test_execution)
    observed = checked["observed"]
    execution, _ = _bound(execution_binding, observed)
    attempt = Path(execution_binding["path"]).resolve().parent
    if (attempt.parent != checked["root"] / "gpu_guarded_leg/attempts"
            or re.fullmatch(r"[0-9a-f]{32}", attempt.name) is None
            or Path(execution_binding["path"]).name != "completion.json"
            or execution.get("schema") != "gvhmr-guarded-leg-execution-v1"
            or execution.get("state") != "complete_actual_cuda_exited"
            or execution.get("training_published") is not False
            or set(execution.get("phases", {})) != {"propose", "select", "apply"}):
        raise ValueError("Select the exact actually completed guarded-leg attempt")
    outputs, output_bindings = {}, {}
    for phase, relative in (("propose", "leg_refinement_proposals/receipt.json"),
                            ("select", "leg_refinement_selection.json"), ("apply", "leg_refinement_receipt.json")):
        item = execution["phases"][phase]["output"]
        if Path(item["path"]).resolve() != checked["repair"] / relative:
            raise ValueError("Actual phase output leaves its exact repair root")
        outputs[phase], _ = _bound(item, observed)
        output_bindings[phase] = item
    policies = _policies(checked, execution, outputs)
    proposals, selection, applied = [outputs[phase] for phase in ("propose", "select", "apply")]
    selected = {key: row for key, row in selection["windows"].items() if row["selected"]}
    total = sum(len(row["selected"]) for row in selected.values())
    if (set(selection["windows"]) != checked["keys"] or set(proposals["windows"]) != checked["keys"]
            or selection["source_union_sha256"] != checked["source_sha"]
            or proposals["source_union_sha256"] != checked["source_sha"]
            or selection["source_proposals_sha256"] != output_bindings["propose"]["sha256"]
            or applied["selection_sha256"] != output_bindings["select"]["sha256"]
            or any(_count(applied.get(key), key) != total for key in ("selected_frames", "updated_frames"))
            or _count(selection.get("selected_frames"), "selected frames") != total
            or _count(applied.get("body_qa_accepted"), "body accepted") != checked["accepted"]
            or set(applied["windows"]) != set(selected)):
        raise ValueError("Actual selected/update/window/body metadata counts differ")
    post_path = checked["repair"] / "final_labels/post_refinement.json"
    post = _load(post_path, observed)
    if any(post.get(key) != applied.get(key) for key in (
        "state", "selected_frames", "updated_frames", "selection_sha256", "policy_sha256", "windows",
        "training_published",
    )):
        raise ValueError("Actual post-refinement metadata differs from application output")
    reports = {}
    for key, row in selected.items():
        sid, name = key.split(":", 1)
        path = checked["repair"] / "final_labels/sessions" / sid / "windows" / name / "fit_report.json"
        report = _load(path, observed)
        item = applied["windows"][key]
        mark = report.get("post_leg_refinement", {})
        if (item.get("fit_report_sha256") != _sha(path) or item.get("frames") != len(row["selected"])
                or mark.get("policy_sha256") != policies["application_entry"]["sha256"]
                or mark.get("selection_sha256") != applied["selection_sha256"]
                or mark.get("frames") != item["frames"]):
            raise ValueError("Actual changed-window report producer/count differs")
        reports[key] = _binding(path)
    request, _ = _bound(execution["request"], observed)
    if request["scope"][0]["manifest"] != checked["baseline"]["source_input"]:
        raise ValueError("Executed source manifest differs from the immutable baseline")
    return dict(schema=PROOF_SCHEMA, state="metadata_checked_native_qa_pending", root=str(checked["root"]),
                baseline=checked["baseline_binding"],
                original_checker=dict(path=str(Path(frozen.__file__).resolve()), sha256=FROZEN_SHA256),
                compatibility_code={role: _binding(path) for role, path in compatibility_files().items()},
                compatibility_implementation=test_execution,
                test_execution=_bound(test_execution, {})[0]["test_execution"],
                leg_execution=execution_binding, request=execution["request"],
                configuration=request["scope"][0]["configuration"], producers=policies, phase_outputs=output_bindings,
                post_refinement=_binding(post_path), changed_window_reports=reports, selected_frames=total,
                updated_frames=total, changed_windows=len(selected), body_qa_accepted=checked["accepted"],
                native_qa_complete=False, gpu_started=False, training_published=False)


def inspect_policy_metadata(formal_root, baseline_binding, execution_binding, test_execution):
    """Cheap source-bound metadata inspection, without the original large-input predicate."""
    root = Path(formal_root).resolve()
    if root == PROJ_ROOT / "outputs" or not root.is_relative_to(PROJ_ROOT / "outputs"):
        raise ValueError("Select the exact formal outputs root")
    baseline, _ = _bound(baseline_binding, {})
    if (Path(baseline_binding["path"]).resolve() != root / "gpu_guarded_leg/baseline/baseline.json"
            or baseline.get("schema") != "gvhmr-formal-preleg-baseline-v1" or baseline.get("state") != "complete"
            or baseline.get("gpu_started") is not False or baseline.get("training_published") is not False):
        raise ValueError("Select the exact immutable formal baseline")
    body, _ = _bound(baseline["body_receipt"], {})
    if body.get("state") != "body_qa_candidate_complete" or body.get("windows") != len(baseline["windows"]):
        raise ValueError("Metadata baseline body/window inventory is incomplete")
    checked = dict(root=root, repair=root / "repair", keys=set(baseline["windows"]), observed={}, baseline=baseline,
                   baseline_binding=baseline_binding,
                   source_sha=baseline["source_union_sha256"], accepted=body["accepted"])
    return _metadata_material(checked, execution_binding, test_execution)


def _validate_proof(receipt, material, repair):
    item = receipt.get("evidence", {}).get(PROOF_ROLE)
    if not isinstance(item, dict):
        raise ValueError("A complete typed source-bound compatibility proof is required")
    path = Path(item["path"]).resolve()
    folder = repair / "evaluation/formal_postleg_cpu/attempts"
    if (path.parent.parent != folder or re.fullmatch(r"[0-9a-f]{32}", path.parent.name) is None
            or path.name != "policy_fingerprint_compatibility.json" or _bound(item, {})[0] != material):
        raise ValueError("Typed compatibility proof differs from actual sources, phase metadata or CPU test evidence")


def validate_compatibility_receipt(formal_root, receipt):
    item = receipt.get("evidence", {}).get(PROOF_ROLE)
    if not isinstance(item, dict):
        raise ValueError("A typed compatibility proof binding is required")
    proof, _ = _bound(item, {})
    material = inspect_policy_metadata(formal_root, receipt["baseline"], receipt["leg_execution"],
                                       proof["compatibility_implementation"])
    _validate_proof(receipt, material, Path(formal_root).resolve() / "repair")
    return proof


def _execution(checked, execution_binding):
    root, repair, observed = checked["root"], checked["repair"], checked["observed"]
    path = Path(execution_binding["path"]).resolve()
    attempt = path.parent
    if (
        path.name != "completion.json"
        or attempt.parent != root / "gpu_guarded_leg/attempts"
        or not re.fullmatch(r"[0-9a-f]{32}", attempt.name)
    ):
        raise ValueError("Execution leaves its exact guarded-leg nonce directory")
    result, _ = _bound(execution_binding, observed)
    if (
        result.get("schema") != "gvhmr-guarded-leg-execution-v1"
        or result.get("state") != "complete_actual_cuda_exited"
        or result.get("training_published") is not False
        or set(result.get("phases", {})) != {"propose", "select", "apply"}
        or not result.get("observed_process_identities")
    ):
        raise ValueError("A complete genuinely exited three-phase execution is required")
    request, _ = _bound(result["request"], observed)
    grant, _ = _bound(result["grant"], observed)
    if (
        Path(result["request"]["path"]).parent.resolve() != root / "gpu_guarded_leg"
        or Path(result["grant"]["path"]).parent.resolve() != root / "gpu_guarded_leg"
        or request.get("schema") != "gvhmr-continuous-production-gpu-request-v1"
        or request.get("authorized_stages_requested") != ["guarded_leg"]
        or len(request.get("scope", [])) != 1
        or grant.get("schema") != "gvhmr-continuous-production-gpu-grant-v1"
        or grant.get("owner") != "GVHMR"
        or grant.get("status") != "held"
        or grant.get("request") != result["request"]
        or grant.get("request_sha256") != result["request"]["sha256"]
    ):
        raise ValueError("Historical executed request/grant bindings differ")
    scope = request["scope"][0]
    if (
        scope.get("stage") != "guarded_leg"
        or Path(scope.get("root", "")).resolve() != repair
        or scope.get("manifest") != checked["baseline"]["source_input"]
        or grant.get("authorized_roots_with_actual_manifest_sha256")
        != {str(repair): dict(manifest_path=scope["manifest"]["path"], manifest_sha256=checked["source_sha"])}
    ):
        raise ValueError("Executed scope is not this exact formal repair/input")
    config, _ = _bound(scope["configuration"], observed)
    if (
        config.get("target_root") != str(repair)
        or config.get("algorithm_changed") is not False
        or config.get("planned_windows") != len(checked["keys"])
    ):
        raise ValueError("Executed configuration uses another target or planned scope")
    ready, _ = _bound(grant["readiness"], observed)
    if (
        ready.get("schema") != "gvhmr-guarded-leg-baseline-readiness-v1"
        or ready.get("state") != "complete"
        or ready.get("request") != result["request"]
        or ready.get("root") != str(repair)
        or ready.get("configuration") != scope["configuration"]
        or ready.get("evidence", {}).get("formal_complete_baseline") != checked["baseline_binding"]
        or ready.get("evidence", {}).get("formal_preleg_qa") != checked["baseline"]["absolute_qa"]
    ):
        raise ValueError("Executed readiness does not bind this genuine baseline and before-QA")
    # This is historical evidence inspection. Expired grant times do not confer
    # any new GPU authorization and do not prevent independent CPU verification.
    context_path = attempt / "context.json"
    context = _load(context_path, observed)
    context_sha = _sha(context_path)
    if (
        context.get("nonce") != attempt.name
        or context.get("job_name") != "Local\\GVHMRGuardedLeg_" + attempt.name
        or context.get("root") != str(repair)
        or context.get("request") != result["request"]
        or context.get("grant") != result["grant"]
    ):
        raise ValueError("Actual execution context leaves its nonce/repair/request/grant")
    for item in context.get("immutable_bindings", []):
        if _sha(item["path"]) != item["sha256"]:
            raise ValueError("Actual execution immutable evidence changed")
        observed[str(Path(item["path"]).resolve())] = item["sha256"]
    preflight, _ = _bound(context["preflight"], observed)
    _exact_path(context["preflight"], attempt / "preflight.json")
    if (
        preflight.get("state") != "complete"
        or preflight.get("request") != result["request"]
        or preflight.get("grant") != result["grant"]
        or preflight.get("readiness") != grant["readiness"]
        or preflight.get("native_and_baseline_evidence_checked") is not True
    ):
        raise ValueError("Actual execution preflight evidence is incomplete")
    _exact_path(result["actual_exit"], attempt / "actual_cuda_exit.json")
    exited, _ = _bound(result["actual_exit"], observed)
    if (
        exited.get("state") != "all_owned_cuda_exited"
        or exited.get("context_sha256") != context_sha
        or exited.get("job_empty") is not True
        or exited.get("gpu0_compute_pids_empty") is not True
        or type(exited.get("observed_process_identities")) is not list
    ):
        raise ValueError("Actual private-job/GPU exit was not proved")
    _exact_path(result["independent_guard"], attempt / "guard_ready.json")
    guard, _ = _bound(result["independent_guard"], observed)
    if (
        guard.get("ready") is not True
        or guard.get("context_sha256") != context_sha
        or guard.get("actual_affinity") != [26, 27]
    ):
        raise ValueError("Actual independent watchdog context was not ready")
    identities = result["observed_process_identities"] + exited["observed_process_identities"] + [guard]
    completed = _load(attempt / "all_phases_complete.json", observed)
    if completed.get("phases") != result["phases"] or completed.get("context") != _binding(context_path):
        raise ValueError("Actual phase completion/context proof differs")
    phase_paths = dict(
        propose=repair / "leg_refinement_proposals/receipt.json",
        select=repair / "leg_refinement_selection.json",
        apply=repair / "leg_refinement_receipt.json",
    )
    commands = scope.get("commands", [])
    if len(commands) != 3 or [c.get("phase") for c in commands] != ["propose", "select", "apply"]:
        raise ValueError("Executed phases do not have all three exact commands")
    outputs = {}
    observed_keys = {(i["pid"], i["create_time"]) for i in result["observed_process_identities"]}
    for item in commands:
        phase = item["phase"]
        phase_info = result["phases"][phase]
        launcher = phase_info.get("launcher", {})
        cuda = phase != "select"
        entry = (
            PROJ_ROOT
            / "tools"
            / {
                "propose": "propose_badminton_leg_refinement.py",
                "select": "select_badminton_leg_refinement.py",
                "apply": "apply_badminton_leg_refinement.py",
            }[phase]
        )
        argv = [str(PROJ_ROOT / ".venv/Scripts/python.exe"), "-u", str(entry), "--root", str(repair)]
        if cuda:
            argv += ["--threads", "2"]
        if (
            item.get("argv") != argv
            or item.get("cuda") is not cuda
            or item.get("entry_binding") != _binding(entry)
            or phase_info.get("actual_job_empty") is not True
            or launcher.get("command") != argv
            or launcher.get("assigned_before_resume") is not True
            or launcher.get("affinity") != ([0, 1] if cuda else [26, 27])
            or (launcher.get("pid"), launcher.get("create_time")) not in observed_keys
        ):
            raise ValueError("Actual phase command, ownership or empty-job evidence differs")
        observed[str(entry.resolve())] = item["entry_binding"]["sha256"]
        if (
            _load(attempt / (phase + ".exit.json"), observed) != phase_info
            or _load(attempt / (phase + ".launch.json"), observed) != launcher
        ):
            raise ValueError("Actual per-phase launch/exit receipt differs")
        _exact_path(phase_info["output"], phase_paths[phase])
        outputs[phase], _ = _bound(phase_info["output"], observed)
        if outputs[phase].get("state") != "complete":
            raise ValueError("An actual leg phase output is incomplete")
        identities.append(launcher)
    for identity in identities:
        _gone(identity)
    _leg_outputs(checked, outputs, result)
    return result


def _leg_outputs(checked, outputs, execution):
    repair, keys, observed = checked["repair"], checked["keys"], checked["observed"]
    proposals, selection, applied = [outputs[p] for p in ("propose", "select", "apply")]
    policies = _policies(checked, execution, outputs)
    for value in (proposals, selection):
        if set(value.get("windows", {})) != keys or value.get("source_union_sha256") != checked["source_sha"]:
            raise ValueError("Leg proposals/selection omit actual formal windows or source")
    if selection.get("source_proposals_sha256") != execution["phases"]["propose"]["output"]["sha256"]:
        raise ValueError("Actual leg selection uses another proposal receipt")
    selected_count = 0
    changed = set()
    for key in keys:
        proposal, _ = _bound(proposals["windows"][key], observed)
        baseline_sha = checked["baseline"]["windows"][key]["report_snapshot"]["sha256"]
        item = selection["windows"][key]
        if (
            proposal.get("source_fit_report_sha256") != baseline_sha
            or item.get("source_fit_report_sha256") != baseline_sha
            or item.get("proposal_path") != proposals["windows"][key]["path"]
            or item.get("proposal_sha256") != proposals["windows"][key]["sha256"]
        ):
            raise ValueError("Leg proposal/selection used another baseline report")
        selected = item.get("selected")
        if type(selected) is not list:
            raise ValueError("Actual selected leg rows are missing")
        selected_count += len(selected)
        if selected:
            changed.add(key)
    if (
        _count(selection.get("selected_frames"), "selected frames") != selected_count
        or applied.get("selected_frames") != selected_count
        or applied.get("updated_frames") != selected_count
        or applied.get("selection_sha256") != execution["phases"]["select"]["output"]["sha256"]
        or applied.get("policy_sha256") != policies["application_entry"]["sha256"]
        or selection.get("refinement_policy_sha256") != proposals.get("policy_sha256")
        or set(applied.get("windows", {})) != changed
        or applied.get("body_qa_accepted") != checked["accepted"]
    ):
        raise ValueError("Actual applied leg selection/update counts differ")
    post = _load(repair / "final_labels/post_refinement.json", observed)
    if any(
        post.get(k) != applied.get(k)
        for k in ("state", "selected_frames", "updated_frames", "selection_sha256", "policy_sha256", "windows")
    ):
        raise ValueError("Actual post-refinement and apply outputs differ")
    for key in keys:
        sid, name = key.split(":", 1)
        window = repair / "final_labels/sessions" / sid / "windows" / name
        report_path = window / "fit_report.json"
        report = _load(report_path, observed)
        _artifacts(report, window, observed)
        if _count(report.get("accepted"), "post-leg accepted") != checked["before"]["windows"][key]["accepted"]:
            raise ValueError("Leg apply changed per-window accepted membership")
        if key in changed:
            entry = applied["windows"][key]
            if (
                entry.get("frames") != len(selection["windows"][key]["selected"])
                or entry.get("fit_report_sha256") != _sha(report_path)
                or report.get("post_leg_refinement", {}).get("selection_sha256") != applied["selection_sha256"]
                or report.get("post_leg_refinement", {}).get("frames") != entry["frames"]
                or report.get("post_leg_refinement", {}).get("policy_sha256")
                != policies["application_entry"]["sha256"]
            ):
                raise ValueError("Actual updated fit report differs from the apply output")
        elif _sha(report_path) != checked["baseline"]["windows"][key]["report_snapshot"]["sha256"]:
            raise ValueError("An unselected formal fit window changed")


def verify_formal_postleg_cpu(
    formal_root, baseline_binding, leg_execution_binding, threads=1, *, compatibility_test_execution=None,
):
    """Check historical CUDA exit, then run only the two existing CPU verifiers."""
    if type(threads) is not int or threads not in (1, 2):
        raise ValueError("Formal CPU verification uses one or two threads")
    _assert_frozen()
    compatibility_test_execution = _resolve_tests(formal_root, compatibility_test_execution)
    checked = _baseline(formal_root, baseline_binding, preleg=False)
    repair, observed = checked["repair"], checked["observed"]
    before_path = repair / "evaluation/v2_before_leg.json"
    if (
        not before_path.exists()
        or not before_path.resolve().is_relative_to(repair)
        or _sha(before_path) != checked["baseline"]["absolute_qa"]["sha256"]
    ):
        raise ValueError("The genuine before-QA file was not installed before leg refinement")
    observed[str(before_path.resolve())] = checked["baseline"]["absolute_qa"]["sha256"]
    _execution(checked, leg_execution_binding)
    material = _metadata_material(checked, leg_execution_binding, compatibility_test_execution)
    _unchanged(observed)
    evaluation = repair / "evaluation"
    if not evaluation.resolve().is_relative_to(repair):
        raise ValueError("CPU verification output leaves the exact repair run")
    final_path = evaluation / "formal_postleg_cpu_receipt.json"
    if final_path.exists():
        receipt = _load(final_path, {})
        if (
            receipt.get("schema") != "gvhmr-formal-postleg-cpu-completion-v1"
            or receipt.get("state") != STATE
            or receipt.get("baseline") != baseline_binding
            or receipt.get("leg_execution") != leg_execution_binding
            or receipt.get("gpu_started") is not False
            or receipt.get("training_published") is not False
            or set(receipt.get("workers", {})) != {"native", "dataset"}
        ):
            raise ValueError("Existing formal CPU completion belongs to different evidence")
        _validate_proof(receipt, material, repair)
        for item in receipt["evidence"].values():
            if _sha(item["path"]) != item["sha256"]:
                raise ValueError("Existing formal CPU completion evidence changed")
        for worker in receipt["workers"].values():
            for item in (worker["entry"], worker["log"]):
                if _sha(item["path"]) != item["sha256"]:
                    raise ValueError("Existing formal CPU worker evidence changed")
        _native_after(checked)
        _dataset_after(checked)
        from .new_source_completion import collect_absolute_qa, compare_same_v2

        after = collect_absolute_qa(repair)
        if after != _load(evaluation / "v2_after_leg.json", {}) or compare_same_v2(checked["before"], after) != _load(
            evaluation / "v2_leg_comparison.json", {}
        ):
            raise ValueError("Existing formal CPU QA no longer matches actual labels")
        return receipt
    attempt = evaluation / "formal_postleg_cpu/attempts" / uuid.uuid4().hex
    if not attempt.resolve().is_relative_to(repair):
        raise ValueError("CPU worker logs leave the exact repair run")
    attempt.mkdir(parents=True, exist_ok=False)
    proof_path = attempt / "policy_fingerprint_compatibility.json"
    _publish(proof_path, _encode(material))
    proof_binding = _binding(proof_path)
    observed[str(proof_path.resolve())] = proof_binding["sha256"]
    workers = {}
    workers["native"] = _run_cpu_check(checked, "native", attempt, threads)
    _native_after(checked)
    _unchanged(observed)
    workers["dataset"] = _run_cpu_check(checked, "dataset", attempt, threads)
    _dataset_after(checked)
    _unchanged(observed)
    from .new_source_completion import collect_absolute_qa, compare_same_v2

    after = collect_absolute_qa(repair)
    comparison = compare_same_v2(checked["before"], after)
    after_reports = {
        key: _binding(
            repair / "final_labels/sessions" / key.split(":")[0] / "windows" / key.split(":", 1)[1] / "fit_report.json"
        )
        for key in checked["keys"]
    }
    if (
        _qa_scope(after, checked["keys"], checked["source_sha"], checked["calibration"], after_reports)
        != checked["accepted"]
    ):
        raise ValueError("Post-leg actual QA accepted scope differs")
    payloads = [
        (evaluation / "v2_after_leg.json", _encode(after)),
        (evaluation / "v2_leg_comparison.json", _encode(comparison)),
    ]
    evidence = dict(
        policy_fingerprint_compatibility=proof_binding,
        before=_binding(before_path),
        native=_binding(repair / "final_labels/native_verification.json"),
        dataset=_binding(repair / "final_labels/datasets/verification.json"),
        post_refinement=_binding(repair / "final_labels/post_refinement.json"),
        leg_receipt=_binding(repair / "leg_refinement_receipt.json"),
    )
    evidence.update({name: _binding(path, payload) for name, (path, payload) in zip(("after", "comparison"), payloads)})
    receipt = dict(
        schema="gvhmr-formal-postleg-cpu-completion-v1",
        state=STATE,
        root=str(checked["root"]),
        baseline=baseline_binding,
        leg_execution=leg_execution_binding,
        source_input=checked["baseline"]["source_input"],
        calibration_digest=checked["calibration"],
        windows=len(checked["keys"]),
        accepted=checked["accepted"],
        workers=workers,
        evidence=evidence,
        gpu_started=False,
        training_published=False,
    )
    payloads += [(final_path, _encode(receipt))]
    for path, payload in payloads:
        _preflight_target(path, payload, evaluation.resolve())
    _unchanged(observed)
    for path, payload in payloads:
        _publish(path, payload)
    return receipt
