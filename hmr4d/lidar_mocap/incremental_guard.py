"""Explicit metadata authority for a generic, source-bound three-phase leg run.

This adapter does not fit, infer, launch CUDA, create a lease, or publish data.
It keeps release parents separate from the actual numeric-import parent map.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import re
import xml.etree.ElementTree as ET

from hmr4d import PROJ_ROOT

from .formal_completion import _baseline
from .formal_import_audit import _parent_preflight
from .formal_native_derive import native_derivation_anchors
from .formal_numeric_import import _load_audit
from .gpu_handoff import utc_time
from .guarded_leg_gpu import (
    ENTRY_NAMES,
    PHASES,
    SHUTDOWN_MARGIN_SECONDS,
    _complete_native,
    _reviewed_identity,
    _runtime_contract_path,
    bound_json,
    identity_alive,
    read_json,
    schema_check,
)
from .incremental_release import load_release_policy, require_binding
from .pilot import binding, check_binding

PROFILE = "gvhmr-incremental-guard-parent-map-v1"
IMPLEMENTATION_SCHEMA = "gvhmr-incremental-guard-implementation-v1"
TEST_SCHEMA = "gvhmr-incremental-guard-cpu-test-v1"
DESCRIPTOR_FIELDS = {
    "schema",
    "authorization_profile",
    "formal_root",
    "release_policy",
    "source_input",
    "import_audit",
    "numeric_import",
    "baseline",
    "formal_preleg_native",
    "formal_preleg_qa",
    "parents",
    "windows",
    "source_scope",
}
WINDOW_FIELDS = {
    "parent_root",
    "source_id",
    "original_window",
    "original_report",
    "current_report",
    "current_window_report",
}
READY_EVIDENCE = {
    "formal_numeric_import",
    "formal_complete_baseline",
    "formal_reviewed_identity",
    "formal_preleg_native_verification",
    "formal_preleg_qa",
    "predecessor_cuda_exit_bundle",
    "independent_guard_ready",
}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    _need(type(value) is int and value >= 0, "Require an exact nonnegative integer: " + name)
    return value


def _affinity(value, expected):
    return isinstance(value, list) and all(type(number) is int for number in value) and value == expected


def _native(value, keys, **options):
    for name in ("verified_windows", "total_windows", "accepted_checked"):
        _integer(value.get(name), name)
    for row in value.get("windows", {}).values():
        _integer(row.get("body", {}).get("accepted_checked"), "native window accepted")
    _complete_native(value, keys, **options)


def _absolute(value):
    _need(isinstance(value, (str, Path)) and bool(str(value)), "Require an explicit absolute path")
    path = Path(value)
    _need(path.is_absolute() and ".." not in path.parts, "Require an absolute path without traversal")
    resolved = path.resolve()
    _need(path == resolved, "Incremental authority cannot use a path alias")
    return resolved


class _Inputs:
    def __init__(self):
        self.items = {}

    def add(self, item):
        require_binding(item)
        path = _absolute(item["path"])
        previous = self.items.get(str(path))
        _need(previous is None or previous == item, "Conflicting immutable incremental bindings")
        check_binding(item)
        self.items[str(path)] = dict(item)
        return item

    def bind(self, path):
        return self.add(binding(_absolute(path)))

    def read(self, item):
        self.add(item)
        result = bound_json(item)
        self.add(item)
        _need(isinstance(result, dict), "Bound incremental metadata must be an object")
        return result

    def recheck(self):
        for item in self.items.values():
            check_binding(item)


def implementation_files():
    """The real new implementation plus its unchanged authorization dependencies."""
    names = dict(
        authorization="hmr4d/lidar_mocap/incremental_guard.py",
        execution="hmr4d/lidar_mocap/incremental_guard_execution.py",
        entry="tools/run_badminton_incremental_guarded_leg.py",
        authorization_tests="tests/test_lidar_incremental_guard.py",
        execution_tests="tests/test_lidar_incremental_guard_execution.py",
        legacy_authorization_leaves="hmr4d/lidar_mocap/guarded_leg_gpu.py",
        legacy_execution_leaves="hmr4d/lidar_mocap/guarded_leg_execution.py",
        windows_job="hmr4d/lidar_mocap/windows_gpu_job.py",
        stage_lease="hmr4d/lidar_mocap/gpu_handoff.py",
        baseline_checker="hmr4d/lidar_mocap/formal_completion.py",
        baseline="hmr4d/lidar_mocap/formal_baseline.py",
        import_audit="hmr4d/lidar_mocap/formal_import_audit.py",
        numeric_import="hmr4d/lidar_mocap/formal_numeric_import.py",
        release_policy="hmr4d/lidar_mocap/incremental_release.py",
        native_witness="hmr4d/lidar_mocap/native_witness.py",
        native_derivation="hmr4d/lidar_mocap/formal_native_derive.py",
        policy_compat="hmr4d/lidar_mocap/formal_completion_policy_compat.py",
    )
    return {role: PROJ_ROOT / name for role, name in names.items()}


def _commands(request):
    scope = request.get("scope", [])
    _need(len(scope) == 1 and scope[0].get("stage") == "guarded_leg", "Require one guarded-leg target")
    root = _absolute(scope[0]["root"])
    base = (PROJ_ROOT / "outputs").resolve()
    _need(root != base and root.is_relative_to(base) and root.name == "repair", "Target leaves exact outputs repair")
    _need(scope[0]["manifest"] == binding(root / "new_source_input.json"), "Source input binding differs")
    commands = []
    for phase, name in zip(PHASES, ENTRY_NAMES):
        entry = PROJ_ROOT / "tools" / name
        argv = [str(PROJ_ROOT / ".venv/Scripts/python.exe"), "-u", str(entry), "--root", str(root)]
        if phase != "select":
            argv += ["--threads", "2"]
        commands.append(dict(phase=phase, cuda=phase != "select", argv=argv, entry_binding=binding(entry)))
    _need(scope[0].get("commands") == commands, "Original numerical command, options or entry changed")
    return root, commands


def inspect_request(path, *, descriptor_path=None, descriptor_sha256=None):
    """Inspect explicit CPU metadata only; this is never authority to launch."""
    _need(descriptor_path is not None and descriptor_sha256 is not None, "Supply explicit descriptor path and SHA256")
    inputs = _Inputs()
    request_path = _absolute(path)
    request_binding = inputs.bind(request_path)
    request = inputs.read(request_binding)
    selected = dict(path=str(_absolute(descriptor_path)), sha256=descriptor_sha256)
    _need(
        request.get("schema") == "gvhmr-continuous-production-gpu-request-v1"
        and request.get("authorization_profile") == PROFILE
        and request.get("incremental_descriptor") == selected
        and request.get("authorized_stages_requested") == ["guarded_leg"]
        and request.get("requested_local_stage_lease_limits_minutes") == {"guarded_leg": 30}
        and request.get("requested_maximum_hours") == 0.5
        and type(request.get("max_concurrent_cuda_trees")) is int
        and request["max_concurrent_cuda_trees"] == 1
        and request.get("requested_cuda_visible_devices") == "0"
        and request.get("gpu_started") is False
        and request.get("training_published") is False,
        "Require an explicit typed incremental guarded-leg request",
    )
    minimum = request.get("minimum_remaining_launch_seconds")
    _need(type(minimum) is int and 0 < minimum <= 1800, "Invalid first-launch time margin")
    utc_time(request["absolute_hard_deadline_utc"])
    root, commands = _commands(request)
    _need(request_path.parent == root.parent / "gpu_guarded_leg", "Request leaves its exact authorization directory")
    descriptor = inputs.read(selected)
    _need(
        set(descriptor) == DESCRIPTOR_FIELDS
        and descriptor.get("schema") == PROFILE
        and descriptor.get("authorization_profile") == PROFILE
        and _absolute(descriptor["formal_root"]) == root.parent
        and descriptor["source_input"] == request["scope"][0]["manifest"],
        "Incremental descriptor scope or closed schema differs",
    )
    for name in (
        "source_input",
        "import_audit",
        "numeric_import",
        "baseline",
        "formal_preleg_native",
        "formal_preleg_qa",
    ):
        inputs.add(descriptor[name])
    policy_binding = inputs.add(descriptor["release_policy"])
    policy = load_release_policy(policy_binding["path"], policy_binding["sha256"])
    for item in policy.dependencies:
        inputs.add(item)
    source = inputs.read(descriptor["source_input"])
    registry_binding = inputs.bind(root.parent / "source/source_registry.json")
    registry = inputs.read(registry_binding)
    scope = descriptor["source_scope"]
    _need(
        isinstance(scope, list)
        and scope
        and all(isinstance(sid, str) and sid for sid in scope)
        and len(scope) == len(set(scope))
        and set(scope) == set(policy.scope["review_sources"])
        and set(registry.get("sources", {})) == set(scope)
        and source.get("release_policy") == policy.binding
        and registry.get("release_policy") == policy.binding
        and source.get("session_splits") == policy.roles
        and source.get("required_nonempty_splits") == policy.required
        and source.get("source_registry_sha256") == registry_binding["sha256"]
        and _absolute(source["source_root"]) == root.parent / "source"
        and source.get("review_only") is not True
        and registry.get("review_only") is not True,
        "Actual source/camera/release policy scope differs",
    )
    for sid in scope:
        _need(
            registry["sources"][sid].get("source_index")
            == policy.scope["review_sources"][sid]["admitted_source_index"],
            "Actual source index differs from the reviewed release policy",
        )
    configuration = inputs.read(request["scope"][0]["configuration"])
    code_inputs = request["scope"][0]["existing_input_and_code_bindings"]
    _need(
        configuration.get("target_root") == str(root)
        and configuration.get("algorithm_changed") is False
        and configuration.get("code_and_inputs") == code_inputs,
        "Numerical policy description differs",
    )
    for item in code_inputs.values():
        inputs.add(item)
    for role, name in (
        ("refinement_policy", "hmr4d/lidar_mocap/leg_refinement.py"),
        ("selection_policy", "hmr4d/lidar_mocap/leg_selection.py"),
        ("application_entry", "tools/apply_badminton_leg_refinement.py"),
    ):
        _need(code_inputs.get(role) == binding(PROJ_ROOT / name), "Original numerical producer changed")
    runtime_path = _runtime_contract_path(request_path, request)
    runtime_binding = inputs.bind(runtime_path)
    runtime = inputs.read(runtime_binding)
    _need(
        runtime.get("request") == request_binding
        and runtime.get("authorization_profile") == PROFILE
        and runtime.get("incremental_descriptor") == selected,
        "Issued runtime contract scope differs",
    )
    cpu = inputs.read(runtime["cpu_allocation"])
    _need(
        cpu.get("request") == request_binding
        and _affinity(cpu["pure_cpu_phases"]["affinity"], [26, 27])
        and _affinity(cpu["cuda_propose_and_apply"]["affinity"], [0, 1]),
        "Original phase CPU/CUDA allocation differs",
    )
    for name in ("grant_schema", "readiness_schema"):
        inputs.add(runtime[name])
    inputs.recheck()
    return dict(
        request=request,
        request_binding=request_binding,
        root=root,
        commands=commands,
        configuration=configuration,
        runtime=runtime,
        runtime_binding=runtime_binding,
        cpu=cpu,
        descriptor=descriptor,
        descriptor_binding=selected,
        policy=policy,
        inputs=inputs,
        authorization_profile=PROFILE,
    )


def verify_implementation(item, checked):
    inputs = checked["inputs"]
    implementation = inputs.read(item)
    expected = {role: binding(path) for role, path in implementation_files().items()}
    _need(
        set(implementation) == {"schema", "request", "descriptor", "runtime_contract", "code", "test_execution"}
        and implementation.get("schema") == IMPLEMENTATION_SCHEMA
        and implementation.get("request") == checked["request_binding"]
        and implementation.get("descriptor") == checked["descriptor_binding"]
        and implementation.get("runtime_contract") == checked["runtime_binding"]
        and implementation.get("code") == expected,
        "The independently tested incremental implementation differs; old guard8 is not this implementation",
    )
    for proof in expected.values():
        inputs.add(proof)
    tested = inputs.read(implementation["test_execution"])
    _need(
        tested.get("schema") == TEST_SCHEMA
        and tested.get("state") == "passed"
        and type(tested.get("exit_code")) is int
        and tested["exit_code"] == 0
        and type(tested.get("passed")) is int
        and tested["passed"] > 0
        and all(type(tested.get(name)) is int and tested[name] == 0 for name in ("failed", "errors", "skipped"))
        and tested.get("gpu_started") is False
        and tested.get("fixture_only") is True
        and tested.get("code") == expected
        and tested.get("test_files"),
        "Actual current-code CPU test evidence is missing or unsuccessful",
    )
    affinity = tested.get("actual_affinity")
    _need(
        isinstance(affinity, list)
        and 0 < len(affinity) <= 2
        and len(affinity) == len(set(affinity))
        and all(type(value) is int and value >= 0 for value in affinity)
        and tested.get("priority") == "BelowNormal"
        and type(tested.get("io_priority")) is int
        and tested["io_priority"] == 1
        and type(tested.get("numerical_threads")) is int
        and tested["numerical_threads"] == 1,
        "CPU test resource evidence differs",
    )
    test_files = tested["test_files"]
    _need(isinstance(test_files, list), "CPU test source bindings must be a list")
    for proof in test_files:
        inputs.add(proof)
    _need(
        len({proof["path"] for proof in test_files}) == len(test_files)
        and all(expected[role] in test_files for role in ("authorization_tests", "execution_tests")),
        "Actual CPU results must bind both current incremental test sources without duplicates",
    )
    inputs.add(tested["stdout"])
    inputs.add(tested["junit"])
    cases = list(ET.parse(tested["junit"]["path"]).iter("testcase"))
    _need(
        len(cases) == tested["passed"]
        and len({(c.get("classname"), c.get("name")) for c in cases}) == len(cases)
        and all(not any(c.find(tag) is not None for tag in ("failure", "error", "skipped")) for c in cases),
        "Actual unique JUnit results differ from the declared CPU tests",
    )
    return implementation


def _parent_map(checked, imported, baseline):
    """Resolve actual audit parent roots; dates never select an authority."""
    inputs, descriptor, root = checked["inputs"], checked["descriptor"], checked["root"]
    audit = _load_audit(root.parent, descriptor["import_audit"])
    _need(audit == inputs.read(descriptor["import_audit"]), "Import audit changed during its original validation")
    keys = baseline["keys"]
    _need(
        imported.get("audit") == descriptor["import_audit"]
        and audit.get("schema") == "badminton-formal-numeric-import-audit-v1"
        and audit.get("state") == "eligible_parent_complete_not_imported"
        and audit.get("formal_root") == str(root.parent)
        and audit.get("source_input") == descriptor["source_input"]
        and set(audit.get("windows", {})) == keys
        and set(imported.get("windows", {})) == keys
        and imported.get("parent_evidence") == audit.get("parent_evidence") == descriptor["parents"],
        "Numeric import/audit/declared parent scope differs",
    )
    for name, path in (
        ("body_plan", root / "final_labels/repair_plan.json"),
        ("formal_registry", root.parent / "source/source_registry.json"),
        ("formal_calibration", root.parent / "source/calibration/calibration_snapshot.json"),
    ):
        _need(audit.get(name) == inputs.bind(path), "Import audit binds another actual input")
    _need(
        inputs.read(audit["formal_calibration"]).get("snapshot_digest") == baseline["calibration"],
        "Actual import calibration differs from the complete baseline",
    )
    inputs.add(audit["input_equivalence"])
    for path, digest in audit.get("teacher_files", {}).items():
        inputs.add(dict(path=path, sha256=digest))
    parents = {}
    for name, evidence in descriptor["parents"].items():
        parent = _absolute(name)
        _need(
            parent != root.parent and not parent.is_relative_to(root.parent) and not root.parent.is_relative_to(parent),
            "Numeric parent aliases or overlaps its target",
        )
        original = _parent_preflight(parent)
        expected = {role: binding(path) for role, path in original["paths"].items()}
        _need(evidence == expected, "Declared parent evidence differs from the actual completed parent")
        for proof in evidence.values():
            inputs.add(proof)
        native = original["values"]["native"]
        _integer(original["values"]["body"].get("windows"), "parent body windows")
        _integer(original["values"]["body"].get("accepted"), "parent body accepted")
        _native(
            native,
            set(original["jobs"]),
            source_sha=binding(original["paths"]["source_input"])["sha256"],
            calibration=original["values"]["calibration"]["snapshot_digest"],
            repair=parent / "repair",
        )
        for proof in native_derivation_anchors(native):
            inputs.add(proof)
        identity = original["values"]["identity"]
        for path, digest in identity["evidence_files"].items():
            inputs.add(dict(path=path, sha256=digest))
        parents[name] = original
    result = {}
    for key, row in audit["windows"].items():
        sid, name = key.split(":", 1)
        parent = _absolute(row["parent_root"])
        _need(
            str(parent) in parents and row.get("source_id") == sid and sid in checked["descriptor"]["source_scope"],
            "Unknown or mismatched numeric parent/source",
        )
        reviewed = checked["policy"].scope["review_sources"][sid]
        _need(
            _absolute(reviewed["review_root"]) == parent
            and reviewed["review_source_input"] == binding(parents[str(parent)]["paths"]["source_input"]),
            "Numeric parent differs from the actual policy-reviewed root/input",
        )
        current = imported["windows"][key]
        expected = dict(
            parent_root=str(parent),
            source_id=sid,
            original_window=row["original_window"],
            original_report=row["original_report"],
            current_report=current["current_report"],
            current_window_report=current["current_window_report"],
        )
        _need(
            set(descriptor["windows"].get(key, {})) == WINDOW_FIELDS
            and descriptor["windows"][key] == expected
            and _absolute(row["original_window"]) == parent / "repair/final_labels/sessions" / sid / "windows" / name
            and _absolute(row["target_window"]) == root / "final_labels/sessions" / sid / "windows" / name
            and row["original_report"] == current["original_report"]
            and current["original_report"] == binding(Path(row["original_window"]) / "fit_report.json")
            and current["current_report"] == baseline["baseline"]["windows"][key]["report_snapshot"]
            and current["current_window_report"] == binding(Path(row["target_window"]) / "fit_report.json"),
            "Actual window/report map differs from its explicit numeric parent",
        )
        old = inputs.read(current["original_report"])
        new = inputs.read(current["current_report"])
        inputs.add(current["current_window_report"])
        teacher = current["teacher_rebinding"]
        _need(
            isinstance(teacher, dict)
            and teacher.get("numerical_values_dtype_shape_identical") is True
            and isinstance(teacher.get("provenance_hash"), str)
            and re.fullmatch("[0-9a-f]{64}", teacher["provenance_hash"]) is not None,
            "Actual teacher provenance rebinding proof is incomplete",
        )
        for role in ("source_manifest", "current_manifest"):
            inputs.add(teacher[role])
        for track in teacher["track_files"]:
            inputs.add(track["source"])
            inputs.add(track["current"])
        _need(
            teacher["source_manifest"] == binding(Path(row["original_window"]) / "teacher/manifest.json")
            and teacher["current_manifest"] == binding(Path(row["target_window"]) / "teacher/manifest.json")
            and new.get("formal_reuse", {}).get("teacher_rebinding") == teacher,
            "Teacher rebinding does not bind the actual imported/current window",
        )
        _need(
            parents[str(parent)]["values"]["native"]["windows"][key]["body"]["report_sha256"]
            == current["original_report"]["sha256"],
            "Imported parent window was not actually native verified",
        )
        numeric = lambda value: {
            name: digest
            for name, digest in value["artifacts_sha256"].items()
            if name == "ptv3.pth" or name.startswith("fits/")
        }
        _need(
            "ptv3.pth" in numeric(old)
            and numeric(old) == numeric(new)
            and all(
                all(type(value.get(field)) is int for value in (row, current, old, new))
                and row[field] == current[field] == old[field] == new[field]
                for field in ("fitted", "accepted")
            ),
            "Imported numeric bytes or actual window counts changed",
        )
        result[key] = expected
    _need(
        set(descriptor["windows"]) == keys and {row["parent_root"] for row in result.values()} == set(parents),
        "Parent map omits or adds windows/parents",
    )
    return result


def validate_readiness_inputs(checked, readiness_binding, implementation_binding):
    inputs, root, descriptor = checked["inputs"], checked["root"], checked["descriptor"]
    ready = inputs.read(readiness_binding)
    schema_check(ready, inputs.read(checked["runtime"]["readiness_schema"]))
    _need(
        ready.get("schema") == "gvhmr-guarded-leg-baseline-readiness-v1"
        and ready.get("state") == "complete"
        and ready.get("authorization_profile") == PROFILE
        and ready.get("incremental_descriptor") == checked["descriptor_binding"]
        and ready.get("request") == checked["request_binding"]
        and ready.get("root") == str(root)
        and ready.get("configuration") == checked["request"]["scope"][0]["configuration"]
        and set(ready.get("evidence", {})) == READY_EVIDENCE,
        "A complete explicitly typed generic readiness envelope is required",
    )
    evidence = {role: inputs.read(proof) for role, proof in ready["evidence"].items()}
    for role, name in (
        ("formal_numeric_import", "numeric_import"),
        ("formal_complete_baseline", "baseline"),
        ("formal_preleg_native_verification", "formal_preleg_native"),
        ("formal_preleg_qa", "formal_preleg_qa"),
    ):
        _need(ready["evidence"][role] == descriptor[name], "Readiness differs from its explicit immutable descriptor")
    baseline = _baseline(root.parent, descriptor["baseline"], preleg=True)
    for path, digest in baseline["observed"].items():
        inputs.add(dict(path=path, sha256=digest))
    _need(
        type(checked["configuration"].get("planned_windows")) is int
        and len(baseline["keys"]) == checked["configuration"]["planned_windows"],
        "Actual body window count differs",
    )
    value = baseline["baseline"]
    _need(
        value["numeric_import"] == descriptor["numeric_import"]
        and value["native_verification"] == descriptor["formal_preleg_native"]
        and value["absolute_qa"] == descriptor["formal_preleg_qa"]
        and value["reviewed_identity"] == ready["evidence"]["formal_reviewed_identity"],
        "Formal preleg/native/QA snapshots differ from the actual baseline",
    )
    _native(
        evidence["formal_preleg_native_verification"],
        baseline["keys"],
        source_sha=baseline["source_sha"],
        calibration=baseline["calibration"],
        repair=root,
    )
    for proof in native_derivation_anchors(evidence["formal_preleg_native_verification"]):
        inputs.add(proof)
    imported = evidence["formal_numeric_import"]
    _need(
        imported.get("schema") == "badminton-formal-numeric-import-v1"
        and imported.get("state") == "complete"
        and imported.get("source_union_sha256") == baseline["source_sha"]
        and imported.get("calibration_digest") == baseline["calibration"]
        and imported.get("numerical_artifacts_byte_identical") is True
        and imported.get("qa_values_unchanged") is True,
        "The actual complete numeric import differs",
    )
    checked["parent_map"] = _parent_map(checked, imported, baseline)
    exits = evidence["predecessor_cuda_exit_bundle"]
    predecessor = checked["request"]["predecessor_release"]
    _need(
        ready["evidence"]["predecessor_cuda_exit_bundle"] == predecessor
        and set(exits) == {"schema", "state", "release_receipts", "process_identities"}
        and exits.get("schema") == "gvhmr-incremental-guard-predecessor-release-v1"
        and exits.get("state") == "complete"
        and isinstance(exits.get("release_receipts"), list)
        and exits["release_receipts"]
        and isinstance(exits.get("process_identities"), list),
        "Actual generic predecessor release evidence is incomplete",
    )
    for proof in exits["release_receipts"]:
        released = inputs.read(proof)
        _need(
            released.get("status") == "released" and released.get("actual_cuda_exited") is True,
            "A predecessor GPU lease has not actually released",
        )
    for identity in exits["process_identities"]:
        _need(
            type(identity.get("pid")) is int
            and identity["pid"] > 0
            and type(identity.get("create_time")) in (float, int)
            and math.isfinite(identity["create_time"])
            and identity["create_time"] > 0
            and not identity_alive(identity),
            "A predecessor identity is invalid or still alive",
        )
    guard = evidence["independent_guard_ready"]
    _need(
        guard.get("schema") == "gvhmr-guarded-leg-guard-readiness-v1"
        and guard.get("state") == "implementation_tested"
        and guard.get("implementation") == implementation_binding,
        "Independent current-code guard readiness is missing",
    )
    verify_implementation(implementation_binding, checked)
    checked["policy"].recheck()
    inputs.recheck()
    return ready


def load_authorization(request_path, grant_path, *, now=None, deep=True, descriptor_path=None, descriptor_sha256=None):
    checked = inspect_request(request_path, descriptor_path=descriptor_path, descriptor_sha256=descriptor_sha256)
    inputs = checked["inputs"]
    grant_binding = inputs.bind(grant_path)
    grant = inputs.read(grant_binding)
    schema_check(grant, inputs.read(checked["runtime"]["grant_schema"]))
    scope = checked["request"]["scope"][0]
    _need(
        grant.get("schema") == "gvhmr-continuous-production-gpu-grant-v1"
        and grant.get("status") == "held"
        and grant.get("owner") == "GVHMR"
        and grant.get("request") == checked["request_binding"]
        and grant.get("request_sha256") == checked["request_binding"]["sha256"]
        and grant.get("authorization_profile") == PROFILE
        and grant.get("incremental_descriptor") == checked["descriptor_binding"]
        and grant.get("allow_local_stage_leases") is True
        and grant.get("native_gpu_returned") is True
        and grant.get("training_published") is False
        and grant.get("authorized_roots_with_actual_manifest_sha256")
        == {
            str(checked["root"]): dict(
                manifest_path=scope["manifest"]["path"], manifest_sha256=scope["manifest"]["sha256"]
            )
        }
        and grant.get("configuration") == scope["configuration"]
        and grant.get("cpu_allocation") == checked["runtime"]["cpu_allocation"]
        and grant.get("predecessor_release") == checked["request"]["predecessor_release"],
        "A real separately issued source-bound incremental grant is required",
    )
    maximum = grant.get("maximum_stage_minutes", {}).get("guarded_leg")
    _need(
        set(grant.get("maximum_stage_minutes", {})) == {"guarded_leg"} and type(maximum) is int and 0 < maximum <= 30,
        "Granted local stage cap exceeds the request",
    )
    start, deadline = utc_time(grant["approved_start_utc"]), utc_time(grant["approved_deadline_utc"])
    hard = utc_time(checked["request"]["absolute_hard_deadline_utc"])
    now = now or datetime.now(timezone.utc)
    _need(
        now.tzinfo is not None and start <= now < deadline <= min(hard, start + timedelta(minutes=30)),
        "Incremental GPU grant is future, expired or exceeds its original fixed deadline",
    )
    verify_implementation(grant["independent_guard"], checked)
    if deep:
        checked["readiness"] = validate_readiness_inputs(checked, grant["readiness"], grant["independent_guard"])
    else:
        inputs.add(grant["readiness"])
    if grant.get("desktop_allowance"):
        inputs.add(grant["desktop_allowance"])
    inputs.add(grant["predecessor_release"])
    inputs.recheck()
    checked.update(
        authorization=dict(
            grant=grant,
            binding=grant_binding,
            request=checked["request_binding"],
            roots=grant["authorized_roots_with_actual_manifest_sha256"],
            start=start,
            deadline=deadline,
        ),
        deadline=deadline,
        not_before=start,
        grant_path=Path(grant_path).resolve(),
        grant_binding=grant_binding,
    )
    return checked


def immutable_anchors(checked):
    """Watch immutable evidence; mutable apply destinations are not permanent anchors."""
    inputs = _Inputs()
    descriptor = checked["descriptor"]
    grant = checked["authorization"]["grant"]
    for item in (
        checked["request_binding"],
        checked["grant_binding"],
        checked["descriptor_binding"],
        checked["runtime_binding"],
        grant["readiness"],
        grant["configuration"],
        grant["cpu_allocation"],
        grant["independent_guard"],
        grant["predecessor_release"],
        checked["runtime"]["grant_schema"],
        checked["runtime"]["readiness_schema"],
    ):
        inputs.add(item)
    for item in checked["policy"].dependencies:
        inputs.add(item)
    for name in (
        "source_input",
        "import_audit",
        "numeric_import",
        "baseline",
        "formal_preleg_native",
        "formal_preleg_qa",
    ):
        inputs.add(descriptor[name])
    for item in checked["request"]["scope"][0]["existing_input_and_code_bindings"].values():
        inputs.add(item)
    baseline = inputs.read(descriptor["baseline"])
    for name in ("body_receipt", "pose_receipt", "native_verification", "reviewed_identity", "absolute_qa", "plan"):
        inputs.add(baseline[name])
    for entry in baseline["windows"].values():
        inputs.add(entry["report_snapshot"])
    imported = inputs.read(descriptor["numeric_import"])
    for entry in imported["windows"].values():
        inputs.add(entry["original_report"])
        inputs.add(entry["current_report"])
    for evidence in descriptor["parents"].values():
        for item in evidence.values():
            inputs.add(item)
        identity = inputs.read(evidence["identity"])
        for path, digest in identity["evidence_files"].items():
            inputs.add(dict(path=path, sha256=digest))
        for item in native_derivation_anchors(inputs.read(evidence["native"])):
            inputs.add(item)
    for item in native_derivation_anchors(inputs.read(descriptor["formal_preleg_native"])):
        inputs.add(item)
    ready = inputs.read(grant["readiness"])
    for item in ready["evidence"].values():
        inputs.add(item)
    predecessor = inputs.read(grant["predecessor_release"])
    for item in predecessor["release_receipts"]:
        inputs.add(item)
    implementation = inputs.read(grant["independent_guard"])
    for item in implementation["code"].values():
        inputs.add(item)
    tested = inputs.read(implementation["test_execution"])
    for item in [*tested["test_files"], tested["junit"], tested["stdout"]]:
        inputs.add(item)
    if grant.get("desktop_allowance"):
        inputs.add(grant["desktop_allowance"])
    # The full readiness predicate reads live target reports/artifacts before
    # propose/apply. The watchdog watches their independent immutable snapshots.
    inputs.recheck()
    return [inputs.items[name] for name in sorted(inputs.items)]
