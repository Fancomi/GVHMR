"""Target-bound authorization for the existing guarded-leg scripts; no fitting policy changes."""

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import time

import psutil

from hmr4d import PROJ_ROOT
from .gpu_handoff import read_continuous_grant, utc_time
from .pilot import binding, check_binding


PHASES = ("propose", "select", "apply")
ENTRY_NAMES = ("propose_badminton_leg_refinement.py", "select_badminton_leg_refinement.py",
               "apply_badminton_leg_refinement.py")
SHUTDOWN_MARGIN_SECONDS = 10


def read_json(path):
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite JSON number")
        return number

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate JSON field: " + key)
            result[key] = value
        return result

    for attempt in range(20):
        try:
            return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_float=finite_float,
                              parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.025)


def bound_json(item):
    if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not re.fullmatch(
            "[0-9a-f]{64}", str(item.get("sha256", ""))):
        raise ValueError("A real path/SHA256 binding is required; pending evidence cannot authorize GPU")
    check_binding(item)
    value = read_json(item["path"])
    check_binding(item)
    return value


def schema_check(value, rule, *, definitions=None):
    """Validate only the closed schema vocabulary used by this bound local contract."""
    allowed = {"$schema", "title", "$defs", "$ref", "allOf", "const", "type", "required", "properties",
               "additionalProperties", "items", "minItems", "uniqueItems", "minLength", "pattern",
               "format", "minimum", "maximum"}
    if set(rule) - allowed:
        raise ValueError("Unsupported local contract schema vocabulary")
    definitions = definitions if definitions is not None else rule.get("$defs", {})
    if "$ref" in rule:
        prefix = "#/$defs/"
        reference = rule["$ref"]
        if not reference.startswith(prefix) or reference[len(prefix):] not in definitions:
            raise ValueError("Unknown or external schema reference")
        schema_check(value, definitions[reference[len(prefix):]], definitions=definitions)
    for part in rule.get("allOf", []):
        schema_check(value, part, definitions=definitions)
    if "const" in rule and (value != rule["const"] or type(value) is not type(rule["const"])):
        raise ValueError("Bound schema constant differs")
    kind = rule.get("type")
    types = {"object": lambda x: type(x) is dict, "array": lambda x: type(x) is list,
             "string": lambda x: type(x) is str, "integer": lambda x: type(x) is int,
             "number": lambda x: type(x) in (int, float) and math.isfinite(x),
             "boolean": lambda x: type(x) is bool, "null": lambda x: x is None}
    if kind is not None and (kind not in types or not types[kind](value)):
        raise ValueError("Wrong local contract field type")
    if isinstance(value, dict):
        if set(rule.get("required", [])) - set(value):
            raise ValueError("Missing required local contract fields")
        properties = rule.get("properties", {})
        if rule.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError("Unknown local contract fields")
        for key in set(value) & set(properties):
            schema_check(value[key], properties[key], definitions=definitions)
    if isinstance(value, list):
        if len(value) < rule.get("minItems", 0):
            raise ValueError("Incomplete contract evidence list")
        if rule.get("uniqueItems") and len({json.dumps(x, sort_keys=True) for x in value}) != len(value):
            raise ValueError("Repeated contract evidence")
        for entry in value:
            if "items" in rule:
                schema_check(entry, rule["items"], definitions=definitions)
    if isinstance(value, str):
        if len(value) < rule.get("minLength", 0) or ("pattern" in rule and not re.search(rule["pattern"], value)):
            raise ValueError("Invalid contract string")
        if "format" in rule:
            if rule["format"] != "date-time":
                raise ValueError("Unsupported contract format")
            utc_time(value)
    if type(value) in (int, float):
        if value < rule.get("minimum", -math.inf) or value > rule.get("maximum", math.inf):
            raise ValueError("Contract number outside its bound")


def approved_commands(request):
    scope = request["scope"]
    if len(scope) != 1 or scope[0].get("stage") != "guarded_leg":
        raise ValueError("Only one exact guarded-leg target is permitted")
    root = Path(scope[0]["root"]).resolve()
    base = (PROJ_ROOT / "outputs").resolve()
    if root == base or not root.is_relative_to(base) or root.name != "repair":
        raise ValueError("Guarded-leg target must be its separate exact repair root")
    if Path(scope[0]["manifest"]["path"]).resolve() != root / "new_source_input.json":
        raise ValueError("Guarded-leg source manifest leaves its exact target")
    expected = []
    for phase, name in zip(PHASES, ENTRY_NAMES):
        entry = PROJ_ROOT / "tools" / name
        argv = [str(PROJ_ROOT / ".venv/Scripts/python.exe"), "-u", str(entry), "--root", str(root)]
        if phase != "select":
            argv += ["--threads", "2"]
        expected.append(dict(phase=phase, cuda=phase != "select", argv=argv, entry_binding=binding(entry)))
    if scope[0].get("commands") != expected:
        raise ValueError("A leg command, script hash, option, interpreter or exact target changed")
    return root, expected


def identity_alive(identity):
    try:
        process = psutil.Process(identity["pid"])
        if process.create_time() != identity["create_time"]:
            return False
        if identity.get("command") is not None and process.cmdline() != identity["command"]:
            raise ValueError("Registered process command changed")
        return process.is_running()
    except psutil.NoSuchProcess:
        return False


def implementation_files():
    return {name: PROJ_ROOT / relative for name, relative in dict(
        authorization="hmr4d/lidar_mocap/guarded_leg_gpu.py",
        execution="hmr4d/lidar_mocap/guarded_leg_execution.py",
        windows_job="hmr4d/lidar_mocap/windows_gpu_job.py",
        entry="tools/run_badminton_guarded_leg.py",
        common_authorization="hmr4d/lidar_mocap/gpu_handoff.py",
    ).items()}


def verify_implementation(item, checked):
    implementation = bound_json(item)
    if (implementation.get("schema") != "gvhmr-guarded-leg-implementation-v1"
            or implementation.get("request") != checked["request_binding"]
            or implementation.get("runtime_contract") != binding(
                Path(checked["request_binding"]["path"]).parent / "issuer_runtime_checks.json")):
        raise ValueError("This independent guard is not bound to the issued target contract")
    expected = {name: binding(path) for name, path in implementation_files().items()}
    if implementation.get("code") != expected:
        raise ValueError("Guard implementation code changed")
    tested = bound_json(implementation["test_execution"])
    if (tested.get("schema") != "gvhmr-guarded-leg-cpu-test-v1" or tested.get("state") != "passed"
            or tested.get("exit_code") != 0 or tested.get("passed", 0) < 1
            or tested.get("failed") != 0 or tested.get("gpu_started") is not False
            or tested.get("actual_affinity") != [26, 27] or tested.get("code") != expected
            or not tested.get("test_files")):
        raise ValueError("Actual isolated CPU guard tests did not pass for this code")
    for proof in tested["test_files"]:
        check_binding(proof)
    return implementation


def inspect_request(path):
    path = Path(path).resolve()
    request = read_json(path)
    if (request.get("schema") != "gvhmr-continuous-production-gpu-request-v1"
            or request.get("authorized_stages_requested") != ["guarded_leg"]
            or request.get("requested_local_stage_lease_limits_minutes") != {"guarded_leg": 30}
            or request.get("requested_maximum_hours") != .5 or request.get("max_concurrent_cuda_trees") != 1
            or request.get("requested_cuda_visible_devices") != "0"
            or request.get("gpu_started") is not False or request.get("training_published") is not False):
        raise ValueError("An explicit unchanged guarded-leg request is required")
    root, commands = approved_commands(request)
    if path.parent != root.parent / "gpu_guarded_leg":
        raise ValueError("Guarded-leg request is outside its separate target authorization directory")
    scope = request["scope"][0]
    configuration = bound_json(scope["configuration"])
    if configuration.get("target_root") != str(root) or configuration.get("algorithm_changed") is not False:
        raise ValueError("Existing guarded-leg configuration/target changed")
    for item in scope["existing_input_and_code_bindings"].values():
        check_binding(item)
    if configuration.get("code_and_inputs") != scope["existing_input_and_code_bindings"]:
        raise ValueError("Guarded policy description differs from actual code/input bindings")
    contract = read_json(path.parent / "issuer_runtime_checks.json")
    if contract.get("request") != binding(path):
        raise ValueError("Runtime contract uses another request")
    cpu = bound_json(contract["cpu_allocation"])
    if (cpu.get("request") != binding(path) or cpu["pure_cpu_phases"]["affinity"] != [26, 27]
            or cpu["cuda_propose_and_apply"]["affinity"] != [0, 1]):
        raise ValueError("Explicit CPU/CUDA affinity allocation differs")
    return dict(request=request, root=root, commands=commands, configuration=configuration,
                runtime=contract, cpu=cpu, request_binding=binding(path))


def _complete_native(value, keys, *, source_sha=None, calibration=None):
    if (value.get("passed") is not True or value.get("state") != "complete"
            or set(value.get("windows", {})) != keys or value.get("verified_windows") != len(keys)
            or value.get("total_windows") != len(keys)):
        raise ValueError("Real complete native window evidence is required")
    if source_sha and value.get("source_union_sha256") != source_sha:
        raise ValueError("Native evidence uses another source input")
    if calibration and value.get("calibration_digest") != calibration:
        raise ValueError("Native evidence uses another camera")


def _reviewed_identity(value, keys, source_sha, calibration):
    if (value.get("review_status") != "reviewed" or set(value.get("windows", {})) != keys
            or value.get("source_union_sha256") != source_sha or value.get("calibration_digest") != calibration
            or not value.get("evidence_files")):
        raise ValueError("Complete actual reviewed identity evidence is required")
    for path, sha in value["evidence_files"].items():
        check_binding(dict(path=path, sha256=sha))


def validate_readiness(checked, grant):
    request, root = checked["request"], checked["root"]
    ready = bound_json(grant["readiness"])
    schema_check(ready, bound_json(checked["runtime"]["readiness_schema"]))
    plan = read_json(root / "final_labels/repair_plan.json")
    keys = {j["source_id"] + ":" + Path(j["source_window"]).name for j in plan["jobs"]}
    if len(keys) != checked["configuration"]["planned_windows"]:
        raise ValueError("Actual formal body plan is incomplete")
    evidence = {role: bound_json(item) for role, item in ready["evidence"].items()}
    baseline_dir = Path(grant["readiness"]["path"]).resolve().parent
    for role in ("formal_preleg_native_verification", "formal_preleg_qa", "formal_complete_baseline"):
        if not Path(ready["evidence"][role]["path"]).resolve().is_relative_to(baseline_dir):
            raise ValueError("Authorization requires immutable separate pre-leg evidence snapshots")
    for day in ("1019", "1020"):
        native, identity = evidence[f"parent_{day}_native"], evidence[f"parent_{day}_identity"]
        parent_keys = {key for key in keys if key.startswith("2026" + day + "_")}
        _complete_native(native, parent_keys)
        _reviewed_identity(identity, parent_keys, native["source_union_sha256"], native["calibration_digest"])
    fit = evidence["parent_1020_full_fit"]
    expected_1020 = len(evidence["parent_1020_native"]["windows"])
    if (fit.get("state") != "fit_windows_complete" or fit.get("completed_windows") != expected_1020
            or fit.get("windows_total") != expected_1020 or fit.get("exit_code") != 0
            or fit.get("source_input_sha256") != evidence["parent_1020_native"]["source_union_sha256"]):
        raise ValueError("Actual final parent fitting completion is required")
    source_sha, calibration = plan["source_union_sha256"], plan["calibration_digest"]
    _complete_native(evidence["formal_preleg_native_verification"], keys,
                     source_sha=source_sha, calibration=calibration)
    _reviewed_identity(evidence["formal_reviewed_identity"], keys, source_sha, calibration)
    qa = evidence["formal_preleg_qa"]
    if (qa.get("schema") != "badminton-native-absolute-qa-v1" or set(qa.get("windows", {})) != keys
            or qa.get("source_union_sha256") != source_sha or qa.get("calibration_digest") != calibration
            or len(qa.get("frames", {})) != evidence["formal_preleg_native_verification"]["accepted_checked"]):
        raise ValueError("Immutable genuine pre-leg QA is incomplete")
    baseline = evidence["formal_complete_baseline"]
    if (baseline.get("schema") != "gvhmr-formal-preleg-baseline-v1" or baseline.get("state") != "complete"
            or baseline.get("source_input") != request["scope"][0]["manifest"]
            or set(baseline.get("windows", {})) != keys):
        raise ValueError("A complete formal baseline inventory is required")
    body = bound_json(baseline["body_receipt"])
    poses = bound_json(baseline["pose_receipt"])
    if any(not Path(baseline[role]["path"]).resolve().is_relative_to(baseline_dir)
           for role in ("body_receipt", "pose_receipt")):
        raise ValueError("Baseline body/native-pose receipts must be separate immutable snapshots")
    if (body.get("state") != "body_qa_candidate_complete" or body.get("windows") != len(keys)
            or body.get("accepted") != len(qa["frames"]) or poses.get("state") != "complete"
            or poses.get("source_union_sha256") != source_sha
            or poses.get("config", {}).get("calibration_digest") != calibration):
        raise ValueError("Formal baseline body/native-pose completion is missing")
    for key, entry in baseline["windows"].items():
        sid, name = key.split(":", 1)
        window = root / "final_labels/sessions" / sid / "windows" / name
        report = bound_json(entry["report_snapshot"])
        if not Path(entry["report_snapshot"]["path"]).resolve().is_relative_to(baseline_dir):
            raise ValueError("A pre-leg fit report snapshot is not immutable/separate")
        if binding(window / "fit_report.json")["sha256"] != entry["report_snapshot"]["sha256"]:
            raise ValueError("Current baseline fit report differs from its immutable pre-leg snapshot")
        if evidence["formal_preleg_native_verification"]["windows"][key]["body"]["report_sha256"] != (
                entry["report_snapshot"]["sha256"]):
            raise ValueError("Baseline inventory was not native verified")
        if (qa["windows"][key].get("fit_report_sha256") != entry["report_snapshot"]["sha256"]
                or qa["windows"][key].get("accepted") != report["accepted"]):
            raise ValueError("Absolute QA was not computed from this immutable baseline window")
        for relative, sha in report["artifacts_sha256"].items():
            path = (window / relative).resolve()
            if Path(relative).is_absolute() or not path.is_relative_to(window.resolve()):
                raise ValueError("Baseline artifact leaves its exact window")
            check_binding(dict(path=str(path), sha256=sha))
    imported = evidence["formal_numeric_import"]
    if (imported.get("schema") != "badminton-formal-numeric-import-v1" or imported.get("state") != "complete"
            or imported.get("source_union_sha256") != source_sha or imported.get("calibration_digest") != calibration
            or imported.get("numerical_artifacts_byte_identical") is not True
            or imported.get("qa_values_unchanged") is not True or set(imported.get("windows", {})) != keys):
        raise ValueError("Actual formal numerical import remains incomplete")
    for key, entry in imported["windows"].items():
        original, current = bound_json(entry["original_report"]), bound_json(entry["current_report"])
        parent = evidence["parent_1019_native"] if key.startswith("20261019_") else evidence["parent_1020_native"]
        if parent["windows"][key]["body"]["report_sha256"] != entry["original_report"]["sha256"]:
            raise ValueError("Imported original was not the actual verified parent window")
        if baseline["windows"][key]["report_snapshot"] != entry["current_report"]:
            raise ValueError("Numerical import and pre-leg baseline use different current reports")
        numeric = lambda report: {relative: sha for relative, sha in report["artifacts_sha256"].items()
                                  if relative == "ptv3.pth" or relative.startswith("fits/")}
        if "ptv3.pth" not in numeric(original) or numeric(original) != numeric(current):
            raise ValueError("Formal imported numerical artifacts changed")
    exits = evidence["predecessor_cuda_exit_bundle"]
    if (exits.get("schema") != "gvhmr-guarded-leg-predecessor-exit-bundle-v1"
            or exits.get("state") != "complete" or exits.get("predecessor_grant") != request["predecessor_continuous_grant"]
            or not exits.get("process_identities") or not exits.get("release_receipts")):
        raise ValueError("Actual preceding CUDA release bundle is required")
    for item in exits["release_receipts"]:
        if bound_json(item).get("status") != "released":
            raise ValueError("A preceding CUDA lease has not actually released")
    if any(identity_alive(item) for item in exits["process_identities"]):
        raise ValueError("A predecessor process is still alive")
    guard = evidence["independent_guard_ready"]
    if (guard.get("schema") != "gvhmr-guarded-leg-guard-readiness-v1" or guard.get("state") != "implementation_tested"
            or guard.get("implementation") != grant["independent_guard"]):
        raise ValueError("Independent tested guard implementation is missing")
    return ready


def load_authorization(request_path, grant_path, *, now=None, deep=True):
    checked = inspect_request(request_path)
    grant_binding = binding(grant_path)
    grant = bound_json(grant_binding)
    schema_check(grant, bound_json(checked["runtime"]["grant_schema"]))
    authorization = read_continuous_grant(request_path, grant_path)
    if (authorization["binding"] != grant_binding or authorization["grant"] != grant
            or authorization["request"] != checked["request_binding"]):
        raise ValueError("Request/grant changed during authorization")
    for item in (grant["configuration"], grant["cpu_allocation"], grant["predecessor_continuous_grant"],
                 grant["readiness"]):
        check_binding(item)
    verify_implementation(grant["independent_guard"], checked)
    now = now or datetime.now(timezone.utc)
    deadline = min(authorization["deadline"], utc_time(checked["request"]["absolute_hard_deadline_utc"]))
    if (authorization["deadline"] != deadline or deadline > authorization["start"] + timedelta(minutes=30)
            or not authorization["start"] <= now < deadline):
        raise ValueError("Guarded-leg authorization is outside its real fixed time window")
    if deep:
        checked["readiness"] = validate_readiness(checked, grant)
    check_binding(grant_binding)
    check_binding(checked["request_binding"])
    checked.update(authorization=authorization, deadline=deadline, grant_path=Path(grant_path).resolve())
    return checked


def immutable_anchors(checked):
    grant, request = checked["authorization"]["grant"], checked["request"]
    ready = bound_json(grant["readiness"])
    values = [checked["request_binding"], checked["authorization"]["binding"],
        grant["readiness"], grant["configuration"], grant["cpu_allocation"], grant["independent_guard"],
        binding(Path(checked["request_binding"]["path"]).parent / "issuer_runtime_checks.json"),
        checked["runtime"]["grant_schema"], checked["runtime"]["readiness_schema"]]
    values += list(ready["evidence"].values())
    baseline = bound_json(ready["evidence"]["formal_complete_baseline"])
    values += [baseline["body_receipt"], baseline["pose_receipt"]]
    values += [entry["report_snapshot"] for entry in baseline["windows"].values()]
    imported = bound_json(ready["evidence"]["formal_numeric_import"])
    values += [entry["original_report"] for entry in imported["windows"].values()]
    values += list(request["scope"][0]["existing_input_and_code_bindings"].values())
    implementation = bound_json(grant["independent_guard"])
    values += list(implementation["code"].values()) + [implementation["test_execution"]]
    values += bound_json(implementation["test_execution"])["test_files"]
    unique = {(x["path"], x["sha256"]): x for x in values}
    return list(unique.values())
