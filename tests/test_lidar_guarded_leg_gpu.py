"""Synthetic CPU authorization cases; no production grant or CUDA launch."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from hmr4d.lidar_mocap import gpu_handoff, guarded_leg_gpu as auth, guarded_leg_execution as execution
from hmr4d.lidar_mocap.pilot import binding


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return binding(path)


def constants(value):
    return dict(type="object", required=list(value), properties={k: dict(const=v) for k, v in value.items()})


@pytest.fixture
def stage(tmp_path, monkeypatch, request):
    runtime_name = getattr(request, "param", None)
    for module in (auth, execution, gpu_handoff):
        monkeypatch.setattr(module, "PROJ_ROOT", tmp_path)
    root = tmp_path / "outputs/formal/repair"
    folder = root.parent / "gpu_guarded_leg"
    baseline_dir = folder / "baseline"
    request_path, grant_path = folder / "request.json", folder / "grant.json"
    source = put(root / "new_source_input.json", dict(schema="test-source-only"))
    keys = ["20261019_152542:rally_1", "20261020_182854:rally_2"]
    jobs = [dict(source_id=k.split(":")[0], source_window="source/" + k.split(":")[1]) for k in keys]
    plan = put(root / "final_labels/repair_plan.json", dict(jobs=jobs, source_union_sha256=source["sha256"],
                                                          calibration_digest="c" * 64))
    code = {}
    for name, path in auth.implementation_files().items():
        code[name] = put(path, dict(test_code=name))
    for name in auth.ENTRY_NAMES:
        put(tmp_path / "tools" / name, dict(test_entry=name))
    commands = []
    for phase, name in zip(auth.PHASES, auth.ENTRY_NAMES):
        entry = tmp_path / "tools" / name
        argv = [str(tmp_path / ".venv/Scripts/python.exe"), "-u", str(entry), "--root", str(root)]
        if phase != "select":
            argv += ["--threads", "2"]
        commands.append(dict(phase=phase, cuda=phase != "select", argv=argv, entry_binding=binding(entry)))
    inputs = dict(plan=plan, source=source)
    config = put(folder / "configuration.json", dict(target_root=str(root), algorithm_changed=False,
                                                     planned_windows=2, code_and_inputs=inputs))
    predecessor = put(folder / "old_grant.json", dict(status="old-test-fixture"))
    sports = put(folder / "sports.json", dict(status="stopped-test-fixture"))
    exits = [put(folder / f"exit_{i}.json", dict(status="released", test_fixture=i)) for i in range(4)]
    request = dict(schema="gvhmr-continuous-production-gpu-request-v1", authorized_stages_requested=["guarded_leg"],
        requested_local_stage_lease_limits_minutes=dict(guarded_leg=30), requested_maximum_hours=.5,
        max_concurrent_cuda_trees=1, requested_cuda_visible_devices="0", gpu_started=False, training_published=False,
        minimum_remaining_launch_seconds=900, absolute_hard_deadline_utc="2026-10-23T11:41:39.650687+00:00",
        sports_stop_receipt=sports, predecessor_continuous_grant=predecessor,
        scope=[dict(root=str(root), manifest=source, stage="guarded_leg", configuration=config,
                    existing_input_and_code_bindings=inputs, commands=commands)])
    if runtime_name is not None:
        request["issuer_runtime_checks_file"] = runtime_name
    request_binding = put(request_path, request)
    cpu = put(folder / "cpu.json", dict(request=request_binding, pure_cpu_phases=dict(affinity=[26, 27]),
                                      cuda_propose_and_apply=dict(affinity=[0, 1])))
    snapshots, originals, parent_natives, identities = {}, {}, {}, {}
    for key in keys:
        sid, name = key.split(":")
        window = root / "final_labels/sessions" / sid / "windows" / name
        tensor = put(window / "ptv3.pth", dict(synthetic_numeric_value=key))
        fit = put(window / "fits/body.npz", dict(synthetic_body_value=key))
        artifacts = {"ptv3.pth": tensor["sha256"], "fits/body.npz": fit["sha256"]}
        report = dict(artifacts_sha256=artifacts, accepted=1)
        put(window / "fit_report.json", report)
        snapshots[key] = dict(report_snapshot=put(baseline_dir / sid / name / "fit_report.json", report))
        originals[key] = put(folder / "parent_reports" / sid / name / "fit_report.json", report)
        day = sid[4:8]
        parent_source = "1" * 64 if day == "1019" else "2" * 64
        native = dict(state="complete", passed=True, windows={key: dict(body=dict(report_sha256=originals[key]["sha256"]))},
                      verified_windows=1, total_windows=1, source_union_sha256=parent_source, calibration_digest="d" * 64)
        parent_natives[day] = put(folder / f"parent_{day}_native.json", native)
        identities[day] = put(folder / f"parent_{day}_identity.json", dict(review_status="reviewed", windows={key: {}},
            source_union_sha256=parent_source, calibration_digest="d" * 64,
            evidence_files={parent_natives[day]["path"]: parent_natives[day]["sha256"]}))
    ready = dict(schema="gvhmr-guarded-leg-baseline-readiness-v1", state="complete", request=request_binding,
                 root=str(root), configuration=config, training_published=False, checks=dict(all_complete=True))
    ready_schema = constants(ready)
    ready_schema["required"].append("evidence")
    roles = ["predecessor_cuda_exit_bundle", "parent_1019_native", "parent_1019_identity", "parent_1020_native",
             "parent_1020_identity", "parent_1020_full_fit", "formal_numeric_import", "formal_complete_baseline",
             "formal_reviewed_identity", "formal_preleg_native_verification", "formal_preleg_qa", "independent_guard_ready"]
    ready_schema["properties"]["evidence"] = dict(type="object", required=roles,
        properties={r: dict(type="object", required=["path", "sha256"], properties=dict(
            path=dict(type="string", minLength=1), sha256=dict(type="string", pattern="^[0-9a-f]{64}$"))) for r in roles},
        additionalProperties=False)
    ready_schema_binding = put(folder / "readiness.schema.json", ready_schema)
    start = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
    grant = dict(schema="gvhmr-continuous-production-gpu-grant-v1", status="held", owner="GVHMR",
        request=request_binding, request_sha256=request_binding["sha256"],
        authorized_roots_with_actual_manifest_sha256={str(root): dict(manifest_path=source["path"], manifest_sha256=source["sha256"])},
        allow_local_stage_leases=True, native_gpu_returned=True, native_exit_receipts=exits, sports_stop_receipt=sports,
        maximum_stage_minutes=dict(guarded_leg=30), training_published=False, configuration=config, cpu_allocation=cpu,
        predecessor_continuous_grant=predecessor)
    grant_schema = constants(grant)
    for name in ("approved_start_utc", "approved_deadline_utc"):
        grant_schema["required"].append(name)
        grant_schema["properties"][name] = dict(type="string", format="date-time")
    for name in ("readiness", "independent_guard"):
        grant_schema["required"].append(name)
        grant_schema["properties"][name] = dict(type="object", required=["path", "sha256"])
    runtime = put(folder / (runtime_name or "issuer_runtime_checks.json"), dict(
        request=request_binding, cpu_allocation=cpu,
        grant_schema=put(folder / "grant.schema.json", grant_schema), readiness_schema=ready_schema_binding))
    test_file = put(tmp_path / "tests/test_guard.py", dict(synthetic_test_source=True))
    tested = put(folder / "cpu_tests.json", dict(schema="gvhmr-guarded-leg-cpu-test-v1", state="passed", passed=1,
        failed=0, exit_code=0, gpu_started=False, actual_affinity=[26, 27], code=code, test_files=[test_file]))
    implementation = put(folder / "implementation.json", dict(schema="gvhmr-guarded-leg-implementation-v1",
        request=request_binding, runtime_contract=runtime, code=code, test_execution=tested))
    formal_native = dict(state="complete", passed=True,
        windows={k: dict(body=dict(report_sha256=snapshots[k]["report_snapshot"]["sha256"])) for k in keys},
        verified_windows=2, total_windows=2, accepted_checked=2,
        source_union_sha256=source["sha256"], calibration_digest="c" * 64)
    native = put(baseline_dir / "native.json", formal_native)
    qa = put(baseline_dir / "qa.json", dict(schema="badminton-native-absolute-qa-v1", windows={k: dict(
        fit_report_sha256=snapshots[k]["report_snapshot"]["sha256"], accepted=1) for k in keys},
        frames={k: {} for k in keys}, source_union_sha256=source["sha256"], calibration_digest="c" * 64))
    baseline = put(baseline_dir / "baseline.json", dict(schema="gvhmr-formal-preleg-baseline-v1", state="complete",
        source_input=source, windows=snapshots,
        body_receipt=put(baseline_dir / "body.json", dict(state="body_qa_candidate_complete", windows=2, accepted=2)),
        pose_receipt=put(baseline_dir / "pose.json", dict(state="complete", source_union_sha256=source["sha256"],
                                                       config=dict(calibration_digest="c" * 64)))))
    imported = put(folder / "numeric_import.json", dict(schema="badminton-formal-numeric-import-v1", state="complete",
        source_union_sha256=source["sha256"], calibration_digest="c" * 64, numerical_artifacts_byte_identical=True,
        qa_values_unchanged=True, windows={k: dict(original_report=originals[k], current_report=snapshots[k]["report_snapshot"]) for k in keys}))
    evidence = dict(parent_1019_native=parent_natives["1019"], parent_1019_identity=identities["1019"],
        parent_1020_native=parent_natives["1020"], parent_1020_identity=identities["1020"],
        parent_1020_full_fit=put(folder / "fit.json", dict(state="fit_windows_complete", completed_windows=1,
                                                       windows_total=1, exit_code=0, source_input_sha256="2" * 64)),
        formal_numeric_import=imported, formal_complete_baseline=baseline,
        formal_reviewed_identity=put(folder / "formal_identity.json", dict(review_status="reviewed", windows={k: {} for k in keys},
            source_union_sha256=source["sha256"], calibration_digest="c" * 64, evidence_files={native["path"]: native["sha256"]})),
        formal_preleg_native_verification=native, formal_preleg_qa=qa,
        predecessor_cuda_exit_bundle=put(folder / "previous.json", dict(schema="gvhmr-guarded-leg-predecessor-exit-bundle-v1",
            state="complete", predecessor_grant=predecessor, release_receipts=exits,
            process_identities=[dict(pid=os.getpid(), create_time=psutil.Process().create_time() - 1)])),
        independent_guard_ready=put(folder / "guard_ready.json", dict(schema="gvhmr-guarded-leg-guard-readiness-v1",
            state="implementation_tested", implementation=implementation)))
    ready["evidence"] = evidence
    ready_binding = put(baseline_dir / "readiness.json", ready)
    grant.update(readiness=ready_binding, independent_guard=implementation, approved_start_utc=start.isoformat(),
                 approved_deadline_utc=(start + timedelta(minutes=30)).isoformat())
    put(grant_path, grant)
    return SimpleNamespace(root=root, folder=folder, request=request, request_path=request_path, grant=grant,
        grant_path=grant_path, start=start, ready=ready, evidence=evidence, keys=set(keys), checked=lambda: auth.load_authorization(
            request_path, grant_path, now=start + timedelta(seconds=1)))


def test_real_bindings_complete_before_authorization_and_no_lease_side_effect(stage):
    checked = stage.checked()
    assert checked["root"] == stage.root
    assert [x["cuda"] for x in checked["commands"]] == [True, False, True]
    assert checked["runtime_binding"] == binding(stage.folder / "issuer_runtime_checks.json")
    assert auth.bound_json(stage.grant["independent_guard"])["runtime_contract"] == checked["runtime_binding"]
    assert not (stage.folder / "guarded_leg_stage_lease.json").exists()
    assert auth.immutable_anchors(checked)


@pytest.mark.parametrize("stage", ["issuer_runtime_checks_v2.json"], indirect=True)
def test_explicit_runtime_contract_binds_actual_request_implementation_and_anchors(stage):
    selected = stage.folder / stage.request["issuer_runtime_checks_file"]
    legacy_request = deepcopy(stage.request)
    legacy_request.pop("issuer_runtime_checks_file")
    old_request = put(stage.folder / "legacy_request.json", legacy_request)
    legacy_contract = dict(auth.read_json(selected), request=old_request)
    default = stage.folder / "issuer_runtime_checks.json"
    old_contract = put(default, legacy_contract)
    old_bytes = default.read_bytes()

    checked = stage.checked()
    assert checked["request_binding"] == binding(stage.request_path)
    assert checked["runtime"]["request"] == checked["request_binding"]
    assert checked["runtime_binding"] == binding(selected)
    implementation = auth.verify_implementation(stage.grant["independent_guard"], checked)
    assert implementation["runtime_contract"] == binding(selected)
    anchors = auth.immutable_anchors(checked)
    assert binding(selected) in anchors
    assert old_contract not in anchors
    assert default.read_bytes() == old_bytes

    implementation["runtime_contract"] = old_contract
    mixed = put(stage.folder / "mixed_implementation.json", implementation)
    with pytest.raises(ValueError, match="issued target contract"):
        auth.verify_implementation(mixed, checked)


@pytest.mark.parametrize("stage", ["issuer_runtime_checks_v2.json"], indirect=True)
@pytest.mark.parametrize("change", ["path", "sha256", "request_bytes"])
def test_selected_runtime_contract_requires_exact_request_path_and_sha(stage, change):
    selected = stage.folder / stage.request["issuer_runtime_checks_file"]
    runtime = auth.read_json(selected)
    if change == "path":
        alias = stage.folder / "same_bytes_request.json"
        alias.write_bytes(stage.request_path.read_bytes())
        runtime["request"] = binding(alias)
    elif change == "sha256":
        runtime["request"]["sha256"] = "f" * 64
    else:
        put(stage.request_path, dict(stage.request, note="new request bytes need their own authority"))
    put(selected, runtime)
    with pytest.raises(ValueError, match="Runtime contract uses another request"):
        auth.inspect_request(stage.request_path)


@pytest.mark.parametrize("stage", ["issuer_runtime_checks_v2.json"], indirect=True)
def test_selected_runtime_contract_change_cannot_be_rebound_after_inspection(stage):
    checked = stage.checked()
    selected = stage.folder / stage.request["issuer_runtime_checks_file"]
    put(selected, dict(checked["runtime"], changed_after_inspection=True))
    with pytest.raises(ValueError, match="Pilot input changed"):
        auth.verify_implementation(stage.grant["independent_guard"], checked)
    with pytest.raises(ValueError, match="Pilot input changed"):
        auth.immutable_anchors(checked)


@pytest.mark.parametrize(
    "name",
    [None, False, 7, [], {}, "", " ", "/outside.json", "C:\\outside.json", "C:outside.json",
     "../outside.json", "..\\outside.json", "sub/contract.json", "sub\\contract.json",
     "contract..json", "contract.txt", "contract\x00.json"],
)
def test_runtime_contract_field_rejects_nonfilenames_before_reading(stage, name):
    put(stage.request_path, dict(stage.request, issuer_runtime_checks_file=name))
    with pytest.raises(ValueError, match="plain same-directory JSON filename"):
        auth.inspect_request(stage.request_path)


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_explicit_runtime_contract_requires_regular_file_without_legacy_fallback(stage, kind):
    name = kind + ".json"
    if kind == "directory":
        (stage.folder / name).mkdir()
    put(stage.request_path, dict(stage.request, issuer_runtime_checks_file=name))
    assert (stage.folder / "issuer_runtime_checks.json").is_file()
    with pytest.raises((FileNotFoundError, ValueError)):
        auth.inspect_request(stage.request_path)


@pytest.mark.parametrize("outside", [False, True], ids=["same-directory-link", "escaping-link"])
def test_runtime_contract_symlinks_are_rejected(stage, outside):
    destination = stage.folder.parent if outside else stage.folder
    target = destination / "symlink_target.json"
    put(target, auth.read_json(stage.folder / "issuer_runtime_checks.json"))
    link = stage.folder / "linked_runtime.json"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip("Actual symlink creation unavailable: " + str(error))
    put(stage.request_path, dict(stage.request, issuer_runtime_checks_file=link.name))
    with pytest.raises(ValueError, match="regular JSON file in the request directory"):
        auth.inspect_request(stage.request_path)


@pytest.mark.parametrize("kind", ["symlink", "resolved_escape"])
def test_runtime_contract_rejects_simulated_symlink_and_resolved_escape(stage, monkeypatch, kind):
    """Exercise both refusal branches even when actual Windows symlink creation is unavailable."""
    target = stage.folder / "simulated_runtime.json"
    put(target, auth.read_json(stage.folder / "issuer_runtime_checks.json"))
    put(stage.request_path, dict(stage.request, issuer_runtime_checks_file=target.name))
    if kind == "symlink":
        is_symlink = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda path: path == target or is_symlink(path))
    else:
        resolve = Path.resolve
        outside = stage.folder.parent / target.name

        def resolved_path(path, *args, **kwargs):
            return outside if path == target else resolve(path, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", resolved_path)
    with pytest.raises(ValueError, match="regular JSON file in the request directory"):
        auth.inspect_request(stage.request_path)


@pytest.mark.parametrize("change", ["binding_escape", "link_after_read"])
def test_runtime_contract_path_is_rechecked_after_simulated_binding_or_read_replacement(stage, monkeypatch, change):
    target = stage.folder / "issuer_runtime_checks.json"
    if change == "binding_escape":
        outside = stage.folder.parent / "outside_runtime.json"
        put(outside, auth.read_json(target))
        original_binding = auth.binding
        monkeypatch.setattr(auth, "binding", lambda path: original_binding(outside if Path(path) == target else path))
        expected = "path changed during inspection"
    else:
        linked = [False]
        original_read = auth.bound_json
        is_symlink = Path.is_symlink

        def read_then_replace(item):
            result = original_read(item)
            if Path(item["path"]) == target:
                linked[0] = True
            return result

        monkeypatch.setattr(auth, "bound_json", read_then_replace)
        monkeypatch.setattr(Path, "is_symlink", lambda path: (path == target and linked[0]) or is_symlink(path))
        expected = "regular JSON file in the request directory"
    with pytest.raises(ValueError, match=expected):
        auth.inspect_request(stage.request_path)


@pytest.mark.parametrize("stage", ["issuer_runtime_checks_v2.json"], indirect=True)
def test_versioned_runtime_contract_cannot_extend_an_expired_grant(stage):
    with pytest.raises(ValueError, match="real fixed time window"):
        auth.load_authorization(stage.request_path, stage.grant_path, now=stage.start + timedelta(minutes=30))


def test_missing_grant_cannot_probe_gpu_create_lease_or_launch(stage, monkeypatch):
    probe = Mock(side_effect=AssertionError("GPU probe should not be reached"))
    monkeypatch.setattr(execution, "cpu_only", lambda: None)
    monkeypatch.setattr(execution, "compute_pids", probe)
    with pytest.raises(FileNotFoundError):
        execution.execute(stage.request_path, stage.folder / "not_issued.json")
    probe.assert_not_called()
    assert not (stage.folder / "guarded_leg_stage_lease.json").exists()
    assert not (stage.folder / "execution.lock.json").exists()


@pytest.mark.parametrize("change", ["failed", "gpu", "cpu", "test_source"])
def test_guard_requires_actual_passed_cpu_tests_for_current_implementation(stage, change):
    implementation = auth.bound_json(stage.grant["independent_guard"])
    tested = auth.bound_json(implementation["test_execution"])
    if change == "failed": tested["failed"] = 1
    if change == "gpu": tested["gpu_started"] = True
    if change == "cpu": tested["actual_affinity"] = [0, 1]
    if change == "test_source": Path(tested["test_files"][0]["path"]).write_bytes(b"changed")
    implementation["test_execution"] = put(Path(implementation["test_execution"]["path"]), tested)
    grant = deepcopy(stage.grant)
    grant["independent_guard"] = put(Path(grant["independent_guard"]["path"]), implementation)
    put(stage.grant_path, grant)
    with pytest.raises(ValueError): stage.checked()


@pytest.mark.parametrize("change", ["target", "option", "interpreter", "script", "extra_scope"])
def test_command_or_target_changes_are_rejected(stage, change):
    request = deepcopy(stage.request)
    if change == "target": request["scope"][0]["commands"][0]["argv"][4] += "_other"
    if change == "option": request["scope"][0]["commands"][0]["argv"] += ["--wait"]
    if change == "interpreter": request["scope"][0]["commands"][0]["argv"][0] = "python"
    if change == "script": request["scope"][0]["commands"][0]["entry_binding"]["sha256"] = "f" * 64
    if change == "extra_scope": request["scope"] *= 2
    with pytest.raises(ValueError): auth.approved_commands(request)


@pytest.mark.parametrize("change", ["pending", "missing_window", "incomplete_fit", "identity", "native", "changed_tensor",
                                  "foreign_import", "live_predecessor", "unreleased", "mutable_qa", "guard"])
def test_incomplete_or_false_readiness_cannot_authorize(stage, change):
    ready = deepcopy(stage.ready)
    if change == "pending":
        ready["evidence"]["formal_numeric_import"] = dict(path=None, sha256=None)
    else:
        role = {"missing_window": "formal_preleg_qa", "incomplete_fit": "parent_1020_full_fit",
            "identity": "parent_1020_identity", "native": "parent_1020_native", "foreign_import": "formal_numeric_import",
            "live_predecessor": "predecessor_cuda_exit_bundle", "unreleased": "predecessor_cuda_exit_bundle",
            "mutable_qa": "formal_preleg_qa", "guard": "independent_guard_ready"}.get(change)
        if role:
            value = auth.bound_json(ready["evidence"][role])
            if change == "missing_window": value["windows"].pop(next(iter(value["windows"])))
            if change == "incomplete_fit": value["completed_windows"] = 0
            if change == "identity": value["review_status"] = "pending"
            if change == "native": value["passed"] = False
            if change == "foreign_import": next(iter(value["windows"].values()))["original_report"]["sha256"] = "0" * 64
            if change == "live_predecessor": value["process_identities"] = [dict(pid=os.getpid(), create_time=psutil.Process().create_time())]
            if change == "unreleased":
                value["release_receipts"] = [put(stage.folder / "held.json", dict(status="held"))]
            if change == "guard": value["state"] = "not_tested"
            dest = Path(ready["evidence"][role]["path"])
            if change == "mutable_qa": dest = stage.root / "evaluation/mutable_qa.json"
            ready["evidence"][role] = put(dest, value)
        else:
            tensor = next((stage.root / "final_labels/sessions").glob("*/windows/*/ptv3.pth"))
            tensor.write_bytes(b"changed")
    grant = deepcopy(stage.grant)
    grant["readiness"] = put(Path(grant["readiness"]["path"]), ready)
    put(stage.grant_path, grant)
    with pytest.raises((ValueError, FileNotFoundError)):
        stage.checked()


@pytest.mark.parametrize("change", ["expired", "future", "long", "naive", "old_grant", "implementation"])
def test_time_old_grant_and_changed_implementation_fail_closed(stage, change):
    grant = deepcopy(stage.grant)
    if change == "expired": grant["approved_deadline_utc"] = (stage.start + timedelta(milliseconds=1)).isoformat()
    if change == "future": grant["approved_start_utc"] = (stage.start + timedelta(minutes=1)).isoformat()
    if change == "long": grant["approved_deadline_utc"] = (stage.start + timedelta(minutes=31)).isoformat()
    if change == "naive": grant["approved_deadline_utc"] = "2026-10-22T12:30:00"
    if change == "old_grant": grant["status"] = "released"
    if change == "implementation": auth.implementation_files()["execution"].write_bytes(b"changed")
    put(stage.grant_path, grant)
    with pytest.raises(ValueError): stage.checked()


def test_first_launch_margin_hard_cutoff_and_reused_pid(stage):
    checked = stage.checked()
    execution._assert_time(checked, first=True, now=stage.start + timedelta(minutes=15))
    with pytest.raises(TimeoutError):
        execution._assert_time(checked, first=True, now=stage.start + timedelta(minutes=15, seconds=1))
    with pytest.raises(TimeoutError):
        execution._assert_time(checked, now=checked["deadline"] - timedelta(seconds=10))
    own = psutil.Process()
    identity = dict(pid=own.pid, create_time=own.create_time(), command=own.cmdline())
    assert auth.identity_alive(identity)
    assert not auth.identity_alive(dict(identity, create_time=identity["create_time"] - 1))
    with pytest.raises(ValueError): auth.identity_alive(dict(identity, command=["other"]))


def test_guard_honors_shorter_local_lease_deadline_without_grant_extension(stage):
    checked = stage.checked()
    lease = gpu_handoff.make_local_stage_lease(checked["authorization"], stage.root, "guarded_leg", now=stage.start)
    lease["approved_deadline_utc"] = (stage.start + timedelta(minutes=20)).isoformat()
    execution._use_lease_deadline(checked, lease)
    assert checked["deadline"] == stage.start + timedelta(minutes=20)
    lease["approved_deadline_utc"] = (stage.start + timedelta(minutes=21)).isoformat()
    with pytest.raises(ValueError): execution._use_lease_deadline(checked, lease)


def test_original_absolute_hard_cap_also_limits_an_otherwise_valid_grant(stage, monkeypatch):
    inspect = auth.inspect_request

    def earlier_cap(path):
        checked = inspect(path)
        checked["request"]["absolute_hard_deadline_utc"] = (stage.start + timedelta(minutes=29)).isoformat()
        return checked

    monkeypatch.setattr(auth, "inspect_request", earlier_cap)
    with pytest.raises(ValueError, match="fixed time window"):
        stage.checked()


def test_json_schema_closed_vocabulary_rejects_pending_timezone_and_extra_scope():
    rule = dict(type="object", required=["proof", "deadline", "single"], additionalProperties=False,
        properties=dict(proof={"$ref": "#/$defs/binding"}, deadline=dict(type="string", format="date-time"),
                        single=dict(allOf=[dict(type="integer", minimum=1, maximum=1), dict(const=1)])),
        **{"$defs": dict(binding=dict(type="object", required=["path", "sha256"], properties=dict(
            path=dict(type="string", minLength=1), sha256=dict(type="string", pattern="^[0-9a-f]{64}$"))))})
    valid = dict(proof=dict(path="actual.json", sha256="a" * 64), deadline="2026-10-22T12:00:00+00:00", single=1)
    auth.schema_check(valid, rule)
    for value in (dict(valid, extra=True), dict(valid, deadline="2026-10-22T12:00:00"), dict(valid, single=True),
                  dict(valid, proof=dict(path=None, sha256=None))):
        with pytest.raises(ValueError): auth.schema_check(value, rule)
    with pytest.raises(ValueError): auth.schema_check(valid, dict(rule, unknownKeyword=True))
    with pytest.raises(ValueError): auth.schema_check(None, {"$ref": "https://invalid/schema"})
    with pytest.raises(ValueError): auth.schema_check([1, 1], dict(type="array", uniqueItems=True))


def test_duplicate_nonfinite_and_changed_bound_json_are_rejected(tmp_path):
    target = tmp_path / "bad.json"
    for text in ('{"status":"held","status":"released"}', '{"deadline":NaN}', '{"value":1e999}'):
        target.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError): auth.read_json(target)
    proof = put(target, dict(complete=True))
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError): auth.bound_json(proof)


def test_watchdog_context_is_bound_to_exact_driver_target_and_anchors(stage):
    checked = stage.checked()
    nonce = "a" * 32
    directory = stage.folder / "attempts" / nonce
    context = dict(nonce=nonce, job_name="Local\\GVHMRGuardedLeg_" + nonce, root=str(stage.root),
        request=binding(stage.request_path), driver=dict(command=["base_python.exe", "-u",
            str(execution.PROJ_ROOT / "tools/run_badminton_guarded_leg.py"), "--execute", "--request",
            str(stage.request_path), "--grant", str(stage.grant_path)]), immutable_bindings=auth.immutable_anchors(checked))
    execution.validate_context(context, directory / "context.json", checked)
    for change in (dict(context, root=str(stage.root) + "_other"), dict(context, job_name="foreign"),
                   dict(context, immutable_bindings=[]), dict(context, driver=dict(command=["other"]))):
        with pytest.raises(ValueError): execution.validate_context(change, directory / "context.json", checked)


def test_applied_receipt_counts_only_changed_windows_and_does_not_require_all_291(stage):
    keys = list(stage.keys)
    selection = dict(state="complete", selected_frames=1, windows={k: {} for k in keys})
    selection_binding = put(stage.root / "leg_refinement_selection.json", selection)
    applied = dict(state="complete", selected_frames=1, updated_frames=1, selection_sha256=selection_binding["sha256"],
                   windows={keys[0]: dict(frames=1)})
    put(stage.root / "leg_refinement_receipt.json", applied)
    put(stage.root / "final_labels/post_refinement.json", applied)
    assert execution._phase_output(stage.root, "apply", 2)
    with pytest.raises(ValueError): execution._assert_clean_outputs(stage.root)
    put(stage.root / "leg_refinement_receipt.json", dict(applied, updated_frames=0))
    with pytest.raises(ValueError): execution._phase_output(stage.root, "apply", 2)


def test_zero_selected_frames_is_an_actual_complete_postpass(stage):
    selection_binding = put(stage.root / "leg_refinement_selection.json", dict(
        state="complete", selected_frames=0, windows={k: {} for k in stage.keys}))
    applied = dict(state="complete", selected_frames=0, updated_frames=0,
                   selection_sha256=selection_binding["sha256"], windows={})
    put(stage.root / "leg_refinement_receipt.json", applied)
    put(stage.root / "final_labels/post_refinement.json", applied)
    assert execution._phase_output(stage.root, "apply", 2)


@pytest.mark.parametrize("change", ["scope", "source", "proposal_hash"])
def test_complete_selection_must_use_exact_native_windows_source_and_proposals(stage, change):
    proposals = put(stage.root / "leg_refinement_proposals/receipt.json", dict(state="complete"))
    selected = dict(state="complete", windows={k: {} for k in stage.keys},
        source_union_sha256=stage.request["scope"][0]["manifest"]["sha256"], source_proposals_sha256=proposals["sha256"])
    if change == "scope": selected["windows"] = dict(foreign_1={}, foreign_2={})
    if change == "source": selected["source_union_sha256"] = "f" * 64
    if change == "proposal_hash": selected["source_proposals_sha256"] = "f" * 64
    put(stage.root / "leg_refinement_selection.json", selected)
    with pytest.raises(ValueError): execution._phase_output(stage.root, "select", 2)


def test_foreign_compute_is_never_ignored_due_to_reused_pid():
    own = psutil.Process()
    observed = {(own.pid, own.create_time() - 1): {}}
    assert execution.foreign_compute_pids({own.pid}, [], observed) == {own.pid}
