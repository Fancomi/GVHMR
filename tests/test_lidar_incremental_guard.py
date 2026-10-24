"""Synthetic, real-byte metadata only. These fixtures never authorize production."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from hmr4d.lidar_mocap import formal_completion, incremental_guard as auth
from hmr4d.lidar_mocap.full_native import pose_config_digest
from hmr4d.lidar_mocap.pipeline import BadmintonPipeline
from test_lidar_incremental_release import make_policy, write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def asset(path, data=b"synthetic CPU fixture, not a model/tensor"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return auth.binding(path)


def commands(project, repair):
    values = []
    for phase, name in zip(auth.PHASES, auth.ENTRY_NAMES):
        entry = project / "tools" / name
        argv = [str(project / ".venv/Scripts/python.exe"), "-u", str(entry), "--root", str(repair)]
        if phase != "select":
            argv += ["--threads", "2"]
        values.append(dict(phase=phase, cuda=phase != "select", argv=argv, entry_binding=auth.binding(entry)))
    return values


def refresh(stage):
    """Rebind a synthetic acyclic request, reaching semantic gates after edits."""
    descriptor = write(stage.descriptor_path, stage.descriptor)
    producers = {
        role: auth.binding(stage.project / name)
        for role, name in (
            ("refinement_policy", "hmr4d/lidar_mocap/leg_refinement.py"),
            ("selection_policy", "hmr4d/lidar_mocap/leg_selection.py"),
            ("application_entry", "tools/apply_badminton_leg_refinement.py"),
        )
    }
    stage.request["incremental_descriptor"] = descriptor
    stage.request["scope"][0].update(
        commands=commands(stage.project, stage.repair), existing_input_and_code_bindings=producers
    )
    stage.configuration["code_and_inputs"] = producers
    configuration = write(stage.folder / "configuration.json", stage.configuration)
    stage.request["scope"][0]["configuration"] = configuration
    request = write(stage.request_path, stage.request)
    cpu = write(
        stage.folder / "cpu.json",
        dict(request=request, pure_cpu_phases=dict(affinity=[26, 27]), cuda_propose_and_apply=dict(affinity=[0, 1])),
    )
    stage.runtime.update(request=request, incremental_descriptor=descriptor, cpu_allocation=cpu)
    runtime = write(stage.folder / "runtime.json", stage.runtime)
    code = {role: auth.binding(path) for role, path in auth.implementation_files().items()}
    stage.test_execution["code"] = code
    tested = write(stage.folder / "tests.json", stage.test_execution)
    stage.implementation.update(
        request=request, descriptor=descriptor, runtime_contract=runtime, code=code, test_execution=tested
    )
    implementation = write(stage.folder / "implementation.json", stage.implementation)
    guard = write(
        stage.folder / "tested-guard.json",
        dict(
            schema="gvhmr-guarded-leg-guard-readiness-v1", state="implementation_tested", implementation=implementation
        ),
    )
    stage.ready.update(request=request, incremental_descriptor=descriptor, configuration=configuration)
    stage.ready["evidence"]["independent_guard_ready"] = guard
    ready = write(stage.folder / "baseline/readiness.json", stage.ready)
    stage.grant.update(
        request=request,
        request_sha256=request["sha256"],
        incremental_descriptor=descriptor,
        configuration=configuration,
        cpu_allocation=cpu,
        readiness=ready,
        independent_guard=implementation,
    )
    write(stage.grant_path, stage.grant)
    stage.descriptor_binding = descriptor
    stage.kwargs = dict(descriptor_path=stage.descriptor_path, descriptor_sha256=descriptor["sha256"])
    return stage


@pytest.fixture
def stage(tmp_path, monkeypatch):
    selected = make_policy(tmp_path)
    source_id = "20261022_161632"
    policy = read(selected["path"])
    scope_path = Path(policy["reviewed_scope"]["path"])
    scope = read(scope_path)
    scope["review_sources"][source_id] = scope["review_sources"].pop("fresh")
    for value in (scope, policy):
        value["new_source_splits"]["train"] = [source_id]
    registry_path = Path(scope["review_sources"][source_id]["review_source_registry"]["path"])
    old_registry = read(registry_path)
    old_registry["sources"][source_id] = old_registry["sources"].pop("fresh")
    write(registry_path, old_registry)
    write(scope_path, scope)
    selected = write(Path(selected["path"]), policy)
    project = tmp_path / "project"
    monkeypatch.setattr(auth, "PROJ_ROOT", project)
    monkeypatch.setattr(formal_completion, "PROJ_ROOT", project)
    for path in auth.implementation_files().values():
        asset(path, b"# synthetic implementation binding; not executed producer source\n")
    for name in [*auth.ENTRY_NAMES, "unused"]:
        asset(project / "tools" / name, b"# synthetic fixed phase entry\n")
    for name in ("leg_refinement", "leg_selection"):
        asset(project / "hmr4d/lidar_mocap" / (name + ".py"))
    window = "window_1"
    key = source_id + ":" + window
    parent = project / "outputs/review"
    parent_repair = parent / "repair"
    parent_initial = parent_repair / "initial_teacher/sessions" / source_id / "windows" / window
    parent_window = parent_repair / "final_labels/sessions" / source_id / "windows" / window
    calibration = "c" * 64
    initial_json = write(parent_initial / "window.json", dict(source_id=source_id))
    parent_jobs = [
        dict(source_id=source_id, source_window=str(parent_initial), window_json_sha256=initial_json["sha256"])
    ]
    registry_path = parent / "source/source_registry.json"
    registry = read(registry_path)
    registry_binding = write(registry_path, registry)
    parent_source_path = parent_repair / "new_source_input.json"
    parent_source = read(parent_source_path)
    parent_source.update(
        jobs=parent_jobs, calibration_digest=calibration, source_registry_sha256=registry_binding["sha256"]
    )
    parent_source_binding = write(parent_source_path, parent_source)
    parent_plan = write(
        parent_repair / "final_labels/repair_plan.json",
        dict(
            jobs=parent_jobs,
            source_union=str(parent_source_path),
            source_union_sha256=parent_source_binding["sha256"],
            calibration_digest=calibration,
        ),
    )
    write(parent / "source/calibration/calibration_snapshot.json", dict(snapshot_digest=calibration))
    case_plan = write(
        parent_repair / "case_plan.json",
        dict(
            source_union=str(parent_source_path),
            source_union_sha256=parent_source_binding["sha256"],
            selected_scans={source_id: [0]},
        ),
    )
    config = dict(calibration_digest=calibration)
    pose = write(
        parent_repair / "pose_receipt.json",
        dict(
            state="complete",
            source_union_sha256=parent_source_binding["sha256"],
            plan_sha256=case_plan["sha256"],
            config=config,
            config_digest=pose_config_digest(config),
        ),
    )
    identity = write(
        parent_repair / "identity_evidence/identity_decisions.json",
        dict(
            schema="badminton-native-identity-decisions-v1",
            review_status="reviewed",
            windows={key: dict(source_window=str(parent_initial))},
            source_union_sha256=parent_source_binding["sha256"],
            calibration_digest=calibration,
            evidence_files={pose["path"]: pose["sha256"]},
        ),
    )
    source_teacher = write(parent_window / "teacher/manifest.json", dict(tracks=[dict(path="track.pt")]))
    source_track = asset(parent_window / "teacher/track.pt")
    numeric = {
        name: asset(parent_window / name)["sha256"]
        for name in ("ptv3.pth", "fits/person.npz", "quality.jsonl", "ptv3.metadata.jsonl")
    }
    original_report = write(parent_window / "fit_report.json", dict(fitted=1, accepted=1, artifacts_sha256=numeric))
    native = write(
        parent_repair / "final_labels/native_verification.json",
        dict(
            state="complete",
            passed=True,
            windows={key: dict(body=dict(report_sha256=original_report["sha256"], accepted_checked=1))},
            total_windows=1,
            verified_windows=1,
            accepted_checked=1,
            source_union_sha256=parent_source_binding["sha256"],
            calibration_digest=calibration,
        ),
    )
    body = write(
        parent_repair / "final_labels/repair_receipt.json",
        dict(state="body_qa_candidate_complete", windows=1, accepted=1),
    )
    parent_paths = dict(
        native=native,
        identity=identity,
        pose=pose,
        body=body,
        source_input=parent_source_binding,
        body_plan=parent_plan,
        calibration=auth.binding(parent / "source/calibration/calibration_snapshot.json"),
        source_registry=registry_binding,
        case_plan=case_plan,
    )
    # Freeze the original real policy metadata again after populating its review
    # root. The policy still goes through the actual closed ParentRelease loader.
    policy = read(selected["path"])
    scope_path = Path(policy["reviewed_scope"]["path"])
    reviewed = read(scope_path)
    for role in ("review_source_input", "review_source_registry"):
        item = reviewed["review_sources"][source_id][role]
        reviewed["review_sources"][source_id][role] = auth.binding(item["path"])
    policy["reviewed_scope"] = write(scope_path, reviewed)
    selected = write(Path(selected["path"]), policy)
    root = project / "outputs/incremental_formal"
    repair, folder = root / "repair", root / "gpu_guarded_leg"
    target_initial = repair / "initial_teacher/sessions" / source_id / "windows" / window
    target_window = repair / "final_labels/sessions" / source_id / "windows" / window
    target_json = write(target_initial / "window.json", dict(source_id=source_id))
    jobs = [dict(source_id=source_id, source_window=str(target_initial), window_json_sha256=target_json["sha256"])]
    formal_registry = write(
        root / "source/source_registry.json",
        dict(
            release_policy=selected,
            review_only=False,
            sources={source_id: dict(source_index=reviewed["review_sources"][source_id]["admitted_source_index"])},
        ),
    )
    source = write(
        repair / "new_source_input.json",
        dict(
            schema="badminton-new-source-repair-input-v1",
            jobs=jobs,
            calibration_digest=calibration,
            source_root=str(root / "source"),
            source_registry_sha256=formal_registry["sha256"],
            release_policy=selected,
            session_splits=dict(train=[source_id], val=[], test=[]),
            required_nonempty_splits=["train"],
        ),
    )
    plan = write(
        repair / "final_labels/repair_plan.json",
        dict(
            jobs=jobs, source_union=source["path"], source_union_sha256=source["sha256"], calibration_digest=calibration
        ),
    )
    formal_calibration = write(root / "source/calibration/calibration_snapshot.json", dict(snapshot_digest=calibration))
    current_teacher = write(target_window / "teacher/manifest.json", dict(tracks=[dict(path="track.pt")]))
    current_track = asset(target_window / "teacher/track.pt")
    teacher = dict(
        source_manifest=source_teacher,
        current_manifest=current_teacher,
        track_files=[dict(source=source_track, current=current_track)],
        provenance_hash="e" * 64,
        numerical_values_dtype_shape_identical=True,
        identity_decision_sha256="a" * 64,
    )
    for name in numeric:
        asset(target_window / name)
    report = write(
        target_window / "fit_report.json",
        dict(fitted=1, accepted=1, artifacts_sha256=numeric, formal_reuse=dict(teacher_rebinding=teacher)),
    )
    report_snapshot = folder / "baseline/reports" / source_id / window / "fit_report.json"
    report_snapshot.parent.mkdir(parents=True)
    report_snapshot.write_bytes(Path(report["path"]).read_bytes())
    snapshot = auth.binding(report_snapshot)
    current_native = dict(
        state="complete",
        passed=True,
        windows={key: dict(body=dict(report_sha256=report["sha256"], accepted_checked=1))},
        total_windows=1,
        verified_windows=1,
        accepted_checked=1,
        source_union_sha256=source["sha256"],
        calibration_digest=calibration,
    )
    current_pose = dict(
        state="complete", source_union_sha256=source["sha256"], config=dict(calibration_digest=calibration)
    )
    current_identity = dict(
        review_status="reviewed",
        windows={key: {}},
        source_union_sha256=source["sha256"],
        calibration_digest=calibration,
        evidence_files={identity["path"]: identity["sha256"]},
    )
    current_qa = dict(
        schema="badminton-native-absolute-qa-v1",
        source_union_sha256=source["sha256"],
        calibration_digest=calibration,
        windows={key: dict(fit_report_sha256=snapshot["sha256"], accepted=1)},
        frames={
            source_id
            + "/"
            + window
            + "/actor/0": dict(final_p2surface_mean_m=0.01, final_p2surface_p90_m=0.02, reprojection_mean_px=1.0)
        },
    )
    frozen = {
        role: write(folder / "baseline" / (role + ".json"), value)
        for role, value in (
            ("native_verification", current_native),
            ("body_receipt", dict(state="body_qa_candidate_complete", windows=1, accepted=1)),
            ("pose_receipt", current_pose),
            ("reviewed_identity", current_identity),
            ("absolute_qa", current_qa),
        )
    }
    write(repair / "final_labels/native_verification.json", current_native)
    digest = BadmintonPipeline._fit_digest(None)
    equivalence = write(root / "formal_input_equivalence.json", dict(synthetic_frozen_equivalence=True))
    audit = write(
        root / "import_evidence/audit.json",
        dict(
            schema="badminton-formal-numeric-import-audit-v1",
            state="eligible_parent_complete_not_imported",
            formal_root=str(root),
            fitted_tensors_imported=False,
            training_published=False,
            source_input=source,
            body_plan=plan,
            formal_registry=formal_registry,
            formal_calibration=formal_calibration,
            input_equivalence=equivalence,
            parent_evidence={str(parent): parent_paths},
            teacher_files={
                proof["path"]: proof["sha256"]
                for proof in (source_teacher, source_track, current_teacher, current_track)
            },
            algorithms=dict(same=dict(current_digest=digest)),
            windows={
                key: dict(
                    parent_root=str(parent),
                    source_id=source_id,
                    original_window=str(parent_window),
                    original_initial_window=str(parent_initial),
                    target_window=str(target_window),
                    target_initial_window=str(target_initial),
                    original_report=original_report,
                    fitted=1,
                    accepted=1,
                    current_fit_algorithm_digest=digest,
                )
            },
        ),
    )
    imported = write(
        root / "formal_numeric_import.json",
        dict(
            schema="badminton-formal-numeric-import-v1",
            state="complete",
            audit=audit,
            source_union_sha256=source["sha256"],
            calibration_digest=calibration,
            numerical_artifacts_byte_identical=True,
            qa_values_unchanged=True,
            parent_evidence={str(parent): parent_paths},
            windows={
                key: dict(
                    original_report=original_report,
                    current_report=snapshot,
                    current_window_report=report,
                    teacher_rebinding=teacher,
                    fitted=1,
                    accepted=1,
                )
            },
        ),
    )
    baseline = write(
        folder / "baseline/baseline.json",
        dict(
            schema="gvhmr-formal-preleg-baseline-v1",
            state="complete",
            training_published=False,
            gpu_started=False,
            source_input=source,
            source_union_sha256=source["sha256"],
            calibration_digest=calibration,
            plan=plan,
            numeric_import=imported,
            windows={key: dict(report_snapshot=snapshot)},
            accepted=1,
            **frozen,
        ),
    )
    descriptor = dict(
        schema=auth.PROFILE,
        authorization_profile=auth.PROFILE,
        formal_root=str(root),
        release_policy=selected,
        source_input=source,
        import_audit=audit,
        numeric_import=imported,
        baseline=baseline,
        formal_preleg_native=frozen["native_verification"],
        formal_preleg_qa=frozen["absolute_qa"],
        parents={str(parent): parent_paths},
        source_scope=[source_id],
        windows={
            key: dict(
                parent_root=str(parent),
                source_id=source_id,
                original_window=str(parent_window),
                original_report=original_report,
                current_report=snapshot,
                current_window_report=report,
            )
        },
    )
    released = write(
        folder / "previous-lease.json", dict(status="released", actual_cuda_exited=True, fixture_only=True)
    )
    predecessor = write(
        folder / "previous.json",
        dict(
            schema="gvhmr-incremental-guard-predecessor-release-v1",
            state="complete",
            release_receipts=[released],
            process_identities=[dict(pid=os.getpid(), create_time=psutil.Process().create_time() - 1)],
        ),
    )
    now = datetime.now(timezone.utc)
    request = dict(
        schema="gvhmr-continuous-production-gpu-request-v1",
        authorization_profile=auth.PROFILE,
        authorized_stages_requested=["guarded_leg"],
        requested_local_stage_lease_limits_minutes=dict(guarded_leg=30),
        requested_maximum_hours=0.5,
        max_concurrent_cuda_trees=1,
        requested_cuda_visible_devices="0",
        gpu_started=False,
        training_published=False,
        minimum_remaining_launch_seconds=1,
        absolute_hard_deadline_utc=(now + timedelta(minutes=30)).isoformat(),
        issuer_runtime_checks_file="runtime.json",
        predecessor_release=predecessor,
        scope=[dict(stage="guarded_leg", root=str(repair), manifest=source)],
    )
    xml = asset(
        folder / "test-results.xml", b'<testsuite><testcase classname="synthetic" name="metadata"/></testsuite>'
    )
    stdout = asset(folder / "test-results.log", b"synthetic fixture test proof, not this run\n")
    test_execution = dict(
        schema=auth.TEST_SCHEMA,
        state="passed",
        passed=1,
        failed=0,
        errors=0,
        skipped=0,
        exit_code=0,
        fixture_only=True,
        gpu_started=False,
        actual_affinity=[16, 17],
        priority="BelowNormal",
        io_priority=1,
        numerical_threads=1,
        test_files=[
            auth.binding(project / "tests/test_lidar_incremental_guard.py"),
            auth.binding(project / "tests/test_lidar_incremental_guard_execution.py"),
        ],
        junit=xml,
        stdout=stdout,
    )
    runtime = dict(
        authorization_profile=auth.PROFILE,
        grant_schema=write(folder / "grant.schema.json", dict(type="object")),
        readiness_schema=write(folder / "readiness.schema.json", dict(type="object")),
    )
    ready = dict(
        schema="gvhmr-guarded-leg-baseline-readiness-v1",
        state="complete",
        authorization_profile=auth.PROFILE,
        root=str(repair),
        evidence=dict(
            formal_numeric_import=imported,
            formal_complete_baseline=baseline,
            formal_reviewed_identity=frozen["reviewed_identity"],
            formal_preleg_native_verification=frozen["native_verification"],
            formal_preleg_qa=frozen["absolute_qa"],
            predecessor_cuda_exit_bundle=predecessor,
        ),
    )
    grant = dict(
        schema="gvhmr-continuous-production-gpu-grant-v1",
        status="held",
        owner="GVHMR",
        authorization_profile=auth.PROFILE,
        allow_local_stage_leases=True,
        native_gpu_returned=True,
        training_published=False,
        authorized_roots_with_actual_manifest_sha256={
            str(repair): dict(manifest_path=source["path"], manifest_sha256=source["sha256"])
        },
        maximum_stage_minutes=dict(guarded_leg=30),
        predecessor_release=predecessor,
        approved_start_utc=now.isoformat(),
        approved_deadline_utc=(now + timedelta(minutes=30)).isoformat(),
    )
    value = SimpleNamespace(
        project=project,
        root=root,
        repair=repair,
        parent=parent,
        folder=folder,
        key=key,
        source=source,
        parent_paths=parent_paths,
        report=report,
        audit=audit,
        baseline=baseline,
        native=frozen["native_verification"],
        descriptor=descriptor,
        descriptor_path=folder / "descriptor.json",
        request=request,
        request_path=folder / "request.json",
        grant=grant,
        grant_path=folder / "grant.json",
        configuration=dict(target_root=str(repair), algorithm_changed=False, planned_windows=1),
        runtime=runtime,
        implementation=dict(schema=auth.IMPLEMENTATION_SCHEMA),
        test_execution=test_execution,
        ready=ready,
        start=now,
    )
    refresh(value)
    value.checked = lambda **options: auth.load_authorization(
        value.request_path, value.grant_path, **value.kwargs, **options
    )
    return value


def test_single_actual_parent_uses_audit_root_not_date_and_preserves_inputs(stage):
    before = {path: auth.binding(path) for path in (stage.request_path, stage.grant_path, stage.descriptor_path)}
    checked = stage.checked()
    assert checked["root"] == stage.repair and set(checked["parent_map"]) == {stage.key}
    assert checked["parent_map"][stage.key]["parent_root"] == str(stage.parent)
    assert before == {path: auth.binding(path) for path in before}
    assert not (stage.folder / "guarded_leg_stage_lease.json").exists()


@pytest.mark.parametrize(
    "field,value", [("parent_root", "wrong"), ("source_id", "20261020_alias"), ("original_report", {})]
)
def test_rebound_descriptor_cannot_forge_parent_map(stage, field, value):
    stage.descriptor["windows"][stage.key][field] = value
    refresh(stage)
    with pytest.raises(ValueError):
        stage.checked()


def test_missing_explicit_descriptor_pair_stops_before_any_grant(stage):
    with pytest.raises(ValueError, match="explicit descriptor"):
        auth.load_authorization(stage.request_path, stage.folder / "missing-grant.json")


def test_readiness_cannot_add_a_foreign_or_missing_window(stage):
    stage.descriptor["windows"]["20261020_false:other"] = deepcopy(stage.descriptor["windows"][stage.key])
    refresh(stage)
    with pytest.raises(ValueError, match="omits or adds"):
        stage.checked()


@pytest.mark.parametrize(
    "change",
    [
        "failed",
        "skipped",
        "old_schema",
        "code",
        "junit",
        "bool_exit",
        "bool_io",
        "bool_threads",
        "missing_test",
        "duplicate_test",
    ],
)
def test_current_actual_implementation_test_evidence_is_required(stage, change):
    if change == "old_schema":
        stage.test_execution["schema"] = "gvhmr-guarded-leg-cpu-test-v1"
    elif change == "code":
        stage.implementation["schema"] = "gvhmr-guarded-leg-implementation-v1"
    elif change == "junit":
        path = Path(stage.test_execution["junit"]["path"])
        stage.test_execution["junit"] = asset(
            path, b'<testsuite><testcase name="bad"><skipped/></testcase></testsuite>'
        )
    elif change in ("bool_exit", "bool_io", "bool_threads"):
        field = {"bool_exit": "exit_code", "bool_io": "io_priority", "bool_threads": "numerical_threads"}[change]
        stage.test_execution[field] = field != "exit_code"
    elif change == "missing_test":
        stage.test_execution["test_files"].pop()
    elif change == "duplicate_test":
        stage.test_execution["test_files"].append(stage.test_execution["test_files"][0])
    else:
        stage.test_execution[change] = 1
    refresh(stage)
    with pytest.raises(ValueError):
        stage.checked()


@pytest.mark.parametrize("change", ["expired", "future", "too_long", "wrong_root", "wrong_profile"])
def test_wrong_or_inactive_grant_fails_without_lease(stage, change):
    if change == "expired":
        stage.grant["approved_deadline_utc"] = (stage.start - timedelta(seconds=1)).isoformat()
    elif change == "future":
        stage.grant["approved_start_utc"] = (stage.start + timedelta(minutes=1)).isoformat()
    elif change == "too_long":
        stage.grant["approved_deadline_utc"] = (stage.start + timedelta(minutes=31)).isoformat()
    elif change == "wrong_root":
        stage.grant["authorized_roots_with_actual_manifest_sha256"] = {}
    else:
        stage.grant["authorization_profile"] = "legacy"
    refresh(stage)
    with pytest.raises(ValueError):
        stage.checked()
    assert not (stage.folder / "guarded_leg_stage_lease.json").exists()


def test_immutable_watchdog_anchors_are_stable_and_exclude_apply_targets(stage):
    deep = stage.checked()
    shallow = stage.checked(deep=False)
    anchors = auth.immutable_anchors(deep)
    assert anchors == auth.immutable_anchors(shallow)
    paths = {item["path"] for item in anchors}
    assert stage.report["path"] not in paths
    assert str(stage.repair / "final_labels/native_verification.json") not in paths
    assert stage.descriptor["windows"][stage.key]["current_report"]["path"] in paths
    assert stage.descriptor["windows"][stage.key]["original_report"]["path"] in paths
    with Path(stage.report["path"]).open("ab") as stream:
        stream.write(b"\n")
    assert auth.immutable_anchors(shallow) == anchors
    with pytest.raises(ValueError):
        stage.checked()


def test_immutable_parent_change_is_caught_before_relaunch(stage):
    checked = stage.checked()
    path = Path(stage.parent_paths["native"]["path"])
    with path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="changed"):
        auth.immutable_anchors(checked)


def test_false_parent_identity_or_native_completion_is_not_a_date_alias(stage):
    path = Path(stage.parent_paths["native"]["path"])
    value = read(path)
    value["passed"] = False
    write(path, value)
    with pytest.raises(ValueError):
        stage.checked()


@pytest.mark.parametrize("change", ["request_concurrency", "planned_windows", "native_count", "actor_pid"])
def test_boolean_is_not_an_integer_authorization_or_completion_count(stage, change):
    if change == "request_concurrency":
        stage.request["max_concurrent_cuda_trees"] = True
    elif change == "planned_windows":
        stage.configuration["planned_windows"] = True
    elif change == "native_count":
        value = read(stage.native["path"])
        value["verified_windows"] = True
        stage.native = write(Path(stage.native["path"]), value)
        stage.descriptor["formal_preleg_native"] = stage.native
        stage.ready["evidence"]["formal_preleg_native_verification"] = stage.native
        baseline = read(stage.baseline["path"])
        baseline["native_verification"] = stage.native
        stage.baseline = write(Path(stage.baseline["path"]), baseline)
        stage.descriptor["baseline"] = stage.baseline
        stage.ready["evidence"]["formal_complete_baseline"] = stage.baseline
    else:
        proof = stage.request["predecessor_release"]
        value = read(proof["path"])
        value["process_identities"][0]["pid"] = True
        proof = write(Path(proof["path"]), value)
        stage.request["predecessor_release"] = proof
        stage.grant["predecessor_release"] = proof
        stage.ready["evidence"]["predecessor_cuda_exit_bundle"] = proof
    refresh(stage)
    with pytest.raises(ValueError):
        stage.checked()


def test_actual_parent_release_consumer_must_have_completed(stage):
    selected = stage.descriptor["release_policy"]
    policy = read(selected["path"])
    scope = read(policy["reviewed_scope"]["path"])
    parent = policy["parent_release"]
    audit = read(parent["final_consumer_audit"]["path"])
    audit["status"] = "pending"
    parent["final_consumer_audit"] = write(Path(parent["final_consumer_audit"]["path"]), audit)
    scope["parent_release"] = deepcopy(parent)
    policy["reviewed_scope"] = write(Path(policy["reviewed_scope"]["path"]), scope)
    stage.descriptor["release_policy"] = write(Path(selected["path"]), policy)
    refresh(stage)
    with pytest.raises(ValueError):
        stage.checked()
