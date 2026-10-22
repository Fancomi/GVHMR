"""Synthetic formal completion/exit contracts; no production data or model load."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import psutil
import pytest

from hmr4d.lidar_mocap import formal_completion as completion
from hmr4d.lidar_mocap import formal_baseline as freezer
from test_lidar_formal_baseline import stage, put, read, bytes_file


@pytest.fixture
def pending(stage, monkeypatch):
    monkeypatch.setattr(completion, "PROJ_ROOT", stage.root.parent.parent)
    receipt = stage.freeze()
    stage.baseline_binding = receipt["baseline"]
    return stage


@pytest.fixture
def finished(pending, monkeypatch):
    stage = pending
    completion.prepare_formal_leg_completion(stage.root, stage.baseline_binding)
    root, repair = stage.root, stage.repair
    folder = root / "gpu_guarded_leg"
    frozen = read(stage.baseline_binding["path"])
    source_sha, calibration = frozen["source_union_sha256"], frozen["calibration_digest"]
    policy_sha = "b" * 64
    proposal_windows = {}
    selection_windows = {}
    for i, key in enumerate(stage.keys):
        sid, name = key.split(":")
        sha = frozen["windows"][key]["report_snapshot"]["sha256"]
        item = put(
            repair / "leg_refinement_proposals" / sid / (name + ".json"),
            dict(source_fit_report_sha256=sha, proposals=[]),
        )
        proposal_windows[key] = item
        selection_windows[key] = dict(
            source_fit_report_sha256=sha,
            proposal_path=item["path"],
            proposal_sha256=item["sha256"],
            selected=[dict(motion_id="person", lidar_index=0)] if i == 0 else [],
        )
    proposal = put(
        repair / "leg_refinement_proposals/receipt.json",
        dict(state="complete", windows=proposal_windows, source_union_sha256=source_sha, policy_sha256=policy_sha),
    )
    selection = put(
        repair / "leg_refinement_selection.json",
        dict(
            state="complete",
            windows=selection_windows,
            selected_frames=1,
            source_union_sha256=source_sha,
            source_proposals_sha256=proposal["sha256"],
            refinement_policy_sha256=policy_sha,
        ),
    )
    changed = stage.current_dirs[stage.keys[0]]
    rows = [json.loads(line) for line in (changed / "quality.jsonl").read_text().splitlines()]
    rows[0]["qa"]["final_p2surface_mean_m"] = 0.024
    rows[0]["qa"]["reprojection_mean_px"] = 3.0
    report = read(changed / "fit_report.json")
    report["artifacts_sha256"]["quality.jsonl"] = bytes_file(
        changed / "quality.jsonl", b"".join(json.dumps(r).encode() + b"\n" for r in rows)
    )
    report["artifacts_sha256"]["fits/person.npz"] = bytes_file(
        changed / "fits/person.npz", b"synthetic corrected leg bytes"
    )
    report["artifacts_sha256"]["ptv3.pth"] = bytes_file(changed / "ptv3.pth", b"synthetic updated body tensor bytes")
    report["post_leg_refinement"] = dict(selection_sha256=selection["sha256"], policy_sha256=policy_sha, frames=1)
    report_binding = put(changed / "fit_report.json", report)
    applied = dict(
        state="complete",
        selected_frames=1,
        updated_frames=1,
        selection_sha256=selection["sha256"],
        policy_sha256=policy_sha,
        body_qa_accepted=2,
        windows={stage.keys[0]: dict(frames=1, fit_report_sha256=report_binding["sha256"])},
        training_published=False,
    )
    apply = put(repair / "leg_refinement_receipt.json", applied)
    post = put(repair / "final_labels/post_refinement.json", dict(applied, source="bounded_native_leg_correspondence"))
    exports = {key.split(":")[0]: dict(accepted=1) for key in stage.keys}
    put(repair / "final_labels/datasets/dataset_manifest.json", dict(accepted=2, exports=exports))
    config = put(
        folder / "configuration.json", dict(target_root=str(repair), algorithm_changed=False, planned_windows=2)
    )
    commands = []
    for phase, script in (
        ("propose", "propose_badminton_leg_refinement.py"),
        ("select", "select_badminton_leg_refinement.py"),
        ("apply", "apply_badminton_leg_refinement.py"),
    ):
        entry = completion.PROJ_ROOT / "tools" / script
        bytes_file(entry, b"# synthetic CPU contract fixture; never executed\n")
        argv = [str(completion.PROJ_ROOT / ".venv/Scripts/python.exe"), "-u", str(entry), "--root", str(repair)]
        if phase != "select":
            argv += ["--threads", "2"]
        commands.append(dict(phase=phase, cuda=phase != "select", argv=argv, entry_binding=freezer._binding(entry)))
    request = put(
        folder / "request.json",
        dict(
            schema="gvhmr-continuous-production-gpu-request-v1",
            authorized_stages_requested=["guarded_leg"],
            scope=[
                dict(
                    stage="guarded_leg",
                    root=str(repair),
                    manifest=frozen["source_input"],
                    configuration=config,
                    commands=commands,
                )
            ],
        ),
    )
    ready = put(
        stage.folder / "readiness_fixture.json",
        dict(
            schema="gvhmr-guarded-leg-baseline-readiness-v1",
            state="complete",
            request=request,
            root=str(repair),
            configuration=config,
            evidence=dict(formal_complete_baseline=stage.baseline_binding, formal_preleg_qa=frozen["absolute_qa"]),
        ),
    )
    grant = put(
        folder / "grant_fixture.json",
        dict(
            schema="gvhmr-continuous-production-gpu-grant-v1",
            owner="GVHMR",
            status="held",
            request=request,
            request_sha256=request["sha256"],
            readiness=ready,
            approved_start_utc="2020-01-01T00:00:00+00:00",
            approved_deadline_utc="2020-01-01T00:30:00+00:00",
            authorized_roots_with_actual_manifest_sha256={
                str(repair): dict(manifest_path=frozen["source_input"]["path"], manifest_sha256=source_sha)
            },
        ),
    )
    nonce = "a" * 32
    attempt = folder / "attempts" / nonce
    preflight = put(
        attempt / "preflight.json",
        dict(
            state="complete", request=request, grant=grant, readiness=ready, native_and_baseline_evidence_checked=True
        ),
    )
    context = put(
        attempt / "context.json",
        dict(
            nonce=nonce,
            root=str(repair),
            job_name="Local\\GVHMRGuardedLeg_" + nonce,
            request=request,
            grant=grant,
            preflight=preflight,
            immutable_bindings=[stage.baseline_binding, frozen["absolute_qa"]],
        ),
    )
    own = psutil.Process()
    identities = [
        dict(
            pid=os.getpid(),
            create_time=own.create_time() - (i + 1) * 100,
            command=command["argv"],
            affinity=[26, 27] if command["phase"] == "select" else [0, 1],
            assigned_before_resume=True,
        )
        for i, command in enumerate(commands)
    ]
    outputs = dict(propose=proposal, select=selection, apply=apply)
    phases = {}
    for command, identity in zip(commands, identities):
        phase = command["phase"]
        phases[phase] = dict(output=outputs[phase], actual_job_empty=True, launcher=identity)
        put(attempt / (phase + ".launch.json"), identity)
        put(attempt / (phase + ".exit.json"), phases[phase])
    put(attempt / "all_phases_complete.json", dict(phases=phases, context=context))
    actual_exit = put(
        attempt / "actual_cuda_exit.json",
        dict(
            state="all_owned_cuda_exited",
            context_sha256=context["sha256"],
            observed_process_identities=identities,
            job_empty=True,
            gpu0_compute_pids_empty=True,
        ),
    )
    guard = put(
        attempt / "guard_ready.json",
        dict(
            pid=os.getpid(),
            create_time=own.create_time() - 1000,
            ready=True,
            context_sha256=context["sha256"],
            actual_affinity=[26, 27],
        ),
    )
    execution = dict(
        schema="gvhmr-guarded-leg-execution-v1",
        state="complete_actual_cuda_exited",
        request=request,
        grant=grant,
        phases=phases,
        actual_exit=actual_exit,
        independent_guard=guard,
        observed_process_identities=identities,
        training_published=False,
    )
    execution_path = attempt / "completion.json"
    execution_binding = put(execution_path, execution)
    stage.execution_path, stage.execution, stage.execution_binding = execution_path, execution, execution_binding
    stage.context, stage.ready, stage.grant, stage.proposal, stage.selection, stage.apply, stage.post = (
        context,
        ready,
        grant,
        proposal,
        selection,
        apply,
        post,
    )
    stage.native_calls = []

    def worker(checked, role, directory, threads):
        stage.native_calls.append(role)
        if role == "native":
            value = read(stage.native["path"])
            for key in stage.keys:
                value["windows"][key]["body"]["report_sha256"] = freezer._sha(
                    stage.current_dirs[key] / "fit_report.json"
                )
            put(Path(stage.native["path"]), value)
        else:
            value = dict(
                calibration_digest=calibration,
                sessions={
                    sid: dict(
                        exported=1,
                        checked=1,
                        max_source_cloud_error_m=0.0,
                        max_rebuilt_mesh_error_m=0.0,
                        max_pelvis_translation_error_m=0.0,
                    )
                    for sid in exports
                },
            )
            put(repair / "final_labels/datasets/verification.json", value)
        log = directory / (role + ".log")
        bytes_file(log, b"synthetic CPU verifier mock; no model executed\n")
        return dict(exit_code=0, entry=commands[0]["entry_binding"], log=freezer._binding(log), hidden_window=True)

    stage.worker = worker
    monkeypatch.setattr(completion, "_run_cpu_check", worker)
    stage.verify = lambda **kwargs: completion.verify_formal_postleg_cpu(
        root, stage.baseline_binding, freezer._binding(execution_path), **kwargs
    )
    return stage


def assert_no_workers(stage):
    assert stage.native_calls == []
    assert not (stage.repair / "evaluation/formal_postleg_cpu_receipt.json").exists()
    assert not (stage.repair / "evaluation/formal_postleg_cpu").exists()


def test_prepare_copies_genuine_bytes_and_reuses_without_rewriting(pending):
    result = completion.prepare_formal_leg_completion(pending.root, pending.baseline_binding)
    target = Path(result["before_qa"]["path"])
    qa = read(pending.baseline_binding["path"])["absolute_qa"]
    assert target.read_bytes() == Path(qa["path"]).read_bytes()
    stamp = target.stat().st_mtime_ns
    assert completion.prepare_formal_leg_completion(pending.root, pending.baseline_binding) == result
    assert target.stat().st_mtime_ns == stamp
    assert result["gpu_started"] is False


@pytest.mark.parametrize("change", ["post", "leg_receipt", "old_qa", "baseline_sha", "current_report", "artifact"])
def test_prepare_refuses_after_leg_old_qa_or_changed_baseline(pending, change):
    stage = pending
    if change == "post":
        put(stage.repair / "final_labels/post_refinement.json", dict(state="partial"))
    if change == "leg_receipt":
        put(stage.repair / "leg_refinement_receipt.json", dict(state="partial"))
    if change == "old_qa":
        bytes_file(stage.repair / "evaluation/v2_before_leg.json", b"different older QA")
    if change == "baseline_sha":
        stage.baseline_binding["sha256"] = "f" * 64
    if change == "current_report":
        (stage.current_dirs[stage.keys[0]] / "fit_report.json").write_bytes(b"{}")
    if change == "artifact":
        (stage.current_dirs[stage.keys[0]] / "fits/person.npz").write_bytes(b"changed")
    with pytest.raises(ValueError):
        completion.prepare_formal_leg_completion(stage.root, stage.baseline_binding)
    assert not (stage.repair / "evaluation/formal_postleg_cpu_receipt.json").exists()


def test_cpu_postleg_accepts_real_verifier_shapes_after_expired_grant_and_is_idempotent(finished):
    stage = finished
    result = stage.verify()
    assert result["state"] == completion.STATE and result["windows"] == 2 and result["accepted"] == 2
    assert stage.native_calls == ["native", "dataset"]
    assert result["gpu_started"] is False and result["training_published"] is False
    after = read(result["evidence"]["after"]["path"])
    comparison = read(result["evidence"]["comparison"]["path"])
    assert comparison["person_scans"] == 2
    assert comparison["metrics"]["reprojection_mean_px"]["after_mean"] == 3.5
    assert set(after["frames"]) == set(read(stage.repair / "evaluation/v2_before_leg.json")["frames"])
    assert stage.verify() == result
    assert stage.native_calls == ["native", "dataset"]


@pytest.mark.parametrize(
    "change",
    [
        "missing_before",
        "old_before",
        "wrong_execution_sha",
        "partial",
        "missing_phase",
        "wrong_nonce",
        "wrong_context_root",
        "wrong_request",
        "wrong_scope",
        "wrong_qa_binding",
        "wrong_baseline_binding",
        "live_pid",
        "not_exited",
        "job_busy",
        "gpu_busy",
        "changed_phase_output",
        "partial_phase",
        "changed_launch",
        "changed_phase_exit",
        "post_partial",
        "wrong_updated_count",
        "changed_unselected",
    ],
)
def test_invalid_exit_scope_or_baseline_evidence_cannot_start_cpu_workers(finished, change):
    stage = finished
    execution = deepcopy(stage.execution)
    before = stage.repair / "evaluation/v2_before_leg.json"
    if change == "missing_before":
        before.unlink()
    if change == "old_before":
        before.write_bytes(b"old QA from another run")
    if change == "partial":
        execution["state"] = "running"
    if change == "missing_phase":
        execution["phases"].pop("apply")
    if change == "wrong_nonce":
        context = read(stage.context["path"])
        context["nonce"] = "b" * 32
        put(Path(stage.context["path"]), context)
    if change == "wrong_context_root":
        context = read(stage.context["path"])
        context["root"] = str(stage.root)
        put(Path(stage.context["path"]), context)
    if change == "wrong_request":
        execution["request"]["sha256"] = "f" * 64
    if change == "wrong_scope":
        request = read(execution["request"]["path"])
        request["scope"][0]["root"] = str(stage.root)
        execution["request"] = put(Path(execution["request"]["path"]), request)
    if change in ("wrong_qa_binding", "wrong_baseline_binding"):
        ready = read(stage.ready["path"])
        role = "formal_preleg_qa" if change == "wrong_qa_binding" else "formal_complete_baseline"
        ready["evidence"][role]["sha256"] = "f" * 64
        grant = read(stage.grant["path"])
        grant["readiness"] = put(Path(stage.ready["path"]), ready)
        execution["grant"] = put(Path(stage.grant["path"]), grant)
    if change == "live_pid":
        execution["observed_process_identities"].append(
            dict(pid=os.getpid(), create_time=psutil.Process().create_time())
        )
    if change in ("not_exited", "job_busy", "gpu_busy"):
        exited = read(execution["actual_exit"]["path"])
        if change == "not_exited":
            exited["state"] = "guarding"
        if change == "job_busy":
            exited["job_empty"] = False
        if change == "gpu_busy":
            exited["gpu0_compute_pids_empty"] = False
        execution["actual_exit"] = put(Path(execution["actual_exit"]["path"]), exited)
    if change == "changed_phase_output":
        Path(stage.apply["path"]).write_bytes(b"{}")
    if change == "partial_phase":
        value = read(stage.apply["path"])
        value["state"] = "applying"
        put(Path(stage.apply["path"]), value)
    if change == "changed_launch":
        put(stage.execution_path.parent / "apply.launch.json", dict(pid=1))
    if change == "changed_phase_exit":
        put(stage.execution_path.parent / "apply.exit.json", dict(actual_job_empty=False))
    if change == "post_partial":
        post = read(stage.post["path"])
        post["state"] = "applying"
        put(Path(stage.post["path"]), post)
    if change == "wrong_updated_count":
        post = read(stage.post["path"])
        post["updated_frames"] = 0
        put(Path(stage.post["path"]), post)
    if change == "changed_unselected":
        (stage.current_dirs[stage.keys[1]] / "fit_report.json").write_bytes(b"{}")
    put(stage.execution_path, execution)
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        if change == "wrong_execution_sha":
            completion.verify_formal_postleg_cpu(
                stage.root, stage.baseline_binding, dict(path=str(stage.execution_path), sha256="f" * 64)
            )
        else:
            stage.verify()
    assert_no_workers(stage)


def test_existing_live_pid_with_a_different_create_time_is_not_terminated_or_rejected(finished):
    # All fixture identities use the real PID with older synthetic creation times.
    # The original identities have exited; the reused current PID belongs to us.
    assert finished.verify()["state"] == completion.STATE
    assert psutil.Process(os.getpid()).is_running()


def test_worker_failure_does_not_publish_or_start_second_verifier(finished, monkeypatch):
    calls = []

    def failing(checked, role, directory, threads):
        calls.append(role)
        raise RuntimeError("synthetic native verifier failure")

    monkeypatch.setattr(completion, "_run_cpu_check", failing)
    with pytest.raises(RuntimeError):
        finished.verify()
    assert calls == ["native"]
    assert not (finished.repair / "evaluation/formal_postleg_cpu_receipt.json").exists()


@pytest.mark.parametrize(
    "change", ["missing_session", "camera", "checked", "exported", "point_error", "mesh_error", "pelvis_error"]
)
def test_dataset_success_requires_actual_native_report_fields_not_a_passed_flag(finished, monkeypatch, change):
    worker = finished.worker

    def invalid_report(checked, role, directory, threads):
        result = worker(checked, role, directory, threads)
        if role == "dataset":
            path = finished.repair / "final_labels/datasets/verification.json"
            report = read(path)
            entry = next(iter(report["sessions"].values()))
            if change == "missing_session":
                report["sessions"].pop(next(iter(report["sessions"])))
            if change == "camera":
                report["calibration_digest"] = "d" * 64
            if change == "checked":
                entry["checked"] = 0
            if change == "exported":
                entry["exported"] = 2
            if change == "point_error":
                entry["max_source_cloud_error_m"] = 0.1
            if change == "mesh_error":
                entry["max_rebuilt_mesh_error_m"] = 0.1
            if change == "pelvis_error":
                entry["max_pelvis_translation_error_m"] = 0.1
            report["passed"] = True
            put(path, report)
        return result

    monkeypatch.setattr(completion, "_run_cpu_check", invalid_report)
    with pytest.raises(ValueError):
        finished.verify()
    assert not (finished.repair / "evaluation/formal_postleg_cpu_receipt.json").exists()


def test_real_cpu_worker_launcher_is_hidden_belownormal_and_disables_cuda(finished, monkeypatch, tmp_path):
    log_dir = tmp_path / "worker_probe"
    log_dir.mkdir()
    bytes_file(completion.PROJ_ROOT / "tools/verify_badminton_native_bound.py", b"# synthetic verifier entry\n")
    allocation = dict(
        schema="gvhmr-formal-cpu-worker-v1", affinity=[26, 27], priority="BelowNormal", cuda_visible_devices="-1"
    )

    def fake_run(command, **kwargs):
        assert command[2] == "-c" and "cpu_affinity([26,27])" in command[3]
        assert command[-2:] == ["--threads", "2"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "2"
        assert kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW
        assert kwargs["creationflags"] & subprocess.BELOW_NORMAL_PRIORITY_CLASS
        kwargs["stdout"].write(json.dumps(allocation).encode() + b"\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(completion.subprocess, "run", fake_run)
    # The fixture replaces the helper; call the original implementation explicitly.
    from importlib.util import spec_from_file_location, module_from_spec

    spec = spec_from_file_location("hmr4d.lidar_mocap.completion_worker_probe", Path(completion.__file__))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PROJ_ROOT = completion.PROJ_ROOT
    checked = dict(repair=finished.repair)
    result = module._run_cpu_check(checked, "native", log_dir, 2)
    assert result["allocation"] == allocation and result["hidden_window"] is True
    assert result["command"][-5:] == ["--root", str(finished.repair), "--verified-formal-cache", "--threads", "2"]


def test_changed_person_frame_set_is_rejected_even_with_same_totals(finished, monkeypatch):
    from hmr4d.lidar_mocap import new_source_completion

    actual = new_source_completion.collect_absolute_qa

    def altered(root):
        result = actual(root)
        result["frames"]["20261019_152542/window_1/different_person/0"] = result["frames"].pop(
            next(iter(result["frames"]))
        )
        return result

    monkeypatch.setattr(new_source_completion, "collect_absolute_qa", altered)
    with pytest.raises(ValueError, match="membership"):
        finished.verify()
    assert not (finished.repair / "evaluation/formal_postleg_cpu_receipt.json").exists()
