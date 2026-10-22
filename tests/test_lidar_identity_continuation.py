"""CPU contracts for canonical identity handoff; no model or CUDA execution."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hmr4d.lidar_mocap import identity_continuation as continuation
from hmr4d.lidar_mocap import gpu_handoff
from hmr4d.lidar_mocap.pilot import binding
from hmr4d.lidar_mocap.repair_execution import native_gpu_command_matches
from test_lidar_gpu_handoff import window


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return binding(path)


@pytest.fixture
def complete(window, monkeypatch):
    request, grant, roots, authority, start = window
    root = roots[1]
    project = root.parent.parent
    monkeypatch.setattr(continuation, "PROJ_ROOT", project)
    source_window = root / "repair/initial_teacher/sessions/20261020_182854/windows/window_1"
    put(
        root / "pilot_input.json",
        dict(
            manifest=binding(
                Path(authority["authorized_roots_with_actual_manifest_sha256"][str(root)]["manifest_path"])
            )
        ),
    )
    manifest = put(source_window / "teacher/manifest.json", dict(tracks=[]))
    jobs = [dict(source_id="20261020_182854", source_window=str(source_window))]
    source = put(root / "repair/new_source_input.json", dict(jobs=jobs, calibration_digest="c" * 64))
    put(root / "teacher_sampling_input.json", dict(sampling="synthetic"))
    evidence = root / "repair/identity_evidence"
    inventory = put(evidence / "boundary_inventory.reviewed.json", dict(review_status="reviewed"))
    put(evidence / "case_plan.json", dict(boundary_inventory=inventory))
    pose = dict(state="complete", source_union_sha256=source["sha256"], config=dict(calibration_digest="c" * 64))
    put(root / "repair/pose_receipt.json", pose)
    put(evidence / "pose_receipt.json", pose)
    cases = []
    statuses = ["same_native_person", "different_native_people", "insufficient_or_ambiguous_native_evidence"]
    for index, status in enumerate(statuses):
        before, after = index + 1, index + 2
        cases.append(
            dict(
                index=index,
                source_window=str(source_window),
                positions=[index, index + 1],
                context_positions=[0, 1, 2, 3],
                event=dict(
                    source_id="20261020_182854",
                    before_motion_id=f"20261020_182854:window_1:{before}:0",
                    after_motion_id=f"20261020_182854:window_1:{after}:0",
                ),
            )
        )
    boundary = put(evidence / "boundaries.json", cases)
    review = dict(
        partial=False,
        cases=[
            dict(
                case,
                status=status,
                missing_native_context=0,
                votes=[dict(original_track_id=index + 1), dict(original_track_id=index + 2)],
            )
            for index, (case, status) in enumerate(zip(cases, statuses))
        ],
        counts={status: 1 for status in statuses},
        inputs={boundary["path"]: boundary["sha256"]},
    )
    put(evidence / "native_identity_review.json", review)
    put(
        evidence / "boundary_inventory.pending.json",
        dict(
            tracklets=[
                dict(source_id="20261020_182854", window="window_1", original_track_id=index) for index in range(1, 5)
            ]
        ),
    )
    policy = dict(
        schema="badminton-new-source-identity-abstention-v1",
        review_status="reviewed",
        action="keep_unproven_original_tracklets_separate",
        source_union_sha256=source["sha256"],
        boundary_inventory_sha256=inventory["sha256"],
        evidence_files={manifest["path"]: manifest["sha256"]},
    )
    policy_binding = put(evidence / "abstention_policy.json", policy)
    decision = dict(
        schema="badminton-native-identity-decisions-v1",
        review_status="reviewed",
        windows={
            "20261020_182854:window_1": dict(
                source_window=str(source_window),
                source_teacher_manifest_sha256=manifest["sha256"],
                links=[dict(tracks=[1, 2], case=0, method="native_opposite_camera_continuity_and_epipolar_pose")],
                cannot_link=[[2, 3]],
                drop_spans=[],
                review_cases=[0, 1, 2],
                reset_tracklets=[1, 2, 3, 4],
            )
        },
        source_union_sha256=source["sha256"],
        calibration_digest="c" * 64,
        unresolved_cases=[],
        abstention_policy=policy_binding["path"],
        abstained_relationship_cases=[2],
        evidence_files={manifest["path"]: manifest["sha256"], policy_binding["path"]: policy_binding["sha256"]},
    )
    put(evidence / "identity_decisions.json", decision)
    for label in ("poses_01", "identity_poses_01"):
        lease = put(
            root / "gpu_leases" / (label + ".json"),
            dict(
                root=str(root),
                owner="GVHMR",
                status="released",
                all_observed_owned_processes_exited=True,
                manifest_sha256=binding(
                    Path(authority["authorized_roots_with_actual_manifest_sha256"][str(root)]["manifest_path"])
                )["sha256"],
            ),
        )
        progress = put(root / (label + "_release_receipt.progress.json"), dict(stage="poses_complete"))
        put(
            root / (label + "_release_receipt.json"),
            dict(
                status="released",
                kind="poses",
                lease_after=lease,
                full_pose_progress=progress,
                observed_process_identities=[dict(pid=999999, create_time=42.0, exited=True)],
            ),
        )
    for name in ("hmr4d/lidar_mocap/identity_continuation.py", continuation.ENTRY):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# synthetic implementation binding only\n")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return start + timedelta(hours=1)

    monkeypatch.setattr(continuation, "datetime", Clock)
    return SimpleNamespace(**locals())


def test_real_completion_reader_and_immutable_freeze_do_not_replay_inference(complete, monkeypatch):
    monkeypatch.setattr(
        continuation.subprocess, "Popen", lambda *a, **k: pytest.fail("Read-only verification launched a worker")
    )
    value = continuation.verify_completed_identity(complete.root)
    assert value["windows"] == 1 and value["geometric_cases"] == 3 and value["abstained"] == 1
    assert value["numerical_body_native_verification_claimed"] is False
    frozen = continuation.freeze_completion(complete.root)
    assert continuation.freeze_completion(complete.root) == frozen
    assert not (complete.root / "gpu_leases/identity_poses_02.json").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "pose_partial",
        "review_partial",
        "review_scope",
        "missing_context",
        "counts",
        "decision_scope",
        "extra_link",
        "lost_cannot_link",
        "lost_reset",
        "lost_abstention",
        "teacher_changed",
        "policy_changed",
        "release_held",
        "release_alive",
        "release_failed",
    ],
)
def test_completed_flags_cannot_bypass_actual_geometry_operations_or_exit_chain(complete, damage):
    evidence = complete.evidence
    if damage == "pose_partial":
        path = evidence / "pose_receipt.json"
        value = json.loads(path.read_bytes())
        value["state"] = "partial"
    elif damage in ("review_partial", "review_scope", "missing_context", "counts"):
        path = evidence / "native_identity_review.json"
        value = deepcopy(complete.review)
        if damage == "review_partial":
            value["partial"] = True
        if damage == "review_scope":
            value["cases"][0]["positions"] = [99, 100]
        if damage == "missing_context":
            value["cases"][0]["missing_native_context"] = 1
        if damage == "counts":
            value["counts"] = {}
    elif damage in ("decision_scope", "extra_link", "lost_cannot_link", "lost_reset", "lost_abstention"):
        path = evidence / "identity_decisions.json"
        value = deepcopy(complete.decision)
        entry = next(iter(value["windows"].values()))
        if damage == "decision_scope":
            value["windows"] = {}
        if damage == "extra_link":
            entry["links"].append(dict(tracks=[3, 4], case=2, method="guessed"))
        if damage == "lost_cannot_link":
            entry["cannot_link"] = []
        if damage == "lost_reset":
            entry["reset_tracklets"].pop()
        if damage == "lost_abstention":
            value["abstained_relationship_cases"] = []
    elif damage == "teacher_changed":
        path = Path(complete.manifest["path"])
        value = dict(tracks=["changed"])
    elif damage == "policy_changed":
        path = evidence / "abstention_policy.json"
        value = dict(complete.policy, action="guess_all_links")
    elif damage == "release_held":
        path = complete.root / "gpu_leases/identity_poses_01.json"
        value = json.loads(path.read_bytes())
        value["status"] = "held"
    else:
        path = complete.root / "identity_poses_01_release_receipt.json"
        value = json.loads(path.read_bytes())
        if damage == "release_alive":
            value["observed_process_identities"][0]["exited"] = False
        else:
            snapshot = Path(value["full_pose_progress"]["path"])
            value["full_pose_progress"] = put(snapshot, dict(stage="failed"))
    put(path, value)
    with pytest.raises(ValueError):
        continuation.verify_completed_identity(complete.root)


def test_canonical_native_worker_remains_visible_to_authorized_native_cleanup(complete):
    lease = dict(path=str(complete.root / "gpu_leases/future.json"), sha256="a" * 64)
    command = continuation.worker_command(complete.root, "observe", 2, lease)
    assert native_gpu_command_matches(command, complete.root)
    assert not native_gpu_command_matches(command, complete.roots[0])
    for role in ("review", "seal"):
        assert not native_gpu_command_matches(continuation.worker_command(complete.root, role, 1), complete.root)
    with pytest.raises(ValueError):
        continuation.worker_command(complete.root, "observe", 2)


def test_existing_controller_state_is_never_replaced(complete):
    path = complete.evidence / "identity_work_progress.json"
    put(path, dict(stage="running", pid=123, create_time=42))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="preserved"):
        continuation.IdentityContinuation(complete.root, complete.request, complete.grant)
    assert path.read_bytes() == before


@pytest.mark.parametrize("damage", [None, "missing", "wrong_root", "released", "expired", "extended", "hash"])
def test_internal_observation_worker_requires_real_bound_local_authority(complete, damage):
    authorization = gpu_handoff.read_continuous_grant(complete.request, complete.grant)
    now = complete.start + timedelta(hours=1)
    lease = gpu_handoff.make_local_stage_lease(authorization, complete.root, "poses", now=now)
    lease["approved_deadline_utc"] = (now + timedelta(minutes=20)).isoformat()
    if damage == "wrong_root":
        lease["root"] = str(complete.roots[0])
    if damage == "released":
        lease["status"] = "released"
    if damage == "expired":
        lease["approved_deadline_utc"] = now.isoformat()
    if damage == "extended":
        lease["approved_deadline_utc"] = (now + timedelta(minutes=21)).isoformat()
    item = put(complete.root / "gpu_leases/future_identity.json", lease)
    if damage == "missing":
        item = None
    if damage == "hash":
        item["sha256"] = "f" * 64
    if damage:
        with pytest.raises(ValueError):
            continuation._worker_authority(complete.root, item)
    else:
        assert continuation._worker_authority(complete.root, item)["binding"] == binding(complete.grant)


def test_worker_rejects_arbitrary_script_before_any_execution(complete, monkeypatch):
    monkeypatch.setattr(continuation.runpy, "run_path", lambda *a, **k: pytest.fail("Arbitrary worker ran"))
    with pytest.raises(ValueError):
        continuation.run_worker(complete.root, "observe", complete.request, 2)


@pytest.mark.parametrize("role", ["review", "seal"])
def test_direct_CPU_worker_also_requires_actual_identity_CUDA_release(complete, monkeypatch, role):
    (complete.root / "identity_poses_01_release_receipt.json").unlink()
    monkeypatch.setattr(
        continuation.runpy, "run_path", lambda *a, **k: pytest.fail("CPU worker started before release")
    )
    with pytest.raises(FileNotFoundError):
        continuation.run_worker(complete.root, role, complete.project / "tools" / continuation.WORKERS[role], 1)


@pytest.mark.parametrize(
    "failure,stage_cap",
    [(None, 20), (None, 10), (None, 15), ("GPU_exit", 20), ("local_deadline", 20), ("observer_exit", 20)],
)
def test_full_canonical_handoff_waits_for_release_before_CPU_seal_and_records_exits(
    complete, monkeypatch, failure, stage_cap
):
    # Only child execution is synthetic. Source/authority/policy/native scope,
    # completion/decision/release checks and real controller flow remain active.
    identity_lease = complete.root / "gpu_leases/identity_poses_01.json"
    identity_release = complete.root / "identity_poses_01_release_receipt.json"
    identity_lease.unlink()
    identity_release.unlink()
    grant = json.loads(complete.grant.read_bytes())
    grant["maximum_stage_minutes"]["poses"] = stage_cap
    put(complete.grant, grant)
    put(complete.request.parent / "live_progress.json", dict(state="waiting_for_actual_reviewed_identity"))
    controller = continuation.IdentityContinuation(complete.root, complete.request, complete.grant)
    clock = [complete.start + timedelta(hours=1)]
    monkeypatch.setattr(controller, "now", lambda: clock[0])
    calls = []

    class Child:
        def __init__(self, role):
            self.role = role
            self.pid = 999999
            self.returncode = None

        def poll(self):
            if self.role == "identity_native_observations" and failure == "local_deadline":
                clock[0] += timedelta(minutes=21)
                return None
            if self.role == "identity_cuda_release":
                assert json.loads(controller.stage_progress.read_bytes())["stage"] == "poses_complete"
                if failure != "observer_exit":
                    lease = json.loads(identity_lease.read_bytes())
                    lease.update(status="released", all_observed_owned_processes_exited=True)
                    after = put(identity_lease, lease)
                    snapshot = put(
                        complete.root / "identity_poses_01_release_receipt.progress.json", dict(stage="poses_complete")
                    )
                    put(
                        identity_release,
                        dict(
                            status="released",
                            kind="poses",
                            lease_after=after,
                            full_pose_progress=snapshot,
                            observed_process_identities=[dict(pid=self.pid, create_time=42.0, exited=True)],
                        ),
                    )
            self.returncode = int(
                (failure == "GPU_exit" and self.role == "identity_native_observations")
                or (failure == "observer_exit" and self.role == "identity_cuda_release")
            )
            return self.returncode

    def spawn(role, command, cuda=False):
        if cuda:
            args = command[command.index("--worker-lease") + 1]
            item = binding(Path(args))
            continuation._worker_authority(complete.root, item)
            lease = json.loads(Path(args).read_bytes())
            assert datetime.fromisoformat(lease["approved_deadline_utc"]) - datetime.fromisoformat(
                lease["approved_start_utc"]
            ) == timedelta(minutes=stage_cap)
        if role == "identity_fixed_camera_geometric_review":
            assert json.loads(identity_lease.read_bytes())["status"] == "released"
            assert identity_release.exists()
        logs = []
        for suffix in (".stdout.log", ".stderr.log"):
            path = complete.evidence / (role + suffix)
            path.write_bytes(b"synthetic child execution, no CUDA/model\n")
            logs.append(str(path))
        execution = dict(
            role=role, cli_pid=999999, cli_create_time=42.0, command=command, stdout=logs[0], stderr=logs[1]
        )
        controller.state["worker_executions"].append(execution)
        calls.append(role)
        return Child(role), execution

    monkeypatch.setattr(controller, "spawn", spawn)
    if failure:
        with pytest.raises((RuntimeError, TimeoutError)):
            controller.run()
        assert controller.state["stage"] == "failed_requires_actual_review"
        assert controller.state["owned_failure_cleanup"][-1]["unrelated_processes_untouched"] is True
        assert "identity_fixed_camera_geometric_review" not in calls
    else:
        result = controller.run()
        assert result == binding(complete.evidence / "native_identity_completion.json")
        assert calls == [
            "identity_native_observations",
            "identity_cuda_release",
            "identity_fixed_camera_geometric_review",
            "identity_conservative_seal",
        ]
        assert all(item["exit_code"] == 0 for item in controller.state["worker_executions"])
        assert controller.state["stage"] == "complete"


@pytest.mark.parametrize("stage_cap,worker_minutes", [(10, 10), (10, 20), (15, 15), (15, 20)])
def test_worker_honors_shorter_original_stage_authority(complete, stage_cap, worker_minutes):
    grant = json.loads(complete.grant.read_bytes())
    grant["maximum_stage_minutes"]["poses"] = stage_cap
    put(complete.grant, grant)
    authority = gpu_handoff.read_continuous_grant(complete.request, complete.grant)
    now = complete.start + timedelta(hours=1)
    lease = gpu_handoff.make_local_stage_lease(authority, complete.root, "poses", now=now)
    lease["approved_deadline_utc"] = (now + timedelta(minutes=worker_minutes)).isoformat()
    item = put(complete.root / "gpu_leases/future_identity.json", lease)
    if worker_minutes > stage_cap:
        with pytest.raises(ValueError):
            continuation._worker_authority(complete.root, item)
    else:
        assert continuation._worker_authority(complete.root, item)["binding"] == binding(complete.grant)


def test_new_implementation_receipt_preserves_prior_actual_completion_bytes(complete):
    first = continuation.freeze_completion(complete.root)
    payload = Path(first["path"]).read_bytes()
    implementation = complete.project / "hmr4d/lidar_mocap/identity_continuation.py"
    implementation.write_bytes(b"# revised implementation, identical actual outcomes\n")
    second = continuation.freeze_completion(complete.root)
    assert second["path"] != first["path"]
    assert Path(first["path"]).read_bytes() == payload
    assert continuation.freeze_completion(complete.root) == second
