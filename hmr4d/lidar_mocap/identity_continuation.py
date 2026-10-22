"""Continue native identity review using genuine pose exits and an existing grant."""

from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import uuid

import psutil

from hmr4d import PROJ_ROOT
from .formal_baseline import _binding, _bound, _encode, _publish, _scope
from .gpu_handoff import make_local_stage_lease, read_continuous_grant
from .pilot import check_binding
from .repair_execution import native_gpu_command_matches, read_live_json, write_progress_json
from .windows_gpu_job import WindowsGPUJob

ENTRY = "tools/continue_badminton_native_identity.py"
WORKERS = dict(
    observe="observe_badminton_repair_poses.py",
    review="review_badminton_identity_views.py",
    seal="build_badminton_identity_decisions.py",
)


def _root(root):
    root = Path(root).resolve()
    outputs = (PROJ_ROOT / "outputs").resolve()
    if root == outputs or not root.is_relative_to(outputs) or not (root / "pilot_input.json").is_file():
        raise ValueError("Select an explicit prepared full-native output run")
    return root


def _release(root, path):
    value = read_live_json(path)
    if (
        value.get("status") != "released"
        or value.get("kind") != "poses"
        or not value.get("observed_process_identities")
        or any(item.get("exited") is not True for item in value["observed_process_identities"])
    ):
        raise ValueError("Actual observed native pose executors must have exited")
    lease, _ = _bound(value["lease_after"], {})
    if (
        not Path(value["lease_after"]["path"]).resolve().is_relative_to(root / "gpu_leases")
        or lease.get("root") != str(root)
        or lease.get("owner") != "GVHMR"
        or lease.get("status") != "released"
        or lease.get("all_observed_owned_processes_exited") is not True
        or lease.get("manifest_sha256") != read_live_json(root / "pilot_input.json")["manifest"]["sha256"]
    ):
        raise ValueError("Actual release uses another run, source or lease")
    progress, _ = _bound(value["full_pose_progress"], {})
    if progress.get("stage") != "poses_complete":
        raise ValueError("A failed or incomplete native pose stage cannot admit identity")
    return value


def verify_completed_identity(root):
    """Validate existing outcomes read-only; never replay inference or sign old gaps."""
    root = _root(root)
    repair, evidence = root / "repair", root / "repair/identity_evidence"
    source = _binding(repair / "new_source_input.json")
    plan = read_live_json(Path(source["path"]))
    keys = set(_scope(plan["jobs"]))
    full_pose = read_live_json(repair / "pose_receipt.json")
    pose = read_live_json(evidence / "pose_receipt.json")
    decision = read_live_json(evidence / "identity_decisions.json")
    review = read_live_json(evidence / "native_identity_review.json")
    boundaries = read_live_json(evidence / "boundaries.json")
    if (
        full_pose.get("state") != "complete"
        or pose.get("state") != "complete"
        or full_pose.get("source_union_sha256") != source["sha256"]
        or pose.get("source_union_sha256") != source["sha256"]
        or pose.get("config") != full_pose.get("config")
        or decision.get("schema") != "badminton-native-identity-decisions-v1"
        or decision.get("review_status") != "reviewed"
        or set(decision.get("windows", {})) != keys
        or decision.get("source_union_sha256") != source["sha256"]
        or decision.get("calibration_digest") != plan["calibration_digest"]
        or decision.get("unresolved_cases")
        or not decision.get("evidence_files")
    ):
        raise ValueError("Complete reviewed identity must cover the exact current native scope")
    if (
        review.get("partial") is not False
        or len(review.get("cases", [])) != len(boundaries)
        or not review.get("inputs")
        or Counter(case["status"] for case in review["cases"]) != review.get("counts")
    ):
        raise ValueError("Actual full native geometric review is incomplete")
    for expected, actual in zip(boundaries, review["cases"]):
        if actual.get("missing_native_context") != 0 or any(
            actual.get(key) != expected[key]
            for key in ("index", "source_window", "event", "positions", "context_positions")
        ):
            raise ValueError("Native geometric cases differ from their frozen scope")
    for name, sha in {**review["inputs"], **decision["evidence_files"]}.items():
        check_binding(dict(path=name, sha256=sha))
    _verify_operations(plan, decision, review, evidence)
    full_release = root / "poses_01_release_receipt.json"
    identity_release = root / "identity_poses_01_release_receipt.json"
    for path in (full_release, identity_release):
        _release(root, path)
    assets = [
        source,
        *(
            _binding(path)
            for path in (
                repair / "pose_receipt.json",
                evidence / "pose_receipt.json",
                evidence / "identity_decisions.json",
                evidence / "native_identity_review.json",
                evidence / "boundaries.json",
                full_release,
                identity_release,
            )
        ),
    ]
    return dict(
        schema="gvhmr-native-identity-completion-v1",
        state="actual_identity_complete",
        root=str(root),
        windows=len(keys),
        geometric_cases=len(boundaries),
        geometric_counts=review["counts"],
        abstained=len(decision.get("abstained_relationship_cases", [])),
        evidence=assets,
        numerical_body_native_verification_claimed=False,
        training_published=False,
    )


def _verify_operations(plan, decision, review, evidence):
    """Reproduce the existing conservative decision contract without writing it."""
    expected = {}
    for key, job in _scope(plan["jobs"]).items():
        source = Path(job["source_window"])
        expected[key] = dict(
            source_window=str(source),
            source_teacher_manifest_sha256=_binding(source / "teacher/manifest.json")["sha256"],
            links=[],
            cannot_link=[],
            drop_spans=[],
            review_cases=[],
            reset_tracklets=[],
        )
    pending = []
    for case in review["cases"]:
        key = case["event"]["source_id"] + ":" + Path(case["source_window"]).name
        ids = [int(case["event"][side + "_motion_id"].split(":")[-2]) for side in ("before", "after")]
        if [vote["original_track_id"] for vote in case["votes"]] != ids:
            raise ValueError("Identity votes refer to other raw tracklets")
        target = expected[key]
        if case["status"] == "same_native_person":
            target["links"].append(
                dict(tracks=ids, case=case["index"], method="native_opposite_camera_continuity_and_epipolar_pose")
            )
        elif case["status"] == "different_native_people":
            target["cannot_link"].append(ids)
        else:
            pending.append(case["index"])
        target["review_cases"].append(case["index"])
    inventory = read_live_json(evidence / "boundary_inventory.pending.json")
    for item in inventory["tracklets"]:
        expected[item["source_id"] + ":" + item["window"]]["reset_tracklets"].append(int(item["original_track_id"]))
    for target in expected.values():
        target["reset_tracklets"] = sorted(set(target["reset_tracklets"]))
    policy_path = evidence / "abstention_policy.json"
    _check_policy(evidence, decision["source_union_sha256"])
    if (
        decision.get("abstention_policy") != str(policy_path.resolve())
        or decision.get("abstained_relationship_cases") != pending
        or decision["windows"] != expected
    ):
        raise ValueError("Actual decision operations differ from complete geometric review and conservative policy")


def _check_policy(evidence, source_sha):
    policy = read_live_json(evidence / "abstention_policy.json")
    if (
        policy.get("schema") != "badminton-new-source-identity-abstention-v1"
        or policy.get("review_status") != "reviewed"
        or policy.get("action") != "keep_unproven_original_tracklets_separate"
        or policy.get("source_union_sha256") != source_sha
        or policy.get("boundary_inventory_sha256") != _binding(evidence / "boundary_inventory.reviewed.json")["sha256"]
        or not policy.get("evidence_files")
    ):
        raise ValueError("Reviewed conservative policy must bind this actual source and inventory")
    for name, sha in policy["evidence_files"].items():
        check_binding(dict(path=name, sha256=sha))
    return policy


def freeze_completion(root):
    """Bind genuine prior execution without launching another model or CUDA stage."""
    value = verify_completed_identity(root)
    root = Path(root).resolve()
    path = root / "repair/identity_evidence/native_identity_completion.json"
    value["implementation"] = [
        _binding(PROJ_ROOT / name) for name in ("hmr4d/lidar_mocap/identity_continuation.py", ENTRY)
    ]
    if path.exists() and path.read_bytes() != _encode(value):
        prior = read_live_json(path)
        if {key: item for key, item in prior.items() if key != "implementation"} != {
            key: item for key, item in value.items() if key != "implementation"
        }:
            raise ValueError("Actual completed identity outcomes differ from their immutable prior receipt")
        revision = hashlib.sha256(_encode(value["implementation"])).hexdigest()
        path = path.parent / "native_identity_completions" / (revision + ".json")
    _publish(path, _encode(value))
    return _binding(path)


def worker_command(root, role, threads, lease=None):
    if role not in WORKERS or threads not in (1, 2):
        raise ValueError("Select an existing identity worker and one or two threads")
    command = [
        str(PROJ_ROOT / ".venv/Scripts/python.exe"),
        "-u",
        str(PROJ_ROOT / ENTRY),
        "--root",
        str(root),
        "--threads",
        str(threads),
        "--worker",
        role,
        "--worker-script",
        str(PROJ_ROOT / "tools" / WORKERS[role]),
    ]
    if role == "observe":
        if lease is None:
            raise ValueError("Native observation worker requires its actual bound local lease")
        command += ["--worker-lease", lease["path"], "--worker-lease-sha256", lease["sha256"]]
    elif lease is not None:
        raise ValueError("CPU identity workers do not acquire CUDA authority")
    return command


def run_worker(root, role, script, threads, lease=None):
    root = _root(root)
    if (
        role not in WORKERS
        or Path(script).resolve() != (PROJ_ROOT / "tools" / WORKERS[role]).resolve()
        or threads not in (1, 2)
    ):
        raise ValueError("Identity worker must name its exact established script")
    cuda = role == "observe"
    if cuda:
        _worker_authority(root, lease)
    elif lease is not None:
        raise ValueError("CPU identity workers do not acquire CUDA authority")
    else:
        _release(root, root / "identity_poses_01_release_receipt.json")
    os.environ.update(
        CUDA_VISIBLE_DEVICES="0" if cuda else "-1",
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS="1",
    )
    process = psutil.Process()
    process.cpu_affinity([0, 1] if cuda else [26, 27])
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    repair, evidence = root / "repair", root / "repair/identity_evidence"
    argv = [str(script), "--root", str(evidence if cuda else repair)]
    if cuda:
        argv += ["--source-root", str(root / "source"), "--reuse-poses-root", str(repair), "--threads", str(threads)]
    elif role == "seal":
        argv += ["--abstention-policy", str(evidence / "abstention_policy.json")]
    old = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old


def _worker_authority(root, item):
    if item is None:
        raise ValueError("Native observation worker requires its bound local lease")
    lease, _ = _bound(item, {})
    grant, _ = _bound(lease["continuous_gpu_grant"], {})
    authorization = read_continuous_grant(grant["request"]["path"], lease["continuous_gpu_grant"]["path"])
    now = datetime.now(timezone.utc)
    start = datetime.fromisoformat(lease["approved_start_utc"])
    end = datetime.fromisoformat(lease["approved_deadline_utc"])
    if (
        lease.get("status") != "held"
        or lease.get("owner") != "GVHMR"
        or lease.get("stage") != "poses"
        or lease.get("root") != str(root)
        or str(root) not in authorization["roots"]
        or lease.get("continuous_gpu_grant") != authorization["binding"]
        or not authorization["start"] <= start <= now < end <= authorization["deadline"]
        or end > start + timedelta(minutes=min(20, authorization["grant"]["maximum_stage_minutes"]["poses"]))
        or not Path(item["path"]).resolve().is_relative_to(root / "gpu_leases")
        or lease.get("manifest_sha256") != authorization["roots"][str(root)]["manifest_sha256"]
    ):
        raise ValueError("Bound identity worker lease does not authorize this actual run/time")
    _release(root, root / "poses_01_release_receipt.json")
    _check_policy(root / "repair/identity_evidence", _binding(root / "repair/new_source_input.json")["sha256"])
    return authorization


class IdentityContinuation:
    """One bounded identity handoff, with CPU review after genuine CUDA release."""

    def __init__(self, root, request, grant, threads=2):
        self.root = _root(root)
        if threads not in (1, 2):
            raise ValueError("Identity continuation uses one or two threads")
        self.threads = threads
        self.authorization = read_continuous_grant(request, grant)
        if str(self.root) not in self.authorization["roots"]:
            raise ValueError("Continuous authority does not admit this root")
        self.queue = Path(request).resolve().parent
        self.repair, self.evidence = self.root / "repair", self.root / "repair/identity_evidence"
        self.state_path = self.evidence / "identity_work_progress.json"
        if self.state_path.exists():
            raise ValueError("Existing identity execution is preserved; use --verify-complete after genuine completion")
        self.lease_path = self.root / "gpu_leases/identity_poses_01.json"
        self.release_path = self.root / "identity_poses_01_release_receipt.json"
        self.stage_progress = self.evidence / "identity_gpu_stage_progress.json"
        self.immutable = [
            _binding(self.evidence / name)
            for name in ("boundary_inventory.reviewed.json", "boundaries.json", "case_plan.json")
        ]
        self.immutable += [
            _binding(self.repair / "new_source_input.json"),
            _binding(self.root / "teacher_sampling_input.json"),
        ]
        self.immutable += [_binding(PROJ_ROOT / name) for name in ("hmr4d/lidar_mocap/identity_continuation.py", ENTRY)]
        process = psutil.Process()
        self.state = dict(
            stage="waiting_for_actual_complete_full_pose_release",
            pid=process.pid,
            create_time=process.create_time(),
            started_utc=self.now().isoformat(),
            CPU_affinity=[26, 27],
            CUDA_affinity=[0, 1],
            immutable_inputs=self.immutable,
            continuous_grant=self.authorization["binding"],
            worker_executions=[],
            training_published=False,
        )
        self.active_cuda = False
        self.owned = []

    @staticmethod
    def now():
        return datetime.now(timezone.utc)

    def update(self, **values):
        self.state.update(values, updated_utc=self.now().isoformat())
        write_progress_json(self.state_path, self.state)

    def check(self):
        check_binding(self.authorization["binding"])
        if self.now() >= self.authorization["deadline"]:
            raise TimeoutError("Original continuous GPU deadline reached")
        for item in self.immutable:
            check_binding(item)

    def spawn(self, role, command, cuda=False):
        stdout, stderr = [self.evidence / (role + suffix) for suffix in (".stdout.log", ".stderr.log")]
        try:
            job = WindowsGPUJob("Local\\GVHMRIdentity_" + uuid.uuid4().hex)
            owned = dict(role=role, job=job, child=None, closed=False)
            self.owned.append(owned)
            with stdout.open("xb") as out, stderr.open("xb") as err:
                # Keep the Job registered while start owns the original Popen
                # HANDLE; startup failures are reaped before a child can resume.
                child, launched = job.start(
                    command,
                    cwd=PROJ_ROOT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="0" if cuda else "-1"),
                    log=out,
                    affinity=[0, 1] if cuda else [26, 27],
                )
                owned.update(child=child, launcher=launched)
                if cuda:
                    self.active_cuda = True
            item = dict(
                role=role,
                cli_pid=launched["pid"],
                cli_create_time=launched["create_time"],
                command=command,
                started_utc=self.now().isoformat(),
                stdout=str(stdout),
                stderr=str(stderr),
                output_streams_merged_into_stdout=True,
                private_job=job.name,
                assigned_before_resume=launched["assigned_before_resume"],
            )
            self.state["worker_executions"].append(item)
            self.update(stage=role, cli_pid=item["cli_pid"], cli_create_time=item["cli_create_time"], command=command)
            return child, item
        except BaseException as error:
            self.abort_owned(error)
            raise

    def abort_owned(self, error):
        """Reap only launchers and descendants in this continuation's Jobs."""
        cleaned, errors = [], []
        for owned in reversed(self.owned):
            if owned["closed"]:
                continue
            result = dict(
                role=owned["role"],
                private_job=owned["job"].name,
                launcher=owned.get("launcher"),
                actual_job_empty=False,
            )
            try:
                owned["job"].terminate()
                owned["job"].wait_empty()
                result["actual_job_empty"] = True
                if owned["child"] is not None:
                    result["launcher_exit_code"] = owned["child"].wait(timeout=5)
                    result["launcher_reaped"] = True
            except BaseException as cleanup_error:
                errors.append(dict(role=owned["role"], error=repr(cleanup_error)))
            finally:
                try:
                    owned["job"].close()
                    if owned["child"] is not None:
                        result["launcher_exit_code"] = owned["child"].wait(timeout=5)
                        result["launcher_reaped"] = True
                except BaseException as close_error:
                    errors.append(dict(role=owned["role"], error=repr(close_error)))
                owned["closed"] = True
            cleaned.append(result)
        self.active_cuda = False
        failure = dict(
            state="failed_own_job_cleanup_unconfirmed" if errors else "failed_own_jobs_stopped",
            error_type=type(error).__name__,
            error=str(error),
            owned_jobs=cleaned,
            cleanup_errors=errors,
            unrelated_processes_untouched=True,
            training_published=False,
        )
        self.state.setdefault("owned_failure_cleanup", []).append(failure)
        # This exclusive file does not depend on the atomic writer that may
        # itself have failed. Reaping precedes every attempt to write evidence.
        try:
            path = self.evidence / "continuation_failures" / (uuid.uuid4().hex + ".json")
            _publish(path, _encode(failure))
            self.state.setdefault("failure_evidence", []).append(_binding(path))
        except BaseException as recording_error:
            self.state.setdefault("failure_recording_errors", []).append(repr(recording_error))
        return failure

    def wait_child(self, child, execution, deadline=None):
        while child.poll() is None:
            self.check()
            if deadline and self.now() >= deadline:
                raise TimeoutError("Identity stage local deadline reached")
            self.update()
            time.sleep(2)
        for owned in self.owned:
            if owned["child"] is child and not owned["closed"]:
                owned["job"].wait_empty()
                child.wait(timeout=5)
                owned["job"].close()
                owned["closed"] = True
        execution.update(
            exit_code=child.returncode,
            completed_utc=self.now().isoformat(),
            stdout=_binding(execution["stdout"]),
            stderr=_binding(execution["stderr"]),
        )
        self.update()
        if child.returncode:
            raise RuntimeError("Identity worker failed: " + execution["role"])

    def run(self):
        process = psutil.Process()
        process.cpu_affinity([26, 27])
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            self.update()
            while True:
                self.check()
                live = read_live_json(self.queue / "live_progress.json")
                if live["state"].startswith("failed"):
                    raise RuntimeError("The sole production coordinator failed")
                if (self.root / "poses_01_release_receipt.json").exists() and live[
                    "state"
                ] == "waiting_for_actual_reviewed_identity":
                    break
                self.update()
                time.sleep(10)
            _release(self.root, self.root / "poses_01_release_receipt.json")
            poses = read_live_json(self.repair / "pose_receipt.json")
            if (
                poses.get("state") != "complete"
                or poses.get("source_union_sha256") != _binding(self.repair / "new_source_input.json")["sha256"]
            ):
                raise ValueError("Actual complete full native poses are required")
            self.immutable += [
                _binding(self.evidence / "abstention_policy.json"),
                _binding(self.repair / "pose_receipt.json"),
                _binding(self.root / "poses_01_release_receipt.json"),
            ]
            _check_policy(self.evidence, _binding(self.repair / "new_source_input.json")["sha256"])
            for process in psutil.process_iter(["pid", "cmdline"]):
                try:
                    if any(
                        native_gpu_command_matches(process.info["cmdline"], Path(name))
                        for name in self.authorization["roots"]
                    ):
                        raise RuntimeError("An authorized predecessor native CUDA executor remains active")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if self.lease_path.exists() or self.release_path.exists():
                raise ValueError("Preserve existing identity GPU lease and release")
            lease = make_local_stage_lease(self.authorization, self.root, "poses", now=self.now())
            deadline = min(
                datetime.fromisoformat(lease["approved_deadline_utc"]),
                datetime.fromisoformat(lease["approved_start_utc"]) + timedelta(minutes=20),
            )
            lease.update(
                approved_deadline_utc=deadline.isoformat(),
                identity_case_plan=_binding(self.evidence / "case_plan.json"),
                reviewed_inventory=_binding(self.evidence / "boundary_inventory.reviewed.json"),
                teacher_sampling_input=_binding(self.root / "teacher_sampling_input.json"),
            )
            write_progress_json(self.lease_path, lease)
            self.update(lease=_binding(self.lease_path))
            child, execution = self.spawn(
                "identity_native_observations",
                worker_command(self.root, "observe", self.threads, _binding(self.lease_path)),
                cuda=True,
            )
            seed = self.evidence / "identity_original_process_seed.json"
            write_progress_json(
                seed,
                dict(
                    root=str(self.root),
                    lease_before=_binding(self.lease_path),
                    command=execution["command"],
                    observed_process_identities=[
                        dict(pid=execution["cli_pid"], create_time=execution["cli_create_time"])
                    ],
                ),
            )
            observer, observed = self.spawn(
                "identity_cuda_release",
                [
                    str(PROJ_ROOT / ".venv/Scripts/python.exe"),
                    "-u",
                    str(PROJ_ROOT / "tools/release_badminton_full_gpu.py"),
                    "--root",
                    str(self.root),
                    "--lease",
                    str(self.lease_path),
                    "--progress",
                    str(self.stage_progress),
                    "--receipt",
                    str(self.release_path),
                    "--kind",
                    "poses",
                    "--seed-evidence",
                    str(seed),
                    "--seed-evidence-sha256",
                    _binding(seed)["sha256"],
                ],
            )
            self.update(
                stage="identity_native_observations",
                actual_GPU_cli_pid=child.pid,
                actual_GPU_cli_create_time=execution["cli_create_time"],
                release_observer_cli_pid=observer.pid,
            )
            self.wait_child(child, execution, deadline)
            if read_live_json(self.evidence / "pose_receipt.json").get("state") != "complete":
                raise ValueError("Complete actual identity observations are required")
            write_progress_json(
                self.stage_progress,
                dict(
                    stage="poses_complete",
                    safe_to_release_gpu=True,
                    actual_pose_receipt=_binding(self.evidence / "pose_receipt.json"),
                    training_published=False,
                ),
            )
            self.wait_child(observer, observed, min(deadline, self.now() + timedelta(seconds=90)))
            _release(self.root, self.release_path)
            self.active_cuda = False
            self.state.pop("actual_GPU_cli_pid", None)
            self.state.pop("actual_GPU_cli_create_time", None)
            self.state.pop("release_observer_cli_pid", None)
            for role, name in (
                ("review", "identity_fixed_camera_geometric_review"),
                ("seal", "identity_conservative_seal"),
            ):
                child, execution = self.spawn(name, worker_command(self.root, role, self.threads))
                self.wait_child(child, execution)
            receipt = freeze_completion(self.root)
            self.update(
                stage="complete",
                completion=receipt,
                decision=_binding(self.evidence / "identity_decisions.json"),
                physical_cross_window_identity_admitted=False,
            )
            return receipt
        except BaseException as error:
            cleanup = self.abort_owned(error)
            try:
                write_progress_json(
                    self.stage_progress,
                    dict(stage="failed", error=str(error), owned_cleanup=cleanup, training_published=False),
                )
                self.update(stage="failed_requires_actual_review", error_type=type(error).__name__, error=str(error))
            except BaseException as recording_error:
                self.state.setdefault("failure_recording_errors", []).append(repr(recording_error))
            raise
