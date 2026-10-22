"""Continue the two approved full runs serially; never synthesize identity/QA reviews."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import urlopen

import psutil

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.gpu_handoff import read_continuous_grant, make_local_stage_lease, stop_owned_cuda_trees
from hmr4d.lidar_mocap.pilot import binding, check_binding
from hmr4d.lidar_mocap.repair_execution import write_progress_json, teacher_batch_metrics, native_gpu_command_matches


def reviewed_identity_ready(root):
    path = root / "repair/identity_evidence/identity_decisions.json"
    if not path.exists():
        return False
    value = json.loads(path.read_bytes())
    if value.get("review_status") != "reviewed":
        return False
    source = root / "repair/new_source_input.json"
    plan = json.loads(source.read_bytes())
    required = {item["source_id"] + ":" + Path(item["source_window"]).name for item in plan["jobs"]}
    if (value.get("source_union_sha256") != binding(source)["sha256"]
            or value.get("calibration_digest") != plan["calibration_digest"]
            or not required <= set(value.get("windows", {})) or not value.get("evidence_files")):
        raise ValueError("Identity review does not cover the current full source/camera/window evidence")
    for name, sha in value["evidence_files"].items():
        check_binding(dict(path=name, sha256=sha))
    return True


class Sequence:
    def __init__(self, request, grant, directory, threads=2, paired_native_rgb=False):
        self.request, self.grant_path, self.directory = request, grant, directory
        self.threads, self.authorization, self.child, self.gpu_root, self.active_progress = threads, None, None, None, None
        self.paired_native_rgb = paired_native_rgb
        self.state_path = directory / "continuous_sequence.json"
        previous = json.loads(self.state_path.read_bytes()) if self.state_path.exists() else {}
        if previous.get("pid"):
            try:
                old = psutil.Process(previous["pid"])
                if old.create_time() == previous.get("create_time"):
                    raise RuntimeError("Another continuous sequence controller is running")
            except psutil.NoSuchProcess:
                pass
        self.events = directory / "sequence_events"
        self.events.mkdir(parents=True, exist_ok=True)
        self.event_index = max([int(path.stem.split("_")[1]) for path in self.events.glob("event_*.json")] or [0])
        self.state = dict(state="waiting_for_continuous_grant", pid=os.getpid(), create_time=psutil.Process().create_time(),
                          request=binding(request), started_utc=datetime.now(timezone.utc).isoformat(),
                          cpu_only_coordinator=True, training_published=False)
        self.last_save = 0.

    def heartbeat(self, force=False):
        if not force and time.monotonic() - self.last_save < 10:
            return
        self.state["heartbeat_utc"] = datetime.now(timezone.utc).isoformat()
        first = Path(json.loads(self.request.read_bytes())["scope"][0]["root"])
        batches = [(int(re.search(r"batch(\d+)", path.name)[1]), path)
                   for path in first.glob("full_teacher_progress.batch*.json")]
        index, current = max(batches, default=(0, None))
        baseline = first / "full_teacher_progress.json"
        if index > 1:
            release_path = first / f"teacher_batch_{index-1:02d}_release_receipt.json"
            if release_path.exists():
                proof = json.loads(release_path.read_bytes())["full_teacher_progress"]
                check_binding(proof)
                baseline = Path(proof["path"])
        if current is not None and current.exists():
            raw = json.loads(current.read_bytes())
            previous = json.loads(baseline.read_bytes())
            self.state["first_teacher_metrics"] = teacher_batch_metrics(raw, previous)
            self.state["first_teacher_progress_path"] = str(current)
            self.state["first_teacher_baseline"] = binding(baseline)
            self.state["first_teacher_active"] = raw.get("active")
            self.state.setdefault("previous_attempts", [dict(progress=binding(baseline), **{
                key: previous.get(key) for key in ("state", "completed_camera_windows", "new_camera_windows_this_batch",
                    "batch_seconds", "started_batch_utc", "completed_batch_utc")})])
        write_progress_json(self.state_path, self.state)
        write_progress_json(self.directory / "live_progress.json", self.state)
        self.last_save = time.monotonic()

    def event(self, kind, root=None, **details):
        self.event_index += 1
        value = dict(sequence=self.event_index, kind=kind, root=str(root) if root else None,
                     utc=datetime.now(timezone.utc).isoformat(), **details)
        path = self.events / f"event_{self.event_index:05d}.json"
        write_progress_json(path, value)
        self.state["last_event"] = binding(path)
        self.heartbeat(True)
        print(json.dumps(value), flush=True)

    def check_window(self):
        if self.authorization is None:
            return
        check_binding(self.authorization["binding"])
        if datetime.now(timezone.utc) >= self.authorization["deadline"]:
            stop_owned_cuda_trees(self.authorization["roots"])
            raise TimeoutError("The fixed continuous GPU deadline was reached")

    def wait(self):
        self.check_window()
        self.heartbeat()
        time.sleep(2)

    def spawn(self, command, log, gpu=False):
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0" if gpu else "-1")
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            environment[key] = str(self.threads)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("ab") as handle:
            return subprocess.Popen(command, cwd=PROJ_ROOT, env=environment, stdout=handle,
                stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)

    def run(self, root, command, role, *, gpu=False, lease=None, progress=None, receipt=None, kind=None):
        self.check_window()
        if gpu:
            for process in psutil.process_iter(["cmdline"]):
                try:
                    if any(native_gpu_command_matches(process.info["cmdline"], Path(name)) for name in self.authorization["roots"]):
                        raise RuntimeError("A prior CUDA executor still owns the continuous window; no concurrent launch")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        self.state.update(state="running_" + role, root=str(root), command=command)
        self.child = self.spawn(command, root / "logs" / (role + ".log"), gpu)
        self.gpu_root = root if gpu else None
        self.active_progress = progress if gpu else None
        self.state.update(cli_pid=self.child.pid, cli_create_time=psutil.Process(self.child.pid).create_time())
        observer = None
        if gpu:
            observer = self.spawn([sys.executable, "-u", str(PROJ_ROOT / "tools/release_badminton_full_gpu.py"),
                "--root", str(root), "--lease", str(lease), "--progress", str(progress), "--receipt", str(receipt),
                "--kind", kind], root / "logs" / (role + "_release.log"))
        self.heartbeat(True)
        started = time.perf_counter()
        while self.child.poll() is None:
            self.wait()
        code = self.child.returncode
        if code:
            if gpu:
                stop_owned_cuda_trees([root])
                value = json.loads(progress.read_bytes()) if progress.exists() else {}
                value.update(state="failed", stage="failed", safe_to_release_gpu=True, exit_code=code)
                write_progress_json(progress, value)
            raise RuntimeError(f"Continuous stage {role} failed with exit {code}; inspect its preserved log")
        if observer:
            timeout = time.monotonic() + 90
            while observer.poll() is None:
                if time.monotonic() > timeout:
                    raise RuntimeError("GPU release observer did not finish; no subsequent GPU stage launched")
                self.wait()
            if observer.returncode:
                raise RuntimeError("GPU release observer failed; retain the lease and review its actual process evidence")
        self.child, self.gpu_root, self.active_progress = None, None, None
        self.event(role + "_complete", root, seconds=time.perf_counter() - started,
                   release=binding(receipt) if receipt else None)

    def command(self, root, stage):
        command = [sys.executable, "-u", str(PROJ_ROOT / "tools/run_badminton_full_native.py"),
                   "--root", str(root), "--stage", stage, "--threads", str(self.threads)]
        if self.paired_native_rgb and stage == "teacher":
            from hmr4d.lidar_mocap.paired_sampling import validate_sampling_input
            path = root / "teacher_sampling_input.json"
            validate_sampling_input(root / "repair", path)
            command += ["--teacher-sampling-input", str(path)]
        return command

    def lease(self, root, stage, label):
        path = root / "gpu_leases" / (label + ".json")
        if path.exists():
            raise ValueError("A local lease already exists; preserve it and use a new batch")
        value = make_local_stage_lease(self.authorization, root, stage)
        if self.paired_native_rgb:
            value["teacher_sampling_input"] = binding(root / "teacher_sampling_input.json")
        if (datetime.fromisoformat(value["approved_deadline_utc"]) - datetime.now(timezone.utc)).total_seconds() < 600:
            raise TimeoutError("Insufficient time remains for a new CUDA stage")
        write_progress_json(path, value)
        return path

    def batches(self, root, stage):
        marker = root / ("dual_teacher_receipt.json" if stage == "teacher" else "full_aggregate_progress.json")
        if marker.exists() and (json.loads(marker.read_bytes()).get("complete") is True
                                or json.loads(marker.read_bytes()).get("state") == "complete_cpu_aggregate"):
            return
        stem = "full_teacher_progress" if stage == "teacher" else "full_fit_progress"
        while True:
            existing = [(int(re.search(r"batch(\d+)", path.name)[1]), path) for path in root.glob(stem + ".batch*.json")]
            previous = max(existing, default=(0, None))
            index, resume = previous[0] + 1, previous[1]
            if resume is not None:
                old_release = root / f"{stage}_batch_{index-1:02d}_release_receipt.json"
                if not old_release.exists():
                    pending = json.loads(resume.read_bytes())
                    if pending.get("state") in ("running", "complete", "yielded_before_deadline", "fit_windows_complete"):
                        self.state.update(state="waiting_for_existing_" + stage + "_batch_real_release",
                                          root=str(root), existing_progress_path=str(resume))
                        self.wait()
                        continue
                    raise ValueError("Prior GPU batch lacks its real release receipt; no new lease created")
                old = json.loads(old_release.read_bytes())
                resume = Path(old["full_teacher_progress" if stage == "teacher" else "full_fit_progress"]["path"])
                check_binding(old["full_teacher_progress" if stage == "teacher" else "full_fit_progress"])
                if marker.exists() and (json.loads(marker.read_bytes()).get("complete") is True
                                        or json.loads(marker.read_bytes()).get("state") == "complete_cpu_aggregate"):
                    return
            progress = root / f"{stem}.batch{index:02d}.json"
            release = root / f"{stage}_batch_{index:02d}_release_receipt.json"
            lease = self.lease(root, stage, f"{stage}_batch_{index:02d}")
            command = self.command(root, stage) + ["--gpu-lease", str(lease), "--progress-output", str(progress)]
            if resume: command += ["--resume-progress", str(resume)]
            self.run(root, command, f"{stage}_batch_{index:02d}", gpu=True, lease=lease,
                     progress=progress, receipt=release, kind=stage)
            outcome = json.loads(progress.read_bytes())["state"]
            if outcome in ("complete", "fit_windows_complete"):
                return

    def cohort(self, root, port):
        while not (root / "body_prepare_cpu_receipt.json").exists():
            self.state.update(state="waiting_for_actual_full_cpu_prepare", root=str(root))
            self.wait()
        self.batches(root, "teacher")
        decisions = root / "repair/identity_evidence/identity_decisions.json"
        if not reviewed_identity_ready(root):
            inventory = root / "repair/identity_evidence/boundary_inventory.pending.json"
            if not inventory.exists():
                self.run(root, self.command(root, "identity"), "identity_inventory_cpu")
            tracklets = len(json.loads(inventory.read_bytes())["tracklets"])
            endpoint = root / "repair/identity_evidence/endpoint_previews/endpoints-0.json"
            if not endpoint.exists():
                self.run(root, [sys.executable, "-u", str(PROJ_ROOT / "tools/prepare_badminton_identity_review.py"),
                    "--root", str(root / "repair"), "--render-endpoints", "--endpoint-limit", str(max(1, tracklets))],
                    "identity_endpoints_cpu")
            self.event("identity_endpoint_review_available", root, inventory=binding(inventory), endpoint=binding(endpoint))
        pose = root / "repair/pose_receipt.json"
        if not pose.exists() or json.loads(pose.read_bytes()).get("state") != "complete":
            lease = self.lease(root, "poses", "poses_01")
            self.run(root, self.command(root, "poses") + ["--gpu-lease", str(lease)], "native_poses",
                gpu=True, lease=lease, progress=root / "pilot_progress.json", receipt=root / "poses_01_release_receipt.json", kind="poses")
        self.event("identity_review_required_before_fitting", root, pose=binding(pose))
        while not reviewed_identity_ready(root):
            self.state.update(state="waiting_for_actual_reviewed_identity", root=str(root))
            self.wait()
        if (root / "fit_reuse_audit.json").exists() and not (root / "fit_reuse_applied_receipt.json").exists():
            self.run(root, [sys.executable, "-u", str(PROJ_ROOT / "tools/apply_badminton_full_fit_reuse.py"),
                           "--root", str(root)], "audited_fit_import_cpu")
        from run_badminton_full_fits import fit_inputs
        fit_inputs(root)  # Complete current poses, reviewed identity and all teacher hashes before acquiring CUDA.
        self.batches(root, "fit")
        aggregate = root / "full_aggregate_progress.json"
        if not aggregate.exists() or json.loads(aggregate.read_bytes()).get("state") != "complete_cpu_aggregate":
            self.run(root, self.command(root, "aggregate"), "full_aggregate_cpu")
        if json.loads((root / "pilot_progress.json").read_bytes()).get("stage") != "verify_complete":
            self.run(root, self.command(root, "verify"), "full_native_verify_cpu")
        viewer = self.spawn(self.command(root, "viewer") + ["--port", str(port)], root / "logs/viewer.log")
        timeout = time.monotonic() + 180
        while True:
            if viewer.poll() is not None:
                raise RuntimeError("Verified full viewer exited; inspect its preserved log")
            try:
                with urlopen(f"http://127.0.0.1:{port}/api/manifest", timeout=2) as response:
                    if response.status == 200: break
            except Exception:
                if time.monotonic() > timeout: raise TimeoutError("Full viewer did not become ready")
                self.wait()
        self.event("native_verified_viewer_ready_pending_visual_quality_review", root,
                   viewer=f"http://127.0.0.1:{port}/", viewer_cli_pid=viewer.pid,
                   quality_visual_review_completed=False, training_published=False)

    def execute(self):
        self.event("sequence_receiver_started")
        while self.authorization is None:
            try:
                self.authorization = read_continuous_grant(self.request, self.grant_path)
            except (FileNotFoundError, ValueError):
                self.wait()
        self.state["grant"] = self.authorization["binding"]
        first_release = Path(json.loads(self.request.read_bytes())["scope"][0]["root"]) / "teacher_batch_02_release_receipt.json"
        while not first_release.exists():
            self.state["state"] = "waiting_for_1019_B02_real_cuda_release"
            self.wait()
        if json.loads(first_release.read_bytes()).get("status") != "released":
            raise ValueError("B02 did not release its actual CUDA lease")
        for index, item in enumerate(json.loads(self.request.read_bytes())["scope"]):
            self.cohort(Path(item["root"]), 8784 + index)
        self.state.update(state="all_compute_and_native_verification_complete_pending_visual_quality_review")
        self.event("all_compute_complete_quality_review_still_required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--grant", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--paired-native-rgb", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to(PROJ_ROOT / "outputs") or args.threads < 1:
        parser.error("Use an independent local output and positive threads")
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    psutil.Process().cpu_affinity([0, 1])
    sequence = Sequence(args.request.resolve(), args.grant.resolve(), directory, args.threads,
                        paired_native_rgb=args.paired_native_rgb)
    try:
        sequence.execute()
    except Exception as error:
        if sequence.gpu_root is not None:
            stop_owned_cuda_trees([sequence.gpu_root])
            if sequence.active_progress is not None:
                path = sequence.active_progress
                progress = json.loads(path.read_bytes()) if path.exists() else {}
                progress.update(state="failed", stage="failed", safe_to_release_gpu=True,
                                exit_code=sequence.child.poll() if sequence.child else None,
                                continuous_sequence_error=dict(type=type(error).__name__, message=str(error)))
                write_progress_json(path, progress)
        sequence.state.update(state="failed_waiting_actual_review", error_type=type(error).__name__, error=str(error))
        sequence.event("sequence_failed_requires_actual_review", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
