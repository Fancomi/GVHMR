"""CPU execution witnesses for genuinely reconstructed native body windows."""

import builtins
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys
import time
import uuid

import psutil

from hmr4d import PROJ_ROOT
from .formal_baseline import _artifacts, _binding, _bound, _encode, _publish, _scope, _sha
from .repair_execution import read_live_json

SCHEMA = "badminton-native-reconstruction-execution-witness-v1"
SOURCE_FILES = (
    "tools/verify_badminton_native_repair.py",
    "tools/verify_badminton_native_bound.py",
    "hmr4d/lidar_mocap/native_witness.py",
    "hmr4d/utils/body_model/contracts.py",
    "hmr4d/lidar_mocap/calibration.py",
    "hmr4d/lidar_mocap/sessions.py",
    "hmr4d/lidar_mocap/source_index.py",
    "hmr4d/lidar_mocap/source_membership.py",
    "hmr4d/lidar_mocap/rally_pipeline.py",
    "hmr4d/lidar_mocap/relocations.py",
    "hmr4d/lidar_mocap/teacher.py",
    "hmr4d/lidar_mocap/new_source_repair.py",
    "hmr4d/lidar_mocap/identity_repair.py",
    "hmr4d/lidar_mocap/identity_continuation.py",
    "hmr4d/lidar_mocap/windows_gpu_job.py",
    "tools/continue_badminton_native_identity.py",
    "hmr4d/lidar_mocap/pilot.py",
    "hmr4d/lidar_mocap/full_native.py",
    "hmr4d/lidar_mocap/pipeline.py",
    "hmr4d/lidar_mocap/formal_baseline.py",
    "hmr4d/lidar_mocap/repair_execution.py",
    "tools/run_badminton_pilot.py",
    "tools/run_badminton_full_native.py",
    "tools/run_badminton_full_sequence.py",
    "hmr4d/lidar_mocap/formal_native_derive.py",
    "hmr4d/lidar_mocap/formal_completion.py",
    "tools/derive_badminton_formal_native.py",
    "tools/finalize_badminton_formal_cpu.py",
)
CUDA_DLL_MARKERS = (
    "cuda",
    "cublas",
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nvrtc",
    "nvjit",
    "nvfuser",
    "nvtools",
    "nvjpeg",
)


def capture_runtime():
    """Hash actual installed CPU implementation files and the neutral model."""
    files = {str((PROJ_ROOT / name).resolve()): _sha(PROJ_ROOT / name) for name in SOURCE_FILES}
    versions = {}
    for name in ("smplx", "torch", "numpy", "joblib", "scipy", "chumpy", "opencv-python", "einops"):
        distribution = importlib.metadata.distribution(name)
        versions[name] = distribution.version
        for relative in distribution.files or ():
            path = Path(distribution.locate_file(relative)).resolve()
            suffix = path.suffix.lower()
            include = suffix in (".py", ".pyd") or (
                suffix == ".dll"
                and (name != "torch" or not any(part in path.name.lower() for part in CUDA_DLL_MARKERS))
            )
            if include and path.is_file():
                files[str(path)] = _sha(path)
    try:
        optional = importlib.metadata.distribution("opt-einsum")
        versions["opt-einsum"] = optional.version
        for relative in optional.files or ():
            path = Path(optional.locate_file(relative)).resolve()
            if path.suffix == ".py" and path.is_file():
                files[str(path)] = _sha(path)
    except importlib.metadata.PackageNotFoundError:
        versions["opt-einsum"] = "not_installed"
    customized = sys.modules.get("sitecustomize")
    if customized is not None and getattr(customized, "__file__", None):
        path = Path(customized.__file__).resolve()
        files[str(path)] = _sha(path)
    interpreter = Path(psutil.Process().exe()).resolve()
    files[str(interpreter)] = _sha(interpreter)
    python_dll = Path(sys.base_prefix) / "python310.dll"
    if python_dll.exists():
        files[str(python_dll.resolve())] = _sha(python_dll)
    model = (PROJ_ROOT / "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl").resolve()
    files[str(model)] = _sha(model)
    return dict(
        models=dict(neutral_smpl=_binding(model)),
        files=files,
        runtime_versions=dict(versions, python=sys.version),
        device="cpu",
        model_type="smpl",
        gender="neutral",
    )


def validate_inventory(value, *, current=None):
    current = capture_runtime() if current is None else current
    if value != current or value.get("device") != "cpu":
        raise ValueError("Actual native model/code/CPU runtime differs from its execution witness")
    for path, sha in value["files"].items():
        if _sha(path) != sha:
            raise ValueError("A witnessed native execution input changed")
    if value["files"].get(value["models"]["neutral_smpl"]["path"]) != value["models"]["neutral_smpl"]["sha256"]:
        raise ValueError("The actual neutral model lacks its runtime file binding")


def _root(repair):
    repair = Path(repair).resolve()
    outputs = (PROJ_ROOT / "outputs").resolve()
    if repair == outputs or not repair.is_relative_to(outputs) or repair.name != "repair":
        raise ValueError("Use an explicit in-run repair directory")
    return repair


def teacher_fingerprints(repair, job):
    repair = Path(repair)
    source = Path(job["source_window"])
    target = repair / "final_labels/sessions" / job["source_id"] / "windows" / source.name
    paths = [
        repair / "final_labels/repair_plan.json",
        repair / "identity_evidence/identity_decisions.json",
        PROJ_ROOT / "tools/verify_badminton_native_repair.py",
        source / "teacher/detections.pt",
    ]
    for folder in (source, target):
        manifest = folder / "teacher/manifest.json"
        data = read_live_json(manifest)
        paths.append(manifest)
        for entry in data["tracks"]:
            path = (folder / "teacher" / entry["path"]).resolve()
            if not path.is_relative_to((folder / "teacher").resolve()):
                raise ValueError("Native teacher track leaves its actual window")
            paths.append(path)
    return {str(path.resolve()): _sha(path) for path in paths}


def _complete(report, repair, *, current_keys=None):
    plan = read_live_json(repair / "final_labels/repair_plan.json")
    jobs = _scope(plan["jobs"])
    keys = set(jobs)
    if (
        report.get("passed") is not True
        or report.get("state") != "complete"
        or set(report.get("windows", {})) != keys
        or report.get("verified_windows") != len(keys)
        or report.get("total_windows") != len(keys)
        or report.get("source_union_sha256") != plan["source_union_sha256"]
        or report.get("calibration_digest") != plan["calibration_digest"]
    ):
        raise ValueError("Actual native verifier did not complete the full source/window scope")
    selected = keys if current_keys is None else set(current_keys)
    if not selected <= keys:
        raise ValueError("A native witness requests windows outside its actual scope")
    accepted = sum(entry["body"]["accepted_checked"] for entry in report["windows"].values())
    for key, entry in report["windows"].items():
        if key not in selected:
            continue
        sid, name = key.split(":", 1)
        path = repair / "final_labels/sessions" / sid / "windows" / name / "fit_report.json"
        fit = read_live_json(path)
        body = entry["body"]
        if body.get("report_sha256") != _sha(path) or body.get("accepted_checked") != fit.get("accepted"):
            raise ValueError("Actual native reconstruction checked another fit report or count")
        _artifacts(fit, path.parent, {})
        if fit["accepted"] and not fit.get("source_lidar_sha256"):
            raise ValueError("Witnessed accepted rows lack their original point-cloud hashes")
        for filename, sha in fit.get("source_lidar_sha256", {}).items():
            if _sha(filename) != sha:
                raise ValueError("A witnessed source point cloud changed")
        for line in (path.parent / "ptv3.metadata.jsonl").read_bytes().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("lidar_path") not in fit.get("source_lidar_sha256", {}):
                raise ValueError("A witnessed accepted point row has no exact cloud binding")
            if not row.get("native_pose_evidence", {}).get("files"):
                raise ValueError("A witnessed accepted point row lacks native pose files")
            for item in row.get("native_pose_evidence", {}).get("files", {}).values():
                if _sha(item["path"]) != item["sha256"]:
                    raise ValueError("A witnessed native observation changed")
        if (
            not isinstance(entry.get("teacher"), dict)
            or not isinstance(body.get("maximums"), dict)
            or any(type(body.get(name)) is not int or body[name] < 0 for name in ("native_scans", "native_pose_files"))
        ):
            raise ValueError("Actual native result lacks its teacher/body measurement inventory")
        if body["accepted_checked"] and (not body["native_scans"] or not body["native_pose_files"]):
            raise ValueError("Accepted native rows lack their real scan/pose coverage")
        for metric, limit in (("source_point_error_m", 1e-6), ("pelvis_error_m", 1e-6), ("rebuilt_mesh_error_m", 1e-4)):
            if body["accepted_checked"] and metric not in body["maximums"]:
                raise ValueError("An actual reconstruction maximum is missing")
            number = body.get("maximums", {}).get(metric, 0)
            if type(number) not in (int, float) or not 0 <= number <= limit:
                raise ValueError("Actual native reconstruction exceeds unchanged limits")
        if entry.get("teacher_inputs_sha256") != teacher_fingerprints(repair, jobs[key]):
            raise ValueError("Actual native teacher/verifier input inventory changed or is incomplete")
    if report.get("accepted_checked") != accepted:
        raise ValueError("Actual native accepted total differs")
    return keys


def validate_native_witness(item, repair, *, current_runtime=None, require_full=True, window_keys=None):
    """Validate a real prior full execution; never promote a pose-stage hash."""
    repair = _root(repair)
    value, _ = _bound(item, {})
    if (
        value.get("schema") != SCHEMA
        or value.get("state") not in ("actual_complete", "actual_subset_complete")
        or value.get("repair_root") != str(repair)
        or value.get("exit_code") != 0
        or value.get("gpu_started") is not False
        or value.get("runtime_before") != value.get("runtime_after")
    ):
        raise ValueError("A genuine first full native reconstruction witness is required")
    cached = set(value.get("cached_window_keys", []))
    actual = set(value.get("actual_window_keys", []))
    if (
        not actual
        or len(cached) != len(value.get("cached_window_keys", []))
        or len(actual) != len(value.get("actual_window_keys", []))
        or cached & actual
        or value.get("preexisting_checked_windows") != len(cached)
        or value.get("all_windows_reconstructed_in_this_execution") is not (not cached)
        or (require_full and (cached or value.get("state") != "actual_complete"))
    ):
        raise ValueError("Cached historical windows cannot acquire a new actual model witness")
    validate_inventory(value["runtime_before"], current=current_runtime)
    model = value["runtime_before"]["models"]["neutral_smpl"]
    reads = value.get("actual_model_file_reads", [])
    if not reads or any(entry.get("model") != model for entry in reads):
        raise ValueError("The actual native constructor did not witness the bound neutral model")
    begin, end = datetime.fromisoformat(value["started_utc"]), datetime.fromisoformat(value["completed_utc"])
    if end < begin or any(not begin <= datetime.fromisoformat(entry["opened_utc"]) <= end for entry in reads):
        raise ValueError("Actual model reads do not belong to the witnessed execution interval")
    result, _ = _bound(value["native_result_snapshot"], {})
    if (
        require_full
        and _sha(repair / "final_labels/native_verification.json") != value["native_result_snapshot"]["sha256"]
    ):
        raise ValueError("Current parent native result differs from its actual execution snapshot")
    selected = actual if window_keys is None else set(window_keys)
    if not selected <= actual or actual | cached != _complete(result, repair, current_keys=selected):
        raise ValueError("Native execution witness omits original reconstructed windows")
    if not require_full:
        current = read_live_json(repair / "final_labels/native_verification.json")
        for key in selected:
            stripped = lambda entry: {k: v for k, v in entry.items() if k != "verification_origin"}
            if stripped(current["windows"][key]) != stripped(result["windows"][key]):
                raise ValueError("A currently actual window differs from its witnessed reconstruction")
    required_inputs = dict(
        plan=repair / "final_labels/repair_plan.json",
        source=repair / "new_source_input.json",
        identity=repair / "identity_evidence/identity_decisions.json",
        pose=repair / "pose_receipt.json",
        calibration=repair / "final_labels/calibration/calibration_snapshot.json",
        source_registry=repair / "final_labels/source_registry.json",
    )
    if set(value.get("inputs", {})) != set(required_inputs):
        raise ValueError("Native model/geometry/source execution input coverage is incomplete")
    for name, expected in value["inputs"].items():
        if expected != _binding(required_inputs[name]):
            raise ValueError("Original native execution plan/source/identity/pose changed: " + name)
    return value, result


def run_bound_native_verification(repair, threads=1, *, verified_cached_keys=None):
    """Invoke the unchanged verifier, tracing the model file actually opened."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or threads not in (1, 2):
        raise ValueError("Native witnesses require explicit CPU-only execution with one or two threads")
    repair = _root(repair)
    result_path = repair / "final_labels/native_verification.json"
    witness_path = repair / "native_verification_witness.json"
    if witness_path.exists() and verified_cached_keys is None:
        value, _ = validate_native_witness(_binding(witness_path), repair)
        return dict(
            state="reused_actual_bound_native", witness=_binding(witness_path), actual=value["native_result_snapshot"]
        )
    existing = read_live_json(result_path) if result_path.exists() else {}
    cached = len(existing.get("windows", {}))
    attempt = repair / "native_witness_attempts" / uuid.uuid4().hex
    attempt.mkdir(parents=True, exist_ok=False)
    if verified_cached_keys is not None:
        allowed = set(verified_cached_keys)
        if not allowed <= set(existing.get("windows", {})):
            raise ValueError("A verified native cache key is not actually present")
        if result_path.exists():
            _publish(attempt / "prior_native_checkpoint.json", result_path.read_bytes())
        existing["windows"] = {key: entry for key, entry in existing.get("windows", {}).items() if key in allowed}
        existing.update(state="partial", passed=False)
        from .repair_execution import write_progress_json

        write_progress_json(result_path, existing)
        cached = len(allowed)
    else:
        allowed = set(existing.get("windows", {}))
    inputs = {
        name: _binding(path)
        for name, path in dict(
            plan=repair / "final_labels/repair_plan.json",
            source=repair / "new_source_input.json",
            identity=repair / "identity_evidence/identity_decisions.json",
            pose=repair / "pose_receipt.json",
            calibration=repair / "final_labels/calibration/calibration_snapshot.json",
            source_registry=repair / "final_labels/source_registry.json",
        ).items()
    }
    before = capture_runtime()
    _publish(attempt / "before.json", _encode(dict(runtime=before, inputs=inputs, cached_windows=cached)))
    original_open, original_argv = builtins.open, sys.argv
    model_path = Path(before["models"]["neutral_smpl"]["path"])
    reads = []

    def observed_open(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file).resolve() == model_path:
            item = dict(model=_binding(model_path), opened_utc=datetime.now(timezone.utc).isoformat())
            reads.append(item)
            _publish(attempt / ("model_open_" + str(len(reads)) + ".json"), _encode(item))
        return original_open(file, *args, **kwargs)

    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    try:
        builtins.open = observed_open
        sys.argv = [
            str(PROJ_ROOT / "tools/verify_badminton_native_repair.py"),
            "--root",
            str(repair),
            "--threads",
            str(threads),
        ]
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        builtins.open, sys.argv = original_open, original_argv
    after = capture_runtime()
    validate_inventory(before, current=after)
    for item in inputs.values():
        if _sha(item["path"]) != item["sha256"]:
            raise ValueError("Actual native execution inputs changed during verification")
    report = read_live_json(result_path)
    keys = _complete(report, repair)
    if (cached and verified_cached_keys is None) or not reads or not (keys - allowed):
        # The native gate still ran, but cached historical windows cannot gain a
        # new model witness and an unobserved loader cannot be attested.
        fallback = dict(
            state="actual_native_complete_witness_unavailable",
            cached_windows=cached,
            actual_model_read_observed=bool(reads),
            historical_model_witness_backfilled=False,
            native_result=_binding(result_path),
            derived_reuse_allowed=False,
        )
        _publish(attempt / "unwitnessed_fallback.json", _encode(fallback))
        return fallback
    snapshot = attempt / "native_result.json"
    _publish(snapshot, result_path.read_bytes())
    process = psutil.Process()
    value = dict(
        schema=SCHEMA,
        state="actual_complete" if not cached else "actual_subset_complete",
        repair_root=str(repair),
        exit_code=0,
        preexisting_checked_windows=cached,
        all_windows_reconstructed_in_this_execution=not cached,
        cached_window_keys=sorted(allowed),
        actual_window_keys=sorted(keys - allowed),
        runtime_before=before,
        runtime_after=after,
        inputs=inputs,
        actual_model_file_reads=reads,
        native_result_snapshot=_binding(snapshot),
        process=dict(pid=process.pid, create_time=process.create_time(), actual_affinity=process.cpu_affinity()),
        started_utc=started_utc,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        actual_wall_seconds=time.perf_counter() - started,
        gpu_started=False,
        training_published=False,
    )
    unique_witness = attempt / "witness.json"
    _publish(unique_witness, _encode(value))
    if not cached and not witness_path.exists():
        _publish(witness_path, _encode(value))
    target = witness_path if verified_cached_keys is None and not cached else unique_witness
    return dict(
        state="actual_native_complete_witnessed" if not cached else "actual_native_subset_witnessed",
        witness=_binding(target),
        actual=value["native_result_snapshot"],
        actual_window_keys=value["actual_window_keys"],
    )
