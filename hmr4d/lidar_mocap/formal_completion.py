"""CPU-only bridge from a genuine pre-leg freeze to post-leg native QA."""

import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

import psutil

from hmr4d import PROJ_ROOT
from .formal_baseline import _artifacts, _binding, _bound, _count, _encode, _preflight_target, _publish, _scope, _sha

STATE = "postleg_native_qa_complete_bilateral_browser_archive_pending"
METRICS = ("final_p2surface_mean_m", "final_p2surface_p90_m", "reprojection_mean_px")


def _load(path, observed):
    return _bound(_binding(path), observed)[0]


def _exact_path(item, path):
    if Path(item["path"]).resolve() != path.resolve():
        raise ValueError("Evidence leaves its exact formal target")


def _unchanged(observed):
    for path, sha in observed.items():
        if _sha(path) != sha:
            raise ValueError("Formal CPU input changed during verification")


def _qa_scope(qa, keys, source_sha, calibration, reports):
    if (
        qa.get("schema") != "badminton-native-absolute-qa-v1"
        or set(qa.get("windows", {})) != keys
        or qa.get("source_union_sha256") != source_sha
        or qa.get("calibration_digest") != calibration
    ):
        raise ValueError("Absolute QA source/camera/window coverage differs")
    counts = dict.fromkeys(keys, 0)
    for key, values in qa.get("frames", {}).items():
        parts = key.split("/", 2)
        window_key = ":".join(parts[:2])
        if len(parts) != 3 or not parts[2] or window_key not in keys or set(values) != set(METRICS):
            raise ValueError("Absolute QA has an unplanned person frame")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values()):
            raise ValueError("Absolute QA values must be finite actual numbers")
        counts[window_key] += 1
    for key, entry in qa["windows"].items():
        if (
            _count(entry.get("accepted"), "QA accepted") != counts[key]
            or entry.get("fit_report_sha256") != reports[key]["sha256"]
        ):
            raise ValueError("Absolute QA per-window count/report does not match real frames")
    return sum(counts.values())


def _baseline(root, baseline_binding, *, preleg):
    root = Path(root).resolve()
    outputs = (PROJ_ROOT / "outputs").resolve()
    if root == outputs or not root.is_relative_to(outputs):
        raise ValueError("Select an explicit formal run inside project outputs")
    repair, observed = root / "repair", {}
    if not repair.resolve().is_relative_to(root):
        raise ValueError("Formal repair target leaves its exact run")
    folder = root / "gpu_guarded_leg/baseline"
    if not folder.resolve().is_relative_to(root):
        raise ValueError("Baseline leaves its exact formal run")
    _exact_path(baseline_binding, folder / "baseline.json")
    value, _ = _bound(baseline_binding, observed)
    if (
        value.get("schema") != "gvhmr-formal-preleg-baseline-v1"
        or value.get("state") != "complete"
        or value.get("training_published") is not False
        or value.get("gpu_started") is not False
    ):
        raise ValueError("A genuine completed formal pre-leg baseline is required")
    _exact_path(value["source_input"], repair / "new_source_input.json")
    source, _ = _bound(value["source_input"], observed)
    _exact_path(value["plan"], repair / "final_labels/repair_plan.json")
    plan, _ = _bound(value["plan"], observed)
    source_sha, calibration = value.get("source_union_sha256"), value.get("calibration_digest")
    keys = set(_scope(plan["jobs"]))
    if (
        source_sha != value["source_input"]["sha256"]
        or plan.get("source_union_sha256") != source_sha
        or plan.get("source_union") != value["source_input"]["path"]
        or not re.fullmatch(r"[0-9a-f]{64}", str(calibration))
        or plan.get("calibration_digest") != calibration
        or source.get("schema") != "badminton-new-source-repair-input-v1"
        or source.get("calibration_digest") != calibration
        or _scope(source["jobs"]) != _scope(plan["jobs"])
        or set(value.get("windows", {})) != keys
    ):
        raise ValueError("Baseline source, camera and full actual body plan differ")
    snapshots = {}
    for role in ("body_receipt", "pose_receipt", "native_verification", "reviewed_identity", "absolute_qa"):
        if not Path(value[role]["path"]).resolve().is_relative_to(folder.resolve()):
            raise ValueError("Genuine pre-leg receipts must be separate baseline snapshots")
        snapshots[role], _ = _bound(value[role], observed)
    body, pose, native, identity, qa = [
        snapshots[k]
        for k in ("body_receipt", "pose_receipt", "native_verification", "reviewed_identity", "absolute_qa")
    ]
    reports = {}
    for key, info in value["windows"].items():
        sid, name = key.split(":", 1)
        item = info["report_snapshot"]
        _exact_path(item, folder / "reports" / sid / name / "fit_report.json")
        report, report_bytes = _bound(item, observed)
        reports[key] = item
        if _count(report.get("accepted"), "frozen report accepted") != qa["windows"][key]["accepted"]:
            raise ValueError("Frozen report/absolute QA accepted count differs")
        if preleg:
            window = repair / "final_labels/sessions" / sid / "windows" / name
            actual, actual_bytes = _bound(_binding(window / "fit_report.json"), observed)
            if (
                actual_bytes != report_bytes
                or actual != report
                or os.path.samefile(item["path"], window / "fit_report.json")
            ):
                raise ValueError("Current pre-leg report differs from its independent frozen snapshot")
            _artifacts(actual, window, observed)
    accepted = _qa_scope(qa, keys, source_sha, calibration, reports)
    if (
        body.get("state") != "body_qa_candidate_complete"
        or _count(body.get("windows"), "body windows") != len(keys)
        or _count(body.get("accepted"), "body accepted") != accepted
        or native.get("passed") is not True
        or native.get("state") != "complete"
        or set(native.get("windows", {})) != keys
        or native.get("verified_windows") != len(keys)
        or native.get("total_windows") != len(keys)
        or native.get("accepted_checked") != accepted
        or native.get("source_union_sha256") != source_sha
        or native.get("calibration_digest") != calibration
    ):
        raise ValueError("Frozen actual body/native completion is incomplete")
    for key in keys:
        checked = native["windows"][key]["body"]
        if (
            checked.get("report_sha256") != reports[key]["sha256"]
            or checked.get("accepted_checked") != qa["windows"][key]["accepted"]
        ):
            raise ValueError("Frozen native check names another per-window report/count")
    if (
        pose.get("state") != "complete"
        or pose.get("source_union_sha256") != source_sha
        or pose.get("config", {}).get("calibration_digest") != calibration
        or identity.get("review_status") != "reviewed"
        or set(identity.get("windows", {})) != keys
        or identity.get("source_union_sha256") != source_sha
        or identity.get("calibration_digest") != calibration
        or not identity.get("evidence_files")
    ):
        raise ValueError("Frozen native pose/identity inputs are incomplete")
    for path, sha in identity["evidence_files"].items():
        if _sha(path) != sha:
            raise ValueError("Frozen reviewed identity witness changed")
        observed[str(Path(path).resolve())] = sha
    return dict(
        root=root,
        repair=repair,
        folder=folder,
        baseline=value,
        baseline_binding=baseline_binding,
        plan=plan,
        keys=keys,
        before=qa,
        accepted=accepted,
        source_sha=source_sha,
        calibration=calibration,
        observed=observed,
    )


def prepare_formal_leg_completion(formal_root, baseline_binding):
    """Install the byte-identical genuine before-QA file before any leg output."""
    repair = Path(formal_root).resolve() / "repair"
    if any(
        path.exists() for path in (repair / "final_labels/post_refinement.json", repair / "leg_refinement_receipt.json")
    ):
        raise ValueError("Cannot install or reconstruct before-QA after leg refinement started")
    checked = _baseline(formal_root, baseline_binding, preleg=True)
    payload = Path(checked["baseline"]["absolute_qa"]["path"]).read_bytes()
    destination = repair / "evaluation/v2_before_leg.json"
    if not destination.resolve().is_relative_to(repair):
        raise ValueError("Before-QA destination leaves its exact repair run")
    _preflight_target(destination, payload, (repair / "evaluation").resolve())
    _unchanged(checked["observed"])
    if any(
        path.exists() for path in (repair / "final_labels/post_refinement.json", repair / "leg_refinement_receipt.json")
    ):
        raise ValueError("Leg refinement started while preparing genuine before-QA")
    _publish(destination, payload)
    return dict(
        state="formal_preleg_qa_installed",
        baseline=baseline_binding,
        before_qa=_binding(destination),
        windows=len(checked["keys"]),
        accepted=checked["accepted"],
        gpu_started=False,
        training_published=False,
    )


def _gone(identity):
    if (
        type(identity.get("pid")) is not int
        or identity["pid"] <= 0
        or type(identity.get("create_time")) not in (int, float)
        or not math.isfinite(identity["create_time"])
        or identity["create_time"] <= 0
    ):
        raise ValueError("Actual process PID/create_time evidence is required")
    try:
        if psutil.Process(identity["pid"]).create_time() == identity["create_time"]:
            raise ValueError("An observed guarded-leg process is still alive")
    except psutil.NoSuchProcess:
        pass


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
    if not re.fullmatch(r"[0-9a-f]{64}", str(proposals.get("policy_sha256", ""))):
        raise ValueError("Actual leg policy hash is missing")
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
        or applied.get("policy_sha256") != proposals.get("policy_sha256")
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
            ):
                raise ValueError("Actual updated fit report differs from the apply output")
        elif _sha(report_path) != checked["baseline"]["windows"][key]["report_snapshot"]["sha256"]:
            raise ValueError("An unselected formal fit window changed")


CPU_BOOTSTRAP = """import json,os,psutil,runpy,sys
p=psutil.Process();p.cpu_affinity([26,27]);p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
assert p.cpu_affinity()==[26,27] and p.nice()==psutil.BELOW_NORMAL_PRIORITY_CLASS
assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
print(json.dumps(dict(schema='gvhmr-formal-cpu-worker-v1',affinity=p.cpu_affinity(),priority='BelowNormal',cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])),flush=True)
entry=sys.argv[1];sys.argv=sys.argv[1:];runpy.run_path(entry,run_name='__main__')
"""


def _run_cpu_check(checked, role, directory, threads):
    repair, labels = checked["repair"], checked["repair"] / "final_labels"
    name, args = {
        "native": ("verify_badminton_native_repair.py", ["--root", str(repair)]),
        "dataset": ("verify_badminton_dataset.py", ["--output-root", str(labels), "--samples-per-session", "64"]),
    }[role]
    entry = PROJ_ROOT / "tools" / name
    entry_binding = _binding(entry)
    command = [
        str(PROJ_ROOT / ".venv/Scripts/python.exe"),
        "-u",
        "-c",
        CPU_BOOTSTRAP,
        str(entry),
        *args,
        "--threads",
        str(threads),
    ]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="-1",
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS="1",
    )
    log = directory / (role + ".log")
    started = time.perf_counter()
    with log.open("xb") as handle:
        completed = subprocess.run(
            command,
            cwd=PROJ_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Formal CPU {role} verification failed ({completed.returncode}); see {log}")
    if _sha(entry) != entry_binding["sha256"]:
        raise ValueError("Actual CPU verifier code changed during execution")
    with log.open("rb") as handle:
        allocation = json.loads(handle.readline())
    if allocation != dict(
        schema="gvhmr-formal-cpu-worker-v1", affinity=[26, 27], priority="BelowNormal", cuda_visible_devices="-1"
    ):
        raise RuntimeError("Actual CPU worker allocation was not verified")
    return dict(
        exit_code=0,
        command=command,
        entry=entry_binding,
        log=_binding(log),
        wall_seconds=time.perf_counter() - started,
        allocation=allocation,
        hidden_window=True,
    )


def _native_after(checked):
    labels = checked["repair"] / "final_labels"
    native = _load(labels / "native_verification.json", {})
    if (
        native.get("passed") is not True
        or native.get("state") != "complete"
        or set(native.get("windows", {})) != checked["keys"]
        or native.get("verified_windows") != len(checked["keys"])
        or native.get("total_windows") != len(checked["keys"])
        or native.get("accepted_checked") != checked["accepted"]
        or native.get("source_union_sha256") != checked["source_sha"]
        or native.get("calibration_digest") != checked["calibration"]
    ):
        raise ValueError("Actual post-leg native verification is incomplete")
    for key in checked["keys"]:
        sid, name = key.split(":", 1)
        body = native["windows"][key]["body"]
        if (
            body.get("report_sha256") != _sha(labels / "sessions" / sid / "windows" / name / "fit_report.json")
            or body.get("accepted_checked") != checked["before"]["windows"][key]["accepted"]
        ):
            raise ValueError("Post-leg native per-window report/count differs")
    return native


def _dataset_after(checked):
    labels = checked["repair"] / "final_labels"
    manifest = _load(labels / "datasets/dataset_manifest.json", {})
    report = _load(labels / "datasets/verification.json", {})
    counts = {}
    for key, value in checked["before"]["windows"].items():
        sid = key.split(":", 1)[0]
        counts[sid] = counts.get(sid, 0) + value["accepted"]
    if set(manifest.get("exports", {})) != set(counts) or manifest.get("accepted") != checked["accepted"]:
        raise ValueError("Actual post-leg dataset source/accepted inventory differs")
    if report.get("calibration_digest") != checked["calibration"] or set(report.get("sessions", {})) != {
        sid for sid, count in counts.items() if count
    }:
        raise ValueError("Actual dataset verification is missing checked sessions or uses another camera")
    for sid, expected in counts.items():
        if manifest["exports"][sid].get("accepted") != expected:
            raise ValueError("Actual dataset per-source count differs")
        if not expected:
            continue
        entry = report["sessions"][sid]
        if entry.get("exported") != expected or entry.get("checked") != min(64, expected):
            raise ValueError("Actual dataset verification sample coverage differs")
        for name, limit in (
            ("max_source_cloud_error_m", 1e-6),
            ("max_rebuilt_mesh_error_m", 1e-4),
            ("max_pelvis_translation_error_m", 1e-6),
        ):
            value = entry.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
                raise ValueError("Actual dataset native point/SMPL reconstruction failed")
    return report


def verify_formal_postleg_cpu(formal_root, baseline_binding, leg_execution_binding, threads=1):
    """Check historical CUDA exit, then run only the two existing CPU verifiers."""
    if type(threads) is not int or threads not in (1, 2):
        raise ValueError("Formal CPU verification uses one or two threads")
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
