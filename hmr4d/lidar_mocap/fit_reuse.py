"""Prove unchanged fit execution despite explicit snapshot-scope metadata."""

import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from hmr4d import PROJ_ROOT


FIT_MODULES = ["hmr4d/lidar_mocap/" + name + ".py" for name in (
    "pipeline", "fitting", "segmentation", "export", "calibration", "sessions", "multiview",
    "pose_provider", "segment_cache")]
FIT_MODULES.append("hmr4d/utils/body_model/contracts.py")


def validate_calibration_reuse(previous, current):
    metadata = {"frozen_utc", "artifacts", "snapshot_digest", "data_relocations", "review_source_ids", "sessions"}
    if ({k: v for k, v in previous.items() if k not in metadata}
            != {k: v for k, v in current.items() if k not in metadata}):
        raise ValueError("Fit reuse changed numerical camera/court/ground/clock or admission inputs")
    for sid, value in previous["sessions"].items():
        if current["sessions"].get(sid) != value:
            raise ValueError("Fit reuse changed an original native session contract")


def calibration_execution_ast(blob):
    tree = ast.parse(blob)
    tree.body = [node for node in tree.body
                 if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_same_inputs")]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CalibrationSnapshot":
            node.body = [method for method in node.body
                         if not (isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)) and method.name == "freeze")]
    return ast.dump(tree, include_attributes=False).encode()


def validate_audited_fit_execution(audit, current_digest):
    if (any(item.get("current_fit_algorithm_digest") != current_digest for item in audit["windows"].values())
            or any(item.get("current_digest") != current_digest for item in audit["algorithms"].values())):
        raise ValueError("Fit execution changed since its reuse audit; repeat the compatibility/input audit before import")


def audit_algorithm_compatibility(original_digest, current_digest, original_commit=None):
    current = {name: (PROJ_ROOT / name).read_bytes().replace(b"\r\n", b"\n") for name in FIT_MODULES}
    if hashlib.sha256(b"".join(current.values())).hexdigest() != current_digest:
        raise ValueError("Current fit execution module list differs from its declared digest")
    if original_digest == current_digest:
        return dict(method="identical_normalized_execution_digest", original_digest=original_digest,
                    current_digest=current_digest, compatible=True)
    if not original_commit:
        raise ValueError("Changed fit digest needs explicit original source revision, never an assumed equivalent version")
    commit = subprocess.check_output(["git", "rev-parse", "--verify", original_commit + "^{commit}"],
                                     cwd=PROJ_ROOT, text=True).strip()
    previous = {name: subprocess.check_output(["git", "show", commit + ":" + name], cwd=PROJ_ROOT)
                .replace(b"\r\n", b"\n") for name in FIT_MODULES}
    if hashlib.sha256(b"".join(previous.values())).hexdigest() != original_digest:
        raise ValueError("Original source revision does not reproduce the completed fit digest")
    changed = [name for name in FIT_MODULES if previous[name] != current[name]]
    if changed != ["hmr4d/lidar_mocap/calibration.py"]:
        raise ValueError("Fit execution changed beyond immutable snapshot freezing; recompute these windows")
    name = changed[0]
    old_ast, new_ast = calibration_execution_ast(previous[name]), calibration_execution_ast(current[name])
    if old_ast != new_ast:
        raise ValueError("Calibration execution outside snapshot freezing changed; recompute these windows")
    return dict(
        compatible=True, method="identical_execution_except_snapshot_freeze_and_input_metadata_check",
        original_digest=original_digest, current_digest=current_digest, original_commit=commit,
        calibration_execution_ast_sha256=hashlib.sha256(new_ast).hexdigest(),
        normalized_module_sha256={name: dict(original=hashlib.sha256(previous[name]).hexdigest(),
                                           current=hashlib.sha256(current[name]).hexdigest()) for name in FIT_MODULES},
        required_input_proof="All numerical camera/court/ground/clock/admission fields, original session contracts, window settings, source points, native observations and fitting configuration must also match",
    )


def audit_full_fit_reuse(root, original_commits):
    from .calibration import CalibrationSnapshot
    from .dual_view import file_sha
    from .fitting import FittingConfig
    from .full_native import validate_window_reuse
    from .pilot import binding, check_binding, validate_full_native_spec
    from .pipeline import BadmintonPipeline, write_json
    from .rally_pipeline import load_run_session, read_json, read_rows
    from .relocations import load_pilot_run_relocations

    root = Path(root).resolve()
    if not root.is_relative_to(PROJ_ROOT / "outputs") or root == PROJ_ROOT / "outputs":
        raise ValueError("Use the independently prepared full-native output")
    frozen = read_json(root / "pilot_input.json")
    check_binding(frozen["manifest"])
    validate_full_native_spec(read_json(frozen["manifest"]["path"]))
    source = root / "source"
    calibration = CalibrationSnapshot.load(source / "calibration/calibration_snapshot.json")
    resolver = load_pilot_run_relocations(source, calibration.data)
    teachers = read_json(root / "teacher_reuse_receipt.json")
    check_binding(teachers["source_input"])
    pose_receipt = read_json(root / "native_pose_reuse_receipt.json")
    if pose_receipt["manifest"] != frozen["manifest"]:
        raise ValueError("Native pose reuse belongs to another full run")
    poses = {str(Path(r["source"]["path"]).resolve()): r for r in pose_receipt["records"]}
    plan = read_json(root / "repair/new_source_input.json")
    jobs = {(j["source_id"], Path(j["source_window"]).name): j for j in plan["jobs"]}
    expected_config = asdict(FittingConfig(distal_pose_prior_weight=.03, lower_limb_reprojection_weight=2))
    current_digest = BadmintonPipeline._fit_digest(None)
    sessions, previous_calibrations, algorithms, checked_poses, checked_clouds = {}, {}, {}, set(), {}
    results = {}
    for key, teacher in teachers["windows"].items():
        sid, name = key.split(":", 1)
        previous_root = Path(teacher["previous_root"])
        old_window = previous_root / "repair/final_labels/sessions" / sid / "windows" / name
        old_initial = previous_root / "repair/initial_teacher/sessions" / sid / "windows" / name
        current_window = Path(jobs[sid, name]["source_window"])
        validate_window_reuse(read_json(old_initial / "window.json"), read_json(current_window / "window.json"))
        check_binding(teacher["native_verification"])
        if read_json(teacher["native_verification"]["path"]).get("passed") is not True:
            raise ValueError("Original reused fit lacks complete native verification")
        old_cal = previous_calibrations.setdefault(str(previous_root), CalibrationSnapshot.load(
            previous_root / "repair/final_labels/calibration/calibration_snapshot.json"))
        validate_calibration_reuse(old_cal.data, calibration.data)
        report = read_json(old_window / "fit_report.json")
        if report["fitting_config"] != expected_config:
            raise ValueError("Original fitted window uses another body-fitting policy")
        digest = report["fit_algorithm_digest"]
        if digest not in algorithms:
            commit = original_commits.get(str(previous_root))
            algorithms[digest] = audit_algorithm_compatibility(digest, current_digest, commit)
        for relative, expected in report["artifacts_sha256"].items():
            if file_sha(old_window / relative) != expected:
                raise ValueError("Original fitted/QA/export artifact changed")
        if sid not in sessions:
            sessions[sid] = load_run_session(source, calibration, sid, relocations=resolver)
        pairs = sessions[sid].pairs()
        rows = read_rows(old_window / "quality.jsonl")
        if len(rows) != report["fitted"] or sum(r["accepted"] for r in rows) != report["accepted"]:
            raise ValueError("Original fit/QA counts disagree")
        selected = set(read_json(current_window / "window.json")["source_pairs"])
        for row in rows:
            scan = row["lidar_index"]
            if (scan not in selected or row["scan_mid_us"] != pairs[scan]["scan_mid_us"]
                    or row["cameras"] != pairs[scan]["cameras"]
                    or row["calibration_digest"] != old_cal.data["snapshot_digest"]):
                raise ValueError("Reused fit has different scan/native pairing or camera scope")
            previous_cloud = row["lidar_path"]
            current_cloud = pairs[scan]["lidar_path"]
            for path in (previous_cloud, current_cloud):
                if path not in checked_clouds:
                    checked_clouds[path] = file_sha(Path(path))
            if (checked_clouds[previous_cloud] != report["source_lidar_sha256"][previous_cloud]
                    or checked_clouds[current_cloud] != checked_clouds[previous_cloud]):
                raise ValueError("Reused native source point cloud changed")
            for entry in row["native_pose_evidence"]["files"].values():
                path = str(Path(entry["path"]).resolve())
                if path not in poses or poses[path]["source"]["sha256"] != entry["sha256"]:
                    raise ValueError("Reused fit's exact pose observation was not rebound from its original evidence")
                if path not in checked_poses:
                    item = poses[path]
                    check_binding(item["source"])
                    check_binding(item["target"])
                    old_record = read_json(item["source"]["path"])
                    new_record = read_json(item["target"]["path"])
                    new_record.pop("reuse_native_evidence")
                    new_record["config_digest"] = old_record["config_digest"]
                    if new_record != old_record:
                        raise ValueError("Native observations changed across full-scope metadata rebinding")
                    checked_poses.add(path)
        results[key] = dict(previous_root=str(previous_root), report=binding(old_window / "fit_report.json"),
            old_window=str(old_window), target_window=str(root / "repair/final_labels/sessions" / sid / "windows" / name),
            source_window=str(current_window), old_initial_window=str(old_initial),
            fitted=report["fitted"], accepted=report["accepted"], original_fit_algorithm_digest=digest,
            current_fit_algorithm_digest=current_digest, eligible=True,
            source_identity_decisions=binding(previous_root / "repair/identity_evidence/identity_decisions.json"))
    result = dict(schema="badminton-full-native-fit-reuse-audit-v1", state="eligible_pending_current_identity_and_pose_completion",
        review_only=True, training_published=False, manifest=frozen["manifest"],
        teacher_reuse=binding(root / "teacher_reuse_receipt.json"), native_pose_reuse=binding(root / "native_pose_reuse_receipt.json"),
        algorithms=algorithms, windows=results, eligible_windows=len(results),
        eligible_fitted_person_scans=sum(r["fitted"] for r in results.values()),
        eligible_qa_accepted=sum(r["accepted"] for r in results.values()),
        exact_native_pose_files_checked=len(checked_poses), source_cloud_files_checked=len(checked_clouds),
        policy="No fitting replay or QA changes claimed: this audit only proves eligibility. Actual import needs completed current poses, reviewed matching local identity operations, immutable numerical teacher outputs and copied SMPL tensors.")
    write_json(root / "fit_reuse_audit.json", result)
    return result


def same_teacher_values(before, after):
    """Require exact values/dtypes outside the explicitly replaced provenance."""
    import numpy as np

    def equal(a, b):
        if hasattr(a, "detach") or isinstance(a, np.ndarray):
            a = a.detach().cpu().numpy() if hasattr(a, "detach") else a
            b = b.detach().cpu().numpy() if hasattr(b, "detach") else b
            return (isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
                    and a.tobytes() == b.tobytes())
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return type(a) is type(b) and a == b

    ignored = {"provenance", "provenance_hash", "temporal_reuse_from"}
    return equal({k: v for k, v in before.items() if k not in ignored},
                 {k: v for k, v in after.items() if k not in ignored})


def rebind_fit_row(row, current_digest, target_window, pose_bindings, teacher_hash):
    result = deepcopy(row)
    result["calibration_digest"] = current_digest
    result["fit_path"] = str(Path(target_window) / "fits" / Path(row["fit_path"]).name)
    result["teacher_sampling"]["provenance_hash"] = teacher_hash
    for camera, item in row["native_pose_evidence"]["files"].items():
        source = str(Path(item["path"]).resolve())
        if source not in pose_bindings or pose_bindings[source]["source"]["sha256"] != item["sha256"]:
            raise ValueError("Reused fit row has no exact original native observation binding")
        result["native_pose_evidence"]["files"][camera] = pose_bindings[source]["target"]
    return result


def apply_full_fit_reuse(root):
    """Copy proven numerical results after full current pose/identity completion."""
    from .calibration import CalibrationSnapshot
    from .dual_view import file_sha
    from .pilot import binding, check_binding, validate_full_native_spec
    from .pipeline import BadmintonPipeline, write_json
    from .pose_ray_filter import RayFilteredPoseProvider
    from .rally_pipeline import read_json, read_rows
    from .teacher import load_teacher_track, run_teacher

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Fit reuse import must run in an explicitly CPU-only process")
    root = Path(root).resolve()
    if not root.is_relative_to(PROJ_ROOT / "outputs") or root == PROJ_ROOT / "outputs":
        raise ValueError("Use the independently prepared full-native output")
    frozen = read_json(root / "pilot_input.json")
    check_binding(frozen["manifest"])
    validate_full_native_spec(read_json(frozen["manifest"]["path"]))
    repair, labels = root / "repair", root / "repair/final_labels"
    audit_path = root / "fit_reuse_audit.json"
    audit = read_json(audit_path)
    if audit["manifest"] != frozen["manifest"]:
        raise ValueError("Fit reuse audit belongs to another full execution")
    validate_audited_fit_execution(audit, BadmintonPipeline._fit_digest(None))
    check_binding(audit["teacher_reuse"])
    check_binding(audit["native_pose_reuse"])
    calibration = CalibrationSnapshot.load(labels / "calibration/calibration_snapshot.json")
    identity_path = repair / "identity_evidence/identity_decisions.json"
    if not identity_path.is_file() or not (repair / "pose_receipt.json").is_file():
        raise ValueError("Complete current full-run native poses and reviewed identity evidence before fit reuse import")
    identity = read_json(identity_path)
    if (identity.get("review_status") != "reviewed" or identity["calibration_digest"] != calibration.data["snapshot_digest"]
            or identity["source_union_sha256"] != file_sha(repair / "new_source_input.json")
            or not identity.get("evidence_files")):
        raise ValueError("Full current identity decisions remain pending")
    for path, expected in identity["evidence_files"].items():
        check_binding(dict(path=path, sha256=expected))
    provider = RayFilteredPoseProvider(repair, calibration)
    poses = read_json(audit["native_pose_reuse"]["path"])
    current_pose_receipt = read_json(repair / "pose_receipt.json")
    if current_pose_receipt["config"] != poses["config"]:
        raise ValueError("Full completed pose inference changed the audited reuse configuration")
    pose_bindings = {str(Path(r["source"]["path"]).resolve()): r for r in poses["records"]}
    applied = {}
    for key, entry in audit["windows"].items():
        check_binding(entry["report"])
        check_binding(entry["source_identity_decisions"])
        original, source, target = map(Path, (entry["old_window"], entry["source_window"], entry["target_window"]))
        old_report = read_json(entry["report"]["path"])
        old_identity = read_json(entry["source_identity_decisions"]["path"])["windows"][key]
        current_identity = identity["windows"][key]
        if any(old_identity.get(k) != current_identity.get(k) for k in
               ("source_teacher_manifest_sha256", "links", "cannot_link", "drop_spans")):
            raise ValueError("Current local identity operations differ; this window needs fresh fitting")
        if (target / "fit_report.json").exists():
            existing = read_json(target / "fit_report.json")
            if existing.get("scope_reuse", {}).get("original_report") != entry["report"]:
                raise ValueError("Fit reuse cannot overwrite an independently fitted current window")
        initial = read_json(source / "teacher/manifest.json")
        if file_sha(source / "teacher/manifest.json") != current_identity["source_teacher_manifest_sha256"]:
            raise ValueError("Audited initial teacher changed before fit reuse")
        provenance = initial["provenance"]
        options = dict(provenance["options"], max_track_gap_frames=6)
        teacher = run_teacher(provenance["video"]["path"], provenance["frame_rows"], provenance["camera"],
            target / "teacher", reuse_preprocess_dir=original / "teacher", identity_decisions=current_identity,
            resume=True, **options)
        old_teacher = read_json(original / "teacher/manifest.json")
        if old_teacher["tracks"] != teacher["tracks"]:
            raise ValueError("Reused repaired teacher has different local track coverage")
        for track in teacher["tracks"]:
            before = load_teacher_track(original / "teacher" / track["path"])
            after = load_teacher_track(target / "teacher" / track["path"])
            if not same_teacher_values(before, after):
                raise ValueError("Repaired teacher numerical values changed; fresh fitting is required")
        config = read_json(source / "window.json")
        settings = dict(config["settings"], share_track_shape=True,
            native_pose_receipt_sha256=provider.digest, native_pose_joint_ray_policy_sha256=provider.policy_sha256)
        if ({k: v for k, v in old_report["settings"].items() if k != "native_pose_receipt_sha256"}
                != {k: v for k, v in settings.items() if k != "native_pose_receipt_sha256"}):
            raise ValueError("Audited body-fitting settings changed")
        numeric = {}
        for relative, expected in old_report["artifacts_sha256"].items():
            if relative in ("quality.jsonl", "ptv3.metadata.jsonl"):
                continue
            old_path, destination = original / relative, target / relative
            if file_sha(old_path) != expected:
                raise ValueError("Original fitted numerical artifact changed")
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old_path, destination)
            if file_sha(destination) != expected:
                raise ValueError("Current fit numerical artifact differs from its original")
            numeric[relative] = expected
        for name in ("quality.jsonl", "ptv3.metadata.jsonl"):
            rows = [rebind_fit_row(row, calibration.data["snapshot_digest"], target, pose_bindings,
                                   teacher["provenance_hash"]) for row in read_rows(original / name)]
            (target / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        write_json(target / "window.json", dict(config, settings=settings,
            body_repair=dict(source_window=str(source), source_union_sha256=identity["source_union_sha256"])))
        report = deepcopy(old_report)
        report.update(fit_algorithm_digest=entry["current_fit_algorithm_digest"], settings=settings,
            teacher_provenance_hash=teacher["provenance_hash"], native_pose_receipt_sha256=provider.digest,
            scope_reuse=dict(original_report=entry["report"], original_fit_algorithm_digest=entry["original_fit_algorithm_digest"],
                compatibility_audit=binding(audit_path), identity_decisions=binding(identity_path),
                numeric_fit_recomputed=False, numerical_artifacts_byte_identical=True,
                original_review_rendering_retained=True))
        report["export"].update(path=str(target / "ptv3.pth"), metadata=str(target / "ptv3.metadata.jsonl"))
        report["artifacts_sha256"] = {name: file_sha(target / name) for name in old_report["artifacts_sha256"]}
        write_json(target / "fit_report.json", report)
        applied[key] = dict(original=entry["report"], current=binding(target / "fit_report.json"),
            unchanged_numerical_artifacts_sha256=numeric, fitted=report["fitted"], accepted=report["accepted"])
    result = dict(schema="badminton-full-native-applied-fit-reuse-v1", review_only=True, training_published=False,
        audit=binding(audit_path), completed_current_pose_receipt=binding(repair / "pose_receipt.json"),
        reviewed_current_identity=binding(identity_path), windows=applied, reused_windows=len(applied),
        reused_fitted_person_scans=sum(v["fitted"] for v in applied.values()),
        reused_qa_accepted=sum(v["accepted"] for v in applied.values()),
        numeric_fit_recomputed=False, qa_modified=False, full_current_native_verification_pending=True)
    write_json(root / "fit_reuse_applied_receipt.json", result)
    return result
