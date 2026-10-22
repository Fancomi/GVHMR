"""CPU-only, proof-bound copies of completed bodies into separately admitted inputs."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

from hmr4d import PROJ_ROOT
from .fit_reuse import rebind_fit_row, same_teacher_values, validate_audited_fit_execution
from .full_native import pose_config_digest, rebind_native_pose_record
from .pilot import binding, check_binding
from .rally_pipeline import read_json, read_rows


def _require_cpu():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Formal numerical import requires an explicitly CPU-only process")


def _target(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Formal import target leaves its independent root")
    return path


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _write_once(path, value):
    path = Path(path)
    blob = _json_bytes(value)
    if path.exists():
        if path.read_bytes() != blob:
            raise ValueError("Preserve the existing different formal import output: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(blob)
    return binding(path)


def _copy_once(source, destination):
    check_binding(source)
    destination = Path(destination)
    if destination.exists():
        if os.path.samefile(source["path"], destination):
            raise ValueError("Formal artifacts and snapshots must be independent copies of their original")
        if binding(destination)["sha256"] != source["sha256"]:
            raise ValueError("Preserve an existing different formal numerical artifact: " + str(destination))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Path(source["path"]).open("rb") as before, destination.open("xb") as after:
            shutil.copyfileobj(before, after, length=1 << 20)
    check_binding(source)
    result = binding(destination)
    if result["sha256"] != source["sha256"]:
        raise ValueError("Copied formal artifact differs from its verified original")
    return result


@contextmanager
def _import_lock(root):
    """A kernel file lock survives as a harmless file and releases on process exit."""
    path = _target(root, root / ".formal_numeric_import.lock")
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _import_status(root, audit_binding):
    from datetime import datetime, timezone
    import psutil
    from .repair_execution import write_progress_json

    with _import_lock(root):
        process = psutil.Process()
        started = time.perf_counter()
        state = dict(
            schema="badminton-formal-numeric-import-progress-v1",
            state="running",
            audit=audit_binding,
            pid=process.pid,
            create_time=process.create_time(),
            actual_affinity=process.cpu_affinity(),
            actual_priority=int(process.nice()),
            gpu_started=False,
            training_published=False,
        )

        def update(**values):
            state.update(
                values,
                actual_wall_seconds=time.perf_counter() - started,
                heartbeat_utc=datetime.now(timezone.utc).isoformat(),
            )
            write_progress_json(_target(root, root / "formal_import_progress.json"), state)

        update(stage="native_pose_import", completed_windows=0)
        try:
            yield update
        except BaseException as error:
            update(state="failed", error_type=type(error).__name__, error=str(error))
            raise


def _load_audit(root, proof):
    check_binding(proof)
    audit = read_json(proof["path"])
    if (
        audit.get("schema") != "badminton-formal-numeric-import-audit-v1"
        or audit.get("state") != "eligible_parent_complete_not_imported"
        or audit.get("formal_root") != str(root)
        or audit.get("fitted_tensors_imported") is not False
        or audit.get("training_published") is not False
    ):
        raise ValueError("A completed exact formal numerical import audit is required")
    expected = dict(
        source_input=root / "repair/new_source_input.json",
        body_plan=root / "repair/final_labels/repair_plan.json",
        formal_registry=root / "source/source_registry.json",
        formal_calibration=root / "source/calibration/calibration_snapshot.json",
    )
    for name, path in expected.items():
        check_binding(audit[name])
        if audit[name] != binding(path):
            raise ValueError("Formal import audit belongs to another frozen input")
    check_binding(audit["input_equivalence"])
    for evidence in audit["parent_evidence"].values():
        for entry in evidence.values():
            check_binding(entry)
    for path, sha in audit["teacher_files"].items():
        check_binding(dict(path=path, sha256=sha))
    from .pipeline import BadmintonPipeline

    validate_audited_fit_execution(audit, BadmintonPipeline._fit_digest(None))
    plan = read_json(audit["body_plan"]["path"])
    keys = {j["source_id"] + ":" + Path(j["source_window"]).name for j in plan["jobs"]}
    if keys != set(audit["windows"]):
        raise ValueError("Formal import audit does not cover every actual planned window")
    for key, entry in audit["windows"].items():
        sid, name = key.split(":", 1)
        expected_window = root / "repair/final_labels/sessions" / sid / "windows" / name
        expected_initial = root / "repair/initial_teacher/sessions" / sid / "windows" / name
        parent = Path(entry["parent_root"]).resolve()
        if (
            str(parent) not in audit["parent_evidence"]
            or parent == root
            or root.is_relative_to(parent)
            or parent.is_relative_to(root)
            or Path(entry["target_window"]).resolve() != expected_window
            or Path(entry["target_initial_window"]).resolve() != expected_initial
            or Path(entry["original_window"]).resolve()
            != parent / "repair/final_labels/sessions" / sid / "windows" / name
            or Path(entry["original_initial_window"]).resolve()
            != parent / "repair/initial_teacher/sessions" / sid / "windows" / name
        ):
            raise ValueError("Audited window leaves its exact parent or formal output scope")
        check_binding(entry["original_report"])
        if entry["original_report"] != binding(Path(entry["original_window"]) / "fit_report.json"):
            raise ValueError("Original report leaves its audited parent window")
    check_binding(proof)
    return audit


def _sessions(root, calibration):
    from .rally_pipeline import load_run_session
    from .relocations import load_run_relocations

    resolver = load_run_relocations(root / "source")
    registry = read_json(root / "source/source_registry.json")
    return {
        sid: load_run_session(root / "source", calibration, sid, relocations=resolver) for sid in registry["sources"]
    }


def _import_native_poses(root, audit, audit_binding, calibration):
    sessions = _sessions(root, calibration)
    config = None
    records = []
    for parent, evidence in audit["parent_evidence"].items():
        receipt = read_json(evidence["pose"]["path"])
        current = dict(receipt["config"], calibration_digest=calibration.data["snapshot_digest"])
        if config is not None and config != current:
            raise ValueError("Parent native observations use different model/options configurations")
        config = current
        parent_registry = read_json(evidence["source_registry"]["path"])
        for sid in parent_registry["sources"]:
            session = sessions[sid]
            for camera in ("cam1", "cam2"):
                rows = session.camera_rows(camera)
                for path in sorted((Path(parent) / "repair/native_poses" / sid / camera).glob("*.json")):
                    original = binding(path)
                    value = read_json(path)
                    index = int(path.stem)
                    if value.get("source_id") != sid or value.get("camera") != camera:
                        raise ValueError("Native observation uses another source/camera")
                    rebound = rebind_native_pose_record(value, receipt["config"], current, rows[index], original)
                    target = _target(root, root / "repair/native_poses" / sid / camera / path.name)
                    records.append(dict(source=original, target=_write_once(target, rebound)))
    by_key = {}
    for entry in records:
        path = Path(entry["target"]["path"])
        key = path.parent.parent.name, path.parent.name, int(path.stem)
        if key in by_key:
            raise ValueError("Two parents claim the same native observation")
        by_key[key] = entry
    cases = read_json(root / "repair/case_plan.json")
    for sid, scans in cases["selected_scans"].items():
        pairs = {p["lidar_index"]: p for p in sessions[sid].pairs()}
        for scan in scans:
            for camera in ("cam1", "cam2"):
                if (sid, camera, pairs[scan]["cameras"][camera]["record_index"]) not in by_key:
                    raise ValueError("Actual formal native camera/scan coverage is incomplete")
    inventory = _write_once(
        root / "repair/formal_pose_import_inventory.json",
        dict(
            schema="badminton-formal-native-pose-import-v1",
            state="complete",
            audit=audit_binding,
            config=config,
            records=records,
            numerical_observations_changed=False,
            training_published=False,
        ),
    )
    pose = _write_once(
        root / "repair/pose_receipt.json",
        dict(
            state="complete",
            config=config,
            config_digest=pose_config_digest(config),
            computed=0,
            reused=len(records),
            source_union_sha256=audit["source_input"]["sha256"],
            plan_sha256=binding(root / "repair/case_plan.json")["sha256"],
            formal_import_inventory=inventory,
            numerical_observations_changed=False,
            training_published=False,
        ),
    )
    return pose, records


def _initial_teacher(root, source, target):
    manifest_path = Path(source) / "manifest.json"
    manifest = read_json(manifest_path)
    files = ["manifest.json", "detections.pt", *(entry["path"] for entry in manifest["tracks"])]
    for relative in files:
        original = _target(source, Path(source) / relative)
        _copy_once(binding(original), _target(root, Path(target) / relative))


def _identity(root, audit, pose_receipt, audit_binding):
    windows, evidence_files, parents, abstained, limits = {}, {}, [], [], []
    for parent, evidence in audit["parent_evidence"].items():
        identity = read_json(evidence["identity"]["path"])
        if identity.get("unresolved_cases"):
            raise ValueError("A parent still has unresolved identity review")
        for path, sha in identity["evidence_files"].items():
            if path in evidence_files and evidence_files[path] != sha:
                raise ValueError("Parents disagree about a reviewed evidence file")
            check_binding(dict(path=path, sha256=sha))
            evidence_files[path] = sha
        evidence_files[evidence["identity"]["path"]] = evidence["identity"]["sha256"]
        parents.append(evidence["identity"])
        abstained += identity.get("abstained_relationship_cases", [])
        limits += identity.get("limits", [])
        for key, before in identity["windows"].items():
            if key in windows or key not in audit["windows"]:
                raise ValueError("Formal reviewed identity windows repeat or leave the audit")
            entry = audit["windows"][key]
            current = deepcopy(before)
            current["source_window"] = entry["target_initial_window"]
            teacher = binding(Path(entry["target_initial_window"]) / "teacher/manifest.json")
            if current["source_teacher_manifest_sha256"] != teacher["sha256"]:
                raise ValueError("Copied initial teacher differs from reviewed identity evidence")
            windows[key] = current
    evidence_files[pose_receipt["path"]] = pose_receipt["sha256"]
    evidence_files[audit_binding["path"]] = audit_binding["sha256"]
    value = dict(
        schema="badminton-native-identity-decisions-v1",
        review_status="reviewed",
        source_union_sha256=audit["source_input"]["sha256"],
        calibration_digest=read_json(audit["formal_calibration"]["path"])["snapshot_digest"],
        windows=windows,
        unresolved_cases=[],
        abstained_relationship_cases=abstained,
        evidence_files=evidence_files,
        parent_decisions=parents,
        limits=limits,
        formal_geometry_equivalence=audit["input_equivalence"],
        training_published=False,
        review_derivation="Unchanged previously reviewed operations; identical physical inputs and explicit metadata rebinding",
    )
    path = root / "repair/identity_evidence/identity_decisions.json"
    return _write_once(path, value), value


def _provenance_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _body_teacher(root, source, target, old_decision, new_decision):
    import torch
    from .identity_repair import decision_digest

    before = read_json(Path(source) / "manifest.json")
    old_hash = _provenance_hash(before["provenance"])
    if before["provenance_hash"] != old_hash or before["provenance"]["options"][
        "identity_decisions_sha256"
    ] != decision_digest(old_decision):
        raise ValueError("Original repaired teacher differs from reviewed identity operations")
    after = deepcopy(before)
    after["provenance"]["options"]["identity_decisions_sha256"] = decision_digest(new_decision)
    new_hash = _provenance_hash(after["provenance"])
    after["provenance_hash"] = new_hash
    if "identity_repair" in after:
        if after["identity_repair"].get("decision_sha256") != decision_digest(old_decision):
            raise ValueError("Original teacher identity receipt differs from its reviewed decision")
        after["identity_repair"]["decision_sha256"] = decision_digest(new_decision)
    proofs = []
    for entry in before["tracks"]:
        old_path = _target(source, Path(source) / entry["path"])
        destination = _target(root, Path(target) / entry["path"])
        value = torch.load(old_path, map_location="cpu", weights_only=False)
        if value.get("provenance_hash") != old_hash:
            raise ValueError("Original teacher track provenance changed")
        current = deepcopy(value)
        current["provenance_hash"] = new_hash
        if "provenance" in current:
            if current["provenance"] != before["provenance"]:
                raise ValueError("Original teacher track has a different provenance payload")
            current["provenance"] = after["provenance"]
        if not same_teacher_values(value, current):
            raise ValueError("Teacher numerical values changed during formal provenance rebinding")
        if destination.exists():
            cached = torch.load(destination, map_location="cpu", weights_only=False)
            if (
                cached.get("provenance_hash") != new_hash
                or not same_teacher_values(current, cached)
                or cached.get("provenance") != current.get("provenance")
                or cached.get("temporal_reuse_from") != current.get("temporal_reuse_from")
            ):
                raise ValueError("Preserve an existing different formal teacher track")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(current, destination)
        proofs.append(dict(source=binding(old_path), current=binding(destination)))
    manifest = _write_once(_target(root, Path(target) / "manifest.json"), after)
    return dict(
        source_manifest=binding(Path(source) / "manifest.json"),
        current_manifest=manifest,
        track_files=proofs,
        provenance_hash=new_hash,
        numerical_values_dtype_shape_identical=True,
        identity_decision_sha256=decision_digest(new_decision),
    )


def import_formal_numeric_bodies(formal_root, audit_binding):
    """Never run teachers/fitting; native reconstruction/aggregation remain separate."""
    _require_cpu()
    root = Path(formal_root).resolve()
    if root == (PROJ_ROOT / "outputs").resolve() or not root.is_relative_to((PROJ_ROOT / "outputs").resolve()):
        raise ValueError("Use the separate prepared formal output root")
    started = time.perf_counter()
    audit = _load_audit(root, audit_binding)
    from .calibration import CalibrationSnapshot
    from .new_source_completion import METRICS

    calibration = CalibrationSnapshot.load(root / "source/calibration/calibration_snapshot.json")
    evidence_dir = _target(root, root / "import_evidence")
    baseline_dir = _target(root, root / "gpu_guarded_leg/baseline")
    with _import_status(root, audit_binding) as update:
        pose_receipt, pose_records = _import_native_poses(root, audit, audit_binding, calibration)
        update(stage="initial_teacher_copy", imported_native_camera_frames=len(pose_records))
        pose_bindings = {str(Path(v["source"]["path"]).resolve()): v for v in pose_records}
        for key, entry in audit["windows"].items():
            _initial_teacher(
                root,
                Path(entry["original_initial_window"]) / "teacher",
                Path(entry["target_initial_window"]) / "teacher",
            )
            sid, name = key.split(":", 1)
            secondary = (
                Path(entry["parent_root"])
                / "repair/secondary_teachers/cam1/sessions"
                / sid
                / "windows"
                / name
                / "teacher"
            )
            _initial_teacher(
                root, secondary, root / "repair/secondary_teachers/cam1/sessions" / sid / "windows" / name / "teacher"
            )
        identity_binding, identity = _identity(root, audit, pose_receipt, audit_binding)
        update(stage="body_numeric_copy", reviewed_identity=identity_binding)
        original_qa = dict(
            schema="badminton-native-absolute-qa-v1",
            source_union_sha256=audit["source_input"]["sha256"],
            calibration_digest=calibration.data["snapshot_digest"],
            frames={},
            windows={},
            scope="Original accepted parent QA values; only source/calibration admission metadata rebound; no metric recomputation",
        )
        applied = {}
        for key, entry in audit["windows"].items():
            original, target = Path(entry["original_window"]), Path(entry["target_window"])
            check_binding(entry["original_report"])
            before = read_json(entry["original_report"]["path"])
            parent_identity = read_json(audit["parent_evidence"][entry["parent_root"]]["identity"]["path"])
            teacher = _body_teacher(
                root,
                original / "teacher",
                target / "teacher",
                parent_identity["windows"][key],
                identity["windows"][key],
            )
            teacher_hash = teacher["provenance_hash"]
            for relative, sha in before["artifacts_sha256"].items():
                if relative not in ("quality.jsonl", "ptv3.metadata.jsonl"):
                    _copy_once(
                        dict(path=str(_target(original, original / relative)), sha256=sha),
                        _target(root, target / relative),
                    )
            for name in ("quality.jsonl", "ptv3.metadata.jsonl"):
                original_binding = dict(path=str(original / name), sha256=before["artifacts_sha256"][name])
                check_binding(original_binding)
                rows = read_rows(original / name)
                rebound = [
                    rebind_fit_row(row, calibration.data["snapshot_digest"], target, pose_bindings, teacher_hash)
                    for row in rows
                ]
                for old, new in zip(rows, rebound):
                    if _json_bytes(old["qa"]) != _json_bytes(new["qa"]):
                        raise ValueError("Formal import changed original QA values")
                blob = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rebound).encode("utf-8")
                destination = _target(root, target / name)
                if destination.exists() and destination.read_bytes() != blob:
                    raise ValueError("Preserve an existing different formal metadata file")
                if not destination.exists():
                    with destination.open("xb") as handle:
                        handle.write(blob)
                check_binding(original_binding)
            settings = dict(before["settings"], native_pose_receipt_sha256=pose_receipt["sha256"])
            config = read_json(Path(entry["target_initial_window"]) / "window.json")
            _write_once(
                target / "window.json",
                dict(
                    config,
                    settings=settings,
                    body_repair=dict(
                        source_window=entry["target_initial_window"],
                        source_union_sha256=audit["source_input"]["sha256"],
                    ),
                ),
            )
            report = deepcopy(before)
            report.update(
                fit_algorithm_digest=entry["current_fit_algorithm_digest"],
                settings=settings,
                teacher_provenance_hash=teacher_hash,
                native_pose_receipt_sha256=pose_receipt["sha256"],
                formal_reuse=dict(
                    audit=audit_binding,
                    input_equivalence=audit["input_equivalence"],
                    original_report=entry["original_report"],
                    identity_decisions=identity_binding,
                    numerical_artifacts_byte_identical=True,
                    numeric_fit_recomputed=False,
                    qa_modified=False,
                ),
            )
            report["formal_reuse"]["teacher_rebinding"] = teacher
            report["export"].update(path=str(target / "ptv3.pth"), metadata=str(target / "ptv3.metadata.jsonl"))
            report["artifacts_sha256"] = {name: binding(target / name)["sha256"] for name in before["artifacts_sha256"]}
            current = _write_once(target / "fit_report.json", report)
            sid, name = key.split(":", 1)
            snapshot = _copy_once(current, baseline_dir / "reports" / sid / name / "fit_report.json")
            accepted = 0
            for row in read_rows(original / "quality.jsonl"):
                if not row["accepted"]:
                    continue
                qa_key = f"{sid}/{name}/{row['motion_id']}/{row['lidar_index']}"
                values = {metric: float(row["qa"][metric]) for metric in METRICS}
                if qa_key in original_qa["frames"] or not all(math.isfinite(v) for v in values.values()):
                    raise ValueError("Original accepted QA identity is duplicated or nonfinite")
                original_qa["frames"][qa_key] = values
                accepted += 1
            if accepted != before["accepted"]:
                raise ValueError("Original accepted QA count changed")
            original_qa["windows"][key] = dict(fit_report_sha256=entry["original_report"]["sha256"], accepted=accepted)
            applied[key] = dict(
                original_report=entry["original_report"],
                current_report=snapshot,
                current_window_report=current,
                teacher_rebinding=teacher,
                fitted=before["fitted"],
                accepted=before["accepted"],
            )
            update(
                stage="body_numeric_copy",
                completed_windows=len(applied),
                windows_total=len(audit["windows"]),
                last_window=key,
            )
        qa_binding = _write_once(evidence_dir / "original_qa_snapshot.json", original_qa)
        _load_audit(root, audit_binding)
        result = dict(
            schema="badminton-formal-numeric-import-v1",
            state="complete",
            audit=audit_binding,
            source_union_sha256=audit["source_input"]["sha256"],
            calibration_digest=calibration.data["snapshot_digest"],
            windows=applied,
            parent_evidence=audit["parent_evidence"],
            completed_native_pose_receipt=pose_receipt,
            reviewed_formal_identity=identity_binding,
            original_qa_snapshot=qa_binding,
            numerical_artifacts_byte_identical=True,
            qa_values_unchanged=True,
            numeric_fit_recomputed=False,
            native_reconstruction_pending=True,
            body_aggregation_pending=True,
            final_admission_pending=True,
            copied_windows=len(applied),
            fitted=sum(v["fitted"] for v in applied.values()),
            accepted=sum(v["accepted"] for v in applied.values()),
            gpu_started=False,
            training_published=False,
        )
        receipt = _write_once(root / "formal_numeric_import.json", result)
        update(
            state="import_complete_body_aggregate_native_pending",
            stage="waiting_cpu_aggregation_and_native_verification",
            result=receipt,
        )
    return dict(result, receipt=receipt, actual_import_seconds=time.perf_counter() - started)
