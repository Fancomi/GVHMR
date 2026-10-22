"""Explicit full-run reuse of unchanged native teacher inputs."""

import inspect
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from hmr4d import PROJ_ROOT
from .calibration import CalibrationSnapshot
from .dual_view import file_sha
from .pipeline import write_json
from .pilot import binding, check_binding, validate_full_native_spec
from .rally_pipeline import load_run_session, read_json
from .relocations import load_pilot_run_relocations
from .segmentation import court_polygon
from .teacher import _jsonable, _normalize_rows, _provenance, run_teacher, validate_provenance


def validate_window_reuse(previous, current):
    keys = ("session_id", "window_id", "left_us", "right_us", "settings", "source_pairs")
    if any(previous.get(key) != current.get(key) for key in keys):
        raise ValueError("Teacher reuse requires the exact original native window and settings")


def teacher_options(settings, calibration, camera):
    defaults = inspect.signature(run_teacher).parameters
    return dict(
        batch_size=settings["batch_size"],
        min_track_frames=defaults["min_track_frames"].default,
        max_track_gap_frames=defaults["max_track_gap_frames"].default,
        inference_window_frames=defaults["inference_window_frames"].default,
        roi_polygon=court_polygon(calibration, camera),
        roi_margin_px=settings.get("roi_margin_px", 40),
        detector_imgsz=defaults["detector_imgsz"].default,
        preproc_precision=settings.get("preproc_precision", "float32"),
    )


def checked_copy_tree(source, target):
    """Copy caches without shared writable links or overwriting differing bytes."""
    files = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("Teacher reuse requires ordinary immutable cache files")
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        destination = target / relative
        expected = file_sha(path)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        if file_sha(destination) != expected:
            raise ValueError("Existing teacher reuse target differs from its frozen source")
        files[relative.as_posix()] = expected
    return files


def seed_full_teacher_caches(root, reuse_roots):
    root = Path(root).resolve()
    if not root.is_relative_to(PROJ_ROOT / "outputs") or root == PROJ_ROOT / "outputs":
        raise ValueError("Use the separately prepared full-run output")
    frozen = read_json(root / "pilot_input.json")
    check_binding(frozen["manifest"])
    spec = read_json(frozen["manifest"]["path"])
    validate_full_native_spec(spec)
    source, repair = root / "source", root / "repair"
    calibration = CalibrationSnapshot.load(source / "calibration/calibration_snapshot.json")
    resolver = load_pilot_run_relocations(source, calibration.data)
    plan = read_json(repair / "new_source_input.json")
    jobs = {(j["source_id"], Path(j["source_window"]).name): j for j in plan["jobs"]}
    sessions, result = {}, {}
    for previous_root in map(lambda p: Path(p).resolve(), reuse_roots):
        if root == previous_root or root.is_relative_to(previous_root):
            raise ValueError("Full reuse output must be independent of every sealed pilot")
        native_path = previous_root / "repair/final_labels/native_verification.json"
        if read_json(native_path).get("passed") is not True:
            raise ValueError("Reuse requires a completed independently native-verified pilot")
        previous = CalibrationSnapshot.load(previous_root / "source/calibration/calibration_snapshot.json")
        if previous.data["configuration_files_sha256"] != calibration.data["configuration_files_sha256"]:
            raise ValueError("Reuse camera/court/ROI/clock bundle differs")
        for camera in ("cam1", "cam2"):
            if _jsonable(previous.camera(camera)) != _jsonable(calibration.camera(camera)):
                raise ValueError("Reuse numerical camera differs")
        prior_registry = read_json(previous_root / "source/source_registry.json")
        registry = read_json(source / "source_registry.json")
        dual = read_json(previous_root / "dual_teacher_receipt.json")
        if not dual.get("complete") or dual["source_input_sha256"] != file_sha(previous_root / "repair/new_source_input.json"):
            raise ValueError("Reuse needs the original independently executed dual-camera teachers")
        previous_plan = read_json(previous_root / "repair/new_source_input.json")
        for old_job in previous_plan["jobs"]:
            sid, name = old_job["source_id"], Path(old_job["source_window"]).name
            if (sid, name) not in jobs:
                continue
            if prior_registry["sources"][sid]["source_index"]["sha256"] != registry["sources"][sid]["source_index"]["sha256"]:
                raise ValueError("Reuse source index differs")
            current_window = Path(jobs[sid, name]["source_window"])
            config = read_json(current_window / "window.json")
            validate_window_reuse(read_json(Path(old_job["source_window"]) / "window.json"), config)
            if sid not in sessions:
                sessions[sid] = load_run_session(source, calibration, sid, relocations=resolver)
            session = sessions[sid]
            entry = dict(previous_root=str(previous_root), native_verification=binding(native_path), cameras={})
            for camera in ("cam1", "cam2"):
                item = dual["windows"][sid + ":" + name][camera]
                check_binding(item)
                old_teacher = Path(item["path"]).parent
                manifest = read_json(item["path"])
                rows = _normalize_rows(session.teacher_rows(camera, config["settings"]["teacher_stride"],
                    config["left_us"] - 1000000, config["right_us"] + 1000000))
                options = teacher_options(config["settings"], calibration, camera)
                _, expected = _provenance(session.video_paths[camera], rows, calibration.camera(camera), options)
                validate_provenance(manifest, expected)
                target = (current_window / "teacher" if camera == config["settings"]["camera"] else
                    repair / "secondary_teachers" / camera / "sessions" / sid / "windows" / name / "teacher")
                hashes = checked_copy_tree(old_teacher, target)
                # The existing resume path validates every saved temporal track
                # against freshly computed current provenance without inference.
                run_teacher(session.video_paths[camera], rows, calibration.camera(camera), target,
                            resume=True, **options)
                entry["cameras"][camera] = dict(source=item, target=binding(target / "manifest.json"),
                                               provenance_hash=expected, copied_files_sha256=hashes)
            key = sid + ":" + name
            if key in result:
                raise ValueError("Two old pilots claim the same full-run teacher window")
            result[key] = entry
    receipt = dict(schema="badminton-full-native-teacher-reuse-v1", review_only=True, training_published=False,
        manifest=frozen["manifest"], source_input=binding(repair / "new_source_input.json"),
        original_configuration_sha256=calibration.data["configuration_files_sha256"],
        reused_windows=len(result), reused_camera_windows=2 * len(result), windows=result,
        policy="Exact native rows, video/weight identities, numerical cameras, window bounds, settings and temporal provenance; unchanged cache bytes copied into the independent run")
    write_json(root / "teacher_reuse_receipt.json", receipt)
    return receipt


def pose_config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def rebind_native_pose_record(record, previous_config, current_config, native_row, source_binding):
    """Rebind scope metadata only after the caller proves identical physical cameras."""
    old = {k: v for k, v in previous_config.items() if k != "calibration_digest"}
    new = {k: v for k, v in current_config.items() if k != "calibration_digest"}
    if old != new or record["config_digest"] != pose_config_digest(previous_config) or record["native"] != native_row:
        raise ValueError("Native pose reuse changed models/options, original frame row or source configuration")
    result = deepcopy(record)
    result["config_digest"] = pose_config_digest(current_config)
    result["reuse_native_evidence"] = dict(source=source_binding,
        previous_config_digest=record["config_digest"], previous_camera_snapshot=previous_config["calibration_digest"],
        method="Identical frozen physical camera files, source index/native frame and model hashes; numerical observations unchanged")
    return result


def seed_full_native_poses(root, reuse_roots):
    root = Path(root).resolve()
    frozen = read_json(root / "pilot_input.json")
    check_binding(frozen["manifest"])
    validate_full_native_spec(read_json(frozen["manifest"]["path"]))
    source, repair = root / "source", root / "repair"
    if not root.is_relative_to(PROJ_ROOT / "outputs") or root == PROJ_ROOT / "outputs":
        raise ValueError("Use the independently prepared full-native root")
    calibration = CalibrationSnapshot.load(source / "calibration/calibration_snapshot.json")
    resolver = load_pilot_run_relocations(source, calibration.data)
    registry = read_json(source / "source_registry.json")
    plan = read_json(repair / "case_plan.json")
    model_paths = dict(
        yolo=PROJ_ROOT / "inputs/checkpoints/yolo/yolov8x.pt",
        coco17_regressor=PROJ_ROOT / "hmr4d/utils/body_model/smpl_coco17_J_regressor.pt",
        vitpose=PROJ_ROOT / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
        smpl_triangle_topology=PROJ_ROOT / "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
    )
    models = {name: binding(path) for name, path in model_paths.items()}
    config = dict(calibration_digest=calibration.data["snapshot_digest"], models=models, precision="float16",
                  detector_imgsz=960, scope="all_in_court_native_detections")
    records, sessions, inventory = [], {}, {}
    for previous_root in map(lambda p: Path(p).resolve(), reuse_roots):
        if root == previous_root or root.is_relative_to(previous_root):
            raise ValueError("Pose reuse must keep the old sealed pilot independent")
        native = previous_root / "repair/final_labels/native_verification.json"
        if read_json(native).get("passed") is not True:
            raise ValueError("Pose reuse requires native-verified original labels")
        previous = CalibrationSnapshot.load(previous_root / "source/calibration/calibration_snapshot.json")
        if previous.data["configuration_files_sha256"] != calibration.data["configuration_files_sha256"]:
            raise ValueError("Pose reuse changed the frozen physical camera bundle")
        if any(_jsonable(previous.camera(c)) != _jsonable(calibration.camera(c)) for c in ("cam1", "cam2")):
            raise ValueError("Pose reuse changed the numerical cameras")
        prior_registry = read_json(previous_root / "source/source_registry.json")
        receipt_path = previous_root / "repair/pose_receipt.json"
        receipt = read_json(receipt_path)
        if (receipt.get("state") != "complete" or receipt["config_digest"] != pose_config_digest(receipt["config"])
                or receipt["config"]["calibration_digest"] != previous.data["snapshot_digest"]):
            raise ValueError("Original native pose receipt changed or is incomplete")
        for sid in prior_registry["sources"]:
            if sid not in plan["selected_scans"]:
                continue
            if prior_registry["sources"][sid]["source_index"]["sha256"] != registry["sources"][sid]["source_index"]["sha256"]:
                raise ValueError("Pose reuse source index differs")
            if sid not in sessions:
                sessions[sid] = load_run_session(source, calibration, sid, relocations=resolver)
            session = sessions[sid]
            pairs = session.pairs()
            for camera in ("cam1", "cam2"):
                wanted = {pairs[i]["cameras"][camera]["record_index"] for i in plan["selected_scans"][sid]}
                rows = session.camera_rows(camera)
                for path in sorted((previous_root / "repair/native_poses" / sid / camera).glob("*.json")):
                    index = int(path.stem)
                    if index not in wanted:
                        continue
                    record = read_json(path)
                    if record["source_id"] != sid or record["camera"] != camera:
                        raise ValueError("Native pose cache has another source/camera")
                    key = (sid, camera, index)
                    if key in inventory:
                        raise ValueError("Two pilots claim one reused native camera frame")
                    original = binding(path)
                    result = rebind_native_pose_record(record, receipt["config"], config, rows[index], original)
                    target = repair / "native_poses" / sid / camera / path.name
                    if target.exists() and read_json(target) != result:
                        raise ValueError("Existing full native pose differs from explicit scope reuse")
                    write_json(target, result)
                    inventory[key] = True
                    records.append(dict(source=original, target=binding(target), receipt=binding(receipt_path)))
    result = dict(schema="badminton-full-native-pose-reuse-v1", state="seeded_partial", review_only=True,
        training_published=False, manifest=frozen["manifest"], config=config, config_digest=pose_config_digest(config),
        reused_native_camera_frames=len(records), records=records,
        numeric_observations_changed=False, original_configuration_sha256=calibration.data["configuration_files_sha256"])
    write_json(root / "native_pose_reuse_receipt.json", result)
    return result
