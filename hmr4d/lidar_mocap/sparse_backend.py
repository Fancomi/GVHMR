"""One fixed-camera native-LiDAR case through the existing fitter and verifier."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .calibration import CalibrationSnapshot
from .pilot import binding, check_binding
from .rally_pipeline import load_run_session, read_json
from .relocations import load_pilot_run_relocations
from .repair_execution import remaining_lease_seconds, write_progress_json
from .sparse_trial_evaluation import statistics
from .teacher import _jsonable, _numpy, load_teacher_track, _save_json


def prepare_backend_input(trial_directory, count=50):
    trial_directory = Path(trial_directory).resolve()
    recipe = read_json(trial_directory / "input.json")
    window = read_json(recipe["source_window"]["path"])
    directory = trial_directory / "backend"
    directory.mkdir(exist_ok=True)
    selected = sorted(window["source_pairs"])[-count:]
    result = dict(schema="gvhmr-sparse-native-fit-trial-v1", root=recipe["root"],
        parent_recipe=binding(trial_directory / "input.json"), source_id=recipe["source_id"],
        selected_source_pairs=selected, output_directory=str(directory), source_window=recipe["source_window"],
        manifest=recipe["manifest"], source_input=recipe["source_input"], calibration=recipe["calibration"],
        paired={name: binding(trial_directory / "paired10" / name / "manifest.json") for name in ("cam1", "cam2")},
        baseline={name: read_json(trial_directory / "baseline30_progress.json")["cameras"][name]["manifest"] for name in ("cam1", "cam2")},
        native_jump_review=binding(trial_directory / "native_jump_review.jpg"),
        native_jump_visually_confirmed=True, iterations=window["settings"]["iterations"],
        distal_pose_prior_weight=.03, lower_limb_reprojection_weight=2., training_published=False,
        identity_scope="Separate original local tracklets; no cross-window/person identity admission")
    path = directory / "input.json"
    if path.exists() and read_json(path) != result:
        raise ValueError("The backend case is already frozen")
    if not path.exists():
        write_progress_json(path, result)
    return binding(path)


def _native_poses(recipe, session, pairs, directory, calibration):
    config = dict(schema="native-observations-from-actual-paired-teacher-crops-v1", source_manifests=recipe["paired"],
                  calibration_digest=calibration.data["snapshot_digest"], virtual_observations=False)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    for name in ("cam1", "cam2"):
        manifest_path = Path(recipe["paired"][name]["path"])
        manifest = read_json(manifest_path)
        observed = {}
        for entry in manifest["tracks"]:
            track = load_teacher_track(manifest_path.parent / entry["path"])
            for index, record in enumerate(_numpy(track["record_index"])):
                observed.setdefault(int(record), []).append((track, index))
        native_rows = {row["record_index"]: row for row in session.camera_rows(name)}
        target = directory / "native_poses" / session.session_id / name
        target.mkdir(parents=True, exist_ok=True)
        for pair in pairs:
            record = pair["cameras"][name]["record_index"]
            values = observed.get(record, [])
            _save_json(target / f"{record:07d}.json", dict(config_digest=digest, native=native_rows[record],
                boxes=[_numpy(track["bbx_xyxy"])[index] for track, index in values],
                keypoints=[_numpy(track["kp2d"])[index] for track, index in values],
                observation_role="actual_native_RGB_crop_never_latent", source_manifest=recipe["paired"][name]))
    _save_json(directory / "case_plan.json", dict(selected_scans={session.session_id: [p["lidar_index"] for p in pairs]}))
    _save_json(directory / "pose_receipt.json", dict(state="complete", config=config, config_digest=digest,
        count=len(pairs) * 2, no_new_image_inference=True, training_published=False))


def run_backend_trial(root, input_path, lease_path):
    import psutil
    import torch
    import smplx
    from hmr4d import PROJ_ROOT
    from .fitting import FittingConfig, SMPLFitter
    from .pipeline import BadmintonPipeline
    from .pose_ray_filter import RayFilteredPoseProvider
    from .sparse_teacher import augment_with_latents
    from tools.verify_badminton_native_repair import verify_body

    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity([0, 1])
    torch.set_num_threads(2)
    root, input_path = Path(root).resolve(), Path(input_path).resolve()
    recipe = read_json(input_path)
    directory = Path(recipe["output_directory"])
    if (Path(recipe["root"]).resolve() != root or not directory.resolve().is_relative_to(root / "benchmarks")
            or not input_path.is_relative_to(directory)):
        raise ValueError("Use the exact independent native-fit experimental namespace")
    for key in ("manifest", "source_input", "calibration", "source_window", "parent_recipe", "native_jump_review"):
        check_binding(recipe[key])
    for cameras in (recipe["paired"], recipe["baseline"]):
        for proof in cameras.values():
            check_binding(proof)
    lease = read_json(lease_path)
    if (lease.get("experiment_input") != binding(input_path) or lease.get("stage") != "fit"
            or Path(lease["root"]).resolve() != root or remaining_lease_seconds(lease, recipe["manifest"]["sha256"]) < 300):
        raise ValueError("The actual backend recipe needs its own held fit lease")
    state = dict(state="running", stage="prepare_native_inputs", recipe=binding(input_path),
                 started_utc=datetime.now(timezone.utc).isoformat(), phases={}, training_published=False)
    progress = directory / "progress.json"
    write_progress_json(progress, state)
    calibration = CalibrationSnapshot.load(recipe["calibration"]["path"])
    relocations = load_pilot_run_relocations(root / "source", calibration.data)
    session = load_run_session(root / "source", calibration, recipe["source_id"], relocations=relocations)
    selected = set(recipe["selected_source_pairs"])
    pairs = [pair for pair in session.pairs() if pair["lidar_index"] in selected]
    if len(pairs) != len(selected):
        raise ValueError("The case must retain exactly the declared actual eligible scans")
    _native_poses(recipe, session, pairs, directory / "observations", calibration)
    poses = RayFilteredPoseProvider(directory / "observations", calibration)
    cfg = FittingConfig(distal_pose_prior_weight=recipe["distal_pose_prior_weight"],
                        lower_limb_reprojection_weight=recipe["lower_limb_reprojection_weight"])
    window = read_json(recipe["source_window"]["path"])
    pipeline = BadmintonPipeline(calibration.data["sessions_root"], "D:/PROJECT/LIDAR/lidar_project", directory)
    pipeline.calibration = calibration
    pipeline.settings = dict(window["settings"], share_track_shape=True,
                             native_pose_receipt_sha256=poses.digest, native_pose_joint_ray_policy_sha256=poses.policy_sha256)
    state["stage"] = "reinfer_only_temporal_states"
    write_progress_json(progress, state)
    target = directory / "paired10/teacher"
    paired = augment_with_latents(Path(recipe["paired"]["cam2"]["path"]).parent, target)
    baseline = read_json(recipe["baseline"]["cam2"]["path"])
    baseline = dict(baseline, tracks=[dict(entry, path=str(Path(recipe["baseline"]["cam2"]["path"]).parent / entry["path"])) for entry in baseline["tracks"]])
    fitter = SMPLFitter(device="cuda", config=cfg)
    for name, teacher, output in (("baseline30", baseline, directory / "baseline30"),
                                   ("paired10", paired, directory / "paired10")):
        if remaining_lease_seconds(read_json(lease_path), recipe["manifest"]["sha256"]) < 180:
            raise ValueError("Backend camera fitting would leave the approved local lease")
        state["stage"] = name + "_native_joint_fit"
        write_progress_json(progress, state)
        output.mkdir(parents=True, exist_ok=True)
        _save_json(output / "window.json", dict(window, source_pairs=recipe["selected_source_pairs"], settings=pipeline.settings))
        started = time.perf_counter()
        report = pipeline._fit_window(session, pairs, teacher, output, fitter, pose_provider=poses)
        rows = [json.loads(row) for row in (output / "quality.jsonl").read_text(encoding="utf-8").splitlines()]
        state["phases"][name] = dict(seconds=time.perf_counter() - started, scan_count=len(pairs),
            fitted=report["fitted"], accepted=report["accepted"], qa_rejected=report["fitted"] - report["accepted"],
            expected_two_person_cells=2 * len(pairs), unfitted_two_person_cells=max(0, 2 * len(pairs) - report["fitted"]),
            rejection_reasons=report["fit_rejection_reasons"], report=binding(output / "fit_report.json"),
            surface_mean_mm=statistics([r["qa"]["final_p2surface_mean_m"] * 1000 for r in rows]),
            surface_p90_mm=statistics([r["qa"]["final_p2surface_p90_m"] * 1000 for r in rows]),
            reprojection_by_camera_px={cam: statistics([r["qa"].get("reprojection_by_camera_px", {}).get(cam)
                for r in rows if r["qa"].get("reprojection_by_camera_px", {}).get(cam) is not None]) for cam in ("cam1", "cam2")})
        write_progress_json(progress, state)
    del fitter
    torch.cuda.empty_cache()
    model = smplx.create(PROJ_ROOT / "inputs/checkpoints/body_models", model_type="smpl", gender="neutral").eval()
    state["stage"] = "native_member_and_SMPL_verification"
    write_progress_json(progress, state)
    for name in state["phases"]:
        state["phases"][name]["native_verification"] = verify_body(directory / name, session, model, calibration.data["snapshot_digest"])
    state.update(state="complete", stage="backend_complete", safe_to_release_gpu=True,
                 completed_utc=datetime.now(timezone.utc).isoformat(), fitting_config=asdict(cfg),
                 camera_parameters_changed=False, physical_identity_admitted=False)
    write_progress_json(progress, state)
    return state
