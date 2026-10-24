"""Apply sealed, continuous leg corrections after the complete baseline has stopped."""

import argparse
import copy
import hashlib
from pathlib import Path
import shutil

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.export import FIELDS, acceptance_reasons, export_joblib
from hmr4d.lidar_mocap.fitting import _projection_loss, point_to_surface_squared
from hmr4d.lidar_mocap.multiview import epipolar_errors, fundamental_matrix
from hmr4d.lidar_mocap.pipeline import BadmintonPipeline, write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json, read_rows, load_run_session
from hmr4d.lidar_mocap.relocations import load_pilot_run_relocations


def load_review_sessions(labels, calibration, selection):
    """Validate all required session contracts before model preparation or writes."""
    source_ids = sorted({key.split(":", 1)[0] for key, item in selection["windows"].items() if item["selected"]})
    if not source_ids:
        return {}
    relocations = (load_pilot_run_relocations(labels, calibration.data)
                   if (labels / "source_registry.json").exists() else None)
    return {sid: load_run_session(labels, calibration, sid, relocations=relocations) for sid in source_ids}


def backup_file(source, target, expected):
    if target.exists():
        if file_sha(target) != expected:
            raise ValueError("Original repair-stage backup changed")
        return
    if file_sha(source) != expected:
        raise ValueError("Source differs from sealed baseline before backup")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temp)
    temp.replace(target)


def revised_qa(row, arrays, source_sample, cal, reg, faces, device):
    qa = copy.deepcopy(row["qa"])
    evidence = copy.deepcopy(row["native_pose_evidence"])
    views = {}
    for cam in ["cam1", "cam2"]:
        views[cam] = np.asarray(
            read_json(evidence["files"][cam]["path"])["keypoints"][evidence[cam + "_detection"]], np.float32
        )
    cam = arrays["candidate"]["camera_permuted"]
    views[cam][[13, 14, 15, 16]] = views[cam][[14, 13, 16, 15]]
    a, b = views["cam1"], views["cam2"]
    valid = (a[:, 2] >= 0.35) & (b[:, 2] >= 0.35)
    errors = epipolar_errors(a, b, fundamental_matrix(cal))
    threshold = max(6.0, min(10.0, 2.5 * float(np.median(errors[5:17][valid[5:17]]))))
    invalid = valid & (errors > threshold)
    a[invalid, 2] = b[invalid, 2] = 0
    evidence.update(
        leg_joint_permutation=dict(camera=cam, indices=[14, 13, 16, 15], original_indices=[13, 14, 15, 16]),
        inconsistent_joint_rays=np.flatnonzero(invalid).tolist(),
        joint_ray_error_px=errors.tolist(),
        joint_ray_threshold_px=threshold,
    )
    tensor = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)
    vertices = tensor(arrays["vertices"])[None]
    cloud = tensor(source_sample["source_cloud"])
    cloud = cloud[torch.linspace(0, len(cloud) - 1, min(4096, len(cloud)), device=device).long()]
    with torch.no_grad():
        distance = point_to_surface_squared(vertices, faces, [cloud])[0].sqrt()
        _, measurements = _projection_loss(
            vertices, reg, cal, {k: tensor(v)[None] for k, v in views.items()}, "cam2", "undistorted", 2.0
        )
    h = (arrays["vertices"] - np.asarray(cal.reference_plane["origin_viewer_m"])) @ np.asarray(
        cal.reference_plane["normal_viewer"]
    )
    mean = float(distance.mean())
    p90 = float(torch.quantile(distance, 0.9))
    floor = float(np.maximum(-h, 0).max())
    qa.update(
        surface_mean_m=mean,
        surface_median_m=float(torch.quantile(distance, 0.5)),
        surface_p90_m=p90,
        surface_p95_m=float(torch.quantile(distance, 0.95)),
        final_p2surface_mean_m=mean,
        final_p2surface_p90_m=p90,
        surface_inlier_fraction_10cm=float((distance <= 0.1).float().mean()),
        surface_improvement_m=qa["teacher_surface_mean_m"] - mean,
        qa_points=len(cloud),
        max_ground_penetration_m=floor,
        floor_penetration_m=floor,
        ground_penetrating_vertex_fraction=float(np.mean(h < -0.02)),
        reprojection_by_camera_px={k: float(v[0][0].cpu()) for k, v in measurements.items()},
        reprojection_valid_joints={k: int(v[1][0].cpu()) for k, v in measurements.items()},
        reprojection_mean_px=float(np.mean([float(v[0][0].cpu()) for v in measurements.values()])),
        finite=bool(np.isfinite(arrays["vertices"]).all()),
        leg_refinement_correction_fraction=arrays["candidate"]["correction_fraction"],
        leg_refinement_max_delta_degrees=arrays["candidate"]["max_delta_angle_deg"],
    )
    # These old conversion-stage measurements do not describe the new pose.
    qa.pop("distal_rotation_from_conversion_deg", None)
    qa.pop("pose_change_from_teacher_matrix_rms", None)
    before = Rotation.from_rotvec(source_sample["body_pose"].reshape(23, 3)).as_matrix()
    after = Rotation.from_rotvec(arrays["body_pose"].reshape(23, 3)).as_matrix()
    qa["distal_rotation_from_previous_fit_deg"] = np.degrees(
        Rotation.from_matrix(after[[6, 7, 9, 10]] @ before[[6, 7, 9, 10]].transpose(0, 2, 1)).magnitude()
    ).tolist()
    return qa, evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--review-sessions-only", action="store_true",
                   help="Validate all bound review session inputs on CPU and exit before models or label writes")
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if args.threads < 1:
        p.error("--threads must be positive")
    if args.review_sessions_only and args.prepare_only:
        p.error("Use --review-sessions-only separately; --prepare-only also computes CUDA QA")
    root = args.root.resolve()
    labels = root / "final_labels"
    plan = read_json(labels / "repair_plan.json")
    selection_path = root / "leg_refinement_selection.json"
    selection = read_json(selection_path)
    baseline = read_json(labels / "repair_receipt.json") if (labels / "repair_receipt.json").exists() else {}
    if not args.prepare_only and (
        baseline.get("windows") != len(plan["jobs"])
        or selection["state"] != "complete"
        or len(selection["windows"]) != len(plan["jobs"])
    ):
        raise ValueError("Full baseline and full leg selection must finish before any label writes")
    if file_sha(plan["source_union"]) != plan["source_union_sha256"]:
        raise ValueError("Original published union changed")
    original_union = read_json(plan["source_union"])
    incremental_policy = None
    if original_union.get("schema") == "badminton-new-source-repair-input-v1" and "release_policy" in original_union:
        from hmr4d.lidar_mocap.incremental_release import policy_from_binding, validate_roles

        incremental_policy = policy_from_binding(original_union["release_policy"])
        if file_sha(labels / "source_registry.json") != original_union["source_registry_sha256"]:
            raise ValueError("New-source registry differs from its prepared input")
        registry = read_json(labels / "source_registry.json")
        roles = validate_roles(original_union["session_splits"], registry["sources"], incremental_policy.required)
        suggested = {name: item["sessions"] for name, item in original_union["suggested_session_splits"].items()}
        if (
            original_union.get("review_only") is True
            or roles != incremental_policy.roles
            or suggested != roles
            or original_union.get("required_nonempty_splits") != incremental_policy.required
        ):
            raise ValueError("New-source roles/required splits differ from the explicit release policy")
    if (
        file_sha(PROJ_ROOT / "hmr4d/lidar_mocap/leg_refinement.py") != selection["refinement_policy_sha256"]
        or file_sha(PROJ_ROOT / "hmr4d/lidar_mocap/leg_selection.py") != selection["selection_policy_sha256"]
    ):
        raise ValueError("Sealed leg policy changed")
    cal = CalibrationSnapshot.load(labels / "calibration/calibration_snapshot.json")
    review_sessions = load_review_sessions(labels, cal, selection)
    if args.review_sessions_only:
        print(dict(state="review_session_contracts_validated", sources=sorted(review_sessions), gpu_started=False))
        return
    torch.set_num_threads(args.threads)
    device = "cuda"
    if read_json(plan["source_union"]).get("schema") == "badminton-new-source-repair-input-v1":
        import smplx
        topology = smplx.create(PROJ_ROOT / "inputs/checkpoints/body_models", model_type="smpl", gender="neutral")
        faces = torch.as_tensor(topology.faces.astype(np.int64), device=device)
    else:
        import base64
        viewer = read_json(PROJ_ROOT / "outputs/badminton_rallies_v1/viewer/manifest.json")
        faces = torch.as_tensor(
            np.frombuffer(base64.b64decode(viewer["faces"]), dtype="<u2").astype(np.int64), device=device
        ).reshape(-1, 3)
    reg = torch.load(
        PROJ_ROOT / "hmr4d/utils/body_model/smpl_coco17_J_regressor.pt", map_location=device, weights_only=False
    )
    selection_sha = file_sha(selection_path)
    policy_sha = file_sha(Path(__file__))
    prepared = {}
    for key, item in selection["windows"].items():
        if not item["selected"]:
            continue
        sid, window = key.split(":")
        source = labels / "sessions" / sid / "windows" / window
        backup = root / "leg_refinement_before" / sid / window
        old_quality = backup / "quality.jsonl" if (backup / "quality.jsonl").exists() else source / "quality.jsonl"
        rows = read_rows(old_quality)
        indexed = {(r["motion_id"], r["lidar_index"]): r for r in rows}
        changes = {}
        for update in item["selected"]:
            row = indexed[update["motion_id"], update["lidar_index"]]
            candidate = update["candidate"]
            path = Path(row["fit_path"])
            old = backup / "fits" / path.name if (backup / "fits" / path.name).exists() else path
            if (
                file_sha(old) != update["source_fit_sha256"]
                or file_sha(candidate["arrays_path"]) != candidate["arrays_sha256"]
            ):
                raise ValueError("Refinement source or candidate changed")
            with np.load(old) as f:
                sample = {name: f[name].copy() for name in f.files}
            with np.load(candidate["arrays_path"]) as f:
                arrays = {name: f[name].copy() for name in f.files}
            arrays["candidate"] = candidate
            qa, evidence = revised_qa(row, arrays, sample, cal, reg, faces, device)
            if acceptance_reasons(qa, len(sample["source_cloud"])):
                raise ValueError("Selected leg candidate fails the unchanged body QA gate")
            changes[row["motion_id"], row["lidar_index"]] = dict(
                update=update, arrays=arrays, sample=sample, qa=qa, evidence=evidence
            )
        prepared[key] = dict(source=source, backup=backup, rows=rows, changes=changes)
    print(
        dict(prepared_windows=len(prepared), prepared_frames=sum(len(w["changes"]) for w in prepared.values())),
        flush=True,
    )
    if args.prepare_only:
        return
    write_json(
        labels / "post_refinement.json",
        dict(state="applying", selection_sha256=selection_sha, policy_sha256=policy_sha),
    )
    receipt = dict(
        state="applying", selection_sha256=selection_sha, policy_sha256=policy_sha, windows={}, training_published=False
    )
    for key, work in prepared.items():
        source, backup = work["source"], work["backup"]
        item = selection["windows"][key]
        base_report = (
            backup / "fit_report.json" if (backup / "fit_report.json").exists() else source / "fit_report.json"
        )
        if file_sha(base_report) != item["source_fit_report_sha256"]:
            raise ValueError("Baseline fit report changed")
        report = read_json(base_report)
        for name in ["fit_report.json", "window.json", "ptv3.pth", "ptv3.metadata.jsonl", "quality.jsonl"]:
            expected = (
                item["source_fit_report_sha256"]
                if name == "fit_report.json"
                else report["artifacts_sha256"].get(name, file_sha(source / name))
            )
            backup_file(source / name, backup / name, expected)
        for (motion, scan), change in work["changes"].items():
            update = change["update"]
            path = Path(update["source_fit_path"])
            backup_file(path, backup / "fits" / path.name, update["source_fit_sha256"])
        exported = joblib.load(backup / "ptv3.pth")
        old_meta = read_rows(backup / "ptv3.metadata.jsonl")
        indices = {(r["motion_id"], r["lidar_index"]): i for i, r in enumerate(old_meta)}
        for row in work["rows"]:
            change = work["changes"].get((row["motion_id"], row["lidar_index"]))
            if change is None:
                continue
            arrays = change["arrays"]
            sample = change["sample"]
            sample.update({name: arrays[name] for name in ["vertices", "joints", "body_pose"]})
            dest = Path(row["fit_path"])
            temp = dest.with_suffix(".tmp")
            with temp.open("wb") as handle:
                np.savez_compressed(handle, **sample)
            temp.replace(dest)
            row["post_leg_refinement"] = dict(
                selection_sha256=selection_sha,
                source_fit_sha256=change["update"]["source_fit_sha256"],
                candidate=change["update"]["candidate"],
                ray_witness=change["update"]["ray_witness"],
                previous_fit_qa=copy.deepcopy(row["qa"]),
                root_pelvis_shape_unchanged=True,
                maximum_correction_step_degrees=20.5,
            )
            row["qa"] = change["qa"]
            row["native_pose_evidence"] = change["evidence"]
            exported["body_pose"][indices[row["motion_id"], row["lidar_index"]]] = torch.from_numpy(arrays["body_pose"])
        rows_index = {(r["motion_id"], r["lidar_index"]): r for r in work["rows"]}
        samples = [
            dict(
                {name: np.asarray(exported[name][i]) for name in FIELDS},
                motion_id=old["motion_id"],
                metadata=rows_index[old["motion_id"], old["lidar_index"]],
            )
            for i, old in enumerate(old_meta)
        ]
        report["export"] = export_joblib(samples, source / "ptv3.pth")
        (source / "quality.jsonl").write_text(
            "".join(__import__("json").dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in work["rows"]),
            encoding="utf-8",
        )
        report["pre_leg_refinement_algorithm_digest"] = report["fit_algorithm_digest"]
        report["fit_algorithm_digest"] = hashlib.sha256(
            (report["fit_algorithm_digest"] + selection_sha + policy_sha).encode()
        ).hexdigest()
        report["post_leg_refinement"] = dict(
            selection_sha256=selection_sha, policy_sha256=policy_sha, frames=len(work["changes"]), backup=str(backup)
        )
        from hmr4d.lidar_mocap.review import make_review

        sid = key.split(":")[0]
        report["review"] = make_review(review_sessions[sid], cal, source)
        for name in ["ptv3.pth", "ptv3.metadata.jsonl", "quality.jsonl"]:
            report["artifacts_sha256"][name] = file_sha(source / name)
        for change in work["changes"].values():
            path = Path(change["update"]["source_fit_path"])
            report["artifacts_sha256"][path.relative_to(source).as_posix()] = file_sha(path)
        write_json(source / "fit_report.json", report)
        receipt["windows"][key] = dict(
            frames=len(work["changes"]), fit_report_sha256=file_sha(source / "fit_report.json")
        )
        write_json(root / "leg_refinement_receipt.json", receipt)
    pipeline = BadmintonPipeline(cal.data["sessions_root"], "D:/PROJECT/LIDAR/lidar_project", labels)
    pipeline.calibration = cal
    aggregate_sources = sorted({j["source_id"] for j in plan["jobs"]})
    if original_union.get("schema") == "badminton-new-source-repair-input-v1":
        aggregate_sources = sorted({sid for ids in original_union["session_splits"].values() for sid in ids})
    aggregate = pipeline.aggregate(aggregate_sources)
    # Keep the existing seven-source session protocol; the legacy aggregate's
    # first-three-source suggestion is insufficient for this candidate run.
    splits = {}
    assigned = []
    for name, reference in original_union["suggested_session_splits"].items():
        parts = []
        for sid in reference["sessions"]:
            item = aggregate["exports"][sid]
            assigned.append(sid)
            parts.append(
                dict(
                    session_id=sid,
                    path=item["path"],
                    metadata=item["metadata"],
                    accepted=item["accepted"],
                    sha256=file_sha(item["path"]),
                    metadata_sha256=file_sha(item["metadata"]),
                )
            )
        splits[name] = dict(sessions=reference["sessions"], accepted=sum(p["accepted"] for p in parts), parts=parts)
    if len(assigned) != len(set(assigned)) or set(assigned) != set(aggregate["exports"]):
        raise ValueError("Frozen session split assignment does not cover the candidate sources exactly once")
    aggregate.update(
        suggested_session_splits=splits,
        source_union_manifest_sha256=plan["source_union_sha256"],
        training_published=False,
        admission="body_QA_candidates_pending_dual_camera_export",
    )
    if original_union.get("schema") == "badminton-new-source-repair-input-v1":
        from hmr4d.lidar_mocap.export import require_nonempty_splits

        if incremental_policy is not None:
            incremental_policy.recheck()
        aggregate["required_nonempty_splits"] = (
            incremental_policy.required if incremental_policy is not None else ["train", "val", "test"]
        )
        require_nonempty_splits(aggregate)
    write_json(labels / "datasets/dataset_manifest.json", aggregate)
    receipt.update(
        state="complete",
        selected_frames=selection["selected_frames"],
        updated_frames=sum(w["frames"] for w in receipt["windows"].values()),
        body_qa_accepted=aggregate["accepted"],
    )
    write_json(root / "leg_refinement_receipt.json", receipt)
    write_json(labels / "post_refinement.json", dict(receipt, source="bounded_native_leg_correspondence"))
    print(
        dict(state="complete", updated_frames=receipt["updated_frames"], updated_windows=len(receipt["windows"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
