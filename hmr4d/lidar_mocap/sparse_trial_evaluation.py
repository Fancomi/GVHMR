"""CPU comparison on identical real native-camera times, without invented GT."""

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .pilot import binding, check_binding
from .rally_pipeline import read_json
from .repair_execution import write_progress_json
from .teacher import _box_iou, _numpy, load_teacher_track, sample_track


def statistics(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return dict(count=int(len(values)), mean=float(values.mean()) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                p95=float(np.percentile(values, 95)) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)


def _joints(model, params):
    import torch

    with torch.no_grad():
        return model(**{key: torch.as_tensor(value, dtype=torch.float32) for key, value in params.items()}).numpy()


def _project(joints, camera):
    K = np.asarray(camera["K"])
    image = joints @ K.T
    return image[..., :2] / image[..., 2:], joints[..., 2] > 0


def evaluate_trial(directory):
    import torch
    from hmr4d.utils.smplx_utils import make_smplx

    torch.set_num_threads(2)
    directory = Path(directory).resolve()
    recipe_path = directory / "input.json"
    recipe = read_json(recipe_path)
    window = read_json(recipe["source_window"]["path"])
    progress = {phase: read_json(directory / (phase + "_progress.json")) for phase in ("baseline30", "paired10")}
    for phase, state in progress.items():
        if state["state"] != "complete" or state["recipe"] != binding(recipe_path):
            raise ValueError("Both trial phases must be complete against exactly the same frozen input")
        for camera in state["cameras"].values():
            check_binding(camera["manifest"])
    model = make_smplx("supermotion_coco17").eval()
    result = dict(schema="gvhmr-paired-rgb-trial-comparison-v1", recipe=binding(recipe_path),
                  cameras={}, training_published=False, formal_sampling_policy_changed=False,
                  limitations=["Observed 2D poses are inference inputs, not independent ground truth",
                               "No LiDAR-fitted neutral SMPL or final surface error is evaluated here",
                               "Track association is geometric correspondence, not reviewed physical identity",
                               "The single initial rally does not establish fast/jump-case quality"])
    detailed = []
    for camera_name, item in recipe["cameras"].items():
        paths = {}
        for phase in progress:
            manifest_path = Path(progress[phase]["cameras"][camera_name]["manifest"]["path"])
            manifest = read_json(manifest_path)
            paths[phase] = [load_teacher_track(manifest_path.parent / track["path"]) for track in manifest["tracks"]]
        observed_rows = item["observed_rows"]
        times = np.asarray([row["common_time_us"] for row in observed_rows], dtype=np.int64)
        scan_times = np.asarray([row["scan_mid_us"] for row in item["scan_bindings"]], dtype=np.int64)
        in_window = (scan_times >= window["left_us"]) & (scan_times <= window["right_us"])
        baseline = [sample_track(track, times) for track in paths["baseline30"]]
        baseline_joints = [_joints(model, sample["smpl_params_incam"]) for sample in baseline]
        paired_joints = [_joints(model, track["smpl_params_incam"]) for track in paths["paired10"]]
        candidate_indices = [{int(t): index for index, t in enumerate(_numpy(track["common_time_us"]))} for track in paths["paired10"]]
        per_phase = {phase: dict(reprojection=[], ankles=[], root_acceleration=[], body_acceleration=[]) for phase in progress}
        tracks_by_candidate, unmatched, matched = {}, 0, 0
        exact_count, exact_kp_distance, errors = 0, [], []
        for position in np.flatnonzero(in_window):
            time_us = int(times[position])
            available = [(index, mapping[time_us]) for index, mapping in enumerate(candidate_indices) if time_us in mapping]
            if not available:
                continue
            valid_baseline = [i for i, sample in enumerate(baseline) if sample["valid"][position]]
            costs = np.ones((len(available), len(valid_baseline)))
            for i, (index, local) in enumerate(available):
                for j, original in enumerate(valid_baseline):
                    costs[i, j] = 1 - _box_iou(_numpy(paths["paired10"][index]["bbx_xyxy"])[local], baseline[original]["bbx_xyxy"][position])
            paired = {}
            if costs.size:
                ii, jj = linear_sum_assignment(costs)
                paired = {int(i): int(j) for i, j in zip(ii, jj) if costs[i, j] < 0.7}
            for i, (index, local) in enumerate(available):
                if i not in paired:
                    unmatched += 1
                    continue
                original = valid_baseline[paired[i]]
                track, sample = paths["paired10"][index], baseline[original]
                kp = _numpy(track["kp2d"])[local]
                joints = dict(baseline30=baseline_joints[original][position], paired10=paired_joints[index][local])
                projected = {phase: _project(value, item["camera"]) for phase, value in joints.items()}
                visible = (kp[:, 2] > 0.5) & projected["baseline30"][1] & projected["paired10"][1]
                entry = dict(camera=camera_name, common_time_us=time_us,
                             scan_mid_us=int(scan_times[position]), native_record_index=observed_rows[position]["record_index"],
                             native_media_pts_us=observed_rows[position]["media_pts_us"],
                             baseline_nearest_record=int(sample["metadata"]["record_index"][position]),
                             baseline_nearest_time_error_us=int(sample["metadata"]["model_time_error_us"][position]),
                             candidate_track=int(track["track_id"]), candidate_segment=int(track["segment_id"]),
                             baseline_track=int(sample["metadata"]["track_id"]),
                             baseline_segment=int(sample["metadata"]["segment_id"]),
                             bbox_iou=float(1 - costs[i, paired[i]]), visible_joint_count=int(visible.sum()))
                same_record = entry["native_record_index"] == entry["baseline_nearest_record"]
                entry["baseline_same_actual_rgb_record"] = same_record
                if same_record:
                    exact_count += 1
                    common_visible = visible & (sample["kp2d"][position, :, 2] > 0.5)
                    exact_kp_distance.extend(np.linalg.norm(kp[common_visible, :2] - sample["kp2d"][position, common_visible, :2], axis=-1))
                for phase in progress:
                    residual = np.linalg.norm(projected[phase][0] - kp[:, :2], axis=-1)
                    per_phase[phase]["reprojection"].extend(residual[visible])
                    ankle_visible = visible[15:17]
                    per_phase[phase]["ankles"].extend(residual[15:17][ankle_visible])
                    entry[phase + "_reprojection_mean_px"] = float(residual[visible].mean()) if visible.any() else None
                errors.append(entry["baseline_nearest_time_error_us"])
                history_key = (index, original)
                tracks_by_candidate.setdefault(history_key, []).append((position, time_us, joints))
                detailed.append(entry)
                matched += 1
        correspondence_changes, segment_boundaries = 0, 0
        for index in range(len(paths["paired10"])):
            sequence = sorted((row[0], original) for (candidate, original), rows in tracks_by_candidate.items() if candidate == index for row in rows)
            for before, after in zip(sequence, sequence[1:]):
                previous = paths["baseline30"][before[1]]
                current = paths["baseline30"][after[1]]
                correspondence_changes += int(previous["track_id"] != current["track_id"])
                segment_boundaries += int(previous["track_id"] == current["track_id"] and before[1] != after[1])
        for sequence in tracks_by_candidate.values():
            for before, current, after in zip(sequence, sequence[1:], sequence[2:]):
                if current[0] != before[0] + 1 or after[0] != current[0] + 1:
                    continue
                dt1, dt2 = (current[1] - before[1]) / 1e6, (after[1] - current[1]) / 1e6
                for phase in progress:
                    poses = [row[2][phase] for row in (before, current, after)]
                    roots = [(pose[11] + pose[12]) / 2 for pose in poses]
                    accel = 2 * ((roots[2] - roots[1]) / dt2 - (roots[1] - roots[0]) / dt1) / (dt1 + dt2)
                    centered = [pose - root for pose, root in zip(poses, roots)]
                    body_accel = 2 * ((centered[2] - centered[1]) / dt2 - (centered[1] - centered[0]) / dt1) / (dt1 + dt2)
                    per_phase[phase]["root_acceleration"].append(np.linalg.norm(accel))
                    per_phase[phase]["body_acceleration"].extend(np.linalg.norm(body_accel, axis=-1))
        timings = {phase: progress[phase]["cameras"][camera_name]["seconds"] for phase in progress}
        result["cameras"][camera_name] = dict(seconds=timings, speedup=timings["baseline30"] / timings["paired10"],
            real_rgb_frames={phase: progress[phase]["cameras"][camera_name]["real_rgb_frames"] for phase in progress},
            matched_real_person_anchors=matched, unmatched_candidate_person_anchors=unmatched,
            exact_same_rgb_person_anchors=exact_count, baseline_nearest_time_error_us=statistics(errors),
            same_actual_rgb_2d_input_distance_px=statistics(exact_kp_distance),
            geometric_correspondence_changes=correspondence_changes,
            baseline_temporal_segment_boundaries=segment_boundaries,
            metrics={phase: {"coco17_reprojection_px": statistics(values["reprojection"]),
                             "ankle_reprojection_px": statistics(values["ankles"]),
                             "pelvis_acceleration_m_s2": statistics(values["root_acceleration"]),
                             "root_centered_coco17_acceleration_m_s2": statistics(values["body_acceleration"])}
                     for phase, values in per_phase.items()})
    write_progress_json(directory / "anchor_comparison.json", dict(recipe=binding(recipe_path), anchors=detailed))
    result["anchor_comparison"] = binding(directory / "anchor_comparison.json")
    write_progress_json(directory / "comparison.json", result)
    return result
