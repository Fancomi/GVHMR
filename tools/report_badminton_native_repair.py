"""Measure old/new geometry on identical native keypoints and repaired point sets."""

import argparse
import base64
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.fitting import point_to_surface_squared
from hmr4d.lidar_mocap.leg_refinement import unordered_leg_errors
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json, read_rows


def stats(values):
    a = np.asarray(values, float)
    return (
        dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)), p90=float(np.percentile(a, 90)))
        if len(a)
        else dict(count=0)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = p.parse_args()
    torch.set_num_threads(2)
    root = args.root
    labels = root / "final_labels"
    plan = read_json(labels / "repair_plan.json")
    cases = read_json(root / "case_plan.json")
    cal = CalibrationSnapshot.load(labels / "calibration/calibration_snapshot.json")
    viewer = read_json(PROJ_ROOT / "outputs/badminton_rallies_v1/viewer/manifest.json")
    faces = torch.as_tensor(
        np.frombuffer(base64.b64decode(viewer["faces"]), dtype="<u2").astype(np.int64), device=args.device
    ).reshape(-1, 3)
    regressor = torch.load(
        PROJ_ROOT / "hmr4d/utils/body_model/smpl_coco17_J_regressor.pt", map_location="cpu", weights_only=False
    ).numpy()
    evidence = root / "evaluation"
    evidence.mkdir(exist_ok=True)
    out = evidence / "metrics.json"
    result = read_json(out) if out.exists() else dict(windows={}, state="partial")
    if result.get("metric_schema_version") != 2:
        result = dict(windows={}, state="partial", metric_schema_version=2)
    for key in list(result["windows"]):
        sid, window = key.split(":")
        report = labels / "sessions" / sid / "windows" / window / "fit_report.json"
        if result["windows"][key].get("fit_report_sha256") != file_sha(report):
            result["windows"].pop(key)
    started = time.perf_counter()
    while len(result["windows"]) < len(plan["jobs"]):
        changed = False
        for job in plan["jobs"]:
            sid = job["source_id"]
            old = Path(job["source_window"])
            key = sid + ":" + old.name
            if key in result["windows"]:
                continue
            new = labels / "sessions" / sid / "windows" / old.name
            if not (new / "fit_report.json").exists():
                continue
            selected = set(cases["selected_scans"][sid])
            after = read_rows(new / "quality.jsonl")
            before = read_rows(old / "quality.jsonl")
            teacher = read_json(new / "teacher/manifest.json")
            aliases = {raw: t["track_id"] for t in teacher["tracks"] for raw in t["source_tracklets"]}
            lookup = defaultdict(list)
            for row in before:
                lookup[
                    row["lidar_index"],
                    aliases.get(int(row["motion_id"].split(":")[-2]), int(row["motion_id"].split(":")[-2])),
                ].append(row)
            observations = []
            counts = Counter()
            for row in after:
                if row["lidar_index"] not in selected:
                    continue
                counts["selected_new_fits"] += 1
                physical = int(row["motion_id"].split(":")[-2])
                oldrows = lookup[row["lidar_index"], physical]
                if len(oldrows) != 1:
                    counts["missing_or_duplicate_old_identity"] += 1
                    continue
                oldrow = oldrows[0]
                counts["matched_same_identity_scan"] += 1
                oldfit = np.load(oldrow["fit_path"])
                newfit = np.load(row["fit_path"])
                v0 = oldfit["vertices"]
                v1 = newfit["vertices"]
                cloud = newfit["source_cloud"]
                with torch.no_grad():
                    ds = point_to_surface_squared(
                        torch.as_tensor(np.stack([v0, v1]), device=args.device),
                        faces,
                        [torch.as_tensor(cloud, device=args.device), torch.as_tensor(cloud, device=args.device)],
                    )
                common = [
                    dict(
                        mean_mm=float(d.sqrt().mean().cpu() * 1000),
                        p90_mm=float(torch.quantile(d.sqrt(), 0.9).cpu() * 1000),
                    )
                    for d in ds
                ]
                record = dict(
                    scan=row["lidar_index"],
                    old_motion=oldrow["motion_id"],
                    new_motion=row["motion_id"],
                    old_accepted=oldrow["accepted"],
                    new_accepted=row["accepted"],
                    common_visible_points=len(cloud),
                    common_points_source="new_fit_actual_assigned_source_cloud",
                    surface_before=common[0],
                    surface_after=common[1],
                    cameras={},
                )
                pose = row.get("native_pose_evidence") or {}
                for cam in ["cam1", "cam2"]:
                    if cam not in pose.get("files", {}) or cam + "_detection" not in pose:
                        continue
                    raw = read_json(pose["files"][cam]["path"])
                    kp = np.asarray(raw["keypoints"][pose[cam + "_detection"]], float)
                    original_kp = kp.copy()
                    mapping = pose.get("leg_joint_permutation")
                    if mapping and mapping["camera"] == cam:
                        kp[mapping["original_indices"]] = kp[mapping["indices"]]
                    valid = kp[:, 2] >= 0.5
                    values = []
                    for vertices in [v0, v1]:
                        uv, depth = cal.project(regressor @ vertices, cam, distort=False)
                        distance = np.linalg.norm(uv - kp[:, :2], axis=1)
                        usable = valid & (depth > 0)
                        ankles = usable.copy()
                        ankles[:15] = False
                        values.append(
                            dict(
                                all_mean_px=float(distance[usable].mean()) if usable.any() else None,
                                ankle_mean_px=float(distance[ankles].mean()) if ankles.any() else None,
                                unordered_leg_mean_px=unordered_leg_errors(
                                    vertices, {cam: original_kp}, cal, regressor
                                )[cam],
                                raw_ankle_mean_px=(
                                    float(
                                        np.linalg.norm(uv - original_kp[:, :2], axis=1)[
                                            (original_kp[:, 2] >= 0.5) & (np.arange(17) >= 15) & (depth > 0)
                                        ].mean()
                                    )
                                    if ((original_kp[:, 2] >= 0.5) & (np.arange(17) >= 15) & (depth > 0)).any()
                                    else None
                                ),
                                joint_count=int(usable.sum()),
                                ankle_count=int(ankles.sum()),
                            )
                        )
                    record["cameras"][cam] = dict(
                        before=values[0],
                        after=values[1],
                        native_record=raw["native"]["record_index"],
                        raw_unmasked_observations=True,
                        native_joint_label_mapping=mapping,
                        rejected_ray_joints=pose.get("inconsistent_joint_rays", []),
                    )
                if sid == "20261009_202713" and physical == 12 and row["lidar_index"] in [1533, 1535, 1543, 1553, 1563]:
                    stages = np.load(
                        PROJ_ROOT / "outputs/badminton_rallies_v1/viewer/quality_audit_v1/conversion_geometry.npz"
                    )
                    i = list(stages["scans"]).index(row["lidar_index"])
                    mask = stages["weights"][:, 10] > 0.25
                    reference = stages["converted"][i, mask]
                    reference = reference - reference.mean(0)
                    record["left_forefoot_direction_change_vs_same_converted_reference_deg"] = [
                        float(np.degrees(Rotation.align_vectors(v[mask] - v[mask].mean(0), reference)[0].magnitude()))
                        for v in [v0, v1]
                    ]
                observations.append(record)
                oldfit.close()
                newfit.close()
            dest = evidence / (sid + "_" + old.name + ".json")
            write_json(dest, observations)
            result["windows"][key] = dict(
                counts=dict(counts),
                observations_path=str(dest),
                fitted_before=len(before),
                fitted_after=len(after),
                fit_report_sha256=file_sha(new / "fit_report.json"),
                accepted_before=sum(r["accepted"] for r in before),
                accepted_after=sum(r["accepted"] for r in after),
            )
            changed = True
            allrows = [r for w in result["windows"].values() for r in read_json(w["observations_path"])]
            cohorts = {
                "all_matched": allrows,
                "both_accepted": [r for r in allrows if r["old_accepted"] and r["new_accepted"]],
            }
            result["cohorts"] = {
                name: dict(
                    person_scans=len(rows),
                    common_surface_mm={
                        v: stats([r["surface_" + v]["mean_mm"] for r in rows]) for v in ["before", "after"]
                    },
                    camera_px={
                        cam: {
                            metric: {
                                v: stats(
                                    [
                                        r["cameras"][cam][v][metric]
                                        for r in rows
                                        if cam in r["cameras"] and r["cameras"][cam][v][metric] is not None
                                    ]
                                )
                                for v in ["before", "after"]
                            }
                            for metric in ["all_mean_px", "ankle_mean_px", "unordered_leg_mean_px", "raw_ankle_mean_px"]
                        }
                        for cam in ["cam1", "cam2"]
                    },
                )
                for name, rows in cohorts.items()
            }
            result.update(
                verified_windows=len(result["windows"]),
                total_windows=len(plan["jobs"]),
                source_union_sha256=cases["source_union_sha256"],
                calibration_digest=cal.data["snapshot_digest"],
                scope="All selected scans: all 587 ankle-review flags and neighbours, track boundaries and ten continuous clips.",
                limits=[
                    "Common points are real assigned points from the new version; ownership is not manual ground truth.",
                    "Both native cameras participate in fitting. Raw high-confidence pose measurements include excluded contradictory joint rays.",
                    "Forefoot angles measure direction change from a common conversion reference, not anatomical error.",
                ],
            )
            write_json(out, result)
            js = root / "comparison/quality.js"
            js.parent.mkdir(exist_ok=True)
            temp = js.with_suffix(".tmp")
            temp.write_text(
                "window.REPAIR_QUALITY=" + __import__("json").dumps(result, separators=(",", ":")) + ";",
                encoding="utf-8",
            )
            temp.replace(js)
            print(
                dict(
                    evaluated_windows=len(result["windows"]),
                    common_person_scans=len(allrows),
                    wall_seconds=time.perf_counter() - started,
                ),
                flush=True,
            )
        if not args.wait:
            break
        if not changed:
            time.sleep(5)
    result["state"] = "complete" if len(result["windows"]) == len(plan["jobs"]) else "partial"
    write_json(out, result)


if __name__ == "__main__":
    main()
