"""Generate guarded native-leg correspondence candidates without changing labels."""

import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
import smplx
import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.leg_refinement import leg_ray_hypotheses, refine_leg_candidates
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json, read_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2")
    p.add_argument("--wait", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    labels = root / "final_labels"
    torch.set_num_threads(2)
    plan = read_json(labels / "repair_plan.json")
    out = root / "leg_refinement_proposals"
    out.mkdir(exist_ok=True)
    cal = CalibrationSnapshot.load(labels / "calibration/calibration_snapshot.json")
    policy = PROJ_ROOT / "hmr4d/lidar_mocap/leg_refinement.py"
    policy_sha = file_sha(policy)
    model = (
        smplx.create(PROJ_ROOT / "inputs/checkpoints/body_models", model_type="smpl", gender="neutral")
        .eval()
        .cuda()
        .requires_grad_(False)
    )
    reg = torch.load(
        PROJ_ROOT / "hmr4d/utils/body_model/smpl_coco17_J_regressor.pt", map_location="cpu", weights_only=False
    ).numpy()
    receipt_path = out / "receipt.json"
    receipt = read_json(receipt_path) if receipt_path.exists() else dict(windows={}, state="partial")
    started = time.perf_counter()
    while len(receipt["windows"]) < len(plan["jobs"]):
        changed = False
        for job in plan["jobs"]:
            sid = job["source_id"]
            window = Path(job["source_window"]).name
            key = sid + ":" + window
            if key in receipt["windows"]:
                continue
            source = labels / "sessions" / sid / "windows" / window
            if not (source / "fit_report.json").exists():
                continue
            directory = out / sid / window
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "proposals.json"
            counts = Counter()
            proposals = []
            for row in read_rows(source / "quality.jsonl"):
                if not row["accepted"]:
                    continue
                ev = row.get("native_pose_evidence") or {}
                if not all(cam in ev.get("files", {}) and cam + "_detection" in ev for cam in ["cam1", "cam2"]):
                    continue
                obs = {
                    cam: np.asarray(
                        read_json(ev["files"][cam]["path"])["keypoints"][ev[cam + "_detection"]], np.float32
                    )
                    for cam in ["cam1", "cam2"]
                }
                check = leg_ray_hypotheses(obs["cam1"], obs["cam2"], cal)
                if check is None:
                    continue
                counts["ray_witness_cases"] += 1
                with np.load(row["fit_path"]) as f:
                    sample = {name: f[name].copy() for name in f.files}
                candidates = refine_leg_candidates(sample, obs, cal, model, reg)
                entries = []
                for i, c in enumerate(candidates):
                    arrays = {name: c.pop(name) for name in ["vertices", "joints", "body_pose"]}
                    artifact = directory / f'{row["lidar_index"]:06d}_{row["motion_id"].split(":")[-2]}_{i}.npz'
                    np.savez_compressed(artifact, **arrays)
                    entries.append(dict(c, arrays_path=str(artifact), arrays_sha256=file_sha(artifact)))
                    counts["accepted_candidate_hypotheses"] += int(c["accepted"])
                proposals.append(
                    dict(
                        lidar_index=row["lidar_index"],
                        motion_id=row["motion_id"],
                        scan_mid_us=row["scan_mid_us"],
                        source_fit_path=row["fit_path"],
                        source_fit_sha256=file_sha(row["fit_path"]),
                        native_pose_evidence=ev,
                        ray_witness=check,
                        candidates=entries,
                    )
                )
                print(
                    dict(
                        source=sid,
                        window=window,
                        scan=row["lidar_index"],
                        accepted_candidates=sum(c["accepted"] for c in entries),
                    ),
                    flush=True,
                )
            write_json(
                target,
                dict(
                    policy_sha256=policy_sha,
                    source_fit_report_sha256=file_sha(source / "fit_report.json"),
                    proposals=proposals,
                    counts=dict(counts),
                ),
            )
            receipt["windows"][key] = dict(path=str(target), sha256=file_sha(target), counts=dict(counts))
            receipt.update(
                policy_sha256=policy_sha,
                source_union_sha256=plan["source_union_sha256"],
                calibration_digest=plan["calibration_digest"],
                windows_complete=len(receipt["windows"]),
                windows_total=len(plan["jobs"]),
                training_published=False,
                applied=False,
            )
            write_json(receipt_path, receipt)
            changed = True
            print(
                dict(proposed_windows=len(receipt["windows"]), wall_seconds=time.perf_counter() - started), flush=True
            )
        if not args.wait:
            break
        if not changed:
            time.sleep(5)
    receipt["state"] = "complete" if len(receipt["windows"]) == len(plan["jobs"]) else "partial"
    write_json(receipt_path, receipt)


if __name__ == "__main__":
    main()
