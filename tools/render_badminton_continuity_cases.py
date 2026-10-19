"""Render native before/after two-camera witnesses of explicit pelvis-jump flags."""

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.expanded_dataset import binding
from hmr4d.lidar_mocap.rally_pipeline import load_run_session, read_json, read_rows
from hmr4d.lidar_mocap.repair_execution import write_progress_json as write_json
from hmr4d.lidar_mocap.teacher import _undistorter


def render(root):
    root = Path(root).resolve()
    audit_path = root / "continuity_audit/audit.json"
    audit = read_json(audit_path)
    labels = root / "final_labels"
    cal = CalibrationSnapshot.load(labels / "calibration/calibration_snapshot.json")
    sessions = {}
    out = root / "continuity_audit/previews"
    out.mkdir(parents=True, exist_ok=True)
    for index, event in enumerate(audit["pelvis_jump_candidates"]):
        sid = event["source_id"]
        if sid not in sessions:
            sessions[sid] = load_run_session(labels, cal, sid)
        rows = read_rows(labels / "sessions" / sid / "windows" / event["window"] / "quality.jsonl")
        selected = [
            next(r for r in rows if r["motion_id"] == event["motion_id"] and r["lidar_index"] == event[key])
            for key in ("before_scan", "after_scan")
        ]
        records, tiles = [], []
        for cam in ("cam1", "cam2"):
            bodies = []
            for row in selected:
                with np.load(row["fit_path"]) as fit:
                    bodies.append(
                        dict(
                            vertices=fit["vertices"].copy(),
                            pelvis=fit["joints"][0].copy(),
                            cloud_center=np.median(fit["source_cloud"], axis=0),
                        )
                    )
            projected = [cal.project(b["vertices"], cam, distort=False)[0] for b in bodies]
            uv = np.concatenate(projected)
            lower = np.maximum(np.floor(uv.min(0) - [90, 90]), 0).astype(int)
            upper = np.minimum(np.ceil(uv.max(0) + [90, 90]), [1920, 1080]).astype(int)
            for side, row, body, points in zip(("before", "after"), selected, bodies, projected):
                record = row["cameras"][cam]["record_index"]
                image = sessions[sid].read_image(cam, record)
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                pose_path = root / "native_poses" / sid / cam / f"{record:07d}.json"
                pose = read_json(pose_path)
                if pose["raw_bgr_sha256"] != digest:
                    raise ValueError("Native continuity witness pixel hash changed")
                image = _undistorter(cal.camera(cam))(image)
                pelvis_uv = cal.project(body["pelvis"][None], cam, distort=False)[0][0]
                cv2.circle(image, tuple(np.rint(pelvis_uv).astype(int)), 9, (0, 220, 255), 2)
                for point in points[::20]:
                    pt = np.rint(point).astype(int)
                    if (pt >= 0).all() and pt[0] < 1920 and pt[1] < 1080:
                        cv2.circle(image, tuple(pt), 1, (255, 180, 0), -1)
                image = image[lower[1] : upper[1], lower[0] : upper[0]]
                scale = min(600 / image.shape[1], 390 / image.shape[0])
                image = cv2.resize(image, None, fx=scale, fy=scale)
                tile = np.full((450, 620, 3), 245, dtype=np.uint8)
                tile[50 : 50 + image.shape[0], (620 - image.shape[1]) // 2 : (620 + image.shape[1]) // 2] = image
                cv2.putText(
                    tile,
                    f'{side} {cam} scan {row["lidar_index"]} jump {event["distance_m"]:.2f}m',
                    (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )
                tiles.append(tile)
                records.append(
                    dict(
                        side=side,
                        camera=cam,
                        native=row["cameras"][cam],
                        raw_bgr_sha256=digest,
                        pose=binding(pose_path),
                        fit=binding(row["fit_path"]),
                        pelvis=body["pelvis"].tolist(),
                        cloud_center=body["cloud_center"].tolist(),
                        teacher_sampling=row["teacher_sampling"],
                        qa=row["qa"],
                    )
                )
        image_path = out / f"case{index:02d}.jpg"
        if not cv2.imwrite(str(image_path), np.concatenate(tiles, axis=1)):
            raise ValueError("Failed to write native witness")
        write_json(
            out / f"case{index:02d}.json",
            dict(case=index, event=event, audit=binding(audit_path), image=binding(image_path), records=records),
        )
        print(dict(case=index, image=str(image_path)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    cv2.setNumThreads(1)
    render(args.root)
