"""Render actual native RGB witnesses for explicit fresh-source identity cases."""

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import load_run_session, read_json
from hmr4d.lidar_mocap.relocations import load_pilot_run_relocations
from hmr4d.lidar_mocap.teacher import _undistorter


def render(root, indices):
    root = Path(root).resolve()
    plan = read_json(root / "new_source_input.json")
    source = Path(plan["source_root"])
    cal = CalibrationSnapshot.load(source / "calibration/calibration_snapshot.json")
    relocations = load_pilot_run_relocations(source, cal.data)
    cases = {c["index"]: c for c in read_json(root / "identity_evidence/boundaries.json")}
    sessions, reports = {}, {}
    output = root / "identity_evidence/case_previews"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in indices:
        existing = output / f"case{index:04d}.json"
        if existing.exists():
            receipt = read_json(existing)
            if (
                receipt["source_input_sha256"] != file_sha(root / "new_source_input.json")
                or file_sha(receipt["image"]) != receipt["image_sha256"]
            ):
                raise ValueError("Existing immutable native case evidence changed")
            rows.append(receipt)
            print(dict(case=index, state="existing_evidence_verified"), flush=True)
            continue
        case = cases[index]
        sid = case["event"]["source_id"]
        report_path = root / "identity_evidence/source_reviews" / (sid + ".json")
        if not report_path.exists():
            report_path = root / "identity_evidence/native_identity_review.json"
        if sid not in reports:
            reports[sid] = read_json(report_path)
            sessions[sid] = load_run_session(source, cal, sid, relocations=relocations)
        review = next(c for c in reports[sid]["cases"] if c["index"] == index)
        session, tiles, records = sessions[sid], [], []
        for side, position, vote in zip(("before", "after"), case["positions"], review["votes"]):
            timeline = [r for r in review["timeline"] if r["original_track_id"] == vote["original_track_id"]]
            if not timeline:
                continue
            selected = min(timeline, key=lambda r: abs(r["position"] - position))
            for cam in ("cam2", "cam1"):
                native_index = selected["native_" + cam]
                pose_path = root / "identity_evidence/native_poses" / sid / cam / f"{native_index:07d}.json"
                pose = read_json(pose_path)
                image = session.read_image(cam, native_index)
                raw_sha = hashlib.sha256(image.tobytes()).hexdigest()
                if raw_sha != pose["raw_bgr_sha256"]:
                    raise ValueError("Preview decoder returned pixels different from the native pose witness")
                image = _undistorter(cal.camera(cam))(image)
                detection = selected[cam + "_detection"]
                box = np.asarray(pose["boxes"][detection], dtype=float) if detection is not None else None
                if box is not None:
                    center = (box[:2] + box[2:]) / 2
                    width, height = (box[2:] - box[:2]) * [2, 1.4]
                    x0, y0 = np.maximum(np.floor(center - [width, height] / np.float64(2)), 0).astype(int)
                    x1, y1 = np.minimum(
                        np.ceil(center + [width, height] / np.float64(2)), [image.shape[1], image.shape[0]]
                    ).astype(int)
                    image = image[y0:y1, x0:x1]
                scale = min(600 / image.shape[1], 400 / image.shape[0])
                image = cv2.resize(image, None, fx=scale, fy=scale)
                tile = np.full((450, 620, 3), 240, dtype=np.uint8)
                tile[45 : 45 + image.shape[0], (620 - image.shape[1]) // 2 : (620 + image.shape[1]) // 2] = image
                title = f"{side} {cam} raw ID {vote['original_track_id']} frame {native_index}"
                cv2.putText(tile, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1)
                tiles.append(tile)
                records.append(
                    dict(
                        side=side,
                        camera=cam,
                        native=pose["native"],
                        raw_bgr_sha256=raw_sha,
                        pose_path=str(pose_path),
                        pose_sha256=file_sha(pose_path),
                        chosen_position=selected["position"],
                        requested_endpoint_position=position,
                    )
                )
        if not tiles:
            continue
        canvas = np.concatenate(tiles, axis=1)
        path = output / f"case{index:04d}.jpg"
        if not cv2.imwrite(str(path), canvas):
            raise ValueError("Native identity preview write failed")
        receipt = dict(
            case=index,
            source_id=sid,
            geometric_status=review["status"],
            review_path=str(report_path),
            review_sha256=file_sha(report_path),
            source_input_sha256=file_sha(root / "new_source_input.json"),
            calibration_digest=plan["calibration_digest"],
            image_space="undistorted_K_preserved_native_crops",
            native=records,
            image=str(path),
            image_sha256=file_sha(path),
        )
        write_json(output / f"case{index:04d}.json", receipt)
        rows.append(receipt)
        print(dict(case=index, image=str(path)), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", type=int, action="append", required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    render(args.root, args.case)


if __name__ == "__main__":
    main()
