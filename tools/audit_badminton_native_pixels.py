"""Audit every test native image and seal independent corrections after unchanged person admission."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import av
import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.dual_view import DualViewGate, detection_reasons, file_sha
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import load_run_session, read_json
from hmr4d.lidar_mocap.rgb_observations import RGBKeypointChecker


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def compare_admissions(before, after, gate, test_ids):
    views = [
        {cam: {p["motion_id"]: p for p in version[cam]["athletes"]} for cam in ("cam1", "cam2")}
        for version in (before, after)
    ]
    ids = set(views[0]["cam1"])
    if any(set(people) != ids for version in views for people in version.values()):
        raise ValueError("Full-frame person sets differ between cameras or rechecks")
    checks = []
    for motion in sorted(ids):
        qa = {version[cam][motion]["fit_accepted"] for version in views for cam in ("cam1", "cam2")}
        if len(qa) != 1:
            raise ValueError("Body QA eligibility changed")
        body_qa = qa.pop()
        states = []
        for version in views:
            reasons = {cam: detection_reasons(version[cam][motion], gate) for cam in ("cam1", "cam2")}
            states.append(dict(admitted=bool(body_qa and not any(reasons.values())), reasons=reasons))
        if states[0]["admitted"] != (motion in test_ids):
            raise ValueError("Original bilateral admission differs from immutable test membership")
        checks.append(
            dict(
                motion_id=motion,
                body_qa_accepted=body_qa,
                before=states[0],
                after=states[1],
                unchanged=states[0]["admitted"] == states[1]["admitted"],
            )
        )
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-manifest",
        type=Path,
        default=PROJ_ROOT / "outputs/badminton_body_repair_v2_paired/training_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJ_ROOT / "outputs/badminton_native_pixel_audit_20261018")
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = read_json(args.training_manifest)
    paired = read_json(manifest["source_manifest"])
    if file_sha(manifest["source_manifest"]) != manifest["source_manifest_sha256"]:
        raise ValueError("Published paired manifest changed")
    metadata = Path(manifest["suggested_session_splits"]["test"]["metadata"])
    if file_sha(metadata) != manifest["suggested_session_splits"]["test"]["metadata_sha256"]:
        raise ValueError("Immutable test metadata changed")
    output = args.output_dir.resolve()
    source = Path(manifest["source_run"]).resolve()
    if (
        not output.is_relative_to(PROJ_ROOT / "outputs")
        or output.is_relative_to(args.training_manifest.resolve().parent)
        or output.is_relative_to(source)
    ):
        raise ValueError("Audit outputs must be separate project outputs")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "native_pixel_corrections.json"
    if destination.exists():
        raise ValueError("Pixel correction supplements are immutable")
    with metadata.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    requests, test_people, published_reports = {}, {}, {}
    for row in rows:
        sid, scan = row["session_id"], row["lidar_index"]
        test_people.setdefault((sid, scan), set()).add(row["motion_id"])
        for cam in ("cam1", "cam2"):
            view = row["dualcam"]["views"][cam]
            native = row["cameras"][cam]
            record = dict(
                scene_id=sid,
                scan=scan,
                camera=cam,
                native=native,
                expected=view["raw_image_sha256"],
                report_path=view["report_path"],
                report_sha256=view["report_sha256"],
            )
            group = requests.setdefault((sid, cam), {})
            i = native["record_index"]
            if i in group and group[i] != record:
                raise ValueError("Test people disagree about one native frame")
            group[i] = record
            published_reports.setdefault((sid, cam), binding(view["report_path"]))
            if published_reports[sid, cam]["sha256"] != view["report_sha256"]:
                raise ValueError("Published native report changed")
    calibration = CalibrationSnapshot.load(manifest["calibration_snapshot"])
    sessions = {sid: load_run_session(source, calibration, sid) for sid in {sid for sid, _ in requests}}
    audit = dict(
        schema="native-pixel-audit/v1",
        status="running",
        metadata=binding(metadata),
        training_manifest=binding(args.training_manifest),
        source_code=binding(__file__),
        av_version=av.__version__,
        av_libraries=av.library_versions,
        unique_scans=len(test_people),
        checked_camera_frames=0,
        expected_camera_frames=sum(map(len, requests.values())),
        camera_results={},
        mismatches=[],
    )
    actual_pixels = {}
    audit_path = output / "test_native_pixel_audit.json"
    for (sid, cam), group in requests.items():
        session = sessions[sid]
        mismatch_count = 0
        for n, (i, bgr) in enumerate(session.iter_video_frames(cam, sorted(group)), 1):
            expected = group[i]
            native = session.camera_rows(cam)[i]
            if any(native[k] != expected["native"][k] for k in ("media_pts_us", "common_time_us", "record_index")):
                raise ValueError("Published native PTS/common time changed")
            actual = hashlib.sha256(bgr.tobytes()).hexdigest()
            actual_pixels[sid, cam, i] = actual
            audit["checked_camera_frames"] += 1
            if actual != expected["expected"]:
                mismatch_count += 1
                audit["mismatches"].append(dict(expected, actual=actual))
            if n % 1000 == 0:
                print(json.dumps(dict(camera=cam, checked=n, total=len(group), mismatches=mismatch_count)), flush=True)
        audit["camera_results"][sid + ":" + cam] = dict(checked=len(group), mismatches=mismatch_count)
    audit["status"] = "complete"
    write_json(audit_path, audit)
    affected = {(r["scene_id"], r["scan"]) for r in audit["mismatches"]}
    old_records = {}
    for (sid, cam), entry in published_reports.items():
        report = read_json(entry["path"])
        journal = Path(report["frames_jsonl"])
        if file_sha(journal) != report["frames_jsonl_sha256"]:
            raise ValueError("Published native journal changed")
        with journal.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if (sid, record["lidar_index"]) in affected:
                    old_records[sid, record["lidar_index"], cam] = record
    gate = DualViewGate(**paired["gate"])
    checker = RGBKeypointChecker(calibration, threads=4, device="cpu", observe_pose=False) if affected else None
    evidence_files, changes = {}, []
    for sid, scan in sorted(affected):
        before = {cam: old_records[sid, scan, cam] for cam in ("cam1", "cam2")}
        reference = before["cam1"]
        frame = dict(
            lidar_index=scan,
            scan_mid_us=reference["scan_mid_us"],
            cameras=reference["paired_cameras"],
            athletes=[
                dict(motion_id=p["motion_id"], fit_path=p["fit_path"], accepted=p["fit_accepted"])
                for p in reference["athletes"]
            ],
        )
        after = {}
        checker.sessions[sid] = sessions[sid]
        for cam in ("cam1", "cam2"):
            checker.select_camera(cam)
            i = frame["cameras"][cam]["record_index"]
            bgr = sessions[sid].read_image(cam, i)
            if hashlib.sha256(bgr.tobytes()).hexdigest() != actual_pixels[sid, cam, i]:
                raise ValueError("Random-seek and complete-stream pixel decoding disagree")
            after[cam] = checker.check_frame(
                dict(id="native_pixel_recheck", session_id=sid, output_root=str(source)), frame, raw_bgr=bgr
            )
            previous = {p["motion_id"]: p for p in before[cam]["athletes"]}
            if any(p["fit_sha256"] != previous[p["motion_id"]]["fit_sha256"] for p in after[cam]["athletes"]):
                raise ValueError("Immutable SMPL fit changed during native recheck")
        checks = compare_admissions(before, after, gate, test_people[sid, scan])
        unchanged = all(item["unchanged"] for item in checks)
        evidence = dict(
            schema="native-pixel-person-admission-recheck/v1",
            scene_id=sid,
            lidar_index=scan,
            scan_mid_us=frame["scan_mid_us"],
            metadata=audit["metadata"],
            training_manifest=audit["training_manifest"],
            gate=asdict(gate),
            models=checker.models,
            code={
                name: binding(PROJ_ROOT / name)
                for name in (
                    "tools/audit_badminton_native_pixels.py",
                    "hmr4d/lidar_mocap/rgb_observations.py",
                    "hmr4d/lidar_mocap/sessions.py",
                    "hmr4d/lidar_mocap/calibration.py",
                )
            },
            original_reports={cam: published_reports[sid, cam] for cam in ("cam1", "cam2")},
            original_views=before,
            verified_views=after,
            people=checks,
            all_person_admissions_unchanged=unchanged,
        )
        path = output / "evidence" / f"{sid}_scan{scan:06d}.json"
        write_json(path, evidence)
        evidence_files[sid, scan] = binding(path)
        if not unchanged:
            changes.append(evidence_files[sid, scan])
        print(json.dumps(dict(rechecked_scene=sid, scan=scan, people=len(checks), unchanged=unchanged)), flush=True)
    if changes:
        write_json(
            output / "admission_changes.json",
            dict(
                status="admission_changes_detected",
                cases=changes,
                all_person_admissions_unchanged=False,
                audit=binding(audit_path),
            ),
        )
        raise ValueError("Some person admissions changed; no pixel correction override was published")
    corrections = []
    for row in audit["mismatches"]:
        native = row["native"]
        corrections.append(
            dict(
                scene_id=row["scene_id"],
                camera=row["camera"],
                record_index=native["record_index"],
                media_pts_us=native["media_pts_us"],
                common_time_us=native["common_time_us"],
                source_frame_no=native["source_frame_no"],
                previous_raw_bgr_sha256=row["expected"],
                verified_raw_bgr_sha256=row["actual"],
                evidence=evidence_files[row["scene_id"], row["scan"]],
                original_report=dict(path=row["report_path"], sha256=row["report_sha256"]),
            )
        )
    result = dict(
        schema="native-pixel-corrections/v1",
        status="complete",
        created_utc=datetime.now(timezone.utc).isoformat(),
        metadata=audit["metadata"],
        training_manifest=audit["training_manifest"],
        all_person_admissions_unchanged=True,
        corrections=corrections,
        full_native_pixel_audit=binding(audit_path),
        rechecked_person_frames=sum(len(read_json(entry["path"])["people"]) for entry in evidence_files.values()),
        reason="Historical native pixel digests are not reproducible with current exact-PTS decoding; images and complete bilateral person admission rechecked without changing published labels or membership.",
    )
    write_json(destination, result)
    print(
        json.dumps(
            dict(
                path=str(destination),
                sha256=file_sha(destination),
                corrections=len(corrections),
                checked_camera_frames=audit["checked_camera_frames"],
                all_person_admissions_unchanged=True,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
