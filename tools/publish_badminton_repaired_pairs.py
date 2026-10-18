"""Observe repaired labels with native detections, then use the standard bilateral publisher."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.dual_view import (
    DualViewGate,
    _bound_output,
    _canonical,
    audit_cached,
    export_reviewed,
    file_sha,
    verify_export,
)
from hmr4d.lidar_mocap.dual_view_observer import _resume_rows, selected_frames, source_selection
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import load_run_session, read_json
from hmr4d.lidar_mocap.repaired_pairing import reassociate_native_frame
from hmr4d.lidar_mocap.rgb_observations import RGBKeypointChecker


def observe(source, output, cache_roots, device):
    dataset, inputs = source_selection(source)
    calibration = CalibrationSnapshot.load(source / "calibration/calibration_snapshot.json")
    reports = {}
    for root in cache_roots:
        manifest = read_json(root / "observations/manifest.json")
        if not manifest["completed"]:
            raise ValueError("Native detector cache is incomplete")
        for path in manifest["reports"]:
            report = read_json(path)
            reports.setdefault((report["session_id"], report["camera_name"]), (Path(path), report))
    selected_reports = [reports[sid, cam] for sid in dataset["exports"] for cam in ("cam1", "cam2")]
    cache_identity = [
        dict(path=str(p), sha256=file_sha(p), frames_sha256=d["frames_jsonl_sha256"]) for p, d in selected_reports
    ]
    identity = dict(
        source=inputs,
        calibration_digest=dataset["calibration_digest"],
        cached_reports=cache_identity,
        reassociation_sha256=file_sha(PROJ_ROOT / "hmr4d/lidar_mocap/repaired_pairing.py"),
        observer_sha256=file_sha(__file__),
        device=device,
    )
    digest = hashlib.sha256(_canonical(identity)).hexdigest()
    target = output / "observations/manifest.json"
    if target.exists():
        previous = read_json(target)
        if previous["identity_sha256"] != digest:
            raise ValueError("Observation inputs or code changed; use a separate run")
        if previous["completed"]:
            return previous
    state = dict(
        schema="badminton-full-native-rgb-observations-v1",
        completed=False,
        created_utc=datetime.now(timezone.utc).isoformat(),
        identity=identity,
        identity_sha256=digest,
        reports=[],
        source_accepted=dataset["accepted"],
        method="repaired_body_native_detection_reassociation",
        models=selected_reports[0][1]["models"],
        missing_native_cache_frames=0,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, state)
    fallback = None
    for sid in dataset["exports"]:
        session = load_run_session(source, calibration, sid)
        frames = selected_frames(source, dataset, sid)
        expected = {f["lidar_index"]: f for f in frames}
        for cam in ("cam1", "cam2"):
            old_path, old = reports[sid, cam]
            video = session.video_paths[cam]
            stat = video.stat()
            if old["source_video"] != dict(path=str(video), size=stat.st_size, mtime_ns=stat.st_mtime_ns):
                raise ValueError("Immutable native video identity changed")
            if old["calibration_digest"] != dataset["calibration_digest"] or old["camera"] != calibration.camera(cam):
                raise ValueError("Immutable detector calibration changed")
            old_rows = Path(old["frames_jsonl"])
            if file_sha(old_rows) != old["frames_jsonl_sha256"]:
                raise ValueError("Immutable native detection journal changed")
            cached = {}
            with old_rows.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row["lidar_index"] in expected:
                        cached[row["lidar_index"]] = {
                            k: v for k, v in row.items() if k not in ("athletes", "detector_observations")
                        }
            evidence = dict(
                report_path=str(old_path),
                report_sha256=file_sha(old_path),
                frames_path=str(old_rows),
                frames_sha256=old["frames_jsonl_sha256"],
                models=old["models"],
                checker_sha256=old["checker_sha256"],
            )
            path = output / "observations" / f"{sid}_{cam}.json"
            rows_path = path.with_suffix(".jsonl")
            done, count, people, seconds = _resume_rows(rows_path, expected, cam)
            missing = sum(f["lidar_index"] not in cached for f in frames)
            state["missing_native_cache_frames"] += missing
            report = dict(
                old,
                completed=False,
                identity_sha256=digest,
                source_video=old["source_video"],
                frames_jsonl=str(rows_path),
                planned_camera_frames=len(frames),
                planned_accepted_person_frames=dataset["exports"][sid]["accepted"],
                completed_camera_frames=count,
                observed_accepted_person_frames=people,
                checker_sha256=identity["reassociation_sha256"],
                device="cpu",
                inference_seconds=seconds,
                immutable_native_source=evidence,
                missing_native_cache_frames=missing,
            )
            state["reports"].append(str(path))
            print(
                json.dumps(dict(source=sid, camera=cam, planned=len(frames), missing_cache=missing, resumed=count)),
                flush=True,
            )
            with rows_path.open("a", encoding="utf-8") as handle:
                for frame in frames:
                    scan = frame["lidar_index"]
                    if scan in done:
                        continue
                    if scan in cached:
                        record = reassociate_native_frame(frame, cached[scan], cam, calibration, evidence)
                    else:
                        if fallback is None:
                            fallback = RGBKeypointChecker(calibration, threads=4, device=device, observe_pose=False)
                        fallback.sessions[sid] = session
                        fallback.select_camera(cam)
                        record = fallback.check_frame(
                            dict(id="repaired_all_native_scans", session_id=sid, output_root=str(source)), frame
                        )
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                    handle.flush()
                    report["completed_camera_frames"] += 1
                    report["observed_accepted_person_frames"] += sum(p["fit_accepted"] for p in record["athletes"])
                    report["inference_seconds"] += record["timing_seconds"]["total"]
                    if report["completed_camera_frames"] % 100 == 0:
                        write_json(path, report)
                        write_json(target, state)
                        print(
                            json.dumps(
                                dict(
                                    source=sid,
                                    camera=cam,
                                    completed=report["completed_camera_frames"],
                                    planned=len(frames),
                                )
                            ),
                            flush=True,
                        )
            report.update(completed=True, frames_jsonl_sha256=file_sha(rows_path))
            write_json(path, report)
            write_json(target, state)
    state["completed"] = True
    write_json(target, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("observe", "publish", "verify"))
    parser.add_argument("--source-root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2/final_labels")
    parser.add_argument("--output-root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2_paired")
    parser.add_argument("--cache-root", type=Path, action="append")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    source, output = _bound_output(args.source_root, args.output_root)
    if args.stage == "verify":
        print(json.dumps(verify_export(output)), flush=True)
        return
    if (output / "datasets/dataset_manifest.json").exists():
        raise ValueError("Published paired datasets are immutable")
    if args.stage == "observe":
        roots = args.cache_root or [
            PROJ_ROOT / "outputs/badminton_dualcam_paired_full_v2",
            PROJ_ROOT / "outputs/badminton_rallies_v1_paired",
        ]
        result = observe(source, output, roots, args.device)
    else:
        path = output / "observations/manifest.json"
        manifest = read_json(path)
        if not manifest["completed"]:
            raise ValueError("All native observations must complete first")
        audit_cached(
            source, output, manifest["reports"], DualViewGate(policy="bilateral_detection"), disk_evidence=True
        )
        result = export_reviewed(source, output, detection_only=True, full_observation_manifest=path)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
