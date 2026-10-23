"""Close expanded pixel differences with fresh both-camera all-person admission.

Freeze native-PTS decoded pixels and recheck their exact retained snapshots.
Local live-decoder differences remain explicit; frozen metadata/labels stay intact.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.dual_view import DualViewGate
from hmr4d.lidar_mocap.expanded_dataset import binding, check_binding
from hmr4d.lidar_mocap.rally_pipeline import load_run_session, read_json
from hmr4d.lidar_mocap.repair_execution import require_gpu_lease, write_progress_json as write_json
from hmr4d.lidar_mocap.relocations import load_pilot_run_relocations
from hmr4d.lidar_mocap.rgb_observations import RGBKeypointChecker
from audit_badminton_native_pixels import compare_admissions


def recheck(audit_path, output, lease_path, device):
    audit_path, output = Path(audit_path).resolve(), Path(output).resolve()
    audit = read_json(audit_path)
    for name in ("assembly", "metadata", "native_pixels"):
        check_binding(audit[name])
    if audit["checked_camera_frames"] != audit["expected_camera_frames"]:
        raise ValueError("Complete native pixel coverage is required")
    assembly = read_json(audit["assembly"]["path"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh independent pixel admission evidence directory")
    output.mkdir(parents=True, exist_ok=True)
    if device != "cpu":
        require_gpu_lease(lease_path)
    requests = {}
    with Path(audit["native_pixels"]["path"]).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            requests.setdefault((record["scene_id"], record["camera"]), {})[record["native"]["record_index"]] = record
    changed = {(r["scene_id"], r["camera"], r["native"]["record_index"]) for r in audit["unknown_differences"]}
    changed.update((r["scene_id"], r["camera"], r["record_index"]) for r in audit["corrections"])
    people, canonical_people, scans, wanted, reports = {}, {}, {}, {}, {}
    with Path(audit["metadata"]["path"]).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sid = row["session_id"]
            scan = row["lidar_index"]
            if not any((sid, cam, row["cameras"][cam]["record_index"]) in changed for cam in ("cam1", "cam2")):
                continue
            key = sid, scan
            source_motion = row.get("canonical_identity", {}).get("source_motion_id", row["motion_id"])
            people.setdefault(key, set()).add(source_motion)
            canonical_people.setdefault(key, set()).add(row["motion_id"])
            scans[key] = row["cameras"]
            for cam in ("cam1", "cam2"):
                wanted.setdefault((sid, cam), set()).add(row["cameras"][cam]["record_index"])
                view = row["dualcam"]["views"][cam]
                reports[sid, cam] = dict(path=view["report_path"], sha256=view["report_sha256"])
    if not scans:
        raise ValueError("No changed native images belong to admitted test scans")
    sessions, calibrations, resolvers = {}, {}, {}
    captures = {}
    source_videos = {}
    for sid in sorted({sid for sid, _ in scans}):
        source = assembly["source_registry"][sid]
        calibration = CalibrationSnapshot.load(source["calibration_snapshot"])
        if calibration.data["snapshot_digest"] != source["calibration_digest"]:
            raise ValueError("Source camera snapshot changed")
        calibrations[sid] = calibration
        run = Path(source["source_run"]).resolve()
        run_key = run, calibration.data["snapshot_digest"]
        if run_key not in resolvers:
            resolvers[run_key] = load_pilot_run_relocations(run, calibration.data)
        sessions[sid] = load_run_session(run, calibration, sid, relocations=resolvers[run_key])
        for cam in ("cam1", "cam2"):
            source_videos[sid + ":" + cam] = binding(sessions[sid].video_paths[cam])
            selected = requests[sid, cam]
            needed = wanted[sid, cam]
            found = set()
            for index, bgr in sessions[sid].iter_video_frames(cam, sorted(selected)):
                if index in needed:
                    actual = hashlib.sha256(bgr.tobytes()).hexdigest()
                    path = output / "native_pixels" / sid / cam / f"{index:07d}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                        raise ValueError("Could not retain exact native evidence pixels")
                    captures[sid, cam, index] = dict(
                        image=binding(path),
                        raw_bgr_sha256=actual,
                        first_audit_raw_bgr_sha256=selected[index]["actual_raw_bgr_sha256"],
                        native=selected[index]["native"],
                        source_video=source_videos[sid + ":" + cam],
                    )
                    found.add(index)
                if found == needed:
                    break
            if found != needed:
                raise ValueError("Did not capture every affected native view")
            print(dict(captured_source=sid, camera=cam, frames=len(found)), flush=True)
    before_records = {}
    for (sid, cam), entry in reports.items():
        check_binding(entry)
        report = read_json(entry["path"])
        check_binding(dict(path=report["frames_jsonl"], sha256=report["frames_jsonl_sha256"]))
        with Path(report["frames_jsonl"]).open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if (sid, record["lidar_index"]) in scans:
                    before_records[sid, record["lidar_index"], cam] = record
    evidence, failures = {}, []
    checkers = {}
    for number, (sid, scan) in enumerate(sorted(scans), 1):
        calibration = calibrations[sid]
        digest = calibration.data["snapshot_digest"]
        if digest not in checkers:
            checkers[digest] = RGBKeypointChecker(calibration, threads=1, device=device, observe_pose=False)
        checker = checkers[digest]
        checker.sessions[sid] = sessions[sid]
        source = assembly["source_registry"][sid]
        paired = read_json(source["paired_manifest"]["path"])
        gate = DualViewGate(**paired["gate"])
        before = {cam: before_records[sid, scan, cam] for cam in ("cam1", "cam2")}
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
        for cam in ("cam1", "cam2"):
            index = frame["cameras"][cam]["record_index"]
            capture = captures[sid, cam, index]
            check_binding(capture["image"])
            bgr = cv2.imread(capture["image"]["path"])
            if hashlib.sha256(bgr.tobytes()).hexdigest() != capture["raw_bgr_sha256"]:
                raise ValueError("Retained native BGR evidence changed")
            checker.select_camera(cam)
            after[cam] = checker.check_frame(
                dict(id="expanded_native_pixel_recheck", session_id=sid, output_root=source["source_run"]),
                frame,
                raw_bgr=bgr,
            )
            if after[cam]["native_bgr_image_sha256"] != capture["raw_bgr_sha256"]:
                raise ValueError("Checker did not evaluate the captured audit pixels")
            old = {p["motion_id"]: p for p in before[cam]["athletes"]}
            if any(p["fit_sha256"] != old[p["motion_id"]]["fit_sha256"] for p in after[cam]["athletes"]):
                raise ValueError("Frozen SMPL fit changed")
        checks = compare_admissions(before, after, gate, people[sid, scan])
        unchanged = all(p["unchanged"] for p in checks)
        proof = dict(
            schema="native-pixel-person-admission-recheck/v1",
            scene_id=sid,
            lidar_index=scan,
            metadata=audit["metadata"],
            assembly=audit["assembly"],
            original_reports={cam: reports[sid, cam] for cam in ("cam1", "cam2")},
            original_views=before,
            verified_views=after,
            people=checks,
            canonical_admitted_motion_ids=sorted(canonical_people[sid, scan]),
            all_person_admissions_unchanged=unchanged,
            gate=asdict(gate),
            models=checker.models,
            captured_pixels={
                cam: captures[sid, cam, frame["cameras"][cam]["record_index"]] for cam in ("cam1", "cam2")
            },
        )
        path = output / "evidence" / f"{sid}_{scan:06d}.json"
        write_json(path, proof)
        evidence[sid, scan] = binding(path)
        if not unchanged:
            failures.append(evidence[sid, scan])
        write_json(
            output / "progress.json",
            dict(
                state="rechecking_full_person_admissions",
                completed=number,
                total=len(scans),
                changed_admission_cases=len(failures),
            ),
        )
        print(
            json.dumps(
                dict(scan=scan, source=sid, completed=number, total=len(scans), people=len(checks), unchanged=unchanged)
            ),
            flush=True,
        )
    corrections = []
    for (sid, scan), views in scans.items():
        for cam, native in views.items():
            key = sid, cam, native["record_index"]
            if (
                key not in changed
                and captures[key]["raw_bgr_sha256"] == requests[sid, cam][native["record_index"]]["expected"]
            ):
                continue
            changed.add(key)
            request = requests[sid, cam][native["record_index"]]
            corrections.append(
                dict(
                    scene_id=sid,
                    camera=cam,
                    **request["native"],
                    previous_raw_bgr_sha256=request["expected"],
                    verified_raw_bgr_sha256=captures[key]["raw_bgr_sha256"],
                    first_audit_raw_bgr_sha256=request["actual_raw_bgr_sha256"],
                    verified_pixel_capture=captures[key],
                    original_report=request["report"],
                    evidence=evidence[sid, scan],
                )
            )
    if {(c["scene_id"], c["camera"], c["record_index"]) for c in corrections} != changed:
        raise ValueError("A changed native image lacks full-person recheck evidence")
    for name in ("assembly", "metadata", "native_pixels"):
        check_binding(audit[name])
    producer = output / "producer.py"
    shutil.copyfile(__file__, producer)
    result = dict(
        schema="badminton-expanded-pixel-admission-recheck-v1",
        status="complete" if not failures else "admissions_changed",
        created_utc=datetime.now(timezone.utc).isoformat(),
        assembly=audit["assembly"],
        metadata=audit["metadata"],
        initial_pixel_audit=binding(audit_path),
        producer=binding(producer),
        checked_camera_frames=audit["checked_camera_frames"],
        all_person_admissions_unchanged=not failures,
        affected_scans=len(scans),
        capture_policy="Freeze exact native-PTS BGR in lossless PNG; snapshot replay is bit-exact. Live decoder bit reproducibility is not asserted.",
        captured_views=[
            dict(scene_id=sid, camera=cam, record_index=index, **capture)
            for (sid, cam, index), capture in captures.items()
        ],
        source_videos=source_videos,
        rechecked_person_frames=sum(len(read_json(p["path"])["people"]) for p in evidence.values()),
        corrections=corrections,
        admission_changes=failures,
    )
    write_json(output / "acceptance.json", result)
    if failures:
        raise ValueError("Bilateral admissions changed; no correction sidecar can be published")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-lease", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if not args.output_root.resolve().is_relative_to(PROJ_ROOT / "outputs"):
        parser.error("Use a fresh independent project outputs directory")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    result = recheck(args.audit, args.output_root, args.gpu_lease, args.device)
    print(
        json.dumps(
            dict(
                status=result["status"], affected_scans=result["affected_scans"], corrections=len(result["corrections"])
            )
        )
    )


if __name__ == "__main__":
    main()
