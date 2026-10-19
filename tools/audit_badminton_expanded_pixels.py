"""Audit every held-out native pixel through its own source calibration/run.

Historical corrections may be bridged only with unchanged original metadata and
hash-bound all-person admission evidence. Unexpected differences fail acceptance.
No tensor, metadata, camera or published historical evidence is rewritten.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import av

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.expanded_dataset import binding, check_binding
from hmr4d.lidar_mocap.rally_pipeline import load_run_session, read_json, read_rows
from hmr4d.lidar_mocap.repair_execution import write_progress_json as write_json


NATIVE_FIELDS = ("record_index", "media_pts_us", "common_time_us", "source_frame_no")


def pixel_requests(rows, registry):
    requests = defaultdict(dict)
    for row in rows:
        sid = row["session_id"]
        if registry[sid]["role"] != "test" or registry[sid]["calibration_digest"] != row["calibration_digest"]:
            raise ValueError("Test row uses the wrong source role or calibration")
        for cam in ("cam1", "cam2"):
            view = row["dualcam"]["views"][cam]
            native = {key: view["native"][key] for key in NATIVE_FIELDS}
            if any(row["cameras"][cam][key] != native[key] for key in NATIVE_FIELDS):
                raise ValueError("Bilateral evidence and paired native image disagree")
            request = dict(
                scene_id=sid,
                camera=cam,
                native=native,
                expected=view["raw_image_sha256"],
                report=dict(path=view["report_path"], sha256=view["report_sha256"]),
            )
            index = native["record_index"]
            group = requests[sid, cam]
            if index in group and group[index] != request:
                raise ValueError("People sharing one native image disagree about its evidence")
            group[index] = request
    return dict(requests)


def historical_corrections(path, assembly):
    if path is None:
        return {}
    correction = read_json(path)
    if (
        correction.get("schema") != "native-pixel-corrections/v1"
        or correction.get("status") != "complete"
        or correction.get("all_person_admissions_unchanged") is not True
    ):
        raise ValueError("Historical supplement is not a complete admission-preserving correction")
    for name in ("metadata", "training_manifest", "full_native_pixel_audit"):
        check_binding(correction[name])
    components = assembly["suggested_session_splits"]["test"]["components"]
    if correction["metadata"] not in [entry["metadata"] for entry in components]:
        raise ValueError("Historical corrected metadata is not an exact unchanged expansion component")
    result = {}
    for item in correction["corrections"]:
        key = item["scene_id"], item["camera"], item["record_index"]
        if key in result:
            raise ValueError("Duplicate historical pixel correction")
        if assembly["source_registry"][key[0]]["generation"] != "old":
            raise ValueError("Historical correction cannot be applied to a new source")
        check_binding(item["evidence"])
        check_binding(item["original_report"])
        evidence = read_json(item["evidence"]["path"])
        if (
            evidence["metadata"] != correction["metadata"]
            or evidence.get("all_person_admissions_unchanged") is not True
            or not all(person.get("unchanged") is True for person in evidence["people"])
        ):
            raise ValueError("Historical full-person admission proof changed")
        for name, digest in (
            ("original_views", "previous_raw_bgr_sha256"),
            ("verified_views", "verified_raw_bgr_sha256"),
        ):
            view = evidence[name][item["camera"]]
            if view["native_bgr_image_sha256"] != item[digest] or any(
                view["camera_native_source"][field] != item[field] for field in NATIVE_FIELDS
            ):
                raise ValueError("Historical pixel proof refers to another native image")
        result[key] = item
    return result


def accepted_pixel(request, actual, corrections):
    if actual == request["expected"]:
        return None
    key = request["scene_id"], request["camera"], request["native"]["record_index"]
    item = corrections.get(key)
    if (
        item is None
        or item["previous_raw_bgr_sha256"] != request["expected"]
        or item["verified_raw_bgr_sha256"] != actual
        or any(item[field] != request["native"][field] for field in NATIVE_FIELDS)
    ):
        raise ValueError("Unapproved native pixel difference: " + str(key))
    return item


def audit(assembly_path, output, correction_path=None):
    assembly_path, output = Path(assembly_path).resolve(), Path(output).resolve()
    frozen = binding(assembly_path)
    assembly = read_json(assembly_path)
    if assembly.get("schema") != "badminton-expanded-assembly-v1" or assembly.get("training_published"):
        raise ValueError("Audit an explicit unpublished full expansion assembly")
    if output.is_relative_to(assembly_path.parent) or assembly_path.parent.is_relative_to(output):
        raise ValueError("Native audit evidence must be separate from the expansion publication")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh independent audit output")
    test = assembly["suggested_session_splits"]["test"]
    for entry in assembly["source_publications"].values():
        check_binding(entry)
    if file_sha(test["metadata"]) != test["metadata_sha256"]:
        raise ValueError("Expansion test metadata changed")
    rows = read_rows(test["metadata"])
    if len(rows) != test["accepted"]:
        raise ValueError("Expansion test metadata count changed")
    requests = pixel_requests(rows, assembly["source_registry"])
    corrections = historical_corrections(correction_path, assembly)
    output.mkdir(parents=True, exist_ok=True)
    source_copy = output / "producer.py"
    shutil.copyfile(__file__, source_copy)
    receipt = dict(
        schema="badminton-expanded-native-pixel-audit-v1",
        status="running",
        assembly=frozen,
        created_utc=datetime.now(timezone.utc).isoformat(),
        metadata=binding(test["metadata"]),
        producer=binding(source_copy),
        av_version=av.__version__,
        av_libraries=av.library_versions,
        unique_scans=len({(r["session_id"], r["lidar_index"]) for r in rows}),
        expected_camera_frames=sum(map(len, requests.values())),
        checked_camera_frames=0,
        source_results={},
        corrections=[],
        unknown_differences=[],
        historical_supplement=binding(correction_path) if correction_path else None,
    )
    reports, used = set(), set()
    journal_path = output / "native_pixels.jsonl"
    with journal_path.open("w", encoding="utf-8") as journal:
        for (sid, cam), group in requests.items():
            source = assembly["source_registry"][sid]
            if file_sha(source["calibration_snapshot"]) != source["calibration_snapshot_sha256"]:
                raise ValueError("Source-specific calibration snapshot changed")
            cal = CalibrationSnapshot.load(source["calibration_snapshot"])
            if cal.data["snapshot_digest"] != source["calibration_digest"]:
                raise ValueError("Source registry and calibration disagree")
            session = load_run_session(Path(source["source_run"]), cal, sid)
            native_rows, checked, applied = session.camera_rows(cam), set(), 0
            for index, bgr in session.iter_video_frames(cam, sorted(group)):
                request = group[index]
                report = request["report"]
                if report["path"] not in reports:
                    check_binding(report)
                    reports.add(report["path"])
                if index in checked or any(native_rows[index][k] != request["native"][k] for k in NATIVE_FIELDS):
                    raise ValueError("Duplicate image or changed native PTS/frame identity")
                actual = hashlib.sha256(bgr.tobytes()).hexdigest()
                record = dict(request, actual_raw_bgr_sha256=actual)
                try:
                    correction = accepted_pixel(request, actual, corrections)
                except ValueError as error:
                    receipt["unknown_differences"].append(dict(record, error=str(error)))
                    correction = None
                if correction is not None:
                    key = sid, cam, index
                    used.add(key)
                    receipt["corrections"].append(correction)
                    applied += 1
                    record["admission_preserving_evidence"] = correction["evidence"]
                journal.write(json.dumps(record) + "\n")
                checked.add(index)
                receipt["checked_camera_frames"] += 1
                if len(checked) % 500 == 0:
                    write_json(
                        output / "progress.json",
                        dict(
                            checked=receipt["checked_camera_frames"],
                            expected=receipt["expected_camera_frames"],
                            current_source=sid,
                            camera=cam,
                            unknown_differences=len(receipt["unknown_differences"]),
                        ),
                    )
            if checked != set(group):
                raise ValueError("Native decoder did not return every requested image")
            receipt["source_results"][sid + ":" + cam] = dict(
                checked=len(checked),
                bridged_corrections=applied,
                source_run=source["source_run"],
                calibration_digest=source["calibration_digest"],
            )
            print(dict(source=sid, camera=cam, checked=len(checked), bridged=applied), flush=True)
    check_binding(frozen)
    receipt.update(
        status="complete" if not receipt["unknown_differences"] else "unapproved_pixel_differences",
        native_pixels=binding(journal_path),
        all_person_admissions_unchanged=not receipt["unknown_differences"],
        unused_historical_corrections=[list(key) for key in corrections if key not in used],
    )
    if receipt["checked_camera_frames"] != receipt["expected_camera_frames"]:
        raise ValueError("Native camera coverage is incomplete")
    write_json(output / "audit.json", receipt)
    if receipt["status"] != "complete":
        raise ValueError("Unexpected native pixel changes need fresh both-camera full-person admission review")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--historical-corrections", type=Path)
    args = parser.parse_args()
    if not args.output_root.resolve().is_relative_to(PROJ_ROOT / "outputs"):
        parser.error("Audit output must stay inside workspace outputs")
    receipt = audit(args.assembly, args.output_root, args.historical_corrections)
    print(json.dumps(dict(status=receipt["status"], checked=receipt["checked_camera_frames"])))


if __name__ == "__main__":
    main()
