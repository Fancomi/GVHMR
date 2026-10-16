"""Rebuild fixed-camera teacher/body labels in a separate, resumable repair version.

Uses the frozen union's original window/scans and exact native pixel caches.
Unchanged temporal inputs reuse their prediction; changed context reruns GVHMR.
Output exports pass body QA only until independently admitted by dual-view export.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import time

import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.body_repair import published_windows
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.fitting import FittingConfig, SMPLFitter
from hmr4d.lidar_mocap.pipeline import BadmintonPipeline, write_json
from hmr4d.lidar_mocap.rally_pipeline import load_run_session, read_json
from hmr4d.lidar_mocap.teacher import _jsonable, run_teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--union", type=Path, default=PROJ_ROOT / "outputs/badminton_full_plus_new_v1/dataset_manifest.json"
    )
    parser.add_argument("--source-root", type=Path, default=PROJ_ROOT / "outputs/badminton_rallies_v1")
    parser.add_argument("--output-root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v1/labels")
    parser.add_argument("--source", action="append")
    parser.add_argument("--window", action="append")
    parser.add_argument("--stage", choices=("prepare", "teacher", "fit", "run"), default="run")
    parser.add_argument("--distal-pose-weight", type=float, default=0.03)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = args.output_root.resolve()
    jobs = [
        (sid, window)
        for sid, window in published_windows(args.union)
        if (not args.source or sid in args.source) and (not args.window or window.name in args.window)
    ]
    if not jobs or not root.is_relative_to((PROJ_ROOT / "outputs").resolve()):
        raise ValueError("Select existing frozen windows and a separate workspace output root")
    if any(root == window or root in window.parents or window in root.parents for _, window in jobs):
        raise ValueError("Repair output must not contain or overwrite a frozen source window")
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "calibration/calibration_snapshot.json").exists():
        shutil.copytree(args.source_root / "calibration", root / "calibration")
    if not (root / "source_registry.json").exists():
        shutil.copyfile(args.source_root / "source_registry.json", root / "source_registry.json")
    calibration = CalibrationSnapshot.load(root / "calibration/calibration_snapshot.json")
    source_calibration = CalibrationSnapshot.load(args.source_root / "calibration/calibration_snapshot.json")
    if calibration.data["snapshot_digest"] != source_calibration.data["snapshot_digest"]:
        raise ValueError("Body repair cannot change cameras/calibration")
    cfg = FittingConfig(distal_pose_prior_weight=args.distal_pose_weight)
    plan = dict(
        schema="badminton-body-repair-plan-v1",
        source_union=str(args.union.resolve()),
        source_union_sha256=file_sha(args.union),
        calibration_digest=calibration.data["snapshot_digest"],
        fitting_config=asdict(cfg),
        training_published=False,
        jobs=[
            dict(source_id=sid, source_window=str(window), window_json_sha256=file_sha(window / "window.json"))
            for sid, window in jobs
        ],
    )
    write_json(root / "repair_plan.json", plan)
    if args.stage == "prepare":
        print(f"Prepared {len(jobs)} frozen windows; no inference or training publication", flush=True)
        return
    pipeline = BadmintonPipeline(calibration.data["sessions_root"], "D:/PROJECT/LIDAR/lidar_project", root)
    pipeline.calibration = calibration
    sessions = {}
    completed = []
    for sid, original in jobs:
        output = root / "sessions" / sid / "windows" / original.name
        output.mkdir(parents=True, exist_ok=True)
        config = read_json(original / "window.json")
        if config["calibration_digest"] != plan["calibration_digest"]:
            raise ValueError("Original window uses another calibration")
        pipeline.settings = dict(config["settings"], share_track_shape=True)
        manifest = read_json(original / "teacher/manifest.json")
        if _jsonable(calibration.camera(pipeline.settings["camera"])) != manifest["provenance"]["camera"]:
            raise ValueError("Original teacher camera differs from the frozen repair camera")
        if read_json(original / "fit_report.json")["teacher_provenance_hash"] != manifest["provenance_hash"]:
            raise ValueError("Original fit and teacher provenance differ")
        options = dict(manifest["provenance"]["options"], max_track_gap_frames=6)
        if args.stage == "fit" and not (output / "teacher/manifest.json").exists():
            raise ValueError("Fit requires the completed repaired teacher stage")
        teacher = run_teacher(
            manifest["provenance"]["video"]["path"],
            manifest["provenance"]["frame_rows"],
            manifest["provenance"]["camera"],
            output / "teacher",
            reuse_preprocess_dir=original / "teacher",
            **options,
        )
        if args.stage == "teacher":
            completed.append(dict(source_id=sid, window_id=original.name, tracks=len(teacher["tracks"])))
            write_json(
                root / "repair_teacher_progress.json",
                dict(
                    completed_windows=len(completed),
                    total_windows=len(jobs),
                    last_window=str(output),
                    training_published=False,
                ),
            )
            continue
        report_path = output / "fit_report.json"
        cached = read_json(report_path) if report_path.exists() else None
        if cached and (
            cached.get("fit_algorithm_digest") == pipeline._fit_digest()
            and cached.get("fitting_config") == asdict(cfg)
            and cached.get("settings") == pipeline.settings
            and cached.get("teacher_provenance_hash") == teacher["provenance_hash"]
            and cached.get("artifacts_sha256")
            and all(file_sha(output / p) == sha for p, sha in cached["artifacts_sha256"].items())
            and all(file_sha(Path(p)) == sha for p, sha in cached.get("source_lidar_sha256", {}).items())
        ):
            completed.append(cached)
            print(f"[validated resume] {sid}/{original.name}", flush=True)
            continue
        if sid not in sessions:
            sessions[sid] = load_run_session(root, calibration, sid)
        session = sessions[sid]
        selected = set(config["source_pairs"])
        pairs = [p for p in session.pairs() if p["lidar_index"] in selected]
        if {p["lidar_index"] for p in pairs} != selected:
            raise ValueError("Native source scan coverage changed")
        write_json(
            output / "window.json",
            dict(
                config,
                settings=pipeline.settings,
                body_repair=dict(source_window=str(original), source_union_sha256=plan["source_union_sha256"]),
            ),
        )
        started = time.perf_counter()
        fitter = SMPLFitter(device=args.device, config=cfg)
        report = pipeline._fit_window(session, pairs, teacher, output, fitter)
        completed.append(report)
        del fitter
        if args.device == "cuda":
            torch.cuda.empty_cache()
        write_json(
            root / "repair_progress.json",
            dict(
                completed_windows=len(completed),
                total_windows=len(jobs),
                last_window=str(output),
                last_wall_seconds=time.perf_counter() - started,
                training_published=False,
            ),
        )
    if args.stage == "teacher":
        write_json(
            root / "repair_teacher_receipt.json",
            dict(
                state="complete",
                windows=len(completed),
                source_union_sha256=plan["source_union_sha256"],
                jobs=completed,
                training_published=False,
            ),
        )
        print(f"Complete: {len(completed)} repaired native teacher windows; fitted labels remain separate", flush=True)
    if args.stage in ("fit", "run"):
        result = pipeline.aggregate(sorted({sid for sid, _ in jobs}))
        write_json(
            root / "repair_receipt.json",
            dict(
                state="body_qa_candidate_complete",
                accepted=result["accepted"],
                windows=len(completed),
                training_published=False,
                body_qa_only=True,
                note="Requires dual-camera admission/export verification before training publication",
            ),
        )
        print(f"Complete: {result['accepted']} body-QA candidate frames; original union unchanged", flush=True)


if __name__ == "__main__":
    main()
