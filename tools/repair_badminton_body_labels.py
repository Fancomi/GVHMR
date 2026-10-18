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
from hmr4d.lidar_mocap.pose_provider import NativePoseProvider
from hmr4d.lidar_mocap.segment_cache import FrozenSegmentCache
from hmr4d.lidar_mocap.pose_ray_filter import RayFilteredPoseProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--union", type=Path, default=PROJ_ROOT / "outputs/badminton_full_plus_new_v1/dataset_manifest.json"
    )
    parser.add_argument("--source-root", type=Path, default=PROJ_ROOT / "outputs/badminton_rallies_v1")
    parser.add_argument("--output-root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v1/labels")
    parser.add_argument("--source", action="append")
    parser.add_argument("--window", action="append")
    parser.add_argument("--case-plan", type=Path)
    parser.add_argument("--stage", choices=("prepare", "teacher", "fit", "run"), default="run")
    parser.add_argument("--distal-pose-weight", type=float, default=0.03)
    parser.add_argument("--lower-limb-reprojection-weight", type=float, default=1.0)
    parser.add_argument("--native-poses-root", type=Path)
    parser.add_argument("--joint-ray-filter", action="store_true")
    parser.add_argument("--seed-teacher-root", type=Path)
    parser.add_argument("--identity-decisions", type=Path)
    parser.add_argument('--partition-worker',action='store_true',help='Disjoint one-source worker; preserve the full plan and defer aggregate export')
    parser.add_argument("--no-segment-cache", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    import cv2
    cv2.setNumThreads(args.threads)
    fresh_sources = read_json(args.union).get("schema") == "badminton-new-source-repair-input-v1"
    if fresh_sources:
        prepared = read_json(args.union)
        if (args.source_root.resolve() != Path(prepared["source_root"]).resolve()
                or file_sha(args.source_root / "source_registry.json") != prepared["source_registry_sha256"]):
            raise ValueError("New-source source-root/registry differs from its prepared input")
    root = args.output_root.resolve()
    if (root/'post_refinement.json').exists():
        raise ValueError('This candidate root contains sealed postprocessed labels; use a fresh output root for baseline fitting')
    jobs = [
        (sid, window)
        for sid, window in published_windows(args.union)
        if (not args.source or sid in args.source) and (not args.window or window.name in args.window)
    ]
    if args.case_plan:
        cases = read_json(args.case_plan)
        if cases["source_union_sha256"] != file_sha(args.union):
            raise ValueError("Case plan uses another union")
        chosen = {(c["source_id"], Path(c["source_window"]).name) for c in cases["clips"]}
        jobs = [(sid, w) for sid, w in jobs if (sid, w.name) in chosen]
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
    cfg = FittingConfig(
        distal_pose_prior_weight=args.distal_pose_weight,
        lower_limb_reprojection_weight=args.lower_limb_reprojection_weight,
    )
    provider = RayFilteredPoseProvider if args.joint_ray_filter else NativePoseProvider
    poses = provider(args.native_poses_root, calibration) if args.native_poses_root and args.stage != "prepare" else None
    if fresh_sources and args.stage not in ("prepare", "teacher"):
        if not args.joint_ray_filter or not args.native_poses_root or not args.identity_decisions:
            raise ValueError("New-source fitting requires native dual poses, ray filtering and reviewed identity evidence")
        identity = read_json(args.identity_decisions)
        if identity.get("review_status") != "reviewed" or not identity.get("evidence_files"):
            raise ValueError("New-source identity evidence remains pending")
        required = {sid + ":" + window.name for sid, window in jobs}
        if not required <= set(identity.get("windows", {})):
            raise ValueError("Some selected new-source identity windows remain pending")
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
    if args.partition_worker:
        if not args.source or len(args.source)!=1 or not (root/'repair_plan.json').exists():
            raise ValueError('A partition worker requires one explicit source and an existing full plan')
        full=read_json(root/'repair_plan.json')
        if full['source_union_sha256']!=plan['source_union_sha256'] or not {(j['source_id'],j['source_window']) for j in plan['jobs']} <= {(j['source_id'],j['source_window']) for j in full['jobs']}:
            raise ValueError('Partition is not part of the frozen full plan')
    else:
        if fresh_sources and (root / "repair_plan.json").exists() and read_json(root / "repair_plan.json") != plan:
            raise ValueError("New-source repair inputs/settings changed; use a new root")
        write_json(root / "repair_plan.json", plan)
    if args.stage == "prepare":
        print(f"Prepared {len(jobs)} frozen windows; no inference or training publication", flush=True)
        return
    pipeline = BadmintonPipeline(calibration.data["sessions_root"], "D:/PROJECT/LIDAR/lidar_project", root)
    pipeline.calibration = calibration
    sessions = {}
    completed = []
    decisions = read_json(args.identity_decisions) if args.identity_decisions else None
    if decisions and decisions["source_union_sha256"] != plan["source_union_sha256"]:
        raise ValueError("Identity decisions refer to another delivered union")
    if decisions and (
        decisions["calibration_digest"] != plan["calibration_digest"]
        or any(file_sha(Path(p)) != sha for p, sha in decisions["evidence_files"].items())
    ):
        raise ValueError("Sealed native identity evidence changed")
    for sid, original in jobs:
        output = root / "sessions" / sid / "windows" / original.name
        output.mkdir(parents=True, exist_ok=True)
        config = read_json(original / "window.json")
        if config["calibration_digest"] != plan["calibration_digest"]:
            raise ValueError("Original window uses another calibration")
        pipeline.settings = dict(config["settings"], share_track_shape=True)
        if poses is not None:
            pipeline.settings["native_pose_receipt_sha256"] = poses.digest
        if args.joint_ray_filter:
            pipeline.settings["native_pose_joint_ray_policy_sha256"] = poses.policy_sha256
        manifest = read_json(original / "teacher/manifest.json")
        if _jsonable(calibration.camera(pipeline.settings["camera"])) != manifest["provenance"]["camera"]:
            raise ValueError("Original teacher camera differs from the frozen repair camera")
        if not fresh_sources and read_json(original / "fit_report.json")["teacher_provenance_hash"] != manifest["provenance_hash"]:
            raise ValueError("Original fit and teacher provenance differ")
        options = dict(manifest["provenance"]["options"], max_track_gap_frames=6)
        decision = decisions["windows"].get(sid + ":" + original.name) if decisions else None
        if decision and decision["source_teacher_manifest_sha256"] != file_sha(original / "teacher/manifest.json"):
            raise ValueError("Identity source teacher changed")
        if args.seed_teacher_root and decision is None and not (output / "teacher/manifest.json").exists():
            seeded = args.seed_teacher_root / "sessions" / sid / "windows" / original.name / "teacher"
            shutil.copytree(seeded, output / "teacher", dirs_exist_ok=True)
        if args.stage == "fit" and not (output / "teacher/manifest.json").exists():
            raise ValueError("Fit requires the completed repaired teacher stage")
        teacher = run_teacher(
            manifest["provenance"]["video"]["path"],
            manifest["provenance"]["frame_rows"],
            manifest["provenance"]["camera"],
            output / "teacher",
            reuse_preprocess_dir=original / "teacher",
            identity_decisions=decision,
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
        segments = None if args.no_segment_cache or fresh_sources else FrozenSegmentCache(original, pairs, plan["calibration_digest"])
        report = pipeline._fit_window(
            session, pairs, teacher, output, fitter, segment_cache=segments, pose_provider=poses
        )
        completed.append(report)
        del fitter
        if args.device == "cuda":
            torch.cuda.empty_cache()
        write_json(
            root / (f'repair_progress_{args.source[0]}.json' if args.partition_worker else 'repair_progress.json'),
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
        if args.partition_worker:
            write_json(root/f'repair_receipt_{args.source[0]}.json',dict(state='partition_complete',windows=len(completed),source=args.source[0],training_published=False))
            print(f'Partition complete: {len(completed)} windows; full aggregate deferred',flush=True)
            return
        aggregate_sources = sorted({sid for sid, _ in jobs})
        if fresh_sources:
            # Keep declared zero-rally sources as empty tensors, never fabricate
            # samples or silently drop them from the exhaustive session split.
            aggregate_sources = sorted({sid for ids in read_json(args.union)["session_splits"].values() for sid in ids})
        result = pipeline.aggregate(aggregate_sources)
        if fresh_sources:
            from hmr4d.lidar_mocap.rally_pipeline import validate_session_splits
            splits = validate_session_splits(read_json(args.union)["session_splits"], result["exports"])
            result["suggested_session_splits"] = {
                name: dict(sessions=sids, accepted=sum(result["exports"][sid]["accepted"] for sid in sids))
                for name, sids in splits.items()}
            result["source_registry"] = str(root / "source_registry.json")
            result["source_registry_sha256"] = file_sha(root / "source_registry.json")
            result["body_policy"] = "native_dual_pose_ray_filter_shared_shape_v2_before_leg_postpass"
            result["training_published"] = False
            result["required_nonempty_splits"] = ["train", "val", "test"]
            from hmr4d.lidar_mocap.export import require_nonempty_splits
            require_nonempty_splits(result)
            write_json(root / "datasets/dataset_manifest.json", result)
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
