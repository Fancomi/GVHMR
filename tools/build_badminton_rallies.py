"""Build SMPL/LiDAR labels for the seven-source sealed rally intake using local uv.

Rerun the same command to validate and resume matching per-window caches.
Never modifies frozen calibration, upstream indexes or original recordings.
"""

import argparse
import json
from pathlib import Path
import time

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.rally_pipeline import RallyPipeline


def publish(pipeline):
    from hmr4d.lidar_mocap.dual_view import audit_cached, export_reviewed, verify_export
    from hmr4d.lidar_mocap.dual_view_observer import observe_all
    from hmr4d.lidar_mocap.rally_pipeline import read_json
    from hmr4d.lidar_mocap.pipeline import write_json

    root = pipeline.output_root
    paired = root.with_name(root.name + "_paired")
    if (paired / "datasets/dataset_manifest.json").exists():
        verify_export(paired)
        result = read_json(paired / "datasets/dataset_manifest.json")
    else:
        result = None
    # Old exact-native observations are imported only after source/time/fit
    # matching. New sources fill missing YOLO detections without pose refitting.
    cache = list((PROJ_ROOT / "outputs/badminton_dualcam_paired_full_v2/observations").glob("*_cam[12].json"))
    if result is None:
        observations = observe_all(root, paired, cache_reports=cache)
        if not observations["completed"]:
            raise ValueError("Native person observations incomplete")
        audit_cached(root, paired, observations["reports"], disk_evidence=True)
        result = export_reviewed(
            root, paired, detection_only=True, full_observation_manifest=paired / "observations/manifest.json"
        )
    state = read_json(root / "build_status.json")
    for sid, entry in result["exports"].items():
        before = state["sources"][sid]["accepted"]
        state["sources"][sid].update(
            state="complete", paired_accepted=entry["accepted"], bilateral_excluded=before - entry["accepted"]
        )
    state.update(
        state="complete",
        paired_accepted=result["accepted"],
        paired_output_root=str(paired),
        dataset_manifest=str(paired / "datasets/dataset_manifest.json"),
    )
    write_json(root / "build_status.json", state)
    # Consumer requested a machine-readable lease release once production GPU
    # stages finish. Never publish release on a failed/incomplete production.
    lease = Path("D:/PROJECT/LIDAR/rollout_lidar_mocap_badminton/.local/gvhmr_data_scale_20261015/gpu_lease.json")
    if lease.parent.is_dir():
        write_json(lease, dict(status="released", owner="GVHMR", note="production GPU completed"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, default=PROJ_ROOT / "outputs/upstream_rally_intake_v1/verification.json")
    parser.add_argument("--output-root", type=Path, default=PROJ_ROOT / "outputs/badminton_rallies_v1")
    parser.add_argument("--baseline-root", type=Path, default=PROJ_ROOT / "outputs/badminton_corner_fix_v1_continuity")
    parser.add_argument("--lidar-project-root", type=Path, default=Path("D:/PROJECT/LIDAR/lidar_project"))
    parser.add_argument("--stage", choices=("prepare", "teacher", "fit", "run", "export", "publish"), default="run")
    parser.add_argument("--wait", action="store_true", help="Publish only: wait for the running body build to finish")
    parser.add_argument(
        "--threads", type=int, default=8, help="CPU tensor threads; limits oversubscription during production"
    )
    args = parser.parse_args()
    published = args.output_root.with_name(args.output_root.name + "_paired") / "datasets/dataset_manifest.json"
    if published.exists() and args.stage in ("run", "fit", "teacher", "export"):
        from hmr4d.lidar_mocap.dual_view import verify_export

        print(json.dumps(verify_export(published.parent.parent), indent=2), flush=True)
        return
    if args.wait:
        if args.stage != "publish":
            parser.error("--wait requires --stage publish")
        status = args.output_root / "build_status.json"
        while True:
            state = json.loads(status.read_text(encoding="utf-8")) if status.exists() else {}
            if state.get("state") in ("body_labels_complete", "complete"):
                break
            if any(s.get("state") == "failed" for s in state.get("sources", {}).values()):
                raise RuntimeError("Body build failed; resolve its actual error and resume before publishing")
            print(
                json.dumps(dict(state="waiting_for_body_build", current_source=state.get("current_source"))), flush=True
            )
            time.sleep(30)
    import torch
    import cv2

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(2)
    pipeline = RallyPipeline(args.intake, args.output_root, args.baseline_root, args.lidar_project_root)
    if args.stage == "prepare":
        result = pipeline.prepare()
    elif args.stage == "export":
        pipeline.prepare()
        result = pipeline.aggregate()
    elif args.stage == "publish":
        pipeline.prepare()
        result = publish(pipeline)
    else:
        result = pipeline.build(args.stage)
        if args.stage in ("run", "fit"):
            result = publish(pipeline)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
