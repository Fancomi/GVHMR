"""Serve versioned repair candidates with the existing native dual-camera/SMPL viewer.

Candidate fit QA counts do not mean bilateral admission or training publication.
Only completed windows under the explicitly selected repair root are included.
"""

import argparse
import copy
from pathlib import Path

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import load_run_session, read_json
from hmr4d.lidar_mocap.rally_viewer import ViewerData, load_display_bbox, serve_viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v1/case_labels")
    parser.add_argument("--source-root", type=Path, default=PROJ_ROOT / "outputs/badminton_rallies_v1")
    parser.add_argument("--display-crop", type=Path)
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    root = args.repair_root.resolve()
    calibration = CalibrationSnapshot.load(root / "calibration/calibration_snapshot.json")
    manifest = copy.deepcopy(read_json(args.source_root / "viewer/manifest.json"))
    if manifest["calibration_digest"] != calibration.data["snapshot_digest"]:
        raise ValueError("Repair and original viewer cameras differ")
    reports = {}
    for path in (root / "sessions").glob("*/windows/*/fit_report.json"):
        reports.setdefault(path.parent.parent.parent.name, []).append(read_json(path))
    if not reports:
        raise ValueError("No completed repair windows")
    manifest["sources"] = {sid: source for sid, source in manifest["sources"].items() if sid in reports}
    sessions, media, state = {}, {}, {}
    total_scans = 0
    for sid, source in manifest["sources"].items():
        source["baseline"] = False
        session = load_run_session(root, calibration, sid)
        sessions[sid] = session
        media.update({(sid, cam): session.video_paths[cam] for cam in ("cam1", "cam2")})
        scans = set()
        for path in (root / "sessions" / sid / "windows").glob("*/quality.jsonl"):
            if not (path.parent / "fit_report.json").exists():
                continue
            import json

            scans.update(json.loads(line)["lidar_index"] for line in path.read_text(encoding="utf-8").splitlines())
        total_scans += len(scans)
        source["rallies"] = [
            clip
            for clip in source["rallies"]
            if any(clip["begin"] <= source["scan_times"][scan] <= clip["end"] for scan in scans)
        ]
        state[sid] = dict(
            baseline=False,
            state="repair_candidate",
            fitted=sum(r["fitted"] for r in reports[sid]),
            accepted=sum(r["accepted"] for r in reports[sid]),
        )
    manifest.update(
        source_count=len(reports),
        rally_count=sum(len(s["rallies"]) for s in manifest["sources"].values()),
        unique_paired_input_scans=total_scans,
        review_only=True,
        rally_window_minutes=sum(c["end"] - c["begin"] for s in manifest["sources"].values() for c in s["rallies"])
        / 60,
    )
    bbox, evidence = load_display_bbox(calibration, args.display_crop)
    manifest["point_filter"]["boxes"] = bbox["boxes"]
    manifest["point_filter"]["display_crop"] = evidence
    output = root / "viewer"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    write_json(
        root / "build_status.json",
        dict(
            state="repair_candidate",
            sources=state,
            training_published=False,
            body_qa_accepted=sum(s["accepted"] for s in state.values()),
        ),
    )
    code = PROJ_ROOT / "hmr4d/lidar_mocap"
    html = (code / "rally_viewer.html").read_text(encoding="utf-8")
    html = html.replace("羽毛球 · 双机 / SMPL 检查", "SMPL 修复候选 · 未发布")
    html = html.replace("7 来源 · 204 回合", f"{manifest['source_count']} 来源 · {manifest['rally_count']} 回合")
    html = html.replace("回合版人帧", "拟合 QA 人帧")
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / "viewer.js").write_bytes((code / "rally_viewer_client.js").read_bytes())
    for name in ("review.js", "review.css"):
        (output / name).write_bytes((code / "review_assets" / name).read_bytes())
    data = ViewerData(root, root, calibration, manifest, sessions, media, viewer_bbox=bbox)
    serve_viewer(data, port=args.port)


if __name__ == "__main__":
    main()
