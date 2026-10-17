"""Export ten synchronous native-frame old/new reviews using the shared renderers."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import numpy as np

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import load_run_session, read_json, read_rows
from hmr4d.lidar_mocap.rally_viewer import ViewerData, load_display_bbox


class WindowViewer(ViewerData):
    def use_window(self, sid, window, colors):
        self.rows = {}
        for row in read_rows(window / "quality.jsonl"):
            self.rows.setdefault(row["lidar_index"], []).append(row)
            self._track_colors[row["motion_id"]] = colors[int(row["motion_id"].split(":")[-2])]

    def quality_rows(self, sid):
        return self.rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2")
    p.add_argument("--labels", type=Path)
    p.add_argument("--wait", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    labels = args.labels or root / "final_labels"
    output = root / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    plan = read_json(root / "case_plan.json")
    original = PROJ_ROOT / "outputs/badminton_rallies_v1"
    cal = CalibrationSnapshot.load(original / "calibration/calibration_snapshot.json")
    base = read_json(original / "viewer/manifest.json")
    assets = PROJ_ROOT / "hmr4d/lidar_mocap/review_assets"
    public = dict(
        schema="native-smpl-before-after-v2",
        source_union_sha256=plan["source_union_sha256"],
        calibration_digest=cal.data["snapshot_digest"],
        faces=base["faces"],
        edges=base["edges"],
        reference_plane=cal.reference_plane,
        clip_count=len(plan["clips"]),
        clips=[],
        limits=[
            "Both cameras participate in the repaired fit; camera residuals are not held-out ground truth.",
            "Surface residual uses each version's own assigned visible points; point ownership may change.",
            "Missing or rejected fits remain explicit. No mesh, point or RGB interpolation.",
        ],
    )
    for c in plan["clips"]:
        public["clips"].append(dict(c, state="pending", frames=50, duration_s=5))

    def save():
        write_json(output / "manifest.json", public)
        (output / "manifest.js").write_text(
            "window.COMPARISON_MANIFEST=" + json.dumps(public, ensure_ascii=False, separators=(",", ":")) + ";",
            encoding="utf-8",
        )

    for src, dst in [
        ("comparison.html", "index.html"),
        ("comparison.js", "comparison.js"),
        ("comparison.css", "comparison.css"),
        ("projection.js", "projection.js"),
        ("review.js", "review.js"),
        ("review.css", "review.css"),
    ]:
        shutil.copyfile(assets / src, output / dst)
    scene = dict(
        base,
        stream_mode=True,
        clip_count=1,
        clip_duration_s=0,
        clips=[
            dict(
                id="compare",
                session_id="compare",
                duration_s=0,
                summary=dict(scans=0, fitted_person_frames=0, accepted=0, mean_surface_mm=None),
            )
        ],
    )
    html = (
        (assets / "index.html")
        .read_text(encoding="utf-8")
        .replace("__MANIFEST__", json.dumps(scene, separators=(",", ":")))
        .replace("<body>", '<body class="stream">')
    )
    (output / "scene.html").write_text(html, encoding="utf-8")
    save()
    bbox, crop = load_display_bbox(
        cal, Path("D:/PROJECT/LIDAR/lidar_project/config/calibration/badminton-20261012/display-crop.json")
    )
    public["display_crop"] = crop
    sessions = {}
    started = time.perf_counter()
    while any(c["state"] != "complete" for c in public["clips"]):
        changed = False
        for c in public["clips"]:
            if c["state"] == "complete":
                continue
            sid = c["source_id"]
            old = Path(c["source_window"])
            new = labels / "sessions" / sid / "windows" / old.name
            if not (new / "fit_report.json").exists():
                continue
            before, after = read_json(old / "fit_report.json"), read_json(new / "fit_report.json")
            if (
                read_json(old / "window.json")["calibration_digest"]
                != read_json(new / "window.json")["calibration_digest"]
            ):
                raise ValueError("Comparison cameras changed")
            if sid not in sessions:
                sessions[sid] = load_run_session(original, cal, sid)
            session = sessions[sid]
            manifest = copy.deepcopy(base)
            manifest["sources"][sid]["baseline"] = False
            palette = [
                [255, 190, 58],
                [80, 165, 255],
                [123, 226, 148],
                [220, 138, 241],
                [255, 125, 120],
                [102, 220, 223],
            ]
            teacher = read_json(new / "teacher/manifest.json")
            aliases = {}
            for t in teacher["tracks"]:
                for raw in t.get("source_tracklets", [t["track_id"]]):
                    aliases[raw] = t["track_id"]
            oldids = {int(r["motion_id"].split(":")[-2]) for r in read_rows(old / "quality.jsonl")}
            newids = {int(r["motion_id"].split(":")[-2]) for r in read_rows(new / "quality.jsonl")}
            ids = sorted({aliases.get(k, k) for k in oldids} | newids)
            colors = {k: palette[i % len(palette)] for i, k in enumerate(ids)}
            a = WindowViewer(original, original, cal, manifest, {sid: session}, {}, bbox)
            b = WindowViewer(labels, labels, cal, manifest, {sid: session}, {}, bbox)
            a.use_window(sid, old, {k: colors[aliases.get(k, k)] for k in oldids})
            b.use_window(sid, new, {k: colors[k] for k in newids})
            b.native_image = a.native_image
            frames = []
            origin = manifest["sources"][sid]["common_origin_us"]
            pairs = session.pairs()
            for scan in c["lidar_indices"]:
                t = (pairs[scan]["scan_mid_us"] - origin) / 1e6
                aa = a.geometry(sid, t)
                bb = b.geometry(sid, t)
                if aa["lidar_index"] != scan or bb["lidar_index"] != scan or aa["native"] != bb["native"]:
                    raise ValueError("Native scan mismatch")
                images = aa.pop("native_images")
                other = bb.pop("native_images")
                if images != other:
                    raise ValueError("Old/new RGB differs")
                for cam, im in images.items():
                    if im["record_index"] != aa["native"][cam]["record_index"]:
                        raise ValueError("Native image mismatch")
                bb["frame"].pop("cloud")
                bb.pop("background_projection")
                offset = (pairs[scan]["scan_mid_us"] - pairs[c["lidar_indices"][0]]["scan_mid_us"]) / 1e6
                frames.append(dict(scan=scan, offset_s=offset, native_images=images, before=aa, after=bb))
            data = dict(
                id=c["id"],
                source_id=sid,
                frames=frames,
                duration_s=5,
                aliases=aliases,
                source_reports={
                    str(path): file_sha(path)
                    for path in [old / "fit_report.json", new / "fit_report.json", new / "teacher/manifest.json"]
                },
            )
            dest = output / c["id"]
            dest.mkdir(exist_ok=True)
            (dest / "data.js").write_text(
                "window.COMPARISON_CLIP_READY("
                + json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + ");",
                encoding="utf-8",
            )
            c.update(
                state="complete",
                data_sha256=file_sha(dest / "data.js"),
                source_reports=data["source_reports"],
                fitted_before=sum(len(f["before"]["frame"]["athletes"]) for f in frames),
                fitted_after=sum(len(f["after"]["frame"]["athletes"]) for f in frames),
            )
            save()
            changed = True
            print(
                dict(clip=c["id"], state="complete", scans=len(frames), wall_seconds=time.perf_counter() - started),
                flush=True,
            )
            for decoder in [a.preview_decoder, b.preview_decoder]:
                for state in decoder.states.values():
                    state["container"].close()
            a._preview_pool.shutdown()
            a._geometry_pool.shutdown()
            b._preview_pool.shutdown()
            b._geometry_pool.shutdown()
        if not args.wait:
            break
        if not changed:
            time.sleep(5)
    public["state"] = "complete" if all(c["state"] == "complete" for c in public["clips"]) else "partial"
    save()


if __name__ == "__main__":
    main()
