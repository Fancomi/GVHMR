"""Serve ten five-second clips using the two verified pilots' native renderers."""

import argparse
from copy import deepcopy
import hashlib
import html
import json
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json
from hmr4d.lidar_mocap.rally_viewer import serve_viewer
from hmr4d.lidar_mocap.relocations import load_run_relocations
from hmr4d.lidar_mocap.source_index import _bound_file


def binding(path):
    path = Path(path).resolve()
    blob = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob))


def fetch(url, path, query=None):
    with urlopen(url.rstrip("/") + path + ("?" + urlencode(query) if query else ""), timeout=60) as response:
        return response.read()


class PilotReview:
    def __init__(self, specifications, output):
        self.viewer_output = output.resolve()
        self.media, self.providers, self.status_sources = {}, {}, {}
        inputs, quality, sources, selections = [], [], {}, []
        first = None
        source_digests = {}
        for root, url, original_rallies in specifications:
            root = root.resolve()
            address = urlparse(url)
            if address.scheme != "http" or address.hostname != "127.0.0.1" or address.path not in ("", "/"):
                raise ValueError("Use the explicit local pilot services")
            labels = root / "repair/final_labels"
            report_path = root / "quality_report.json"
            report = read_json(report_path)
            dataset_path = labels / "datasets/dataset_manifest.json"
            dataset = read_json(dataset_path)
            native_path = labels / "native_verification.json"
            if (report.get("review_only") is not True or report.get("training_published") is not False
                    or dataset.get("review_only") is not True or dataset.get("training_published") is not False
                    or read_json(native_path).get("passed") is not True
                    or binding(dataset_path)["sha256"] != report["dataset"]["sha256"]):
                raise ValueError("Both pilot datasets require completed native verification and bound quality reports")
            calibration = CalibrationSnapshot.load(labels / "calibration/calibration_snapshot.json")
            live = json.loads(fetch(url, "/api/manifest"))
            if (live.get("review_only") is not True or live.get("training_published") is not False
                    or live["calibration_digest"] != report["calibration_digest"]
                    or live["calibration_digest"] != calibration.data["snapshot_digest"]
                    or len(live["sources"]) != 1
                    or set(live["sources"]) != set(report["selection"]["sources"])):
                raise ValueError("The live service does not match its verified pilot")
            runtime_entry = live["runtime_relocations"]
            if binding(runtime_entry["path"])["sha256"] != runtime_entry["sha256"]:
                raise ValueError("Live pilot runtime binding changed")
            runtime = read_json(runtime_entry["path"])
            if (Path(runtime["source_registry"]["path"]).resolve() != labels / "source_registry.json"
                    or binding(labels / "source_registry.json")["sha256"] != runtime["source_registry"]["sha256"]):
                raise ValueError("The live renderer serves another body-output root")
            if first is None:
                first = (calibration, live)
            elif (calibration.data["configuration_files_sha256"] != first[0].data["configuration_files_sha256"]
                    or live["faces"] != first[1]["faces"] or live["edges"] != first[1]["edges"]
                    or live["data_relocations"] != first[1]["data_relocations"]):
                raise ValueError("Pilots differ in original camera bundle, SMPL topology or source relocation")
            registry = read_json(labels / "source_registry.json")
            table = live["data_relocations"]
            resolver = load_run_relocations(labels, Path(registry["data_root"]) / table["file"], table["sha256"])
            wanted = set(original_rallies)
            found = set()
            for sid, source in live["sources"].items():
                if sid in sources:
                    raise ValueError("A source cannot route to two pilot services")
                selected = []
                for rally in source["rallies"]:
                    if rally["original_rally_index"] not in wanted:
                        continue
                    start = max(0, rally["begin"]) + 0.1
                    end = start + 5
                    if end > rally["end"] - 0.05:
                        raise ValueError("A selected rally cannot supply five continuous seconds")
                    indices = [i for i, t in enumerate(source["scan_times"]) if start <= t <= end]
                    if len(indices) < 45:
                        raise ValueError("A selected preview lacks actual native scan coverage")
                    clip = dict(rally, index=len(selected), begin=start, end=end,
                                source_begin_frame=indices[0], source_frame_count=len(indices))
                    selected.append(clip)
                    selections.append(dict(source_id=sid, original_rally_index=rally["original_rally_index"],
                                           begin=start, end=end, scans=len(indices)))
                    found.add(rally["original_rally_index"])
                if not selected:
                    continue
                sources[sid] = dict(source, rallies=selected, baseline=True)
                self.providers[sid] = url
                self.status_sources[sid] = dict(baseline=True, state="fitted", rallies=len(selected),
                    rally_unique_paired_scans=sum(c["source_frame_count"] for c in selected),
                    fitted=report["cohorts"]["all_fitted"]["person_scans"],
                    accepted=report["cohorts"]["accepted"]["person_scans"])
                source_digests[sid] = calibration.data["snapshot_digest"]
                entry = registry["sources"][sid]["source_index"]
                if binding(entry["path"])["sha256"] != entry["sha256"]:
                    raise ValueError("Pilot source index changed")
                index = read_json(entry["path"])
                for cam in ("cam1", "cam2"):
                    self.media[sid, cam] = _bound_file(Path(registry["data_root"]), index["native_videos"][cam],
                                                      relocations=resolver)
            if wanted != found:
                raise ValueError("An explicitly selected original rally is absent")
            inputs.append(dict(root=str(root), url=url, quality_report=binding(report_path),
                               native_verification=binding(native_path), dataset=binding(dataset_path)))
            cohort = report["cohorts"]["all_fitted"]
            outlier_path = root / "outlier_review.json"
            ground_cases = read_json(outlier_path)["ground_cases"] if outlier_path.is_file() else []
            worst_url = None
            if ground_cases:
                case = max(ground_cases, key=lambda case: case["ground_mm"])
                worst_url = url.rstrip("/") + "/?" + urlencode(dict(
                    source=report["selection"]["sources"][0], scan=case["scan"]))
            quality.append(dict(sources=report["selection"]["sources"],
                source_seconds=report["selection"]["selected_seconds"],
                surface_mm=cohort["surface_and_ground_mm"]["surface_mean_m"]["mean"],
                cam1_px=cohort["raw_native_camera_px"]["cam1"]["mean_px"]["mean"],
                cam2_px=cohort["raw_native_camera_px"]["cam2"]["mean_px"]["mean"],
                max_ground_mm=cohort["surface_and_ground_mm"]["max_ground_penetration_m"]["max"],
                worst_ground_url=worst_url))
        if len(selections) != 10:
            raise ValueError("This review requires exactly ten explicit five-second clips")
        self.manifest = deepcopy(first[1])
        self.manifest.update(schema="badminton-multi-pilot-native-review-v1", sources=sources,
            source_count=len(sources), declared_source_count=len(sources), empty_sources={}, rally_count=10,
            rally_window_minutes=50 / 60, unique_paired_input_scans=sum(s["scans"] for s in selections),
            calibration_digest=None, source_calibration_digests=source_digests,
            original_configuration_sha256=first[0].data["configuration_files_sha256"],
            review_only=True, training_published=False, review_inputs=inputs, preview_selection=selections,
            quality_scope="Quality metrics cover both complete pilot datasets, not only the ten previews",
            quality=quality)
        self.manifest.pop("runtime_relocations", None)
        self.viewer_output.mkdir(parents=True, exist_ok=True)
        write_json(self.viewer_output / "manifest.json", self.manifest)
        code = PROJ_ROOT / "hmr4d/lidar_mocap"
        page = (code / "rally_viewer.html").read_text(encoding="utf-8")
        page = page.replace("羽毛球 · 双机 / SMPL 检查", "R11 · 十段连续评审 · 未发布")
        page = page.replace("7 来源 · 204 回合", "2 来源 · 10 × 5秒")
        page = page.replace("回合版数据 · 7 来源 / 204 回合", "R11 评审候选 · 2 来源 / 10 个片段")
        page = page.replace("回合版人帧", "全段 QA 人帧")
        page = page.replace("本页展示密封回合子集，合格人帧包含旧场次的回合复用和新增场次，不代表所有历史数据累计量。",
                            "本页仅展示评审候选；拟合 QA 通过不代表正式训练接纳。")
        rows = "".join(f"<tr><td>{html.escape(', '.join(q['sources']))}</td><td>{q['source_seconds']:.2f}s</td>"
            f"<td>{q['surface_mm']:.2f}mm</td><td>{q['cam1_px']:.2f}px</td><td>{q['cam2_px']:.2f}px</td>"
            f"<td>{q['max_ground_mm']:.2f}mm "
            + (f'<a href="{html.escape(q["worst_ground_url"])}" target="_blank" rel="noopener">查看坏例</a>' if q["worst_ground_url"] else "")
            + "</td></tr>" for q in quality)
        table = ("<p>两段完整数据约178秒；当前仅精选十段，每段5秒。以下误差覆盖完整两段，不是仅这50秒。"
                 "表面残差不是人体真值误差；地面穿透坏例仍保留。来源内的轨迹不代表跨回合身份已确认。</p>"
                 "<table><thead><tr><th>来源</th><th>时长</th><th>表面均值</th><th>cam1原始重投影</th>"
                 "<th>cam2原始重投影</th><th>最大地面穿透</th></tr></thead><tbody>" + rows + "</tbody></table>")
        page = page.replace('<div class="details-panel">', '<div class="details-panel">' + table)
        (self.viewer_output / "index.html").write_text(page, encoding="utf-8")
        for name, path in [("viewer.js", code / "rally_viewer_client.js"),
                           ("review.js", code / "review_assets/review.js"),
                           ("review.css", code / "review_assets/review.css")]:
            (self.viewer_output / name).write_bytes(path.read_bytes())

    def status(self):
        return dict(state="review_only", review_only=True, training_published=False,
                    body_qa_accepted=sum(s["accepted"] for s in self.status_sources.values()),
                    sources=self.status_sources)

    def geometry(self, sid, seconds, scope="roi"):
        return fetch(self.providers[sid], "/api/geometry", dict(source=sid, t=seconds, scope=scope))

    def geometry_batch(self, sid, seconds, count, scope="roi"):
        return fetch(self.providers[sid], "/api/bundles", dict(source=sid, t=seconds, count=count, scope=scope))

    def overlay(self, sid, seconds):
        return fetch(self.providers[sid], "/api/overlay", dict(source=sid, t=seconds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-root", type=Path, default=PROJ_ROOT / "outputs/badminton_1019_r11_pilot_v1")
    parser.add_argument("--second-root", type=Path, default=PROJ_ROOT / "outputs/badminton_1019_r11_pilot_second_v1")
    parser.add_argument("--first-url", default="http://127.0.0.1:8781/")
    parser.add_argument("--second-url", default="http://127.0.0.1:8782/")
    parser.add_argument("--output", type=Path, default=PROJ_ROOT / "outputs/badminton_1019_r11_review/viewer")
    parser.add_argument("--port", type=int, default=8783)
    args = parser.parse_args()
    output = args.output.resolve()
    if (not output.is_relative_to(PROJ_ROOT / "outputs") or output == PROJ_ROOT / "outputs"
            or any(output.is_relative_to(r.resolve()) for r in (args.first_root, args.second_root))):
        parser.error("Review output must stay inside this workspace's outputs directory")
    data = PilotReview([(args.first_root, args.first_url, [2, 3, 4, 5, 6]),
                        (args.second_root, args.second_url, [1, 15, 23, 32, 38])], args.output)
    serve_viewer(data, port=args.port)


if __name__ == "__main__":
    main()
