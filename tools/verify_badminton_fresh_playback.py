"""Verify ten real five-second native dual-camera/SMPL/cloud clips in Chrome."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.canonical_tracks import checked_review, remap_row
from hmr4d.lidar_mocap.expanded_dataset import binding, check_binding
from hmr4d.lidar_mocap.rally_pipeline import read_json
from hmr4d.lidar_mocap.repair_execution import write_progress_json as write_json
from verify_badminton_viewer import visible_rgb_checks


def verify(root, url, canonical_path=None, output_root=None):
    root = Path(root).resolve()
    output = Path(output_root).resolve() if output_root else root / "browser_acceptance"
    output.mkdir(parents=True, exist_ok=True)
    frozen = binding(root / "new_source_completion_receipt.json")
    source = read_json(frozen["path"])
    if source.get("state") != "labels_published_viewer_prepared":
        raise ValueError("Finish all native/body/bilateral gates before browser acceptance")
    for path, sha in source["evidence"].items():
        check_binding(dict(path=path, sha256=sha))
    for attempt in range(120):
        try:
            with urlopen(url + "api/manifest", timeout=2) as response:
                manifest = json.load(response)
            break
        except OSError:
            if attempt == 119:
                raise
            time.sleep(0.5)
    if manifest.get("declared_source_count") != 6 or set(manifest.get("empty_sources", {})) != {"20261015_164748"}:
        raise ValueError("Fresh viewer must explicitly retain the declared zero source without synthetic playback")
    if len(manifest["sources"]) != 5 or not all(s["baseline"] for s in manifest["sources"].values()):
        raise ValueError("Every nonempty fresh source must expose actual stored body labels")
    clips = []
    canonical_binding, boundary_rows, boundary_results = None, {}, []
    if canonical_path:
        canonical_binding = binding(canonical_path)
        if manifest.get("canonical_publication") != canonical_binding:
            raise ValueError("Browser is not serving the selected canonical publication")
        canonical = read_json(canonical_path)
        review_binding = canonical["canonical_identity_revision"]["reviewed_boundaries"]
        cuts = checked_review(review_binding["path"])
        review = read_json(review_binding["path"])
        admitted_boundary_keys = set()
        original_boundary_ids = {c["event"]["motion_id"] for c in review["cases"]}
        for sid in {c["event"]["source_id"] for c in review["cases"]}:
            with Path(canonical["exports"][sid]["metadata"]).open(encoding="utf-8") as stream:
                for line in stream:
                    if any(motion in line for motion in original_boundary_ids):
                        row = json.loads(line)
                        admitted_boundary_keys.add((row["motion_id"], row["lidar_index"]))
        for case in review["cases"]:
            event = case["event"]
            path = root / "final_labels/sessions" / event["source_id"] / "windows" / event["window"] / "quality.jsonl"
            wanted = {event["before_scan"], event["after_scan"]}
            selected = {}
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if event["motion_id"] not in line:
                        continue
                    row = json.loads(line)
                    if row["motion_id"] == event["motion_id"] and row["lidar_index"] in wanted:
                        selected[row["lidar_index"]] = row
            boundary_rows[case["case"]] = selected
        focused = set()
        for case in review["cases"]:
            event = case["event"]
            key = event["source_id"], event["window"], event["motion_id"]
            if case["verdict"] != "different_person" or key in focused:
                continue
            item = manifest["sources"][event["source_id"]]
            t = (boundary_rows[case["case"]][event["before_scan"]]["scan_mid_us"] - item["common_origin_us"]) / 1e6
            eligible = [r for r in item["rallies"] if r["begin"] <= t <= r["end"] and r["end"] - r["begin"] >= 5.5]
            if not eligible:
                continue
            rally = eligible[0]
            begin = max(rally["begin"] + 0.1, min(t - 2.5, rally["end"] - 5.1))
            clips.append(
                dict(
                    source_id=event["source_id"],
                    rally_index=rally["index"],
                    begin=begin,
                    duration_seconds=5,
                    regression_case=case["case"],
                )
            )
            focused.add(key)
    for sid, item in manifest["sources"].items():
        rallies = [r for r in item["rallies"] if r["end"] - r["begin"] >= 5.5]
        if len(rallies) < 2:
            raise ValueError("Each source needs two genuine continuous five-second native clips")
        selected_rallies = (
            (
                max(
                    rallies,
                    key=lambda r: min(
                        [abs((r["begin"] + r["end"]) / 2 - c["begin"]) for c in clips if c["source_id"] == sid] or [1]
                    ),
                ),
            )
            if canonical_path
            else (rallies[0], rallies[-1])
        )
        for rally in selected_rallies:
            clips.append(
                dict(source_id=sid, rally_index=rally["index"], begin=rally["begin"] + 0.2, duration_seconds=5)
            )
    if canonical_path and len(clips) < 10:
        for sid, item in manifest["sources"].items():
            for rally in item["rallies"]:
                if len(clips) == 10:
                    break
                if rally["end"] - rally["begin"] >= 5.5 and not any(
                    c["source_id"] == sid and c["rally_index"] == rally["index"] for c in clips
                ):
                    clips.append(
                        dict(source_id=sid, rally_index=rally["index"], begin=rally["begin"] + 0.2, duration_seconds=5)
                    )
    if len(clips) != 10:
        raise ValueError("Exactly ten genuine five-second clips are required")
    plan = dict(
        schema="badminton-fresh-browser-clips-v1",
        source_completion=frozen,
        canonical_publication=canonical_binding,
        selection=(
            "Native identity-boundary regressions plus temporally diverse original rallies across all five nonempty sources"
            if canonical_path
            else "First and last sufficiently long original rally of each nonempty source; all five sources covered"
        ),
        clips=clips,
        declared_empty_sources=manifest["empty_sources"],
    )
    write_json(output / "clip_plan.json", plan)
    errors, results = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"]
        )
        page = browser.new_page(viewport=dict(width=1366, height=768))
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_function("window.viewerReady === true", timeout=180000)
        page.select_option("#mode", "paired")
        for index, clip in enumerate(clips):
            page.evaluate("async(sid)=>{await viewer.selectSource(sid); return true}", clip["source_id"])
            page.evaluate("async(index)=>{await viewer.selectRally(index); return true}", clip["rally_index"])
            page.evaluate("async(t)=>{await viewer.seek(t); return true}", clip["begin"])
            page.wait_for_function("!frameBusy && lastPairedFrame && lastPairedFrame.athletes.length>0", timeout=90000)
            initial_rgb = visible_rgb_checks(page)
            assert all(r["visible"] and r["different_rgba_components"] == 0 for r in initial_rgb), initial_rgb
            page.locator("#fitView").click()
            page.evaluate(
                """()=>{
                window.freshSamples=[];
                window.freshSampler=setInterval(()=>{
                    if(frameBusy||!playing||!committedPairedFrame)return;
                    const scene=document.querySelector('#scene').contentWindow.REVIEW_STATE;
                    if(!scene)return;
                    freshSamples.push({scan:committedPairedFrame.scan,time:viewer.commonTime(),sceneScan:scene.scan,
                        meshCount:scene.meshCount,vertexCount:scene.vertexCount,webglError:scene.webglError,
                        projections:projectionStates.map(p=>({scan:p.scan,error:p.webglError})),
                        athleteIds:lastPairedFrame.athletes.map(a=>a.motion_id||a.id)});
                },40);
            }"""
            )
            page.locator("#loop").uncheck()
            page.locator("#play").click()
            page.wait_for_function("playing && !preparing", timeout=180000)
            start_time = page.evaluate("viewer.commonTime()")
            page.wait_for_function("(t)=>viewer.commonTime()>=t+5", arg=start_time, timeout=90000)
            page.evaluate("viewer.pause()")
            page.wait_for_function("!playing && !preparing && !frameBusy", timeout=45000)
            page.evaluate("clearInterval(freshSampler)")
            samples = page.evaluate("freshSamples")
            final_rgb = visible_rgb_checks(page)
            state = page.frame_locator("#scene").locator("body").evaluate("()=>REVIEW_STATE")
            end_time = page.evaluate("viewer.commonTime()")
            scans = sorted({s["scan"] for s in samples})
            write_json(
                output / f"clip{index+1:02d}_diagnostics.json", dict(clip=clip, samples=samples, final_state=state)
            )
            assert len(scans) >= 20 and end_time - start_time >= 4.9, (clip, len(scans), start_time, end_time)
            valid_samples = [
                s
                for s in samples
                if s["scan"] == s["sceneScan"]
                and not s["webglError"]
                and s["meshCount"] == len(s["athleteIds"])
                and s["vertexCount"] == s["meshCount"] * 6890
                and all(p["scan"] == s["scan"] and not p["error"] for p in s["projections"])
            ]
            assert len(valid_samples) == len(samples), (clip, [s for s in samples if s not in valid_samples][:3])
            assert all(
                s["scan"] == s["sceneScan"]
                and not s["webglError"]
                and s["meshCount"] == len(s["athleteIds"])
                and s["vertexCount"] == s["meshCount"] * 6890
                and all(p["scan"] == s["scan"] and not p["error"] for p in s["projections"])
                for s in samples
            ), (clip, [s for s in samples if s["scan"] != s["sceneScan"]][:3])
            assert all(r["visible"] and r["different_rgba_components"] == 0 for r in final_rgb), final_rgb
            labelled_scans = {s["scan"] for s in samples if s["meshCount"] > 0}
            assert len(labelled_scans) >= 20 and not state["webglError"], state
            screenshot = output / f"clip{index+1:02d}.png"
            page.screenshot(path=str(screenshot), full_page=False)
            result = dict(
                **clip,
                passed=True,
                actual_begin=start_time,
                actual_end=end_time,
                unique_committed_scans=len(scans),
                labelled_scans=len(labelled_scans),
                explicit_unlabelled_scans=sorted({s["scan"] for s in samples if s["meshCount"] == 0}),
                scans=scans,
                samples=samples,
                initial_visible_rgb=initial_rgb,
                final_visible_rgb=final_rgb,
                full_smpl_vertices=state["vertexCount"],
                mesh_count=state["meshCount"],
                point_count=state["pointCount"],
                screenshot=binding(screenshot),
            )
            results.append(result)
            write_json(
                output / "progress.json",
                dict(
                    state="running",
                    completed_clips=len(results),
                    total_clips=len(clips),
                    last_clip=dict(source=clip["source_id"], rally=clip["rally_index"]),
                ),
            )
            print(
                json.dumps(
                    dict(
                        clip=index + 1,
                        source=clip["source_id"],
                        unique_scans=len(scans),
                        elapsed=end_time - start_time,
                        passed=True,
                    )
                ),
                flush=True,
            )
        if canonical_path:
            for case in review["cases"]:
                event = case["event"]
                sid = event["source_id"]
                page.evaluate("async(sid)=>{await viewer.selectSource(sid);return true}", sid)
                observed = []
                for side, key in (("before", "before_scan"), ("after", "after_scan")):
                    original = boundary_rows[case["case"]][event[key]]
                    expected = remap_row(
                        dict(original, track_id=original["motion_id"].split(":")[2]), cuts, review_binding["sha256"]
                    )["motion_id"]
                    t = (original["scan_mid_us"] - manifest["sources"][sid]["common_origin_us"]) / 1e6
                    page.evaluate("async(t)=>{await viewer.seek(t);return true}", t)
                    page.wait_for_function("!frameBusy", timeout=45000)
                    actors = page.evaluate("lastPairedFrame.athletes.map(a=>({motion_id:a.motion_id,color:a.color}))")
                    present = any(a["motion_id"] == expected for a in actors)
                    should_exist = (expected, event[key]) in admitted_boundary_keys
                    assert present == should_exist, (case["case"], side, expected, actors)
                    rgb = visible_rgb_checks(page)
                    assert all(r["visible"] and r["different_rgba_components"] == 0 for r in rgb), rgb
                    observed.append(
                        dict(
                            side=side,
                            scan=event[key],
                            canonical_motion_id=expected,
                            present=present,
                            actors=actors,
                            visible_rgb=rgb,
                        )
                    )
                if case["verdict"] == "different_person":
                    assert observed[0]["canonical_motion_id"] != observed[1]["canonical_motion_id"]
                boundary_results.append(
                    dict(case=case["case"], verdict=case["verdict"], passed=True, observations=observed)
                )
        version = browser.version
        browser.close()
    if errors or len(results) != 10:
        raise ValueError("Browser reported errors or did not complete ten clips: " + str(errors))
    check_binding(frozen)
    producer = output / "producer.py"
    shutil.copyfile(__file__, producer)
    receipt = dict(
        schema="badminton-fresh-browser-acceptance-v1",
        passed=True,
        completed_utc=datetime.now(timezone.utc).isoformat(),
        source_completion=frozen,
        canonical_publication=canonical_binding,
        native_boundary_regressions=boundary_results,
        producer=binding(producer),
        clip_plan=binding(output / "clip_plan.json"),
        viewer_manifest=binding(root / "final_labels/viewer/manifest.json"),
        url=url,
        browser_version=version,
        declared_source_count=6,
        played_sources=5,
        clips=len(results),
        minimum_clip_seconds=5,
        results=results,
        page_errors=errors,
        limits=[
            "Actual independent Chrome playback; both cameras, full stored SMPL and native cloud remain synchronized.",
            "Visible RGB is byte-compared to the PTS-bound rendered image at clip endpoints.",
            "Internal camera/point residuals are not independent anatomical ground truth.",
            "Scans with no admitted body remain explicitly empty; no geometry or labels are interpolated.",
        ],
    )
    write_json(output / "acceptance.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8773/")
    parser.add_argument("--canonical-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if not args.root.resolve().is_relative_to(PROJ_ROOT / "outputs"):
        parser.error("Select an explicit fresh production outputs root")
    if args.output_root and not args.output_root.resolve().is_relative_to(args.root.resolve()):
        parser.error("Fresh acceptance evidence must stay inside its repair run")
    receipt = verify(args.root, args.url, args.canonical_manifest, args.output_root)
    print(json.dumps(dict(passed=receipt["passed"], clips=receipt["clips"], played_sources=receipt["played_sources"])))


if __name__ == "__main__":
    main()
