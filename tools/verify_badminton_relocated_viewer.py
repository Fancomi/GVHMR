"""Check relocated canonical source reads through the actual running browser."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

from hmr4d.lidar_mocap.rally_pipeline import read_json
from verify_badminton_viewer import visible_rgb_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8774/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--without-canonical-publication", action="store_true")
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(args.url + "api/manifest", timeout=20) as response:
        manifest = json.load(response)
    publication = manifest.get("canonical_publication")
    if not args.without_canonical_publication:
        publication_path = Path(publication["path"])
        assert hashlib.sha256(publication_path.read_bytes()).hexdigest() == publication["sha256"]
        admitted = read_json(publication_path)
    errors, results = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        for sid, source in manifest["sources"].items():
            if args.without_canonical_publication:
                rally = source["rallies"][0]
                target = (max(0, rally["begin"]) + rally["end"]) / 2
                scan = min(range(len(source["scan_times"])), key=lambda i: abs(source["scan_times"][i] - target))
                seconds = source["scan_times"][scan]
            else:
                with Path(admitted["exports"][sid]["metadata"]).open(encoding="utf-8") as stream:
                    row = json.loads(next(stream))
                scan = row["lidar_index"]
                seconds = (row["scan_mid_us"] - source["common_origin_us"]) / 1e6
            page = browser.new_page(viewport=dict(width=1366, height=768))
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(args.url + f"?source={sid}&rally={source['rallies'][0]['index']}&scan={scan}")
            page.wait_for_function("window.viewerReady===true && !frameBusy", timeout=90000)
            page.evaluate("async(t)=>{await viewer.seek(t);return true}", seconds)
            page.wait_for_function("!frameBusy && lastPairedFrame.athletes.length>0", timeout=45000)
            state = page.frame_locator("#scene").locator("body").evaluate("()=>REVIEW_STATE")
            frame = page.evaluate("""()=>({source:committedPairedFrame.sourceId,scan:committedPairedFrame.scan,
                points:lastPairedFrame.frame.point_count,
                projections:Object.fromEntries(Object.entries(lastPairedFrame.background_projection).map(([k,v])=>[k,v.length])),
                projected_meshes:lastPairedFrame.athletes.map(a=>[a.cameras.cam1.mesh.length,a.cameras.cam2.mesh.length]),
                native_pts_errors:['cam1','cam2'].map(c=>Math.abs(lastPairedFrame.native_images[c].media_pts_us/1e6-lastPairedFrame.native[c].pts)),
                projection_states:projectionStates})""")
            rgb = visible_rgb_checks(page)
            assert frame["source"] == sid and frame["scan"] == scan == state["scan"], frame
            assert frame["points"] == state["pointCount"] > 0, state
            assert state["meshCount"] > 0 and state["vertexCount"] == state["meshCount"] * 6890, state
            assert all(n == 6890 for pair in frame["projected_meshes"] for n in pair), frame
            assert all(n > 0 for n in frame["projections"].values()), frame
            assert max(frame["native_pts_errors"]) < 0.001 and not state["webglError"], frame
            assert all(s["scan"] == scan and not s["webglError"] for s in frame["projection_states"]), frame
            assert all(c["visible"] and c["different_rgba_components"] == 0 for c in rgb), rgb
            screenshot = args.output.parent / f"{args.output.stem}_{sid}.png"
            page.screenshot(path=str(screenshot))
            results.append(dict(source_id=sid, passed=True, scene=state, frame=frame, rgb=rgb, screenshot=str(screenshot)))
            print(dict(source_id=sid, scan=scan, passed=True), flush=True)
            page.close()
        version = browser.version
        browser.close()
    assert not errors, errors
    result = dict(passed=True, url=args.url, data_relocations=manifest["data_relocations"],
                  started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                  runtime_relocations=manifest["runtime_relocations"], canonical_publication=publication,
                  declared_sources=manifest.get("declared_source_count", manifest["source_count"]),
                  empty_sources=manifest.get("empty_sources", {}),
                  cases=results, page_errors=errors, browser_version=version)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
