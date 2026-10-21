"""Play every selected short pilot rally in real Chrome with exact native RGB checks."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from playwright.sync_api import sync_playwright

from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json
from verify_badminton_viewer import visible_rgb_checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:8781/")
    p.add_argument("--output", type=Path)
    p.add_argument("--original-rallies", type=int, nargs="+", help="Explicit original rally numbers to play")
    p.add_argument("--outlier-scan", type=int)
    p.add_argument("--outlier-source")
    args = p.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    root = args.root.resolve()
    combined = not (root / "pilot_input.json").is_file()
    if combined:
        frozen = read_json(root / "viewer/manifest.json")
        if (frozen.get("schema") != "badminton-multi-pilot-native-review-v1"
                or frozen.get("review_only") is not True or frozen.get("training_published") is not False):
            raise ValueError("Select a verified multi-pilot review")
        for source in frozen["review_inputs"]:
            for name in ("quality_report", "native_verification", "dataset"):
                item = source[name]
                if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                    raise ValueError("Combined pilot review input changed")
    elif read_json(root / "pilot_input.json").get("review_only") is not True:
        raise ValueError("Select an explicitly review-only pilot")
    output = args.output.resolve() if args.output else root / "browser_checks"
    if not output.is_relative_to(root):
        p.error("Pilot browser evidence must stay inside its run")
    output.mkdir(exist_ok=True)
    results, errors = [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"]
        )
        initial = browser.new_page(viewport=dict(width=1366, height=768))
        initial.goto(args.url)
        initial.wait_for_function("window.viewerReady===true", timeout=90000)
        manifest = initial.evaluate("manifest")
        if manifest.get("review_only") is not True or manifest.get("training_published") is not False:
            raise ValueError("Browser is not serving the review-only pilot")
        initial.close()
        selected = [
            (sid, rally)
            for sid, source in manifest["sources"].items()
            for rally in source["rallies"]
            if args.original_rallies is None or rally["original_rally_index"] in args.original_rallies
        ]
        if not selected or (
            args.original_rallies is not None
            and {r["original_rally_index"] for _, r in selected} != set(args.original_rallies)
        ):
            raise ValueError("Selected original rally numbers must exist in the served manifest")
        for sid, source in manifest["sources"].items():
            for _, rally in [(s, r) for s, r in selected if s == sid]:
                if rally["end"] - rally["begin"] < (4.99 if combined else 5.3):
                    raise ValueError("Pilot rally is too short for an actual five-second check")
                # Fresh default canvases keep each exact RGBA check on the same
                # rendering backend rather than triggering repeated-readback migration.
                page = browser.new_page(viewport=dict(width=1366, height=768))
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(args.url + f"?source={sid}&rally={rally['index']}")
                page.wait_for_function("window.viewerReady===true", timeout=90000)
                page.evaluate("async(i)=>{await viewer.selectRally(i);return true}", rally["index"])
                page.evaluate("async(t)=>{await viewer.seek(t);return true}", rally["begin"] + (0 if combined else 0.1))
                page.wait_for_function("!frameBusy && lastPairedFrame.athletes.length===2", timeout=45000)
                before_rgb = visible_rgb_checks(page)
                assert all(r["visible"] and r["different_rgba_components"] == 0 for r in before_rgb), before_rgb
                page.locator("#fitView").click()
                page.evaluate(
                    """()=>{window.pilotSamples=[];window.pilotSampler=setInterval(()=>{
                  if(frameBusy||!playing||!committedPairedFrame)return;
                  const s=document.querySelector('#scene').contentWindow.REVIEW_STATE;if(!s)return;
                  pilotSamples.push({scan:committedPairedFrame.scan,time:viewer.commonTime(),scene:s.scan,
                    vertices:s.vertexCount,meshes:s.meshCount,error:s.webglError,
                    projections:projectionStates.map(x=>({scan:x.scan,error:x.webglError}))});},40);} """
                )
                page.locator("#loop").uncheck()
                page.locator("#play").click()
                page.wait_for_function("playing && !preparing", timeout=90000)
                begin = page.evaluate("viewer.commonTime()")
                page.wait_for_function("(t)=>viewer.commonTime()>=t+5", arg=begin, timeout=90000)
                page.evaluate("viewer.pause()")
                page.wait_for_function("!playing && !preparing && !frameBusy", timeout=45000)
                page.evaluate("clearInterval(pilotSampler)")
                samples = page.evaluate("pilotSamples")
                after_rgb = visible_rgb_checks(page)
                end = page.evaluate("viewer.commonTime()")
                scans = {s["scan"] for s in samples}
                assert end - begin >= 4.9 and len(scans) >= 20, (begin, end, len(scans))
                assert all(
                    s["scan"] == s["scene"]
                    and s["meshes"] == 2
                    and s["vertices"] == 13780
                    and not s["error"]
                    and all(x["scan"] == s["scan"] and not x["error"] for x in s["projections"])
                    for s in samples
                ), samples[-3:]
                assert all(r["visible"] and r["different_rgba_components"] == 0 for r in after_rgb), after_rgb
                layout = page.evaluate(
                    """()=>({height:innerHeight,scroll:document.documentElement.scrollHeight,
                     bottom:document.querySelector('footer').getBoundingClientRect().bottom})"""
                )
                assert layout["bottom"] <= layout["height"] + 1, layout
                page.screenshot(path=str(output / f"{sid}_rally{rally['index']}.png"))
                result = dict(
                    source_id=sid,
                    rally=rally["index"],
                    original_rally_index=rally["original_rally_index"],
                    passed=True,
                    actual_begin=begin,
                    actual_end=end,
                    unique_scans=len(scans),
                    samples=samples,
                    initial_rgb=before_rgb,
                    final_rgb=after_rgb,
                    desktop_layout=layout,
                )
                results.append(result)
                write_json(
                    output / "continuous_progress.json", dict(completed=len(results), rallies=len(selected))
                )
                print(dict(rally=rally["index"], scans=len(scans), seconds=end - begin, passed=True), flush=True)
                page.close()
        outlier = None
        if args.outlier_scan is not None or "20261019_152542" in manifest["sources"]:
            outlier_sid = args.outlier_source or ("20261019_152542" if args.outlier_scan is None else next(iter(manifest["sources"])))
            outlier_scan = 443 if args.outlier_scan is None else args.outlier_scan
            outlier_source = manifest["sources"][outlier_sid]
            outlier_time = outlier_source["scan_times"][outlier_scan]
            outlier_rally = next((r for r in outlier_source["rallies"] if r["begin"] <= outlier_time <= r["end"]),
                                 outlier_source["rallies"][0])
            worst = browser.new_page(viewport=dict(width=1366, height=768))
            worst.goto(args.url + f"?source={outlier_sid}&rally={outlier_rally['index']}&scan={outlier_scan}")
            worst.wait_for_function("window.viewerReady===true && !frameBusy", timeout=90000)
            worst.locator("#fitView").click()
            screenshot = output / f"ground_outlier_scan{outlier_scan}.png"
            worst.screenshot(path=str(screenshot))
            worst.close()
            outlier = dict(source_id=outlier_sid, scan=outlier_scan, screenshot=str(screenshot))
        version = browser.version
        browser.close()
    assert not errors and len(results) == len(selected), errors
    write_json(
        output / "continuous_verification.json",
        dict(
            passed=True,
            started_utc=started,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            url=args.url,
            review_only=True,
            training_published=False,
            data_relocations=manifest.get("data_relocations"),
            runtime_relocations=manifest.get("runtime_relocations"),
            browser_version=version,
            clips=len(results),
            selected_original_rallies=args.original_rallies,
            outlier_case=outlier,
            results=results,
            page_errors=errors,
            native_rgb_comparison="Exact same-backend RGBA, no tolerance; fresh default canvases for every rally",
        ),
    )


if __name__ == "__main__":
    main()
