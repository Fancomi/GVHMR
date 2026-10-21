"""Hold actual frame requests and prove pause freezes the displayed native scan."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from playwright.sync_api import sync_playwright
import psutil

from hmr4d import PROJ_ROOT
from verify_badminton_viewer import visible_rgb_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8781/")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    process.cpu_affinity([0, 1])
    started = datetime.now(timezone.utc).isoformat()
    code = (PROJ_ROOT / "hmr4d/lidar_mocap/rally_viewer_client.js").read_bytes()
    held, errors = [], []
    counter = [0]
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=dict(width=1366, height=768))
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/viewer.js", lambda route: route.fulfill(body=code, content_type="application/javascript"))
        page.goto(args.url)
        page.wait_for_function("window.viewerReady===true && !frameBusy", timeout=90000)
        page.locator("#loop").uncheck()

        def delay(route):
            counter[0] += 1
            if counter[0] <= 3:
                route.continue_()
            else:
                held.append(route)

        page.route("**/api/bundles?*", delay)
        page.locator("#play").click()
        page.wait_for_function("playing && !preparing", timeout=60000)
        page.wait_for_function("playing && prefetchTasks.size===3 && [...prefetchTasks.values()].every(t=>t.remaining===4) && [...readyFrames.keys()].every(i=>i<=lastData.lidar_index)", timeout=60000)
        assert len(held) == 3, len(held)
        before = page.evaluate("({scan:committedPairedFrame.scan,time:lastData.frame.offset_s})")
        page.wait_for_timeout(2100)
        starved = page.evaluate("({scan:committedPairedFrame.scan,time:lastData.frame.offset_s,clock:reviewTime,next:source.scan_times[committedPairedFrame.scan+1]})")
        assert held and starved["scan"] == before["scan"], (len(held), before, starved)
        assert starved["clock"] <= starved["next"] + 1e-6, starved
        geometry = []
        page.on("request", lambda request: geometry.append(request.url) if "/api/geometry?" in request.url else None)
        page.locator("#play").click()
        page.wait_for_function("!playing && !preparing && !frameBusy")
        paused = page.evaluate("committedPairedFrame.scan")
        assert paused == before["scan"] and not geometry, (before, paused, geometry)
        for route in held:
            route.continue_()
        held.clear()
        page.unroute("**/api/bundles?*", delay)
        page.wait_for_timeout(600)
        assert page.evaluate("committedPairedFrame.scan") == paused
        page.locator("#play").click()
        page.wait_for_function("(n)=>committedPairedFrame.scan>=n+3", arg=paused, timeout=60000)
        page.evaluate("viewer.pause()")
        page.wait_for_function("!playing && !frameBusy")
        state = page.frame_locator("#scene").locator("body").evaluate("()=>REVIEW_STATE")
        scan = page.evaluate("committedPairedFrame.scan")
        projections = page.evaluate("projectionStates")
        rgb = visible_rgb_checks(page)
        assert scan > paused and state["scan"] == scan and not state["webglError"]
        assert all(p["scan"] == scan and not p["webglError"] for p in projections)
        assert all(r["visible"] and r["different_rgba_components"] == 0 for r in rgb), rgb
        assert not errors, errors
        args.output.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(args.output.with_suffix(".png")))
        args.output.write_text(json.dumps(dict(passed=True, url=args.url, started_utc=started,
            completed_utc=datetime.now(timezone.utc).isoformat(), starvation=starved,
            paused_scan=paused, resumed_scan=scan, pause_geometry_requests=geometry,
            exact_visible_rgb=rgb, projections=projections, page_errors=errors), indent=2), encoding="utf-8")
        browser.close()
    print("Buffered pause/resume with exact native RGB passed", flush=True)


if __name__ == "__main__":
    main()
