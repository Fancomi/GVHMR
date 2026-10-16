"""Check the reported 1009 opening-seconds color regression in independent Chrome."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from verify_badminton_viewer import visible_rgb_checks


def snapshot(page):
    value = page.evaluate("""()=>({scan:lastPairedFrame.lidar_index,
        actors:lastPairedFrame.athletes.map(a=>({id:a.motion_id,color:a.color,accepted:a.accepted})),
        frameActors:lastPairedFrame.frame.athletes.map(a=>({id:a.motion_id,color:a.color})),
        scene:document.getElementById('scene').contentWindow.REVIEW_STATE,projections:projectionStates,
        footer:document.getElementById('fitInfo').textContent})""")
    assert value["actors"]
    assert {a["id"]: a["color"] for a in value["actors"]} == {a["id"]: a["color"] for a in value["frameActors"]}
    assert value["scene"]["scan"] == value["scan"] and not value["scene"]["webglError"]
    assert all(s["scan"] == value["scan"] and not s["webglError"] for s in value["projections"])
    value["visible_rgb"] = visible_rgb_checks(page)
    assert all(r["visible"] and r["different_rgba_components"] == 0 for r in value["visible_rgb"])
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8767")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/badminton_label_audit_v2/stable_color_browser.json")
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=dict(width=1366, height=768))
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        base = args.url.rstrip("/")
        page.goto(base + "/?source=20261009_202713&rally=1&scan=1535", wait_until="domcontentloaded")
        page.wait_for_function("window.viewerReady===true", timeout=180000)
        linked = snapshot(page)
        assert linked["scan"] == 1535
        checks = []
        for offset in (0, 1, 1.2, 2, 3, 4):
            page.evaluate("(dt)=>viewer.seek(152.01425+dt)", offset)
            value = snapshot(page)
            far = next(a for a in value["actors"] if ":12:0" in a["id"])
            assert far["color"] == [79, 163, 255]
            checks.append(value)
            if offset == 1.2:
                page.screenshot(path=str(args.output.parent / "stable_tracks_1535.png"))
        # This old-full scan lies outside the rally subset; its deep link must
        # still select the exact stored scan and disable rally looping.
        page.goto(base + "/?source=20261012_163452&rally=1&scan=8280", wait_until="domcontentloaded")
        page.wait_for_function("window.viewerReady===true", timeout=180000)
        outside = snapshot(page)
        assert outside["scan"] == 8280 and not page.is_checked("#loop")
        assert not errors, errors
        args.output.write_text(
            json.dumps(dict(passed=True, exact_scan_links=[linked, outside], checks=checks, errors=errors), indent=2),
            encoding="utf-8",
        )
        browser.close()
    print("PASS: stable track12 colors, exact scan links, same-scan 3D/dual GL and native RGB pixels")


if __name__ == "__main__":
    main()
