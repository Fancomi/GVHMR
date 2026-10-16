"""Measure actual complete native-bundle commits, crop membership and RGB identity."""

import argparse
import json
from pathlib import Path
import time
from urllib.request import urlopen

import numpy as np
from playwright.sync_api import sync_playwright

from verify_badminton_viewer import visible_rgb_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8767/")
    parser.add_argument("--source", action="append")
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/badminton_rallies_v1/viewer/buffer_verification.json")
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i in range(180):
        try:
            with urlopen(args.url, timeout=2) as r:
                break
        except OSError:
            time.sleep(1)
    checks = []
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=dict(width=1500, height=1050))
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_function("window.viewerReady===true", timeout=180000)
        manifest = page.evaluate("manifest")
        for sid in args.source or manifest["sources"]:
            page.evaluate("(sid)=>viewer.selectSource(sid)", sid)
            longest = max(manifest["sources"][sid]["rallies"], key=lambda r: r["end"] - r["begin"])
            page.evaluate("(r)=>viewer.selectRally(r)", longest["index"])
            page.locator("#loop").uncheck()
            page.evaluate("(t)=>viewer.seek(t)", max(0, longest["begin"] + 0.3))
            initial = page.evaluate("({scan:committedPairedFrame.scan,time:lastPairedFrame.frame.offset_s})")
            began = time.perf_counter()
            page.evaluate("viewer.play()")
            startup = time.perf_counter() - began
            decoder_before = page.evaluate("fetch('/api/status').then(r=>r.json()).then(s=>s.preview_decoder)")
            page.wait_for_timeout(round(args.seconds * 1000))
            metrics = page.evaluate("viewer.bufferState()")
            before_pause = page.evaluate("({scan:committedPairedFrame.scan,time:lastPairedFrame.frame.offset_s})")
            decoder_after = page.evaluate("fetch('/api/status').then(r=>r.json()).then(s=>s.preview_decoder)")
            page.evaluate("viewer.pause()")
            rgb = visible_rgb_checks(page)
            stamps = [c["time"] for c in metrics["commits"]]
            gaps = np.diff(stamps)
            renderer = page.evaluate(
                '(()=>{const g=overlay1.getContext("webgl2"),e=g.getExtension("WEBGL_debug_renderer_info");return e?g.getParameter(e.UNMASKED_RENDERER_WEBGL):"unknown"})()'
            )
            result = dict(
                source_id=sid,
                seconds=args.seconds,
                startup_seconds=startup,
                complete_commits=len(stamps),
                source_advance_seconds=before_pause["time"] - initial["time"],
                begin_scan=initial["scan"],
                end_scan=before_pause["scan"],
                median_commit_gap_ms=float(np.median(gaps)) if len(gaps) else None,
                p95_commit_gap_ms=float(np.percentile(gaps, 95)) if len(gaps) else None,
                max_commit_gap_ms=float(max(gaps)) if len(gaps) else None,
                ready_frames=metrics["ready"],
                buffer_underruns=metrics["buffer_underruns"],
                buffer_wait_ms=metrics["buffer_wait_ms"],
                gaps_over_250ms=len(metrics["starvation_gaps"]),
                renderer=renderer,
                visible_rgb=rgb,
                scope=metrics["scope"],
                decoder_before=decoder_before,
                decoder_after=decoder_after,
                steady_playback_seeks=decoder_after["seeks"] - decoder_before["seeks"],
            )
            result["passed"] = len(stamps) >= args.seconds * 6.25 and all(
                c["different_rgba_components"] == 0 for c in rgb
            )
            checks.append(result)
            print(json.dumps(result), flush=True)
            args.output.write_text(
                json.dumps(
                    dict(passed=all(r["passed"] for r in checks) and not errors, checks=checks, page_errors=errors),
                    indent=2,
                ),
                encoding="utf-8",
            )
        browser.close()
    if errors or not all(r["passed"] for r in checks):
        raise AssertionError("Playback commit/pixel verification failed")


if __name__ == "__main__":
    main()
