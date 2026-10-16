"""Exercise actual H264 playback/seeking in an independent installed Chrome.

Install optional browser driver with `uv pip install playwright`; Chrome itself
is used via channel=chrome, so no bundled browser download is required.
"""

import argparse
import json
from pathlib import Path
import time
from urllib.request import urlopen

from playwright.sync_api import sync_playwright


def visible_rgb_checks(page):
    """Compare the actual visible RGB canvases to the PTS-bound decoded JPEGs."""
    return page.evaluate("""async()=>{const result=[];
    for(const [i,cam] of ['cam1','cam2'].entries()){
      const source=lastPairedFrame.native_images[cam],c=document.getElementById('nativeFrame'+(i+1));
      const bitmap=await createImageBitmap(await(await fetch('data:image/jpeg;base64,'+source.jpeg_base64)).blob());
      const ref=document.createElement('canvas');ref.width=c.width;ref.height=c.height;
      ref.getContext('2d',{alpha:false}).drawImage(bitmap,0,0,ref.width,ref.height);bitmap.close();
      const a=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
      const b=ref.getContext('2d').getImageData(0,0,ref.width,ref.height).data;
      let different=0;for(let k=0;k<a.length;k++)if(a[k]!==b[k])different++;
      result.push({camera:cam,record_index:source.record_index,media_pts_us:source.media_pts_us,
       raw_bgr_sha256:source.raw_bgr_sha256,native_jpeg_sha256:source.jpeg_sha256,different_rgba_components:different,
       visible:c.style.visibility==='visible'});
    }return result;}""")


def verify_atomic_loading(page):
    page.evaluate("viewer.selectSource('20261012_163452')")
    page.select_option("#mode", "paired")
    page.evaluate("viewer.seek(32.4377)")
    before = page.evaluate("committedPairedFrame.scan")
    held = []
    page.route("**/api/geometry?*", lambda route: held.append(route))
    try:
        page.evaluate("viewer.seek(33.0377); true")
        page.wait_for_function("frameBusy")
        page.wait_for_timeout(80)
        during = page.evaluate("committedPairedFrame.scan")
        scene = page.frame_locator("#scene").locator("body").evaluate("()=>REVIEW_STATE.scan")
        rgb = visible_rgb_checks(page)
        assert held and during == before == scene and all(c["different_rgba_components"] == 0 for c in rgb), (
            before,
            during,
            scene,
            rgb,
        )
        held[0].continue_()
        held.clear()
        page.wait_for_function("!frameBusy", timeout=45000)
    finally:
        for route in held:
            route.continue_()
        page.unroute("**/api/geometry?*")
    after = page.evaluate("committedPairedFrame.scan")
    assert after > before, (before, after)
    page.evaluate("viewer.play()")
    page.wait_for_timeout(850)
    page.evaluate("viewer.pause()")
    paused = page.evaluate("committedPairedFrame.scan")
    scene = page.frame_locator("#scene").locator("body").evaluate("()=>REVIEW_STATE.scan")
    projections = page.evaluate("projectionStates")
    final_rgb = visible_rgb_checks(page)
    assert paused == scene and all(c["scan"] == paused and not c["webglError"] for c in projections)
    assert all(c["visible"] and c["different_rgba_components"] == 0 for c in final_rgb)
    return dict(
        passed=True,
        before_scan=before,
        loading_scan=during,
        after_seek_scan=after,
        paused_scan=paused,
        retained_previous_complete_frame_while_loading=True,
        loading_rgb=rgb,
        paused_rgb=final_rgb,
        projections=projections,
    )


def verify_controls(page, output):
    """Use real controls and iframe keyboard focus, with viewport/FOV checks."""
    page.set_viewport_size(dict(width=1366, height=768))
    page.locator("#play").click()
    page.wait_for_function("playing && !preparing", timeout=45000)
    first = page.evaluate("committedPairedFrame.scan")
    page.wait_for_timeout(1500)
    after_click = page.evaluate("committedPairedFrame.scan")
    assert after_click > first + 3, (first, after_click)
    page.locator("#play").click()
    page.wait_for_function("!playing && !preparing && !frameBusy")
    page.frame_locator("#scene").locator("#scene").click()
    first = page.evaluate("committedPairedFrame.scan")
    page.keyboard.press("Space")
    page.wait_for_function("playing", timeout=45000)
    page.wait_for_timeout(1300)
    after_space = page.evaluate("committedPairedFrame.scan")
    assert after_space > first + 3, (first, after_space)
    page.keyboard.press("Space")
    page.wait_for_function("!playing && !preparing && !frameBusy")
    page.wait_for_timeout(300)
    step_before = page.evaluate("committedPairedFrame.scan")
    page.keyboard.press("ArrowRight")
    page.wait_for_function("(n)=>!frameBusy && committedPairedFrame.scan===n+1", arg=step_before, timeout=45000)
    page.keyboard.press("ArrowLeft")
    page.wait_for_function("(n)=>!frameBusy && committedPairedFrame.scan===n", arg=step_before, timeout=45000)
    page.locator("h1").click()
    page.keyboard.press("Space")
    page.wait_for_function("playing", timeout=45000)
    page.keyboard.press("Space")
    page.wait_for_function("!playing && !frameBusy")
    seek = page.locator("#seek")
    bounds = seek.bounding_box()
    seek.click(position=dict(x=bounds["width"] / 2, y=bounds["height"] / 2))
    page.wait_for_function("!frameBusy && !scrubbing")
    sought = page.evaluate("({actual:lastPairedFrame.frame.offset_s,expected:(Math.max(0,clip.begin)+clip.end)/2})")
    assert abs(sought["actual"] - sought["expected"]) < 0.2, sought
    page.keyboard.press("Space")
    page.wait_for_function("playing", timeout=45000)
    page.keyboard.press("Space")
    page.wait_for_function("!playing && !frameBusy")
    rgb = visible_rgb_checks(page)
    assert all(c["visible"] and c["different_rgba_components"] == 0 for c in rgb)
    # Hold preparation requests: a second real click must cancel the pending
    # start, and late responses must not restart playback behind the user.
    page.evaluate("viewer.selectSource('20261012_163452')")
    held = []
    page.route("**/api/bundles?*", lambda route: held.append(route))
    try:
        page.locator("#play").click()
        page.wait_for_function("preparing")
        page.wait_for_timeout(150)
        page.locator("#play").click()
        page.wait_for_function("!preparing && !playing && !frameBusy")
        for route in held:
            route.continue_()
        held.clear()
    finally:
        for route in held:
            route.continue_()
        page.unroute("**/api/bundles?*")
    page.wait_for_timeout(300)
    assert page.evaluate("!playing && !preparing && document.getElementById('play').textContent==='播放'")
    # A short rally must actually wrap and continue; switching sources must
    # not leave the play button in its preparation state.
    page.evaluate("viewer.selectSource('20261012_160054')")
    page.locator("#loop").check()
    page.evaluate(
        "window.controlScans=[];window.controlSampler=setInterval(()=>controlScans.push(committedPairedFrame.scan),60)"
    )
    page.locator("#play").click()
    page.wait_for_function("playing", timeout=45000)
    page.wait_for_function("controlScans.some((v,i)=>i>0&&v<controlScans[i-1]-5)", timeout=20000)
    page.wait_for_function("playing && !preparing", timeout=45000)
    page.locator("#play").click()
    page.wait_for_function("!playing && !frameBusy")
    page.evaluate("clearInterval(controlSampler)")
    loop_scans = page.evaluate("controlScans")
    viewports = []
    for width, height in ((1366, 768), (1920, 1080), (1024, 768), (390, 844)):
        page.set_viewport_size(dict(width=width, height=height))
        page.locator("#fitView").click()
        page.wait_for_timeout(300)
        layout = page.evaluate(
            """()=>({scrollY,viewport:innerHeight,
        scrollHeight:document.documentElement.scrollHeight,
        boxes:Object.fromEntries(['.inspection','#scene','.view:first-child','.view:last-child','#seek','#play','#fitInfo'].map(s=>[s,document.querySelector(s).getBoundingClientRect().toJSON()]))})"""
        )
        assert all(
            b["left"] >= -1 and b["top"] >= -1 and b["right"] <= width + 1 and b["bottom"] <= height + 1
            for b in layout["boxes"].values()
        ), (width, height, layout)
        assert layout["boxes"]["#seek"]["bottom"] < layout["boxes"][".inspection"]["top"]
        assert layout["scrollHeight"] <= height + 1 and layout["scrollY"] == 0, layout
        fov = (
            page.frame_locator("#scene")
            .locator("body")
            .evaluate(
                """()=>{
        const m=matrix(),f=clip.frames[index];let outside=0,count=0,maxNdc=0;
        for(const a of f.athletes)for(let i=0;i<a._vertices.length;i+=3){
         const [x,y,z]=a._vertices.slice(i,i+3),w=m[3]*x+m[7]*y+m[11]*z+m[15];
         const nx=(m[0]*x+m[4]*y+m[8]*z+m[12])/w,ny=(m[1]*x+m[5]*y+m[9]*z+m[13])/w;
         count++;maxNdc=Math.max(maxNdc,Math.abs(nx),Math.abs(ny));if(w<=0||Math.abs(nx)>1.001||Math.abs(ny)>1.001)outside++;}
        const c=canvas.getBoundingClientRect(),controls=document.querySelector('.controls').getBoundingClientRect();
        return {count,outside,maxNdc,canvas:c.toJSON(),controls:controls.toJSON(),height:innerHeight,scrollHeight:document.documentElement.scrollHeight};}"""
            )
        )
        assert fov["count"] > 0 and fov["outside"] == 0 and fov["controls"]["bottom"] <= fov["height"] + 1, fov
        page.screenshot(path=str(output.with_name(f"controls_{width}x{height}.png")), full_page=False)
        viewports.append(dict(width=width, height=height, layout=layout, full_mesh_fov=fov, passed=True))
    return dict(
        passed=True,
        button_play_advanced_to=after_click,
        iframe_space_advanced_to=after_space,
        parent_space_toggle=True,
        timeline_space_toggle=True,
        native_scan_arrow_steps=True,
        preparation_cancelled_without_late_restart=True,
        source_switch_and_rally_loop=True,
        loop_scan_range=[min(loop_scans), max(loop_scans)],
        seek=sought,
        visible_native_rgb=rgb,
        viewports=viewports,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8767/")
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/badminton_rallies_v1/viewer/browser_verification.json")
    )
    parser.add_argument("--atomic-only", action="store_true", help="Only loading/pause/pixel transaction regression")
    parser.add_argument(
        "--controls-only", action="store_true", help="Real play/keyboard/seek and viewport/FOV regression"
    )
    args = parser.parse_args()
    errors = []
    checks = []
    overlays = []
    cloud_checks = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(180):
        try:
            with urlopen(args.url, timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            if attempt == 179:
                raise
            time.sleep(1)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"]
        )
        page = browser.new_page(viewport=dict(width=1500, height=1050))
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url, wait_until="domcontentloaded")
        page.wait_for_function("window.viewerReady === true", timeout=180000)
        if args.controls_only:
            result = verify_controls(page, args.output)
            result.update(page_errors=errors, browser_version=browser.version, url=args.url)
            assert not errors, errors
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            browser.close()
            print(
                json.dumps(dict(passed=True, viewports=len(result["viewports"]), report=str(args.output))), flush=True
            )
            return
        if args.atomic_only:
            result = verify_atomic_loading(page)
            result.update(page_errors=errors, browser_version=browser.version, url=args.url)
            assert not errors, errors
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            page.screenshot(path=str(args.output.with_suffix(".png")), full_page=True)
            browser.close()
            print(json.dumps(result))
            return
        page.select_option("#mode", "native")
        manifest = page.evaluate('fetch("/api/manifest").then(r=>r.json())')
        for sid, source in manifest["sources"].items():
            page.select_option("#mode", "native")
            page.evaluate("(sid)=>window.viewer.selectSource(sid)", sid)
            # Beginning, central and final rally exercise distant MP4 ranges.
            candidates = source["rallies"]
            for index in sorted({0, len(candidates) // 2, len(candidates) - 1}):
                rally = candidates[index]
                page.evaluate("(i)=>window.viewer.selectRally(i)", rally["index"])
                page.wait_for_function("[cam1,cam2].every(v=>!v.seeking && v.readyState>=2)", timeout=45000)
                before = page.evaluate("[cam1,cam2].map(v=>v.currentTime)")
                page.evaluate("window.viewer.play()")
                page.wait_for_timeout(1300)
                after = page.evaluate("[cam1,cam2].map(v=>v.currentTime)")
                sync_error = page.evaluate("Math.abs(viewer.commonTime(1)-viewer.commonTime(2))")
                quality = page.evaluate(
                    "[cam1,cam2].map(v=>({width:v.videoWidth,height:v.videoHeight,error:v.error&&v.error.message,decoded:v.getVideoPlaybackQuality().totalVideoFrames}))"
                )
                page.evaluate("window.viewer.pause()")
                seek_t = min(source["duration"] - 0.1, max(0, rally["begin"] + 0.65))
                page.evaluate("(t)=>window.viewer.seek(t)", seek_t)
                page.wait_for_function("[cam1,cam2].every(v=>!v.seeking && v.readyState>=2)", timeout=45000)
                actual = page.evaluate("window.viewer.commonTime()")
                passed = (
                    all(b > a + 0.4 for a, b in zip(before, after))
                    and abs(actual - seek_t) < 0.06
                    and sync_error < 0.06
                    and all(
                        q["width"] == 1920 and q["height"] == 1080 and not q["error"] and q["decoded"] > 0
                        for q in quality
                    )
                )
                check = dict(
                    source_id=sid,
                    rally_index=rally["index"],
                    before_seconds=before,
                    after_seconds=after,
                    seek_requested=seek_t,
                    seek_actual=actual,
                    playback_common_time_error_ms=sync_error * 1000,
                    decode=quality,
                    passed=passed,
                )
                checks.append(check)
                print(json.dumps(check), flush=True)
                if not passed:
                    raise AssertionError(f'Native video playback/seek failed: {sid} rally{rally["index"]}')
            if source["baseline"]:
                page.evaluate("window.viewer.pairedFrame()")
                page.wait_for_function("[cam1,cam2].every(v=>!v.seeking)", timeout=30000)
                overlay = page.evaluate(
                    "window.lastPairedFrame&&({state:lastPairedFrame.state,athletes:lastPairedFrame.athletes.length,native:lastPairedFrame.native})"
                )
                if not overlay or overlay["state"] != "available" or overlay["athletes"] < 1:
                    # A counted rally may end with no segmented person; try its
                    # center rather than claiming unbuilt geometry is present.
                    page.evaluate("(t)=>window.viewer.seek(t)", (candidates[-1]["begin"] + candidates[-1]["end"]) / 2)
                    page.evaluate("window.viewer.pairedFrame()")
                    overlay = page.evaluate(
                        "window.lastPairedFrame&&({state:lastPairedFrame.state,athletes:lastPairedFrame.athletes.length})"
                    )
                if not overlay or overlay["state"] != "available":
                    raise AssertionError(f"Baseline exact overlay missing: {sid}")
                page.wait_for_function("[cam1,cam2].every(v=>!v.seeking)", timeout=30000)
                errors_pts = page.evaluate(
                    "['cam1','cam2'].map(c=>Math.abs(lastPairedFrame.native_images[c].media_pts_us/1e6-lastPairedFrame.native[c].pts))"
                )
                if max(errors_pts) > 0.001:
                    raise AssertionError(f"Overlay native PTS mismatch: {sid}")
                overlays.append(
                    dict(source_id=sid, athletes=overlay["athletes"], native_pts_error_seconds=errors_pts, passed=True)
                )
                scene = page.frame_locator("#scene")
                state = scene.locator("body").evaluate("()=>window.REVIEW_STATE")
                if state["meshCount"] < 1 or state["vertexCount"] != state["meshCount"] * 6890 or state["webglError"]:
                    raise AssertionError(f"Full SMPL WebGL mesh missing: {sid} {state}")
                initial_scan = state["scan"]
                page.evaluate("viewer.play()")
                page.wait_for_timeout(1800)
                page.evaluate("viewer.pause()")
                page.wait_for_function("!frameBusy", timeout=30000)
                final_state = scene.locator("body").evaluate("()=>window.REVIEW_STATE")
                if final_state["scan"] <= initial_scan + 3 or final_state["webglError"]:
                    raise AssertionError(f"3D scan playback did not advance: {sid} {final_state}")
                overlays[-1].update(
                    full_mesh_vertices=state["vertexCount"],
                    mesh_count=state["meshCount"],
                    original_cloud_points=state["pointCount"],
                    displayed_background_points=state["displayedBackgroundPoints"],
                    play_begin_scan=initial_scan,
                    play_end_scan=final_state["scan"],
                    webgl_error=final_state["webglError"],
                )
                page.select_option("#mode", "native")
            # Every source must show an actual raw scan in 3D and project it
            # into both native cameras, including sources awaiting body fits.
            page.select_option("#mode", "paired")
            await_idle = "!frameBusy"
            page.wait_for_function(await_idle, timeout=30000)
            page.evaluate("(t)=>viewer.seek(t)", (candidates[0]["begin"] + candidates[0]["end"]) / 2)
            page.wait_for_function(await_idle, timeout=30000)
            state = page.frame_locator("#scene").locator("body").evaluate("()=>window.REVIEW_STATE")
            projected_mesh_vertices = page.evaluate(
                "lastPairedFrame.athletes.map(a=>[a.cameras.cam1.mesh.length,a.cameras.cam2.mesh.length])"
            )
            if any(count != 6890 for pair_counts in projected_mesh_vertices for count in pair_counts):
                raise AssertionError(f"Camera overlay lacks full projected SMPL: {sid}")
            observed = page.evaluate(
                "({source_id:committedPairedFrame.sourceId,scan:lastPairedFrame.lidar_index,points:lastPairedFrame.frame.point_count,projection_counts:Object.fromEntries(Object.entries(lastPairedFrame.background_projection).map(([k,v])=>[k,v.length])),native_pts_errors:['cam1','cam2'].map(c=>Math.abs(lastPairedFrame.native_images[c].media_pts_us/1e6-lastPairedFrame.native[c].pts))})"
            )
            valid = (
                state["scan"] == observed["scan"]
                and observed["source_id"] == sid
                and state["pointCount"] == observed["points"]
                and observed["points"] > 0
                and all(v > 0 for v in observed["projection_counts"].values())
                and max(observed["native_pts_errors"]) < 0.001
                and state["webglError"] == 0
            )
            if not valid:
                raise AssertionError(f"Seven-source raw scan/projection transaction failed: {sid}")
            rgb_checks = visible_rgb_checks(page)
            if any(not c["visible"] or c["different_rgba_components"] for c in rgb_checks):
                raise AssertionError(f"Visible native RGB does not match committed scan: {sid} {rgb_checks}")
            cloud_checks.append(
                dict(
                    passed=True,
                    **observed,
                    projected_mesh_vertices=projected_mesh_vertices,
                    webgl_error=state["webglError"],
                    visible_native_rgb=rgb_checks,
                )
            )
            roi_scan = observed["scan"]
            roi_count = observed["points"]
            page.locator("#allPoints").check()
            page.wait_for_function("!frameBusy && lastPairedFrame.frame.point_scope==='all'", timeout=45000)
            full_cloud = page.evaluate(
                "({scan:lastPairedFrame.lidar_index,count:lastPairedFrame.frame.point_count,raw:lastPairedFrame.frame.raw_point_count})"
            )
            assert full_cloud["scan"] == roi_scan and full_cloud["count"] == full_cloud["raw"] >= roi_count
            page.locator("#allPoints").uncheck()
            page.wait_for_function("!frameBusy && lastPairedFrame.frame.point_scope==='roi'", timeout=45000)
            assert page.evaluate("lastPairedFrame.frame.point_count") == roi_count
            cloud_checks[-1].update(
                roi_points=roi_count, all_source_points=full_cloud["raw"], point_scope_toggle_passed=True
            )
        page.screenshot(path=str(args.output.with_suffix(".png")), full_page=True)
        version = browser.version
        browser.close()
    result = dict(
        passed=not errors and all(c["passed"] for c in checks),
        browser="independent Chrome headless",
        browser_version=version,
        url=args.url,
        sources=len(manifest["sources"]),
        rallies=manifest["rally_count"],
        playback_seek_cases=len(checks),
        checks=checks,
        exact_overlay_checks=overlays,
        raw_cloud_and_projection_checks=cloud_checks,
        page_errors=errors,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        raise AssertionError(result)
    print(
        json.dumps(dict(passed=True, sources=result["sources"], cases=len(checks), report=str(args.output))), flush=True
    )


if __name__ == "__main__":
    main()
