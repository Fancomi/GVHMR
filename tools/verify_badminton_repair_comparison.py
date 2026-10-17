"""Exercise real Chrome playback, native RGB equality and six-view scan commits."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def rgb_check(page):
    return page.evaluate("""async()=>{
      const result=[];
      for(let i=1;i<=2;i++){
        const im=COMPARISON_LAST_BUNDLE.data.native_images['cam'+i];
        const bitmap=await createImageBitmap(await(await fetch('data:image/jpeg;base64,'+im.jpeg_base64)).blob());
        const ref=document.createElement('canvas');ref.width=1920;ref.height=1080;ref.getContext('2d').drawImage(bitmap,0,0);bitmap.close();
        const expected=ref.getContext('2d').getImageData(0,0,1920,1080).data;
        for(const prefix of ['old','new']){
          const actual=document.getElementById(prefix+'Rgb'+i).getContext('2d').getImageData(0,0,1920,1080).data;
          let different=0;for(let j=0;j<actual.length;j++)if(actual[j]!==expected[j])different++;
          result.push({camera:'cam'+i,version:prefix,different_rgba_components:different,record_index:im.record_index});
        }
      }return result;
    }""")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8772/")
    p.add_argument(
        "--output", type=Path, default=Path("outputs/badminton_body_repair_v2/comparison/browser_verification.json")
    )
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    errors = []
    checks = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"]
        )
        page = browser.new_page(viewport=dict(width=1600, height=1000))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.url)
        page.wait_for_function("window.COMPARISON_STATE", timeout=90000)
        clips = page.evaluate("COMPARISON_MANIFEST.clips")
        if not args.allow_partial:
            assert all(c["state"] == "complete" for c in clips)
        for i, c in enumerate(clips):
            if c["state"] != "complete":
                continue
            page.evaluate("(i)=>comparison.load(i)", i)
            page.wait_for_function(
                "(id)=>COMPARISON_STATE?.clip===id && COMPARISON_STATE.index===0 && !busy", arg=c["id"], timeout=90000
            )
            page.locator("#loop").uncheck()
            playback_start = page.evaluate("performance.now()")
            page.evaluate("comparison.play()")
            page.wait_for_function("COMPARISON_STATE.index===49 && !playing", timeout=45000)
            page.evaluate("comparison.pause()")
            page.wait_for_function("!busy", timeout=20000)
            state = page.evaluate("comparison.state()")
            rgb = rgb_check(page)
            playback = [r for r in state["commits"] if r["clip"] == c["id"] and r["time"] >= playback_start]
            assert [r["scan"] for r in playback] == c["lidar_indices"][1:], playback
            assert all(r["different_rgba_components"] == 0 for r in rgb), rgb
            assert all(s["scan"] == state["scan"] and not s["webglError"] for s in state["scenes"]), state
            assert all(s["scan"] == state["scan"] and not s["webglError"] for s in state["projections"]), state
            assert all(s["vertexCount"] == s["meshCount"] * 6890 for s in state["scenes"])
            for target in [49, 12, 25]:
                page.evaluate("(i)=>comparison.seek(i)", target)
                page.wait_for_function("(i)=>COMPARISON_STATE.index===i && !busy", arg=target, timeout=45000)
            # Hold one offscreen scene; no visible view may advance early.
            page.evaluate("""()=>{
              window.savedRender=workerRender;window.heldRender=null;
              window.frozen3d=['old3d','new3d'].map(id=>document.getElementById(id).toDataURL());
              workerRender=(which,...args)=>which===1&&args[0].lidar_index===clip.frames[40].scan?new Promise(resolve=>{window.heldRender=()=>savedRender(which,...args).then(resolve);}):savedRender(which,...args);
              comparison.seek(40);return true;
            }""")
            page.wait_for_function("window.heldRender!==null", timeout=20000)
            assert page.evaluate(
                'COMPARISON_STATE.index===25 && ["old3d","new3d"].every((id,i)=>document.getElementById(id).toDataURL()===frozen3d[i])'
            )
            assert all(r["different_rgba_components"] == 0 for r in rgb_check(page))
            page.evaluate("heldRender();workerRender=savedRender;true")
            page.wait_for_function("COMPARISON_STATE.index===40 && !busy", timeout=30000)
            page.locator("#zoom").check()
            page.locator("#zoom").uncheck()
            page.locator("#points").uncheck()
            page.wait_for_function("!busy", timeout=30000)
            page.locator("#points").check()
            page.wait_for_function("!busy", timeout=30000)
            page.locator("#failed").check()
            page.wait_for_function("!busy", timeout=30000)
            page.locator("#failed").uncheck()
            page.wait_for_function("!busy", timeout=30000)
            page.screenshot(path=str(args.output.with_name(c["id"] + ".png")))
            checks.append(
                dict(
                    clip=c["id"],
                    played_to_scan=state["scan"],
                    played_to_index=state["index"],
                    native_rgb=rgb,
                    playback_consecutive_scans=len(playback),
                    playback_wall_seconds=(playback[-1]["time"] - playback_start) / 1000,
                    six_view_scan_sync=True,
                    full_smpl_vertices=True,
                    seek_indices=[49, 12, 25],
                    atomic_loading=True,
                )
            )
        layout = []
        for width, height in [(1920, 1080), (1600, 1000), (1366, 768), (900, 700)]:
            page.set_viewport_size(dict(width=width, height=height))
            page.wait_for_function("!busy", timeout=30000)
            boxes = page.locator("#views figure").evaluate_all(
                "(els)=>els.map(e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,right:r.right,bottom:r.bottom}})"
            )
            assert all(
                b["x"] >= 0 and b["y"] >= 0 and b["right"] <= width + 1 and b["bottom"] <= height + 1 for b in boxes
            ), (width, height, boxes)
            layout.append(dict(viewport=[width, height], six_panels_visible=True))
        result = dict(
            passed=not errors and len(checks) == sum(c["state"] == "complete" for c in clips),
            complete=len(checks),
            total=len(clips),
            partial=args.allow_partial,
            checks=checks,
            layout=layout,
            page_errors=errors,
            browser_version=browser.version,
        )
        browser.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert result["passed"], result
    print(dict(passed=True, clips=len(checks), report=str(args.output)))


if __name__ == "__main__":
    main()
