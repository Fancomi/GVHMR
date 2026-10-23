"""Verify all declared sources through real five-second native playback clips."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.canonical_tracks import checked_review, remap_row
from hmr4d.lidar_mocap.expanded_dataset import binding, check_binding
from hmr4d.lidar_mocap.rally_pipeline import read_json
from hmr4d.lidar_mocap.repair_execution import write_progress_json as write_json
from verify_badminton_viewer import visible_rgb_checks
from badminton_viewer_contracts import check_viewer_coverage, complete_clip_plan


def _canonical_context(root, canonical_path, *, release_policy=None, release_policy_sha256=None):
    """Check a real canonical view before any browser or boundary planning."""
    if (release_policy is None) != (release_policy_sha256 is None):
        raise ValueError("Release policy path and SHA256 must be supplied together")
    root = Path(root).resolve()
    input_path = root / "new_source_input.json"
    current = read_json(input_path) if input_path.is_file() else {}
    policy = None
    if current.get("release_policy") is not None or release_policy is not None:
        if release_policy is None:
            raise ValueError("V2 playback requires an explicit release policy path and SHA256")
        from hmr4d.lidar_mocap.incremental_release import load_release_policy

        policy = load_release_policy(release_policy, release_policy_sha256)
        if current.get("release_policy") != policy.binding:
            raise ValueError("Playback source input differs from its explicit release policy")
        if canonical_path is None:
            raise ValueError("V2 playback requires its actual reviewed canonical publication")
    if canonical_path is None:
        return None, None, policy, None
    selected = binding(canonical_path)
    canonical = read_json(selected["path"])
    proof = None
    if canonical.get("schema") == "badminton-canonical-bilateral-publication-v2":
        if policy is None:
            raise ValueError("A no-change publication requires the explicit v2 release policy")
        from hmr4d.lidar_mocap.canonical_no_change import verify_no_change_publication

        proof = verify_no_change_publication(
            canonical, release_policy=release_policy, release_policy_sha256=release_policy_sha256
        )
    elif (
        canonical.get("schema") != "badminton-canonical-bilateral-publication-v1"
        or canonical.get("canonical_identity_revision", {}).get("mode") == "reviewed_no_changes"
    ):
        raise ValueError("Select the exact reviewed cut or typed no-change canonical publication")
    if policy is not None:
        body = binding(root / "final_labels/datasets/dataset_manifest.json")
        if (
            canonical.get("release_policy") != policy.binding
            or canonical.get("source_dataset_manifest") != body["path"]
            or canonical.get("source_dataset_manifest_sha256") != body["sha256"]
        ):
            raise ValueError("V2 playback canonical view uses another current full-body publication")
        policy.recheck()
    check_binding(selected)
    return selected, canonical, policy, proof


def verify(
    root, url, canonical_path=None, output_root=None, *, cdp_url, release_policy=None, release_policy_sha256=None
):
    root = Path(root).resolve()
    selected_policy = dict(release_policy=release_policy, release_policy_sha256=release_policy_sha256)
    canonical_binding, canonical, policy, no_change_proof = _canonical_context(root, canonical_path, **selected_policy)
    no_changes = no_change_proof is not None
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
    coverage = check_viewer_coverage(root / "final_labels", manifest)
    served_manifest_path = output / "served_viewer_manifest.json"
    write_json(served_manifest_path, manifest)
    clips = []
    boundary_rows, boundary_results = {}, []
    if canonical_path:
        if manifest.get("canonical_publication") != canonical_binding:
            raise ValueError("Browser is not serving the selected canonical publication")
    if canonical_path and not no_changes:
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
    clips = complete_clip_plan(manifest, clips)
    required_clips = len(clips)
    if required_clips < coverage["minimum_clips"]:
        raise ValueError("Every source needs two genuine five-second clips")
    plan = dict(
        schema="badminton-fresh-browser-clips-v1",
        source_completion=frozen,
        canonical_publication=canonical_binding,
        selection=(
            "Native identity-boundary regressions plus temporally diverse original rallies across every nonempty source"
            if canonical_path and not no_changes
            else "First and last sufficiently long original rally of each nonempty source"
        ),
        clips=clips,
        declared_empty_sources=manifest["empty_sources"],
        frozen_source_coverage=coverage,
    )
    if policy is not None:
        plan.update(
            release_policy=policy.binding, canonical_mode="reviewed_no_changes" if no_changes else "reviewed_cuts"
        )
        if no_changes:
            plan["canonical_no_change_proof"] = canonical["canonical_identity_revision"]["verification"]
    write_json(output / "clip_plan.json", plan)
    errors, results = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(cdp_url)
        context = browser.new_context(viewport=dict(width=1366, height=768))
        page = context.new_page()
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
            page.evaluate("""()=>{
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
            }""")
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
        if canonical_path and not no_changes:
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
        context.close()
    if errors or len(results) != required_clips:
        raise ValueError("Browser reported errors or did not complete the frozen-source clip plan: " + str(errors))
    check_binding(frozen)
    for evidence in coverage["evidence"]:
        check_binding(evidence)
    with urlopen(url + "api/manifest", timeout=10) as response:
        if json.load(response) != manifest:
            raise ValueError("The served viewer manifest changed during actual browser acceptance")
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
        viewer_manifest=binding(served_manifest_path),
        url=url,
        browser_version=version,
        declared_source_count=coverage["declared_source_count"],
        played_sources=len({clip["source_id"] for clip in results}),
        frozen_source_coverage=coverage,
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
    if policy is not None:
        receipt.update(
            release_policy=policy.binding, canonical_mode="reviewed_no_changes" if no_changes else "reviewed_cuts"
        )
        policy.recheck()
        for path, sha in source["evidence"].items():
            check_binding(dict(path=path, sha256=sha))
        check_binding(canonical_binding)
        if no_changes:
            from hmr4d.lidar_mocap.canonical_no_change import validate_no_change_acceptance

            receipt["canonical_no_change_proof"] = canonical["canonical_identity_revision"]["verification"]
            validate_no_change_acceptance(canonical, no_change_proof["continuity_audit"], receipt, **selected_policy)
    write_json(output / "acceptance.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8773/")
    parser.add_argument("--canonical-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cdp-url", help=argparse.SUPPRESS)
    parser.add_argument("--browser-session-helper", type=Path, help="Existing dedicated Chrome lifecycle helper")
    parser.add_argument("--release-policy", type=Path, help="Explicit frozen v2 release policy")
    parser.add_argument("--release-policy-sha256", help="Exact SHA256 of the explicit v2 policy")
    args = parser.parse_args()
    if (args.release_policy is None) != (args.release_policy_sha256 is None):
        parser.error("--release-policy and --release-policy-sha256 must be supplied together")
    if not args.root.resolve().is_relative_to(PROJ_ROOT / "outputs"):
        parser.error("Select an explicit fresh production outputs root")
    if args.output_root and not args.output_root.resolve().is_relative_to(args.root.resolve()):
        parser.error("Fresh acceptance evidence must stay inside its repair run")
    if not args.cdp_url:
        node = shutil.which("node")
        if node is None:
            parser.error("Node is required to run the shared dedicated Chrome lifecycle helper")
        command = [
            node,
            str(Path(__file__).with_name("check_badminton_fresh_playback.mjs")),
            "--python",
            sys.executable,
            *sys.argv[1:],
        ]
        raise SystemExit(subprocess.run(command, check=False).returncode)
    parsed = urlsplit(args.cdp_url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port:
        parser.error("CDP must use the dedicated loopback browser endpoint")
    receipt = verify(
        args.root,
        args.url,
        args.canonical_manifest,
        args.output_root,
        cdp_url=args.cdp_url,
        release_policy=args.release_policy,
        release_policy_sha256=args.release_policy_sha256,
    )
    print(json.dumps(dict(passed=receipt["passed"], clips=receipt["clips"], played_sources=receipt["played_sources"])))


if __name__ == "__main__":
    main()
