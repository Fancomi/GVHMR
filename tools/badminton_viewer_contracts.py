"""Shared frozen-scope adapters for native viewer command entries and acceptance."""

from pathlib import Path

from hmr4d.lidar_mocap.expanded_dataset import binding, check_binding
from hmr4d.lidar_mocap.rally_pipeline import read_json
from hmr4d.lidar_mocap.rally_viewer import load_viewer_intake
from hmr4d.lidar_mocap.relocations import load_pilot_run_relocations, load_run_relocations


def viewer_relocations(root, manifest=None, sha256=None, calibration_data=None):
    if (manifest is None) != (sha256 is None):
        raise ValueError("--data-relocations and --data-relocations-sha256 must be supplied together")
    registry = read_json(Path(root) / "source_registry.json")
    if registry.get("schema") != "badminton-formal-derived-source-registry-v1":
        return load_run_relocations(root, manifest, sha256)
    resolver = load_pilot_run_relocations(root, calibration_data)
    if manifest is not None:
        supplied = load_run_relocations(root, manifest, sha256)
        if resolver is None or supplied.root != resolver.root or supplied.binding != resolver.binding:
            raise ValueError("Explicit viewer relocations differ from the frozen formal input")
    return resolver


def check_viewer_coverage(labels_root, manifest):
    registry, _, _, coverage = load_viewer_intake(labels_root)
    camera = binding(Path(labels_root) / "calibration/calibration_snapshot.json")
    if manifest.get("calibration_digest") != read_json(camera["path"])["snapshot_digest"]:
        raise ValueError("Viewer camera differs from the frozen source calibration")
    declared = set(registry["sources"])
    empty = set(coverage["empty_sources"])
    active = declared - empty
    if (manifest.get("declared_source_count") != len(declared)
            or set(manifest.get("empty_sources", {})) != empty
            or set(manifest.get("sources", {})) != active
            or manifest.get("source_count") != len(active)):
        raise ValueError("Viewer source coverage differs from its frozen registry/intake/ranges")
    if not active or not all(item.get("baseline") is True for item in manifest["sources"].values()):
        raise ValueError("Every nonempty fresh source must expose actual stored body labels")
    return dict(
        declared_source_count=len(declared), active_sources=sorted(active), empty_sources=sorted(empty),
        minimum_clips=2 * len(active), evidence=coverage["evidence"] + [camera],
    )


def complete_clip_plan(manifest, focus_clips=()):
    """Keep every identity regression and cover two real rallies of every source."""
    clips = list(focus_clips)
    for sid, item in manifest["sources"].items():
        rallies = [r for r in item["rallies"] if r["end"] - r["begin"] >= 5.5]
        if len(rallies) < 2:
            raise ValueError("Each source needs two genuine continuous five-second native clips")
        if not focus_clips:
            selected = [rallies[0], rallies[-1]]
        else:
            selected = []
            existing = [c for c in clips if c["source_id"] == sid]
            while len(existing) + len(selected) < 2:
                used = {c["rally_index"] for c in existing} | {r["index"] for r in selected}
                remaining = [r for r in rallies if r["index"] not in used]
                if not remaining:
                    raise ValueError("Insufficient distinct native rallies for source playback coverage")
                times = [c["begin"] for c in existing] + [r["begin"] for r in selected]
                selected.append(max(remaining, key=lambda r: min(
                    [abs((r["begin"] + r["end"]) / 2 - t) for t in times] or [1],
                )))
        for rally in selected:
            clips.append(dict(
                source_id=sid, rally_index=rally["index"], begin=rally["begin"] + 0.2, duration_seconds=5,
            ))
    return clips


def validate_browser_source_acceptance(browser, canonical_binding):
    """Recheck the producer's actual frozen scope when sealing a new archive."""
    check_binding(canonical_binding)
    canonical = read_json(canonical_binding["path"])
    body = dict(path=canonical["source_dataset_manifest"], sha256=canonical["source_dataset_manifest_sha256"])
    check_binding(body)
    check_binding(browser["viewer_manifest"])
    viewer = read_json(browser["viewer_manifest"]["path"])
    coverage = check_viewer_coverage(Path(body["path"]).parent.parent, viewer)
    current = browser.get("frozen_source_coverage")
    formal = read_json(Path(body["path"]).parent.parent / "source_registry.json").get("schema") == (
        "badminton-formal-derived-source-registry-v1"
    )
    if formal and current is None:
        raise ValueError("Formal browser acceptance requires its independently frozen source coverage")
    if current is not None and current != coverage:
        raise ValueError("Browser source coverage changed after actual playback")
    clips = browser.get("results", [])
    if (browser.get("passed") is not True or browser.get("minimum_clip_seconds", 0) < 5
            or browser.get("page_errors") or browser.get("clips") != len(clips)
            or len(clips) < coverage["minimum_clips"]
            or browser.get("declared_source_count") != coverage["declared_source_count"]
            or browser.get("played_sources") != len(coverage["active_sources"])):
        raise ValueError("Real continuous native browser coverage is incomplete")
    if current is None and len(clips) != coverage["minimum_clips"]:
        raise ValueError("Legacy browser acceptance must retain its original clip count")
    counts = {sid: 0 for sid in coverage["active_sources"]}
    for clip in clips:
        sid = clip["source_id"]
        if (sid not in counts or clip.get("passed") is not True
                or clip.get("actual_end", 0) - clip.get("actual_begin", 0) < 4.9
                or clip.get("unique_committed_scans", 0) < 20):
            raise ValueError("Browser clip lacks real native source/time/scan evidence")
        counts[sid] += 1
    if any(count < (2 if current is not None else 1) for count in counts.values()):
        raise ValueError("Browser playback did not cover every nonempty source")
    return coverage
