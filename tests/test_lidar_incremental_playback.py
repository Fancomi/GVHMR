"""Checked native playback planning metadata; never opens a browser or a model."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.canonical_tracks import checked_review
from test_lidar_canonical_no_change import make_case, pair, publish
from test_lidar_incremental_release import bind, write

sys.path.insert(0, str(PROJ_ROOT / "tools"))
from badminton_viewer_contracts import complete_clip_plan


def load_playback():
    path = PROJ_ROOT / "tools/verify_badminton_fresh_playback.py"
    spec = importlib.util.spec_from_file_location("_incremental_playback_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


playback = load_playback()


def context(case, **kwargs):
    return playback._canonical_context(case["repair"], case["output"] / "datasets/dataset_manifest.json", **kwargs)


def test_typed_no_change_planning_rechecks_actual_full_subset_without_cut_fabrication(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch)
    manifest = publish(case)
    audit_bytes = case["audit_path"].read_bytes()
    selected, actual, policy, proof = context(case, **pair(case))
    assert selected == bind(case["output"] / "datasets/dataset_manifest.json")
    assert actual == manifest
    assert policy.binding == case["selected"]
    assert (proof["full_accepted"], proof["accepted"]) == (4, 3)
    assert proof["identity_changes"] == 0
    assert case["audit_path"].read_bytes() == audit_bytes
    assert json.loads(audit_bytes)["body_manifest"]["path"] == manifest["source_dataset_manifest"]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("kwargs", [{}, dict(release_policy="missing"), dict(release_policy_sha256="0" * 64)])
def test_v2_no_default_or_half_policy_pair_before_any_browser(tmp_path, monkeypatch, kwargs):
    case = make_case(tmp_path, monkeypatch)
    publish(case)
    called = []
    monkeypatch.setattr(playback, "sync_playwright", lambda: called.append(True))
    with pytest.raises(ValueError, match="policy"):
        playback.verify(
            case["repair"],
            "http://127.0.0.1:1/",
            case["output"] / "datasets/dataset_manifest.json",
            cdp_url="http://127.0.0.1:1",
            **kwargs,
        )
    assert called == []
    assert not (case["repair"] / "browser_acceptance").exists()


def test_v2_requires_canonical_publication_even_with_valid_explicit_policy(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="canonical publication"):
        playback._canonical_context(case["repair"], None, **pair(case))


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(schema="unrecognized"),
        lambda m: m.update(schema=True),
        lambda m: m["canonical_identity_revision"].update(mode=True),
        lambda m: m["canonical_identity_revision"].update(mode="reviewed_cuts"),
        lambda m: m["canonical_identity_revision"].update(identity_keys_unchanged=1),
        lambda m: m["canonical_identity_revision"].update(verification=dict(path="missing", sha256="0" * 64)),
    ],
)
def test_rebound_unknown_or_flag_only_no_change_does_not_plan_browser(tmp_path, monkeypatch, change):
    case = make_case(tmp_path, monkeypatch)
    manifest = publish(case)
    change(manifest)
    write(case["output"] / "datasets/dataset_manifest.json", manifest)
    with pytest.raises(ValueError):
        context(case, **pair(case))


def test_mutated_real_source_after_publication_is_rejected_before_browser(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch)
    manifest = publish(case)
    metadata = Path(manifest["exports"]["fresh"]["metadata"])
    metadata.write_bytes(metadata.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="dependency changed"):
        context(case, **pair(case))


def test_no_change_retains_two_distinct_five_second_native_rallies_per_source():
    viewer = dict(
        sources=dict(
            source1=dict(rallies=[dict(index=i, begin=i * 20, end=i * 20 + 8) for i in range(4)]),
            source2=dict(rallies=[dict(index=i, begin=i * 30, end=i * 30 + 8) for i in range(3)]),
        )
    )
    clips = complete_clip_plan(viewer)
    assert len(clips) == 4
    for sid in viewer["sources"]:
        chosen = [clip for clip in clips if clip["source_id"] == sid]
        assert len({clip["rally_index"] for clip in chosen}) == 2
        assert all(clip["duration_seconds"] == 5 for clip in chosen)
    short = deepcopy(viewer)
    short["sources"]["source2"]["rallies"] = [dict(index=0, begin=0, end=5.4)]
    with pytest.raises(ValueError, match="genuine"):
        complete_clip_plan(short)


def test_legacy_cut_context_does_not_require_or_invent_a_policy(tmp_path):
    manifest = dict(
        schema="badminton-canonical-bilateral-publication-v1",
        canonical_identity_revision=dict(reviewed_boundaries=dict(path="not-opened-before-cut-review", sha256="x")),
    )
    selected = write(tmp_path / "legacy.json", manifest)
    actual_binding, actual, policy, proof = playback._canonical_context(tmp_path, selected["path"])
    assert actual_binding == selected and actual == manifest and policy is proof is None
    review = write(
        tmp_path / "no-fake-cut.json",
        dict(schema="badminton-reviewed-canonical-track-boundaries-v1", review_status="pending"),
    )
    with pytest.raises(ValueError, match="explicit reviewed"):
        checked_review(review["path"])


def test_legacy_uncanonical_planning_remains_available(tmp_path):
    assert playback._canonical_context(tmp_path, None) == (None, None, None, None)


@pytest.mark.parametrize(
    "tool,args",
    [
        ("verify_badminton_fresh_playback.py", ["--root", str(PROJ_ROOT / "outputs/not-used")]),
        (
            "seal_badminton_expansion.py",
            [
                "--assembly",
                "missing",
                "--browser",
                "missing",
                "--continuity",
                "missing",
                "--pixels",
                "missing",
                "--sequence-report",
                "missing",
                "--release-id",
                "not-used",
            ],
        ),
    ],
)
def test_cli_rejects_half_pair_before_assets_or_browser(tool, args):
    result = subprocess.run(
        [sys.executable, "-B", str(PROJ_ROOT / "tools" / tool), *args, "--release-policy", "missing"],
        cwd=str(PROJ_ROOT.parent),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "must be supplied together" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("tool", ["verify_badminton_fresh_playback.py", "seal_badminton_expansion.py"])
def test_cli_help_outside_repository_preserves_literal_policy_pair(tool):
    result = subprocess.run(
        [sys.executable, "-B", str(PROJ_ROOT / "tools" / tool), "--help"],
        cwd=str(PROJ_ROOT.parent),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--release-policy" in result.stdout and "--release-policy-sha256" in result.stdout
