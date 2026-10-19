from copy import deepcopy
import json

import pytest

from hmr4d.lidar_mocap.canonical_tracks import checked_review, remap_row


def row(scan=10, motion="scene:window:13:0"):
    return dict(
        motion_id=motion,
        track_id="13",
        lidar_index=scan,
        fit_path="native-fit.npz",
        source_segment_indices=[4, 8],
        qa=dict(surface=0.0178),
        cameras=dict(cam1=dict(record_index=100), cam2=dict(record_index=102)),
    )


def test_confirmed_boundary_changes_only_canonical_fields_and_preserves_source():
    source = row()
    previous = deepcopy(source)
    changed = remap_row(source, {source["motion_id"]: [dict(scan=10, track_id=1000001, case=3)]}, "bound-review")
    assert source == previous
    assert changed["motion_id"] == "scene:window:1000001:0"
    assert changed["canonical_identity"]["source_motion_id"] == source["motion_id"]
    assert {k: v for k, v in changed.items() if k not in ("motion_id", "track_id", "canonical_identity")} == {
        k: v for k, v in source.items() if k not in ("motion_id", "track_id")
    }


def test_cut_does_not_change_earlier_frame_or_same_numeric_id_in_other_source():
    cuts = {"scene:window:13:0": [dict(scan=10, track_id=1000001, case=3)]}
    for source in (row(9), row(10, "other:window:13:0"), row(10, "scene:other:13:0")):
        assert remap_row(source, cuts, "review") == source


def test_separate_unknown_continuations_are_never_forcibly_rejoined():
    cuts = {"scene:window:13:0": [dict(scan=10, track_id=1000001, case=1), dict(scan=11, track_id=1000002, case=2)]}
    identities = [remap_row(row(scan), cuts, "review")["motion_id"] for scan in (9, 10, 11)]
    assert len(set(identities)) == 3


def test_pending_review_cannot_change_canonical_identity(tmp_path):
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps(dict(schema="badminton-reviewed-canonical-track-boundaries-v1", review_status="pending")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="explicit reviewed"):
        checked_review(path)
