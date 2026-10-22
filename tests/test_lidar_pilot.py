"""Bounded pilot scopes never manufacture release partitions or admission."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from hmr4d.lidar_mocap.calibration import _CONFIG_FILES
from hmr4d.lidar_mocap.dual_view import export_reviewed
from hmr4d.lidar_mocap.dual_view_observer import source_selection
from hmr4d.lidar_mocap.pilot import (
    require_review_group,
    validate_pilot_spec,
    validate_full_native_spec,
    review_rally_windows,
    validate_native_rally,
    binding,
)
from hmr4d.lidar_mocap.pipeline import BadmintonPipeline
from hmr4d.lidar_mocap.rally_pipeline import validate_session_splits


def spec():
    return dict(
        schema="badminton-review-pilot-v1",
        review_only=True,
        training_published=False,
        camera_label="R11",
        bundle_sha256={name: "a" * 64 for name in _CONFIG_FILES},
        source_indexes={"actual": {"file": "actual.json", "sha256": "b" * 64}},
        review_authorization=dict(path="camera-review.json", sha256="d" * 64),
        clips=[
            dict(
                source_id="actual",
                begin_common_us=1000000,
                end_common_us=21000000,
                selection_status="native_rally_confirmed",
                eligible_paired_png_indices=[7, 8, 10],
                evidence_files=[dict(path="real-native.json", sha256="c" * 64)],
            )
        ],
    )


def test_one_genuine_short_rally_is_allowed_before_second_source_or_120_seconds():
    actual = spec()
    previous = deepcopy(actual)
    assert validate_pilot_spec(actual) == dict(sources=["actual"], selected_seconds=20.0, selected_scans=3)
    assert actual == previous


def full_spec(tmp_path):
    value = spec()
    audit = tmp_path / "coverage.json"
    audit.write_text('{"actual_unique_coverage":true}', encoding="utf-8")
    value.update(
        schema="badminton-native-rally-fit-v1",
        execution_scope="all_effective_native_rallies",
        user_confirmation="Fit all valid 1019 native rallies; keep training publication separate.",
        coverage_audit=binding(audit),
    )
    value["clips"][0]["end_common_us"] = 251000000
    return value


def test_full_native_scope_allows_long_run_without_relaxing_short_pilot(tmp_path):
    value = full_spec(tmp_path)
    assert validate_full_native_spec(value) == dict(sources=["actual"], selected_seconds=250.0, selected_scans=3)
    with pytest.raises(ValueError):
        validate_pilot_spec(value)
    value["schema"] = "badminton-review-pilot-v1"
    with pytest.raises(ValueError, match="180"):
        validate_pilot_spec(value)
    with pytest.raises(ValueError):
        validate_full_native_spec(value)


@pytest.mark.parametrize("change", ["scope", "confirmation", "audit", "source_omission", "published", "overlap"])
def test_full_native_scope_requires_bound_actual_coverage_and_review_status(tmp_path, change):
    value = full_spec(tmp_path)
    if change == "scope":
        value["execution_scope"] = "unchecked_all_frames"
    if change == "confirmation":
        value["user_confirmation"] = " "
    if change == "audit":
        Path(value["coverage_audit"]["path"]).write_text("changed", encoding="utf-8")
    if change == "source_omission":
        value["source_indexes"]["omitted"] = dict(file="omitted.json", sha256="f" * 64)
    if change == "published":
        value["training_published"] = True
    if change == "overlap":
        value["clips"].append(dict(value["clips"][0], eligible_paired_png_indices=[20, 21]))
    with pytest.raises(ValueError):
        validate_full_native_spec(value)


def test_pilot_teacher_windows_follow_real_rallies_and_do_not_add_idle_minute_frames():
    pairs = [dict(lidar_index=i, scan_mid_us=1000000 + i * 100000, status="paired") for i in range(100)]
    rows = [
        dict(begin_common_us=2000000, end_common_us=3000000, eligible_paired_png_indices=[10, 11, 19]),
        dict(begin_common_us=8000000, end_common_us=9000000, eligible_paired_png_indices=[70, 79]),
    ]
    windows = review_rally_windows(pairs, rows)
    assert len(windows) == 2 and sum(len(w[3]) for w in windows) == 5
    assert [(w[1], w[2]) for w in windows] == [(2000000, 3000000), (8000000, 9000000)]


def native_rally_fixture(root):
    analysis = root / "analysis.json"
    analysis.write_text('{"native_result":true}')
    result = binding(analysis)
    table = dict(
        schema="badminton-analysis-clips-v1",
        source_id="actual",
        analysis_sha256=result["sha256"],
        planner_sha256="planner",
        clips=[dict(index=2, kind="badminton_rally", source_begin_frame=7, source_frame_count=4)],
    )
    receipt = dict(
        source_id="actual",
        status="completed",
        exit_code=0,
        source_manifest_sha256="new-crop",
        analysis_sha256=result["sha256"],
        planner_sha256="planner",
    )
    entries = [result]
    for name, value in [("clips.json", table), ("receipt.json", receipt)]:
        p = root / name
        p.write_text(json.dumps(value))
        entries.append(binding(p))
    return dict(spec()["clips"][0], rally_index=2, evidence_files=entries), table, receipt


def test_native_rally_evidence_is_bound_to_the_actual_new_crop_and_analysis(tmp_path):
    clip, _, _ = native_rally_fixture(tmp_path)
    validate_native_rally(clip, "new-crop")


@pytest.mark.parametrize(
    "change", ["old_crop", "failed_native", "outside_rally", "wrong_planner", "not_rally", "missing_analysis"]
)
def test_presence_or_old_crop_native_results_do_not_authorize_a_pilot_rally(tmp_path, change):
    clip, table, receipt = native_rally_fixture(tmp_path)
    if change == "old_crop":
        receipt["source_manifest_sha256"] = "old-crop"
    if change == "failed_native":
        receipt["exit_code"] = 1
    if change == "outside_rally":
        clip["eligible_paired_png_indices"] = [7, 11]
    if change == "wrong_planner":
        receipt["planner_sha256"] = "unrelated"
    if change == "not_rally":
        table["clips"][0]["kind"] = "people_present"
    if change == "missing_analysis":
        clip["evidence_files"] = clip["evidence_files"][1:]
    for name, value in [("clips.json", table), ("receipt.json", receipt)]:
        p = tmp_path / name
        p.write_text(json.dumps(value))
        clip["evidence_files"] = [binding(p) if Path(item["path"]) == p else item for item in clip["evidence_files"]]
    with pytest.raises(ValueError):
        validate_native_rally(clip, "new-crop")


@pytest.mark.parametrize(
    "change",
    [
        "published",
        "not_review",
        "no_camera",
        "old_bundle",
        "presence_only",
        "no_evidence",
        "no_source",
        "float_time",
        "reversed_time",
        "too_long",
        "duplicate_index",
        "bool_index",
        "no_indices",
        "duplicate_clip",
    ],
)
def test_pilot_cannot_expand_scope_or_fabricate_rally_coverage(change):
    value = spec()
    clip = value["clips"][0]
    if change == "published":
        value["training_published"] = True
    if change == "not_review":
        value["review_only"] = False
    if change == "no_camera":
        value.pop("camera_label")
    if change == "old_bundle":
        value["bundle_sha256"].pop("bbox.json")
    if change == "presence_only":
        clip["selection_status"] = "people_present"
    if change == "no_evidence":
        clip["evidence_files"] = []
    if change == "no_source":
        clip["source_id"] = "unbound"
    if change == "float_time":
        clip["begin_common_us"] = 1e6
    if change == "reversed_time":
        clip["end_common_us"] = 1000000
    if change == "too_long":
        clip["end_common_us"] = 182000000
    if change == "duplicate_index":
        clip["eligible_paired_png_indices"] = [7, 7]
    if change == "bool_index":
        clip["eligible_paired_png_indices"] = [True, 8]
    if change == "no_indices":
        clip["eligible_paired_png_indices"] = []
    if change == "duplicate_clip":
        value["clips"].append(deepcopy(clip))
    with pytest.raises(ValueError):
        validate_pilot_spec(value)


def test_review_source_group_does_not_relax_formal_release_partitions():
    assert require_review_group({"review": ["actual"]}, {"actual": {}}) == {"review": ["actual"]}
    with pytest.raises(ValueError):
        validate_session_splits({"train": ["actual"], "val": [], "test": []}, {"actual": {}})
    for wrong in ({"review": ["actual", "actual"]}, {"train": ["actual"]}, {"review": ["missing"]}):
        with pytest.raises(ValueError):
            require_review_group(wrong, {"actual": {}})


def test_review_aggregate_never_creates_a_fake_train_partition(tmp_path):
    pipeline = BadmintonPipeline.__new__(BadmintonPipeline)
    pipeline.output_root = tmp_path
    pipeline.calibration = Mock(data=dict(sessions={"actual": {}}, snapshot_digest="fixed"))
    with patch.object(pipeline, "_load"):
        result = pipeline.aggregate(["actual"], review_only=True)
    assert set(result["suggested_session_splits"]) == {"review"}
    assert result["review_only"] is True and result["training_published"] is False
    assert result["status"] == "review_only" and result["accepted"] == 0


@pytest.mark.parametrize("dataset_flag", [True, False])
def test_formal_observer_and_publisher_reject_even_a_stripped_pilot_flag(tmp_path, dataset_flag):
    source = tmp_path / "source"
    (source / "datasets").mkdir(parents=True)
    (source / "calibration").mkdir()
    (source / "datasets/dataset_manifest.json").write_text(json.dumps(dict(review_only=dataset_flag)))
    (source / "calibration/calibration_snapshot.json").write_text('{"review_only":true}')
    with pytest.raises(ValueError, match="Review-only pilot"):
        source_selection(source)
    output = tmp_path / "must-not-be-published"
    with pytest.raises(ValueError, match="Review-only pilot"):
        export_reviewed(source, output)
    assert not output.exists()
