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
    validate_no_fit_source,
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
    value["clips"][0]["rally_index"] = 1
    return value


def test_full_native_scope_allows_long_run_without_relaxing_short_pilot(tmp_path):
    value = full_spec(tmp_path)
    assert validate_full_native_spec(value) == dict(sources=["actual"], selected_seconds=250.0, selected_scans=3)
    with pytest.raises(ValueError):
        validate_pilot_spec(value)
    value["schema"] = "badminton-review-pilot-v1"
    with pytest.raises(ValueError, match="180"):
        validate_pilot_spec(value)


@pytest.mark.parametrize("name", ["source_indexes", "excluded_source_indexes"])
@pytest.mark.parametrize("entry", [dict(path="actual.json", sha256="b" * 64), dict(file="", sha256="b" * 64)])
def test_full_native_source_indexes_use_the_existing_bound_file_contract(tmp_path, name, entry):
    value = full_spec(tmp_path)
    value[name] = dict(actual=entry)
    previous = deepcopy(value)
    with pytest.raises(ValueError, match="requires a file/sha256 binding relative to data_root"):
        validate_full_native_spec(value)
    assert value == previous
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


def test_disconnected_valid_native_segments_keep_one_original_rally_without_duplicate_scans(tmp_path):
    value = full_spec(tmp_path)
    original, _, _ = native_rally_fixture(tmp_path)
    value["clips"] = [
        dict(original, segment_index=0, begin_common_us=1650000, end_common_us=1850000,
             eligible_paired_png_indices=[7, 8]),
        dict(original, segment_index=1, begin_common_us=1950000, end_common_us=2050000,
             eligible_paired_png_indices=[10]),
    ]
    assert validate_full_native_spec(value)["selected_scans"] == 3
    for clip in value["clips"]:
        assert clip["rally_index"] == 2
        validate_native_rally(clip, "new-crop")
    pairs = [dict(lidar_index=i, scan_mid_us=1000000 + i * 100000, status="paired") for i in range(20)]
    windows = review_rally_windows(pairs, value["clips"])
    assert [[p["lidar_index"] for p in w[3]] for w in windows] == [[7, 8], [10]]
    value["clips"][1]["segment_index"] = 0
    with pytest.raises(ValueError, match="segment is declared twice"):
        validate_full_native_spec(value)
    value["clips"][1]["segment_index"] = 1
    value["clips"][0]["contiguous_windows"] = []
    with pytest.raises(ValueError, match="flat native"):
        validate_full_native_spec(value)


def test_real_no_fit_source_closes_native_coverage_without_fabricating_any_fit_window(tmp_path):
    clip, _, receipt = native_rally_fixture(tmp_path)
    index = tmp_path / "no_fit_index.json"
    index.write_text(json.dumps(dict(source_id="empty_actual", source_kind="derived_native_roi_fixed24",
                                    source_manifest=dict(sha256="new-crop"))), encoding="utf-8")
    coverage = tmp_path / "empty_coverage.json"
    coverage.write_text('{"assigned_unique_scans":0}', encoding="utf-8")
    table_path = tmp_path / "empty_clips.json"
    analysis_entry = clip["evidence_files"][0]
    table = dict(schema="badminton-analysis-clips-v1", source_id="empty_actual", clips=[],
                 analysis_sha256=analysis_entry["sha256"], planner_sha256="planner")
    table_path.write_text(json.dumps(table), encoding="utf-8")
    receipt_path = tmp_path / "empty_native_receipt.json"
    receipt.update(source_id="empty_actual")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    marker = tmp_path / "empty_actual.no-fit-needed.json"
    proof = dict(schema="badminton-native-no-fit-needed-v1", source_id="empty_actual", status="no_fit_needed",
        eligible_paired_scan_count=0, reason="Actual native analysis found no eligible rallies",
        source_index=binding(index), source_index_sha256=binding(index)["sha256"], coverage_audit=binding(coverage),
        evidence_files=[analysis_entry, binding(table_path), binding(receipt_path)])
    marker.write_text(json.dumps(proof), encoding="utf-8")
    assert validate_no_fit_source(binding(marker), source_id="empty_actual")["eligible_paired_scan_count"] == 0
    value = full_spec(tmp_path)
    value.update(excluded_source_indexes=dict(empty_actual=dict(file="no_fit_index.json", sha256=binding(index)["sha256"])),
                 no_fit_sources=dict(empty_actual=binding(marker)))
    result = validate_full_native_spec(value)
    assert result["sources"] == ["actual"] and result["excluded_no_fit_sources"] == ["empty_actual"]
    assert result["selected_scans"] == 3 and len(value["clips"]) == 1
    value["clips"] = []
    with pytest.raises(ValueError, match="actual native-confirmed"):
        validate_full_native_spec(value)
    receipt["exit_code"] = 1
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    proof["evidence_files"][-1] = binding(receipt_path)
    marker.write_text(json.dumps(proof), encoding="utf-8")
    with pytest.raises(ValueError, match="completed actual native"):
        validate_no_fit_source(binding(marker), source_id="empty_actual")


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
