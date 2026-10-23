"""Actual small CPU proofs at the browser/archive seam; no production release."""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap import incremental_release as inc
from hmr4d.lidar_mocap.export import FIELDS
from test_lidar_canonical_no_change import browser, make_case, pair, publish
from test_lidar_incremental_release import bind, make_policy, write


def load_seal():
    path = PROJ_ROOT / "tools/seal_badminton_expansion.py"
    spec = importlib.util.spec_from_file_location("_incremental_seal_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seal_tool = load_seal()


def gate_fixture(tmp_path, monkeypatch):
    """Real policy/subset/continuity checks with explicitly synthetic browser metadata."""
    case = make_case(tmp_path, monkeypatch)
    labels = case["labels"]
    ranges = labels / "ranges.jsonl"
    ranges.write_text(json.dumps(dict(source_id="fresh")) + "\n", encoding="utf-8")
    intake = write(
        labels / "intake.json",
        dict(
            prepared_ranges=bind(ranges),
            sources=dict(fresh=dict(paired_unique_png_frames=4)),
            effective_window_minutes=1,
        ),
    )
    registry = json.loads((labels / "source_registry.json").read_text())
    registry["intake"] = intake
    write(labels / "source_registry.json", registry)
    canonical = publish(case)
    canonical_binding = bind(case["output"] / "datasets/dataset_manifest.json")
    policy = inc.load_release_policy(**dict(path=case["selected"]["path"], sha256=case["selected"]["sha256"]))
    viewer = dict(
        calibration_digest="e" * 64,
        declared_source_count=1,
        source_count=1,
        empty_sources={},
        canonical_publication=canonical_binding,
        sources=dict(
            fresh=dict(
                baseline=True,
                rallies=[dict(index=index, begin=index * 20, end=index * 20 + 8) for index in range(2)],
            )
        ),
    )
    evidence = browser(case, canonical)
    evidence["schema"] = "badminton-fresh-browser-acceptance-v1"
    evidence["viewer_manifest"] = write(labels / "viewer/manifest.json", viewer)
    evidence["frozen_source_coverage"] = seal_tool._viewer_contracts.check_viewer_coverage(labels, viewer)
    evidence["release_policy"] = policy.binding
    for clip in evidence["results"]:
        begin = clip["rally_index"] * 20 + 0.2
        clip.update(begin=begin, actual_begin=begin, actual_end=begin + 5, duration_seconds=5, labelled_scans=30)
        clip["samples"] = [
            dict(
                scan=index,
                sceneScan=index,
                meshCount=1,
                vertexCount=6890,
                webglError=0,
                projections=[dict(scan=index, error=0), dict(scan=index, error=0)],
                athleteIds=["synthetic-only"],
            )
            for index in range(30)
        ]
        views = [dict(camera=camera, visible=True, different_rgba_components=0) for camera in ("cam1", "cam2")]
        clip.update(initial_visible_rgb=views, final_visible_rgb=deepcopy(views))
    plan = dict(
        schema="badminton-fresh-browser-clips-v1",
        source_completion=evidence["source_completion"],
        canonical_publication=canonical_binding,
        canonical_mode="reviewed_no_changes",
        canonical_no_change_proof=evidence["canonical_no_change_proof"],
        release_policy=policy.binding,
        frozen_source_coverage=evidence["frozen_source_coverage"],
        clips=[
            {key: clip[key] for key in ("source_id", "rally_index", "begin", "duration_seconds")}
            for clip in evidence["results"]
        ],
    )
    evidence["clip_plan"] = write(case["repair"] / "synthetic_clip_plan.json", plan)
    browser_path = case["repair"] / "synthetic_browser.json"
    write(browser_path, evidence)
    parent = policy.parent["bindings"]
    source = policy.scope["review_sources"]["fresh"]
    contract = dict(
        schema=inc.ASSEMBLY,
        status="complete",
        sealed=True,
        release_policy=policy.binding,
        canonical_data_root=policy.data["canonical_data_root"],
        new_repair_root=str(case["repair"]),
        old_accepted=policy.parent["total"],
        old_split_counts=policy.parent["counts"],
        old_source_splits=policy.parent["roles"],
        new_source_splits=policy.roles,
        required_new_nonempty_splits=policy.required,
        new_sources=dict(
            fresh=dict(
                final_admitted_count=3,
                repair_root=str(case["repair"]),
                admitted_source_index=source["admitted_source_index"],
                camera_applicability=source["camera_applicability"],
                calibration_snapshot=bind(labels / "calibration/calibration_snapshot.json"),
                calibration_digest="e" * 64,
                formal_body_input=bind(case["repair"] / "new_source_input.json"),
            )
        ),
        new_final_admitted_counts=dict(train=3, val=0, test=0, total=3),
        canonical_mode="reviewed_no_changes",
        paired_schema=dict(
            dtype="float32",
            target_body_model="neutral_smpl_24_joints",
            coordinate_space="lidar_viewer_xyz_m",
            transl_convention="shaped_pelvis_position",
            **{name: ["N", *shape] for name, shape in FIELDS.items()},
        ),
        bindings=dict(
            old_archive=parent["archive_manifest"],
            old_dataset_manifest=parent["dataset_manifest"],
            old_training_manifest=parent["training_manifest"],
            parent_final_consumer_contract=parent["final_consumer_contract"],
            parent_final_consumer_audit=parent["final_consumer_audit"],
            new_source_input=bind(case["repair"] / "new_source_input.json"),
            new_source_registry=bind(case["root"] / "source/source_registry.json"),
            new_source_splits=policy.binding,
            producer_confirmation=bind(case["completion_path"]),
            producer_joint_manifest=canonical_binding,
        ),
    )
    contract_path = case["repair"] / "synthetic_assembly_input.json"
    contract_binding = write(contract_path, contract)
    splits = deepcopy(policy.parent["dataset"]["suggested_session_splits"])
    splits["train"].update(accepted=5, sessions=policy.parent["roles"]["train"] + ["fresh"])
    assembly = dict(
        schema="badminton-expanded-assembly-v1",
        training_published=False,
        accepted=9,
        old_accepted=6,
        new_accepted=3,
        all_old_rows_retained=True,
        all_tensors_equal_ordered_source_concatenation=True,
        all_metadata_bytes_equal_ordered_source_concatenation=True,
        release_policy=policy.binding,
        consumer_contract=contract_binding,
        source_publications=dict(old=parent["dataset_manifest"], new=canonical_binding),
        suggested_session_splits=splits,
    )
    assembly_path = case["repair"] / "synthetic_assembly.json"
    assembly_binding = write(assembly_path, assembly)
    initial = write(
        case["repair"] / "synthetic_pixel_audit.json",
        dict(checked_camera_frames=0, expected_camera_frames=0, unknown_differences=[], corrections=[]),
    )
    pixels_path = case["repair"] / "synthetic_pixels.json"
    write(
        pixels_path,
        dict(
            status="complete",
            all_person_admissions_unchanged=True,
            admission_changes=[],
            assembly=assembly_binding,
            initial_pixel_audit=initial,
            checked_camera_frames=0,
            metadata=bind(Path(splits["test"]["metadata"])),
            corrections=[],
            source_videos={},
        ),
    )
    return dict(
        case,
        policy=policy,
        canonical=canonical,
        assembly=assembly,
        assembly_path=assembly_path,
        contract=contract,
        contract_path=contract_path,
        browser=evidence,
        browser_path=browser_path,
        pixels_path=pixels_path,
    )


def check(case, **kwargs):
    return seal_tool.validate_acceptance(
        case["assembly_path"], case["browser_path"], case["audit_path"], case["pixels_path"], **kwargs
    )


def test_real_parent_bridge_and_policy_archive_independent_exact_bytes_without_legacy_receipt(tmp_path):
    selected = make_policy(tmp_path)
    policy = inc.load_release_policy(selected["path"], selected["sha256"])
    archive = seal_tool.Archive(tmp_path / "copy", "synthetic")
    seal_tool._archive_release_provenance(archive, dict(bindings={}), policy)
    expected = dict(
        release_policy=policy.binding,
        model_bridge=policy.parent["model_bridge"],
        reviewed_scope=policy.data["reviewed_scope"],
    )
    for name, original in expected.items():
        copied = archive.stage / ("provenance/" + name + ".json")
        assert copied.read_bytes() == Path(original["path"]).read_bytes()
        assert bind(copied)["sha256"] == original["sha256"]
        assert not os.path.samefile(copied, original["path"])
    for name, original in policy.parent["bindings"].items():
        copied = archive.stage / ("provenance/parent_release/" + name + ".json")
        assert copied.read_bytes() == Path(original["path"]).read_bytes()
        assert not os.path.samefile(copied, original["path"])
    assert len(archive.artifacts) == 8


def test_legacy_parent_receipt_bridge_keeps_original_layout(tmp_path):
    bridge = write(tmp_path / "bridge.json", dict(legacy=True))
    receipt = write(tmp_path / "receipt.json", dict(model_bridge=bridge))
    archive = seal_tool.Archive(tmp_path / "copy", "legacy")
    seal_tool._archive_release_provenance(archive, dict(bindings=dict(parent_receipt=receipt)), None)
    assert (archive.stage / "provenance/model_bridge.json").read_bytes() == Path(bridge["path"]).read_bytes()
    assert [item["path"] for item in archive.artifacts] == ["provenance/model_bridge.json"]


@pytest.mark.parametrize("kwargs", [{}, dict(release_policy="missing"), dict(release_policy_sha256="0" * 64)])
def test_v2_pair_is_explicit_and_complete_before_archive(tmp_path, monkeypatch, kwargs):
    case = gate_fixture(tmp_path, monkeypatch)
    entered = []
    monkeypatch.setattr(seal_tool, "Archive", lambda *a: entered.append(a))
    args = SimpleNamespace(
        assembly=case["assembly_path"],
        browser=case["browser_path"],
        continuity=case["audit_path"],
        pixels=case["pixels_path"],
        **kwargs,
    )
    with pytest.raises(ValueError, match="policy"):
        seal_tool.seal(args)
    assert entered == []


def test_typed_no_change_gate_preserves_full_body_audit_and_three_return_values(tmp_path, monkeypatch):
    case = gate_fixture(tmp_path, monkeypatch)
    original = case["audit_path"].read_bytes()
    frozen, actual, _ = check(case, **pair(case))
    assert actual["accepted"] == 9
    assert set(frozen) == {"assembly", "browser", "continuity", "pixels"}
    assert case["audit_path"].read_bytes() == original
    assert json.loads(original)["body_manifest"] != actual["source_publications"]["new"]
    assert case["browser"]["native_boundary_regressions"] == []
    proof = json.loads(Path(case["canonical"]["canonical_identity_revision"]["verification"]["path"]).read_text())
    assert (proof["full_accepted"], proof["accepted"]) == (4, 3)


@pytest.mark.parametrize(
    "change",
    [
        lambda a: a.update(accepted=True),
        lambda a: a.update(new_accepted=True),
        lambda a: a.update(old_accepted=True),
        lambda a: a.update(release_policy=dict(path="wrong", sha256="0" * 64)),
        lambda a: a["suggested_session_splits"]["val"].update(accepted=3),
        lambda a: a["suggested_session_splits"]["train"]["sessions"].reverse(),
        lambda a: a.update(all_old_rows_retained=False),
    ],
)
def test_v2_rebound_wrong_population_policy_role_or_retention_rejected(tmp_path, monkeypatch, change):
    case = gate_fixture(tmp_path, monkeypatch)
    change(case["assembly"])
    write(case["assembly_path"], case["assembly"])
    with pytest.raises(ValueError):
        check(case, **pair(case))


@pytest.mark.parametrize(
    "change",
    [
        lambda b: b.update(native_boundary_regressions=[dict(passed=True)]),
        lambda b: b.pop("canonical_no_change_proof"),
        lambda b: b["results"][1].update(rally_index=0),
        lambda b: b["results"][0].update(actual_end=1),
        lambda b: b["results"][0].update(unique_committed_scans=1),
        lambda b: b.update(schema="unrecognized"),
        lambda b: b.update(release_policy=dict(path="wrong", sha256="0" * 64)),
        lambda b: b["results"][0].update(actual_begin=float("inf")),
        lambda b: b["results"][0].pop("samples"),
        lambda b: b["results"][0]["samples"][0].update(vertexCount=1),
        lambda b: b["results"][0]["samples"][0].update(sceneScan=True),
        lambda b: b["results"][0]["samples"][0].update(projections=[]),
        lambda b: b["results"][0].update(labelled_scans=1),
        lambda b: b["results"][0]["initial_visible_rgb"][0].update(visible=False),
        lambda b: b["results"][0]["final_visible_rgb"][1].update(camera="cam1"),
        lambda b: b["results"][0]["final_visible_rgb"][0].update(different_rgba_components=False),
    ],
)
def test_no_change_does_not_relax_actual_native_browser_requirements(tmp_path, monkeypatch, change):
    case = gate_fixture(tmp_path, monkeypatch)
    change(case["browser"])
    write(case["browser_path"], case["browser"])
    with pytest.raises(ValueError):
        check(case, **pair(case))


def test_rebound_plan_cannot_choose_unknown_rally_or_leave_original_native_interval(tmp_path, monkeypatch):
    case = gate_fixture(tmp_path, monkeypatch)
    path = Path(case["browser"]["clip_plan"]["path"])
    plan = json.loads(path.read_text())
    plan["clips"][0]["begin"] = 1000
    case["browser"]["results"][0]["begin"] = 1000
    case["browser"]["clip_plan"] = write(path, plan)
    write(case["browser_path"], case["browser"])
    with pytest.raises(ValueError, match="outside its actual frozen native rally"):
        check(case, **pair(case))


def test_no_change_does_not_relax_pixel_admission_gate(tmp_path, monkeypatch):
    case = gate_fixture(tmp_path, monkeypatch)
    pixels = json.loads(case["pixels_path"].read_text())
    pixels["all_person_admissions_unchanged"] = False
    write(case["pixels_path"], pixels)
    with pytest.raises(ValueError, match="admission"):
        check(case, **pair(case))


def test_archive_no_change_provenance_keeps_true_full_audit_instead_of_fake_boundaries(tmp_path, monkeypatch):
    case = gate_fixture(tmp_path, monkeypatch)
    check(case, **pair(case))
    archive = seal_tool.Archive(tmp_path / "copy", "nochange")
    continuity = json.loads(case["audit_path"].read_text())
    seal_tool._archive_canonical_evidence(archive, case["canonical"], continuity, case["policy"], case["contract"])
    assert (archive.stage / "provenance/full_continuity_audit.json").read_bytes() == case["audit_path"].read_bytes()
    assert (archive.stage / "provenance/reviewed_continuity.json").read_bytes() == case["review_path"].read_bytes()
    assert not (archive.stage / "provenance/reviewed_track_boundaries.json").exists()


def test_late_parent_bridge_change_is_rejected_before_copy(tmp_path):
    selected = make_policy(tmp_path)
    policy = inc.load_release_policy(selected["path"], selected["sha256"])
    bridge = Path(policy.parent["model_bridge"]["path"])
    bridge.write_bytes(bridge.read_bytes() + b" ")
    archive = seal_tool.Archive(tmp_path / "copy", "changed")
    with pytest.raises(ValueError, match="dependency changed"):
        seal_tool._archive_release_provenance(archive, {}, policy)
    assert archive.artifacts == []
