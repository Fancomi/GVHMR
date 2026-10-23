"""Apply validates real typed review sessions before touching models or labels."""

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.pilot import binding
from hmr4d.lidar_mocap.relocations import DataRelocations
from test_lidar_pilot import full_spec
from test_lidar_pilot_relocations import write
from test_lidar_relocations import SID, relocated_source_fixture


spec = importlib.util.spec_from_file_location("tested_apply_leg", PROJ_ROOT / "tools/apply_badminton_leg_refinement.py")
apply = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apply)


@pytest.fixture
def review_run(tmp_path):
    data = tmp_path / "DATA"
    data.mkdir()
    index_path, index_sha, resolver = relocated_source_fixture(data)
    labels = tmp_path / "repair/final_labels"
    value = dict(full_spec(tmp_path), data_root=str(data), data_relocations=dict(resolver.binding))
    value["source_indexes"] = {SID: dict(file=index_path.relative_to(data).as_posix(), sha256=index_sha)}
    value["clips"][0]["source_id"] = SID
    value["review_authorization"] = write(tmp_path / "camera_review.json", dict(
        schema="badminton-calibration-review-status-v1", baseline=value["camera_label"], usage="review_only",
        camera_pilot_authorized=True, body_labels_production_admitted=False, training_published=False,
        result_files=value["bundle_sha256"], user_confirmation="Synthetic frozen camera review"))
    original = write(tmp_path / "pilot.json", value)
    registry = dict(review_only=True, training_published=False, data_root=str(data),
        data_relocations=dict(resolver.binding), pilot_manifest=original, coverage_audit=value["coverage_audit"],
        execution_scope=value["execution_scope"], sources={SID: dict(source_index=dict(
            path=str(index_path), sha256=index_sha))})
    write(labels / "source_registry.json", registry)
    calibration = SimpleNamespace(data=dict(sessions_root=str(data / "sessions"), review_only=True,
        configuration_files_sha256=value["bundle_sha256"], data_relocations=dict(resolver.binding)))
    selection = dict(windows={SID + ":first": dict(selected=[dict(synthetic=True)]),
                              SID + ":second": dict(selected=[dict(synthetic=True)]),
                              "unused:third": dict(selected=[])})
    return SimpleNamespace(**locals())


def test_real_relocated_session_is_validated_once_per_source_and_original_reads_remain_byte_bound(
        review_run, monkeypatch):
    fixture = review_run
    assert not fixture.index_path.exists()
    actual = apply.load_run_session
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs["relocations"])
        assert isinstance(kwargs["relocations"], DataRelocations)
        return actual(*args, **kwargs)

    monkeypatch.setattr(apply, "load_run_session", load)
    sessions = apply.load_review_sessions(fixture.labels, fixture.calibration, fixture.selection)
    assert set(sessions) == {SID} and len(calls) == 1
    session = sessions[SID]
    assert dict(session.data_relocations.binding) == dict(fixture.resolver.binding)
    assert len(session.pairs()) == 3
    np.testing.assert_array_equal(session.read_cloud(session.pairs()[0], viewer=False), [[1, 2, 3]])
    assert not (fixture.labels.parent / "leg_refinement_before").exists()


@pytest.mark.parametrize("change", ["camera_binding", "registry_binding", "manifest_bytes", "mapped_source_bytes"])
def test_real_wrong_relocation_contract_is_rejected_before_any_model_or_application_write(
        review_run, monkeypatch, change):
    fixture = review_run
    if change == "camera_binding":
        fixture.calibration.data["data_relocations"]["bytes"] = 0
    elif change == "registry_binding":
        fixture.registry["data_relocations"]["sha256"] = "0" * 64
        write(fixture.labels / "source_registry.json", fixture.registry)
    elif change == "manifest_bytes":
        (fixture.data / fixture.resolver.binding["file"]).write_bytes(b"changed")
    else:
        reference = fixture.registry["sources"][SID]["source_index"]
        mapped = fixture.resolver.resolve(dict(file=fixture.index_path.relative_to(fixture.data).as_posix(),
                                              sha256=reference["sha256"]))
        mapped.write_bytes(b"changed actual relocated source")
    root = fixture.labels.parent
    selected = {key: item for key, item in fixture.selection["windows"].items() if item["selected"]}
    selection = dict(state="complete", windows=selected,
        refinement_policy_sha256=binding(PROJ_ROOT / "hmr4d/lidar_mocap/leg_refinement.py")["sha256"],
        selection_policy_sha256=binding(PROJ_ROOT / "hmr4d/lidar_mocap/leg_selection.py")["sha256"])
    source = write(root / "source_union.json", dict(schema="synthetic-not-executed"))
    write(fixture.labels / "repair_plan.json", dict(jobs=[{}, {}], source_union=source["path"],
                                                   source_union_sha256=source["sha256"]))
    write(fixture.labels / "repair_receipt.json", dict(windows=2))
    write(root / "leg_refinement_selection.json", selection)
    monkeypatch.setattr(apply.CalibrationSnapshot, "load", lambda path: fixture.calibration)
    monkeypatch.setattr(apply.torch, "set_num_threads", lambda value: pytest.fail("Model/QA preparation ran too soon"))
    monkeypatch.setattr(apply, "backup_file", lambda *args: pytest.fail("Backup/application ran too soon"))
    monkeypatch.setattr(apply, "write_json", lambda *args: pytest.fail("Application marker ran too soon"))
    monkeypatch.setattr(sys, "argv", [str(PROJ_ROOT / "tools/apply_badminton_leg_refinement.py"), "--root", str(root)])
    before = {path: path.read_bytes() for path in (root / "leg_refinement_selection.json",
              fixture.labels / "repair_plan.json", fixture.labels / "repair_receipt.json")}
    with pytest.raises((ValueError, FileNotFoundError)):
        apply.main()
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert not (fixture.labels / "post_refinement.json").exists()
    assert not (root / "leg_refinement_before").exists()
    assert not (root / "leg_refinement_receipt.json").exists()


def test_all_selected_sources_are_preloaded_before_any_application_even_if_later_source_fails(
        review_run, monkeypatch):
    fixture = review_run
    selection = deepcopy(fixture.selection)
    selection["windows"]["zz_missing_source:last"] = dict(selected=[{}])
    loaded = []
    actual = apply.load_run_session

    def load(*args, **kwargs):
        loaded.append(args[2])
        return actual(*args, **kwargs)

    monkeypatch.setattr(apply, "load_run_session", load)
    with pytest.raises(KeyError):
        apply.load_review_sessions(fixture.labels, fixture.calibration, selection)
    assert loaded == [SID, "zz_missing_source"]
    assert not (fixture.labels.parent / "leg_refinement_before").exists()


def test_legacy_session_without_registry_keeps_none_resolver_and_no_relocation_discovery(tmp_path, monkeypatch):
    labels = tmp_path / "labels"
    selection = dict(windows={"legacy:window": dict(selected=[{}])})
    session = object()
    calls = []
    monkeypatch.setattr(apply, "load_pilot_run_relocations", lambda *args: pytest.fail("Legacy table discovery"))
    monkeypatch.setattr(apply, "load_run_session", lambda *args, **kwargs: calls.append(kwargs) or session)
    assert apply.load_review_sessions(labels, SimpleNamespace(data={}), selection) == dict(legacy=session)
    assert calls == [dict(relocations=None)]


def test_no_selected_windows_need_no_review_session(tmp_path, monkeypatch):
    monkeypatch.setattr(apply, "load_run_session", lambda *args, **kwargs: pytest.fail("Unselected session load"))
    selection = dict(windows={"unused:a": dict(selected=[])})
    assert apply.load_review_sessions(tmp_path, SimpleNamespace(data={}), selection) == {}


def test_explicit_cpu_review_only_mode_loads_real_sessions_and_returns_before_model_or_any_write(
        review_run, monkeypatch, capsys):
    fixture = review_run
    root = fixture.labels.parent
    selected = {key: item for key, item in fixture.selection["windows"].items() if item["selected"]}
    selection = dict(state="complete", windows=selected,
        refinement_policy_sha256=binding(PROJ_ROOT / "hmr4d/lidar_mocap/leg_refinement.py")["sha256"],
        selection_policy_sha256=binding(PROJ_ROOT / "hmr4d/lidar_mocap/leg_selection.py")["sha256"])
    source = write(root / "source_union.json", dict(schema="synthetic-not-executed"))
    write(fixture.labels / "repair_plan.json", dict(jobs=[{}, {}], source_union=source["path"],
                                                   source_union_sha256=source["sha256"]))
    write(fixture.labels / "repair_receipt.json", dict(windows=2))
    write(root / "leg_refinement_selection.json", selection)
    monkeypatch.setattr(apply.CalibrationSnapshot, "load", lambda path: fixture.calibration)
    monkeypatch.setattr(apply.torch, "set_num_threads", lambda value: pytest.fail("CPU-only check reached models/QA"))
    monkeypatch.setattr(apply, "write_json", lambda *args: pytest.fail("CPU-only check attempted a write"))
    monkeypatch.setattr(sys, "argv", [str(PROJ_ROOT / "tools/apply_badminton_leg_refinement.py"),
                                    "--root", str(root), "--review-sessions-only", "--threads", "1"])
    apply.main()
    assert "review_session_contracts_validated" in capsys.readouterr().out
    assert not (fixture.labels / "post_refinement.json").exists()
    assert not (root / "leg_refinement_before").exists()
