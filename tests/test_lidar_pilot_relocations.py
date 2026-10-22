"""A new review pilot freezes its explicit mapping without loosening default readers."""

import hashlib
import json
from pathlib import Path
import runpy
from unittest.mock import patch

import pytest

from hmr4d.lidar_mocap import calibration as calibration_module
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.relocations import (
    DataRelocations,
    load_manifest_relocations,
    load_pilot_run_relocations,
)
from hmr4d.lidar_mocap.rally_pipeline import load_run_session
from hmr4d.lidar_mocap.sessions import SessionIndex

from test_lidar_calibration import SnapshotFixture
from test_lidar_relocations import manifest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return dict(path=str(path), sha256=digest(path))


def bound_run(root):
    data = root / "DATA"
    data.mkdir()
    resolver = manifest(data, {})
    spec = dict(
        schema="badminton-review-pilot-v1",
        review_only=True,
        training_published=False,
        data_root=str(data),
        data_relocations=dict(resolver.binding),
    )
    original = write(root / "original-pilot.json", spec)
    run = root / "run"
    registry = dict(
        review_only=True,
        training_published=False,
        pilot_manifest=original,
        data_root=str(data),
        data_relocations=dict(resolver.binding),
        sources={},
    )
    write(run / "source_registry.json", registry)
    cal = dict(sessions_root=str(data / "sessions"), data_relocations=dict(resolver.binding))
    return run, resolver, spec, registry, cal


def test_manifest_binding_is_the_only_activation_source_and_checks_byte_count(tmp_path):
    run, resolver, spec, _, _ = bound_run(tmp_path)
    loaded = load_manifest_relocations(spec)
    assert dict(loaded.binding) == dict(resolver.binding)
    assert load_manifest_relocations(dict(spec, data_relocations=None)) is None
    with pytest.raises(ValueError, match="byte binding"):
        load_manifest_relocations(dict(spec, data_relocations=dict(resolver.binding, bytes=0)))
    bad = dict(spec, data_relocations=dict(resolver.binding, file="../escape.json"))
    with pytest.raises(ValueError, match="canonical"):
        load_manifest_relocations(bad)


def test_frozen_run_helper_requires_original_manifest_registry_and_camera_same_binding(tmp_path):
    run, resolver, _, registry, cal = bound_run(tmp_path)
    loaded = load_pilot_run_relocations(run, cal)
    assert dict(loaded.binding) == dict(resolver.binding)
    with pytest.raises(ValueError, match="explicitly bound"):
        load_run_session(run, type("Camera", (), dict(data=cal))(), "unused")
    assert (run / "source_registry.json").is_file()


@pytest.mark.parametrize("change", [None, "coverage_registry", "coverage_file", "scope_registry"])
def test_full_native_mapping_requires_original_full_scope_and_unchanged_coverage(tmp_path, change):
    from test_lidar_pilot import full_spec

    run, resolver, original, registry, cal = bound_run(tmp_path)
    spec = dict(full_spec(tmp_path), data_root=original["data_root"], data_relocations=original["data_relocations"])
    registry["pilot_manifest"] = write(tmp_path / "original-pilot.json", spec)
    registry.update(execution_scope=spec["execution_scope"], coverage_audit=spec["coverage_audit"])
    if change == "coverage_registry":
        registry["coverage_audit"] = dict(spec["coverage_audit"], sha256="0" * 64)
    if change == "coverage_file":
        Path(spec["coverage_audit"]["path"]).write_text("changed", encoding="utf-8")
    if change == "scope_registry":
        registry.pop("execution_scope")
    write(run / "source_registry.json", registry)
    if change is None:
        assert dict(load_pilot_run_relocations(run, cal).binding) == dict(resolver.binding)
    else:
        with pytest.raises(ValueError):
            load_pilot_run_relocations(run, cal)


@pytest.mark.parametrize("changed_scope", [False, True])
def test_full_native_source_loader_checks_full_contract_without_formal_admission(tmp_path, changed_scope):
    from test_lidar_pilot import full_spec

    run, resolver, original, registry, cal = bound_run(tmp_path)
    spec = dict(full_spec(tmp_path), data_root=original["data_root"], data_relocations=original["data_relocations"])
    registry.update(
        pilot_manifest=write(tmp_path / "original-pilot.json", spec),
        execution_scope="changed" if changed_scope else spec["execution_scope"],
        coverage_audit=spec["coverage_audit"],
        sources=dict(actual=dict(source_index=dict(path=str(tmp_path / "actual.json"), sha256="b" * 64))),
    )
    cal.update(review_only=True, configuration_files_sha256=spec["bundle_sha256"])
    write(run / "source_registry.json", registry)
    with patch("hmr4d.lidar_mocap.pilot.validate_camera_review"), patch(
        "hmr4d.lidar_mocap.source_index.SourceIndexedSession.load_source_index"
    ) as loader:
        if changed_scope:
            with pytest.raises(ValueError, match="complete coverage"):
                load_run_session(run, type("Camera", (), dict(data=cal))(), "actual", relocations=resolver)
            loader.assert_not_called()
        else:
            load_run_session(run, type("Camera", (), dict(data=cal))(), "actual", relocations=resolver)
            assert loader.call_args.kwargs["require_admitted"] is False
            assert loader.call_args.kwargs["relocations"] is resolver


@pytest.mark.parametrize(
    "change", ["original", "strip_registry", "wrong_registry", "wrong_camera", "strip_camera", "wrong_root"]
)
def test_run_helper_rejects_unbound_or_changed_relocation_inputs(tmp_path, change):
    run, resolver, _, registry, cal = bound_run(tmp_path)
    if change == "original":
        Path(registry["pilot_manifest"]["path"]).write_bytes(b"changed")
    elif change == "strip_registry":
        registry.pop("data_relocations")
    elif change == "wrong_registry":
        registry["data_relocations"] = dict(resolver.binding, sha256="0" * 64)
    elif change == "wrong_camera":
        cal["data_relocations"] = dict(resolver.binding, bytes=0)
    elif change == "strip_camera":
        cal.pop("data_relocations")
    else:
        registry["data_root"] = str(tmp_path / "other")
    write(run / "source_registry.json", registry)
    with pytest.raises(ValueError):
        load_pilot_run_relocations(run, cal)


def test_old_pilot_without_explicit_map_does_not_discover_an_existing_manifest(tmp_path):
    run, resolver, spec, registry, cal = bound_run(tmp_path)
    spec.pop("data_relocations")
    registry["pilot_manifest"] = write(tmp_path / "original-pilot.json", spec)
    registry.pop("data_relocations")
    cal.pop("data_relocations")
    write(run / "source_registry.json", registry)
    assert (resolver.root / resolver.binding["file"]).is_file()
    assert load_pilot_run_relocations(run, cal) is None


@pytest.mark.parametrize("override", [False, True])
def test_pilot_viewer_entry_uses_frozen_manifest_or_explicit_override_once(tmp_path, override):
    from hmr4d import PROJ_ROOT
    import hmr4d.lidar_mocap.pilot as pilot
    import hmr4d.lidar_mocap.relocations as relocation_module

    script = PROJ_ROOT / "tools/run_badminton_pilot.py"
    argv = [str(script), "--root", str(tmp_path / "run"), "--stage", "viewer", "--port", "8782"]
    if override:
        argv += ["--data-relocations", str(tmp_path / "explicit.json"), "--data-relocations-sha256", "a" * 64]
    resolver = object()
    with (
        patch("sys.argv", argv),
        patch.object(pilot, "serve_pilot") as serve,
        patch.object(relocation_module, "load_pilot_run_relocations", return_value=resolver) as frozen_load,
        patch.object(relocation_module, "load_run_relocations", return_value=resolver) as explicit_load,
    ):
        runpy.run_path(str(script), run_name="__main__")
    assert serve.call_args.kwargs["relocations"] is resolver
    assert serve.call_args.args[1] == 8782
    assert frozen_load.call_count == (0 if override else 1)
    assert explicit_load.call_count == (1 if override else 0)


def test_snapshot_freezes_mapping_and_exact_camera_bytes_after_logical_clock_is_absent(tmp_path):
    fixture = SnapshotFixture(tmp_path)
    data = fixture.sessions.parent
    session = SessionIndex.load(fixture.session)

    def reference(path):
        return dict(file=path.relative_to(data).as_posix(), sha256=digest(path), bytes=path.stat().st_size)

    old_clock = f"upstream/2026-10-12/{fixture.sid}/original-clock.json"
    resolver = manifest(data, {old_clock: reference(fixture.clock_path)})
    hashes = {name: digest(fixture.base / name) for name in calibration_module._CONFIG_FILES}
    session.source_index = dict(
        candidate_bundle_sha256=hashes,
        crop_binding=dict(bbox_sha256=hashes["bbox.json"]),
        camera_calibration_admitted=False,
        source_manifest=reference(fixture.manifest_path),
        common_clock=dict(reference(fixture.clock_path), file=old_clock),
        canonical_video_sidecars={
            camera: reference(fixture.video / (camera + ".frames.jsonl")) for camera in ("cam1", "cam2")
        },
    )
    index = data / "test-index.json"
    index.write_text("{}")
    assert not (data / old_clock).exists()
    original_five = {name: (fixture.base / name).read_bytes() for name in calibration_module._CONFIG_FILES}

    def freeze(output, selected_resolver=resolver):
        return CalibrationSnapshot.freeze(
            fixture.project,
            fixture.sessions,
            fixture.workspace / output,
            calibration_dir=fixture.base,
            source_indexes={fixture.sid: reference(index)},
            data_root=data,
            review_source_ids=[fixture.sid],
            relocations=selected_resolver,
        )

    with (
        patch.object(calibration_module, "_WORKSPACE_ROOT", fixture.workspace),
        patch("hmr4d.lidar_mocap.source_index.SourceIndexedSession.load_source_index", return_value=session) as loader,
    ):
        result = freeze("mapped")
        assert loader.call_args.kwargs["relocations"] is resolver
        assert result.data["data_relocations"] == dict(resolver.binding)
        assert result.data["configuration_files_sha256"] == hashes
        assert result.data["production_admitted"] is False and result.data["review_only"] is True
        for name, blob in original_five.items():
            assert (result.snapshot_json.parent / "evidence" / name).read_bytes() == blob
            assert (fixture.base / name).read_bytes() == blob
        assert (result.snapshot_json.parent / "evidence/data-relocations.json").read_bytes() == (
            data / resolver.binding["file"]
        ).read_bytes()
        assert freeze("mapped").data["snapshot_digest"] == result.data["snapshot_digest"]
        with pytest.raises(FileNotFoundError):
            freeze("unmapped", None)
        second_path = data / "another-relocations.json"
        second_path.write_bytes((data / resolver.binding["file"]).read_bytes())
        second = DataRelocations(data, second_path, digest(second_path))
        with pytest.raises(ValueError, match="another adoption"):
            freeze("mapped", second)


def test_calibration_relocation_cannot_be_bound_to_an_unrelated_data_root(tmp_path):
    data = tmp_path / "DATA"
    data.mkdir()
    resolver = manifest(data, {})
    with pytest.raises(ValueError, match="explicit DATA"):
        CalibrationSnapshot.freeze(tmp_path / "project", data / "sessions", tmp_path / "out", relocations=resolver)


def test_native_teacher_cli_preserves_signed_window_id_as_one_argument(tmp_path):
    from hmr4d import PROJ_ROOT
    import hmr4d.lidar_mocap.new_source_repair as fresh

    script = PROJ_ROOT / "tools/prepare_badminton_new_source_repair.py"
    signed = "-00000033308_000005333394"
    argv = [str(script), "--root", str(tmp_path / "repair"), "--source-root", str(tmp_path / "source"),
            "--stage", "teacher", "--source", "actual", "--window=" + signed, "--camera", "cam2", "--threads", "2"]
    with patch("sys.argv", argv), patch.object(fresh, "initial_teacher_window", return_value=dict(tracks=[])) as teacher:
        runpy.run_path(str(script), run_name="__main__")
    assert teacher.call_args.args[1:3] == ("actual", signed)
    assert teacher.call_args.kwargs["camera"] == "cam2"
