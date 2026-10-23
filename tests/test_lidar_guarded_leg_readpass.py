"""Actual Windows synthetic locks and CPU capability lifecycle; no models/CUDA."""

from copy import deepcopy
from contextlib import nullcontext
import codecs
import ctypes
from ctypes import wintypes
from datetime import datetime, timedelta, timezone
import json
import mmap
import os
from pathlib import Path
import pickle
import struct
from types import SimpleNamespace

import pytest

from hmr4d.lidar_mocap import guarded_leg_gpu as auth, guarded_leg_execution as execution
from hmr4d.lidar_mocap import verified_read_pass as readpass
from hmr4d.lidar_mocap.pilot import binding
from test_lidar_guarded_leg_gpu import put, stage


@pytest.fixture
def locks(tmp_path):
    if os.name != "nt":
        pytest.skip("Actual Windows share-mode test requires Windows")
    with readpass.WindowsReadLocks(roots=[tmp_path]) as value:
        yield value


def test_actual_leaf_lock_allows_read_and_denies_write_truncate_delete_replace_and_rename(tmp_path, locks):
    path = tmp_path / "input.bin"
    path.write_bytes(b"unchanged")
    locks.pin(path)
    assert path.read_bytes() == b"unchanged"
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"replacement")
    for operation in (lambda: path.write_bytes(b"changed"), path.unlink,
                      lambda: path.rename(tmp_path / "renamed.bin"), lambda: replacement.replace(path)):
        with pytest.raises(PermissionError):
            operation()
    locks.assert_held()
    locks.close()
    path.write_bytes(b"after close")


def test_actual_existing_writer_and_writable_mapping_prevent_any_successful_pin(tmp_path, locks):
    path = tmp_path / "input.bin"
    path.write_bytes(b"x" * 4096)
    with path.open("r+b") as writer:
        with pytest.raises(OSError) as error:
            locks.pin(path)
        assert error.value.winerror == 32
        mapped = mmap.mmap(writer.fileno(), 0, access=mmap.ACCESS_WRITE)
    try:
        with pytest.raises(OSError) as error:
            locks.pin(path)
        assert error.value.winerror == 32
    finally:
        mapped.close()
    locks.pin(path)
    locks.assert_held()


def test_actual_ancestor_pins_deny_directory_rename_but_allow_future_grant_creation(tmp_path, locks):
    directory = tmp_path / "inputs"
    directory.mkdir()
    path = directory / "input.json"
    path.write_text("{}", encoding="utf-8")
    locks.pin(path)
    with pytest.raises(PermissionError):
        directory.rename(tmp_path / "renamed")
    # A directory pin alone does not protect a future child. The grant must remain writable.
    (directory / "grant.json").write_text("{}", encoding="utf-8")
    assert (directory / "grant.json").read_text(encoding="utf-8") == "{}"
    temporary = directory / "grant.json.tmp"
    temporary.write_text('{"fresh":true}', encoding="utf-8")
    temporary.replace(directory / "grant.json")
    assert (directory / "grant.json").read_text(encoding="utf-8") == '{"fresh":true}'
    locks.assert_held()


def make_junction(alias, target):
    alias.mkdir()
    kernel = readpass._kernel()
    handle = kernel.CreateFileW(str(alias), 0x40000000, 7, None, 3, 0x02000000 | 0x00200000, None)
    if handle == readpass._INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        substitute, display = ("\\??\\" + str(target)).encode("utf-16-le"), str(target).encode("utf-16-le")
        names = substitute + b"\0\0" + display + b"\0\0"
        data = struct.pack("<IHHHHHH", 0xa0000003, 8 + len(names), 0, 0, len(substitute),
                           len(substitute) + 2, len(display)) + names
        payload = ctypes.create_string_buffer(data)
        returned = wintypes.DWORD()
        if not kernel.DeviceIoControl(handle, 0x000900a4, payload, len(data), None, 0, ctypes.byref(returned), None):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


def test_actual_junction_and_resolved_leaf_retain_real_handles_and_block_data_writes(tmp_path, locks):
    target, alias = tmp_path / "actual_runtime", tmp_path / "runtime_alias"
    target.mkdir()
    path = target / "input.json"
    path.write_text("{}", encoding="utf-8")
    try:
        make_junction(alias, target)
    except OSError as error:
        pytest.skip("Actual Windows junction creation was unavailable: " + str(error))
    locks.pin(alias / path.name)
    assert os.path.normcase(str(path)) in locks.files
    assert os.path.normcase(str(alias / path.name)) in locks.files
    assert locks.files[os.path.normcase(str(path))]["file_id"] == locks.files[
        os.path.normcase(str(alias / path.name))]["file_id"]
    with pytest.raises(PermissionError):
        path.write_bytes(b"changed")
    with pytest.raises(PermissionError):
        alias.rmdir()
    with pytest.raises(PermissionError):
        target.rename(tmp_path / "renamed_runtime")
    item = locks.directories[os.path.normcase(str(alias))]
    payload = ctypes.create_string_buffer(struct.pack("<IHH", 0xa0000003, 0, 0))
    returned = wintypes.DWORD()
    assert not locks.kernel.DeviceIoControl(item["handle"], 0x000900ac, payload, 8, None, 0,
                                           ctypes.byref(returned), None)
    assert ctypes.get_last_error() == 5
    write_handle = locks.kernel.CreateFileW(str(alias), 0x40000000, 7, None, 3, 0x02000000 | 0x00200000, None)
    assert write_handle == readpass._INVALID_HANDLE
    assert ctypes.get_last_error() == 32
    locks.assert_held()


def test_actual_attribute_only_junction_change_remains_possible_and_is_rejected_at_boundary(tmp_path, locks):
    target, alias = tmp_path / "actual_input", tmp_path / "input_alias"
    target.mkdir()
    leaf = target / "input.bin"
    leaf.write_bytes(b"canonical bytes remain held")
    try:
        make_junction(alias, target)
    except OSError as error:
        pytest.skip("Actual Windows junction creation was unavailable: " + str(error))
    locks.pin(alias / leaf.name)
    handle = locks.kernel.CreateFileW(str(alias), 0x100, 7, None, 3, 0x02000000 | 0x00200000, None)
    assert handle != readpass._INVALID_HANDLE
    try:
        payload = ctypes.create_string_buffer(struct.pack("<IHH", 0xa0000003, 0, 0))
        returned = wintypes.DWORD()
        changed = locks.kernel.DeviceIoControl(handle, 0x000900ac, payload, 8, None, 0,
                                               ctypes.byref(returned), None)
        if not changed:
            pytest.skip("This filesystem denied attribute-only reparse deletion: " + str(ctypes.get_last_error()))
    finally:
        locks.kernel.CloseHandle(handle)
    with pytest.raises(ValueError, match="ancestor handle changed identity or file type"):
        locks.assert_held()
    with pytest.raises(PermissionError):
        leaf.write_bytes(b"canonical data write is still forbidden")


def test_actual_nonempty_canonical_ancestor_cannot_become_a_junction_with_attribute_only_handle(tmp_path, locks):
    directory, target = tmp_path / "ordinary_input", tmp_path / "different_input"
    directory.mkdir()
    target.mkdir()
    leaf = directory / "input.bin"
    leaf.write_bytes(b"held ordinary child")
    locks.pin(leaf)
    handle = locks.kernel.CreateFileW(str(directory), 0x100, 7, None, 3, 0x02000000 | 0x00200000, None)
    assert handle != readpass._INVALID_HANDLE
    try:
        substitute, display = ("\\??\\" + str(target)).encode("utf-16-le"), str(target).encode("utf-16-le")
        names = substitute + b"\0\0" + display + b"\0\0"
        data = struct.pack("<IHHHHHH", 0xa0000003, 8 + len(names), 0, 0, len(substitute),
                           len(substitute) + 2, len(display)) + names
        payload, returned = ctypes.create_string_buffer(data), wintypes.DWORD()
        assert not locks.kernel.DeviceIoControl(handle, 0x000900a4, payload, len(data), None, 0,
                                               ctypes.byref(returned), None)
        assert ctypes.get_last_error() == 145  # ERROR_DIR_NOT_EMPTY; its real child remains pinned.
    finally:
        locks.kernel.CloseHandle(handle)
    assert leaf.read_bytes() == b"held ordinary child"
    locks.assert_held()


def test_actual_hardlink_alias_write_is_blocked_and_release_cannot_unlock_parent_alias(tmp_path, locks):
    current, parent = tmp_path / "current.bin", tmp_path / "parent.bin"
    current.write_bytes(b"same actual file")
    os.link(current, parent)
    locks.pin(current)
    locks.pin(parent)
    assert locks.files[os.path.normcase(str(current))]["file_id"] == locks.files[
        os.path.normcase(str(parent))]["file_id"]
    with pytest.raises(PermissionError):
        parent.write_bytes(b"changed")
    with pytest.raises(ValueError, match="immutable or unselected"):
        locks.release_exact([current])
    with pytest.raises(PermissionError):
        current.write_bytes(b"changed")


def test_actual_release_is_exact_and_retains_unselected_original_input(tmp_path, locks):
    selected, original = tmp_path / "selected.bin", tmp_path / "original.bin"
    selected.write_bytes(b"selected")
    original.write_bytes(b"original")
    locks.pin(selected)
    locks.pin(original)
    locks.release_exact([selected])
    selected.write_bytes(b"updated")
    with pytest.raises(PermissionError):
        original.write_bytes(b"changed")
    locks.assert_held()


@pytest.mark.parametrize("name", ["../outside.bin", "sub/../../outside.bin", "input.bin:alternate"])
def test_lock_path_boundary_rejects_traversal_and_alternate_streams(tmp_path, locks, name):
    with pytest.raises(ValueError):
        locks.pin(tmp_path / name)


def test_lock_limit_or_missing_input_fails_closed_and_context_closes_prior_handles(tmp_path):
    if os.name != "nt":
        pytest.skip("Actual Windows sharing test requires Windows")
    path = tmp_path / "held.bin"
    path.write_bytes(b"held")
    with pytest.raises(FileNotFoundError):
        with readpass.WindowsReadLocks(roots=[tmp_path]) as value:
            value.pin(path)
            value.pin(tmp_path / "missing.bin")
    path.write_bytes(b"released after failure")
    with readpass.WindowsReadLocks(roots=[tmp_path], maximum_handles=1) as value:
        with pytest.raises(OSError, match="handle limit"):
            value.pin(path)


def test_python_read_observer_pins_before_read_and_does_not_substitute_bytes(tmp_path, locks):
    path = tmp_path / "input.json"
    path.write_bytes(b'{"actual":true}')
    observed = []

    def pin(target):
        observed.append(target)
        locks.pin(target)

    with readpass.observe_reads(pin):
        assert path.read_bytes() == b'{"actual":true}'
        with pytest.raises(ValueError, match="must not write"):
            path.write_bytes(b"changed")
    assert observed == [path]
    assert path.read_bytes() == b'{"actual":true}'
    with pytest.raises(PermissionError):
        path.write_bytes(b"changed")


def test_stat_only_discovery_and_original_deep_resolver_use_same_actual_mapped_file_without_extra_hash(
        tmp_path, locks, monkeypatch):
    import msvcrt
    from hmr4d.lidar_mocap import relocations
    from hmr4d.lidar_mocap.source_index import _bound_file

    data_root = tmp_path / "data"
    sid = "20261019_152542"
    target = data_root / "sessions" / sid / "lidar/cloud.png"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"synthetic mapped actual point bytes")
    expected = binding(target)
    logical = "upstream/2026-10-19/" + sid + "/cloud.png"
    mapped = dict(file=target.relative_to(data_root).as_posix(), sha256=expected["sha256"], bytes=target.stat().st_size)
    manifest = put(data_root / "relocations.json", dict(
        schema="badminton-data-relocations-v1", entries={logical: mapped}))
    resolver = relocations.DataRelocations(data_root, manifest["path"], manifest["sha256"])
    reference = dict(file=logical, sha256=expected["sha256"], bytes=mapped["bytes"])
    with monkeypatch.context() as patch:
        patch.setattr(relocations, "_file_sha", lambda path: pytest.fail(
            "Name discovery must not add a mapped body read"))
        path = readpass._stat_source_path(data_root, reference, resolver)
        assert path == target
        locks.pin(path)
        with pytest.raises(ValueError, match="frozen relocation binding"):
            readpass._stat_source_path(data_root, dict(reference, sha256="f" * 64), resolver)
        with pytest.raises(ValueError, match="case alias"):
            readpass._stat_source_path(data_root, dict(reference, file=logical.upper()), resolver)
    # This is the unchanged deep resolver, with its actual SHA read restored.
    assert _bound_file(data_root, reference, verify=False, relocations=resolver) == target
    with target.open("rb") as actual_reader:
        identity = locks._info(msvcrt.get_osfhandle(actual_reader.fileno()))["file_id"]
        assert identity == locks.files[os.path.normcase(str(path))]["file_id"]
        assert actual_reader.read() == b"synthetic mapped actual point bytes"


@pytest.fixture
def cpu_stage(stage, monkeypatch):
    if os.name != "nt":
        pytest.skip("Live capability requires actual Windows read handles")
    # Preimport the unchanged validator consumers before the synthetic lock scope;
    # production files and installed packages must never receive test read locks.
    from hmr4d.lidar_mocap import formal_native_derive, relocations, source_index
    codecs.lookup("utf-8-sig")

    native_locks = readpass.WindowsReadLocks
    instances = []

    def synthetic_locks(**options):
        value = native_locks(roots=[stage.root.parents[2]], **options)
        instances.append(value)
        return value

    monkeypatch.setattr(readpass, "WindowsReadLocks", synthetic_locks)
    monkeypatch.setattr(readpass, "PROJ_ROOT", stage.root.parents[2])
    monkeypatch.setattr(readpass, "_runtime_names", lambda: dict(files=[], versions=dict(synthetic=True)))
    clock = stage.start - timedelta(minutes=1)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock

    monkeypatch.setattr(readpass, "datetime", FixedDatetime)
    data_root = stage.folder / "synthetic_data"
    png = data_root / "unselected.png"
    put(png, dict(synthetic_png=True, selected_scan=False))
    mapping = put(data_root / "unselected.members.json", dict(synthetic_member=True))
    frame = dict(file=png.name, png_sha256=binding(png)["sha256"], png_bytes=png.stat().st_size,
                 member_mapping=dict(file=Path(mapping["path"]).name, sha256=mapping["sha256"]))
    sid = next(iter(stage.keys)).split(":")[0]
    index = put(data_root / "index.json", dict(frames=[frame], parent_source_kind="original_fullraw_fixed24"))
    old_registry = put(stage.folder / "old_source/source_registry.json", dict(
        data_root=str(data_root), sources={sid: dict(source_index=index)}))
    new_registry = put(stage.folder / "new_source/source_registry.json", dict(
        data_root=str(data_root), sources={sid: dict(source_index=index)}))
    equivalence = put(stage.folder / "equivalence.json", dict(
        sources={sid: dict(original_registry=old_registry, original_index=index, admitted_index=index)},
        evidence=dict(formal_registry=new_registry)))
    audit = put(stage.folder / "audit.json", dict(input_equivalence=equivalence, algorithms={}))
    imported = auth.bound_json(stage.evidence["formal_numeric_import"])
    imported["audit"] = audit
    stage.evidence["formal_numeric_import"] = put(stage.folder / "numeric_import.json", imported)
    stage.ready["evidence"] = stage.evidence
    readiness = put(stage.folder / "baseline/readiness.json", stage.ready)
    stage.grant["readiness"] = readiness
    put(stage.grant_path, stage.grant)
    checked = auth.inspect_request(stage.request_path)
    bundle = put(stage.folder / "pregrant.json", dict(
        schema="gvhmr-guarded-leg-pregrant-v1", state="cpu_inputs_only_not_authorized",
        request=binding(stage.request_path), runtime_contract=checked["runtime_binding"],
        readiness=readiness, independent_guard=stage.grant["independent_guard"],
        cpu_wait_deadline_utc=(stage.start + timedelta(hours=1)).isoformat(),
        gpu_started=False, training_published=False))
    stage.bundle = bundle
    stage.instances = instances
    stage.unselected_png = png
    stage.member_map = Path(mapping["path"])
    for key in sorted(stage.keys):
        source_id, name = key.split(":", 1)
        window = stage.root / "final_labels/sessions" / source_id / "windows" / name
        rows = [dict(motion_id="athlete1", lidar_index=0, fit_path=str(window / "fits/body.npz"))]
        (window / "quality.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        (window / "ptv3.metadata.jsonl").write_text("{}\n", encoding="utf-8")
    stage.prepare = lambda: readpass.VerifiedReadPass.prepare(stage.request_path, bundle["path"], bundle["sha256"])
    return stage


def test_original_deep_runs_once_and_locks_all_stat_only_inputs_before_ready(cpu_stage, monkeypatch):
    actual = auth.validate_readiness_inputs
    calls = []

    def complete(*args):
        calls.append(args)
        return actual(*args)

    monkeypatch.setattr(auth, "validate_readiness_inputs", complete)
    original_open = Path.open

    def no_extra_stat_only_bytes(path, mode="r", *args, **kwargs):
        if path == cpu_stage.unselected_png and "r" in mode:
            pytest.fail("An unselected stat-only PNG must not gain an extra full-file body read")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_extra_stat_only_bytes)
    with cpu_stage.prepare() as capability:
        assert len(calls) == 1
        assert capability.state == "ready"
        receipt = auth.bound_json(capability.receipt)
        assert receipt["state"] == "cpu_ready_waiting_for_grant"
        assert receipt["driver"] == capability.owner
        assert receipt["gpu_started"] is False
        inventory = auth.bound_json(receipt["read_set"])
        paths = {item["path"] for item in inventory["files"]}
        assert str(cpu_stage.unselected_png) in paths
        assert str(cpu_stage.member_map) in paths
        png_entry = next(item for item in inventory["files"] if item["path"] == str(cpu_stage.unselected_png))
        assert png_entry["stat_only_declared"] is True and png_entry["python_read_observed"] is False
        assert "sha256" not in png_entry and inventory["additional_content_hashes_computed"] is False
        assert all(not item["missing"] and item["locked"] == item["declared"]
                   for item in inventory["stat_only_sources"].values())
        with pytest.raises(PermissionError):
            cpu_stage.unselected_png.unlink()
        with pytest.raises(PermissionError):
            cpu_stage.member_map.write_bytes(b"changed")
        with pytest.raises(TypeError, match="cannot be serialized"):
            pickle.dumps(capability)
        assert not (cpu_stage.folder / "guarded_leg_stage_lease.json").exists()
    cpu_stage.unselected_png.write_bytes(b"released only after capability closes")


@pytest.mark.parametrize("first_alias_read", ["pregrant", "during_deep"])
def test_first_deep_cannot_issue_ready_through_a_junction_even_if_alias_would_later_restore(
        cpu_stage, monkeypatch, first_alias_read):
    alias = cpu_stage.folder / "cpu_input_alias"
    try:
        make_junction(alias, cpu_stage.folder)
    except OSError as error:
        pytest.skip("Actual Windows junction creation was unavailable: " + str(error))
    calls = []

    def deep(*args):
        calls.append(args)
        (alias / "pregrant.json").read_bytes()

    monkeypatch.setattr(auth, "validate_readiness_inputs", deep)
    pregrant = alias / "pregrant.json" if first_alias_read == "pregrant" else Path(cpu_stage.bundle["path"])
    with pytest.raises(ValueError, match="canonical entity paths without junction/reparse aliases"):
        readpass.VerifiedReadPass.prepare(cpu_stage.request_path, pregrant, cpu_stage.bundle["sha256"])
    assert len(calls) == (first_alias_read == "during_deep") and cpu_stage.instances[-1].closed
    assert not list((cpu_stage.folder / "cpu_read_passes").glob("*/cpu_ready.json"))


@pytest.mark.parametrize("error", [ValueError("deep validator failed"), KeyboardInterrupt("cancelled")])
def test_deep_failure_or_baseexception_closes_all_handles_and_never_emits_ready(cpu_stage, monkeypatch, error):
    def fail(*args):
        raise error

    monkeypatch.setattr(auth, "validate_readiness_inputs", fail)
    with pytest.raises(type(error)):
        cpu_stage.prepare()
    assert cpu_stage.instances[-1].closed
    assert not list((cpu_stage.folder / "cpu_read_passes").glob("*/cpu_ready.json"))
    cpu_stage.unselected_png.write_bytes(b"released")


def checked_grant(stage, capability):
    checked = auth.load_authorization(stage.request_path, stage.grant_path, now=stage.start + timedelta(seconds=1),
                                      deep=False)
    desktop = put(stage.folder / "fresh_desktop.json", dict(
        schema="gvhmr-guarded-leg-wddm-desktop-identities-v1", state="frozen_observed_desktop_only",
        request=checked["request_binding"], observed_utc=stage.start.isoformat(),
        valid_until_utc=(stage.start + timedelta(hours=1)).isoformat(),
        future_or_reused_PID_exempt=False, desktop_identities=[]))
    checked["authorization"]["grant"].update(
        verified_read_pass=capability.receipt, desktop_allowance=desktop,
        desktop_allowance_fixture_only=False)
    return checked


def test_fresh_grant_can_use_only_same_live_capability_and_exact_inputs(cpu_stage):
    with cpu_stage.prepare() as capability:
        checked = checked_grant(cpu_stage, capability)
        assert capability.validate_grant(checked) == cpu_stage.ready
        assert capability.check_baseline(checked) == capability.current_artifacts
        changed = deepcopy(checked)
        changed["authorization"]["grant"]["readiness"]["sha256"] = "f" * 64
        with pytest.raises(ValueError, match="does not match"):
            capability.validate_grant(changed)
        changed = deepcopy(checked)
        changed["authorization"]["binding"] = dict(path="another", sha256="f" * 64)
        with pytest.raises(ValueError, match="renew, extend or switch"):
            capability.validate_grant(changed)
        capability.owner["create_time"] -= 1
        with pytest.raises(ValueError, match="same live driver"):
            capability.validate_grant(checked)


def test_closed_or_consumed_capability_cannot_reauthorize_even_if_all_bytes_are_unchanged(cpu_stage):
    with cpu_stage.prepare() as capability:
        checked = checked_grant(cpu_stage, capability)
        capability.validate_grant(checked)
        capability.state = "apply_invalidated"
        with pytest.raises(ValueError, match="same live driver"):
            capability.validate_grant(checked)
    with pytest.raises(ValueError, match="same live driver"):
        capability.validate_grant(checked)


def test_grant_start_cannot_precede_completed_deep_cpu_verification(cpu_stage):
    with cpu_stage.prepare() as capability:
        checked = checked_grant(cpu_stage, capability)
        checked["authorization"]["start"] = cpu_stage.start - timedelta(hours=1)
        with pytest.raises(ValueError, match="must start after"):
            capability.validate_grant(checked)


def test_existing_writable_mapping_in_stat_only_source_prevents_ready_receipt(cpu_stage):
    with cpu_stage.unselected_png.open("r+b") as writer:
        mapped = mmap.mmap(writer.fileno(), 0, access=mmap.ACCESS_WRITE)
    try:
        with pytest.raises(OSError) as error:
            cpu_stage.prepare()
        assert error.value.winerror == 32
    finally:
        mapped.close()
    assert cpu_stage.instances[-1].closed
    assert not list((cpu_stage.folder / "cpu_read_passes").glob("*/cpu_ready.json"))


def test_missing_stat_only_member_fails_closed_before_any_ready_receipt(cpu_stage):
    cpu_stage.member_map.unlink()
    with pytest.raises(FileNotFoundError):
        cpu_stage.prepare()
    assert cpu_stage.instances[-1].closed
    assert not list((cpu_stage.folder / "cpu_read_passes").glob("*/cpu_ready.json"))


def test_copied_serialized_receipt_cannot_recreate_live_in_memory_capability(cpu_stage):
    with cpu_stage.prepare() as capability:
        checked = checked_grant(cpu_stage, capability)
        serialized = auth.bound_json(capability.receipt)
        copied = readpass.VerifiedReadPass()
        copied.receipt = capability.receipt
        copied.owner = serialized["driver"]
        copied.nonce = serialized["nonce"]
        with pytest.raises(ValueError, match="same live driver"):
            copied.validate_grant(checked)


def test_runtime_member_change_after_ready_is_rejected_without_rebuilding_context(cpu_stage, monkeypatch):
    with cpu_stage.prepare() as capability:
        checked = checked_grant(cpu_stage, capability)
        monkeypatch.setattr(readpass, "_runtime_names", lambda: dict(files=["new runtime member"], versions={}))
        with pytest.raises(ValueError, match="inventory names or versions"):
            capability.validate_grant(checked)


def test_actual_baseline_bytes_are_read_again_with_held_locks_at_each_boundary(cpu_stage, monkeypatch):
    with cpu_stage.prepare() as capability:
        original = readpass.check_binding
        seen = []

        def record(item):
            seen.append(item["path"])
            return original(item)

        monkeypatch.setattr(readpass, "check_binding", record)
        capability.check_baseline(checked_grant(cpu_stage, capability))
        assert set(capability.current_artifacts) <= set(seen)


def selection_for_apply(stage):
    key = sorted(stage.keys)[0]
    sid, name = key.split(":", 1)
    window = stage.root / "final_labels/sessions" / sid / "windows" / name
    fit = window / "fits/body.npz"
    arrays = put(stage.root / "leg_refinement_proposals/candidate_arrays.npz", dict(synthetic_candidate=True))
    selected = dict(motion_id="athlete1", lidar_index=0, source_fit_path=str(fit),
                    source_fit_sha256=binding(fit)["sha256"],
                    candidate=dict(arrays_path=arrays["path"], arrays_sha256=arrays["sha256"]))
    put(stage.root / "leg_refinement_selection.json", dict(
        state="complete", selected_frames=1,
        windows={item: dict(selected=[selected] if item == key else []) for item in stage.keys}))
    return window, fit, arrays


def test_apply_consumes_capability_only_after_real_baseline_recheck_and_releases_exact_current_paths(cpu_stage):
    with cpu_stage.prepare() as capability:
        checked = checked_grant(cpu_stage, capability)
        window, fit, arrays = selection_for_apply(cpu_stage)
        released = capability.before_apply(checked)
        assert capability.state == "apply_invalidated"
        assert str(fit) in released
        fit.write_bytes(b"only selected current fit may change")
        (window / "quality.jsonl").write_text("updated\n", encoding="utf-8")
        with pytest.raises(PermissionError):
            cpu_stage.unselected_png.write_bytes(b"cannot change source")
        with pytest.raises(PermissionError):
            Path(arrays["path"]).write_bytes(b"cannot change selected candidate")
        remaining_fit = next(path for path in capability.current_artifacts
                             if path.endswith("body.npz") and path != str(fit))
        with pytest.raises(PermissionError):
            Path(remaining_fit).write_bytes(b"cannot change unselected fit")
        with pytest.raises(ValueError, match="same live driver"):
            capability.before_apply(checked)


def test_apply_rejects_current_target_sharing_a_file_id_with_any_protected_parent_input(cpu_stage):
    key = sorted(cpu_stage.keys)[0]
    sid, name = key.split(":", 1)
    fit = cpu_stage.root / "final_labels/sessions" / sid / "windows" / name / "fits/body.npz"
    parent = cpu_stage.folder / "parent_numeric_copy.npz"
    os.link(fit, parent)
    with cpu_stage.prepare() as capability:
        capability.locks.pin(parent)
        checked = checked_grant(cpu_stage, capability)
        selection_for_apply(cpu_stage)
        with pytest.raises(ValueError, match="immutable or unselected"):
            capability.before_apply(checked)
        assert capability.state == "ready"
        with pytest.raises(PermissionError):
            parent.write_bytes(b"must remain protected")
        with pytest.raises(PermissionError):
            fit.write_bytes(b"must remain protected")


def test_apply_does_not_release_a_current_path_that_also_has_an_immutable_evidence_role(cpu_stage):
    key = sorted(cpu_stage.keys)[0]
    sid, name = key.split(":", 1)
    fit = cpu_stage.root / "final_labels/sessions" / sid / "windows" / name / "fits/body.npz"
    identity_binding = cpu_stage.evidence["formal_reviewed_identity"]
    identity = auth.bound_json(identity_binding)
    identity["evidence_files"][str(fit)] = binding(fit)["sha256"]
    cpu_stage.evidence["formal_reviewed_identity"] = put(Path(identity_binding["path"]), identity)
    readiness = put(Path(cpu_stage.grant["readiness"]["path"]), cpu_stage.ready)
    cpu_stage.grant["readiness"] = readiness
    put(cpu_stage.grant_path, cpu_stage.grant)
    bundle = auth.bound_json(cpu_stage.bundle)
    bundle["readiness"] = readiness
    cpu_stage.bundle.update(put(Path(cpu_stage.bundle["path"]), bundle))
    with cpu_stage.prepare() as capability:
        checked = checked_grant(cpu_stage, capability)
        window, fit, arrays = selection_for_apply(cpu_stage)
        with pytest.raises(ValueError, match="immutable input role"):
            capability.before_apply(checked)
        with pytest.raises(PermissionError):
            fit.write_bytes(b"still immutable")


def test_cpu_wait_without_grant_does_not_probe_gpu_or_create_lease_job_or_launch(cpu_stage, monkeypatch):
    class Waiting:
        wait_deadline = datetime.now(timezone.utc) - timedelta(seconds=1)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

    value = Waiting()
    monkeypatch.setattr(readpass.VerifiedReadPass, "prepare", lambda *args: value)
    monkeypatch.setattr(execution, "cpu_only", lambda: None)
    for name in ("compute_pids", "WindowsGPUJob", "make_local_stage_lease", "load_authorization"):
        monkeypatch.setattr(execution, name, lambda *args, **kwargs: pytest.fail(
            "No GPU authority exists while waiting"))
    with pytest.raises(TimeoutError, match="CPU waiting deadline"):
        execution.execute(cpu_stage.request_path, cpu_stage.folder / "not_issued.json",
                          pregrant_path=cpu_stage.bundle["path"], pregrant_sha256=cpu_stage.bundle["sha256"])
    assert value.closed


@pytest.mark.parametrize("change", ["none", "wrong_driver", "wrong_pregrant_command", "pregrant_grant_start"])
def test_independent_watchdog_binds_receipt_to_exact_live_driver_and_canonical_pregrant(cpu_stage, monkeypatch, change):
    with cpu_stage.prepare() as capability:
        checked = checked_grant(cpu_stage, capability)
        command = ["python.exe", "-u", str(execution.PROJ_ROOT / "tools/run_badminton_guarded_leg.py"),
                   "--execute", "--request", str(cpu_stage.request_path), "--grant", str(cpu_stage.grant_path),
                   "--pregrant", cpu_stage.bundle["path"], "--pregrant-sha256", cpu_stage.bundle["sha256"]]
        driver = dict(capability.owner, command=command)
        receipt = dict(auth.bound_json(capability.receipt), driver=driver)
        if change == "wrong_driver":
            receipt["driver"] = dict(driver, create_time=driver["create_time"] - 1)
        if change == "wrong_pregrant_command":
            command[11] = "f" * 64
        if change == "pregrant_grant_start":
            checked["authorization"]["start"] = cpu_stage.start - timedelta(hours=1)
        checked["authorization"]["grant"]["verified_read_pass"] = put(
            cpu_stage.folder / "watchdog_cpu_ready.json", receipt)
        monkeypatch.setattr(auth, "identity_alive", lambda value: value == driver)
        if change == "none":
            assert readpass.validate_live_receipt(checked, driver) == receipt
            nonce = "a" * 32
            context = dict(nonce=nonce, job_name="Local\\GVHMRGuardedLeg_" + nonce, root=str(cpu_stage.root),
                request=checked["request_binding"], driver=driver, immutable_bindings=auth.immutable_anchors(checked))
            execution.validate_context(context, cpu_stage.folder / "attempts" / nonce / "context.json", checked)
        else:
            with pytest.raises(ValueError):
                readpass.validate_live_receipt(checked, driver)


def test_cpu_pass_remains_held_through_owned_job_cleanup_when_launch_is_aborted(cpu_stage, monkeypatch):
    capability = cpu_stage.prepare()
    checked = checked_grant(cpu_stage, capability)
    put(cpu_stage.grant_path, checked["authorization"]["grant"])
    now = cpu_stage.start + timedelta(seconds=1)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(execution, "datetime", FixedDatetime)
    monkeypatch.setattr(execution, "cpu_only", lambda: None)
    monkeypatch.setattr(readpass.VerifiedReadPass, "prepare", lambda *args: capability)
    loads, emptied = [], []

    def load(request, grant, *, deep=True):
        loads.append(deep)
        return auth.load_authorization(request, grant, now=now, deep=deep)

    monkeypatch.setattr(execution, "load_authorization", load)
    from hmr4d.lidar_mocap.gpu_handoff import make_local_stage_lease
    monkeypatch.setattr(execution, "make_local_stage_lease", lambda *args: make_local_stage_lease(*args, now=now))
    monkeypatch.setattr(execution, "compute_pids", lambda: set())

    class Job:
        def __init__(self, name):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.wait_empty()

        def deadline_guard(self, seconds):
            assert 0 < seconds <= 1800
            return nullcontext()

        def pids(self):
            return []

        def start(self, *args, **kwargs):
            capability.locks.assert_held()
            raise RuntimeError("synthetic stop before any actual child or CUDA launch")

        def terminate(self):
            capability.locks.assert_held()

        def wait_empty(self):
            assert not capability.locks.closed
            emptied.append(True)

    def watchdog_ready(command, **kwargs):
        path = Path(command[command.index("--context") + 1])
        process = execution.psutil.Process()
        put(path.parent / "guard_ready.json", dict(pid=process.pid, create_time=process.create_time(),
            context_sha256=binding(path)["sha256"], ready=True))
        return SimpleNamespace(poll=lambda: None, wait=lambda **kw: 0)

    monkeypatch.setattr(execution, "WindowsGPUJob", Job)
    monkeypatch.setattr(execution.subprocess, "Popen", watchdog_ready)
    with pytest.raises(RuntimeError, match="before any actual child"):
        execution.execute(cpu_stage.request_path, cpu_stage.grant_path,
                          pregrant_path=cpu_stage.bundle["path"], pregrant_sha256=cpu_stage.bundle["sha256"])
    assert loads == [False, False]
    assert emptied and capability.locks.closed
    lease = auth.read_json(cpu_stage.folder / "guarded_leg_stage_lease.json")
    assert lease["status"] == "released" and lease["approved_deadline_utc"] == cpu_stage.grant["approved_deadline_utc"]
    assert not list((cpu_stage.folder / "attempts").glob("*/propose.launch.json"))


def test_readpass_implementation_is_in_every_required_code_binding():
    assert auth.implementation_files()["verified_read_pass"] == (
        auth.PROJ_ROOT / "hmr4d/lidar_mocap/verified_read_pass.py")
