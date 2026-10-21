"""Relocated originals stay byte-bound with historical files absent."""

from dataclasses import FrozenInstanceError
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from hmr4d.lidar_mocap.relocations import DataRelocations, load_relocations


SID = "20261019_152542"
OLD = f"upstream/2026-10-19/{SID}/original.bin"
TARGET = f"sessions/{SID}/meta/originals/original.bin"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def put(root, name, data=b"original bytes"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return dict(file=name, sha256=sha(data), bytes=len(data))


def manifest(root, entries):
    binding = put(
        root,
        "relocations.json",
        json.dumps(dict(schema="badminton-data-relocations-v1", entries=entries)).encode(),
    )
    return DataRelocations(root, root / binding["file"], binding["sha256"])


def test_target_always_wins_even_when_old_path_still_exists_and_has_changed(tmp_path):
    original = put(tmp_path, OLD)
    target = put(tmp_path, TARGET)
    resolver = manifest(tmp_path, {OLD: target})
    (tmp_path / OLD).write_bytes(b"obsolete copy")
    assert resolver.resolve(original) == tmp_path / TARGET
    assert resolver.resolve(original).read_bytes() == b"original bytes"


@pytest.mark.parametrize("change", ["missing", "changed", "size"])
def test_broken_target_never_falls_back_to_existing_historical_copy(tmp_path, change):
    original = put(tmp_path, OLD)
    target = put(tmp_path, TARGET)
    resolver = manifest(tmp_path, {OLD: target})
    if change == "missing":
        (tmp_path / TARGET).unlink()
    elif change == "changed":
        (tmp_path / TARGET).write_bytes(b"changed! bytes")
    else:
        (tmp_path / TARGET).write_bytes(b"short")
    with pytest.raises((ValueError, FileNotFoundError)):
        resolver.resolve(original)
    assert (tmp_path / OLD).read_bytes() == b"original bytes"


@pytest.mark.parametrize("change", ["sha", "size", "bool_size", "missing_sha", "null_size"])
def test_mapping_cannot_override_immutable_reference_identity(tmp_path, change):
    original = dict(put(tmp_path, OLD))
    target = put(tmp_path, TARGET)
    resolver = manifest(tmp_path, {OLD: target})
    if change == "sha":
        original["sha256"] = "0" * 64
    elif change == "size":
        original["bytes"] += 1
    elif change == "bool_size":
        original["bytes"] = True
    elif change == "missing_sha":
        original.pop("sha256")
    else:
        original["bytes"] = None
    with pytest.raises(ValueError):
        resolver.resolve(original)


def test_raw_index_without_declared_size_still_checks_target_size_and_hash(tmp_path):
    target = put(tmp_path, TARGET)
    resolver = manifest(tmp_path, {OLD: target})
    assert resolver.resolve(dict(file=OLD, sha256=target["sha256"])) == tmp_path / TARGET
    (tmp_path / TARGET).write_bytes(b"bad")
    with pytest.raises(ValueError):
        resolver.resolve(dict(file=OLD, sha256=target["sha256"]))


def test_manifest_is_loaded_once_and_tables_are_immutable(tmp_path):
    target = put(tmp_path, TARGET)
    resolver = manifest(tmp_path, {OLD: target})
    original = dict(target, file=OLD)
    (tmp_path / "relocations.json").write_text("{}")
    assert resolver.resolve(original) == tmp_path / TARGET
    with pytest.raises(TypeError):
        resolver._entries[OLD]["file"] = "anything"
    with pytest.raises(TypeError):
        resolver.binding["sha256"] = "anything"
    with pytest.raises(FrozenInstanceError):
        resolver.root = Path("anything")


def test_shared_target_requires_same_source_sha_and_size(tmp_path):
    target = put(tmp_path, TARGET)
    alias = f"upstream/2026-10-19/{SID}/another/original.bin"
    resolver = manifest(tmp_path, {OLD: target, alias: target})
    assert resolver.resolve(dict(target, file=alias)) == resolver.resolve(dict(target, file=OLD))
    with pytest.raises(ValueError, match="Conflicting identities"):
        manifest(tmp_path, {OLD: target, alias: dict(target, sha256="0" * 64)})


@pytest.mark.parametrize(
    "old,target",
    [
        ("../outside", TARGET),
        ("/absolute/file", TARGET),
        (OLD.replace("/", "\\"), TARGET),
        (OLD, TARGET + "/../escape"),
        (OLD, "sessions/20261019_152750/original.bin"),
        (OLD, "calibration/original.bin"),
        (OLD.replace("2026-10-19", "2026-10-18"), TARGET),
        ("collections/any/device-time-sources/20261019_152542/original.bin", TARGET),
    ],
)
def test_out_of_scope_and_noncanonical_paths_are_rejected(tmp_path, old, target):
    with pytest.raises(ValueError):
        manifest(tmp_path, {old: dict(file=target, sha256="0" * 64, bytes=0)})


@pytest.mark.parametrize("sid", ["20261010_192753", "20261015_161256", "20261015_162740"])
def test_explicit_unique_device_time_source_domain(tmp_path, sid):
    old = f"collections/native/device-time-sources/{sid}/roi/cloud.png"
    target = put(tmp_path, f"sessions/{sid}/output-roi/cloud.png")
    resolver = manifest(tmp_path, {old: target})
    assert resolver.resolve(dict(target, file=old)) == tmp_path / target["file"]


def test_manifest_hash_and_duplicate_json_keys_are_rejected(tmp_path):
    target = put(tmp_path, TARGET)
    resolver = manifest(tmp_path, {OLD: target})
    path = tmp_path / "relocations.json"
    with pytest.raises(ValueError, match="manifest SHA256"):
        DataRelocations(tmp_path, path, "0" * 64)
    payload = b'{"schema":"badminton-data-relocations-v1","entries":{},"entries":{}}'
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="Duplicate relocation JSON"):
        DataRelocations(tmp_path, path, sha(payload))
    assert resolver.binding["file"] == "relocations.json"


@pytest.mark.parametrize("case", ["source", "target"])
def test_case_aliases_are_rejected(tmp_path, case):
    target = put(tmp_path, TARGET)
    if case == "source":
        entries = {OLD: target, OLD.replace("original", "Original"): target}
    else:
        alias = OLD.replace("original", "another")
        entries = {OLD: target, alias: dict(target, file=TARGET.replace("original.bin", "Original.bin"))}
    with pytest.raises(ValueError, match="case alias"):
        manifest(tmp_path, entries)


def test_input_case_alias_cannot_bypass_a_mapped_read(tmp_path):
    original = put(tmp_path, OLD)
    target = put(tmp_path, TARGET)
    resolver = manifest(tmp_path, {OLD: target})
    with pytest.raises(ValueError, match="reference case alias"):
        resolver.resolve(dict(original, file=OLD.replace("original", "Original")))


def test_symlink_cannot_escape_declared_session(tmp_path):
    actual = put(tmp_path, f"sessions/20261019_152750/original.bin")
    link = tmp_path / TARGET
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(tmp_path / actual["file"])
    except OSError:
        pytest.skip("Local Windows policy does not permit file symlinks")
    resolver = manifest(tmp_path, {OLD: dict(actual, file=TARGET)})
    with pytest.raises(ValueError, match="canonical session"):
        resolver.resolve(dict(actual, file=OLD))


def test_activation_requires_both_explicit_arguments_and_has_no_default_discovery(tmp_path):
    manifest(tmp_path, {})
    assert load_relocations(tmp_path) is None
    with pytest.raises(ValueError, match="together"):
        load_relocations(tmp_path, tmp_path / "relocations.json")
    with pytest.raises(ValueError, match="together"):
        load_relocations(tmp_path, expected_sha256="0" * 64)


def relocated_source_fixture(root, sid=SID, device_only=False):
    from test_lidar_source_clock_mapping import reference, source_fixture

    prefix = (
        f"collections/native/device-time-sources/{sid}"
        if device_only
        else f"upstream/{sid[:4]}-{sid[4:6]}-{sid[6:8]}/{sid}"
    )
    source = root / prefix
    source.mkdir(parents=True)
    index, clock = source_fixture(source)
    index["source_id"] = clock["session_id"] = sid
    index["common_clock"] = reference(source / "clock.json", clock)
    index["original_ptp_validation"] = reference(
        source / "ptp.json", dict(passed=True, common_index_sha256=index["common_clock"]["sha256"])
    )

    def prefix_references(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "file":
                    value[key] = f"{prefix}/{item}"
                else:
                    prefix_references(item)
        elif isinstance(value, list):
            for item in value:
                prefix_references(item)

    prefix_references(index)
    (source / "index.json").write_text(json.dumps(index), encoding="utf-8")
    index_sha = sha((source / "index.json").read_bytes())
    entries = {}
    for path in source.iterdir():
        if path.name.endswith(".raw.json"):
            subdir = "meta/raw"
        elif path.suffix == ".png":
            subdir = "output-fullraw"
        else:
            subdir = "meta/originals"
        target = put(root, f"sessions/{sid}/{subdir}/{path.name}", path.read_bytes())
        entries[path.relative_to(root).as_posix()] = target
    resolver = manifest(root, entries)
    shutil.rmtree(source)
    # Neighbours of the physical clock cannot override its original logical
    # raw-index references; the actual raw files occupy meta/raw instead.
    for track in clock["tracks"]:
        put(root, f"sessions/{sid}/meta/originals/{track['raw_index']}", b"wrong physical neighbour")
    return root / prefix / "index.json", index_sha, resolver


@pytest.mark.parametrize("device_only", [False, True])
def test_actual_source_prepare_and_cloud_reads_work_after_historical_tree_is_removed(tmp_path, device_only):
    import numpy as np

    from hmr4d.lidar_mocap.source_index import SourceIndexedSession, prepare_source_index

    sid = "20261015_162740" if device_only else SID
    index_path, index_sha, resolver = relocated_source_fixture(tmp_path, sid, device_only)
    assert not index_path.exists()
    session = SourceIndexedSession.load_source_index(
        index_path, tmp_path, expected_sha256=index_sha, require_admitted=False, relocations=resolver
    )
    assert len(session.pairs()) == 3 and len(session.original_lidar_clock_rows) == 4
    assert session.pairs()[1]["reasons"] == ["empty_foreground"]
    assert all(Path(pair["lidar_path"]).is_relative_to(tmp_path / "sessions" / sid) for pair in session.pairs())
    np.testing.assert_array_equal(session.read_cloud(session.pairs()[0], viewer=False), [[1, 2, 3]])
    report = prepare_source_index(index_path, tmp_path, index_sha, relocations=resolver)
    assert report["data_relocations"] == dict(resolver.binding)
    assert report["paired"] == 2 and report["original_clock_rows"] == 4
    assert report["camera_calibration_admitted"] is False
    assert report["body_fitting_started"] is False and report["source_files_modified"] is False
    for item in report["fullraw_evidence"]["raw_timestamp_indices"]:
        assert item["file"].startswith("collections/" if device_only else "upstream/")
        assert "sessions/" not in item["file"]
    with pytest.raises(FileNotFoundError):
        SourceIndexedSession.load_source_index(index_path, tmp_path, expected_sha256=index_sha, require_admitted=False)
    with pytest.raises(ValueError, match="no fitting admission"):
        SourceIndexedSession.load_source_index(index_path, tmp_path, expected_sha256=index_sha, relocations=resolver)


def test_fullraw_member_verification_reads_relocated_original_parent_cells(tmp_path):
    from test_lidar_source_clock_mapping import members_fixture

    from hmr4d.lidar_mocap.source_index import _bound_file
    from hmr4d.lidar_mocap.source_membership import verify_fullraw_members

    import numpy as np

    frame, values = members_fixture(tmp_path)
    parent = tmp_path / "parent.png"
    old = f"upstream/2026-10-19/{SID}/parent.png"
    target = put(tmp_path, f"sessions/{SID}/output-fullraw/parent.png", parent.read_bytes())
    parent.unlink()
    frame["parent_png"]["file"] = old
    np.savez_compressed(tmp_path / "members.npz", **values)
    frame["member_mapping"] = put(tmp_path, "bound.npz", (tmp_path / "members.npz").read_bytes())
    resolver = manifest(tmp_path, {old: target})
    result = verify_fullraw_members(
        frame,
        tmp_path / "child.png",
        tmp_path,
        dict(point_width=4, point_height=1),
        _bound_file,
        relocations=resolver,
    )
    assert result["encoded_coordinates_bit_exact"] is True
    assert result["parent_scope"] == "fullraw_exported_clouds"
    assert result["maps_to_full_pcap_returns"] is False


def test_resolver_for_different_data_root_is_not_silently_ignored(tmp_path):
    from hmr4d.lidar_mocap.source_index import _bound_file

    resolver = manifest(tmp_path, {})
    other = tmp_path / "other-data"
    other.mkdir()
    entry = put(other, "asset", b"same basename")
    with pytest.raises(ValueError, match="different DATA root"):
        _bound_file(other, entry, relocations=resolver)


def test_derived_crop_reconstructs_relocated_parent_members_and_original_clock_chain(tmp_path):
    import cv2
    import numpy as np

    from hmr4d.lidar_mocap.source_index import prepare_source_index

    old_index, index_sha, resolver = relocated_source_fixture(tmp_path)
    logical_index = old_index.relative_to(tmp_path).as_posix()
    physical_index = resolver.resolve(dict(file=logical_index, sha256=index_sha))
    original = json.loads(physical_index.read_bytes())
    parent_ref = dict(file=logical_index, sha256=index_sha, bytes=physical_index.stat().st_size)
    child = deepcopy(original)
    child.update(
        source_kind="derived_native_roi_fixed24",
        parent_source_kind="original_fullraw_fixed24",
        parent_scope="fullraw_exported_clouds",
        maps_to_full_pcap_returns=False,
        crop_binding_sha256="c" * 64,
        parent_source_index_identity=parent_ref,
        parent_inputs=dict(source_index=parent_ref, source_manifest=original["source_manifest"]),
    )
    original_clock = resolver.resolve(original["common_clock"])
    child["common_clock"] = put(tmp_path, "collections/frozen-pilot/clock.json", original_clock.read_bytes())
    child["source_manifest"] = put(
        tmp_path, f"sessions/{SID}/output-crop/manifest.json", json.dumps(dict(kind="derived", frame_count=3)).encode()
    )
    for frame, parent in zip(child["frames"], original["frames"]):
        parent_ref = dict(
            file=parent["file"],
            sha256=parent["png_sha256"],
            bytes=parent["png_bytes"],
            valid_points=parent["point_count"],
        )
        parent_path = resolver.resolve(parent_ref)
        child_ref = put(
            tmp_path, f"sessions/{SID}/output-crop/frame-{frame['frame_index']}.png", parent_path.read_bytes()
        )
        frame.update(file=child_ref["file"], parent_png=parent_ref, crop_binding_sha256=child["crop_binding_sha256"])
        image = cv2.imread(str(parent_path), cv2.IMREAD_UNCHANGED).reshape(3, 2, 3)
        grid = np.flatnonzero(np.any(image != 0, axis=-1).all(axis=0)).astype(np.int64)
        mapping_path = tmp_path / f"sessions/{SID}/output-crop/member-{frame['frame_index']}.npz"
        np.savez_compressed(
            mapping_path,
            schema=np.asarray("badminton-native-roi-member-map-v1"),
            dtype=np.asarray("int64"),
            parent_scope=np.asarray("fullraw_exported_clouds"),
            point_index_space=np.asarray("decoded_fixed24_valid_returns_before_raw_to_viewer"),
            grid_index_space=np.asarray("zero-based row-major cell in one coordinate band"),
            parent_png_sha256=np.asarray(parent_ref["sha256"]),
            child_png_sha256=np.asarray(child_ref["sha256"]),
            crop_binding_sha256=np.asarray(child["crop_binding_sha256"]),
            parent_valid_to_grid=grid,
            parent_valid_to_child=np.arange(len(grid), dtype=np.int64),
            child_valid_to_parent=np.arange(len(grid), dtype=np.int64),
        )
        frame["member_mapping"] = dict(
            put(tmp_path, mapping_path.relative_to(tmp_path).as_posix(), mapping_path.read_bytes()),
            schema="badminton-native-roi-member-map-v1",
            parent_valid_points=len(grid),
            child_valid_points=len(grid),
        )
    index_ref = put(tmp_path, "child-index.json", json.dumps(child).encode())
    report = prepare_source_index(tmp_path / index_ref["file"], tmp_path, index_ref["sha256"], relocations=resolver)
    assert report["fullraw_evidence"]["derived"] is True
    assert report["fullraw_evidence"]["original_clock_for_raw_indices"] == original["common_clock"]
    assert len(report["sampled_member_mapping_verifications"]) == 3
    for item in report["sampled_member_mapping_verifications"].values():
        assert item["encoded_coordinates_bit_exact"] is True
        assert item["maps_to_full_pcap_returns"] is False


def test_prepare_cli_requires_both_relocation_arguments_before_reading_inputs(tmp_path):
    import os
    import subprocess
    import sys

    from hmr4d import PROJ_ROOT

    command = [
        sys.executable,
        str(PROJ_ROOT / "tools/prepare_badminton_source.py"),
        "--source-index",
        str(tmp_path / "missing-index.json"),
        "--data-root",
        str(tmp_path),
        "--report",
        str(PROJ_ROOT / "outputs/unused-relocation-report.json"),
        "--data-relocations",
        str(tmp_path / "relocations.json"),
    ]
    result = subprocess.run(
        command,
        cwd=PROJ_ROOT,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    assert result.returncode == 2
    assert "must be supplied together" in result.stderr
    assert "FileNotFoundError" not in result.stderr
