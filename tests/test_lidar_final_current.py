"""Portable current-CROP identity/security tests; no GPU, training or old parent."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from hmr4d.lidar_mocap.final_current import FinalCurrentRelease, prepare_final_current, _identity, _json, _undo_lineage


POINT = "decoded_fixed24_valid_returns_before_raw_to_viewer"
GRID = "zero-based row-major cell in one coordinate band"


class PortableFixture:
    def __init__(self, root):
        self.root = root
        self.sid = "20261101_000000"
        self.directory = f"sessions/{self.sid}/output-lidar/lidar_{self.sid}"
        self.old_directory = (f"E:/Datas/badminton-recoding/collections/crop/"
                              f"sources/{self.sid}/output-lidar/lidar_{self.sid}")
        self.map_directory = f"sessions/{self.sid}/member-maps/primary"
        self.old_map_directory = f"E:/Datas/badminton-recoding/collections/crop/sources/{self.sid}/member-maps"
        grid = dict(coordinate_encoding="fixed24", stored_xyz_space="lidar_sensor_xyz", point_width=4,
                    point_height=2, scale=1000, center=256, image_size=[4, 6])
        native = dict(**grid, frames=[], frame_count=3, frame_timestamps=[1.0, 2.0, 3.0],
                      frame_capture_timestamps=[dict(device_first_packet_us=i * 100) for i in range(3)])
        old_frames, new_frames, rows, identities = [], [], [], []
        for number in range(3):
            png = f"frame_{number:06d}.png"
            # All three encoded axes valid at original J grid cells 2 and 7.
            codes = np.zeros((3, 8), dtype=np.uint32)
            codes[:, [2, 7]] = np.asarray([[256001, 256002], [256003, 256004], [256005, 256006]])
            image = np.stack([codes & 255, (codes >> 8) & 255, codes >> 16], axis=-1).reshape(6, 4, 3).astype(np.uint8)
            ok, encoded = cv2.imencode(".png", image)
            assert ok
            png_ref = self.write_bytes(f"{self.directory}/{png}", encoded.tobytes())
            path = self.root / f"{self.map_directory}/frame_{number:06d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, parent_valid_to_child=np.array([-1, 0, -1, 1], dtype=np.int64),
                                child_valid_to_parent=np.array([1, 3], dtype=np.int64),
                                parent_valid_to_grid=np.array([0, 2, 4, 7], dtype=np.int64),
                                schema="badminton-native-roi-member-map-v1", dtype="int64",
                                parent_scope="fullraw_exported_clouds",
                                point_index_space=POINT, grid_index_space=GRID, parent_png_sha256="a" * 64,
                                child_png_sha256=png_ref["sha256"], crop_binding_sha256="b" * 64)
            mapping = dict(self.ref(path.relative_to(root).as_posix()), schema="badminton-native-roi-member-map-v1",
                           dtype="int64", parent_valid_points=4, child_valid_points=2,
                           point_index_space=POINT, grid_index_space=GRID)
            old = dict(frame_index=number, file=f"{self.old_directory}/{png}", png_sha256=png_ref["sha256"],
                       png_bytes=png_ref["bytes"], point_count=2, original_frame_id=number + 17,
                       frame_map_lidar_frame=number, scan_mid_us=100 + number * 100,
                       cameras={"cam1": dict(status="unpaired", record_index=None)},
                       member_mapping=dict(mapping, file=f"{self.old_map_directory}/frame_{number:06d}.npz"),
                       crop_binding_sha256="b" * 64,
                       parent_png=dict(file=f"{self.old_directory}/retired_{number}.png",
                                       sha256="a" * 64, bytes=999, valid_points=4),
                       parent_valid_to_grid_sha256=hashlib.sha256(
                           np.array([0, 2, 4, 7], dtype="<i8").tobytes()).hexdigest())
            new = dict(old, file=png_ref["file"], member_mapping=mapping,
                       parent_png=dict(original_file=old["parent_png"]["file"],
                                       reference_scope="lineage_only_not_a_live_payload_dependency",
                                       sha256="a" * 64, bytes=999, valid_points=4))
            old_frames.append(old)
            new_frames.append(new)
            native["frames"].append(dict(file=png, sha256=png_ref["sha256"], bytes=png_ref["bytes"], kept_points=2))
            rows.append([number, png, 2])
            identities.append([png_ref["sha256"], png_ref["bytes"]])
        old_native_ref = self.write_json("provenance/files/native.json", native)
        self.mapping = dict(schema="badminton-final-review-path-map-v1", files=[], directories=[
            dict(original_directory=self.old_directory, file=self.directory),
            dict(original_directory=self.old_map_directory, file=self.map_directory)])
        old_native_identity = dict(old_native_ref, file=self.old_directory + "/manifest.json")
        self.mapping["files"].append(dict(old_native_identity, original_file=old_native_identity["file"],
                                          file=old_native_ref["file"]))
        self.original = dict(schema="badminton-source-index-v1", source_id=self.sid, fixed24=grid,
                             frame_count=3, frames=old_frames, source_manifest=old_native_identity,
                             parent_scope="fullraw_exported_clouds", parent_input_data_root="D:/retired-original",
                             coordinate_contract=dict(M_baked_into_stored_xyz=False,
                                                      application_raw_to_viewer_M=np.eye(3).tolist()),
                             camera_calibration_admitted=False, body_labels_production_admitted=False)
        old_index_ref = self.write_json("provenance/files/index.json", self.original)
        old_index_identity = dict(old_index_ref, file="E:/Datas/badminton-recoding/collections/crop/source-index.json")
        self.mapping["files"].append(dict(old_index_identity, original_file=old_index_identity["file"],
                                          file=old_index_ref["file"]))
        new_native = copy.deepcopy(native)
        for row in new_native["frames"]:
            row["file"] = self.directory + "/" + row["file"]
        new_native["provenance"] = dict(original_manifest_ref=old_native_ref, path_only_portability_derivation=True)
        native_ref = self.write_json(self.directory + "/manifest.json", new_native)
        self.index = dict(self.original, schema="badminton-final-current-source-index-v1",
                          original_schema=self.original["schema"],
                          frames=new_frames, data_root=".", source_manifest=native_ref,
                          provenance=dict(original_source_index_ref=old_index_ref,
                                          historical_parent_payload_refs_are_lineage_only=True,
                                          scientific_values_and_current_point_members_preserved=True))
        del self.index["parent_input_data_root"]
        self.index["original_parent_input_data_root"] = "D:/retired-original"
        old_helper = dict(schema="badminton-final-review-source-v1", source_id=self.sid,
                          png_directory=self.old_directory,
                          source_index_data_root="E:/Datas/badminton-recoding", source_index_ref=old_index_identity,
                          frame_count=3, fixed24=grid, frame_fields=["frame_index", "png_basename", "point_count"],
                          camera_pair_fields=[], frames=rows, png_sha256_and_bytes_by_frame=identities,
                          camera_calibration_admitted=False,
                          geometry=dict(viewer_M=np.eye(3).tolist(), camera_states=None),
                          timing=dict(clock_quality="software_arrival", exposure_sync=False))
        old_helper_ref = self.write_json("provenance/files/helper.json", old_helper)
        self.helper = dict(old_helper, png_directory=self.directory, canonical_png_directory=self.directory,
                           source_index_data_root=".", provenance=dict(original_selected_helper_ref=old_helper_ref,
                                                                      original_base_helper_ref=old_helper_ref))
        ids = [f"20261101_{n:06d}" for n in range(35)]
        sessions = [dict(id=sid, frame_count=3, events_present=False, events_frame_count=None,
                         source_manifest_ref={}) for sid in ids]
        old_master = dict(schema="badminton-final-review-manifest-v1", recording_root="E:/Datas/badminton-recoding",
                          counts=dict(sources=35), sessions=copy.deepcopy(sessions))
        self.master = dict(old_master, sessions=sessions)
        self.release = dict(schema="badminton-final-review-release-v1", dataset_root=".", release_id="test-final",
                            origin_recording_root=old_master["recording_root"],
                            original_manifest_ref=self.write_json("provenance/files/master.json", old_master),
                            sources=[dict(source_id=sid, helper_ref={}, source_index_ref={}) for sid in ids])
        self.path = root / "metadata/final-review-release.json"
        self.publish()

    def ref(self, name):
        raw = (self.root / name).read_bytes()
        return dict(file=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def write_bytes(self, name, raw):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return self.ref(name)

    def write_json(self, name, value):
        return self.write_bytes(name, json.dumps(value, allow_nan=False).encode())

    def publish(self):
        index_ref = self.write_json(f"metadata/source-indices/{self.sid}.json", self.index)
        self.helper["source_index_ref"] = index_ref
        helper_ref = self.write_json(f"metadata/sources/{self.sid}.json", self.helper)
        for row, master in zip(self.release["sources"], self.master["sessions"]):
            row.update(helper_ref=helper_ref, source_index_ref=index_ref)
            master["source_manifest_ref"] = helper_ref
        self.release["canonical_manifest_ref"] = self.write_json("metadata/master.json", self.master)
        self.release["path_map_ref"] = self.write_json("metadata/path-map.json", self.mapping)
        self.write_json("metadata/final-review-release.json", self.release)

    def source(self):
        return FinalCurrentRelease(self.path, self.root).source(self.sid)


@pytest.fixture
def portable(tmp_path):
    return PortableFixture(tmp_path)


def test_real_current_decode_without_parent_and_single_M(portable):
    source = portable.source()
    points, proof = source.read_cloud(1)
    assert points.dtype == np.float32 and points.shape == (2, 3)
    assert np.allclose(points, [[.001, .003, .005], [.002, .004, .006]], atol=2e-5)
    viewer, evidence = source.read_cloud(1, viewer=True)
    assert np.array_equal(viewer, points @ np.eye(3))
    assert proof["original_J_to_current_inverse_verified"] is True
    assert proof["parent_payload_reopened"] is False and evidence["application_M_count"] == 1
    report = prepare_final_current(portable.path, portable.root, portable.sid)
    assert len(report["samples"]) == 3 and report["training_admitted"] is False
    assert report["camera_exposure_admission_issued"] is False


@pytest.mark.parametrize("mutation", ["time", "M", "frame", "source", "bool", "parent_is_child",
                                      "provenance", "schema", "variant"])
def test_changed_science_is_rejected_even_with_rebound_release(portable, mutation):
    if mutation == "time":
        portable.index["frames"][0]["scan_mid_us"] += 1
    elif mutation == "M":
        portable.index["coordinate_contract"]["application_raw_to_viewer_M"][0][0] += .01
    elif mutation == "frame":
        portable.index["frames"][0]["original_frame_id"] += 1
    elif mutation == "source":
        portable.index["source_id"] = "20261101_000001"
    elif mutation == "bool":
        portable.index["camera_calibration_admitted"] = 0
    elif mutation == "parent_is_child":
        frame = portable.index["frames"][0]
        frame["parent_png"] = dict(file=frame["file"], bytes=frame["png_bytes"],
                                   sha256=frame["png_sha256"], valid_points=2)
    elif mutation == "provenance":
        portable.index["provenance"]["original_source_index_ref"]["sha256"] = "f" * 64
    elif mutation == "schema":
        portable.index["schema"] = "badminton-source-index-v1"
    else:
        portable.helper["png_directory"] = f"annotations/manual-hit-corpus-v1/foreground/{portable.sid}"
    portable.publish()
    with pytest.raises(ValueError):
        portable.source()


@pytest.mark.parametrize("mutation", ["inverse", "grid", "selected", "dtype", "parent_sha", "child_sha"])
def test_bound_map_tampering_is_rejected(portable, mutation):
    source = portable.source()
    frame = source.index["frames"][0]
    ref = frame["member_mapping"]
    path = portable.root / ref["file"]
    with np.load(path, allow_pickle=False) as z:
        values = {k: z[k] for k in z.files}
    if mutation == "inverse":
        values["parent_valid_to_child"][0] = 0
    elif mutation == "grid":
        values["parent_valid_to_grid"][1] = 1
    elif mutation == "selected":
        values["child_valid_to_parent"] = np.array([3, 1], dtype=np.int64)
    elif mutation == "dtype":
        values["child_valid_to_parent"] = values["child_valid_to_parent"].astype(np.int32)
    else:
        values["parent_png_sha256" if mutation == "parent_sha" else "child_png_sha256"] = np.array("f" * 64)
    np.savez_compressed(path, **values)
    # Unit probe of the independent math/scope guards after a bound NPZ read.
    # The separate in-memory-context test proves callers cannot do this in a
    # normal source handle; here the fixture also rebinds that guard explicitly.
    ref.update(portable.ref(ref["file"]))
    source._frame_identities = (_identity(frame), *source._frame_identities[1:])
    with pytest.raises(ValueError):
        source.read_cloud(0)


def test_changed_png_bytes_rejected(portable):
    source = portable.source()
    path = portable.root / portable.index["frames"][0]["file"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="byte length"):
        source.read_cloud(0)


def test_in_memory_context_changes_cannot_bypass_original_index(portable):
    source = portable.source()
    source.index["frames"][0]["scan_mid_us"] += 1
    with pytest.raises(ValueError, match="changed in memory"):
        source.read_cloud(0)


def test_decoder_cannot_return_different_bits_than_hashed_png(portable, monkeypatch):
    source = portable.source()
    monkeypatch.setattr("hmr4d.lidar_mocap.final_current.decode_fixed24",
                        lambda *args: np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="hash-verified"):
        source.read_cloud(0)


def test_lossless_lineage_collision_contract():
    original = dict(file="sessions/cam1.mp4", original_file="upstream/cam1.mp4", sha256="a" * 64, bytes=123)
    portable = dict(original_file=original["original_file"], lineage_file=original["file"],
                    reference_scope="lineage_only_not_a_live_payload_dependency", sha256="a" * 64, bytes=123)
    assert _undo_lineage(portable) == original
    with pytest.raises(ValueError, match="Colliding"):
        _undo_lineage(dict(lineage_file="old-file", reference_scope="lineage_only_not_a_live_payload_dependency"))


@pytest.mark.parametrize("name", ["../escape", "C:/escape", "/escape", "a\\b", "a/../b", "a//b",
                                  "a/./b", "a/stream:ads", "a/b. "])
def test_path_escape_rejected(portable, name):
    release = FinalCurrentRelease(portable.path, portable.root)
    with pytest.raises(ValueError):
        release.relative(name)


def test_symlink_escape_rejected(portable, tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    link = portable.root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Platform denies unprivileged directory symlinks")
    with pytest.raises(ValueError, match="escapes"):
        FinalCurrentRelease(portable.path, portable.root).relative("link/file.json")


def test_duplicate_map_and_source_rejected(portable):
    portable.mapping["directories"].append(copy.deepcopy(portable.mapping["directories"][0]))
    portable.publish()
    with pytest.raises(ValueError, match="Duplicate"):
        portable.source()
    portable.mapping["directories"].pop()
    portable.release["sources"][1]["source_id"] = portable.sid
    portable.publish()
    with pytest.raises(ValueError, match="duplicate"):
        portable.source()


def test_release_pin_duplicate_json_and_lineage_only_live_open(portable):
    with pytest.raises(ValueError, match="SHA256"):
        FinalCurrentRelease(portable.path, portable.root, "f" * 64)
    with pytest.raises(ValueError, match="Duplicate"):
        _json(b'{"frames":[],"frames":[]}')
    with pytest.raises(ValueError, match="Non-finite"):
        _json(b'{"clock":NaN}')
    source = portable.source()
    with pytest.raises(ValueError, match="live"):
        source.release.document(source.index["frames"][0]["parent_png"])


def test_cli_modes_do_not_relax_legacy_mode(tmp_path):
    script = Path(__file__).resolve().parents[1] / "tools/prepare_badminton_source.py"
    for options in (["--source-index", "x", "--final-current-release", "y"],
                    ["--final-current-release", "y"], ["--source-index", "x", "--source-id", "20261101_000000"]):
        result = subprocess.run([sys.executable, str(script), *options, "--data-root", str(tmp_path),
                                 "--report", "outputs/test-final-current/report.json"], text=True, capture_output=True)
        assert result.returncode == 2


def _retirement_ref(portable, document):
    reference = portable.write_json("metadata/point-retirement.json", document)
    portable.index["point_mapping_retirement_ref"] = reference
    portable.helper["point_mapping_retirement_ref"] = reference
    portable.release["point_mapping_retirement_ref"] = reference
    portable.publish()


def _current_only(portable, *, remove_maps=True):
    previous = copy.deepcopy(portable.index)
    previous_ref = portable.write_json("provenance/files/previous-portable-v1.json", previous)
    counts = {f"{role}_npz_{field}": 0 for role in ("primary", "immediate", "prior") for field in ("count", "bytes")}
    maps = {"member_mapping": "primary", "immediate_member_mapping": "immediate", "prior_member_mapping": "prior"}
    for frame in previous["frames"]:
        for name, role in maps.items():
            if name in frame:
                counts[f"{role}_npz_count"] += 1
                counts[f"{role}_npz_bytes"] += frame[name]["bytes"]
                if remove_maps:
                    (portable.root / frame[name]["file"]).unlink()
    if remove_maps:
        (portable.root / portable.map_directory).rmdir()
        (portable.root / portable.map_directory).parent.rmdir()
    counts.update(total_npz_count=sum(counts[f"{r}_npz_count"] for r in ("primary", "immediate", "prior")),
                  total_npz_bytes=sum(counts[f"{r}_npz_bytes"] for r in ("primary", "immediate", "prior")),
                  retained_png_count=3, retained_png_bytes=sum(f["png_bytes"] for f in previous["frames"]))
    rows = []
    for row in portable.release["sources"]:
        values = counts if row["source_id"] == portable.sid else {key: 0 for key in counts}
        rows.append(dict(source_id=row["source_id"], previous_canonical_source_index_ref=previous_ref,
                         mapping_schema_before_retirement=previous.get("mapping_schema"), **values))
    retirement = dict(schema="badminton-point-mapping-retirement-v1", complete=True,
                      status="excluded_from_current_dataset_by_user", point_mapping_available=False,
                      physical_removal_status="execution_recorded_separately", sources=rows,
                      totals=dict(source_count=35, **{key: sum(row[key] for row in rows) for key in counts}))
    portable.index.update(schema="badminton-final-current-source-index-v2", point_mapping_available=False,
                          point_mapping_status="retired_by_user",
                          point_identity_scope="current_png_valid_returns_and_grid_cells_only")
    for key in ("point_mapping_available", "point_mapping_status", "point_identity_scope"):
        portable.helper[key] = portable.index[key]
    p = portable.index["provenance"]
    del p["scientific_values_and_current_point_members_preserved"]
    p.update(current_png_points_geometry_clocks_and_frame_values_preserved=True,
             previous_canonical_source_index_ref=previous_ref)
    for frame in portable.index["frames"]:
        for name in maps:
            frame.pop(name, None)
    portable.mapping["directories"] = [row for row in portable.mapping["directories"]
                                        if row["original_directory"] != portable.old_map_directory]
    _retirement_ref(portable, retirement)
    return retirement, previous


def test_v2_returns_real_current_grid_without_original_maps(portable):
    _current_only(portable)
    assert not (portable.root / f"sessions/{portable.sid}/member-maps").exists()
    source = portable.source()
    points, grid, proof = source.read_current_cloud(0)
    assert points.shape == (2, 3) and points.dtype == np.float32
    assert grid.dtype == np.int64 and grid.tolist() == [2, 7]
    assert grid.tolist() != list(range(len(points)))
    assert np.allclose(points, [[.001, .003, .005], [.002, .004, .006]], atol=2e-5)
    assert proof["scan_mid_us"] == 100 and proof["original_frame_id"] == 17
    assert proof["cameras"] == portable.index["frames"][0]["cameras"]
    assert proof["point_mapping_available"] is False
    assert proof["original_point_identity_available"] is False
    assert proof["original_J_to_current_inverse_verified"] is False
    assert proof["original_point_membership_verified"] is False
    assert proof["training_admitted"] is False and proof["camera_exposure_admission_issued"] is False
    assert proof["point_mapping_retirement"]["physical_removal_status"] == "execution_recorded_separately"
    with pytest.raises(ValueError, match="retired by the user"):
        source.read_cloud(0)
    report = prepare_final_current(portable.path, portable.root, portable.sid)
    assert report["schema"] == "gvhmr-final-current-cpu-inspection-v2"
    assert report["samples"][0]["current_grid_indices"] == [2, 7]
    assert report["training_admitted"] is False and report["original_point_membership_verified"] is False
    assert "scientific_path_only_equivalence_verified" not in report
    assert report["non_membership_scientific_equivalence_verified"] is True


def test_v2_logical_retirement_does_not_claim_physical_deletion(portable):
    _current_only(portable, remove_maps=False)
    source = portable.source()
    assert list((portable.root / portable.map_directory).glob("*.npz"))
    assert source.read_current_cloud(1)[2]["point_mapping_available"] is False
    with pytest.raises(ValueError, match="retired"):
        source.read_cloud(1)


def test_missing_v1_map_never_falls_back_to_current_only(portable):
    source = portable.source()
    (portable.root / portable.index["frames"][0]["member_mapping"]["file"]).unlink()
    with pytest.raises(FileNotFoundError):
        source.read_cloud(0)
    with pytest.raises(ValueError, match="explicit v2"):
        source.read_current_cloud(0)


@pytest.mark.parametrize("mutation", ["scope", "available", "active_map", "time", "M", "provenance"])
def test_v2_cannot_waive_science_or_reactivate_maps(portable, mutation):
    _, previous = _current_only(portable)
    if mutation == "scope":
        portable.index["point_identity_scope"] = "original_J"
    elif mutation == "available":
        portable.index["point_mapping_available"] = True
    elif mutation == "active_map":
        portable.index["frames"][0]["member_mapping"] = previous["frames"][0]["member_mapping"]
    elif mutation == "time":
        portable.index["frames"][0]["scan_mid_us"] += 1
    elif mutation == "M":
        portable.index["coordinate_contract"]["application_raw_to_viewer_M"][0][0] += .01
    else:
        portable.index["provenance"]["scientific_values_and_current_point_members_preserved"] = True
    portable.publish()
    with pytest.raises(ValueError):
        portable.source()


@pytest.mark.parametrize("mutation", ["counts", "previous_time", "previous_source", "old_map_sha", "physical_claim"])
def test_v2_retirement_requires_real_hash_bound_history(portable, mutation):
    retirement, previous = _current_only(portable)
    if mutation == "counts":
        retirement["sources"][0]["primary_npz_bytes"] += 1
        retirement["totals"]["primary_npz_bytes"] += 1
    elif mutation == "physical_claim":
        retirement["physical_removal_status"] = "all_files_physically_deleted"
    else:
        if mutation == "previous_time":
            previous["frames"][0]["scan_mid_us"] += 1
        elif mutation == "previous_source":
            previous["source_id"] = "20261101_000001"
        else:
            previous["frames"][0]["member_mapping"]["sha256"] = "f" * 64
        previous_ref = portable.write_json("provenance/files/previous-portable-v1.json", previous)
        portable.index["provenance"]["previous_canonical_source_index_ref"] = previous_ref
        retirement["sources"][0]["previous_canonical_source_index_ref"] = previous_ref
    _retirement_ref(portable, retirement)
    with pytest.raises(ValueError):
        portable.source()


def test_v2_still_rejects_changed_current_png_and_context(portable):
    _current_only(portable)
    source = portable.source()
    source.index["frames"][0]["cameras"] = {}
    with pytest.raises(ValueError, match="changed in memory"):
        source.read_current_cloud(0)
    source = portable.source()
    path = portable.root / portable.index["frames"][0]["file"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="byte length"):
        source.read_current_cloud(0)


def test_cli_selects_v2_declared_current_cloud_mode(portable, monkeypatch, capsys):
    import builtins
    import importlib.util

    original_import = builtins.__import__

    def no_legacy_source_import(name, *args, **kwargs):
        if name == "hmr4d.lidar_mocap.source_index":
            raise AssertionError("Current-only CLI must not import the legacy source/body reader")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_legacy_source_import)
    _current_only(portable)
    script = Path(__file__).resolve().parents[1] / "tools/prepare_badminton_source.py"
    spec = importlib.util.spec_from_file_location("prepare_current_cli_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = portable.root / "outputs/current-v2.json"
    monkeypatch.setattr(module, "PROJ_ROOT", portable.root)
    monkeypatch.setattr(sys, "argv", [str(script), "--final-current-release", str(portable.path),
                                     "--source-id", portable.sid, "--data-root", str(portable.root),
                                     "--report", str(report)])
    module.main()
    output = json.loads(report.read_text())
    assert output["point_mapping_available"] is False
    assert output["samples"][1]["current_grid_indices"] == [2, 7]
    assert output["training_admitted"] is False
    assert "current_png_readable_without_original_point_maps" in capsys.readouterr().out
