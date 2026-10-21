"""Original packet rows and emitted clouds have separate, evidenced identities."""

from copy import deepcopy
import hashlib
import json

import cv2
import numpy as np
import pytest

from hmr4d.lidar_mocap.source_index import (
    SourceIndexedSession,
    _bound_file,
    prepare_source_index,
    validate_coordinate_contract,
    validate_packet_row_mapping,
    verify_fullraw_evidence,
)
from hmr4d.lidar_mocap.source_membership import verify_fullraw_members


def contract(count=4, selected=(0, 1, 2)):
    rows = [
        dict(
            frame_id=100 + i * 7,
            common_time_us=10000 + i * 20000,
            end_common_time_us=19000 + i * 20000,
            start_packet_index=i * 50,
            end_packet_index=i * 50 + 40,
            pcap_start_offset=i * 1000,
        )
        for i in range(count)
    ]
    frames, maps = [], {}
    for ordinal, source in enumerate(selected):
        row = rows[source]
        item = dict(
            source_common_clock_row_index=source,
            source_common_clock_row=deepcopy(row),
            device_first_us=row["common_time_us"],
            device_last_us=row["end_common_time_us"],
            scan_mid_us=row["common_time_us"] + 4500,
        )
        frames.append(dict(item, frame_index=ordinal, original_frame_id=row["frame_id"]))
        maps[40 + ordinal] = dict(item, lidar_frame=40 + ordinal)
    missing = sorted(set(range(count)) - set(selected))
    omissions = [
        dict(
            source_common_clock_row_index=i,
            source_common_clock_row=deepcopy(rows[i]),
            reason="unclosed_final_scan_not_emitted",
        )
        for i in missing
    ]
    index = dict(
        source_kind="original_fullraw_fixed24",
        frame_count=len(frames),
        frames=frames,
        lidar_common_clock_row_indices=list(selected),
        unexported_packet_clock_rows=omissions,
    )
    return index, rows, maps, list(maps), []


def test_exact_packet_row_contract_retains_final_unemitted_row_and_nonordinal_frame_ids():
    args = contract()
    before = deepcopy(args)
    omitted, mapping = validate_packet_row_mapping(*args)
    assert mapping == [0, 1, 2] and omitted[0]["source_common_clock_row"]["frame_id"] == 121
    assert args == before


def test_one_to_one_export_needs_no_packet_omission():
    args = contract(selected=(0, 1, 2, 3))
    assert validate_packet_row_mapping(*args) == ([], [0, 1, 2, 3])


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "reorder",
        "bool",
        "float",
        "out_of_range",
        "length",
        "frame_id",
        "frame_packet",
        "map_packet",
        "map_time",
        "embedded_type",
        "wrong_reason",
        "omitted_packet",
        "undeclared",
        "mapping_missing",
        "mapping_null",
    ],
)
def test_corrupted_or_inferred_packet_mapping_is_rejected(change):
    args = contract()
    index, rows, maps, selected, _ = args
    if change == "duplicate":
        index["lidar_common_clock_row_indices"][1] = 0
    if change == "reorder":
        index["lidar_common_clock_row_indices"] = [1, 0, 2]
    if change == "bool":
        index["lidar_common_clock_row_indices"][1] = True
    if change == "float":
        index["lidar_common_clock_row_indices"][1] = 1.0
    if change == "out_of_range":
        index["lidar_common_clock_row_indices"][2] = 4
    if change == "length":
        index["lidar_common_clock_row_indices"].pop()
    if change == "frame_id":
        index["frames"][1]["original_frame_id"] += 1
    if change == "frame_packet":
        index["frames"][1]["source_common_clock_row"]["pcap_start_offset"] += 1
    if change == "map_packet":
        maps[selected[1]]["source_common_clock_row"]["end_packet_index"] += 1
    if change == "map_time":
        maps[selected[1]]["scan_mid_us"] += 1
    if change == "embedded_type":
        index["frames"][1]["source_common_clock_row"]["start_packet_index"] = 50.0
    if change == "wrong_reason":
        index["unexported_packet_clock_rows"][0]["reason"] = "missing_png"
    if change == "omitted_packet":
        index["unexported_packet_clock_rows"][0]["source_common_clock_row"]["pcap_start_offset"] += 1
    if change == "undeclared":
        index["unexported_packet_clock_rows"] = []
    if change == "mapping_missing":
        index.pop("lidar_common_clock_row_indices")
    if change == "mapping_null":
        index["lidar_common_clock_row_indices"] = None
    with pytest.raises(ValueError):
        validate_packet_row_mapping(*args)


@pytest.mark.parametrize("selected", [(0, 2, 3), (0, 1)])
def test_an_interior_omission_or_two_missing_packet_rows_cannot_be_relabelled_as_a_tail(selected):
    with pytest.raises(ValueError, match="final unclosed"):
        validate_packet_row_mapping(*contract(selected=selected))


def test_legacy_camera_omission_contract_is_separate_and_unchanged():
    assert validate_packet_row_mapping(dict(frames=[{}]), [], {0: {}}, [0], []) == ([], None)


def test_fullraw_cannot_launder_an_interior_packet_omission_as_camera_coverage():
    args = list(contract(selected=(1, 2, 3)))
    args[0]["unexported_packet_clock_rows"] = []
    args[4] = [dict(original_frame_id=args[1][0]["frame_id"])]
    with pytest.raises(ValueError, match="camera-coverage omissions"):
        validate_packet_row_mapping(*args)


def reference(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def evidence(tmp_path):
    index = contract()[0]
    manifest = reference(tmp_path / "manifest.json", dict(kind="full", frame_count=3))
    raw = reference(tmp_path / "raw.json", [1, 2, 3, 4])
    clock = dict(tracks=[dict(device_id="lidar1", raw_index=raw["file"], raw_index_sha256=raw["sha256"])])
    clock_ref = reference(tmp_path / "clock.json", clock)
    ptp = reference(
        tmp_path / "ptp.json", dict(passed=True, evidence_incomplete=False, common_index_sha256=clock_ref["sha256"])
    )
    proof = dict(
        status="completed",
        exit_code=0,
        complete_pcap_read=True,
        frame_count=3,
        source_manifest_sha256=manifest["sha256"],
        unexported_packet_clock_rows=index["unexported_packet_clock_rows"],
    )
    index.update(
        source_manifest=manifest,
        original_ptp_validation=ptp,
        fullraw_decode_evidence=reference(tmp_path / "proof.json", proof),
    )
    return index, clock, proof


def test_completed_full_export_binds_the_original_manifest_ptp_and_raw_indexes(tmp_path):
    index, clock, _ = evidence(tmp_path)
    result = verify_fullraw_evidence(index, tmp_path, tmp_path / "clock.json", clock)
    assert result["parent_scope"] == "fullraw_exported_clouds"
    assert result["maps_to_full_pcap_returns"] is False and not result["derived"]


@pytest.mark.parametrize(
    "change",
    [
        "status",
        "exit",
        "bool_exit",
        "incomplete",
        "count",
        "manifest",
        "reason",
        "ptp_missing",
        "ptp_changed",
        "raw_changed",
        "no_mapping",
    ],
)
def test_unbound_or_incomplete_fullraw_evidence_is_rejected(tmp_path, change):
    index, clock, proof = evidence(tmp_path)
    if change == "status":
        proof["status"] = "running"
    if change == "exit":
        proof["exit_code"] = 1
    if change == "bool_exit":
        proof["exit_code"] = False
    if change == "incomplete":
        proof["complete_pcap_read"] = False
    if change == "count":
        proof["frame_count"] = 4
    if change == "manifest":
        proof["source_manifest_sha256"] = "another_manifest"
    if change == "reason":
        proof["unexported_packet_clock_rows"] = []
    if change == "ptp_missing":
        (tmp_path / "ptp.json").unlink()
    if change == "ptp_changed":
        (tmp_path / "ptp.json").write_text("{}")
    if change == "raw_changed":
        (tmp_path / "raw.json").write_text("changed")
    if change == "no_mapping":
        index.pop("lidar_common_clock_row_indices")
    index["fullraw_decode_evidence"] = reference(tmp_path / "proof.json", proof)
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_fullraw_evidence(index, tmp_path, tmp_path / "clock.json", clock)


def test_reconstructed_export_requires_original_completion_logs(tmp_path):
    index, clock, proof = evidence(tmp_path)
    proof.update(
        status="reconstructed_from_completion_log",
        exit_code_source="Recorded historical exit receipt",
        frame_limit=None,
        completion_logs=[reference(tmp_path / "completion.json", dict(written=3, status="Done"))],
    )
    index["fullraw_decode_evidence"] = reference(tmp_path / "proof.json", proof)
    verify_fullraw_evidence(index, tmp_path, tmp_path / "clock.json", clock)
    (tmp_path / "completion.json").write_text("changed original log")
    with pytest.raises(ValueError, match="SHA256|size mismatch"):
        verify_fullraw_evidence(index, tmp_path, tmp_path / "clock.json", clock)


def bytes_reference(path, data, root):
    path.write_bytes(data)
    return dict(file=path.relative_to(root).as_posix(), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def source_fixture(root):
    index, clock_rows, maps, selected, _ = contract()
    index.update(
        schema="badminton-source-index-candidate-v1",
        source_id="fullraw-fixture",
        camera_calibration_admitted=False,
        production_admitted=False,
        camera_applicability=dict(status="pending"),
        raw_clock_rows=len(clock_rows),
        fixed24=dict(
            coordinate_encoding="fixed24",
            stored_xyz_space="lidar_sensor_xyz",
            scale=1000,
            center=256,
            point_width=2,
            point_height=1,
            image_width=2,
            image_height=3,
        ),
        coordinate_contract=dict(
            M_baked_into_stored_xyz=False, application_raw_to_viewer_M=[[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        ),
    )
    tracks = []
    for camera in ("cam1", "cam2"):
        raw = [
            dict(common_time_us=row["common_time_us"] + 4500, media_pts_us=i * 20000, frame_no=900 + i)
            for i, row in enumerate(clock_rows)
        ]
        raw_ref = reference(root / (camera + ".raw.json"), raw)
        tracks.append(dict(device_id=camera, rows=raw, raw_index=raw_ref["file"], raw_index_sha256=raw_ref["sha256"]))
        sidecar = [dict(record_index=i, media_pts=row["media_pts_us"]) for i, row in enumerate(raw)]
        index.setdefault("canonical_video_sidecars", {})[camera] = bytes_reference(
            root / (camera + ".frames.jsonl"), "\n".join(json.dumps(row) for row in sidecar).encode(), root
        )
        index.setdefault("native_videos", {})[camera] = bytes_reference(
            root / (camera + ".mp4"), b"metadata-only fixture", root
        )
    lidar_ref = reference(root / "lidar.raw.json", clock_rows)
    tracks.insert(
        0, dict(device_id="lidar1", rows=clock_rows, raw_index=lidar_ref["file"], raw_index_sha256=lidar_ref["sha256"])
    )
    clock = dict(
        schema="external_gm_common_timeline_v1",
        session_id=index["source_id"],
        tracks=tracks,
        common_overlap_begin_us=14500,
        common_overlap_end_us=74500,
    )
    index["common_clock"] = reference(root / "clock.json", clock)
    for i, frame in enumerate(index["frames"]):
        encoded = np.zeros((3, 2), np.uint32)
        if i != 1:
            encoded[:, 0] = [257000, 258000, 259000]
        bgr = np.stack((encoded & 255, (encoded >> 8) & 255, (encoded >> 16) & 255), axis=-1).astype(np.uint8)
        ok, png = cv2.imencode(".png", bgr)
        assert ok
        png_ref = bytes_reference(root / f"frame_{i:06d}.png", png.tobytes(), root)
        frame.update(
            file=png_ref["file"],
            png_sha256=png_ref["sha256"],
            png_bytes=png_ref["bytes"],
            point_count=0 if i == 1 else 1,
            frame_map_lidar_frame=selected[i],
            cameras={},
        )
        for camera in ("cam1", "cam2"):
            expected = dict(
                status="matched",
                record_index=i,
                media_pts_us=i * 20000,
                video_common_us=frame["scan_mid_us"],
                time_error_us=0,
            )
            frame["cameras"][camera] = expected
        maps[selected[i]]["cameras"] = deepcopy(frame["cameras"])
    index["frame_map"] = dict(
        bytes_reference(root / "frame-map.jsonl", "\n".join(json.dumps(row) for row in maps.values()).encode(), root),
        rows=3,
    )
    index["source_manifest"] = reference(root / "manifest.json", dict(kind="full", frame_count=3))
    index["fullraw_decode_evidence"] = reference(
        root / "proof.json",
        dict(
            status="completed",
            exit_code=0,
            complete_pcap_read=True,
            frame_count=3,
            source_manifest_sha256=index["source_manifest"]["sha256"],
            unexported_packet_clock_rows=index["unexported_packet_clock_rows"],
        ),
    )
    index["original_ptp_validation"] = reference(
        root / "ptp.json", dict(passed=True, common_index_sha256=index["common_clock"]["sha256"])
    )
    reference(root / "index.json", index)
    return index, clock


def test_actual_loader_retains_empty_cloud_and_unemitted_tail_inside_camera_coverage(tmp_path):
    index, clock = source_fixture(tmp_path)
    session = SourceIndexedSession.load_source_index(tmp_path / "index.json", tmp_path, require_admitted=False)
    assert len(session.pairs()) == 3 and len(session.original_lidar_clock_rows) == 4
    assert len(session.common_clock["tracks"][0]["rows"]) == 4
    assert session.pairs()[1]["reasons"] == ["empty_foreground"]
    assert [p["source_frame_map_lidar_frame"] for p in session.pairs()] == [40, 41, 42]
    assert [p["source_common_clock_row_index"] for p in session.pairs()] == [0, 1, 2]
    assert session.camera_applicability is None
    np.testing.assert_array_equal(session.read_cloud(session.pairs()[0], viewer=False), [[1, 2, 3]])
    np.testing.assert_array_equal(session.read_cloud(session.pairs()[0], viewer=True), [[-2, 1, 3]])
    report = prepare_source_index(tmp_path / "index.json", tmp_path)
    assert report["frames"] == 3 and report["original_clock_rows"] == 4 and report["paired"] == 2
    assert report["camera_calibration_admitted"] is False and report["body_fitting_started"] is False
    assert report["calibration_applicability"] == "pending" and report["preparation_seconds"] >= 0
    with pytest.raises(ValueError, match="no fitting admission"):
        SourceIndexedSession.load_source_index(tmp_path / "index.json", tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("consumer_application_count_required", 2),
        ("consumer_application_count_required", True),
        ("application_M_applied_by_this_index_builder", True),
        ("M_baked_into_stored_xyz", True),
    ],
)
def test_fullraw_alias_compatibility_never_allows_double_m(tmp_path, field, value):
    index, _ = source_fixture(tmp_path)
    index["coordinate_contract"][field] = value
    with pytest.raises(ValueError):
        validate_coordinate_contract(index)


@pytest.mark.parametrize("field,value", [("image_size", [2, 1]), ("image_width", 3), ("image_height", 6)])
def test_fullraw_grid_aliases_must_agree(tmp_path, field, value):
    index, _ = source_fixture(tmp_path)
    index["fixed24"][field] = value
    with pytest.raises(ValueError, match="grid/image"):
        validate_coordinate_contract(index)


def members_fixture(root, empty=False):
    parent = np.zeros((3, 4, 3), np.uint8)
    parent[:, [0, 1, 3], 2] = 4
    parent[:, [0, 1, 3], 1] = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
    selected = np.asarray([] if empty else [0, 2], np.int64)
    original_grid = np.asarray([0, 1, 3], np.int64)
    child = np.zeros_like(parent)
    child[:, original_grid[selected]] = parent[:, original_grid[selected]]
    refs = []
    for name, image in [("parent", parent), ("child", child)]:
        ok, png = cv2.imencode(".png", image)
        assert ok
        refs.append(bytes_reference(root / (name + ".png"), png.tobytes(), root))
    inverse = np.full(3, -1, np.int64)
    inverse[selected] = np.arange(len(selected))
    values = dict(
        schema=np.asarray("badminton-native-roi-member-map-v1"),
        dtype=np.asarray("int64"),
        parent_scope=np.asarray("fullraw_exported_clouds"),
        point_index_space=np.asarray("decoded_fixed24_valid_returns_before_raw_to_viewer"),
        grid_index_space=np.asarray("zero-based row-major cell in one coordinate band"),
        parent_png_sha256=np.asarray(refs[0]["sha256"]),
        child_png_sha256=np.asarray(refs[1]["sha256"]),
        crop_binding_sha256=np.asarray("crop-binding"),
        parent_valid_to_child=inverse,
        child_valid_to_parent=selected,
        parent_valid_to_grid=original_grid,
    )
    frame = dict(
        parent_png=dict(refs[0], valid_points=3),
        png_sha256=refs[1]["sha256"],
        point_count=len(selected),
        crop_binding_sha256="crop-binding",
    )
    return frame, values


@pytest.mark.parametrize("empty", [False, True])
def test_derived_member_map_is_exact_encoded_subset_and_retains_empty_clouds(tmp_path, empty):
    frame, values = members_fixture(tmp_path, empty)
    np.savez_compressed(tmp_path / "members.npz", **values)
    frame["member_mapping"] = bytes_reference(tmp_path / "bound.npz", (tmp_path / "members.npz").read_bytes(), tmp_path)
    result = verify_fullraw_members(
        frame, tmp_path / "child.png", tmp_path, dict(point_width=4, point_height=1), _bound_file
    )
    assert result["encoded_coordinates_bit_exact"] and result["maps_to_full_pcap_returns"] is False
    assert result["child_valid_points"] == (0 if empty else 2)


@pytest.mark.parametrize(
    "change", ["scope", "crop_binding", "grid", "inverse", "reorder", "dtype", "encoded_coordinate"]
)
def test_member_lineage_rejects_rebinding_wrong_scope_or_changed_encoded_return(tmp_path, change):
    frame, values = members_fixture(tmp_path)
    if change == "scope":
        values["parent_scope"] = np.asarray("full_pcap_returns")
    if change == "crop_binding":
        values["crop_binding_sha256"] = np.asarray("different-crop")
    if change == "grid":
        values["parent_valid_to_grid"][1] = 2
    if change == "inverse":
        values["parent_valid_to_child"][1] = 0
    if change == "reorder":
        values["child_valid_to_parent"] = values["child_valid_to_parent"][::-1]
    if change == "dtype":
        values["child_valid_to_parent"] = values["child_valid_to_parent"].astype(np.int32)
    if change == "encoded_coordinate":
        image = cv2.imread(str(tmp_path / "child.png"), cv2.IMREAD_UNCHANGED)
        image[0, 0, 0] += 1
        ok, png = cv2.imencode(".png", image)
        assert ok
        changed = bytes_reference(tmp_path / "child.png", png.tobytes(), tmp_path)
        frame["png_sha256"] = changed["sha256"]
        values["child_png_sha256"] = np.asarray(changed["sha256"])
    np.savez_compressed(tmp_path / "members.npz", **values)
    frame["member_mapping"] = bytes_reference(tmp_path / "bound.npz", (tmp_path / "members.npz").read_bytes(), tmp_path)
    with pytest.raises(ValueError):
        verify_fullraw_members(
            frame, tmp_path / "child.png", tmp_path, dict(point_width=4, point_height=1), _bound_file
        )


def derived_evidence_fixture(root):
    original, clock = source_fixture(root)
    parent_index = bytes_reference(root / "parent-index.json", (root / "index.json").read_bytes(), root)
    child = deepcopy(original)
    child.update(
        source_kind="derived_native_roi_fixed24",
        parent_source_kind="original_fullraw_fixed24",
        parent_scope="fullraw_exported_clouds",
        maps_to_full_pcap_returns=False,
        crop_binding_sha256="crop-binding",
        parent_source_index_identity=deepcopy(parent_index),
        parent_inputs=dict(source_index=parent_index, source_manifest=deepcopy(original["source_manifest"])),
    )
    child["source_manifest"] = reference(root / "child-manifest.json", dict(frame_count=3, kind="derived"))
    for frame, parent in zip(child["frames"], original["frames"]):
        frame["crop_binding_sha256"] = "crop-binding"
        frame["parent_png"] = dict(
            file=parent["file"],
            sha256=parent["png_sha256"],
            bytes=parent["png_bytes"],
            valid_points=parent["point_count"],
        )
        frame["member_mapping"] = dict(
            bytes_reference(root / f"member-{frame['frame_index']}.npz", b"bound member fixture", root),
            schema="badminton-native-roi-member-map-v1",
            parent_valid_points=parent["point_count"],
            child_valid_points=frame["point_count"],
        )
    return child, clock


def test_derived_fullraw_proof_binds_original_parent_not_the_crop_manifest(tmp_path):
    child, clock = derived_evidence_fixture(tmp_path)
    result = verify_fullraw_evidence(child, tmp_path, tmp_path / "clock.json", clock)
    assert result["derived"] is True
    assert result["original_manifest"] == child["parent_inputs"]["source_manifest"]
    assert result["original_manifest"] != child["source_manifest"]


@pytest.mark.parametrize(
    "change", ["scope", "pcap_scope", "parent_identity", "parent_manifest", "mapped_rows", "parent_png", "member_count"]
)
def test_derived_fullraw_parent_lineage_cannot_be_rebound(tmp_path, change):
    child, clock = derived_evidence_fixture(tmp_path)
    if change == "scope":
        child["parent_scope"] = "already_lossy_native_roi"
    if change == "pcap_scope":
        child["maps_to_full_pcap_returns"] = True
    if change == "parent_identity":
        child["parent_source_index_identity"]["sha256"] = "different-parent"
    if change == "parent_manifest":
        child["parent_inputs"]["source_manifest"] = child["source_manifest"]
    if change == "mapped_rows":
        child["lidar_common_clock_row_indices"] = [0, 1, 3]
    if change == "parent_png":
        child["frames"][0]["parent_png"]["sha256"] = "different-png"
    if change == "member_count":
        child["frames"][0]["member_mapping"]["child_valid_points"] = 2
    with pytest.raises(ValueError):
        verify_fullraw_evidence(child, tmp_path, tmp_path / "clock.json", clock)
