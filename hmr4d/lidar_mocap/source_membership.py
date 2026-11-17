"""Verify emitted-cloud crop members, including an explicit retired-parent proof."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import cv2
import numpy as np


def verify_fullraw_members(frame, child_path, data_root, grid, bound_file, *, relocations=None, retired_evidence=None):
    if retired_evidence is not None:
        return _verify_retired_members(frame, child_path, data_root, grid, bound_file,
                                       retired_evidence, relocations=relocations)
    mapping = frame["member_mapping"]
    options = dict(relocations=relocations) if relocations is not None else {}
    path = bound_file(data_root, mapping, **options)
    parent_path = bound_file(data_root, frame["parent_png"], **options)
    width, height = grid["point_width"], grid["point_height"]
    images = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in (parent_path, child_path)]
    if any(image is None or image.dtype != np.uint8 or image.shape != (height * 3, width, 3) for image in images):
        raise ValueError("Fullraw member verification requires the original RGB8 fixed24 grids")
    parent, child = [image.reshape(3, width * height, 3) for image in images]
    parent_grid, child_grid = [np.flatnonzero(np.any(image != 0, axis=-1).all(axis=0)) for image in (parent, child)]
    with np.load(path, allow_pickle=False) as data:
        expected = dict(
            schema="badminton-native-roi-member-map-v1",
            dtype="int64",
            parent_scope="fullraw_exported_clouds",
            point_index_space="decoded_fixed24_valid_returns_before_raw_to_viewer",
            grid_index_space="zero-based row-major cell in one coordinate band",
            parent_png_sha256=frame["parent_png"]["sha256"],
            child_png_sha256=frame["png_sha256"],
            crop_binding_sha256=frame["crop_binding_sha256"],
        )
        if any(data[name].shape != () or data[name].item() != value for name, value in expected.items()):
            raise ValueError("Fullraw member map scope or PNG/crop binding differs")
        parent_to_child = data["parent_valid_to_child"]
        selected = data["child_valid_to_parent"]
        original_grid = data["parent_valid_to_grid"]
        if (
            any(value.dtype != np.int64 or value.ndim != 1 for value in (parent_to_child, selected, original_grid))
            or len(parent_to_child) != len(parent_grid)
            or len(original_grid) != len(parent_grid)
            or len(selected) != len(child_grid)
            or len(child_grid) != frame["point_count"]
            or len(parent_grid) != frame["parent_png"]["valid_points"]
            or not np.array_equal(original_grid, parent_grid)
            or np.any((selected < 0) | (selected >= len(parent_grid)))
            or np.any(np.diff(selected) <= 0)
        ):
            raise ValueError("Fullraw member map does not cover actual emitted-cloud valid returns")
        inverse = np.full(len(parent_grid), -1, dtype=np.int64)
        inverse[selected] = np.arange(len(selected), dtype=np.int64)
        if (
            not np.array_equal(parent_to_child, inverse)
            or not np.array_equal(child_grid, parent_grid[selected])
            or not np.array_equal(child[:, child_grid], parent[:, parent_grid[selected]])
        ):
            raise ValueError("Derived crop is not an exact encoded-cell subset of its emitted fullraw PNG")
    return dict(
        member_mapping=mapping,
        parent_png=frame["parent_png"],
        parent_valid_points=len(parent_grid),
        child_valid_points=len(child_grid),
        encoded_coordinates_bit_exact=True,
        parent_scope="fullraw_exported_clouds",
        maps_to_full_pcap_returns=False,
    )


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _reference(data_root, entry, bound_file, relocations=None):
    _require(isinstance(entry, dict) and isinstance(entry.get("file"), str),
             "Retired-parent proof requires a file reference")
    _require(isinstance(entry.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is not None,
             "Retired-parent proof requires an explicit SHA256")
    _require(type(entry.get("bytes")) is int and entry["bytes"] >= 0,
             "Retired-parent proof requires an explicit byte length")
    # Existing immutable QA receipts use absolute DATA paths. Resolve only
    # their spelling, preserving digest/length and the resolver's byte checks.
    return bound_file(data_root, dict(entry, file=_root_name(data_root, entry["file"])), relocations=relocations)


def _document(data_root, entry, bound_file, relocations=None):
    path = _reference(data_root, entry, bound_file, relocations)
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate key in retired-parent evidence")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs)


def _same_reference(left, right):
    return (isinstance(left, dict) and isinstance(right, dict)
            and left.get("sha256") == right.get("sha256")
            and type(left.get("bytes")) is int and left.get("bytes") == right.get("bytes"))


def _root_name(data_root, value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (Path(data_root) / path).resolve()
    _require(path.is_relative_to(data_root), "Retired-parent evidence escapes DATA")
    return path.relative_to(data_root).as_posix()


def _collection_reference(collection, entry):
    name = entry.get("file") if isinstance(entry, dict) else None
    _require(isinstance(name, str) and name and "\\" not in name and ":" not in name
             and not name.startswith("/") and all(part not in ("", ".", "..") for part in name.split("/")),
             "Crop collection references must be relative POSIX paths")
    return dict(entry, file=(PurePosixPath(collection["file"]).parent / name).as_posix())


@dataclass(frozen=True)
class _RetiredParentEvidence:
    reference: dict
    index_header: bytes
    frame_hashes: tuple
    summary: dict


def verify_retired_parent_evidence(index, data_root, bound_file, *, relocations=None):
    """Bind a new read-only overlay to immutable crop and actual retirement evidence.

    This does not read or reconstruct the original parent coordinates. The
    complete original valid/grid identities remain independently checked when
    each current child is read; historical encoded checks retain their scope.
    """
    reference = index.get("retired_parent_payload")
    _require(index.get("source_kind") == "derived_native_roi_fixed24"
             and index.get("parent_source_kind") == "original_fullraw_fixed24",
             "Retired-parent opt-in is only for the declared derived fullraw reader")
    proof = _document(data_root, reference, bound_file, relocations)
    _require(proof.get("schema") == "badminton-retired-fullraw-parent-evidence-v1"
             and proof.get("source_id") == index.get("source_id")
             and proof.get("parent_payload_required_at_runtime") is False
             and proof.get("original_parent_xyz_reconstruction_claimed") is False,
             "Retired-parent proof source or access scope differs")
    base = _document(data_root, proof.get("base_source_index"), bound_file, relocations)
    overlay_base = {key: value for key, value in index.items() if key != "retired_parent_payload"}
    _require("retired_parent_payload" not in base and _canonical(base) == _canonical(overlay_base),
             "Retired-parent overlay changes its immutable source index")
    frames = index.get("frames", [])
    count = index.get("frame_count")
    _require(type(count) is int and count > 0 and len(frames) == count,
             "Retired-parent proof requires all original crop frames")

    collection_ref = proof.get("crop_collection")
    collection = _document(data_root, collection_ref, bound_file, relocations)
    retained_ref = proof.get("retained_crop_source_index")
    retained = _document(data_root, retained_ref, bound_file, relocations)
    _require(collection.get("schema") == "badminton-derived-native-roi-collection-v1"
             and collection.get("status") == "crop_complete_media_freeze_required"
             and collection.get("parent_scope") == "fullraw_exported_clouds"
             and collection.get("maps_to_full_pcap_returns") is False
             and collection.get("source_files_modified") is False,
             "Retired-parent proof requires a completed original encoded-subset crop")
    entries = [entry for entry in collection.get("sources", []) if entry.get("source_id") == index["source_id"]]
    _require(len(entries) == 1, "Retired-parent crop source is missing or ambiguous")
    source = entries[0]
    expected_retained_ref = _collection_reference(collection_ref, source.get("source_index"))
    _require(_same_reference(expected_retained_ref, retained_ref)
             and _root_name(data_root, expected_retained_ref["file"]) == _root_name(data_root, retained_ref["file"])
             and source.get("frame_count") == count
             and source.get("crop_binding_sha256") == index.get("crop_binding_sha256")
             and retained.get("source_id") == index["source_id"] and retained.get("frame_count") == count
             and retained.get("crop_binding_sha256") == index.get("crop_binding_sha256")
             and len(retained.get("frames", [])) == count,
             "Retired-parent completed crop/index binding differs")
    _require(_canonical(collection.get("candidate_bundle_sha256")) == _canonical(index.get("candidate_bundle_sha256")),
             "Retired-parent original crop bundle changed")
    approval_ref = _collection_reference(collection_ref, collection.get("approval"))
    approval = _document(data_root, approval_ref, bound_file, relocations)
    _require(approval.get("schema") == "badminton-native-roi-crop-approval-v1"
             and approval.get("status") == "user_approved"
             and approval.get("scope") == "derived_native_roi_crop"
             and _canonical(approval.get("candidate_bundle_sha256"))
             == _canonical(index.get("candidate_bundle_sha256")),
             "Retired-parent original crop approval differs")
    for name, entry in collection.get("applied_calibration", {}).items():
        _require(entry.get("sha256") == collection["candidate_bundle_sha256"].get(name),
                 "Retired-parent original applied bundle differs")
        _reference(data_root, _collection_reference(collection_ref, entry), bound_file, relocations)
    _require(set(collection.get("applied_calibration", {})) == set(index["candidate_bundle_sha256"]),
             "Retired-parent original crop bundle is incomplete")
    for frame, old in zip(frames, retained["frames"]):
        normalized = dict(old, file=_collection_reference(collection_ref, {"file": old["file"]})["file"],
                          member_mapping=_collection_reference(collection_ref, old["member_mapping"]))
        _require(_canonical(frame) == _canonical(normalized),
                 "Retained crop frame, member map or time metadata differs")
    _require(source.get("totals", {}).get("parent_valid_points") == sum(f["parent_png"]["valid_points"] for f in frames)
             and source.get("totals", {}).get("child_valid_points") == sum(f["point_count"] for f in frames),
             "Retired-parent crop original/retained totals differ")

    qa_ref = proof.get("full_sequence_validation")
    qa = _document(data_root, qa_ref, bound_file, relocations)
    _require(qa.get("schema") == "badminton-existing-crop-full-frame-verification-v1"
             and qa.get("source_id") == index["source_id"]
             and qa.get("status") == "full_frame_read_only_validation_complete"
             and qa.get("decision") == "reuse_exact_original_policy"
             and qa.get("reusable_without_resubset") is True
             and type(qa.get("original_frames_N")) is int and qa["original_frames_N"] == count
             and type(qa.get("verified_crop_frames_N")) is int and qa["verified_crop_frames_N"] == count,
             "Retired-parent full sequence validation is incomplete or belongs to another source")
    for flag in ("full_original_frame_coverage_passed", "full_original_valid_map_coverage_and_bijection_passed",
                 "actual_existing_png_and_npz_SHA256_and_bytes_passed", "fixed24_RGB8_800x2340_float32_decode_passed",
                 "original_M_float64_equal_to_final", "immutable_old_maps_retained"):
        _require(qa.get(flag) is True, "Retired-parent full sequence validation lacks " + flag)
    for flag in ("retired_parent_encoded_bytes_restored", "retired_parent_deleted_returns_recovered",
                 "inputs_modified"):
        _require(qa.get(flag) is False, "Retired-parent validation may not restore or change original inputs")
    for field, expected in (("old_source_index", retained_ref), ("old_source_manifest", index["source_manifest"]),
                            ("original_parent_index_snapshot", index["parent_inputs"]["source_index"]),
                            ("original_parent_manifest_snapshot", index["parent_inputs"]["source_manifest"]),
                            ("original_common_clock", index["common_clock"]), ("old_approval", approval_ref)):
        _require(_same_reference(qa.get(field), expected), "Retired-parent validation input binding differs: " + field)
    _require(qa.get("original_enriched_map_SHA256") == index["frame_map"]["sha256"],
             "Retired-parent validation frame map changed")
    trace_ref = qa.get("full_frame_validation_trace")
    trace_path = _reference(data_root, trace_ref, bound_file, relocations)
    _require(type(trace_ref.get("rows")) is int and trace_ref["rows"] == count,
             "Retired-parent full validation trace row count differs")
    with trace_path.open(encoding="utf-8-sig") as stream:
        for ordinal, frame in enumerate(frames):
            line = stream.readline()
            _require(bool(line.strip()), "Retired-parent trace is incomplete")
            row = json.loads(line)
            expected = dict(frame_index=frame["frame_index"], original_frame_id=frame["original_frame_id"],
                            frame_map_lidar_frame=frame.get("frame_map_lidar_frame", frame["frame_index"]),
                            original_valid_N=frame["parent_png"]["valid_points"],
                            retained_current_N=frame["point_count"],
                            retained_final_N=frame["point_count"], new_removed_N=0, PNG_SHA256=frame["png_sha256"],
                            NPZ_SHA256=frame["member_mapping"]["sha256"],
                            original_grid_int64le_SHA256=frame["parent_valid_to_grid_sha256"],
                            full_original_current_map_bijection_passed=True, encoded_layout_and_zeroed_grid_passed=True,
                            original_device_time_and_camera_pair_unchanged=True,
                            original_parent_recomputed_by_official_crop=(
                                qa["original_encoded_parent_byte_comparison_newly_executed"]))
            _require(_canonical({key: row.get(key) for key in expected}) == _canonical(expected),
                     "Retired-parent complete trace changes frame/grid/encoded/time evidence")
            _require(type(frame["frame_index"]) is int and frame["frame_index"] == ordinal,
                     "Retired-parent crop frame order changed")
        _require(not stream.read().strip(), "Retired-parent trace has extra frame rows")

    history = None
    state = proof.get("access_transition_state")
    if state == "historically_retired":
        _require(qa.get("original_encoded_parent_byte_comparison_newly_executed") is False
                 and type(qa.get("totals", {}).get("original_parent_pngs_revalidated")) is int
                 and qa["totals"]["original_parent_pngs_revalidated"] == 0,
                 "Historical retirement may not claim a new parent payload replay")
        binding = _document(data_root, proof.get("historical_retirement_binding"), bound_file, relocations)
        _require(binding.get("schema") == "badminton-canonical-crop-view-binding-v1"
                 and binding.get("source_id") == index["source_id"]
                 and binding.get("parent_payload_retained") is False
                 and binding.get("clock_and_coordinates_unchanged") is True
                 and binding.get("state") == "active_with_retired_parent_payload"
                 and binding.get("crop_binding_sha256") == index["crop_binding_sha256"]
                 and binding.get("retained_source_index", {}).get("sha256") == retained_ref["sha256"]
                 and binding.get("legacy_manifest", {}).get("sha256")
                 == index["parent_inputs"]["source_manifest"]["sha256"],
                 "Historical retirement binding differs from this source lineage")
        retired_ref = binding.get("retirement_receipt", {})
        retired_ref = dict(file=retired_ref.get("data_relative_path"), sha256=retired_ref.get("sha256"))
        # The historical protocol predates byte lengths; its bound digest is
        # still checked by the existing reader, without fabricating a size.
        retired_path = bound_file(data_root, retired_ref, relocations=relocations)
        retired = json.loads(retired_path.read_text(encoding="utf-8-sig"))
        entry = retired.get("sources", {}).get(index["source_id"], {})
        _require(retired.get("schema") == "badminton-lidar-parent-retirement-receipt-v1"
                 and retired.get("state") == "complete" and retired.get("verification", {}).get("state") == "passed"
                 and entry.get("parent_payload_retained") is False and entry.get("retired_parent_frames") == count
                 and entry.get("retained_crop_source_index_sha256") == retained_ref["sha256"]
                 and entry.get("original_parent_manifest_sha256")
                 == index["parent_inputs"]["source_manifest"]["sha256"],
                 "Actual historical retirement does not cover this source")
        historical = _document(data_root, proof.get("historical_encoded_evidence"), bound_file, relocations)
        entries = [entry for entry in historical.get("sources", []) if entry.get("source_id") == index["source_id"]]
        _require(historical.get("schema") == "badminton-crop-fullcontent-verification-v1"
                 and historical.get("status") == "passed" and len(entries) == 1,
                 "Historical encoded evidence does not identify this source")
        history = entries[0]
        _require(history.get("status") == "passed" and history.get("errors") == []
                 and history.get("frame_count") == count and history.get("crop_png_sha_verified") == count
                 and history.get("member_npz_sha_verified") == count
                 and history.get("npz_inverse_and_parent_grid_verified") == count
                 and history.get("frame_map_rows_verified") == count
                 and type(history.get("parent_png_all_content_replayed")) is bool,
                 "Historical child/map evidence is incomplete")
    elif state == "verified_ready_for_retirement":
        _require("historical_retirement_binding" not in proof and "historical_encoded_evidence" not in proof
                 and qa.get("original_encoded_parent_byte_comparison_newly_executed") is True
                 and type(qa.get("totals", {}).get("original_parent_pngs_revalidated")) is int
                 and qa["totals"]["original_parent_pngs_revalidated"] == count,
                 "Prospective retirement requires the actual complete current parent comparison")
    else:
        raise ValueError("Unsupported retired-parent access transition")
    summary = dict(reference=dict(reference), source_id=index["source_id"], access_transition_state=state,
                   base_source_index=proof["base_source_index"], crop_collection=collection_ref,
                   retained_crop_source_index=retained_ref, full_sequence_validation=qa_ref,
                   parent_payload_required_at_runtime=False, parent_payload_reopened=False,
                   original_parent_xyz_reconstruction_claimed=False,
                   original_encoded_parent_byte_comparison_newly_executed=(
                       qa["original_encoded_parent_byte_comparison_newly_executed"]),
                   historical_parent_encoded_evidence=history)
    return _RetiredParentEvidence(dict(reference), _index_header(index),
                                  tuple(hashlib.sha256(_canonical(frame)).hexdigest() for frame in frames), summary)


def _index_header(index):
    return _canonical({key: value for key, value in index.items() if key != "frames"})


def verify_retired_index_binding(index, evidence):
    """Reject changing source, clock, geometry, admission or proof after loading."""
    _require(isinstance(evidence, _RetiredParentEvidence) and _index_header(index) == evidence.index_header,
             "Retired-parent source index metadata changed after verification")


def _verify_retired_members(frame, child_path, data_root, grid, bound_file, evidence, *, relocations=None):
    _require(isinstance(evidence, _RetiredParentEvidence), "A verified retired-parent context is required")
    ordinal = frame.get("frame_index")
    _require(type(ordinal) is int and 0 <= ordinal < len(evidence.frame_hashes)
             and hashlib.sha256(_canonical(frame)).hexdigest() == evidence.frame_hashes[ordinal],
             "Retired-parent frame no longer matches its immutable base index")
    _reference(data_root, evidence.reference, bound_file, relocations)
    expected_child = dict(file=frame["file"], sha256=frame["png_sha256"], bytes=frame["png_bytes"])
    actual_child = _reference(data_root, expected_child, bound_file, relocations)
    _require(Path(child_path).resolve() == actual_child.resolve(), "Retired-parent child path differs")
    map_path = _reference(data_root, frame["member_mapping"], bound_file, relocations)
    width, height = grid["point_width"], grid["point_height"]
    image = cv2.imread(str(actual_child), cv2.IMREAD_UNCHANGED)
    _require(image is not None and image.dtype == np.uint8 and image.shape == (height * 3, width, 3),
             "Retired-parent member verification requires the original RGB8 fixed24 child grid")
    child = image.reshape(3, width * height, 3)
    child_grid = np.flatnonzero(np.any(child != 0, axis=-1).all(axis=0))
    with np.load(map_path, allow_pickle=False) as data:
        expected = dict(schema="badminton-native-roi-member-map-v1", dtype="int64",
                        parent_scope="fullraw_exported_clouds",
                        point_index_space="decoded_fixed24_valid_returns_before_raw_to_viewer",
                        grid_index_space="zero-based row-major cell in one coordinate band",
                        parent_png_sha256=frame["parent_png"]["sha256"], child_png_sha256=frame["png_sha256"],
                        crop_binding_sha256=frame["crop_binding_sha256"])
        _require(all(data[name].shape == () and data[name].item() == value for name, value in expected.items()),
                 "Retired-parent member map scope or original PNG binding differs")
        inverse, selected, original_grid = (data[name] for name in
                                           ("parent_valid_to_child", "child_valid_to_parent", "parent_valid_to_grid"))
        count = frame["parent_png"]["valid_points"]
        _require(type(count) is int and count >= 0
                 and all(value.dtype == np.int64 and value.ndim == 1 for value in (inverse, selected, original_grid))
                 and len(inverse) == len(original_grid) == count
                 and len(selected) == len(child_grid) == frame["point_count"]
                 and not np.any((original_grid < 0) | (original_grid >= width * height))
                 and not np.any(np.diff(original_grid) <= 0)
                 and not np.any((selected < 0) | (selected >= count)) and not np.any(np.diff(selected) <= 0)
                 and hashlib.sha256(original_grid.astype("<i8", copy=False).tobytes()).hexdigest()
                 == frame["parent_valid_to_grid_sha256"],
                 "Retired-parent map does not preserve the complete original valid/grid identities")
        expected_inverse = np.full(count, -1, dtype=np.int64)
        expected_inverse[selected] = np.arange(len(selected), dtype=np.int64)
        _require(np.array_equal(inverse, expected_inverse) and np.array_equal(child_grid, original_grid[selected]),
                 "Retired-parent crop has invalid inverse or retained original grid membership")
    return dict(member_mapping=frame["member_mapping"], parent_png=frame["parent_png"],
                parent_valid_points=count, child_valid_points=len(child_grid),
                parent_payload_reopened=False, original_parent_xyz_reconstructed=False,
                current_child_png_hash_verified=True, current_complete_original_grid_and_inverse_verified=True,
                original_parent_encoded_equality_replayed_this_read=False,
                historical_crop_construction="completed_exact_encoded_subset_crop",
                retired_parent_payload=evidence.summary, parent_scope="fullraw_exported_clouds",
                maps_to_full_pcap_returns=False)
