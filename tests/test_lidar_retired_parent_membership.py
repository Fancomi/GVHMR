"""An explicit, byte-bound retirement overlay never weakens the legacy reader."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from hmr4d.lidar_mocap.source_index import SourceIndexedSession, _bound_file, verify_fullraw_evidence
from hmr4d.lidar_mocap.source_membership import verify_fullraw_members, verify_retired_parent_evidence


def sha(data):
    return hashlib.sha256(data).hexdigest()


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
    path.write_bytes(data)
    return dict(file=name, sha256=sha(data), bytes=len(data))


class RetirementFixture:
    def __init__(self, root, state="verified_ready_for_retirement", empty=False, map_change=None):
        self.root, self.state = root, state
        self.sid = "20261020_191011"
        self.grid = dict(point_width=4, point_height=1)
        self.parent = np.zeros((3, 4, 3), np.uint8)
        self.parent[:, [0, 1, 3], 2] = 4
        self.parent[:, [0, 1, 3], 1] = [[10, 11, 12], [20, 21, 22], [30, 31, 32]]
        self.original_grid = np.asarray([0, 1, 3], np.int64)
        selected = np.asarray([] if empty else [0, 2], np.int64)
        self.child = np.zeros_like(self.parent)
        self.child[:, self.original_grid[selected]] = self.parent[:, self.original_grid[selected]]
        parent_ref = put(root, "parent.png", cv2.imencode(".png", self.parent)[1].tobytes())
        child_ref = put(root, "crop/child.png", cv2.imencode(".png", self.child)[1].tobytes())
        inverse = np.full(3, -1, np.int64)
        inverse[selected] = np.arange(len(selected))
        self.map_values = dict(
            schema=np.asarray("badminton-native-roi-member-map-v1"), dtype=np.asarray("int64"),
            parent_scope=np.asarray("fullraw_exported_clouds"),
            point_index_space=np.asarray("decoded_fixed24_valid_returns_before_raw_to_viewer"),
            grid_index_space=np.asarray("zero-based row-major cell in one coordinate band"),
            parent_png_sha256=np.asarray(parent_ref["sha256"]), child_png_sha256=np.asarray(child_ref["sha256"]),
            crop_binding_sha256=np.asarray("crop-binding"), parent_valid_to_child=inverse,
            child_valid_to_parent=selected, parent_valid_to_grid=self.original_grid.copy(),
        )
        if map_change == "inverse":
            self.map_values["parent_valid_to_child"][1] = 0
        if map_change == "grid":
            self.map_values["parent_valid_to_grid"][1] = 2
        if map_change == "dtype":
            self.map_values["parent_valid_to_grid"] = self.original_grid.astype(np.int32)
        if map_change == "selected":
            self.map_values["child_valid_to_parent"] = np.asarray([0, 1], np.int64)
        if map_change == "parent_sha":
            self.map_values["parent_png_sha256"] = np.asarray("0" * 64)
        map_ref = self.write_map()
        frame = dict(file=child_ref["file"], frame_index=0, original_frame_id=17,
                     png_sha256=child_ref["sha256"], png_bytes=child_ref["bytes"], point_count=len(selected),
                     parent_png=dict(parent_ref, valid_points=3), member_mapping=map_ref,
                     parent_valid_to_grid_sha256=sha(self.original_grid.astype("<i8").tobytes()),
                     parent_valid_to_grid_dtype="int64_little_endian", crop_binding_sha256="crop-binding",
                     device_first_us=100, device_last_us=200, scan_mid_us=150,
                     cameras=dict(cam1=dict(record_index=1, video_common_us=150),
                                  cam2=dict(record_index=2, video_common_us=150)))
        manifests = {k: put(root, k + ".json", dict(name=k))
                     for k in ("parent-index", "parent-manifest", "child-manifest", "clock", "frame-map")}
        self.base = dict(schema="badminton-source-index-v1", source_id=self.sid,
                         source_kind="derived_native_roi_fixed24", parent_source_kind="original_fullraw_fixed24",
                         parent_scope="fullraw_exported_clouds", maps_to_full_pcap_returns=False,
                         frame_count=1, frames=[frame], crop_binding_sha256="crop-binding",
                         source_manifest=manifests["child-manifest"], common_clock=manifests["clock"],
                         frame_map=manifests["frame-map"], parent_inputs=dict(source_index=manifests["parent-index"],
                         source_manifest=manifests["parent-manifest"]), coordinate_contract=dict(M=np.eye(3).tolist()),
                         candidate_bundle_sha256={})
        applied = {}
        for name in ("M", "K1", "K2", "E1", "E2"):
            ref = put(root, "crop/" + name + ".json", dict(name=name, values=np.eye(3).tolist()))
            self.base["candidate_bundle_sha256"][name] = ref["sha256"]
            applied[name] = dict(ref, file=Path(ref["file"]).name)
        self.approval = dict(schema="badminton-native-roi-crop-approval-v1", status="user_approved",
                             scope="derived_native_roi_crop",
                             candidate_bundle_sha256=self.base["candidate_bundle_sha256"])
        approval_ref = put(root, "crop/approval.json", self.approval)
        retained = deepcopy(self.base)
        retained["frames"][0]["file"] = "child.png"
        retained["frames"][0]["member_mapping"]["file"] = "members.npz"
        retained_ref = put(root, "crop/index.json", retained)
        collection = dict(schema="badminton-derived-native-roi-collection-v1",
                          status="crop_complete_media_freeze_required", parent_scope="fullraw_exported_clouds",
                          maps_to_full_pcap_returns=False, source_files_modified=False,
                          candidate_bundle_sha256=self.base["candidate_bundle_sha256"], applied_calibration=applied,
                          approval=dict(approval_ref, file="approval.json"), sources=[dict(source_id=self.sid,
                          source_index=dict(retained_ref, file="index.json"), frame_count=1,
                          crop_binding_sha256="crop-binding", totals=dict(parent_valid_points=3,
                          child_valid_points=len(selected)))])
        collection_ref = put(root, "crop/collection.json", collection)
        compared = state == "verified_ready_for_retirement"
        trace = dict(frame_index=0, original_frame_id=17, frame_map_lidar_frame=0, original_valid_N=3,
                     retained_current_N=len(selected), retained_final_N=len(selected), new_removed_N=0,
                     PNG_SHA256=child_ref["sha256"], NPZ_SHA256=map_ref["sha256"],
                     original_grid_int64le_SHA256=frame["parent_valid_to_grid_sha256"],
                     full_original_current_map_bijection_passed=True, encoded_layout_and_zeroed_grid_passed=True,
                     original_device_time_and_camera_pair_unchanged=True,
                     original_parent_recomputed_by_official_crop=compared)
        self.trace = trace
        trace_ref = put(root, "trace.jsonl", (json.dumps(trace) + "\n").encode())
        self.qa = dict(schema="badminton-existing-crop-full-frame-verification-v1", source_id=self.sid,
                       status="full_frame_read_only_validation_complete", decision="reuse_exact_original_policy",
                       reusable_without_resubset=True, original_frames_N=1, verified_crop_frames_N=1,
                       old_source_index=retained_ref, old_source_manifest=self.base["source_manifest"],
                       original_parent_index_snapshot=self.base["parent_inputs"]["source_index"],
                       original_parent_manifest_snapshot=self.base["parent_inputs"]["source_manifest"],
                       original_common_clock=self.base["common_clock"], old_approval=approval_ref,
                       original_enriched_map_SHA256=self.base["frame_map"]["sha256"],
                       full_frame_validation_trace=dict(trace_ref, rows=1),
                       original_encoded_parent_byte_comparison_newly_executed=compared,
                       totals=dict(original_parent_pngs_revalidated=int(compared)))
        for name in ("full_original_frame_coverage_passed", "full_original_valid_map_coverage_and_bijection_passed",
                     "actual_existing_png_and_npz_SHA256_and_bytes_passed",
                     "fixed24_RGB8_800x2340_float32_decode_passed",
                     "original_M_float64_equal_to_final", "immutable_old_maps_retained"):
            self.qa[name] = True
        for name in ("retired_parent_encoded_bytes_restored", "retired_parent_deleted_returns_recovered",
                     "inputs_modified"):
            self.qa[name] = False
        self.proof = dict(schema="badminton-retired-fullraw-parent-evidence-v1", source_id=self.sid,
                          base_source_index=put(root, "base.json", self.base), retained_crop_source_index=retained_ref,
                          crop_collection=collection_ref, full_sequence_validation=put(root, "qa.json", self.qa),
                          parent_payload_required_at_runtime=False, original_parent_xyz_reconstruction_claimed=False,
                          access_transition_state=state)
        if state == "historically_retired":
            retired = dict(schema="badminton-lidar-parent-retirement-receipt-v1", state="complete",
                           verification=dict(state="passed"), sources={self.sid: dict(parent_payload_retained=False,
                           retired_parent_frames=1, retained_crop_source_index_sha256=retained_ref["sha256"],
                           original_parent_manifest_sha256=self.base["parent_inputs"]["source_manifest"]["sha256"])})
            ref = put(root, "retirement.json", retired)
            self.binding = dict(schema="badminton-canonical-crop-view-binding-v1", source_id=self.sid,
                                parent_payload_retained=False, clock_and_coordinates_unchanged=True,
                                state="active_with_retired_parent_payload", crop_binding_sha256="crop-binding",
                                retained_source_index=retained_ref,
                                legacy_manifest=self.base["parent_inputs"]["source_manifest"],
                                retirement_receipt=dict(data_relative_path=ref["file"], sha256=ref["sha256"]))
            self.history = dict(schema="badminton-crop-fullcontent-verification-v1", status="passed", sources=[dict(
                source_id=self.sid, status="passed", errors=[], frame_count=1, crop_png_sha_verified=1,
                member_npz_sha_verified=1, npz_inverse_and_parent_grid_verified=1,
                frame_map_rows_verified=1,
                parent_png_all_content_replayed=False,
                prior_native_pixel_sample_checks_preserved="21 samples across seven sources")])
            self.proof["historical_retirement_binding"] = put(root, "binding.json", self.binding)
            self.proof["historical_encoded_evidence"] = put(root, "history.json", self.history)
        self.index = deepcopy(self.base)
        self.publish_proof()

    def write_map(self):
        path = self.root / "crop/members.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.map_values)
        return put(self.root, "crop/members.npz", path.read_bytes())

    def publish_proof(self):
        self.index["retired_parent_payload"] = put(self.root, "proof.json", self.proof)

    def context(self):
        return verify_retired_parent_evidence(self.index, self.root, _bound_file)

    def read(self, context=None, frame=None):
        session = SourceIndexedSession()
        session.source_index = dict(self.index, frames=[frame or self.index["frames"][0]])
        session.data_root, session.data_relocations = self.root, None
        session.fullraw_evidence = dict(derived=True)
        session._retired_parent_context, session.member_mapping_verifications = context, {}
        session.manifest, session.Q = self.grid, np.eye(3)
        pair = dict(lidar_index=0, lidar_path=self.root / "crop/child.png",
                    kept_points=self.base["frames"][0]["point_count"])
        points = session.read_cloud(pair, viewer=False)
        return points, session.member_mapping_verifications[0]


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("state", ["verified_ready_for_retirement", "historically_retired"])
def test_retirement_keeps_current_points_and_original_ids_without_reopening_parent(tmp_path, empty, state):
    fixture = RetirementFixture(tmp_path, state, empty)
    before, strict = fixture.read()
    assert strict["encoded_coordinates_bit_exact"] is True
    context = fixture.context()
    (tmp_path / "parent.png").unlink()  # Only this generated test copy is retired.
    with pytest.raises(FileNotFoundError):
        fixture.read()
    after, result = fixture.read(context)
    assert np.array_equal(before.view(np.uint32), after.view(np.uint32))
    assert result["parent_payload_reopened"] is False
    assert result["current_child_png_hash_verified"] is True
    assert result["current_complete_original_grid_and_inverse_verified"] is True
    assert "encoded_coordinates_bit_exact" not in result
    assert result["original_parent_xyz_reconstructed"] is False
    if state == "historically_retired":
        historical = result["retired_parent_payload"]["historical_parent_encoded_evidence"]
        assert historical["parent_png_all_content_replayed"] is False
        assert historical["prior_native_pixel_sample_checks_preserved"] == "21 samples across seven sources"


@pytest.mark.parametrize("field", ["scan_mid_us", "original_frame_id", "cameras", "point_count"])
def test_overlay_must_preserve_every_old_frame_field(tmp_path, field):
    fixture = RetirementFixture(tmp_path)
    fixture.index["frames"][0][field] = None
    with pytest.raises(ValueError, match="immutable source index"):
        fixture.context()


@pytest.mark.parametrize("change", ["base_sha", "base_type", "geometry", "source", "runtime_parent", "restore_xyz",
                                    "qa_count", "qa_parent_count", "qa_manifest", "trace_time", "trace_map",
                                    "proof_sha"])
def test_proof_rejects_source_geometry_time_and_binding_changes(tmp_path, change):
    fixture = RetirementFixture(tmp_path)
    if change == "base_sha":
        fixture.proof["base_source_index"]["sha256"] = "0" * 64
    if change == "base_type":
        fixture.index["frame_count"] = 1.0
    if change == "geometry":
        fixture.index["coordinate_contract"]["M"][0][0] = 2.0
    if change == "source":
        fixture.proof["source_id"] = "20261020_200346"
    if change == "runtime_parent":
        fixture.proof["parent_payload_required_at_runtime"] = True
    if change == "restore_xyz":
        fixture.proof["original_parent_xyz_reconstruction_claimed"] = True
    if change.startswith("qa_"):
        if change == "qa_count":
            fixture.qa["verified_crop_frames_N"] = 0
        elif change == "qa_parent_count":
            fixture.qa["totals"]["original_parent_pngs_revalidated"] = 0
        else:
            fixture.qa["original_parent_manifest_snapshot"] = fixture.base["common_clock"]
        fixture.proof["full_sequence_validation"] = put(tmp_path, "qa.json", fixture.qa)
    if change.startswith("trace_"):
        fixture.trace[
            "original_device_time_and_camera_pair_unchanged" if change == "trace_time" else "NPZ_SHA256"] = False
        fixture.qa["full_frame_validation_trace"] = dict(
            put(tmp_path, "trace.jsonl", (json.dumps(fixture.trace) + "\n").encode()), rows=1)
        fixture.proof["full_sequence_validation"] = put(tmp_path, "qa.json", fixture.qa)
    fixture.publish_proof()
    if change == "proof_sha":
        fixture.index["retired_parent_payload"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        fixture.context()


@pytest.mark.parametrize("change", ["proof", "child", "map", "frame", "grid", "inverse", "dtype", "selected"])
def test_runtime_never_accepts_changed_current_payload_or_identity(tmp_path, change):
    fixture = RetirementFixture(tmp_path)
    context = fixture.context()
    frame = deepcopy(fixture.index["frames"][0])
    if change in ("proof", "child", "map"):
        path = tmp_path / {"proof": "proof.json", "child": "crop/child.png", "map": "crop/members.npz"}[change]
        path.write_bytes(path.read_bytes() + b"changed")
    elif change == "frame":
        frame["scan_mid_us"] += 1
    else:
        if change == "grid":
            fixture.map_values["parent_valid_to_grid"][1] = 2
        if change == "inverse":
            fixture.map_values["parent_valid_to_child"][1] = 0
        if change == "dtype":
            fixture.map_values["parent_valid_to_grid"] = fixture.original_grid.astype(np.int32)
        if change == "selected":
            fixture.map_values["child_valid_to_parent"] = np.asarray([0, 1], np.int64)
        fixture.write_map()
    with pytest.raises(ValueError):
        fixture.read(context, frame)


@pytest.mark.parametrize("change", ["inverse", "grid", "dtype", "selected", "parent_sha"])
def test_even_hash_bound_maps_must_preserve_original_grid_and_inverse(tmp_path, change):
    fixture = RetirementFixture(tmp_path, map_change=change)
    context = fixture.context()
    with pytest.raises(ValueError, match="map|inverse|binding"):
        fixture.read(context)


@pytest.mark.parametrize("change", ["source", "geometry", "clock", "proof", "admission"])
def test_verified_context_cannot_be_reused_after_index_metadata_changes(tmp_path, change):
    fixture = RetirementFixture(tmp_path)
    context = fixture.context()
    if change == "source":
        fixture.index["source_id"] = "another-source"
    if change == "geometry":
        fixture.index["coordinate_contract"]["M"][0][0] = 2.0
    if change == "clock":
        fixture.index["common_clock"] = fixture.base["source_manifest"]
    if change == "proof":
        fixture.index["retired_parent_payload"].update(put(tmp_path, "other-proof.json", fixture.qa))
    if change == "admission":
        fixture.index["production_admitted"] = True
    with pytest.raises(ValueError, match="metadata changed"):
        fixture.read(context)


@pytest.mark.parametrize("change", ["different_retired_source", "fake_new_parent_replay", "not_retired",
                                    "history_missing_source"])
def test_historical_receipt_must_actually_cover_the_source(tmp_path, change):
    fixture = RetirementFixture(tmp_path, "historically_retired")
    if change == "different_retired_source":
        fixture.binding["source_id"] = "another-source"
    if change == "not_retired":
        fixture.binding["parent_payload_retained"] = True
    if change in ("different_retired_source", "not_retired"):
        fixture.proof["historical_retirement_binding"] = put(tmp_path, "binding.json", fixture.binding)
    if change == "fake_new_parent_replay":
        fixture.qa["original_encoded_parent_byte_comparison_newly_executed"] = True
        fixture.proof["full_sequence_validation"] = put(tmp_path, "qa.json", fixture.qa)
    if change == "history_missing_source":
        fixture.history["sources"] = []
        fixture.proof["historical_encoded_evidence"] = put(tmp_path, "history.json", fixture.history)
    fixture.publish_proof()
    with pytest.raises(ValueError):
        fixture.context()


def test_opt_in_cannot_attach_to_other_parent_or_direct_source_kind(tmp_path):
    fixture = RetirementFixture(tmp_path)
    fixture.index["parent_source_kind"] = "canonical_fullraw_pcap_fixed24_export"
    with pytest.raises(ValueError, match="derived original-fullraw"):
        verify_fullraw_evidence(fixture.index, tmp_path, None, {})
    with pytest.raises(ValueError, match="declared derived fullraw"):
        fixture.context()
