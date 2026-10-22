from copy import deepcopy
import unittest

from hmr4d.lidar_mocap.formal_reuse import (
    GEOMETRY_FIELDS,
    validate_formal_geometry,
    validate_formal_source,
)


class FormalReuseTests(unittest.TestCase):
    def snapshots(self):
        session = dict(manifest_sha256="points", frame_count=2, frame_map_sha256="exposures",
                       source_clock_audit=dict(common_index_sha256="clock", tracks={"cam1": "a", "cam2": "b"}))
        old = {key: dict(value=key) for key in GEOMETRY_FIELDS}
        old.update(review_only=True, sessions={"source": session}, snapshot_digest="review",
                   configuration_files_sha256={"calibration.json": "review-camera"})
        new = deepcopy(old)
        new.pop("review_only")
        new.update(snapshot_digest="formal", configuration_files_sha256={"calibration.json": "formal-camera"})
        return old, new

    def source_indexes(self):
        old = dict(source_id="source", source_kind="derived_native_roi_fixed24", frame_count=2,
            fixed24=dict(scale=.001), frames=[dict(png_sha256="p0"), dict(png_sha256="p1")],
            source_manifest=dict(sha256="points"), common_clock=dict(sha256="clock"),
            frame_map=dict(sha256="exposures"), native_videos=dict(cam1="a", cam2="b"),
            canonical_video_sidecars=dict(cam1="as", cam2="bs"), candidate_bundle_sha256={"camera": "original"},
            coordinate_contract=dict(M_baked_into_stored_xyz=False, application_raw_to_viewer_M=[[1]],
                                     source_requested_M=[[1]]),
            crop_binding=dict(bbox_sha256="bbox", parent_source_index_sha256="parent"))
        new = deepcopy(old)
        new.update(camera_calibration_admitted=True, accepted_bundle_sha256={"camera": "admitted"},
                   camera_applicability=dict(file="actual-certificate", sha256="certificate"))
        new["coordinate_contract"]["physical_camera_applicability"] = "sampled_visual_accepted"
        return old, new

    def test_separate_admission_metadata_with_identical_physics_and_sources(self):
        old, new = self.snapshots()
        new["sessions"]["additional_formal_source"] = deepcopy(old["sessions"]["source"])
        validate_formal_geometry(old, new, ["source"])
        validate_formal_source(*self.source_indexes())

    def test_any_physical_camera_ground_timing_or_crop_change_blocks_reuse(self):
        old, new = self.snapshots()
        for field in GEOMETRY_FIELDS:
            changed = deepcopy(new)
            changed[field] = {"changed": True}
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "physical calibration input"):
                    validate_formal_geometry(old, changed, ["source"])

    def test_remaining_review_scope_or_changed_native_clock_blocks_reuse(self):
        old, new = self.snapshots()
        with self.assertRaisesRegex(ValueError, "separately admitted"):
            validate_formal_geometry(old, old, ["source"])
        new["sessions"]["source"]["source_clock_audit"]["tracks"]["cam2"] = "another-exposure-index"
        with self.assertRaisesRegex(ValueError, "native session contract"):
            validate_formal_geometry(old, new, ["source"])

    def test_modified_point_members_exposures_or_candidate_lineage_blocks_reuse(self):
        old, new = self.source_indexes()
        changes = [("frames", [dict(png_sha256="p0"), dict(png_sha256="other")]),
                   ("frame_map", dict(sha256="another-exposure")),
                   ("native_videos", dict(cam1="a", cam2="replacement")),
                   ("candidate_bundle_sha256", dict(camera="replacement"))]
        for key, value in changes:
            changed = deepcopy(new)
            changed[key] = value
            with self.subTest(field=key):
                with self.assertRaisesRegex(ValueError, "original source contract"):
                    validate_formal_source(old, changed)
        new["crop_binding"]["parent_source_index_sha256"] = "another-parent"
        with self.assertRaisesRegex(ValueError, "original crop lineage"):
            validate_formal_source(old, new)


if __name__ == "__main__":
    unittest.main()
