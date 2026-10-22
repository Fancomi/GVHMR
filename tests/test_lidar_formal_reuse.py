from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from hmr4d.lidar_mocap.formal_reuse import (
    GEOMETRY_FIELDS,
    validate_formal_geometry,
    validate_formal_source,
)
from hmr4d.lidar_mocap.relocations import load_pilot_run_relocations


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


class FormalRelocationTests(unittest.TestCase):
    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(value).encode() if isinstance(value, dict) else value
        path.write_bytes(blob)
        return dict(path=str(path), sha256=hashlib.sha256(blob).hexdigest())

    def fixture(self, root):
        data, run = root / "data", root / "formal"
        sid = "20261019_152542"
        target = data / "sessions" / sid / "meta/original.bin"
        target_binding = self.put(target, b"actual original")
        original = dict(file=f"upstream/2026-10-19/{sid}/original.bin",
                        sha256=target_binding["sha256"], bytes=target.stat().st_size)
        table = self.put(data / "relocations.json", dict(schema="badminton-data-relocations-v1", entries={
            original["file"]: dict(file=target.relative_to(data).as_posix(),
                                   sha256=original["sha256"], bytes=original["bytes"])}))
        relocation = dict(file="relocations.json", sha256=table["sha256"], bytes=(data / "relocations.json").stat().st_size)
        index = self.put(data / "admitted/source.json", {})
        bundle = {name: "a" * 64 for name in ("calibration.json", "bbox.json", "sources.json",
                                              "roi-config.txt", "sync-policy.json")}
        handoff = dict(schema="badminton-camera-admission-handoff-v1", data_root=str(data),
            camera_calibration_admitted=True, body_labels_production_admitted=False, training_published=False,
            accepted_bundle_sha256=bundle,
            source_indexes={sid: dict(file="admitted/source.json", sha256=index["sha256"], bytes=2)})
        handoff_binding = self.put(data / "camera-handoff.json", handoff)
        intake = dict(schema="badminton-formal-derived-intake-v1", execution_scope="all_effective_native_rallies",
            data_root=str(data), camera_calibration_admitted=True, body_labels_production_admitted=False,
            training_published=False, camera_admission_handoff=handoff_binding)
        intake_binding = self.put(run / "intake.json", intake)
        registry = dict(schema="badminton-formal-derived-source-registry-v1", data_root=str(data),
            data_relocations=relocation, intake=intake_binding, camera_admission_handoff=handoff_binding,
            sources={sid: dict(source_index=index)})
        self.put(run / "source_registry.json", registry)
        calibration = dict(sessions_root=str(data / "sessions"), data_relocations=relocation,
                           configuration_files_sha256=bundle, sessions={sid: {}})
        return run, registry, calibration, original

    def test_formal_explicit_mapping_resolves_deleted_history_through_bound_intake(self):
        with TemporaryDirectory() as directory:
            run, _, calibration, original = self.fixture(Path(directory))
            resolver = load_pilot_run_relocations(run, calibration)
            self.assertEqual(resolver.resolve(original).read_bytes(), b"actual original")
            self.assertFalse((resolver.root / original["file"]).exists())

    def test_untyped_or_changed_formal_source_cannot_enable_relocations(self):
        with TemporaryDirectory() as directory:
            run, registry, calibration, _ = self.fixture(Path(directory))
            changed = deepcopy(registry)
            changed["schema"] = "unrecognized-formal-intake"
            self.put(run / "source_registry.json", changed)
            with self.assertRaisesRegex(ValueError, "explicit frozen review pilot"):
                load_pilot_run_relocations(run, calibration)
            registry["sources"]["20261019_152542"]["source_index"]["sha256"] = "b" * 64
            self.put(run / "source_registry.json", registry)
            with self.assertRaisesRegex(ValueError, "source differs"):
                load_pilot_run_relocations(run, calibration)

    def test_changed_handoff_or_review_camera_snapshot_blocks_formal_mapping(self):
        with TemporaryDirectory() as directory:
            run, registry, calibration, _ = self.fixture(Path(directory))
            calibration["review_only"] = True
            with self.assertRaisesRegex(ValueError, "camera snapshot"):
                load_pilot_run_relocations(run, calibration)
            calibration.pop("review_only")
            Path(registry["camera_admission_handoff"]["path"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "Frozen formal admission input changed"):
                load_pilot_run_relocations(run, calibration)


if __name__ == "__main__":
    unittest.main()
