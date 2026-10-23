from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from hmr4d.lidar_mocap.expanded_dataset import binding, check_binding
from hmr4d.lidar_mocap.pipeline import write_json
from tools.audit_badminton_expanded_pixels import accepted_pixel, historical_corrections, pixel_requests


def row():
    native = dict(record_index=10, media_pts_us=100, common_time_us=1000, source_frame_no=123)
    view = dict(native=native, raw_image_sha256="old", report_path="report.json", report_sha256="report-sha")
    return dict(
        session_id="new",
        lidar_index=3,
        calibration_digest="new-cal",
        cameras={cam: deepcopy(native) for cam in ("cam1", "cam2")},
        dualcam=dict(views={cam: deepcopy(view) for cam in ("cam1", "cam2")}),
    )


class ExpandedPixelsTests(unittest.TestCase):
    def request(self):
        return pixel_requests([row()], {"new": dict(role="test", calibration_digest="new-cal")})["new", "cam1"][10]

    def correction(self):
        return dict(
            scene_id="new",
            camera="cam1",
            **self.request()["native"],
            previous_raw_bgr_sha256="old",
            verified_raw_bgr_sha256="verified",
        )

    def test_unchanged_pixels_require_no_supplement(self):
        self.assertIsNone(accepted_pixel(self.request(), "old", {}))

    def test_only_exact_native_bound_correction_is_accepted(self):
        correction = self.correction()
        self.assertEqual(accepted_pixel(self.request(), "verified", {("new", "cam1", 10): correction}), correction)
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            accepted_pixel(self.request(), "unknown", {("new", "cam1", 10): correction})

    def test_correction_cannot_move_to_another_native_pts(self):
        correction = self.correction()
        correction["media_pts_us"] += 1
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            accepted_pixel(self.request(), "verified", {("new", "cam1", 10): correction})

    def test_source_specific_calibration_and_role_are_required(self):
        for role, digest in (("train", "new-cal"), ("test", "old-cal")):
            with self.assertRaisesRegex(ValueError, "role or calibration"):
                pixel_requests([row()], {"new": dict(role=role, calibration_digest=digest)})

    def test_people_sharing_pixels_must_agree(self):
        a, b = row(), row()
        b["dualcam"]["views"]["cam1"]["raw_image_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "sharing one native"):
            pixel_requests([a, b], {"new": dict(role="test", calibration_digest="new-cal")})

    def test_paired_pts_and_observer_pts_cannot_disagree(self):
        a = row()
        a["cameras"]["cam1"]["media_pts_us"] += 1
        with self.assertRaisesRegex(ValueError, "paired native"):
            pixel_requests([a], {"new": dict(role="test", calibration_digest="new-cal")})


class HistoricalPixelsTests(unittest.TestCase):
    def fixture(self, root, *, junction=False):
        canonical = root / "canonical"
        canonical.mkdir()
        metadata = canonical / "metadata.jsonl"
        metadata.write_text('{"synthetic_metadata":true}\n', encoding="utf-8")
        original_path = metadata
        if junction:
            alias = root / "historical_LABEL"
            if os.name == "nt":
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(alias), str(canonical)],
                    check=True,
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                self.addCleanup(lambda: alias.rmdir() if alias.exists() else None)
            else:
                alias.symlink_to(canonical, target_is_directory=True)
                self.addCleanup(alias.unlink)
            original_path = alias / metadata.name
        original = dict(path=str(original_path), sha256=binding(metadata)["sha256"])
        evidence_path, report_path, training_path = root / "evidence.json", root / "report.json", root / "training.json"
        write_json(report_path, {})
        write_json(training_path, {})
        native = dict(record_index=10, media_pts_us=100, common_time_us=1000, source_frame_no=123)
        evidence = dict(
            metadata=original,
            all_person_admissions_unchanged=True,
            people=[dict(unchanged=True)],
            original_views=dict(cam1=dict(native_bgr_image_sha256="old", camera_native_source=native)),
            verified_views=dict(cam1=dict(native_bgr_image_sha256="verified", camera_native_source=native)),
        )
        write_json(evidence_path, evidence)
        item = dict(
            scene_id="old",
            camera="cam1",
            **native,
            previous_raw_bgr_sha256="old",
            verified_raw_bgr_sha256="verified",
            evidence=binding(evidence_path),
            original_report=binding(report_path),
        )
        correction = dict(
            schema="native-pixel-corrections/v1",
            status="complete",
            all_person_admissions_unchanged=True,
            metadata=original,
            training_manifest=binding(training_path),
            full_native_pixel_audit=binding(report_path),
            corrections=[item],
        )
        sidecar = root / "sidecar.json"
        write_json(sidecar, correction)
        assembly = dict(
            suggested_session_splits=dict(test=dict(components=[dict(metadata=binding(metadata))])),
            source_registry=dict(old=dict(generation="old", role="test")),
        )
        return sidecar, correction, assembly, metadata

    def test_existing_literal_metadata_binding_is_preserved(self):
        with TemporaryDirectory() as directory:
            sidecar, correction, assembly, _ = self.fixture(Path(directory))
            self.assertEqual(
                historical_corrections(sidecar, assembly), {("old", "cam1", 10): correction["corrections"][0]}
            )

    def test_real_directory_alias_requires_same_file_and_checks_current_component(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        sidecar, correction, assembly, metadata = self.fixture(Path(temporary.name), junction=True)
        before = sidecar.read_bytes()
        original = Path(correction["metadata"]["path"])
        self.assertNotEqual(str(original), str(metadata))
        self.assertEqual(original.resolve(strict=True), metadata.resolve(strict=True))
        self.assertTrue(original.samefile(metadata))
        with patch("tools.audit_badminton_expanded_pixels.check_binding", wraps=check_binding) as checked:
            self.assertEqual(
                historical_corrections(sidecar, assembly), {("old", "cam1", 10): correction["corrections"][0]}
            )
        current = assembly["suggested_session_splits"]["test"]["components"][0]["metadata"]
        self.assertIn(current, [call.args[0] for call in checked.call_args_list])
        self.assertNotIn(correction["metadata"], [call.args[0] for call in checked.call_args_list])
        self.assertEqual(sidecar.read_bytes(), before)

    def test_identical_bytes_at_another_file_are_not_a_path_alias(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar, correction, assembly, metadata = self.fixture(root)
            copy = root / "unrelated_metadata.jsonl"
            shutil.copyfile(metadata, copy)
            self.assertEqual(binding(copy)["sha256"], correction["metadata"]["sha256"])
            self.assertFalse(copy.samefile(metadata))
            correction["metadata"]["path"] = str(copy)
            write_json(sidecar, correction)
            with self.assertRaisesRegex(ValueError, "exact unchanged expansion component"):
                historical_corrections(sidecar, assembly)

    def test_missing_historical_or_current_path_cannot_be_replaced_by_its_digest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            sidecar, correction, assembly, metadata = self.fixture(root)
            correction["metadata"]["path"] = str(root / "missing_old_path.jsonl")
            write_json(sidecar, correction)
            with self.assertRaisesRegex(ValueError, "exact unchanged expansion component"):
                historical_corrections(sidecar, assembly)
            correction["metadata"]["path"] = str(metadata)
            write_json(sidecar, correction)
            assembly["suggested_session_splits"]["test"]["components"][0]["metadata"]["path"] = str(
                root / "missing_current.jsonl"
            )
            with self.assertRaisesRegex(ValueError, "exact unchanged expansion component"):
                historical_corrections(sidecar, assembly)

    def test_expected_digest_must_match_on_both_bindings_and_current_bytes(self):
        with TemporaryDirectory() as directory:
            sidecar, original, assembly, _ = self.fixture(Path(directory))
            for old_digest, current_digest in (
                ("0" * 64, original["metadata"]["sha256"]),
                (original["metadata"]["sha256"], "0" * 64),
                ("0" * 64, "0" * 64),
            ):
                correction, current = deepcopy(original), deepcopy(assembly)
                correction["metadata"]["sha256"] = old_digest
                current["suggested_session_splits"]["test"]["components"][0]["metadata"]["sha256"] = current_digest
                write_json(sidecar, correction)
                with self.subTest(old=old_digest, current=current_digest), self.assertRaises(ValueError):
                    historical_corrections(sidecar, current)

    def test_changed_current_component_bytes_are_rejected(self):
        with TemporaryDirectory() as directory:
            sidecar, _, assembly, metadata = self.fixture(Path(directory))
            metadata.write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen dependency"):
                historical_corrections(sidecar, assembly)

    def test_original_source_generation_and_test_role_remain_required(self):
        with TemporaryDirectory() as directory:
            sidecar, _, assembly, _ = self.fixture(Path(directory))
            for change in (dict(generation="new", role="test"), dict(generation="old", role="train")):
                current = deepcopy(assembly)
                current["source_registry"]["old"] = change
                with self.subTest(change=change), self.assertRaises(ValueError):
                    historical_corrections(sidecar, current)

    def test_sidecar_and_native_witness_must_preserve_every_person_admission(self):
        with TemporaryDirectory() as directory:
            sidecar, original, assembly, _ = self.fixture(Path(directory))
            correction = dict(original, all_person_admissions_unchanged=False)
            write_json(sidecar, correction)
            with self.assertRaisesRegex(ValueError, "admission-preserving"):
                historical_corrections(sidecar, assembly)
            evidence_path = Path(original["corrections"][0]["evidence"]["path"])
            evidence = json.loads(evidence_path.read_text())
            evidence["people"][0]["unchanged"] = False
            write_json(evidence_path, evidence)
            original["corrections"][0]["evidence"] = binding(evidence_path)
            write_json(sidecar, original)
            with self.assertRaisesRegex(ValueError, "full-person admission"):
                historical_corrections(sidecar, assembly)

    def test_native_time_identity_cannot_change_with_a_valid_path_alias(self):
        with TemporaryDirectory() as directory:
            sidecar, correction, assembly, _ = self.fixture(Path(directory))
            correction["corrections"][0]["media_pts_us"] += 1
            write_json(sidecar, correction)
            with self.assertRaisesRegex(ValueError, "another native image"):
                historical_corrections(sidecar, assembly)


if __name__ == "__main__":
    unittest.main()
