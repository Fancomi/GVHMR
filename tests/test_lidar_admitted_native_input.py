"""Synthetic accepted-parent/review-scope contracts; no production data or models."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hmr4d.lidar_mocap import admitted_native_input as module
from hmr4d.lidar_mocap import calibration
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.pilot import validate_full_native_spec
from test_lidar_calibration import SnapshotFixture, write_json


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def reseal(path, data):
    """Make mutations self-consistent so a stale digest cannot mask a weak guard."""
    data = deepcopy(data)
    data.pop("snapshot_digest", None)
    data["snapshot_digest"] = calibration._sha(calibration._canonical(data))
    write_json(path, data)


class AdmittedNativeInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gvhmr-admitted-review-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = SnapshotFixture(self.root)
        self.workspace = self.fixture.workspace
        workspace = patch.object(calibration, "_WORKSPACE_ROOT", self.workspace)
        workspace.start()
        self.addCleanup(workspace.stop)
        self.fixture.sources.update(camera_calibration_admitted=True, body_labels_production_admitted=False)
        write_json(self.fixture.base / "sources.json", self.fixture.sources)
        self.parent = self.fixture.freeze()
        self.parent_path = self.parent.snapshot_json
        self.parent_bytes = self.parent_path.read_bytes()
        self.sid = self.fixture.sid
        self.hashes = deepcopy(self.parent.data["configuration_files_sha256"])
        self.authority_path = self.root / "authority.json"
        self.authority = dict(
            schema="badminton-calibration-review-status-v1", baseline="synthetic R11",
            usage="review_only", camera_pilot_authorized=True, body_labels_production_admitted=False,
            training_published=False, result_files=self.hashes, user_confirmation="Review the synthetic native range",
        )
        write_json(self.authority_path, self.authority)
        coverage = self.root / "coverage.json"
        write_json(coverage, dict(schema="badminton-native-rally-coverage-audit-v1", assigned_unique_scans=3))
        evidence = self.root / "native-evidence.json"
        write_json(evidence, dict(scope="synthetic fixture only; no native execution claimed"))
        self.manifest_path = self.root / "native-manifest.json"
        self.spec = dict(
            schema="badminton-native-rally-fit-v1", review_only=True, training_published=False,
            camera_label="synthetic R11", data_root=str(self.root / "data"),
            lidar_project_root=str(self.fixture.project), calibration_dir=str(self.fixture.base),
            bundle_sha256=self.hashes, review_authorization=binding(self.authority_path),
            execution_scope="all_effective_native_rallies", user_confirmation="Synthetic test scope",
            coverage_audit=binding(coverage),
            source_indexes={self.sid: dict(file="indexes/synthetic.json", sha256="a" * 64)},
            clips=[dict(source_id=self.sid, rally_index=1, segment_index=0,
                        begin_common_us=99000, end_common_us=301001,
                        eligible_paired_png_indices=[0, 1, 2], selection_status="native_rally_confirmed",
                        evidence_files=[binding(evidence)])],
        )
        write_json(self.manifest_path, self.spec)
        scope = validate_full_native_spec(self.spec)
        self.proof_path = self.root / "accepted-freeze-proof.json"
        self.proof = dict(
            schema="badminton-1022-strict-accepted-freeze-proof-v2",
            status="strict_accepted_freeze_completed_review_prepare_not_invoked", actual_exit_code=0,
            accepted_snapshot=binding(self.parent_path), manifest=binding(self.manifest_path),
            accepted_bundle_sha256=self.hashes,
            source_rows=sum(row["frame_count"] for row in self.parent.data["sessions"].values()),
            native_rallies=len(self.spec["clips"]), selected_scans=scope["selected_scans"],
            selected_seconds=scope["selected_seconds"], model_loaded=False, gpu_started=False,
            body_labels_production_admitted=False, training_published=False,
        )
        write_json(self.proof_path, self.proof)
        self.serial = 0

    def derive(self):
        self.serial += 1
        return module.derive_review_snapshot(
            self.parent_path, self.manifest_path, self.workspace / f"review-{self.serial}",
            accepted_freeze_proof=binding(self.proof_path),
        )

    def verify(self, child):
        return module.verify_review_snapshot(
            self.parent_path, child.snapshot_json, self.manifest_path,
            accepted_freeze_proof=binding(self.proof_path),
        )

    def rebind_manifest(self):
        write_json(self.manifest_path, self.spec)
        self.proof["manifest"] = binding(self.manifest_path)
        write_json(self.proof_path, self.proof)

    def test_derivation_preserves_parent_geometry_and_artifact_bytes(self):
        child = self.derive()
        self.assertIsInstance(child, CalibrationSnapshot)
        self.assertEqual(self.parent_path.read_bytes(), self.parent_bytes)
        self.assertIs(child.data["review_only"], True)
        self.assertIs(child.data["production_admitted"], False)
        self.assertIs(child.data["training_published"], False)
        self.assertEqual(child.data["review_source_ids"], [self.sid])
        self.assertEqual(child.data["declared_calibration_source_ids"], [self.sid])
        mutable = {"review_only", "review_source_ids", "declared_calibration_source_ids",
                   "production_admitted", "training_published", "admitted_review_derivation", "snapshot_digest"}
        before = {key: value for key, value in self.parent.data.items() if key not in mutable}
        after = {key: value for key, value in child.data.items() if key not in mutable}
        self.assertEqual(calibration._canonical(before), calibration._canonical(after))
        self.assertNotEqual(self.parent.data["snapshot_digest"], child.data["snapshot_digest"])
        lineage = child.data["admitted_review_derivation"]
        self.assertEqual(lineage["schema"], "badminton-admitted-review-calibration-derivation-v1")
        for name, expected in (("parent_accepted_snapshot", binding(self.parent_path)),
                               ("full_native_manifest", binding(self.manifest_path)),
                               ("review_authorization", binding(self.authority_path)),
                               ("accepted_freeze_proof", binding(self.proof_path))):
            for key, value in expected.items():
                self.assertEqual(lineage[name][key], value)
        for artifact in self.parent.data["artifacts"].values():
            self.assertEqual((self.parent_path.parent / artifact["path"]).read_bytes(),
                             (child.snapshot_json.parent / artifact["path"]).read_bytes())
        self.assertEqual(self.verify(child)["status"], "review_snapshot_verified_metadata_only")

    def test_rejects_resigned_immutable_geometry_clock_session_and_artifact_metadata(self):
        mutations = {
            "camera": lambda d: d["cameras"]["cam1"]["K"][0].__setitem__(0, d["cameras"]["cam1"]["K"][0][0] + 17.0),
            "integer-to-float clock": lambda d: d["timing_policy"]["extra_offset_us"].__setitem__("cam1", 0.0),
            "boolean-to-integer": lambda d: d["physical_limits"].__setitem__("per_point_scan_times_available", 0),
            "session type": lambda d: d["sessions"][self.sid].__setitem__("frame_count", 3.0),
            "clock identity": lambda d: d["sessions"][self.sid]["source_clock_audit"].__setitem__(
                "common_index_sha256", "f" * 64),
            "artifact provenance": lambda d: next(iter(d["artifacts"].values())).__setitem__(
                "source_path", "different-input"),
            "unknown field": lambda d: d.__setitem__("undeclared_override", True),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                child = self.derive()
                data = deepcopy(child.data)
                mutate(data)
                reseal(child.snapshot_json, data)
                with self.assertRaises(ValueError):
                    self.verify(child)

    def test_rejects_scope_and_purpose_escalation_even_with_new_digest(self):
        mutations = {
            "not review": lambda d: d.__setitem__("review_only", False),
            "review type": lambda d: d.__setitem__("review_only", 1),
            "production": lambda d: d.__setitem__("production_admitted", True),
            "production type": lambda d: d.__setitem__("production_admitted", 0),
            "training": lambda d: d.__setitem__("training_published", True),
            "training type": lambda d: d.__setitem__("training_published", 0),
            "scope expansion": lambda d: d.__setitem__("review_source_ids", [self.sid, "20261022_161632"]),
            "duplicate scope": lambda d: d.__setitem__("review_source_ids", [self.sid, self.sid]),
            "declared source": lambda d: d.__setitem__("declared_calibration_source_ids", ["20261022_161632"]),
            "missing lineage": lambda d: d.pop("admitted_review_derivation"),
            "lineage schema": lambda d: d["admitted_review_derivation"].__setitem__("schema", "other"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                child = self.derive()
                data = deepcopy(child.data)
                mutate(data)
                reseal(child.snapshot_json, data)
                with self.assertRaises(ValueError):
                    self.verify(child)

    def test_rejects_changed_artifact_even_if_its_sha_and_child_digest_are_updated(self):
        child = self.derive()
        data = deepcopy(child.data)
        artifact = data["artifacts"][f"frame_maps/{self.sid}.jsonl"]
        path = child.snapshot_json.parent / artifact["path"]
        path.write_bytes(path.read_bytes() + b"\n")
        artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        reseal(child.snapshot_json, data)
        with self.assertRaises(ValueError):
            self.verify(child)
        self.assertEqual(self.parent_path.read_bytes(), self.parent_bytes)

    def test_rejects_stale_child_digest(self):
        child = self.derive()
        data = deepcopy(child.data)
        data["snapshot_digest"] = self.parent.data["snapshot_digest"]
        write_json(child.snapshot_json, data)
        with self.assertRaises(ValueError):
            self.verify(child)

    def test_rejects_changed_bound_parent_authority_manifest_or_proof(self):
        child = self.derive()
        for path in (self.parent_path, self.authority_path, self.manifest_path, self.proof_path):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                try:
                    path.write_bytes(original + b"\n")
                    with self.assertRaises(ValueError):
                        self.verify(child)
                finally:
                    path.write_bytes(original)

    def test_rejects_invalid_freeze_proof_even_when_binding_is_fresh(self):
        original = deepcopy(self.proof)
        mutations = {
            "schema": lambda d: d.__setitem__("schema", "other"),
            "not completed": lambda d: d.__setitem__("status", "pending"),
            "exit": lambda d: d.__setitem__("actual_exit_code", 1),
            "exit type": lambda d: d.__setitem__("actual_exit_code", False),
            "parent": lambda d: d.__setitem__("accepted_snapshot", binding(self.authority_path)),
            "manifest": lambda d: d.__setitem__("manifest", binding(self.authority_path)),
            "bundle": lambda d: d["accepted_bundle_sha256"].__setitem__("sources.json", "f" * 64),
            "source rows": lambda d: d.__setitem__("source_rows", 2),
            "source rows type": lambda d: d.__setitem__("source_rows", 3.0),
            "rallies": lambda d: d.__setitem__("native_rallies", 2),
            "scan count": lambda d: d.__setitem__("selected_scans", 2),
            "seconds": lambda d: d.__setitem__("selected_seconds", d["selected_seconds"] + .000001),
            "model": lambda d: d.__setitem__("model_loaded", True),
            "model type": lambda d: d.__setitem__("model_loaded", 0),
            "gpu": lambda d: d.__setitem__("gpu_started", True),
            "body": lambda d: d.__setitem__("body_labels_production_admitted", True),
            "training": lambda d: d.__setitem__("training_published", True),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                proof = deepcopy(original)
                mutate(proof)
                write_json(self.proof_path, proof)
                with self.assertRaises(ValueError):
                    self.derive()
        write_json(self.proof_path, original)

    def test_rejects_candidate_or_review_parent_with_rebound_proof(self):
        original = deepcopy(self.parent.data)
        mutations = {
            "candidate": lambda d: d["provenance"].__setitem__("camera_calibration_admitted", False),
            "camera type": lambda d: d["provenance"].__setitem__("camera_calibration_admitted", 1),
            "body": lambda d: d["provenance"].__setitem__("body_labels_production_admitted", True),
            "review": lambda d: d.__setitem__("review_only", True),
            "production": lambda d: d.__setitem__("production_admitted", True),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                data = deepcopy(original)
                mutate(data)
                reseal(self.parent_path, data)
                self.proof["accepted_snapshot"] = binding(self.parent_path)
                write_json(self.proof_path, self.proof)
                with self.assertRaises(ValueError):
                    self.derive()
        self.parent_path.write_bytes(self.parent_bytes)

    def test_rejects_self_consistent_manifest_for_another_source(self):
        other = "20261022_161632"
        self.spec["source_indexes"] = {other: self.spec["source_indexes"][self.sid]}
        self.spec["clips"][0]["source_id"] = other
        self.rebind_manifest()
        with self.assertRaises(ValueError):
            self.derive()

    def test_rejects_bundle_confusion_even_with_matching_authority(self):
        self.spec["bundle_sha256"] = deepcopy(self.hashes)
        self.spec["bundle_sha256"]["sources.json"] = "f" * 64
        self.authority["result_files"] = self.spec["bundle_sha256"]
        write_json(self.authority_path, self.authority)
        self.spec["review_authorization"] = binding(self.authority_path)
        self.rebind_manifest()
        with self.assertRaises(ValueError):
            self.derive()

    def test_requires_new_independent_output(self):
        for output in (self.parent_path.parent, self.parent_path.parent / "child", self.workspace):
            with self.subTest(output=str(output)):
                with self.assertRaises(ValueError):
                    module.derive_review_snapshot(self.parent_path, self.manifest_path, output,
                                                  accepted_freeze_proof=binding(self.proof_path))
        child = self.derive()
        with self.assertRaises(ValueError):
            module.derive_review_snapshot(self.parent_path, self.manifest_path, child.snapshot_json.parent,
                                          accepted_freeze_proof=binding(self.proof_path))

    def test_freeze_proof_accepts_equivalent_windows_path_spelling_but_not_another_binding(self):
        for key in ("accepted_snapshot", "manifest"):
            self.proof[key]["path"] = Path(self.proof[key]["path"]).as_posix()
        write_json(self.proof_path, self.proof)
        child = self.derive()
        self.assertEqual(self.verify(child)["status"], "review_snapshot_verified_metadata_only")
        correct = deepcopy(self.proof)
        for key in ("accepted_snapshot", "manifest"):
            for corruption in ("digest", "path"):
                with self.subTest(binding=key, corruption=corruption):
                    proof = deepcopy(correct)
                    if corruption == "digest":
                        proof[key]["sha256"] = "f" * 64
                    else:
                        proof[key] = binding(self.authority_path)
                        proof[key]["path"] = self.authority_path.as_posix()
                    write_json(self.proof_path, proof)
                    with self.assertRaises(ValueError):
                        self.derive()

    def test_parent_rejects_integer_review_flags_preset_scope_and_integer_production(self):
        original = deepcopy(self.parent.data)
        mutations = {
            "review zero": lambda d: d.__setitem__("review_only", 0),
            "review one": lambda d: d.__setitem__("review_only", 1),
            "preset review sources": lambda d: d.__setitem__("review_source_ids", []),
            "preset declared sources": lambda d: d.__setitem__("declared_calibration_source_ids", [self.sid]),
            "production zero": lambda d: d["provenance"].__setitem__("production_admitted", 0),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                data = deepcopy(original)
                mutate(data)
                reseal(self.parent_path, data)
                self.proof["accepted_snapshot"] = binding(self.parent_path)
                write_json(self.proof_path, self.proof)
                with self.assertRaises(ValueError):
                    self.derive()

    def prepare_fixture(self, name, *, indices=None, native_count=3, missing_view=False, outside_time=False):
        """A real frozen miniature parent and bound native evidence; the source reader alone is synthetic."""
        run = self.workspace / "outputs" / name
        shutil.copytree(self.parent_path.parent, run / "accepted-base")
        parent = run / "accepted-base/calibration_snapshot.json"
        directory = self.root / "data" / name
        analysis = directory / "analysis.json"
        write_json(analysis, dict(sport="badminton", frame_count=3, synthetic_test_only=True))
        planner_sha = "e" * 64
        index = dict(
            source_id=self.sid, source_kind="derived_native_roi_fixed24",
            crop_binding=dict(bbox_sha256=self.hashes["bbox.json"]), accepted_bundle_sha256=self.hashes,
            camera_calibration_admitted=True, body_labels_production_admitted=False, production_admitted=False,
            source_manifest=dict(sha256="b" * 64), common_clock=dict(sha256="c" * 64),
            frame_map=dict(sha256="d" * 64),
        )
        index_path = directory / "source-index.json"
        write_json(index_path, index)
        planned = directory / "clips.json"
        write_json(planned, dict(
            schema="badminton-analysis-clips-v1", sport="badminton", source_id=self.sid,
            analysis_sha256=binding(analysis)["sha256"], planner_sha256=planner_sha,
            clips=[dict(index=1, kind="badminton_rally", source_begin_frame=0, source_frame_count=native_count)],
        ))
        receipt = directory / "receipt.json"
        write_json(receipt, dict(
            source_id=self.sid, status="completed", exit_code=0,
            source_manifest_sha256=index["source_manifest"]["sha256"],
            analysis_sha256=binding(analysis)["sha256"], planner_sha256=planner_sha,
        ))
        spec = deepcopy(self.spec)
        spec["source_indexes"] = {self.sid: dict(
            file=index_path.relative_to(Path(spec["data_root"])).as_posix(), sha256=binding(index_path)["sha256"])}
        spec["clips"][0]["eligible_paired_png_indices"] = indices or [0, 1, 2]
        spec["clips"][0]["evidence_files"] = [binding(analysis), binding(planned), binding(receipt)]
        write_json(self.manifest_path, spec)
        proof = deepcopy(self.proof)
        proof["accepted_snapshot"] = binding(parent)
        proof["manifest"] = binding(self.manifest_path)
        scope = validate_full_native_spec(spec)
        proof.update(selected_scans=scope["selected_scans"], selected_seconds=scope["selected_seconds"])
        write_json(self.proof_path, proof)
        pairs = [dict(lidar_index=i, scan_mid_us=100000 * (i + 1), status="paired") for i in range(3)]
        if missing_view:
            pairs[1]["status"] = "unmatched"
        if outside_time:
            pairs[2]["scan_mid_us"] = spec["clips"][0]["end_common_us"]
        return run, parent, SimpleNamespace(source_index=index, pairs=lambda: pairs)

    def test_prepare_rejects_native_range_pairing_and_time_before_source_is_written(self):
        cases = (
            ("outside-native", dict(indices=[0, 1, 3])),
            ("outside-source", dict(indices=[0, 1, 3], native_count=4)),
            ("missing-view", dict(missing_view=True)),
            ("outside-time", dict(outside_time=True)),
        )
        for name, options in cases:
            with self.subTest(name=name):
                run, parent, session = self.prepare_fixture(name, **options)
                with patch.object(module, "PROJ_ROOT", self.workspace), patch.object(
                    module.SourceIndexedSession, "load_source_index", return_value=session
                ) as reader, patch.object(module, "prepare_new_sources") as source_builder, patch(
                    "hmr4d.lidar_mocap.repair_execution.prepare_full_native_body_plan"
                ) as body_builder:
                    with self.assertRaises(ValueError):
                        module.prepare_admitted_full_native(
                            self.manifest_path, run, parent,
                            accepted_freeze_proof=binding(self.proof_path), threads=1,
                        )
                    self.assertIs(reader.call_args.kwargs["require_admitted"], True)
                    source_builder.assert_not_called()
                    body_builder.assert_not_called()
                self.assertFalse((run / "source").exists())
                self.assertFalse((run / "repair").exists())
                self.assertFalse((run / "pilot_input.json").exists())

    def test_prepare_retains_only_complete_review_group_before_the_real_body_boundary(self):
        run, parent, session = self.prepare_fixture("valid-prefix")

        class ReachedSourceBuilder(Exception):
            pass

        def inspect_source(source, repair, **kwargs):
            registry = json.loads((source / "source_registry.json").read_bytes())
            splits = json.loads((source / "session_splits.json").read_bytes())
            intake = json.loads((source / "pilot_intake.json").read_bytes())
            ranges = [json.loads(line) for line in (source / "pilot_ranges.jsonl").read_text().splitlines()]
            self.assertEqual(splits, dict(review=[self.sid]))
            self.assertEqual(set(registry["sources"]), {self.sid})
            self.assertIs(registry["sources"][self.sid]["baseline"], False)
            self.assertIs(registry["review_only"], True)
            self.assertIs(registry["training_published"], False)
            self.assertIs(intake["review_only"], True)
            self.assertIs(intake["training_published"], False)
            self.assertEqual(ranges[0]["eligible_paired_png_indices"], [0, 1, 2])
            self.assertEqual(ranges[0]["original_rally_index"], 1)
            self.assertEqual(ranges[0]["segment_index"], 0)
            self.assertEqual(repair, run / "repair")
            raise ReachedSourceBuilder

        with patch.object(module, "PROJ_ROOT", self.workspace), patch.object(
            module.SourceIndexedSession, "load_source_index", return_value=session
        ), patch.object(module, "prepare_new_sources", side_effect=inspect_source), patch(
            "hmr4d.lidar_mocap.repair_execution.prepare_full_native_body_plan"
        ) as body_builder:
            with self.assertRaises(ReachedSourceBuilder):
                module.prepare_admitted_full_native(
                    self.manifest_path, run, parent,
                    accepted_freeze_proof=binding(self.proof_path), threads=1,
                )
            body_builder.assert_not_called()
        self.assertFalse((run / "body_prepare_cpu_receipt.json").exists())
        self.assertFalse((run / "pilot_progress.json").exists())


if __name__ == "__main__":
    unittest.main()
