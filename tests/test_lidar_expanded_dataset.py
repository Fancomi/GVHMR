from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.expanded_dataset import (
    CONTRACT,
    _validate_new_completion,
    assemble,
    binding,
    contiguous_sequences,
    merge_role,
    validate_union_contract,
    verify_expanded_base,
)
from hmr4d.lidar_mocap.export import FIELDS, export_joblib
from hmr4d.lidar_mocap.pipeline import write_json


def sample(sid, value=1, scan=10, segment=0, digest="old-cal"):
    motion = f"{sid}:window:3:{segment}"
    row = dict(
        session_id=sid,
        motion_id=motion,
        window_id="window",
        track_id=3,
        lidar_index=scan,
        scan_mid_us=scan * 100000,
        calibration_digest=digest,
    )
    return dict(
        {name: np.full(shape, value, dtype=np.float32) for name, shape in FIELDS.items()},
        motion_id=motion,
        metadata=row,
    )


def export(root, sid, samples):
    entry = export_joblib(samples, root / (sid + ".pth"))
    entry.update(
        sha256=binding(entry["path"])["sha256"], metadata_sha256=binding(entry["metadata"])["sha256"], sessions=[sid]
    )
    return entry


class ExpandedDatasetTests(unittest.TestCase):
    def test_exact_old_prefix_and_metadata_bytes_with_different_calibrations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old = export(root / "old", "a", [sample("a", 1)])
            new = export(root / "new", "b", [sample("b", 2, digest="new-cal")])
            old_bytes = Path(old["metadata"]).read_bytes()
            new_bytes = Path(new["metadata"]).read_bytes()
            merged, rows = merge_role([old, new], root / "result/train.pth", {"a": "old-cal", "b": "new-cal"})
            self.assertEqual(Path(merged["metadata"]).read_bytes(), old_bytes + new_bytes)
            self.assertEqual(merged["original_role_prefix_samples"], 1)
            data = joblib.load(merged["path"])
            for name in FIELDS:
                self.assertTrue(torch.equal(data[name][:1], joblib.load(old["path"])[name]))
                self.assertTrue(torch.equal(data[name][1:], joblib.load(new["path"])[name]))
            self.assertEqual([r["session_id"] for r in rows], ["a", "b"])

    def test_duplicate_canonical_person_scan_is_rejected_across_segments(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            entry = export(root, "a", [sample("a", segment=0), sample("a", segment=1)])
            with self.assertRaisesRegex(ValueError, "Duplicate canonical"):
                merge_role([entry], root / "result/train.pth", {"a": "old-cal"})
            self.assertFalse((root / "result/train.pth").exists())

    def test_source_calibration_mismatch_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            entry = export(root, "a", [sample("a")])
            with self.assertRaisesRegex(ValueError, "source or calibration"):
                merge_role([entry], root / "result/train.pth", {"a": "new-cal"})

    def make_publications(self, root, *, formal=False):
        contract = dict(old_accepted=3, old_split_counts={r: 1 for r in ("train", "val", "test")})
        publications = []
        for generation in ("old", "new"):
            repair = root / "formal/repair" if formal and generation == "new" else root / (generation + "_repair")
            body = repair / "final_labels"
            source = body / "datasets/dataset_manifest.json"
            write_json(source, {})
            write_json(body / "calibration/calibration_snapshot.json", dict(snapshot_digest=generation + "-cal"))
            paired = repair.with_name(repair.name + "_paired")
            manifest = dict(**CONTRACT, status="complete", accepted=3, calibration_digest=generation + "-cal")
            manifest.update(
                source_dataset_manifest=str(source), source_dataset_manifest_sha256=binding(source)["sha256"]
            )
            manifest["exports"], manifest["suggested_session_splits"] = {}, {}
            roles = {}
            for role in ("train", "val", "test"):
                sid = generation + "_" + role
                entry = export(paired / "datasets", sid, [sample(sid, digest=generation + "-cal")])
                manifest["exports"][sid] = entry
                manifest["suggested_session_splits"][role] = deepcopy(entry)
                roles[role] = [sid]
            if generation == "new":
                sid = "new_zero"
                manifest["exports"][sid] = export(paired / "datasets", sid, [])
                manifest["suggested_session_splits"]["train"]["sessions"].append(sid)
                roles["train"].append(sid)
                contract["new_repair_root"] = str(repair)
            contract[generation + "_source_splits"] = roles
            path = paired / "datasets/dataset_manifest.json"
            write_json(path, manifest)
            publications.append((path, manifest))
        input_path, split_path = root / "input.json", root / "splits.json"
        write_json(input_path, {})
        write_json(split_path, {})
        contract["bindings"] = dict(
            old_dataset_manifest=binding(publications[0][0]),
            new_source_input=binding(input_path),
            new_source_splits=binding(split_path),
        )
        new_path = publications[1][0]
        write_json(
            Path(contract["new_repair_root"]) / "new_source_completion_receipt.json",
            dict(training_published=True, evidence={str(new_path): binding(new_path)["sha256"]}),
        )
        contract_path = root / "contract.json"
        write_json(contract_path, contract)
        return publications, contract, contract_path

    def test_full_assembly_retains_zero_source_and_remains_unpublished(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            publications, _, contract_path = self.make_publications(root)

            def load_calibration(path):
                generation = "new" if "new_repair" in str(path) else "old"
                return SimpleNamespace(data=dict(snapshot_digest=generation + "-cal"))

            with (
                patch("hmr4d.lidar_mocap.expanded_dataset.CalibrationSnapshot.load", side_effect=load_calibration),
                patch("hmr4d.lidar_mocap.expanded_dataset._verify_export_data", return_value={}),
            ):
                result = assemble(publications[0][0], publications[1][0], contract_path, root / "union")
            self.assertEqual(result["accepted"], 6)
            self.assertEqual(result["source_registry"]["new_zero"]["accepted"], 0)
            self.assertEqual(result["source_registry"]["new_zero"]["role"], "train")
            self.assertFalse(result["training_published"])
            self.assertFalse((root / "union/training_manifest.json").exists())
            self.assertFalse((root / "union/datasets/dataset_manifest.json").exists())
            self.assertEqual(result["source_registry"]["new_test"]["calibration_digest"], "new-cal")

    def test_truncation_and_whole_source_role_changes_are_rejected(self):
        with TemporaryDirectory() as directory:
            publications, contract, _ = self.make_publications(Path(directory))
            old, new = [m for _, m in publications]
            self.assertEqual(validate_union_contract(old, new, contract)["new_zero"], "train")
            truncated = dict(contract, old_accepted=2)
            with self.assertRaisesRegex(ValueError, "complete old"):
                validate_union_contract(old, new, truncated)
            changed = deepcopy(contract)
            changed["new_source_splits"]["val"] = ["new_test"]
            with self.assertRaisesRegex(ValueError, "Whole-source"):
                validate_union_contract(old, new, changed)

    def test_sequences_keep_missing_scans_time_gaps_and_identity_segments_separate(self):
        rows = [sample("a", scan=s)["metadata"] for s in (10, 11, 13, 14)]
        rows[-1]["scan_mid_us"] += 300000
        rows.append(sample("a", scan=15, segment=1)["metadata"])
        chunks = contiguous_sequences(rows)
        self.assertEqual([c["scans"] for c in chunks], [[10, 11], [13], [14], [15]])
        self.assertEqual(sorted(i for c in chunks for i in c["indices"]), list(range(5)))

    def make_expanded_base(self, root):
        publications, contract, _ = self.make_publications(root)
        old_path, original = publications[0]
        old = deepcopy(original)
        old.update(schema="badminton-expanded-dataset-v1", training_published=True)
        for name in ("source_dataset_manifest", "source_dataset_manifest_sha256", "calibration_digest"):
            old.pop(name)
        archive = root / "archive"
        old["canonical_archive"] = str(archive)
        registry, calibrations, archive_splits = {}, {}, {}
        for role in ("train", "val", "test"):
            sid = "old_" + role
            digest = "second-cal" if role == "val" else "old-cal"
            entry = export(root / "base_parts", sid, [sample(sid, digest=digest)])
            old["exports"][sid] = deepcopy(entry)
            old["suggested_session_splits"][role] = deepcopy(entry)
            cal_path = root / "calibrations" / digest / "calibration_snapshot.json"
            write_json(cal_path, dict(snapshot_digest=digest))
            cal_ref = binding(cal_path)
            registry[sid] = dict(
                role=role, accepted=1, calibration_digest=digest,
                calibration_snapshot=str(cal_path), calibration_snapshot_sha256=cal_ref["sha256"],
                generation="new" if role == "test" else "old", source_run="unchanged original lineage",
            )
            relative = Path("calibration") / digest / "calibration_snapshot.json"
            destination = archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cal_path, destination)
            calibrations[digest] = dict(path=relative.as_posix(), bytes=destination.stat().st_size,
                                        sha256=cal_ref["sha256"])
            refs = {}
            for field, hash_field, name in (("path", "sha256", "data.pth"),
                                             ("metadata", "metadata_sha256", "metadata.jsonl")):
                destination = archive / role / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(entry[field], destination)
                old["suggested_session_splits"][role][field] = str(destination)
                refs["tensor" if field == "path" else field] = dict(
                    path=destination.relative_to(archive).as_posix(), bytes=destination.stat().st_size,
                    sha256=entry[hash_field],
                )
            archive_splits[role] = dict(frames=1, declared_source_sessions=[sid], **refs)
        old["source_registry"] = registry
        old["source_publications"] = dict(original=binding(old_path))
        archive_manifest = archive / "manifest.json"
        write_json(archive_manifest, dict(schema="lidar2smpl-release-v1", status="complete",
                                         total_person_frames=3, source_registry=registry,
                                         splits=archive_splits, source_calibrations=calibrations))
        old_path = root / "expanded/datasets/dataset_manifest.json"
        write_json(old_path, old)
        contract["bindings"]["old_dataset_manifest"] = binding(old_path)
        contract["bindings"]["old_archive"] = binding(archive_manifest)
        contract_path = root / "expanded_contract.json"
        write_json(contract_path, contract)
        return (old_path, old), publications[1], contract, contract_path

    @staticmethod
    def expanded_calibration(path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return SimpleNamespace(data=data if "snapshot_digest" in data else dict(snapshot_digest="new-cal"))

    def test_expansion_retains_complete_archived_base_with_two_calibrations(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (old_path, old), (new_path, _), _, contract_path = self.make_expanded_base(root)
            with (patch("hmr4d.lidar_mocap.expanded_dataset.CalibrationSnapshot.load",
                        side_effect=self.expanded_calibration),
                  patch("hmr4d.lidar_mocap.expanded_dataset._verify_export_data", return_value={}) as verify):
                result = assemble(old_path, new_path, contract_path, root / "next_expansion")
            self.assertEqual(verify.call_count, 1)
            self.assertEqual(result["accepted"], 6)
            self.assertEqual(result["source_registry"]["old_val"]["calibration_digest"], "second-cal")
            for sid, before in old["source_registry"].items():
                self.assertEqual(result["source_registry"][sid], dict(before, generation="old"))
            self.assertEqual(old["source_registry"]["old_test"]["generation"], "new")
            for role in ("train", "val", "test"):
                before = old["suggested_session_splits"][role]
                after = result["suggested_session_splits"][role]
                self.assertTrue(Path(after["metadata"]).read_bytes().startswith(Path(before["metadata"]).read_bytes()))
                self.assertEqual(after["original_role_prefix_samples"], 1)
                for field in FIELDS:
                    self.assertTrue(torch.equal(joblib.load(after["path"])[field][:1],
                                                joblib.load(before["path"])[field]))

    def test_expanded_base_rejects_wrong_archive_payload_or_missing_archive_contract(self):
        with TemporaryDirectory() as directory:
            (old_path, old), _, contract, _ = self.make_expanded_base(Path(directory))
            missing = deepcopy(contract)
            missing["bindings"].pop("old_archive")
            with self.assertRaisesRegex(ValueError, "explicit old archive"):
                verify_expanded_base(old, missing)
            changed = deepcopy(old)
            changed["suggested_session_splits"]["train"]["metadata_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "role payload"):
                verify_expanded_base(changed, contract)
            self.assertTrue(old_path.exists())

    def test_expanded_base_rejects_source_calibration_substitution(self):
        with TemporaryDirectory() as directory:
            (_, old), _, contract, _ = self.make_expanded_base(Path(directory))
            with patch("hmr4d.lidar_mocap.expanded_dataset.CalibrationSnapshot.load",
                       return_value=SimpleNamespace(data=dict(snapshot_digest="substituted-cal"))):
                with self.assertRaisesRegex(ValueError, "Archived source calibration"):
                    verify_expanded_base(old, contract)

    def test_expanded_base_rejects_an_archive_member_outside_archive(self):
        with TemporaryDirectory() as directory:
            (_, old), _, contract, _ = self.make_expanded_base(Path(directory))
            archive_path = Path(contract["bindings"]["old_archive"]["path"])
            archive = json.loads(archive_path.read_text(encoding="utf-8"))
            archive["splits"]["train"]["tensor"]["path"] = "../outside.pth"
            write_json(archive_path, archive)
            contract["bindings"]["old_archive"] = binding(archive_path)
            with self.assertRaisesRegex(ValueError, "escapes its archive"):
                verify_expanded_base(old, contract)

    def test_draft_or_unsealed_joint_contract_cannot_assemble(self):
        with TemporaryDirectory() as directory:
            publications, contract, _ = self.make_publications(Path(directory))
            old, new = [m for _, m in publications]
            for schema, status, sealed in (
                ("expanded-1019-1020-consumer-contract-draft/v1", "draft_unsealed", False),
                ("expanded-1019-1020-consumer-contract/v1", "complete", False),
                ("expanded-1019-1020-consumer-contract/v1", "draft", True),
                ("expanded-1019-1020-assembly-input/v1", "complete", False),
                (None, "complete", True),
            ):
                with self.subTest(schema=schema, status=status, sealed=sealed):
                    with self.assertRaisesRegex(ValueError, "unsealed consumer draft"):
                        validate_union_contract(old, new, dict(contract, schema=schema, status=status, sealed=sealed))

    def test_sealed_joint_input_requires_final_source_counts_and_actual_publication_binding(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            publications, contract, _ = self.make_publications(root)
            old, new = [m for _, m in publications]
            contract.update(schema="expanded-1019-1020-assembly-input/v1", status="complete", sealed=True,
                            new_sources={sid: dict(final_admitted_count=e["accepted"])
                                         for sid, e in new["exports"].items()})
            self.assertEqual(validate_union_contract(old, new, contract)["new_zero"], "train")
            for value in (None, True, 99):
                changed = deepcopy(contract)
                changed["new_sources"]["new_train"]["final_admitted_count"] = value
                with self.subTest(count=value):
                    with self.assertRaisesRegex(ValueError, "count is pending or differs"):
                        validate_union_contract(old, new, changed)
            contract["bindings"]["producer_confirmation"] = binding(
                Path(contract["new_repair_root"]) / "new_source_completion_receipt.json")
            contract["bindings"]["producer_joint_manifest"] = binding(publications[0][0])
            path = root / "joint_contract.json"
            write_json(path, contract)
            with self.assertRaisesRegex(ValueError, "different new publication"):
                assemble(publications[0][0], publications[1][0], path, root / "wrong_union")
            self.assertFalse((root / "wrong_union").exists())
            contract["bindings"]["producer_joint_manifest"] = binding(publications[1][0])
            write_json(path, contract)
            with (patch("hmr4d.lidar_mocap.expanded_dataset.CalibrationSnapshot.load",
                        side_effect=lambda p: SimpleNamespace(data=dict(
                            snapshot_digest="new-cal" if "new_repair" in str(p) else "old-cal"))),
                  patch("hmr4d.lidar_mocap.expanded_dataset._verify_export_data", return_value={})):
                result = assemble(publications[0][0], publications[1][0], path, root / "valid_joint_union")
            self.assertEqual(result["source_publications"]["new"], binding(publications[1][0]))
            self.assertEqual(result["new_completion"], contract["bindings"]["producer_confirmation"])
            self.assertFalse(result["training_published"])

    def make_formal_completion(self, root):
        publications, contract, contract_path = self.make_publications(root, formal=True)
        parent_path, publication = publications[1]
        repair = Path(contract["new_repair_root"])
        labels = repair / "final_labels"
        jobs = [
            dict(source_id=sid, source_window=str(repair / "initial_teacher" / sid / "window"))
            for sid, entry in publication["exports"].items()
            if entry["accepted"]
        ]
        source_path = repair / "new_source_input.json"
        write_json(
            source_path, dict(schema="badminton-new-source-repair-input-v1", calibration_digest="new-cal", jobs=jobs)
        )
        source = binding(source_path)
        write_json(
            repair / "case_plan.json",
            dict(
                source_union=source["path"],
                source_union_sha256=source["sha256"],
                calibration_digest="new-cal",
                jobs=jobs,
            ),
        )
        write_json(
            labels / "source_registry.json",
            dict(
                schema="badminton-formal-derived-source-registry-v1",
                sources={sid: {} for sid in publication["exports"]},
            ),
        )
        body = dict(
            accepted=6,
            calibration_digest="new-cal",
            source_registry_sha256=binding(labels / "source_registry.json")["sha256"],
            exports={
                sid: export(
                    labels / "datasets",
                    sid,
                    [sample(sid, scan=scan, digest="new-cal") for scan in (10, 11)] if entry["accepted"] else [],
                )
                for sid, entry in publication["exports"].items()
            },
        )
        body_path = labels / "datasets/dataset_manifest.json"
        write_json(body_path, body)
        observations_path = parent_path.parent.parent / "observations/manifest.json"
        write_json(
            observations_path,
            dict(
                completed=True,
                source_accepted=6,
                identity=dict(calibration_digest="new-cal", source=dict(dataset_manifest=binding(body_path))),
            ),
        )
        publication.update(
            source_dataset_manifest=str(body_path),
            source_dataset_manifest_sha256=binding(body_path)["sha256"],
            full_observation_manifest=str(observations_path),
            full_observation_manifest_sha256=binding(observations_path)["sha256"],
        )
        write_json(parent_path, publication)
        write_json(
            parent_path.parent.parent / "verification.json",
            dict(
                accepted=3,
                sessions={sid: dict(accepted=entry["accepted"]) for sid, entry in publication["exports"].items()},
            ),
        )
        write_json(labels / "viewer/manifest.json", dict(calibration_digest="new-cal"))
        evidence_paths = dict(
            before=repair / "evaluation/v2_before_leg.json",
            native=labels / "native_verification.json",
            dataset=labels / "datasets/verification.json",
            post_refinement=labels / "post_refinement.json",
            leg_receipt=repair / "leg_refinement_receipt.json",
            after=repair / "evaluation/v2_after_leg.json",
            comparison=repair / "evaluation/v2_leg_comparison.json",
        )
        for name, path in evidence_paths.items():
            write_json(path, dict(synthetic_report=name, accepted=6))
        baseline_path = repair.parent / "gpu_guarded_leg/baseline/baseline.json"
        execution_path = repair.parent / "gpu_guarded_leg/attempts/actual_fixture/completion.json"
        write_json(baseline_path, dict(source_input=source))
        write_json(execution_path, dict(synthetic_completed_execution=True))
        allocation = dict(
            schema="gvhmr-formal-cpu-worker-v1", affinity=[26, 27], priority="BelowNormal", cuda_visible_devices="-1"
        )
        workers = {}
        for role, entry in (("native", "verify_badminton_native_bound.py"), ("dataset", "verify_badminton_dataset.py")):
            log = repair / "evaluation/formal_postleg_cpu/attempts/actual_fixture" / (role + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(json.dumps(allocation) + "\nsynthetic CPU fixture only\n", encoding="utf-8")
            workers[role] = dict(
                exit_code=0,
                allocation=allocation,
                hidden_window=True,
                entry=binding(PROJ_ROOT / "tools" / entry),
                log=binding(log),
            )
        receipt_path = repair / "evaluation/formal_postleg_cpu_receipt.json"
        receipt = dict(
            schema="gvhmr-formal-postleg-cpu-completion-v1",
            state="postleg_native_qa_complete_bilateral_browser_archive_pending",
            root=str(repair.parent),
            source_input=source,
            calibration_digest="new-cal",
            windows=len(jobs),
            accepted=6,
            baseline=binding(baseline_path),
            leg_execution=binding(execution_path),
            workers=workers,
            evidence={name: binding(path) for name, path in evidence_paths.items()},
            gpu_started=False,
            training_published=False,
        )
        write_json(receipt_path, receipt)
        completion = dict(
            state="labels_published_viewer_prepared",
            training_published=False,
            bilateral_labels_published=True,
            archive_published=False,
            consumer_admitted=False,
            dataset_manifest=str(parent_path),
            accepted=3,
            formal_postleg_cpu_receipt=binding(receipt_path),
        )
        paths = [
            parent_path,
            receipt_path,
            body_path,
            observations_path,
            parent_path.parent.parent / "verification.json",
            labels / "viewer/manifest.json",
        ]
        paths.extend(evidence_paths[name] for name in ("before", "native", "leg_receipt", "after", "comparison"))
        completion["evidence"] = {str(path): binding(path)["sha256"] for path in paths}
        write_json(repair / "new_source_completion_receipt.json", completion)
        contract["bindings"]["new_source_input"] = source
        contract.update(
            schema="expanded-1019-1020-assembly-input/v1",
            status="complete",
            sealed=True,
            new_sources={
                sid: dict(final_admitted_count=entry["accepted"]) for sid, entry in publication["exports"].items()
            },
        )
        contract["bindings"]["producer_confirmation"] = binding(repair / "new_source_completion_receipt.json")
        contract["bindings"]["producer_joint_manifest"] = binding(parent_path)
        write_json(contract_path, contract)
        return publications, contract_path, repair, completion, receipt

    def rebind_cpu_receipt(self, repair, completion, receipt):
        path = repair / "evaluation/formal_postleg_cpu_receipt.json"
        write_json(path, receipt)
        completion["formal_postleg_cpu_receipt"] = binding(path)
        completion["evidence"][str(path)] = binding(path)["sha256"]

    def test_formal_assembly_reuses_proven_cpu_receipt_with_different_body_and_bilateral_counts(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            publications, contract_path, repair, completion, _ = self.make_formal_completion(root)
            with (
                patch(
                    "hmr4d.lidar_mocap.expanded_dataset.CalibrationSnapshot.load", side_effect=self.expanded_calibration
                ),
                patch("hmr4d.lidar_mocap.expanded_dataset._verify_export_data", return_value={}),
                patch(
                    "hmr4d.lidar_mocap.formal_completion.verify_formal_postleg_cpu",
                    side_effect=AssertionError("Do not rerun formal CPU/context verification"),
                ) as cpu,
                patch("subprocess.run", side_effect=AssertionError("Do not start a worker")) as launch,
            ):
                result = assemble(publications[0][0], publications[1][0], contract_path, root / "formal_union")
            cpu.assert_not_called()
            launch.assert_not_called()
            self.assertEqual(result["accepted"], 6)
            self.assertEqual(result["new_accepted"], completion["accepted"])
            self.assertEqual(result["new_completion"], binding(repair / "new_source_completion_receipt.json"))
            self.assertFalse(result["training_published"])
            self.assertFalse((root / "formal_union/training_manifest.json").exists())
            self.assertEqual(json.loads((repair / "new_source_completion_receipt.json").read_text()), completion)

    def test_formal_canonical_assembly_checks_the_original_bilateral_parent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            publications, contract_path, repair, _, _ = self.make_formal_completion(root)
            original_path, publication = publications[1]
            review_path, verification_path = repair / "review.json", repair / "canonical_verification.json"
            write_json(review_path, dict(synthetic_review=True))
            write_json(verification_path, dict(numeric_values_unchanged=True))
            canonical = dict(
                publication,
                canonical_identity_revision=dict(
                    source_publication=binding(original_path),
                    reviewed_boundaries=binding(review_path),
                    verification=binding(verification_path),
                ),
            )
            canonical_path = repair / "canonical/datasets/dataset_manifest.json"
            write_json(canonical_path, canonical)
            contract = json.loads(contract_path.read_text())
            contract["bindings"]["producer_joint_manifest"] = binding(canonical_path)
            write_json(contract_path, contract)
            with (
                patch(
                    "hmr4d.lidar_mocap.expanded_dataset.CalibrationSnapshot.load", side_effect=self.expanded_calibration
                ),
                patch("hmr4d.lidar_mocap.expanded_dataset._verify_export_data", return_value={}),
                patch("hmr4d.lidar_mocap.canonical_tracks.verify_canonical_publication", return_value={}),
            ):
                result = assemble(publications[0][0], canonical_path, contract_path, root / "canonical_union")
            self.assertEqual(result["source_publications"]["new"], binding(canonical_path))
            self.assertEqual(result["new_accepted"], publication["accepted"])

    def test_formal_contract_cannot_keep_a_foreign_input_or_confirmation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            publications, contract_path, _, _, _ = self.make_formal_completion(root)
            contract = json.loads(contract_path.read_text())
            foreign = root / "old_1020_scope.json"
            write_json(foreign, dict(old_scope=True))
            for name in ("new_source_input", "producer_confirmation", "schema"):
                changed = deepcopy(contract)
                if name == "schema":
                    changed["schema"] = "expanded-1015-consumer-contract/v1"
                else:
                    changed["bindings"][name] = binding(foreign)
                write_json(contract_path, changed)
                with (
                    patch(
                        "hmr4d.lidar_mocap.expanded_dataset.CalibrationSnapshot.load",
                        side_effect=self.expanded_calibration,
                    ),
                    patch("hmr4d.lidar_mocap.expanded_dataset._verify_export_data", return_value={}),
                    self.subTest(field=name),
                    self.assertRaisesRegex(ValueError, "sealed joint contract"),
                ):
                    assemble(publications[0][0], publications[1][0], contract_path, root / (name + "_union"))
                self.assertFalse((root / (name + "_union") / "assembly_manifest.json").exists())

    def test_formal_pending_and_false_training_publication_claims_are_rejected(self):
        with TemporaryDirectory() as directory:
            publications, _, repair, completion, _ = self.make_formal_completion(Path(directory))
            parent, publication = publications[1]
            for change in (
                dict(state="bilateral_observation_pending"),
                dict(training_published=True),
                dict(bilateral_labels_published=False),
                dict(archive_published=True),
                dict(consumer_admitted=True),
                dict(accepted=6),
                dict(accepted=True),
                dict(dataset_manifest="foreign.json"),
            ):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    _validate_new_completion(repair, dict(completion, **change), binding(parent), publication)

    def test_formal_receipt_semantics_cannot_be_replaced_by_a_new_hash(self):
        with TemporaryDirectory() as directory:
            publications, _, repair, original, receipt = self.make_formal_completion(Path(directory))
            parent, publication = publications[1]
            for change in (
                dict(schema="unrecognized"),
                dict(state="pending"),
                dict(root=str(repair)),
                dict(source_input=dict(receipt["source_input"], sha256="0" * 64)),
                dict(calibration_digest="other-camera"),
                dict(windows=2),
                dict(windows=True),
                dict(accepted=3),
                dict(gpu_started=True),
                dict(training_published=True),
                dict(workers={}),
                dict(evidence={}),
            ):
                changed, completion = dict(receipt, **change), deepcopy(original)
                self.rebind_cpu_receipt(repair, completion, changed)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    _validate_new_completion(repair, completion, binding(parent), publication)

    def test_formal_requires_existing_exact_cpu_receipt_and_worker_evidence(self):
        with TemporaryDirectory() as directory:
            publications, _, repair, original, receipt = self.make_formal_completion(Path(directory))
            parent, publication = publications[1]
            for field, value in (
                ("exit_code", 1),
                ("exit_code", False),
                ("allocation", {}),
                ("hidden_window", False),
                ("entry", receipt["workers"]["dataset"]["entry"]),
                ("log", receipt["workers"]["dataset"]["log"]),
            ):
                changed, completion = deepcopy(receipt), deepcopy(original)
                changed["workers"]["native"][field] = value
                self.rebind_cpu_receipt(repair, completion, changed)
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    _validate_new_completion(repair, completion, binding(parent), publication)
            self.rebind_cpu_receipt(repair, original, receipt)
            wrong = deepcopy(original)
            wrong["formal_postleg_cpu_receipt"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "Frozen dependency"):
                _validate_new_completion(repair, wrong, binding(parent), publication)
            wrong = deepcopy(original)
            foreign = repair / "foreign_receipt.json"
            write_json(foreign, receipt)
            wrong["formal_postleg_cpu_receipt"] = binding(foreign)
            with self.assertRaisesRegex(ValueError, "exact existing"):
                _validate_new_completion(repair, wrong, binding(parent), publication)
            wrong = deepcopy(original)
            del wrong["formal_postleg_cpu_receipt"]
            with self.assertRaisesRegex(ValueError, "exact existing"):
                _validate_new_completion(repair, wrong, binding(parent), publication)

    def test_formal_rejects_foreign_parent_observations_and_bilateral_counts(self):
        with TemporaryDirectory() as directory:
            publications, _, repair, completion, _ = self.make_formal_completion(Path(directory))
            parent, publication = publications[1]
            for change in (
                dict(source_dataset_manifest=str(repair / "foreign_body.json")),
                dict(source_dataset_manifest_sha256="0" * 64),
                dict(full_observation_manifest=str(repair / "foreign_observations.json")),
                dict(full_observation_manifest_sha256="0" * 64),
            ):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    _validate_new_completion(repair, completion, binding(parent), dict(publication, **change))
            observed = Path(publication["full_observation_manifest"])
            original = json.loads(observed.read_text())
            for change in (
                dict(completed=False),
                dict(source_accepted=3),
                dict(identity=dict(calibration_digest="other-camera")),
            ):
                write_json(observed, dict(original, **change))
                changed = deepcopy(completion)
                changed["evidence"][str(observed)] = binding(observed)["sha256"]
                current = dict(publication, full_observation_manifest_sha256=binding(observed)["sha256"])
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "bilateral observations"):
                    _validate_new_completion(repair, changed, binding(parent), current)

    def test_formal_source_registry_and_bilateral_verification_must_remain_bound(self):
        with TemporaryDirectory() as directory:
            publications, _, repair, completion, _ = self.make_formal_completion(Path(directory))
            parent, publication = publications[1]
            verification_path = parent.parent.parent / "verification.json"
            verification = json.loads(verification_path.read_text())
            for change in (
                dict(accepted=6),
                dict(sessions={}),
                dict(sessions=dict(verification["sessions"], new_train=dict(accepted=2))),
            ):
                write_json(verification_path, dict(verification, **change))
                current = deepcopy(completion)
                current["evidence"][str(verification_path)] = binding(verification_path)["sha256"]
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "bilateral verification"):
                    _validate_new_completion(repair, current, binding(parent), publication)
            write_json(verification_path, verification)
            registry_path = repair / "final_labels/source_registry.json"
            registry = json.loads(registry_path.read_text())
            registry["sources"]["new_train"]["changed"] = True
            write_json(registry_path, registry)
            with self.assertRaisesRegex(ValueError, "source/calibration/window/body"):
                _validate_new_completion(repair, completion, binding(parent), publication)

    def test_formal_rechecks_existing_proof_bindings_before_completing_assembly(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            publications, contract_path, _, _, _ = self.make_formal_completion(root)
            observation_path = Path(publications[1][1]["full_observation_manifest"])
            called = []

            def merge_then_change(*args):
                result = merge_role(*args)
                if not called:
                    observation_path.write_text("{}", encoding="utf-8")
                called.append(True)
                return result

            with (
                patch(
                    "hmr4d.lidar_mocap.expanded_dataset.CalibrationSnapshot.load", side_effect=self.expanded_calibration
                ),
                patch("hmr4d.lidar_mocap.expanded_dataset._verify_export_data", return_value={}),
                patch("hmr4d.lidar_mocap.expanded_dataset.merge_role", side_effect=merge_then_change),
                self.assertRaisesRegex(ValueError, "Frozen dependency"),
            ):
                assemble(publications[0][0], publications[1][0], contract_path, root / "changed_union")
            self.assertFalse((root / "changed_union/assembly_manifest.json").exists())

    def test_expanded_old_generation_allows_exact_historical_pixel_supplement_without_rewriting_base(self):
        from tools.audit_badminton_expanded_pixels import historical_corrections

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (old_path, old), (new_path, _), _, contract_path = self.make_expanded_base(root)
            old_binding = binding(old_path)
            with (
                patch(
                    "hmr4d.lidar_mocap.expanded_dataset.CalibrationSnapshot.load", side_effect=self.expanded_calibration
                ),
                patch("hmr4d.lidar_mocap.expanded_dataset._verify_export_data", return_value={}),
            ):
                result = assemble(old_path, new_path, contract_path, root / "next_expansion")
            metadata = result["suggested_session_splits"]["test"]["components"][0]["metadata"]
            native = dict(record_index=7, media_pts_us=700000, common_time_us=800000, source_frame_no=7)
            evidence_path, report_path = root / "pixel_evidence.json", root / "original_report.json"
            write_json(report_path, {})
            write_json(
                evidence_path,
                dict(
                    metadata=metadata,
                    all_person_admissions_unchanged=True,
                    people=[dict(unchanged=True)],
                    original_views=dict(cam1=dict(native_bgr_image_sha256="a" * 64, camera_native_source=native)),
                    verified_views=dict(cam1=dict(native_bgr_image_sha256="b" * 64, camera_native_source=native)),
                ),
            )
            correction = dict(
                scene_id="old_test",
                camera="cam1",
                **native,
                previous_raw_bgr_sha256="a" * 64,
                verified_raw_bgr_sha256="b" * 64,
                evidence=binding(evidence_path),
                original_report=binding(report_path),
            )
            supplement = root / "historical_corrections.json"
            write_json(
                supplement,
                dict(
                    schema="native-pixel-corrections/v1",
                    status="complete",
                    all_person_admissions_unchanged=True,
                    metadata=metadata,
                    training_manifest=old_binding,
                    full_native_pixel_audit=binding(report_path),
                    corrections=[correction],
                ),
            )
            with self.assertRaisesRegex(ValueError, "new source"):
                historical_corrections(supplement, dict(result, source_registry=old["source_registry"]))
            self.assertEqual(historical_corrections(supplement, result), {("old_test", "cam1", 7): correction})
            self.assertEqual(binding(old_path), old_binding)
            self.assertEqual(old["source_registry"]["old_test"]["generation"], "new")


if __name__ == "__main__":
    unittest.main()
