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

from hmr4d.lidar_mocap.expanded_dataset import (
    CONTRACT,
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

    def make_publications(self, root):
        contract = dict(old_accepted=3, old_split_counts={r: 1 for r in ("train", "val", "test")})
        publications = []
        for generation in ("old", "new"):
            repair = root / (generation + "_repair")
            body = repair / "final_labels"
            source = body / "datasets/dataset_manifest.json"
            write_json(source, {})
            write_json(body / "calibration/calibration_snapshot.json", {})
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
            ):
                with self.subTest(schema=schema, status=status, sealed=sealed):
                    with self.assertRaisesRegex(ValueError, "unsealed consumer draft"):
                        validate_union_contract(old, new, dict(contract, schema=schema, status=status, sealed=sealed))


if __name__ == "__main__":
    unittest.main()
