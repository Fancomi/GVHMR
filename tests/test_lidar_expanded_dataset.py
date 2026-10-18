from copy import deepcopy
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
