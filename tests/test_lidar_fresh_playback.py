"""Frozen source coverage and actual-rally selection across legacy/formal viewers."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hmr4d.lidar_mocap.rally_viewer import load_viewer_intake

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from badminton_viewer_contracts import (
    check_viewer_coverage, complete_clip_plan, validate_browser_source_acceptance, viewer_relocations,
)


class FrozenViewerCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "calibration").mkdir()
        (self.root / "calibration/calibration_snapshot.json").write_text(
            json.dumps(dict(snapshot_digest="frozen_camera")), encoding="utf-8",
        )

    def write_bound(self, name, data, *, rows=False):
        path = self.root / name
        text = "\n".join(json.dumps(row) for row in data) if rows else json.dumps(data)
        path.write_text(text, encoding="utf-8")
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def fixture(self, count, *, formal=False, empty=()):
        ids = [f"source{index}" for index in range(count)]
        ranges = [dict(
            source_id=sid, eligible_paired_png_indices=[1, 2], source_frame_count=2,
            first_scan_mid_us=100, last_scan_mid_us=200,
        ) for sid in ids if sid not in empty]
        prepared = self.write_bound("ranges.jsonl", ranges, rows=True)
        intake = dict(prepared_ranges=prepared)
        registry = dict(sources={sid: dict(baseline=False) for sid in ids})
        if formal:
            intake.update(schema="badminton-formal-derived-intake-v1", rallies=len(ranges),
                          selected_scans=2 * len(ranges))
            registry["schema"] = "badminton-formal-derived-source-registry-v1"
        else:
            intake.update(sources={sid: dict(paired_unique_png_frames=0 if sid in empty else 2) for sid in ids},
                          effective_window_minutes=4)
        registry["intake"] = self.write_bound("intake.json", intake)
        self.write_bound("source_registry.json", registry)
        manifest = dict(
            calibration_digest="frozen_camera",
            declared_source_count=count, source_count=count - len(empty),
            empty_sources={sid: dict(reason="no_selected_paired_scans") for sid in empty},
            sources={sid: dict(baseline=True) for sid in ids if sid not in empty},
        )
        return registry, intake, ranges, manifest

    def test_legacy_six_source_intake_retains_real_empty_source(self):
        _, _, _, manifest = self.fixture(6, empty=("source5",))
        coverage = check_viewer_coverage(self.root, manifest)
        self.assertEqual(coverage["minimum_clips"], 10)
        self.assertEqual(coverage["empty_sources"], ["source5"])

    def test_formal_seven_sources_derive_counts_without_legacy_fields(self):
        self.fixture(7, formal=True)
        registry, intake, rows, coverage = load_viewer_intake(self.root)
        self.assertNotIn("sources", intake)
        self.assertEqual(len(registry["sources"]), 7)
        self.assertEqual(len(rows), 7)
        self.assertEqual(sum(coverage["paired_scans"].values()), 14)
        self.assertAlmostEqual(coverage["rally_window_minutes"], 700 / 60e6)
        self.assertEqual(len(coverage["evidence"]), 3)

    def test_seven_sources_need_fourteen_clips(self):
        _, _, _, manifest = self.fixture(7, formal=True)
        self.assertEqual(check_viewer_coverage(self.root, manifest)["minimum_clips"], 14)

    def test_one_new_source_needs_two_clips(self):
        _, _, _, manifest = self.fixture(1, formal=True)
        self.assertEqual(check_viewer_coverage(self.root, manifest)["minimum_clips"], 2)

    def test_api_cannot_choose_its_own_coverage(self):
        _, _, _, manifest = self.fixture(7, formal=True)
        for change in (dict(declared_source_count=6), dict(source_count=6),
                       dict(empty_sources={"source6": {}}),
                       dict(sources={sid: value for sid, value in manifest["sources"].items() if sid != "source6"})):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "coverage"):
                check_viewer_coverage(self.root, dict(manifest, **change))

    def test_changed_ranges_are_rejected_even_if_api_counts_match(self):
        _, _, _, manifest = self.fixture(7, formal=True)
        with (self.root / "ranges.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n{}")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            check_viewer_coverage(self.root, manifest)

    def test_formal_intake_totals_must_match_real_members(self):
        registry, intake, _, _ = self.fixture(7, formal=True)
        intake["selected_scans"] += 1
        registry["intake"] = self.write_bound("intake.json", intake)
        self.write_bound("source_registry.json", registry)
        with self.assertRaisesRegex(ValueError, "totals"):
            load_viewer_intake(self.root)

    def test_nonstored_body_source_is_not_admitted(self):
        _, _, _, manifest = self.fixture(7, formal=True)
        manifest["sources"]["source2"]["baseline"] = False
        with self.assertRaisesRegex(ValueError, "stored body"):
            check_viewer_coverage(self.root, manifest)

    def test_api_cannot_substitute_a_different_camera(self):
        _, _, _, manifest = self.fixture(7, formal=True)
        manifest["calibration_digest"] = "other_camera"
        with self.assertRaisesRegex(ValueError, "camera"):
            check_viewer_coverage(self.root, manifest)

    def browser_receipt(self, *, formal=True):
        _, _, _, manifest = self.fixture(7 if formal else 6, formal=formal, empty=() if formal else ("source5",))
        (self.root / "datasets").mkdir()
        (self.root / "viewer").mkdir()
        body = self.write_bound("datasets/dataset_manifest.json", dict(accepted=123))
        canonical = self.write_bound("canonical.json", dict(
            source_dataset_manifest=body["path"], source_dataset_manifest_sha256=body["sha256"],
        ))
        viewer = self.write_bound("viewer/manifest.json", manifest)
        coverage = check_viewer_coverage(self.root, manifest)
        clips = [dict(source_id=sid, passed=True, actual_begin=0.2, actual_end=5.3, unique_committed_scans=50)
                 for sid in coverage["active_sources"] for _ in range(2)]
        receipt = dict(passed=True, page_errors=[], minimum_clip_seconds=5, clips=len(clips), results=clips,
                       viewer_manifest=viewer, declared_source_count=coverage["declared_source_count"],
                       played_sources=len(coverage["active_sources"]))
        if formal:
            receipt["frozen_source_coverage"] = coverage
        return receipt, canonical

    def test_archive_gate_accepts_actual_seven_source_fourteen_clip_proof(self):
        receipt, canonical = self.browser_receipt()
        self.assertEqual(validate_browser_source_acceptance(receipt, canonical)["minimum_clips"], 14)

    def test_archive_gate_preserves_legacy_ten_clip_proof(self):
        receipt, canonical = self.browser_receipt(formal=False)
        self.assertEqual(validate_browser_source_acceptance(receipt, canonical)["minimum_clips"], 10)

    def test_archive_gate_rejects_old_ten_clip_count_for_seven_sources(self):
        receipt, canonical = self.browser_receipt()
        receipt["results"] = receipt["results"][:10]
        receipt["clips"] = 10
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_browser_source_acceptance(receipt, canonical)

    def test_archive_gate_rejects_undercovered_source_despite_fourteen_total(self):
        receipt, canonical = self.browser_receipt()
        receipt["results"][-1]["source_id"] = "source0"
        with self.assertRaisesRegex(ValueError, "every"):
            validate_browser_source_acceptance(receipt, canonical)

    def test_archive_gate_rejects_missing_frozen_formal_coverage(self):
        receipt, canonical = self.browser_receipt()
        del receipt["frozen_source_coverage"]
        with self.assertRaisesRegex(ValueError, "frozen source coverage"):
            validate_browser_source_acceptance(receipt, canonical)

    def test_archive_gate_rejects_short_or_synthetic_playback(self):
        receipt, canonical = self.browser_receipt()
        receipt["results"][0]["actual_end"] = 4
        with self.assertRaisesRegex(ValueError, "native source/time/scan"):
            validate_browser_source_acceptance(receipt, canonical)


class ActualClipSelectionTests(unittest.TestCase):
    def manifest(self, count=7):
        return dict(sources={f"s{index}": dict(rallies=[
            dict(index=r, begin=r * 20, end=r * 20 + 8) for r in range(5)
        ]) for index in range(count)})

    def test_first_last_rallies_cover_all_seven_sources(self):
        clips = complete_clip_plan(self.manifest())
        self.assertEqual(len(clips), 14)
        for sid in self.manifest()["sources"]:
            self.assertEqual([c["rally_index"] for c in clips if c["source_id"] == sid], [0, 4])

    def test_identity_focus_does_not_leave_other_sources_undercovered(self):
        focused = [dict(source_id="s0", rally_index=r, begin=r * 20 + 1, regression_case=r) for r in range(4)]
        clips = complete_clip_plan(self.manifest(), focused)
        self.assertEqual(clips[:4], focused)
        self.assertEqual(len(clips), 16)
        self.assertTrue(all(sum(c["source_id"] == sid for c in clips) >= 2 for sid in self.manifest()["sources"]))

    def test_short_rallies_cannot_become_synthetic_five_second_clips(self):
        manifest = self.manifest(1)
        manifest["sources"]["s0"]["rallies"] = [dict(index=0, begin=0, end=5)]
        with self.assertRaisesRegex(ValueError, "genuine"):
            complete_clip_plan(manifest)


class TypedViewerRelocationTests(unittest.TestCase):
    def test_formal_entry_uses_existing_typed_adapter(self):
        expected = object()
        with patch("badminton_viewer_contracts.read_json", return_value=dict(
            schema="badminton-formal-derived-source-registry-v1",
        )), patch("badminton_viewer_contracts.load_pilot_run_relocations", return_value=expected) as typed:
            self.assertIs(viewer_relocations("root"), expected)
            typed.assert_called_once_with("root", None)

    def test_partial_explicit_binding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "together"):
            viewer_relocations("root", "relocations.json")


class SealImportTests(unittest.TestCase):

    def launch(self, code, *args, cwd=None):
        root = Path(__file__).resolve().parents[1]
        return subprocess.run(
            [sys.executable, "-B", "-c", code, str(root), *map(str, args)],
            cwd=cwd or root.parent,
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    def test_spec_import_from_outside_repository_loads_exact_sibling_without_tools_path(self):
        code = """
import importlib.util, json, sys, types
from pathlib import Path
root = Path(sys.argv[1])
tools = root / 'tools'
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != tools]
sys.path.insert(0, str(root))
assert not Path.cwd().resolve().is_relative_to(root)
assert all(Path(p or '.').resolve() != tools for p in sys.path)
# An unrelated already imported module must not replace this tool's sibling.
sys.modules['badminton_viewer_contracts'] = types.SimpleNamespace(validate_browser_source_acceptance=None)
before = list(sys.path)
spec = importlib.util.spec_from_file_location('_consumer_readonly_seal_gate', tools / 'seal_badminton_expansion.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
assert sys.path == before
assert callable(gate.validate_acceptance)
assert callable(gate.validate_browser_source_acceptance)
assert Path(gate.validate_browser_source_acceptance.__code__.co_filename).resolve() == tools / 'badminton_viewer_contracts.py'
import torch
assert not torch.cuda.is_initialized()
print(json.dumps(dict(spec_import=True, exact_sibling=True, tools_on_path=False, cuda_initialized=False)))
"""
        result = self.launch(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["exact_sibling"])

    def test_real_consumer_spec_gate_import_from_consumer_repository(self):
        root = Path(__file__).resolve().parents[1]
        consumer = root.parent.parent / "LIDAR/rollout_lidar_mocap_badminton"
        if not (consumer / "src/data/gvhmr_joint_validator.py").is_file():
            self.skipTest("The independent consumer checkout is unavailable")
        code = """
import json, sys
from pathlib import Path
from types import SimpleNamespace
root, consumer = map(Path, sys.argv[1:3])
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != root / 'tools']
sys.path.insert(0, str(consumer))
from src.data.gvhmr_joint_validator import producer_seal_gate
gate = producer_seal_gate(SimpleNamespace(gvhmr_root=root))
assert callable(gate.validate_acceptance)
assert Path(gate.validate_browser_source_acceptance.__code__.co_filename).resolve() == root / 'tools/badminton_viewer_contracts.py'
import torch
assert not torch.cuda.is_initialized()
print(json.dumps(dict(real_consumer_gate=True, cuda_initialized=False)))
"""
        result = self.launch(code, consumer, cwd=consumer)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["real_consumer_gate"])

    def test_seal_cli_help_from_outside_repository(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "-B", str(root / "tools/seal_badminton_expansion.py"), "--help"],
            cwd=root.parent,
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--sequence-report", result.stdout)


if __name__ == "__main__":
    unittest.main()
