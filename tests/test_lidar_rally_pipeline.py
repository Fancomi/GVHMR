"""CPU edge cases for exact rally selection and missing original frame counters."""

import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import tempfile

import numpy as np

from hmr4d.lidar_mocap.rally_pipeline import select_window_pairs, validate_session_splits, RallyPipeline
from hmr4d.lidar_mocap.teacher import _normalize_rows, sample_track


class RallySelectionTests(unittest.TestCase):
    def test_new_source_split_is_disjoint_and_exhaustive(self):
        good = dict(train=["a", "b", "c", "d"], val=["e"], test=["f"])
        self.assertEqual(validate_session_splits(good, "abcdef"), good)
        for bad in (dict(train=["a"], val=["a"], test=["f"]),
                    dict(train=["a", "b", "c"], val=["e"], test=["f"]),
                    dict(train=["a", "b", "c", "d"], val=[], test=["e", "f"])):
            with self.assertRaises(ValueError):
                validate_session_splits(bad, "abcdef")

    def test_independent_prepare_never_reads_missing_legacy_baseline(self):
        from hmr4d.lidar_mocap import rally_pipeline as module
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pipeline = RallyPipeline.__new__(RallyPipeline)
            pipeline.output_root = root / "run"
            pipeline.baseline_root = root / "does-not-exist"
            pipeline.reuse_baseline = False
            pipeline.calibration_dir = Path("config/calibration/badminton-20261015")
            pipeline.lidar_project_root = root / "lidar"
            pipeline.sessions_root = root / "sessions"
            pipeline.session_splits = dict(train=["a"], val=["b"], test=["c"])
            pipeline.data_root = root
            pipeline.intake_path = root / "intake.json"
            pipeline.intake_path.write_text("{}")
            pipeline.intake = dict(acquisition_manifest_sha256="a", rally_inventory_sha256="b",
                                   effective_window_minutes=1, sources={})
            pipeline.rallies = []
            pipeline.source_entries = {}
            frozen = Mock(data=dict(snapshot_digest="new"))
            with patch.object(module.CalibrationSnapshot, "freeze", return_value=frozen) as freeze:
                state = pipeline.prepare()
                self.assertEqual(state["state"], "prepared")
                self.assertEqual(freeze.call_args.kwargs["calibration_dir"], pipeline.calibration_dir)
                pipeline.session_splits = dict(train=["b"], val=["a"], test=["c"])
                with self.assertRaisesRegex(ValueError, "split changed"):
                    pipeline.prepare()

    def test_exact_union_retains_terminal_midpoint_and_skips_rest_frames(self):
        pairs = [dict(lidar_index=i, scan_mid_us=1000 + i * 100000, status="paired") for i in range(8)]
        # The last scan midpoint is later than the inventory's scan-start
        # accounting endpoint. Only explicit PNG membership selects fitting.
        windows = select_window_pairs(pairs, {1, 2, 5, 6}, seconds=0.3)
        self.assertEqual([p["lidar_index"] for w in windows for p in w[3]], [1, 2, 5, 6])
        self.assertEqual(len({p["lidar_index"] for w in windows for p in w[3]}), 4)

    def test_lost_pairing_or_missing_png_fails_instead_of_silent_drop(self):
        pairs = [dict(lidar_index=1, scan_mid_us=1000, status="excluded")]
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            select_window_pairs(pairs, {1})
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            select_window_pairs([], {2})

    def test_optional_original_counter_never_fabricates_native_timing(self):
        rows = [
            dict(record_index=i * 2, common_time_us=1000000 + i * 33334, media_pts_us=i * 33334, source_frame_no=None)
            for i in range(3)
        ]
        normalized = _normalize_rows(rows)
        self.assertTrue(all(r["source_frame_no"] is None for r in normalized))
        self.assertEqual(normalized[1]["record_index"], 2)
        with self.assertRaises(TypeError):
            _normalize_rows([{**rows[0], "common_time_us": None}])


if __name__ == "__main__":
    unittest.main()
