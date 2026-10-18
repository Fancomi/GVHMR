"""CPU edge cases for exact rally selection and missing original frame counters."""

import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import tempfile

import numpy as np

from hmr4d.lidar_mocap.rally_pipeline import select_window_pairs, validate_session_splits, validate_split_windows, RallyPipeline
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

    def test_prepared_windows_reject_empty_test_but_allow_empty_individual_source(self):
        pairs = [dict(lidar_index=i, scan_mid_us=1000000 + i * 100000, status="paired") for i in range(3)]
        sources = {sid: dict(fit_windows=len(select_window_pairs(pairs, eligible)))
                   for sid, eligible in dict(a={0}, b={1}, c={2}, empty=set()).items()}
        with self.assertRaisesRegex(ValueError, "split.*test"):
            validate_split_windows(dict(train=["a", "b"], val=["c"], test=["empty"]), sources)
        counts = validate_split_windows(dict(train=["a", "empty"], val=["b"], test=["c"]), sources)
        self.assertEqual(counts, dict(train=1, val=1, test=1))
        for name in ("train", "val", "test"):
            splits = dict(train=["a"], val=["b"], test=["c"])
            splits[name] = ["empty"]
            with self.assertRaisesRegex(ValueError, name):
                validate_split_windows(splits, sources)

    def test_required_partitions_recheck_actual_body_and_bilateral_counts(self):
        from hmr4d.lidar_mocap.export import require_nonempty_splits
        dataset = dict(exports={sid:dict(accepted=count) for sid,count in dict(a=2,b=1,c=1,empty=0).items()},
                       suggested_session_splits=dict(train=dict(sessions=["a","empty"]),
                                                     val=dict(sessions=["b"]), test=dict(sessions=["c"])),
                       required_nonempty_splits=["train","val","test"])
        self.assertEqual(require_nonempty_splits(dataset), dict(train=2,val=1,test=1))
        dataset["exports"]["c"]["accepted"] = 0  # No body QA or no final bilateral admission.
        with self.assertRaisesRegex(ValueError, "required split: test"):
            require_nonempty_splits(dataset)
        dataset.pop("required_nonempty_splits")
        self.assertEqual(require_nonempty_splits(dataset), {})  # Historical manifests remain compatible.

    def test_zero_rally_source_exports_empty_real_tensor_and_metadata(self):
        import joblib
        from hmr4d.lidar_mocap.pipeline import BadmintonPipeline
        with tempfile.TemporaryDirectory() as temp:
            pipeline = BadmintonPipeline.__new__(BadmintonPipeline)
            pipeline.output_root = Path(temp)
            pipeline.calibration = Mock(data=dict(sessions={"empty": {}}, snapshot_digest="test"))
            with patch.object(pipeline, "_load"):
                result = pipeline.aggregate(["empty"])
            entry = result["exports"]["empty"]
            self.assertEqual(entry["accepted"], 0)
            self.assertEqual(result["selected_windows"]["empty"], [])
            self.assertEqual(tuple(joblib.load(entry["path"])["pc"].shape), (0, 256, 3))
            self.assertEqual(Path(entry["metadata"]).read_text(), "")

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
            with patch.object(module.CalibrationSnapshot, "freeze", return_value=frozen):
                with self.assertRaisesRegex(ValueError, "No usable prepared"):
                    pipeline.prepare()
                self.assertFalse((pipeline.output_root / "session_splits.json").exists())
                self.assertFalse((pipeline.output_root / "build_status.json").exists())
            with patch.object(module.CalibrationSnapshot, "freeze", return_value=frozen) as freeze, \
                 patch.object(module, "validate_split_windows", return_value=dict(train=1, val=1, test=1)):
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
