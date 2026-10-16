"""CPU edge cases for exact rally selection and missing original frame counters."""

import unittest

import numpy as np

from hmr4d.lidar_mocap.rally_pipeline import select_window_pairs
from hmr4d.lidar_mocap.teacher import _normalize_rows, sample_track


class RallySelectionTests(unittest.TestCase):
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
