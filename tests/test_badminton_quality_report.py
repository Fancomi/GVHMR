"""The audit must not count pre-job smoke or unfinished windows as throughput."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "badminton_quality_report", Path(__file__).resolve().parents[1] / "tools/report_badminton_quality.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CompletedWindowScopeTests(unittest.TestCase):
    def test_reject_prejob_smoke_future_report_and_keep_short_terminal_window(self):
        # A duration-limited smoke uses window_seconds=60 too. Checking only
        # the configured length would inflate this job's measured throughput.
        from datetime import datetime, timezone

        start = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)
        watermark = datetime(2026, 10, 14, 16, 0, tzinfo=timezone.utc)
        settings = {"window_seconds": 60, "fit_batch_size": 128}
        pairing = {"settings": settings, "calibration_digest": "adopted"}
        progress = {"updated_utc": watermark.isoformat()}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def add(sid, left, right, modified, batch=128):
                directory = root / "sessions" / sid / "windows" / f"{left:012d}_{right:012d}"
                directory.mkdir(parents=True)
                config = {"session_id": sid, "window_id": directory.name, "left_us": left,
                          "right_us": right, "calibration_digest": "adopted",
                          "settings": dict(settings, fit_batch_size=batch)}
                (directory / "window.json").write_text(json.dumps(config), encoding="utf-8")
                path = directory / "fit_report.json"
                path.write_text("{}", encoding="utf-8")
                os.utime(path, (modified, modified))

            add("first", 0, 6000000, start.timestamp() - 60, batch=8)
            add("second", 0, 6000000, start.timestamp() - 30)  # same batch, pre-job
            add("first", 0, 60000000, start.timestamp() + 600)
            add("first", 60000000, 65000000, start.timestamp() + 1200)  # real terminal
            add("first", 65000000, 125000000, watermark.timestamp() + 1)  # newly complete
            selected, rejected = module.completed_windows(root, pairing, progress, start.isoformat())
            self.assertEqual([(entry[0]["left_us"], entry[0]["right_us"]) for entry in selected],
                             [(0, 60000000), (60000000, 65000000)])
            self.assertEqual(rejected["different_job_settings"], 1)
            self.assertEqual(rejected["completed_before_current_job"], 1)
            self.assertEqual(rejected["completed_after_progress_watermark"], 1)

    def test_naive_timestamp_is_not_silently_treated_as_utc(self):
        with self.assertRaisesRegex(ValueError, "UTC offset"):
            module.parse_utc("2026-10-14T16:00:00")


if __name__ == "__main__":
    unittest.main()
