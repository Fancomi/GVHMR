import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hmr4d.lidar_mocap.repair_execution import record_execution, repair_efficiency


class RepairExecutionTests(unittest.TestCase):
    def test_resume_preserves_original_launch_and_baseline_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch("hmr4d.lidar_mocap.repair_execution.time.time", return_value=100):
                record_execution(root)
            with patch("hmr4d.lidar_mocap.repair_execution.time.time", return_value=160):
                record_execution(root, baseline_finished=True)
            with patch("hmr4d.lidar_mocap.repair_execution.time.time", return_value=220):
                record_execution(root, baseline_finished=True)
                efficiency = repair_efficiency(root, 600, 120)
            self.assertEqual(efficiency["baseline_fit_wall_seconds"], 60)
            self.assertEqual(efficiency["wall_seconds"], 120)
            self.assertEqual(efficiency["fitted_person_scans_per_wall_second"], 5)

    def test_disposable_logs_are_not_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "execution_timing.json").write_text(
                json.dumps(dict(started_unix_s=100, baseline_finished_unix_s=150)), encoding="utf-8"
            )
            with patch("hmr4d.lidar_mocap.repair_execution.time.time", return_value=200):
                self.assertEqual(repair_efficiency(root, 100, 50)["wall_seconds"], 100)

    def test_clock_reversal_cannot_publish_negative_efficiency(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "execution_timing.json").write_text(
                json.dumps(dict(started_unix_s=100, baseline_finished_unix_s=150)), encoding="utf-8"
            )
            with patch("hmr4d.lidar_mocap.repair_execution.time.time", return_value=140):
                with self.assertRaisesRegex(ValueError, "out of order"):
                    repair_efficiency(root, 100, 50)
