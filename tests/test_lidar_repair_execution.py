import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hmr4d.lidar_mocap.repair_execution import (
    record_execution,
    repair_efficiency,
    require_gpu_lease,
    write_progress_json,
)


class RepairExecutionTests(unittest.TestCase):
    def test_transient_windows_progress_lock_retries_identical_atomic_payload(self):
        payload = dict(state="running", computed=1825)
        with (
            patch(
                "hmr4d.lidar_mocap.pipeline.write_json", side_effect=[PermissionError(), PermissionError(), None]
            ) as writer,
            patch("hmr4d.lidar_mocap.repair_execution.time.sleep") as sleep,
        ):
            write_progress_json("progress.json", payload)
            self.assertEqual(writer.call_count, 3)
            self.assertTrue(all(call.args == ("progress.json", payload) for call in writer.call_args_list))
            self.assertEqual(sleep.call_count, 2)

    def test_persistent_lock_and_other_io_errors_are_not_hidden(self):
        with (
            patch("hmr4d.lidar_mocap.pipeline.write_json", side_effect=PermissionError()) as writer,
            patch("hmr4d.lidar_mocap.repair_execution.time.sleep"),
        ):
            with self.assertRaises(PermissionError):
                write_progress_json("progress.json", {}, attempts=3)
            self.assertEqual(writer.call_count, 3)
        with patch("hmr4d.lidar_mocap.pipeline.write_json", side_effect=OSError("disk full")) as writer:
            with self.assertRaisesRegex(OSError, "disk full"):
                write_progress_json("progress.json", {})
            self.assertEqual(writer.call_count, 1)

    def test_foreign_gpu_lease_owner_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lease.json"
            path.write_text(json.dumps(dict(status="held", owner="Sonata")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not held by GVHMR"):
                require_gpu_lease(path)

    def test_released_gpu_lease_cannot_start_a_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lease.json"
            path.write_text(json.dumps(dict(status="released", owner="GVHMR")), encoding="utf-8")
            with self.assertRaises(ValueError):
                require_gpu_lease(path)

    def test_held_production_gpu_lease_is_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "lease.json"
            path.write_text(json.dumps(dict(status="held", owner="GVHMR")), encoding="utf-8")
            require_gpu_lease(path)

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
