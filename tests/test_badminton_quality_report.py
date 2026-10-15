"""The audit must not count pre-job smoke or unfinished windows as throughput."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


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

    def test_completed_export_includes_selected_prejob_cache_and_excludes_smoke(self):
        from datetime import datetime, timezone

        start = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)
        settings = {"window_seconds": 60, "fit_batch_size": 128}
        pairing = {"settings": settings, "calibration_digest": "adopted"}
        progress = {"updated_utc": "2026-10-14T16:00:00+00:00"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exported = set()
            for left, right, offset, included in [(0, 6000000, -60, False),
                                                  (0, 60000000, -30, True),
                                                  (60000000, 65000000, 600, True)]:
                name = f"{left:012d}_{right:012d}"
                directory = root / "sessions/first/windows" / name
                directory.mkdir(parents=True)
                config = {"session_id": "first", "window_id": name, "left_us": left,
                          "right_us": right, "calibration_digest": "adopted", "settings": settings}
                (directory / "window.json").write_text(json.dumps(config), encoding="utf-8")
                report = directory / "fit_report.json"
                report.write_text("{}", encoding="utf-8")
                os.utime(report, (start.timestamp() + offset, start.timestamp() + offset))
                if included:
                    exported.add(("first", name))
            selected, rejected = module.completed_windows(root, pairing, progress, start.isoformat(), exported)
            self.assertEqual({("first", entry[2].name) for entry in selected}, exported)
            self.assertEqual(rejected["not_selected_for_export"], 1)
            with self.assertRaisesRegex(ValueError, "Exported windows"):
                module.completed_windows(root, pairing, progress, start.isoformat(), exported | {("first", "missing")})


class StandaloneQualityReportTests(unittest.TestCase):
    def test_completed_job_report_uses_declared_inputs_and_preserves_optional_cam1(self):
        from datetime import datetime, timezone

        start = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)
        settings = {"window_seconds": 60, "fit_batch_size": 128}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window = root / "sessions/first/windows/000000000000_000060000000"
            window.mkdir(parents=True)

            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")

            write(root / "pairing_report.json", {"settings": settings, "calibration_digest": "adopted"})
            write(root / "job_status.json", {"state": "complete", "started_utc": start.isoformat(),
                                            "completed_windows": 1})
            write(root / "progress.json", {"updated_utc": "2026-10-14T15:02:00+00:00"})
            write(root / "datasets/dataset_manifest.json", {"calibration_digest": "adopted", "accepted": 1,
                                                            "selected_windows": {"first": [window.name]}})
            write(window / "window.json", {"session_id": "first", "window_id": window.name,
                                          "left_us": 0, "right_us": 60000000, "calibration_digest": "adopted",
                                          "settings": settings, "source_pairs": [3, 4]})
            report_path = window / "fit_report.json"
            write(report_path, {"fitted": 2, "accepted": 1})
            os.utime(report_path, (start.timestamp() + 60, start.timestamp() + 60))
            rows = []
            for index, accepted, mean, p90 in ((3, True, .02, .04), (4, False, .09, .17)):
                fit_path = window / f"{index:06d}.npz"
                np.savez(fit_path, shape=np.zeros(10, dtype=np.float32))
                qa = {"finite": True, "final_p2surface_mean_m": mean, "final_p2surface_p90_m": p90,
                      "floor_penetration_m": 0., "reprojection_mean_px": 3.5,
                      "reprojection_valid_joints": {"cam2": 17}}
                rows.append({"motion_id": "first:window:track1", "lidar_index": index,
                             "fit_path": str(fit_path), "source_point_count": 120, "accepted": accepted,
                             "qa": qa, "rejection_reasons": [] if accepted else ["surface_outside_limit"]})
            (window / "quality.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            # No calibration worktree, historical experiments, models or source
            # videos are present: ordinary QC must work from these job records.
            with mock.patch.object(module, "read_json", wraps=module.read_json) as read:
                report = module.build_report(root)
            self.assertTrue(all(Path(call.args[0]).is_relative_to(root) for call in read.call_args_list))
            self.assertEqual(report["sample_counts"]["all_fitted"], 2)
            self.assertEqual(report["sample_counts"]["passed"], 1)
            self.assertAlmostEqual(report["quality"]["passed"]["surface_stages"]["final_smpl_all_triangles"]
                                   ["frame_mean_m"]["mean"], .02)
            self.assertEqual(report["efficiency"]["wall_seconds"], 120.)
            self.assertEqual(report["efficiency"]["source_seconds_per_wall_second"], .5)
            self.assertEqual(report["shape_audit"]["audited_samples"], 2)
            self.assertIsNone(report["cam1_rgb_check"])
            markdown = module.markdown_report(report)
            self.assertNotIn("old_calibration_fitting_only_benchmarks", report)
            self.assertNotIn("candidate_faces", markdown)

            cam1_path = root / "independent_rgb/cam1.json"
            cam1 = {"calibration_digest": "adopted", "completed": True, "not_ground_truth": True,
                    "not_fed_back_to_fitting": True,
                    "summary": {"checked_clips": 1, "checked_frames": 2, "measured_athlete_frames": 2,
                                "athlete_frame_candidates": 2, "eligible_joints_in_measured_frames": 34,
                                "coverage_of_candidate_17_joints": 1.,
                                "pixel_error": {"mean": 2., "median": 1.5, "p90": 3.},
                                "focal_normalized_error": {"mean": .002}}}
            write(cam1_path, cam1)
            with_cam1 = module.build_report(root, cam1_report=cam1_path)
            self.assertEqual(with_cam1["sample_counts"], report["sample_counts"])
            self.assertEqual(with_cam1["efficiency"], report["efficiency"])
            self.assertEqual(with_cam1["cam1_rgb_check"]["summary"], cam1["summary"])
            self.assertTrue(with_cam1["cam1_rgb_check"]["provenance"]["not_fed_back_to_fitting"])
            self.assertIn("2.000/1.500/3.000px", module.markdown_report(with_cam1))
            cam1["calibration_digest"] = "another-calibration"
            write(cam1_path, cam1)
            with self.assertRaisesRegex(ValueError, "different frozen calibrations"):
                module.build_report(root, cam1_report=cam1_path)


if __name__ == "__main__":
    unittest.main()
