"""Fresh-source completion preserves real gates without requiring legacy fitted bodies."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from hmr4d.lidar_mocap import new_source_completion as module
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.dual_view import file_sha


class CompletionTests(unittest.TestCase):
    def test_absolute_qa_keeps_original_acceptance_threshold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels = root / "final_labels"
            window = labels / "sessions/s/windows/w"
            write_json(labels / "repair_plan.json", dict(jobs=[dict(source_id="s", source_window="w")],
                       source_union_sha256="u", calibration_digest="c"))
            write_json(labels / "repair_receipt.json", dict(windows=1))
            qa = dict(final_p2surface_mean_m=.02, final_p2surface_p90_m=.04, reprojection_mean_px=3,
                      finite=True, floor_penetration_m=.01, reprojection_valid_joints=dict(cam2=10))
            row = dict(accepted=True, qa=qa, source_point_count=100, motion_id="m", lidar_index=0)
            window.mkdir(parents=True)
            path = window / "quality.jsonl"
            path.write_text(json.dumps(row)+"\n")
            write_json(window / "fit_report.json", dict(accepted=1, artifacts_sha256={"quality.jsonl":file_sha(path)}))
            before = module.collect_absolute_qa(root)
            self.assertEqual(len(before["frames"]), 1)
            row["qa"]["final_p2surface_mean_m"] = .0651
            path.write_text(json.dumps(row)+"\n")
            write_json(window / "fit_report.json", dict(accepted=1, artifacts_sha256={"quality.jsonl":file_sha(path)}))
            with self.assertRaisesRegex(ValueError, "unchanged QA"):
                module.collect_absolute_qa(root)
            after = copy.deepcopy(before)
            after["frames"] = {}
            with self.assertRaisesRegex(ValueError, "membership"):
                module.compare_same_v2(before, after)

    def test_pending_bilateral_observer_cannot_seal_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            labels = root / "final_labels"
            source = root / "new.json"
            write_json(source, dict(schema="badminton-new-source-repair-input-v1"))
            write_json(labels / "repair_plan.json", dict(source_union=str(source), source_union_sha256=file_sha(source), jobs=[]))
            write_json(labels / "post_refinement.json", dict(state="complete"))
            qa = dict(frames={}, windows={}, source_union_sha256="u", calibration_digest="c")
            write_json(root / "evaluation/v2_before_leg.json", qa)
            commands = []
            def run(command, **kwargs):
                name = Path(command[2]).name
                commands.append(name)
                if name == "verify_badminton_native_repair.py":
                    write_json(labels / "native_verification.json", dict(passed=True, verified_windows=0))
                if name == "observe_badminton_dual_view.py":
                    write_json(root.with_name("run_paired") / "observations/manifest.json", dict(completed=False))
                return Mock(returncode=0)
            with patch.object(module, "collect_absolute_qa", return_value=qa), patch.object(module.subprocess, "run", side_effect=run):
                module.finish_new_sources(root)
            self.assertEqual(commands, ["verify_badminton_native_repair.py", "verify_badminton_dataset.py", "observe_badminton_dual_view.py"])
            self.assertFalse((root / "new_source_completion_receipt.json").exists())
            self.assertFalse((root.with_name("run_paired") / "datasets/dataset_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
