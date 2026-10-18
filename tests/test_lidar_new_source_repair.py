"""CPU-only tests for fresh-source repair planning and immutable window boundaries."""
import tempfile
import unittest
import runpy
from pathlib import Path
from unittest.mock import Mock, patch

from hmr4d.lidar_mocap import new_source_repair as module
from hmr4d.lidar_mocap.body_repair import published_windows
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.dual_view import file_sha


class NewSourceTests(unittest.TestCase):
    def test_prepare_no_old_fit_or_gpu_and_rejects_changed_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source = workspace / "outputs/source"
            root = workspace / "outputs/repair"
            write_json(source / "calibration/calibration_snapshot.json", {})
            ranges = source / "ranges.jsonl"
            ranges.write_text('{"source_id":"s","eligible_paired_png_indices":[0,1]}\n')
            intake = source / "intake.json"
            write_json(intake, dict(prepared_ranges=dict(path=str(ranges), sha256=file_sha(ranges))))
            registry = dict(intake=dict(path=str(intake), sha256=file_sha(intake)), sources=dict(s=dict(baseline=False)))
            write_json(source / "source_registry.json", registry)
            write_json(source / "session_splits.json", dict(train=["s"], val=["v"], test=["t"]))
            cal = Mock(data=dict(snapshot_digest="frozen", sessions_root=str(workspace / "missing"),
                                 lidar_project_root=str(workspace / "lidar")))
            session = Mock()
            session.pairs.return_value = [dict(lidar_index=i, status="paired", scan_mid_us=1000000+i*100000)
                                          for i in range(2)]
            pipe = Mock(settings=dict(camera="cam2", teacher_stride=2, batch_size=4))
            with patch.object(module, "PROJ_ROOT", workspace), patch.object(module, "BadmintonPipeline", return_value=pipe), \
                 patch.object(module.CalibrationSnapshot, "load", return_value=cal), \
                 patch.object(module, "load_run_session", return_value=session):
                first = module.prepare_new_sources(source, root)
                self.assertEqual(module.prepare_new_sources(source, root), first)
                self.assertEqual(first["identity_status"], "pending_native_detection_evidence")
                jobs = published_windows(root / "new_source_input.json")
                self.assertEqual(len(jobs), 1)
                self.assertFalse((jobs[0][1] / "fit_report.json").exists())
                (jobs[0][1] / "window.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "window changed"):
                    published_windows(root / "new_source_input.json")
                with self.assertRaisesRegex(ValueError, "input changed"):
                    module.prepare_new_sources(source, root)

    def test_worker_forwards_explicit_new_input_source_and_thread_limit(self):
        import subprocess
        from hmr4d import PROJ_ROOT
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "new.json"
            write_json(input_path, dict(schema="badminton-new-source-repair-input-v1"))
            write_json(root / "final_labels/repair_plan.json", dict(source_union=str(input_path), jobs=[]))
            script = PROJ_ROOT / "tools/run_badminton_repair_windows.py"
            argv = [str(script), "--root", str(root), "--union", str(input_path),
                    "--source-root", str(root / "source"), "--threads", "1"]
            with patch("sys.argv", argv), patch.object(subprocess, "run", return_value=Mock(returncode=0)) as launch, \
                 patch("hmr4d.lidar_mocap.repair_execution.record_execution"):
                runpy.run_path(str(script), run_name="__main__")
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index("--union")+1], str(input_path.resolve()))
            self.assertEqual(command[command.index("--source-root")+1], str((root / "source").resolve()))
            self.assertEqual(command[command.index("--threads")+1], "1")
            argv = [str(script), "--root", str(root)]
            with patch("sys.argv", argv), patch.object(subprocess, "run") as launch, \
                 patch("hmr4d.lidar_mocap.repair_execution.record_execution"):
                with self.assertRaisesRegex(ValueError, "explicit --union"):
                    runpy.run_path(str(script), run_name="__main__")
                launch.assert_not_called()

    def test_frozen_writer_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            module.write_frozen(path, dict(identity_status="pending"))
            with self.assertRaises(ValueError):
                module.write_frozen(path, dict(identity_status="approved"))


if __name__ == "__main__":
    unittest.main()
