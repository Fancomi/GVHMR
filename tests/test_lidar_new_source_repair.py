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

    def test_formal_plan_requires_actual_equivalence_and_preserves_native_rally_windows(self):
        import json

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            source, root = workspace / "outputs/source", workspace / "outputs/repair"
            camera = source / "calibration/calibration_snapshot.json"
            write_json(camera, {})
            roles = dict(train=["s"], val=["v"], test=["t"])
            ranges = source / "ranges.jsonl"
            ranges.write_text("".join(json.dumps(dict(source_id=sid, eligible_paired_png_indices=[0, 1]))+"\n"
                                      for ids in roles.values() for sid in ids))
            intake = source / "intake.json"
            write_json(intake, dict(prepared_ranges=dict(path=str(ranges), sha256=file_sha(ranges))))
            registry_path = source / "source_registry.json"
            registry = dict(schema="badminton-formal-derived-source-registry-v1",
                intake=dict(path=str(intake), sha256=file_sha(intake)), camera_admission_handoff=dict(path="bound"),
                sources={sid: dict(baseline=False) for ids in roles.values() for sid in ids})
            write_json(registry_path, registry)
            write_json(source / "session_splits.json", roles)
            cal = Mock(data=dict(snapshot_digest="formal", sessions_root=str(workspace / "sessions"),
                                 lidar_project_root=str(workspace / "lidar")))
            session = Mock()
            pairs = [dict(lidar_index=i, status="paired", scan_mid_us=1000000+i*100000) for i in range(2)]
            session.pairs.return_value = pairs
            windows = [(str(i), i*100000, (i+1)*100000, [pair]) for i, pair in enumerate(pairs)]
            pipe = Mock(settings=dict(camera="cam2", teacher_stride=2, batch_size=4))
            proof_path = source / "actual_equivalence.json"
            write_json(proof_path, dict(schema="badminton-review-formal-input-equivalence-v1", state="complete",
                evidence=dict(formal_registry=dict(path=str(registry_path), sha256=file_sha(registry_path)),
                              formal_calibration=dict(path=str(camera), sha256=file_sha(camera))),
                sources={sid: {} for sid in registry["sources"]}, role_assignment=roles))
            proof = dict(path=str(proof_path), sha256=file_sha(proof_path))
            with (patch.object(module, "PROJ_ROOT", workspace),
                  patch.object(module, "BadmintonPipeline", return_value=pipe),
                  patch.object(module.CalibrationSnapshot, "load", return_value=cal),
                  patch.object(module, "load_pilot_run_relocations", return_value=None),
                  patch.object(module, "load_run_session", return_value=session),
                  patch("hmr4d.lidar_mocap.pilot.review_rally_windows", return_value=windows)):
                with self.assertRaisesRegex(ValueError, "requires its completed actual input-equivalence"):
                    module.prepare_new_sources(source, root)
                self.assertFalse(root.exists())
                plan = module.prepare_new_sources(source, root, formal_equivalence=proof)
                self.assertEqual(len(plan["jobs"]), 6)
                self.assertEqual(plan["required_nonempty_splits"], ["train", "val", "test"])
                self.assertEqual(plan["formal_derivation"]["input_equivalence"], proof)
                self.assertNotIn("review_only", plan)
                self.assertFalse(plan["training_published"])


if __name__ == "__main__":
    unittest.main()
