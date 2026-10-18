from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from hmr4d import PROJ_ROOT
from tools import advance_badminton_new_source as supervisor


class FreshSupervisorFailureTests(unittest.TestCase):
    def run_failure(self, root, state):
        args = [
            "advance",
            "--root",
            str(root),
            "--source-root",
            str(PROJ_ROOT / "outputs/source"),
            "--gpu-lease",
            str(PROJ_ROOT / "outputs/lease.json"),
        ]
        with (
            patch.object(sys, "argv", args),
            patch.object(supervisor, "execute", side_effect=ValueError("bad receipt")),
            patch.object(Path, "exists", return_value=True),
            patch.object(supervisor, "read_json", return_value=state),
            patch.object(supervisor, "write_json") as writer,
        ):
            with self.assertRaisesRegex(ValueError, "bad receipt"):
                supervisor.main()
            return writer

    def test_validation_failure_persists_failed_state_for_cpu_monitor(self):
        state = dict(schema="badminton-fresh-production-stage-v1", stage="waiting_initial_teachers")
        writer = self.run_failure(PROJ_ROOT / "outputs/fake_new_source", state)
        self.assertEqual(writer.call_args.args[1]["stage"], "failed")
        self.assertEqual(writer.call_args.args[1]["failed_stage"], "waiting_initial_teachers")
        self.assertEqual(writer.call_args.args[1]["error_type"], "ValueError")

    def test_foreign_state_is_not_rewritten(self):
        writer = self.run_failure(PROJ_ROOT / "outputs/fake_new_source", dict(schema="other", stage="complete"))
        writer.assert_not_called()

    def test_non_output_path_is_not_written_on_failure(self):
        state = dict(schema="badminton-fresh-production-stage-v1", stage="waiting_initial_teachers")
        writer = self.run_failure(PROJ_ROOT / "docs", state)
        writer.assert_not_called()
