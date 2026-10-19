from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hmr4d.lidar_mocap.expanded_dataset import binding
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json
from tools import advance_badminton_body_production as module


class BodySupervisorTests(unittest.TestCase):
    def fixture(self, workspace):
        root, source = workspace / "outputs/repair", workspace / "outputs/source"
        policy, lease = root / "policy.json", root / "lease.json"
        write_json(
            root / "new_source_input.json", dict(schema="badminton-new-source-repair-input-v1", source_root=str(source))
        )
        identity = root / "identity_evidence/identity_decisions.json"
        write_json(identity, dict(review_status="pending"))
        write_json(root / "pose_receipt.json", {})
        write_json(
            root / "production_stage.json",
            dict(
                stage="awaiting_identity_case_adjudication",
                identity_decisions=str(identity),
                identity_decisions_sha256=binding(identity)["sha256"],
            ),
        )
        write_json(policy, {})
        write_json(lease, dict(status="held", owner="GVHMR"))
        args = SimpleNamespace(root=root, source_root=source, gpu_lease=lease, abstention_policy=policy, threads=1)
        return root, args

    def test_only_explicit_reviewed_policy_advances_to_real_browser_gate(self):
        with TemporaryDirectory() as temp:
            workspace = Path(temp)
            root, args = self.fixture(workspace)
            commands = []

            def child(command, **kwargs):
                commands.append(command)
                name = Path(command[2]).name
                if name == "build_badminton_identity_decisions.py":
                    self.assertIn("--abstention-policy", command)
                    write_json(
                        root / "identity_evidence/identity_decisions.json",
                        dict(review_status="reviewed", unresolved_cases=[]),
                    )
                if name == "finish_badminton_body_repair.py":
                    artifact = root / "fixture_qa.json"
                    write_json(artifact, dict(passed=True))
                    write_json(
                        root / "new_source_completion_receipt.json",
                        dict(
                            state="labels_published_viewer_prepared",
                            accepted=3,
                            evidence={str(artifact): binding(artifact)["sha256"]},
                        ),
                    )
                return SimpleNamespace(returncode=0)

            with (
                patch.object(module, "PROJ_ROOT", workspace),
                patch.object(module.subprocess, "run", side_effect=child),
            ):
                module.execute(args)
            self.assertEqual(len(commands), 4)
            fitting = next(c for c in commands if Path(c[2]).name == "run_badminton_repair_windows.py")
            self.assertIn("--union", fitting)
            finish = next(c for c in commands if Path(c[2]).name == "finish_badminton_body_repair.py")
            self.assertEqual(finish[finish.index("--observer-max-new-frames") + 1], "1000000")
            state = read_json(root / "production_stage.json")
            self.assertEqual(state["stage"], "awaiting_browser_playback_verification")
            self.assertFalse(state["training_published"])
            self.assertFalse(state["browser_playback_verified"])
            self.assertEqual(read_json(args.gpu_lease)["status"], "held")

    def test_changed_identity_on_resume_stops_before_any_model_process(self):
        with TemporaryDirectory() as temp:
            workspace = Path(temp)
            root, args = self.fixture(workspace)
            identity = root / "identity_evidence/identity_decisions.json"
            write_json(
                root / "body_production_inputs.json",
                dict(
                    source_input=binding(root / "new_source_input.json"),
                    identity_decisions=binding(identity),
                    policy=binding(args.abstention_policy),
                    native_pose_receipt=binding(root / "pose_receipt.json"),
                ),
            )
            write_json(identity, dict(changed=True))
            with patch.object(module, "PROJ_ROOT", workspace), patch.object(module.subprocess, "run") as runner:
                with self.assertRaisesRegex(ValueError, "dependency changed"):
                    module.execute(args)
            runner.assert_not_called()
            self.assertEqual(read_json(root / "production_stage.json")["stage"], "failed")


if __name__ == "__main__":
    unittest.main()
