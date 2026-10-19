"""Identity adapters preserve ambiguity and never inherit scene-specific exceptions."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hmr4d.lidar_mocap import new_source_identity as module
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.identity_repair import apply_identity_decisions


class IdentityTests(unittest.TestCase):
    def test_explicit_abstention_splits_unproven_existing_links_without_claiming_different_people(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "identity_evidence"
            source, inventory = root / "input.json", root / "inventory.json"
            write_json(source, {})
            write_json(inventory, {})
            write_json(root / "case_plan.json", dict(source_union=str(source), source_union_sha256=file_sha(source)))
            write_json(
                evidence / "case_plan.json",
                dict(boundary_inventory=str(inventory), boundary_inventory_sha256=file_sha(inventory)),
            )
            window = root / "initial/w"
            write_json(window / "teacher/manifest.json", {})
            case = dict(
                index=0,
                source_window=str(window),
                positions=[0, 1],
                context_positions=[0, 1],
                event=dict(source_id="s", before_motion_id="s:w:7:0", after_motion_id="s:w:8:0"),
            )
            write_json(evidence / "boundaries.json", [case])
            write_json(evidence / "pose_receipt.json", dict(state="complete", config=dict(calibration_digest="c")))
            write_json(
                evidence / "native_identity_review.json",
                dict(
                    partial=False,
                    cases=[
                        dict(
                            case,
                            status="insufficient_or_ambiguous_native_evidence",
                            votes=[dict(original_track_id=7), dict(original_track_id=8)],
                        )
                    ],
                    inputs={str(source): file_sha(source)},
                ),
            )
            write_json(
                evidence / "boundary_inventory.pending.json",
                dict(
                    source_union_sha256=file_sha(source),
                    tracklets=[dict(source_id="s", window="w", original_track_id=i) for i in (7, 8)],
                ),
            )
            policy = root / "policy.json"
            write_json(
                policy,
                dict(
                    schema="badminton-new-source-identity-abstention-v1",
                    review_status="reviewed",
                    action="keep_unproven_original_tracklets_separate",
                    source_union_sha256=file_sha(source),
                    boundary_inventory_sha256=file_sha(inventory),
                    evidence_files={str(source): file_sha(source)},
                ),
            )
            with patch.object(module, "published_windows", return_value=[("s", window)]):
                result = module.seal_identity(root, policy)
            self.assertEqual(result["state"], "reviewed")
            self.assertEqual(result["abstained"], 1)
            decisions = module.read_json(evidence / "identity_decisions.json")
            self.assertEqual(decisions["abstained_relationship_cases"], [0])
            target = decisions["windows"]["s:w"]
            self.assertEqual(target["links"], [])
            self.assertEqual(target["cannot_link"], [])
            self.assertEqual(target["reset_tracklets"], [7, 8])
            tracks = {
                7: [
                    dict(position=i, original_track_id=k, bbx_xyxy=np.array([0, 0, 10, 30]))
                    for i, k in enumerate((7, 8))
                ]
            }
            separated, _ = apply_identity_decisions(tracks, target)
            self.assertEqual(set(separated), {7, 8})
            source.write_text('{"changed":true}')
            with patch.object(module, "published_windows", return_value=[("s", window)]):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    module.seal_identity(root, policy)

    def test_endpoint_inventory_is_pending_and_has_no_implicit_decisions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.json"
            write_json(source, {})
            write_json(root / "case_plan.json", dict(source_union=str(source), source_union_sha256=file_sha(source)))
            window = root / "initial/w"
            write_json(window / "teacher/manifest.json", dict(provenance=dict(frame_rows=[dict(record_index=0)])))
            with (
                patch.object(module, "published_windows", return_value=[("s", window)]),
                patch.object(module, "native_teacher_mapping", return_value=np.array([[0, 0, 7, 7]])),
            ):
                result = module.prepare_identity(root)
            self.assertEqual(result["state"], "pending_boundary_inventory")
            inventory = module.read_json(result["path"])
            self.assertEqual(inventory["cases"], [])
            self.assertEqual(inventory["review_status"], "pending")
            self.assertFalse((root / "identity_evidence/identity_decisions.json").exists())

    def test_unknown_stays_pending_and_known_status_has_no_manual_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "identity_evidence"
            source = root / "input.json"
            write_json(source, {})
            write_json(root / "case_plan.json", dict(source_union=str(source), source_union_sha256=file_sha(source)))
            inventory = root / "inventory.json"
            write_json(inventory, {})
            write_json(
                evidence / "case_plan.json",
                dict(boundary_inventory=str(inventory), boundary_inventory_sha256=file_sha(inventory)),
            )
            window = root / "initial/w"
            write_json(window / "teacher/manifest.json", {})
            case = dict(
                index=0,
                source_window=str(window),
                positions=[0, 1],
                context_positions=[0, 1],
                event=dict(source_id="s", before_motion_id="s:w:7:0", after_motion_id="s:w:8:0"),
            )
            write_json(evidence / "boundaries.json", [case])
            write_json(evidence / "pose_receipt.json", dict(state="complete", config=dict(calibration_digest="c")))
            for status, expected in (
                ("insufficient_or_ambiguous_native_evidence", "pending"),
                ("different_native_people", "reviewed"),
            ):
                write_json(
                    evidence / "native_identity_review.json",
                    dict(
                        partial=False,
                        cases=[dict(case, status=status, votes=[dict(original_track_id=7), dict(original_track_id=8)])],
                        inputs={str(source): file_sha(source)},
                    ),
                )
                with patch.object(module, "published_windows", return_value=[("s", window)]):
                    self.assertEqual(module.seal_identity(root)["state"], expected)
            decisions = module.read_json(evidence / "identity_decisions.json")
            self.assertEqual(decisions["windows"]["s:w"]["cannot_link"], [[7, 8]])
            self.assertEqual(decisions["windows"]["s:w"]["links"], [])


if __name__ == "__main__":
    unittest.main()
