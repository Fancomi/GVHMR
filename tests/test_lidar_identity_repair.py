import numpy as np
import unittest

from hmr4d.lidar_mocap.identity_repair import apply_identity_decisions


def observation(pos, old, height=100):
    return dict(
        position=pos, original_track_id=old, bbx_xyxy=np.array([0, 0, 50, height], np.float32), detection_confidence=0.9
    )


class IdentityRepairTests(unittest.TestCase):
    def test_mixed_tail_and_duplicate_do_not_enter_temporal_context(self):
        tracks = {
            1: [observation(0, 1), observation(1, 1), observation(2, 1)],
            2: [observation(1, 2, 50), observation(2, 2, 50)],
        }
        decision = dict(
            links=[dict(tracks=[1, 2])], drop_spans=[dict(track_id=1, begin=2, end=9, reason="mixed_person")]
        )
        result, receipt = apply_identity_decisions(tracks, decision)
        assert [(o["position"], o["original_track_id"]) for o in result[1]] == [(0, 1), (1, 1), (2, 2)]
        assert len(receipt["removed_observations"]) == len(receipt["duplicate_observations"]) == 1
        assert np.array_equal(result[1][-1]["bbx_xyxy"], tracks[2][-1]["bbx_xyxy"])

    def test_distinct_people_cannot_be_joined_transitively(self):
        tracks = {i: [observation(0, i)] for i in [1, 2, 3]}
        with self.assertRaisesRegex(ValueError, "witnessed people"):
            apply_identity_decisions(
                tracks, dict(links=[dict(tracks=[1, 2]), dict(tracks=[2, 3])], cannot_link=[[1, 3]])
            )

    def test_native_review_can_undo_a_wrong_single_camera_merge(self):
        tracks={11:[observation(0,11)],23:[observation(0,23),observation(1,25),observation(3,29)],27:[observation(2,27)]}
        decision=dict(reset_tracklets=[23,25,29],links=[dict(tracks=[11,25]),dict(tracks=[25,29]),dict(tracks=[23,27])],cannot_link=[[11,23]])
        result,_=apply_identity_decisions(tracks,decision)
        assert {o['original_track_id'] for o in result[11]}=={11,25,29}
        assert {o['original_track_id'] for o in result[23]}=={23,27}
