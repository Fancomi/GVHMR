import unittest

from hmr4d.lidar_mocap.track_colors import PALETTE, track_color_registry


class TrackColorTests(unittest.TestCase):
    def test_person_remains_blue_when_another_track_disappears(self):
        a, b, c, d = [f"source:window:{i}:0" for i in (8, 12, 16, 20)]
        rows = lambda *ids: [dict(motion_id=i) for i in ids]
        frames = {1523: rows(a, b, c), 1533: rows(a, b), 1535: rows(b, d), 1540: rows(d, b)}
        colors = track_color_registry(frames)
        self.assertEqual(colors[b], PALETTE[1])
        self.assertEqual(colors[a], PALETTE[0])
        self.assertEqual(colors[d], PALETTE[0])
        self.assertNotEqual(colors[b], colors[d])

    def test_gap_and_segment_do_not_reassign_same_detector_track(self):
        a, b, again = "s:w:1:0", "s:w:2:0", "s:w:1:1"
        colors = track_color_registry({0: [dict(motion_id=a)], 1: [dict(motion_id=b)], 2: [dict(motion_id=again)]})
        self.assertEqual(colors[a], colors[again])
        self.assertNotEqual(colors[a], colors[b])

    def test_more_than_four_simultaneous_tracks_have_distinct_colors(self):
        keys = [f"s:w:{i}:0" for i in range(8)]
        colors = track_color_registry({0: [dict(motion_id=k) for k in keys]})
        self.assertEqual(len(set(colors.values())), 8)


if __name__ == "__main__":
    unittest.main()
