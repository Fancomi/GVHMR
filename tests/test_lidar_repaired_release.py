import unittest

from tools.seal_badminton_repaired_release import real_sequences


def row(scan, time, motion="session:window:12:0"):
    return dict(lidar_index=scan, scan_mid_us=time, motion_id=motion)


class RepairedSequenceTests(unittest.TestCase):
    def test_missing_scan_splits_sequence_without_inventing_a_sample(self):
        rows = [row(13, 1300000), row(10, 1000000), row(11, 1100000)]
        chunks = real_sequences(rows)
        self.assertEqual([c["scans"] for c in chunks], [[10, 11], [13]])
        self.assertEqual(sorted(i for c in chunks for i in c["indices"]), [0, 1, 2])

    def test_track_segments_are_never_stitched(self):
        chunks = real_sequences([row(10, 1000000), row(11, 1100000, "session:window:12:1")])
        self.assertEqual(len(chunks), 2)

    def test_large_time_gap_breaks_even_consecutive_indices(self):
        chunks = real_sequences([row(10, 1000000), row(11, 1300000)])
        self.assertEqual([c["scans"] for c in chunks], [[10], [11]])
