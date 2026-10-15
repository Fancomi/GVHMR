import copy
from types import SimpleNamespace
import unittest

from hmr4d.lidar_mocap.rally_inventory import camera_bound,union_intervals,verify_rally_range


class NativeRallyRangeTests(unittest.TestCase):
    def setUp(self):
        self.pairs=[]
        for i in range(4):
            start=1000+i*1000
            cameras={cam:dict(status='matched',record_index=10*i+j,media_pts_us=10000*i+j,
                              common_time_us=start+200+j,time_error_us=j) for j,cam in enumerate(('cam1','cam2'))}
            self.pairs.append(dict(lidar_index=i,lidar_frame_id=101+i,source_frame_map_lidar_frame=10+i,
                                   scan_start_us=start,scan_mid_us=start+200,cameras=cameras,status='paired'))
        self.session=SimpleNamespace(pairs=lambda:self.pairs,common_clock=dict(common_overlap_begin_us=1100,common_overlap_end_us=4500))
        self.clip=dict(source_id='recording',index=1,original_rally_index=7,source_begin_frame=1,source_frame_count=2,
                       common_begin_us=2000,common_end_us=3000,both_camera_matched_frames=2,
                       camera_bounds={cam:dict(first=camera_bound(self.pairs[1]['cameras'][cam]),last=camera_bound(self.pairs[2]['cameras'][cam])) for cam in ('cam1','cam2')})

    def test_exact_png_range_retains_terminal_scan_after_accounting_end(self):
        result=verify_rally_range(self.clip,self.session)
        self.assertEqual(result['source_png_indices'],[1,2])
        self.assertEqual(result['source_map_indices'],[11,12])
        self.assertEqual(result['source_original_frame_ids'],[102,103])
        self.assertEqual(result['eligible_paired_png_indices'],[1,2])
        self.assertGreater(result['last_scan_mid_us'],result['accounting_common_end_us'])
        self.assertFalse(result['bilateral_person_detection_qualified'])
        self.assertFalse(result['body_fitting_started'])

    def test_tampered_bounds_timestamps_and_out_of_range_png_are_rejected(self):
        for change in [dict(source_begin_frame=1.0),dict(source_frame_count=5),dict(common_end_us=3000.0),
                       dict(common_end_us=3100),dict(both_camera_matched_frames=1)]:
            with self.subTest(change=change),self.assertRaises(ValueError):verify_rally_range({**self.clip,**change},self.session)
        changed=copy.deepcopy(self.clip);changed['camera_bounds']['cam2']['first']['record_index']=999
        with self.assertRaisesRegex(ValueError,'native index/PTS/common time'):verify_rally_range(changed,self.session)

    def test_missing_camera_frame_is_not_a_qualified_input_frame(self):
        self.pairs[2]['cameras']['cam2']['status']='missing_clock';self.pairs[2]['status']='excluded'
        self.clip['both_camera_matched_frames']=1
        for cam in ('cam1','cam2'):
            self.clip['camera_bounds'][cam]['last']=camera_bound(self.pairs[1]['cameras'][cam])
        result=verify_rally_range(self.clip,self.session)
        self.assertEqual(result['source_png_indices'],[1,2])
        self.assertEqual(result['eligible_paired_png_indices'],[1])

    def test_camera_summary_uses_joint_matches_without_truncating_parent_range(self):
        self.pairs[1]['cameras']['cam2']['status']='outside_video_time_range';self.pairs[1]['status']='excluded'
        self.clip['both_camera_matched_frames']=1
        for cam in ('cam1','cam2'):
            self.clip['camera_bounds'][cam]['first']=camera_bound(self.pairs[2]['cameras'][cam])
        result=verify_rally_range(self.clip,self.session)
        self.assertEqual(result['source_png_indices'],[1,2])
        self.assertEqual(result['eligible_paired_png_indices'],[2])

    def test_overlapping_and_touching_rallies_are_counted_once(self):
        merged=union_intervals([[300,330],[50,150],[0,100],[150,180]])
        self.assertEqual(merged,[[0,180],[300,330]])
        self.assertEqual(sum(b-a for a,b in merged),210)
        with self.assertRaises(ValueError):union_intervals([[100,50]])


if __name__=='__main__':unittest.main()
