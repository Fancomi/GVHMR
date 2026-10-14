import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from hmr4d.lidar_mocap.export import acceptance_reasons, export_joblib
from hmr4d.lidar_mocap.segmentation import farthest_sample


class TrainingPairTests(unittest.TestCase):
    def good_qa(self):
        return {'final_p2surface_mean_m':.02,'final_p2surface_p90_m':.04,'finite':True,
                'floor_penetration_m':.01,'reprojection_mean_px':5,'reprojection_valid_joints':{'cam2':17}}

    def test_missing_or_nonfinite_supervision_is_rejected(self):
        good = self.good_qa()
        self.assertEqual(acceptance_reasons(good,200),[])
        for key in ('floor_penetration_m','reprojection_mean_px','final_p2surface_mean_m','final_p2surface_p90_m'):
            bad = {**good,key:float('nan')}
            self.assertTrue(acceptance_reasons(bad,200))
            del bad[key]
            self.assertTrue(acceptance_reasons(bad,200))
        self.assertTrue(acceptance_reasons({**good,'reprojection_valid_joints':{'cam2':0}},200))

    def test_real_indices_unique_even_duplicate_xyz(self):
        points = np.ones((300,3),np.float32)
        samples, indices = farthest_sample(points)
        self.assertEqual(len(np.unique(indices)),256)
        np.testing.assert_array_equal(samples,points[indices])
        short = np.arange(15,dtype=np.float32).reshape(5,3)
        samples,indices = farthest_sample(short)
        self.assertEqual(samples.shape,(256,3))
        self.assertEqual(set(indices),set(range(5)))

    def test_joblib_preserves_contract_and_real_cloud(self):
        sample = {'pc':np.arange(768,dtype=np.float32).reshape(256,3)/1000,
                  'root_pose':np.zeros((1,3)), 'body_pose':np.zeros((23,3)),
                  'shape':np.zeros(10),'transl':np.ones(3),
                  'motion_id':'session:window:person','metadata':{'source':'real'}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pairs.pth'
            export_joblib([sample],path)
            loaded = joblib.load(path)
            self.assertEqual(tuple(loaded['body_pose'].shape),(1,23,3))
            np.testing.assert_array_equal(loaded['pc'][0].numpy(),sample['pc'])
            self.assertTrue(path.with_suffix('.metadata.jsonl').exists())


if __name__=='__main__':
    unittest.main()
