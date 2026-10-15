import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hmr4d.lidar_mocap.source_index import SourceIndexedSession, _bound_file, validate_coordinate_contract


class ExternalSourceContractTests(unittest.TestCase):
    def setUp(self):
        self.index=dict(fixed24=dict(coordinate_encoding='fixed24',stored_xyz_space='lidar_sensor_xyz',
                                     scale=1000,center=256,point_width=2,point_height=1,image_size=[2,3]),
                        coordinate_contract=dict(M_baked_into_stored_xyz=False,consumer_application_count_required=1,
                                                 application_M_applied_by_this_index_builder=False,
                                                 application_raw_to_viewer_M=[[.984690,0,.174316],[0,1,0],[-.174316,0,.984690]]))

    def test_raw_coordinates_require_one_explicit_matrix_and_actual_inverse(self):
        matrix=validate_coordinate_contract(self.index)
        raw=np.array([[2.,3.,4.]])
        viewer=raw@matrix.T
        np.testing.assert_allclose(viewer@np.linalg.inv(matrix).T,raw,atol=1e-12)
        self.assertGreater(np.max(np.abs(viewer@matrix-raw)),1e-8)

    def test_baked_twice_or_invalid_grid_matrix_is_rejected(self):
        for field,value in [('M_baked_into_stored_xyz',True),('consumer_application_count_required',2),
                            ('application_M_applied_by_this_index_builder',True),('application_raw_to_viewer_M',np.zeros((3,3)).tolist())]:
            index=copy.deepcopy(self.index);index['coordinate_contract'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):validate_coordinate_contract(index)
        index=copy.deepcopy(self.index);index['fixed24']['image_size']=[2,1]
        with self.assertRaises(ValueError):validate_coordinate_contract(index)

    def test_pending_applicability_cannot_admit_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'index.json'
            path.write_text(json.dumps(dict(schema='badminton-source-index-candidate-v1',
                camera_calibration_admitted=False,production_admitted=False)))
            with self.assertRaisesRegex(ValueError,'pending.*no fitting admission'):
                SourceIndexedSession.load_source_index(path,root)

    def test_root_escape_and_changed_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'source.bin';path.write_bytes(b'original')
            entry=dict(file='source.bin',sha256=hashlib.sha256(b'original').hexdigest(),bytes=8)
            self.assertEqual(_bound_file(root,entry),path)
            path.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):_bound_file(root,entry)
            with self.assertRaisesRegex(ValueError,'escapes'):_bound_file(root,dict(file='../outside'))


if __name__=='__main__':unittest.main()
