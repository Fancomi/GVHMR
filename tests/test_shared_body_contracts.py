import unittest
from pathlib import Path
import torch
from hmr4d.utils.body_model.contracts import map_smplx_vertices, mapped_smpl_geometry
from hmr4d.utils.geo.hmr_cam import resolve_camera_intrinsics


class SharedContractTests(unittest.TestCase):
    def test_sparse_mapping_matches_framewise_reference_and_gradients(self):
        indices = torch.stack((torch.arange(6890),torch.arange(6890)))
        mapping = torch.sparse_coo_tensor(indices,torch.ones(6890),(6890,10475))
        vertices = torch.randn(2,10475,3,requires_grad=True)
        regressor = torch.zeros(24,6890)
        regressor[:,0] = 1
        mapped,joints = mapped_smpl_geometry(vertices,mapping,regressor)
        reference = torch.stack([mapping@v for v in vertices])
        torch.testing.assert_close(mapped,reference)
        torch.testing.assert_close(joints,reference[:,:1].expand(-1,24,-1))
        mapped.sum().backward()
        self.assertTrue(torch.isfinite(vertices.grad).all())
        self.assertEqual(map_smplx_vertices(vertices[0],mapping).shape,(6890,3))

    def test_invalid_camera_values_fail_consistently(self):
        for value in ([[0,0,0],[0,1,0],[0,0,1]],
                      [[1,0,float('nan')],[0,1,0],[0,0,1]],
                      [[1,0,0],[0,1,0],[0,0,2]],{'fx':1,'fy':-1,'cx':0,'cy':0}):
            with self.assertRaises(ValueError):
                resolve_camera_intrinsics(value,1920,1080)
        k = resolve_camera_intrinsics({'fx':1167,'fy':1167,'cx':960,'cy':540},1920,1080)
        self.assertEqual(tuple(k.shape),(3,3))


if __name__=='__main__':
    unittest.main()
