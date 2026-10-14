import unittest
import numpy as np
from hmr4d.lidar_mocap.review import fill_triangle_union


class ProjectedSilhouetteTests(unittest.TestCase):
    def test_overlapping_front_back_faces_keep_solid_interior(self):
        triangle=np.array([[1,1],[10,1],[1,10]],np.int32)
        mask=np.zeros((12,12),np.uint8)
        fill_triangle_union(mask,[triangle,triangle[::-1]],1)
        self.assertEqual(mask[3,3],1)
        self.assertEqual(mask[11,11],0)
        self.assertGreater(np.count_nonzero(mask),40)

    def test_shared_edges_and_rgb_overlay_preserve_union(self):
        faces=np.array([[[1,1],[9,1],[9,9]],[[1,1],[9,9],[1,9]]],np.int32)
        image=np.zeros((12,12,3),np.uint8)
        fill_triangle_union(image,np.concatenate([faces,faces[:,::-1]]),(12,34,56))
        np.testing.assert_array_equal(image[2:9,2:9],np.broadcast_to([12,34,56],(7,7,3)))
        np.testing.assert_array_equal(image[11,11],[0,0,0])


if __name__=='__main__':unittest.main()
