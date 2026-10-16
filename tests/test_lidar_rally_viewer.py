import base64
import hashlib
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from hmr4d.lidar_mocap.rally_viewer import ViewerData


class NativeImageBindingTests(unittest.TestCase):
    def test_rgb_is_bound_to_real_record_and_native_pts(self):
        images = [np.full((24, 32, 3), value, dtype=np.uint8) for value in (20, 170)]
        rows = [
            dict(media_pts_us=90123, common_time_us=1700000000000010),
            dict(media_pts_us=106807, common_time_us=1700000000016694),
        ]
        session = SimpleNamespace(camera_rows=lambda cam: rows, read_image=lambda cam, i: images[i].copy())
        viewer = ViewerData(None, None, None, None, {"s": session}, {})
        first = viewer.native_image("s", "cam1", 0)
        second = viewer.native_image("s", "cam1", 1)
        self.assertEqual(second["record_index"], 1)
        self.assertEqual(second["media_pts_us"], 106807)
        self.assertEqual(second["raw_bgr_sha256"], hashlib.sha256(images[1].tobytes()).hexdigest())
        self.assertNotEqual(first["raw_bgr_sha256"], second["raw_bgr_sha256"])
        blob = base64.b64decode(second["jpeg_base64"])
        self.assertEqual(second["jpeg_sha256"], hashlib.sha256(blob).hexdigest())
        decoded = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(decoded, images[1])


if __name__ == "__main__":
    unittest.main()
