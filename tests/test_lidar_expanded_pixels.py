from copy import deepcopy
import unittest

from tools.audit_badminton_expanded_pixels import accepted_pixel, pixel_requests


def row():
    native = dict(record_index=10, media_pts_us=100, common_time_us=1000, source_frame_no=123)
    view = dict(native=native, raw_image_sha256="old", report_path="report.json", report_sha256="report-sha")
    return dict(
        session_id="new",
        lidar_index=3,
        calibration_digest="new-cal",
        cameras={cam: deepcopy(native) for cam in ("cam1", "cam2")},
        dualcam=dict(views={cam: deepcopy(view) for cam in ("cam1", "cam2")}),
    )


class ExpandedPixelsTests(unittest.TestCase):
    def request(self):
        return pixel_requests([row()], {"new": dict(role="test", calibration_digest="new-cal")})["new", "cam1"][10]

    def correction(self):
        return dict(
            scene_id="new",
            camera="cam1",
            **self.request()["native"],
            previous_raw_bgr_sha256="old",
            verified_raw_bgr_sha256="verified",
        )

    def test_unchanged_pixels_require_no_supplement(self):
        self.assertIsNone(accepted_pixel(self.request(), "old", {}))

    def test_only_exact_native_bound_correction_is_accepted(self):
        correction = self.correction()
        self.assertEqual(accepted_pixel(self.request(), "verified", {("new", "cam1", 10): correction}), correction)
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            accepted_pixel(self.request(), "unknown", {("new", "cam1", 10): correction})

    def test_correction_cannot_move_to_another_native_pts(self):
        correction = self.correction()
        correction["media_pts_us"] += 1
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            accepted_pixel(self.request(), "verified", {("new", "cam1", 10): correction})

    def test_source_specific_calibration_and_role_are_required(self):
        for role, digest in (("train", "new-cal"), ("test", "old-cal")):
            with self.assertRaisesRegex(ValueError, "role or calibration"):
                pixel_requests([row()], {"new": dict(role=role, calibration_digest=digest)})

    def test_people_sharing_pixels_must_agree(self):
        a, b = row(), row()
        b["dualcam"]["views"]["cam1"]["raw_image_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "sharing one native"):
            pixel_requests([a, b], {"new": dict(role="test", calibration_digest="new-cal")})

    def test_paired_pts_and_observer_pts_cannot_disagree(self):
        a = row()
        a["cameras"]["cam1"]["media_pts_us"] += 1
        with self.assertRaisesRegex(ValueError, "paired native"):
            pixel_requests([a], {"new": dict(role="test", calibration_digest="new-cal")})


if __name__ == "__main__":
    unittest.main()
