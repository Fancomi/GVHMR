from copy import deepcopy
import unittest

from hmr4d.lidar_mocap.dual_view import DualViewGate
from tools.audit_badminton_native_pixels import compare_admissions


def person(motion, body_qa=True, confidence=0.9):
    return dict(
        motion_id=motion,
        fit_accepted=body_qa,
        detector_index=0,
        independent_detection_bbox_xyxy=[10, 10, 50, 90],
        independent_detection_confidence=confidence,
        association_bbox_iou=0.8,
        association_margin=0.7,
    )


def views(people):
    return {cam: dict(athletes=deepcopy(people)) for cam in ("cam1", "cam2")}


class NativePixelCorrectionTests(unittest.TestCase):
    def test_previously_excluded_person_becoming_admitted_is_detected(self):
        before = views([person("a"), person("b", confidence=0.2)])
        after = views([person("a"), person("b")])
        checks = compare_admissions(before, after, DualViewGate(), {"a"})
        self.assertFalse(next(c for c in checks if c["motion_id"] == "b")["unchanged"])

    def test_body_qa_rejected_person_cannot_enter_through_pixel_correction(self):
        before = views([person("a"), person("b", body_qa=False, confidence=0.2)])
        after = views([person("a"), person("b", body_qa=False)])
        checks = compare_admissions(before, after, DualViewGate(), {"a"})
        self.assertTrue(all(c["unchanged"] for c in checks))
        self.assertFalse(next(c for c in checks if c["motion_id"] == "b")["after"]["admitted"])

    def test_incomplete_person_set_fails_closed(self):
        before = views([person("a"), person("b")])
        after = views([person("a"), person("b")])
        after["cam2"]["athletes"].pop()
        with self.assertRaisesRegex(ValueError, "person sets"):
            compare_admissions(before, after, DualViewGate(), {"a", "b"})

    def test_wrong_original_membership_fails_closed(self):
        before = views([person("a")])
        with self.assertRaisesRegex(ValueError, "immutable test membership"):
            compare_admissions(before, deepcopy(before), DualViewGate(), set())
