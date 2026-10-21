import numpy as np

from hmr4d.lidar_mocap.pilot_report import outlier_review


def row(motion, scan, time_us, x):
    return dict(source_id="example", motion_id=motion, scan=scan, scan_mid_us=time_us,
                pelvis=np.array([x, 0, 0]), accepted=True, qa={}, raw_camera={})


def test_pelvis_screen_uses_elapsed_clock_and_never_crosses_identity_or_scan_gaps():
    rows = [row("fast", 0, 0, 0), row("fast", 1, 20_000, 0.4),
            row("slow", 0, 0, 0), row("slow", 1, 500_000, 0.4),
            row("gap", 0, 0, 0), row("gap", 3, 300_000, 10),
            row("different", 4, 400_000, 20)]
    result = outlier_review(rows)
    assert result["pelvis_steps"]["count"] == 2
    assert result["pelvis_steps"]["candidate_count"] == 1
    candidate = result["pelvis_steps"]["candidates"][0]
    assert candidate["motion_id"] == "fast" and candidate["speed_m_s"] == 20
    assert result["training_published"] is False


def test_ground_bad_case_keeps_rejected_rows_and_raw_camera_evidence():
    bad = row("actor", 10, 1_000_000, 0)
    bad.update(accepted=False, raw_camera={"cam1": {"mean_px": 30}})
    bad["qa"] = dict(max_ground_penetration_m=0.11, surface_mean_m=0.04,
                     reprojection_by_camera_px={"cam1": 8})
    result = outlier_review([bad])
    assert result["ground_over_20mm"] == result["ground_over_50mm"] == 1
    assert result["ground_cases"][0]["accepted"] is False
    assert result["ground_cases"][0]["raw_camera"]["cam1"]["mean_px"] == 30
