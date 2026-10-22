"""Coverage must retain missing cells and expose additional detections separately."""

import pytest

from hmr4d.lidar_mocap.full_coverage import coverage_counts


def test_nominal_coverage_cannot_hide_missing_people_with_extra_rows():
    result = coverage_counts([2, 1, 0, 3], [2, 1, 0, 2])
    assert result["nominal_person_scan_cells"] == 8
    assert result["fitted"] == 6
    assert result["gross_unfilled_before_qa"] == 3
    assert result["additional_fitted_over_nominal"] == 1
    assert result["net_unfilled_before_qa"] == 2
    assert result["qa_rejected"] == 1
    assert result["net_unfilled_after_qa"] == 3
    assert result["fitted_people_histogram"] == {"0": 1, "1": 1, "2": 1, "3": 1}


@pytest.mark.parametrize("fitted,accepted", [([1], [2]), ([1], [-1]), ([1, 2], [1])])
def test_invalid_coverage_counts_fail(fitted, accepted):
    with pytest.raises(ValueError):
        coverage_counts(fitted, accepted)
