"""Measured source, fitting and final person-pair counts for sealed rally runs."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from .pipeline import write_json
from .rally_pipeline import read_json, read_rows, window_directory


def report_rally_delivery(source_root, output_root, manifest):
    source_root = Path(source_root)
    output_root = Path(output_root)
    source = read_json(source_root / "datasets/dataset_manifest.json")
    registry = read_json(source_root / "source_registry.json")
    intake = read_json(registry["intake"]["path"])
    state = read_json(source_root / "build_status.json")
    by_source = {}
    all_surface = []
    all_p90 = []
    all_reprojection = []
    bilateral_exclusions = Counter()
    for row in read_rows(output_root / "excluded.jsonl"):
        bilateral_exclusions[row["session_id"]] += 1
    for sid, entry in manifest["exports"].items():
        rows = read_rows(entry["metadata"])
        surface = [r["qa"]["final_p2surface_mean_m"] * 1000 for r in rows]
        p90 = [r["qa"]["final_p2surface_p90_m"] * 1000 for r in rows]
        reprojection = [r["qa"]["reprojection_mean_px"] for r in rows if r["qa"]["reprojection_mean_px"] is not None]
        eligible = {i for a, b in intake["sources"][sid]["unique_paired_png_ranges_half_open"] for i in range(a, b)}
        fitted = []
        segmentation = []
        window_reports = []
        for window in source["selected_windows"][sid]:
            directory = window_directory(source_root, source, sid, window)
            fitted.extend(r for r in read_rows(directory / "quality.jsonl") if r["lidar_index"] in eligible)
            report = read_json(directory / "fit_report.json")
            window_reports.append(report)
            segmentation.extend(r for r in report["segmentation_exclusions"] if r["lidar_index"] in eligible)
        by_source[sid] = dict(
            baseline_reused=intake["sources"][sid]["baseline"],
            raw_scan_frames=state["sources"][sid]["raw_scans"],
            rallies=intake["sources"][sid]["rallies"],
            rally_window_minutes=intake["sources"][sid]["effective_window_seconds"] / 60,
            unique_paired_rally_scan_frames=len(eligible),
            fitted_person_frames=len(fitted),
            body_qa_accepted_person_frames=source["exports"][sid]["accepted"],
            body_qa_excluded_person_frames=len(fitted) - source["exports"][sid]["accepted"],
            body_qa_reasons_overlapping=dict(Counter(reason for r in fitted for reason in r["rejection_reasons"])),
            segmentation_excluded_person_scan_candidates=len(segmentation),
            segmentation_reasons=dict(Counter(r["reason"] for r in segmentation)),
            bilateral_excluded_person_frames=bilateral_exclusions[sid],
            final_paired_person_frames=len(rows),
            surface_mean_mm=float(np.mean(surface)) if surface else None,
            mean_of_frame_surface_p90_mm=float(np.mean(p90)) if p90 else None,
            cam2_optimization_residual_mean_px=float(np.mean(reprojection)) if reprojection else None,
        )
        all_surface.extend(surface)
        all_p90.extend(p90)
        all_reprojection.extend(reprojection)
    result = dict(
        schema="badminton-sealed-rally-delivery-report-v1",
        sources=by_source,
        rally_window_minutes=intake["effective_window_minutes"],
        unique_paired_input_scan_frames=sum(len_["paired_unique_png_frames"] for len_ in intake["sources"].values()),
        fitted_person_frames=sum(r["fitted_person_frames"] for r in by_source.values()),
        body_qa_accepted_person_frames=source["accepted"],
        final_paired_person_frames=manifest["accepted"],
        splits={
            name: dict(sessions=s["sessions"], person_frames=s["accepted"])
            for name, s in manifest["suggested_session_splits"].items()
        },
        surface_mean_mm=float(np.mean(all_surface)) if all_surface else None,
        mean_of_frame_surface_p90_mm=float(np.mean(all_p90)) if all_p90 else None,
        cam2_optimization_residual_mean_px=float(np.mean(all_reprojection)) if all_reprojection else None,
        metric_limit="Visible surface and participating-camera optimization residuals of fitted pseudo-labels; not anatomical ground truth",
        time_limit="Rally minutes include preroll/cooldown. Input scan frames and final person-frame labels are separate quantities",
        split_limit="User requested new split; previously trained-on 154726 is now test, so this is not an untouched blind test",
    )
    write_json(output_root / "quality_summary.json", result)
    return result
