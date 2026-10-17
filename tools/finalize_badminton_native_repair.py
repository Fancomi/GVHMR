"""Seal the completed repair only after all windows, native checks and HTML pass."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json
from hmr4d.lidar_mocap.repair_execution import repair_efficiency


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2")
    p.add_argument("--wait", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    labels = root / "final_labels"
    out = root / "comparison"
    plan = read_json(labels / "repair_plan.json")
    expected = len(plan["jobs"])
    required = [
        labels / "repair_receipt.json",
        labels / "native_verification.json",
        root / "evaluation/metrics.json",
        out / "manifest.json",
        root / "leg_refinement_receipt.json",
        labels / "datasets/verification.json",
    ]
    while True:
        values = [read_json(path) if path.exists() else {} for path in required]
        fit, contract, metrics, viewer, legs, aggregate_check = values
        ready = (
            fit.get("windows") == expected
            and contract.get("passed")
            and contract.get("verified_windows") == expected
            and metrics.get("verified_windows") == expected
            and metrics.get("state") == "complete"
            and viewer.get("state") == "complete"
            and all(c["state"] == "complete" for c in viewer.get("clips", []))
            and len(viewer.get("clips", [])) == 10
            and legs.get("state") == "complete"
        )
        if ready:
            ready = (
                len(aggregate_check.get("sessions", {})) == 7
                and all(
                    s["max_source_cloud_error_m"] <= 1e-6
                    and s["max_rebuilt_mesh_error_m"] <= 1e-4
                    and s["max_pelvis_translation_error_m"] <= 1e-6
                    for s in aggregate_check["sessions"].values()
                )
                and metrics.get("metric_schema_version") == 2
                and all(
                    contract["windows"][key]["body"]["report_sha256"] == entry["fit_report_sha256"]
                    and metrics["windows"][key].get("fit_report_sha256") == entry["fit_report_sha256"]
                    for key, entry in legs["windows"].items()
                )
                and all(file_sha(path) == sha for c in viewer["clips"] for path, sha in c["source_reports"].items())
            )
        if ready:
            break
        if not args.wait:
            print("Incomplete: full fits, native verification, metrics or ten clips still pending")
            return
        time.sleep(5)
    if file_sha(plan["source_union"]) != plan["source_union_sha256"]:
        raise ValueError("Original delivered union changed")
    if file_sha(Path(viewer["display_crop"]["path"])) != viewer["display_crop"]["sha256"]:
        raise ValueError("Frozen display crop changed")
    if (
        viewer["calibration_digest"] != plan["calibration_digest"]
        or contract["calibration_digest"] != plan["calibration_digest"]
    ):
        raise ValueError("Frozen calibration changed")
    assets = PROJ_ROOT / "hmr4d/lidar_mocap/review_assets"
    diagnostics={}
    for window in metrics['windows'].values():
        for row in read_json(window['observations_path']):
            for version,motion in [('before',row['old_motion']),('after',row['new_motion'])]:
                cameras=row['cameras']
                diagnostics[version+'|'+motion+'|'+str(row['scan'])]=dict(
                    cam1=cameras.get('cam1',{}).get(version,{}).get('unordered_leg_mean_px'),
                    cam2=cameras.get('cam2',{}).get(version,{}).get('unordered_leg_mean_px'),
                    inconsistent=version=='after' and any(any(i>=13 for i in c.get('rejected_ray_joints',[])) for c in cameras.values()),
                    corrected=version=='after' and any(c.get('native_joint_label_mapping') for c in cameras.values()))
    (out/'diagnostics.js').write_text('window.REPAIR_DIAGNOSTICS='+json.dumps(diagnostics,separators=(',',':'))+';',encoding='utf-8')
    for src, dest in [
        ("comparison.html", "index.html"),
        ("comparison.js", "comparison.js"),
        ("comparison.css", "comparison.css"),
        ("projection.js", "projection.js"),
        ("review.js", "review.js"),
        ("review.css", "review.css"),
    ]:
        shutil.copyfile(assets / src, out / dest)
    for url, name in [
        ("http://127.0.0.1:8772/", "browser_verification.json"),
        ((out / "index.html").as_uri(), "offline_browser_verification.json"),
    ]:
        path = out / name
        subprocess.run(
            [
                sys.executable,
                str(PROJ_ROOT / "tools/verify_badminton_repair_comparison.py"),
                "--url",
                url,
                "--output",
                str(path),
            ],
            check=True,
        )
        receipt = read_json(path)
        if not receipt["passed"] or receipt["complete"] != 10 or receipt["partial"]:
            raise ValueError("Full browser review failed")
    old_fitted = old_accepted = new_fitted = new_accepted = 0
    teacher = Counter()
    maxima = Counter()
    foot = []
    for job in plan["jobs"]:
        sid = job["source_id"]
        old = Path(job["source_window"])
        new = labels / "sessions" / sid / "windows" / old.name
        a, b = read_json(old / "fit_report.json"), read_json(new / "fit_report.json")
        old_fitted += a["fitted"]
        old_accepted += a["accepted"]
        new_fitted += b["fitted"]
        new_accepted += b["accepted"]
        check = contract["windows"][sid + ":" + old.name]
        teacher.update(check["teacher"])
        for key, value in check["body"]["maximums"].items():
            maxima[key] = max(maxima[key], value)
    for window in metrics["windows"].values():
        for row in read_json(window["observations_path"]):
            if "left_forefoot_direction_change_vs_same_converted_reference_deg" in row:
                foot.append(row)
    seconds = sum(
        (
            read_json(Path(j["source_window"]) / "window.json")["right_us"]
            - read_json(Path(j["source_window"]) / "window.json")["left_us"]
        )
        / 1e6
        for j in plan["jobs"]
    )
    result = dict(
        schema="badminton-native-body-repair-complete-v2",
        state="complete",
        created_utc=datetime.now(timezone.utc).isoformat(),
        source_union_sha256=plan["source_union_sha256"],
        original_delivered_union_person_scans=79776,
        calibration_digest=plan["calibration_digest"],
        windows=expected,
        clips=10,
        clip_seconds=5,
        fitted_before=old_fitted,
        fitted_after=new_fitted,
        body_qa_before=old_accepted,
        body_qa_after=new_accepted,
        training_published=False,
        teacher_counts=dict(teacher),
        max_contract_errors_m=dict(maxima),
        efficiency=repair_efficiency(root, new_fitted, seconds),
        native_leg_refinement=legs,
        common_observation_cohorts=metrics["cohorts"],
        foot_examples=foot,
        identity_review=dict(
            actual_id_boundary_cases=64,
            native_continuity_confirmed=56,
            additional_native_RGB_review_confirmed=3,
            different_people=2,
            mixed_boundary_cases=3,
            limits="Reviewed boundaries and detected contaminated spans; not cross-session athlete recognition or a global identity ground-truth guarantee.",
        ),
        evidence_files={
            str(path): file_sha(path)
            for path in required
            + [
                out / "browser_verification.json",
                out / "offline_browser_verification.json",
                root / "identity_evidence/identity_decisions.json",
                root / "identity_evidence/manual_review_receipt.json",
            ]
        },
        limits=metrics["limits"]
        + [
            "Dual-view constraints cover declared review scans and contexts; other rebuilt scans retain primary-camera observations.",
            "Body QA counts are not bilateral training admission. The original 79,776-sample union and training queue remain unchanged.",
        ],
    )
    write_json(root / "completion_receipt.json", result)
    blocks = []
    for name, cohort in metrics["cohorts"].items():
        rows = []
        surface = cohort["common_surface_mm"]
        rows.append(("Same visible points / mean mm", surface["before"], surface["after"]))
        for cam in ["cam1", "cam2"]:
            value = cohort["camera_px"][cam]["ankle_mean_px"]
            rows.append((cam + " ankle / px", value["before"], value["after"]))
            value = cohort["camera_px"][cam]["unordered_leg_mean_px"]
            rows.append((cam + " knee/ankle set / px", value["before"], value["after"]))
        body = "".join(
            "<tr><td>"
            + title
            + "</td>"
            + "".join(
                f'<td>{v.get("mean",0):.2f} / {v.get("median",0):.2f} / {v.get("p90",0):.2f} (n={v["count"]})</td>'
                for v in [a, b]
            )
            + "</tr>"
            for title, a, b in rows
        )
        blocks.append(
            f'<h2>{name} · {cohort["person_scans"]} matched person-scans</h2><table><tr><th>Metric</th><th>Before mean / median / P90</th><th>After mean / median / P90</th></tr>{body}</table>'
        )
    footrows = "".join(
        f'<tr><td>{r["scan"]}</td><td>{r["left_forefoot_direction_change_vs_same_converted_reference_deg"][0]:.2f}</td><td>{r["left_forefoot_direction_change_vs_same_converted_reference_deg"][1]:.2f}</td></tr>'
        for r in foot
    )
    page = '<!doctype html><meta charset="utf-8"><title>SMPL repair quality</title><style>body{font:15px system-ui;margin:32px;max-width:1300px;background:#101824;color:#e2eaf5}a{color:#8bcfff}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #43556c;text-align:left}pre{white-space:pre-wrap}</style>'
    page += f'<h1>身份 / 脚部修复结果</h1><p><a href="index.html">10 × 5 秒同步新旧对比</a></p><p>{expected} 个窗口；拟合 {old_fitted} → {new_fitted} 人帧；人体拟合 QA {old_accepted} → {new_accepted}。原训练集 79,776 份保持不变。</p>'
    page += (
        "".join(blocks)
        + f"<h2>脚部方向变化 · 相同模型转换参考</h2><p>该角度不是解剖真值误差；同时检查两机原图轮廓。保留所有五个固定帧，包括与参考方向差异增大的帧。</p><table><tr><th>scan</th><th>Before deg</th><th>After deg</th></tr>{footrows}</table>"
    )
    page += "<h2>校验与评估范围</h2><pre>" + html.escape(json.dumps(result, ensure_ascii=False, indent=2)) + "</pre>"
    (out / "report.html").write_text(page, encoding="utf-8")
    write_json(root/'end_to_end_receipt.json',dict(state='complete',completion_receipt=str(root/'completion_receipt.json'),
               sha256=file_sha(root/'completion_receipt.json'),training_published=False))
    print(
        dict(state="complete", windows=expected, body_qa_after=new_accepted, html=str(out / "index.html")), flush=True
    )


if __name__ == "__main__":
    main()
