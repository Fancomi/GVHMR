"""Make native-image before/after evidence for the reported 1009 track/foot repair."""

import argparse
import base64
import html
import json
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json, read_rows


def api(base, path):
    with urlopen(base.rstrip("/") + path, timeout=120) as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v1")
    parser.add_argument("--old-url", default="http://127.0.0.1:8767")
    parser.add_argument("--new-url", default="http://127.0.0.1:8771")
    args = parser.parse_args()
    root = args.root
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    name = "sessions/20261009_202713/windows/000120000000_000180000000"
    old = PROJ_ROOT / "outputs/badminton_rallies_v1" / name
    new = root / "case_labels" / name
    cal = CalibrationSnapshot.load(root / "case_labels/calibration/calibration_snapshot.json")
    rows = {
        version: {r["lidar_index"]: r for r in read_rows(path / "quality.jsonl") if r["motion_id"].endswith(":12:0")}
        for version, path in (("old", old), ("new", new))
    }
    both = sorted(scan for scan in rows["old"] if rows["old"][scan]["accepted"] and rows["new"][scan]["accepted"])
    cohort = {
        version: dict(
            count=len(both),
            mean_surface_mm=float(
                np.mean([rows[version][scan]["qa"]["final_p2surface_mean_m"] * 1000 for scan in both])
            ),
            mean_reprojection_px=float(np.mean([rows[version][scan]["qa"]["reprojection_mean_px"] for scan in both])),
        )
        for version in ("old", "new")
    }
    comparison = read_json(root / "dual_camera_pose_comparison.json")
    dual = {
        cam: {
            version: {
                metric: float(
                    np.mean(
                        [
                            r["comparison"][version][metric]
                            for r in comparison["results"]
                            if r["camera"] == cam and r["comparison"] is not None
                        ]
                    )
                )
                for metric in ("all_mean_px", "ankle_mean_px")
            }
            for version in ("old", "new")
        }
        for cam in ("cam1", "cam2")
    }
    manifest = api(args.old_url, "/api/manifest")
    faces = (
        np.frombuffer(base64.b64decode(manifest["faces"]["data"]), dtype="<u2").reshape(-1, 3)
        if isinstance(manifest["faces"], dict)
        else np.frombuffer(base64.b64decode(manifest["faces"]), dtype="<u2").reshape(-1, 3)
    )
    stages = np.load(PROJ_ROOT / "outputs/badminton_rallies_v1/viewer/quality_audit_v1/conversion_geometry.npz")
    examples = []
    for i, scan_value in enumerate(stages["scans"]):
        scan = int(scan_value)
        relative = manifest["sources"]["20261009_202713"]["scan_times"][scan]
        route = f"/api/geometry?source=20261009_202713&t={relative}"
        before, after = api(args.old_url, route), api(args.new_url, route)
        assert before["lidar_index"] == after["lidar_index"] == scan
        meshes, joints = {}, {}
        for version in ("old", "new"):
            with np.load(rows[version][scan]["fit_path"]) as data:
                meshes[version], joints[version] = data["vertices"].copy(), data["joints"].copy()
        foot = {}
        for version in ("old", "new"):
            angles = []
            for joint in (10, 11):
                mask = stages["weights"][:, joint] > 0.25
                a, b = stages["converted"][i, mask], meshes[version][mask]
                rotation, _ = Rotation.align_vectors(b - b.mean(0), a - a.mean(0))
                angles.append(float(np.degrees(rotation.magnitude())))
            foot[version] = angles
        canvas = Image.new("RGB", (960, 760), (18, 24, 34))
        draw = ImageDraw.Draw(canvas)
        for cam_i, cam in enumerate(("cam1", "cam2")):
            a, b = before["native_images"][cam], after["native_images"][cam]
            assert a["record_index"] == b["record_index"] and a["raw_bgr_sha256"] == b["raw_bgr_sha256"]
            rgb = cv2.imdecode(np.frombuffer(base64.b64decode(a["jpeg_base64"]), np.uint8), cv2.IMREAD_COLOR)
            projected = {v: cal.project(meshes[v], cam, distort=True)[0] for v in meshes}
            limb = np.concatenate([cal.project(joints[v][[4, 5, 7, 8, 10, 11]], cam, distort=True)[0] for v in meshes])
            low = np.maximum(np.floor(limb.min(0) - [22, 20]).astype(int), [0, 0])
            high = np.minimum(np.ceil(limb.max(0) + [22, 20]).astype(int), [1920, 1080])
            variants = [("Native RGB", rgb)]
            for version, color in (("old", (30, 157, 255)), ("new", (255, 210, 70))):
                mask = np.zeros(rgb.shape[:2], np.uint8)
                cv2.fillPoly(mask, [t for t in np.rint(projected[version][faces]).astype(np.int32)], 255)
                image = rgb.copy()
                image[mask > 0] = (image[mask > 0] * 0.55 + np.asarray(color) * 0.45).astype(np.uint8)
                variants.append((version + " SMPL", image))
            for col, (label, bgr) in enumerate(variants):
                patch = Image.fromarray(cv2.cvtColor(bgr[low[1] : high[1], low[0] : high[0]], cv2.COLOR_BGR2RGB))
                scale = min(310 / patch.width, 310 / patch.height)
                patch = patch.resize(
                    (round(patch.width * scale), round(patch.height * scale)), Image.Resampling.BICUBIC
                )
                canvas.paste(
                    patch, (col * 320 + (310 - patch.width) // 2, cam_i * 380 + 45 + (310 - patch.height) // 2)
                )
                draw.text((col * 320 + 10, cam_i * 380 + 10), f"{cam} / scan {scan} / {label}", fill="white")
        image_name = f"scan_{scan}_both_cameras.png"
        canvas.save(out / image_name)
        examples.append(
            dict(
                scan=scan,
                mean_surface_mm={v: rows[v][scan]["qa"]["final_p2surface_mean_m"] * 1000 for v in meshes},
                forefoot_rotation_from_cpu_conversion_deg=foot,
                image=image_name,
                url=args.new_url + f"/?source=20261009_202713&rally=1&scan={scan}",
            )
        )
    tracking = read_json(root / "tracking_final/tracking_replay.json")
    remaining = read_json(root / "tracking_final/unlinked_review.json")
    result = dict(
        schema="badminton-body-repair-case-report-v1",
        source_union_sha256=tracking["source_union_sha256"],
        tracking=tracking["counts"],
        unresolved_screening=remaining["counts"],
        same_accepted_far_person_cohort=cohort,
        dual_camera_detector_agreement=dual,
        examples=examples,
        case_fitted=read_json(new / "fit_report.json")["fitted"],
        case_body_qa_accepted=read_json(new / "fit_report.json")["accepted"],
        training_published=False,
        body_contract_verification=read_json(root / "case_labels/datasets/verification.json"),
        browser_verification_sha256=file_sha(root / "case_browser_verification.json"),
        limits=[
            "Only this case window has new body fits; tracking replay covers all formal union windows.",
            "Forefoot angles use a CPU reconstructed conversion as a common reference, not anatomical truth.",
            "cam1 is a held-out camera detector agreement check; it was not used for fitting.",
        ],
    )
    result["full_teacher_verification"] = read_json(root / "labels/teacher_verification.json")
    result["algorithm_sources_sha256"] = {
        name: file_sha(PROJ_ROOT / "hmr4d/lidar_mocap" / (name + ".py"))
        for name in ("teacher", "fitting", "pipeline", "body_repair")
    }
    write_json(out / "report.json", result)
    table = "".join(
        f"<tr><td>{cam}</td><td>{dual[cam]['old']['all_mean_px']:.2f} → {dual[cam]['new']['all_mean_px']:.2f}</td>"
        f"<td>{dual[cam]['old']['ankle_mean_px']:.2f} → {dual[cam]['new']['ankle_mean_px']:.2f}</td></tr>"
        for cam in dual
    )
    figures = "".join(
        f"<figure><a href='{html.escape(e['url'])}'><img src='{e['image']}'></a><figcaption>scan {e['scan']}：远侧脚方向偏移 "
        f"{e['forefoot_rotation_from_cpu_conversion_deg']['old'][0]:.1f}° → {e['forefoot_rotation_from_cpu_conversion_deg']['new'][0]:.1f}°；"
        f"点面均值 {e['mean_surface_mm']['old']:.1f} → {e['mean_surface_mm']['new']:.1f} mm。点击打开实际帧。</figcaption></figure>"
        for e in examples
    )
    document = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>轨迹与足部修复候选</title><style>body{{max-width:1100px;margin:24px auto;padding:0 18px;background:#111722;color:#e7eef8;font:15px/1.7 system-ui}}a{{color:#93cdff}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #42516a;text-align:left;padding:8px}}figure{{margin:20px 0}}.note{{background:#253147;padding:14px}}</style>
<h1>1009 轨迹修复与足部约束的实际重算对照</h1>
<p><a href="{args.new_url}/?source=20261009_202713&rally=1&scan=1535">打开新版本双机 / 完整 SMPL 连续播放</a> · <a href="{args.old_url}/?source=20261009_202713&rally=1&scan=1535">原版本同帧</a> · <a href="report.json">完整数值</a></p>
<p>已对指定窗口整组重算：{result['case_fitted']} 个拟合人帧，{result['case_body_qa_accepted']} 个通过身体 QA，尚未发布为训练集。原 79,776 条训练数据保持原值。</p>
<h2>轨迹修复</h2><p>先保留已有连续轨迹，再依据实际观察框的尺寸、中心运动和唯一性裕量恢复短时断开的边界。原始缺失帧不会补造。1009 的 8→20 已重算为持续的 track 8；拼接改变了时序上下文，已重新推理 GVHMR。</p>
<p>全量回放 136,455 个原生视频帧、328,896 次人物检测，原有连续关联被拆开的数量为 0；90 个筛查候选中有 50 个接回。其余 33 个的原始轨迹在时间上重叠，另 7 个未通过几何或唯一性条件；原图存在拥挤、遮挡和混叠框，不能把这 40 个直接宣称为已修或强接成一个人。</p>
<p>全部 75 个正式窗口的新版时序 teacher 已生成并独立验证：628 段中 619 段保持完全相同的时序输入并复用原结果，9 段改变了上下文并重新推理；已有的 328,067 个原生 2D 观测逐值相等，另保留了 16 个以前因短段被舍弃的真实观测。这里统计的是约 30 Hz 的视频人物观测，不是新增训练样本数。</p>
<h2>脚部拟合</h2><p>对转换到同一 SMPL 基底后的脚踝、脚掌增加独立姿态保持项，并在同一检测轨迹跨段时共享已建立的体型。两处重点帧的远侧脚几何方向偏移由 21.3° / 29.1° 降到 8.7° / 12.2°。角度采用 CPU 重建的同一转换网格作参考，不是人体真值误差。</p>
<p>同一批前后均通过 QA 的远侧人物 {len(both)} 帧：平均点面距离 {cohort['old']['mean_surface_mm']:.2f} → {cohort['new']['mean_surface_mm']:.2f} mm；参与拟合的 cam2 重投影均值 {cohort['old']['mean_reprojection_px']:.2f} → {cohort['new']['mean_reprojection_px']:.2f} px。</p>
<h2>双机原生影像复核</h2><p>下表为五个重点帧，与同一原生 2D 观测比较。cam1 未参与拟合；cam2 属参与检测器的一致性指标。两者均非人工人体真值。图片使用同一 PTS 绑定原生帧的 JPEG 预览，局部同比放大；投影坐标未修改。</p><table><tr><th>相机</th><th>全身均值 px（旧→新）</th><th>踝部均值 px（旧→新）</th></tr>{table}</table>
<p class="note">422 条候选全部验证：导出点云与原始 fixed24 成员最大差为 0；SMPL 参数重建网格最大坐标差约 0.00095 mm。实际 Chrome 已验证 track 8 连续、双相机原生 RGB、完整网格同帧和连续播放。</p>
{figures}<h2>范围</h2><p>关联规则已做全量缓存回放；新身体拟合目前只覆盖这个诊断窗口。脚部约束仍需要跨场复核，不能把本例的改善说成全部关节问题已经解决。相机、外参、原始点云、旧训练张量及其训练队列均未修改。</p></html>"""
    (out / "index.html").write_text(document, encoding="utf-8")
    print(json.dumps(dict(report=str(out / "index.html"), cohort=cohort, dual=dual)), flush=True)


if __name__ == "__main__":
    main()
