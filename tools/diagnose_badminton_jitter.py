"""CPU-only diagnosis of measured-return motion in a completed review clip.

No source coordinates, calibration, fits, or review frames are modified. Motion
compensation below is a diagnostic comparison only; output points are never
smoothed, resampled, deskewed, or rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.sessions import SessionIndex
from hmr4d.lidar_mocap.teacher import sample_track


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numbers(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return None
    return dict(min=float(a.min()), median=float(np.median(a)), p90=float(np.quantile(a, .9)),
                max=float(a.max()), mean=float(a.mean()))


def cloud_stats(points):
    return dict(point_count=len(points), centroid_m=points.mean(0).tolist(),
                median_m=np.median(points, axis=0).tolist(),
                quantile_probabilities=[.02, .1, .5, .9, .98],
                quantile_xyz_m=np.quantile(points, [.02, .1, .5, .9, .98], axis=0).tolist())


def nearest_coverage(previous, current, shift=None):
    previous = previous if shift is None else previous + shift
    forward = cKDTree(current).query(previous, k=1)[0]
    reverse = cKDTree(previous).query(current, k=1)[0]
    both = np.r_[forward, reverse]
    return dict(symmetric_mean_m=float(both.mean()), symmetric_p50_m=float(np.median(both)),
                symmetric_p90_m=float(np.quantile(both, .9)),
                within_3cm=float((both <= .03).mean()), within_5cm=float((both <= .05).mean()),
                within_10cm=float((both <= .1).mean()))


def projection_stats(points, calibration, camera):
    uv, depth = calibration.project(points, camera, distort=True)
    good = np.isfinite(uv).all(1) & (depth > 0)
    if not good.any():
        return None
    return dict(valid_returns=int(good.sum()), median_uv_px=np.median(uv[good], axis=0).tolist(),
                quantile_uv_px=np.quantile(uv[good], [.02, .5, .98], axis=0).tolist(),
                depth_m=numbers(depth[good]))


def rotation_steps(before, after):
    first = Rotation.from_rotvec(np.asarray(before).reshape(-1, 3))
    second = Rotation.from_rotvec(np.asarray(after).reshape(-1, 3))
    angle = (first.inv() * second).magnitude() * 180 / np.pi
    return dict(mean_deg=float(angle.mean()), max_deg=float(angle.max()))


def _array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def undistorted_to_brown(uv, camera):
    k = np.asarray(camera['K'], float)
    homogeneous = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(k).T
    x, y = homogeneous[:, 0] / homogeneous[:, 2], homogeneous[:, 1] / homogeneous[:, 2]
    k1, k2, p1, p2, k3 = camera['distortion_opencv']
    r2 = x*x + y*y
    radial = 1 + r2*(k1+r2*(k2+r2*k3))
    xd = x*radial + 2*p1*x*y+p2*(r2+2*x*x)
    yd = y*radial + p1*(r2+2*y*y)+2*p2*x*y
    image = np.column_stack([xd, yd, np.ones(len(uv))]) @ k.T
    return image[:, :2] / image[:, 2:]


def exploratory_foot_lag(track, entries, camera):
    """Compare actual video ankles and visible low-cloud proxies, not fitted mesh.

    Their anatomical/surface mismatch prevents this from estimating a verified
    physical clock offset. Removing one constant image bias does not change any
    source points. Profiles are reported to expose ambiguity, not apply a fix.
    """
    times = _array(track['common_time_us']).astype(np.int64)
    kp = _array(track['kp2d'])
    good = (kp[:, [15, 16], 2] > .25).all(1)
    if good.sum() < 30:
        return dict(status='insufficient_independent_ankle_observations')
    anchors = undistorted_to_brown(kp[:, [15, 16], :2].mean(1), camera)
    cloud_times = np.array([e['scan_mid_us'] for e in entries], dtype=np.int64)
    cloud_uv = np.array([e['low_cloud_median_brown_uv_px']['cam2'] for e in entries], dtype=float)
    profiles = []
    for offset_us in range(-150000, 150001, 5000):
        requested = cloud_times + offset_us
        right = np.searchsorted(times, requested)
        left = np.maximum(right-1, 0)
        right = np.minimum(right, len(times)-1)
        valid = ((requested >= times[0]) & (requested <= times[-1]) & good[left] & good[right]
                 & (times[right]-times[left] <= 36000) & np.isfinite(cloud_uv).all(1))
        if valid.sum() < 30:
            continue
        fraction = np.divide(requested-times[left], times[right]-times[left],
                             out=np.zeros(len(entries), float), where=times[right] != times[left])
        video_uv = anchors[left]*(1-fraction[:, None])+anchors[right]*fraction[:, None]
        residual = cloud_uv[valid]-video_uv[valid]
        residual -= np.median(residual, axis=0)
        profiles.append(dict(offset_ms=offset_us/1000, usable_scans=int(valid.sum()),
                             demeaned_median_error_px=float(np.median(np.linalg.norm(residual, axis=1)))))
    if not profiles:
        return dict(status='insufficient_overlapping_proxy_observations')
    best = min(profiles, key=lambda p: p['demeaned_median_error_px'])
    near = [p['offset_ms'] for p in profiles if p['demeaned_median_error_px'] <= best['demeaned_median_error_px']+1]
    zero = next(p for p in profiles if p['offset_ms'] == 0)
    return dict(status='exploratory_proxy_only_not_a_calibration', best_offset_ms=best['offset_ms'],
                offset_sign='compare cloud(t) with actual cam2 ankles(t + offset)',
                within_1px_of_best_ms=[min(near), max(near)], zero_offset=zero, best=best,
                profiles=profiles,
                limits=['Low cloud surface is not the anatomical ankle midpoint.',
                        'Visibility and alternating stance can mimic temporal offsets.',
                        'Only five seconds and no independent cam1 person keypoints.',
                        'No physical exposure offset is established and no offset is applied.'])


def diagnose(output_root, clip_id):
    output_root = Path(output_root).resolve()
    manifest_path = output_root/'sequence_review/manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    clip = next(c for c in manifest['clips'] if c['id'] == clip_id)
    calibration = CalibrationSnapshot.load(manifest['calibration_snapshot'])
    session = SessionIndex.load(Path(calibration.data['sessions_root'])/clip['session_id'],
                                calibration.frame_map_path(clip['session_id']))
    window = Path(clip['source_window'])
    metadata = {(r['lidar_index'], r['motion_id']): r for r in
                (json.loads(line) for line in (window/'quality.jsonl').read_text(encoding='utf-8').splitlines())}
    teacher_manifest = json.loads((window/'teacher/manifest.json').read_text(encoding='utf-8'))
    teacher_tracks = {}
    for row in teacher_manifest['tracks']:
        name = f"{clip['session_id']}:{window.name}:{row['track_id']}:{row['segment_id']}"
        teacher_tracks[name] = torch.load(window/'teacher'/row['path'], map_location='cpu', weights_only=False)
    targets = [frame['scan_mid_us'] for frame in clip['frames']]
    teacher_samples = {name: sample_track(track, targets) for name, track in teacher_tracks.items()}
    per_frame, people = [], defaultdict(list)
    previous_raw = previous_viewer = None
    previous_people = {}
    identity_candidates = []
    plane_origin = np.asarray(calibration.reference_plane['origin_viewer_m'])
    plane_normal = np.asarray(calibration.reference_plane['normal_viewer'])
    for frame_position, frame in enumerate(clip['frames']):
        index = frame['lidar_index']
        pair = session.pairs()[index]
        raw = session.read_cloud(pair, viewer=False)
        viewer = session.read_cloud(pair)
        scan = dict(lidar_index=index, source_png=pair['lidar_path'], source_png_sha256=digest(pair['lidar_path']),
                    scan_start_us=pair['scan_start_us'], scan_end_us=pair['scan_end_us'],
                    scan_mid_us=pair['scan_mid_us'], scan_duration_us=pair['scan_duration_us'],
                    cameras=pair['cameras'], raw=cloud_stats(raw), viewer=cloud_stats(viewer), athletes=[])
        if previous_raw is not None:
            scan['raw_nearest_previous'] = nearest_coverage(previous_raw, raw)
            scan['viewer_nearest_previous'] = nearest_coverage(previous_viewer, viewer)
            scan['period_us'] = pair['scan_mid_us']-per_frame[-1]['scan_mid_us']
        current_people, memberships = {}, {}
        for athlete in frame['athletes']:
            motion_id = athlete['motion_id']
            meta = metadata[(index, motion_id)]
            with np.load(athlete['fit_path'], allow_pickle=False) as data:
                arrays = {key: data[key].copy() for key in data.files}
            sample = teacher_samples[motion_id]
            arrays['_teacher_root'] = sample['smpl_params_incam']['global_orient'][frame_position]
            arrays['_teacher_body'] = sample['smpl_params_incam']['body_pose'][frame_position]
            arrays['_teacher_camera_translation_viewer'] = calibration.camera_to_viewer(
                sample['smpl_params_incam']['transl'][frame_position], 'cam2')
            cloud, source_indices = arrays['source_cloud'], arrays['source_indices']
            if len(source_indices) != len(cloud) or source_indices.min() < 0 or source_indices.max() >= len(viewer):
                raise ValueError(f'Invalid actual source index at scan {index}/{motion_id}')
            source_error = float(np.abs(cloud-viewer[source_indices]).max())
            if source_error != 0:
                raise ValueError(f'Cloud is not exactly the recorded source subset: {index}/{motion_id}')
            heights = (cloud-plane_origin) @ plane_normal
            low_cloud = cloud[heights <= np.quantile(heights, .12)]
            low_uv = {}
            for camera in ('cam1', 'cam2'):
                uv, depth = calibration.project(low_cloud, camera, distort=True)
                good = (depth > 0) & np.isfinite(uv).all(1)
                low_uv[camera] = np.median(uv[good], axis=0).tolist() if good.any() else [None, None]
            entry = dict(motion_id=motion_id, lidar_index=index, scan_mid_us=pair['scan_mid_us'],
                         accepted=athlete['accepted'], fit_path=athlete['fit_path'], fit_sha256=digest(athlete['fit_path']),
                         source_indices_count=len(source_indices), unique_source_indices=int(len(np.unique(source_indices))),
                         source_indices_sha256=hashlib.sha256(source_indices.astype('<i8').tobytes()).hexdigest(),
                         exact_recorded_subset=True, max_source_error_m=source_error,
                         cloud=cloud_stats(cloud), pelvis_m=arrays['transl'].tolist(),
                         cloud_median_minus_pelvis_m=(np.median(cloud, axis=0)-arrays['transl']).tolist(),
                         projection={c: projection_stats(cloud, calibration, c) for c in ('cam1', 'cam2')},
                         low_cloud_median_brown_uv_px=low_uv, teacher_sampling=meta['teacher_sampling'],
                         teacher_sample_valid=bool(sample['valid'][frame_position]),
                         teacher_camera_translation_viewer_m=arrays['_teacher_camera_translation_viewer'].tolist(),
                         qa=meta['qa'])
            current_people[motion_id] = arrays
            memberships[motion_id] = set(source_indices.tolist())
            if motion_id in previous_people:
                before = previous_people[motion_id]
                dc = np.median(cloud, axis=0)-np.median(before['source_cloud'], axis=0)
                dp = arrays['transl']-before['transl']
                dj = np.linalg.norm(arrays['joints']-before['joints'], axis=1)
                entry['step'] = dict(point_count_fraction_change=float(len(cloud)/len(before['source_cloud'])-1),
                                     median_step_m=float(np.linalg.norm(dc)), median_step_xyz_m=dc.tolist(),
                                     pelvis_step_m=float(np.linalg.norm(dp)), pelvis_step_xyz_m=dp.tolist(),
                                     median_minus_pelvis_step_m=float(np.linalg.norm(dc-dp)),
                                     joint_step_m=numbers(dj),
                                     root_rotation=rotation_steps(before['root_pose'], arrays['root_pose']),
                                     body_rotation=rotation_steps(before['body_pose'], arrays['body_pose']),
                                     teacher_root_rotation=rotation_steps(before['_teacher_root'], arrays['_teacher_root']),
                                     teacher_body_rotation=rotation_steps(before['_teacher_body'], arrays['_teacher_body']),
                                     teacher_camera_translation_step_m=float(np.linalg.norm(
                                         arrays['_teacher_camera_translation_viewer']-before['_teacher_camera_translation_viewer'])),
                                     nearest_previous=nearest_coverage(before['source_cloud'], cloud),
                                     nearest_after_pelvis_translation=nearest_coverage(before['source_cloud'], cloud, dp),
                                     nearest_after_cloud_median_translation=nearest_coverage(before['source_cloud'], cloud, dc))
                for camera in ('cam1', 'cam2'):
                    p0 = projection_stats(before['source_cloud'], calibration, camera)['median_uv_px']
                    p1 = entry['projection'][camera]['median_uv_px']
                    entry['step'][camera+'_cloud_median_pixel_step'] = float(np.linalg.norm(np.asarray(p1)-p0))
            people[motion_id].append(entry)
            scan['athletes'].append(entry)
        overlap = []
        names = sorted(memberships)
        for i, name in enumerate(names):
            for other in names[i+1:]:
                common = len(memberships[name] & memberships[other])
                if common:
                    overlap.append(dict(motion_ids=[name, other], shared_source_return_count=common))
        scan['cross_person_source_overlap'] = overlap
        if previous_people and set(previous_people) == set(current_people):
            ids = sorted(current_people)
            before = np.array([np.median(previous_people[k]['source_cloud'], axis=0) for k in ids])
            after = np.array([np.median(current_people[k]['source_cloud'], axis=0) for k in ids])
            costs = np.linalg.norm(before[:, None]-after[None, :], axis=-1)
            left, right = linear_sum_assignment(costs)
            swaps = [(ids[i], ids[j]) for i, j in zip(left, right) if i != j]
            if swaps:
                identity_candidates.append(dict(lidar_index=index, nearest_centroid_swaps=swaps))
        previous_raw, previous_viewer, previous_people = raw, viewer, current_people
        per_frame.append(scan)
    summaries = {}
    for name, entries in people.items():
        steps = [r['step'] for r in entries if 'step' in r]
        ranked = sorted((r for r in entries if 'step' in r),
                        key=lambda r: r['step']['median_minus_pelvis_step_m'], reverse=True)
        track = teacher_tracks[name]
        times = _array(track['common_time_us']).astype(np.int64)
        orientation = _array(track['smpl_params_incam']['global_orient'])
        body = _array(track['smpl_params_incam']['body_pose'])
        local = (times >= clip['left_us']-40000) & (times <= clip['right_us']+40000)
        local_idx = np.flatnonzero(local)
        summary = dict(frames=len(entries), accepted=sum(r['accepted'] for r in entries),
                       point_count=numbers([r['cloud']['point_count'] for r in entries]),
                       median_step_m=numbers([s['median_step_m'] for s in steps]),
                       pelvis_step_m=numbers([s['pelvis_step_m'] for s in steps]),
                       median_minus_pelvis_step_m=numbers([s['median_minus_pelvis_step_m'] for s in steps]),
                       relative_point_count_change=numbers([abs(s['point_count_fraction_change']) for s in steps]),
                       coverage_within_5cm=numbers([s['nearest_previous']['within_5cm'] for s in steps]),
                       coverage_after_pelvis_translation_within_5cm=numbers(
                           [s['nearest_after_pelvis_translation']['within_5cm'] for s in steps]),
                       ranked_cloud_vs_pelvis_displacement=[dict(lidar_index=r['lidar_index'], **r['step']) for r in ranked[:8]],
                       teacher_period_us=numbers(np.diff(times[local])),
                       teacher_identity_ambiguous_count=int(_array(track.get('identity_ambiguous', np.zeros(len(times))))[local].sum()),
                       teacher_root_step_deg=numbers([(Rotation.from_rotvec(orientation[i-1]).inv()*
                                                     Rotation.from_rotvec(orientation[i])).magnitude()*180/np.pi
                                                     for i in local_idx[1:]]),
                       teacher_body_max_step_deg=numbers([rotation_steps(body[i-1], body[i])['max_deg'] for i in local_idx[1:]]),
                       teacher_nearest_error_us=numbers([r['teacher_sampling']['model_time_error_us'] for r in entries]),
                       teacher_bracket_gap_us=numbers([r['teacher_sampling']['teacher_right_time_us']-
                                                       r['teacher_sampling']['teacher_left_time_us'] for r in entries]),
                       all_scan_teacher_samples_valid=all(r['teacher_sample_valid'] for r in entries),
                       cam2_exploratory_time_proxy=exploratory_foot_lag(track, entries, calibration.camera('cam2')))
        # Report how much real translation can happen within a single measured
        # scan; this is a scale estimate, not per-return deskew evidence.
        dt = np.diff([r['scan_mid_us'] for r in entries])/1e6
        speed = np.array([s['pelvis_step_m'] for s in steps])/dt
        duration = np.median([r['scan_duration_us'] for r in per_frame])/1e6
        summary['fitted_pelvis_speed_mps'] = numbers(speed)
        summary['estimated_pelvis_motion_over_scan_m'] = numbers(speed*duration)
        summaries[name] = summary
    report = dict(schema='badminton-measured-cloud-jitter-diagnostic-v1', clip_id=clip_id,
                  session_id=clip['session_id'], source_window=str(window), scan_count=len(per_frame),
                  native_lidar_indices=[r['lidar_index'] for r in per_frame],
                  calibration_snapshot=str(calibration.snapshot_json), calibration_digest=calibration.data['snapshot_digest'],
                  review_manifest=str(manifest_path), review_manifest_sha256=digest(manifest_path),
                  source_geometry_modified=False, gpu_used=False,
                  scan_period_us=numbers([r['period_us'] for r in per_frame if 'period_us' in r]),
                  scan_duration_us=numbers([r['scan_duration_us'] for r in per_frame]),
                  native_camera_time_error_us={c: numbers([r['cameras'][c]['time_error_us'] for r in per_frame]) for c in ('cam1', 'cam2')},
                  raw_point_count=numbers([r['raw']['point_count'] for r in per_frame]),
                  all_displayed_clouds_exact_recorded_subsets=all(r['exact_recorded_subset'] for rows in people.values() for r in rows),
                  frames_with_cross_person_source_overlap=[r['lidar_index'] for r in per_frame if r['cross_person_source_overlap']],
                  centroid_identity_swap_candidates=identity_candidates,
                  review_display=dict(camera_pose='one fixed view computed over all clip vertices, unless user drags',
                                      geometry_interpolation=False, timestamp_based_native_scan_steps=True,
                                      measured_cloud_color='yellow', temporal_sampling_hz=10,
                                      source='hmr4d/lidar_mocap/review_assets/review.js orient/draw/animate'),
                  interpretation_limits=['Frame-to-frame source_indices are compact valid-return indices, not persistent physical point IDs.',
                                         'No source per-return timestamp exists in fixed24, so true scan deskew is not available here.',
                                         'A constant rigid extrinsic changes projection/absolute placement but cannot create discontinuous raw XYZ motion.',
                                         'Cloud median may move with visible surface/arm/leg coverage even when pelvis is stable.',
                                         'Fitted pelvis is not independent ground truth; it already uses the measured cloud.',
                                         'Exploratory time proxy is anatomical/surface-confounded and must not be adopted as an exposure calibration.'],
                  athletes=summaries, frames=per_frame)
    jump_candidates = []
    for name, entries in people.items():
        for entry in entries:
            step = entry.get('step', {})
            if (step.get('pelvis_step_m', 0) > .20 and step.get('median_step_m', 1) < .12
                    and step['nearest_previous']['within_5cm'] > .75):
                jump_candidates.append(dict(motion_id=name, lidar_index=entry['lidar_index'],
                                            pelvis_step_m=step['pelvis_step_m'], cloud_median_step_m=step['median_step_m'],
                                            cloud_5cm_coverage=step['nearest_previous']['within_5cm'],
                                            fitted_root_step_deg=step['root_rotation']['max_deg'],
                                            teacher_root_step_deg=step['teacher_root_rotation']['max_deg'],
                                            interpretation='Fitted motion exceeds the relatively persistent measured cloud; inspect fit continuity and real articulation.'))
    report['fitted_jump_candidates'] = jump_candidates
    proxies = [a['cam2_exploratory_time_proxy'] for a in summaries.values() if 'best_offset_ms' in a['cam2_exploratory_time_proxy']]
    report['physical_time_offset_established'] = False
    report['exploratory_proxy_best_offset_spread_ms'] = [min(a['best_offset_ms'] for a in proxies),
                                                       max(a['best_offset_ms'] for a in proxies)] if proxies else None
    return report


def markdown(report):
    line = [f"# {report['clip_id']} 实测点抖动诊断", '',
            f"会话 `{report['session_id']}`，原生扫描 {report['native_lidar_indices'][0]}–{report['native_lidar_indices'][-1]}，"
            f"共 {report['scan_count']} 帧；{len(report['athletes'])} 名运动员。", '',
            f"200 份显示点云逐帧与原 PNG 解码后的实际 source_indices 精确对应：{report['all_displayed_clouds_exact_recorded_subsets']}。"
            f"跨运动员共享原始返回点的扫描：{report['frames_with_cross_person_source_overlap']}；"
            f"几何近邻提示的身份互换：{report['centroid_identity_swap_candidates']}。", '',
            f"扫描持续时间中位数 {report['scan_duration_us']['median']/1000:.3f} ms；"
            f"扫描中心间隔中位数 {report['scan_period_us']['median']/1000:.3f} ms。", '',
            '| 轨迹 | 点数范围 | 云中位数步长 P90 (cm) | pelvis 步长 P90 (cm) | 云减 pelvis 步长 P90 (cm) | 5cm 覆盖中位数 | 疑似变化扫描 |',
            '|---|---:|---:|---:|---:|---:|---|']
    for name, a in report['athletes'].items():
        scan = ', '.join(str(r['lidar_index']) for r in a['ranked_cloud_vs_pelvis_displacement'][:5])
        line.append(f"| {name.split(':')[-2]}:{name.split(':')[-1]} | {a['point_count']['min']:.0f}–{a['point_count']['max']:.0f} | "
                    f"{a['median_step_m']['p90']*100:.2f} | {a['pelvis_step_m']['p90']*100:.2f} | "
                    f"{a['median_minus_pelvis_step_m']['p90']*100:.2f} | {a['coverage_within_5cm']['median']:.3f} | {scan} |")
    line += ['', '上述扫描按云中位数相对拟合 pelvis 的步长排序，用于逐帧复查；真实挥拍、迈步和可见表面改变也会产生该指标，不能直接标成测量错误。', '']
    if report['fitted_jump_candidates']:
        line += ['拟合连续性值得优先复查：', '']
        for row in report['fitted_jump_candidates']:
            line.append(f"- 扫描 {row['lidar_index']} / 轨迹 {row['motion_id'].split(':')[-2]}：pelvis 步长 "
                        f"{row['pelvis_step_m']*100:.2f} cm，原云中位数只动 {row['cloud_median_step_m']*100:.2f} cm，"
                        f"相邻原云5cm覆盖 {row['cloud_5cm_coverage']:.3f}；拟合/教师根旋转步长 "
                        f"{row['fitted_root_step_deg']:.2f}/{row['teacher_root_step_deg']:.2f}°。")
    line += ['',
             '播放使用约10 Hz原扫描阶跃，摄像机视图在切入clip时根据整个clip固定，没有逐帧自动缩放；黄色点不作几何插值。', '',
             '常量内外参可以纠正叠加位置和Brown投影，但无法消除原始XYZ本身的帧间变化。扫描约66 ms内的人体运动可能形成不同肢体位置的混合；目前fixed24不保留逐返回点时间，不能可靠deskew。', '',
             '时间偏移仅作低点云表面与实际cam2脚踝的探索性运动代理比较。人体表面与关节点并非同一观测、脚步和遮挡改变可模拟偏移，不能据此采用校准偏移。cam1没有独立关节观测，不估计其物理时差。', '']
    for name, a in report['athletes'].items():
        lag = a['cam2_exploratory_time_proxy']
        if 'best_offset_ms' in lag:
            line.append(f"- 轨迹 {name.split(':')[-2]}：代理最优 {lag['best_offset_ms']:+.0f} ms，"
                        f"距最优1 px内范围 {lag['within_1px_of_best_ms']} ms，零偏移残差 "
                        f"{lag['zero_offset']['demeaned_median_error_px']:.2f} px、最优 "
                        f"{lag['best']['demeaned_median_error_px']:.2f} px；此数值不等于曝光时差。")
    line += ['', '源PNG路径/hash、每帧真实点数/索引hash、完整扫描起止、nearest视频PTS及教师插值/SO(3)统计保存在同目录 `jitter_stats.json`。本诊断未修改源数据、校准、拟合、复查页面或实测XYZ。', '']
    return '\n'.join(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('outputs/badminton_corner_fix_v1'))
    parser.add_argument('--clip', default='clip10')
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = diagnose(args.output_root, args.clip)
    destination = args.output_root/'calibration_refinement'
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'jitter_stats.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (destination/'jitter_stats.md').write_text(markdown(report), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('clip_id', 'scan_count', 'all_displayed_clouds_exact_recorded_subsets',
                                                  'frames_with_cross_person_source_overlap', 'centroid_identity_swap_candidates')}, ensure_ascii=False))
    print(str((destination/'jitter_stats.json').resolve()))


if __name__ == '__main__':
    main()
