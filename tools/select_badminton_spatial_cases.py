"""Select spatially diverse native scans from completed, read-only fits.

Positions are medians of actual source_cloud points after the exact frozen
viewer -> raw -> registered court transform. SMPL pelvis never determines a
position. No source XYZ, calibration, identity, or native clock is rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from hmr4d.lidar_mocap.calibration import CalibrationSnapshot


X_BOUNDS, Y_BOUNDS = (-3.05, 3.05), (-6.7, 6.7)
GRID = (3, 6)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid_cell(xy):
    x, y = xy
    if not X_BOUNDS[0] <= x <= X_BOUNDS[1] or not Y_BOUNDS[0] <= y <= Y_BOUNDS[1]:
        return None
    return [min(2, int((x-X_BOUNDS[0])/(X_BOUNDS[1]-X_BOUNDS[0])*3)),
            min(5, int((y-Y_BOUNDS[0])/(Y_BOUNDS[1]-Y_BOUNDS[0])*6))]


def completed_windows(root, digest):
    eligible, excluded = [], []
    for path in sorted((root/'sessions').glob('*/windows/*/fit_report.json')):
        config = json.loads((path.parent/'window.json').read_text(encoding='utf-8'))
        report = json.loads(path.read_text(encoding='utf-8'))
        if config['calibration_digest'] != digest:
            excluded.append(dict(window=str(path.parent), reason='calibration_digest_mismatch'))
            continue
        if not (path.parent/'quality.jsonl').exists() or report.get('fitted', 0) < 1:
            excluded.append(dict(window=str(path.parent), reason='no_completed_quality_rows'))
            continue
        eligible.append((config, path.parent))
    result = []
    for config, path in sorted(eligible, key=lambda item: -(item[0]['right_us']-item[0]['left_us'])):
        sid = config['session_id']
        if any(old['session_id'] == sid and old['left_us'] <= config['left_us']
               and old['right_us'] >= config['right_us'] for old, _ in result):
            excluded.append(dict(window=str(path), reason='contained_smoke_superseded_by_completed_longer_window'))
            continue
        if any(old['session_id'] == sid and max(old['left_us'], config['left_us'])
               < min(old['right_us'], config['right_us']) for old, _ in result):
            raise ValueError('Partially overlapping windows have unverified cross-window identity')
        result.append((config, path))
    return result, excluded


def select(root, target_count, context_seconds=2):
    root = Path(root).resolve()
    calibration = CalibrationSnapshot.load(root/'calibration/calibration_snapshot.json')
    source_m = calibration.Q
    source_inverse = np.linalg.inv(source_m)
    registration = calibration.data['court_registration']
    registered_q = np.asarray(registration['R_raw_lidar_to_court'], dtype=float)
    registered_offset = np.asarray(registration['t_raw_lidar_to_court_m'], dtype=float)
    windows, excluded_windows = completed_windows(root, calibration.data['snapshot_digest'])
    old_report_path = root/'calibration_refinement/joint_alignment/joint_alignment.json'
    old_report = json.loads(old_report_path.read_text(encoding='utf-8'))
    already_evaluated = {(row['session_id'], row['lidar_index']) for row in old_report['measurements']}
    frames, positions, candidates, groups = {}, [], [], {}
    counts, sessions = Counter(), defaultdict(lambda: dict(windows=[], fitted_rows=0, eligible_anchors=0))
    shape_min, shape_max = np.full(10, np.inf), np.full(10, -np.inf)
    context_us = round(context_seconds*1e6)
    source_reports = []
    for config, window in windows:
        sid = config['session_id']
        sessions[sid]['windows'].append(dict(source_window=str(window), duration_s=(config['right_us']-config['left_us'])/1e6))
        source_reports.append(dict(path=str(window/'fit_report.json'), sha256=sha(window/'fit_report.json')))
        with (window/'quality.jsonl').open(encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        track_times = defaultdict(list)
        for row in rows:
            if row.get('fit_path'):
                track_times[row['motion_id']].append(row['scan_mid_us'])
        track_times = {name: np.unique(np.asarray(times, np.int64)) for name, times in track_times.items()}
        for row in rows:
            counts['quality_rows'] += 1
            sessions[sid]['fitted_rows'] += 1
            key = (sid, row['lidar_index'])
            record = frames.setdefault(key, dict(session_id=sid, source_window=str(window),
                frame={name: row[name] for name in ('scan_mid_us','lidar_index','lidar_path','cameras')},
                court_centres=[], motion_ids=set()))
            if record['source_window'] != str(window):
                raise ValueError('A native scan was assigned to more than one completed window')
            record['frame'].setdefault('athletes', []).append({name: row[name] for name in
                ('motion_id','fit_path','accepted','qa','rejection_reasons')})
            record['motion_ids'].add(row['motion_id'])
            if key in already_evaluated:
                counts['previously_evaluated_athlete_rows'] += 1
                continue
            if not row.get('accepted'):
                counts['rejected_fit_athlete_rows'] += 1
                continue
            path = Path(row['fit_path'])
            if not path.is_file():
                counts['missing_fit_file'] += 1
                continue
            with np.load(path, allow_pickle=False) as fit:
                cloud, shape = fit['source_cloud'], fit['shape']
            if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) < 80 or not np.isfinite(cloud).all():
                counts['invalid_actual_source_cloud'] += 1
                continue
            if shape.shape != (10,) or not np.isfinite(shape).all():
                counts['invalid_shape'] += 1
                continue
            if len(cloud) != row['source_point_count']:
                raise ValueError('Stored source cloud count differs from its quality evidence')
            shape_min, shape_max = np.minimum(shape_min, shape), np.maximum(shape_max, shape)
            # Componentwise medians and rotations do not commute. Transform all
            # actual points first; only then compute the court-space median.
            raw = cloud.astype(np.float64) @ source_inverse.T
            court = raw @ registered_q.T + registered_offset
            centre = np.median(court, axis=0)
            cell = grid_cell(centre[:2])
            position = dict(motion_id=row['motion_id'], median_viewer_xyz_m=np.median(cloud, axis=0).tolist(),
                            median_court_xyz_m=centre.tolist(), court_xy_m=centre[:2].tolist(),
                            grid_cell=cell, source_point_count=len(cloud),
                            position_source='median_after_transform_of_all_actual_source_cloud_returns')
            record['court_centres'].append(position)
            positions.append(dict(session_id=sid, lidar_index=row['lidar_index'], **position))
            counts['finite_actual_cloud_positions'] += 1
            if cell is None:
                counts['outside_normative_court_centres'] += 1
                continue
            times = track_times[row['motion_id']]
            before = np.searchsorted(times, row['scan_mid_us']-context_us, side='right')-1
            after = np.searchsorted(times, row['scan_mid_us']+context_us, side='left')
            if before < 0 or after >= len(times):
                counts['insufficient_two_sided_context'] += 1
                continue
            if np.max(np.diff(times[before:after+1]), initial=0) > 250000:
                counts['context_tracking_or_cloud_gap_over_250ms'] += 1
                continue
            candidate = dict(frame_key=key, source_window=str(window), session_id=sid,
                             motion_id=row['motion_id'], xy=centre[:2], grid_cell=cell,
                             context_start_us=int(times[before]), context_end_us=int(times[after]),
                             source_point_count=len(cloud), scan_mid_us=row['scan_mid_us'])
            candidates.append(candidate)
            sessions[sid]['eligible_anchors'] += 1
        print(f"[spatial] {sid}/{window.name}: {len(rows)} fitted rows, {len(candidates)} eligible anchors", flush=True)
    selected, selected_keys = [], set()

    def compatible(candidate):
        if candidate['frame_key'] in selected_keys:
            return False
        candidate_frame = frames[candidate['frame_key']]
        for old in selected:
            if candidate['source_window'] == old['source_window'] and abs(candidate['scan_mid_us']-old['scan_mid_us']) < 1000000:
                if candidate_frame['motion_ids'] & frames[old['frame_key']]['motion_ids']:
                    return False
        return True

    def distance(candidate):
        if not selected:
            return float(np.linalg.norm(candidate['xy']))
        return min(float(np.linalg.norm(candidate['xy']-old['xy'])) for old in selected)

    def choose(pool):
        choices = [candidate for candidate in pool if compatible(candidate)]
        if not choices:
            return False
        best = max(choices, key=lambda c: (distance(c), c['source_point_count'], -c['scan_mid_us']))
        selected.append(best)
        selected_keys.add(best['frame_key'])
        return True

    # Reserve real additional observations in the less-covered sessions before
    # cell coverage. Session3's six seconds may only support two independent
    # frames with the required +/-2s fitting context; never invent extra slots.
    quota = {'20261012_161544': 4, '20261012_163452': 2}
    for sid, count in quota.items():
        for _ in range(count):
            if len(selected) >= target_count or not choose([c for c in candidates if c['session_id'] == sid]):
                break
    occupied = sorted({tuple(candidate['grid_cell']) for candidate in candidates})
    for cell in occupied:
        if len(selected) >= target_count:
            break
        if cell not in {tuple(candidate['grid_cell']) for candidate in selected}:
            choose([candidate for candidate in candidates if tuple(candidate['grid_cell']) == cell])
    while len(selected) < target_count and choose(candidates):
        pass
    cases = []
    for number, candidate in enumerate(selected, 1):
        record = frames[candidate['frame_key']]
        cases.append(dict(key=f'spatial{number:02d}', clip_id=f'spatial{number:02d}',
                          session_id=record['session_id'], source_window=record['source_window'],
                          frame=record['frame'], court_centres=record['court_centres'],
                          selected_optimization_athlete_motion_id=candidate['motion_id'],
                          selected_court_xy_m=candidate['xy'].tolist(), grid_cell=candidate['grid_cell'],
                          available_context=dict(start_us=candidate['context_start_us'], end_us=candidate['context_end_us'],
                                                 required_before_after_s=context_seconds,
                                                 identity_scope='same source_window and motion_id; no cross-window identity join')))
    selected_cells = sorted({tuple(case['grid_cell']) for case in cases})
    actual_evaluation_cells = sorted({tuple(p['grid_cell']) for case in cases for p in case['court_centres'] if p['grid_cell'] is not None})
    xy = np.array([p['court_xy_m'] for p in positions])
    for sid, detail in sessions.items():
        selected_sid = [case for case in cases if case['session_id'] == sid]
        detail['selected_native_frames'] = len(selected_sid)
        detail['selected_anchor_cells'] = sorted({tuple(case['grid_cell']) for case in selected_sid})
        own_positions = np.array([p['court_xy_m'] for p in positions if p['session_id'] == sid])
        detail['actual_court_xy_range_m'] = [own_positions.min(0).tolist(), own_positions.max(0).tolist()] if len(own_positions) else None
    result = dict(schema='badminton-native-spatial-cases-v1', calibration_digest=calibration.data['snapshot_digest'],
                  source_root=str(root), source_geometry_or_time_modified=False, GPU_used=False,
                  position_definition='median in registered court coordinates AFTER transforming every source_cloud point',
                  transform=dict(SourceM_raw_to_viewer=source_m.tolist(), SourceM_exact_inverse=source_inverse.tolist(),
                                 registered_Q_raw_to_court=registered_q.tolist(), registered_q_m=registered_offset.tolist(),
                                 row_vector_formula='court=(source_cloud_viewer @ inv(SourceM).T) @ registered_Q.T + registered_q'),
                  normative_court_bounds_m=dict(x=list(X_BOUNDS), y=list(Y_BOUNDS)), grid_shape=list(GRID),
                  candidate_athlete_positions=len(positions), eligible_optimization_anchors=len(candidates),
                  eligible_unique_scan_frames=len({c['frame_key'] for c in candidates}),
                  requested_native_frames=target_count, selected_native_frames=len(cases),
                  previous_evaluated_unique_scans=len(already_evaluated), previous_report_sha256=sha(old_report_path),
                  occupied_candidate_cells=[list(cell) for cell in occupied], selected_anchor_cells=[list(cell) for cell in selected_cells],
                  actual_evaluated_person_cells=[list(cell) for cell in actual_evaluation_cells],
                  uncovered_occupied_cells=[list(cell) for cell in occupied if cell not in actual_evaluation_cells],
                  empty_candidate_cells=[[x,y] for x in range(3) for y in range(6) if (x,y) not in occupied],
                  candidate_court_xy_range_m=[xy.min(0).tolist(), xy.max(0).tolist()] if len(xy) else None,
                  shape_statistics=dict(dimension=10, minimum_beta=shape_min.tolist(), maximum_beta=shape_max.tolist(),
                                        used_for_position=False), exclusions=dict(counts), excluded_windows=excluded_windows,
                  completed_source_reports=source_reports, sessions=dict(sessions),
                  temporal_diversity='unique native scan; shared same-window motion_ids separated by at least1s',
                  selection='reserve real session2/session3 observations, occupied3x6 cells, then spatial farthest anchors',
                  limits=['Normative court registration is not independently surveyed.',
                          'Source-cloud surface median is an observed position proxy, not an anatomical body centre.',
                          'Only completed frozen-baseline fit windows are available; session3 has only6s.',
                          'No unobserved court cell is populated; no nativeXYZ is synthesized.',
                          'Same-window native cloud context is required; no identities are joined across windows.'],
                  cases=cases)
    return result, positions


def svg(report, positions):
    width, height = 740, 1000
    def point(xy):
        return 70+(xy[0]+3.05)/6.1*600, 60+(6.7-xy[1])/13.4*860
    colors = {'20261012_154726':'#77a9df', '20261012_161544':'#ffa75d', '20261012_163452':'#9ee48c'}
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#111a25"/>',
             '<text x="35" y="30" fill="white" font-size="19">Actual measured cloud centres · 3 × 6 court cells</text>',
             '<rect x="70" y="60" width="600" height="860" fill="none" stroke="#eeeeee"/>']
    for x in range(1, 3):
        xx, _ = point([X_BOUNDS[0]+x*(X_BOUNDS[1]-X_BOUNDS[0])/3, 0])
        lines.append(f'<path d="M{xx:.2f},60V920" stroke="#495463"/>')
    for y in range(1, 6):
        _, yy = point([0, Y_BOUNDS[0]+y*(Y_BOUNDS[1]-Y_BOUNDS[0])/6])
        lines.append(f'<path d="M70,{yy:.2f}H670" stroke="#495463"/>')
    for p in positions:
        if p['grid_cell'] is None:
            continue
        x,y = point(p['court_xy_m'])
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.25" fill="{colors[p["session_id"]]}" opacity=".14"/>')
    for case in report['cases']:
        x,y = point(case['selected_court_xy_m'])
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{colors[case["session_id"]]}" stroke="white"/>')
        lines.append(f'<text x="{x+9:.2f}" y="{y-7:.2f}" fill="white" font-size="12">{case["key"][7:]}</text>')
    lines += ['<text x="35" y="952" fill="#d9e5ee" font-size="13">Blue: session1 · orange: session2 · green: session3 · circles: selected anchors</text>',
              '<text x="35" y="976" fill="#d9e5ee" font-size="13">Court outline is normative; all plotted points come from actual source clouds.</text>', '</svg>']
    return '\n'.join(lines)


def markdown(report):
    lines = ['# 实测人体点云空间选帧', '',
             f"选择 {report['selected_native_frames']} 个新增原生扫描；排除既有 {report['previous_evaluated_unique_scans']} 个已评估扫描。"
             f"候选实际人体位置 {report['candidate_athlete_positions']}，满足双侧2s上下文的优化锚点 {report['eligible_optimization_anchors']}。", '',
             f"3×6网格实际occupied {len(report['occupied_candidate_cells'])}/18；选中锚点覆盖 {len(report['selected_anchor_cells'])}/18；"
             f"所选扫描全部人体实际覆盖 {len(report['actual_evaluated_person_cells'])}/18。没有候选的cell：{report['empty_candidate_cells']}。", '',
             '位置来自每份source_cloud全部实际返回点：先用SourceM的精确逆还原raw，再用冻结registered_Q/q转court，最后在court坐标取median。没有使用SMPL pelvis，componentwise median与旋转不交换。', '',
             '| 会话 | 已完成窗口秒数 | 候选优化锚点 | 新扫描 | 实际位置XY范围 (m) |', '|---|---|---:|---:|---|']
    for sid, detail in report['sessions'].items():
        durations = ', '.join(f"{window['duration_s']:.1f}" for window in detail['windows'])
        lines.append(f"| {sid} | {durations} | "
                     f"{detail['eligible_anchors']} | {detail['selected_native_frames']} | {detail['actual_court_xy_range_m']} |")
    lines += ['', '同window中共现motion_id的扫描相隔至少1s，所有优化锚点均有本window双侧至少2s真实cloud上下文；不拼接跨window身份。session2可用已完成60s，session3仅6s，所以有上下文约束后的新扫描数量有限。', '',
              f"实际court XY总范围：{report['candidate_court_xy_range_m']}。外沿为X±3.05m、Y±6.7m；outside centre数量 {report['exclusions'].get('outside_normative_court_centres',0)}，未硬填无数据区域。", '',
              f"源shape维数10、全部有限；β统计留在JSON，未参与空间位置或填补。排除统计：`{json.dumps(report['exclusions'],ensure_ascii=False)}`。", '',
              'SVG只绘制实际点云经冻结刚性坐标变换后的位置及规范场地边界；规范场地不是独立实测survey。选帧是开发优化证据，mask/关节仍是模型预测，不能把改进解释成3D真值精度。', '',
              'JSON cases可直接送双相机原图mask评估；每frame保存原始时间、路径、nativecamera行以及所有已拟合运动员，selected_optimization_athlete_motion_id标记空间优化锚点。源素材、config、active与任何core均未改动。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('outputs/badminton_corner_fix_v1'))
    parser.add_argument('--count', type=int, default=24)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    report, positions = select(args.output_root, args.count)
    directory = args.output_root/'calibration_refinement'
    path = directory/'spatial_cases.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f"[ready] {path.resolve()}", flush=True)
    (directory/'spatial_cases.md').write_text(markdown(report), encoding='utf-8')
    (directory/'spatial_cases.svg').write_text(svg(report, positions), encoding='utf-8')
    print(json.dumps({'selected':report['selected_native_frames'],'occupied_cells':len(report['occupied_candidate_cells']),
                      'selected_cells':len(report['selected_anchor_cells']),
                      'sessions':{s:d['selected_native_frames'] for s,d in report['sessions'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
