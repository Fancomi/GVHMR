"""Isolated clip10 fitting continuity comparison on preserved teacher/clouds.

The original calibration, observations, fitted files and XYZ are read-only.
All variants fix the original neutral-SMPL beta to isolate motion fitting.
Candidate poses are written only to fit_continuity_benchmark artifacts.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix

from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.export import acceptance_reasons
from hmr4d.lidar_mocap.fitting import FittingConfig, SMPLFitter
from hmr4d.lidar_mocap.teacher import load_teacher_track, sample_track


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "max": float(values.max())} if len(values) else None


def summarize_final(output_root):
    """Keep the single-track comparison schema while attaching regression data."""
    directory = Path(output_root).resolve()/'calibration_refinement'
    selected_key = 'batch128_weight0.2'
    track1_path = directory/'fit_continuity_benchmark.torso1.json'
    report = json.loads(track1_path.read_text(encoding='utf-8'))
    baseline1 = json.loads((directory/'fit_continuity_benchmark.json').read_text(encoding='utf-8'))
    report['controlled_baseline'] = baseline1['variants']['batch8_weight0']
    report['cold_large_batch_baseline'] = baseline1['variants']['batch128_weight0']
    report['final_choice'] = {'variant':selected_key,'bootstrap_frames':8,'later_batch_size':128,
                              'config':report['variants'][selected_key]['config'],
                              'motion_reference':'Real observed cloud median minus anatomically continuous video pelvis/spine/neck centre offset; no raw XYZ modification',
                              'adoption_basis':'Track1 continuity, point-to-surface and independent Cam1 consistency improve substantially; small other-track residual changes reported below; no absolute speed gate'}
    report['cross_track_regression'] = []
    arrays = {}
    with np.load(directory/'fit_continuity_benchmark.torso1.npz') as stored:
        arrays.update({name:stored[name].copy() for name in stored.files})
    for track_id in (2,3,4):
        baseline_path = directory/f'fit_continuity_benchmark.track{track_id}.json'
        candidate_path = directory/f'fit_continuity_benchmark.torso{track_id}.json'
        old = json.loads(baseline_path.read_text(encoding='utf-8'))
        new = json.loads(candidate_path.read_text(encoding='utf-8'))
        baseline = old['variants']['batch128_weight0']
        candidate = new['variants'][selected_key]
        report['cross_track_regression'].append({'track_id':track_id,'motion_id':new['motion_id'],
                                                 'baseline_report':str(baseline_path),'candidate_report':str(candidate_path),
                                                 'baseline':baseline,'candidate':candidate,
                                                 'surface_mean_change_mm':(candidate['mean_surface_m']-baseline['mean_surface_m'])*1000,
                                                 'average_frame_p90_change_mm':(candidate['mean_of_frame_p90_surface_m']-baseline['mean_of_frame_p90_surface_m'])*1000,
                                                 'cam2_fit_change_px':candidate['mean_cam2_fitting_reprojection_px']-baseline['mean_cam2_fitting_reprojection_px'],
                                                 'cam1_detector_mean_change_px':candidate['rgb_fixed_observation_check']['cam1']['joint_pixel_error']['mean']-baseline['rgb_fixed_observation_check']['cam1']['joint_pixel_error']['mean'],
                                                 'candidate_array_prefix':f'track{track_id}_{selected_key}'})
        with np.load(directory/f'fit_continuity_benchmark.torso{track_id}.npz') as stored:
            arrays.update({f'track{track_id}_{name}':stored[name].copy() for name in stored.files})
    report['measurement_limits'] = [
        'Cloud median is an observed motion proxy, not ground-truth pelvis position; video torso offsets compensate trunk posture',
        'RGB residuals reuse the original associations and eligibility masks; detections are not manual GT and retain selection/calibration bias',
        'Surface mean and average frameP90 describe visible points only; this is not a measured 3D pose MPJPE',
        'This benchmark fixes each original beta to isolate motion fitting; new shape-free pipeline integration is a separate test',
        'Motion/root teacher comparisons preserve observed quick action rather than requiring low absolute speed',
    ]
    prefix = directory/'fit_continuity_benchmark.final'
    Path(str(prefix)+'.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(Path(str(prefix)+'.npz'),**arrays)
    cases=[(1,report['controlled_baseline'],report['variants'][selected_key])]
    cases += [(entry['track_id'],entry['baseline'],entry['candidate']) for entry in report['cross_track_regression']]
    lines=['Final: bootstrap8 -> batch128; continuity=.2, root-motion=.03, reprojection=1.0, cloud_torso reference; old beta fixed.','',
           '| Track | Surface mean old/new mm | Average frameP90 old/new mm | Cam2 fit old/new px | Cam1 detector mean old/new px | Pelvis-cloud motion residual P90 old/new cm |',
           '|---|---:|---:|---:|---:|---:|']
    for track_id,old,new in cases:
        lines.append(f"| {track_id} | {old['mean_surface_m']*1000:.3f}/{new['mean_surface_m']*1000:.3f} | {old['mean_of_frame_p90_surface_m']*1000:.3f}/{new['mean_of_frame_p90_surface_m']*1000:.3f} | {old['mean_cam2_fitting_reprojection_px']:.3f}/{new['mean_cam2_fitting_reprojection_px']:.3f} | {old['rgb_fixed_observation_check']['cam1']['joint_pixel_error']['mean']:.3f}/{new['rgb_fixed_observation_check']['cam1']['joint_pixel_error']['mean']:.3f} | {old['pelvis_vs_cloud_step_residual_m']['p90']*100:.3f}/{new['pelvis_vs_cloud_step_residual_m']['p90']*100:.3f} |")
    old44=next(row for row in report['original_stored_fit']['per_scan_motion'] if row['lidar_index']==44)
    new44=next(row for row in report['variants'][selected_key]['per_scan_motion'] if row['lidar_index']==44)
    report['scan44_highlight'] = {'old':old44,'new':new44,
                                 'note':'Cloud median step is a proxy, not pelvis GT; root step compared with actual teacher relative rotation'}
    Path(str(prefix)+'.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    lines.extend(['',f"Track1 scan44 pelvis step {old44['pelvis_step_m']*100:.3f} -> {new44['pelvis_step_m']*100:.3f} cm; original cloud median step {new44['cloud_median_step_m']*100:.3f} cm. Root step {old44['root_step_deg']:.3f} -> {new44['root_step_deg']:.3f} deg; teacher {new44['teacher_root_step_deg']:.3f} deg.",
                  '',f"Track1 context60 fitting time: original8-batch {report['controlled_baseline']['seconds']:.3f}s, cold128 {report['cold_large_batch_baseline']['seconds']:.3f}s, bootstrap8->128 {report['variants'][selected_key]['seconds']:.3f}s. Fitting/QA only; no teacher/I/O timing.",
                  '','Cloud/teacher motion are references, not pelvis or pose GT. No geometry interpolation, XYZ smoothing, absolute speed gate or calibration change. Small residual trades for tracks2/3/4 are retained in the table. Fixed-beta comparison does not replace shape-free end-to-end verification.'])
    Path(str(prefix)+'.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('final report',str(prefix),flush=True)


def rotation_angle(rotation):
    cosine = (np.trace(rotation, axis1=-2, axis2=-1) - 1) * .5
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def rgb_comparison(coco, rows, observations, calibration, motion_id):
    by_scan = {row['lidar_index']: i for i, row in enumerate(rows)}
    results = {}
    for camera, report in observations.items():
        values = []
        frames = []
        for frame in report['frames']:
            if frame['lidar_index'] not in by_scan:
                continue
            for athlete in frame['athletes']:
                if athlete['motion_id'] != motion_id or athlete.get('status') != 'measured':
                    continue
                index = by_scan[frame['lidar_index']]
                keypoints = np.asarray(athlete['all_coco17_observations'], dtype=np.float64)
                eligible = np.asarray(athlete['eligible_mask'], dtype=bool)
                projected, _ = calibration.project(coco[index], camera, distort=False)
                residual = np.linalg.norm(projected - keypoints[:, :2], axis=-1)
                values.extend(residual[eligible].tolist())
                frames.append({'lidar_index': frame['lidar_index'], 'eligible_joints': int(eligible.sum()),
                               'mean_px': float(residual[eligible].mean())})
        results[camera] = {'joint_pixel_error': distribution(values), 'frames': frames,
                           'fixed_original_association_and_eligibility': True,
                           'pixel_space': 'undistorted_pixels_K_preserved',
                           'not_ground_truth': True}
    return results


def summarize(result, rows, mask, cloud_centres, teacher_root, observations, calibration, motion_id):
    root = axis_angle_to_matrix(torch.as_tensor(result['root_pose']).reshape(-1, 3)).numpy()
    body = axis_angle_to_matrix(torch.as_tensor(result['body_pose']).reshape(-1, 23, 3)).numpy()
    pelvis = np.asarray(result['transl'])
    indices = np.flatnonzero(mask)
    step_indices = indices[1:]
    previous_indices = indices[:-1]
    pelvis_step = pelvis[step_indices] - pelvis[previous_indices]
    cloud_step = cloud_centres[step_indices] - cloud_centres[previous_indices]
    root_delta = root[step_indices] @ root[previous_indices].transpose(0, 2, 1)
    teacher_delta = teacher_root[step_indices] @ teacher_root[previous_indices].transpose(0, 2, 1)
    relative_error = root_delta @ teacher_delta.transpose(0, 2, 1)
    body_delta = body[step_indices] @ body[previous_indices].transpose(0, 1, 3, 2)
    frame_qa = [result['qa'][i] for i in indices]
    per_scan = []
    for j, index in enumerate(step_indices):
        per_scan.append({'lidar_index': rows[index]['lidar_index'],
                         'pelvis_step_m': float(np.linalg.norm(pelvis_step[j])),
                         'cloud_median_step_m': float(np.linalg.norm(cloud_step[j])),
                         'pelvis_vs_cloud_step_residual_m': float(np.linalg.norm(pelvis_step[j] - cloud_step[j])),
                         'root_step_deg': float(rotation_angle(root_delta[j])),
                         'teacher_root_step_deg': float(rotation_angle(teacher_delta[j])),
                         'root_motion_error_deg': float(rotation_angle(relative_error[j])),
                         'qa': result['qa'][index]})
    return {'clip_person_scans': len(indices),
            'mean_surface_m': float(np.mean([q['final_p2surface_mean_m'] for q in frame_qa])),
            'mean_of_frame_p90_surface_m': float(np.mean([q['final_p2surface_p90_m'] for q in frame_qa])),
            'mean_cam2_fitting_reprojection_px': float(np.mean([q['reprojection_mean_px'] for q in frame_qa])),
            'accepted_count': sum(not acceptance_reasons(result['qa'][i], rows[i]['source_point_count']) for i in indices),
            'pelvis_step_m': distribution(np.linalg.norm(pelvis_step, axis=-1)),
            'cloud_median_step_m': distribution(np.linalg.norm(cloud_step, axis=-1)),
            'pelvis_vs_cloud_step_residual_m': distribution(np.linalg.norm(pelvis_step - cloud_step, axis=-1)),
            'root_step_deg': distribution(rotation_angle(root_delta)),
            'root_motion_error_deg': distribution(rotation_angle(relative_error)),
            'body_mean_joint_step_deg': distribution(rotation_angle(body_delta).mean(-1)),
            'rgb_fixed_observation_check': rgb_comparison(np.asarray(result['coco17']), rows, observations, calibration, motion_id),
            'per_scan_motion': per_scan}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path('outputs/badminton_corner_fix_v1'))
    parser.add_argument('--output-prefix', type=Path)
    parser.add_argument('--weights', nargs='+', type=float, default=[0., .2])
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[8, 128])
    parser.add_argument('--root-motion-weight', type=float, default=.03)
    parser.add_argument('--track-id', type=int, default=1)
    parser.add_argument('--bootstrap-frames', type=int, default=0)
    parser.add_argument('--candidate-reprojection-weight', type=float, default=.3)
    parser.add_argument('--no-warm-start', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--summarize-final', action='store_true')
    args = parser.parse_args()
    if args.summarize_final:
        summarize_final(args.output_root)
        return
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    root = args.output_root.resolve()
    prefix = args.output_prefix or root/'calibration_refinement/fit_continuity_benchmark'
    prefix.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/'sequence_review/manifest.json').read_text(encoding='utf-8'))
    clip = next(clip for clip in manifest['clips'] if clip['id']=='clip10')
    window = Path(clip['source_window'])
    motion_id = next(key for key in clip['motion_colors'] if key.endswith(f':{args.track_id}:0'))
    rows = [json.loads(line) for line in (window/'quality.jsonl').read_text(encoding='utf-8').splitlines()]
    rows = sorted((row for row in rows if row['motion_id']==motion_id), key=lambda row: row['scan_mid_us'])
    calibration = CalibrationSnapshot.load(root/'calibration/calibration_snapshot.json')
    original = {key: [] for key in ('root_pose','body_pose','shape','transl','vertices','joints')}
    clouds = []
    source_hashes = {}
    for row in rows:
        path = Path(row['fit_path'])
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as fit:
            for key in original:
                original[key].append(fit[key].copy())
            clouds.append(fit['source_cloud'].copy())
    original = {key: np.stack(values) for key, values in original.items()}
    original['qa'] = [row['qa'] for row in rows]
    times = np.array([row['scan_mid_us'] for row in rows], dtype=np.int64)
    clip_mask = (times>=clip['left_us']) & (times<clip['right_us'])
    track = load_teacher_track(window/'teacher'/rows[0]['teacher'])
    sampled = sample_track(track, times)
    if not sampled['valid'].all():
        raise ValueError('Benchmark encountered invalid original teacher association')
    observations = {}
    for camera in ('cam1','cam2'):
        path = root/f'calibration_refinement/rgb_keypoints_{camera}_clip10.json'
        if path.exists():
            check = json.loads(path.read_text(encoding='utf-8'))
            if check.get('completed'):
                observations[camera] = check
    cloud_centres = np.stack([np.median(cloud, axis=0) for cloud in clouds])
    teacher_root = axis_angle_to_matrix(torch.from_numpy(sampled['smpl_params_incam']['global_orient'])).numpy()
    teacher_root = calibration.camera_rotation_to_viewer('cam2')[None] @ teacher_root
    fitter = SMPLFitter(device=args.device)
    regressor = fitter.coco_regressor.detach().cpu().numpy()
    original['coco17'] = np.einsum('jv,tvc->tjc', regressor, original['vertices'])
    report = {'schema':'badminton-fit-continuity-benchmark-v1','calibration_digest':manifest['calibration_digest'],
              'source_window':str(window),'motion_id':motion_id,'original_source_fits_sha256':source_hashes,
              'clip_indices':[row['lidar_index'] for row, valid in zip(rows,clip_mask) if valid],
              'fixed_beta':original['shape'][0].tolist(),
              'scope':'All original track frames fitted as context; only the original clip10 50 scans scored. Original SMPL beta fixed in every variant; no XYZ/source/calibration changes. Configured batch128 has only 60 available track frames.',
              'original_stored_fit':summarize(original,rows,clip_mask,cloud_centres,teacher_root,observations,calibration,motion_id),
              'variants':{}}
    saved = {'scan_mid_us':times,'lidar_index':np.array([row['lidar_index'] for row in rows]),'clip_mask':clip_mask}
    for batch_size in args.batch_sizes:
        for weight in args.weights:
            name = f'batch{batch_size}_weight{weight:g}'
            fitter.config = replace(FittingConfig(),continuity_weight=weight,
                                    root_motion_weight=args.root_motion_weight,
                                    reprojection_weight=args.candidate_reprojection_weight if weight>0 else .3,
                                    continuity_warm_start=not args.no_warm_start)
            previous = {'shape':original['shape'][0], 'shape_fixed':True}
            pieces = []
            actual_batches = []
            if args.device=='cuda':
                torch.cuda.synchronize()
            started = time.perf_counter()
            ranges = []
            first = 0
            while first<len(rows):
                size = args.bootstrap_frames if first==0 and args.bootstrap_frames>0 else batch_size
                last = min(first+size,len(rows))
                ranges.append((first,last))
                first = last
            for first,last in ranges:
                actual_batches.append(last-first)
                params = {key:value[first:last] for key,value in sampled['smpl_params_incam'].items()}
                result = fitter.fit_batch(params,clouds[first:last],calibration,keypoints=sampled['kp2d'][first:last],
                                          previous=previous,timestamps_us=times[first:last],iterations=80)
                if weight>0:
                    valid = [i for i,q in enumerate(result['qa']) if not acceptance_reasons(q,len(clouds[first+i]))]
                    if valid:
                        index = valid[-1]
                        previous.update({key:result[key][index] for key in ('root_pose','body_pose','transl')})
                        previous.update({key:result['continuity'][key][index] for key in
                                         ('scan_time_us','aligned_transl','teacher_root_pose','teacher_body_pose')})
                pieces.append(result)
            if args.device=='cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            combined = {key:torch.cat([result[key] for result in pieces]).numpy()
                        for key in ('root_pose','body_pose','shape','transl','vertices','joints','coco17')}
            combined['qa'] = [q for result in pieces for q in result['qa']]
            summary = summarize(combined,rows,clip_mask,cloud_centres,teacher_root,observations,calibration,motion_id)
            report['variants'][name] = dict(summary,seconds=elapsed,actual_batches=actual_batches,config=asdict(fitter.config))
            for key in ('root_pose','body_pose','shape','transl','vertices','joints','coco17'):
                saved[f'{name}_{key}'] = combined[key]
            Path(str(prefix)+'.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
            print(name,json.dumps({key:summary[key] for key in ('mean_surface_m','mean_of_frame_p90_surface_m','mean_cam2_fitting_reprojection_px','pelvis_vs_cloud_step_residual_m','root_motion_error_deg')},allow_nan=False),flush=True)
    np.savez_compressed(Path(str(prefix)+'.npz'),**saved)
    lines=['Fixed old beta; original calibration/source/teacher; 50 original clip10 scans scored.','',
           '| Variant | Surface mean / average frameP90 mm | Cam2 fit px | Pelvis-cloud step P90 / max cm | Root-motion P90 deg |','|---|---:|---:|---:|---:|']
    for name,value in [('original_stored_fit',report['original_stored_fit']),*report['variants'].items()]:
        step=value['pelvis_vs_cloud_step_residual_m']; angle=value['root_motion_error_deg']
        lines.append(f"| {name} | {value['mean_surface_m']*1000:.2f} / {value['mean_of_frame_p90_surface_m']*1000:.2f} | {value['mean_cam2_fitting_reprojection_px']:.2f} | {step['p90']*100:.2f} / {step['max']*100:.2f} | {angle['p90']:.2f} |")
        for camera,check in value['rgb_fixed_observation_check'].items():
            pixel=check['joint_pixel_error']
            lines.append(f"{name} {camera} independent fixed observations: {pixel}")
    lines.extend(['','No absolute velocity threshold or XYZ smoothing. RGB masks/association remain frozen from original fits; detector consistency is not ground truth and retains selection/calibration bias.'])
    Path(str(prefix)+'.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('written',str(prefix),flush=True)


if __name__=='__main__':
    main()
