"""Render the same recorded clip before/after an isolated fitting correction."""
import argparse
import copy
import json
from pathlib import Path
import cv2
import numpy as np
import smplx
import torch
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.export import acceptance_reasons
from hmr4d.lidar_mocap.fitting import point_to_surface_squared
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.sequence_review import export_clip,review_selection_digest,write_page
from hmr4d.lidar_mocap.teacher import load_teacher_track,sample_track
from hmr4d.utils.body_model.contracts import forward_smpl_at_pelvis


def compare_all_tracks(args, root, out, benchmark, saved, before, calibration):
    """Keep the controlled fixed-shape experiment separate from production."""
    workspace = Path(__file__).resolve().parents[1]
    original = copy.deepcopy(next(c for c in before['clips'] if c['id']=='clip10'))
    candidate = copy.deepcopy(original)
    names = sorted(original['motion_colors'])
    models = smplx.create(workspace/'inputs/checkpoints/body_models', model_type='smpl', gender='neutral')
    faces = np.asarray(models.faces, dtype=np.uint16)
    regressor = torch.load(workspace/'hmr4d/utils/body_model/smpl_coco17_J_regressor.pt', map_location='cpu').float()
    variants = {benchmark['motion_id']: (args.variant, benchmark['variants'][args.variant], '')}
    for entry in benchmark['cross_track_regression']:
        track = entry['track_id']
        variants[entry['motion_id']] = (entry['candidate_array_prefix'], entry['candidate'], f'track{track}_')
    if set(variants) != set(names):
        raise ValueError('The benchmark must cover every displayed athlete')
    out.mkdir(parents=True, exist_ok=True)
    fit_dir = out/'corrected_fits'; fit_dir.mkdir(exist_ok=True)
    checked = []
    for frame in candidate['frames']:
        for athlete in frame['athletes']:
            prefix, report, time_prefix = variants[athlete['motion_id']]
            scan = frame['lidar_index']
            indices = np.flatnonzero(saved[time_prefix+'lidar_index'] == scan)
            if len(indices) != 1 or int(saved[time_prefix+'scan_mid_us'][indices[0]]) != frame['scan_mid_us']:
                raise ValueError('Benchmark geometry does not match the recorded scan clock')
            i = indices[0]
            with np.load(athlete['fit_path']) as source:
                values = {key:source[key].copy() for key in source.files}
            for key in ('root_pose','body_pose','shape','transl','vertices','joints'):
                values[key] = saved[f'{prefix}_{key}'][i].copy()
            # The first displayed frame has no velocity row; measure its new
            # geometry instead of borrowing the old geometry's QA numbers.
            by_scan = {row['lidar_index']:row['qa'] for row in report['per_scan_motion']}
            qa = copy.deepcopy(by_scan.get(scan, athlete['qa']))
            if scan not in by_scan:
                verts = torch.from_numpy(values['vertices'])[None]
                distance = point_to_surface_squared(verts, torch.as_tensor(faces.astype(np.int64)),
                                                    [torch.from_numpy(values['source_cloud'])])[0].clamp_min(0).sqrt()
                qa['final_p2surface_mean_m'] = float(distance.mean())
                qa['final_p2surface_p90_m'] = float(torch.quantile(distance, .9))
                qa['surface_mean_m']=qa['final_p2surface_mean_m']
                qa['surface_p90_m']=qa['final_p2surface_p90_m']
                quality = [json.loads(s) for s in (Path(original['source_window'])/'quality.jsonl').read_text('utf-8').splitlines()]
                row = next(r for r in quality if r['motion_id']==athlete['motion_id'] and r['lidar_index']==scan)
                teacher = load_teacher_track(Path(original['source_window'])/'teacher'/row['teacher'])
                observed = sample_track(teacher, [frame['scan_mid_us']])['kp2d'][0]
                uv, depth = calibration.project((regressor @ verts[0]).numpy(), 'cam2', distort=False)
                valid = (observed[:,2]>=.25)&(depth>0)
                qa['reprojection_mean_px'] = float(np.linalg.norm(uv[valid]-observed[valid,:2], axis=1).mean()) if valid.any() else None
                qa['reprojection_by_camera_px']={'cam2':qa['reprojection_mean_px']}
                qa['reprojection_valid_joints']={'cam2':int(valid.sum())}
                plane = calibration.reference_plane
                height = (values['vertices']-np.asarray(plane['origin_viewer_m'])) @ np.asarray(plane['normal_viewer'])
                qa['floor_penetration_m'] = float(np.maximum(-height,0).max())
                qa['finite'] = bool(np.isfinite(values['vertices']).all())
            reasons = acceptance_reasons(qa, len(values['source_cloud']))
            track = athlete['motion_id'].split(':')[-2]
            path = fit_dir/f'{scan:06d}_track{track}.npz'
            np.savez_compressed(path, **values)
            checked.append((athlete['motion_id'], scan))
            athlete.update(fit_path=str(path), qa=qa, accepted=not reasons, rejection_reasons=reasons)
    clips = [original, candidate]
    labels = ['原拟合', '连续性修正 · 四人 · 固定原体型']
    if args.production_root:
        production_root = args.production_root.resolve()
        production_calibration = json.loads((production_root/'calibration/calibration_snapshot.json').read_text('utf-8'))
        for key, value in calibration.data.items():
            if key not in {'frozen_utc','snapshot_digest'} and production_calibration.get(key) != value:
                raise ValueError(f'Production calibration differs in {key}')
        window = production_root/'sessions'/original['session_id']/'windows'/Path(original['source_window']).name
        rows = [json.loads(s) for s in (window/'quality.jsonl').read_text('utf-8').splitlines()]
        by_pair = {(r['motion_id'],r['lidar_index']):r for r in rows}
        production = copy.deepcopy(original)
        production['source_window'] = str(window)
        for frame in production['frames']:
            for athlete in frame['athletes']:
                row = by_pair[(athlete['motion_id'],frame['lidar_index'])]
                if any(row[k] != frame[k] for k in ('scan_mid_us','lidar_path','cameras')):
                    raise ValueError('Production source pairing changed')
                athlete.update({k:row[k] for k in ('fit_path','qa','accepted','rejection_reasons')})
        clips.append(production); labels.append('当前训练流水线 · 四人 · 新拟合体型')
    errors = []
    for number, (clip,label) in enumerate(zip(clips,labels), 1):
        clip.update(id=f'clip{number:02d}', label=label)
        rows = [a for f in clip['frames'] for a in f['athletes']]
        clip['summary'].update(accepted=sum(a['accepted'] for a in rows),
            mean_surface_mm=float(np.mean([a['qa']['final_p2surface_mean_m'] for a in rows])*1000),
            mean_frame_p90_mm=float(np.mean([a['qa']['final_p2surface_p90_m'] for a in rows])*1000))
        arrays = []
        for old_frame, frame in zip(original['frames'], clip['frames']):
            original_by_id = {a['motion_id']:a for a in old_frame['athletes']}
            for athlete in frame['athletes']:
                with np.load(athlete['fit_path']) as fit, np.load(original_by_id[athlete['motion_id']]['fit_path']) as source:
                    for key in ('source_cloud','source_indices','pc'):
                        np.testing.assert_array_equal(fit[key], source[key])
                    if number==2: np.testing.assert_array_equal(fit['shape'], source['shape'])
                    arrays.append({k:fit[k].copy() for k in ('root_pose','body_pose','shape','transl','vertices','joints')})
        with torch.no_grad():
            for start in range(0,len(arrays),32):
                batch=arrays[start:start+32]
                tensor=lambda key:torch.from_numpy(np.stack([v[key] for v in batch]))
                vertices,joints=forward_smpl_at_pelvis(models,tensor('root_pose'),tensor('body_pose'),tensor('shape'),tensor('transl'))
                error=max(float((vertices-tensor('vertices')).abs().max()),float((joints-tensor('joints')).abs().max()))
                if error>3e-6: raise ValueError('Stored geometry is inconsistent with native SMPL parameters')
                errors.append(error)
    manifest = {k:v for k,v in before.items() if k not in {'clips','quality_report','cam1_check'}}
    manifest.update(state='complete',clip_count=len(clips),clips=clips,rendered_clips=[c['id'] for c in clips],
        review_title='20261012_163452 · 四人骨盆连续性对照',
        review_description='同一段 5 秒、同一实测点云与双机帧：四人全部替换。固定原体型实验与当前训练流水线分别展示。',
        all_tracks_comparison={'affected_motion_ids':names,'corrected_person_frames':len(checked),
            'source_cloud_and_indices_unchanged':True,'fixed_shape_control':True,
            'production_refits_shape':bool(args.production_root),'geometry_interpolated':False,
            'calibration_geometry_changed':False,'previous_review_only_changed_track':benchmark['motion_id']})
    manifest['selection_sha256']=review_selection_digest(manifest)
    write_json(out/'manifest.json',manifest)
    for clip in clips: export_clip(clip,calibration,out,faces,1280)
    write_page(out,manifest,faces)
    (out/'index.html').replace(out/'viewer.html')
    assets=workspace/'hmr4d/lidar_mocap/review_assets'
    (out/'index.html').write_text((assets/'synchronized.html').read_text('utf-8').replace('__MANIFEST__',
        json.dumps({k:v for k,v in manifest.items() if k!='clips'} | {'clips':[{k:v for k,v in c.items() if k!='frames'} for c in clips]},ensure_ascii=False)),encoding='utf-8')
    (out/'synchronized.js').write_bytes((assets/'synchronized.js').read_bytes())
    diagnosis_path=root/'calibration_refinement/all_tracks_diagnosis/diagnosis.json'
    if diagnosis_path.exists():
        diagnosis=json.loads(diagnosis_path.read_text('utf-8'))
        write_json(out/'diagnosis.json',diagnosis)
        (out/'diagnosis.js').write_text('window.SYNC_DIAGNOSIS='+json.dumps(diagnosis,ensure_ascii=False,allow_nan=False)+';\n',encoding='utf-8')
    write_json(out/'data_verification.json',{'passed':True,'corrected_person_frames':len(checked),
        'all_motion_ids':names,'original_cloud_indices_pc_exact':True,'fixed_shape_control_exact':True,
        'native_smpl_max_error_m':max(errors),'production_refits_shape':bool(args.production_root),
        'calibration_geometry_changed':False,'geometry_interpolated':False})
    print(json.dumps({'review':str(out/'index.html'),'corrected_person_frames':len(checked),'clips':len(clips)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=Path('outputs/badminton_corner_fix_v1'))
    parser.add_argument('--benchmark-prefix',type=Path,required=True)
    parser.add_argument('--variant',required=True)
    parser.add_argument('--review-dir',type=Path)
    parser.add_argument('--all-tracks',action='store_true')
    parser.add_argument('--production-root',type=Path)
    args=parser.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1)
    root=args.output_root.resolve();out=(args.review_dir or root/'calibration_refinement/comparison').resolve()
    workspace=Path(__file__).resolve().parents[1]
    if not out.is_relative_to(workspace):raise ValueError('Comparison outputs must stay in the GVHMR workspace')
    benchmark=json.loads(Path(str(args.benchmark_prefix)+'.json').read_text(encoding='utf-8'))
    saved=np.load(Path(str(args.benchmark_prefix)+'.npz'))
    before=json.loads((root/'sequence_review/manifest.json').read_text(encoding='utf-8'))
    clip=next(c for c in before['clips'] if c['id']=='clip10');new=copy.deepcopy(clip)
    calibration=CalibrationSnapshot.load(root/'calibration/calibration_snapshot.json')
    assert benchmark['calibration_digest']==calibration.data['snapshot_digest']==before['calibration_digest']
    if args.all_tracks:
        compare_all_tracks(args,root,out,benchmark,saved,before,calibration)
        return
    variant=benchmark['variants'][args.variant]; motion_id=benchmark['motion_id']
    index={int(scan):i for i,scan in enumerate(saved['lidar_index'])}
    qa_by_scan={r['lidar_index']:r['qa'] for r in variant['per_scan_motion']}
    model=smplx.create(workspace/'inputs/checkpoints/body_models',model_type='smpl',gender='neutral')
    faces=np.asarray(model.faces,dtype=np.uint16);del model
    regressor=torch.load(workspace/'hmr4d/utils/body_model/smpl_coco17_J_regressor.pt',map_location='cpu').float()
    quality=[json.loads(line) for line in (Path(clip['source_window'])/'quality.jsonl').read_text(encoding='utf-8').splitlines()]
    row_by_scan={r['lidar_index']:r for r in quality if r['motion_id']==motion_id}
    teacher=load_teacher_track(Path(clip['source_window'])/'teacher'/row_by_scan[clip['frames'][0]['lidar_index']]['teacher'])
    sampled=sample_track(teacher,[f['scan_mid_us'] for f in clip['frames']])
    out.mkdir(parents=True,exist_ok=True);fit_dir=out/'corrected_fits';fit_dir.mkdir(exist_ok=True)
    checked=[]
    for fi,frame in enumerate(new['frames']):
        for athlete in frame['athletes']:
            if athlete['motion_id']!=motion_id:continue
            scan=frame['lidar_index'];i=index[scan]
            with np.load(athlete['fit_path']) as fit:values={key:fit[key].copy() for key in fit.files}
            original_points=values['source_cloud'].copy();original_indices=values['source_indices'].copy()
            for key in ('root_pose','body_pose','shape','transl','vertices','joints'):
                values[key]=saved[f'{args.variant}_{key}'][i].copy()
            if scan in qa_by_scan:
                qa=copy.deepcopy(qa_by_scan[scan])
            else:
                # The motion table excludes the first frame's velocity. Compute
                # its actual new surface/keypoint residual instead of copying
                # the former geometry's metrics onto a corrected body.
                qa=copy.deepcopy(athlete['qa']);verts=torch.from_numpy(values['vertices'])[None]
                distance=point_to_surface_squared(verts,torch.as_tensor(faces.astype(np.int64)),[torch.from_numpy(original_points)])[0].clamp_min(0).sqrt()
                qa['final_p2surface_mean_m']=float(distance.mean());qa['final_p2surface_p90_m']=float(torch.quantile(distance,.9))
                coco=(regressor @ verts[0]).numpy();uv,depth=calibration.project(coco,'cam2',distort=False)
                kp=sampled['kp2d'][fi];valid=(kp[:,2]>=.25)&(depth>0)
                qa['reprojection_mean_px']=float(np.linalg.norm(uv[valid]-kp[valid,:2],axis=1).mean()) if valid.any() else None
                qa['reprojection_by_camera_px']={'cam2':qa['reprojection_mean_px']};qa['reprojection_valid_joints']={'cam2':int(valid.sum())}
                plane=calibration.reference_plane;height=(values['vertices']-np.asarray(plane['origin_viewer_m'])) @ np.asarray(plane['normal_viewer'])
                qa['floor_penetration_m']=float(np.maximum(-height,0).max());qa['finite']=bool(np.isfinite(values['vertices']).all())
                qa['timed_continuity']=True
            reasons=acceptance_reasons(qa,len(original_points));path=fit_dir/f'{scan:06d}_corrected.npz'
            np.savez_compressed(path,**values)
            np.testing.assert_array_equal(values['source_cloud'],original_points);np.testing.assert_array_equal(values['source_indices'],original_indices)
            athlete.update(fit_path=str(path),qa=qa,accepted=not reasons,rejection_reasons=reasons)
            checked.append(scan)
    rows=[a for f in new['frames'] for a in f['athletes']]
    new['summary'].update(accepted=sum(a['accepted'] for a in rows),
                          mean_surface_mm=float(np.mean([a['qa']['final_p2surface_mean_m'] for a in rows])*1000),
                          mean_frame_p90_mm=float(np.mean([a['qa']['final_p2surface_p90_m'] for a in rows])*1000))
    clip=copy.deepcopy(clip);clip.update(id='clip01',label='修改前 · 原拟合')
    new.update(id='clip02',label='连续性修正 · 同一份点云')
    manifest={k:v for k,v in before.items() if k!='clips'}
    # Full-run statistics belong to the original ten-clip review. This page
    # measures one isolated case and must not present stale results/links.
    manifest.pop('quality_report',None);manifest.pop('cam1_check',None)
    manifest.update(state='complete',clip_count=2,clips=[clip,new],rendered_clips=['clip01','clip02'],
                    review_title='最后片段 · 拟合连续性对照',
                    review_description='原始点云、摄像机帧与标定完全相同。仅更改问题轨迹的 SMPL 拟合，未平滑实测点。',
                    refinement_comparison={'variant':args.variant,'affected_motion_id':motion_id,
                       'before':benchmark['original_stored_fit'],'after':variant,
                       'source_cloud_and_indices_unchanged':True,'corrected_scan_indices':checked,
                       'calibration_changed':False,'geometry_interpolated':False})
    manifest['selection_sha256']=review_selection_digest(manifest);write_json(out/'manifest.json',manifest)
    for item in manifest['clips']:export_clip(item,calibration,out,faces,1280)
    write_page(out,manifest,faces)
    write_json(out/'data_verification.json',{'passed':True,'corrected_fits':len(checked),
                                           'original_cloud_and_indices_exact':True,'calibration_changed':False,
                                           'benchmark_variant':args.variant})
    print(json.dumps({'review':str(out/'index.html'),'corrected_fits':len(checked),'variant':args.variant},indent=2))


if __name__=='__main__':main()
