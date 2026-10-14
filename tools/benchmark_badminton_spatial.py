"""Prepare/fit diverse recorded badminton contexts without modifying sources.

``prepare`` is CPU-only. ``fit`` is an explicitly separate stage so the parent
can sequence GPU fitting and frozen-mask inference. Contexts share the selected
old neutral-SMPL beta; close selections on the same track are merged.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_key(motion_id, lidar_index):
    return f'{motion_id}|{int(lidar_index)}'


def geometry_signature(calibration):
    # Frozen timestamps change snapshot digests without changing geometry.
    return {name:calibration.data[name] for name in ('cameras','Q_raw_to_viewer','reference_plane')}


def prepare(cases_path, output_dir, *, context_seconds=6., min_side_seconds=2.,
            only_cases=None, reprojection_weight=None):
    from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
    from hmr4d.lidar_mocap.fitting import FittingConfig
    from hmr4d.lidar_mocap.sessions import decode_fixed24
    from hmr4d.lidar_mocap.teacher import load_teacher_track,sample_track

    source = read_json(cases_path)
    config=FittingConfig()
    if reprojection_weight is not None:
        if not np.isfinite(reprojection_weight) or reprojection_weight<=0:
            raise ValueError('Reprojection weight must be positive and finite')
        config=replace(config,reprojection_weight=float(reprojection_weight))
    requested=set(only_cases or [])
    available={case.get('clip_id',case.get('id')) for case in source['cases']}
    if requested-available:
        raise ValueError(f'Unknown requested cases: {sorted(requested-available)}')
    output_dir = Path(output_dir).resolve()
    selected, excluded = [], []
    window_rows = {}
    raw_manifests = {}
    raw_clouds = {}
    references = defaultdict(list)
    for case in source['cases']:
        identity = case.get('clip_id',case.get('id'))
        if requested and identity not in requested:
            continue
        frame = case['frame']
        motion = case['selected_optimization_athlete_motion_id']
        window = Path(case['source_window']).resolve()
        if window not in window_rows:
            window_rows[window] = [json.loads(line) for line in (window/'quality.jsonl').read_text(encoding='utf-8').splitlines()]
        rows = sorted((row for row in window_rows[window] if row['motion_id']==motion), key=lambda row:row['scan_mid_us'])
        position = next((i for i,row in enumerate(rows) if row['lidar_index']==frame['lidar_index']),None)
        if position is None:
            excluded.append({'case_id':identity,'reason':'selected_original_fit_missing'})
            continue
        row = rows[position]
        if row['scan_mid_us']!=frame['scan_mid_us']:
            raise ValueError('Selection and original fit timestamp differ')
        with np.load(row['fit_path']) as original:
            beta = original['shape'].copy()
            cloud = original['source_cloud'].copy()
            indices = original['source_indices'].copy()
        if beta.shape!=(10,) or not np.isfinite(beta).all() or len(cloud)<80:
            excluded.append({'case_id':identity,'reason':'invalid_original_shape_or_insufficient_actual_returns'})
            continue
        # Check the selected point set directly against the original PNG, not
        # merely against another generated fit. No source recording is written.
        lidar_path = Path(row['lidar_path'])
        manifest_path = lidar_path.parent/'manifest.json'
        if manifest_path not in raw_manifests:
            raw_manifests[manifest_path]=read_json(manifest_path)
        manifest = raw_manifests[manifest_path]
        if lidar_path not in raw_clouds:
            Q=np.asarray(manifest['lidar_rotation'],dtype=np.float64).reshape(3,3)
            raw_clouds[lidar_path]=(decode_fixed24(lidar_path,manifest)@Q.T).astype(np.float32)
        decoded=raw_clouds[lidar_path]
        if len(indices)!=len(cloud) or np.min(indices)<0 or np.max(indices)>=len(decoded):
            raise ValueError('Selected source indices do not identify original valid returns')
        origin_error=float(np.max(np.abs(decoded[indices]-cloud)))
        if origin_error>1e-6:
            raise ValueError('Stored cloud differs from its original decoded LiDAR returns')
        # Restrict to one continuous observed/associated track segment.
        left_index,right_index=position,position
        max_gap_us=round(config.continuity_max_gap_s*1e6)
        while left_index>0 and rows[left_index]['scan_mid_us']-rows[left_index-1]['scan_mid_us']<=max_gap_us:
            left_index-=1
        while right_index+1<len(rows) and rows[right_index+1]['scan_mid_us']-rows[right_index]['scan_mid_us']<=max_gap_us:
            right_index+=1
        moment=row['scan_mid_us']
        left=max(moment-round(context_seconds*.5e6),rows[left_index]['scan_mid_us'])
        right=min(moment+round(context_seconds*.5e6),rows[right_index]['scan_mid_us'])
        before=(moment-left)/1e6;after=(right-moment)/1e6
        if min(before,after)+1e-6<min_side_seconds:
            excluded.append({'case_id':identity,'reason':'insufficient_continuous_track_context',
                             'before_s':before,'after_s':after,'motion_id':motion})
            continue
        entry={'case_id':identity,'source_window':str(window),'motion_id':motion,
               'lidar_index':row['lidar_index'],'scan_mid_us':moment,'original_fit_path':row['fit_path'],
               'original_fit_sha256':sha256(row['fit_path']),'raw_lidar_sha256':sha256(lidar_path),
               'point_origin_max_error_m':origin_error,'source_point_count':len(cloud),
               'selected_court_xy_m':case.get('selected_court_xy_m'),'grid_cell':case.get('grid_cell'),
               'before_s':before,'after_s':after,'left_us':left,'right_us':right,
               'fixed_beta':beta.tolist(),'teacher_path':str(window/'teacher'/row['teacher'])}
        beta_digest=hashlib.sha256(beta.astype(np.float32).tobytes()).hexdigest()
        references[(window,motion,beta_digest)].append((entry,rows))
        selected.append(entry)

    groups=[]
    for (window,motion,beta_digest),entries in references.items():
        ranges=[]
        for entry,rows in sorted(entries,key=lambda pair:pair[0]['left_us']):
            if ranges and entry['left_us']<=ranges[-1]['right_us']:
                ranges[-1]['right_us']=max(ranges[-1]['right_us'],entry['right_us'])
                ranges[-1]['cases'].append(entry)
            else:
                ranges.append({'left_us':entry['left_us'],'right_us':entry['right_us'],'cases':[entry]})
        for interval in ranges:
            rows=entries[0][1]
            context=[row for row in rows if interval['left_us']<=row['scan_mid_us']<=interval['right_us']]
            teacher_path=Path(context[0]['teacher'])
            if not teacher_path.is_absolute():
                teacher_path=window/'teacher'/teacher_path
            track=load_teacher_track(teacher_path)
            times=np.array([row['scan_mid_us'] for row in context],dtype=np.int64)
            sampled=sample_track(track,times)
            if not sampled['valid'].all() or len(context)<8:
                for entry in interval['cases']:
                    excluded.append({'case_id':entry['case_id'],'reason':'invalid_or_short_original_teacher_context'})
                continue
            snapshot=window.parents[3]/'calibration/calibration_snapshot.json'
            calibration=CalibrationSnapshot.load(snapshot)
            fields={}
            for row in context:
                with np.load(row['fit_path']) as original:
                    cloud=original['source_cloud']
                    if len(cloud)<80 or not np.isfinite(cloud).all():
                        raise ValueError('Invalid original cloud in the selected continuous context')
                fields[row['fit_path']]=sha256(row['fit_path'])
            group_id=f'context{len(groups)+1:03d}'
            groups.append({'group_id':group_id,'source_window':str(window),'motion_id':motion,
                           'left_us':interval['left_us'],'right_us':interval['right_us'],
                           'source_seconds':(int(times[-1])-int(times[0]))/1e6,'frame_count':len(context),
                           'case_ids':[entry['case_id'] for entry in interval['cases']],
                           'selected_lidar_indices':[entry['lidar_index'] for entry in interval['cases']],
                           'fixed_beta':interval['cases'][0]['fixed_beta'],'fixed_beta_sha256':beta_digest,
                           'teacher_path':str(teacher_path),'teacher_sha256':sha256(teacher_path),
                           'snapshot_path':str(snapshot),'calibration_digest':calibration.data['snapshot_digest'],
                           'geometry_signature':geometry_signature(calibration),
                           'original_fit_sha256':fields,'rows':context})
    ready_ids={case_id for group in groups for case_id in group['case_ids']}
    selected=[entry for entry in selected if entry['case_id'] in ready_ids]
    report={'schema':'badminton-spatial-fitting-plan-v1','stage':'prepared_cpu_only',
            'cases_path':str(Path(cases_path).resolve()),'cases_sha256':sha256(cases_path),
            'source_case_count':len(source['cases']),'ready_case_count':len(selected),
            'context_groups':groups,'selected_cases':selected,'excluded_cases':excluded,
            'context_frame_count':sum(group['frame_count'] for group in groups),
            'config':asdict(config),'bootstrap_frames':8,'later_batch_size':128,
            'requested_case_ids':sorted(requested),'configuration_override':{'reprojection_weight':config.reprojection_weight},
            'requested_context_seconds':context_seconds,'minimum_each_side_seconds':min_side_seconds,
            'point_origin':'Actual stored segmented returns; selected sets verified against original fixed24 PNG + manifest rotation',
            'beta_scope':'Original selected-frame neutral-SMPL beta fixed; contexts with different beta are not merged',
            'source_and_original_outputs_modified':False,'gpu_initialized':False}
    write_json(output_dir/'plan.json',report)
    return report


def motion_summary(pelvis,cloud_centres,root_pose,times_us,teacher_rotation):
    import torch
    from pytorch3d.transforms import axis_angle_to_matrix
    rotations=axis_angle_to_matrix(torch.as_tensor(root_pose).reshape(-1,3)).numpy()
    step=np.diff(pelvis,axis=0);cloud_step=np.diff(cloud_centres,axis=0)
    residual=np.linalg.norm(step-cloud_step,axis=-1)
    root_delta=rotations[1:]@rotations[:-1].transpose(0,2,1)
    angles=np.degrees(np.arccos(np.clip((np.trace(root_delta,axis1=-2,axis2=-1)-1)/2,-1,1)))
    teacher_delta=teacher_rotation[1:]@teacher_rotation[:-1].transpose(0,2,1)
    discrepancy=root_delta@teacher_delta.transpose(0,2,1)
    rotation_error=np.degrees(np.arccos(np.clip((np.trace(discrepancy,axis1=-2,axis2=-1)-1)/2,-1,1)))
    dt=np.diff(np.asarray(times_us,dtype=np.int64)).astype(np.float64)/1e6
    metric=lambda value:{'mean':float(value.mean()),'p90':float(np.quantile(value,.9)),'max':float(value.max())} if len(value) else None
    return {'pelvis_step_m':metric(np.linalg.norm(step,axis=-1)),
            'scan_dt_seconds':metric(dt),'pelvis_minus_cloud_step_m':metric(residual),
            'pelvis_minus_cloud_motion_per_100ms_m':metric(residual*.1/dt),
            'root_step_deg':metric(angles),'root_relative_teacher_step_error_deg':metric(rotation_error),
            'reference_limit':'Cloud median is an observed proxy, not ground-truth pelvis; no absolute speed gate'}


def finalize_report(output_dir, report):
    selected=report['selected_cases']
    def averages(name):
        result=[]
        for prefix in ('old','new'):
            values=[entry[f'{prefix}_qa'].get(name) for entry in selected]
            values=[value for value in values if value is not None and np.isfinite(value)]
            result.append(float(np.mean(values)) if values else None)
        return result
    report['selected_frame_comparison']={
        'count':len(selected),
        'old_accepted':sum(entry['old_accepted'] for entry in selected),
        'new_accepted':sum(entry['new_accepted'] for entry in selected),
        'surface_mean_old_new_m':averages('final_p2surface_mean_m'),
        'average_frame_p90_old_new_m':averages('final_p2surface_p90_m'),
        'cam2_fitting_reprojection_mean_old_new_px':averages('reprojection_mean_px'),
        'baseline_metrics':'Original stored full-triangle QA; candidate full-triangle QA from this run',
        'p90_definition':'Arithmetic average of per-frame visible-return P90 values, not pooled point P90',
        'cam2_limit':'Cam2 is an optimization constraint; independent frozen RGB masks are evaluated separately',
    }
    review=[]
    for entry in selected:
        old,new=entry['old_qa'],entry['new_qa']
        delta={'surface_mean_mm':(new['final_p2surface_mean_m']-old['final_p2surface_mean_m'])*1000,
               'frame_p90_mm':(new['final_p2surface_p90_m']-old['final_p2surface_p90_m'])*1000,
               'cam2_fitting_px':new['reprojection_mean_px']-old['reprojection_mean_px']}
        entry['paired_metric_changes']=delta
        failed=[name for name,limit in (('surface_mean_mm',1.),('frame_p90_mm',2.),('cam2_fitting_px',1.)) if delta[name]>limit]
        if failed:
            review.append({'case_ids':entry['case_ids'],'motion_id':entry['motion_id'],
                           'lidar_index':entry['lidar_index'],'changes':delta,'review_metrics':failed})
    report['paired_regression_review']={
        'scope':'Diagnostic paired comparison guards proposed in body_joint_evaluation, separate from training acceptance; independent RGB/masks must assess these trades',
        'limits_increases':{'surface_mean_mm':1.,'frame_p90_mm':2.,'cam2_fitting_px':1.},
        'cases_exceeding_guards':review,
    }
    contexts=report['contexts']
    report['context_motion_comparison']={
        'count':len(contexts),
        'p90_residual_improved_count':sum(context['new_motion']['pelvis_minus_cloud_step_m']['p90']<context['old_motion']['pelvis_minus_cloud_step_m']['p90'] for context in contexts),
        'average_context_p90_pelvis_cloud_step_residual_old_new_m':[
            float(np.mean([context[f'{prefix}_motion']['pelvis_minus_cloud_step_m']['p90'] for context in contexts]))
            for prefix in ('old','new')],
        'limit':'Equally weighted mean of context P90 values; cloud motion is not pelvis ground truth',
    }
    write_json(output_dir/'report.json',report)
    summary=report['selected_frame_comparison']
    lines=['Spatial fitting comparison: fixed existing beta, unchanged camera calibration and source returns.','',
           f"{len(selected)} selected people/frames in {len(report['contexts'])} continuous contexts; old/new accepted {summary['old_accepted']}/{summary['new_accepted']}.",'',
           '| Metric | Original | Candidate |','|---|---:|---:|']
    for label,key,factor in (
        ('Visible point-to-triangle mean (mm)','surface_mean_old_new_m',1000),
        ('Average frame P90 (mm)','average_frame_p90_old_new_m',1000),
        ('Cam2 fitting reprojection mean (px)','cam2_fitting_reprojection_mean_old_new_px',1),
    ):
        values=summary[key]
        formatted=[f'{value*factor:.3f}' if value is not None else 'unavailable' for value in values]
        lines.append(f'| {label} | {formatted[0]} | {formatted[1]} |')
    lines.extend(['','| Context | Frames | Time (s) | Pelvis-cloud motion residual P90 old/new (cm) |',
                  '|---|---:|---:|---:|'])
    for context in report['contexts']:
        old=context['old_motion']['pelvis_minus_cloud_step_m']['p90']*100
        new=context['new_motion']['pelvis_minus_cloud_step_m']['p90']*100
        lines.append(f"| {context['group_id']} | {context['frame_count']} | {context['seconds']:.3f} | {old:.3f}/{new:.3f} |")
    lines.extend(['','Paired regression checks below are diagnostic review guards, separate from training acceptance.','',
                  '| Selected case | Surface mean change (mm) | Frame P90 change (mm) | Cam2 change (px) |',
                  '|---|---:|---:|---:|'])
    for case in review:
        changes=case['changes']
        lines.append(f"| {', '.join(case['case_ids'])} | {changes['surface_mean_mm']:+.3f} | {changes['frame_p90_mm']:+.3f} | {changes['cam2_fitting_px']:+.3f} |")
    lines.extend(['','Cloud median motion is an observed proxy, not pelvis ground truth. No speed gate, source XYZ smoothing or camera refinement. Each context uses real timestamps and bootstrap8 -> batch128 warm state. Surface metrics measure observed points only; Cam2 detector residual is not independent pose accuracy. Other people in the RGB frames retain their original fits. Frozen independent mask/RGB comparison is a separate evaluation.'])
    (output_dir/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def verify_output_artifacts(report):
    """Check saved interchange arrays and their actual unmodified point sets."""
    expected={'root_pose':(1,3),'body_pose':(23,3),'shape':(10,),
              'transl':(3,),'vertices':(6890,3),'joints':(24,3),'coco17':(17,3)}
    maximum_pelvis_error=0.
    for entry in report['selected_cases']:
        with np.load(entry['old_fit_path']) as old,np.load(entry['corrected_fit_path']) as new:
            for name in ('pc','source_cloud','source_indices','shape'):
                if old[name].dtype!=new[name].dtype or not np.array_equal(old[name],new[name]):
                    raise ValueError(f'Original {name} changed in {entry["corrected_fit_path"]}')
            for name,shape in expected.items():
                if new[name].shape!=shape or new[name].dtype!=np.float32 or not np.isfinite(new[name]).all():
                    raise ValueError(f'Invalid saved neutral SMPL {name}')
            error=float(np.max(np.abs(new['joints'][0]-new['transl'])))
            maximum_pelvis_error=max(maximum_pelvis_error,error)
            if error>2e-6 or int(new['scan_mid_us'])!=entry['scan_mid_us']:
                raise ValueError('Saved pelvis convention or source scan timestamp changed')
    return {'checked_selection_count':len(report['selected_cases']),
            'source_arrays_and_fixed_shape_exact':True,'neutral_smpl_arrays_finite_float32':True,
            'timestamp_preserved':True,'maximum_pelvis_joint0_error_m':maximum_pelvis_error}


def fit(plan_path,output_dir,device='cuda'):
    import torch
    from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
    from hmr4d.lidar_mocap.export import acceptance_reasons
    from hmr4d.lidar_mocap.fitting import FittingConfig,SMPLFitter
    from hmr4d.lidar_mocap.teacher import load_teacher_track,sample_track
    from pytorch3d.transforms import axis_angle_to_matrix

    torch.set_num_threads(4)
    torch.manual_seed(2026)
    plan=read_json(plan_path);output_dir=Path(output_dir).resolve()
    config=replace(FittingConfig(),reprojection_weight=plan['config']['reprojection_weight'])
    if plan['cases_sha256']!=sha256(plan['cases_path']) or plan['config']!=asdict(config):
        raise ValueError('Selection or fitting configuration changed after preparation')
    result_report={'schema':'badminton-spatial-fitting-result-v1','status':'running',
                   'plan':str(Path(plan_path).resolve()),'plan_sha256':sha256(plan_path),
                   'candidate_config':plan['config'],'fixed_original_beta':True,
                   'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                   'corrected_fit_paths':{},'corrected_fit_paths_by_motion':{},
                   'selected_cases':[],'contexts':[],'excluded_cases':plan['excluded_cases']}
    write_json(output_dir/'report.json',result_report)
    fitter=SMPLFitter(device=device,config=config)
    for group in plan['context_groups']:
        rows=group['rows'];calibration=CalibrationSnapshot.load(group['snapshot_path'])
        if geometry_signature(calibration)!=group['geometry_signature']:
            raise ValueError('Physical calibration changed after spatial plan')
        if sha256(group['teacher_path'])!=group['teacher_sha256']:
            raise ValueError('Teacher changed after spatial preparation')
        clouds=[];originals=[]
        for row in rows:
            if sha256(row['fit_path'])!=group['original_fit_sha256'][row['fit_path']]:
                raise ValueError('Original fitting input changed after preparation')
            with np.load(row['fit_path']) as stored:
                originals.append({key:stored[key].copy() for key in stored.files})
                clouds.append(stored['source_cloud'].copy())
        times=np.array([row['scan_mid_us'] for row in rows],dtype=np.int64)
        sampled=sample_track(load_teacher_track(group['teacher_path']),times)
        if not sampled['valid'].all():
            raise ValueError('Original teacher no longer validates')
        teacher_rotation=axis_angle_to_matrix(torch.from_numpy(sampled['smpl_params_incam']['global_orient'])).numpy()
        teacher_rotation=calibration.camera_rotation_to_viewer('cam2')[None]@teacher_rotation
        previous={'shape':np.array(group['fixed_beta'],dtype=np.float32),'shape_fixed':True}
        pieces=[];ranges=[];first=0;started=time.perf_counter()
        while first<len(rows):
            size=128 if previous.get('scan_time_us') is not None else 8
            last=min(first+size,len(rows));ranges.append([first,last])
            current=fitter.fit_batch({key:value[first:last] for key,value in sampled['smpl_params_incam'].items()},
                                     clouds[first:last],calibration,keypoints=sampled['kp2d'][first:last],
                                     previous=previous,timestamps_us=times[first:last],iterations=80)
            valid=[j for j,qa in enumerate(current['qa']) if not acceptance_reasons(qa,len(clouds[first+j]))]
            if valid:
                j=valid[-1]
                previous.update({key:current[key][j] for key in ('root_pose','body_pose','transl')})
                previous.update({key:current['continuity'][key][j] for key in
                                 ('scan_time_us','aligned_transl','teacher_root_pose','teacher_body_pose')})
            pieces.append(current);first=last
        combined={key:torch.cat([piece[key] for piece in pieces]).numpy()
                  for key in ('root_pose','body_pose','shape','transl','vertices','joints','coco17')}
        qa=[item for piece in pieces for item in piece['qa']]
        centre=np.stack([np.median(cloud,axis=0) for cloud in clouds])
        context_result={'group_id':group['group_id'],'motion_id':group['motion_id'],'case_ids':group['case_ids'],
                        'source_window':group['source_window'],
                        'frame_count':len(rows),'actual_batch_ranges':ranges,'seconds':time.perf_counter()-started,
                        'old_motion':motion_summary(np.stack([old['transl'] for old in originals]),centre,
                                                    np.stack([old['root_pose'] for old in originals]),times,teacher_rotation),
                        'new_motion':motion_summary(combined['transl'],centre,combined['root_pose'],times,teacher_rotation),
                        'surface_mean_old_new_m':[float(np.mean([row['qa']['final_p2surface_mean_m'] for row in rows])),float(np.mean([q['final_p2surface_mean_m'] for q in qa]))],
                        'mean_frame_p90_old_new_m':[float(np.mean([row['qa']['final_p2surface_p90_m'] for row in rows])),float(np.mean([q['final_p2surface_p90_m'] for q in qa]))]}
        result_report['contexts'].append(context_result)
        selected_set=set(group['selected_lidar_indices'])
        for i,row in enumerate(rows):
            if row['lidar_index'] not in selected_set:
                continue
            output_path=output_dir/'fits'/group['group_id']/f"{row['lidar_index']:06d}.npz"
            output_path.parent.mkdir(parents=True,exist_ok=True)
            payload={key:combined[key][i] for key in combined}
            payload.update({key:originals[i][key] for key in ('pc','source_cloud','source_indices')})
            payload['scan_mid_us']=np.array(row['scan_mid_us'],dtype=np.int64)
            np.savez_compressed(output_path,**payload)
            key=fit_key(row['motion_id'],row['lidar_index'])
            result_report['corrected_fit_paths'][key]=str(output_path)
            result_report['corrected_fit_paths_by_motion'].setdefault(row['motion_id'],{})[str(row['lidar_index'])]=str(output_path)
            entry={'motion_id':row['motion_id'],'lidar_index':row['lidar_index'],'scan_mid_us':row['scan_mid_us'],
                   'case_ids':[case['case_id'] for case in plan['selected_cases'] if case['motion_id']==row['motion_id'] and case['lidar_index']==row['lidar_index']],
                   'old_fit_path':row['fit_path'],'corrected_fit_path':str(output_path),'corrected_fit_sha256':sha256(output_path),
                   'old_qa':row['qa'],'new_qa':qa[i],'new_accepted':not acceptance_reasons(qa[i],len(clouds[i])),
                   'old_accepted':row['accepted'],'old_rejection_reasons':row.get('rejection_reasons',[]),
                   'new_rejection_reasons':acceptance_reasons(qa[i],len(clouds[i])),'source_point_count':len(clouds[i]),
                   'calibration_digest':calibration.data['snapshot_digest'],
                   'coordinate_space':'viewer_m','translation_convention':'shaped_pelvis_position',
                   'shape_fixed':True,'source_arrays_unchanged':True}
            write_json(output_path.with_suffix('.qa.json'),entry)
            result_report['selected_cases'].append(entry)
        write_json(output_dir/'report.json',result_report)
        print(f"{group['group_id']}: {len(rows)} original scans, {len(selected_set)} selections, {context_result['seconds']:.2f}s",flush=True)
    result_report['status']='complete'
    result_report['corrected_selection_count']=len(result_report['corrected_fit_paths'])
    result_report['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    result_report['scope']='Fixed original beta; actual selected-frame original returns/indices preserved. Context motion and surface metrics are proxies, not pelvis/pose GT. Camera calibration unchanged.'
    result_report['unselected_people']='Other athletes in the selected RGB frames retain their original fits; only selected optimization people have corrected_fit_paths overrides'
    result_report['saved_artifact_verification']=verify_output_artifacts(result_report)
    finalize_report(output_dir,result_report)
    return result_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','fit'))
    parser.add_argument('--cases',type=Path,default=Path('outputs/badminton_corner_fix_v1/calibration_refinement/spatial_cases.json'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/badminton_corner_fix_v1/calibration_refinement/spatial_fitting'))
    parser.add_argument('--context-seconds',type=float,default=6.)
    parser.add_argument('--min-side-seconds',type=float,default=2.)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--only-case',action='append',default=[],help='Repeat to fit only specified selection IDs')
    parser.add_argument('--reprojection-weight',type=float,help='Isolated RGB-loss override stored in the prepared plan')
    args=parser.parse_args()
    if args.stage=='prepare':
        value=prepare(args.cases,args.output_dir,context_seconds=args.context_seconds,min_side_seconds=args.min_side_seconds,
                      only_cases=args.only_case,reprojection_weight=args.reprojection_weight)
        print(json.dumps({'plan':str(args.output_dir/'plan.json'),'ready_cases':value['ready_case_count'],
                          'contexts':len(value['context_groups']),'frames':value['context_frame_count'],
                          'excluded':value['excluded_cases'],'gpu_initialized':False},ensure_ascii=False))
    else:
        prepared=read_json(args.output_dir/'plan.json')
        if args.only_case and sorted(set(args.only_case))!=prepared.get('requested_case_ids',[]):
            parser.error('fit uses the prepared case selection; prepare a separate plan to change it')
        if args.reprojection_weight is not None and args.reprojection_weight!=prepared['config']['reprojection_weight']:
            parser.error('fit uses the prepared RGB weight; prepare a separate plan to change it')
        value=fit(args.output_dir/'plan.json',args.output_dir,args.device)
        print(json.dumps({'report':str(args.output_dir/'report.json'),'corrected_fit_paths':value['corrected_fit_paths']},ensure_ascii=False))


if __name__=='__main__':
    main()
