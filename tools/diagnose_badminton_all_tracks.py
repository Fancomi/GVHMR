"""CPU audit of all four native-scan pelvis trajectories and displayed versions.

Actual cloud centres are visibility-dependent motion proxies, not ground-truth
pelvis positions. Acceleration is numerical differentiation of estimates, not
a physical jump classifier. No source data, fits, or comparison page is edited.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


SID = '20261012_163452'
WINDOW = '000000000000_000006000000'
VARIANT = 'batch128_weight0.2'
COLORS = {1: ('orange/yellow-like', [255,157,48]), 2: ('blue', [79,163,255]),
          3: ('green', [128,224,134]), 4: ('purple', [216,132,226])}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(array):
    array = np.asarray(array, float)
    return dict(mean=float(array.mean()), median=float(np.median(array)),
                p90=float(np.quantile(array,.9)), max=float(array.max()), min=float(array.min()))


def read_rows(window):
    with (window/'quality.jsonl').open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def production_window_path(root, value):
    value=Path(value)
    if value.is_absolute() or value.is_dir():
        return value.resolve()
    return (Path(root)/'sessions'/SID/'windows'/value).resolve()


def box_iou(left,right):
    low=np.maximum(left[:2],right[:2]);high=np.minimum(left[2:],right[2:])
    intersection=float(np.maximum(high-low,0).prod())
    return intersection/max(float(np.maximum(left[2:]-left[:2],0).prod()+
                                  np.maximum(right[2:]-right[:2],0).prod()-intersection),1e-9)


def observed_bbox(window,row,camera,cache):
    """Use the recorded teacher observation, normalized to native Brown pixels."""
    path=Path(row['teacher'])
    if not path.is_absolute():path=window/'teacher'/path
    if path not in cache:
        import torch
        cache[path]=torch.load(path,map_location='cpu',weights_only=False)
    teacher=cache[path]
    records=np.asarray(teacher['record_index'])
    requested=row['teacher_sampling']['record_index']
    positions=np.flatnonzero(records==requested)
    if len(positions)!=1:raise ValueError('Teacher bbox lacks its exact native record identity')
    at=int(positions[0]);box=np.asarray(teacher['bbx_xyxy'][at],float)
    if box.shape!=(4,) or not np.isfinite(box).all():raise ValueError('Invalid observed teacher bbox')
    points=np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]])
    K=np.asarray(camera['K']);rays=np.column_stack([points,np.ones(4)])@np.linalg.inv(K).T
    x,y=rays[:,0]/rays[:,2],rays[:,1]/rays[:,2];k1,k2,p1,p2,k3=camera['distortion_opencv']
    r2=x*x+y*y;radial=1+r2*(k1+r2*(k2+r2*k3))
    xd=x*radial+2*p1*x*y+p2*(r2+2*x*x);yd=y*radial+p1*(r2+2*y*y)+2*p2*x*y
    native=np.column_stack([xd,yd,np.ones(4)])@K.T;uv=native[:,:2]/native[:,2:]
    return np.r_[uv.min(0),uv.max(0)]


def verify_and_match(root,production_root,old_window,new_window,old_rows,new_rows,old_by,new_arrays):
    """Validate real PNG membership, then match people on shared native scans.

    Window-local track/segment suffixes are reported as hints only. Assignment
    uses same-scan actual-return overlap, 3D cloud position and observed bboxes.
    """
    from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
    from hmr4d.lidar_mocap.sessions import SessionIndex
    old_cal=CalibrationSnapshot.load(root/'calibration/calibration_snapshot.json')
    new_cal=CalibrationSnapshot.load(production_root/'calibration/calibration_snapshot.json')
    old_session=SessionIndex.load(Path(old_cal.data['sessions_root'])/SID,old_cal.frame_map_path(SID))
    new_session=SessionIndex.load(Path(new_cal.data['sessions_root'])/SID,new_cal.frame_map_path(SID))
    hashes=[json.loads((w/'fit_report.json').read_text(encoding='utf-8'))['source_lidar_sha256']
            for w in (old_window,new_window)]
    raw_cache={};source_validation=[];teacher_cache={};by_scan=[defaultdict(list),defaultdict(list)]
    for version,(window,rows,session,cal) in enumerate(((old_window,old_rows,old_session,old_cal),(new_window,new_rows,new_session,new_cal))):
        for row in rows:
            scan=row['lidar_index'];pair=session.pairs()[scan];path=Path(row['lidar_path']).resolve()
            if path!=Path(pair['lidar_path']).resolve():raise ValueError('Fit uses a different original PNG than its native scan')
            if row['scan_mid_us']!=pair['scan_mid_us']:raise ValueError('Fit native scan timestamp mismatch')
            for name in ('cam1','cam2'):
                for field in ('record_index','media_pts_us','common_time_us'):
                    if row['cameras'][name][field]!=pair['cameras'][name][field]:raise ValueError('Fit camera/native PTS evidence mismatch')
            if path not in raw_cache:
                digest=sha(path)
                raw_cache[path]=(session.read_cloud(pair,viewer=False),digest)
            raw,digest=raw_cache[path]
            declared=hashes[version].get(str(path))
            if declared!=digest:raise ValueError('Original PNG differs from the completed fit source SHA256')
            expected=(raw@session.Q.T).astype(np.float32)
            if version==0:
                track=int(row['motion_id'].rsplit(':',2)[-2]);data=old_by[(track,scan)]
            else:data=new_arrays[row['fit_path']]
            indices=data['source_indices']
            if len(indices)!=len(data['source_cloud']) or len(indices)==0 or indices.min()<0 or indices.max()>=len(raw):
                raise ValueError('Person source indices are outside the actual original PNG')
            if not np.array_equal(expected[indices],data['source_cloud']):raise ValueError('Person cloud is not an exact real PNG subset')
            bbox=observed_bbox(window,row,cal.camera('cam2'),teacher_cache)
            by_scan[version][scan].append(dict(row=row,data=data,source_indices=set(indices.tolist()),
                                               median=np.median(data['source_cloud'],axis=0),bbox=bbox))
            source_validation.append((version,scan))
    matches,unmatched,match_rows={},[],[]
    for scan in sorted(set(by_scan[0])&set(by_scan[1])):
        old,new=by_scan[0][scan],by_scan[1][scan]
        score=np.zeros((len(old),len(new)));evidence={}
        for i,a in enumerate(old):
            for j,b in enumerate(new):
                shared=len(a['source_indices']&b['source_indices'])
                overlap=shared/max(min(len(a['source_indices']),len(b['source_indices'])),1)
                distance=float(np.linalg.norm(a['median']-b['median']))
                iou=box_iou(a['bbox'],b['bbox'])
                score[i,j]=.6*overlap+.2*iou+.2*np.exp(-(distance/.5)**2)
                evidence[i,j]=dict(shared_returns=shared,source_overlap_over_minimum_count=overlap,
                                   cloud_median_distance_m=distance,observed_native_Brown_bbox_iou=iou)
        left,right=linear_sum_assignment(-score)
        accepted_old=set()
        for i,j in zip(left,right):
            ev=evidence[i,j]
            alternatives=np.r_[np.delete(score[i],j),np.delete(score[:,j],i)]
            margin=float(score[i,j]-alternatives.max()) if len(alternatives) else 1.
            record=dict(lidar_index=scan,old_motion_id=old[i]['row']['motion_id'],
                        production_motion_id=new[j]['row']['motion_id'],assignment_margin=margin,**ev)
            strong=(ev['source_overlap_over_minimum_count']>=.30 and ev['cloud_median_distance_m']<=.6
                    and ev['observed_native_Brown_bbox_iou']>=.10 and margin>=.08)
            record['status']='identity_verified' if strong else 'excluded_ambiguous_or_insufficient_identity_evidence'
            if strong:
                canonical=int(old[i]['row']['motion_id'].rsplit(':',2)[-2]);matches[(canonical,scan)]=new[j]['data'];accepted_old.add(i)
                record['personal_subset_exactly_equal']=np.array_equal(old[i]['data']['source_cloud'],new[j]['data']['source_cloud'])
                record['personal_source_indices_exactly_equal']=np.array_equal(old[i]['data']['source_indices'],new[j]['data']['source_indices'])
            match_rows.append(record)
        for i,a in enumerate(old):
            if i not in accepted_old:unmatched.append(dict(lidar_index=scan,old_motion_id=a['row']['motion_id']))
    for scan in sorted(set(by_scan[0])-set(by_scan[1])):
        unmatched.extend(dict(lidar_index=scan,old_motion_id=a['row']['motion_id']) for a in by_scan[0][scan])
    verified=[row for row in match_rows if row['status']=='identity_verified']
    new_group_matches=defaultdict(Counter)
    for row in verified:new_group_matches[row['production_motion_id']][row['old_motion_id']]+=1
    full_groups={}
    for motion_id in sorted({row['motion_id'] for row in new_rows}):
        own=sorted([row for row in new_rows if row['motion_id']==motion_id],key=lambda row:row['scan_mid_us'])
        samples=[new_arrays[row['fit_path']] for row in own]
        labels=new_group_matches.get(motion_id,{})
        full=dict(native_frames=len(own),accepted=sum(row['accepted'] for row in own),
                  native_scan_range=[own[0]['lidar_index'],own[-1]['lidar_index']],
                  old_identity_support_counts=dict(labels),
                  canonical_old_identity=next(iter(labels)) if len(labels)==1 else None,
                  identity_scope='same production motion segment; canonical name anchored only by verified old-window overlap')
        if len(own)>=3:
            clouds=[sample['source_cloud'] for sample in samples]
            cov=[source_coverage(clouds[i],clouds[i+1]) for i in range(len(clouds)-1)]
            metrics,_=temporal_metrics(np.array([sample['transl'] for sample in samples]),
                                      np.array([np.median(cloud,axis=0) for cloud in clouds]),
                                      np.array([row['scan_mid_us'] for row in own],np.int64),
                                      np.array([row['lidar_index'] for row in own]),cov)
            full['own_production_input_metrics']=metrics
            full['maximum_native_gap_us']=int(np.diff([row['scan_mid_us'] for row in own]).max())
        full_groups[motion_id]=full
    common=set(by_scan[0])&set(by_scan[1])
    return matches,dict(production_window=str(new_window),old_native_scans=len(by_scan[0]),production_native_scans=len(by_scan[1]),
        common_original_PNG_scans=len(common),actual_original_PNG_SHA256_and_person_indices_verified=True,
        old_person_source_checks=sum(v==0 for v,s in source_validation),production_person_source_checks=sum(v==1 for v,s in source_validation),
        source_capture_Q_equal=np.array_equal(old_session.Q,new_session.Q),
        raw_entire_scan_source_constant_on_shared_scans=True,
        identity_match_method='one-to-one actual same-PNG source-return overlap plus3D median distance plus observed native Brown bbox IoU; IDs alone never match',
        verified_person_matches=len(verified),personal_subsets_changed=sum(not row['personal_subset_exactly_equal'] for row in verified),
        excluded_identity_matches=unmatched,matches=match_rows,
        full_production_motion_segments=full_groups,
        benchmark_limit='Only original0..6s fixed-old-beta inputs. Full production outside this overlap has no old benchmark counterpart.')


def source_coverage(before, after):
    distances = np.r_[cKDTree(after).query(before,k=1)[0], cKDTree(before).query(after,k=1)[0]]
    return dict(symmetric_p90_m=float(np.quantile(distances,.9)), within5cm=float((distances<=.05).mean()))


def temporal_metrics(pelvis, proxy, times, scans, coverage):
    dt = np.diff(times)/1e6
    dp, dc = np.diff(pelvis,axis=0), np.diff(proxy,axis=0)
    velocity, proxy_velocity = dp/dt[:,None], dc/dt[:,None]
    acceleration_dt = (dt[1:]+dt[:-1])/2
    acceleration = np.diff(velocity,axis=0)/acceleration_dt[:,None]
    proxy_acceleration = np.diff(proxy_velocity,axis=0)/acceleration_dt[:,None]
    residual = np.linalg.norm(dp-dc,axis=1)
    ranked = np.argsort(residual)[::-1][:8]
    events = []
    for i in ranked:
        events.append(dict(previous_scan=int(scans[i]), lidar_index=int(scans[i+1]),
                           scan_mid_us=int(times[i+1]), source_offset_s=float((times[i+1]-times[0])/1e6),
                           pelvis_step_m=float(np.linalg.norm(dp[i])), cloud_proxy_step_m=float(np.linalg.norm(dc[i])),
                           pelvis_step_xyz_m=dp[i].tolist(), cloud_proxy_step_xyz_m=dc[i].tolist(),
                           step_residual_m=float(residual[i]), source_cloud_coverage=coverage[i],
                           acceleration_mps2=float(np.linalg.norm(acceleration[i-1])) if i>0 else None,
                           relative_acceleration_mps2=float(np.linalg.norm(acceleration[i-1]-proxy_acceleration[i-1])) if i>0 else None,
                           classification='ranked estimate-vs-visible-surface discrepancy; not a confirmed physical jump'))
    return dict(pelvis_step_m=stats(np.linalg.norm(dp,axis=1)), cloud_proxy_step_m=stats(np.linalg.norm(dc,axis=1)),
                pelvis_speed_mps=stats(np.linalg.norm(velocity,axis=1)),
                cloud_proxy_speed_mps=stats(np.linalg.norm(proxy_velocity,axis=1)),
                pelvis_minus_cloud_proxy_step_m=stats(residual),
                numerical_pelvis_acceleration_mps2=stats(np.linalg.norm(acceleration,axis=1)),
                numerical_cloud_proxy_acceleration_mps2=stats(np.linalg.norm(proxy_acceleration,axis=1)),
                numerical_relative_acceleration_mps2=stats(np.linalg.norm(acceleration-proxy_acceleration,axis=1)),
                ranked_discrepancies=events), dict(origin_us=int(times[0]), times_us=times[1:].tolist(), scans=scans[1:].tolist(),
                                                pelvis_step_m=np.linalg.norm(dp,axis=1).tolist(),
                                                proxy_step_m=np.linalg.norm(dc,axis=1).tolist(), residual_m=residual.tolist())


def parse_data_js(path):
    source = path.read_text(encoding='utf-8')
    return json.loads(source[source.find('(')+1:source.rfind(');')])


def diagnose(root, production_root, production_window=None):
    root, production_root = Path(root).resolve(), Path(production_root).resolve()
    old_window=root/'sessions'/SID/'windows'/WINDOW
    new_window=(production_root/'sessions'/SID/'windows'/WINDOW if production_window is None
                else production_window_path(production_root,production_window))
    old_rows, new_rows = read_rows(old_window), read_rows(new_window)
    old_by, new_by, new_arrays = {}, {}, {}
    for dest, rows in ((old_by, old_rows), (new_by, new_rows)):
        for row in rows:
            track = int(row['motion_id'].rsplit(':',2)[-2])
            with np.load(row['fit_path'],allow_pickle=False) as data:
                loaded = {key:data[key].copy() for key in
                    ('transl','joints','shape','source_cloud','source_indices','pc')}
                dest[(track,row['lidar_index'])]=loaded
                if dest is new_by:new_arrays[row['fit_path']]=loaded
    production_verification=None
    if production_window is not None:
        new_by,production_verification=verify_and_match(root,production_root,old_window,new_window,
                                                        old_rows,new_rows,old_by,new_arrays)
    benchmark_path = root/'calibration_refinement/fit_continuity_benchmark.final.npz'
    # The '.final' token is part of the basename, never replace it with_suffix.
    with np.load(benchmark_path,allow_pickle=False) as data:
        benchmark = {key:data[key].copy() for key in data.files if not key.endswith('_vertices')}
    sequence_path = root/'sequence_review/manifest.json'
    sequence = json.loads(sequence_path.read_text(encoding='utf-8'))
    clip = next(row for row in sequence['clips'] if row['id']=='clip10')
    clip_scans = {row['lidar_index'] for row in clip['frames']}
    calibration = json.loads((root/'calibration/calibration_snapshot.json').read_text(encoding='utf-8'))
    plane = calibration['reference_plane'];origin=np.asarray(plane['origin_viewer_m']);normal=np.asarray(plane['normal_viewer'])
    report_tracks, curves, source_checks = {}, {}, defaultdict(int)
    for track in range(1,5):
        prefix = '' if track==1 else f'track{track}_'
        scans = benchmark[prefix+'lidar_index'].astype(int)
        times = benchmark[prefix+'scan_mid_us'].astype(np.int64)
        matched=np.array([(track,int(scan)) in new_by for scan in scans])
        missing=scans[~matched].tolist()
        if matched.sum()<3:
            report_tracks[track]=dict(track_id=track,color_name=COLORS[track][0],
                                     comparison_status='insufficient_identity_verified_support',
                                     missing_production_scans=missing)
            continue
        benchmark_selected=np.flatnonzero(matched)
        scans,times=scans[matched],times[matched]
        old = [old_by[(track,int(s))] for s in scans]
        new = [new_by[(track,int(s))] for s in scans]
        proxy = np.array([np.median(d['source_cloud'],axis=0) for d in old])
        new_proxy=np.array([np.median(d['source_cloud'],axis=0) for d in new])
        low_height = np.array([np.quantile((d['source_cloud']-origin)@normal,.05) for d in old])
        torso = []
        for d in old:
            h=(d['source_cloud']-origin)@normal;middle=d['source_cloud'][(h>=.6)&(h<=1.5)]
            torso.append(np.median(middle,axis=0) if len(middle)>=30 else np.median(d['source_cloud'],axis=0))
        torso=np.asarray(torso)
        coverage = [source_coverage(old[i]['source_cloud'],old[i+1]['source_cloud']) for i in range(len(old)-1)]
        production_coverage=[source_coverage(new[i]['source_cloud'],new[i+1]['source_cloud']) for i in range(len(new)-1)]
        for a,b in zip(old,new):
            equal=True
            for key in ('source_cloud','source_indices','pc'):
                if not np.array_equal(a[key],b[key]):
                    equal=False
                    if production_window is None:raise ValueError(f'Old/new production source geometry differs: track{track}/{key}')
            source_checks['old_production_exact_cloud_and_index_and_FPS_pairs' if equal else 'verified_real_input_personal_subsets_or_FPS_changed']+=1
        trajectories={'old':np.array([d['transl'] for d in old]),
                      'fixed_beta_benchmark':benchmark[prefix+VARIANT+'_transl'][benchmark_selected],
                      'production':np.array([d['transl'] for d in new])}
        mask=np.array([s in clip_scans for s in scans])
        if mask.sum()<3:
            report_tracks[track]=dict(track_id=track,color_name=COLORS[track][0],
                                     comparison_status='insufficient_identity_verified_clip_support',
                                     missing_production_scans=missing)
            continue
        summaries={};curves[track]={}
        for variant,pelvis in trajectories.items():
            input_proxy=new_proxy if variant=='production' else proxy
            input_coverage=production_coverage if variant=='production' else coverage
            metrics,curve=temporal_metrics(pelvis[mask],input_proxy[mask],times[mask],scans[mask],
                                         [input_coverage[i] for i in range(len(input_coverage)) if mask[i] and mask[i+1]])
            full,_=temporal_metrics(pelvis,input_proxy,times,scans,input_coverage)
            metrics['input_cloud_proxy']='own verified production subset' if variant=='production' else 'original fixed benchmark input subset'
            torso_metrics,_=temporal_metrics(pelvis[mask],torso[mask],times[mask],scans[mask],
                                             [coverage[i] for i in range(len(coverage)) if mask[i] and mask[i+1]])
            metrics['height_band_surface_proxy_residual_m']=torso_metrics['pelvis_minus_cloud_proxy_step_m']
            summaries[variant]=dict(clip50=metrics,window60=full)
            curves[track][variant]=curve
        display_color=clip['motion_colors'][f'{SID}:{WINDOW}:{track}:0']
        report_tracks[track]=dict(track_id=track, display_color_rgb=display_color, color_name=COLORS[track][0],
            clip_native_scans=scans[mask].tolist(), requested_clip_native_scans=sorted(clip_scans),
            missing_production_scans=missing, window_native_scans=scans.tolist(), variants=summaries,
            full_cloud_surface_proxy='median of actual visible LiDAR returns; not anatomical pelvis ground truth',
            height_band_proxy='median of actual returns .6..1.5m above frozen plane; still visibility/posture dependent',
            low_surface_height_q05_m=stats(low_height[mask]),
            production_beta_max_difference_from_old=float(np.max(np.abs(np.array([d['shape'] for d in new])-np.array([d['shape'] for d in old])))),
            production_beta_max_temporal_range=float(np.ptp(np.array([d['shape'] for d in new]),axis=0).max()),
            benchmark_beta_max_difference_from_old=float(np.max(np.abs(benchmark[prefix+VARIANT+'_shape'][benchmark_selected]-np.array([d['shape'] for d in old])))),
            source_cloud_nearest_5cm_coverage=stats([c['within5cm'] for c in coverage]),
            timeline=[dict(lidar_index=int(s),scan_mid_us=int(t),clip_offset_s=float((t-clip['left_us'])/1e6),
                           proxy_m=p.tolist(),height_band_proxy_m=tor.tolist(),low_height_q05_m=float(lh),
                           proxy_by_variant_m={v:(new_proxy[i] if v=='production' else p).tolist() for v in trajectories},
                           pelvis_by_variant_m={v:values[i].tolist() for v,values in trajectories.items()})
                      for i,(s,t,p,tor,lh) in enumerate(zip(scans,times,proxy,torso,low_height))])
    comparison_dir=root/'calibration_refinement/comparison'
    comparison_manifest=json.loads((comparison_dir/'manifest.json').read_text(encoding='utf-8'))
    displayed={}
    for displayed_clip in comparison_manifest['clips']:
        payload=parse_data_js(comparison_dir/displayed_clip['id']/'data.js')
        checks={}
        for frame in payload['frames']:
            for actor in frame['athletes']:
                track=int(actor['motion_id'].rsplit(':',2)[-2]);scan=frame['lidar_index'];prefix='' if track==1 else f'track{track}_'
                lookup={int(s):i for i,s in enumerate(benchmark[prefix+'lidar_index'])}
                joints=np.frombuffer(base64.b64decode(actor['joints']),dtype='<f4').reshape(-1,3)
                cloud=np.frombuffer(base64.b64decode(actor['cloud']),dtype='<f4').reshape(-1,3)
                old=old_by[(track,scan)];new=new_by.get((track,scan))
                if not np.array_equal(cloud,old['source_cloud']):raise ValueError('Displayed source cloud was changed')
                values=checks.setdefault(track,dict(person_scans=0,max_joint_difference_old_m=0.,max_joint_difference_final_benchmark_m=0.,max_joint_difference_production_m=0.))
                values['person_scans']+=1
                targets=[('old',old['joints']),('final_benchmark',benchmark[prefix+VARIANT+'_joints'][lookup[scan]])]
                if new is not None:targets.append(('production',new['joints']))
                else:values['production_identity_unmatched_scans']=values.get('production_identity_unmatched_scans',0)+1
                for name,target in targets:
                    field=f'max_joint_difference_{name}_m';values[field]=max(values[field],float(np.abs(joints-target).max()))
        displayed[displayed_clip['id']]=checks
        for values in checks.values():
            if values.get('production_identity_unmatched_scans',0)==values['person_scans']:
                values['max_joint_difference_production_m']=None
    clip_index={frame['lidar_index']:i for i,frame in enumerate(clip['frames'])}
    requested_groups=[(1,[14,15]),(1,[37,38]),(1,[39,40]),(2,[23,24,25])]
    specified_events=[]
    for interpretation in ('native_lidar_index','display_zero_based_frame_index','display_one_based_frame_number'):
        for track,requested in requested_groups:
            if interpretation=='native_lidar_index':
                scans=requested
            else:
                offset=0 if interpretation=='display_zero_based_frame_index' else 1
                scans=[clip['frames'][number-offset]['lidar_index'] for number in requested]
            prefix='' if track==1 else f'track{track}_'
            lookup={int(s):i for i,s in enumerate(benchmark[prefix+'lidar_index'])}
            timeline={row['lidar_index']:row for row in report_tracks.get(track,{}).get('timeline',[])}
            steps=[]
            for previous,current in zip(scans,scans[1:]):
                if previous not in timeline or current not in timeline:
                    steps.append(dict(previous_lidar_index=previous,lidar_index=current,
                                      status='excluded_missing_identity_verified_production_support'))
                    continue
                before,after=timeline[previous],timeline[current]
                dt=(after['scan_mid_us']-before['scan_mid_us'])/1e6
                cloud_delta=np.asarray(after['proxy_m'])-before['proxy_m']
                coverage=source_coverage(old_by[(track,previous)]['source_cloud'],old_by[(track,current)]['source_cloud'])
                variants={}
                for variant in ('old','fixed_beta_benchmark','production'):
                    own_cloud_delta=np.asarray(after['proxy_by_variant_m'][variant])-before['proxy_by_variant_m'][variant]
                    displacement=np.asarray(after['pelvis_by_variant_m'][variant])-before['pelvis_by_variant_m'][variant]
                    metric=dict(pelvis_step_m=float(np.linalg.norm(displacement)),pelvis_delta_xyz_m=displacement.tolist(),
                                pelvis_speed_mps=float(np.linalg.norm(displacement)/dt),
                                cloud_proxy_step_m=float(np.linalg.norm(own_cloud_delta)),
                                cloud_proxy_step_residual_m=float(np.linalg.norm(displacement-own_cloud_delta)))
                    if previous-1 in timeline:
                        preceding=timeline[previous-1];previous_dt=(before['scan_mid_us']-preceding['scan_mid_us'])/1e6
                        v0=(np.asarray(before['pelvis_by_variant_m'][variant])-preceding['pelvis_by_variant_m'][variant])/previous_dt
                        acc=(displacement/dt-v0)/((dt+previous_dt)/2)
                        c0=(np.asarray(before['proxy_by_variant_m'][variant])-preceding['proxy_by_variant_m'][variant])/previous_dt
                        cloud_acc=(own_cloud_delta/dt-c0)/((dt+previous_dt)/2)
                        metric.update(numerical_acceleration_mps2=float(np.linalg.norm(acc)),
                                      numerical_acceleration_xyz_mps2=acc.tolist(),
                                      numerical_cloud_relative_acceleration_mps2=float(np.linalg.norm(acc-cloud_acc)))
                    variants[variant]=metric
                steps.append(dict(previous_lidar_index=previous,lidar_index=current,
                                  previous_display_zero_index=clip_index.get(previous),display_zero_index=clip_index.get(current),
                                  previous_display_one_frame=clip_index[previous]+1,display_one_frame=clip_index[current]+1,
                                  scan_mid_us=after['scan_mid_us'],clip_offset_s=after['clip_offset_s'],
                                  dt_s=dt,cloud_proxy_step_m=float(np.linalg.norm(cloud_delta)),
                                  cloud_proxy_delta_xyz_m=cloud_delta.tolist(),source_cloud_coverage=coverage,variants=variants))
            specified_events.append(dict(number_interpretation=interpretation,user_requested_numbers=requested,
                                         track_id=track,color=COLORS[track][0],resolved_native_scans=scans,steps=steps))
    report=dict(schema='badminton-all-track-motion-diagnosis-v1',session_id=SID, source_window=WINDOW,
                audited_versions=['original clip10','fit_continuity_benchmark.final all4 fixed-old-beta tracks',f'production continuity {new_window.name}'],
                gpu_used=False,source_geometry_or_comparison_modified=False,
                comparison_manifest_sha256=sha(comparison_dir/'manifest.json'),
                benchmark_final_report_sha256=sha(root/'calibration_refinement/fit_continuity_benchmark.final.json'),
                benchmark_final_arrays_sha256=sha(benchmark_path),
                production_fit_report_sha256=sha(new_window/'fit_report.json'),
                source_checks=dict(source_checks), displayed_comparison=displayed,
                displayed_comparison_only_track1_updated=all(displayed['clip02'][t]['max_joint_difference_old_m']==0 for t in (2,3,4)),
                yellow_points_note='Yellow is actual source_cloud for all persons and has no pelvis marker. Track1 mesh is orange/yellow-like; blue is track2.',
                temporal_step_rule='Differentiate actual scan_mid_us, not index/fps; native ~100ms scans remain discrete.',
                interpretation_limits=['Cloud medians and height-band medians are visible-surface proxies, not GT pelvis positions.',
                                       'Numerical acceleration may include estimator noise, visible-surface changes, and genuine articulation.',
                                       'No independent3D measurement confirms an airborne jump; no absolute speed/acceleration rejection gate is proposed.',
                                       'Original cloud coverage and RGB/stance context must be inspected before treating a rapid motion as estimation error.',
                                       'Benchmark fixes old betas; production fits its own per-track beta, so they are separate comparisons.',
                                       'The stale comparison actually leaves tracks2/3/4 old; correcting display is necessary before judging their refinement.'],
                specified_user_events=specified_events,
                production_window=str(new_window),production_window_identity_verification=production_verification,
                tracks=report_tracks)
    return report,curves


def markdown(report):
    lines=['# 四轨迹骨盆与显示版本诊断','',
           '当前comparison仅更换橙色轨迹1；蓝色2、绿色3、紫色4后片段的关节仍与原NPZ逐float32完全相同。它们与final四轨迹benchmark的最大关节差分别约41.8/7.7/22.2cm，前后视觉差异小有明确的旧数据原因。','',
           '| 轨迹/颜色 | pelvis步长最大 old→production (cm) | pelvis−云代理步长P90 old→benchmark→production (cm) | production residual max (cm) |',
           '|---|---:|---:|---:|']
    for track,a in report['tracks'].items():
        if 'variants' not in a:
            lines.append(f"| {track}/{a['color_name']} | identity support不足 | 未比较 | 未比较 |")
            continue
        old,bench,new=[a['variants'][v]['clip50'] for v in ('old','fixed_beta_benchmark','production')]
        lines.append(f"| {track}/{a['color_name']} | {old['pelvis_step_m']['max']*100:.2f}→{new['pelvis_step_m']['max']*100:.2f} | "
                     f"{old['pelvis_minus_cloud_proxy_step_m']['p90']*100:.2f}→{bench['pelvis_minus_cloud_proxy_step_m']['p90']*100:.2f}→{new['pelvis_minus_cloud_proxy_step_m']['p90']*100:.2f} | {new['pelvis_minus_cloud_proxy_step_m']['max']*100:.2f} |")
    lines += ['', '用户指定事件首先按native lidar_index核对；同时列出页面帧索引，避免数字误对：','',
              '| 轨迹 | native scan | 页面index(从0)/帧号(从1) | 原云步长cm | pelvis步长 old→production cm | 云代理残差 old→production cm |',
              '|---|---|---|---:|---:|---:|']
    for event in report['specified_user_events']:
        if event['number_interpretation']!='native_lidar_index':
            continue
        for step in event['steps']:
            if 'variants' not in step:
                lines.append(f"| {event['track_id']}/{event['color']} | {step['previous_lidar_index']}→{step['lidar_index']} | 身份/源帧支持不足 | 未比较 | 未比较 | 未比较 |")
                continue
            old,new=step['variants']['old'],step['variants']['production']
            lines.append(f"| {event['track_id']}/{event['color']} | {step['previous_lidar_index']}→{step['lidar_index']} | "
                         f"{step['previous_display_zero_index']}→{step['display_zero_index']} / {step['previous_display_one_frame']}→{step['display_one_frame']} | "
                         f"{step['cloud_proxy_step_m']*100:.2f} | {old['pelvis_step_m']*100:.2f}→{new['pelvis_step_m']*100:.2f} | "
                         f"{old['cloud_proxy_step_residual_m']*100:.2f}→{new['cloud_proxy_step_residual_m']*100:.2f} |")
    lines += ['', '若14/15等数字指页面index或1-based帧号，JSON另外保存两套解析后的native scans及完整old/benchmark/production量化。', '',
              '蓝色轨迹2的新生产剩余差异排名：', '']
    blue=report['tracks'].get(2,{})
    blue_events=blue.get('variants',{}).get('production',{}).get('clip50',{}).get('ranked_discrepancies',[])
    for event in blue_events[:5]:
        lines.append(f"- scan {event['lidar_index']}，时间 {event['scan_mid_us']}us：pelvis步长{event['pelvis_step_m']*100:.2f}cm，"
                     f"原云代理步长{event['cloud_proxy_step_m']*100:.2f}cm，差异{event['step_residual_m']*100:.2f}cm；"
                     f"原云5cm覆盖{event['source_cloud_coverage']['within5cm']:.3f}。")
    exact=report['source_checks'].get('old_production_exact_cloud_and_index_and_FPS_pairs',0)
    changed=report['source_checks'].get('verified_real_input_personal_subsets_or_FPS_changed',0)
    lines += ['', f"{exact}份old/production个人source_cloud、source_indices和256点FPS样本完全一致；{changed}份子集或FPS发生改变，分别保留真实来源，production用自己的可见表面代理。beta在benchmark固定old值，production是自己的每轨固定beta，因此分别列版本。", '']
    verification=report.get('production_window_identity_verification')
    if verification:
        lines += [f"新窗口 `{verification['production_window']}`：{verification['production_native_scans']}个native scans，"
                  f"{verification['production_person_source_checks']}个人子集均核验实际PNG+indices；共同原扫描{verification['common_original_PNG_scans']}，PNG SHA/source整扫保持不变。"
                  f"身份验证匹配{verification['verified_person_matches']}，子集变化{verification['personal_subsets_changed']}，身份不足排除{len(verification['excluded_identity_matches'])}。", '',
                  '人物匹配依据同一PNG的真实返回点重合、3D云位置和原视频teacher bbox的native Brown IoU，不依赖track编号猜身份。新full范围按自己的motion segment独立统计；没有旧窗口重合证据的segment不硬命名为旧轨迹。', '']
    lines += ['',
              '约10Hz帧阶跃会让真实迈步/挥拍运动看起来离散。cloud median不是GT：可见躯干/手臂/腿面积变化也可移动中位数；高度带代理同样受遮挡和姿态影响。加速度只是按真实scan时间对估计/表面代理数值微分，不能当作真实跳跃或异常标签。', '',
              '目前四轨生产的云代理残差P90约1.9–2.4cm；旧轨迹1约30cm的估计跳已经降低，但不能把真实约9cm/100ms动作强行平滑。先在更新后的三版本四轨同源页面对照RGB和真实云，再决定是否需调整拟合。', '',
              '完整60scan/clip50统计、逐scan时间、数值加速度和排名保存在diagnosis.json；motion.svg仅绘制已有真实云代理及已存模型估计，不改坐标或补帧。', '']
    return '\n'.join(lines)


def plot(curves,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(4,2,figsize=(13,13),sharex=True)
    styles={'old':('#aeb6c0','old'),'fixed_beta_benchmark':('#d9933a','fixed beta benchmark'),'production':('#3387d5','production')}
    for track in range(1,5):
        if not curves.get(track):
            continue
        for variant,curve in curves[track].items():
            t=(np.asarray(curve['times_us'])-curve['origin_us'])/1e6;color,label=styles[variant]
            axes[track-1,0].plot(t,np.asarray(curve['pelvis_step_m'])*100,color=color,label=label)
            axes[track-1,1].plot(t,np.asarray(curve['residual_m'])*100,color=color,label=label)
        curve=curves[track]['old'];t=(np.asarray(curve['times_us'])-curve['origin_us'])/1e6
        axes[track-1,0].plot(t,np.asarray(curve['proxy_step_m'])*100,color='#c2ac13',linestyle=':',label='actual cloud surface proxy')
        for column in range(2):
            axes[track-1,column].grid(alpha=.25);axes[track-1,column].set_ylabel(f'Track {track} ({COLORS[track][0]}) / cm')
            for scan in ({1:[15,38,40],2:[24,25]}.get(track,[])):
                at=curve['scans'].index(scan)
                axes[track-1,column].axvline(t[at],color='#666666',alpha=.25,linewidth=.8)
                axes[track-1,column].text(t[at],.98,f'scan{scan}',rotation=90,va='top',fontsize=7,
                                         transform=axes[track-1,column].get_xaxis_transform())
    axes[0,0].set_title('Native-scan pelvis step / visible surface proxy step')
    axes[0,1].set_title('Pelvis-step minus visible cloud-proxy step (not GT error)')
    axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    axes[-1,0].set_xlabel('Original clip elapsed time / s');axes[-1,1].set_xlabel('Original clip elapsed time / s')
    figure.tight_layout();figure.savefig(path);plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=Path('outputs/badminton_corner_fix_v1'))
    parser.add_argument('--production-root',type=Path,default=Path('outputs/badminton_corner_fix_v1_continuity'))
    parser.add_argument('--production-window',type=Path,help='Completed native window basename or full path; explicit identity/source validation')
    parser.add_argument('--output-dir',type=Path,help='Dedicated generated diagnosis directory')
    args=parser.parse_args();report,curves=diagnose(args.output_root,args.production_root,args.production_window)
    output=args.output_dir or args.output_root/'calibration_refinement/all_tracks_diagnosis'
    output=Path(output).resolve()
    if not output.is_relative_to(Path(__file__).resolve().parents[1]/'outputs'):
        raise ValueError('Generated diagnosis must remain under the GVHMR outputs directory')
    output.mkdir(parents=True,exist_ok=True)
    (output/'diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (output/'diagnosis.md').write_text(markdown(report),encoding='utf-8')
    plot(curves,output/'motion.svg')
    print(json.dumps({'comparison_only_track1_updated':report['displayed_comparison_only_track1_updated'],
                      'exact_source_pairs':report['source_checks'],'path':str((output/'diagnosis.json').resolve())},ensure_ascii=False))


if __name__=='__main__':main()
