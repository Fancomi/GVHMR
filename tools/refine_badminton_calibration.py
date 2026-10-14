"""Test small global camera corrections against fixed RGB and court evidence.

Writes candidates and regression evidence only. It never changes the adopted
selection, original returns, per-person transforms or original timestamps.
Dynamic SMPL is a noisy proxy, not measured 3D calibration ground truth.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot


def project(points,camera,q):
    xyz=np.asarray(points) @ (np.asarray(camera['R_raw_lidar_to_camera']) @ np.linalg.inv(q)).T+camera['t_m']
    x,y=xyz[...,0]/xyz[...,2],xyz[...,1]/xyz[...,2]
    k1,k2,p1,p2,k3=camera['distortion_opencv'];r=x*x+y*y;rad=1+r*(k1+r*(k2+r*k3))
    u=x*rad+2*p1*x*y+p2*(r+2*x*x);v=y*rad+p1*(r+2*y*y)+2*p2*x*y
    h=np.stack([u,v,np.ones_like(u)],-1) @ np.asarray(camera['K']).T
    return h[...,:2]/h[...,2:]


def to_raw(uv,camera):
    uv=np.asarray(uv); k=np.asarray(camera['K']);xy=np.column_stack([uv[:,:2],np.ones(len(uv))]) @ np.linalg.inv(k).T
    x,y=xy[:,0],xy[:,1];k1,k2,p1,p2,k3=camera['distortion_opencv'];r=x*x+y*y;rad=1+r*(k1+r*(k2+r*k3))
    h=np.column_stack([x*rad+2*p1*x*y+p2*(r+2*x*x),y*rad+p1*(r+2*y*y)+2*p2*x*y,np.ones(len(uv))]) @ k.T
    return h[:,:2]/h[:,2:]


def summary(errors):
    values=np.asarray(errors,float)
    return {'count':int(values.size),'mean_px':float(values.mean()) if values.size else None,
            'median_px':float(np.median(values)) if values.size else None,
            'p90_px':float(np.quantile(values,.9)) if values.size else None,
            'rms_px':float(np.sqrt(np.mean(values**2))) if values.size else None}


def observe(report,exclude_last_clip=False,stable_only=False):
    records=[]
    regressor=None
    unstable={14,15,16,37,38,39,43,44,45,52,53,54}
    for frame in report['frames']:
        if exclude_last_clip and frame['clip_id']=='clip10':continue
        for a in frame['athletes']:
            if not a.get('fit_accepted',False):continue
            if stable_only and a['motion_id'].endswith(':1:0') and frame['lidar_index'] in unstable:continue
            if 'eligible_mask' in a:
                mask=np.asarray(a['eligible_mask'],bool)
            else:
                mask=np.zeros(17,dtype=bool)
                for joint in a.get('joint_metrics',[]):mask[joint['joint_index']]=joint['eligible']
            if mask.sum()<6:continue
            uv=np.asarray(a['all_coco17_observations'],float)
            raw=np.asarray(a.get('all_coco17_observations_raw_distorted',to_raw(uv,report['camera'])),float)[:,:2]
            if 'fixed_smpl_coco17_viewer_m' in a:
                xyz=np.asarray(a['fixed_smpl_coco17_viewer_m'],float)
            else:
                if regressor is None:regressor=torch.load('hmr4d/utils/body_model/smpl_coco17_J_regressor.pt',map_location='cpu').numpy()
                with np.load(a['fit_path']) as fit:xyz=regressor @ fit['vertices']
            records.append({'motion_id':a['motion_id'],'index':frame['lidar_index'],
                            'time_us':frame.get('observation_common_time_us',frame['cam1_native_source']['common_time_us'] if 'cam1_native_source' in frame else 0),
                            'xyz':xyz,'uv':raw,'mask':mask,'confidence':uv[:,2]})
    return records


def body_errors(records,camera,q):
    return np.concatenate([np.linalg.norm(project(r['xyz'],camera,q)-r['uv'],axis=1)[r['mask']] for r in records])


def changed_camera(base,delta):
    camera=copy.deepcopy(base);dr=Rotation.from_rotvec(delta[:3]).as_matrix()
    camera['R_raw_lidar_to_camera']=(dr @ np.asarray(base['R_raw_lidar_to_camera'])).tolist()
    camera['t_m']=(dr @ np.asarray(base['t_m'])+delta[3:6]).tolist()
    if len(delta)>6:
        k=np.asarray(base['K']).copy();k[0,0]*=np.exp(delta[6]);k[1,1]*=np.exp(delta[6]);camera['K']=k.tolist()
        d=np.asarray(base['distortion_opencv']).copy();d[0]+=delta[7];d[1]+=delta[8];camera['distortion_opencv']=d.tolist()
    camera['R_viewer_to_camera']=None  # Removed before publishing canonical values.
    camera.pop('R_viewer_to_camera',None)
    return camera


def spatial_candidates(calibration,context,train,validation):
    base=calibration.camera('cam1');q=calibration.Q
    anchors=[a for a in context['static_court_anchors'] if a['camera']=='cam1']
    def court(camera,use):
        rows=[a for a in anchors if a['use']==use]
        return np.linalg.norm(project([a['fixed_viewer_xyz_m'] for a in rows],camera,q)-[a['measured_native_distorted_uv'] for a in rows],axis=1)
    baseline={'clip10_stable_proxy':summary(body_errors(train,base,q)),
              'other_clips_proxy':summary(body_errors(validation,base,q)),
              'court_train':summary(court(base,'train')),'court_development_holdout':summary(court(base,'development_holdout'))}
    # The current camera already has strong static evidence. Small bounded
    # corrections cannot chase per-frame model noise without losing that check.
    scales=np.array([.001,.001,.001,.01,.01,.01,.002,.001,.01])
    bounds=np.array([.003,.003,.003,.03,.03,.03,.005,.003,.03])
    joint_weights=np.array([.4,.4,.4,.3,.3,1,1,.7,.7,.3,.3,1,1,.8,.8,.6,.6])
    static=[a for a in anchors if a['use']=='train'];n=sum(r['mask'].sum() for r in train)
    results=[]
    for dimensions,name in [(6,'small_global_pose'),(9,'small_global_pose_focal_radial')]:
        def residual(delta):
            camera=changed_camera(base,delta);body=[]
            for r in train:
                diff=project(r['xyz'],camera,q)-r['uv'];weight=np.sqrt(r['confidence']*joint_weights)
                body.extend((diff[r['mask']]*weight[r['mask'],None]/8/np.sqrt(n)*2).ravel())
            xyz=[a['fixed_viewer_xyz_m'] for a in static];uv=[a['measured_native_distorted_uv'] for a in static]
            weights=np.array([np.sqrt(a['goal_weight'])/a['sigma_px'] for a in static])
            fixed=((project(xyz,camera,q)-uv)*weights[:,None]/np.sqrt(len(static))).ravel()
            return np.concatenate([body,fixed,delta/scales[:dimensions]*.3])
        fit=least_squares(residual,np.zeros(dimensions),bounds=(-bounds[:dimensions],bounds[:dimensions]),loss='soft_l1',f_scale=1,max_nfev=150)
        camera=changed_camera(base,fit.x)
        metrics={'clip10_stable_proxy':summary(body_errors(train,camera,q)),
                 'other_clips_proxy':summary(body_errors(validation,camera,q)),
                 'court_train':summary(court(camera,'train')),'court_development_holdout':summary(court(camera,'development_holdout'))}
        checks={'meaningful_clip10_proxy_improvement':metrics['clip10_stable_proxy']['mean_px']<=baseline['clip10_stable_proxy']['mean_px']*.95,
                'other_clips_do_not_regress':metrics['other_clips_proxy']['mean_px']<=baseline['other_clips_proxy']['mean_px']+.1,
                'static_train_does_not_regress':metrics['court_train']['rms_px']<=baseline['court_train']['rms_px']+.1,
                'static_development_does_not_regress':metrics['court_development_holdout']['rms_px']<=baseline['court_development_holdout']['rms_px']+.1}
        results.append({'name':name,'delta':fit.x.tolist(),'optimizer_success':bool(fit.success),
                        'camera':camera,'metrics':metrics,'checks':checks,'passes_numeric_regression':all(checks.values()),
                        'adopted':False,'adoption_limit':'Dynamic SMPL is not measured 3D ground truth; original active calibration remains unchanged'})
    return {'baseline':baseline,'candidates':results,'xyz_moved':False,'camera_model_scope':'one global K/d/R/t per camera; cam2 entirely unchanged'}


def time_profiles(calibration,manifest,reports):
    clip=next(c for c in manifest['clips'] if c['id']=='clip10')
    regressor=torch.load('hmr4d/utils/body_model/smpl_coco17_J_regressor.pt',map_location='cpu').numpy()
    trajectories={}
    for f in clip['frames']:
        for a in f['athletes']:
            if not a['accepted']:continue
            # Exclude independently diagnosed root jumps from clock fitting.
            if a['motion_id'].endswith(':1:0') and f['lidar_index'] in {14,15,16,37,38,39,43,44,45,52,53,54}:continue
            with np.load(a['fit_path']) as fit:joints=regressor @ fit['vertices']
            trajectories.setdefault(a['motion_id'],[]).append((f['scan_mid_us'],joints))
    slow=np.array([5,6,11,12]); offsets=np.arange(-50,51,5,dtype=float)
    result={}
    for name,report in reports.items():
        per_track={};camera=calibration.camera(name)
        for r in observe(report,stable_only=True):
            if r['motion_id'] not in trajectories:continue
            trajectory=trajectories[r['motion_id']];ts=np.array([v[0] for v in trajectory],dtype=np.int64);xyz=np.array([v[1] for v in trajectory]);mask=r['mask'][slow]
            if not mask.any():continue
            errors=[];valid=True
            for delta in offsets:
                query=r['time_us']-delta*1000;right=np.searchsorted(ts,query)
                if right==0 or right==len(ts) or ts[right]-ts[right-1]>150000:valid=False;break
                alpha=(query-ts[right-1])/(ts[right]-ts[right-1]);joints=(1-alpha)*xyz[right-1]+alpha*xyz[right]
                e=np.linalg.norm(project(joints[slow],camera,calibration.Q)-r['uv'][slow],axis=1)[mask]
                errors.append(e)
            if valid:per_track.setdefault(r['motion_id'],[]).append(errors)
        profiles=[]
        for motion,values in per_track.items():
            means=[float(np.concatenate([v[i] for v in values]).mean()) for i in range(len(offsets))]
            best=int(np.argmin(means));near=[float(offsets[i]) for i,v in enumerate(means) if v<=means[best]+.5]
            profiles.append({'motion_id':motion,'joint_count':sum(len(v[0]) for v in values),'offset_ms':offsets.tolist(),
                             'mean_error_px':means,'best_camera_query_offset_ms':float(offsets[best]),
                             'within_half_pixel_ms_range':[min(near),max(near)]})
        result[name]={'profiles':profiles,'adopted_offset_ms':0,'diagnostic_only':True,
                      'definition':'camera frame query common_time = lidar midpoint + offset; interpolate only model trajectory for diagnostic, never original XYZ',
                      'limit':'5Hz observed RGB and 10Hz proxy mesh, occlusion and missing per-point times make fast dynamics/physical exposure unverified'}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-snapshot',type=Path,required=True)
    p.add_argument('--evidence-dir',type=Path,default=Path('outputs/badminton_corner_fix_v1/calibration_refinement'))
    p.add_argument('--review-dir',type=Path,default=Path('outputs/badminton_corner_fix_v1/sequence_review'));a=p.parse_args()
    c=CalibrationSnapshot.load(a.baseline_snapshot);context=json.loads((a.evidence_dir/'calibration_context.json').read_text());report=json.loads((a.evidence_dir/'rgb_keypoints_cam1_clip10.json').read_text());old=json.loads((a.review_dir/'cam1_check.json').read_text());manifest=json.loads((a.review_dir/'manifest.json').read_text())
    train=observe(report,stable_only=True);validation=observe(old,exclude_last_clip=True)
    reports={name:json.loads((a.evidence_dir/f'rgb_keypoints_{name}_clip10.json').read_text()) for name in ['cam1','cam2']}
    assert all(r['calibration_digest']==c.data['snapshot_digest'] and r['completed'] for r in reports.values())
    result={'schema':'badminton-global-calibration-candidate-diagnostic-v1','parent_calibration_digest':c.data['snapshot_digest'],
            'parent_active_selection_sha256':c.data['active_selection_sha256'],'spatial':spatial_candidates(c,context,train,validation),
            'temporal':time_profiles(c,manifest,reports),'active_selection_changed':False,
            'limits':['Court XYZ are normative-plane priors, not surveyed/measured raw LiDAR anchors','RGB keypoints are detector estimates; fixed SMPL proxy was fitted with baseline cam2','Previously reviewed development cases are regression, not a blind independent test']}
    path=a.evidence_dir/'global_calibration_candidates.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'report':str(path),'baseline':result['spatial']['baseline'],'candidates':[{k:v for k,v in x.items() if k not in ['camera','delta']} for x in result['spatial']['candidates']]},indent=2))


if __name__=='__main__':main()
