"""Bounded global R/t refinement using spatially diverse real-cloud RGB masks.

Each camera has one rigid transform for all people/frames; original K/d, XYZ,
source M, court plane and clock maps remain fixed. Mask predictions are noisy
one-sided image evidence; static court anchors and a small parameter prior
prevent fitting per-person warp or unconstrained mask containment.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.pipeline import write_json
from refine_badminton_calibration import changed_camera,project,summary


def anchor_errors(anchors,camera,q,use):
    rows=[a for a in anchors if a['use']==use]
    return np.linalg.norm(project([a['fixed_viewer_xyz_m'] for a in rows],camera,q)-[a['measured_native_distorted_uv'] for a in rows],axis=1)


def mask_distances(records,camera,q):
    values=[]
    for record in records:
        uv=project(record['xyz'],camera,q)
        distance=map_coordinates(record['outside'],[uv[:,1],uv[:,0]],order=1,
                                 mode='constant',cval=1000.,prefilter=False,output=np.float64)
        values.append(distance)
    return values


def refine(camera,q,anchors,records):
    # Dimensionless optimizer coordinates give stable derivatives for metres
    # and radians. Bilinear distance interpolation avoids pixel-rounding loss.
    scales=np.array([.001,.001,.001,.01,.01,.01]);train=[a for a in anchors if a['use']=='train']
    weights=np.array([np.sqrt(a['goal_weight'])/a['sigma_px'] for a in train])
    xyz=[a['fixed_viewer_xyz_m'] for a in train];uv=[a['measured_native_distorted_uv'] for a in train]
    cell_counts=Counter(tuple(r['cell']) for r in records)
    sample_count=sum(len(r['xyz']) for r in records)
    def residual(z):
        candidate=changed_camera(camera,z*scales)
        court=((project(xyz,candidate,q)-uv)*weights[:,None]/np.sqrt(len(train))).ravel()
        image=[]
        for record,distance in zip(records,mask_distances(records,candidate,q)):
            # Equal occupied-cell influence, without copying/creating points.
            weight=np.sqrt(record['confidence']/cell_counts[tuple(record['cell'])])
            image.extend(np.minimum(distance,30.)/3*weight/np.sqrt(max(sample_count,1))*3)
        return np.concatenate([court,image,z*.4])
    fit=least_squares(residual,np.zeros(6),bounds=(-np.full(6,3.),np.full(6,3.)),
                      loss='soft_l1',f_scale=1.,max_nfev=100)
    candidate=changed_camera(camera,fit.x*scales)
    centre=-np.asarray(candidate['R_raw_lidar_to_camera']).T @ np.asarray(candidate['t_m'])
    candidate['camera_center_raw_lidar_m']=centre.tolist();candidate['camera_center_viewer_m']=(q @ centre).tolist()
    details={'optimizer_success':bool(fit.success),'nfev':fit.nfev,'delta_rotvec_rad_then_translation_m':(fit.x*scales).tolist(),
             'optimization_person_frames':len(records),'optimization_true_points':sample_count,
             'occupied_cells':sorted([list(cell) for cell in cell_counts]),
             'court_train_before_px':summary(anchor_errors(anchors,camera,q,'train')),
             'court_train_after_px':summary(anchor_errors(anchors,candidate,q,'train')),
             'court_dev_before_px':summary(anchor_errors(anchors,camera,q,'development_holdout')),
             'court_dev_after_px':summary(anchor_errors(anchors,candidate,q,'development_holdout')),
             'sampled_point_outside_distance_before_px':summary(np.concatenate(mask_distances(records,camera,q))),
             'sampled_point_outside_distance_after_px':summary(np.concatenate(mask_distances(records,candidate,q)))}
    return candidate,details


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-snapshot',type=Path,default=Path('outputs/badminton_corner_fix_v1/calibration/calibration_snapshot.json'))
    p.add_argument('--evidence-dir',type=Path,default=Path('outputs/badminton_corner_fix_v1/calibration_refinement'))
    p.add_argument('--evaluation-dir',type=Path);p.add_argument('--points-per-person',type=int,default=128)
    a=p.parse_args();cv2.setNumThreads(2);c=CalibrationSnapshot.load(a.baseline_snapshot);e=a.evidence_dir.resolve()
    if a.points_per_person<1:raise ValueError('points-per-person must be positive')
    if not e.is_relative_to(Path(__file__).resolve().parents[1]):raise ValueError('Candidate outputs must remain in GVHMR workspace')
    out=a.evaluation_dir or e/'spatial_alignment';selection=json.loads((e/'spatial_cases.json').read_text(encoding='utf-8'))
    report=json.loads((out/'joint_alignment.json').read_text(encoding='utf-8'));context=json.loads((e/'calibration_context.json').read_text(encoding='utf-8'))
    if report['calibration_digest']!=c.data['snapshot_digest']:raise ValueError('Mask evidence uses another calibration')
    measurements={(r['clip_id'],r['lidar_index'],r['camera'],r['motion_id']):r for r in report['measurements'] if r['variant']=='baseline'}
    cameras={};details={};excluded=[];started=time.perf_counter()
    for name in ('cam1','cam2'):
        records=[]
        for case in selection['cases']:
            frame=case['frame'];motion=case['selected_optimization_athlete_motion_id']
            row=measurements[(case['clip_id'],frame['lidar_index'],name,motion)]
            if not row['mask_matched'] or row['mask_confidence']<.5 or not row['fit_accepted']:
                excluded.append({'key':case['key'],'camera':name,'reason':'ambiguous_or_low_confidence_mask_or_rejected_source_fit'});continue
            athlete=next(v for v in frame['athletes'] if v['motion_id']==motion)
            with np.load(athlete['fit_path']) as stored:cloud=stored['source_cloud'].copy()
            indices=np.unique(np.linspace(0,len(cloud)-1,min(len(cloud),a.points_per_person)).astype(int))
            cache=out/'masks'/f"{case['key']}_{name}.npz"
            with np.load(cache) as observation:mask=observation['masks'][row['association']['detection_index']].astype(bool)
            outside=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            records.append({'xyz':cloud[indices],'outside':outside,'confidence':row['mask_confidence'],'cell':case['grid_cell']})
        if len(records)<8:raise ValueError(f'{name}: insufficient diverse confident masks for a global correction')
        anchors=[v for v in context['static_court_anchors'] if v['camera']==name]
        cameras[name],details[name]=refine(c.camera(name),c.Q,anchors,records)
    result={'schema':'badminton-spatial-mask-camera-candidates-v1','parent_calibration_digest':c.data['snapshot_digest'],
            'evaluator_report_sha256':hashlib.sha256((out/'joint_alignment.json').read_bytes()).hexdigest(),
            'spatial_selection_sha256':hashlib.sha256((e/'spatial_cases.json').read_bytes()).hexdigest(),
            'optimization_wall_s':time.perf_counter()-started,'selection':str((e/'spatial_cases.json').resolve()),
            'candidates':[{'name':'spatial_mask_pose','cameras':cameras,'details':details,'adopted':False}],
            'excluded':excluded,'source_xyz_changed':False,'K_and_distortion_changed':False,'clock_changed':False,
            'limits':['Instance masks are predictions and may have boundary/occlusion bias','Court plane XYZ are normative priors, not surveyed anchors','This small bounded optimizer does not establish a global optimum','Adoption requires evaluating every original point on original regression cases plus static court and fitted-body checks']}
    path=e/'spatial_camera_candidates.json';write_json(path,result)
    print(json.dumps({'report':str(path),'candidates':result['candidates'],'excluded':excluded},indent=2))


if __name__=='__main__':main()
