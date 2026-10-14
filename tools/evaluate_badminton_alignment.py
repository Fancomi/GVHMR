"""Compare fixed court, native RGB person masks and real LiDAR/SMPL geometry.

RGB masks are independent model predictions, not ground truth. Associations
are established once from baseline geometry and held fixed across candidates.
Original XYZ, timestamps and the adopted camera selection are never changed.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import time
import cv2
import numpy as np
import smplx
import torch
from ultralytics import YOLO
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.sessions import SessionIndex
from hmr4d.lidar_mocap.review import fill_triangle_union
from check_badminton_cam1 import associate_boxes,clip_box
from refine_badminton_calibration import project


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def summary(values):
    values=np.asarray(values,dtype=float)
    return {'count':int(values.size),'mean':float(values.mean()) if values.size else None,
            'median':float(np.median(values)) if values.size else None,
            'p90':float(np.quantile(values,.9)) if values.size else None}


def pixel_points(xyz,camera,q,width,height):
    uv=project(xyz,camera,q)
    depth=(np.asarray(xyz) @ (np.asarray(camera['R_raw_lidar_to_camera']) @ np.linalg.inv(q)).T+camera['t_m'])[:,2]
    good=np.isfinite(uv).all(1)&(depth>0)
    xy=np.zeros_like(uv,dtype=np.int64);xy[good]=np.rint(uv[good]).astype(np.int64)
    inside=good&(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)
    return uv,depth,xy,inside


def silhouette(vertices,faces,camera,q,width,height):
    uv,depth,_,_=pixel_points(vertices,camera,q,width,height)
    good=(depth[faces]>0).all(1)&np.isfinite(uv[faces]).all((1,2))
    result=np.zeros((height,width),dtype=np.uint8)
    # fillPoly over several contours applies the even/odd rule: overlapping
    # front/back triangles can erase the body interior. A silhouette is their
    # union, so rasterize each convex triangle into the same destination.
    fill_triangle_union(result,np.rint(uv[faces[good]]).astype(np.int32),1)
    return result.astype(bool)


def read_cases(manifest):
    cases=[]
    for clip in manifest['clips']:
        indices=range(0,len(clip['frames']),2) if clip['id']=='clip10' else sorted({0,len(clip['frames'])//2,len(clip['frames'])-1})
        for index in indices:
            frame=clip['frames'][index]
            cases.append({'clip_id':clip['id'],'session_id':clip['session_id'],'frame':frame,
                          'key':f"{clip['id']}_{frame['lidar_index']:06d}"})
    return cases


def final_geometry(bench_prefix):
    report=json.loads(Path(str(bench_prefix)+'.json').read_text(encoding='utf-8'))
    data=np.load(Path(str(bench_prefix)+'.npz'));variant='batch128_weight0.2';final={}
    scans=np.asarray(data['lidar_index'],dtype=int)
    for track in range(1,5):
        prefix='' if track==1 else f'track{track}_'
        key=prefix+variant+'_vertices'
        if key not in data:raise ValueError('Final four-track benchmark missing '+key)
        for index,scan in enumerate(scans):final[(track,int(scan))]=data[key][index].copy()
    return final,report


def mask_observation(model,image,cache,identity,options):
    if cache.exists():
        with np.load(cache) as saved:
            if str(saved['identity'])!=identity:raise ValueError('Mask cache provenance differs')
            return saved['masks'].astype(bool),saved['boxes'],saved['confidence']
    prediction=model.predict(source=image,classes=[0],conf=options['confidence'],iou=.7,
                             imgsz=options['imgsz'],device=options['device'],retina_masks=True,verbose=False)[0]
    masks=prediction.masks.data.cpu().numpy().astype(bool) if prediction.masks is not None else np.zeros((0,*image.shape[:2]),bool)
    if masks.shape[1:]!=image.shape[:2]:raise ValueError('Mask pixels must use the original native image')
    boxes=prediction.boxes.xyxy.cpu().numpy();confidence=prediction.boxes.conf.cpu().numpy()
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,masks=masks,boxes=boxes,confidence=confidence,identity=identity)
    return masks,boxes,confidence


def evaluate_case(case,camera_name,image,masks,boxes,confidence,cameras,calibration,faces,final,output,court,corrected_fits=None):
    height,width=image.shape[:2];frame=case['frame'];base=cameras['baseline'][camera_name]
    geometry=[];mesh_boxes=[];members=[]
    for athlete in frame['athletes']:
        with np.load(athlete['fit_path']) as stored:
            vertices=stored['vertices'].copy();cloud=stored['source_cloud'].copy()
        track=int(athlete['motion_id'].rsplit(':',2)[-2])
        corrected=final.get((track,frame['lidar_index']),vertices) if case['clip_id']=='clip10' else vertices
        corrected_path=(corrected_fits or {}).get(f"{athlete['motion_id']}|{frame['lidar_index']}")
        if corrected_path:
            with np.load(corrected_path) as stored:
                np.testing.assert_array_equal(stored['source_cloud'],cloud)
                corrected=stored['vertices'].copy()
        uv,depth,_,_=pixel_points(vertices,base,calibration.Q,width,height)
        box=clip_box(uv,depth,width,height)
        geometry.append((athlete,vertices,corrected,cloud))
        if box is not None:mesh_boxes.append(box);members.append(len(geometry)-1)
    matches,rejections=associate_boxes(mesh_boxes,boxes)
    matched={members[k]:v for k,v in matches.items()};rows=[];overlays={}
    for variant,camera_set in cameras.items():
        camera=camera_set[camera_name];overlay=image.copy();layer=image.copy()
        for mask in masks:layer[mask]=np.array([80,190,80],dtype=np.uint8)
        overlay=cv2.addWeighted(layer,.2,overlay,.8,0)
        for segment in court.get('court_line_segments',[]):
            points=segment.get('viewer_xyz_m',segment.get('dense_viewer_xyz_m'))
            if points is None:continue
            uv,depth,_,_=pixel_points(points,camera,calibration.Q,width,height)
            if (depth>0).all() and np.isfinite(uv).all():cv2.polylines(overlay,[np.rint(uv).astype(np.int32)],False,(255,255,0),2,cv2.LINE_AA)
        for index,(athlete,vertices,corrected,cloud) in enumerate(geometry):
            selected=corrected if variant.startswith('continuity') else vertices
            uv,_,xy,in_image=pixel_points(cloud,camera,calibration.Q,width,height)
            overlay[xy[in_image,1],xy[in_image,0]]=(0,255,255)
            body=silhouette(selected,faces,camera,calibration.Q,width,height)
            contours,_=cv2.findContours(body.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay,contours,-1,(255,80,210),1,cv2.LINE_AA)
            row={'variant':variant,'camera':camera_name,'clip_id':case['clip_id'],'session_id':case['session_id'],
                 'lidar_index':frame['lidar_index'],'scan_mid_us':frame['scan_mid_us'],'native_camera_row':frame['cameras'][camera_name],
                 'motion_id':athlete['motion_id'],'fit_accepted':athlete['accepted'],'source_point_count':len(cloud),
                 'points_in_original_image':int(in_image.sum()),'mask_matched':index in matched}
            if index in matched:
                match=matched[index];mask=masks[match['detection_index']]
                # One independent original-image mask for all camera/fit candidates.
                outside_distance=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                visible_xy=xy[in_image];hits=mask[visible_xy[:,1],visible_xy[:,0]]
                distances=outside_distance[visible_xy[:,1],visible_xy[:,0]]
                intersection=int((mask&body).sum());union=int((mask|body).sum())
                row.update(association=match,mask_confidence=float(confidence[match['detection_index']]),
                           mask_area_px=int(mask.sum()),lidar_points_inside_mask=int(hits.sum()),
                           lidar_points_within_3px_mask=int((distances<=3).sum()),
                           lidar_mask_hit_rate=float(hits.mean()) if hits.size else None,
                           lidar_mask_3px_hit_rate=float((distances<=3).mean()) if hits.size else None,
                           lidar_outside_mask_distance_px=summary(distances),
                           smpl_mask_iou=intersection/max(union,1),smpl_silhouette_area_px=int(body.sum()))
            rows.append(row)
        cv2.putText(overlay,f"{case['key']} {camera_name} {variant} | cyan court / green RGB mask / yellow LiDAR / pink SMPL",(15,27),cv2.FONT_HERSHEY_SIMPLEX,.5,(245,245,245),1,cv2.LINE_AA)
        # Store baseline and selected body correction; all candidate metrics are
        # retained even though the gallery stays compact.
        if variant in ('baseline','continuity_baseline','continuity_projection_retry'):
            relative=f"images/{case['key']}_{camera_name}_{variant}.jpg";path=output/relative;path.parent.mkdir(exist_ok=True)
            resized=cv2.resize(overlay,(1280,round(height*1280/width)))
            if not cv2.imwrite(str(path),resized,[cv2.IMWRITE_JPEG_QUALITY,87]):raise OSError('Cannot write review image')
            overlays[variant]=relative
    return rows,overlays,{'detections':len(boxes),'baseline_athletes':len(geometry),'matched':len(matched),'rejected_associations':rejections}


def aggregate(rows):
    result={}
    for variant in sorted({r['variant'] for r in rows}):
        result[variant]={}
        for camera in ('cam1','cam2'):
            all_rows=[r for r in rows if r['variant']==variant and r['camera']==camera]
            eligible=[r for r in all_rows if r['mask_matched'] and r['points_in_original_image']>0]
            points=sum(r['points_in_original_image'] for r in eligible)
            result[variant][camera]={'athlete_frames':len(all_rows),'mask_matched_frames':len(eligible),
                'association_coverage':len(eligible)/max(len(all_rows),1),'original_lidar_points_scored':points,
                'point_weighted_exact_mask_hit_rate':sum(r['lidar_points_inside_mask'] for r in eligible)/max(points,1),
                'point_weighted_3px_mask_hit_rate':sum(r['lidar_points_within_3px_mask'] for r in eligible)/max(points,1),
                'per_athlete_mask_hit_rate':summary([r['lidar_mask_hit_rate'] for r in eligible]),
                'per_athlete_outside_distance_mean_px':summary([r['lidar_outside_mask_distance_px']['mean'] for r in eligible]),
                'smpl_mask_iou':summary([r['smpl_mask_iou'] for r in eligible])}
    return result


def fixed_rgb_checks(evidence,cameras,calibration,final):
    regressor=torch.load('hmr4d/utils/body_model/smpl_coco17_J_regressor.pt',map_location='cpu').numpy()
    result={variant:{} for variant in cameras}
    for camera in ('cam1','cam2'):
        observed=json.loads((evidence/f'rgb_keypoints_{camera}_clip10.json').read_text(encoding='utf-8'))
        errors={variant:[] for variant in cameras}
        for frame in observed['frames']:
            for athlete in frame['athletes']:
                if not athlete['fit_accepted']:continue
                mask=np.asarray(athlete['eligible_mask'],bool);uv=np.asarray(athlete['all_coco17_observations_raw_distorted'])[:,:2]
                old=np.asarray(athlete['fixed_smpl_coco17_viewer_m']);track=int(athlete['motion_id'].rsplit(':',2)[-2])
                new=regressor @ final[(track,frame['lidar_index'])]
                for variant,camera_set in cameras.items():
                    xyz=old if variant=='baseline' else new
                    error=np.linalg.norm(project(xyz,camera_set[camera],calibration.Q)-uv,axis=1)
                    errors[variant].extend(error[mask].tolist())
        for variant,values in errors.items():result[variant][camera]=summary(values)
    return {'pixel_space':'original_native_distorted_Brown_pixels','fixed_original_association_and_eligibility':True,
            'scope':'clip10 fixed independently observed keypoints; physical nearest native camera times preserved',
            'not_ground_truth':True,'variants':result}


def write_gallery(output,report):
    aggregates=report['mask_alignment'];table=[]
    labels={'baseline':'原标定／原拟合','continuity_baseline':'原标定／连续拟合',
            'continuity_small_global_pose':'小幅外参候选／连续拟合',
            'continuity_small_global_pose_focal_radial':'外参及镜头候选／连续拟合',
            'continuity_spatial_mask_pose':'多位置联合外参候选／连续拟合',
            'continuity_projection_retry':'更强 RGB 候选／连续拟合'}
    for variant,cameras in aggregates.items():
        for camera,r in cameras.items():
            iou=r['smpl_mask_iou']['mean'];iou_label=f'{iou:.3f}' if iou is not None else '未评估'
            table.append(f"<tr><td>{html.escape(labels.get(variant,variant))}</td><td>{camera}</td><td>{r['mask_matched_frames']}/{r['athlete_frames']}</td><td>{r['point_weighted_exact_mask_hit_rate']:.2%}</td><td>{r['point_weighted_3px_mask_hit_rate']:.2%}</td><td>{iou_label}</td></tr>")
    pictures=[]
    for case in report['cases']:
        views=[]
        for camera,images in case['images'].items():
            for variant,path in images.items():views.append(f'<figure><figcaption>{camera} · {labels.get(variant,variant)}</figcaption><img loading="lazy" src="{path}" /></figure>')
        pictures.append(f"<details><summary>{case['key']} · {case['session_id']}</summary><div class='grid'>{''.join(views)}</div></details>")
    page='''<!doctype html><meta charset="utf-8"><title>双相机、LiDAR 与 SMPL 联合评估</title><style>body{background:#101720;color:#d7e4ef;font:16px system-ui;margin:32px}a{color:#73c3ff}table{border-collapse:collapse}td,th{border:1px solid #344052;padding:9px}details{margin:12px 0;border:1px solid #344052;padding:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}img{width:100%}pre{white-space:pre-wrap}</style><h1>双相机、LiDAR 与 SMPL 联合评估</h1><p>青色：冻结球场线的 Brown 投影；绿色：原图模型预测人体 mask；黄色：真实 LiDAR 点；粉色：SMPL 轮廓。人物关联固定；没有移动原始点云或时间戳。</p><p>Mask 来自实例分割模型，遮挡、远处小人体和边界误差均影响覆盖率。3 px 容差与严格命中同时报告；点云覆盖率不等同于稠密 mask IoU。SMPL mask IoU 单独列出。</p><table><tr><th>候选</th><th>相机</th><th>有 mask 的人体帧</th><th>LiDAR 严格命中</th><th>3 px 容差命中</th><th>SMPL mask IoU</th></tr>'''+''.join(table)+'</table><details><summary>静态画线投影、最终点到表面误差和同观察集 RGB 对照</summary><pre>'+html.escape(json.dumps({'court':report['court_evaluation'],'body':report['body_evaluation'],'rgb':report['fixed_rgb_checks']},indent=2,ensure_ascii=False))+'</pre></details>'+''.join(pictures)+'<p><a href="joint_alignment.json">完整指标、输入哈希与限制</a></p>'
    if report.get('spatial_selection'):
        coverage='<h2>空间分层选帧</h2><p>新增位置来自真实人体点云的场地平面中心。仅填实际有观测的网格；新增位置参与优化，原十组片段作回归。</p><img style="max-width:800px" src="../spatial_cases.svg"><details><summary>选帧来源与实际覆盖</summary><pre>'+html.escape(json.dumps({k:v for k,v in report['spatial_selection'].items() if k!='cases'},indent=2,ensure_ascii=False))+'</pre></details>'
        page=page.replace('<table><tr><th>候选</th>',coverage+'<table><tr><th>候选</th>',1)
    if report.get('selection'):
        selection='<p>'+html.escape(report['selection'].get('description_zh',report['selection']['reason']))+'</p>'
        page=page.replace('<h1>双相机、LiDAR 与 SMPL 联合评估</h1>','<h1>双相机、LiDAR 与 SMPL 联合评估</h1>'+selection,1)
    (output/'index.html').write_text(page,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,default=Path('outputs/badminton_corner_fix_v1'))
    p.add_argument('--model',type=Path,required=True);p.add_argument('--imgsz',type=int,default=1920);p.add_argument('--confidence',type=float,default=.25)
    p.add_argument('--device',default='0');p.add_argument('--benchmark-prefix',type=Path)
    p.add_argument('--additional-cases',type=Path);p.add_argument('--evaluation-dir',type=Path)
    p.add_argument('--only-spatial',action='store_true',help='Prepare added optimization masks before the full regression comparison')
    p.add_argument('--corrected-fits',type=Path);p.add_argument('--camera-candidates',type=Path)
    p.add_argument('--retry-fits',type=Path,help='Add isolated stronger-RGB comparisons without mixing evaluation cohorts');a=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(2);root=a.output_root.resolve();evidence=root/'calibration_refinement';out=(a.evaluation_dir or evidence/'joint_alignment').resolve();out.mkdir(parents=True,exist_ok=True)
    workspace=Path(__file__).resolve().parents[1]
    if not out.is_relative_to(workspace):raise ValueError('Outputs must remain in GVHMR workspace')
    calibration=CalibrationSnapshot.load(root/'calibration/calibration_snapshot.json')
    manifest_path=root/'sequence_review/manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    candidates=json.loads((evidence/'global_calibration_candidates.json').read_text(encoding='utf-8'))
    court=json.loads((evidence/'court_evaluation.json').read_text(encoding='utf-8'))
    body_path=evidence/'body_joint_evaluation.json'
    final,benchmark=final_geometry(a.benchmark_prefix or evidence/'fit_continuity_benchmark.final')
    camera_base={name:calibration.camera(name) for name in ('cam1','cam2')};cameras={'baseline':camera_base,'continuity_baseline':camera_base}
    for candidate in candidates['spatial']['candidates']:
        cameras['continuity_'+candidate['name']]={'cam1':candidate['camera'],'cam2':camera_base['cam2']}
    if a.camera_candidates:
        for candidate in json.loads(a.camera_candidates.read_text(encoding='utf-8'))['candidates']:
            cameras['continuity_'+candidate['name']]=candidate['cameras']
    corrected_fits=json.loads(a.corrected_fits.read_text(encoding='utf-8'))['corrected_fit_paths'] if a.corrected_fits else {}
    retry_report=json.loads(a.retry_fits.read_text(encoding='utf-8')) if a.retry_fits else None
    retry_fits=retry_report['corrected_fit_paths'] if retry_report else {};retry_rows=[]
    options={'imgsz':a.imgsz,'confidence':a.confidence,'device':a.device,'retina_masks':True,'classes':[0]}
    model_info={'path':str(a.model.resolve()),'sha256':sha(a.model),'ultralytics':__import__('ultralytics').__version__,'options':options}
    model=YOLO(a.model)
    if model.task!='segment':raise ValueError('A detection bbox checkpoint cannot supply instance masks')
    topology=smplx.create(workspace/'inputs/checkpoints/body_models',model_type='smpl',gender='neutral');faces=np.asarray(topology.faces,dtype=np.int64);del topology
    cases=read_cases(manifest);regression_keys={c['key'] for c in cases};groups=defaultdict(list)
    additional=json.loads(a.additional_cases.read_text(encoding='utf-8')) if a.additional_cases else None
    if a.only_spatial and not additional:raise ValueError('--only-spatial requires --additional-cases')
    if additional:cases=([] if a.only_spatial else cases)+additional['cases']
    for case in cases:
        for camera in ('cam1','cam2'):groups[(case['session_id'],camera)].append(case)
    rows=[];case_reports={c['key']:{'key':c['key'],'session_id':c['session_id'],'clip_id':c['clip_id'],'images':{},'association':{}} for c in cases};started=time.perf_counter()
    for (sid,camera),group in groups.items():
        session=SessionIndex.load(Path(calibration.data['sessions_root'])/sid,calibration.frame_map_path(sid));by_index=defaultdict(list)
        for case in group:by_index[case['frame']['cameras'][camera]['record_index']].append(case)
        for index,image in session.iter_video_frames(camera,sorted(by_index)):
            for case in by_index[index]:
                identity=hashlib.sha256(json.dumps({'model':model_info,'video':str(session.video_paths[camera]),'video_bytes':session.video_paths[camera].stat().st_size,'video_mtime_ns':session.video_paths[camera].stat().st_mtime_ns,'native_row':case['frame']['cameras'][camera]},sort_keys=True).encode()).hexdigest()
                cache=out/'masks'/f"{case['key']}_{camera}.npz"
                existing=evidence/'joint_alignment/masks'/cache.name
                if not cache.exists() and existing.exists():cache=existing
                masks,boxes,confidence=mask_observation(model,image,cache,identity,options)
                result,images,association=evaluate_case(case,camera,image,masks,boxes,confidence,cameras,calibration,faces,final,out,court,corrected_fits)
                if any(f"{athlete['motion_id']}|{case['frame']['lidar_index']}" in retry_fits for athlete in case['frame']['athletes']):
                    trial,extra,_=evaluate_case(case,camera,image,masks,boxes,confidence,
                                               {'baseline':camera_base,'continuity_projection_retry':camera_base},
                                               calibration,faces,final,out,court,retry_fits)
                    retry_rows.extend(v for v in trial if v['variant']=='continuity_projection_retry')
                    images.update({k:v for k,v in extra.items() if k=='continuity_projection_retry'})
                rows+=result;case_reports[case['key']]['images'][camera]=images;case_reports[case['key']]['association'][camera]=association
                print(f"[mask] {case['key']} {camera}: {association['matched']}/{association['baseline_athletes']} matched",flush=True)
    body=json.loads(body_path.read_text(encoding='utf-8'))
    body['joint_camera_mask_evaluation_status']='complete_in_this_report'
    report={'schema':'badminton-court-mask-surface-joint-evaluation-v1','calibration_digest':calibration.data['snapshot_digest'],'active_selection_changed':False,
            'evaluator_sha256':sha(__file__),'silhouette_rule':'union_of_individually_rasterized_projected_triangles',
            'manifest_sha256':sha(manifest_path),'segmentation_model':model_info,'benchmark_sha256':sha(Path(str(a.benchmark_prefix or evidence/'fit_continuity_benchmark.final')+'.json')),
            'inference_and_evaluation_wall_s':time.perf_counter()-started,'native_camera_images':2*len(cases),'cases':[case_reports[c['key']] for c in cases],
            'mask_alignment':aggregate(rows),'court_evaluation':court,'body_evaluation':body,
            'regression_mask_alignment':aggregate([r for r in rows if f"{r['clip_id']}_{r['lidar_index']:06d}" in regression_keys]),
            'optimization_mask_alignment':aggregate([r for r in rows if f"{r['clip_id']}_{r['lidar_index']:06d}" not in regression_keys]) if additional else None,
            'spatial_selection':additional,'corrected_fit_paths':corrected_fits,
            'spatial_fit_evaluation':json.loads(a.corrected_fits.read_text(encoding='utf-8')) if a.corrected_fits else None,
            'camera_refinement_evaluation':json.loads(a.camera_candidates.read_text(encoding='utf-8')) if a.camera_candidates else None,
            'optional_projection_retry':{'report':retry_report,'measurements':retry_rows,'scope':'Isolated source contexts; not pooled with the full evaluation cohort'} if retry_report else None,
            'fixed_rgb_checks':fixed_rgb_checks(evidence,cameras,calibration,final),'measurements':rows,
            'limits':['Court XYZ are normative court-plane priors, not surveyed LiDAR anchors','RGB instance masks and keypoints are model predictions, not anatomical ground truth','Segmentation associations are frozen from original geometry; mask boundaries and occlusions remain uncertain','Cloud points may be visible to LiDAR but occluded from a camera; all points are scored without model-based visibility filtering','Point-to-surface residual is one-sided visible-surface consistency, not hidden-surface or 3D-joint accuracy','These reviewed cases provide regression and development evidence, not a blind test or proof of a global optimum']}
    write_json(out/'joint_alignment.json',report);write_gallery(out,report)
    print(json.dumps({'review':str(out/'index.html'),'mask_alignment':report['mask_alignment'],'native_camera_images':report['native_camera_images']},indent=2))


if __name__=='__main__':main()
