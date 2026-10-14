"""Portable, continuous reviews of recorded LiDAR frames and fitted SMPL.

Rendering is CPU-only. Each display frame is an original scan and the two
independently paired native video frames. Geometry is never interpolated.
"""
from __future__ import annotations

import base64
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import smplx

from .calibration import CalibrationSnapshot
from .pipeline import write_json
from .review import EDGES,fill_triangle_union
from .sessions import SessionIndex


COLORS = [(255, 157, 48), (79, 163, 255), (128, 224, 134), (216, 132, 226)]


def review_selection_digest(manifest):
    """Bind frame selection independently of incremental rendering state."""
    payload = {'calibration_digest':manifest['calibration_digest'],'clips':[]}
    for clip in manifest['clips']:
        frames = [{k:frame[k] for k in ('scan_mid_us','lidar_index','lidar_path','cameras')}
                  for frame in clip['frames']]
        for dest,frame in zip(frames,clip['frames']):
            dest['athletes'] = [{k:a[k] for k in ('motion_id','fit_path','accepted')}
                                for a in frame['athletes']]
        payload['clips'].append({'id':clip['id'],'session_id':clip['session_id'],
                                 'left_us':clip['left_us'],'right_us':clip['right_us'],'frames':frames})
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()


def packed(array, dtype='<f4'):
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode('ascii')


def completed_windows(output_root, calibration):
    """Exclude contained smoke windows and mismatched adopted geometry."""
    grouped = defaultdict(list)
    for path in (output_root/'sessions').glob('*/windows/*/fit_report.json'):
        config = json.loads((path.parent/'window.json').read_text(encoding='utf-8'))
        if config['calibration_digest'] != calibration.data['snapshot_digest']:
            raise ValueError('A review cannot mix calibration versions')
        report = json.loads(path.read_text(encoding='utf-8'))
        grouped[report['session_id']].append((config['right_us']-config['left_us'],config,path.parent))
    result = []
    for sid, candidates in sorted(grouped.items()):
        chosen = []
        for _, config, path in sorted(candidates,key=lambda v:v[0],reverse=True):
            if any(old['left_us']<=config['left_us'] and old['right_us']>=config['right_us'] for old in chosen):
                continue
            if any(max(old['left_us'],config['left_us'])<min(old['right_us'],config['right_us']) for old in chosen):
                raise ValueError('Partially overlapping completed windows are ambiguous')
            chosen.append(config)
            result.append((sid,config,path))
    return sorted(result,key=lambda v:(v[0],v[1]['left_us']))


def choose_windows(windows, count, seconds):
    eligible = [w for w in windows if w[1]['right_us']-w[1]['left_us']>=seconds*1e6]
    if not eligible:
        raise ValueError('No completed fitted window covers the requested duration')
    # Cover every session first, then evenly cover the other completed windows.
    by_session = defaultdict(list)
    for w in eligible:
        by_session[w[0]].append(w)
    selected = [v[len(v)//2] for v in by_session.values()]
    for w in eligible:
        if len(selected)>=count:
            break
        if w not in selected:
            selected.append(w)
    selected = selected[:count]
    allocation = [1]*len(selected)
    while sum(allocation)<count:
        # Extra clips go to windows with room for disjoint intervals.
        choices = [(w[1]['right_us']-w[1]['left_us'])/allocation[i] for i,w in enumerate(selected)]
        idx = int(np.argmax(choices))
        if choices[idx]<seconds*1e6*(allocation[idx]+1)/allocation[idx]:
            raise ValueError('Too little fitted footage for disjoint requested clips')
        allocation[idx]+=1
    return sorted(zip(selected,allocation),key=lambda v:(v[0][0],v[0][1]['left_us']))


def plan_clips(output_root, calibration, count=10, seconds=5):
    clips = []
    for (sid,config,window),amount in choose_windows(completed_windows(output_root,calibration),count,seconds):
        session = SessionIndex.load(Path(calibration.data['sessions_root'])/sid,calibration.frame_map_path(sid))
        by_scan = defaultdict(list)
        for line in (window/'quality.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            by_scan[row['lidar_index']].append(row)
        pairs = session.pairs()
        length = config['right_us']-config['left_us']
        # Midpoints of evenly spaced slots avoid selecting by fitting quality.
        offsets = (np.arange(amount)+.5)*length/amount - seconds*1e6/2
        for offset in offsets:
            requested = int(config['left_us']+offset)
            remaining = [p for p in pairs if requested<=p['scan_mid_us']<config['right_us']]
            if not remaining:
                raise ValueError('No native scan starts this clip')
            start = remaining[0]['scan_mid_us']
            end = start+round(seconds*1e6)
            if end>config['right_us']:
                raise ValueError('Clip extends beyond the completed window')
            frames = []
            for pair in remaining:
                if pair['scan_mid_us']>=end:
                    break
                if any(pair['cameras'][c]['record_index'] is None for c in ('cam1','cam2')):
                    raise ValueError('Both native camera frames are needed for review')
                athletes = [{k:r[k] for k in ('motion_id','fit_path','accepted','qa','rejection_reasons')}
                            for r in by_scan[pair['lidar_index']]]
                frames.append({'scan_mid_us':pair['scan_mid_us'],'offset_s':(pair['scan_mid_us']-start)/1e6,
                               'lidar_index':pair['lidar_index'],'lidar_path':pair['lidar_path'],
                               'scan_start_us':pair['scan_start_us'],'scan_end_us':pair['scan_end_us'],
                               'pair_status':pair['status'],'cameras':pair['cameras'],'athletes':athletes})
            names = sorted({a['motion_id'] for f in frames for a in f['athletes']})
            rows = [a for f in frames for a in f['athletes']]
            clips.append({'id':f'clip{len(clips)+1:02d}','session_id':sid,'source_window':str(window),
                          'left_us':start,'right_us':end,'duration_s':seconds,'frames':frames,
                          'motion_colors':{name:COLORS[i%len(COLORS)] for i,name in enumerate(names)},
                          'summary':{'scans':len(frames),'fitted_person_frames':len(rows),
                                     'accepted':sum(r['accepted'] for r in rows),
                                     'scans_without_fits':sum(not f['athletes'] for f in frames),
                                     'mean_surface_mm':float(np.mean([r['qa']['final_p2surface_mean_m'] for r in rows])*1000) if rows else None,
                                     'mean_frame_p90_mm':float(np.mean([r['qa']['final_p2surface_p90_m'] for r in rows])*1000) if rows else None}})
    return clips


def _draw_overlay(image, frame, calibration, camera, faces, colors):
    """The same full stored mesh is projected to both original Brown images."""
    layer = image.copy()
    geometries = []
    for athlete in frame['athletes']:
        with np.load(athlete['fit_path']) as fit:
            vertices,joints,cloud = fit['vertices'],fit['joints'],fit['source_cloud']
        color = tuple(reversed(colors[athlete['motion_id']]))
        uv,depth = calibration.project(vertices,camera,distort=True)
        valid = (depth[faces]>0).all(1)&np.isfinite(uv[faces]).all((1,2))
        triangles = np.rint(uv[faces[valid]]).astype(np.int32)
        # Front/back faces overlap; fillPoly's even/odd multi-contour rule
        # would cut holes in an otherwise solid projected mesh.
        fill_triangle_union(layer,triangles,color)
        geometries.append((uv,depth,joints,cloud,color))
    image = cv2.addWeighted(layer,.23,image,.77,0)
    for uv,depth,joints,cloud,color in geometries:
        thin = faces[::12]
        thin = thin[(depth[thin]>0).all(1)&np.isfinite(uv[thin]).all((1,2))]
        cv2.polylines(image,list(np.rint(uv[thin]).astype(np.int32)),True,color,1,cv2.LINE_AA)
        cuv,cdepth = calibration.project(cloud,camera,distort=True)
        valid = (cdepth>0)&np.isfinite(cuv).all(1)
        xy = np.rint(cuv[valid]).astype(np.int64)
        xy = xy[(xy[:,0]>=0)&(xy[:,0]<image.shape[1])&(xy[:,1]>=0)&(xy[:,1]<image.shape[0])]
        image[xy[:,1],xy[:,0]] = (0,255,255)
        juv,jdepth = calibration.project(joints,camera,distort=True)
        for a,b in EDGES:
            if min(jdepth[a],jdepth[b])>0:
                cv2.line(image,tuple(np.rint(juv[a]).astype(int)),tuple(np.rint(juv[b]).astype(int)),color,2,cv2.LINE_AA)
    return image


def export_clip(clip, calibration, review_dir, faces, image_width=1280):
    directory = review_dir/clip['id']
    directory.mkdir(parents=True,exist_ok=True)
    sid = clip['session_id']
    session = SessionIndex.load(Path(calibration.data['sessions_root'])/sid,calibration.frame_map_path(sid))
    pairs = session.pairs()
    geometry_frames = []
    for frame in clip['frames']:
        cloud = session.read_cloud(pairs[frame['lidar_index']])
        athletes = []
        for athlete in frame['athletes']:
            with np.load(athlete['fit_path']) as fit:
                athletes.append({'motion_id':athlete['motion_id'],'accepted':athlete['accepted'],
                                 'vertices':packed(fit['vertices']),'joints':packed(fit['joints']),
                                 'cloud':packed(fit['source_cloud']),'qa':athlete['qa'],
                                 'rejection_reasons':athlete['rejection_reasons'],
                                 'color':clip['motion_colors'][athlete['motion_id']]})
        geometry_frames.append({'offset_s':frame['offset_s'],'lidar_index':frame['lidar_index'],
                                'scan_mid_us':frame['scan_mid_us'],'cameras':frame['cameras'],
                                'cloud':packed(cloud),'point_count':len(cloud),'athletes':athletes})
    for camera in ('cam1','cam2'):
        frames_by_index = defaultdict(list)
        for i,frame in enumerate(clip['frames']):
            frames_by_index[frame['cameras'][camera]['record_index']].append(i)
        for index,image in session.iter_video_frames(camera,sorted(frames_by_index)):
            for i in frames_by_index[index]:
                frame = clip['frames'][i]
                annotated = _draw_overlay(image.copy(),frame,calibration,camera,faces,clip['motion_colors'])
                if image_width<annotated.shape[1]:
                    annotated = cv2.resize(annotated,(image_width,round(annotated.shape[0]*image_width/annotated.shape[1])),interpolation=cv2.INTER_AREA)
                cv2.putText(annotated,f"{camera} | scan {frame['lidar_index']} | {frame['offset_s']:.3f}s | yellow: measured LiDAR",
                            (15,27),cv2.FONT_HERSHEY_SIMPLEX,.55,(245,245,245),1,cv2.LINE_AA)
                relative = f"{clip['id']}/{i:03d}_{camera}.jpg"
                if not cv2.imwrite(str(review_dir/relative),annotated,[cv2.IMWRITE_JPEG_QUALITY,85]):
                    raise IOError('Unable to write review image')
                geometry_frames[i][camera+'_image'] = relative
    payload = {'id':clip['id'],'duration_s':clip['duration_s'],'summary':clip['summary'],'frames':geometry_frames}
    path = directory/'data.js'
    path.write_text('window.REVIEW_CLIP_READY('+json.dumps(payload,separators=(',',':'),allow_nan=False)+');\n',encoding='utf-8')
    return path


def write_page(review_dir,manifest,faces):
    assets = Path(__file__).with_name('review_assets')
    report_file = review_dir/'quality_efficiency.json'
    report = json.loads(report_file.read_text(encoding='utf-8')) if report_file.exists() else None
    cam1_file = review_dir/'cam1_check.json'
    cam1 = json.loads(cam1_file.read_text(encoding='utf-8')) if cam1_file.exists() else None
    public = {k:v for k,v in manifest.items() if k!='clips'}
    public['clips'] = [{k:v for k,v in c.items() if k!='frames'} for c in manifest['clips']]
    public.update(faces=packed(faces,'<u2'),quality_report=report,cam1_check=cam1,
                  reference_plane=manifest['reference_plane'],edges=EDGES)
    html = (assets/'index.html').read_text(encoding='utf-8').replace('__MANIFEST__',json.dumps(public,separators=(',',':'),ensure_ascii=False,allow_nan=False))
    (review_dir/'index.html').write_text(html,encoding='utf-8')
    for name in ('review.js','review.css'):
        (review_dir/name).write_bytes((assets/name).read_bytes())


def render_sequences(output_root,review_dir,*,count=10,seconds=5,image_width=1280,html_only=False):
    output_root,review_dir = Path(output_root).resolve(),Path(review_dir).resolve()
    workspace = Path(__file__).resolve().parents[2]
    if not review_dir.is_relative_to(workspace):
        raise ValueError('Generated review must stay in the GVHMR workspace')
    cv2.setNumThreads(1)
    calibration = CalibrationSnapshot.load(output_root/'calibration/calibration_snapshot.json')
    model = smplx.create(workspace/'inputs/checkpoints/body_models',model_type='smpl',gender='neutral')
    faces = np.asarray(model.faces,dtype=np.uint16)
    del model
    review_dir.mkdir(parents=True,exist_ok=True)
    if html_only:
        manifest = json.loads((review_dir/'manifest.json').read_text(encoding='utf-8'))
        manifest['selection_sha256'] = review_selection_digest(manifest)
        write_json(review_dir/'manifest.json',manifest)
    else:
        clips = plan_clips(output_root,calibration,count,seconds)
        manifest = {'schema':'badminton-continuous-review-v1','state':'rendering',
                    'output_root':str(output_root),'calibration_snapshot':str(calibration.snapshot_json),
                    'calibration_digest':calibration.data['snapshot_digest'],'geometry_variant':calibration.data.get('geometry_variant'),
                    'reference_plane':calibration.reference_plane,'clip_count':count,'clip_duration_s':seconds,
                    'display_cadence':'original LiDAR scan timestamps (~10Hz); nearest measured camera frames; no geometry interpolation',
                    'image_display_width':image_width,'clips':clips,'rendered_clips':[]}
        manifest['selection_sha256'] = review_selection_digest(manifest)
        write_json(review_dir/'manifest.json',manifest)
        for clip in clips:
            export_clip(clip,calibration,review_dir,faces,image_width)
            manifest['rendered_clips'].append(clip['id'])
            write_json(review_dir/'manifest.json',manifest)
            print(f"[review] {clip['id']} {clip['session_id']}: {len(clip['frames'])} native scans, {clip['summary']['fitted_person_frames']} person frames",flush=True)
        manifest['state']='complete'
        write_json(review_dir/'manifest.json',manifest)
    write_page(review_dir,manifest,faces)
    return {'path':str(review_dir/'index.html'),'clips':len(manifest['clips']),'state':manifest['state']}
