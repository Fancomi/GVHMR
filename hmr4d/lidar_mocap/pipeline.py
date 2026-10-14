"""Resumable calibrated badminton video/LiDAR -> SMPL training pairs."""
from __future__ import annotations
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from .calibration import CalibrationSnapshot
from .sessions import SessionIndex
from .segmentation import court_polygon, farthest_sample, segment_people
from .export import acceptance_reasons, export_joblib


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1<<20), b''):
            digest.update(block)
    return digest.hexdigest()


class BadmintonPipeline:
    def __init__(self, sessions_root, lidar_project_root, output_root, *, camera='cam2',
                 teacher_stride=2, batch_size=4, fit_batch_size=16, iterations=80,
                 window_seconds=60, min_points=80):
        self.sessions_root = Path(sessions_root).resolve()
        self.lidar_project_root = Path(lidar_project_root).resolve()
        self.output_root = Path(output_root).resolve()
        workspace = Path(__file__).resolve().parents[2]
        if not self.output_root.is_relative_to(workspace):
            raise ValueError('Outputs must remain in the GVHMR workspace')
        self.settings = dict(camera=camera,teacher_stride=teacher_stride,batch_size=batch_size,
                             fit_batch_size=fit_batch_size,iterations=iterations,
                             window_seconds=window_seconds,min_points=min_points)
        self.calibration = None

    def _load(self):
        if self.calibration is None:
            self.calibration = CalibrationSnapshot.load(self.output_root/'calibration/calibration_snapshot.json')
        return self.calibration

    def _session(self, sid):
        c = self._load()
        path = self.sessions_root/sid
        expected = c.data['sessions'][sid]['manifest_sha256']
        manifest_path = path/'output-lidar'/f'lidar_{sid}'/'manifest.json'
        if _hash_file(manifest_path) != expected:
            raise ValueError(f'Source manifest changed after calibration freeze: {sid}')
        return SessionIndex.load(path,c.frame_map_path(sid))

    def prepare(self):
        self.calibration = CalibrationSnapshot.freeze(self.lidar_project_root,self.sessions_root,
                                                       self.output_root/'calibration')
        summary = {'schema':'badminton-smpl-pair-run-v1','prepared_utc':datetime.now(timezone.utc).isoformat(),
                   'sessions_root':str(self.sessions_root),'output_root':str(self.output_root),
                   'calibration_snapshot':str(self.calibration.snapshot_json),
                   'calibration_digest':self.calibration.data['snapshot_digest'],'settings':self.settings,
                   'sessions':{},'physical_limits':self.calibration.data['physical_limits']}
        for sid in sorted(self.calibration.data['sessions']):
            session = self._session(sid)
            pairs = session.pairs()
            directory = self.output_root/'sessions'/sid
            directory.mkdir(parents=True,exist_ok=True)
            pairs_path = directory/'pairs.jsonl'
            pairs_path.write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in pairs),encoding='utf-8')
            evidence_paths = [session.meta_dir/f'{sid}_common_clock.json']
            evidence_paths += [session.video_dir/f'{cam}.frames.jsonl' for cam in ('cam1','cam2')]
            source_evidence = {str(p):_hash_file(p) for p in evidence_paths}
            stats = {'source_scan_count':len(pairs),'paired_nonempty_scan_count':sum(p['status']=='paired' for p in pairs),
                     'exclusion_reasons':dict(Counter(r for p in pairs for r in p['reasons'])),
                     'pairs':str(pairs_path),'source_evidence_sha256':source_evidence,
                     'video_files':{cam:{'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
                                    for cam,p in session.video_paths.items()},
                     'teacher_roi_polygon':court_polygon(self.calibration,self.settings['camera'])}
            write_json(directory/'pairing_report.json',stats)
            summary['sessions'][sid] = stats
        summary['paired_nonempty_scan_count'] = sum(v['paired_nonempty_scan_count'] for v in summary['sessions'].values())
        write_json(self.output_root/'pairing_report.json',summary)
        return summary

    def _windows(self, session, start_seconds=0, duration_seconds=None):
        pairs = [p for p in session.pairs() if p['status']=='paired']
        if not pairs:
            return
        base = min(p['scan_mid_us'] for p in pairs)
        lower = base + round(start_seconds*1e6)
        upper = max(p['scan_mid_us'] for p in pairs)+1
        if duration_seconds is not None:
            upper = min(upper,lower+round(duration_seconds*1e6))
        window_us = round(self.settings['window_seconds']*1e6)
        if window_us <= 0:
            raise ValueError('window_seconds must be positive')
        for left in range(lower,upper,window_us):
            right = min(left+window_us,upper)
            selected = [p for p in pairs if left <= p['scan_mid_us'] < right]
            if selected:
                yield f'{left-base:012d}_{right-base:012d}', left,right,selected

    def run(self, session_ids=None, *, start_seconds=0, duration_seconds=None, stage='run', resume=True):
        from .teacher import run_teacher
        self._load()
        sessions = session_ids or sorted(self.calibration.data['sessions'])
        results = []
        fitter = None
        for sid in sessions:
            session = self._session(sid)
            for window_id,left,right,pairs in self._windows(session,start_seconds,duration_seconds):
                output = self.output_root/'sessions'/sid/'windows'/window_id
                output.mkdir(parents=True,exist_ok=True)
                camera = self.settings['camera']
                rows = session.teacher_rows(camera,self.settings['teacher_stride'],left-1000000,right+1000000)
                config = {'schema':'badminton-window-v1','session_id':sid,'window_id':window_id,
                          'left_us':left,'right_us':right,'calibration_digest':self.calibration.data['snapshot_digest'],
                          'settings':self.settings,'source_pairs':[p['lidar_index'] for p in pairs]}
                config_path = output/'window.json'
                if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
                    raise ValueError(f'Window settings differ; use a separate output directory: {output}')
                write_json(config_path,config)
                report_path = output/'fit_report.json'
                if resume and stage=='run' and report_path.exists():
                    results.append(json.loads(report_path.read_text(encoding='utf-8')))
                    print(f'[resume] {sid}/{window_id}',flush=True)
                    continue
                print(f'[{stage}] {sid}/{window_id}: {len(pairs)} scans, {len(rows)} source video frames',flush=True)
                if fitter is not None:
                    # Teacher stages need the GPU capacity released by fitting.
                    del fitter
                    fitter = None
                    import torch
                    torch.cuda.empty_cache()
                if stage=='fit':
                    teacher_manifest = json.loads((output/'teacher/manifest.json').read_text(encoding='utf-8'))
                else:
                    teacher_manifest = run_teacher(session.video_paths[camera],rows,self.calibration.camera(camera),
                                                   output/'teacher',batch_size=self.settings['batch_size'],resume=resume,
                                                   roi_polygon=court_polygon(self.calibration,camera))
                if stage=='teacher':
                    results.append(teacher_manifest)
                    continue
                from .fitting import SMPLFitter
                fitter = SMPLFitter(device='cuda')
                result = self._fit_window(session,pairs,teacher_manifest,output,fitter)
                results.append(result)
                write_json(self.output_root/'progress.json',{'updated_utc':datetime.now(timezone.utc).isoformat(),
                           'last_session':sid,'last_window':window_id,'completed_in_this_call':len(results),
                           'accepted_in_this_call':sum(r.get('accepted',0) for r in results)})
        report = {'stage':stage,'windows':results,'accepted':sum(r.get('accepted',0) for r in results)}
        write_json(self.output_root/'last_run_report.json',report)
        return report

    def _fit_window(self, session, pairs, teacher_manifest, output, fitter):
        import torch
        from .teacher import load_teacher_track,sample_track
        times = [p['scan_mid_us'] for p in pairs]
        tracks = []
        for entry in teacher_manifest['tracks']:
            path = Path(entry['path'])
            if not path.is_absolute():
                path = output/'teacher'/path
            track = load_teacher_track(path)
            sampled = sample_track(track,times)
            tracks.append((entry,track,sampled))
        by_track = defaultdict(list)
        exclusions = []
        for i,pair in enumerate(pairs):
            visible = []
            for k,(entry,track,sampled) in enumerate(tracks):
                if bool(sampled['valid'][i]):
                    visible.append({'track_id':str(k),'bbox':np.asarray(sampled['bbx_xyxy'][i]),
                                    'kp2d':np.asarray(sampled['kp2d'][i])})
            cloud = session.read_cloud(pair)
            segments = segment_people(cloud,visible,self.calibration,self.settings['camera'],
                                      min_points=self.settings['min_points'])
            for k in visible:
                index = int(k['track_id'])
                segment = segments[k['track_id']]
                if segment['status']=='segmented':
                    by_track[index].append((i,pair,segment))
                else:
                    exclusions.append({'lidar_index':pair['lidar_index'],'track':index,'reason':segment['status']})
        accepted_samples = []
        reviews = []
        for k,frames in by_track.items():
            entry,track,sampled = tracks[k]
            track_name = f"{session.session_id}:{output.name}:{entry['track_id']}:{entry['segment_id']}"
            previous = None
            for start in range(0,len(frames),self.settings['fit_batch_size']):
                batch = frames[start:start+self.settings['fit_batch_size']]
                indices = [f[0] for f in batch]
                params = {name:value[indices] for name,value in sampled['smpl_params_incam'].items()}
                clouds = [f[2]['points'] for f in batch]
                result = fitter.fit_batch(params,clouds,self.calibration,camera=self.settings['camera'],
                                          keypoints=sampled['kp2d'][indices],previous=previous,
                                          iterations=self.settings['iterations'])
                if previous is None and len(batch)>=8 and sum(not acceptance_reasons(q,len(clouds[j])) for j,q in enumerate(result['qa'])) >= 4:
                    previous = {'shape':result['shape'][0],'shape_fixed':True}
                for j,(_,pair,segment) in enumerate(batch):
                    qa = result['qa'][j]
                    reasons = acceptance_reasons(qa,len(segment['points']))
                    metadata = {'session_id':session.session_id,'motion_id':track_name,
                                'lidar_index':pair['lidar_index'],'lidar_frame_id':pair['lidar_frame_id'],
                                'lidar_path':pair['lidar_path'],'scan_mid_us':pair['scan_mid_us'],
                                'scan_start_us':pair['scan_start_us'],'scan_end_us':pair['scan_end_us'],
                                'cameras':pair['cameras'],'clock_quality':pair['clock_quality'],
                                'calibration_digest':self.calibration.data['snapshot_digest'],
                                'coordinate_space':'lidar_viewer_xyz_m','qa':qa,
                                'source_point_count':len(segment['points']),'accepted':not reasons,
                                'rejection_reasons':reasons,'teacher':str(entry['path']),
                                'teacher_sampling':{name:(value[indices[j]].item() if isinstance(value,np.ndarray) else value)
                                                    for name,value in sampled['metadata'].items()}}
                    reviews.append(metadata)
                    sample = {name:np.asarray(result[name][j],dtype=np.float32)
                              for name in ('root_pose','body_pose','shape','transl')}
                    sample['pc'],sample_indices = farthest_sample(segment['points'])
                    metadata['source_segment_indices'] = segment['indices'][sample_indices].tolist()
                    sample.update(motion_id=track_name,metadata=metadata)
                    # Retain fitted parameters for review even when rejected.
                    frame_path = output/'fits'/f"{pair['lidar_index']:06d}_t{k:03d}.npz"
                    frame_path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(frame_path,**{n:sample[n] for n in ('pc','root_pose','body_pose','shape','transl')},
                                        vertices=np.asarray(result['vertices'][j]),joints=np.asarray(result['joints'][j]),
                                        source_indices=segment['indices'],source_cloud=segment['points'])
                    metadata['fit_path'] = str(frame_path)
                    if not reasons:
                        accepted_samples.append(sample)
                print(f'[fit] {track_name} {start+len(batch)}/{len(frames)}; accepted={len(accepted_samples)}',flush=True)
        export = export_joblib(accepted_samples,output/'ptv3.pth')
        (output/'quality.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in reviews),encoding='utf-8')
        report = {'session_id':session.session_id,'window':output.name,'scan_count':len(pairs),
                  'teacher_tracks':len(tracks),'fitted':len(reviews),'accepted':len(accepted_samples),
                  'fit_rejection_reasons':dict(Counter(r for q in reviews for r in q['rejection_reasons'])),
                  'segmentation_exclusions':exclusions,'export':export}
        write_json(output/'fit_report.json',report)
        return report
