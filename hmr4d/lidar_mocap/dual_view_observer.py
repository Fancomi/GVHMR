"""Resumable native RGB observations for every existing accepted person-frame.

One shared detector/pose model observes both cameras. Streaming native decoding
checks actual PTS; all fitted athletes (including rejected labels) participate
in association and foreign-person visibility. No body or camera is optimized.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from hmr4d import PROJ_ROOT
from .calibration import CalibrationSnapshot
from .dual_view import _bound_output, _canonical, _read, file_sha
from .rgb_observations import RGBKeypointChecker, add_raw_pixel_observations, write_json
from .rally_pipeline import window_directory, load_run_session


def source_selection(source):
    dataset_path = source/'datasets/dataset_manifest.json'
    dataset = _read(dataset_path)
    snapshot=source/'calibration/calibration_snapshot.json'
    if dataset.get('review_only') or (snapshot.exists() and _read(snapshot).get('review_only')):
        raise ValueError('Review-only pilot cannot be used as a training export source')
    sources = dict(dataset_manifest=dict(path=str(dataset_path), sha256=file_sha(dataset_path)),
                   calibration=file_sha(source/'calibration/calibration_snapshot.json'), files=[])
    if dataset.get('source_registry'):
        path=Path(dataset['source_registry'])
        if file_sha(path)!=dataset['source_registry_sha256']:raise ValueError('Source registry changed')
        sources['files'].append(dict(path=str(path),sha256=file_sha(path)))
    for sid, entry in dataset['exports'].items():
        for name in ('path', 'metadata'):
            path = Path(entry[name])
            sources['files'].append(dict(path=str(path), sha256=file_sha(path)))
        for window in dataset['selected_windows'][sid]:
            path = window_directory(source,dataset,sid,window)/'quality.jsonl'
            sources['files'].append(dict(path=str(path), sha256=file_sha(path)))
            teacher_directory=path.parent/'teacher'
            for name in ('manifest.json','detections.pt'):
                teacher_path=teacher_directory/name
                sources['files'].append(dict(path=str(teacher_path),sha256=file_sha(teacher_path)))
    return dataset, sources


def selected_frames(source, dataset, sid):
    accepted = {}
    with Path(dataset['exports'][sid]['metadata']).open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            key = (int(row['lidar_index']), row['motion_id'])
            if key in accepted:
                raise ValueError('Duplicate source accepted person-frame')
            accepted[key] = row
    frames = {}
    found = set()
    for window in dataset['selected_windows'][sid]:
        path = window_directory(source,dataset,sid,window)/'quality.jsonl'
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                index = int(row['lidar_index'])
                frame = frames.setdefault(index, dict(lidar_index=index, scan_mid_us=row['scan_mid_us'],
                                                       cameras=row['cameras'], athletes=[]))
                if frame['scan_mid_us'] != row['scan_mid_us'] or frame['cameras'] != row['cameras']:
                    raise ValueError('Source fitted people disagree about native frame/time')
                key = (index, row['motion_id'])
                if key in accepted:
                    if row != accepted[key]:
                        raise ValueError('Source window and exported person metadata differ')
                    found.add(key)
                frame['athletes'].append(dict(motion_id=row['motion_id'], fit_path=row['fit_path'],
                                               accepted=key in accepted, qa=row['qa']))
    if found != set(accepted):
        raise ValueError('Selected fit windows do not cover every accepted source label')
    return [frame for _,frame in sorted(frames.items()) if any(p['accepted'] for p in frame['athletes'])]


def _resume_rows(path, expected, camera):
    """Recover only an interrupted final write; reject changed/duplicate records."""
    done, frame_count, person_count, seconds = set(), 0, 0, 0.
    if not path.exists():
        return done, frame_count, person_count, seconds
    with path.open('r+b') as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                handle.truncate(start)
                break
            row = json.loads(line)
            index = row['lidar_index']
            frame = expected.get(index)
            if index in done or frame is None or row['camera_name'] != camera:
                raise ValueError('Unexpected or duplicated observation in resume journal')
            if (row['scan_mid_us'] != frame['scan_mid_us'] or row['paired_cameras'] != frame['cameras']
                    or {p['motion_id'] for p in row['athletes']} != {p['motion_id'] for p in frame['athletes']}):
                raise ValueError('Observation journal source changed')
            done.add(index)
            frame_count += 1
            person_count += sum(p['fit_accepted'] for p in row['athletes'])
            seconds += row['timing_seconds']['total']
    return done, frame_count, person_count, seconds


def _cached_frames(paths, calibration_digest):
    cache = {}
    for path in paths:
        path = Path(path).resolve()
        report = _read(path)
        report_sha=file_sha(path)
        if report['calibration_digest'] != calibration_digest:
            raise ValueError('Cached detections use a different frozen calibration')
        if 'frames_jsonl' in report:
            rows_path = Path(report['frames_jsonl'])
            rows_sha = file_sha(rows_path)
            def rows():
                with rows_path.open('rb') as handle:
                    for line in handle:
                        if not line.endswith(b'\n'):break
                        yield json.loads(line)
            frames = rows()
        else:
            rows_path, rows_sha, frames = path, file_sha(path), report['frames']
        for frame in frames:
            key=(frame['session_id'],frame['camera_name'],frame['lidar_index'])
            frame['reused_native_evidence']=dict(report_path=str(path),report_sha256=report_sha,
                                                   frames_path=str(rows_path),frames_sha256=rows_sha,
                                                   models=report['models'],checker_sha256=report['checker_sha256'])
            cache.setdefault(key,frame)
    return cache


def _cache_matches(record, frame):
    if (record['scan_mid_us'] != frame['scan_mid_us'] or record['paired_cameras'] != frame['cameras']
            or {p['motion_id'] for p in record['athletes']} != {p['motion_id'] for p in frame['athletes']}):
        return False
    expected={p['motion_id']:p for p in frame['athletes']}
    for person in record['athletes']:
        source=expected[person['motion_id']]
        if (Path(person['fit_path']).resolve() != Path(source['fit_path']).resolve()
                or person['fit_sha256'] != file_sha(source['fit_path'])):
            return False
        person['fit_accepted']=source['accepted']
    return True


def teacher_detections(source, dataset, sid, camera, calibration, session):
    """Reuse only exact native-frame detections; never treat interpolation as presence."""
    import torch
    result={}
    for window in dataset['selected_windows'][sid]:
        directory=window_directory(source,dataset,sid,window)
        config=_read(directory/'window.json')
        if config['settings']['camera'] != camera:
            continue
        manifest_path=directory/'teacher/manifest.json'
        manifest=_read(manifest_path)
        provenance=manifest['provenance']
        expected_hash=hashlib.sha256(_canonical(provenance)).hexdigest()
        if (expected_hash!=manifest['provenance_hash'] or config['calibration_digest']!=calibration.data['snapshot_digest']
                or provenance['camera']!=calibration.camera(camera)):
            raise ValueError('Native teacher detections have mismatched calibration/provenance')
        video=session.video_paths[camera];stat=video.stat()
        if provenance['video']!=dict(path=str(video.resolve()),size=stat.st_size,mtime_ns=stat.st_mtime_ns):
            raise ValueError('Native teacher video identity changed')
        path=directory/'teacher/detections.pt'
        cache=torch.load(path,map_location='cpu')
        if cache['provenance_hash']!=expected_hash or cache['schema']!=manifest['schema']:
            raise ValueError('Teacher detection cache is not bound to its provenance manifest')
        rows=provenance['frame_rows']
        if len(rows)!=len(cache['detections']):
            raise ValueError('Teacher detections and native frame rows differ in length')
        evidence=dict(cache_path=str(path.resolve()),cache_sha256=file_sha(path),
                      manifest_path=str(manifest_path.resolve()),manifest_sha256=file_sha(manifest_path),
                      provenance_hash=expected_hash)
        for row,detections in zip(rows,cache['detections']):
            native=session._camera_rows[camera][row['record_index']]
            if any(row[k]!=native[k] for k in ('record_index','common_time_us','media_pts_us','source_frame_no')):
                raise ValueError('Teacher detector native index/PTS/common time does not match frozen frame map')
            result.setdefault(row['record_index'],dict(detections=detections,evidence=evidence))
    return result


def observe_all(source_root, output_root, device='cuda:0', pose_batch_size=4,
                detector_imgsz=960, threads=8, max_new_frames=None, cache_reports=()):
    source, output = _bound_output(source_root, output_root)
    dataset, sources = source_selection(source)
    calibration = CalibrationSnapshot.load(source/'calibration/calibration_snapshot.json')
    if dataset['calibration_digest'] != calibration.data['snapshot_digest']:
        raise ValueError('Source labels and frozen calibration differ')
    settings = dict(device=device, pose_batch_size=pose_batch_size, detector_imgsz=detector_imgsz,
                    threads=threads, flip_test=False, observation_scope='person_detection_and_association_only')
    cache_sources=[]
    for path in cache_reports:
        report=_read(path)
        item=dict(path=str(Path(path).resolve()),sha256=file_sha(path))
        if 'frames_jsonl' in report:
            item.update(frames_jsonl=report['frames_jsonl'],frames_jsonl_sha256=file_sha(report['frames_jsonl']))
        cache_sources.append(item)
    identity = dict(source=sources, settings=settings,
                    checker_sha256=file_sha(PROJ_ROOT/'hmr4d/lidar_mocap/rgb_observations.py'),
                    observer_sha256=file_sha(__file__), calibration_digest=dataset['calibration_digest'],cache_sources=cache_sources)
    identity_sha = hashlib.sha256(_canonical(identity)).hexdigest()
    observer_path = output/'observations/manifest.json'
    if observer_path.exists() and _read(observer_path)['identity_sha256'] != identity_sha:
        raise ValueError('Resume inputs/settings/checker changed; use a new output folder')
    started = time.perf_counter()
    checker = RGBKeypointChecker(calibration, threads, detector_imgsz, False, device=device,
                                 pose_batch_size=pose_batch_size,observe_pose=False)
    cache=_cached_frames(cache_reports,dataset['calibration_digest'])
    state = dict(schema='badminton-full-native-rgb-observations-v1', completed=False,
                 created_utc=datetime.now(timezone.utc).isoformat(), identity=identity,
                 identity_sha256=identity_sha, models=checker.models, reports=[],
                 source_accepted=dataset['accepted'], model_loading_seconds=time.perf_counter()-started)
    observer_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(observer_path, state)
    new_frames = 0
    for sid in dataset['exports']:
        checker.sessions[sid]=load_run_session(source,calibration,sid)
        frames = selected_frames(source, dataset, sid)
        frame_lookup = {f['lidar_index']:f for f in frames}
        for camera in ('cam1', 'cam2'):
            checker.select_camera(camera)
            report_path = output/'observations'/f'{sid}_{camera}.json'
            rows_path = report_path.with_suffix('.jsonl')
            done, completed_frames, completed_people, measured_seconds = _resume_rows(rows_path, frame_lookup, camera)
            session = checker.session(sid)
            teacher_cache=teacher_detections(source,dataset,sid,camera,calibration,session)
            video = session.video_paths[camera]
            video_stat = video.stat()
            video_identity = dict(path=str(video), size=video_stat.st_size, mtime_ns=video_stat.st_mtime_ns)
            report = dict(schema='badminton-native-rgb-journal-v1', completed=False,
                          calibration_digest=dataset['calibration_digest'], models=checker.models,
                          checker_sha256=identity['checker_sha256'], identity_sha256=identity_sha,
                          session_id=sid, camera_name=camera, camera=checker.camera,
                          frames_jsonl=str(rows_path), source_video=video_identity,
                          planned_camera_frames=len(frames), planned_accepted_person_frames=dataset['exports'][sid]['accepted'],
                          completed_camera_frames=completed_frames, observed_accepted_person_frames=completed_people,
                          inference_seconds=measured_seconds, device=device)
            report['exact_native_teacher_cache_frames_available']=sum(f['cameras'][camera]['record_index'] in teacher_cache for f in frames)
            if report_path.exists() and _read(report_path)['source_video'] != video_identity:
                raise ValueError('Native source video changed since observing began')
            state['reports'].append(str(report_path))
            reused=0
            with rows_path.open('a',encoding='utf-8') as handle:
                for frame in frames:
                    key=(sid,camera,frame['lidar_index'])
                    if frame['lidar_index'] in done or key not in cache:
                        continue
                    record=cache[key]
                    if not _cache_matches(record,frame):
                        continue
                    handle.write(json.dumps(record,allow_nan=False)+'\n')
                    done.add(frame['lidar_index']);reused+=1
                    report['completed_camera_frames']+=1
                    report['observed_accepted_person_frames']+=sum(p['fit_accepted'] for p in record['athletes'])
            report['cached_camera_frames_imported_this_run']=reused
            grouped = defaultdict(list)
            for frame in frames:
                if frame['lidar_index'] not in done:
                    grouped[frame['cameras'][camera]['record_index']].append(frame)
            stop = max_new_frames is not None and new_frames >= max_new_frames
            if not stop:
                with rows_path.open('a', encoding='utf-8') as handle:
                    stream = session.iter_video_frames(camera, sorted(grouped))
                    try:
                        for native_index, bgr in stream:
                            for frame in grouped[native_index]:
                                clip = dict(id='full_source_session', session_id=sid, output_root=str(source))
                                cached=teacher_cache.get(native_index)
                                record = checker.check_frame(clip, frame, raw_bgr=bgr,
                                    cached_detections=cached['detections'] if cached else None)
                                if cached:record['teacher_detection_evidence']=cached['evidence']
                                add_raw_pixel_observations(dict(camera=checker.camera, frames=[record]))
                                handle.write(json.dumps(record, allow_nan=False)+'\n')
                                handle.flush()
                                new_frames += 1
                                report['completed_camera_frames'] += 1
                                report['observed_accepted_person_frames'] += sum(p['fit_accepted'] for p in record['athletes'])
                                report['inference_seconds'] += record['timing_seconds']['total']
                                if new_frames % 25 == 0 or new_frames <= 3:
                                    write_json(report_path, report)
                                    write_json(observer_path, state)
                                    print(json.dumps(dict(session=sid,camera=camera,
                                        completed_camera_frames=report['completed_camera_frames'],
                                        planned_camera_frames=len(frames), new_frames=new_frames,
                                        wall_seconds=time.perf_counter()-started,
                                        frame_seconds=record['timing_seconds'])), flush=True)
                                if max_new_frames is not None and new_frames >= max_new_frames:
                                    stop = True
                                    break
                            if stop:
                                break
                    finally:
                        stream.close()
            report['completed'] = report['completed_camera_frames'] == len(frames)
            if report['completed']:
                report['frames_jsonl_sha256'] = file_sha(rows_path)
            write_json(report_path, report)
            write_json(observer_path, state)
            if stop:
                state.update(last_run_seconds=time.perf_counter()-started, last_run_new_frames=new_frames)
                write_json(observer_path, state)
                return state
    state.update(completed=True, last_run_seconds=time.perf_counter()-started, last_run_new_frames=new_frames)
    write_json(observer_path, state)
    return state
