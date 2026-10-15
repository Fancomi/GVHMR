"""Explicit external source-index reader for CPU preparation and contract checks.

This adapter does not change default sessions, calibration admission or source
manifests. Candidate indexes may be inspected, but cannot authorize body fitting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dual_view import file_sha
from .sessions import SessionIndex, _integer, _read_json


def _bound_file(data_root, entry, verify=True):
    path=(data_root/entry['file']).resolve()
    if not path.is_relative_to(data_root):
        raise ValueError('External source path escapes the explicit data root')
    if not path.is_file():raise FileNotFoundError(path)
    if entry.get('bytes') is not None and path.stat().st_size!=entry['bytes']:
        raise ValueError(f'External source size mismatch: {path}')
    if verify and file_sha(path)!=entry['sha256']:
        raise ValueError(f'External source SHA256 mismatch: {path}')
    return path


def validate_coordinate_contract(index):
    grid=index['fixed24'];contract=index['coordinate_contract']
    if (grid['coordinate_encoding']!='fixed24' or grid['stored_xyz_space']!='lidar_sensor_xyz'
            or grid['scale']!=1000 or grid['center']!=256
            or contract['M_baked_into_stored_xyz'] is not False
            or contract['consumer_application_count_required']!=1
            or contract['application_M_applied_by_this_index_builder'] is not False):
        raise ValueError('Expected raw fixed24 XYZ and exactly one consumer application of explicit M')
    w,h=grid['point_width'],grid['point_height']
    if (isinstance(w,bool) or isinstance(h,bool) or not isinstance(w,int) or not isinstance(h,int)
            or min(w,h)<=0 or grid['image_size']!=[w,3*h]):
        raise ValueError('External fixed24 grid/image dimensions disagree')
    matrix=np.asarray(contract['application_raw_to_viewer_M'],dtype=float)
    if (matrix.shape!=(3,3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix@matrix.T,np.eye(3),atol=2e-4) or np.linalg.det(matrix)<=0):
        raise ValueError('External application M must be finite and preserve handedness')
    return matrix


class SourceIndexedSession(SessionIndex):
    @classmethod
    def load_source_index(cls,index_path,data_root,*,expected_sha256=None,require_admitted=True):
        index_path=Path(index_path).resolve();data_root=Path(data_root).resolve()
        digest=file_sha(index_path)
        if expected_sha256 and digest!=expected_sha256:
            raise ValueError('External source-index SHA256 mismatch')
        index=_read_json(index_path)
        if index['schema'] not in ('badminton-source-index-candidate-v1','badminton-source-index-v1'):
            raise ValueError('Unsupported external source-index schema')
        if require_admitted and not (index.get('camera_calibration_admitted') is True and index.get('production_admitted') is True):
            raise ValueError('External source calibration applicability is pending; preparation only, no fitting admission')
        self=cls();self.source_index=index;self.source_index_path=index_path;self.source_index_sha256=digest
        self.session_id=index['source_id'];self.session_dir=data_root/'sessions'/self.session_id
        self.Q=validate_coordinate_contract(index)
        self.lidar_dir=data_root
        _bound_file(data_root,index['source_manifest'])
        clock_path=_bound_file(data_root,index['common_clock'])
        frame_map_path=_bound_file(data_root,index['frame_map'])
        self.video_paths={name:_bound_file(data_root,index['native_videos'][name]) for name in ('cam1','cam2')}
        sidecars={name:_bound_file(data_root,index['canonical_video_sidecars'][name]) for name in self.video_paths}
        self.video_dir=sidecars['cam1'].parent
        if any(path!=self.video_dir/f'{name}.frames.jsonl' for name,path in sidecars.items()):
            raise ValueError('External native sidecars must retain canonical camera names in one directory')
        frames=index['frames']
        if len(frames)!=_integer(index['frame_count'],'external frame_count'):
            raise ValueError('External frame list does not cover frame_count')
        normalized=[];captures=[]
        for i,frame in enumerate(frames):
            if _integer(frame['frame_index'],'external frame_index')!=i:
                raise ValueError('External PNG ordinal must be contiguous without synthetic frames')
            _integer(frame['original_frame_id'],'external original_frame_id')
            start=_integer(frame['device_first_us'],'external device_first_us')
            end=_integer(frame['device_last_us'],'external device_last_us')
            if start is None or end is None or end<start or frame['scan_mid_us']!=start+(end-start)//2:
                raise ValueError('External LiDAR midpoint is not bound to exact device packet times')
            _bound_file(data_root,dict(file=frame['file'],bytes=frame['png_bytes']),verify=False)
            normalized.append(dict(frame_id=frame['original_frame_id'],file=frame['file'],kept_points=frame['point_count']))
            captures.append(dict(device_first_packet_us=start,device_last_packet_us=end))
        self.manifest=dict(index['fixed24'],coordinate_space='lidar_sensor_xyz',lidar_rotation=self.Q.tolist(),
                           frame_count=len(frames),frames=normalized,frame_capture_timestamps=captures)
        self.meta_dir=clock_path.parent;self.common_clock=_read_json(clock_path)
        if self.common_clock.get('schema')!='external_gm_common_timeline_v1' or self.common_clock.get('session_id')!=self.session_id:
            raise ValueError('External common-clock schema/session mismatch')
        self._tracks={}
        for track in self.common_clock['tracks']:
            name=track['device_id']
            if name in self._tracks:raise ValueError('Duplicate external clock track')
            self._tracks[name]=track
        self._camera_rows={name:self._load_camera(name) for name in self.video_paths}
        self._camera_known={};self._camera_times={}
        for name,rows in self._camera_rows.items():
            known=[r for r in rows if r['common_time_us'] is not None]
            times=[r['common_time_us'] for r in known]
            if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Non-monotonic external camera clock')
            self._camera_known[name]=known;self._camera_times[name]=times
        self._lidar_clock={}
        for row in self._tracks.get('lidar1',{}).get('rows',[]):
            frame_id=_integer(row.get('frame_id'),'external clock frame_id')
            if frame_id is None or frame_id in self._lidar_clock:raise ValueError('Missing/duplicate external clock frame_id')
            self._lidar_clock[frame_id]=row
        self._frozen=self._load_frozen_map(frame_map_path)
        self._pairs=self._build_pairs()
        for frame,pair in zip(frames,self._pairs):
            for camera,expected in frame['cameras'].items():
                actual=pair['cameras'][camera]
                for name in ('status','record_index','media_pts_us','time_error_us'):
                    if actual[name]!=expected[name]:raise ValueError('External source index and independently checked pairing disagree')
                if actual['common_time_us']!=expected['video_common_us']:
                    raise ValueError('External source-index camera common time disagrees')
        return self

    def read_cloud(self,pair,viewer=True):
        frame=self.source_index['frames'][pair['lidar_index']]
        if file_sha(pair['lidar_path'])!=frame['png_sha256']:
            raise ValueError('External original PNG SHA256 mismatch')
        return super().read_cloud(pair,viewer)


def prepare_source_index(index_path,data_root,expected_sha256=None,sample_count=6):
    session=SourceIndexedSession.load_source_index(index_path,data_root,expected_sha256=expected_sha256,require_admitted=False)
    pairs=session.pairs();indices=np.linspace(0,len(pairs)-1,min(sample_count,len(pairs)),dtype=int)
    samples=[]
    for i in indices:
        pair=pairs[i];raw=session.read_cloud(pair,viewer=False);viewer=(raw@session.Q.T).astype(np.float32)
        recovered=viewer@np.linalg.inv(session.Q).T
        samples.append(dict(lidar_index=int(i),points=len(raw),raw_to_viewer_applications=1,
                            inverse_roundtrip_max_m=float(np.max(np.abs(recovered-raw))) if len(raw) else 0.))
    return dict(schema='badminton-external-source-prepare-v1',source_id=session.session_id,
                source_index=str(session.source_index_path),source_index_sha256=session.source_index_sha256,
                frames=len(pairs),paired=sum(p['status']=='paired' for p in pairs),samples=samples,
                calibration_applicability=session.source_index['coordinate_contract']['physical_camera_applicability'],
                camera_calibration_admitted=session.source_index['camera_calibration_admitted'],
                production_admitted=session.source_index['production_admitted'],
                body_fitting_started=False,source_files_modified=False)
