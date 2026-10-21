"""Explicit external source-index reader for CPU preparation and contract checks.

This adapter does not change default sessions, calibration admission or source
manifests. Candidate indexes may be inspected, but cannot authorize body fitting.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .dual_view import file_sha
from .relocations import require_relocation_root
from .sessions import SessionIndex, _integer, _read_json, _read_jsonl


def _exact_object(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def validate_packet_row_mapping(index, clock_rows, frozen, selected, camera_omitted):
    """Reconcile emitted clouds against original packet rows, without rewriting either."""
    indices = index.get("lidar_common_clock_row_indices")
    packet_omitted = index.get("unexported_packet_clock_rows", [])
    if indices is None:
        if "lidar_common_clock_row_indices" in index or packet_omitted:
            raise ValueError("Packet omissions require explicit cloud-to-original-clock row indices")
        if any(
            "source_common_clock_row_index" in row or "source_common_clock_row" in row
            for row in [*index["frames"], *frozen.values()]
        ):
            raise ValueError("Embedded packet rows require explicit cloud-to-original-clock row indices")
        return [], None
    if (
        not isinstance(indices, list)
        or len(indices) != len(index["frames"])
        or any(type(i) is not int or not 0 <= i < len(clock_rows) for i in indices)
        or any(b <= a for a, b in zip(indices, indices[1:]))
    ):
        raise ValueError("Explicit packet row indices must be ordered unique integers covering actual clouds")
    if not isinstance(packet_omitted, list):
        raise ValueError("Packet omissions must be an explicit list")
    if camera_omitted and (
        index.get("source_kind") == "original_fullraw_fixed24"
        or index.get("parent_source_kind") == "original_fullraw_fixed24"
    ):
        raise ValueError("Fullraw emitted-cloud mapping cannot hide packet rows as camera-coverage omissions")
    for frame, map_index, source_index in zip(index["frames"], selected, indices):
        row = clock_rows[source_index]
        if type(frame["original_frame_id"]) is not int or frame["original_frame_id"] != row["frame_id"]:
            raise ValueError("Mapped cloud original frame ID differs from original packet row")
        for item in (frame, frozen[map_index]):
            if (
                type(item.get("source_common_clock_row_index")) is not int
                or item["source_common_clock_row_index"] != source_index
                or not _exact_object(item.get("source_common_clock_row"), row)
            ):
                raise ValueError("Mapped cloud/frame-map original packet row object differs")
            start, end = row["common_time_us"], row["end_common_time_us"]
            for name, expected in (
                ("device_first_us", start),
                ("device_last_us", end),
                ("scan_mid_us", start + (end - start) // 2),
            ):
                if type(item.get(name)) is not int or item[name] != expected:
                    raise ValueError("Mapped cloud/frame-map packet timing differs")
    covered = set(indices)
    camera_ids = {row["original_frame_id"] for row in camera_omitted}
    camera_indices = {i for i, row in enumerate(clock_rows) if row["frame_id"] in camera_ids}
    if covered & camera_indices:
        raise ValueError("Camera-coverage omissions overlap emitted clouds")
    missing = sorted(set(range(len(clock_rows))) - covered - camera_indices)
    expected = (
        [
            dict(
                source_common_clock_row_index=missing[0],
                source_common_clock_row=clock_rows[missing[0]],
                reason="unclosed_final_scan_not_emitted",
            )
        ]
        if missing == [len(clock_rows) - 1]
        else []
    )
    if (missing and missing != [len(clock_rows) - 1]) or not _exact_object(packet_omitted, expected):
        raise ValueError("Only an exactly documented final unclosed packet row may lack an emitted cloud")
    return packet_omitted, indices


def verify_fullraw_evidence(index, data_root, clock_path, clock, *, relocations=None):
    """Verify export/PTP evidence separately from camera or body admission."""
    direct = index.get("source_kind") == "original_fullraw_fixed24"
    derived = index.get("parent_source_kind") == "original_fullraw_fixed24"
    if not (direct or derived):
        return None
    if "lidar_common_clock_row_indices" not in index:
        raise ValueError("Fullraw sources require explicit original packet row mapping")
    original = index
    manifest_entry = index["source_manifest"]
    if derived:
        if (
            index.get("parent_scope") != "fullraw_exported_clouds"
            or index.get("maps_to_full_pcap_returns") is not False
        ):
            raise ValueError("Derived fullraw members refer to emitted-cloud returns, never full PCAP returns")
        parents = index["parent_inputs"]
        original = _read_json(_bound_file(data_root, parents["source_index"], relocations=relocations))
        manifest_entry = parents["source_manifest"]
        if (
            original.get("source_kind") != "original_fullraw_fixed24"
            or parents["source_index"]["sha256"] != index["parent_source_index_identity"]["sha256"]
            or original["source_manifest"]["sha256"] != manifest_entry["sha256"]
            or original["common_clock"]["sha256"] != index["common_clock"]["sha256"]
            or original["frame_map"]["sha256"] != index["frame_map"]["sha256"]
            or original["frame_count"] != index["frame_count"]
            or len(original["frames"]) != index["frame_count"]
        ):
            raise ValueError("Derived fullraw parent index/manifest/clock/map binding differs")
        for name in ("lidar_common_clock_row_indices", "unexported_packet_clock_rows"):
            if not _exact_object(original.get(name), index.get(name)):
                raise ValueError("Derived fullraw original packet mapping changed")
        for child, parent in zip(index["frames"], original["frames"]):
            expected = dict(
                file=parent["file"],
                sha256=parent["png_sha256"],
                bytes=parent["png_bytes"],
                valid_points=parent["point_count"],
            )
            mapping = child.get("member_mapping", {})
            if (
                child.get("parent_png") != expected
                or mapping.get("schema") != "badminton-native-roi-member-map-v1"
                or mapping.get("parent_valid_points") != parent["point_count"]
                or mapping.get("child_valid_points") != child["point_count"]
                or not index.get("crop_binding_sha256")
                or child.get("crop_binding_sha256") != index["crop_binding_sha256"]
            ):
                raise ValueError("Derived fullraw parent PNG/member-map lineage differs")
            _bound_file(data_root, mapping, verify=False, relocations=relocations)
    manifest = _read_json(_bound_file(data_root, manifest_entry, relocations=relocations))
    if manifest.get("kind") != "full" or manifest.get("frame_count") != index["frame_count"]:
        raise ValueError("Fullraw proof must refer to the complete emitted-cloud manifest")
    proof_path = _bound_file(data_root, index["fullraw_decode_evidence"], relocations=relocations)
    proof = _read_json(proof_path)
    if (
        proof.get("status") not in ("completed", "reconstructed_from_completion_log")
        or type(proof.get("exit_code")) is not int
        or proof["exit_code"] != 0
        or proof.get("complete_pcap_read") is not True
        or proof.get("frame_count") != index["frame_count"]
        or proof.get("source_manifest_sha256") != manifest_entry["sha256"]
        or not _exact_object(
            proof.get("unexported_packet_clock_rows", []), index.get("unexported_packet_clock_rows", [])
        )
    ):
        raise ValueError("Fullraw completion evidence does not bind the emitted clouds and original packet omission")
    if proof["status"] == "reconstructed_from_completion_log":
        if (
            not proof.get("exit_code_source")
            or "frame_limit" not in proof
            or proof["frame_limit"] is not None
            or not proof.get("completion_logs")
        ):
            raise ValueError("Reconstructed fullraw completion requires explicit exit/no-limit/log evidence")
        for log in proof["completion_logs"]:
            _bound_file(data_root, log, relocations=relocations)
    ptp_path = _bound_file(data_root, index["original_ptp_validation"], relocations=relocations)
    ptp = _read_json(ptp_path)
    if (
        ptp.get("passed") is not True
        or ptp.get("evidence_incomplete")
        or ptp.get("common_index_sha256") != file_sha(clock_path)
    ):
        raise ValueError("Fullraw original PTP evidence does not bind the complete original clock")
    raw_indexes = []
    # Frozen crops retain the original clock bytes, including raw_index paths
    # relative to its original directory. Resolve that directory through the
    # already-bound parent index, never through the relocated clock snapshot.
    raw_clock_path = (_bound_file(data_root, original["common_clock"], relocations=relocations)
                      if derived else clock_path)
    for track in clock["tracks"]:
        if relocations is None:
            raw = (raw_clock_path.parent / track["raw_index"]).resolve()
            entry = dict(file=str(raw.relative_to(data_root)), sha256=track["raw_index_sha256"])
        else:
            # The original clock bytes name raw indexes relative to the old
            # logical location, even when this clock itself has moved.
            entry = relocations.relative_binding(
                original["common_clock"]["file"], track["raw_index"], track["raw_index_sha256"])
        _bound_file(data_root, entry, relocations=relocations)
        raw_indexes.append(entry)
    return dict(
        parent_scope="fullraw_exported_clouds",
        maps_to_full_pcap_returns=False,
        decode_evidence=index["fullraw_decode_evidence"],
        original_ptp_validation=index["original_ptp_validation"],
        original_manifest=manifest_entry,
        original_clock_for_raw_indices=(original["common_clock"] if derived else index["common_clock"]),
        raw_timestamp_indices=raw_indexes,
        derived=derived,
    )
def _bound_file(data_root, entry, verify=True, *, relocations=None):
    require_relocation_root(data_root, relocations)
    path=(relocations.resolve(entry) if relocations is not None else (data_root/entry['file']).resolve())
    if not path.is_relative_to(data_root):
        raise ValueError('External source path escapes the explicit data root')
    if not path.is_file():raise FileNotFoundError(path)
    if entry.get('bytes') is not None and path.stat().st_size!=entry['bytes']:
        raise ValueError(f'External source size mismatch: {path}')
    # Mapped resolution has already checked both the immutable reference and
    # actual target SHA/bytes. Do not hash large native videos twice.
    if verify and (relocations is None or not relocations.has_mapping(entry['file'])) and file_sha(path)!=entry['sha256']:
        raise ValueError(f'External source SHA256 mismatch: {path}')
    return path


def validate_coordinate_contract(index):
    grid=index['fixed24'];contract=index['coordinate_contract']
    fullraw=(index.get('source_kind')=='original_fullraw_fixed24'
             or index.get('parent_source_kind')=='original_fullraw_fixed24')
    # Original full exports declare raw storage and M, rather than the legacy
    # consumer-policy aliases. The single application is enforced below/in
    # read_cloud; explicit conflicting aliases are still rejected.
    applications=contract.get('consumer_application_count_required',1 if fullraw else None)
    applied=contract.get('application_M_applied_by_this_index_builder',False if fullraw else None)
    if (grid['coordinate_encoding']!='fixed24' or grid['stored_xyz_space']!='lidar_sensor_xyz'
            or grid['scale']!=1000 or grid['center']!=256
            or contract['M_baked_into_stored_xyz'] is not False
            or type(applications) is not int or applications!=1 or applied is not False):
        raise ValueError('Expected raw fixed24 XYZ and exactly one consumer application of explicit M')
    w,h=grid['point_width'],grid['point_height']
    size=grid.get('image_size',[grid.get('image_width'),grid.get('image_height')] if fullraw else None)
    if (isinstance(w,bool) or isinstance(h,bool) or not isinstance(w,int) or not isinstance(h,int)
            or min(w,h)<=0 or size!=[w,3*h]
            or ('image_width' in grid and grid['image_width']!=w)
            or ('image_height' in grid and grid['image_height']!=3*h)):
        raise ValueError('External fixed24 grid/image dimensions disagree')
    matrix=np.asarray(contract['application_raw_to_viewer_M'],dtype=float)
    if (matrix.shape!=(3,3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix@matrix.T,np.eye(3),atol=2e-4) or np.linalg.det(matrix)<=0):
        raise ValueError('External application M must be finite and preserve handedness')
    return matrix


def require_camera_applicability(index):
    # Body-label admission happens after fitting/QA. Requiring it to admit a
    # source would incorrectly make new, unfitted recordings impossible to use.
    if (index.get('camera_calibration_admitted') is not True
            or index.get('coordinate_contract',{}).get('physical_camera_applicability')!='sampled_visual_accepted'):
        raise ValueError('External source calibration applicability is pending; preparation only, no fitting admission')


def video_sidecar_contract(index):
    generic=index.get('video_sidecars');canonical=index.get('canonical_video_sidecars')
    if generic is not None:
        contract=index.get('video_index_contract',{})
        if (contract.get('kind')!='dataset_only_device_clock' or contract.get('canonical_receive_admitted') is not False):
            raise ValueError('Dataset-only sidecars require explicit device-clock scope and no canonical receive admission')
        if (contract.get('device_clock_status')!='verified_common_ptp_no_loss'
                or not isinstance(contract.get('device_clock_evidence'),dict)):
            raise ValueError('Dataset-only sidecars cannot admit missing, unknown or lost device-clock synchronization')
        if canonical is not None and generic!=canonical:
            raise ValueError('Conflicting generic and canonical video sidecar bindings')
        return 'video_sidecars',generic
    if canonical is None:raise ValueError('External source lacks native video index sidecars')
    return 'canonical_video_sidecars',canonical


def verify_camera_applicability(index,data_root,*,relocations=None):
    require_camera_applicability(index)
    entry=index.get('camera_applicability')
    if not isinstance(entry,dict):raise ValueError('Admitted external camera source needs an applicability certificate')
    path=_bound_file(data_root,entry,relocations=relocations);certificate=_read_json(path)
    if (certificate.get('schema')!='badminton-camera-applicability-v1'
            or certificate.get('accepted_calibration_sha256')!=index['accepted_calibration_sha256']
            or certificate.get('accepted_bundle_sha256')!=index['accepted_bundle_sha256']
            or not np.array_equal(certificate.get('application_raw_to_viewer_M'),index['coordinate_contract']['application_raw_to_viewer_M'])):
        raise ValueError('Camera applicability certificate calibration/M binding mismatch')
    source=certificate.get('sources',{}).get(index['source_id'])
    if not source or source.get('status')!='sampled_visual_accepted' or source.get('camera_calibration_admitted') is not True:
        raise ValueError('Camera applicability certificate does not admit this recording')
    sidecar_key,_=video_sidecar_contract(index)
    for name in ('source_manifest','common_clock','frame_map','native_videos',sidecar_key):
        if source.get(name)!=index[name]:raise ValueError(f'Camera applicability certificate source binding mismatch: {name}')
    if index.get('video_index_contract') is not None and source.get('video_index_contract')!=index['video_index_contract']:
        raise ValueError('Camera applicability certificate does not bind dataset-only video indexing policy')
    evidence=source.get('review_evidence')
    if not isinstance(evidence,list) or not evidence:
        raise ValueError('Camera applicability certificate lacks native review evidence')
    for review in evidence:_bound_file(data_root,review,relocations=relocations)
    return dict(file=str(path),sha256=entry['sha256'],status=source['status'],
                review_limits=source.get('review_limits'),body_labels_production_admitted=source.get('body_labels_production_admitted'))


class SourceIndexedSession(SessionIndex):
    @classmethod
    def load_source_index(cls,index_path,data_root,*,expected_sha256=None,require_admitted=True,relocations=None):
        data_root=Path(data_root).resolve()
        require_relocation_root(data_root,relocations)
        logical_index_path=Path(index_path).absolute()
        if relocations is not None and logical_index_path.is_relative_to(data_root):
            index_path=relocations.resolve(dict(file=logical_index_path.relative_to(data_root).as_posix(),
                                               sha256=expected_sha256))
        else:
            index_path=Path(index_path).resolve()
        digest=file_sha(index_path)
        if expected_sha256 and digest!=expected_sha256:
            raise ValueError('External source-index SHA256 mismatch')
        index=_read_json(index_path)
        if index['schema'] not in ('badminton-source-index-candidate-v1','badminton-source-index-v1'):
            raise ValueError('Unsupported external source-index schema')
        if require_admitted:require_camera_applicability(index)
        self=cls();self.source_index=index;self.source_index_path=index_path;self.source_index_sha256=digest
        self.data_relocations=relocations;self.source_index_logical_path=logical_index_path
        self.camera_applicability=(verify_camera_applicability(index,data_root,relocations=relocations)
                                   if require_admitted or index.get('camera_calibration_admitted') is True else None)
        self.session_id=index['source_id'];self.session_dir=data_root/'sessions'/self.session_id
        self.data_root=data_root
        self.Q=validate_coordinate_contract(index)
        self.lidar_dir=data_root
        _bound_file(data_root,index['source_manifest'],relocations=relocations)
        clock_path=_bound_file(data_root,index['common_clock'],relocations=relocations)
        frame_map_path=_bound_file(data_root,index['frame_map'],relocations=relocations)
        self.video_paths={name:_bound_file(data_root,index['native_videos'][name],relocations=relocations)
                          for name in ('cam1','cam2')}
        _,sidecar_bindings=video_sidecar_contract(index)
        if 'video_sidecars' in index:
            _bound_file(data_root,index['video_index_contract']['device_clock_evidence'],relocations=relocations)
        sidecars={name:_bound_file(data_root,sidecar_bindings[name],relocations=relocations) for name in self.video_paths}
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
            cloud_entry=dict(file=frame['file'],bytes=frame['png_bytes'])
            if relocations is not None:cloud_entry['sha256']=frame['png_sha256']
            cloud_path=_bound_file(data_root,cloud_entry,verify=False,relocations=relocations)
            cloud_file=(cloud_path.relative_to(data_root).as_posix() if relocations is not None else frame['file'])
            normalized.append(dict(frame_id=frame['original_frame_id'],file=cloud_file,kept_points=frame['point_count']))
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
        self.original_lidar_clock_rows=self._tracks.get('lidar1',{}).get('rows',[])
        previous_start=None
        for row in self.original_lidar_clock_rows:
            frame_id=_integer(row.get('frame_id'),'external clock frame_id')
            if frame_id is None or frame_id in self._lidar_clock:raise ValueError('Missing/duplicate external clock frame_id')
            if 'lidar_common_clock_row_indices' in index:
                start=_integer(row.get('common_time_us'),'original packet start')
                last=_integer(row.get('end_common_time_us'),'original packet end')
                if start is None or last is None or last<start or (previous_start is not None and start<=previous_start):
                    raise ValueError('Original packet clock rows are missing, reversed or non-monotonic')
                previous_start=start
            self._lidar_clock[frame_id]=row
        if 'raw_clock_rows' in index and _integer(index['raw_clock_rows'],'raw_clock_rows')!=len(self.original_lidar_clock_rows):
            raise ValueError('Declared original packet clock row count differs')
        map_rows=_read_jsonl(frame_map_path)
        self._frozen={_integer(row['lidar_frame'],'external original map index'):row for row in map_rows}
        if (len(self._frozen)!=len(map_rows) or len(map_rows)!=index['frame_map']['rows']
                or None in self._frozen):
            raise ValueError('External original frame map contains duplicate/missing indexes or wrong row count')
        selected=[_integer(frame.get('frame_map_lidar_frame',i),'external PNG-to-map index') for i,frame in enumerate(frames)]
        if len(set(selected))!=len(selected) or any(i not in self._frozen for i in selected):
            raise ValueError('External PNG ordinals do not map uniquely to original clock/map rows')
        omitted=index.get('nonexported_original_clock_rows',[])
        declared={row['lidar_frame'] for row in omitted}
        if len(declared)!=len(omitted) or set(self._frozen)-set(selected)!=declared:
            raise ValueError('Missing source PNGs need explicit original frame-map omission evidence')
        begin=self.common_clock['common_overlap_begin_us'];end=self.common_clock['common_overlap_end_us']
        for row in omitted:
            clock=self._lidar_clock[row['original_frame_id']]
            start=clock['common_time_us'];last=clock['end_common_time_us'];mid=start+(last-start)//2
            if (row.get('exported_png_present') is not False or row['device_first_us']!=start
                    or row['device_last_us']!=last or row['scan_mid_us']!=mid or begin<=mid<=end):
                raise ValueError('An omitted source frame is inside usable common coverage or has invalid packet timing')
            frozen=self._frozen[row['lidar_frame']]
            if any(frozen.get(k)!=row[k] for k in ('device_first_us','device_last_us','scan_mid_us')):
                raise ValueError('Omitted original frame-map timing differs from clock evidence')
            for camera in self.video_paths:
                nearest=self._frozen_camera(camera,row['lidar_frame'],mid)
                if nearest['status']!='outside_video_time_range' or row['cameras'][camera]!=frozen['cameras'][camera]:
                    raise ValueError('An omitted source frame could affect paired camera coverage')
        packet_omitted,mapping=validate_packet_row_mapping(index,self.original_lidar_clock_rows,self._frozen,selected,omitted)
        self.fullraw_evidence=verify_fullraw_evidence(index,data_root,clock_path,self.common_clock,relocations=relocations)
        self.member_mapping_verifications={}
        if packet_omitted and self.fullraw_evidence is None:
            raise ValueError('Unexported packet rows require the explicit fullraw completion/PTP contract')
        exported_ids={frame['original_frame_id'] for frame in frames}
        omitted_ids={row['original_frame_id'] for row in omitted}
        packet_ids={row['source_common_clock_row']['frame_id'] for row in packet_omitted}
        if omitted_ids & packet_ids or set(self._lidar_clock)-exported_ids!=omitted_ids|packet_ids:
            raise ValueError('Original device clock has unexplained missing source frames')
        self.lidar_source_row_mapping=(dict(original_rows=len(self.original_lidar_clock_rows),mapped_rows=len(frames),
             lidar_common_clock_row_indices=mapping,unexported_packet_clock_rows=packet_omitted,
             nonexported_camera_coverage_rows=len(omitted)) if mapping is not None else None)
        self._lidar_clock={key:value for key,value in self._lidar_clock.items() if key in exported_ids}
        self._frozen={i:self._frozen[original] for i,original in enumerate(selected)}
        self._pairs=self._build_pairs()
        for frame,pair in zip(frames,self._pairs):
            pair['source_frame_map_lidar_frame']=frame.get('frame_map_lidar_frame',frame['frame_index'])
            if mapping is not None:
                source_index=mapping[frame['frame_index']]
                pair['source_common_clock_row_index']=source_index
                pair['source_common_clock_row']=self.original_lidar_clock_rows[source_index]
            for camera,expected in frame['cameras'].items():
                actual=pair['cameras'][camera]
                for name in ('status','record_index','media_pts_us','time_error_us'):
                    if actual[name]!=expected[name]:raise ValueError('External source index and independently checked pairing disagree')
                if actual['common_time_us']!=expected['video_common_us']:
                    raise ValueError('External source-index camera common time disagrees')
        return self

    def read_cloud(self,pair,viewer=True):
        frame=self.source_index['frames'][pair['lidar_index']]
        if self.data_relocations is not None:
            actual=_bound_file(self.data_root,dict(file=frame['file'],sha256=frame['png_sha256'],bytes=frame['png_bytes']),
                               relocations=self.data_relocations)
            if Path(pair['lidar_path']).resolve()!=actual:
                raise ValueError('Relocated cloud pair no longer names its bound input')
        elif file_sha(pair['lidar_path'])!=frame['png_sha256']:
            raise ValueError('External original PNG SHA256 mismatch')
        points=super().read_cloud(pair,viewer)
        if self.fullraw_evidence and self.fullraw_evidence['derived']:
            from .source_membership import verify_fullraw_members
            self.member_mapping_verifications[pair['lidar_index']]=verify_fullraw_members(
                frame,pair['lidar_path'],self.data_root,self.manifest,_bound_file,relocations=self.data_relocations)
        return points


def prepare_source_index(index_path,data_root,expected_sha256=None,sample_count=6,*,relocations=None):
    started=time.perf_counter()
    session=SourceIndexedSession.load_source_index(index_path,data_root,expected_sha256=expected_sha256,
                                                   require_admitted=False,relocations=relocations)
    pairs=session.pairs();indices=np.linspace(0,len(pairs)-1,min(sample_count,len(pairs)),dtype=int)
    samples=[]
    for i in indices:
        pair=pairs[i];raw=session.read_cloud(pair,viewer=False);viewer=(raw@session.Q.T).astype(np.float32)
        recovered=viewer@np.linalg.inv(session.Q).T
        samples.append(dict(lidar_index=int(i),points=len(raw),raw_to_viewer_applications=1,
                            inverse_roundtrip_max_m=float(np.max(np.abs(recovered-raw))) if len(raw) else 0.))
    report=dict(schema='badminton-external-source-prepare-v1',source_id=session.session_id,
                source_index=str(session.source_index_path),source_index_sha256=session.source_index_sha256,
                frames=len(pairs),paired=sum(p['status']=='paired' for p in pairs),samples=samples,
                original_clock_rows=len(session.original_lidar_clock_rows),
                nonexported_original_clock_rows=session.source_index.get('nonexported_original_clock_rows',[]),
                lidar_source_row_mapping=session.lidar_source_row_mapping,
                unexported_packet_clock_rows=session.source_index.get('unexported_packet_clock_rows',[]),
                fullraw_evidence=session.fullraw_evidence,
                sampled_member_mapping_verifications=session.member_mapping_verifications,
                preparation_seconds=time.perf_counter()-started,
                calibration_applicability=session.source_index['coordinate_contract'].get('physical_camera_applicability','pending'),
                camera_applicability_certificate=session.camera_applicability,
                video_index_contract=session.source_index.get('video_index_contract'),
                camera_calibration_admitted=session.source_index['camera_calibration_admitted'],
                production_admitted=session.source_index['production_admitted'],
                body_fitting_started=False,source_files_modified=False)
    if relocations is not None:
        report['data_relocations']=dict(relocations.binding)
        report['source_index_logical_path']=str(session.source_index_logical_path)
    return report
