"""Verify sealed native rally windows without fitting or altering recordings."""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

from .dual_view import file_sha
from .sessions import SessionIndex, _integer, _read_json
from .source_index import SourceIndexedSession, _bound_file


def union_intervals(intervals):
    merged=[]
    for begin,end in sorted(intervals):
        _integer(begin,'rally common begin');_integer(end,'rally common end')
        if begin is None or end is None:raise ValueError('Missing rally common interval')
        if end<begin:raise ValueError('Reversed rally common interval')
        if merged and begin<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([begin,end])
    return merged


def camera_bound(row):
    return dict(status=row['status'],record_index=row['record_index'],media_pts_us=row['media_pts_us'],
                video_common_us=row['common_time_us'],time_error_us=row['time_error_us'])


def verify_rally_range(clip,session):
    begin=_integer(clip['source_begin_frame'],'rally PNG begin')
    count=_integer(clip['source_frame_count'],'rally PNG count')
    _integer(clip['common_begin_us'],'rally common begin');_integer(clip['common_end_us'],'rally common end')
    pairs=session.pairs()
    if begin is None or count is None or begin<0 or count<1 or begin+count>len(pairs):
        raise ValueError('Rally parent PNG frame range is invalid')
    selected=pairs[begin:begin+count]
    first,last=selected[0],selected[-1]
    clock=session.common_clock
    expected_begin=max(first['scan_start_us'],clock['common_overlap_begin_us'])
    expected_end=min(last['scan_start_us'],clock['common_overlap_end_us'])
    if (expected_end<expected_begin or clip['common_begin_us']!=expected_begin
            or clip['common_end_us']!=expected_end):
        raise ValueError('Rally accounting interval disagrees with original device frame boundaries')
    both=[p for p in selected if all(p['cameras'][cam]['status']=='matched' for cam in ('cam1','cam2'))]
    if len(both)!=clip['both_camera_matched_frames']:
        raise ValueError('Rally camera-match count disagrees with source frame rows')
    if not both:raise ValueError('Rally has no jointly matched native camera bounds')
    for camera in ('cam1','cam2'):
        if clip['camera_bounds'][camera]!=dict(first=camera_bound(both[0]['cameras'][camera]),last=camera_bound(both[-1]['cameras'][camera])):
            raise ValueError('Rally camera bounds disagree with original native index/PTS/common time')
    return dict(source_id=clip['source_id'],review_index=clip['index'],source_rally_index=clip['index'],original_rally_index=clip['original_rally_index'],
                source_begin_frame=begin,source_frame_count=count,
                source_map_indices=[p.get('source_frame_map_lidar_frame',p['lidar_index']) for p in selected],
                source_png_indices=[p['lidar_index'] for p in selected],
                source_original_frame_ids=[p['lidar_frame_id'] for p in selected],
                eligible_paired_png_indices=[p['lidar_index'] for p in selected if p['status']=='paired'],
                accounting_common_begin_us=expected_begin,accounting_common_end_us=expected_end,
                first_scan_mid_us=first['scan_mid_us'],last_scan_mid_us=last['scan_mid_us'],
                frame_selection_policy='Exact original parent PNG range; accounting endpoints are not scan-midpoint fitting filters',
                bilateral_person_detection_qualified=False,body_fitting_started=False)


def verify_inventory(acquisition_path,data_root,expected_sha256=None):
    acquisition_path=Path(acquisition_path).resolve();data_root=Path(data_root).resolve()
    digest=file_sha(acquisition_path)
    if expected_sha256 and digest!=expected_sha256:raise ValueError('Acquisition manifest SHA256 mismatch')
    acquisition=_read_json(acquisition_path)
    if acquisition.get('schema')!='badminton-upstream-acquisition-manifest-v1':
        raise ValueError('Unsupported acquisition manifest schema')
    for name in ('source_pool','analysis_processing_config','processing_proof','frame_map_proof','native_roi_import_proof'):
        _bound_file(data_root,acquisition[name])
    inventory_path=_bound_file(data_root,acquisition['effective_rally_inventory'])
    inventory=_read_json(inventory_path)
    if inventory.get('schema')!='native-rally-source-pool-v1' or inventory.get('sport')!='badminton':
        raise ValueError('Unsupported native rally inventory')
    if inventory.get('qualified_training_minutes') is not None:
        raise ValueError('Rally windows cannot certify body-label training minutes')
    certificate_path=_bound_file(data_root,acquisition['camera_applicability_certificate'])
    certificate=_read_json(certificate_path)
    payload_hashes={name:entry['sha256'] for name,entry in acquisition['accepted_processing_payloads'].items()}
    for entry in acquisition['accepted_processing_payloads'].values():_bound_file(data_root,entry)
    registry=_read_json(_bound_file(data_root,acquisition['accepted_processing_payloads']['sources.json']))
    package=Path(registry['input_package'])
    if not package.parts or package.parts[0]!=acquisition['sport']:
        raise ValueError('Baseline source registry has an unexpected logical data package')
    baseline_maps={entry['source_id']:entry for entry in registry['frame_maps']}
    if certificate['accepted_bundle_sha256']!=payload_hashes:
        raise ValueError('Acquisition processing payloads differ from the camera applicability certificate')
    proof_path=_bound_file(data_root,acquisition['native_rally_analysis_proof'])
    proof=_read_json(proof_path)
    if any(proof['rally_inventory'].get(k)!=acquisition['effective_rally_inventory'].get(k) for k in ('file','sha256','bytes')):
        raise ValueError('Native analysis proof is bound to a different rally inventory')
    snapshots={r['file']:r for r in proof['all_sources_report_snapshots']}
    sources={s['source_id']:s for s in inventory['sources']}
    if len(sources)!=len(inventory['sources']):raise ValueError('Duplicate rally source')
    new_sources={s['source_id']:s for s in acquisition['sources']}
    if set(new_sources)!={sid for sid,s in sources.items() if not s['baseline']}:
        raise ValueError('Acquisition and rally inventory have different new sources')
    clips=inventory['uniform_review_order'];seen=set();grouped=defaultdict(list);review_positions={}
    for position,clip in enumerate(clips,1):
        _integer(clip['index'],'rally review index');_integer(clip['original_rally_index'],'original rally index')
        if (clip['index']<1 or clip.get('kind')!='badminton_rally'
                or clip.get('time_basis')!='source_timestamp'
                or clip.get('bilateral_person_detection_not_yet_qualified') is not True):
            raise ValueError('Rally order/type/source-time/admission declaration is invalid')
        key=clip['source_id'],clip['original_rally_index']
        if key in seen or key[0] not in sources:raise ValueError('Duplicate/unknown native rally')
        seen.add(key);grouped[key[0]].append(clip);review_positions[key]=position
    prepared=[];results={};new_window_us=0;total_window_us=0
    for sid,source in sources.items():
        snapshot_entry=source['rally_analysis_snapshot']
        if snapshots.get(snapshot_entry['file'])!=snapshot_entry or snapshot_entry['sha256']!=source['rally_analysis_sha256']:
            raise ValueError('Rally source and immutable analysis proof differ')
        analysis=_read_json(_bound_file(data_root,snapshot_entry))
        native={r.get('original_index',r['index']):r for r in analysis['badminton_report']['rallies']
                if r.get('assessment',{}).get('counted') is not False}
        if len(native)!=source['rally_count'] or len(grouped[sid])!=len(native):
            raise ValueError('Inventory dropped or duplicated a counted native rally')
        if source['baseline']:
            parent=(data_root/source['parent_session']).resolve()
            _bound_file(data_root,dict(file=str((parent/'output-lidar'/f'lidar_{sid}'/'manifest.json').relative_to(data_root)),sha256=source['source_manifest_sha256']))
            _bound_file(data_root,dict(file=str((parent/'meta'/sid/f'{sid}_common_clock.json').relative_to(data_root)),sha256=source['common_clock_sha256']))
            entry=baseline_maps[sid]
            if entry['sha256']!=source['frame_map_sha256']:
                raise ValueError('Baseline rally frame-map SHA256 mismatch')
            relative=Path(*package.parts[1:])/entry['path']
            map_path=_bound_file(data_root,dict(file=str(relative),sha256=entry['sha256']))
            session=SessionIndex.load(parent,map_path)
        else:
            if source['source_index']!=new_sources[sid]['source_index']:
                raise ValueError('Acquisition and rally source-index bindings differ')
            index_path=_bound_file(data_root,source['source_index'])
            session=SourceIndexedSession.load_source_index(index_path,data_root,expected_sha256=source['source_index']['sha256'])
            for name in ('source_manifest','common_clock','frame_map'):
                if session.source_index[name]['sha256']!=source[name+'_sha256']:
                    raise ValueError('Rally source index does not bind the declared source files')
        if analysis['frame_count']!=len(session.pairs()):raise ValueError('Native analysis parent frame count changed')
        intervals=[];rally_rows=[]
        for clip in grouped[sid]:
            rally=native.get(clip['original_rally_index'])
            if (rally is None or clip['index']!=rally['index'] or clip['source_begin_frame']!=rally['start_frame']
                    or clip['source_frame_count']!=rally['end_frame']-rally['start_frame']+1
                    or not math.isclose(clip['source_start_seconds'],rally['start_time_seconds'],rel_tol=0,abs_tol=1e-7)):
                raise ValueError('Rally range does not reproduce its counted immutable native analysis')
            row=verify_rally_range(clip,session)
            row['review_index']=review_positions[sid,clip['original_rally_index']]
            prepared.append(row);rally_rows.append(row)
            intervals.append([row['accounting_common_begin_us'],row['accounting_common_end_us']])
        union=union_intervals(intervals)
        window_us=sum(end-begin for begin,end in union)
        if union!=source['common_window_union_us'] or not math.isclose(window_us/1e6,source['effective_window_seconds'],rel_tol=0,abs_tol=1e-6):
            raise ValueError('Native rally duration counted overlapping frames more than once')
        total_window_us+=window_us
        if not source['baseline']:new_window_us+=window_us
        unique_png={i for r in rally_rows for i in r['source_png_indices']}
        paired_png={i for r in rally_rows for i in r['eligible_paired_png_indices']}
        results[sid]=dict(baseline=source['baseline'],rallies=len(rally_rows),effective_window_seconds=window_us/1e6,
                          selected_png_occurrences=sum(r['source_frame_count'] for r in rally_rows),
                          unique_selected_png_frames=len(unique_png),paired_unique_png_frames=len(paired_png),
                          unique_png_ranges_half_open=union_intervals([[i,i+1] for i in unique_png]),
                          unique_paired_png_ranges_half_open=union_intervals([[i,i+1] for i in paired_png]))
    summary=inventory['summary']
    if (summary['rally_count']!=len(clips) or summary['completed_sources']!=len(sources) or inventory['pending']
            or not math.isclose(total_window_us/60e6,summary['effective_window_minutes'],rel_tol=0,abs_tol=1e-9)
            or not math.isclose(new_window_us/60e6,summary['new_window_minutes'],rel_tol=0,abs_tol=1e-9)):
        raise ValueError('Rally summary counts/durations disagree with native source ranges')
    prepared.sort(key=lambda r:r['review_index'])
    if file_sha(acquisition_path)!=digest or file_sha(inventory_path)!=acquisition['effective_rally_inventory']['sha256']:
        raise ValueError('Sealed acquisition or rally inventory changed during verification')
    return dict(schema='badminton-rally-intake-verification-v1',acquisition_manifest=str(acquisition_path),
                acquisition_manifest_sha256=digest,rally_inventory=str(inventory_path),rally_inventory_sha256=file_sha(inventory_path),
                sources=results,rallies=len(prepared),effective_window_minutes=total_window_us/60e6,
                new_window_minutes=new_window_us/60e6,qualified_training_minutes=None,
                body_fitting_started=False,source_files_modified=False),prepared
