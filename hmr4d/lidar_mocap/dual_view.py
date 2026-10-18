"""Auditable dual-camera person-presence subsets of existing SMPL/LiDAR labels.

Timestamp matching alone never qualifies a person. The default rule requires a
corresponding native detection in both cameras. Optional whole-body diagnostics
and review do not add gates to this rule. Source tensors/fits are not optimized.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import html
import json
from pathlib import Path
import sqlite3

import cv2
import joblib
import numpy as np
import torch

from hmr4d import PROJ_ROOT
from .calibration import CalibrationSnapshot
from .export import FIELDS, acceptance_reasons, export_joblib
from .pipeline import write_json
from .sessions import SessionIndex


@dataclass(frozen=True)
class DualViewGate:
    policy: str = 'bilateral_detection'
    detection_confidence: float = .35
    detection_minimum_bbox_iou: float = .1
    detection_minimum_association_margin: float = .08
    body_confidence: float = .5
    minimum_bbox_iou: float = .5
    minimum_association_margin: float = .08
    boundary_margin_px: int = 5
    mean_reprojection_px: float = 20.
    p90_reprojection_px: float = 40.


def detection_reasons(person, gate=None):
    """Current training rule: this person must have a native detection in each view."""
    gate = gate or DualViewGate()
    box = np.asarray(person.get('independent_detection_bbox_xyxy'), dtype=float)
    detection_index=person.get('detector_index')
    confidence=person.get('independent_detection_confidence',0)
    iou,margin=person.get('association_bbox_iou',0),person.get('association_margin',0)
    if (isinstance(detection_index,bool) or not isinstance(detection_index,int) or detection_index<0 or box.shape != (4,)
            or not np.isfinite(box).all() or not (box[2:] > box[:2]).all()
            or confidence is None or not np.isfinite(confidence) or confidence < gate.detection_confidence):
        return ['missing_corresponding_native_person_detection']
    if (iou is None or margin is None or not np.isfinite([iou,margin]).all()
            or iou < gate.detection_minimum_bbox_iou or margin < gate.detection_minimum_association_margin):
        return ['no_unique_same_person_detection_association']
    return []


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _key(session_id, row):
    return session_id, int(row['lidar_index']), row['motion_id']


def view_reasons(person, uv, depth, raw_uv, size, gate):
    """Operational whole-body completeness, not a claim of anatomical truth."""
    reasons = []
    observations = np.asarray(person.get('all_coco17_observations'), dtype=float)
    projected = np.asarray(person.get('all_coco17_fixed_fit_projection'), dtype=float)
    eligible = np.asarray(person.get('eligible_mask'), dtype=bool)
    if (observations.shape != (17, 3) or projected.shape != (17, 2) or eligible.shape != (17,)
            or not np.isfinite(observations).all() or not np.isfinite(projected).all()):
        return ['missing_or_nonfinite_native_body_observations'], {}
    margin = gate.boundary_margin_px
    low, high = np.array([margin, margin]), np.asarray(size) - margin - 1
    box = np.asarray(person.get('independent_detection_bbox_xyxy'), dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or not (box[:2] >= low).all() or not (box[2:] <= high).all():
        reasons.append('independent_person_bbox_truncated')
    if (not np.isfinite(uv).all() or not np.isfinite(raw_uv).all() or not np.isfinite(depth).all()
            or not (depth > 0).all() or not (uv >= low).all() or not (uv <= high).all()
            or not (raw_uv >= low).all() or not (raw_uv <= high).all()):
        reasons.append('native_smpl_whole_body_outside_image')
    if person.get('association_bbox_iou', 0) < gate.minimum_bbox_iou:
        reasons.append('insufficient_independent_bbox_overlap')
    if person.get('association_margin', 0) < gate.minimum_association_margin:
        reasons.append('ambiguous_same_person_association')
    # COCO 5..16 are both shoulders/elbows/wrists/hips/knees/ankles. Six high
    # confidence points alone cannot establish an entirely observed body.
    observed_body = observations[5:]
    body_in_view = ((observed_body[:, :2] >= low).all(1) & (observed_body[:, :2] <= high).all(1))
    if not (eligible[5:].all() and body_in_view.all() and (observed_body[:, 2] >= gate.body_confidence).all()):
        reasons.append('incomplete_torso_or_limb_observations')
    if person.get('other_athlete_occluded_joint_count') != 0:
        reasons.append('foreign_person_occlusion_or_missing_visibility_evidence')
    errors = np.linalg.norm(projected[5:] - observed_body[:, :2], axis=1)
    mean, p90 = float(errors.mean()), float(np.quantile(errors, .9))
    if mean > gate.mean_reprojection_px or p90 > gate.p90_reprojection_px:
        reasons.append('bilateral_body_reprojection_outside_limit')
    if person.get('status') != 'measured':
        reasons.append('independent_observation_not_measured')
    return reasons, dict(body_joint_indices=list(range(5, 17)), body_joint_count=12,
                         eligible_body_joint_count=int(eligible[5:].sum()),
                         body_mean_reprojection_px=mean, body_p90_reprojection_px=p90,
                         whole_mesh_in_raw_and_undistorted_image='native_smpl_whole_body_outside_image' not in reasons,
                         independent_bbox_untruncated='independent_person_bbox_truncated' not in reasons,
                         foreign_person_occluded_joint_count=person.get('other_athlete_occluded_joint_count'),
                         association_bbox_iou=person.get('association_bbox_iou'),
                         association_margin=person.get('association_margin'))


class _EvidenceIndex:
    """Disk-backed join: a full observation set need not live in Python RAM."""
    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.execute('CREATE TABLE IF NOT EXISTS evidence (sid TEXT, scan INTEGER, motion TEXT, camera TEXT, record TEXT, PRIMARY KEY(sid,scan,motion,camera))')
        self.connection.execute('DELETE FROM evidence')

    def get(self, key, default=None):
        rows = self.connection.execute('SELECT camera,record FROM evidence WHERE sid=? AND scan=? AND motion=?',key)
        return {camera:json.loads(record) for camera,record in rows}

    def put(self, key, camera, record):
        self.connection.execute('INSERT INTO evidence VALUES (?,?,?,?,?)',(*key,camera,json.dumps(record,allow_nan=False)))

    def commit(self):
        self.connection.commit()

    def close(self):
        self.connection.close()


def _report_frames(report):
    if 'frames_jsonl' not in report:
        yield from report['frames']
        return
    path = Path(report['frames_jsonl'])
    if file_sha(path) != report['frames_jsonl_sha256']:
        raise ValueError('Native observation journal changed after completion')
    count, people = 0, 0
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            frame = json.loads(line)
            count += 1
            people += sum(p['fit_accepted'] for p in frame['athletes'])
            yield frame
    if count != report['planned_camera_frames'] or people != report['planned_accepted_person_frames']:
        raise ValueError('Native observation journal does not cover every planned source person-frame')


def _load_evidence(paths, calibration_digest, index_path=None):
    evidence = _EvidenceIndex(index_path) if index_path else defaultdict(dict)
    provenance = []
    for path in paths:
        path = Path(path).resolve()
        report = _read(path)
        if not report.get('completed') or report.get('calibration_digest') != calibration_digest:
            raise ValueError(f'Incomplete or mismatched native RGB evidence: {path}')
        provenance.append(dict(path=str(path), sha256=file_sha(path), models=report.get('models'),
                               checker_sha256=report.get('checker_sha256'),
                               frames_jsonl=report.get('frames_jsonl'), frames_jsonl_sha256=report.get('frames_jsonl_sha256')))
        for frame in _report_frames(report):
            camera = frame['camera_name']
            if camera not in ('cam1', 'cam2') or frame.get('image_source') != 'original_native_video_not_review_overlay':
                raise ValueError('Evidence must be an original native camera observation')
            for person in frame['athletes']:
                key = _key(frame['session_id'], {**person, 'lidar_index':frame['lidar_index']})
                record = dict(person=person, frame={k:v for k,v in frame.items() if k not in ('athletes', 'detector_observations')},
                              report_path=str(path), report_sha256=provenance[-1]['sha256'])
                previous = evidence.get(key,{}).get(camera)
                if previous:
                    before = previous['person'].get('all_coco17_observations')
                    after = person.get('all_coco17_observations')
                    observations_equal = before is None and after is None
                    if before is not None and after is not None:
                        observations_equal = np.allclose(before, after, atol=1e-3, rtol=0)
                    if (previous['person']['fit_sha256'] != person['fit_sha256']
                            or previous['frame']['native_bgr_image_sha256'] != frame['native_bgr_image_sha256']
                            or not observations_equal):
                        raise ValueError(f'Conflicting duplicate independent evidence: {key} {camera}')
                else:
                    if index_path:
                        evidence.put(key,camera,record)
                    else:
                        evidence[key][camera] = record
        if index_path:
            evidence.commit()
    return evidence, provenance


def _bound_output(source_root, output_root):
    source, output = Path(source_root).resolve(), Path(output_root).resolve()
    if not output.is_relative_to(PROJ_ROOT/'outputs') or output.is_relative_to(source):
        raise ValueError('Strict export must use a separate project outputs folder')
    output.mkdir(parents=True, exist_ok=True)
    return source, output


def _require_unpublished(output):
    if (output/'datasets/dataset_manifest.json').exists():
        raise ValueError('Published strict datasets are immutable; use a separate output folder')


def audit_cached(source_root, output_root, report_paths, gate=DualViewGate(), disk_evidence=False):
    source, output = _bound_output(source_root, output_root)
    _require_unpublished(output)
    if gate.policy not in ('full_body','bilateral_detection'):
        raise ValueError('Unknown dual-camera qualification policy')
    dataset = _read(source/'datasets/dataset_manifest.json')
    calibration = CalibrationSnapshot.load(source/'calibration/calibration_snapshot.json')
    if dataset['calibration_digest'] != calibration.data['snapshot_digest']:
        raise ValueError('Source dataset and frozen geometry differ')
    evidence, reports = _load_evidence(report_paths, dataset['calibration_digest'],
                                      output/'native_evidence.sqlite' if disk_evidence else None)
    counts, failures, candidates = Counter(), Counter(), []
    regressor_path = PROJ_ROOT/'hmr4d/utils/body_model/smpl_coco17_J_regressor.pt'
    regressor = torch.load(regressor_path, map_location='cpu').float().numpy() if gate.policy=='full_body' else None
    exclusion_path = output/'excluded.jsonl'
    with exclusion_path.open('w', encoding='utf-8') as excluded:
        for sid, entry in dataset['exports'].items():
            with Path(entry['metadata']).open(encoding='utf-8') as rows:
                for source_index, line in enumerate(rows):
                    metadata = json.loads(line)
                    counts['source_accepted'] += 1
                    key = _key(sid, metadata)
                    pair = evidence.get(key, {})
                    reasons, views = [], {}
                    if set(pair) != {'cam1', 'cam2'}:
                        reasons.append('missing_dual_camera_person_observations')
                    else:
                        counts['with_bilateral_person_observations'] += 1
                        fit_path = Path(metadata['fit_path']).resolve()
                        fit_roots=[Path(p).resolve() for p in dataset.get('fit_roots',[str(source/'sessions')])]
                        if not any(fit_path.is_relative_to(root) for root in fit_roots):
                            raise ValueError('Source fit escapes the selected frozen run')
                        fit_sha = file_sha(fit_path)
                        if gate.policy=='full_body':
                            with np.load(fit_path, allow_pickle=False) as fit:
                                vertices = fit['vertices'].astype(np.float64)
                            joints = regressor @ vertices
                        reasons += ['source_fit:' + reason for reason in acceptance_reasons(metadata['qa'], metadata['source_point_count'])]
                        for camera, record in pair.items():
                            person, frame = record['person'], record['frame']
                            native = metadata['cameras'][camera]
                            expected = frame['camera_native_source']
                            if (person['fit_sha256'] != fit_sha or frame['scan_mid_us'] != metadata['scan_mid_us']
                                    or any(native[k] != expected[k] for k in ('record_index','media_pts_us','common_time_us'))
                                    or native['status'] != 'matched' or abs(native['time_error_us']) > 8500):
                                raise ValueError(f'Native image/fit/source binding mismatch: {key} {camera}')
                            if gate.policy == 'bilateral_detection':
                                rejected = detection_reasons(person,gate)
                                facts=dict(person_detection_present=not rejected,
                                           association_bbox_iou=person.get('association_bbox_iou'),
                                           association_margin=person.get('association_margin'),
                                           body_visibility_diagnostics='not_recomputed_presence_only_rule')
                            else:
                                if not np.allclose(joints, person['fixed_smpl_coco17_viewer_m'], atol=1e-6, rtol=0):
                                    raise ValueError('Cached joints do not match the selected native SMPL')
                                uv, depth = calibration.project(vertices, camera, distort=False)
                                raw_uv, _ = calibration.project(vertices, camera, distort=True)
                                projected, _ = calibration.project(joints, camera, distort=False)
                                if not np.allclose(projected, person['all_coco17_fixed_fit_projection'], atol=1e-3, rtol=0):
                                    raise ValueError('Cached joint projection is not the selected native SMPL')
                                rejected, facts = view_reasons(person, uv, depth, raw_uv, calibration.camera(camera)['image_size'], gate)
                                facts['full_body_diagnostic_reasons'] = rejected
                                facts['person_detection_present'] = not detection_reasons(person,gate)
                            reasons += [camera+':'+reason for reason in rejected]
                            views[camera] = dict(facts=facts, observation=person, native=frame['camera_native_source'],
                                                 raw_image_sha256=frame['native_bgr_image_sha256'],
                                                 undistorted_rgb_sha256=frame['undistorted_rgb_image_sha256'],
                                                 report_path=record['report_path'],report_sha256=record['report_sha256'])
                    if reasons:
                        failures.update(set(reasons))
                        excluded.write(json.dumps(dict(session_id=sid, source_index=source_index, motion_id=metadata['motion_id'],
                                                       lidar_index=metadata['lidar_index'], reasons=reasons))+'\n')
                    else:
                        identifier = f'c{len(candidates)+1:04d}'
                        candidates.append(dict(id=identifier, session_id=sid, source_index=source_index, source_metadata=metadata,
                                               source_fit_sha256=fit_sha, dualcam=views))
    (output/'candidates.jsonl').write_text(''.join(json.dumps(c,allow_nan=False)+'\n' for c in candidates),encoding='utf-8')
    if gate.policy == 'full_body':counts['automatic_complete_body_pass'] = len(candidates)
    counts['automatic_person_pair_pass'] = len(candidates)
    counts['automatic_fail_after_observation'] = counts['with_bilateral_person_observations'] - len(candidates)
    source_exports = {sid:dict(path=entry['path'],sha256=file_sha(entry['path']),
                               metadata=entry['metadata'],metadata_sha256=file_sha(entry['metadata']))
                      for sid,entry in dataset['exports'].items()}
    result = dict(schema='badminton-dualcam-audit-v1', state='geometry_and_observations_checked',
                  source_root=str(source), source_dataset_manifest_sha256=file_sha(source/'datasets/dataset_manifest.json'),
                  source_exports=source_exports,
                  calibration_digest=dataset['calibration_digest'], gate=asdict(gate),
                  gate_sha256=hashlib.sha256(_canonical(asdict(gate))).hexdigest(), counts=dict(counts),
                  failure_counts_overlapping=dict(failures), evidence_reports=reports,
                  candidate_file_sha256=file_sha(output/'candidates.jsonl'),
                  candidate_session_counts=dict(Counter(c['session_id'] for c in candidates)))
    write_json(output/'audit.json',result)
    if disk_evidence:
        evidence.close()
    return result


def render_candidate_crops(source_root, output_root):
    source, output = _bound_output(source_root, output_root)
    _require_unpublished(output)
    audit = _read(output/'audit.json')
    candidates = [json.loads(line) for line in (output/'candidates.jsonl').read_text(encoding='utf-8').splitlines()]
    if file_sha(output/'candidates.jsonl') != audit['candidate_file_sha256']:
        raise ValueError('Audit candidates changed before native image verification')
    calibration = CalibrationSnapshot.load(source/'calibration/calibration_snapshot.json')
    sessions, maps, groups = {}, {}, defaultdict(list)
    crop_dir = output/'crops';crop_dir.mkdir(exist_ok=True)
    for candidate in candidates:
        for camera, record in candidate['dualcam'].items():
            groups[(candidate['session_id'],camera,record['native']['record_index'])].append(candidate)
    for (sid,camera,index), group in groups.items():
        if sid not in sessions:
            sessions[sid] = SessionIndex.load(Path(calibration.data['sessions_root'])/sid,calibration.frame_map_path(sid))
        raw = sessions[sid].read_image(camera,index)
        sha = hashlib.sha256(raw.tobytes()).hexdigest()
        if (sid,camera) not in maps:
            model = calibration.camera(camera)
            K = np.asarray(model['K']);d = np.asarray(model['distortion_opencv'])
            maps[sid,camera] = cv2.initUndistortRectifyMap(K,d,None,K,tuple(model['image_size']),cv2.CV_32FC1)
        rgb = cv2.remap(raw[:,:,::-1],*maps[sid,camera],cv2.INTER_LINEAR)
        rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
        for candidate in group:
            record = candidate['dualcam'][camera]
            if sha != record['raw_image_sha256'] or rgb_sha != record['undistorted_rgb_sha256']:
                raise ValueError('Current native video pixels differ from independent observation evidence')
            box = np.asarray(record['observation']['independent_detection_bbox_xyxy'])
            padding = (box[2:]-box[:2])*.12
            low = np.maximum(np.floor(box[:2]-padding).astype(int),0)
            high = np.minimum(np.ceil(box[2:]+padding).astype(int),[rgb.shape[1],rgb.shape[0]])
            crop_path = crop_dir/f'{candidate["id"]}_{camera}.png'
            if not cv2.imwrite(str(crop_path),rgb[low[1]:high[1],low[0]:high[0],::-1]):
                raise IOError('Cannot save native-person review crop')
    rows = []
    for candidate in candidates:
        meta = candidate['source_metadata']
        rows.append(f'<tr><td>{candidate["id"]}<br>{html.escape(candidate["session_id"])}<br>scan {meta["lidar_index"]}<br>{html.escape(meta["motion_id"])}</td>'
                    + ''.join(f'<td><img src="crops/{candidate["id"]}_{camera}.png" height="280"></td>' for camera in ('cam1','cam2'))+'</tr>')
    description=('All candidates have a corresponding independent person detection in both native cameras; '
                 'individual visual review is not required by this qualification rule.'
                 if audit['gate'].get('policy')=='bilateral_detection' else
                 'All candidates passed both independent 12-limb/torso and whole-image extent gates.')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Native dual-camera person evidence</title>'
                                   '<style>body{font-family:sans-serif}td{padding:8px;border-bottom:1px solid #999}img{max-width:320px;object-fit:contain}</style>'
                                   '<h1>Native dual-camera body evidence</h1><p>Calibrated native RGB crops, without rendered SMPL overlays. '
                                   + description+'</p><table>'+''.join(rows)+'</table>',encoding='utf-8')
    audit.update(state='native_images_verified',native_camera_frames=len(groups),candidate_crops=2*len(candidates))
    write_json(output/'audit.json',audit)
    return audit


def export_reviewed(source_root, output_root, review_path=None, *, detection_only=None, full_observation_manifest=None):
    source, output = _bound_output(source_root, output_root)
    _require_unpublished(output)
    audit = _read(output/'audit.json')
    if detection_only is None:detection_only=audit['gate'].get('policy','full_body')=='bilateral_detection'
    valid_state = audit['state'] == 'native_images_verified'
    if detection_only and full_observation_manifest:
        observations = _read(full_observation_manifest)
        if not observations['completed'] or observations['source_accepted'] != audit['counts']['source_accepted']:
            raise ValueError('Full export requires completed native observations of every source label')
        if set(observations['reports']) != {r['path'] for r in audit['evidence_reports']}:
            raise ValueError('Full observation manifest and audited evidence reports differ')
        frozen_inputs={entry['path']:entry['sha256'] for entry in observations['identity']['source']['files']}
        for entry in audit['source_exports'].values():
            if (frozen_inputs.get(entry['path'])!=entry['sha256']
                    or frozen_inputs.get(entry['metadata'])!=entry['metadata_sha256']):
                raise ValueError('Source training tensors/metadata changed after native observation began')
        valid_state = True
    if not valid_state or file_sha(output/'candidates.jsonl') != audit['candidate_file_sha256']:
        raise ValueError('Export needs verified native camera images and unchanged candidates')
    if detection_only and audit['gate']['policy'] != 'bilateral_detection':
        raise ValueError('Automatic export requires the bilateral detection policy')
    reviews = {}
    for line in (Path(review_path).read_text(encoding='utf-8').splitlines() if review_path else []):
        row = json.loads(line)
        if row['id'] in reviews:raise ValueError('Duplicate completeness review decision')
        reviews[row['id']] = row
    candidates = [json.loads(line) for line in (output/'candidates.jsonl').read_text(encoding='utf-8').splitlines()]
    dataset = _read(source/'datasets/dataset_manifest.json')
    if file_sha(source/'datasets/dataset_manifest.json') != audit['source_dataset_manifest_sha256']:
        raise ValueError('Source dataset changed after independent auditing')
    chosen, rejected = defaultdict(list), Counter()
    for candidate in candidates:
        if detection_only:
            for camera in ('cam1','cam2'):
                if detection_reasons(candidate['dualcam'][camera]['observation'],DualViewGate(**audit['gate'])):
                    raise ValueError('Audited candidate lacks a corresponding person detection')
            chosen[candidate['session_id']].append((candidate,None))
            continue
        review = reviews.get(candidate['id'])
        if not review or review.get('both_native_views_complete') is not True:
            rejected['missing_or_failed_native_full_body_review'] += 1;continue
        if not isinstance(review.get('reviewer'),str) or not review['reviewer'].strip():
            raise ValueError('Native completeness review requires an explicit reviewer')
        for camera in ('cam1','cam2'):
            if review.get(camera+'_crop_sha256') != file_sha(output/'crops'/f'{candidate["id"]}_{camera}.png'):
                raise ValueError('Native visual decision is not bound to the displayed crop')
        chosen[candidate['session_id']].append((candidate,review))
    exports = {}
    for sid, entry in dataset['exports'].items():
        expected = audit['source_exports'][sid]
        if file_sha(entry['path']) != expected['sha256'] or file_sha(entry['metadata']) != expected['metadata_sha256']:
            raise ValueError('Source tensors/metadata changed after auditing')
        data = joblib.load(entry['path'])
        samples = []
        for candidate, review in chosen[sid]:
            index = candidate['source_index'];metadata = candidate['source_metadata'].copy()
            if data['motion_id'][index] != metadata['motion_id']:
                raise ValueError('Source tensor index does not match its person-frame metadata')
            motion = metadata['motion_id'].split(':')
            metadata.update(scene_id=sid,person_id=metadata['motion_id'],person_identity_scope='window_track_not_verified_across_sessions',
                            window_id=motion[1],track_id=motion[2],segment_id=motion[3],frame_index=metadata['lidar_index'],
                            source_tensor_path=entry['path'],source_tensor_index=index,
                            dualcam=dict(schema='native-dualcam-completeness-v1',qualified=True,gate= audit['gate'],
                                         qualification_policy=audit['gate'].get('policy','full_body'),
                                         gate_sha256=audit['gate_sha256'],views=candidate['dualcam'],native_visual_review=review,
                                         visual_review_status='not_required_detection_presence_rule' if detection_only else 'individually_reviewed'))
            sample = {name:data[name][index].numpy() for name in FIELDS}
            sample.update(motion_id=metadata['motion_id'],metadata=metadata)
            samples.append(sample)
        exports[sid] = export_joblib(samples,output/'datasets'/f'{sid}.pth')
        exports[sid].update(sha256=file_sha(exports[sid]['path']),metadata_sha256=file_sha(exports[sid]['metadata']))
    splits = {}
    for name, original in dataset['suggested_session_splits'].items():
        sids = original['sessions']
        if len(sids)==1:
            splits[name] = dict(exports[sids[0]],sessions=sids)
        else:
            merged=[]
            for sid in sids:
                data=joblib.load(exports[sid]['path'])
                rows=[json.loads(line) for line in Path(exports[sid]['metadata']).read_text(encoding='utf-8').splitlines()]
                for index,row in enumerate(rows):
                    sample={key:data[key][index].numpy() for key in FIELDS}
                    sample.update(motion_id=data['motion_id'][index],metadata=row);merged.append(sample)
            split=export_joblib(merged,output/'datasets'/f'{name}.pth')
            splits[name]=dict(split,sessions=sids,sha256=file_sha(split['path']),metadata_sha256=file_sha(split['metadata']))
    scope = 'full_existing_source_dataset' if full_observation_manifest else 'pilot_existing_native_observation_evidence'
    manifest = dict(schema='badminton-ptv3-dualcam-dataset-v1',status='complete',scope=scope,
                    accepted=sum(entry['accepted'] for entry in exports.values()),exports=exports,suggested_session_splits=splits,
                    counts_before_after={**audit['counts'],'after_qualification':sum(e['accepted'] for e in exports.values())},
                    excluded_after_automatic_gate=dict(rejected),calibration_digest=audit['calibration_digest'],
                    source_dataset_manifest=str(source/'datasets/dataset_manifest.json'),
                    source_dataset_manifest_sha256=audit['source_dataset_manifest_sha256'],
                    gate=audit['gate'],gate_sha256=audit['gate_sha256'],audit=str(output/'audit.json'),
                    audit_sha256=file_sha(output/'audit.json'),
                    qualification_policy=audit['gate'].get('policy','full_body'),
                    hard_rule='Exclude this person-frame if either native camera lacks a uniquely associated person detection' if detection_only else 'Full-body automatic gates plus paired native visual review',
                    manual_review_required=not detection_only,
                    review_path=str(Path(review_path).resolve()) if review_path else None,
                    review_sha256=file_sha(review_path) if review_path else None,
                    full_observation_manifest=str(Path(full_observation_manifest).resolve()) if full_observation_manifest else None,
                    full_observation_manifest_sha256=file_sha(full_observation_manifest) if full_observation_manifest else None,
                    target_body_model='neutral_smpl_24_joints',gender='neutral',shape_basis='SMPL fitted, not SMPL-X betas',
                    coordinate_space='lidar_viewer_xyz_m',point_count=256,transl_convention='shaped_pelvis_position',
                    split_limit='Whole sessions disjoint; cross-session physical identities unverified; not unseen-person generalization',
                    observation_limit='Independent native person detections with fixed-fit identity association; diagnostic pose/visibility/reprojection fields are not anatomical ground truth')
    # The consumer watches status=complete. Validate all values before the
    # atomic manifest write, so training cannot start on an unchecked export.
    verification=_verify_export_data(manifest)
    if dataset.get('scope')=='sealed_seven_source_rally_subset':
        from .rally_report import report_rally_delivery
        manifest['scope']='full_sealed_seven_source_rally_subset'
        report_rally_delivery(source,output,manifest)
        manifest['quality_summary']=str(output/'quality_summary.json')
        manifest['quality_summary_sha256']=file_sha(output/'quality_summary.json')
    manifest_path=output/'datasets/dataset_manifest.json'
    write_json(manifest_path,manifest)
    verification.update(dataset_manifest=str(manifest_path),dataset_manifest_sha256=file_sha(manifest_path))
    write_json(output/'verification.json',verification)
    return manifest


def verify_export(output_root):
    """Verify the delivered training values and person-level dual-view evidence."""
    output=Path(output_root).resolve()
    manifest_path=output/'datasets/dataset_manifest.json'
    manifest=_read(manifest_path)
    if manifest.get('status')!='complete':raise ValueError('Dataset is not complete')
    result=_verify_export_data(manifest)
    result.update(dataset_manifest=str(manifest_path),dataset_manifest_sha256=file_sha(manifest_path))
    write_json(output/'verification.json',result)
    return result


def _verify_export_data(manifest):
    source=_read(manifest['source_dataset_manifest'])
    if file_sha(manifest['source_dataset_manifest'])!=manifest['source_dataset_manifest_sha256']:
        raise ValueError('Original source manifest changed')
    if manifest.get('audit_sha256') and file_sha(manifest['audit'])!=manifest['audit_sha256']:
        raise ValueError('Dataset audit changed')
    if manifest.get('quality_summary_sha256') and file_sha(manifest['quality_summary'])!=manifest['quality_summary_sha256']:
        raise ValueError('Dataset quality report changed')
    result=dict(sessions={})
    for sid,entry in manifest['exports'].items():
        tensor_sha,metadata_sha=file_sha(entry['path']),file_sha(entry['metadata'])
        if (entry.get('sha256') and tensor_sha!=entry['sha256']) or (entry.get('metadata_sha256') and metadata_sha!=entry['metadata_sha256']):
            raise ValueError('Delivered tensor/metadata hash mismatch')
        data,original=joblib.load(entry['path']),joblib.load(source['exports'][sid]['path'])
        rows=[json.loads(line) for line in Path(entry['metadata']).read_text(encoding='utf-8').splitlines()]
        if len(rows)!=entry['accepted'] or data['motion_id']!=[r['motion_id'] for r in rows]:
            raise ValueError('Delivered metadata count/order mismatch')
        indices=torch.tensor([r['source_tensor_index'] for r in rows],dtype=torch.long)
        for name in FIELDS:
            if not torch.equal(data[name],original[name][indices]):
                raise ValueError(f'Delivered {name} changed original training values')
        for row in rows:
            if original['motion_id'][row['source_tensor_index']]!=row['motion_id']:
                raise ValueError('Delivered source person/index mismatch')
            if manifest['qualification_policy']=='bilateral_detection':
                for camera in ('cam1','cam2'):
                    if detection_reasons(row['dualcam']['views'][camera]['observation'],DualViewGate(**manifest['gate'])):
                        raise ValueError('Delivered label lacks corresponding detection in both cameras')
        result['sessions'][sid]=dict(accepted=len(rows),exact_tensor_subset=True,bilateral_person_presence_verified=True,
                                     tensor_sha256=tensor_sha,metadata_sha256=metadata_sha,
                                     export_hashes_bound_in_manifest='sha256' in entry and 'metadata_sha256' in entry)
    result['accepted']=sum(e['accepted'] for e in result['sessions'].values())
    if source.get("required_nonempty_splits"):
        from .export import require_nonempty_splits
        require_nonempty_splits(dict(manifest, required_nonempty_splits=source["required_nonempty_splits"]))
    if result['accepted']!=manifest['accepted']:raise ValueError('Delivered total differs from manifest')
    assigned=set()
    for name,split in manifest.get('suggested_session_splits',{}).items():
        sids=split['sessions']
        if assigned.intersection(sids) or len(sids)!=len(set(sids)):
            raise ValueError('Session leakage between delivered splits')
        assigned.update(sids)
        if len(sids)<=1:continue
        if file_sha(split['path'])!=split['sha256'] or file_sha(split['metadata'])!=split['metadata_sha256']:
            raise ValueError('Merged split artifact hash mismatch')
        merged=joblib.load(split['path']);parts=[joblib.load(manifest['exports'][sid]['path']) for sid in sids]
        if merged['motion_id']!=[motion for part in parts for motion in part['motion_id']]:
            raise ValueError('Merged split person/order mismatch')
        for field in FIELDS:
            if not torch.equal(merged[field],torch.cat([part[field] for part in parts],dim=0)):
                raise ValueError('Merged split changed source values: '+field)
        result.setdefault('merged_splits',{})[name]=dict(accepted=len(merged['motion_id']),exact_session_concatenation=True)
    return result
