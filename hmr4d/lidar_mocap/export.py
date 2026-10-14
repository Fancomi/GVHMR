"""Quality gate and the existing PTv3 single-frame SMPL/joblib contract."""
import json
from pathlib import Path
import joblib
import numpy as np
import torch


FIELDS = {'pc': (256,3), 'root_pose': (1,3), 'body_pose': (23,3),
          'shape': (10,), 'transl': (3,)}


def acceptance_reasons(qa, point_count, *, mean_limit=.065, p90_limit=.12):
    reasons = []
    if point_count < 80:
        reasons.append('fewer_than_80_human_returns')
    for key, limit in (('final_p2surface_mean_m',mean_limit), ('final_p2surface_p90_m',p90_limit)):
        value = qa.get(key)
        if value is None or not np.isfinite(value) or value > limit:
            reasons.append(key + '_outside_limit')
    if qa.get('finite') is not True:
        reasons.append('nonfinite_fit')
    floor = qa.get('floor_penetration_m')
    if floor is None or not np.isfinite(floor) or floor > .12:
        reasons.append('floor_penetration')
    projection = qa.get('reprojection_mean_px')
    if projection is None or not np.isfinite(projection) or projection > 50:
        reasons.append('primary_reprojection_over_50px')
    if max(qa.get('reprojection_valid_joints',{}).values(),default=0) < 6:
        reasons.append('fewer_than_6_reprojection_joints')
    return reasons


def validate_sample(sample):
    for name, shape in FIELDS.items():
        value = np.asarray(sample[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid {name}: {value.shape}; expected finite {shape}')
    if np.max(np.abs(np.asarray(sample['shape']))) > 5:
        raise ValueError('SMPL shape exceeds configured beta bound')


def export_joblib(samples, output_path):
    """Only accepted real-cloud samples are passed here by the pipeline."""
    if not samples:
        return {'accepted': 0, 'path': None}
    for sample in samples:
        validate_sample(sample)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    data = {name: torch.from_numpy(np.stack([s[name] for s in samples]).astype(np.float32)) for name in FIELDS}
    data['motion_id'] = [s['motion_id'] for s in samples]
    temporary = output_path.with_suffix(output_path.suffix+'.tmp')
    joblib.dump(data, temporary)
    temporary.replace(output_path)
    metadata_path = output_path.with_suffix('.metadata.jsonl')
    metadata_path.write_text(''.join(json.dumps(s['metadata'],ensure_ascii=False)+'\n' for s in samples),encoding='utf-8')
    return {'accepted':len(samples),'path':str(output_path),'metadata':str(metadata_path),
            'fields':{k:list(v.shape) for k,v in data.items() if torch.is_tensor(v)},
            'coordinate_space':'lidar_viewer_xyz_m', 'transl_convention':'shaped_pelvis_position',
            'cloud_source':'segmented_actual_fixed24_returns'}
