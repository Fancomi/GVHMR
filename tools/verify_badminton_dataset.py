"""Verify exported labels against original fixed24 returns and native SMPL."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import smplx
import torch
from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.rally_pipeline import load_run_session
from hmr4d.utils.body_model.contracts import forward_smpl_at_pelvis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=Path('outputs/badminton'))
    parser.add_argument('--samples-per-session',type=int,default=16)
    parser.add_argument('--report',type=Path,help='Write verification here when inspecting an immutable historical run')
    args = parser.parse_args()
    if args.samples_per_session<1:
        parser.error('samples-per-session must be positive')
    torch.set_num_threads(4)
    calibration = CalibrationSnapshot.load(args.output_root/'calibration/calibration_snapshot.json')
    manifest = json.loads((args.output_root/'datasets/dataset_manifest.json').read_text(encoding='utf-8'))
    model = smplx.create(PROJ_ROOT/'inputs/checkpoints/body_models',model_type='smpl',gender='neutral').eval()
    report = {'scope':'Original recorded fixed24 returns and native neutral SMPL reconstruction',
              'sessions':{},'calibration_digest':calibration.data['snapshot_digest']}
    for sid,entry in manifest['exports'].items():
        if not entry['accepted']:
            continue
        data = joblib.load(entry['path'])
        rows = [json.loads(line) for line in Path(entry['metadata']).read_text(encoding='utf-8').splitlines()]
        if len(rows)!=len(data['pc']):
            raise ValueError('Metadata and tensor sample counts differ')
        session = load_run_session(args.output_root,calibration,sid)
        pairs = session.pairs()
        selected = np.unique(np.linspace(0,len(rows)-1,min(args.samples_per_session,len(rows))).astype(int))
        cloud_error = mesh_error = pelvis_error = 0
        for index in selected:
            row = rows[index]
            cloud = session.read_cloud(pairs[row['lidar_index']])
            expected = cloud[np.asarray(row['source_segment_indices'],dtype=np.int64)]
            cloud_error = max(cloud_error,float(np.max(np.abs(data['pc'][index].numpy()-expected))))
            with torch.no_grad():
                vertices,joints = forward_smpl_at_pelvis(model,data['root_pose'][index:index+1],
                                                       data['body_pose'][index:index+1],data['shape'][index:index+1],
                                                       data['transl'][index:index+1])
            with np.load(row['fit_path']) as fit:
                mesh_error = max(mesh_error,float(np.max(np.abs(vertices[0].numpy()-fit['vertices']))))
            pelvis_error = max(pelvis_error,float((joints[:,0]-data['transl'][index:index+1]).abs().max()))
        if cloud_error>1e-6 or mesh_error>1e-4 or pelvis_error>1e-6:
            raise ValueError(f'Actual source/body contract mismatch {sid}: {cloud_error}, {mesh_error}, {pelvis_error}')
        report['sessions'][sid] = {'exported':len(rows),'checked':len(selected),'max_source_cloud_error_m':cloud_error,
                                   'max_rebuilt_mesh_error_m':mesh_error,'max_pelvis_translation_error_m':pelvis_error}
    path = args.report or args.output_root/'datasets/verification.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
