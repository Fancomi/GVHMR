import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import torch

from hmr4d.lidar_mocap.dual_view import DualViewGate, detection_reasons, view_reasons, export_reviewed, file_sha
from hmr4d.lidar_mocap.export import FIELDS
from hmr4d.lidar_mocap.rgb_observations import RGBKeypointChecker
from hmr4d.lidar_mocap.dual_view_observer import _resume_rows


class WholeBodyGateTests(unittest.TestCase):
    def setUp(self):
        self.uv = np.array([[100.,100.],[200.,500.]])
        points = np.tile([150.,250.,.9],(17,1))
        self.person = dict(all_coco17_observations=points.tolist(),
                           all_coco17_fixed_fit_projection=points[:,:2].tolist(),
                           eligible_mask=[True]*17,independent_detection_bbox_xyxy=[90,80,220,520],
                           association_bbox_iou=.8,association_margin=.3,
                           other_athlete_occluded_joint_count=0,status='measured')

    def reasons(self, person=None, raw=None):
        return view_reasons(person or self.person,self.uv,np.ones(2),self.uv if raw is None else raw,
                            [1920,1080],DualViewGate())[0]

    def test_complete_body_does_not_require_all_facial_points(self):
        person = copy.deepcopy(self.person)
        person['eligible_mask'][:5] = [False]*5
        for point in person['all_coco17_observations'][:5]:point[2]=.1
        self.assertEqual(self.reasons(person),[])

    def test_six_observations_cannot_qualify_an_incomplete_body(self):
        person = copy.deepcopy(self.person)
        person['eligible_mask'][10:] = [False]*7
        self.assertIn('incomplete_torso_or_limb_observations',self.reasons(person))

    def test_missing_pose_cannot_qualify_from_timestamps(self):
        person = copy.deepcopy(self.person)
        person['all_coco17_observations'] = None
        self.assertIn('missing_or_nonfinite_native_body_observations',self.reasons(person))

    def test_bounds_occlusion_association_and_reprojection_fail_closed(self):
        for change, expected in [({'independent_detection_bbox_xyxy':[0,0,220,520]},'independent_person_bbox_truncated'),
                                 ({'other_athlete_occluded_joint_count':1},'foreign_person_occlusion_or_missing_visibility_evidence'),
                                 ({'association_margin':.01},'ambiguous_same_person_association')]:
            with self.subTest(reason=expected):
                self.assertIn(expected,self.reasons({**self.person,**change}))
        self.assertIn('native_smpl_whole_body_outside_image',self.reasons(raw=np.array([[0.,0.],[200.,500.]])))
        person = copy.deepcopy(self.person)
        person['all_coco17_fixed_fit_projection'] = (np.asarray(person['all_coco17_fixed_fit_projection'])+100).tolist()
        self.assertIn('bilateral_body_reprojection_outside_limit',self.reasons(person))


class ForeignMeshVisibilityTests(unittest.TestCase):
    def test_bbox_broad_phase_preserves_triangle_hits_and_depth(self):
        checker = RGBKeypointChecker.__new__(RGBKeypointChecker)
        checker.camera = dict(K=[[100,0,100],[0,100,100],[0,0,1]])
        checker.faces = np.array([[0,1,2]])
        triangle = np.array([[-1,-1,3],[1,-1,3],[0,1,3]],float)
        uv = np.array([[100,100],[170,100],[100,120],[100,100]],float)
        blocked = checker._other_mesh_occludes(uv,np.array([5,5,5,2]),[triangle,triangle],0)
        np.testing.assert_array_equal(blocked,[True,False,True,False])
        np.testing.assert_array_equal(checker._other_mesh_occludes(uv,np.ones(4)*5,[triangle],0),[False]*4)


class BilateralPresenceTests(unittest.TestCase):
    def setUp(self):
        self.person=dict(detector_index=0,independent_detection_bbox_xyxy=[10,20,100,300],
                         independent_detection_confidence=.7,association_bbox_iou=.6,association_margin=.2)

    def test_detection_rule_does_not_require_pose_or_manual_visibility(self):
        self.assertEqual(detection_reasons(self.person),[])
        self.assertEqual(detection_reasons({**self.person,'eligible_mask':[False]*17,
            'other_athlete_occluded_joint_count':8,'status':'excluded','all_coco17_observations':None}),[])

    def test_missing_view_ambiguous_identity_and_nonfinite_evidence_fail(self):
        for change in [dict(detector_index=None),dict(detector_index=-1),dict(association_margin=.01),
                       dict(independent_detection_confidence=float('nan')),dict(association_bbox_iou=float('nan'))]:
            with self.subTest(change=change):self.assertTrue(detection_reasons({**self.person,**change}))


class ExportIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'outputs/source'
        self.output = self.root/'outputs/strict'
        (self.source/'datasets').mkdir(parents=True)
        (self.output/'crops').mkdir(parents=True)
        self.patcher = patch('hmr4d.lidar_mocap.dual_view.PROJ_ROOT',self.root)
        self.patcher.start();self.addCleanup(self.patcher.stop)
        self.sid = 'session'
        self.motion = 'session:window:1:0'
        self.data = {name:torch.arange(2*int(np.prod(shape))).reshape(2,*shape).float()/1000 for name,shape in FIELDS.items()}
        self.data['motion_id'] = ['session:window:2:0',self.motion]
        tensor = self.source/'datasets/source.pth';joblib.dump(self.data,tensor)
        metadata = self.source/'datasets/source.metadata.jsonl';metadata.write_text('{}\n{}\n')
        dataset = dict(exports={self.sid:dict(path=str(tensor),metadata=str(metadata))},suggested_session_splits={'train':dict(sessions=[self.sid])})
        manifest = self.source/'datasets/dataset_manifest.json';manifest.write_text(json.dumps(dataset))
        self.candidate = dict(id='c0001',session_id=self.sid,source_index=1,
                              source_metadata=dict(motion_id=self.motion,lidar_index=77),dualcam={})
        candidates = self.output/'candidates.jsonl';candidates.write_text(json.dumps(self.candidate)+'\n')
        audit = dict(state='native_images_verified',candidate_file_sha256=file_sha(candidates),
                     source_dataset_manifest_sha256=file_sha(manifest),source_exports={self.sid:dict(sha256=file_sha(tensor),metadata_sha256=file_sha(metadata))},
                     gate={},gate_sha256='gate',counts={},calibration_digest='frozen')
        (self.output/'audit.json').write_text(json.dumps(audit))
        self.review = dict(id='c0001',reviewer='test reviewer',both_native_views_complete=True)
        for camera in ('cam1','cam2'):
            path = self.output/'crops'/f'c0001_{camera}.png';path.write_bytes(b'example native crop')
            self.review[camera+'_crop_sha256']=file_sha(path)
        self.review_path = self.output/'review.jsonl'
        self.save_review()

    def save_review(self):
        self.review_path.write_text(json.dumps(self.review)+'\n')

    def test_subset_keeps_exact_points_pose_shape_and_pelvis(self):
        result = export_reviewed(self.source,self.output,self.review_path)
        self.assertEqual(result['accepted'],1)
        actual = joblib.load(result['exports'][self.sid]['path'])
        for name in FIELDS:
            self.assertTrue(torch.equal(actual[name],self.data[name][1:2]))
        self.assertEqual(actual['motion_id'],[self.motion])
        with self.assertRaisesRegex(ValueError,'immutable'):
            export_reviewed(self.source,self.output,self.review_path)

    def test_review_needs_named_reviewer_and_both_unchanged_crop_hashes(self):
        self.review['reviewer']='';self.save_review()
        with self.assertRaisesRegex(ValueError,'explicit reviewer'):
            export_reviewed(self.source,self.output,self.review_path)
        self.review['reviewer']='reviewer';self.save_review()
        (self.output/'crops/c0001_cam2.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'displayed crop'):
            export_reviewed(self.source,self.output,self.review_path)

    def test_changed_source_tensor_is_rejected_before_export(self):
        with (self.source/'datasets/source.pth').open('ab') as handle:handle.write(b'changed')
        with self.assertRaisesRegex(ValueError,'Source tensors/metadata changed'):
            export_reviewed(self.source,self.output,self.review_path)

    def test_failed_final_validation_does_not_publish_a_complete_manifest(self):
        with patch('hmr4d.lidar_mocap.dual_view._verify_export_data',side_effect=ValueError('invalid final values')):
            with self.assertRaisesRegex(ValueError,'invalid final values'):
                export_reviewed(self.source,self.output,self.review_path)
        self.assertFalse((self.output/'datasets/dataset_manifest.json').exists())

    def test_automatic_export_needs_both_detections_but_no_review(self):
        person=dict(detector_index=0,independent_detection_bbox_xyxy=[10,20,100,300],
                    independent_detection_confidence=.7,association_bbox_iou=.6,association_margin=.2)
        self.candidate['dualcam']={cam:dict(observation=person) for cam in ('cam1','cam2')}
        path=self.output/'candidates.jsonl';path.write_text(json.dumps(self.candidate)+'\n')
        audit_path=self.output/'audit.json';audit=json.loads(audit_path.read_text())
        audit['gate']={'policy':'bilateral_detection'};audit['candidate_file_sha256']=file_sha(path)
        audit_path.write_text(json.dumps(audit))
        result=export_reviewed(self.source,self.output,detection_only=True)
        self.assertEqual(result['accepted'],1);self.assertFalse(result['manual_review_required'])
        metadata=json.loads(Path(result['exports'][self.sid]['metadata']).read_text())
        self.assertIsNone(metadata['dualcam']['native_visual_review'])

    def test_full_export_rejects_source_changed_since_observation_started(self):
        path=self.output/'audit.json';audit=json.loads(path.read_text())
        audit.update(counts={'source_accepted':2},evidence_reports=[])
        entry=audit['source_exports'][self.sid]
        entry.update(path=str(self.source/'datasets/source.pth'),metadata=str(self.source/'datasets/source.metadata.jsonl'))
        path.write_text(json.dumps(audit))
        observations=dict(completed=True,source_accepted=2,reports=[],identity=dict(source=dict(files=[
            dict(path=entry['path'],sha256='older-original-tensor-sha'),dict(path=entry['metadata'],sha256=entry['metadata_sha256'])])))
        observation_path=self.output/'observer.json';observation_path.write_text(json.dumps(observations))
        with self.assertRaisesRegex(ValueError,'changed after native observation began'):
            export_reviewed(self.source,self.output,detection_only=True,full_observation_manifest=observation_path)

    def test_resume_repairs_only_partial_tail_and_rejects_changed_source(self):
        frame=dict(lidar_index=1,scan_mid_us=100,cameras={},athletes=[dict(motion_id='a')])
        record=dict(lidar_index=1,scan_mid_us=100,paired_cameras={},camera_name='cam1',
                    athletes=[dict(motion_id='a',fit_accepted=True)],timing_seconds=dict(total=.1))
        path=self.output/'observations.jsonl';valid=(json.dumps(record)+'\n').encode()
        path.write_bytes(valid+b'{"interrupted":')
        self.assertEqual(_resume_rows(path,{1:frame},'cam1'),({1},1,1,.1))
        self.assertEqual(path.read_bytes(),valid)
        with self.assertRaisesRegex(ValueError,'source changed'):
            _resume_rows(path,{1:{**frame,'scan_mid_us':200}},'cam1')


if __name__ == '__main__':unittest.main()
