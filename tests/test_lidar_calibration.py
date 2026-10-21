"""Canonical bundle, raw recording identity and projection contracts; CPU only."""
import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from hmr4d.lidar_mocap import calibration as module
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.sessions import SessionIndex


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return sha(path.read_bytes())


def compact_frame_map(session):
    rows = []
    for pair in session.pairs():
        rows.append({"lidar_frame": pair["lidar_index"], "device_first_us": pair["scan_start_us"],
                     "device_last_us": pair["scan_end_us"], "scan_mid_us": pair["scan_mid_us"],
                     "cameras": {name: {"status": match["status"], "record_index": match["record_index"],
                                         "media_pts_us": match["media_pts_us"], "video_common_us": match["common_time_us"],
                                         "time_error_us": match["time_error_us"]}
                                 for name, match in pair["cameras"].items()}})
    return b"".join(json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n" for row in rows)


class SnapshotFixture:
    """Five canonical files and a complete miniature native recording."""

    def __init__(self, root):
        self.workspace = root / "gvhmr"
        self.workspace.mkdir()
        self.project = root / "lidar_project"
        self.sessions = root / "sessions"
        self.output = self.workspace / "snapshot"
        self.base = self.project / "config/calibration/badminton-20261012"
        self.sid = "20261012_154726"
        self.session = self.sessions / self.sid
        self.lidar = self.session / "output-lidar" / f"lidar_{self.sid}"
        self.video = self.session / "output-zcam" / f"zcam_{self.sid}"
        self.meta = self.session / "meta" / self.sid
        for directory in (self.lidar, self.video, self.meta):
            directory.mkdir(parents=True)
        angle = .17
        self.q = np.round([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]], 7)
        self.manifest_path = self.lidar / "manifest.json"
        self.clock_path = self.meta / f"{self.sid}_common_clock.json"
        self._make_recording()

        theta = .3
        raw_r = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        camera = dict(K=[[1000., 0, 960.], [0, 1100., 540.], [0, 0, 1]],
                      distortion_opencv=[-.03, .01, .002, -.003, -.001],
                      R_raw_lidar_to_camera=raw_r.tolist(),
                      R_viewer_to_camera=(raw_r @ np.linalg.inv(self.q)).tolist(), t_m=[.4, -.2, 1.3])
        self.policy = dict(schema="selected-device-common-sync-v1", variant="device_mid",
                           clock_axis="device_reported_timestamps_in_host_seeded_PHC_epoch",
                           lidar_reference="scan_midpoint",
                           target_formula="(device_first_packet_us + device_last_packet_us) // 2",
                           timestamp_unit="integer_microseconds", video_selection="nearest_ZCAM_common_NTP_timestamp",
                           tie_break="earlier_record_index", extra_offset_us=dict(cam1=0, cam2=0),
                           media_binding="canonical_media_pts_us_for_selected_record_index",
                           missing_time_behavior="unmatched; never reuse an endpoint frame or infer a receive-clock fallback",
                           physical_exposure_accuracy_verified=False, per_point_scan_times_available=False)
        offset = np.array([0., 0., 2.8])
        origin = -self.q @ offset
        normal = np.linalg.inv(self.q).T @ np.array([0., 0., 1.])
        normal /= np.linalg.norm(normal)
        axes = self.q @ np.diag([-1., -1., 1.])
        left, _, right = np.linalg.svd(axes)
        rotation = left @ right
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arcsin(-rotation[2, 0])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        self.plane = dict(origin_viewer_m=origin.tolist(), normal_viewer=normal.tolist(),
                          court_to_raw_R=np.eye(3).tolist(), court_origin_raw_m=(-offset).tolist(),
                          ground_bbox_centre_normal_offset_m=-.03, ground_bbox_upper_surface_margin_m=.105,
                          note="Normative frozen plane; no independently surveyed floor")
        self.bbox = dict(version=1, boxes=[dict(id="region_1", role="ground", type="Ignore", enabled=True,
                         center=(origin-.03*normal).tolist(), size=[6.1, 13.4, .27],
                         rotation_deg=np.rad2deg([roll, pitch, yaw]).tolist()),
                         dict(id="region_2", role="net", type="Ignore", enabled=True,
                              center=[10., 0., -1.9], size=[6.1, .6, 1.85], rotation_deg=[0., 0., 3.])])
        self.config = dict(schema="badminton-calibration-result-v1", sport="badminton", image_size=[1920, 1080],
                           coordinate_convention="p_destination = R @ p_source + t; column vectors, metres",
                           lidar_rotation=self.q.reshape(-1).tolist(), cameras=dict(cam1=camera, cam2=copy.deepcopy(camera)),
                           court_registration=dict(R_raw_lidar_to_court=np.eye(3).tolist(), t_raw_lidar_to_court_m=offset.tolist()),
                           court_reference_plane=self.plane, timing_policy=self.policy,
                           court=dict(width_m=6.1, length_m=13.4, line_width_m=.04,
                                      dimension_basis="Regulatory prior; not independently surveyed"),
                           source_id=self.sid, valid_source_ids=[self.sid], production_admitted=False)
        write_json(self.base / "calibration.json", self.config)
        bbox_hash = write_json(self.base / "bbox.json", self.bbox)
        policy_hash = write_json(self.base / "sync-policy.json", self.policy)
        (self.base / "roi-config.txt").write_text("# Original rounded capture matrix, metres\n"
            + "lidar_rotation=" + ",".join(format(x, ".17g") for x in self.q.reshape(-1))
            + "\nbbox_json=bbox.json\n", encoding="utf-8")
        self.sources = dict(schema="badminton-calibration-sources-v1", workflow_branch="calibration/workflow",
                            input_package="badminton/calibration/20261012", recordings_package="badminton/sessions",
                            input_manifest_sha256="a"*64, input_case_sha256="b"*64,
                            accepted_source_calibration_sha256="c"*64,
                            accepted_bbox_sha256=bbox_hash, sync_policy_sha256=policy_hash,
                            valid_source_ids=[self.sid], frame_maps=[], production_admitted=False,
                            point_clouds=dict(coordinate_encoding="fixed24", coordinate_frame="raw LiDAR",
                                              source_members_preserved=True),
                            validation_inputs=dict(accepted_camera_or_timing_change=False))
        self.refresh_source_map()

    def _make_recording(self):
        mids = (100000, 200000, 300000)
        captures = [dict(device_first_packet_us=t-1000, device_last_packet_us=t+1000) for t in mids]
        frames = [dict(frame_id=i+1, file=f"frame_{i:06d}.png", kept_points=1) for i in range(len(mids))]
        manifest = dict(format="png-sequence", coordinate_encoding="fixed24", coordinate_space="lidar_sensor_xyz",
                        lidar_rotation=self.q.reshape(-1).tolist(), frame_count=len(mids), frames=frames,
                        frame_pattern="frame_%06d.png", frame_capture_timestamps=captures,
                        frame_timestamps=[t/1e6 for t in mids], fps=10., point_width=2, point_height=1,
                        scale=1000, center=256)
        write_json(self.manifest_path, manifest)
        encoded = np.array([[257000, 0], [258000, 0], [259000, 0]], dtype=np.uint32)
        rgb = np.stack(((encoded >> 16) & 255, (encoded >> 8) & 255, encoded & 255), axis=-1).astype(np.uint8)
        for frame in frames:
            Image.fromarray(rgb).save(self.lidar / frame["file"])
        tracks = [dict(device_id="lidar1", rows=[dict(frame_id=i+1, common_time_us=t-1000,
                       end_common_time_us=t+1000) for i, t in enumerate(mids)])]
        for name in ("cam1", "cam2"):
            rows = [dict(common_time_us=t, media_pts_us=i*100000, frame_no=601+i) for i, t in enumerate(mids)]
            tracks.append(dict(device_id=name, rows=rows))
            sidecar = [dict(record_index=i, media_pts=row["media_pts_us"], media_time_base_num=1,
                           media_time_base_den=1000000, ssp_frame_no=row["frame_no"], discontinuity=False,
                           synthesized=False, connection_segment=0) for i, row in enumerate(rows)]
            (self.video / f"{name}.frames.jsonl").write_text(
                "\n".join(json.dumps(row) for row in sidecar)+"\n", encoding="utf-8")
            # freeze must index clocks/PTS without decoding media or using a GPU.
            (self.video / f"{name}.mp4").write_bytes(b"not decoded by calibration freeze")
        write_json(self.clock_path, dict(schema="external_gm_common_timeline_v1", session_id=self.sid,
                   common_overlap_begin_us=mids[0], common_overlap_end_us=mids[-1],
                   physical_accuracy_verified=False, exposure_offset_calibrated=False, tracks=tracks))

    def refresh_source_map(self):
        blob = compact_frame_map(SessionIndex.load(self.session))
        self.expected_map = blob
        self.sources["frame_maps"] = [dict(source_id=self.sid, path=f"frame-map/{self.sid}.jsonl",
                                          rows=3, sha256=sha(blob))]
        write_json(self.base / "sources.json", self.sources)

    def freeze(self):
        return CalibrationSnapshot.freeze(self.project, self.sessions, self.output)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gvhmr-calibration-test-")
        self.fixture = SnapshotFixture(Path(self.temp.name))
        self.workspace_patch = patch.object(module, "_WORKSPACE_ROOT", self.fixture.workspace)
        self.workspace_patch.start()

    def tearDown(self):
        self.workspace_patch.stop()
        self.temp.cleanup()

    def test_explicit_independent_bundle_preserves_default_and_rejects_escape(self):
        f = self.fixture
        alternate = f.project / "config/calibration/badminton-20261015"
        shutil.copytree(f.base, alternate)
        first = f.freeze()
        separate = CalibrationSnapshot.freeze(f.project, f.sessions, f.workspace / "independent",
                                              calibration_dir=alternate)
        self.assertEqual(first.data["configuration_files_sha256"], separate.data["configuration_files_sha256"])
        self.assertEqual(separate.data["artifacts"]["evidence/calibration.json"]["source_path"],
                         str(alternate / "calibration.json"))
        with self.assertRaisesRegex(ValueError, "escapes"):
            CalibrationSnapshot.freeze(f.project, f.sessions, f.workspace / "bad",
                                       calibration_dir=f.workspace)

    def test_explicit_index_freeze_uses_imported_pairs_without_canonical_sessions(self):
        from hmr4d.lidar_mocap import source_index as source_module
        f = self.fixture
        def binding(path):
            return dict(file=str(path.relative_to(f.sessions)), sha256=sha(path.read_bytes()))
        session = SessionIndex.load(f.session)
        hashes = {name: sha((f.base / name).read_bytes()) for name in module._CONFIG_FILES}
        session.source_index = dict(
            accepted_calibration_sha256=hashes["calibration.json"], accepted_bundle_sha256=hashes,
            source_manifest=binding(f.manifest_path), common_clock=binding(f.clock_path),
            canonical_video_sidecars={cam: binding(f.video / (cam + ".frames.jsonl")) for cam in ("cam1", "cam2")})
        index = f.sessions / "index.json"
        index.write_text("{}")
        with patch.object(source_module.SourceIndexedSession, "load_source_index", return_value=session):
            result = CalibrationSnapshot.freeze(
                f.project, f.sessions / "missing-layout", f.output, calibration_dir=f.base,
                source_indexes={f.sid: binding(index)}, data_root=f.sessions)
        self.assertEqual(result.data["sessions"][f.sid]["frame_count"], len(session.pairs()))
        session.source_index["accepted_calibration_sha256"] = "0" * 64
        with patch.object(source_module.SourceIndexedSession, "load_source_index", return_value=session):
            with self.assertRaisesRegex(ValueError, "another calibration"):
                CalibrationSnapshot.freeze(f.project, f.sessions, f.workspace / "wrong-index",
                    source_indexes={f.sid: binding(index)}, data_root=f.sessions)

    def test_review_subset_freezes_only_bound_ready_source_without_admitting_production(self):
        from hmr4d.lidar_mocap import source_index as source_module
        f=self.fixture
        f.config['valid_source_ids']=[f.sid,'not-yet-exported']
        f.sources['valid_source_ids']=[f.sid,'not-yet-exported']
        f.sources['frame_maps'].append(dict(source_id='not-yet-exported',rows=4,sha256='f'*64))
        write_json(f.base/'calibration.json',f.config)
        write_json(f.base/'sources.json',f.sources)
        hashes={name:sha((f.base/name).read_bytes()) for name in module._CONFIG_FILES}
        def binding(path):
            return dict(file=str(path.relative_to(f.sessions)),sha256=sha(path.read_bytes()))
        session=SessionIndex.load(f.session)
        session.source_index=dict(candidate_bundle_sha256=hashes,crop_binding=dict(bbox_sha256=hashes['bbox.json']),
            camera_calibration_admitted=False,source_manifest=binding(f.manifest_path),common_clock=binding(f.clock_path),
            canonical_video_sidecars={cam:binding(f.video/(cam+'.frames.jsonl')) for cam in ('cam1','cam2')})
        index=f.sessions/'index.json'
        index.write_text('{}')
        with patch.object(source_module.SourceIndexedSession,'load_source_index',return_value=session) as load:
            result=CalibrationSnapshot.freeze(f.project,f.sessions,f.output,calibration_dir=f.base,
                source_indexes={f.sid:binding(index)},data_root=f.sessions,review_source_ids=[f.sid])
        self.assertEqual(set(result.data['sessions']),{f.sid})
        self.assertTrue(result.data['review_only'])
        self.assertFalse(result.data['production_admitted'])
        self.assertFalse(load.call_args.kwargs['require_admitted'])
        self.assertFalse(session.source_index['camera_calibration_admitted'])
        self.assertEqual(result.data['configuration_files_sha256'],hashes)
        with patch.object(source_module.SourceIndexedSession,'load_source_index',return_value=session):
            with self.assertRaisesRegex(ValueError,'cover the calibration'):
                CalibrationSnapshot.freeze(f.project,f.sessions,f.workspace/'formal',calibration_dir=f.base,
                    source_indexes={f.sid:binding(index)},data_root=f.sessions)

    def test_portable_review_bundle_uses_real_parent_and_authorization_without_accepted_fields(self):
        from hmr4d.lidar_mocap import source_index as source_module
        f = self.fixture
        def binding(path):
            return dict(file=str(path.relative_to(f.sessions)), sha256=sha(path.read_bytes()))
        parent = f.sessions / "original-index.json"
        parent.write_text('{"original":true}')
        identity = binding(parent)
        f.config.pop("valid_source_ids")
        write_json(f.base / "calibration.json", f.config)
        sources = dict(schema="badminton-calibration-sources-v1", usage="review_only",
            valid_source_ids=[f.sid], sources=[dict(source_id=f.sid, source_index=identity['file'],
            source_index_sha256=identity['sha256'])], production_admitted=False,
            camera_calibration_admitted=False, body_labels_production_admitted=False)
        write_json(f.base / "sources.json", sources)
        hashes = {name: sha((f.base / name).read_bytes()) for name in module._CONFIG_FILES}
        auth = f.base / "review-status.json"
        write_json(auth, dict(schema="badminton-calibration-review-status-v1", usage="review_only",
            camera_pilot_authorized=True, body_labels_production_admitted=False, training_published=False,
            result_files=hashes, user_confirmation="Review these actual short rallies"))
        authority = dict(path=str(auth), sha256=sha(auth.read_bytes()))
        session = SessionIndex.load(f.session)
        session.source_index = dict(candidate_bundle_sha256=hashes, crop_binding=dict(bbox_sha256=hashes['bbox.json']),
            parent_source_index_identity=identity, frame_count=3, frame_map=dict(rows=3),
            source_manifest=binding(f.manifest_path), common_clock=binding(f.clock_path),
            canonical_video_sidecars={cam:binding(f.video/(cam+'.frames.jsonl')) for cam in ('cam1','cam2')})
        index = f.sessions / "derived-index.json"
        index.write_text('{}')
        def freeze(name, **extra):
            return CalibrationSnapshot.freeze(f.project, f.sessions, f.workspace/name, calibration_dir=f.base,
                source_indexes={f.sid:binding(index)}, data_root=f.sessions, **extra)
        with patch.object(source_module.SourceIndexedSession, 'load_source_index', return_value=session):
            result = freeze('portable', review_source_ids=[f.sid], review_authorization=authority)
            self.assertEqual(result.data['configuration_files_sha256'], hashes)
            self.assertEqual(result.data['provenance'], sources)
            self.assertNotIn('accepted_source_calibration_sha256', result.data['provenance'])
            self.assertIn('evidence/review-authorization.json', result.data['artifacts'])
            self.assertEqual(result.frame_map_path(f.sid).read_bytes(), f.expected_map)
            self.assertFalse(result.data['production_admitted'])
            with self.assertRaisesRegex(ValueError, 'accepted_bbox_sha256'):
                freeze('formal')
            with self.assertRaisesRegex(ValueError, 'authorization'):
                freeze('no-authority', review_source_ids=[f.sid])
            session.source_index['parent_source_index_identity'] = dict(identity,sha256='f'*64)
            with self.assertRaisesRegex(ValueError, 'parent differs'):
                freeze('wrong-parent', review_source_ids=[f.sid], review_authorization=authority)
            session.source_index['parent_source_index_identity'] = identity
            auth.write_bytes(auth.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                freeze('changed-authority', review_source_ids=[f.sid], review_authorization=authority)

    def test_canonical_five_files_and_original_source_sha_are_distinct(self):
        snapshot = self.fixture.freeze()
        self.assertEqual(snapshot.data["schema"], module._CANONICAL_SCHEMA)
        self.assertEqual({p.name for p in self.fixture.base.iterdir()},
                         {"calibration.json", "bbox.json", "roi-config.txt", "sources.json", "sync-policy.json"})
        self.assertNotEqual(self.fixture.sources["accepted_source_calibration_sha256"],
                            sha((self.fixture.base / "calibration.json").read_bytes()))
        self.assertEqual(snapshot.frame_map_path(self.fixture.sid).read_bytes(), self.fixture.expected_map)
        np.testing.assert_array_equal(snapshot.Q, self.fixture.q)
        self.assertFalse(snapshot.data["production_admitted"])
        self.assertEqual(snapshot.data["provenance"]["accepted_source_calibration_sha256"], "c"*64)
        self.assertEqual(snapshot.data["configuration_files_sha256"]["calibration.json"],
                         sha((self.fixture.base / "calibration.json").read_bytes()))
        evidence = snapshot.data["sessions"][self.fixture.sid]["source_evidence_sha256"]
        expected_sources = [self.fixture.manifest_path, self.fixture.clock_path,
                            self.fixture.video / "cam1.frames.jsonl", self.fixture.video / "cam2.frames.jsonl"]
        self.assertEqual(evidence, {str(path.resolve()): sha(path.read_bytes()) for path in expected_sources})
        for camera in ("cam1", "cam2"):
            for key in ("K", "distortion_opencv", "R_raw_lidar_to_camera", "t_m"):
                np.testing.assert_array_equal(snapshot.camera(camera)[key], self.fixture.config["cameras"][camera][key])
            self.assertEqual(snapshot.camera(camera)["image_size"], [1920, 1080])
            self.assertEqual(snapshot.camera(camera)["projection_model"], "opencv_pinhole_brown5")

    def test_exact_rounded_q_roundtrip_and_orientation(self):
        snapshot = self.fixture.freeze()
        points = np.array([[1, -.4, 2], [-.5, .1, 5], [3, 4, 30.]])
        viewer = snapshot.camera_to_viewer(points, "cam1")
        np.testing.assert_allclose(snapshot.viewer_to_camera(viewer, "cam1"), points, atol=1e-13)
        camera = snapshot.camera("cam1")
        wrong = viewer @ snapshot.Q @ np.asarray(camera["R_raw_lidar_to_camera"]).T + camera["t_m"]
        self.assertGreater(np.max(np.abs(wrong-points)), 1e-8)
        rotation = snapshot.camera_rotation_to_viewer("cam1")
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(rotation), 1, places=14)

    def test_brown_radial_and_tangential_distortion(self):
        snapshot = self.fixture.freeze()
        viewer = snapshot.camera_to_viewer([[1., -.4, 2.]], "cam1")
        uv, depth = snapshot.project(viewer, "cam1")
        x, y = .5, -.2
        r2 = x*x+y*y
        k1, k2, p1, p2, k3 = snapshot.camera("cam1")["distortion_opencv"]
        radial = 1+k1*r2+k2*r2*r2+k3*r2*r2*r2
        xd = x*radial+2*p1*x*y+p2*(r2+2*x*x)
        yd = y*radial+p1*(r2+2*y*y)+2*p2*x*y
        np.testing.assert_allclose(uv[0], [1000*xd+960, 1100*yd+540], atol=1e-10)
        np.testing.assert_allclose(depth, [2.])
        plain, _ = snapshot.project(viewer, "cam1", distort=False)
        np.testing.assert_allclose(plain[0], [1460, 320], atol=1e-10)

    def test_torch_projection_gradients_and_transform(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is unavailable")
        snapshot = self.fixture.freeze()
        points = torch.tensor([[1., -.4, 2.], [-.2, .3, 4.]], dtype=torch.float64, requires_grad=True)
        viewer = snapshot.camera_to_viewer(points, "cam1")
        torch.testing.assert_close(snapshot.viewer_to_camera(viewer, "cam1"), points, atol=1e-13, rtol=0)
        for distort in (True, False):
            uv, depth = snapshot.project_torch(viewer, "cam1", distort=distort)
            expected, expected_depth = snapshot.project(viewer.detach().numpy(), "cam1", distort=distort)
            np.testing.assert_allclose(uv.detach().numpy(), expected, atol=1e-10)
            np.testing.assert_allclose(depth.detach().numpy(), expected_depth, atol=1e-13)
        uv.sum().backward()
        self.assertTrue(torch.isfinite(points.grad).all().item())
        self.assertGreater(points.grad.abs().sum().item(), 0)

    def test_freeze_is_idempotent_and_later_config_does_not_change_snapshot(self):
        snapshot = self.fixture.freeze()
        digest = snapshot.data["snapshot_digest"]
        self.assertEqual(self.fixture.freeze().data["snapshot_digest"], digest)
        config = copy.deepcopy(self.fixture.config)
        config["revision_note"] = "later canonical publication"
        write_json(self.fixture.base / "calibration.json", config)
        self.assertEqual(CalibrationSnapshot.load(snapshot.snapshot_json).data["snapshot_digest"], digest)
        with self.assertRaises(ValueError):
            self.fixture.freeze()

    def test_published_result_file_hashes_are_bound_to_actual_canonical_bytes(self):
        self.fixture.sources["result_files"] = {
            name: sha((self.fixture.base / name).read_bytes())
            for name in module._CONFIG_FILES if name != "sources.json"
        }
        write_json(self.fixture.base / "sources.json", self.fixture.sources)
        self.fixture.freeze()
        path = self.fixture.base / "calibration.json"
        path.write_bytes(path.read_bytes()+b" ")
        self.fixture.output = self.fixture.workspace / "fresh-snapshot"
        with self.assertRaisesRegex(ValueError, "result file SHA256"):
            self.fixture.freeze()

    def test_source_manifest_change_rejected_on_existing_freeze(self):
        self.fixture.freeze()
        self.fixture.manifest_path.write_bytes(self.fixture.manifest_path.read_bytes()+b" ")
        with self.assertRaises(ValueError):
            self.fixture.freeze()

    def test_source_manifest_race_is_rejected_before_publication(self):
        original = Path.read_bytes
        raced = []
        target = self.fixture.manifest_path.resolve()
        def read_and_change(path):
            blob = original(path)
            if path.resolve() == target and not raced:
                raced.append(True)
                path.write_bytes(blob+b" ")
            return blob
        with patch.object(Path, "read_bytes", read_and_change):
            with self.assertRaises(ValueError):
                self.fixture.freeze()
        self.assertTrue(raced, "Race injection must actually read the native manifest")
        self.assertFalse((self.fixture.output / "calibration_snapshot.json").exists())

    def test_frozen_artifact_tampering_rejected(self):
        snapshot = self.fixture.freeze()
        snapshot.frame_map_path(self.fixture.sid).write_bytes(b"{}\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            CalibrationSnapshot.load(snapshot.snapshot_json)

    def test_missing_camera_and_incomplete_d5_rejected(self):
        path = self.fixture.base / "calibration.json"
        for change in ("camera", "d5"):
            with self.subTest(change=change):
                config = copy.deepcopy(self.fixture.config)
                if change == "camera":
                    del config["cameras"]["cam2"]
                else:
                    config["cameras"]["cam1"]["distortion_opencv"] = [0., 0., 0., 0.]
                write_json(path, config)
                with self.assertRaises((ValueError, KeyError)):
                    self.fixture.freeze()
        self.assertFalse((self.fixture.output / "calibration_snapshot.json").exists())

    def test_source_mismatch_m_rejected(self):
        manifest = json.loads(self.fixture.manifest_path.read_text(encoding="utf-8"))
        manifest["lidar_rotation"] = np.eye(3).reshape(-1).tolist()
        write_json(self.fixture.manifest_path, manifest)
        with self.assertRaises(ValueError):
            self.fixture.freeze()

    def test_embedded_or_nonmidpoint_timing_rejected(self):
        config_path = self.fixture.base / "calibration.json"
        config = copy.deepcopy(self.fixture.config)
        config["timing_policy"] = copy.deepcopy(config["timing_policy"])
        config["timing_policy"]["extra_offset_us"]["cam1"] = 50_000
        write_json(config_path, config)
        with self.assertRaises(ValueError):
            self.fixture.freeze()
        # Even mutually consistent files cannot silently change the adopted
        # scan-midpoint/zero-offset timing contract.
        changed = copy.deepcopy(self.fixture.policy)
        changed["lidar_reference"] = "scan_start"
        config["timing_policy"] = changed
        write_json(config_path, config)
        self.fixture.sources["sync_policy_sha256"] = write_json(self.fixture.base / "sync-policy.json", changed)
        write_json(self.fixture.base / "sources.json", self.fixture.sources)
        with self.assertRaises(ValueError):
            self.fixture.freeze()

    def test_rebuilt_frame_map_sha_is_required(self):
        self.fixture.sources["frame_maps"][0]["sha256"] = "0"*64
        write_json(self.fixture.base / "sources.json", self.fixture.sources)
        with self.assertRaises(ValueError):
            self.fixture.freeze()

    def test_changed_camera_clock_or_sidecar_rejected(self):
        clock = json.loads(self.fixture.clock_path.read_text(encoding="utf-8"))
        clock["tracks"][1]["rows"][1]["common_time_us"] += 100
        write_json(self.fixture.clock_path, clock)
        with self.assertRaises(ValueError):
            self.fixture.freeze()

    def test_native_sidecar_byte_change_rejected_on_existing_freeze(self):
        self.fixture.freeze()
        path = self.fixture.video / "cam1.frames.jsonl"
        path.write_bytes(path.read_bytes()+b"\n")
        with self.assertRaises(ValueError):
            self.fixture.freeze()

    def test_bbox_and_sync_file_sha_are_required(self):
        for name in ("bbox.json", "sync-policy.json"):
            with self.subTest(file=name):
                path = self.fixture.base / name
                previous = path.read_bytes()
                path.write_bytes(previous+b" ")
                with self.assertRaises(ValueError):
                    self.fixture.freeze()
                path.write_bytes(previous)

    def test_reference_plane_is_bound_to_court_registration(self):
        config = copy.deepcopy(self.fixture.config)
        config["court_reference_plane"]["origin_viewer_m"][2] += .2
        write_json(self.fixture.base / "calibration.json", config)
        with self.assertRaises(ValueError):
            self.fixture.freeze()
        self.assertFalse((self.fixture.output / "calibration_snapshot.json").exists())

    def test_legacy_v1_bundle_remains_loadable_without_live_source(self):
        snapshot = self.fixture.freeze()
        data = copy.deepcopy(snapshot.data)
        data["schema"] = module._SCHEMA
        data["active_selection"] = {"calibration_sha256": self.fixture.sources["accepted_source_calibration_sha256"]}
        data["reference_plane"]["note"] = "Legacy corner-correction development plane description"
        data["reference_plane"]["source"] = "Legacy one-metric-plane note; numerical geometry unchanged"
        data["geometry_variant"] = "corner_fix_shared_plane"
        payload = copy.deepcopy(data)
        payload.pop("snapshot_digest", None)
        data["snapshot_digest"] = sha(module._canonical(payload))
        legacy_path = snapshot.snapshot_json
        write_json(legacy_path, data)
        frozen_bytes = legacy_path.read_bytes()
        # A frozen bundle must not need source-root .local or current config.
        with patch.object(CalibrationSnapshot, "freeze", side_effect=AssertionError("load invoked freeze")):
            legacy = CalibrationSnapshot.load(legacy_path)
        self.assertEqual(legacy.data["schema"], module._SCHEMA)
        np.testing.assert_array_equal(legacy.Q, snapshot.Q)
        np.testing.assert_allclose(legacy.project([[0., 0., 1.]], "cam1")[0],
                                   snapshot.project([[0., 0., 1.]], "cam1")[0])
        # Canonical republishing changes description/packaging, not the
        # accepted camera/plane/source geometry or the old frozen run identity.
        reused = self.fixture.freeze()
        self.assertEqual(reused.data["snapshot_digest"], data["snapshot_digest"])
        self.assertEqual(legacy_path.read_bytes(), frozen_bytes)

    def test_output_cannot_escape_workspace(self):
        with self.assertRaisesRegex(ValueError, "inside the GVHMR"):
            CalibrationSnapshot.freeze(self.fixture.project, self.fixture.sessions, Path(self.temp.name) / "elsewhere")


if __name__ == "__main__":
    unittest.main()
