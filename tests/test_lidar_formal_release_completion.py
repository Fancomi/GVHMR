"""Small CPU fixtures for formal completion, exact teacher reuse and typed readers."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from hmr4d.lidar_mocap import dual_view_observer as observer
from hmr4d.lidar_mocap import formal_completion
from hmr4d.lidar_mocap import new_source_completion as completion
from hmr4d.lidar_mocap import rally_viewer
from tools import audit_badminton_expanded_pixels as pixels
from tools import render_badminton_continuity_cases as continuity
from tools import recheck_badminton_expanded_pixels as recheck
from test_lidar_formal_baseline import stage, read
from test_lidar_formal_completion import pending, finished


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return dict(path=str(path), sha256=completion.file_sha(path))


@pytest.fixture
def completed(finished):
    result = finished.verify()
    labels = finished.repair / "final_labels"
    put(labels / "source_registry.json", dict(schema="badminton-formal-derived-source-registry-v1"))
    finished.cpu_path = finished.repair / "evaluation/formal_postleg_cpu_receipt.json"
    finished.cpu_sha = completion.file_sha(finished.cpu_path)
    finished.cpu_receipt = result
    return finished


def finish(stage, **kwargs):
    return completion.finish_new_sources(
        stage.repair,
        formal_postleg_cpu_receipt=stage.cpu_path,
        formal_postleg_cpu_receipt_sha256=stage.cpu_sha,
        **kwargs,
    )


def test_formal_pending_reuses_real_strict_checks_without_new_workers_or_qa_writes(completed, monkeypatch):
    snapshot = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in [completed.cpu_path, *(completed.repair / "evaluation").glob("v2_*.json")]
    }
    monkeypatch.setattr(completion.subprocess, "run", lambda *a, **k: pytest.fail("No subprocess may start"))
    result = finish(completed)
    assert result["state"] == "bilateral_observation_pending"
    assert result["observer_started"] is False and result["training_published"] is False
    assert completed.native_calls == ["native", "dataset"]
    assert not (completed.repair / "new_source_completion_receipt.json").exists()
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == old for p, old in snapshot.items())


@pytest.mark.parametrize(
    "change", ["sha", "missing", "foreign_path", "partial", "root", "windows", "camera", "input", "published", "worker"]
)
def test_invalid_explicit_receipt_never_reaches_a_worker(completed, monkeypatch, change):
    value = deepcopy(completed.cpu_receipt)
    if change == "sha":
        completed.cpu_sha = "0" * 64
    elif change == "missing":
        completed.cpu_path.unlink()
    elif change == "foreign_path":
        completed.cpu_path = completed.repair / "foreign.json"
        completed.cpu_sha = put(completed.cpu_path, value)["sha256"]
    else:
        if change == "partial":
            value["state"] = "running"
        elif change == "root":
            value["root"] = str(completed.root.parent / "another_formal")
        elif change == "windows":
            value["windows"] += 1
        elif change == "camera":
            value["calibration_digest"] = "f" * 64
        elif change == "input":
            value["source_input"]["sha256"] = "f" * 64
        elif change == "published":
            value["training_published"] = True
        elif change == "worker":
            value["workers"].pop("native")
        completed.cpu_sha = put(completed.cpu_path, value)["sha256"]
    monkeypatch.setattr(
        formal_completion, "verify_formal_postleg_cpu", lambda *a, **k: pytest.fail("Invalid receipt reached verifier")
    )
    with pytest.raises(ValueError):
        finish(completed)
    assert completed.native_calls == ["native", "dataset"]


def test_changed_bound_cpu_evidence_is_rejected_by_original_strict_verifier(completed):
    evidence = completed.cpu_receipt["evidence"]["comparison"]["path"]
    Path(evidence).write_bytes(b"{}")
    with pytest.raises(ValueError, match="evidence changed"):
        finish(completed)
    assert completed.native_calls == ["native", "dataset"]


def test_formal_cannot_fall_back_to_legacy_without_explicit_receipt(completed):
    with pytest.raises(ValueError, match="require their explicit"):
        completion.finish_new_sources(completed.repair)
    with pytest.raises(ValueError, match="supplied together"):
        completion.finish_new_sources(completed.repair, formal_postleg_cpu_receipt=completed.cpu_path)


@pytest.mark.skipif(os.name != "nt", reason="Windows receipt-handle deletion semantics")
def test_receipt_cannot_disappear_while_original_verifier_selects_reuse(completed, monkeypatch):
    original = formal_completion.verify_formal_postleg_cpu

    def checked(*args, **kwargs):
        with pytest.raises(PermissionError):
            completed.cpu_path.unlink()
        return original(*args, **kwargs)

    monkeypatch.setattr(formal_completion, "verify_formal_postleg_cpu", checked)
    assert finish(completed)["state"] == "bilateral_observation_pending"
    assert completed.native_calls == ["native", "dataset"]


def observations(stage, completed=True):
    labels = stage.repair / "final_labels"
    paired = stage.repair.with_name("repair_paired")
    path = paired / "observations/manifest.json"
    value = dict(
        completed=completed,
        source_accepted=stage.cpu_receipt["accepted"],
        identity=dict(
            calibration_digest=stage.cpu_receipt["calibration_digest"],
            source=dict(
                dataset_manifest=dict(
                    path=str(labels / "datasets/dataset_manifest.json"),
                    sha256=completion.file_sha(labels / "datasets/dataset_manifest.json"),
                )
            ),
        ),
    )
    put(path, value)
    return paired, path, value


@pytest.mark.parametrize("change", ["partial", "count", "camera", "source"])
def test_incomplete_or_foreign_bilateral_observations_cannot_publish(completed, monkeypatch, change):
    paired, path, value = observations(completed, completed=change != "partial")
    if change == "count":
        value["source_accepted"] += 1
    elif change == "camera":
        value["identity"]["calibration_digest"] = "f" * 64
    elif change == "source":
        value["identity"]["source"]["dataset_manifest"]["sha256"] = "f" * 64
    put(path, value)
    monkeypatch.setattr(completion.subprocess, "run", lambda *a, **k: pytest.fail("Observer/export must not start"))
    if change == "partial":
        assert finish(completed)["state"] == "bilateral_observation_pending"
    else:
        with pytest.raises(ValueError, match="another formal"):
            finish(completed)
    assert not (completed.repair / "new_source_completion_receipt.json").exists()


@pytest.mark.parametrize("already_published", [False, True])
@pytest.mark.parametrize("change_during_viewer", [False, True])
def test_formal_bilateral_completion_delegates_export_and_keeps_training_pending(
    completed, monkeypatch, already_published, change_during_viewer
):
    labels = completed.repair / "final_labels"
    paired, observed, _ = observations(completed)
    manifest_path = paired / "datasets/dataset_manifest.json"
    publication = dict(
        status="complete",
        accepted=2,
        source_dataset_manifest=str(labels / "datasets/dataset_manifest.json"),
        source_dataset_manifest_sha256=completion.file_sha(labels / "datasets/dataset_manifest.json"),
        calibration_digest=completed.cpu_receipt["calibration_digest"],
        full_observation_manifest=str(observed),
        full_observation_manifest_sha256=completion.file_sha(observed),
    )
    if already_published:
        put(manifest_path, publication)
    commands = []

    def run(command, **kwargs):
        assert Path(command[2]).name == "export_badminton_dual_view.py"
        commands.append(command[3])
        if command[3] == "finalize":
            put(manifest_path, publication)
        put(paired / "verification.json", dict(accepted=2, dataset_manifest_sha256=completion.file_sha(manifest_path)))
        return SimpleNamespace(returncode=0)

    resolver = object()
    loader = Mock(return_value=resolver)
    monkeypatch.setattr(completion, "load_pilot_run_relocations", loader)
    monkeypatch.setattr(completion.subprocess, "run", run)

    def viewer(root, baseline, *, relocations):
        assert root == baseline == labels and relocations is resolver
        put(labels / "viewer/manifest.json", dict(source_count=2))
        if change_during_viewer:
            Path(completed.cpu_receipt["evidence"]["leg_receipt"]["path"]).write_bytes(b"{}")

    monkeypatch.setattr(rally_viewer, "prepare_viewer", viewer)
    if change_during_viewer:
        with pytest.raises(ValueError, match="evidence changed before bilateral"):
            finish(completed)
        assert not (completed.repair / "new_source_completion_receipt.json").exists()
        return
    result = finish(completed)
    assert commands == (["verify"] if already_published else ["finalize", "verify"])
    assert result["state"] == "labels_published_viewer_prepared"
    assert result["training_published"] is False and result["bilateral_labels_published"] is True
    assert result["archive_published"] is False and result["consumer_admitted"] is False
    assert result["browser_playback_verified"] is False
    assert result["evidence"][str(completed.cpu_path)] == completed.cpu_sha
    loader.assert_called_once_with(labels)
    assert completed.native_calls == ["native", "dataset"]


def test_legacy_finish_retains_its_native_dataset_and_training_receipt_behavior(completed, monkeypatch):
    labels = completed.repair / "final_labels"
    put(labels / "source_registry.json", dict(schema="legacy"))
    paired = completed.repair.with_name("repair_paired")
    put(paired / "observations/manifest.json", dict(completed=True))
    calls = []

    def run(command, **kwargs):
        calls.append((Path(command[2]).name, command[3]))
        if Path(command[2]).name == "export_badminton_dual_view.py" and command[3] == "finalize":
            put(paired / "datasets/dataset_manifest.json", dict(accepted=2))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(completion.subprocess, "run", run)
    monkeypatch.setattr(completion, "load_pilot_run_relocations", lambda *args: None)
    monkeypatch.setattr(rally_viewer, "prepare_viewer", lambda *args, **kwargs: None)
    result = completion.finish_new_sources(completed.repair)
    assert result["training_published"] is True and "formal_postleg_cpu_receipt" not in result
    assert [name for name, _ in calls] == [
        "verify_badminton_native_repair.py",
        "verify_badminton_dataset.py",
        "export_badminton_dual_view.py",
        "export_badminton_dual_view.py",
    ]


def test_legacy_source_cannot_consume_a_formal_postleg_receipt(completed):
    put(completed.repair / "final_labels/source_registry.json", dict(schema="legacy"))
    with pytest.raises(ValueError, match="legacy inputs cannot reuse"):
        finish(completed)


@pytest.fixture
def teachers(tmp_path):
    source = tmp_path / "repair/final_labels"
    sid, window = "source", "window"
    directory = source / "sessions" / sid / "windows" / window
    primary = source.parent / "initial_teacher/sessions" / sid / "windows" / window
    secondary = source.parent / "secondary_teachers/cam1/sessions" / sid / "windows" / window
    config = dict(
        settings=dict(camera="cam2"), calibration_digest="current", body_repair=dict(source_window=str(primary))
    )
    put(directory / "window.json", config)
    calibration = SimpleNamespace(
        data=dict(snapshot_digest="current"), camera=lambda camera: dict(name=camera, K="exact")
    )
    rows = {
        camera: [
            dict(record_index=i, media_pts_us=100 + i, common_time_us=1000 + i, source_frame_no=10 + i)
            for i in range(3)
        ]
        for camera in ("cam1", "cam2")
    }
    video_paths = {}
    paths = {}
    for camera, target in (("cam1", secondary), ("cam2", primary)):
        video = tmp_path / (camera + ".fixture")
        video.write_bytes(b"synthetic stat identity; never decoded")
        video_paths[camera] = video
        stat = video.stat()
        provenance = dict(
            camera=calibration.camera(camera),
            frame_rows=[rows[camera][0], rows[camera][2]],
            video=dict(path=str(video.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns),
        )
        digest = hashlib.sha256(observer._canonical(provenance)).hexdigest()
        teacher = target / "teacher"
        put(
            teacher / "manifest.json",
            dict(status="complete", schema="synthetic-teacher", provenance=provenance, provenance_hash=digest),
        )
        torch.save(
            dict(
                schema="synthetic-teacher",
                provenance_hash=digest,
                detections=[[dict(bbx_xyxy=[1, 2, 3, 4], detection_confidence=0.9)], []],
            ),
            teacher / "detections.pt",
        )
        paths[camera] = teacher
    session = SimpleNamespace(video_paths=video_paths, _camera_rows=rows)
    return SimpleNamespace(
        source=source,
        sid=sid,
        window=window,
        directory=directory,
        primary=primary,
        secondary=secondary,
        paths=paths,
        calibration=calibration,
        session=session,
        dataset=dict(selected_windows={sid: [window]}, exports={sid: dict(accepted=1)}),
    )


@pytest.mark.parametrize("camera", ["cam1", "cam2"])
def test_initial_and_secondary_teachers_reuse_only_exact_observed_records(teachers, camera):
    result = observer.teacher_detections(
        teachers.source, teachers.dataset, teachers.sid, camera, teachers.calibration, teachers.session
    )
    assert set(result) == {0, 2}
    assert result[0]["detections"][0]["detection_confidence"] == 0.9
    assert result[2]["detections"] == []
    assert 1 not in result  # No nearest-frame or interpolation admission.
    assert result[0]["evidence"]["cache_path"] == str((teachers.paths[camera] / "detections.pt").resolve())


@pytest.mark.parametrize(
    "change", ["camera", "calibration", "video", "time", "pts", "source_frame", "provenance", "partial", "duplicate"]
)
def test_wrong_teacher_bindings_are_rejected(teachers, change):
    path = teachers.paths["cam1"] / "manifest.json"
    manifest = read(path)
    if change == "camera":
        manifest["provenance"]["camera"]["name"] = "cam2"
    elif change == "calibration":
        config = read(teachers.directory / "window.json")
        config["calibration_digest"] = "another"
        put(teachers.directory / "window.json", config)
    elif change == "video":
        manifest["provenance"]["video"]["size"] += 1
    elif change in ("time", "pts", "source_frame"):
        field = dict(time="common_time_us", pts="media_pts_us", source_frame="source_frame_no")[change]
        teachers.session._camera_rows["cam1"][0][field] += 1
    elif change == "partial":
        manifest["status"] = "running"
    elif change == "duplicate":
        manifest["provenance"]["frame_rows"][1] = deepcopy(manifest["provenance"]["frame_rows"][0])
    digest = hashlib.sha256(observer._canonical(manifest["provenance"])).hexdigest()
    manifest["provenance_hash"] = "f" * 64 if change == "provenance" else digest
    put(path, manifest)
    cache_path = teachers.paths["cam1"] / "detections.pt"
    cache = torch.load(cache_path, map_location="cpu")
    cache["provenance_hash"] = digest
    torch.save(cache, cache_path)
    with pytest.raises(ValueError):
        observer.teacher_detections(
            teachers.source, teachers.dataset, teachers.sid, "cam1", teachers.calibration, teachers.session
        )


def test_missing_secondary_cache_has_no_fabricated_detection(teachers):
    (teachers.paths["cam1"] / "detections.pt").unlink()
    assert (
        observer.teacher_detections(
            teachers.source, teachers.dataset, teachers.sid, "cam1", teachers.calibration, teachers.session
        )
        == {}
    )


def test_teacher_source_cannot_escape_its_selected_repair_window(teachers):
    config = read(teachers.directory / "window.json")
    config["body_repair"]["source_window"] = str(teachers.source.parent.parent / "foreign")
    put(teachers.directory / "window.json", config)
    with pytest.raises(ValueError, match="leaves the declared"):
        observer.teacher_detections(
            teachers.source, teachers.dataset, teachers.sid, "cam2", teachers.calibration, teachers.session
        )


def formal_teacher_inputs(teachers):
    source = teachers.source
    snapshot = put(source / "calibration/calibration_snapshot.json", teachers.calibration.data)
    index = dict(path=str(source / "admitted-index.json"), sha256="a" * 64)
    registry = put(
        source / "source_registry.json",
        dict(schema="badminton-formal-derived-source-registry-v1", sources={teachers.sid: dict(source_index=index)}),
    )
    proof_path = source.parent.parent / "formal_input_equivalence.json"
    proof = dict(
        schema="badminton-review-formal-input-equivalence-v1",
        state="complete",
        training_published=False,
        fitted_tensors_imported=False,
        evidence=dict(formal_registry=registry, formal_calibration=snapshot),
        sources={
            teachers.sid: dict(
                original_inputs_exact=True, cameras_court_ground_bbox_clock_exact=True, admitted_index=index
            )
        },
    )
    binding = put(proof_path, proof)
    prepared = dict(
        schema="badminton-new-source-repair-input-v1",
        calibration_digest="current",
        source_registry_sha256=registry["sha256"],
        formal_derivation=dict(input_equivalence=binding),
    )
    input_binding = put(source.parent / "new_source_input.json", prepared)
    put(
        source / "repair_plan.json",
        dict(
            source_union=input_binding["path"],
            source_union_sha256=input_binding["sha256"],
            calibration_digest="current",
        ),
    )
    return proof_path, proof


def test_formal_teacher_cache_retains_existing_physical_input_mapping(teachers):
    formal_teacher_inputs(teachers)
    result = observer.teacher_detections(
        teachers.source, teachers.dataset, teachers.sid, "cam1", teachers.calibration, teachers.session
    )
    proof = result[0]["evidence"]["formal_input_equivalence"]
    assert proof["proof"]["path"] == str(teachers.source.parent.parent / "formal_input_equivalence.json")


@pytest.mark.parametrize("change", ["missing", "stale", "camera", "physical", "index", "schema", "incomplete"])
def test_formal_teacher_cache_cannot_reuse_unproved_equivalence(teachers, change):
    path, proof = formal_teacher_inputs(teachers)
    if change == "missing":
        path.unlink()
    elif change == "camera":
        proof["evidence"]["formal_calibration"]["sha256"] = "f" * 64
    elif change == "physical":
        proof["sources"][teachers.sid]["cameras_court_ground_bbox_clock_exact"] = False
    elif change == "index":
        proof["sources"][teachers.sid]["admitted_index"]["sha256"] = "f" * 64
    elif change == "schema":
        proof["schema"] = "unproved"
    elif change == "incomplete":
        proof["state"] = "running"
    if change not in ("missing", "stale"):
        binding = put(path, proof)
        prepared_path = teachers.source.parent / "new_source_input.json"
        prepared = read(prepared_path)
        prepared["formal_derivation"]["input_equivalence"] = binding
        input_binding = put(prepared_path, prepared)
        plan_path = teachers.source / "repair_plan.json"
        plan = read(plan_path)
        plan["source_union_sha256"] = input_binding["sha256"]
        put(plan_path, plan)
    elif change == "stale":
        path.write_text("{}", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        observer.teacher_detections(
            teachers.source, teachers.dataset, teachers.sid, "cam1", teachers.calibration, teachers.session
        )


def test_source_selection_binds_both_actual_initial_detector_caches(teachers):
    source = teachers.source
    tensor = source / "datasets/source.fixture"
    tensor.parent.mkdir(parents=True, exist_ok=True)
    tensor.write_bytes(b"synthetic tensor identity only")
    metadata = source / "datasets/source.jsonl"
    metadata.write_bytes(b"{}\n")
    dataset = dict(teachers.dataset, calibration_digest="current")
    dataset["exports"][teachers.sid].update(path=str(tensor), metadata=str(metadata))
    put(source / "datasets/dataset_manifest.json", dataset)
    put(source / "calibration/calibration_snapshot.json", teachers.calibration.data)
    (teachers.directory / "quality.jsonl").write_bytes(b"{}\n")
    _, evidence = observer.source_selection(source)
    bound = {entry["path"] for entry in evidence["files"]}
    assert all(
        str(path / name) in bound for path in teachers.paths.values() for name in ("manifest.json", "detections.pt")
    )
    assert str(teachers.directory / "teacher/detections.pt") not in bound


def test_observer_loads_one_typed_resolver_for_every_session(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "observations"
    source.mkdir()
    output.mkdir()
    video = tmp_path / "stat.fixture"
    video.write_bytes(b"not a video")
    dataset = dict(calibration_digest="camera", exports={"a": dict(accepted=0), "b": dict(accepted=0)}, accepted=0)
    calibration = SimpleNamespace(data=dict(snapshot_digest="camera"))
    monkeypatch.setattr(observer, "_bound_output", lambda *args: (source, output))
    monkeypatch.setattr(observer, "source_selection", lambda *args: (dataset, {}))
    monkeypatch.setattr(observer.CalibrationSnapshot, "load", lambda *args: calibration)
    resolver = object()
    loader = Mock(return_value=resolver)
    monkeypatch.setattr(observer, "load_pilot_run_relocations", loader)
    monkeypatch.setattr(observer, "_formal_teacher_proof", lambda *args: None)
    monkeypatch.setattr(observer, "selected_frames", lambda *args: [])
    monkeypatch.setattr(observer, "teacher_detections", lambda *args, **kwargs: {})
    session = SimpleNamespace(
        video_paths={cam: video for cam in ("cam1", "cam2")}, iter_video_frames=lambda *a: (row for row in ())
    )
    sessions = Mock(return_value=session)
    monkeypatch.setattr(observer, "load_run_session", sessions)

    class Checker:
        def __init__(self, *args, **kwargs):
            self.models, self.sessions, self.camera = {}, {}, {}

        def select_camera(self, camera):
            self.camera = dict(name=camera)

        def session(self, sid):
            return self.sessions[sid]

    monkeypatch.setattr(observer, "RGBKeypointChecker", Checker)
    assert observer.observe_all(source, output, device="cpu", threads=1)["completed"] is True
    loader.assert_called_once_with(source, calibration.data)
    assert sessions.call_count == 2
    assert all(call.kwargs == dict(relocations=resolver) for call in sessions.call_args_list)


def test_continuity_renderer_passes_its_typed_resolver_before_reading_pixels(tmp_path, monkeypatch):
    root = tmp_path / "repair"
    put(root / "continuity_audit/audit.json", dict(pelvis_jump_candidates=[dict(source_id="s")]))
    calibration = SimpleNamespace(data={})
    monkeypatch.setattr(continuity.CalibrationSnapshot, "load", lambda *args: calibration)
    resolver = object()
    loader = Mock(return_value=resolver)
    monkeypatch.setattr(continuity, "load_pilot_run_relocations", loader)
    session = Mock(side_effect=RuntimeError("stop before assets"))
    monkeypatch.setattr(continuity, "load_run_session", session)
    with pytest.raises(RuntimeError, match="stop before assets"):
        continuity.render(root)
    loader.assert_called_once_with(root / "final_labels", calibration.data)
    session.assert_called_once_with(root / "final_labels", calibration, "s", relocations=resolver)


def native_row(sid, digest, report, pixel_digest):
    native = dict(record_index=0, media_pts_us=100, common_time_us=1000, source_frame_no=10)
    view = dict(
        native=native, raw_image_sha256=pixel_digest, report_path=str(report), report_sha256=completion.file_sha(report)
    )
    return dict(
        session_id=sid,
        lidar_index=0,
        calibration_digest=digest,
        motion_id="m",
        cameras={cam: deepcopy(native) for cam in ("cam1", "cam2")},
        dualcam=dict(views={cam: deepcopy(view) for cam in ("cam1", "cam2")}),
    )


def test_expanded_pixel_audit_reuses_one_typed_session_for_both_cameras(tmp_path, monkeypatch):
    bgr = np.zeros((2, 3, 3), np.uint8)
    digest = hashlib.sha256(bgr.tobytes()).hexdigest()
    report = tmp_path / "report.json"
    put(report, {})
    row = native_row("s", "cal", report, digest)
    metadata = tmp_path / "test.jsonl"
    metadata.write_text(json.dumps(row) + "\n", encoding="utf-8")
    calibration_path = tmp_path / "calibration.json"
    cal_binding = put(calibration_path, {})
    source_run = tmp_path / "run"
    registry = dict(
        s=dict(
            role="test",
            calibration_digest="cal",
            calibration_snapshot=str(calibration_path),
            calibration_snapshot_sha256=cal_binding["sha256"],
            source_run=str(source_run),
        )
    )
    assembly_path = tmp_path / "assembly/assembly.json"
    put(
        assembly_path,
        dict(
            schema="badminton-expanded-assembly-v1",
            training_published=False,
            source_publications={},
            source_registry=registry,
            suggested_session_splits=dict(
                test=dict(metadata=str(metadata), metadata_sha256=completion.file_sha(metadata), accepted=1)
            ),
        ),
    )
    cal = SimpleNamespace(data=dict(snapshot_digest="cal"))
    monkeypatch.setattr(pixels.CalibrationSnapshot, "load", lambda *args: cal)
    resolver = object()
    loader = Mock(return_value=resolver)
    monkeypatch.setattr(pixels, "load_pilot_run_relocations", loader)
    session = SimpleNamespace(
        camera_rows=lambda camera: [row["cameras"][camera]], iter_video_frames=lambda *args: iter([(0, bgr)])
    )
    load_session = Mock(return_value=session)
    monkeypatch.setattr(pixels, "load_run_session", load_session)
    result = pixels.audit(assembly_path, tmp_path / "pixel_audit")
    assert result["checked_camera_frames"] == result["expected_camera_frames"] == 2
    loader.assert_called_once_with(source_run, cal.data)
    load_session.assert_called_once_with(source_run, cal, "s", relocations=resolver)


def test_pixel_recheck_reuses_one_typed_resolver_across_formal_sources(tmp_path, monkeypatch):
    report = tmp_path / "stop_report.json"
    put(report, {})
    bgr = np.zeros((2, 3, 3), np.uint8)
    digest = hashlib.sha256(bgr.tobytes()).hexdigest()
    rows = [native_row(sid, "cal", report, digest) for sid in ("a", "b")]
    metadata = tmp_path / "test.jsonl"
    metadata.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    native_pixels = tmp_path / "native.jsonl"
    records = [
        dict(
            scene_id=row["session_id"],
            camera=cam,
            native=row["cameras"][cam],
            actual_raw_bgr_sha256=digest,
            expected="old",
        )
        for row in rows
        for cam in ("cam1", "cam2")
    ]
    native_pixels.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    calibration = tmp_path / "calibration.json"
    put(calibration, {})
    run = tmp_path / "source_run"
    assembly_path = tmp_path / "assembly.json"
    assembly = dict(
        source_registry={
            sid: dict(calibration_snapshot=str(calibration), calibration_digest="cal", source_run=str(run))
            for sid in ("a", "b")
        }
    )
    assembly_binding = put(assembly_path, assembly)
    audit_path = tmp_path / "audit.json"
    put(
        audit_path,
        dict(
            assembly=assembly_binding,
            metadata=dict(path=str(metadata), sha256=completion.file_sha(metadata)),
            native_pixels=dict(path=str(native_pixels), sha256=completion.file_sha(native_pixels)),
            checked_camera_frames=4,
            expected_camera_frames=4,
            unknown_differences=records,
            corrections=[],
        ),
    )
    cal = SimpleNamespace(data=dict(snapshot_digest="cal"))
    monkeypatch.setattr(recheck.CalibrationSnapshot, "load", lambda *args: cal)
    resolver = object()
    loader = Mock(return_value=resolver)
    monkeypatch.setattr(recheck, "load_pilot_run_relocations", loader)
    video = tmp_path / "video.fixture"
    video.write_bytes(b"not a video")
    session = SimpleNamespace(
        video_paths={cam: video for cam in ("cam1", "cam2")}, iter_video_frames=lambda *args: iter([(0, bgr)])
    )
    sessions = Mock(return_value=session)
    monkeypatch.setattr(recheck, "load_run_session", sessions)
    original = recheck.check_binding

    def check(entry):
        if entry["path"] == str(report):
            raise RuntimeError("stop before detector")
        return original(entry)

    monkeypatch.setattr(recheck, "check_binding", check)
    monkeypatch.setattr(recheck, "require_gpu_lease", lambda *args: pytest.fail("CPU cannot request a GPU lease"))
    monkeypatch.setattr(recheck, "RGBKeypointChecker", lambda *args: pytest.fail("No real model in this fixture"))
    with pytest.raises(RuntimeError, match="stop before detector"):
        recheck.recheck(audit_path, tmp_path / "recheck", tmp_path / "unused_cpu_lease", "cpu")
    loader.assert_called_once_with(run, cal.data)
    assert sessions.call_count == 2
    assert all(call.kwargs == dict(relocations=resolver) for call in sessions.call_args_list)
