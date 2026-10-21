"""Runtime relocation stays explicit and preserves sealed run/camera artifacts."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.relocations import load_run_relocations, validate_run_relocations
from hmr4d.lidar_mocap.rally_pipeline import audit_legacy_session_paths, load_run_session
from hmr4d.lidar_mocap.rally_viewer import prepare_viewer, write_viewer_runtime_relocations

from test_lidar_relocations import SID, manifest, put


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def runtime_fixture(root):
    data = root / "DATA"
    data.mkdir()
    resolver = manifest(data, {})
    run = root / "run"
    index = json_file(data / "index.json", {})
    index_sha = hashlib.sha256(index.read_bytes()).hexdigest()
    registry = dict(
        data_root=str(data),
        sources={SID: dict(baseline=False, source_index=dict(path=str(index), sha256=index_sha))},
        intake=dict(path=str(run / "intake.json")),
    )
    registry_path = json_file(run / "source_registry.json", registry)
    calibration_data = dict(snapshot_digest="camera", sessions_root=str(data / "sessions"), bbox=dict(boxes=[]))
    snapshot_path = json_file(run / "calibration/calibration_snapshot.json", calibration_data)
    calibration = SimpleNamespace(data=calibration_data, snapshot_json=snapshot_path, reference_plane={})
    rows = [dict(record_index=i, common_time_us=1000 + i * 1000, media_pts_us=i * 1000) for i in range(2)]
    session = SimpleNamespace(
        source_index_sha256=index_sha,
        source_index_path=index,
        source_index_logical_path=index,
        camera_rows=lambda camera: rows,
        pairs=lambda: [dict(scan_mid_us=row["common_time_us"]) for row in rows],
        video_paths={camera: data / (camera + ".mp4") for camera in ("cam1", "cam2")},
    )
    return run, resolver, registry, calibration, session


def test_run_loader_passes_only_the_explicit_resolver_and_preserves_camera_gate(tmp_path):
    run, resolver, registry, calibration, session = runtime_fixture(tmp_path)
    with patch(
        "hmr4d.lidar_mocap.rally_pipeline.SourceIndexedSession.load_source_index", return_value=session
    ) as loader:
        assert load_run_session(run, calibration, SID, relocations=resolver) is session
        assert loader.call_args.kwargs["relocations"] is resolver
        assert loader.call_args.kwargs["require_admitted"] is True
        assert loader.call_args.kwargs["expected_sha256"] == session.source_index_sha256
        assert load_run_session(run, calibration, SID) is session
        assert loader.call_args.kwargs["relocations"] is None


@pytest.mark.parametrize(
    "change", ["data_root", "camera_root", "registry_binding", "camera_binding", "missing_resolver"]
)
def test_wrong_runtime_root_or_binding_is_rejected_before_source_loading(tmp_path, change):
    run, resolver, registry, calibration, _ = runtime_fixture(tmp_path)
    if change == "data_root":
        registry["data_root"] = str(tmp_path / "other")
    elif change == "camera_root":
        calibration.data["sessions_root"] = str(tmp_path / "other/sessions")
    elif change == "registry_binding":
        registry["data_relocations"] = dict(resolver.binding, sha256="0" * 64)
    elif change == "camera_binding":
        calibration.data["data_relocations"] = dict(resolver.binding, bytes=0)
    else:
        registry["data_relocations"] = dict(resolver.binding)
        resolver = None
    json_file(run / "source_registry.json", registry)
    with patch("hmr4d.lidar_mocap.rally_pipeline.SourceIndexedSession.load_source_index") as loader:
        with pytest.raises(ValueError):
            load_run_session(run, calibration, SID, relocations=resolver)
        loader.assert_not_called()


def test_identical_declared_binding_is_accepted_without_rewriting_registry(tmp_path):
    run, resolver, registry, calibration, _ = runtime_fixture(tmp_path)
    registry["data_relocations"] = calibration.data["data_relocations"] = dict(resolver.binding)
    validate_run_relocations(registry, calibration.data, resolver)
    original = (run / "source_registry.json").read_bytes()
    loaded = load_run_relocations(run, resolver.root / resolver.binding["file"], resolver.binding["sha256"])
    assert dict(loaded.binding) == dict(resolver.binding)
    assert (run / "source_registry.json").read_bytes() == original
    assert load_run_relocations(tmp_path / "missing-run") is None
    with pytest.raises(ValueError, match="together"):
        load_run_relocations(tmp_path / "missing-run", manifest=tmp_path / "missing.json")


def test_runtime_receipt_binds_sealed_files_without_changing_them(tmp_path):
    run, resolver, registry, calibration, session = runtime_fixture(tmp_path)
    before = {path: path.read_bytes() for path in (run / "source_registry.json", calibration.snapshot_json)}
    output = tmp_path / "independent-viewer"
    binding = write_viewer_runtime_relocations(run, calibration, resolver, {SID: session}, output)
    receipt = json.loads(Path(binding["path"]).read_bytes())
    assert receipt["data_relocations"] == dict(resolver.binding)
    assert receipt["readers"][SID]["source_index_sha256"] == session.source_index_sha256
    for key, path in (
        ("source_registry", run / "source_registry.json"),
        ("calibration_snapshot", calibration.snapshot_json),
    ):
        assert receipt[key]["sha256"] == hashlib.sha256(before[path]).hexdigest()
        assert path.read_bytes() == before[path]
    assert receipt["original_data_modified"] is False
    assert not (run / "viewer").exists()


def test_changed_registry_or_snapshot_cannot_receive_a_successful_runtime_receipt(tmp_path):
    run, resolver, registry, calibration, session = runtime_fixture(tmp_path)
    registry["sources"][SID]["source_index"]["sha256"] = "0" * 64
    json_file(run / "source_registry.json", registry)
    with pytest.raises(ValueError, match="registry changed"):
        write_viewer_runtime_relocations(run, calibration, resolver, {SID: session}, tmp_path / "new-viewer")
    assert not (tmp_path / "new-viewer").exists()
    json_file(calibration.snapshot_json, dict(calibration.data, snapshot_digest="changed"))
    with pytest.raises(ValueError, match="snapshot changed"):
        write_viewer_runtime_relocations(run, calibration, resolver, {SID: session}, tmp_path / "new-viewer")


def test_prepare_viewer_uses_separate_output_and_explicit_source_resolver(tmp_path):
    run, resolver, registry, calibration, session = runtime_fixture(tmp_path)
    ranges = run / "ranges.jsonl"
    ranges.write_text(
        json.dumps(
            dict(
                source_id=SID,
                source_rally_index=1,
                review_index=1,
                original_rally_index=1,
                first_scan_mid_us=1000,
                last_scan_mid_us=2000,
                source_begin_frame=0,
                source_frame_count=2,
            )
        ),
        encoding="utf-8",
    )
    json_file(
        run / "intake.json",
        dict(
            prepared_ranges=dict(path=str(ranges)),
            effective_window_minutes=0.1,
            sources={SID: dict(paired_unique_png_frames=2)},
        ),
    )
    original_viewer = put(run, "viewer/index.html", b"original accepted viewer")
    before_registry = (run / "source_registry.json").read_bytes()
    output = tmp_path / "separate/viewer"
    with (
        patch("hmr4d.lidar_mocap.rally_viewer.CalibrationSnapshot.load", return_value=calibration),
        patch("hmr4d.lidar_mocap.rally_viewer.load_run_session", return_value=session) as loader,
        patch("smplx.create", return_value=SimpleNamespace(faces=np.array([[0, 1, 2]]))),
    ):
        viewer = prepare_viewer(run, run, relocations=resolver, viewer_output=output)
    assert loader.call_args.kwargs["relocations"] is resolver
    assert viewer.viewer_output == output
    assert Path(viewer.manifest["runtime_relocations"]["path"]) == output / "runtime_relocations.json"
    assert viewer.manifest["calibration_digest"] == "camera"
    assert (run / original_viewer["file"]).read_bytes() == b"original accepted viewer"
    assert (run / "source_registry.json").read_bytes() == before_registry
    assert (output / "viewer.js").is_file()


def legacy_session(data):
    session_dir = data / "sessions" / SID
    video_dir = session_dir / "output-zcam" / ("zcam_" + SID)
    for camera in ("cam1", "cam2"):
        put(data, (video_dir.relative_to(data) / (camera + ".frames.jsonl")).as_posix())
    return SimpleNamespace(
        session_id=SID,
        lidar_dir=session_dir / "output-lidar" / ("lidar_" + SID),
        meta_dir=session_dir / "meta" / SID,
        video_dir=video_dir,
        video_paths={camera: video_dir / (camera + ".mp4") for camera in ("cam1", "cam2")},
        pairs=lambda: [dict(lidar_path=str(session_dir / "output-lidar/cloud.png"))],
        _tracks={},
    )


def test_legacy_path_audit_only_claims_the_actual_legacy_dependencies(tmp_path):
    resolver = manifest(tmp_path, {})
    session = legacy_session(tmp_path)
    audit = audit_legacy_session_paths(session, resolver)
    assert audit["relocation_applied"] is False
    assert audit["historical_relocation_dependencies"] == []
    session.pairs = lambda: [dict(lidar_path=str(tmp_path / f"upstream/2026-10-19/{SID}/cloud.png"))]
    with pytest.raises(ValueError, match="still depends on historical"):
        audit_legacy_session_paths(session, resolver)


@pytest.mark.parametrize(
    "script,required",
    [
        ("serve_badminton_rallies.py", []),
        ("serve_badminton_body_repair.py", []),
        ("serve_badminton_canonical_tracks.py", ["--manifest", "missing.json"]),
    ],
)
def test_serve_clis_expose_flags_and_reject_partial_activation_before_file_reads(script, required):
    command = [sys.executable, str(PROJ_ROOT / "tools" / script)]
    options = dict(
        cwd=PROJ_ROOT,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    result = subprocess.run(command + ["--help"], **options)
    assert result.returncode == 0 and "--data-relocations-sha256" in result.stdout
    assert "--viewer-output" in result.stdout
    result = subprocess.run(command + required + ["--data-relocations", "missing.json"], **options)
    assert result.returncode == 2 and "must be supplied together" in result.stderr
    assert "FileNotFoundError" not in result.stderr
