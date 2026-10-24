"""Tiny synthetic metadata/real NumPy renderer checks; no production labels."""

import base64
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hmr4d.lidar_mocap.calibration import CalibrationSnapshot, _canonical
from hmr4d.lidar_mocap.formal_ground_display import (
    COMMON_FIELDS,
    GEOMETRY_FIELDS,
    SCHEMA,
    SOURCE,
    configure_formal_ground_display,
    read_formal_ground_display,
)
from hmr4d.lidar_mocap.rally_viewer import GROUND_DISPLAY_SCHEMA, badminton_detection_bbox


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return bind(path)


def bind(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "formal/repair/final_labels"
    old = tmp_path / "review/repair/final_labels"
    bbox = dict(
        boxes=[
            dict(id="ground", type="Ignore", enabled=True, center=[0, 0, 0], size=[6, 12, 0.2], rotation_deg=[0, 0, 0]),
            dict(id="net", type="Ignore", enabled=True, center=[100, 100, 100], size=[0.1, 0.1, 0.1]),
        ]
    )
    snapshots = []
    for folder, review in [(root, False), (old, True)]:
        artifact = put(folder / "calibration/evidence/bbox.json", bbox)
        camera = dict(
            R_raw_lidar_to_camera=np.eye(3).tolist(),
            t_m=[0, 0, 2],
            K=[[700, 0, 960], [0, 700, 540], [0, 0, 1]],
            distortion_opencv=[0, 0, 0, 0, 0],
            image_size=[1920, 1080],
            projection_model="opencv_pinhole_brown5",
        )
        snapshot = {key: dict(fixture_only=key) for key in GEOMETRY_FIELDS}
        snapshot.update(
            schema="gvhmr-canonical-lidar-calibration-v2",
            cameras={"cam1": camera, "cam2": dict(camera, t_m=[1, 0, 2])},
            Q_raw_to_viewer=np.eye(3).tolist(),
            reference_plane=dict(origin_viewer_m=[0, 0, 0], normal_viewer=[0, 0, 1]),
            bbox=bbox,
            artifacts={"bbox": dict(path="evidence/bbox.json", sha256=artifact["sha256"])},
            sessions={SOURCE: dict(frame_map="unused.jsonl")},
        )
        if review:
            snapshot["review_only"] = True
        snapshot["snapshot_digest"] = hashlib.sha256(_canonical(snapshot)).hexdigest()
        put(folder / "calibration/calibration_snapshot.json", snapshot)
        snapshots.append(snapshot)
    source_entry = dict(
        baseline=False,
        source_index=dict(path=str(tmp_path / "source.json"), sha256="a" * 64),
        source_manifest_sha256="b" * 64,
        common_clock_sha256="c" * 64,
        frame_map_sha256="d" * 64,
    )
    bodies = []
    for folder, review, calibration in [(root, False, snapshots[0]), (old, True, snapshots[1])]:
        registry = put(
            folder / "source_registry.json",
            dict(
                schema=(
                    "badminton-rally-run-source-registry-v1"
                    if review
                    else "badminton-formal-derived-source-registry-v1"
                ),
                sources={SOURCE: source_entry},
            ),
        )
        body = dict(
            schema="badminton-review-only-body-labels-v1" if review else "badminton-ptv3-dataset-v1",
            exports={SOURCE: dict(accepted=1)},
            accepted=1,
            training_published=False,
            calibration_digest=calibration["snapshot_digest"],
            selected_windows={SOURCE: ["window"]},
        )
        if review:
            body["review_only"] = True
            body.update(source_registry=registry["path"], source_registry_sha256=registry["sha256"])
        else:
            source_root = root.parent.parent / "source"
            put(source_root / "source_registry.json", json.loads(Path(registry["path"]).read_text()))
            source_input = put(
                root.parent / "new_source_input.json",
                dict(
                    schema="badminton-new-source-repair-input-v1",
                    source_root=str(source_root),
                    source_registry_sha256=registry["sha256"],
                    calibration_digest=calibration["snapshot_digest"],
                ),
            )
            body["source_union_manifest_sha256"] = source_input["sha256"]
        bodies.append(put(folder / "datasets/dataset_manifest.json", body))
    body_binding, old_binding = bodies
    parent = dict(
        schema=GROUND_DISPLAY_SCHEMA,
        sport="badminton",
        scope="background_display_only",
        source_ids=[SOURCE],
        base_bbox_sha256=bind(root / "calibration/evidence/bbox.json")["sha256"],
        ground_ignore_id="ground",
        ground_positive_local_z_extension_m=0.015,
        dataset_manifest=old_binding,
        preview_only=True,
        formal_crop_unchanged=True,
    )
    parent_binding = put(tmp_path / "accepted-ground.json", parent)
    postleg = put(
        root.parent / "evaluation/formal_postleg_cpu_receipt.json",
        dict(
            schema="gvhmr-formal-postleg-cpu-completion-v1",
            state="postleg_native_qa_complete_bilateral_browser_archive_pending",
            training_published=False,
            source_input=source_input,
        ),
    )
    paired_path = root.parent.with_name("repair_paired") / "datasets/dataset_manifest.json"
    paired = put(
        paired_path,
        dict(
            source_dataset_manifest=body_binding["path"],
            source_dataset_manifest_sha256=body_binding["sha256"],
            accepted=1,
        ),
    )
    finish = put(
        root.parent / "new_source_completion_receipt.json",
        dict(
            state="labels_published_viewer_prepared",
            training_published=False,
            bilateral_labels_published=True,
            accepted=1,
            dataset_manifest=paired["path"],
            formal_postleg_cpu_receipt=postleg,
            evidence={x["path"]: x["sha256"] for x in [body_binding, postleg, paired]},
        ),
    )
    canonical = put(
        root.parent / "canonical/datasets/dataset_manifest.json",
        dict(
            schema="badminton-canonical-bilateral-publication-v2",
            canonical_identity_revision=dict(mode="reviewed_no_changes"),
            source_dataset_manifest=body_binding["path"],
            source_dataset_manifest_sha256=body_binding["sha256"],
            exports={SOURCE: dict(accepted=1)},
            accepted=1,
        ),
    )
    settings = {key: copy.deepcopy(parent[key]) for key in COMMON_FIELDS}
    settings.update(
        schema=SCHEMA,
        dataset_manifest=body_binding,
        parent_display_crop=parent_binding,
        formal_completion=finish,
        canonical_publication=canonical,
    )
    sidecar = tmp_path / "formal-ground.json"
    put(sidecar, settings)
    return SimpleNamespace(
        root=root,
        old=old,
        sidecar=sidecar,
        settings=settings,
        body=body_binding,
        canonical=canonical,
        snapshots=snapshots,
        tmp=tmp_path,
        source_input=source_input,
    )


def load(case):
    return read_formal_ground_display(case.sidecar, body_path=Path(case.body["path"]), canonical_binding=case.canonical)


def test_real_formal_metadata_has_no_fabricated_review_flag(case):
    evidence = load(case)
    assert "review_only" not in json.loads(Path(case.body["path"]).read_text())
    assert "source_registry" not in json.loads(Path(case.body["path"]).read_text())
    assert "source_registry_sha256" not in json.loads(Path(case.body["path"]).read_text())
    assert (
        evidence["settings"]["dataset_manifest"]
        != json.loads(Path(case.settings["parent_display_crop"]["path"]).read_text())["dataset_manifest"]
    )
    assert evidence["settings"]["schema"] == SCHEMA


def rebind_body_graph(case):
    """Keep unrelated real bindings fresh so a negative reaches its predicate."""
    case.body = bind(Path(case.body["path"]))
    canonical_path = Path(case.canonical["path"])
    canonical = json.loads(canonical_path.read_text())
    canonical["source_dataset_manifest_sha256"] = case.body["sha256"]
    case.canonical = put(canonical_path, canonical)
    finish_path = Path(case.settings["formal_completion"]["path"])
    finish = json.loads(finish_path.read_text())
    paired_path = Path(finish["dataset_manifest"])
    paired = json.loads(paired_path.read_text())
    paired["source_dataset_manifest_sha256"] = case.body["sha256"]
    paired_binding = put(paired_path, paired)
    finish["evidence"][case.body["path"]] = case.body["sha256"]
    finish["evidence"][paired_binding["path"]] = paired_binding["sha256"]
    case.settings.update(
        dataset_manifest=case.body, canonical_publication=case.canonical, formal_completion=put(finish_path, finish)
    )
    put(case.sidecar, case.settings)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", GROUND_DISPLAY_SCHEMA),
        ("source_ids", ["20261019_152542"]),
        ("preview_only", False),
        ("formal_crop_unchanged", False),
        ("ground_positive_local_z_extension_m", True),
        ("ground_positive_local_z_extension_m", 0.005),
        ("unexpected", True),
    ],
)
def test_closed_formal_sidecar_rejects_wrong_scope_before_viewer_writes(case, field, value):
    put(case.sidecar, dict(case.settings, **{field: value}))
    with pytest.raises(ValueError):
        load(case)
    assert not (case.tmp / "viewer").exists()


@pytest.mark.parametrize(
    "field", ["dataset_manifest", "parent_display_crop", "formal_completion", "canonical_publication"]
)
def test_stale_real_bound_input_rejected(case, field):
    Path(case.settings[field]["path"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        load(case)


@pytest.mark.parametrize(
    "field,value", [("training_published", True), ("review_only", True), ("schema", "other"), ("exports", {})]
)
def test_even_freshly_rebound_wrong_body_rejected(case, field, value):
    p = Path(case.body["path"])
    v = json.loads(p.read_text())
    v[field] = value
    case.settings["dataset_manifest"] = put(p, v)
    put(case.sidecar, case.settings)
    with pytest.raises(ValueError, match="formal body"):
        load(case)


@pytest.mark.parametrize("field", ["cameras", "bbox", "court", "Q_raw_to_viewer", "timing_policy"])
def test_rebound_camera_court_bbox_matrix_or_timing_change_rejected(case, field):
    p = case.root / "calibration/calibration_snapshot.json"
    v = json.loads(p.read_text())
    v[field] = dict(changed=True)
    put(p, v)
    with pytest.raises(ValueError, match="geometry changed"):
        load(case)


def test_rebound_foreign_native_registry_and_formal_finish_rejected(case):
    p = case.root / "source_registry.json"
    v = json.loads(p.read_text())
    v["sources"][SOURCE]["source_index"]["sha256"] = "f" * 64
    registry = put(p, v)
    source_path = Path(case.source_input["path"])
    source = json.loads(source_path.read_text())
    put(Path(source["source_root"]) / "source_registry.json", v)
    source["source_registry_sha256"] = registry["sha256"]
    case.source_input = put(source_path, source)
    finish_path = Path(case.settings["formal_completion"]["path"])
    finish = json.loads(finish_path.read_text())
    cpu_path = Path(finish["formal_postleg_cpu_receipt"]["path"])
    cpu = json.loads(cpu_path.read_text())
    cpu["source_input"] = case.source_input
    finish["formal_postleg_cpu_receipt"] = put(cpu_path, cpu)
    finish["evidence"][str(cpu_path)] = bind(cpu_path)["sha256"]
    case.settings["formal_completion"] = put(finish_path, finish)
    body_path = Path(case.body["path"])
    body = json.loads(body_path.read_text())
    body["source_union_manifest_sha256"] = case.source_input["sha256"]
    put(body_path, body)
    rebind_body_graph(case)
    with pytest.raises(ValueError, match="native source lineage changed"):
        load(case)


@pytest.mark.parametrize("adjacent", [True, False])
def test_both_registry_copies_must_match_the_real_union_sha(case, adjacent):
    source = json.loads(Path(case.source_input["path"]).read_text())
    p = case.root / "source_registry.json" if adjacent else Path(source["source_root"]) / "source_registry.json"
    p.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bound bytes changed"):
        load(case)


def test_body_union_hash_cannot_disagree_with_real_cpu_source_input(case):
    p = Path(case.body["path"])
    body = json.loads(p.read_text())
    body["source_union_manifest_sha256"] = "f" * 64
    put(p, body)
    rebind_body_graph(case)
    with pytest.raises(ValueError, match="source input binding mismatch"):
        load(case)


@pytest.mark.parametrize(
    "field,value",
    [("state", "pending"), ("training_published", True), ("bilateral_labels_published", False), ("accepted", True)],
)
def test_freshly_rebound_unfinished_or_bool_count_receipt_rejected(case, field, value):
    p = Path(case.settings["formal_completion"]["path"])
    finish = json.loads(p.read_text())
    finish[field] = value
    case.settings["formal_completion"] = put(p, finish)
    put(case.sidecar, case.settings)
    with pytest.raises(ValueError):
        load(case)


def test_freshly_rebound_wrong_postleg_type_rejected(case):
    p = Path(case.settings["formal_completion"]["path"])
    finish = json.loads(p.read_text())
    postleg_path = Path(finish["formal_postleg_cpu_receipt"]["path"])
    postleg = json.loads(postleg_path.read_text())
    postleg["schema"] = "review-only"
    finish["formal_postleg_cpu_receipt"] = put(postleg_path, postleg)
    finish["evidence"][str(postleg_path)] = bind(postleg_path)["sha256"]
    case.settings["formal_completion"] = put(p, finish)
    put(case.sidecar, case.settings)
    with pytest.raises(ValueError, match="postleg binding"):
        load(case)


def canonical_module(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location(
        "formal_ground_canonical_test", tools / "serve_badminton_canonical_tracks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_viewer(case, monkeypatch):
    module = canonical_module(monkeypatch)
    cloud = np.array([[2, 3, 0.105], [2, 3, 0.105], [2, 3, 0.11], [2, 3, 0.13], [2, 3, 0.09]], np.float32)
    rows = []
    for track, ordinal, accepted in [(1, 1, True), (2, 2, False)]:
        fit_path = case.root / f"sessions/{SOURCE}/windows/window/fits/fit{track}.npz"
        fit_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            fit_path,
            source_indices=np.array([ordinal], np.int64),
            source_cloud=cloud[[ordinal]],
            vertices=np.repeat(cloud[[ordinal]], 6890, axis=0),
            joints=np.repeat(cloud[[ordinal]], 24, axis=0),
        )
        rows.append(
            dict(
                fit_path=str(fit_path),
                motion_id=f"{SOURCE}:window:{track}:0",
                track_id=track,
                lidar_index=0,
                accepted=accepted,
                rejection_reasons=[] if accepted else ["rejected"],
                qa=dict(final_p2surface_mean_m=0.01),
            )
        )
    quality = case.root / f"sessions/{SOURCE}/windows/window/quality.jsonl"
    quality.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    put(quality.parent / "fit_report.json", dict(fitted=2, accepted=1))
    camera = dict(status="matched", record_index=0, media_pts_us=100, common_time_us=100)
    pair = dict(lidar_index=0, scan_mid_us=100, cameras={cam: dict(camera) for cam in ["cam1", "cam2"]})
    session = SimpleNamespace(
        pairs=lambda: [pair],
        read_cloud=lambda pair: cloud.copy(),
        camera_rows=lambda cam: [camera],
        read_image=lambda cam, index: np.full((16, 16, 3), 100, np.uint8),
    )
    manifest = dict(sources={SOURCE: dict(common_origin_us=100, baseline=True)}, point_filter=dict(default="roi"))
    output = case.tmp / "viewer"
    output.mkdir(exist_ok=True)
    (output / "index.html").write_text('<div class="counts"></div>', encoding="utf-8")
    calibration = CalibrationSnapshot.load(case.root / "calibration/calibration_snapshot.json")
    viewer = module.CanonicalViewer(
        case.root, case.root, calibration, manifest, {SOURCE: session}, {}, viewer_output=output
    )
    viewer.canonical_admitted = {SOURCE: {(rows[0]["motion_id"], 0): rows[0]}}
    viewer.canonical_frames = {}
    return viewer, cloud, rows


def test_all_formal_members_including_rejected_are_protected_with_identical_people_projection_rgb(case, monkeypatch):
    evidence = load(case)
    viewer, cloud, rows = make_viewer(case, monkeypatch)
    body_before = Path(case.body["path"]).read_bytes()
    original = viewer.geometry(SOURCE, 0)
    all_before = viewer.geometry(SOURCE, 0, "all")
    person_before = [copy.deepcopy(viewer.render_fit(row["fit_path"])) for row in rows]
    matrices_before = copy.deepcopy(viewer.calibration.data)
    configure_formal_ground_display(viewer, evidence)
    viewer.render_fit.cache_clear()
    changed = viewer.geometry(SOURCE, 0)
    np.testing.assert_array_equal(
        np.frombuffer(base64.b64decode(changed["frame"]["cloud"]), "<f4").reshape(-1, 3), cloud[[1, 2, 3]]
    )
    assert changed["frame"]["background_filter"]["removed_background_points"] == 1
    assert changed["frame"]["background_filter"]["protected_members"] == 2
    assert changed["frame"]["background_filter"]["removed_fit_members"] == 0
    assert len(changed["frame"]["athletes"]) == 1  # rejected fit is protected, not admitted
    assert changed["frame"]["athletes"] == original["frame"]["athletes"]
    assert changed["athletes"] == original["athletes"]
    assert changed["native_images"] == original["native_images"]
    assert viewer.geometry(SOURCE, 0, "all") == all_before
    assert [viewer.render_fit(row["fit_path"]) for row in rows] == person_before
    assert viewer.calibration.data == matrices_before
    assert Path(case.body["path"]).read_bytes() == body_before
    assert "review_only" not in viewer.manifest
    assert viewer.manifest["training_published"] is False


def test_late_bound_input_mutation_or_output_in_body_refuses_configuration(case, monkeypatch):
    evidence = load(case)
    viewer, _, _ = make_viewer(case, monkeypatch)
    viewer.viewer_output = case.root / "viewer"
    with pytest.raises(ValueError, match="scope"):
        configure_formal_ground_display(viewer, evidence)
    viewer.viewer_output = case.tmp / "viewer"
    Path(case.settings["formal_completion"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        configure_formal_ground_display(viewer, evidence)


@pytest.fixture
def native_video(tmp_path):
    """Lossless real container with known BGR bytes and independently set PTS."""
    import av

    path = tmp_path / "native-colors.mkv"
    y, x = np.indices((16, 16))
    images = [
        np.stack(((x * 7 + i * 13) % 256, (y * 9 + i * 19) % 256, (x + y + i * 67) % 256), axis=-1).astype(np.uint8)
        for i in range(3)
    ]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=10)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "bgr0"
        stream.codec_context.thread_count = 1
        for i, image in enumerate(images):
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = i, Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    rows = [dict(media_pts_us=i * 100_000, common_time_us=i * 100_000 + 123) for i in range(3)]
    session = SimpleNamespace(
        session_id=SOURCE, video_paths={cam: path for cam in ("cam1", "cam2")}, camera_rows=lambda cam: rows
    )
    return session, rows, images


def test_safe_native_decoder_preserves_raw_bgr_ordinals_and_rejects_wrong_pts(monkeypatch, native_video):
    module = canonical_module(monkeypatch)
    session, rows, images = native_video
    decoder = module.SingleThreadNativePreviewDecoder()
    decoded = list(decoder.frames(session, "cam1", [0, 2]))
    assert [index for index, image in decoded] == [0, 2]
    for index, image in decoded:
        assert hashlib.sha256(image.tobytes()).hexdigest() == hashlib.sha256(images[index].tobytes()).hexdigest()
        assert image.flags.c_contiguous
    assert decoder.diagnostics() == dict(seeks=1, delivered_frames=2)
    assert list(decoder.frames(session, "cam2", [])) == []
    assert decoder.diagnostics() == dict(seeks=1, delivered_frames=2)
    rows[2]["media_pts_us"] -= 1000
    with pytest.raises(ValueError, match="PTS mismatch"):
        list(decoder.frames(session, "cam2", [2]))
    assert decoder.diagnostics()["delivered_frames"] == 2


def test_safe_native_image_cache_preserves_pts_sha_and_jpeg_bgr(case, monkeypatch, native_video):
    import cv2

    module = canonical_module(monkeypatch)
    viewer, _, _ = make_viewer(case, monkeypatch)
    session, rows, images = native_video
    viewer.sessions[SOURCE] = session
    module.configure_native_decode(viewer, 1)
    assert viewer.manifest["native_decode"] == dict(
        threads=1, thread_type="SLICE", profile="pyav_single_thread_slice_v1"
    )
    ok, expected_jpeg = cv2.imencode(".jpg", images[2], [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    for camera in ("cam1", "cam2"):
        result = viewer.native_image(SOURCE, camera, 2)
        assert result["record_index"] == 2
        assert result["media_pts_us"] == rows[2]["media_pts_us"] == 200_000
        assert result["common_time_us"] == rows[2]["common_time_us"] == 200_123
        assert result["raw_bgr_sha256"] == hashlib.sha256(images[2].tobytes()).hexdigest()
        assert base64.b64decode(result["jpeg_base64"]) == expected_jpeg.tobytes()
        assert result["jpeg_sha256"] == hashlib.sha256(expected_jpeg.tobytes()).hexdigest()
        assert viewer.native_image(SOURCE, camera, 2) is result
    assert viewer.preview_decoder.diagnostics() == dict(seeks=2, delivered_frames=2)


def test_native_decode_opt_in_preserves_default_and_rejects_warm_cache(case, monkeypatch, native_video):
    module = canonical_module(monkeypatch)
    viewer, _, _ = make_viewer(case, monkeypatch)
    original_decoder = viewer.preview_decoder
    module.configure_native_decode(viewer, None)
    assert viewer.preview_decoder is original_decoder
    assert "native_decode" not in viewer.manifest
    for invalid in (True, 2, "1"):
        with pytest.raises(ValueError, match="explicitly 1"):
            module.configure_native_decode(viewer, invalid)
        assert viewer.preview_decoder is original_decoder
    viewer.native_image(SOURCE, "cam1", 0)
    with pytest.raises(ValueError, match="before any native frame"):
        module.configure_native_decode(viewer, 1)
    viewer._native_images.clear()
    session, _, _ = native_video
    viewer.sessions[SOURCE] = session
    module.configure_native_decode(viewer, 1)
    list(viewer.preview_decoder.frames(session, "cam1", [0]))
    assert not viewer._native_images
    with pytest.raises(ValueError, match="before any native frame"):
        module.configure_native_decode(viewer, 1)
