"""Synthetic whole-import checks preserve bytes, values, identities and QA."""

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hmr4d.lidar_mocap import formal_import_audit as audit, formal_numeric_import as importer
from hmr4d.lidar_mocap.identity_repair import decision_digest
from hmr4d.lidar_mocap.teacher import COORDINATE_SPACE, SCHEMA
import test_lidar_formal_reuse as reuse_fixtures
from test_lidar_formal_import_audit import case as audited_case, write, rows, refresh_report


@pytest.mark.parametrize("changed", [None, "handoff", "mapping", "asset", "calibration"])
def test_import_sessions_use_only_the_frozen_formal_relocation_chain(tmp_path, monkeypatch, changed):
    from hmr4d.lidar_mocap.source_index import SourceIndexedSession

    _, registry, calibration, original = reuse_fixtures.FormalRelocationTests().fixture(tmp_path)
    root = tmp_path / "import"
    write(root / "source/source_registry.json", registry)
    target = Path(registry["data_root"]) / "sessions/20261019_152542/meta/original.bin"
    if changed == "handoff":
        Path(registry["camera_admission_handoff"]["path"]).write_bytes(b"changed")
    elif changed == "mapping":
        (Path(registry["data_root"]) / registry["data_relocations"]["file"]).write_bytes(b"changed")
    elif changed == "asset":
        target.write_bytes(b"changed")
    elif changed == "calibration":
        calibration["data_relocations"] = dict(calibration["data_relocations"], sha256="0" * 64)

    loaded = []

    def load_index(path, data_root, *, expected_sha256, require_admitted, relocations):
        assert require_admitted is True
        assert Path(path) == Path(registry["sources"]["20261019_152542"]["source_index"]["path"])
        assert expected_sha256 == registry["sources"]["20261019_152542"]["source_index"]["sha256"]
        assert Path(data_root) == relocations.root
        assert not (relocations.root / original["file"]).exists()
        asset = relocations.resolve(original)
        assert asset == target and asset.read_bytes() == b"actual original"
        loaded.append(asset)
        return asset

    # Keep the real formal intake, camera handoff, relocation checks and run
    # loader. Only the unrelated source-index/camera certificate parser is a stub.
    monkeypatch.setattr(SourceIndexedSession, "load_source_index", load_index)
    if changed is None:
        assert importer._sessions(root, SimpleNamespace(data=calibration)) == {"20261019_152542": target}
        assert loaded == [target]
    else:
        with pytest.raises(ValueError):
            importer._sessions(root, SimpleNamespace(data=calibration))
        assert loaded == []


@pytest.fixture
def bundle(audited_case, monkeypatch):
    case = audited_case
    project = case.formal.parent.parent
    monkeypatch.setattr(importer, "PROJ_ROOT", project)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setattr(
        importer,
        "_sessions",
        lambda root, calibration: {
            sid: audit.load_run_session(root / "source", calibration, sid) for sid in case.per_source
        },
    )
    originals = {}
    for parent in case.parents:
        identity_path = parent / "repair/identity_evidence/identity_decisions.json"
        identity = audit.read_json(identity_path)
        for sid, item in case.per_source.items():
            if item["parent"] != parent:
                continue
            provenance = dict(
                schema=SCHEMA,
                algorithm_version=2,
                video=dict(path="native.mp4", bytes=42),
                frame_rows=[dict(record_index=0, common_time_us=50, media_pts_us=100)],
                camera=dict(K=[1, 1]),
                options=dict(max_track_gap_frames=6),
                weights=[],
                coordinate_space=COORDINATE_SPACE,
            )
            source_hash = importer._provenance_hash(provenance)
            manifest = dict(
                schema=SCHEMA,
                coordinate_space=COORDINATE_SPACE,
                provenance=provenance,
                provenance_hash=source_hash,
                tracks=[dict(path="track.pt")],
            )
            initial = item["initial"] / "teacher"
            payload = dict(
                schema=SCHEMA,
                coordinate_space=COORDINATE_SPACE,
                provenance=provenance,
                provenance_hash=source_hash,
                body_pose=torch.arange(69, dtype=torch.float32).reshape(1, 69),
                pc=torch.arange(768, dtype=torch.float64).reshape(1, 256, 3),
            )
            write(initial / "manifest.json", manifest)
            torch.save(payload, initial / "track.pt")
            secondary = parent / "repair/secondary_teachers/cam1/sessions" / sid / "windows/0_1/teacher"
            secondary.mkdir(parents=True)
            for name in ("manifest.json", "track.pt", "detections.pt"):
                (secondary / name).write_bytes((initial / name).read_bytes())
            key = item["key"]
            identity["windows"][key]["source_teacher_manifest_sha256"] = audit.file_sha(initial / "manifest.json")
            body_provenance = deepcopy(provenance)
            body_provenance["options"]["identity_decisions_sha256"] = decision_digest(identity["windows"][key])
            body_hash = importer._provenance_hash(body_provenance)
            body_payload = dict(payload, provenance=body_provenance, provenance_hash=body_hash)
            body_manifest = dict(
                manifest,
                provenance=body_provenance,
                provenance_hash=body_hash,
                identity_repair=dict(decision_sha256=decision_digest(identity["windows"][key]), observations_after=1),
            )
            write(item["window"] / "teacher/manifest.json", body_manifest)
            torch.save(body_payload, item["window"] / "teacher/track.pt")
            quality = audit.read_rows(item["window"] / "quality.jsonl")
            for number, row in enumerate(quality):
                row.update(motion_id=f"{sid}:1:{number}", teacher_sampling=dict(provenance_hash=body_hash))
            rows(item["window"] / "quality.jsonl", quality)
            rows(item["window"] / "ptv3.metadata.jsonl", quality[:1])
            report_path = item["window"] / "fit_report.json"
            report = audit.read_json(report_path)
            report.update(
                teacher_provenance_hash=body_hash,
                export=dict(
                    path=str(item["window"] / "ptv3.pth"), metadata=str(item["window"] / "ptv3.metadata.jsonl")
                ),
            )
            write(report_path, report)
            refresh_report(case, sid)
            originals[key] = dict(quality=quality, teacher_payload=body_payload)
        write(identity_path, identity)
        native_path = parent / "repair/final_labels/native_verification.json"
        native = audit.read_json(native_path)
        for entry in native["windows"].values():
            entry["teacher_inputs_sha256"] = {path: audit.file_sha(path) for path in entry["teacher_inputs_sha256"]}
        write(native_path, native)
    source = case.formal / "repair/new_source_input.json"
    write(
        case.formal / "repair/case_plan.json",
        dict(
            source_union=str(source),
            source_union_sha256=audit.file_sha(source),
            selected_scans={sid: [0] for sid in case.per_source},
        ),
    )
    actual_audit = audit.audit_formal_numeric_import(case.formal, case.proof, case.parents)
    audit_proof = importer._write_once(case.formal / "import_evidence/audit.json", actual_audit)
    return SimpleNamespace(case=case, audit=actual_audit, proof=audit_proof, originals=originals)


def test_complete_cpu_import_is_exact_for_bodies_points_qa_and_teacher_values(bundle):
    result = importer.import_formal_numeric_bodies(bundle.case.formal, bundle.proof)
    assert result["state"] == "complete" and result["copied_windows"] == 3
    assert result["accepted"] == 3 and result["fitted"] == 6
    assert result["body_aggregation_pending"] and result["native_reconstruction_pending"]
    assert result["training_published"] is False and result["numeric_fit_recomputed"] is False
    for key, entry in bundle.audit["windows"].items():
        original, target = Path(entry["original_window"]), Path(entry["target_window"])
        before, after = audit.read_json(original / "fit_report.json"), audit.read_json(target / "fit_report.json")
        for relative in ("ptv3.pth", "fits/0.npz", "fits/1.npz"):
            assert (original / relative).read_bytes() == (target / relative).read_bytes()
            assert not os.path.samefile(original / relative, target / relative)
        assert before["accepted"] == after["accepted"] and before["fitted"] == after["fitted"]
        for old, new in zip(bundle.originals[key]["quality"], audit.read_rows(target / "quality.jsonl")):
            assert old["qa"] == new["qa"] and old["accepted"] == new["accepted"]
            assert old["motion_id"] == new["motion_id"]
            assert old["cameras"] == new["cameras"]
            assert new["calibration_digest"] == "formal"
        payload = torch.load(target / "teacher/track.pt", map_location="cpu", weights_only=False)
        assert importer.same_teacher_values(bundle.originals[key]["teacher_payload"], payload)
        assert payload["provenance_hash"] != bundle.originals[key]["teacher_payload"]["provenance_hash"]
        current_identity = audit.read_json(result["reviewed_formal_identity"]["path"])
        teacher_manifest = audit.read_json(target / "teacher/manifest.json")
        assert teacher_manifest["identity_repair"]["decision_sha256"] == decision_digest(
            current_identity["windows"][key]
        )
        snapshot = Path(result["windows"][key]["current_report"]["path"])
        assert snapshot.read_bytes() == (target / "fit_report.json").read_bytes()
    assert len(audit.read_json(result["original_qa_snapshot"]["path"])["frames"]) == 3
    # A second verified CPU call preserves the original immutable receipt and values.
    repeated = importer.import_formal_numeric_bodies(bundle.case.formal, bundle.proof)
    assert repeated["receipt"] == result["receipt"]
    assert not (bundle.case.formal / "gpu_guarded_leg/baseline/guarded_leg_readiness.json").exists()


@pytest.mark.parametrize("changed", ["quality", "numeric", "teacher", "parent_identity", "scope", "audit"])
def test_changed_original_or_out_of_scope_output_prevents_complete_import(bundle, changed):
    entry = next(iter(bundle.audit["windows"].values()))
    if changed == "quality":
        (Path(entry["original_window"]) / "quality.jsonl").write_text("[]", encoding="utf-8")
    elif changed == "numeric":
        (Path(entry["original_window"]) / "fits/0.npz").write_bytes(b"different numeric fit")
    elif changed == "teacher":
        (Path(entry["original_window"]) / "teacher/track.pt").write_bytes(b"changed")
    elif changed == "parent_identity":
        identity = next(iter(bundle.audit["parent_evidence"].values()))["identity"]
        Path(identity["path"]).write_text("{}", encoding="utf-8")
    elif changed == "scope":
        updated = deepcopy(bundle.audit)
        next(iter(updated["windows"].values()))["target_window"] += "_other"
        proof = bundle.case.formal / "import_evidence/out_of_scope_audit.json"
        write(proof, updated)
        bundle.proof = audit.binding(proof)
    else:
        Path(bundle.proof["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        importer.import_formal_numeric_bodies(bundle.case.formal, bundle.proof)
    assert not (bundle.case.formal / "formal_numeric_import.json").exists()


def test_parent_numeric_hardlink_is_rejected_before_refinement_can_modify_the_parent(bundle):
    entry = next(iter(bundle.audit["windows"].values()))
    original = Path(entry["original_window"]) / "ptv3.pth"
    destination = Path(entry["target_window"]) / "ptv3.pth"
    destination.parent.mkdir(parents=True)
    os.link(original, destination)
    before = original.read_bytes()
    with pytest.raises(ValueError, match="independent copies"):
        importer.import_formal_numeric_bodies(bundle.case.formal, bundle.proof)
    assert original.read_bytes() == before
    assert not (bundle.case.formal / "formal_numeric_import.json").exists()


def test_import_requires_explicit_cpu_environment_and_never_overwrites_conflicting_bytes(bundle, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        importer.import_formal_numeric_bodies(bundle.case.formal, bundle.proof)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    entry = next(iter(bundle.audit["windows"].values()))
    conflict = Path(entry["target_window"]) / "ptv3.pth"
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b"independent fitted current output")
    with pytest.raises(ValueError, match="existing different"):
        importer.import_formal_numeric_bodies(bundle.case.formal, bundle.proof)
    assert conflict.read_bytes() == b"independent fitted current output"
    assert not (bundle.case.formal / "formal_numeric_import.json").exists()
