"""Actual tiny CPU tensors and original verifiers; no production admission."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pytest

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap import canonical_no_change as nc
from hmr4d.lidar_mocap.canonical_tracks import checked_review
from hmr4d.lidar_mocap.dual_view import DualViewGate, _verify_export_data
from hmr4d.lidar_mocap.export import FIELDS, export_joblib
from test_lidar_incremental_release import bind as original_bind, make_policy, write


def bind(path):
    return original_bind(Path(path))


def exported(path, samples):
    entry = export_joblib(samples, path)
    entry.update(sha256=bind(entry["path"])["sha256"], metadata_sha256=bind(entry["metadata"])["sha256"])
    return entry


def samples(step):
    result = []
    for index, scan in enumerate(range(10, 14)):
        motion = "fresh:window:3:0"
        row = dict(
            session_id="fresh",
            motion_id=motion,
            window_id="window",
            track_id=3,
            lidar_index=scan,
            scan_mid_us=scan * 100000,
            calibration_digest="e" * 64,
            source_segment_indices=[1, 2],
            cameras={cam: dict(record_index=scan) for cam in ("cam1", "cam2")},
            teacher_sampling=dict(backend="paired_rgb10"),
        )
        values = {name: np.full(shape, index / 10, dtype=np.float32) for name, shape in FIELDS.items()}
        values["transl"] = np.array([index * step, 0, 0], dtype=np.float32)
        result.append(dict(values, motion_id=motion, metadata=row))
    return result


def refresh(case):
    """Rebind changed synthetic bytes so negative tests reach semantic gates."""
    source_path, completion_path = case["source_path"], case["completion_path"]
    review = json.loads(case["review_path"].read_text())
    completion = json.loads(completion_path.read_text())
    completion["evidence"][str(source_path)] = bind(source_path)["sha256"]
    write(completion_path, completion)
    review.update(source_publication=bind(source_path), source_completion=bind(completion_path))
    write(case["review_path"], review)
    return case


def source_change(case, change):
    source = json.loads(case["source_path"].read_text())
    change(source)
    write(case["source_path"], source)
    return refresh(case)


def metadata_change(case, change):
    source = json.loads(case["source_path"].read_text())
    path = Path(source["exports"]["fresh"]["metadata"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    change(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    source["exports"]["fresh"]["metadata_sha256"] = bind(path)["sha256"]
    source["suggested_session_splits"]["train"]["metadata_sha256"] = bind(path)["sha256"]
    write(case["source_path"], source)
    return refresh(case)


def make_case(tmp_path, monkeypatch, *, step=0.1):
    selected = make_policy(tmp_path)
    policy = nc.incremental_release.load_release_policy(selected["path"], selected["sha256"])
    root = tmp_path / "project/outputs/no_change_fixture"
    repair, labels = root / "repair", root / "repair/final_labels"
    paired = root / "repair_paired"
    registry = dict(
        schema="badminton-new-source-registry-v1",
        release_policy=selected,
        sources=dict(fresh=dict(source_index=policy.scope["review_sources"]["fresh"]["admitted_source_index"])),
    )
    registry_binding = write(root / "source/source_registry.json", registry)
    write(labels / "source_registry.json", registry)
    write(
        repair / "new_source_input.json",
        dict(
            schema="badminton-new-source-repair-input-v1",
            source_root=str(root / "source"),
            source_registry_sha256=registry_binding["sha256"],
            release_policy=selected,
            session_splits=policy.roles,
            required_nonempty_splits=policy.required,
        ),
    )
    write(labels / "calibration/calibration_snapshot.json", dict(snapshot_digest="e" * 64))
    full = samples(step)
    fit = root / "synthetic_fit.npz"
    np.savez(fit, vertices=np.zeros((3, 3)), joints=np.zeros((1, 3)), source_cloud=np.zeros((256, 3)))
    for sample in full:
        sample["metadata"]["fit_path"] = str(fit)
    window = labels / "sessions/fresh/windows/window"
    local = exported(window / "ptv3.pth", full)
    report = write(
        window / "fit_report.json",
        dict(
            accepted=4,
            artifacts_sha256={name: bind(window / name)["sha256"] for name in ("ptv3.pth", "ptv3.metadata.jsonl")},
        ),
    )
    body = exported(labels / "datasets/fresh.pth", full)
    body_manifest = write(
        labels / "datasets/dataset_manifest.json",
        dict(
            status="complete",
            accepted=4,
            exports=dict(fresh=body),
            calibration_digest="e" * 64,
            selected_windows=dict(fresh=["window"]),
            required_nonempty_splits=["train"],
        ),
    )
    native = write(
        labels / "native_verification.json",
        dict(
            passed=True,
            state="complete",
            accepted_checked=4,
            calibration_digest="e" * 64,
            windows={"fresh:window": dict(body=dict(report_sha256=report["sha256"]))},
        ),
    )
    # Run the unchanged aggregate equality and full continuity algorithm.
    monkeypatch.syspath_prepend(str(PROJ_ROOT / "tools"))
    from audit_badminton_fresh_continuity import audit

    audit(repair)
    audit_path = repair / "continuity_audit/audit.json"
    chosen = []
    for index in (0, 2, 3):
        sample = deepcopy(full[index])
        sample["metadata"].update(
            source_tensor_index=index,
            source_tensor_path=body["path"],
            dualcam=dict(
                views={
                    camera: dict(
                        observation=dict(
                            detector_index=0,
                            independent_detection_bbox_xyxy=[10, 20, 110, 220],
                            independent_detection_confidence=0.9,
                            association_bbox_iou=0.8,
                            association_margin=0.5,
                        )
                    )
                    for camera in ("cam1", "cam2")
                }
            ),
        )
        chosen.append(sample)
    qualified = exported(paired / "datasets/fresh.pth", chosen)
    splits = dict(train=dict(qualified, sessions=["fresh"]))
    for role in ("val", "test"):
        splits[role] = dict(exported(paired / "datasets" / (role + ".pth"), []), sessions=[])
    source_path = paired / "datasets/dataset_manifest.json"
    source_binding = write(
        source_path,
        dict(
            schema="badminton-ptv3-dualcam-dataset-v1",
            status="complete",
            accepted=3,
            exports=dict(fresh=qualified),
            suggested_session_splits=splits,
            calibration_digest="e" * 64,
            source_dataset_manifest=body_manifest["path"],
            source_dataset_manifest_sha256=body_manifest["sha256"],
            qualification_policy="bilateral_detection",
            gate=asdict(DualViewGate()),
            target_body_model="neutral_smpl_24_joints",
            gender="neutral",
            coordinate_space="lidar_viewer_xyz_m",
            point_count=256,
            transl_convention="shaped_pelvis_position",
            opaque_original_field=dict(kept="exactly"),
        ),
    )
    source = json.loads(source_path.read_text())
    assert _verify_export_data(source)["accepted"] == 3
    completion_path = repair / "new_source_completion_receipt.json"
    completion = write(
        completion_path,
        dict(
            state="labels_published_viewer_prepared",
            training_published=True,
            accepted=3,
            dataset_manifest=str(source_path),
            evidence={source_binding["path"]: source_binding["sha256"], native["path"]: native["sha256"]},
        ),
    )
    audit_document = json.loads(audit_path.read_text())
    cases = []
    raw_sha = hashlib.sha256(np.zeros((2, 2, 3), dtype=np.uint8).tobytes()).hexdigest()
    for index, event in enumerate(audit_document["pelvis_jump_candidates"]):
        image = root / "synthetic_native_review" / (str(index) + ".png")
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"synthetic test image; never native production evidence")
        records = []
        for side in ("before", "after"):
            for camera in ("cam1", "cam2"):
                pose = write(image.with_name(f"{index}-{side}-{camera}.json"), dict(raw_bgr_sha256=raw_sha))
                records.append(dict(side=side, camera=camera, pose=pose, fit=bind(fit), raw_bgr_sha256=raw_sha))
        witness = write(
            image.with_suffix(".json"),
            dict(case=index, event=event, audit=bind(audit_path), image=bind(image), records=records),
        )
        cases.append(dict(case=index, event=event, verdict="same_person", evidence=witness, image=bind(image)))
    review_path = repair / "reviewed_no_changes.json"
    write(
        review_path,
        dict(
            schema=nc.REVIEW,
            review_status="root_reviewed",
            reviewed_by="synthetic-test-root",
            reviewed_utc="2026-10-23T19:26:11Z",
            source_publication=source_binding,
            source_completion=completion,
            continuity_audit=bind(audit_path),
            cases=cases,
            identity_changes=0,
            training_authorized=False,
            physical_identity_ground_truth=False,
        ),
    )
    return dict(
        selected=selected,
        root=root,
        repair=repair,
        labels=labels,
        source_path=source_path,
        completion_path=completion_path,
        review_path=review_path,
        audit_path=audit_path,
        output=repair / "canonical_no_change_v2",
        original_verify=_verify_export_data,
    )


@pytest.fixture
def case(tmp_path, monkeypatch):
    return make_case(tmp_path, monkeypatch)


def pair(case):
    return dict(release_policy=case["selected"]["path"], release_policy_sha256=case["selected"]["sha256"])


def publish(case):
    return nc.publish_no_change_view(case["source_path"], case["review_path"], case["output"], **pair(case))


def rebound_proof(case, manifest, change):
    path = Path(manifest["canonical_identity_revision"]["verification"]["path"])
    proof = json.loads(path.read_text())
    change(proof)
    write(path, proof)
    manifest["canonical_identity_revision"]["verification"] = bind(path)
    return manifest


def test_real_verifiers_publish_exact_no_rewrite_subset_and_preserve_both_counts(case):
    before = {str(path): bind(path)["sha256"] for path in case["root"].rglob("*") if path.is_file()}
    source = json.loads(case["source_path"].read_text())
    manifest = publish(case)
    proof = nc.verify_no_change_publication(manifest, **pair(case))
    assert proof["accepted"] == 3 and proof["full_accepted"] == 4
    assert proof["sessions"]["fresh"]["exact_body_subset"] is True
    assert proof["sessions"]["fresh"]["numeric_values_bit_exact"] is True
    assert proof["identity_changes"] == 0 and proof["unresolved_jump_candidates"] == []
    assert manifest["exports"] == source["exports"]
    assert manifest["suggested_session_splits"] == source["suggested_session_splits"]
    assert manifest["opaque_original_field"] == source["opaque_original_field"]
    assert {name: bind(name)["sha256"] for name in before} == before
    published = sorted(
        path.relative_to(case["output"]).as_posix() for path in case["output"].rglob("*") if path.is_file()
    )
    assert published == [
        "canonical_verification.json",
        "datasets/dataset_manifest.json",
    ]
    assert json.loads(case["audit_path"].read_text())["body_manifest"]["path"] == source["source_dataset_manifest"]
    assert set(proof) == nc.PROOF_FIELDS
    assert set(proof["sessions"]["fresh"]) == nc.SESSION_FIELDS


def test_actual_flagged_full_audit_requires_every_native_same_person_witness(tmp_path, monkeypatch):
    case = make_case(tmp_path, monkeypatch, step=1.0)
    review = json.loads(case["review_path"].read_text())
    assert len(review["cases"]) == 3
    assert nc.verify_no_change_publication(publish(case), **pair(case))["identity_changes"] == 0
    legacy = write(
        case["root"] / "legacy_review.json",
        dict(
            schema="badminton-reviewed-canonical-track-boundaries-v1",
            review_status="reviewed",
            audit=review["continuity_audit"],
            cases=review["cases"],
        ),
    )
    with pytest.raises(ValueError, match="no reviewed cross-person"):
        checked_review(legacy["path"])


@pytest.mark.parametrize(
    "change",
    [
        lambda review: review.update(extra=True),
        lambda review: review.update(review_status="reviewed"),
        lambda review: review.update(identity_changes=False),
        lambda review: review.update(identity_changes=1),
        lambda review: review.update(training_authorized=True),
        lambda review: review.update(physical_identity_ground_truth=True),
        lambda review: review.update(reviewed_by=""),
        lambda review: review.update(reviewed_utc="2026-10-23T19:26:11"),
    ],
)
def test_closed_review_and_truthful_flags(case, change):
    review = json.loads(case["review_path"].read_text())
    change(review)
    write(case["review_path"], review)
    with pytest.raises(ValueError):
        publish(case)
    assert not case["output"].exists()


@pytest.mark.parametrize(
    "change",
    [
        lambda review: review.update(cases=[]),
        lambda review: review["cases"].pop(),
        lambda review: review["cases"][0].update(verdict="unknown"),
        lambda review: review["cases"][0].update(verdict="different_person"),
        lambda review: review["cases"][0].update(case=False),
        lambda review: review["cases"][0].update(extra=True),
        lambda review: review["cases"].append(deepcopy(review["cases"][0])),
        lambda review: review["cases"][0]["event"].update(after_scan=999),
    ],
)
def test_flags_cannot_be_hidden_by_zero_cut_claim(tmp_path, monkeypatch, change):
    case = make_case(tmp_path, monkeypatch, step=1.0)
    review = json.loads(case["review_path"].read_text())
    change(review)
    write(case["review_path"], review)
    with pytest.raises(ValueError):
        publish(case)


@pytest.mark.parametrize(
    "field,value",
    [
        ("motion_id", "foreign:window:3:0"),
        ("session_id", "foreign"),
        ("lidar_index", 999),
        ("scan_mid_us", 999),
        ("source_tensor_index", True),
        ("source_tensor_index", 999),
        ("calibration_digest", "foreign"),
        ("canonical_identity", dict(source_motion_id="foreign")),
        ("track_id", 999),
        ("window_id", "foreign"),
        ("segment_id", "foreign"),
        ("person_id", "foreign"),
        ("scene_id", "foreign"),
        ("frame_index", True),
        ("teacher_sampling", dict(backend="foreign")),
    ],
)
def test_rebound_foreign_or_changed_subset_is_rejected(case, field, value):
    metadata_change(case, lambda rows: rows[0].update({field: value}))
    with pytest.raises(ValueError):
        publish(case)


def test_real_original_numeric_comparison_rejects_freshly_rebound_changed_tensor(case):
    def change(source):
        entry = source["exports"]["fresh"]
        data = joblib.load(entry["path"])
        data["body_pose"][0, 0, 0] += 0.25
        joblib.dump(data, entry["path"])
        entry["sha256"] = bind(entry["path"])["sha256"]
        source["suggested_session_splits"]["train"]["sha256"] = entry["sha256"]

    source_change(case, change)
    with pytest.raises(ValueError, match="numeric values"):
        publish(case)


def test_original_bilateral_detection_verifier_is_executed(case):
    def changed(rows):
        rows[0]["dualcam"]["views"]["cam2"]["observation"]["independent_detection_confidence"] = 0

    metadata_change(case, changed)
    with pytest.raises(ValueError, match="corresponding detection"):
        publish(case)


@pytest.mark.parametrize(
    "field,value",
    [
        ("state", "running"),
        ("accepted", True),
        ("dataset_manifest", "foreign"),
        ("training_published", False),
    ],
)
def test_source_completion_cannot_be_replaced_by_pending_or_wrong_receipt(case, field, value):
    completion = json.loads(case["completion_path"].read_text())
    completion[field] = value
    write(case["completion_path"], completion)
    review = json.loads(case["review_path"].read_text())
    review["source_completion"] = bind(case["completion_path"])
    write(case["review_path"], review)
    with pytest.raises(ValueError):
        publish(case)


@pytest.mark.parametrize(
    "role,field,value",
    [
        ("native", "passed", False),
        ("native", "state", "running"),
        ("native", "accepted_checked", True),
        ("native", "accepted_checked", 3),
        ("aggregate", "checked", 3),
        ("aggregate", "passed", False),
        ("aggregate", "all_point_and_label_tensors_equal_native_verified_windows", False),
        ("audit", "screening_passed_no_flags", False),
        ("audit", "mutual_swap_candidates", [dict(swap=True)]),
        ("audit", "counts", dict(person_frames=3, pelvis_jump_candidates=0, mutual_swap_candidates=0)),
    ],
)
def test_rebound_native_aggregate_or_full_audit_cannot_hide_incomplete_scope(case, role, field, value):
    audit = json.loads(case["audit_path"].read_text())
    if role == "audit":
        audit[field] = value
    else:
        name = "native_verification" if role == "native" else "aggregate_native_contract"
        path = Path(audit[name]["path"])
        document = json.loads(path.read_text())
        document[field] = value
        audit[name] = write(path, document)
        if role == "native":
            completion = json.loads(case["completion_path"].read_text())
            completion["evidence"][str(path)] = audit[name]["sha256"]
            write(case["completion_path"], completion)
    write(case["audit_path"], audit)
    review = json.loads(case["review_path"].read_text())
    review.update(continuity_audit=bind(case["audit_path"]), source_completion=bind(case["completion_path"]))
    write(case["review_path"], review)
    with pytest.raises(ValueError):
        publish(case)


def test_empty_role_cannot_hide_foreign_numeric_rows(case):
    def change(source):
        entry = source["suggested_session_splits"]["val"]
        joblib.dump(joblib.load(source["exports"]["fresh"]["path"]), entry["path"])
        entry["sha256"] = bind(entry["path"])["sha256"]

    source_change(case, change)
    with pytest.raises(ValueError, match="empty role"):
        publish(case)


@pytest.mark.parametrize(
    "field,value",
    [
        ("review_only", True),
        ("coordinate_space", "foreign"),
        ("point_count", 80),
        ("target_body_model", "foreign"),
    ],
)
def test_original_body_contract_cannot_be_overridden_by_new_marker(case, field, value):
    source_change(case, lambda source: source.update({field: value}))
    with pytest.raises(ValueError):
        publish(case)


def test_original_native_window_bytes_are_rechecked_without_new_native_execution(case):
    path = case["labels"] / "sessions/fresh/windows/window/ptv3.pth"
    path.write_bytes(path.read_bytes() + b"changed-after-original-audit")
    with pytest.raises(ValueError, match="dependency changed"):
        publish(case)


def test_hardlink_alias_cannot_substitute_source_tensor(case):
    def change(source):
        entry = source["exports"]["fresh"]
        alias = Path(entry["path"]).with_name("same_bytes_alias.pth")
        alias.hardlink_to(entry["path"])
        entry["path"] = str(alias)

    source_change(case, change)
    with pytest.raises(ValueError, match="aliases or foreign"):
        publish(case)


@pytest.mark.parametrize(
    "field,value",
    [
        ("identity_changes", False),
        ("identity_changes", 1),
        ("full_accepted", 3),
        ("accepted", True),
        ("numeric_values_unchanged", False),
        ("identity_keys_unchanged", False),
        ("unresolved_jump_candidates", [dict(hidden=True)]),
        ("extra", True),
        ("training_authorized", True),
        ("physical_identity_ground_truth", True),
    ],
)
def test_rebound_flag_only_or_non_closed_proof_is_rejected(case, field, value):
    manifest = rebound_proof(case, publish(case), lambda proof: proof.update({field: value}))
    with pytest.raises(ValueError):
        nc.verify_no_change_publication(manifest, **pair(case))


@pytest.mark.parametrize(
    "change",
    [
        lambda manifest: manifest.update(extra=True),
        lambda manifest: manifest.update(opaque_original_field=dict(kept="changed")),
        lambda manifest: manifest["canonical_identity_revision"].update(extra=True),
        lambda manifest: manifest["canonical_identity_revision"].update(mode="reviewed_cuts"),
        lambda manifest: manifest["canonical_identity_revision"].update(identity_keys_unchanged=1),
        lambda manifest: manifest.update(schema="badminton-canonical-bilateral-publication-v1"),
    ],
)
def test_closed_publication_and_legacy_dispatch(case, change):
    manifest = publish(case)
    change(manifest)
    with pytest.raises(ValueError):
        nc.verify_no_change_publication(manifest, **pair(case))


def test_changed_source_after_publication_and_before_seal_is_rejected(case):
    manifest = publish(case)
    path = Path(manifest["exports"]["fresh"]["metadata"])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="dependency changed"):
        nc.verify_no_change_publication(manifest, **pair(case))


def test_post_verifier_mutation_is_caught_by_final_byte_recheck(case, monkeypatch):
    def verify(source):
        result = case["original_verify"](source)
        path = Path(source["exports"]["fresh"]["metadata"])
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(nc, "_verify_export_data", verify)
    with pytest.raises(ValueError, match="dependency changed"):
        publish(case)
    assert not case["output"].exists()


def test_foreign_source_hash_and_unknown_verifier_code_are_rejected(case):
    manifest = publish(case)
    foreign = write(case["root"] / "foreign.json", dict(not_the_producer=True))
    rebound_proof(case, manifest, lambda proof: proof.update(producer=foreign))
    with pytest.raises(ValueError, match="source-bound"):
        nc.verify_no_change_publication(manifest, **pair(case))


def test_explicit_actual_policy_is_required_and_rechecked(case):
    with pytest.raises(TypeError):
        nc.publish_no_change_view(case["source_path"], case["review_path"], case["output"])
    with pytest.raises(ValueError):
        nc.publish_no_change_view(
            case["source_path"],
            case["review_path"],
            case["output"],
            release_policy=case["selected"]["path"],
            release_policy_sha256="0" * 64,
        )
    manifest = publish(case)
    path = Path(case["selected"]["path"])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="dependency changed"):
        nc.verify_no_change_publication(manifest, **pair(case))


def test_fresh_output_only_and_no_alias(case):
    publish(case)
    with pytest.raises(ValueError, match="fresh empty"):
        publish(case)
    with pytest.raises(ValueError, match="traversal"):
        nc.publish_no_change_view(
            case["source_path"].parent / ".." / "datasets/dataset_manifest.json",
            case["review_path"],
            case["output"].with_name("other"),
            **pair(case),
        )


def browser(case, manifest):
    canonical = bind(case["output"] / "datasets/dataset_manifest.json")
    evidence = write(case["root"] / "synthetic_browser_evidence.json", dict(synthetic=True))
    return dict(
        passed=True,
        page_errors=[],
        canonical_publication=canonical,
        canonical_mode=nc.MODE,
        canonical_no_change_proof=manifest["canonical_identity_revision"]["verification"],
        source_completion=json.loads(case["review_path"].read_text())["source_completion"],
        native_boundary_regressions=[],
        frozen_source_coverage=dict(active_sources=["fresh"], empty_sources=[]),
        declared_source_count=1,
        played_sources=1,
        clips=2,
        minimum_clip_seconds=5,
        results=[
            dict(
                source_id="fresh",
                rally_index=index,
                passed=True,
                actual_begin=index * 10,
                actual_end=index * 10 + 5,
                unique_committed_scans=30,
                screenshot=evidence,
            )
            for index in range(2)
        ],
        producer=evidence,
        clip_plan=evidence,
        viewer_manifest=evidence,
    )


def test_acceptance_binds_original_full_audit_and_actual_clip_metadata(case):
    manifest = publish(case)
    proof = nc.validate_no_change_acceptance(manifest, bind(case["audit_path"]), browser(case, manifest), **pair(case))
    assert proof["full_accepted"] == 4
    assert proof["continuity_audit"] == bind(case["audit_path"])


@pytest.mark.parametrize(
    "change",
    [
        lambda b: b.update(passed=False),
        lambda b: b.update(page_errors=["error"]),
        lambda b: b.update(native_boundary_regressions=[dict(passed=True)]),
        lambda b: b.update(canonical_mode="reviewed_cuts"),
        lambda b: b.pop("canonical_no_change_proof"),
        lambda b: b.update(results=[], clips=0),
        lambda b: b["results"][0].update(source_id="foreign"),
        lambda b: b["results"][1].update(rally_index=0),
        lambda b: b["results"][0].update(actual_end=1),
        lambda b: b["results"][0].update(unique_committed_scans=True),
    ],
)
def test_no_flag_only_browser_acceptance(case, change):
    manifest = publish(case)
    actual = browser(case, manifest)
    change(actual)
    with pytest.raises(ValueError):
        nc.validate_no_change_acceptance(manifest, bind(case["audit_path"]), actual, **pair(case))


def test_cli_requires_explicit_policy_before_reading_assets():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(PROJ_ROOT / "tools/publish_badminton_canonical_no_change.py"),
            "--source-manifest",
            "missing.json",
            "--reviewed-continuity",
            "missing.json",
            "--output-root",
            "missing",
        ],
        cwd=str(PROJ_ROOT.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--release-policy" in result.stderr
