"""Small synthetic metadata only; no production policy or body admission is created."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hmr4d.lidar_mocap import incremental_release as inc
from hmr4d.lidar_mocap.formal_reuse import GEOMETRY_FIELDS, SOURCE_FIELDS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return bind(path)


def bind(path):
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def member(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return dict(path=relative, sha256=bind(path)["sha256"], bytes=path.stat().st_size)


def make_policy(root):
    project, raw, archive = root / "project", root / "raw", root / "data/releases/parent"
    review = project / "outputs/review"
    old_roles = {role: ["old_" + role] for role in inc.ROLES}
    prior_roles = {role: ["prior_" + role] for role in inc.ROLES}
    registry, splits, exports = {}, {}, {}
    calibration = member(archive, "calibrations/base.json", b'{"snapshot_digest":"base"}\n')
    for role in inc.ROLES:
        sids = old_roles[role] + prior_roles[role]
        tensor = member(archive, role + "/data.pth", ("synthetic " + role).encode())
        metadata = member(archive, role + "/metadata.jsonl", b'{"original":0}\n{"original":1}\n')
        splits[role] = dict(frames=2, declared_source_sessions=sids, tensor=tensor, metadata=metadata)
        for sid in sids:
            registry[sid] = dict(
                role=role,
                accepted=1,
                generation="new",
                calibration_digest="base",
                calibration_snapshot_sha256=calibration["sha256"],
                untouched_lineage=dict(sid=sid),
            )
            exports[sid] = dict(accepted=1)
    bridge = member(archive, "provenance/model_bridge.json", b'{"actual_synthetic_bridge":true}\n')
    archive_binding = write(
        archive / "manifest.json",
        dict(
            schema="lidar2smpl-release-v1",
            release_id="parent",
            status="complete",
            total_person_frames=6,
            splits=splits,
            source_registry=registry,
            source_calibrations=dict(base=calibration),
            artifacts=[bridge],
        ),
    )
    common = dict(
        status="complete",
        training_published=True,
        accepted=6,
        canonical_archive=str(archive),
        source_registry=registry,
        exports=exports,
        suggested_session_splits={},
    )
    for role, item in splits.items():
        common["suggested_session_splits"][role] = dict(
            accepted=2,
            sessions=item["declared_source_sessions"],
            path=str(archive / item["tensor"]["path"]),
            sha256=item["tensor"]["sha256"],
            metadata=str(archive / item["metadata"]["path"]),
            metadata_sha256=item["metadata"]["sha256"],
        )
    dataset = write(
        project / "outputs/base/datasets/dataset_manifest.json", dict(common, schema="badminton-expanded-dataset-v1")
    )
    training = write(
        project / "outputs/base/training_manifest.json", dict(common, schema="badminton-expanded-training-v1")
    )
    verification = write(
        archive / "verification.json",
        dict(
            passed=True,
            release_id="parent",
            manifest_sha256=archive_binding["sha256"],
            splits={role: dict(frames=2) for role in inc.ROLES},
        ),
    )
    seal = write(
        project / "outputs/base/release_receipt.json",
        dict(
            schema="badminton-expanded-release-receipt-v1",
            status="complete",
            accepted=6,
            old_accepted=3,
            new_accepted=3,
            splits={role: 2 for role in inc.ROLES},
            all_old_rows_retained=True,
            source_numeric_values_unchanged=True,
            archive_manifest=archive_binding,
            dataset_manifest=dataset,
            training_manifest=training,
            archive_verification=verification,
        ),
    )
    prior_contract = write(
        root / "parent_final_contract.json",
        dict(
            schema="expanded-1019-1020-consumer-contract/v1",
            status="complete",
            sealed=True,
            old_accepted=3,
            old_split_counts={role: 1 for role in inc.ROLES},
            old_source_splits=old_roles,
            new_source_splits=prior_roles,
            new_sources={sid: dict(final_admitted_count=1) for sids in prior_roles.values() for sid in sids},
            new_final_admitted_counts=dict(train=1, val=1, test=1, total=3),
            final_delivery=dict(archive_manifest=archive_binding, dataset_manifest=dataset, training_manifest=training),
            release_evidence=dict(seal=seal),
        ),
    )
    prior_audit = write(
        root / "parent_final_audit.json",
        dict(
            status="final_cpu_validation_passed_no_training_started",
            validation_stage="final",
            contract=prior_contract,
            training_authorized=False,
            gpu_used=False,
            formal_receipt_written=False,
            old_retained_bit_exact=3,
            new_retained_bit_exact=3,
            new_accepted_by_source={sids[0]: 1 for sids in prior_roles.values()},
        ),
    )
    parent = dict(
        archive_manifest=archive_binding,
        dataset_manifest=dataset,
        training_manifest=training,
        final_consumer_contract=prior_contract,
        final_consumer_audit=prior_audit,
    )
    manifest = write(raw / "review_manifest.json", dict(schema="badminton-native-rally-fit-v1", review_only=True))
    certificate = write(
        raw / "camera.json", dict(camera_calibration_admitted=True, body_labels_production_admitted=False)
    )
    index = {name: {"unchanged": name} for name in SOURCE_FIELDS}
    index.update(
        source_id="fresh",
        frame_count=1,
        frames=[dict(ordinal=0)],
        coordinate_contract=dict(M_baked_into_stored_xyz=False, application_raw_to_viewer_M=1, source_requested_M=1),
        crop_binding=dict(bbox_sha256="same"),
        accepted_bundle_sha256={"calibration.json": "a" * 64},
        accepted_calibration_sha256="a" * 64,
    )
    admitted_index = write(raw / "admitted/fresh.json", index)
    pilot = write(review / "pilot_input.json", dict(review_only=True, manifest=manifest))
    reg = write(
        review / "source/source_registry.json",
        dict(
            review_only=True,
            pilot_manifest=manifest,
            data_root=str(raw),
            sources=dict(fresh=dict(baseline=False, source_index=admitted_index)),
        ),
    )
    session = dict(manifest_sha256="unchanged", frame_count=1, frame_map_sha256="same", source_clock_audit={})
    geometry = {name: {"unchanged": name} for name in GEOMETRY_FIELDS}
    cal = dict(
        geometry,
        snapshot_digest="f" * 64,
        configuration_files_sha256={"calibration.json": "a" * 64},
        sessions=dict(fresh=session),
        artifacts={},
    )
    accepted = write(review / "accepted-base/calibration_snapshot.json", cal)
    write(
        review / "source/calibration/calibration_snapshot.json", dict(cal, snapshot_digest="e" * 64, review_only=True)
    )
    proof = write(raw / "accepted_freeze.json", dict(strict_accepted_freeze=True))
    name = "000000000000_000000000100"
    window = review / "repair/initial_teacher/sessions/fresh/windows" / name
    config = write(
        window / "window.json",
        dict(
            schema="badminton-window-v1",
            session_id="fresh",
            window_id=name,
            left_us=100,
            right_us=200,
            calibration_digest="e" * 64,
            settings=dict(camera="cam2", teacher_stride=1),
            source_pairs=[0],
        ),
    )
    source_input = write(
        review / "repair/new_source_input.json",
        dict(
            schema="badminton-new-source-repair-input-v1",
            review_only=True,
            source_root=str(review / "source"),
            source_registry_sha256=reg["sha256"],
            calibration_digest="e" * 64,
            session_splits=dict(review=["fresh"]),
            jobs=[dict(source_id="fresh", source_window=str(window), window_json_sha256=config["sha256"])],
        ),
    )
    scope = dict(
        schema=inc.SCOPE,
        review_status="root_reviewed",
        scope_id="synthetic-scope",
        reviewed_by="test-root",
        reviewed_utc="2026-10-23T00:00:00Z",
        parent_release=parent,
        training_authorized=False,
        new_source_splits=dict(train=["fresh"], val=[], test=[]),
        required_new_nonempty_splits=["train"],
        required_union_nonempty_splits=list(inc.ROLES),
        review_sources=dict(
            fresh=dict(
                review_root=str(review),
                pilot_input=pilot,
                review_manifest=manifest,
                review_source_registry=reg,
                review_source_input=source_input,
                accepted_parent_snapshot=accepted,
                accepted_freeze_proof=proof,
                admitted_source_index=admitted_index,
                camera_applicability=certificate,
            )
        ),
    )
    scope_binding = write(root / "scope.json", scope)
    policy = dict(
        schema=inc.POLICY,
        status="frozen_root_reviewed",
        policy_id="synthetic-policy",
        canonical_data_root=str(root / "data"),
        raw_data_root=str(raw),
        gvhmr_root=str(project),
        consumer_root=str(root / "consumer"),
        reviewed_scope=scope_binding,
        parent_release=parent,
        new_source_splits=deepcopy(scope["new_source_splits"]),
        required_new_nonempty_splits=["train"],
        required_union_nonempty_splits=list(inc.ROLES),
        training_authorized=False,
    )
    return write(root / "policy.json", policy)


def change_policy(binding, change, *, scope_change=None):
    path = Path(binding["path"])
    policy = json.loads(path.read_text())
    if scope_change:
        scope_path = Path(policy["reviewed_scope"]["path"])
        scope = json.loads(scope_path.read_text())
        scope_change(scope)
        policy["reviewed_scope"] = write(scope_path, scope)
    change(policy)
    return write(path, policy)


def test_actual_parent_metadata_and_train_only_policy(tmp_path):
    selected = make_policy(tmp_path)
    result = inc.load_release_policy(**dict(path=selected["path"], sha256=selected["sha256"]))
    assert result.parent["total"] == 6
    assert result.roles == dict(train=["fresh"], val=[], test=[])
    assert result.required == ["train"]
    assert Path(result.parent["model_bridge"]["path"]).read_bytes() == b'{"actual_synthetic_bridge":true}\n'
    assert "parent_receipt" not in result.parent["bindings"]
    assert not (tmp_path / "project/outputs/formal").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "pending"),
        ("training_authorized", True),
        ("new_source_registry", {"future": True}),
        ("new_final_admitted_counts", {}),
    ],
)
def test_policy_cannot_unlock_pending_or_future_outputs(tmp_path, field, value):
    selected = change_policy(make_policy(tmp_path), lambda data: data.update({field: value}))
    with pytest.raises(ValueError):
        inc.load_release_policy(selected["path"], selected["sha256"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "passed"),
        ("validation_stage", "assembly"),
        ("training_authorized", True),
        ("old_retained_bit_exact", 2),
        ("new_accepted_by_source", dict(prior_train=True, prior_val=1, prior_test=1)),
        ("contract", {"path": "foreign", "sha256": "0" * 64}),
    ],
)
def test_fresh_audit_hash_does_not_replace_actual_parent_final_semantics(tmp_path, field, value):
    selected = make_policy(tmp_path)
    policy = json.loads(Path(selected["path"]).read_text())
    audit_path = Path(policy["parent_release"]["final_consumer_audit"]["path"])
    audit = json.loads(audit_path.read_text())
    audit[field] = value
    fresh = write(audit_path, audit)
    selected = change_policy(
        selected,
        lambda data: data["parent_release"].update(final_consumer_audit=fresh),
        scope_change=lambda data: data["parent_release"].update(final_consumer_audit=fresh),
    )
    with pytest.raises(ValueError, match="Parent|Incremental"):
        inc.load_release_policy(selected["path"], selected["sha256"])


def test_policy_rechecks_live_metadata_after_success(tmp_path):
    selected = make_policy(tmp_path)
    result = inc.load_release_policy(selected["path"], selected["sha256"])
    Path(result.scope["review_sources"]["fresh"]["camera_applicability"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="Frozen incremental dependency changed"):
        result.recheck()


def test_wrong_explicit_sha_fails_before_parent_reads(tmp_path):
    selected = make_policy(tmp_path)
    Path(tmp_path / "parent_final_audit.json").unlink()
    with pytest.raises(ValueError, match="Frozen incremental dependency changed"):
        inc.load_release_policy(selected["path"], "0" * 64)


@pytest.mark.parametrize(
    "roles",
    [
        dict(train=["fresh"], val=["fresh"], test=[]),
        dict(train=[], val=["fresh"], test=[]),
        dict(train=["foreign"], val=[], test=[]),
    ],
)
def test_reviewed_scope_cannot_duplicate_move_or_replace_source(tmp_path, roles):
    selected = change_policy(
        make_policy(tmp_path),
        lambda data: data.update(new_source_splits=roles),
        scope_change=lambda data: data.update(new_source_splits=roles),
    )
    with pytest.raises(ValueError):
        inc.load_release_policy(selected["path"], selected["sha256"])


def test_model_bridge_change_is_not_a_new_runtime_receipt(tmp_path):
    selected = make_policy(tmp_path)
    (tmp_path / "data/releases/parent/provenance/model_bridge.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="model bridge"):
        inc.load_release_policy(selected["path"], selected["sha256"])


@pytest.mark.parametrize("tool", ["prepare_badminton_formal_increment.py", "assemble_badminton_expansion.py"])
def test_canonical_help_runs_from_outside_checkout_without_models(tmp_path, tool):
    path = Path(inc.__file__).resolve().parents[2] / "tools" / tool
    result = subprocess.run(
        [sys.executable, "-B", str(path), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stderr
    assert "--release-policy" in result.stdout and "--release-policy-sha256" in result.stdout
    assert not any(tmp_path.iterdir())


def test_formal_cli_missing_policy_sha_stops_before_outputs(tmp_path):
    path = Path(inc.__file__).resolve().parents[2] / "tools/prepare_badminton_formal_increment.py"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(path),
            "--release-policy",
            str(tmp_path / "missing.json"),
            "--root",
            str(tmp_path / "output"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 2 and "--release-policy-sha256" in result.stderr
    assert not (tmp_path / "output").exists()
