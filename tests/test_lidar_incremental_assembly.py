"""Real small tensor/metadata concatenation behind separately tested policy validation."""

from copy import deepcopy
import json
from pathlib import Path

import joblib
import pytest
import torch

from hmr4d.lidar_mocap import expanded_dataset as expanded, incremental_release as inc
from hmr4d.lidar_mocap.export import FIELDS
import test_lidar_expanded_dataset as legacy_expanded
from test_lidar_incremental_release import write


@pytest.fixture
def increment(tmp_path, monkeypatch):
    helper = legacy_expanded.ExpandedDatasetTests()
    (old_path, old), _, _, _ = helper.make_expanded_base(tmp_path / "base")
    publications, _, repair, completion, receipt = helper.make_formal_completion(tmp_path / "fresh")
    paired_path, paired = publications[1]
    labels, source_root = repair / "final_labels", repair.parent / "source"
    selected = write(tmp_path / "policy.json", dict(schema=inc.POLICY))
    index = write(tmp_path / "actual_index.json", dict(actual_admitted_source=True))
    certificate = write(tmp_path / "actual_camera.json", dict(actual_camera=True))
    registry = dict(
        schema="badminton-formal-derived-source-registry-v1",
        release_policy=selected,
        sources=dict(new_train=dict(source_index=index)),
    )
    write(source_root / "source_registry.json", registry)
    write(labels / "source_registry.json", registry)
    roles = dict(train=["new_train"], val=[], test=[])
    input_path = repair / "new_source_input.json"
    source = json.loads(input_path.read_text())
    source.update(
        jobs=[job for job in source["jobs"] if job["source_id"] == "new_train"],
        source_root=str(source_root),
        source_registry_sha256=expanded.binding(source_root / "source_registry.json")["sha256"],
        release_policy=selected,
        session_splits=roles,
        required_nonempty_splits=["train"],
    )
    write(input_path, source)
    plan_path = labels / "repair_plan.json"
    plan = json.loads(plan_path.read_text())
    plan.update(jobs=source["jobs"], source_union_sha256=expanded.binding(input_path)["sha256"])
    write(plan_path, plan)
    body_path = labels / "datasets/dataset_manifest.json"
    body = json.loads(body_path.read_text())
    body.update(
        exports=dict(new_train=body["exports"]["new_train"]),
        accepted=2,
        source_registry_sha256=expanded.binding(labels / "source_registry.json")["sha256"],
    )
    write(body_path, body)
    observations_path = Path(paired["full_observation_manifest"])
    observations = json.loads(observations_path.read_text())
    observations["source_accepted"] = 2
    observations["identity"]["source"]["dataset_manifest"] = expanded.binding(body_path)
    write(observations_path, observations)
    paired.update(
        exports=dict(new_train=paired["exports"]["new_train"]),
        accepted=1,
        source_dataset_manifest_sha256=expanded.binding(body_path)["sha256"],
        full_observation_manifest_sha256=expanded.binding(observations_path)["sha256"],
    )
    paired["suggested_session_splits"] = dict(
        train=dict(paired["exports"]["new_train"], sessions=["new_train"]),
        val=legacy_expanded.export(tmp_path / "empty", "val", []),
        test=legacy_expanded.export(tmp_path / "empty", "test", []),
    )
    for role in ("val", "test"):
        paired["suggested_session_splits"][role]["sessions"] = []
    write(paired_path, paired)
    verification_path = paired_path.parent.parent / "verification.json"
    write(verification_path, dict(accepted=1, sessions=dict(new_train=dict(accepted=1))))
    receipt.update(source_input=expanded.binding(input_path), windows=1, accepted=2)
    helper.rebind_cpu_receipt(repair, completion, receipt)
    completion["accepted"] = 1
    for path in (paired_path, body_path, observations_path, verification_path):
        completion["evidence"][str(path)] = expanded.binding(path)["sha256"]
    write(repair / "new_source_completion_receipt.json", completion)
    review = write(repair / "review.json", dict(synthetic_review=True))
    verified = write(repair / "canonical_verification.json", dict(synthetic_original_verifier=True))
    canonical = dict(
        paired,
        schema="badminton-canonical-bilateral-publication-v1",
        release_policy=selected,
        canonical_identity_revision=dict(
            source_publication=expanded.binding(paired_path), reviewed_boundaries=review, verification=verified
        ),
    )
    new_path = repair / "canonical/datasets/dataset_manifest.json"
    write(new_path, canonical)
    old_training = write(tmp_path / "parent_training.json", dict(old, schema="badminton-expanded-training-v1"))
    parent_contract = write(tmp_path / "parent_final_contract.json", dict(previous_actual_contract=True))
    parent_audit = write(tmp_path / "parent_final_audit.json", dict(previous_actual_audit=True))
    parent_bindings = dict(
        archive_manifest=expanded.binding(Path(old["canonical_archive"]) / "manifest.json"),
        dataset_manifest=expanded.binding(old_path),
        training_manifest=old_training,
        final_consumer_contract=parent_contract,
        final_consumer_audit=parent_audit,
    )
    policy = inc.IncrementalRelease(
        selected,
        dict(canonical_data_root=str(tmp_path), new_source_splits=roles, required_new_nonempty_splits=["train"]),
        dict(review_sources=dict(new_train=dict(admitted_source_index=index, camera_applicability=certificate))),
        dict(
            total=3,
            counts={role: 1 for role in inc.ROLES},
            roles={role: ["old_" + role] for role in inc.ROLES},
            bindings=parent_bindings,
        ),
        [selected, *parent_bindings.values()],
    )

    def explicit_policy(path, sha256):
        assert str(Path(path).resolve()) == selected["path"] and sha256 == selected["sha256"]
        policy.recheck()
        return policy

    monkeypatch.setattr(inc, "load_release_policy", explicit_policy)
    monkeypatch.setattr(expanded.CalibrationSnapshot, "load", helper.expanded_calibration)
    monkeypatch.setattr(expanded, "_verify_export_data", lambda manifest: {})
    from hmr4d.lidar_mocap import canonical_tracks

    monkeypatch.setattr(canonical_tracks, "verify_canonical_publication", lambda manifest: dict(original_called=True))
    bindings = dict(
        old_archive=parent_bindings["archive_manifest"],
        old_dataset_manifest=parent_bindings["dataset_manifest"],
        old_training_manifest=old_training,
        parent_final_consumer_contract=parent_contract,
        parent_final_consumer_audit=parent_audit,
        new_source_input=expanded.binding(input_path),
        new_source_registry=expanded.binding(source_root / "source_registry.json"),
        new_source_splits=selected,
        producer_confirmation=expanded.binding(repair / "new_source_completion_receipt.json"),
        producer_joint_manifest=expanded.binding(new_path),
    )
    contract = dict(
        schema=inc.ASSEMBLY,
        status="complete",
        sealed=True,
        release_policy=selected,
        canonical_data_root=str(tmp_path),
        new_repair_root=str(repair),
        old_accepted=3,
        old_split_counts=policy.parent["counts"],
        old_source_splits=policy.parent["roles"],
        new_source_splits=roles,
        required_new_nonempty_splits=["train"],
        new_final_admitted_counts=dict(train=1, val=0, test=0, total=1),
        new_sources=dict(
            new_train=dict(
                final_admitted_count=1,
                repair_root=str(repair),
                admitted_source_index=index,
                camera_applicability=certificate,
                calibration_snapshot=expanded.binding(labels / "calibration/calibration_snapshot.json"),
                calibration_digest="new-cal",
                formal_body_input=expanded.binding(input_path),
            )
        ),
        canonical_mode="reviewed_cuts",
        paired_schema=dict(
            dtype="float32",
            target_body_model="neutral_smpl_24_joints",
            coordinate_space="lidar_viewer_xyz_m",
            transl_convention="shaped_pelvis_position",
            **{name: ["N", *shape] for name, shape in FIELDS.items()},
        ),
        bindings=bindings,
    )
    contract_path = tmp_path / "assembly_input.json"
    write(contract_path, contract)
    return dict(
        old_path=old_path,
        old=old,
        new_path=new_path,
        contract_path=contract_path,
        contract=contract,
        policy=selected,
        output=tmp_path / "union",
        registry=source_root / "source_registry.json",
    )


def assemble(case, **overrides):
    return expanded.assemble(
        case["old_path"],
        case["new_path"],
        case["contract_path"],
        case["output"],
        **dict(
            dict(release_policy=case["policy"]["path"], release_policy_sha256=case["policy"]["sha256"]), **overrides
        ),
    )


def test_train_only_increment_retains_real_old_payloads_and_lineage(increment):
    result = assemble(increment)
    assert result["release_policy"] == increment["policy"] and result["accepted"] == 4
    assert result["new_accepted"] == 1 and result["old_accepted"] == 3 and result["training_published"] is False
    for role in inc.ROLES:
        before = increment["old"]["suggested_session_splits"][role]
        after = result["suggested_session_splits"][role]
        original_bytes = Path(before["metadata"]).read_bytes()
        assert Path(after["metadata"]).read_bytes().startswith(original_bytes)
        for name in FIELDS:
            assert torch.equal(joblib.load(after["path"])[name][:1], joblib.load(before["path"])[name])
        if role != "train":
            assert after["accepted"] == before["accepted"] == 1
            assert Path(after["metadata"]).read_bytes() == original_bytes
            assert after["sha256"] == before["sha256"]
            assert after["metadata_sha256"] == before["metadata_sha256"]
            assert not Path(after["path"]).samefile(before["path"])
            assert not Path(after["metadata"]).samefile(before["metadata"])
    for sid, entry in increment["old"]["source_registry"].items():
        assert result["source_registry"][sid] == dict(entry, generation="old")


def test_v2_requires_both_explicit_policy_arguments(increment):
    with pytest.raises(ValueError, match="both explicit"):
        assemble(increment, release_policy_sha256=None)
    assert not increment["output"].exists()


@pytest.mark.parametrize(
    "change",
    [
        dict(required_new_nonempty_splits=["train", "val"]),
        dict(new_final_admitted_counts=dict(train=1, val=1, test=0, total=2)),
        dict(new_final_admitted_counts=dict(train=True, val=0, test=0, total=1)),
        dict(old_split_counts=dict(train=True, val=1, test=1)),
        dict(canonical_mode="reviewed_no_changes"),
    ],
)
def test_v2_rejects_fake_new_val_or_unimplemented_no_cut(increment, change):
    write(increment["contract_path"], dict(increment["contract"], **change))
    with pytest.raises(ValueError):
        assemble(increment)
    assert not increment["output"].exists()


def test_v2_cannot_rebind_a_foreign_current_source_registry(increment):
    registry = json.loads(increment["registry"].read_text())
    registry["release_policy"] = dict(increment["policy"], sha256="0" * 64)
    fresh = write(increment["registry"], registry)
    changed = deepcopy(increment["contract"])
    changed["bindings"]["new_source_registry"] = fresh
    write(increment["contract_path"], changed)
    with pytest.raises(ValueError, match="source input/registry"):
        assemble(increment)
    assert not increment["output"].exists()


@pytest.mark.parametrize("wrong", [None, dict(path="foreign", sha256="0" * 64)])
def test_v2_publication_must_bind_the_explicit_policy(increment, wrong):
    publication = json.loads(increment["new_path"].read_text())
    publication["release_policy"] = wrong
    changed = deepcopy(increment["contract"])
    changed["bindings"]["producer_joint_manifest"] = write(increment["new_path"], publication)
    write(increment["contract_path"], changed)
    with pytest.raises(ValueError, match="publication differs"):
        assemble(increment)
    assert not increment["output"].exists()


def test_zero_increment_copies_parent_encoding_even_when_reserialization_changes_sha(increment):
    policy = inc.load_release_policy(increment["policy"]["path"], increment["policy"]["sha256"])
    archive_path = Path(policy.parent["bindings"]["archive_manifest"]["path"])
    archive = json.loads(archive_path.read_text())
    for role in ("val", "test"):
        entry = increment["old"]["suggested_session_splits"][role]
        data = joblib.load(entry["path"])
        joblib.dump(data, entry["path"], compress=3, protocol=2)
        entry["sha256"] = expanded.binding(entry["path"])["sha256"]
        member = archive["splits"][role]["tensor"]
        member.update(sha256=entry["sha256"], bytes=Path(entry["path"]).stat().st_size)
        empty = json.loads(increment["new_path"].read_text())["suggested_session_splits"][role]
        legacy, _ = expanded.merge_role(
            [entry, empty],
            increment["output"].parent / ("legacy-" + role + ".pth"),
            {sid: item["calibration_digest"] for sid, item in increment["old"]["source_registry"].items()},
        )
        assert legacy["sha256"] != entry["sha256"]
        assert all(torch.equal(joblib.load(legacy["path"])[name], data[name]) for name in FIELDS)
    fresh_archive = write(archive_path, archive)
    fresh_dataset = write(increment["old_path"], increment["old"])
    changed = deepcopy(increment["contract"])
    changed["bindings"].update(old_archive=fresh_archive, old_dataset_manifest=fresh_dataset)
    policy.parent["bindings"].update(archive_manifest=fresh_archive, dataset_manifest=fresh_dataset)
    replacements = {item["path"]: item for item in (fresh_archive, fresh_dataset)}
    policy.dependencies[:] = [replacements.get(item["path"], item) for item in policy.dependencies]
    write(increment["contract_path"], changed)
    result = assemble(increment)
    for role in ("val", "test"):
        original, actual = increment["old"]["suggested_session_splits"][role], result["suggested_session_splits"][role]
        assert actual["sha256"] == original["sha256"]
        assert actual["metadata_sha256"] == original["metadata_sha256"]
        assert not Path(actual["path"]).samefile(original["path"])
        assert not Path(actual["metadata"]).samefile(original["metadata"])
