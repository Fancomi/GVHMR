"""Strict policy/contract seam and real tiny no-change verifier; no production assets."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hmr4d.lidar_mocap import canonical_no_change as nc, expanded_dataset as expanded, incremental_release as inc
from hmr4d.lidar_mocap.export import FIELDS
from test_lidar_canonical_no_change import case, pair, publish
from test_lidar_incremental_release import bind, write


@pytest.fixture
def no_change_contract(case):
    case["output"] = case["repair"] / "canonical_no_change_v2"
    manifest = publish(case)
    policy = inc.load_release_policy(case["selected"]["path"], case["selected"]["sha256"])
    parent = policy.parent["bindings"]
    source = policy.scope["review_sources"]["fresh"]
    registry = case["root"] / "source/source_registry.json"
    input_path = case["repair"] / "new_source_input.json"
    manifest_path = case["output"] / "datasets/dataset_manifest.json"
    contract = dict(
        schema=inc.ASSEMBLY,
        status="complete",
        sealed=True,
        release_policy=policy.binding,
        canonical_data_root=policy.data["canonical_data_root"],
        new_repair_root=str(case["repair"]),
        old_accepted=policy.parent["total"],
        old_split_counts=policy.parent["counts"],
        old_source_splits=policy.parent["roles"],
        new_source_splits=policy.roles,
        required_new_nonempty_splits=policy.required,
        new_final_admitted_counts=dict(train=3, val=0, test=0, total=3),
        canonical_mode="reviewed_no_changes",
        new_sources=dict(
            fresh=dict(
                final_admitted_count=3,
                repair_root=str(case["repair"]),
                admitted_source_index=source["admitted_source_index"],
                camera_applicability=source["camera_applicability"],
                calibration_snapshot=bind(case["labels"] / "calibration/calibration_snapshot.json"),
                calibration_digest=manifest["calibration_digest"],
                formal_body_input=bind(input_path),
            )
        ),
        paired_schema=dict(
            dtype="float32",
            target_body_model="neutral_smpl_24_joints",
            coordinate_space="lidar_viewer_xyz_m",
            transl_convention="shaped_pelvis_position",
            **{name: ["N", *shape] for name, shape in FIELDS.items()},
        ),
        bindings=dict(
            old_archive=parent["archive_manifest"],
            old_dataset_manifest=parent["dataset_manifest"],
            old_training_manifest=parent["training_manifest"],
            parent_final_consumer_contract=parent["final_consumer_contract"],
            parent_final_consumer_audit=parent["final_consumer_audit"],
            new_source_input=bind(input_path),
            new_source_registry=bind(registry),
            new_source_splits=policy.binding,
            producer_confirmation=bind(case["completion_path"]),
            producer_joint_manifest=bind(manifest_path),
        ),
    )
    contract_path = case["root"] / "assembly-input.json"
    write(contract_path, contract)
    return dict(
        case=case,
        manifest=manifest,
        manifest_path=manifest_path,
        policy=policy,
        contract=contract,
        contract_path=contract_path,
        output=case["root"] / "assembly",
    )


def test_actual_no_change_contract_and_original_numeric_verifier(no_change_contract):
    value = no_change_contract
    inc.validate_contract(value["contract"], value["policy"])
    proof = nc.verify_no_change_publication(value["manifest"], **pair(value["case"]))
    assert proof["source_completion"] == value["contract"]["bindings"]["producer_confirmation"]
    assert proof["source_publication"] == value["manifest"]["canonical_identity_revision"]["source_publication"]
    assert proof["accepted"] == 3 and proof["full_accepted"] == 4


@pytest.mark.parametrize(
    "change",
    [
        "policy",
        "completion",
        "source",
        "extra_marker",
        "extra_proof",
        "count",
        "mode",
        "pending",
        "geometry",
        "identity",
    ],
)
def test_no_change_is_a_bound_typed_branch_not_a_mode_string(no_change_contract, change):
    value = no_change_contract
    manifest, contract = deepcopy(value["manifest"]), deepcopy(value["contract"])
    marker = manifest["canonical_identity_revision"]
    proof = json.loads(Path(marker["verification"]["path"]).read_text())
    if change == "policy":
        manifest["release_policy"] = dict(value["policy"].binding, sha256="0" * 64)
    elif change == "completion":
        proof["source_completion"] = bind(value["case"]["review_path"])
    elif change == "source":
        copied = value["case"]["root"] / "foreign-same-bytes.json"
        copied.write_bytes(Path(marker["source_publication"]["path"]).read_bytes())
        proof["source_publication"] = marker["source_publication"] = bind(copied)
    elif change == "extra_marker":
        marker["untyped"] = True
    elif change == "extra_proof":
        proof["untyped"] = True
    elif change == "count":
        proof["accepted"] = True
    elif change == "pending":
        manifest["status"] = "pending"
    elif change == "geometry":
        marker["numeric_geometry_unchanged"] = 1
    elif change == "identity":
        marker["identity_keys_unchanged"] = False
    else:
        marker["mode"] = "reviewed_cuts"
    marker["verification"] = write(Path(marker["verification"]["path"]), proof)
    contract["bindings"]["producer_joint_manifest"] = write(value["manifest_path"], manifest)
    with pytest.raises(ValueError):
        inc.validate_contract(contract, value["policy"])


class ReachedMerge(Exception):
    pass


@pytest.mark.parametrize("wrong_return", [False, True])
def test_assembly_dispatch_runs_real_no_change_verifier_and_binds_returned_completion(
    no_change_contract,
    monkeypatch,
    wrong_return,
):
    """Isolate old union/base stages; execute the complete new-source numeric verifier and caller checks."""
    value = no_change_contract
    policy, contract = value["policy"], value["contract"]
    inc.validate_contract(contract, policy)
    roles = {sid: role for role, sids in policy.parent["roles"].items() for sid in sids}
    roles.update({sid: role for role, sids in policy.roles.items() for sid in sids})
    monkeypatch.setattr(expanded, "validate_union_contract", lambda *args, **kwargs: roles)
    monkeypatch.setattr(
        expanded, "verify_expanded_base", lambda *args: (policy.parent["archive"]["source_registry"], {}, [])
    )
    monkeypatch.setattr(
        expanded.CalibrationSnapshot, "load", lambda path: SimpleNamespace(data=json.loads(Path(path).read_text()))
    )
    original = nc.verify_no_change_publication
    calls = []

    def verify(manifest, **kwargs):
        assert kwargs == pair(value["case"])
        proof = original(manifest, **kwargs)
        calls.append(deepcopy(proof))
        if wrong_return:
            proof["source_completion"] = bind(value["case"]["review_path"])
        return proof

    monkeypatch.setattr(nc, "verify_no_change_publication", verify)
    monkeypatch.setattr(expanded, "merge_role", lambda *args, **kwargs: (_ for _ in ()).throw(ReachedMerge()))
    with pytest.raises(ValueError if wrong_return else ReachedMerge):
        expanded.assemble(
            policy.parent["bindings"]["dataset_manifest"]["path"],
            value["manifest_path"],
            value["contract_path"],
            value["output"],
            release_policy=policy.binding["path"],
            release_policy_sha256=policy.binding["sha256"],
        )
    assert len(calls) == 1 and calls[0]["accepted"] == 3 and calls[0]["full_accepted"] == 4
    assert not value["output"].exists()
