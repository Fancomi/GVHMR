"""Explicit v2 incremental policy and sealed-parent metadata, without model execution."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

POLICY = "badminton-incremental-release-policy-v2"
SCOPE = "badminton-incremental-release-scope-v2"
ASSEMBLY = "expanded-source-increment-assembly-input/v2"
FINAL = "expanded-source-increment-consumer-contract/v2"
ROLES = ("train", "val", "test")
PARENT_FIELDS = (
    "archive_manifest",
    "dataset_manifest",
    "training_manifest",
    "final_consumer_contract",
    "final_consumer_audit",
)
POLICY_FIELDS = {
    "schema",
    "status",
    "policy_id",
    "canonical_data_root",
    "raw_data_root",
    "gvhmr_root",
    "consumer_root",
    "reviewed_scope",
    "parent_release",
    "new_source_splits",
    "required_new_nonempty_splits",
    "required_union_nonempty_splits",
    "training_authorized",
}
SCOPE_FIELDS = {
    "schema",
    "review_status",
    "scope_id",
    "reviewed_by",
    "reviewed_utc",
    "parent_release",
    "review_sources",
    "new_source_splits",
    "required_new_nonempty_splits",
    "required_union_nonempty_splits",
    "training_authorized",
}
REVIEW_FIELDS = {
    "review_root",
    "pilot_input",
    "review_manifest",
    "review_source_registry",
    "review_source_input",
    "accepted_parent_snapshot",
    "accepted_freeze_proof",
    "admitted_source_index",
    "camera_applicability",
}
CONTRACT_BINDINGS = {
    "old_archive",
    "old_dataset_manifest",
    "old_training_manifest",
    "parent_final_consumer_contract",
    "parent_final_consumer_audit",
    "new_source_input",
    "new_source_registry",
    "new_source_splits",
    "producer_confirmation",
    "producer_joint_manifest",
}
CONTRACT_FIELDS = {
    "schema",
    "status",
    "sealed",
    "release_policy",
    "canonical_data_root",
    "new_repair_root",
    "old_accepted",
    "old_split_counts",
    "old_source_splits",
    "new_source_splits",
    "required_new_nonempty_splits",
    "new_sources",
    "new_final_admitted_counts",
    "canonical_mode",
    "paired_schema",
    "bindings",
}
ACTUAL_SOURCE_FIELDS = {
    "final_admitted_count",
    "repair_root",
    "admitted_source_index",
    "camera_applicability",
    "calibration_snapshot",
    "calibration_digest",
    "formal_body_input",
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    need(type(value) is int and value >= 0, "Incremental counts must be exact nonnegative integers")
    return value


def count_map(value, names):
    need(isinstance(value, dict) and set(value) == set(names), "Incremental count keys differ")
    return {name: integer(value[name]) for name in names}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def require_binding(item):
    need(
        isinstance(item, dict)
        and set(item) == {"path", "sha256"}
        and isinstance(item["path"], str)
        and Path(item["path"]).is_absolute()
        and isinstance(item["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None,
        "Incremental evidence requires an exact absolute path/SHA256 binding",
    )
    return item


def validate_roles(roles, sources, required):
    need(isinstance(roles, dict) and set(roles) == set(ROLES), "Incremental roles require train/val/test keys")
    need(
        isinstance(required, list)
        and bool(required)
        and len(required) == len(set(required))
        and all(role in ROLES for role in required)
        and "train" in required,
        "Incremental required new roles must explicitly include train",
    )
    need(all(isinstance(value, list) for value in roles.values()), "Incremental roles must be source lists")
    ids = [sid for role in ROLES for sid in roles[role]]
    need(all(isinstance(sid, str) and sid for sid in ids), "Invalid incremental source identity")
    need(len(ids) == len(set(ids)) and set(ids) == set(sources), "Incremental roles must cover each source once")
    need(all(roles[role] for role in required), "A required incremental source role is empty")
    return {role: list(roles[role]) for role in ROLES}


@dataclass
class IncrementalRelease:
    binding: dict
    data: dict
    scope: dict
    parent: dict
    dependencies: list

    @property
    def roles(self):
        return self.data["new_source_splits"]

    @property
    def required(self):
        return self.data["required_new_nonempty_splits"]

    def recheck(self):
        for item in self.dependencies:
            need(file_sha(item["path"]) == item["sha256"], "Frozen incremental dependency changed: " + item["path"])


def load_release_policy(path, sha256):
    """Validate explicit immutable metadata; full numeric admission remains a separate pass."""
    selected = require_binding(dict(path=str(Path(path).resolve()), sha256=sha256))
    observed, documents = {}, {}

    def checked(item):
        require_binding(item)
        identity = str(Path(item["path"]).resolve())
        if identity in observed:
            need(observed[identity]["sha256"] == item["sha256"], "Conflicting incremental file bindings")
        else:
            need(file_sha(item["path"]) == item["sha256"], "Frozen incremental dependency changed: " + item["path"])
            observed[identity] = dict(item)
        return item

    def document(item):
        checked(item)
        identity = str(Path(item["path"]).resolve())
        if identity not in documents:
            payload = Path(item["path"]).read_bytes()
            need(hashlib.sha256(payload).hexdigest() == item["sha256"], "Incremental JSON changed while reading")
            value = json.loads(payload.decode("utf-8-sig"))
            need(isinstance(value, dict), "Incremental metadata must be a JSON object")
            documents[identity] = value
        return documents[identity]

    policy = document(selected)
    need(
        set(policy) == POLICY_FIELDS and policy.get("schema") == POLICY, "Unsupported or non-closed incremental policy"
    )
    need(
        policy["status"] == "frozen_root_reviewed" and policy["training_authorized"] is False,
        "An explicitly frozen reviewed policy without training authorization is required",
    )
    for name in ("policy_id",):
        need(isinstance(policy[name], str) and bool(policy[name]), "Missing incremental policy identifier")
    for name in ("canonical_data_root", "raw_data_root", "gvhmr_root", "consumer_root"):
        need(isinstance(policy[name], str) and Path(policy[name]).is_absolute(), "Incremental roots must be absolute")
    scope = document(policy["reviewed_scope"])
    need(
        set(scope) == SCOPE_FIELDS
        and scope.get("schema") == SCOPE
        and scope["review_status"] == "root_reviewed"
        and scope["training_authorized"] is False,
        "An exact root-reviewed incremental scope is required",
    )
    for name in ("scope_id", "reviewed_by", "reviewed_utc"):
        need(isinstance(scope[name], str) and bool(scope[name]), "Missing actual incremental scope review")
    for name in (
        "parent_release",
        "new_source_splits",
        "required_new_nonempty_splits",
        "required_union_nonempty_splits",
    ):
        need(policy[name] == scope[name], "Policy differs from its root-reviewed scope: " + name)
    need(policy["required_union_nonempty_splits"] == list(ROLES), "The complete union must retain all nonempty roles")
    need(isinstance(scope["review_sources"], dict) and bool(scope["review_sources"]), "Missing actual review sources")
    validate_roles(policy["new_source_splits"], scope["review_sources"], policy["required_new_nonempty_splits"])
    for sid, entry in scope["review_sources"].items():
        need(isinstance(entry, dict) and set(entry) == REVIEW_FIELDS, "Review source evidence is not closed")
        root = Path(entry["review_root"])
        need(root.is_absolute(), "Review roots must be absolute")
        for name in REVIEW_FIELDS - {"review_root"}:
            checked(entry[name])
        for name, relative in (
            ("pilot_input", "pilot_input.json"),
            ("review_source_registry", "source/source_registry.json"),
            ("review_source_input", "repair/new_source_input.json"),
            ("accepted_parent_snapshot", "accepted-base/calibration_snapshot.json"),
        ):
            need(
                Path(entry[name]["path"]).resolve() == (root / relative).resolve(), "Review evidence uses another root"
            )
        pilot, registry, source = (
            document(entry[name]) for name in ("pilot_input", "review_source_registry", "review_source_input")
        )
        need(
            pilot.get("manifest") == entry["review_manifest"] and pilot.get("review_only") is True,
            "Review pilot manifest/scope differs",
        )
        need(
            registry.get("review_only") is True
            and registry.get("pilot_manifest") == entry["review_manifest"]
            and registry.get("sources", {}).get(sid, {}).get("source_index") == entry["admitted_source_index"],
            "Review source registry differs from the accepted reviewed source",
        )
        need(
            source.get("schema") == "badminton-new-source-repair-input-v1"
            and source.get("review_only") is True
            and Path(source.get("source_root", "")).resolve() == (root / "source").resolve()
            and source.get("source_registry_sha256") == entry["review_source_registry"]["sha256"],
            "Review body input differs from its frozen source registry",
        )
    parent_bindings = policy["parent_release"]
    need(
        isinstance(parent_bindings, dict) and set(parent_bindings) == set(PARENT_FIELDS),
        "Require five actual parent release bindings",
    )
    archive, dataset, training, contract, audit = (document(parent_bindings[name]) for name in PARENT_FIELDS)
    archive_root = Path(parent_bindings["archive_manifest"]["path"]).resolve().parent
    need(
        archive_root.is_relative_to(Path(policy["canonical_data_root"]).resolve()),
        "Parent archive leaves canonical root",
    )
    need(
        archive.get("schema") == "lidar2smpl-release-v1" and archive.get("status") == "complete",
        "Parent archive is not a complete sealed release",
    )
    need(
        contract.get("schema") in ("expanded-1019-1020-consumer-contract/v1", FINAL)
        and contract.get("status") == "complete"
        and contract.get("sealed") is True,
        "Parent requires its actual sealed final consumer contract",
    )
    need(
        audit.get("status") == "final_cpu_validation_passed_no_training_started"
        and audit.get("validation_stage") == "final"
        and audit.get("contract") == parent_bindings["final_consumer_contract"]
        and all(audit.get(name) is False for name in ("training_authorized", "gpu_used", "formal_receipt_written")),
        "Parent requires its actual successful final consumer audit and exact contract",
    )
    for name in ("archive_manifest", "dataset_manifest", "training_manifest"):
        need(
            contract.get("final_delivery", {}).get(name) == parent_bindings[name],
            "Parent final delivery differs: " + name,
        )
    counts, roles = {}, {}
    registry = archive.get("source_registry", {})
    need(isinstance(registry, dict) and bool(registry), "Parent source registry is missing")
    for role in ROLES:
        item = archive["splits"][role]
        counts[role], roles[role] = integer(item["frames"]), item["declared_source_sessions"]
        need(counts[role] > 0, "The complete parent must retain every nonempty role")
        for kind in ("tensor", "metadata"):
            member = item[kind]
            relative = Path(member["path"])
            target = (archive_root / relative).resolve()
            need(
                not relative.is_absolute() and target.is_relative_to(archive_root), "Parent payload leaves its archive"
            )
            require_binding(dict(path=str(target), sha256=member["sha256"]))
            need(integer(member["bytes"]) == target.stat().st_size, "Parent payload size differs")
        need(
            sum(integer(registry[sid]["accepted"]) for sid in roles[role]) == counts[role]
            and all(registry[sid]["role"] == role for sid in roles[role]),
            "Parent source counts/roles differ",
        )
    validate_roles(roles, registry, list(ROLES))
    calibrations = archive.get("source_calibrations", {})
    for source in registry.values():
        reference = calibrations.get(source.get("calibration_digest"))
        need(
            isinstance(reference, dict) and reference.get("sha256") == source.get("calibration_snapshot_sha256"),
            "Parent source calibration differs from its archive",
        )
    total = sum(counts.values())
    need(integer(archive["total_person_frames"]) == total, "Parent archive total differs")
    for label, publication in (("dataset", dataset), ("training", training)):
        need(
            publication.get("schema") == "badminton-expanded-" + label + "-v1"
            and publication.get("status") == "complete"
            and publication.get("training_published") is True
            and publication.get("accepted") == total
            and publication.get("source_registry") == registry
            and set(publication.get("exports", {})) == set(registry)
            and Path(publication.get("canonical_archive", "")).resolve() == archive_root,
            "Parent producer publication differs from its actual archive",
        )
        for role in ROLES:
            entry, original = publication["suggested_session_splits"][role], archive["splits"][role]
            need(
                entry["accepted"] == counts[role] and entry["sessions"] == roles[role],
                "Parent publication role differs",
            )
            for field, sha_name, kind in (("path", "sha256", "tensor"), ("metadata", "metadata_sha256", "metadata")):
                need(
                    Path(entry[field]).resolve() == (archive_root / original[kind]["path"]).resolve()
                    and entry[sha_name] == original[kind]["sha256"],
                    "Parent publication payload differs",
                )
    new_counts = {sid: integer(item["final_admitted_count"]) for sid, item in contract["new_sources"].items()}
    count_map(contract["old_split_counts"], ROLES)
    parent_new_counts = count_map(contract["new_final_admitted_counts"], (*ROLES, "total"))
    need(
        parent_new_counts
        == dict(
            {role: sum(new_counts[sid] for sid in contract["new_source_splits"][role]) for role in ROLES},
            total=sum(new_counts.values()),
        ),
        "Parent final contract admitted counts differ",
    )
    integer(audit.get("old_retained_bit_exact"))
    integer(audit.get("new_retained_bit_exact"))
    count_map(audit.get("new_accepted_by_source"), new_counts)
    need(
        audit.get("old_retained_bit_exact") == integer(contract["old_accepted"])
        and audit.get("new_retained_bit_exact") == sum(new_counts.values())
        and audit.get("new_accepted_by_source") == new_counts
        and contract["old_accepted"] + sum(new_counts.values()) == total,
        "Parent successful audit population differs from its archive",
    )
    for role in ROLES:
        need(
            roles[role] == contract["old_source_splits"][role] + contract["new_source_splits"][role]
            and counts[role]
            == contract["old_split_counts"][role] + sum(new_counts[sid] for sid in contract["new_source_splits"][role]),
            "Parent final contract role differs",
        )
    seal = document(contract["release_evidence"]["seal"])
    need(
        seal.get("schema") == "badminton-expanded-release-receipt-v1"
        and seal.get("status") == "complete"
        and seal.get("accepted") == total
        and seal.get("old_accepted") == contract["old_accepted"]
        and seal.get("new_accepted") == sum(new_counts.values())
        and seal.get("splits") == counts
        and seal.get("all_old_rows_retained") is True
        and seal.get("source_numeric_values_unchanged") is True,
        "Parent seal is incomplete or differs from the final audit",
    )
    for name in ("archive_manifest", "dataset_manifest", "training_manifest"):
        need(seal.get(name) == parent_bindings[name], "Parent seal names another publication")
    verification = document(seal["archive_verification"])
    need(
        verification.get("passed") is True
        and verification.get("release_id") == archive.get("release_id")
        and verification.get("manifest_sha256") == parent_bindings["archive_manifest"]["sha256"]
        and all(verification.get("splits", {}).get(role, {}).get("frames") == counts[role] for role in ROLES),
        "Parent archive verification differs from the actual sealed archive",
    )
    bridges = [item for item in archive["artifacts"] if item["path"] == "provenance/model_bridge.json"]
    need(len(bridges) == 1, "Parent archive must bind exactly one model bridge artifact")
    bridge_path = (archive_root / bridges[0]["path"]).resolve()
    need(
        bridge_path.is_relative_to(archive_root) and bridge_path.stat().st_size == integer(bridges[0]["bytes"]),
        "Parent model bridge leaves its archive or has changed size",
    )
    bridge = checked(dict(path=str(bridge_path), sha256=bridges[0]["sha256"]))
    need(not set(registry).intersection(scope["review_sources"]), "Old/new incremental source scopes overlap")
    result = IncrementalRelease(
        selected,
        policy,
        scope,
        dict(
            bindings=parent_bindings,
            archive=archive,
            dataset=dataset,
            training=training,
            contract=contract,
            audit=audit,
            counts=counts,
            roles=roles,
            total=total,
            model_bridge=bridge,
        ),
        list(observed.values()),
    )
    result.recheck()
    return result


def policy_from_binding(item):
    require_binding(item)
    return load_release_policy(item["path"], item["sha256"])


def validate_contract(contract, policy):
    """Check v2 readiness against real policy/base metadata, never infer a policy."""
    from .export import FIELDS

    need(
        set(contract) == CONTRACT_FIELDS
        and contract.get("schema") == ASSEMBLY
        and contract.get("status") == "complete"
        and contract.get("sealed") is True,
        "An actual sealed v2 assembly input is required",
    )
    need(contract.get("release_policy") == policy.binding, "Assembly uses another explicit release policy")
    integer(contract["old_accepted"])
    count_map(contract["old_split_counts"], ROLES)
    count_map(contract["new_final_admitted_counts"], (*ROLES, "total"))
    need(
        contract.get("canonical_data_root") == policy.data["canonical_data_root"]
        and contract.get("old_accepted") == policy.parent["total"]
        and contract.get("old_split_counts") == policy.parent["counts"]
        and contract.get("old_source_splits") == policy.parent["roles"]
        and contract.get("new_source_splits") == policy.roles
        and contract.get("required_new_nonempty_splits") == policy.required,
        "Incremental contract changes its frozen base or whole-source scope",
    )
    items = contract.get("bindings", {})
    need(set(items) == CONTRACT_BINDINGS, "V2 requires exact actual bindings without a legacy parent receipt")
    for item in items.values():
        require_binding(item)
        need(file_sha(item["path"]) == item["sha256"], "Incremental contract binding changed")
    for name, parent_name in (
        ("old_archive", "archive_manifest"),
        ("old_dataset_manifest", "dataset_manifest"),
        ("old_training_manifest", "training_manifest"),
        ("parent_final_consumer_contract", "final_consumer_contract"),
        ("parent_final_consumer_audit", "final_consumer_audit"),
    ):
        need(items[name] == policy.parent["bindings"][parent_name], "Incremental parent binding differs")
    need(items["new_source_splits"] == policy.binding, "V2 source roles must bind its explicit policy")
    need(set(contract.get("new_sources", {})) == set(policy.scope["review_sources"]), "Incremental source set differs")
    for source in contract["new_sources"].values():
        need(
            isinstance(source, dict) and set(source) == ACTUAL_SOURCE_FIELDS,
            "Actual incremental source metadata is not closed",
        )
        for name in ("admitted_source_index", "camera_applicability", "calibration_snapshot", "formal_body_input"):
            require_binding(source[name])
            need(file_sha(source[name]["path"]) == source[name]["sha256"], "Actual incremental source evidence changed")
    totals = {
        role: sum(integer(contract["new_sources"][sid]["final_admitted_count"]) for sid in policy.roles[role])
        for role in ROLES
    }
    need(
        contract.get("new_final_admitted_counts") == dict(totals, total=sum(totals.values()))
        and all(totals[role] > 0 for role in policy.required),
        "Incremental admitted counts are pending or differ",
    )
    mode = contract.get("canonical_mode")
    need(mode in ("reviewed_cuts", "reviewed_no_changes"), "Unsupported explicit canonical mode")
    if mode == "reviewed_no_changes":
        from . import canonical_no_change as no_change

        def document(item):
            require_binding(item)
            payload = Path(item["path"]).read_bytes()
            need(hashlib.sha256(payload).hexdigest() == item["sha256"], "No-change contract evidence changed")
            value = json.loads(payload)
            need(isinstance(value, dict), "No-change contract metadata must be an object")
            return value

        manifest = document(items["producer_joint_manifest"])
        marker = manifest.get("canonical_identity_revision")
        need(
            manifest.get("schema") == no_change.PUBLICATION
            and manifest.get("status") == "complete"
            and manifest.get("release_policy") == policy.binding
            and isinstance(marker, dict)
            and set(marker) == no_change.MARKER_FIELDS
            and marker.get("mode") == mode
            and marker.get("numeric_geometry_unchanged") is True
            and marker.get("identity_keys_unchanged") is True,
            "Require the exact typed no-change publication for this policy",
        )
        proof = document(marker["verification"])
        require_binding(marker["source_publication"])
        need(
            set(proof) == no_change.PROOF_FIELDS
            and proof.get("schema") == no_change.PROOF
            and proof.get("state") == "complete"
            and proof.get("release_policy") == policy.binding
            and proof.get("source_completion") == items["producer_confirmation"]
            and proof.get("source_publication") == marker["source_publication"]
            and proof.get("reviewed_continuity") == marker["reviewed_continuity"]
            and Path(marker["source_publication"]["path"]).resolve()
            == Path(contract["new_repair_root"]).with_name("repair_paired").resolve() / "datasets/dataset_manifest.json"
            and integer(proof.get("accepted")) == integer(manifest.get("accepted")) == sum(totals.values())
            and integer(proof.get("full_accepted")) >= proof["accepted"]
            and set(proof.get("sessions", {})) == set(contract["new_sources"])
            and all(
                integer(proof["sessions"][sid].get("accepted")) == item["final_admitted_count"]
                for sid, item in contract["new_sources"].items()
            ),
            "No-change proof differs from its exact assembly source/completion/policy/population",
        )
    paired = contract["paired_schema"]
    need(
        paired.get("dtype") == "float32"
        and paired.get("target_body_model") == "neutral_smpl_24_joints"
        and paired.get("coordinate_space") == "lidar_viewer_xyz_m"
        and paired.get("transl_convention") == "shaped_pelvis_position"
        and all(paired.get(name) == ["N", *shape] for name, shape in FIELDS.items()),
        "Unsupported incremental paired body/coordinate/shape contract",
    )
    need(
        not any(name in contract for name in ("final_delivery", "release_evidence", "assembly_input")),
        "Assembly input cannot depend on later final outputs",
    )
    policy.recheck()
