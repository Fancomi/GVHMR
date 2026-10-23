"""Separate native-confirmed cross-person track boundaries without changing bodies.

This is a hash-bound identity view of an immutable bilaterally admitted source.
It changes canonical keys and real sequence membership, never numeric pose/cloud
values, observations, cameras, admission counts, or the original publication.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import joblib
import torch

from .dual_view import _verify_export_data, file_sha
from .expanded_dataset import (
    CONTRACT,
    _validate_new_completion,
    binding,
    check_binding,
    load_export,
    merge_role,
    sample_key,
    source_roles,
)
from .export import FIELDS
from .rally_pipeline import read_json, read_rows
from .repair_execution import write_progress_json as write_json


class _CutBindings:
    """Hold the explicit v2 inputs until the publication completion marker."""

    def __init__(self):
        self.items = {}

    def add(self, item):
        from .incremental_release import require_binding

        require_binding(item)
        path = str(Path(item["path"]).resolve())
        previous = self.items.get(os.path.normcase(path))
        if previous is not None and previous != item:
            raise ValueError("Conflicting canonical dependency bindings")
        check_binding(item)
        self.items[os.path.normcase(path)] = dict(item)
        return item

    def bind(self, path):
        return self.add(binding(path))

    def read(self, item):
        self.add(item)
        value = read_json(item["path"])
        check_binding(item)
        return value

    def entry(self, entry):
        for name, sha in (("path", "sha256"), ("metadata", "metadata_sha256")):
            self.add(dict(path=entry[name], sha256=entry[sha]))

    def recheck(self):
        for item in self.items.values():
            check_binding(item)


def _incremental_source(source_path, source, review_path, policy):
    """Validate current v2 metadata without rerunning native/model verification."""
    frozen = _CutBindings()
    for item in policy.dependencies:
        frozen.add(item)
    source_binding = frozen.bind(source_path)
    if frozen.read(source_binding) != source:
        raise ValueError("Canonical source changed while its inputs were selected")
    review = frozen.read(frozen.bind(review_path))
    if (
        source.get("schema") != "badminton-ptv3-dualcam-dataset-v1"
        or source.get("status") != "complete"
        or source.get("review_only") is True
        or "canonical_identity_revision" in source
        or any(source.get(name) != value for name, value in CONTRACT.items())
        or ("release_policy" in source and source["release_policy"] != policy.binding)
        or "mode" in review
    ):
        raise ValueError("V2 cuts require the actual original completed bilateral publication")
    source_roles(source, required_nonempty=policy.required)
    if {role: entry["sessions"] for role, entry in source["suggested_session_splits"].items()} != policy.roles:
        raise ValueError("Canonical source roles differ from the explicit release policy")
    body_binding = dict(path=source["source_dataset_manifest"], sha256=source["source_dataset_manifest_sha256"])
    body = frozen.read(body_binding)
    labels = Path(body_binding["path"]).resolve().parent.parent
    repair = labels.parent
    if labels.name != "final_labels" or Path(source_path).resolve() != (
        repair.with_name(repair.name + "_paired") / "datasets/dataset_manifest.json"
    ):
        raise ValueError("Canonical source is not the actual final-label paired publication")
    registry_binding = frozen.bind(repair.parent / "source/source_registry.json")
    registry = frozen.read(registry_binding)
    source_input = frozen.read(frozen.bind(repair / "new_source_input.json"))
    calibration = frozen.read(frozen.bind(labels / "calibration/calibration_snapshot.json"))
    if (
        source_input.get("schema") != "badminton-new-source-repair-input-v1"
        or source_input.get("review_only") is True
        or registry.get("review_only") is True
        or source_input.get("release_policy") != policy.binding
        or registry.get("release_policy") != policy.binding
        or source_input.get("session_splits") != policy.roles
        or source_input.get("required_nonempty_splits") != policy.required
        or Path(source_input.get("source_root", "")).resolve() != Path(registry_binding["path"]).parent
        or source_input.get("source_registry_sha256") != registry_binding["sha256"]
        or set(registry.get("sources", {})) != set(policy.scope["review_sources"])
        or set(source["exports"]) != set(body["exports"])
        or set(source["exports"]) != set(registry.get("sources", {}))
        or body.get("review_only") is True
        or source["calibration_digest"] != body.get("calibration_digest")
        or source["calibration_digest"] != calibration.get("snapshot_digest")
    ):
        raise ValueError("Canonical source input/registry/calibration differs from its actual policy scope")
    for sid, entry in registry["sources"].items():
        if entry.get("source_index") != policy.scope["review_sources"][sid]["admitted_source_index"]:
            raise ValueError("Canonical source admission differs from the reviewed policy scope")
    for publication in (source, body):
        for entry in publication["exports"].values():
            frozen.entry(entry)
            load_export(entry)
    for entry in source["suggested_session_splits"].values():
        frozen.entry(entry)
    completion = frozen.read(frozen.bind(repair / "new_source_completion_receipt.json"))
    if (
        completion.get("state") != "labels_published_viewer_prepared"
        or completion.get("dataset_manifest") != source_binding["path"]
        or type(completion.get("accepted")) is not int
        or completion["accepted"] != source["accepted"]
        or not isinstance(completion.get("evidence"), dict)
    ):
        raise ValueError("Canonical source completion does not seal this publication")
    for path, sha in completion["evidence"].items():
        frozen.add(dict(path=path, sha256=sha))
    for item in _validate_new_completion(repair, completion, source_binding, source):
        frozen.add(item)
    audit = frozen.read(review["audit"])
    native = frozen.read(audit["native_verification"])
    aggregate = frozen.read(audit["aggregate_native_contract"])
    count = body.get("accepted")
    if (
        audit.get("schema") != "badminton-fresh-continuity-audit-v1"
        or audit.get("state") != "complete"
        or audit.get("body_manifest") != body_binding
        or type(count) is not int
        or audit.get("counts", {}).get("person_frames") != count
        or native.get("state") != "complete"
        or native.get("passed") is not True
        or native.get("accepted_checked") != count
        or native.get("calibration_digest") != source["calibration_digest"]
        or Path(audit["native_verification"]["path"]).resolve() != labels / "native_verification.json"
        or completion["evidence"].get(audit["native_verification"]["path"]) != audit["native_verification"]["sha256"]
        or aggregate.get("passed") is not True
        or aggregate.get("checked") != count
        or aggregate.get("all_point_and_label_tensors_equal_native_verified_windows") is not True
        or aggregate.get("native_verification") != audit["native_verification"]["path"]
        or aggregate.get("native_verification_sha256") != audit["native_verification"]["sha256"]
        or set(audit.get("sources", {})) != set(body["exports"])
    ):
        raise ValueError("Canonical cut review does not bind the completed full native continuity audit")
    for sid, entry in body["exports"].items():
        expected = [
            dict(path=entry[name], sha256=entry[sha])
            for name, sha in (("path", "sha256"), ("metadata", "metadata_sha256"))
        ]
        if (
            audit["sources"][sid].get("person_frames") != entry["accepted"]
            or audit["sources"][sid].get("dependencies") != expected
        ):
            raise ValueError("Full continuity audit source/count bindings differ")
    # checked_review owns the unchanged real-cut identity rule. Keep every byte
    # it consumed live until the final completion marker, including pixel proof.
    for case in review["cases"]:
        frozen.add(case["image"])
        witness = frozen.read(case["evidence"])
        for record in witness["records"]:
            frozen.add(record["pose"])
            frozen.add(record["fit"])
    frozen.recheck()
    return frozen


def _empty_role(entry, destination):
    """Copy a real empty role; never manufacture an empty tensor or metadata."""
    if type(entry.get("accepted")) is not int or entry["accepted"] != 0 or entry.get("sessions") != []:
        raise ValueError("Only an actual zero-count role with no sources may use the v2 empty-copy path")
    load_export(entry)
    result = dict(entry)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name, sha, target in (
        ("path", "sha256", destination),
        ("metadata", "metadata_sha256", destination.with_suffix(".metadata.jsonl")),
    ):
        original = Path(entry[name]).resolve()
        size = original.stat().st_size
        with original.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        if original.samefile(target) or target.stat().st_size != size or file_sha(target) != entry[sha]:
            raise ValueError("Canonical empty-role copy is not an independent byte-identical original")
        result[name] = str(target)
    load_export(result)
    return result


def _verify_empty_role(previous, entry):
    for value in (previous, entry):
        if type(value.get("accepted")) is not int or value["accepted"] != 0 or value.get("sessions") != []:
            raise ValueError("Canonical empty role changed its original membership")
        load_export(value)
    for name, sha in (("path", "sha256"), ("metadata", "metadata_sha256")):
        original, revised = Path(previous[name]), Path(entry[name])
        if (
            original.samefile(revised)
            or original.stat().st_size != revised.stat().st_size
            or entry[sha] != previous[sha]
        ):
            raise ValueError("Canonical empty role must independently preserve original bytes")


def checked_review(path):
    plan = read_json(path)
    if (
        plan.get("schema") != "badminton-reviewed-canonical-track-boundaries-v1"
        or plan.get("review_status") != "reviewed"
    ):
        raise ValueError("Canonical changes require explicit reviewed native evidence")
    check_binding(plan["audit"])
    audit = read_json(plan["audit"]["path"])
    known = audit["pelvis_jump_candidates"]
    cuts, identities = {}, set()
    for case in plan["cases"]:
        check_binding(case["evidence"])
        check_binding(case["image"])
        witness = read_json(case["evidence"]["path"])
        event = case["event"]
        if witness["event"] != event or known[case["case"]] != event or witness["image"] != case["image"]:
            raise ValueError("Canonical review and native before/after witness disagree")
        if {(r["side"], r["camera"]) for r in witness["records"]} != {
            (s, c) for s in ("before", "after") for c in ("cam1", "cam2")
        } or len(witness["records"]) != 4:
            raise ValueError("Require original before/after witnesses from both cameras")
        for record in witness["records"]:
            for name in ("pose", "fit"):
                check_binding(record[name])
            pose = read_json(record["pose"]["path"])
            if pose["raw_bgr_sha256"] != record["raw_bgr_sha256"]:
                raise ValueError("Native pixel provenance changed")
        if case["verdict"] == "same_person":
            continue
        if case["verdict"] != "different_person" or event["after_scan"] != event["before_scan"] + 1:
            raise ValueError("Only witnessed cross-person boundaries may split canonical identity")
        target = int(case["new_track_id"])
        if target < 1000000 or target in identities:
            raise ValueError("Canonical split IDs must be unique and separate from native detector IDs")
        identities.add(target)
        cuts.setdefault(event["motion_id"], []).append(
            dict(scan=event["after_scan"], track_id=target, case=case["case"])
        )
    for values in cuts.values():
        values.sort(key=lambda value: value["scan"])
        if len({v["scan"] for v in values}) != len(values):
            raise ValueError("Duplicate canonical boundary")
    if not cuts:
        raise ValueError("There are no reviewed cross-person changes to apply")
    return cuts


def remap_row(row, cuts, review_sha):
    original = row["motion_id"]
    action = next((cut for cut in reversed(cuts.get(original, [])) if int(row["lidar_index"]) >= cut["scan"]), None)
    if action is None:
        return row
    pieces = original.split(":")
    if len(pieces) != 4 or int(pieces[2]) != int(row["track_id"]):
        raise ValueError("Unexpected source canonical identity format")
    pieces[2] = str(action["track_id"])
    return dict(
        row,
        motion_id=":".join(pieces),
        track_id=action["track_id"],
        canonical_identity=dict(
            source_motion_id=original,
            source_track_id=row["track_id"],
            boundary_case=action["case"],
            reviewed_boundaries_sha256=review_sha,
        ),
    )


def verify_canonical_publication(manifest):
    marker = manifest["canonical_identity_revision"]
    for name in ("source_publication", "reviewed_boundaries"):
        check_binding(marker[name])
    original = read_json(marker["source_publication"]["path"])
    policy, frozen = None, None
    if "release_policy" not in manifest and any(
        not entry["sessions"] for entry in original["suggested_session_splits"].values()
    ):
        raise ValueError("Empty canonical roles require an explicit policy-bound publication")
    if "release_policy" in manifest:
        from .incremental_release import policy_from_binding

        policy = policy_from_binding(manifest["release_policy"])
        if (
            manifest.get("schema") != "badminton-canonical-bilateral-publication-v1"
            or "mode" in manifest
            or "mode" in marker
            or marker.get("numeric_geometry_unchanged") is not True
            or marker.get("strategy") != "split_native_confirmed_cross_person_boundaries_no_unproven_join"
        ):
            raise ValueError("V2 canonical cuts cannot use a no-change or unknown publication mode")
        frozen = _incremental_source(
            marker["source_publication"]["path"], original, marker["reviewed_boundaries"]["path"], policy
        )
        expected = dict(
            original,
            schema="badminton-canonical-bilateral-publication-v1",
            exports=manifest["exports"],
            suggested_session_splits=manifest["suggested_session_splits"],
            canonical_identity_revision=marker,
            release_policy=policy.binding,
        )
        if manifest != expected:
            raise ValueError("V2 canonical publication changed original non-identity fields")
    original_verification = _verify_export_data(original)
    cuts = checked_review(marker["reviewed_boundaries"]["path"])
    review_sha = marker["reviewed_boundaries"]["sha256"]
    result = dict(
        sessions={},
        numeric_values_unchanged=True,
        admission_membership_unchanged=True,
        split_only_no_unproven_join=True,
    )
    if manifest["accepted"] != original["accepted"] or set(manifest["exports"]) != set(original["exports"]):
        raise ValueError("Canonical revision changed source membership")
    for sid, entry in manifest["exports"].items():
        previous = original["exports"][sid]
        for key, digest in (("path", "sha256"), ("metadata", "metadata_sha256")):
            if file_sha(entry[key]) != entry[digest]:
                raise ValueError("Canonical artifact changed")
        a, b = joblib.load(previous["path"]), joblib.load(entry["path"])
        rows, changed = read_rows(previous["metadata"]), read_rows(entry["metadata"])
        expected = [remap_row(row, cuts, review_sha) for row in rows]
        if changed != expected or b["motion_id"] != [row["motion_id"] for row in expected]:
            raise ValueError("Canonical metadata does not reproduce the reviewed rule")
        if any(not torch.equal(a[key], b[key]) for key in FIELDS):
            raise ValueError("An identity-only revision changed actual cloud or SMPL numbers")
        if len({sample_key(row) for row in changed}) != len(changed):
            raise ValueError("Canonical revision introduced duplicate person-scans")
        result["sessions"][sid] = dict(
            accepted=len(changed),
            changed_canonical_keys=sum(x["motion_id"] != y["motion_id"] for x, y in zip(rows, changed)),
            numeric_values_bit_exact=True,
        )
    result.update(accepted=original_verification["accepted"], reviewed_boundary_count=sum(map(len, cuts.values())))
    if policy is not None:
        source_roles(manifest, required_nonempty=policy.required)
        if {role: entry["sessions"] for role, entry in manifest["suggested_session_splits"].items()} != policy.roles:
            raise ValueError("Canonical revision changed policy whole-source roles/order")
        for entry in manifest["exports"].values():
            frozen.entry(entry)
        for role, entry in manifest["suggested_session_splits"].items():
            if not entry["sessions"]:
                _verify_empty_role(original["suggested_session_splits"][role], entry)
            else:
                data, rows = load_export(entry)
                parts = [load_export(manifest["exports"][sid]) for sid in entry["sessions"]]
                if rows != [row for _, local in parts for row in local] or any(
                    not torch.equal(data[name], torch.cat([part[name] for part, _ in parts], dim=0)) for name in FIELDS
                ):
                    raise ValueError("Canonical role changed source values or metadata order")
            frozen.entry(entry)
        policy.recheck()
        frozen.recheck()
    return result


def publish_canonical_view(source_path, review_path, output_root, *, release_policy=None, release_policy_sha256=None):
    if (release_policy is None) != (release_policy_sha256 is None):
        raise ValueError("Supply release-policy path and SHA256 together")
    source_path, review_path, output = map(lambda p: Path(p).resolve(), (source_path, review_path, output_root))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Keep original and revised publications immutable; use a fresh destination")
    source = read_json(source_path)
    policy, frozen = None, None
    if release_policy is not None:
        from .incremental_release import load_release_policy

        policy = load_release_policy(release_policy, release_policy_sha256)
        frozen = _incremental_source(source_path, source, review_path, policy)
    elif "release_policy" in source:
        raise ValueError("A v2 source requires its explicit release-policy path and SHA256")
    _verify_export_data(source)
    cuts = checked_review(review_path)
    review = binding(review_path)
    manifest = deepcopy(source)
    manifest["schema"] = "badminton-canonical-bilateral-publication-v1"
    manifest["canonical_identity_revision"] = dict(
        source_publication=binding(source_path),
        reviewed_boundaries=review,
        numeric_geometry_unchanged=True,
        strategy="split_native_confirmed_cross_person_boundaries_no_unproven_join",
    )
    if policy is not None:
        manifest["release_policy"] = deepcopy(policy.binding)
    exports = {}
    for sid, entry in source["exports"].items():
        data, rows = joblib.load(entry["path"]), read_rows(entry["metadata"])
        changed = [remap_row(row, cuts, review["sha256"]) for row in rows]
        if any(int(row["track_id"]) >= 1000000 for row in rows):
            raise ValueError("Canonical split range collides with an existing source track")
        value = dict(data, motion_id=[row["motion_id"] for row in changed])
        path = output / "datasets" / (sid + ".pth")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(value, path)
        metadata = path.with_suffix(".metadata.jsonl")
        with metadata.open("w", encoding="utf-8", newline="\n") as stream:
            for row in changed:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        exports[sid] = dict(
            entry, path=str(path), metadata=str(metadata), sha256=file_sha(path), metadata_sha256=file_sha(metadata)
        )
        write_json(output / "progress.json", dict(state="writing_canonical_sources", completed_sources=list(exports)))
        print(
            dict(
                source=sid,
                accepted=len(rows),
                changed=sum(x["motion_id"] != y["motion_id"] for x, y in zip(rows, changed)),
            ),
            flush=True,
        )
    manifest["exports"] = exports
    digests = {sid: source["calibration_digest"] for sid in exports}
    splits = {}
    for role, entry in source["suggested_session_splits"].items():
        parts = [dict(exports[sid], sessions=[sid]) for sid in entry["sessions"]]
        destination = output / "datasets" / (role + ".pth")
        if policy is not None and not parts:
            splits[role] = _empty_role(entry, destination)
        else:
            splits[role], _ = merge_role(parts, destination, digests)
    manifest["suggested_session_splits"] = splits
    proof = verify_canonical_publication(manifest)
    write_json(output / "canonical_verification.json", proof)
    manifest["canonical_identity_revision"]["verification"] = binding(output / "canonical_verification.json")
    if frozen is not None:
        for entry in [*exports.values(), *splits.values()]:
            frozen.entry(entry)
        frozen.add(manifest["canonical_identity_revision"]["verification"])
        policy.recheck()
        frozen.recheck()
    write_json(output / "datasets/dataset_manifest.json", manifest)
    return manifest
