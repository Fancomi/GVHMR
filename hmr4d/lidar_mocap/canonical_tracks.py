"""Separate native-confirmed cross-person track boundaries without changing bodies.

This is a hash-bound identity view of an immutable bilaterally admitted source.
It changes canonical keys and real sequence membership, never numeric pose/cloud
values, observations, cameras, admission counts, or the original publication.
"""

from copy import deepcopy
import json
from pathlib import Path

import joblib
import torch

from .dual_view import _verify_export_data, file_sha
from .expanded_dataset import binding, check_binding, merge_role, sample_key
from .export import FIELDS
from .rally_pipeline import read_json, read_rows
from .repair_execution import write_progress_json as write_json


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
    return result


def publish_canonical_view(source_path, review_path, output_root):
    source_path, review_path, output = map(lambda p: Path(p).resolve(), (source_path, review_path, output_root))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Keep original and revised publications immutable; use a fresh destination")
    source = read_json(source_path)
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
        splits[role], _ = merge_role(parts, output / "datasets" / (role + ".pth"), digests)
    manifest["suggested_session_splits"] = splits
    proof = verify_canonical_publication(manifest)
    write_json(output / "canonical_verification.json", proof)
    manifest["canonical_identity_revision"]["verification"] = binding(output / "canonical_verification.json")
    write_json(output / "datasets/dataset_manifest.json", manifest)
    return manifest
