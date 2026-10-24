"""Assemble full repaired publications without altering labels, roles or metadata.

Assembly is deliberately provisional: native pixel and browser acceptance must
be completed before either consumer-facing manifest is published.
"""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil

import joblib
import torch

from hmr4d import PROJ_ROOT

from .calibration import CalibrationSnapshot
from .dual_view import _verify_export_data, file_sha
from .export import FIELDS
from .pipeline import write_json
from .rally_pipeline import read_json, read_rows


ROLES = ("train", "val", "test")
JOINT_CONTRACTS = (
    "expanded-1019-1020-assembly-input/v1",
    "expanded-1019-1020-consumer-contract/v1",
    "expanded-source-increment-assembly-input/v2",
)
CONTRACT = dict(
    qualification_policy="bilateral_detection",
    target_body_model="neutral_smpl_24_joints",
    gender="neutral",
    coordinate_space="lidar_viewer_xyz_m",
    point_count=256,
    transl_convention="shaped_pelvis_position",
)


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_sha(path))


def check_binding(entry):
    if file_sha(entry["path"]) != entry["sha256"]:
        raise ValueError("Frozen dependency changed: " + entry["path"])


def source_roles(manifest, *, required_nonempty=ROLES):
    roles = {}
    if set(manifest["suggested_session_splits"]) != set(ROLES):
        raise ValueError("Exactly train, val and test are required")
    for role, entry in manifest["suggested_session_splits"].items():
        for sid in entry["sessions"]:
            if sid in roles:
                raise ValueError("A source is assigned more than once: " + sid)
            roles[sid] = role
        count = sum(manifest["exports"][sid]["accepted"] for sid in entry["sessions"])
        if count != entry["accepted"] or (role in required_nonempty and count < 1):
            raise ValueError("Partition count differs from source counts or is empty: " + role)
    if set(roles) != set(manifest["exports"]):
        raise ValueError("Every declared source, including zero sources, needs a role")
    if sum(e["accepted"] for e in manifest["exports"].values()) != manifest["accepted"]:
        raise ValueError("Publication count differs from its sources")
    return roles


def validate_union_contract(old, new, contract, *, incremental=None):
    schema = contract.get("schema", "")
    if (
        not isinstance(schema, str)
        or "draft" in schema
        or (schema in JOINT_CONTRACTS and (contract.get("status") != "complete" or contract.get("sealed") is not True))
    ):
        raise ValueError("An unsealed consumer draft cannot assemble a formal expansion")
    for manifest in (old, new):
        if manifest.get("status") != "complete" or any(manifest.get(k) != v for k, v in CONTRACT.items()):
            raise ValueError("Only complete publications with the agreed body/detection contract can be merged")
    if schema == "expanded-source-increment-assembly-input/v2":
        if incremental is None:
            raise ValueError("V2 assembly requires its explicitly selected release policy")
        from .incremental_release import validate_contract

        validate_contract(contract, incremental)
    elif incremental is not None or schema == "expanded-source-increment-consumer-contract/v2":
        raise ValueError("Select an explicit v2 assembly input, never a final contract or legacy policy override")
    old_roles = source_roles(old)
    new_roles = source_roles(new, required_nonempty=incremental.required if incremental is not None else ROLES)
    if set(old_roles).intersection(new_roles):
        raise ValueError("Old and new publications contain overlapping sources")
    if old["accepted"] != contract["old_accepted"]:
        raise ValueError("Use the complete old qualified dataset, not a truncated training view")
    if {r: old["suggested_session_splits"][r]["accepted"] for r in ROLES} != contract["old_split_counts"]:
        raise ValueError("Complete old role counts changed")
    for tag, manifest in (("old", old), ("new", new)):
        for role in ROLES:
            if manifest["suggested_session_splits"][role]["sessions"] != contract[tag + "_source_splits"][role]:
                raise ValueError("Whole-source split or order changed: " + tag + ":" + role)
    if schema in JOINT_CONTRACTS:
        sources = contract.get("new_sources", {})
        if set(sources) != set(new["exports"]):
            raise ValueError("Joint input must declare every actual new admitted source")
        for sid, entry in new["exports"].items():
            count = sources[sid].get("final_admitted_count")
            if isinstance(count, bool) or not isinstance(count, int) or count != entry["accepted"]:
                raise ValueError("Joint input source count is pending or differs from the new publication")
    return dict(old_roles, **new_roles)


def sample_key(row):
    # Detector identities are canonical only inside their native fitting window.
    # Segment IDs must not allow two labels for the same canonical person/scan.
    return row["session_id"], row["window_id"], int(row["track_id"]), int(row["lidar_index"])


def contiguous_sequences(rows):
    tracks = defaultdict(list)
    for index, row in enumerate(rows):
        tracks[row["session_id"], row["motion_id"]].append((int(row["lidar_index"]), int(row["scan_mid_us"]), index))
    result = []
    for (sid, motion), frames in sorted(tracks.items()):
        chunk = []
        for frame in sorted(frames):
            if chunk and (frame[0] != chunk[-1][0] + 1 or not 0 < frame[1] - chunk[-1][1] <= 150000):
                result.append(
                    dict(session_id=sid, motion_id=motion, scans=[v[0] for v in chunk], indices=[v[2] for v in chunk])
                )
                chunk = []
            chunk.append(frame)
        if chunk:
            result.append(
                dict(session_id=sid, motion_id=motion, scans=[v[0] for v in chunk], indices=[v[2] for v in chunk])
            )
    return result


def load_export(entry):
    if file_sha(entry["path"]) != entry["sha256"] or file_sha(entry["metadata"]) != entry["metadata_sha256"]:
        raise ValueError("Source tensor/metadata changed")
    data, rows = joblib.load(entry["path"]), read_rows(entry["metadata"])
    if len(rows) != entry["accepted"] or data["motion_id"] != [r["motion_id"] for r in rows]:
        raise ValueError("Source metadata count or tensor order differs")
    for name, shape in FIELDS.items():
        value = data[name]
        if value.dtype != torch.float32 or tuple(value.shape) != (len(rows), *shape) or not torch.isfinite(value).all():
            raise ValueError("Source violates the finite float32 tensor contract: " + name)
    return data, rows


def merge_role(entries, destination, source_digests, *, preserve_old_bytes=False):
    """Append exact new rows after the full old role, preserving metadata bytes."""
    parts, rows, keys = [], [], set()
    for entry in entries:
        data, local = load_export(entry)
        for row in local:
            sid = row["session_id"]
            if sid not in entry["sessions"] or row["calibration_digest"] != source_digests[sid]:
                raise ValueError("Role metadata has the wrong source or calibration")
            key = sample_key(row)
            if key in keys:
                raise ValueError("Duplicate canonical person/scan; refusing to pad or silently discard a label")
            keys.add(key)
        parts.append(data)
        rows.extend(local)
    if preserve_old_bytes:
        if (
            len(entries) != 2
            or entries[1]["accepted"] != 0
            or parts[1]["motion_id"]
            or len(rows) != entries[0]["accepted"]
            or Path(entries[1]["metadata"]).stat().st_size != 0
        ):
            raise ValueError("An unchanged role requires an actually empty new tensor and metadata")
        merged = parts[0]
    else:
        merged = {name: torch.cat([part[name] for part in parts]) for name in FIELDS}
        merged["motion_id"] = [m for part in parts for m in part["motion_id"]]
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pth.tmp")
    if preserve_old_bytes:
        shutil.copyfile(entries[0]["path"], temporary)
    else:
        joblib.dump(merged, temporary)
    temporary.replace(destination)
    metadata = destination.with_suffix(".metadata.jsonl")
    temporary = metadata.with_suffix(".jsonl.tmp")
    with temporary.open("wb") as output:
        for entry in entries:
            with Path(entry["metadata"]).open("rb") as source:
                source.seek(0, 2)
                if source.tell():
                    source.seek(-1, 2)
                    if source.read(1) != b"\n":
                        raise ValueError("Preserving metadata bytes requires newline-terminated source rows")
                source.seek(0)
                shutil.copyfileobj(source, output)
    temporary.replace(metadata)
    entry = dict(
        accepted=len(rows),
        path=str(destination),
        metadata=str(metadata),
        fields={name: list(merged[name].shape) for name in FIELDS},
        sessions=[sid for e in entries for sid in e["sessions"]],
        coordinate_space=CONTRACT["coordinate_space"],
        transl_convention=CONTRACT["transl_convention"],
        cloud_source="segmented_actual_fixed24_returns",
        sha256=file_sha(destination),
        metadata_sha256=file_sha(metadata),
        original_role_prefix_samples=entries[0]["accepted"],
        components=[
            dict(tensor=binding(e["path"]), metadata=binding(e["metadata"]), accepted=e["accepted"]) for e in entries
        ],
    )
    if preserve_old_bytes and (
        entry["sha256"] != entries[0]["sha256"]
        or entry["metadata_sha256"] != entries[0]["metadata_sha256"]
        or destination.samefile(entries[0]["path"])
        or metadata.samefile(entries[0]["metadata"])
    ):
        raise ValueError("An unchanged role must independently preserve both complete parent payload bytes")
    check, check_rows = load_export(entry)
    offset = 0
    for part in parts:
        end = offset + len(part["motion_id"])
        if check["motion_id"][offset:end] != part["motion_id"] or any(
            not torch.equal(check[name][offset:end], part[name]) for name in FIELDS
        ):
            raise ValueError("Merged tensor changed source values")
        offset = end
    if check_rows != rows:
        raise ValueError("Merged metadata changed source values")
    return entry, rows


def _archive_member(root, entry):
    relative = Path(entry["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Canonical base artifact escapes its archive")
    if path.stat().st_size != entry["bytes"]:
        raise ValueError("Canonical base artifact size changed")
    return path


def verify_expanded_base(manifest, contract):
    """Reuse a sealed multi-calibration base; role payload hashes are checked by merge_role."""
    archive_binding = contract.get("bindings", {}).get("old_archive")
    if not archive_binding:
        raise ValueError("An expanded base requires the consumer's explicit old archive binding")
    check_binding(archive_binding)
    archive_path = Path(archive_binding["path"]).resolve()
    root = archive_path.parent
    archive = read_json(archive_path)
    if (archive.get("schema") != "lidar2smpl-release-v1" or archive.get("status") != "complete"
            or manifest.get("training_published") is not True
            or Path(manifest.get("canonical_archive", "")).resolve() != root
            or archive.get("total_person_frames") != manifest["accepted"]
            or manifest["accepted"] != contract["old_accepted"]):
        raise ValueError("Expanded base is not the contracted complete canonical archive")
    registry = manifest.get("source_registry", {})
    if not registry or registry != archive.get("source_registry") or set(registry) != set(manifest["exports"]):
        raise ValueError("Expanded base source registry differs from its canonical archive")
    roles = source_roles(manifest)
    for role in ROLES:
        entry, original = manifest["suggested_session_splits"][role], archive["splits"][role]
        if (entry["accepted"] != original["frames"] or entry["accepted"] != contract["old_split_counts"][role]
                or entry["sessions"] != original["declared_source_sessions"]
                or entry["sessions"] != contract["old_source_splits"][role]):
            raise ValueError("Expanded base role membership differs from its canonical archive")
        for field, hash_field, member in (("path", "sha256", "tensor"),
                                           ("metadata", "metadata_sha256", "metadata")):
            path = _archive_member(root, original[member])
            if Path(entry[field]).resolve() != path or entry[hash_field] != original[member]["sha256"]:
                raise ValueError("Expanded base role payload differs from its canonical archive")
    dependencies, calibrations = [archive_binding], {}
    for digest, reference in archive["source_calibrations"].items():
        path = _archive_member(root, reference)
        item = dict(path=str(path), sha256=reference["sha256"])
        check_binding(item)
        calibration = CalibrationSnapshot.load(path)
        if calibration.data["snapshot_digest"] != digest or calibration.data.get("review_only"):
            raise ValueError("Archived source calibration has a different digest or review-only scope")
        dependencies.append(item)
        calibrations[digest] = reference["sha256"]
    for sid, entry in registry.items():
        if (entry["role"] != roles[sid] or entry["accepted"] != manifest["exports"][sid]["accepted"]
                or calibrations.get(entry["calibration_digest"]) != entry["calibration_snapshot_sha256"]):
            raise ValueError("Expanded source count, role or calibration differs from its archive")
        item = dict(path=entry["calibration_snapshot"], sha256=entry["calibration_snapshot_sha256"])
        if item not in dependencies:
            check_binding(item)
            dependencies.append(item)
    publications = manifest.get("source_publications", {})
    if not publications:
        raise ValueError("Expanded base lacks its original source-publication lineage")
    for item in publications.values():
        check_binding(item)
        dependencies.append(item)
    return registry, dict(
        accepted=manifest["accepted"], canonical_archive=archive_binding,
        source_calibrations=list(calibrations), source_registry_matches_archive=True,
        role_payload_checks="Full tensor/metadata SHA and row checks performed by merge_role before assembly completes",
    ), dependencies


def _validate_new_completion(repair, completion, parent_publication, publication):
    """Bind an already verified formal completion without rerunning its CPU/native passes."""
    evidence = completion.get("evidence", {})
    if not isinstance(evidence, dict) or evidence.get(parent_publication["path"]) != parent_publication["sha256"]:
        raise ValueError("New source completion does not seal this dataset")
    labels = repair / "final_labels"
    registry_path = labels / "source_registry.json"
    formal = registry_path.exists() and read_json(registry_path).get("schema") == (
        "badminton-formal-derived-source-registry-v1"
    )
    if not formal:
        if not completion.get("training_published") or completion.get("formal_postleg_cpu_receipt") is not None:
            raise ValueError("New source completion does not seal this dataset")
        return []
    if (
        repair.name != "repair"
        or completion.get("state") != "labels_published_viewer_prepared"
        or completion.get("training_published") is not False
        or completion.get("bilateral_labels_published") is not True
        or completion.get("archive_published") is not False
        or completion.get("consumer_admitted") is not False
        or completion.get("dataset_manifest") != parent_publication["path"]
        or type(completion.get("accepted")) is not int
        or completion["accepted"] != publication["accepted"]
    ):
        raise ValueError("Formal completion must seal bilateral labels while archive/consumer admission stays pending")
    dependencies = []

    def bound(item, expected=None):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
            or (expected is not None and Path(item["path"]).resolve() != expected.resolve())
        ):
            raise ValueError("Formal completion requires exact existing evidence paths and SHA256 bindings")
        check_binding(item)
        if item not in dependencies:
            dependencies.append(item)
        return item

    def completed(path):
        return bound(dict(path=str(path), sha256=evidence.get(str(path))), path)

    registry_binding = bound(binding(registry_path))
    registry = read_json(registry_binding["path"])
    plan_path = labels / "repair_plan.json"
    plan = read_json(bound(binding(plan_path))["path"])
    source = bound(
        dict(path=plan["source_union"], sha256=plan["source_union_sha256"]), repair / "new_source_input.json"
    )
    source_input = read_json(source["path"])
    body_path = labels / "datasets/dataset_manifest.json"
    body_binding = completed(body_path)
    body = read_json(body_path)
    paired = repair.with_name(repair.name + "_paired")
    bound(parent_publication, paired / "datasets/dataset_manifest.json")
    receipt_binding = bound(
        completion.get("formal_postleg_cpu_receipt"), repair / "evaluation/formal_postleg_cpu_receipt.json"
    )
    if completed(Path(receipt_binding["path"])) != receipt_binding:
        raise ValueError("Formal completion does not bind its postleg CPU receipt")
    receipt = read_json(receipt_binding["path"])
    if "source_registry_sha256" in body:
        body_registry_matches = body["source_registry_sha256"] == registry_binding["sha256"]
    else:
        body_registry_matches = (
            body.get("schema") == "badminton-ptv3-dataset-v1"
            and body.get("source_union_manifest_sha256") == source["sha256"]
            and source_input.get("source_registry_sha256") == registry_binding["sha256"]
        )
    if (
        receipt.get("schema") != "gvhmr-formal-postleg-cpu-completion-v1"
        or receipt.get("state") != "postleg_native_qa_complete_bilateral_browser_archive_pending"
        or Path(receipt.get("root", "")).resolve() != repair.parent
        or receipt.get("source_input") != source
        or source_input.get("schema") != "badminton-new-source-repair-input-v1"
        or receipt.get("calibration_digest") != plan["calibration_digest"]
        or receipt.get("calibration_digest") != source_input.get("calibration_digest")
        or receipt.get("calibration_digest") != publication["calibration_digest"]
        or receipt.get("calibration_digest") != body.get("calibration_digest")
        or type(receipt.get("windows")) is not int
        or receipt["windows"] != len(plan["jobs"])
        or receipt["windows"] != len(source_input["jobs"])
        or type(receipt.get("accepted")) is not int
        or receipt["accepted"] != body.get("accepted")
        or receipt["accepted"] < publication["accepted"]
        or receipt.get("gpu_started") is not False
        or receipt.get("training_published") is not False
        or set(receipt.get("workers", {})) != {"native", "dataset"}
        or set(body.get("exports", {})) != set(publication["exports"])
        or not body_registry_matches
        or set(registry["sources"]) != set(publication["exports"])
        or {job["source_id"] for job in source_input["jobs"]} - set(registry["sources"])
        or publication.get("source_dataset_manifest") != str(body_path)
        or publication.get("source_dataset_manifest_sha256") != body_binding["sha256"]
    ):
        raise ValueError("Formal postleg CPU receipt differs from the actual source/calibration/window/body scope")
    bound(receipt.get("baseline"), repair.parent / "gpu_guarded_leg/baseline/baseline.json")
    execution = bound(receipt.get("leg_execution"))
    execution_path = Path(execution["path"]).resolve()
    if (
        execution_path.name != "completion.json"
        or execution_path.parent.parent != repair.parent / "gpu_guarded_leg/attempts"
    ):
        raise ValueError("Formal CPU receipt must bind its actual guarded execution completion")
    expected_evidence = dict(
        before=repair / "evaluation/v2_before_leg.json",
        native=labels / "native_verification.json",
        dataset=labels / "datasets/verification.json",
        post_refinement=labels / "post_refinement.json",
        leg_receipt=repair / "leg_refinement_receipt.json",
        after=repair / "evaluation/v2_after_leg.json",
        comparison=repair / "evaluation/v2_leg_comparison.json",
    )
    compatibility_role = "policy_fingerprint_compatibility"
    has_compatibility = compatibility_role in receipt.get("evidence", {})
    if has_compatibility:
        expected_evidence[compatibility_role] = Path(bound(receipt["evidence"][compatibility_role])["path"])
    if set(receipt.get("evidence", {})) != set(expected_evidence):
        raise ValueError("Formal postleg CPU receipt is missing its actual verification evidence")
    if has_compatibility:
        from .formal_completion_policy_compat import validate_compatibility_receipt

        proof = validate_compatibility_receipt(repair.parent, receipt)

        def proof_bindings(value):
            if isinstance(value, dict):
                if set(value) == {"path", "sha256"}:
                    bound(value)
                else:
                    for item in value.values():
                        proof_bindings(item)
            elif isinstance(value, list):
                for item in value:
                    proof_bindings(item)

        proof_bindings(proof)

        def implementation_bindings(implementation_binding):
            payload = Path(implementation_binding["path"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != implementation_binding["sha256"]:
                raise ValueError("The verified compatibility implementation changed while reading")
            implementation = json.loads(payload)
            for item in (
                *implementation["code_archive"].values(),
                implementation["test_execution"],
                *implementation["test_outputs"].values(),
                *(entry["archived"] for entry in implementation["test_sources"]),
                implementation["original_checker"]["archived"],
            ):
                bound(item)

        implementation_bindings(proof["compatibility_implementation"])
    for name, path in expected_evidence.items():
        item = bound(receipt["evidence"][name], path)
        if name in ("before", "native", "leg_receipt", "after", "comparison") and completed(path) != item:
            raise ValueError("Formal completion and postleg CPU evidence differ")
    allocation = dict(
        schema="gvhmr-formal-cpu-worker-v1", affinity=[26, 27], priority="BelowNormal", cuda_visible_devices="-1"
    )
    for role, worker in receipt["workers"].items():
        if role == "native" and worker.get("state") == "reused_completed_native_witness":
            if not has_compatibility:
                raise ValueError("Native witness reuse requires the verified compatibility proof")
            from .formal_completion_policy_compat import _resume_native

            observed = {}
            checked = dict(root=repair.parent, repair=repair, baseline_binding=receipt["baseline"], observed=observed)
            if _resume_native(checked, execution, worker.get("resume"), proof) != worker:
                raise ValueError("Native witness reuse differs from the original exact metadata reconstruction")
            bound(worker.get("entry"), PROJ_ROOT / "tools/verify_badminton_native_bound.py")
            bound(worker.get("log"))
            for path, sha in observed.items():
                bound(dict(path=path, sha256=sha))
            resume = read_json(worker["resume"]["path"])
            proof_bindings(resume)
            implementation_bindings(resume["previous_implementation"])
            for item in worker["actual_native_witnesses"]:
                bound(item)
            continue
        if (
            type(worker.get("exit_code")) is not int
            or worker["exit_code"] != 0
            or worker.get("allocation") != allocation
            or worker.get("hidden_window") is not True
        ):
            raise ValueError("Formal CPU worker did not complete in its verified CPU allocation")
        entry = "verify_badminton_native_bound.py" if role == "native" else "verify_badminton_dataset.py"
        if role == "dataset" and has_compatibility and "dataset_entry" in proof["compatibility_code"]:
            expected_entry = bound(
                proof["compatibility_code"]["dataset_entry"], PROJ_ROOT / "tools/verify_badminton_formal_dataset.py"
            )
            if worker.get("entry") != expected_entry:
                raise ValueError("Dataset worker differs from its verified source-bound compatibility entry")
            entry = "verify_badminton_formal_dataset.py"
        bound(worker.get("entry"), PROJ_ROOT / "tools" / entry)
        log = bound(worker.get("log"))
        log_path = Path(log["path"]).resolve()
        if (
            log_path.name != role + ".log"
            or log_path.parent.parent != repair / "evaluation/formal_postleg_cpu/attempts"
        ):
            raise ValueError("Formal CPU worker log leaves its actual completion attempt")
    observations_path = paired / "observations/manifest.json"
    observations_binding = completed(observations_path)
    observations = read_json(observations_path)
    if (
        observations.get("completed") is not True
        or observations.get("source_accepted") != receipt["accepted"]
        or observations.get("identity", {}).get("calibration_digest") != receipt["calibration_digest"]
        or observations.get("identity", {}).get("source", {}).get("dataset_manifest") != body_binding
        or publication.get("full_observation_manifest") != str(observations_path)
        or publication.get("full_observation_manifest_sha256") != observations_binding["sha256"]
    ):
        raise ValueError("Formal bilateral observations do not bind the completed CPU body publication")
    verification = read_json(completed(paired / "verification.json")["path"])
    if (
        verification.get("accepted") != publication["accepted"]
        or set(verification.get("sessions", {})) != set(publication["exports"])
        or any(
            verification["sessions"][sid].get("accepted") != item["accepted"]
            for sid, item in publication["exports"].items()
        )
    ):
        raise ValueError("Formal bilateral verification differs from its actual publication")
    completed(labels / "viewer/manifest.json")
    return dependencies


def assemble(old_path, new_path, contract_path, output_root, *, release_policy=None, release_policy_sha256=None):
    old_path, new_path, contract_path, output = map(Path, (old_path, new_path, contract_path, output_root))
    dependencies = [binding(path) for path in (old_path, new_path, contract_path)]
    old, new, contract = map(read_json, (old_path, new_path, contract_path))
    incremental = None
    if contract.get("schema") == "expanded-source-increment-assembly-input/v2":
        if release_policy is None or release_policy_sha256 is None:
            raise ValueError("V2 assembly requires both explicit release policy path and SHA256")
        from .incremental_release import load_release_policy

        incremental = load_release_policy(release_policy, release_policy_sha256)
        dependencies.extend(incremental.dependencies)
    elif release_policy is not None or release_policy_sha256 is not None:
        raise ValueError("Legacy assembly cannot override its policy")
    old_binding = contract["bindings"]["old_dataset_manifest"]
    check_binding(old_binding)
    if old_path.resolve() != Path(old_binding["path"]).resolve():
        raise ValueError("Consumer contract names a different original dataset")
    check_binding(contract["bindings"]["new_source_input"])
    check_binding(contract["bindings"]["new_source_splits"])
    dependencies.extend(contract["bindings"][name] for name in ("new_source_input", "new_source_splits"))
    roles = validate_union_contract(old, new, contract, incremental=incremental)
    if incremental is not None:
        repair = Path(contract["new_repair_root"]).resolve()
        if new.get("release_policy") != incremental.binding:
            raise ValueError("V2 publication differs from the explicitly selected release policy")
        if contract["canonical_mode"] == "reviewed_cuts":
            if new.get("schema") != "badminton-canonical-bilateral-publication-v1" or not new.get(
                "canonical_identity_revision"
            ):
                raise ValueError("V2 cut mode requires the actual verified canonical bilateral publication")
        source_binding = contract["bindings"]["new_source_input"]
        registry_binding = contract["bindings"]["new_source_registry"]
        source_input, current_registry = map(read_json, (source_binding["path"], registry_binding["path"]))
        if (
            Path(source_binding["path"]).resolve() != repair / "new_source_input.json"
            or Path(registry_binding["path"]).resolve() != repair.parent / "source/source_registry.json"
            or Path(source_input.get("source_root", "")).resolve() != repair.parent / "source"
            or source_input.get("source_registry_sha256") != registry_binding["sha256"]
            or source_input.get("release_policy") != incremental.binding
            or current_registry.get("release_policy") != incremental.binding
            or source_input.get("session_splits") != incremental.roles
            or source_input.get("required_nonempty_splits") != incremental.required
            or set(current_registry.get("sources", {})) != set(incremental.scope["review_sources"])
        ):
            raise ValueError("V2 assembly source input/registry differs from its exact current policy")
        dependencies.extend(contract["bindings"].values())
    if contract.get("schema") in JOINT_CONTRACTS:
        for name in ("producer_joint_manifest", "producer_confirmation"):
            item = contract["bindings"].get(name)
            if not item:
                raise ValueError("Joint input lacks its actual " + name + " binding")
            check_binding(item)
            dependencies.append(item)
            if name == "producer_joint_manifest" and (
                Path(item["path"]).resolve() != new_path.resolve() or item["sha256"] != dependencies[1]["sha256"]
            ):
                raise ValueError("Joint input names a different new publication")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Assembly requires an empty destination; completed and provisional runs stay immutable")
    source_registry, verification = {}, {}
    for tag, manifest, path in (("old", old, old_path), ("new", new, new_path)):
        if tag == "old" and manifest.get("schema") in (
            "badminton-expanded-dataset-v1",
            "badminton-expanded-training-v1",
        ):
            registry, verification[tag], base_dependencies = verify_expanded_base(manifest, contract)
            base_run = path.resolve().parent
            if base_run.name == "datasets":
                base_run = base_run.parent
            if output.resolve().is_relative_to(base_run) or base_run.is_relative_to(output.resolve()):
                raise ValueError("Expansion output must be separate from the complete canonical base")
            source_registry.update({sid: dict(entry, generation="old") for sid, entry in registry.items()})
            dependencies.extend(base_dependencies)
            continue
        source_path = Path(manifest["source_dataset_manifest"])
        if file_sha(source_path) != manifest["source_dataset_manifest_sha256"]:
            raise ValueError("Body source manifest changed")
        source_run = source_path.resolve().parent.parent
        if output.resolve().is_relative_to(source_run) or source_run.is_relative_to(output.resolve()):
            raise ValueError("Expansion output must be separate from source labels")
        calibration_path = source_run / "calibration/calibration_snapshot.json"
        dependencies.extend(binding(p) for p in (source_path, calibration_path))
        calibration = CalibrationSnapshot.load(calibration_path)
        if calibration.data["snapshot_digest"] != manifest["calibration_digest"]:
            raise ValueError("Publication and source calibration disagree")
        # This verifier reads source publications without rewriting verification.json.
        revision = manifest.get("canonical_identity_revision")
        if revision:
            if tag != "new":
                raise ValueError("The complete old publication remains immutable")
            if incremental is not None and contract["canonical_mode"] == "reviewed_no_changes":
                from .canonical_no_change import verify_no_change_publication

                for name in ("source_publication", "reviewed_continuity", "verification"):
                    check_binding(revision[name])
                    dependencies.append(revision[name])
                proof = verify_no_change_publication(
                    manifest,
                    release_policy=incremental.binding["path"],
                    release_policy_sha256=incremental.binding["sha256"],
                )
                if (
                    proof["release_policy"] != incremental.binding
                    or proof["source_completion"] != contract["bindings"]["producer_confirmation"]
                    or proof["source_publication"] != revision["source_publication"]
                    or proof["accepted"] != manifest["accepted"]
                ):
                    raise ValueError("No-change proof differs from this exact assembly source/completion/policy")
                for item in (
                    proof["producer"],
                    *proof["verifiers"].values(),
                    *(
                        proof[name]
                        for name in (
                            "source_completion",
                            "continuity_audit",
                            "native_verification",
                            "aggregate_native_contract",
                        )
                    ),
                    *(
                        session[name]
                        for session in proof["sessions"].values()
                        for name in ("source_tensor", "source_metadata", "body_tensor", "body_metadata")
                    ),
                ):
                    check_binding(item)
                    if item not in dependencies:
                        dependencies.append(item)
                verification[tag] = proof
            else:
                from .canonical_tracks import verify_canonical_publication

                for name in ("source_publication", "reviewed_boundaries", "verification"):
                    check_binding(revision[name])
                    dependencies.append(revision[name])
                verification[tag] = verify_canonical_publication(manifest)
        else:
            verification[tag] = _verify_export_data(manifest)
        for sid, export in manifest["exports"].items():
            if tag == "new" and incremental is not None:
                declared = contract["new_sources"][sid]
                reviewed = incremental.scope["review_sources"][sid]
                if (
                    declared.get("formal_body_input") != contract["bindings"]["new_source_input"]
                    or Path(declared.get("repair_root", "")).resolve() != Path(contract["new_repair_root"]).resolve()
                    or declared.get("admitted_source_index") != reviewed["admitted_source_index"]
                    or declared.get("camera_applicability") != reviewed["camera_applicability"]
                    or declared.get("calibration_snapshot") != binding(calibration_path)
                    or declared.get("calibration_digest") != calibration.data["snapshot_digest"]
                    or current_registry["sources"][sid].get("source_index") != reviewed["admitted_source_index"]
                ):
                    raise ValueError("V2 actual source/calibration/admission lineage differs from its reviewed scope")
            source_registry[sid] = dict(
                role=roles[sid],
                accepted=export["accepted"],
                generation=tag,
                source_run=str(source_run),
                calibration_snapshot=str(calibration_path),
                calibration_snapshot_sha256=file_sha(calibration_path),
                calibration_digest=manifest["calibration_digest"],
                paired_manifest=binding(path),
                body_dataset_manifest=binding(source_path),
            )
    repair = Path(contract["new_repair_root"]).resolve()
    expected_new = repair.with_name(repair.name + "_paired") / "datasets/dataset_manifest.json"
    revision = new.get("canonical_identity_revision")
    if revision:
        parent_publication = revision["source_publication"]
        if Path(parent_publication["path"]).resolve() != expected_new or not new_path.resolve().is_relative_to(repair):
            raise ValueError("Canonical identity revision must derive from this fresh repair publication")
    else:
        if new_path.resolve() != expected_new:
            raise ValueError("New publication does not belong to the agreed new repair run")
        parent_publication = binding(new_path)
    completion_path = repair / "new_source_completion_receipt.json"
    completion_binding = binding(completion_path)
    dependencies.append(completion_binding)
    completion = read_json(completion_path)
    publication = read_json(parent_publication["path"])
    completion_evidence = _validate_new_completion(repair, completion, parent_publication, publication)
    if completion_evidence:
        source = next(item for item in completion_evidence if Path(item["path"]) == repair / "new_source_input.json")
        if (
            contract.get("schema") not in JOINT_CONTRACTS
            or contract["bindings"]["new_source_input"] != source
            or contract["bindings"]["producer_confirmation"] != completion_binding
        ):
            raise ValueError("Formal assembly requires the sealed joint contract's exact current source and completion")
    dependencies.extend(completion_evidence)
    digests = {sid: e["calibration_digest"] for sid, e in source_registry.items()}
    splits, sequences = {}, {}
    for role in ROLES:
        entries = [manifest["suggested_session_splits"][role] for manifest in (old, new)]
        destination = output / "datasets" / (role + ".pth")
        if incremental is not None and contract["new_final_admitted_counts"][role] == 0:
            entry, rows = merge_role(entries, destination, digests, preserve_old_bytes=True)
        else:
            entry, rows = merge_role(entries, destination, digests)
        counts = Counter(row["session_id"] for row in rows)
        if any(counts[sid] != source_registry[sid]["accepted"] for sid in entry["sessions"]):
            raise ValueError("Merged role does not retain every qualified source sample")
        splits[role] = entry
        chunks = contiguous_sequences(rows)
        path = output / (role + "_sequences.jsonl")
        path.write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
        if sorted(i for c in chunks for i in c["indices"]) != list(range(len(rows))):
            raise ValueError("Sequence index does not cover real rows exactly once")
        sequences[role] = dict(**binding(path), sequences=len(chunks), samples=len(rows))
    result = dict(
        schema="badminton-expanded-assembly-v1",
        status="awaiting_native_pixel_and_browser_acceptance",
        training_published=False,
        **CONTRACT,
        accepted=sum(e["accepted"] for e in splits.values()),
        exports=dict(old["exports"], **new["exports"]),
        suggested_session_splits=splits,
        source_registry=source_registry,
        source_publications={"old": dependencies[0], "new": dependencies[1]},
        consumer_contract=dependencies[2],
        new_completion=completion_binding,
        source_verification=verification,
        sequences=sequences,
        old_accepted=old["accepted"],
        new_accepted=new["accepted"],
        all_old_rows_retained=True,
        all_tensors_equal_ordered_source_concatenation=True,
        all_metadata_bytes_equal_ordered_source_concatenation=True,
        split_limit="Whole sources disjoint; cross-source physical identities are unverified",
    )
    if incremental is not None:
        result["release_policy"] = incremental.binding
    for entry in dependencies:
        check_binding(entry)
    write_json(output / "assembly_manifest.json", result)
    return result
