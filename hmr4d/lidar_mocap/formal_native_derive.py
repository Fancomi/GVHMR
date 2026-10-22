"""Strictly derive unchanged native checks from genuinely witnessed parents."""

from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import runpy

from hmr4d import PROJ_ROOT
from .fit_reuse import rebind_fit_row, same_teacher_values
from .formal_baseline import _artifacts, _binding, _bound, _encode, _publish, _scope, _sha
from .formal_import_audit import audit_formal_numeric_import
from .formal_numeric_import import _load_audit, _provenance_hash
from .full_native import rebind_native_pose_record
from .identity_repair import decision_digest
from .native_witness import (
    capture_runtime,
    teacher_fingerprints,
    validate_native_witness,
    run_bound_native_verification,
)
from .repair_execution import read_live_json, write_progress_json
from .rally_pipeline import read_rows

SCHEMA = "badminton-formal-derived-native-equivalence-v1"


def has_derived_native(value):
    return bool(
        value.get("derived_native_proof")
        or any(
            entry.get("verification_origin", {}).get("mode") == "derived" for entry in value.get("windows", {}).values()
        )
    )


def _actual_witnesses(repair, native, runtime):
    entries = native.get("windows", {})
    groups = {}
    for key, entry in entries.items():
        origin = entry.get("verification_origin", {})
        if origin.get("mode") == "actual":
            item = origin["witness"]
            token = (item["path"], item["sha256"])
            groups.setdefault(token, dict(item=item, keys=set()))["keys"].add(key)
        elif origin.get("mode") != "derived":
            raise ValueError("Every native window needs its explicit actual/derived source")
    declared = native.get("actual_native_witnesses", [])
    if {(item["path"], item["sha256"]) for item in declared} != set(groups) or len(declared) != len(groups):
        raise ValueError("Actual native coverage lacks its exact execution witness inventory")
    for group in groups.values():
        validate_native_witness(
            group["item"], repair, current_runtime=runtime, require_full=False, window_keys=group["keys"]
        )
    return [group["item"] for group in groups.values()]


def _formal_root(root):
    root = Path(root).resolve()
    outputs = (PROJ_ROOT / "outputs").resolve()
    if root == outputs or not root.is_relative_to(outputs):
        raise ValueError("Select an explicit separate formal output run")
    return root


def _read(item):
    return _bound(item, {})[0]


def _initial_teacher(source, target):
    before = read_live_json(source / "manifest.json")
    files = ["manifest.json", "detections.pt", *(entry["path"] for entry in before["tracks"])]
    for name in files:
        a, b = (source / name).resolve(), (target / name).resolve()
        if not a.is_relative_to(source.resolve()) or not b.is_relative_to(target.resolve()):
            raise ValueError("Initial native teacher file leaves its exact scope")
        if _sha(a) != _sha(b) or os.path.samefile(a, b):
            raise ValueError("Formal initial teacher differs from its independent original copy")


def _body_teacher(source, target, old_decision, new_decision):
    import torch

    before = read_live_json(source / "manifest.json")
    after = read_live_json(target / "manifest.json")
    expected = deepcopy(before)
    if before["provenance_hash"] != _provenance_hash(before["provenance"]) or before["provenance"]["options"].get(
        "identity_decisions_sha256"
    ) != decision_digest(old_decision):
        raise ValueError("Original witnessed teacher identity/provenance changed")
    expected["provenance"]["options"]["identity_decisions_sha256"] = decision_digest(new_decision)
    expected["provenance_hash"] = _provenance_hash(expected["provenance"])
    if "identity_repair" in expected:
        if expected["identity_repair"].get("decision_sha256") != decision_digest(old_decision):
            raise ValueError("Original teacher repair operations changed")
        expected["identity_repair"]["decision_sha256"] = decision_digest(new_decision)
    if after != expected:
        raise ValueError("Formal teacher changed beyond explicit identity provenance rebinding")
    for entry in before["tracks"]:
        a, b = (source / entry["path"]).resolve(), (target / entry["path"]).resolve()
        if not a.is_relative_to(source.resolve()) or not b.is_relative_to(target.resolve()):
            raise ValueError("Repaired teacher track leaves its native window")
        old_sha, new_sha = _sha(a), _sha(b)
        left, right = torch.load(a, map_location="cpu", weights_only=False), torch.load(
            b, map_location="cpu", weights_only=False
        )
        if (
            not same_teacher_values(left, right)
            or set(left) != set(right)
            or left.get("temporal_reuse_from") != right.get("temporal_reuse_from")
            or left.get("provenance_hash") != before["provenance_hash"]
            or right.get("provenance_hash") != after["provenance_hash"]
            or ("provenance" in right and right["provenance"] != after["provenance"])
            or ("provenance" in left and left["provenance"] != before["provenance"])
            or _sha(a) != old_sha
            or _sha(b) != new_sha
        ):
            raise ValueError("Actual teacher tensors, temporal context, dtype, shape or provenance changed")
    return after["provenance_hash"]


def _context(root, import_binding):
    imported = _read(import_binding)
    if (
        imported.get("schema") != "badminton-formal-numeric-import-v1"
        or imported.get("state") != "complete"
        or imported.get("numerical_artifacts_byte_identical") is not True
        or imported.get("qa_values_unchanged") is not True
        or imported.get("numeric_fit_recomputed") is not False
    ):
        raise ValueError("Complete exact formal numerical import is required")
    audit = _load_audit(root, imported["audit"])
    commits = {
        key.rsplit(":", 1)[0]: value["original_commit"]
        for key, value in audit["algorithms"].items()
        if value.get("original_commit")
    }
    fresh = audit_formal_numeric_import(
        root, audit["input_equivalence"], list(audit["parent_evidence"]), original_commits=commits
    )
    if fresh != audit or set(imported["windows"]) != set(audit["windows"]):
        raise ValueError("Current full geometry/source/role/algorithm/point equivalence no longer matches its audit")
    pose = _read(imported["completed_native_pose_receipt"])
    identity = _read(imported["reviewed_formal_identity"])
    if (
        _binding(root / "repair/pose_receipt.json") != imported["completed_native_pose_receipt"]
        or _binding(root / "repair/identity_evidence/identity_decisions.json") != imported["reviewed_formal_identity"]
    ):
        raise ValueError("Actual formal pose/identity differs from the numerical import")
    inventory = _read(pose["formal_import_inventory"])
    parents = {parent: _read(item["pose"]) for parent, item in audit["parent_evidence"].items()}
    if inventory.get("config") != pose["config"] or inventory.get("numerical_observations_changed") is not False:
        raise ValueError("Native pose import inventory changed")
    pose_records = {}
    for item in inventory["records"]:
        source = item["source"]["path"]
        if source in pose_records:
            raise ValueError("Native pose source identity is duplicated")
        pose_records[source] = item
        original, current = _read(item["source"]), _read(item["target"])
        parent = [name for name in parents if Path(source).resolve().is_relative_to(Path(name) / "repair/native_poses")]
        if len(parent) != 1:
            raise ValueError("Native observation lacks one exact audited parent")
        expected = rebind_native_pose_record(
            original, parents[parent[0]]["config"], pose["config"], original["native"], item["source"]
        )
        if current != expected:
            raise ValueError("Actual formal native observations differ numerically from their parent")
    return dict(
        root=root,
        imported=imported,
        import_binding=import_binding,
        audit=audit,
        pose=pose,
        identity=identity,
        inventory=inventory,
        pose_records=pose_records,
    )


def _equivalent_window(context, key):
    root, imported, audit = context["root"], context["imported"], context["audit"]
    entry = audit["windows"][key]
    record = imported["windows"][key]
    original, target = Path(entry["original_window"]), Path(entry["target_window"])
    before, current = _read(record["original_report"]), _read(record["current_report"])
    if (
        record["original_report"] != entry["original_report"]
        or _sha(target / "fit_report.json") != record["current_report"]["sha256"]
        or before["accepted"] != current["accepted"]
        or before["fitted"] != current["fitted"]
        or _artifacts(before, original, {}) != _artifacts(current, target, {})
    ):
        raise ValueError("Actual formal numerical body artifacts or fit-report membership changed")
    parent_identity = _read(audit["parent_evidence"][entry["parent_root"]]["identity"])
    old_decision = parent_identity["windows"][key]
    decision = context["identity"]["windows"][key]
    if decision != dict(old_decision, source_window=entry["target_initial_window"]):
        raise ValueError("Formal reviewed identity operations differ from the witnessed parent")
    _initial_teacher(
        Path(entry["original_initial_window"]) / "teacher", Path(entry["target_initial_window"]) / "teacher"
    )
    teacher_hash = _body_teacher(original / "teacher", target / "teacher", old_decision, decision)
    for filename in ("quality.jsonl", "ptv3.metadata.jsonl"):
        left, right = read_rows(original / filename), read_rows(target / filename)
        expected = [
            rebind_fit_row(row, imported["calibration_digest"], target, context["pose_records"], teacher_hash)
            for row in left
        ]
        if right != expected:
            raise ValueError("Formal person/scan/point-index/QA/pose rows changed beyond explicit lineage rebinding")
    return record["current_report"]


def _expected(context, key, witness, parent_native):
    report = _equivalent_window(context, key)
    model = witness["runtime_before"]["models"]["neutral_smpl"]
    if context["pose"]["config"]["models"].get("smpl_triangle_topology") != model:
        raise ValueError("Actual reconstruction model differs from the formal model contract")
    parent = context["audit"]["windows"][key]["parent_root"]
    if (
        _binding(Path(parent) / "repair/final_labels/native_verification.json")["sha256"]
        != witness["native_result_snapshot"]["sha256"]
    ):
        raise ValueError("Witness belongs to another parent native validation")
    result = deepcopy(parent_native["windows"][key])
    result["body"]["report_sha256"] = report["sha256"]
    result["teacher_inputs_sha256"] = teacher_fingerprints(
        context["root"] / "repair", _scope(_read(context["audit"]["body_plan"])["jobs"])[key]
    )
    return result


def prepare_derived_native(root, import_binding, parent_witnesses):
    """Seed only proven windows; missing witnesses remain actual-check work."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Derived native preparation is explicitly CPU-only")
    root = _formal_root(root)
    context = _context(root, import_binding)
    audit = context["audit"]
    if set(parent_witnesses) - set(audit["parent_evidence"]):
        raise ValueError("An explicitly supplied native witness is outside the admitted parents")
    runtime = capture_runtime() if parent_witnesses else None
    valid, fallback, derived = {}, {}, {}
    for parent in audit["parent_evidence"]:
        keys = {key for key, value in audit["windows"].items() if value["parent_root"] == parent}
        item = parent_witnesses.get(parent)
        if item is None:
            fallback.update(
                {
                    key: "Original native model/runtime execution witness absent; actual full check required"
                    for key in keys
                }
            )
            continue
        try:
            witness, native = validate_native_witness(item, Path(parent) / "repair", current_runtime=runtime)
        except (ValueError, FileNotFoundError, KeyError) as error:
            fallback.update({key: str(error) for key in keys})
            continue
        valid[parent] = item
        for key in sorted(keys):
            try:
                derived[key] = _expected(context, key, witness, native)
            except (ValueError, FileNotFoundError, KeyError) as error:
                fallback[key] = str(error)
    if not derived:
        return dict(
            state="actual_native_check_required",
            derived_windows=0,
            actual_required=fallback,
            original_native_check_required=True,
            gpu_started=False,
        )
    proof = dict(
        schema=SCHEMA,
        state="complete_equivalence_checked",
        formal_root=str(root),
        numeric_import=import_binding,
        audit=context["imported"]["audit"],
        parent_witnesses=valid,
        windows=derived,
        actual_required=fallback,
        runtime=runtime,
        derivation="Reuse actual parent numerical maxima only after full input equivalence; no reconstructed SMPL or points were recomputed here",
        numeric_reconstruction_recomputed=False,
        gpu_started=False,
        training_published=False,
    )
    proof_path = root / "import_evidence/derived_native_equivalence.json"
    _publish(proof_path, _encode(proof))
    proof_binding = _binding(proof_path)
    native_path = root / "repair/final_labels/native_verification.json"
    windows = {
        key: dict(
            value,
            verification_origin=dict(
                mode="derived",
                proof=proof_binding,
                parent_witness=valid[audit["windows"][key]["parent_root"]],
                maximums_source="actual_parent_reconstruction",
            ),
        )
        for key, value in derived.items()
    }
    result = dict(
        windows=windows,
        passed=len(windows) == len(audit["windows"]),
        state="complete" if len(windows) == len(audit["windows"]) else "partial",
        verified_windows=len(windows),
        total_windows=len(audit["windows"]),
        accepted_checked=sum(item["body"]["accepted_checked"] for item in windows.values()),
        source_union_sha256=context["imported"]["source_union_sha256"],
        calibration_digest=context["imported"]["calibration_digest"],
        derived_native_proof=proof_binding,
        verification_coverage=dict(
            actual_window_keys=[],
            derived_window_keys=sorted(windows),
            maximums_for_derived_windows_are_parent_measurements=True,
        ),
        actual_native_witnesses=[],
        limits=[
            "Derived windows retain actual parent maxima and full equivalence provenance; uncovered windows require actual native reconstruction"
        ],
    )
    validate_native_derivations(root / "repair", result, allow_partial=True)
    _publish(native_path, _encode(result))
    return dict(
        state="proven_native_windows_seeded_actual_remainder_required",
        proof=proof_binding,
        derived_windows=len(derived),
        actual_required=fallback,
        gpu_started=False,
    )


def validate_native_derivations(repair, native, *, allow_partial=False, postleg=False):
    """Read-only guard check of every currently derived window and full chain."""
    repair = Path(repair).resolve()
    canonical_proof = repair.parent / "import_evidence/derived_native_equivalence.json"
    has_actual = bool(native.get("actual_native_witnesses"))
    if canonical_proof.exists() and not native.get("derived_native_proof") and not has_actual:
        raise ValueError("A derived native provenance envelope was removed")
    if not has_derived_native(native) and not has_actual:
        return []
    root = _formal_root(repair.parent)
    runtime = capture_runtime()
    _actual_witnesses(repair, native, runtime)
    derived_keys = {
        key for key, entry in native["windows"].items() if entry.get("verification_origin", {}).get("mode") == "derived"
    }
    actual_keys = set(native["windows"]) - derived_keys
    coverage = dict(
        actual_window_keys=sorted(actual_keys),
        derived_window_keys=sorted(derived_keys),
        maximums_for_derived_windows_are_parent_measurements=True,
    )
    if not allow_partial and native.get("verification_coverage") != coverage:
        raise ValueError("Actual-versus-derived native window coverage is incomplete")
    if not has_derived_native(native):
        imported_path = root / "formal_numeric_import.json"
        if imported_path.exists():
            _context(root, _binding(imported_path))
        if derived_keys or (
            not allow_partial and (native.get("passed") is not True or native.get("state") != "complete")
        ):
            raise ValueError("Actual full reconstruction did not supersede missing derived evidence")
        return native_derivation_anchors(native)
    proof_binding = native.get("derived_native_proof")
    proof = _read(proof_binding)
    if (
        proof.get("schema") != SCHEMA
        or proof.get("state") != "complete_equivalence_checked"
        or proof.get("formal_root") != str(root)
        or proof.get("numeric_reconstruction_recomputed") is not False
    ):
        raise ValueError("A complete explicit derived-native equivalence proof is required")
    context = _context(root, proof["numeric_import"])
    if proof.get("audit") != context["imported"]["audit"]:
        raise ValueError("Derived native proof changed the full original input equivalence")
    keys = set(context["audit"]["windows"])
    if not allow_partial and (
        set(native["windows"]) != keys or native.get("passed") is not True or native.get("state") != "complete"
    ):
        raise ValueError("Derived native evidence does not complete the actual full window scope")
    if (
        native.get("source_union_sha256") != context["imported"]["source_union_sha256"]
        or native.get("calibration_digest") != context["imported"]["calibration_digest"]
    ):
        raise ValueError("Derived native evidence names another source or geometry")
    if runtime != proof["runtime"]:
        raise ValueError("Derived native model/algorithm/runtime changed")
    witnesses = {
        parent: validate_native_witness(item, Path(parent) / "repair", current_runtime=runtime)
        for parent, item in proof["parent_witnesses"].items()
    }
    missing = set(proof["windows"]) - derived_keys
    if derived_keys - set(proof["windows"]) or (missing and not postleg):
        raise ValueError("Derived per-window coverage was removed or substituted")
    for key in missing:
        sid, name = key.split(":", 1)
        report = read_live_json(repair / "final_labels/sessions" / sid / "windows" / name / "fit_report.json")
        if (
            (
                _sha(repair / "final_labels/sessions" / sid / "windows" / name / "fit_report.json")
                == proof["windows"][key]["body"]["report_sha256"]
            )
            or not report.get("post_leg_refinement")
            or (key not in native["windows"] and not allow_partial)
        ):
            raise ValueError("Only actually reverified post-leg changed windows can replace derived checks")
    used = set()
    for key, entry in native["windows"].items():
        origin = entry.get("verification_origin", {})
        if origin.get("mode") != "derived":
            continue
        if key not in proof["windows"] or origin.get("proof") != proof_binding:
            raise ValueError("A derived native window lacks its exact equivalence proof")
        parent = context["audit"]["windows"][key]["parent_root"]
        if origin.get("parent_witness") != proof["parent_witnesses"].get(parent) or parent not in witnesses:
            raise ValueError("A derived window lacks its actual original reconstruction witness")
        expected = _expected(context, key, *witnesses[parent])
        if (
            proof["windows"][key] != expected
            or {k: v for k, v in entry.items() if k != "verification_origin"} != expected
        ):
            raise ValueError("Derived native maxima/counts/report/teacher/numerical inputs changed")
        used.add(parent)
    # These immutable envelopes bind the model/point/teacher checks. Large
    # mutable current fit artifacts are revalidated above, not watched after apply.
    assets = native_derivation_anchors(native)
    return assets


def native_derivation_anchors(native):
    """Bound immutable provenance envelopes; current mutable fits stay separate."""
    if not has_derived_native(native) and not native.get("actual_native_witnesses"):
        return []
    assets, items = [], list(native.get("actual_native_witnesses", []))
    if native.get("derived_native_proof"):
        proof_binding = native["derived_native_proof"]
        proof = _read(proof_binding)
        assets += [proof_binding, proof["numeric_import"], proof["audit"]]
        items += list(proof["parent_witnesses"].values())
    for item in items:
        witness = _read(item)
        assets.extend([item, witness["native_result_snapshot"], witness["runtime_before"]["models"]["neutral_smpl"]])
        for name in ("tools/verify_badminton_native_repair.py", "hmr4d/utils/body_model/contracts.py"):
            path = str((PROJ_ROOT / name).resolve())
            assets.append(dict(path=path, sha256=witness["runtime_before"]["files"][path]))
    return assets


def refresh_native_coverage(repair, *, postleg=False):
    path = Path(repair) / "final_labels/native_verification.json"
    native = read_live_json(path)
    if not has_derived_native(native) and not native.get("actual_native_witnesses"):
        return native
    derived = sorted(
        key for key, item in native["windows"].items() if item.get("verification_origin", {}).get("mode") == "derived"
    )
    actual = sorted(set(native["windows"]) - set(derived))
    coverage = dict(
        actual_window_keys=actual,
        derived_window_keys=derived,
        maximums_for_derived_windows_are_parent_measurements=True,
    )
    candidate = dict(native, verification_coverage=coverage)
    validate_native_derivations(repair, candidate, postleg=postleg)
    if candidate != native:
        write_progress_json(path, candidate)
    native = candidate
    return native


def run_required_actual_native(repair, threads=1, *, postleg=False):
    """Let the original verifier really check every unproven/changed window."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or threads not in (1, 2):
        raise ValueError("Actual fallback native verification requires CPU-only execution")
    repair = Path(repair).resolve()
    path = repair / "final_labels/native_verification.json"
    native = read_live_json(path) if path.exists() else dict(windows={})
    runtime = capture_runtime()
    plan = read_live_json(repair / "final_labels/repair_plan.json")
    jobs = _scope(plan["jobs"])
    valid = {}
    for key, entry in native.get("windows", {}).items():
        if key not in jobs:
            continue
        sid, name = key.split(":", 1)
        report = repair / "final_labels/sessions" / sid / "windows" / name / "fit_report.json"
        if _sha(report) == entry["body"]["report_sha256"] and entry.get(
            "teacher_inputs_sha256"
        ) == teacher_fingerprints(repair, jobs[key]):
            valid[key] = entry
    candidate = dict(native, windows=valid, state="partial", passed=False)
    # A whole actual witness may predate explicit per-window origin tags.
    canonical = repair / "native_verification_witness.json"
    if canonical.exists():
        try:
            value, _ = validate_native_witness(
                _binding(canonical), repair, current_runtime=runtime, require_full=False, window_keys=set(valid)
            )
            for key in set(value["actual_window_keys"]) & set(valid):
                candidate["windows"][key] = dict(
                    valid[key], verification_origin=dict(mode="actual", witness=_binding(canonical))
                )
            candidate["actual_native_witnesses"] = [_binding(canonical)]
        except (ValueError, FileNotFoundError, KeyError):
            pass
    try:
        validate_native_derivations(repair, candidate, allow_partial=True, postleg=postleg)
        allowed = {
            key
            for key, entry in candidate["windows"].items()
            if entry.get("verification_origin", {}).get("mode") in ("actual", "derived")
        }
    except (ValueError, FileNotFoundError, KeyError):
        allowed = set()
        if candidate.get("derived_native_proof"):
            candidate["superseded_derived_native_proof"] = candidate.pop("derived_native_proof")
        candidate.pop("actual_native_witnesses", None)
    # Preserve the original checkpoint, then admit only cryptographically proven
    # caches. All other windows are truly rebuilt by the unchanged verifier.
    prior = repair / "native_actual_fallback" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json")
    if path.exists():
        _publish(prior, path.read_bytes())
    candidate["windows"] = {key: entry for key, entry in candidate["windows"].items() if key in allowed}
    write_progress_json(path, candidate)
    execution = run_bound_native_verification(repair, threads, verified_cached_keys=allowed)
    native = read_live_json(path)
    if execution.get("witness"):
        value = _read(execution["witness"])
        for key in value["actual_window_keys"]:
            native["windows"][key]["verification_origin"] = dict(mode="actual", witness=execution["witness"])
    actual_items = {
        (item["verification_origin"]["witness"]["path"], item["verification_origin"]["witness"]["sha256"]): item[
            "verification_origin"
        ]["witness"]
        for item in native["windows"].values()
        if item.get("verification_origin", {}).get("mode") == "actual"
    }
    native["actual_native_witnesses"] = list(actual_items.values())
    derived_keys = sorted(
        key for key, entry in native["windows"].items() if entry.get("verification_origin", {}).get("mode") == "derived"
    )
    native["verification_coverage"] = dict(
        actual_window_keys=sorted(set(native["windows"]) - set(derived_keys)),
        derived_window_keys=derived_keys,
        maximums_for_derived_windows_are_parent_measurements=True,
    )
    validate_native_derivations(repair, native, postleg=postleg)
    write_progress_json(path, native)
    return native
