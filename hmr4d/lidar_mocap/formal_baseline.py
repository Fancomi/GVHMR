"""Freeze genuine, completed formal body evidence before the guarded leg stage.

This CPU wrapper consumes an explicit numerical-import receipt. It does not
perform fitting, native verification, GPU authorization or training admission.
"""

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

from hmr4d import PROJ_ROOT


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite JSON value")
    return result


def _json(payload):
    return json.loads(
        payload.decode("utf-8-sig"),
        object_pairs_hook=_pairs,
        parse_float=_float,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON value")),
    )


def _encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")


def _binding(path, payload=None):
    return dict(
        path=str(Path(path).resolve()), sha256=_sha(path) if payload is None else hashlib.sha256(payload).hexdigest()
    )


def _bound(item, observed):
    if (
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not Path(item["path"]).is_absolute()
        or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))
    ):
        raise ValueError("An explicit absolute path and SHA256 binding is required")
    path = Path(item["path"])
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != item["sha256"]:
        raise ValueError("Bound JSON file changed: " + str(path))
    observed[str(path.resolve())] = item["sha256"]
    value = _json(payload)
    if type(value) is not dict:
        raise ValueError("Evidence must be a JSON object")
    return value, payload


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError("Invalid actual count: " + name)
    return value


def _scope(jobs):
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("A nonempty actual formal window plan is required")
    result = {}
    for job in jobs:
        sid, name = job["source_id"], Path(job["source_window"]).name
        if (
            not isinstance(sid, str)
            or not re.fullmatch(r"[0-9]{8}_[0-9]{6}", sid)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", name)
            or name in (".", "..")
        ):
            raise ValueError("Unsafe actual source/window identifier")
        key = sid + ":" + name
        if key in result:
            raise ValueError("Duplicate formal planned window")
        result[key] = job
    return result


def _artifacts(report, window, observed):
    artifacts = report.get("artifacts_sha256")
    if not isinstance(artifacts, dict) or not {"ptv3.pth", "quality.jsonl", "ptv3.metadata.jsonl"} <= set(artifacts):
        raise ValueError("A complete actual fit artifact inventory is required")
    for relative, sha in artifacts.items():
        if (
            not isinstance(relative, str)
            or PurePosixPath(relative).as_posix() != relative
            or PureWindowsPath(relative).drive
            or "\\" in relative
            or PureWindowsPath(relative).root
            or ".." in PurePosixPath(relative).parts
        ):
            raise ValueError("Fit artifact leaves its exact window")
        path = (window / relative).resolve()
        if not path.is_relative_to(window.resolve()) or not re.fullmatch(r"[0-9a-f]{64}", str(sha)):
            raise ValueError("Invalid bounded fit artifact")
        if _sha(path) != sha:
            raise ValueError("Actual fit artifact changed: " + str(path))
        observed[str(path)] = sha
    return {name: sha for name, sha in artifacts.items() if name == "ptv3.pth" or name.startswith("fits/")}


def _row_qa(window):
    result = {}
    for line in (window / "quality.jsonl").read_bytes().splitlines():
        if not line.strip():
            continue
        row = _json(line)
        key = (row["motion_id"], row["lidar_index"])
        if key in result or type(row["accepted"]) is not bool:
            raise ValueError("Duplicate or invalid actual quality row")
        result[key] = {name: row[name] for name in ("accepted", "source_point_count", "qa")}
    return result


def _preflight_target(path, payload, baseline):
    if not path.resolve().is_relative_to(baseline):
        raise ValueError("Baseline snapshot leaves its fixed output directory")
    if path.exists() and (not path.is_file() or path.read_bytes() != payload):
        raise ValueError("An existing immutable baseline target differs: " + str(path))


def _publish(path, payload):
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("An existing immutable baseline target changed")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation never replaces an existing evidence file. A partial
    # failed write cannot become a complete wrapper or successful receipt.
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ValueError("Concurrent immutable baseline target differs")


def freeze_formal_baseline(root, numeric_import, numeric_import_sha256):
    """Validate all formal windows and freeze their pre-leg evidence on CPU.

    ``root`` is the formal run directory containing ``repair``. Numerical import
    must already have copied each current report to the fixed baseline/reports
    path. Original and current reports may differ in provenance; every numerical
    artifact and every original QA row must remain equal.
    """
    root = Path(root).resolve()
    outputs = (PROJ_ROOT / "outputs").resolve()
    if root == outputs or not root.is_relative_to(outputs):
        raise ValueError("Select an explicit formal run inside project outputs")
    repair = root / "repair"
    labels = repair / "final_labels"
    baseline = root / "gpu_guarded_leg/baseline"
    if not baseline.resolve().is_relative_to(root):
        raise ValueError("Baseline destination leaves the exact formal run")
    baseline = baseline.resolve()
    if any(path.exists() for path in (labels / "post_refinement.json", repair / "leg_refinement_receipt.json")):
        raise ValueError("A genuine pre-leg baseline cannot be reconstructed after refinement started")
    observed = {}
    imported_binding = dict(path=str(Path(numeric_import).resolve()), sha256=numeric_import_sha256)
    imported, _ = _bound(imported_binding, observed)
    source_binding = _binding(repair / "new_source_input.json")
    source, _ = _bound(source_binding, observed)
    plan_binding = _binding(labels / "repair_plan.json")
    plan, _ = _bound(plan_binding, observed)
    keys = set(_scope(plan["jobs"]))
    source_sha, calibration = plan.get("source_union_sha256"), plan.get("calibration_digest")
    if (
        plan.get("source_union") != source_binding["path"]
        or source_sha != source_binding["sha256"]
        or not re.fullmatch(r"[0-9a-f]{64}", str(calibration))
        or source.get("schema") != "badminton-new-source-repair-input-v1"
        or source.get("calibration_digest") != calibration
        or _scope(source["jobs"]) != _scope(plan["jobs"])
    ):
        raise ValueError("Formal source, camera and exact body plan differ")
    if (
        imported.get("schema") != "badminton-formal-numeric-import-v1"
        or imported.get("state") != "complete"
        or imported.get("source_union_sha256") != source_sha
        or imported.get("calibration_digest") != calibration
        or imported.get("numerical_artifacts_byte_identical") is not True
        or imported.get("qa_values_unchanged") is not True
        or set(imported.get("windows", {})) != keys
    ):
        raise ValueError("A complete exact formal numerical import is required")

    paths = dict(
        native=labels / "native_verification.json",
        body=labels / "repair_receipt.json",
        pose=repair / "pose_receipt.json",
        identity=repair / "identity_evidence/identity_decisions.json",
    )
    inputs, values, payloads = {}, {}, {}
    for role, path in paths.items():
        inputs[role] = _binding(path)
        values[role], payloads[role] = _bound(inputs[role], observed)
    native, body, poses, identity = [values[role] for role in ("native", "body", "pose", "identity")]
    if (
        native.get("passed") is not True
        or native.get("state") != "complete"
        or set(native.get("windows", {})) != keys
        or type(native.get("verified_windows")) is not int
        or native["verified_windows"] != len(keys)
        or type(native.get("total_windows")) is not int
        or native["total_windows"] != len(keys)
        or native.get("source_union_sha256") != source_sha
        or native.get("calibration_digest") != calibration
    ):
        raise ValueError("Complete actual native verification of every formal window is required")
    if (
        poses.get("state") != "complete"
        or poses.get("source_union_sha256") != source_sha
        or poses.get("config", {}).get("calibration_digest") != calibration
        or poses.get("config_digest")
        != hashlib.sha256(json.dumps(poses["config"], sort_keys=True).encode()).hexdigest()
    ):
        raise ValueError("Complete actual native poses use another source or camera config")
    if (
        identity.get("review_status") != "reviewed"
        or set(identity.get("windows", {})) != keys
        or identity.get("source_union_sha256") != source_sha
        or identity.get("calibration_digest") != calibration
        or not isinstance(identity.get("evidence_files"), dict)
        or not identity["evidence_files"]
    ):
        raise ValueError("Every formal window needs genuinely reviewed identity evidence")
    for path, sha in identity["evidence_files"].items():
        if _sha(path) != sha or not re.fullmatch(r"[0-9a-f]{64}", str(sha)):
            raise ValueError("Reviewed identity evidence changed")
        observed[str(Path(path).resolve())] = sha
    if (
        body.get("state") != "body_qa_candidate_complete"
        or type(body.get("windows")) is not int
        or body["windows"] != len(keys)
    ):
        raise ValueError("Actual formal body QA is incomplete")

    windows = {}
    accepted_total = 0
    for key in sorted(keys):
        sid, name = key.split(":", 1)
        window = labels / "sessions" / sid / "windows" / name
        entry = imported["windows"][key]
        original, _ = _bound(entry["original_report"], observed)
        current, current_bytes = _bound(entry["current_report"], observed)
        snapshot_path = baseline / "reports" / sid / name / "fit_report.json"
        actual_path = window / "fit_report.json"
        actual, actual_bytes = _bound(_binding(actual_path), observed)
        if (
            Path(entry["current_report"]["path"]).resolve() != snapshot_path
            or current_bytes != actual_bytes
            or current != actual
            or os.path.samefile(snapshot_path, actual_path)
        ):
            raise ValueError("Current report is not the independent byte-identical fixed pre-leg snapshot")
        before = _artifacts(original, Path(entry["original_report"]["path"]).parent, observed)
        after = _artifacts(current, window, observed)
        if before != after:
            raise ValueError("Formal numerical ptv3/fits artifacts changed during import")
        accepted = _count(current.get("accepted"), "fit accepted")
        if accepted != _count(original.get("accepted"), "original accepted"):
            raise ValueError("Import changed actual accepted membership")
        if _row_qa(Path(entry["original_report"]["path"]).parent) != _row_qa(window):
            raise ValueError("Imported baseline row QA values changed")
        checked = native["windows"][key].get("body", {})
        if (
            checked.get("report_sha256") != entry["current_report"]["sha256"]
            or type(checked.get("accepted_checked")) is not int
            or checked["accepted_checked"] != accepted
        ):
            raise ValueError("This exact fit report and accepted rows were not native verified")
        for path, sha in native["windows"][key].get("teacher_inputs_sha256", {}).items():
            if _sha(path) != sha:
                raise ValueError("Native verification teacher input changed")
            observed[str(Path(path).resolve())] = sha
        windows[key] = dict(report_snapshot=entry["current_report"])
        accepted_total += accepted
    if (
        _count(body.get("accepted"), "body accepted") != accepted_total
        or _count(native.get("accepted_checked"), "native accepted") != accepted_total
    ):
        raise ValueError("Actual full body/native accepted totals differ")

    # The existing CPU collector checks the original QA gates, every artifact,
    # duplicate accepted person scans and the actual report/row count per window.
    from .new_source_completion import collect_absolute_qa

    qa = collect_absolute_qa(repair)
    if (
        set(qa["windows"]) != keys
        or len(qa["frames"]) != accepted_total
        or qa["source_union_sha256"] != source_sha
        or qa["calibration_digest"] != calibration
    ):
        raise ValueError("Recomputed absolute QA does not cover the real formal scope")
    for key, entry in qa["windows"].items():
        if (
            entry["fit_report_sha256"] != windows[key]["report_snapshot"]["sha256"]
            or entry["accepted"] != native["windows"][key]["body"]["accepted_checked"]
        ):
            raise ValueError("Recomputed per-window accepted count/report differs")
    if "original_qa_snapshot" in imported:
        before_qa, _ = _bound(imported["original_qa_snapshot"], observed)
        if (
            before_qa.get("schema") != "badminton-native-absolute-qa-v1"
            or before_qa.get("frames") != qa["frames"]
            or set(before_qa.get("windows", {})) != keys
            or any(
                before_qa["windows"][key].get("accepted") != qa["windows"][key]["accepted"]
                or before_qa["windows"][key].get("fit_report_sha256")
                != imported["windows"][key]["original_report"]["sha256"]
                for key in keys
            )
        ):
            raise ValueError("Recomputed QA values differ from the import's original QA snapshot")

    names = dict(
        native="native_verification.json",
        body="body_receipt.json",
        pose="pose_receipt.json",
        identity="identity_decisions.json",
        qa="absolute_qa.json",
    )
    payloads["qa"] = _encode(qa)
    snapshots = {role: _binding(baseline / names[role], data) for role, data in payloads.items()}
    result = dict(
        schema="gvhmr-formal-preleg-baseline-v1",
        state="complete",
        source_input=source_binding,
        source_union_sha256=source_sha,
        calibration_digest=calibration,
        body_receipt=snapshots["body"],
        pose_receipt=snapshots["pose"],
        native_verification=snapshots["native"],
        absolute_qa=snapshots["qa"],
        reviewed_identity=snapshots["identity"],
        numeric_import=imported_binding,
        plan=plan_binding,
        windows=windows,
        accepted=accepted_total,
        training_published=False,
        gpu_started=False,
    )
    result_bytes = _encode(result)
    result_binding = _binding(baseline / "baseline.json", result_bytes)
    receipt = dict(
        schema="gvhmr-formal-preleg-freezer-receipt-v1",
        state="complete",
        root=str(root),
        source_input=source_binding,
        numeric_import=imported_binding,
        plan=plan_binding,
        inputs=inputs,
        baseline=result_binding,
        snapshots=snapshots,
        windows=len(keys),
        accepted=accepted_total,
        training_published=False,
        gpu_started=False,
    )
    targets = [(baseline / names[role], payload) for role, payload in payloads.items()]
    targets += [(baseline / "baseline.json", result_bytes), (baseline / "freezer_receipt.json", _encode(receipt))]
    for path, payload in targets:
        _preflight_target(path, payload, baseline)
        for role, original in paths.items():
            if path.exists() and os.path.samefile(path, original):
                raise ValueError("Evidence snapshots must be independent copies of actual receipts")
    for path, sha in observed.items():
        if _sha(path) != sha:
            raise ValueError("Baseline input changed during CPU validation")
    for path, payload in targets:
        _publish(path, payload)
    return receipt
