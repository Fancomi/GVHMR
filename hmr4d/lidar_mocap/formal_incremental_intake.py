"""Prepare independent formal metadata from one accepted, unchanged review parent."""

from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys

from hmr4d import PROJ_ROOT
from .incremental_release import load_release_policy, need


def prepare_formal_increment(policy_path, policy_sha256, root):
    """Explicit CPU metadata preparation; never fit, run a teacher or publish labels."""
    from .admitted_native_input import verify_review_snapshot
    from .calibration import CalibrationSnapshot
    from .expanded_dataset import binding, check_binding
    from .formal_numeric_import import _copy_once, _write_once
    from .formal_reuse import audit_formal_inputs, validate_formal_geometry
    from .pilot import review_rally_windows
    from .rally_pipeline import read_json, read_rows
    from .relocations import load_pilot_run_relocations
    from .source_index import SourceIndexedSession

    policy = load_release_policy(policy_path, policy_sha256)
    need(Path(policy.data["gvhmr_root"]).resolve() == PROJ_ROOT.resolve(), "Policy uses another producer checkout")
    root = Path(root).resolve()
    outputs = (PROJ_ROOT / "outputs").resolve()
    need(root != outputs and root.is_relative_to(outputs), "Use a separate formal outputs root")
    need(not root.exists() or not any(root.iterdir()), "Preserve prior preparation; use a fresh formal root")
    entries = policy.scope["review_sources"]
    parents = {str(Path(item["review_root"]).resolve()) for item in entries.values()}
    snapshots = {
        (item["accepted_parent_snapshot"]["path"], item["accepted_parent_snapshot"]["sha256"])
        for item in entries.values()
    }
    need(len(parents) == len(snapshots) == 1, "This v2 intake requires one complete accepted review parent")
    parent = Path(next(iter(parents)))
    need(
        not root.is_relative_to(parent) and not parent.is_relative_to(root),
        "Formal output cannot nest with its review parent",
    )
    first = next(iter(entries.values()))
    for item in entries.values():
        for name in (
            "pilot_input",
            "review_manifest",
            "review_source_registry",
            "review_source_input",
            "accepted_parent_snapshot",
            "accepted_freeze_proof",
        ):
            need(item[name] == first[name], "One parent must use one exact frozen review/camera input")
    parent_cal = CalibrationSnapshot.load(first["accepted_parent_snapshot"]["path"])
    review_cal = CalibrationSnapshot.load(parent / "source/calibration/calibration_snapshot.json")
    snapshot_proof = verify_review_snapshot(
        parent_cal.snapshot_json,
        review_cal.snapshot_json,
        first["review_manifest"]["path"],
        accepted_freeze_proof=first["accepted_freeze_proof"],
    )
    old_registry = read_json(first["review_source_registry"]["path"])
    old_plan = read_json(first["review_source_input"]["path"])
    need(
        set(old_registry["sources"]) == set(entries) == set(parent_cal.data["sessions"]),
        "The complete review and accepted camera must cover the exact policy source scope",
    )
    validate_formal_geometry(review_cal.data, parent_cal.data, entries)
    resolver = load_pilot_run_relocations(parent / "source", review_cal.data)
    intake = read_json(old_registry["intake"]["path"])
    check_binding(old_registry["intake"])
    check_binding(intake["prepared_ranges"])
    ranges = read_rows(intake["prepared_ranges"]["path"])
    need(
        {row["source_id"] for row in ranges} == set(entries), "Native ranges differ from the complete incremental scope"
    )
    sessions, window_configs, selected = {}, {}, {}
    old_jobs = {(job["source_id"], Path(job["source_window"]).name): job for job in old_plan["jobs"]}
    need(len(old_jobs) == len(old_plan["jobs"]), "Original review plan repeats a native window")
    for sid, item in entries.items():
        session = SourceIndexedSession.load_source_index(
            item["admitted_source_index"]["path"],
            policy.data["raw_data_root"],
            expected_sha256=item["admitted_source_index"]["sha256"],
            require_admitted=True,
            relocations=resolver,
        )
        need(
            session.camera_applicability
            and dict(path=session.camera_applicability["file"], sha256=session.camera_applicability["sha256"])
            == item["camera_applicability"],
            "Actual native camera certificate differs from the reviewed accepted source",
        )
        need(
            session.source_index.get("accepted_bundle_sha256") == parent_cal.data["configuration_files_sha256"],
            "Admitted source uses another physical camera bundle",
        )
        sessions[sid] = session
        source_ranges = [row for row in ranges if row["source_id"] == sid]
        windows = review_rally_windows(session.pairs(), source_ranges)
        selected[sid] = sorted({scan for row in source_ranges for scan in row["eligible_paired_png_indices"]})
        for name, left, right, pairs in windows:
            job = old_jobs.get((sid, name))
            need(job is not None, "Original complete review plan lacks an actual native window")
            original = Path(job["source_window"]).resolve()
            need(
                original == parent / "repair/initial_teacher/sessions" / sid / "windows" / name,
                "Review window leaves its exact parent",
            )
            config_path = original / "window.json"
            need(binding(config_path)["sha256"] == job["window_json_sha256"], "Original native window changed")
            config = read_json(config_path)
            need(
                config["session_id"] == sid
                and config["window_id"] == name
                and config["calibration_digest"] == review_cal.data["snapshot_digest"]
                and config["left_us"] == left
                and config["right_us"] == right
                and config["source_pairs"] == [pair["lidar_index"] for pair in pairs],
                "Original window differs from the exact native range/time/scan scope",
            )
            window_configs[sid, name] = dict(config, calibration_digest=parent_cal.data["snapshot_digest"])
    need(set(window_configs) == set(old_jobs), "Formal windows cannot omit or extend the complete review plan")
    policy.recheck()
    source, repair = root / "source", root / "repair"
    for artifact in parent_cal.data["artifacts"].values():
        relative = Path(artifact["path"])
        original = (parent_cal.snapshot_json.parent / relative).resolve()
        need(
            not relative.is_absolute() and original.is_relative_to(parent_cal.snapshot_json.parent),
            "Accepted camera artifact leaves its frozen snapshot",
        )
        _copy_once(dict(path=str(original), sha256=artifact["sha256"]), source / "calibration" / relative)
    _copy_once(first["accepted_parent_snapshot"], source / "calibration/calibration_snapshot.json")
    _write_once(root / "import_evidence/accepted_review_snapshot.json", snapshot_proof)
    range_binding = _copy_once(intake["prepared_ranges"], source / "formal_ranges.jsonl")
    data_root = Path(policy.data["raw_data_root"]).resolve()
    index_refs = {}
    for sid, item in entries.items():
        path = Path(item["admitted_source_index"]["path"]).resolve()
        need(path.is_relative_to(data_root), "Admitted source index leaves the explicit raw data root")
        index_refs[sid] = dict(
            file=path.relative_to(data_root).as_posix(), sha256=item["admitted_source_index"]["sha256"]
        )
    handoff = _write_once(
        source / "camera_admission_handoff.json",
        dict(
            schema="badminton-camera-admission-handoff-v1",
            data_root=str(data_root),
            source_indexes=index_refs,
            accepted_bundle_sha256=parent_cal.data["configuration_files_sha256"],
            camera_calibration_admitted=True,
            body_labels_production_admitted=False,
            training_published=False,
            release_policy=policy.binding,
            accepted_parent_snapshot=first["accepted_parent_snapshot"],
            accepted_freeze_proof=first["accepted_freeze_proof"],
        ),
    )
    formal_intake = dict(
        schema="badminton-formal-derived-intake-v1",
        state="camera_and_native_ranges_prepared",
        data_root=str(data_root),
        camera_admission_handoff=handoff,
        parent_review_manifests=[first["review_manifest"]],
        prepared_ranges=range_binding,
        source_role_contract=policy.binding,
        release_policy=policy.binding,
        execution_scope="all_effective_native_rallies",
        rallies=len(ranges),
        selected_scans=sum(map(len, selected.values())),
        camera_calibration_admitted=True,
        body_labels_production_admitted=False,
        production_admitted=False,
        training_published=False,
    )
    intake_binding = _write_once(source / "formal_intake.json", formal_intake)
    registry = dict(
        schema="badminton-formal-derived-source-registry-v1",
        data_root=str(data_root),
        sources=deepcopy(old_registry["sources"]),
        intake=intake_binding,
        camera_admission_handoff=handoff,
        release_policy=policy.binding,
        execution_scope="all_effective_native_rallies",
        training_published=False,
    )
    if old_registry.get("data_relocations") is not None:
        registry["data_relocations"] = deepcopy(old_registry["data_relocations"])
    _write_once(source / "source_registry.json", registry)
    _write_once(source / "session_splits.json", policy.roles)
    equivalence = audit_formal_inputs(
        [parent],
        source,
        policy.binding["path"],
        release_policy=policy.binding["path"],
        release_policy_sha256=policy.binding["sha256"],
    )
    equivalence_binding = _write_once(root / "formal_input_equivalence.json", equivalence)
    jobs = []
    for (sid, name), config in window_configs.items():
        target = repair / "initial_teacher/sessions" / sid / "windows" / name
        config_binding = _write_once(target / "window.json", config)
        jobs.append(dict(source_id=sid, source_window=str(target), window_json_sha256=config_binding["sha256"]))
    initial = repair / "initial_teacher"
    shutil.copytree(source / "calibration", initial / "calibration")
    _copy_once(binding(source / "source_registry.json"), initial / "source_registry.json")
    plan = dict(
        schema="badminton-new-source-repair-input-v1",
        source_root=str(source),
        calibration_digest=parent_cal.data["snapshot_digest"],
        jobs=jobs,
        source_registry_sha256=binding(source / "source_registry.json")["sha256"],
        session_splits=policy.roles,
        suggested_session_splits={role: dict(sessions=ids) for role, ids in policy.roles.items()},
        purpose="formal_native_rally_derivation",
        required_nonempty_splits=policy.required,
        release_policy=policy.binding,
        training_published=False,
        identity_status="pending_native_detection_evidence",
        formal_derivation=dict(input_equivalence=equivalence_binding, camera_admission_handoff=handoff),
    )
    if registry.get("data_relocations") is not None:
        plan["data_relocations"] = registry["data_relocations"]
    input_binding = _write_once(repair / "new_source_input.json", plan)
    endpoints = _write_once(repair / "unresolved_native_endpoints.json", [])
    _write_once(
        repair / "case_plan.json",
        dict(
            schema="badminton-body-repair-cases-v2",
            source_union=input_binding["path"],
            source_union_sha256=input_binding["sha256"],
            clips=[],
            selected_scans=selected,
            unresolved_native_endpoints=endpoints["path"],
            track_candidates=[],
        ),
    )
    command = [
        sys.executable,
        "-B",
        str(PROJ_ROOT / "tools/repair_badminton_body_labels.py"),
        "--stage",
        "prepare",
        "--union",
        input_binding["path"],
        "--source-root",
        str(source),
        "--output-root",
        str(repair / "final_labels"),
        "--native-poses-root",
        str(repair),
        "--joint-ray-filter",
        "--distal-pose-weight",
        "0.03",
        "--lower-limb-reprojection-weight",
        "2",
        "--device",
        "cpu",
        "--threads",
        "1",
    ]
    import os

    subprocess.run(
        command,
        cwd=PROJ_ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"),
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
        | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
    )
    body = read_json(repair / "final_labels/repair_plan.json")
    need(
        body["jobs"] == jobs
        and body["source_union_sha256"] == input_binding["sha256"]
        and body["calibration_digest"] == plan["calibration_digest"],
        "Original body planner returned different inputs",
    )
    policy.recheck()
    return _write_once(
        root / "formal_increment_preparation.json",
        dict(
            schema="badminton-formal-increment-preparation-v2",
            state="formal_metadata_prepared_body_import_pending",
            release_policy=policy.binding,
            source_registry=binding(source / "source_registry.json"),
            source_input=input_binding,
            input_equivalence=equivalence_binding,
            body_plan=binding(repair / "final_labels/repair_plan.json"),
            windows=len(jobs),
            model_loaded=False,
            gpu_started=False,
            training_published=False,
        ),
    )
