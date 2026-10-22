"""Synthetic CPU proofs for the completed-parent-only formal numeric audit."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hmr4d.lidar_mocap import formal_import_audit as audit
from hmr4d.lidar_mocap.fitting import FittingConfig


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def asset(path, content=b"synthetic numerical payload; never deserialized"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.fixture
def case(tmp_path, monkeypatch):
    project = tmp_path / "project"
    formal = project / "outputs/formal"
    parents = [project / "outputs/parent_1019", project / "outputs/parent_1020"]
    monkeypatch.setattr(audit, "PROJ_ROOT", project)
    monkeypatch.setattr(
        audit.CalibrationSnapshot, "load", staticmethod(lambda path: SimpleNamespace(data=audit.read_json(path)))
    )
    monkeypatch.setattr(audit.BadmintonPipeline, "_fit_digest", lambda self: "current_algorithm")
    monkeypatch.setattr(audit, "load_pilot_run_relocations", lambda source, calibration: None)
    algorithm_calls = []

    def compatible(original, current, commit=None):
        algorithm_calls.append((original, current, commit))
        if original != current and not commit:
            raise ValueError("Changed fit digest needs explicit original source revision")
        return dict(compatible=True, original_digest=original, current_digest=current, original_commit=commit)

    monkeypatch.setattr(audit, "audit_algorithm_compatibility", compatible)
    verifier = project / "tools/verify_badminton_native_repair.py"
    asset(verifier, b"frozen verifier")
    asset(project / "hmr4d/lidar_mocap/pose_ray_filter.py", b"frozen ray policy")
    policy = audit.file_sha(project / "hmr4d/lidar_mocap/pose_ray_filter.py")
    config = asdict(FittingConfig(distal_pose_prior_weight=0.03, lower_limb_reprojection_weight=2))
    role_map = dict(train=["a"], val=["b"], test=["c"])
    role_contract = tmp_path / "role_contract.json"
    write(role_contract, dict(new_source_splits=role_map))
    handoff = tmp_path / "camera_handoff.json"
    write(handoff, dict(actual_admitted_camera=True))
    camera_certificate = tmp_path / "camera_certificate.json"
    write(camera_certificate, dict(camera_calibration_admitted=True))
    sessions, per_source, formal_sources, proof_sources, parent_manifests = {}, {}, {}, {}, []
    formal_jobs = []
    common = {field: {"unchanged": field} for field in audit.GEOMETRY_FIELDS}
    formal_cal = dict(common, review_only=False, snapshot_digest="formal", sessions={})

    for parent, source_ids in zip(parents, (["a"], ["b", "c"])):
        old_cal = dict(
            common,
            review_only=True,
            snapshot_digest=parent.name,
            sessions={
                sid: dict(
                    manifest_sha256=sid, frame_count=1, frame_map_sha256=sid, source_clock_audit={"original": sid}
                )
                for sid in source_ids
            },
        )
        formal_cal["sessions"].update(old_cal["sessions"])
        old_cal_path = parent / "source/calibration/calibration_snapshot.json"
        write(old_cal_path, old_cal)
        write(parent / "repair/final_labels/calibration/calibration_snapshot.json", old_cal)
        manifest = tmp_path / (parent.name + "_manifest.json")
        write(manifest, dict(source_ids=source_ids))
        manifest_binding = audit.binding(manifest)
        parent_manifests.append(manifest_binding)
        write(parent / "pilot_input.json", dict(manifest=manifest_binding))
        old_sources, old_jobs, identity_windows, native_windows = {}, [], {}, {}
        pose_config = dict(calibration_digest=parent.name, models={"pose": "exact"}, precision="float16")

        for sid in source_ids:
            old_index = tmp_path / (sid + "_review_index.json")
            admitted_index = tmp_path / (sid + "_formal_index.json")
            index = {field: {"original": sid} for field in audit.SOURCE_FIELDS}
            index.update(
                source_id=sid,
                frame_count=1,
                frames=[dict(ordinal=0)],
                coordinate_contract=dict(
                    M_baked_into_stored_xyz=False, application_raw_to_viewer_M=1, source_requested_M=1
                ),
                crop_binding=dict(bbox_sha256="same"),
            )
            write(old_index, index)
            write(admitted_index, dict(index, camera_calibration_admitted=True))
            old_sources[sid] = dict(source_index=audit.binding(old_index), baseline=False)
            formal_sources[sid] = dict(source_index=audit.binding(admitted_index), baseline=False)
            initial = parent / "repair/initial_teacher/sessions" / sid / "windows/0_1"
            target_initial = formal / "repair/initial_teacher/sessions" / sid / "windows/0_1"
            window = parent / "repair/final_labels/sessions" / sid / "windows/0_1"
            window_config = dict(
                schema="badminton-window-v1",
                session_id=sid,
                window_id="0_1",
                left_us=0,
                right_us=100,
                calibration_digest=parent.name,
                source_pairs=[0],
                settings=dict(camera="cam2", teacher_stride=1),
            )
            write(initial / "window.json", window_config)
            write(target_initial / "window.json", dict(window_config, calibration_digest="formal"))
            old_jobs.append(
                dict(
                    source_id=sid,
                    source_window=str(initial),
                    window_json_sha256=audit.file_sha(initial / "window.json"),
                )
            )
            formal_jobs.append(
                dict(
                    source_id=sid,
                    source_window=str(target_initial),
                    window_json_sha256=audit.file_sha(target_initial / "window.json"),
                )
            )
            for folder in (initial, window):
                write(folder / "teacher/manifest.json", dict(tracks=[dict(path="track.pt")], provenance_hash="old"))
                asset(folder / "teacher/track.pt")
            asset(initial / "teacher/detections.pt")
            key = sid + ":0_1"
            identity_windows[key] = dict(
                source_window=str(initial),
                source_teacher_manifest_sha256=audit.file_sha(initial / "teacher/manifest.json"),
                links=[],
                cannot_link=[],
                drop_spans=[],
            )
            cloud = tmp_path / (sid + "_lidar.png")
            asset(cloud, b"encoded original source cloud")
            cameras = {camera: dict(record_index=0, matched=True) for camera in ("cam1", "cam2")}
            native_rows = {camera: dict(record_index=0, common_time_us=50, media_pts_us=100) for camera in cameras}
            evidence = {}
            for camera in cameras:
                path = parent / "repair/native_poses" / sid / camera / "0000000.json"
                write(
                    path,
                    dict(
                        source_id=sid,
                        camera=camera,
                        config_digest=audit.pose_config_digest(pose_config),
                        native=native_rows[camera],
                        keypoints=[[[1.0, 2.0, 0.9]]],
                    ),
                )
                evidence[camera] = audit.binding(path)
            qa = dict(
                finite=True,
                final_p2surface_mean_m=0.02,
                final_p2surface_p90_m=0.03,
                floor_penetration_m=0.01,
                reprojection_mean_px=3.0,
                reprojection_valid_joints=dict(cam1=8, cam2=8),
            )
            quality = []
            for number, accepted in enumerate((True, False)):
                fit = window / "fits" / f"{number}.npz"
                asset(fit)
                quality.append(
                    dict(
                        session_id=sid,
                        lidar_index=0,
                        scan_mid_us=50,
                        cameras=cameras,
                        calibration_digest=parent.name,
                        lidar_path=str(cloud),
                        fit_path=str(fit),
                        qa=deepcopy(qa),
                        source_point_count=100,
                        accepted=accepted,
                        rejection_reasons=[] if accepted else ["track_shape_not_yet_established"],
                        native_pose_evidence=dict(files=evidence),
                    )
                )
            rows(window / "quality.jsonl", quality)
            rows(window / "ptv3.metadata.jsonl", quality[:1])
            asset(window / "ptv3.pth")
            session = SimpleNamespace(
                pairs=lambda cloud=cloud, cameras=cameras: [
                    dict(lidar_index=0, scan_mid_us=50, cameras=cameras, lidar_path=str(cloud))
                ],
                camera_rows=lambda camera, native_rows=native_rows: [native_rows[camera]],
            )
            sessions[sid] = session
            per_source[sid] = dict(
                parent=parent,
                initial=initial,
                target_initial=target_initial,
                window=window,
                key=key,
                config=window_config,
                cloud=cloud,
            )
            native_windows[key] = dict(body=dict(accepted_checked=1), teacher_inputs_sha256={})

        registry_path = parent / "source/source_registry.json"
        write(registry_path, dict(review_only=True, pilot_manifest=manifest_binding, sources=old_sources))
        source_plan = dict(
            schema="badminton-new-source-repair-input-v1",
            review_only=True,
            source_root=str(parent / "source"),
            calibration_digest=parent.name,
            jobs=old_jobs,
            source_registry_sha256=audit.file_sha(registry_path),
        )
        source_path = parent / "repair/new_source_input.json"
        write(source_path, source_plan)
        source_sha = audit.file_sha(source_path)
        body_plan = dict(
            schema="badminton-body-repair-plan-v1",
            source_union=str(source_path),
            source_union_sha256=source_sha,
            calibration_digest=parent.name,
            fitting_config=config,
            jobs=old_jobs,
        )
        write(parent / "repair/final_labels/repair_plan.json", body_plan)
        cases_path = parent / "repair/case_plan.json"
        write(
            cases_path,
            dict(
                source_union=str(source_path),
                source_union_sha256=source_sha,
                selected_scans={sid: [0] for sid in source_ids},
            ),
        )
        identity_evidence = parent / "repair/identity_evidence/witness.json"
        write(identity_evidence, dict(reviewed=True))
        identity_path = parent / "repair/identity_evidence/identity_decisions.json"
        write(
            identity_path,
            dict(
                schema="badminton-native-identity-decisions-v1",
                review_status="reviewed",
                source_union_sha256=source_sha,
                calibration_digest=parent.name,
                windows=identity_windows,
                evidence_files={str(identity_evidence): audit.file_sha(identity_evidence)},
            ),
        )
        pose_path = parent / "repair/pose_receipt.json"
        write(
            pose_path,
            dict(
                state="complete",
                config=pose_config,
                config_digest=audit.pose_config_digest(pose_config),
                plan_sha256=audit.file_sha(cases_path),
                source_union_sha256=source_sha,
            ),
        )
        for sid in source_ids:
            p = per_source[sid]
            settings = dict(
                p["config"]["settings"],
                share_track_shape=True,
                native_pose_joint_ray_policy_sha256=policy,
                native_pose_receipt_sha256=audit.file_sha(pose_path),
            )
            write(p["window"] / "window.json", dict(p["config"], settings=settings))
            paths = [
                p["window"] / name
                for name in ("quality.jsonl", "ptv3.metadata.jsonl", "ptv3.pth", "fits/0.npz", "fits/1.npz")
            ]
            report = dict(
                session_id=sid,
                window="0_1",
                scan_count=1,
                fitted=2,
                accepted=1,
                fitting_config=config,
                fit_algorithm_digest="current_algorithm",
                settings=settings,
                teacher_provenance_hash="old",
                native_pose_receipt_sha256=audit.file_sha(pose_path),
                artifacts_sha256={path.relative_to(p["window"]).as_posix(): audit.file_sha(path) for path in paths},
                source_lidar_sha256={str(p["cloud"]): audit.file_sha(p["cloud"])},
            )
            write(p["window"] / "fit_report.json", report)
            native_windows[p["key"]]["body"]["report_sha256"] = audit.file_sha(p["window"] / "fit_report.json")
            teachers = [
                parent / "repair/final_labels/repair_plan.json",
                identity_path,
                verifier,
                p["initial"] / "teacher/detections.pt",
            ]
            teachers += [
                folder / "teacher" / name
                for folder in (p["initial"], p["window"])
                for name in ("manifest.json", "track.pt")
            ]
            native_windows[p["key"]]["teacher_inputs_sha256"] = {str(path): audit.file_sha(path) for path in teachers}
            proof_sources[sid] = dict(
                original_registry=audit.binding(registry_path),
                original_calibration=audit.binding(old_cal_path),
                original_index=old_sources[sid]["source_index"],
                admitted_index=formal_sources[sid]["source_index"],
                camera_applicability=audit.binding(camera_certificate),
                source_frames=1,
                original_inputs_exact=True,
                cameras_court_ground_bbox_clock_exact=True,
            )
        write(
            parent / "repair/final_labels/native_verification.json",
            dict(
                passed=True,
                state="complete",
                windows=native_windows,
                verified_windows=len(old_jobs),
                total_windows=len(old_jobs),
                accepted_checked=len(old_jobs),
                source_union_sha256=source_sha,
                calibration_digest=parent.name,
            ),
        )
        write(
            parent / "repair/final_labels/repair_receipt.json",
            dict(state="body_qa_candidate_complete", accepted=len(old_jobs), windows=len(old_jobs)),
        )

    monkeypatch.setattr(audit, "load_run_session", lambda source, calibration, sid, **kw: sessions[sid])
    write(formal / "source/calibration/calibration_snapshot.json", formal_cal)
    write(formal / "repair/final_labels/calibration/calibration_snapshot.json", formal_cal)
    ranges_path = formal / "source/formal_ranges.jsonl"
    rows(ranges_path, [dict(source_id=sid) for sid in formal_sources])
    intake_path = formal / "source/formal_intake.json"
    write(
        intake_path,
        dict(
            schema="badminton-formal-derived-intake-v1",
            camera_admission_handoff=audit.binding(handoff),
            source_role_contract=audit.binding(role_contract),
            parent_review_manifests=parent_manifests,
            prepared_ranges=audit.binding(ranges_path),
        ),
    )
    write(
        formal / "source/source_registry.json",
        dict(
            schema="badminton-formal-derived-source-registry-v1",
            sources=formal_sources,
            intake=audit.binding(intake_path),
            camera_admission_handoff=audit.binding(handoff),
        ),
    )
    write(formal / "source/session_splits.json", role_map)
    proof_path = formal / "formal_input_equivalence.json"
    write(
        proof_path,
        dict(
            schema="badminton-review-formal-input-equivalence-v1",
            state="complete",
            training_published=False,
            fitted_tensors_imported=False,
            geometry_fields=list(audit.GEOMETRY_FIELDS),
            source_fields=list(audit.SOURCE_FIELDS),
            role_assignment=role_map,
            sources=proof_sources,
            evidence=dict(
                formal_registry=audit.binding(formal / "source/source_registry.json"),
                formal_calibration=audit.binding(formal / "source/calibration/calibration_snapshot.json"),
                role_contract=audit.binding(role_contract),
            ),
        ),
    )
    proof_binding = audit.binding(proof_path)
    source_path = formal / "repair/new_source_input.json"
    write(
        source_path,
        dict(
            schema="badminton-new-source-repair-input-v1",
            source_root=str(formal / "source"),
            calibration_digest="formal",
            source_registry_sha256=audit.file_sha(formal / "source/source_registry.json"),
            jobs=formal_jobs,
            purpose="formal_native_rally_derivation",
            session_splits=role_map,
            formal_derivation=dict(input_equivalence=proof_binding, camera_admission_handoff=audit.binding(handoff)),
        ),
    )
    write(
        formal / "repair/final_labels/repair_plan.json",
        dict(
            schema="badminton-body-repair-plan-v1",
            source_union=str(source_path),
            source_union_sha256=audit.file_sha(source_path),
            calibration_digest="formal",
            jobs=formal_jobs,
            fitting_config=config,
        ),
    )
    return SimpleNamespace(
        formal=formal, parents=parents, proof=proof_binding, per_source=per_source, algorithm_calls=algorithm_calls
    )


def run(case, **kwargs):
    return audit.audit_formal_numeric_import(case.formal, case.proof, case.parents, **kwargs)


def change_json(path, mutate):
    value = audit.read_json(path)
    mutate(value)
    write(path, value)


def refresh_report(case, sid):
    p = case.per_source[sid]
    report_path = p["window"] / "fit_report.json"
    report = audit.read_json(report_path)
    report["artifacts_sha256"] = {
        relative: audit.file_sha(p["window"] / relative) for relative in report["artifacts_sha256"]
    }
    write(report_path, report)
    change_json(
        p["parent"] / "repair/final_labels/native_verification.json",
        lambda value: value["windows"][p["key"]]["body"].update(report_sha256=audit.file_sha(report_path)),
    )


def test_complete_parents_return_read_only_bound_inventory(case):
    before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in case.formal.parent.rglob("*")
        if path.is_file()
    }
    result = run(case)
    after = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in case.formal.parent.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert result["schema"] == "badminton-formal-numeric-import-audit-v1"
    assert result["state"] == "eligible_parent_complete_not_imported"
    assert result["formal_root"] == str(case.formal)
    assert result["eligible_windows"] == 3 and result["eligible_fitted"] == 6 and result["eligible_accepted"] == 3
    assert result["fitted_tensors_imported"] is False and result["training_published"] is False
    assert set(result["windows"]) == {"a:0_1", "b:0_1", "c:0_1"}
    assert set(next(iter(result["parent_evidence"].values()))) == {
        "native",
        "identity",
        "pose",
        "body",
        "source_input",
        "body_plan",
        "calibration",
        "source_registry",
        "case_plan",
    }
    assert result["teacher_files"]


@pytest.mark.parametrize(
    "receipt,field,value",
    [
        ("final_labels/native_verification.json", "passed", False),
        ("final_labels/native_verification.json", "state", "partial"),
        ("final_labels/native_verification.json", "total_windows", 1),
        ("identity_evidence/identity_decisions.json", "review_status", "pending"),
        ("pose_receipt.json", "state", "partial"),
        ("pose_receipt.json", "config_digest", "changed"),
        ("pose_receipt.json", "source_union_sha256", "changed"),
        ("final_labels/repair_receipt.json", "accepted", 99),
        ("final_labels/repair_receipt.json", "state", "partition_complete"),
    ],
)
def test_any_incomplete_parent_blocks_before_all_numerical_asset_reads(case, monkeypatch, receipt, field, value):
    change_json(case.parents[1] / "repair" / receipt, lambda record: record.update({field: value}))
    actual_hash = audit.file_sha

    def hash_metadata_only(path):
        assert Path(path).suffix not in {".pt", ".pth", ".npz"}, "Read numerical asset before parent completion barrier"
        return actual_hash(path)

    monkeypatch.setattr(audit, "file_sha", hash_metadata_only)
    with pytest.raises(ValueError, match="Parent"):
        run(case)


@pytest.mark.parametrize("scope", ["native", "identity"])
def test_parent_receipts_must_cover_all_planned_keys(case, scope):
    path = case.parents[1] / (
        "repair/final_labels/native_verification.json"
        if scope == "native"
        else "repair/identity_evidence/identity_decisions.json"
    )
    change_json(path, lambda record: record["windows"].pop("c:0_1"))
    with pytest.raises(ValueError, match="Parent"):
        run(case)


def test_changed_review_evidence_is_rejected(case):
    asset(case.parents[1] / "repair/identity_evidence/witness.json", b"changed")
    with pytest.raises(ValueError, match="Frozen dependency changed"):
        run(case)


def test_formal_requires_separate_root_and_two_distinct_parents(case):
    with pytest.raises(ValueError, match="distinct"):
        audit.audit_formal_numeric_import(case.formal, case.proof, [case.parents[0]] * 2)
    with pytest.raises(ValueError, match="nest"):
        audit.audit_formal_numeric_import(case.parents[0] / "child", case.proof, case.parents)


@pytest.mark.parametrize("mutation", ["schema", "state", "field_coverage", "source_lineage"])
def test_actual_equivalence_proof_cannot_be_partial_or_bind_other_parents(case, mutation):
    def edit(proof):
        if mutation == "schema":
            proof["schema"] = "untyped"
        elif mutation == "state":
            proof["state"] = "pending"
        elif mutation == "field_coverage":
            proof["geometry_fields"].pop()
        else:
            proof["sources"]["c"]["original_inputs_exact"] = False

    change_json(case.proof["path"], edit)
    case.proof = audit.binding(case.proof["path"])
    change_json(
        case.formal / "repair/new_source_input.json",
        lambda plan: plan["formal_derivation"].update(input_equivalence=case.proof),
    )
    change_json(
        case.formal / "repair/final_labels/repair_plan.json",
        lambda plan: plan.update(source_union_sha256=audit.file_sha(case.formal / "repair/new_source_input.json")),
    )
    with pytest.raises(ValueError):
        run(case)


@pytest.mark.parametrize("target", ["fits/0.npz", "ptv3.pth", "teacher/track.pt"])
def test_changed_existing_numerical_bytes_are_rejected_without_deserialization(case, target):
    asset(case.per_source["c"]["window"] / target, b"changed asset")
    with pytest.raises(ValueError, match="Frozen dependency changed"):
        run(case)


def test_artifact_inventory_cannot_escape_original_window(case):
    p = case.per_source["c"]
    outside = p["window"].parent / "outside.pt"
    asset(outside)
    change_json(
        p["window"] / "fit_report.json",
        lambda report: report["artifacts_sha256"].update({"../outside.pt": audit.file_sha(outside)}),
    )
    refresh_report(case, "c")
    with pytest.raises(ValueError, match="escapes"):
        run(case)


def test_report_must_be_exactly_native_verified(case):
    change_json(case.per_source["c"]["window"] / "fit_report.json", lambda report: report.update(accepted=100))
    with pytest.raises(ValueError, match="differs from native"):
        run(case)


def test_joint_ray_policy_is_checked_even_when_fit_algorithm_digest_matches(case):
    p = case.per_source["c"]
    for path in (p["window"] / "fit_report.json", p["window"] / "window.json"):
        change_json(path, lambda value: value["settings"].update(native_pose_joint_ray_policy_sha256="changed"))
    refresh_report(case, "c")
    with pytest.raises(ValueError, match="joint-ray policy"):
        run(case)


def test_changed_algorithm_requires_explicit_original_commit(case):
    p = case.per_source["c"]
    change_json(p["window"] / "fit_report.json", lambda report: report.update(fit_algorithm_digest="old_algorithm"))
    refresh_report(case, "c")
    with pytest.raises(ValueError, match="explicit original"):
        run(case)
    result = run(case, original_commits={str(p["parent"]): "actual_compatible_commit"})
    assert result["windows"]["c:0_1"]["original_fit_algorithm_digest"] == "old_algorithm"
    assert ("old_algorithm", "current_algorithm", "actual_compatible_commit") in case.algorithm_calls


@pytest.mark.parametrize("field,value", [("scan_mid_us", 51), ("calibration_digest", "another"), ("lidar_index", 5)])
def test_quality_scan_scope_must_match_exact_formal_source(case, field, value):
    p = case.per_source["c"]
    quality = audit.read_rows(p["window"] / "quality.jsonl")
    quality[0][field] = value
    rows(p["window"] / "quality.jsonl", quality)
    rows(p["window"] / "ptv3.metadata.jsonl", quality[:1])
    refresh_report(case, "c")
    with pytest.raises(ValueError, match="scan/native/camera"):
        run(case)


def test_original_qa_gate_still_rejects_bad_accepted_row(case):
    p = case.per_source["c"]
    quality = audit.read_rows(p["window"] / "quality.jsonl")
    quality[0]["qa"]["floor_penetration_m"] = 0.5
    rows(p["window"] / "quality.jsonl", quality)
    rows(p["window"] / "ptv3.metadata.jsonl", quality[:1])
    refresh_report(case, "c")
    with pytest.raises(ValueError, match="Unchanged body QA"):
        run(case)


def test_source_cloud_and_native_teacher_inventory_are_actual_sha_bound(case):
    asset(case.per_source["c"]["cloud"], b"changed original source cloud")
    with pytest.raises(ValueError, match="source point-cloud bytes"):
        run(case)


def test_native_teacher_inventory_cannot_omit_a_track_file(case):
    p = case.per_source["c"]

    def edit(native):
        native["windows"][p["key"]]["teacher_inputs_sha256"].pop(str(p["window"] / "teacher/track.pt"))

    change_json(p["parent"] / "repair/final_labels/native_verification.json", edit)
    with pytest.raises(ValueError, match="exact original teacher"):
        run(case)


def test_rejected_rows_retain_their_original_decision_without_new_qa_thresholds(case):
    p = case.per_source["c"]
    quality = audit.read_rows(p["window"] / "quality.jsonl")
    quality[1]["qa"]["floor_penetration_m"] = 0.5
    rows(p["window"] / "quality.jsonl", quality)
    refresh_report(case, "c")
    assert run(case)["eligible_accepted"] == 3
    assert audit.read_rows(p["window"] / "quality.jsonl")[1]["rejection_reasons"] == ["track_shape_not_yet_established"]


@pytest.mark.parametrize("change", ["settings", "source_pairs", "new_field"])
def test_formal_initial_window_must_match_every_original_field_except_calibration(case, change):
    p = case.per_source["c"]
    path = p["target_initial"] / "window.json"

    def edit(value):
        if change == "settings":
            value["settings"]["teacher_stride"] = 2
        elif change == "source_pairs":
            value["source_pairs"] = [0, 1]
        else:
            value["unproved_extra_setting"] = True

    change_json(path, edit)
    source_path = case.formal / "repair/new_source_input.json"

    def update_source(value):
        for job in value["jobs"]:
            if job["source_id"] == "c":
                job["window_json_sha256"] = audit.file_sha(path)

    change_json(source_path, update_source)
    change_json(
        case.formal / "repair/final_labels/repair_plan.json",
        lambda value: value.update(
            jobs=audit.read_json(source_path)["jobs"], source_union_sha256=audit.file_sha(source_path)
        ),
    )
    with pytest.raises(ValueError, match="Formal initial window changed"):
        run(case)


def test_formal_roles_must_match_equivalence_and_bound_role_contract(case):
    change_json(case.formal / "source/session_splits.json", lambda value: value.update(train=["b"], val=["a"]))
    with pytest.raises(ValueError, match="source roles changed"):
        run(case)


def test_formal_derivation_cannot_bind_another_equivalence_proof(case):
    source_path = case.formal / "repair/new_source_input.json"
    change_json(source_path, lambda value: value["formal_derivation"]["input_equivalence"].update(sha256="0" * 64))
    change_json(
        case.formal / "repair/final_labels/repair_plan.json",
        lambda value: value.update(source_union_sha256=audit.file_sha(source_path)),
    )
    with pytest.raises(ValueError, match="Formal derivation"):
        run(case)
