"""Read-only eligibility proof for importing completed review bodies into a formal run."""

from dataclasses import asdict
from pathlib import Path

from hmr4d import PROJ_ROOT
from .calibration import CalibrationSnapshot
from .dual_view import file_sha
from .expanded_dataset import binding, check_binding
from .export import acceptance_reasons
from .fit_reuse import audit_algorithm_compatibility, validate_calibration_reuse
from .fitting import FittingConfig
from .formal_reuse import GEOMETRY_FIELDS, SOURCE_FIELDS, validate_formal_geometry, validate_formal_source
from .full_native import pose_config_digest
from .pipeline import BadmintonPipeline
from .rally_pipeline import load_run_session, read_json, read_rows, validate_session_splits
from .relocations import load_pilot_run_relocations


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _confined(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    _require(path != root and path.is_relative_to(root), "Formal import dependency escapes its declared root")
    return path


def _checked(entry):
    _require(isinstance(entry, dict) and isinstance(entry.get("path"), str), "Missing formal import file binding")
    digest = entry.get("sha256")
    _require(
        isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
        "Invalid formal import SHA256 binding",
    )
    check_binding(entry)
    return entry


def _matches(entry, path):
    _checked(entry)
    _require(entry == binding(path), "Formal import proof binds another input file")


def _jobs(plan, root):
    _require(isinstance(plan.get("jobs"), list) and plan["jobs"], "Formal import requires a complete nonempty job plan")
    result = {}
    for job in plan["jobs"]:
        sid, path = job["source_id"], _confined(job["source_window"], root / "repair/initial_teacher")
        _require(
            path == root / "repair/initial_teacher/sessions" / sid / "windows" / path.name,
            "Job uses another source window",
        )
        key = sid + ":" + path.name
        _require(key not in result, "Formal import job plan repeats a window")
        _matches(dict(path=str(path / "window.json"), sha256=job["window_json_sha256"]), path / "window.json")
        result[key] = path
    return result


def _plan_contract(root, source_input, body_plan, calibration, registry):
    source_path = root / "repair/new_source_input.json"
    _require(
        source_input.get("schema") == "badminton-new-source-repair-input-v1"
        and body_plan.get("schema") == "badminton-body-repair-plan-v1",
        "Unexpected formal import plan schema",
    )
    _require(
        Path(source_input["source_root"]).resolve() == root / "source"
        and source_input["source_registry_sha256"] == file_sha(root / "source/source_registry.json")
        and source_input["calibration_digest"] == calibration.data["snapshot_digest"]
        and Path(body_plan["source_union"]).resolve() == source_path
        and body_plan["source_union_sha256"] == file_sha(source_path)
        and body_plan["calibration_digest"] == calibration.data["snapshot_digest"]
        and body_plan["jobs"] == source_input["jobs"],
        "Body/source plan bindings differ",
    )
    _require(
        set(registry["sources"]) == set(calibration.data["sessions"]), "Registry and calibration source scopes differ"
    )
    _require(
        body_plan["fitting_config"]
        == asdict(FittingConfig(distal_pose_prior_weight=0.03, lower_limb_reprojection_weight=2)),
        "Body fitting configuration changed",
    )
    labels_cal = CalibrationSnapshot.load(root / "repair/final_labels/calibration/calibration_snapshot.json")
    validate_calibration_reuse(calibration.data, labels_cal.data)
    _require(
        labels_cal.data["snapshot_digest"] == calibration.data["snapshot_digest"],
        "Body labels use another calibration snapshot",
    )


def _parent_preflight(root):
    """Only read plans and receipts: never inspect fitted or teacher tensor assets here."""
    paths = dict(
        native=root / "repair/final_labels/native_verification.json",
        identity=root / "repair/identity_evidence/identity_decisions.json",
        pose=root / "repair/pose_receipt.json",
        body=root / "repair/final_labels/repair_receipt.json",
        source_input=root / "repair/new_source_input.json",
        body_plan=root / "repair/final_labels/repair_plan.json",
        calibration=root / "source/calibration/calibration_snapshot.json",
        source_registry=root / "source/source_registry.json",
        case_plan=root / "repair/case_plan.json",
    )
    _require(
        all(p.is_file() for p in paths.values()), "Every parent must complete native, identity, pose and body receipts"
    )
    values = {name: read_json(path) for name, path in paths.items()}
    native, identity, pose, body = (values[k] for k in ("native", "identity", "pose", "body"))
    plan, body_plan, registry = (values[k] for k in ("source_input", "body_plan", "source_registry"))
    jobs = _jobs(plan, root)
    keys, digest, source_sha = set(jobs), values["calibration"]["snapshot_digest"], file_sha(paths["source_input"])
    _require(
        registry.get("review_only") is True and plan.get("review_only") is True,
        "Formal numeric import parents must be explicit review roots",
    )
    _require(
        native.get("passed") is True
        and native.get("state") == "complete"
        and set(native.get("windows", {})) == keys
        and native.get("verified_windows") == len(keys)
        and native.get("total_windows") == len(keys)
        and native.get("source_union_sha256") == source_sha
        and native.get("calibration_digest") == digest,
        "Parent native verification is incomplete or has another full planned scope",
    )
    _require(
        identity.get("schema") == "badminton-native-identity-decisions-v1"
        and identity.get("review_status") == "reviewed"
        and set(identity.get("windows", {})) == keys
        and identity.get("source_union_sha256") == source_sha
        and identity.get("calibration_digest") == digest
        and bool(identity.get("evidence_files")),
        "Parent identity evidence is incomplete or unreviewed",
    )
    _require(
        pose.get("state") == "complete"
        and pose.get("source_union_sha256") == source_sha
        and pose.get("plan_sha256") == file_sha(paths["case_plan"])
        and pose.get("config", {}).get("calibration_digest") == digest
        and pose.get("config_digest") == pose_config_digest(pose.get("config", {})),
        "Parent pose receipt is incomplete or changed source/calibration/config",
    )
    _require(
        body.get("state") == "body_qa_candidate_complete"
        and body.get("windows") == len(keys)
        and body.get("accepted") == native.get("accepted_checked")
        and sum(w["body"]["accepted_checked"] for w in native["windows"].values()) == native["accepted_checked"],
        "Parent body receipt differs from complete native verification",
    )
    _require(
        body_plan.get("jobs") == plan["jobs"]
        and body_plan.get("source_union_sha256") == source_sha
        and Path(body_plan["source_union"]).resolve() == paths["source_input"]
        and body_plan.get("calibration_digest") == digest,
        "Parent body plan changed",
    )
    cases = values["case_plan"]
    _require(
        cases.get("source_union_sha256") == source_sha
        and Path(cases["source_union"]).resolve() == paths["source_input"]
        and set(cases.get("selected_scans", {})) == set(registry["sources"]),
        "Parent case plan changed",
    )
    for key, initial in jobs.items():
        _require(
            Path(identity["windows"][key]["source_window"]).resolve() == initial,
            "Parent identity window differs from its complete plan",
        )
    return dict(paths=paths, values=values, jobs=jobs)


def _teacher_inventory(parent, key, initial, window):
    native = parent["values"]["native"]["windows"][key]
    expected = [
        parent["paths"]["body_plan"],
        parent["paths"]["identity"],
        PROJ_ROOT / "tools/verify_badminton_native_repair.py",
        initial / "teacher/detections.pt",
    ]
    for folder in (initial, window):
        teacher = folder / "teacher"
        manifest = teacher / "manifest.json"
        expected.append(manifest)
        data = read_json(manifest)
        _require(isinstance(data.get("tracks"), list), "Native-verified teacher manifest has no track inventory")
        expected.extend(_confined(teacher / item["path"], teacher) for item in data["tracks"])
    original = native.get("teacher_inputs_sha256", {})
    canonical = {str(Path(path).resolve()): sha for path, sha in original.items()}
    _require(
        len(canonical) == len(original) and set(canonical) == {str(p.resolve()) for p in expected},
        "Parent native verification lacks its exact original teacher inputs",
    )
    for path, sha in canonical.items():
        _checked(dict(path=path, sha256=sha))
    initial_manifest = initial / "teacher/manifest.json"
    _require(
        file_sha(initial_manifest) == parent["values"]["identity"]["windows"][key]["source_teacher_manifest_sha256"],
        "Reviewed initial teacher manifest changed",
    )
    return canonical


def audit_formal_numeric_import(formal_root, input_equivalence_binding, parent_roots, original_commits=None):
    """Return import eligibility only; no output directories, tensors or receipts are written."""
    root = Path(formal_root).resolve()
    parents = [Path(p).resolve() for p in parent_roots]
    outputs = (PROJ_ROOT / "outputs").resolve()
    _require(root != outputs and root.is_relative_to(outputs), "Formal import needs a separate project outputs root")
    from .incremental_release import POLICY, policy_from_binding, validate_roles

    incremental = None
    if len(parents) != 2:
        _checked(input_equivalence_binding)
        initial_proof = read_json(input_equivalence_binding["path"])
        role_binding = initial_proof.get("evidence", {}).get("role_contract")
        if isinstance(role_binding, dict):
            _checked(role_binding)
            if read_json(role_binding["path"]).get("schema") == POLICY:
                incremental = policy_from_binding(role_binding)
    _require(
        bool(parents) and len(set(parents)) == len(parents) and (len(parents) == 2 or incremental is not None),
        "Exactly two explicit distinct review parent roots are required",
    )
    for parent in parents:
        _require(
            parent != outputs
            and parent.is_relative_to(outputs)
            and root != parent
            and not root.is_relative_to(parent)
            and not parent.is_relative_to(root),
            "Formal output cannot nest within or overwrite a parent root",
        )
    _require(
        all(not left.is_relative_to(right) for left in parents for right in parents if left != right),
        "Review parent roots cannot nest",
    )
    # This barrier precedes all reads/hashes of numerical assets, including teachers.
    preflight = {str(parent): _parent_preflight(parent) for parent in parents}
    for item in preflight.values():
        for path, sha in item["values"]["identity"]["evidence_files"].items():
            _checked(dict(path=path, sha256=sha))

    _checked(input_equivalence_binding)
    proof = read_json(input_equivalence_binding["path"])
    _require(
        proof.get("schema") == "badminton-review-formal-input-equivalence-v1"
        and proof.get("state") == "complete"
        and proof.get("training_published") is False
        and proof.get("fitted_tensors_imported") is False
        and proof.get("geometry_fields") == list(GEOMETRY_FIELDS)
        and proof.get("source_fields") == list(SOURCE_FIELDS),
        "A complete actual input-equivalence proof is required",
    )
    source, repair = root / "source", root / "repair"
    registry = read_json(source / "source_registry.json")
    calibration = CalibrationSnapshot.load(source / "calibration/calibration_snapshot.json")
    _require(
        registry.get("schema") == "badminton-formal-derived-source-registry-v1"
        and registry.get("review_only") is not True
        and calibration.data.get("review_only") is not True,
        "Formal import requires separately admitted formal inputs",
    )
    _matches(proof["evidence"]["formal_registry"], source / "source_registry.json")
    _matches(proof["evidence"]["formal_calibration"], source / "calibration/calibration_snapshot.json")
    for entry in proof["evidence"].values():
        _checked(entry)
    role_binding = proof["evidence"]["role_contract"]
    if read_json(role_binding["path"]).get("schema") == POLICY:
        incremental = incremental or policy_from_binding(role_binding)
    if incremental is not None:
        _require(
            proof["evidence"].get("release_policy") == incremental.binding
            and registry.get("release_policy") == incremental.binding
            and set(map(str, parents))
            == {str(Path(item["review_root"]).resolve()) for item in incremental.scope["review_sources"].values()},
            "Formal numeric parents or source registry differ from the exact v2 policy",
        )
    roles = read_json(source / "session_splits.json")
    _require(
        roles == proof["role_assignment"] == read_json(proof["evidence"]["role_contract"]["path"])["new_source_splits"],
        "Formal source roles changed",
    )
    if incremental is None:
        validate_session_splits(roles, registry["sources"])
    else:
        validate_roles(roles, registry["sources"], incremental.required)
        _require(roles == incremental.roles, "Formal numeric source roles differ from the v2 policy")
    _checked(registry["intake"])
    _checked(registry["camera_admission_handoff"])
    intake = read_json(registry["intake"]["path"])
    _require(
        intake.get("schema") == "badminton-formal-derived-intake-v1"
        and intake.get("camera_admission_handoff") == registry["camera_admission_handoff"]
        and intake.get("source_role_contract") == proof["evidence"]["role_contract"],
        "Formal manifest/intake changed",
    )
    _checked(intake["prepared_ranges"])
    if incremental is not None:
        _require(intake.get("release_policy") == incremental.binding, "Formal intake changed its v2 policy")
    plan, body_plan = read_json(repair / "new_source_input.json"), read_json(repair / "final_labels/repair_plan.json")
    _plan_contract(root, plan, body_plan, calibration, registry)
    if incremental is not None:
        _require(
            plan.get("release_policy") == incremental.binding
            and plan.get("required_nonempty_splits") == incremental.required,
            "Formal body input changed its explicit v2 policy/nonempty roles",
        )
    _require(
        plan.get("purpose") == "formal_native_rally_derivation"
        and plan.get("review_only") is not True
        and plan.get("formal_derivation")
        == dict(
            input_equivalence=input_equivalence_binding, camera_admission_handoff=registry["camera_admission_handoff"]
        )
        and plan.get("session_splits") == roles,
        "Formal derivation is not bound to the actual equivalence proof",
    )
    jobs = _jobs(plan, root)
    by_source, parent_evidence, parent_manifests = {}, {}, []
    for parent in parents:
        item = preflight[str(parent)]
        old_registry = item["values"]["source_registry"]
        previous = CalibrationSnapshot.load(item["paths"]["calibration"])
        _plan_contract(parent, item["values"]["source_input"], item["values"]["body_plan"], previous, old_registry)
        validate_formal_geometry(previous.data, calibration.data, old_registry["sources"])
        _checked(old_registry["pilot_manifest"])
        _require(
            read_json(parent / "pilot_input.json")["manifest"] == old_registry["pilot_manifest"],
            "Parent frozen manifest differs from its source registry",
        )
        parent_manifests.append(old_registry["pilot_manifest"])
        for sid, entry in old_registry["sources"].items():
            _require(
                sid not in by_source and sid in registry["sources"], "Parent source coverage is duplicate or extraneous"
            )
            lineage = proof["sources"].get(sid, {})
            _require(
                lineage.get("original_registry") == binding(item["paths"]["source_registry"])
                and lineage.get("original_calibration") == binding(item["paths"]["calibration"])
                and lineage.get("original_index") == entry["source_index"]
                and lineage.get("admitted_index") == registry["sources"][sid]["source_index"]
                and lineage.get("original_inputs_exact") is True
                and lineage.get("cameras_court_ground_bbox_clock_exact") is True,
                "Formal equivalence source lineage differs from the explicit parent",
            )
            for name in ("original_index", "admitted_index", "camera_applicability"):
                _checked(lineage[name])
            original_index = read_json(lineage["original_index"]["path"])
            admitted_index = read_json(lineage["admitted_index"]["path"])
            validate_formal_source(original_index, admitted_index)
            _require(
                lineage.get("source_frames") == admitted_index.get("frame_count") == len(admitted_index["frames"]),
                "Formal equivalence source frame coverage changed",
            )
            by_source[sid] = (parent, previous)
        parent_evidence[str(parent)] = {name: binding(path) for name, path in item["paths"].items()}
    _require(
        set(by_source) == set(registry["sources"]) == set(proof["sources"]), "Parents do not cover all formal sources"
    )
    _require(
        sorted(intake["parent_review_manifests"], key=lambda e: e["path"])
        == sorted(parent_manifests, key=lambda e: e["path"]),
        "Formal intake binds different review manifests",
    )
    expected_keys = {key for item in preflight.values() for key in item["jobs"]}
    _require(set(jobs) == expected_keys, "Formal windows differ from complete parent job coverage")
    current_digest = BadmintonPipeline._fit_digest(None)
    original_commits = {str(Path(p).resolve()): value for p, value in (original_commits or {}).items()}
    algorithms, results, teacher_files, sessions, clouds = {}, {}, {}, {}, {}
    resolver = load_pilot_run_relocations(source, calibration.data)
    for key, target_initial in jobs.items():
        sid, name = key.split(":", 1)
        parent, previous = by_source[sid]
        item = preflight[str(parent)]
        original_initial = item["jobs"][key]
        original = parent / "repair/final_labels/sessions" / sid / "windows" / name
        target = root / "repair/final_labels/sessions" / sid / "windows" / name
        before, after = read_json(original_initial / "window.json"), read_json(target_initial / "window.json")
        _require(
            before.get("calibration_digest") == previous.data["snapshot_digest"]
            and after.get("calibration_digest") == calibration.data["snapshot_digest"]
            and {k: v for k, v in before.items() if k != "calibration_digest"}
            == {k: v for k, v in after.items() if k != "calibration_digest"},
            "Formal initial window changed numerical inputs",
        )
        report_path = original / "fit_report.json"
        native = item["values"]["native"]["windows"][key]
        _require(
            file_sha(report_path) == native["body"]["report_sha256"],
            "Original fit report differs from native verification",
        )
        report = read_json(report_path)
        _require(
            report.get("session_id") == sid
            and report.get("window") == name
            and report.get("fitting_config") == body_plan["fitting_config"],
            "Original report fitting configuration changed",
        )
        body_window = read_json(original / "window.json")
        ray_policy = file_sha(PROJ_ROOT / "hmr4d/lidar_mocap/pose_ray_filter.py")
        expected_settings = dict(
            after["settings"], share_track_shape=True, native_pose_joint_ray_policy_sha256=ray_policy
        )
        settings = report.get("settings", {})
        _require(
            body_window.get("settings") == settings
            and settings.get("native_pose_joint_ray_policy_sha256") == ray_policy
            and settings.get("native_pose_receipt_sha256") == parent_evidence[str(parent)]["pose"]["sha256"]
            and {k: v for k, v in settings.items() if k != "native_pose_receipt_sha256"} == expected_settings
            and all(
                body_window.get(k) == before[k]
                for k in ("session_id", "window_id", "left_us", "right_us", "source_pairs", "calibration_digest")
            ),
            "Original body fitting settings or joint-ray policy changed",
        )
        digest = report["fit_algorithm_digest"]
        algorithm_key = str(parent) + ":" + digest
        if algorithm_key not in algorithms:
            algorithms[algorithm_key] = audit_algorithm_compatibility(
                digest, current_digest, original_commits.get(str(parent))
            )
        artifacts = report.get("artifacts_sha256", {})
        _require(
            {"quality.jsonl", "ptv3.metadata.jsonl", "ptv3.pth"} <= set(artifacts),
            "Incomplete fitted artifact inventory",
        )
        for relative, expected in artifacts.items():
            _checked(dict(path=str(_confined(original / relative, original)), sha256=expected))
        rows = read_rows(original / "quality.jsonl")
        accepted = [row for row in rows if row.get("accepted") is True]
        _require(
            len(rows) == report["fitted"]
            and len(accepted) == report["accepted"] == native["body"]["accepted_checked"]
            and read_rows(original / "ptv3.metadata.jsonl") == accepted,
            "Original fitted/accepted quality rows disagree",
        )
        _require(
            report.get("native_pose_receipt_sha256") == parent_evidence[str(parent)]["pose"]["sha256"],
            "Original report native pose receipt changed",
        )
        teacher_files.update(_teacher_inventory(item, key, original_initial, original))
        _require(
            report.get("teacher_provenance_hash") == read_json(original / "teacher/manifest.json")["provenance_hash"],
            "Original report teacher provenance changed",
        )
        if sid not in sessions:
            sessions[sid] = load_run_session(source, calibration, sid, relocations=resolver)
        session = sessions[sid]
        pairs, selected = session.pairs(), set(after["source_pairs"])
        _require(report.get("scan_count") == len(selected), "Original report scan coverage changed")
        for scan in selected:
            path = pairs[scan]["lidar_path"]
            _require(path in report["source_lidar_sha256"], "Original report omitted a selected source point cloud")
        for path, expected in report["source_lidar_sha256"].items():
            if path not in clouds:
                clouds[path] = file_sha(path)
            _require(clouds[path] == expected, "Original source point-cloud bytes changed")
        for row in rows:
            scan = row["lidar_index"]
            _require(
                row.get("session_id") == sid
                and scan in selected
                and row["scan_mid_us"] == pairs[scan]["scan_mid_us"]
                and row["cameras"] == pairs[scan]["cameras"]
                and row["calibration_digest"] == previous.data["snapshot_digest"],
                "Reused scan/native/camera scope changed",
            )
            fit = _confined(row["fit_path"], original)
            _require(
                fit.relative_to(original).as_posix() in artifacts, "Fitted row references an unbound numeric asset"
            )
            if row["accepted"]:
                _require(
                    not acceptance_reasons(row["qa"], row["source_point_count"]) and row.get("rejection_reasons") == [],
                    "Unchanged body QA rejects an admitted row",
                )
            for path in (row["lidar_path"], pairs[scan]["lidar_path"]):
                if path not in clouds:
                    clouds[path] = file_sha(path)
            _require(
                clouds[row["lidar_path"]]
                == report["source_lidar_sha256"].get(row["lidar_path"])
                == clouds[pairs[scan]["lidar_path"]],
                "Original source point-cloud bytes changed",
            )
            pose_files = row.get("native_pose_evidence", {}).get("files", {})
            _require(
                bool(pose_files) and set(pose_files) <= {"cam1", "cam2"},
                "Original row lacks exact native pose evidence",
            )
            for camera, entry in pose_files.items():
                path = _confined(entry["path"], parent / "repair/native_poses" / sid / camera)
                _checked(entry)
                record = read_json(path)
                index = pairs[scan]["cameras"][camera]["record_index"]
                _require(
                    record.get("source_id") == sid
                    and record.get("camera") == camera
                    and record.get("config_digest") == item["values"]["pose"]["config_digest"]
                    and record.get("native") == session.camera_rows(camera)[index],
                    "Original exact native pose evidence changed",
                )
        results[key] = dict(
            parent_root=str(parent),
            source_id=sid,
            original_window=str(original),
            original_initial_window=str(original_initial),
            target_window=str(target),
            target_initial_window=str(target_initial),
            original_report=binding(report_path),
            fitted=report["fitted"],
            accepted=report["accepted"],
            original_fit_algorithm_digest=digest,
            current_fit_algorithm_digest=current_digest,
        )
    if incremental is not None:
        incremental.recheck()
    return dict(
        schema="badminton-formal-numeric-import-audit-v1",
        state="eligible_parent_complete_not_imported",
        formal_root=str(root),
        source_input=binding(repair / "new_source_input.json"),
        body_plan=binding(repair / "final_labels/repair_plan.json"),
        input_equivalence=dict(input_equivalence_binding),
        formal_registry=binding(source / "source_registry.json"),
        formal_calibration=binding(source / "calibration/calibration_snapshot.json"),
        parent_evidence=parent_evidence,
        windows=results,
        algorithms=algorithms,
        teacher_files=teacher_files,
        fitted_tensors_imported=False,
        training_published=False,
        eligible_windows=len(results),
        eligible_fitted=sum(v["fitted"] for v in results.values()),
        eligible_accepted=sum(v["accepted"] for v in results.values()),
    )
