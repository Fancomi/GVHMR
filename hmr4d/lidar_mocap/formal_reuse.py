"""Prove review/admitted input equivalence without altering either frozen run."""

from pathlib import Path
import time


GEOMETRY_FIELDS = (
    "cameras", "Q_raw_to_viewer", "reference_plane", "bbox", "court_registration", "court",
    "timing_policy", "geometry_variant", "bbox_rotation_convention", "bbox_geometry",
    "coordinate_convention", "physical_limits", "Q_rounding",
)
SESSION_FIELDS = ("manifest_sha256", "frame_count", "frame_map_sha256", "source_clock_audit")
SOURCE_FIELDS = (
    "source_id", "source_kind", "frame_count", "fixed24", "frames", "source_manifest",
    "common_clock", "frame_map", "native_videos", "canonical_video_sidecars", "video_sidecars",
    "video_index_contract", "parent_source_index_identity", "parent_inputs", "candidate_bundle_sha256",
)


def validate_formal_geometry(previous, current, source_ids):
    """This comparison never grants admission; the caller must verify real certificates."""
    if previous.get("review_only") is not True or current.get("review_only") is True:
        raise ValueError("Require a frozen review input and a separately admitted formal snapshot")
    if (not source_ids or set(source_ids) != set(previous["sessions"])
            or not set(source_ids) <= set(current["sessions"])):
        raise ValueError("Review/formal source scope differs")
    for name in GEOMETRY_FIELDS:
        if name not in previous or previous[name] != current.get(name):
            raise ValueError("Formal reuse changed physical calibration input: " + name)
    for sid in source_ids:
        for name in SESSION_FIELDS:
            before, after = previous["sessions"][sid], current["sessions"][sid]
            if name not in before or before[name] != after.get(name):
                raise ValueError("Formal reuse changed original native session contract: " + sid + ":" + name)


def validate_formal_source(previous, current):
    """Compare full source/clock/exposure/member declarations across admission metadata."""
    for name in SOURCE_FIELDS:
        if previous.get(name) != current.get(name):
            raise ValueError("Formal reuse changed original source contract: " + name)
    for name in ("M_baked_into_stored_xyz", "application_raw_to_viewer_M", "source_requested_M"):
        if name not in previous["coordinate_contract"] or (
                previous["coordinate_contract"][name] != current["coordinate_contract"].get(name)):
            raise ValueError("Formal reuse changed source coordinate contract: " + name)
    for name in ("bbox_sha256", "application_raw_to_viewer_M", "parent_source_index_sha256",
                 "parent_source_manifest_sha256", "candidate_bundle_sha256"):
        if previous.get("crop_binding", {}).get(name) != current.get("crop_binding", {}).get(name):
            raise ValueError("Formal reuse changed original crop lineage: " + name)


def audit_formal_inputs(review_roots, formal_source_root, role_contract_path, *,
                        data_relocations=None, data_relocations_sha256=None):
    from .calibration import CalibrationSnapshot
    from .expanded_dataset import binding, check_binding
    from .rally_pipeline import load_run_session, read_json, validate_session_splits
    from .relocations import load_pilot_run_relocations, load_run_relocations, validate_run_relocations

    started = time.perf_counter()
    formal = Path(formal_source_root).resolve()
    registry_path = formal / "source_registry.json"
    registry = read_json(registry_path)
    calibration_path = formal / "calibration/calibration_snapshot.json"
    calibration = CalibrationSnapshot.load(calibration_path)
    if registry.get("review_only") is True or calibration.data.get("review_only") is True:
        raise ValueError("Actual formal source registry and snapshot are required")
    roles = read_json(role_contract_path)["new_source_splits"]
    validate_session_splits(roles, registry["sources"])
    if read_json(formal / "session_splits.json") != roles:
        raise ValueError("Formal role assignment differs from the frozen whole-source draft")
    resolver = load_run_relocations(formal, data_relocations, data_relocations_sha256)
    validate_run_relocations(registry, calibration.data, resolver)
    evidence = dict(formal_registry=binding(registry_path), formal_calibration=binding(calibration_path),
                    role_contract=binding(role_contract_path))
    if resolver is not None:
        evidence["formal_data_relocations"] = binding(resolver.root / resolver.binding["file"])
    results = {}
    for root in review_roots:
        root = Path(root).resolve()
        source = root / "source"
        old_registry_path = source / "source_registry.json"
        old_registry = read_json(old_registry_path)
        old_path = source / "calibration/calibration_snapshot.json"
        old_calibration = CalibrationSnapshot.load(old_path)
        if old_registry.get("review_only") is not True:
            raise ValueError("Original review source registry is required")
        validate_formal_geometry(old_calibration.data, calibration.data, old_registry["sources"])
        old_resolver = load_pilot_run_relocations(source, old_calibration.data)
        for sid, original in old_registry["sources"].items():
            if sid in results:
                raise ValueError("Two review roots repeat a source")
            check_binding(original["source_index"])
            current = registry["sources"][sid]
            check_binding(current["source_index"])
            old_session = load_run_session(source, old_calibration, sid, relocations=old_resolver)
            session = load_run_session(formal, calibration, sid, relocations=resolver)
            validate_formal_source(old_session.source_index, session.source_index)
            if (session.source_index.get("accepted_bundle_sha256") != calibration.data["configuration_files_sha256"]
                    or session.source_index.get("accepted_calibration_sha256")
                    != calibration.data["configuration_files_sha256"]["calibration.json"]):
                raise ValueError("Formal source certificate uses another actual camera bundle")
            # load_run_session -> load_source_index(require_admitted=True) already
            # verifies the complete camera certificate, real evidence and source bindings.
            if not session.camera_applicability:
                raise ValueError("Formal source has no verified real camera applicability certificate")
            certificate = dict(path=session.camera_applicability["file"],
                               sha256=session.camera_applicability["sha256"])
            results[sid] = dict(original_registry=binding(old_registry_path),
                original_calibration=binding(old_path),
                original_index=original["source_index"], admitted_index=current["source_index"],
                camera_applicability=certificate, source_frames=len(session.source_index["frames"]),
                original_inputs_exact=True, cameras_court_ground_bbox_clock_exact=True)
    if set(results) != set(registry["sources"]):
        raise ValueError("Review evidence does not cover every actual formal source")
    for item in evidence.values():
        check_binding(item)
    return dict(schema="badminton-review-formal-input-equivalence-v1", state="complete", evidence=evidence,
        sources=results, geometry_fields=list(GEOMETRY_FIELDS), source_fields=list(SOURCE_FIELDS),
        role_assignment=roles, training_published=False, fitted_tensors_imported=False,
        full_native_rgb_pixels_redecoded=False, report_seconds=time.perf_counter() - started,
        scope="Input/camera-admission equivalence only; fit/QA/identity/member imports require separate proof")
