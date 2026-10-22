"""Bounded review-only rallies using the existing native v2 fit and viewers."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from hmr4d import PROJ_ROOT
from .calibration import CalibrationSnapshot, _CONFIG_FILES
from .dual_view import file_sha
from .new_source_repair import prepare_new_sources, write_frozen
from .pipeline import write_json
from .rally_pipeline import load_run_session, read_json
from .repair_execution import require_gpu_lease
from .relocations import load_manifest_relocations, load_pilot_run_relocations
from .source_index import _bound_file


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=file_sha(path))


def check_binding(entry):
    if file_sha(entry["path"]) != entry["sha256"]:
        raise ValueError("Pilot input changed: " + entry["path"])


def validate_pilot_spec(spec):
    return _validate_review_rally_spec(spec, "badminton-review-pilot-v1", maximum_seconds=180)


def validate_full_native_spec(spec):
    """Full execution uses its own frozen scope; the short pilot stays bounded."""
    for name in ("source_indexes", "excluded_source_indexes"):
        entries = spec.get(name, {})
        if not isinstance(entries, dict):
            raise ValueError(f"Full native {name} must be a source-to-binding mapping")
        for sid, entry in entries.items():
            if (not isinstance(entry, dict) or not isinstance(entry.get("file"), str)
                    or not entry["file"].strip() or not isinstance(entry.get("sha256"), str)
                    or not entry["sha256"].strip()):
                raise ValueError(
                    f"Full native {name}[{sid}] requires a file/sha256 binding relative to data_root; "
                    "path/sha256 receipt bindings cannot be used as source indexes"
                )
    if (
        spec.get("execution_scope") != "all_effective_native_rallies"
        or not isinstance(spec.get("user_confirmation"), str)
        or not spec["user_confirmation"].strip()
        or not isinstance(spec.get("coverage_audit"), dict)
        or not spec["coverage_audit"].get("path")
    ):
        raise ValueError("Full native execution requires its actual user scope and bound coverage audit")
    check_binding(spec["coverage_audit"])
    scope = _validate_review_rally_spec(spec, "badminton-native-rally-fit-v1", maximum_seconds=None)
    if set(scope["sources"]) != set(spec["source_indexes"]):
        raise ValueError("Full native scope must account for every declared source index")
    excluded, no_fit = spec.get("excluded_source_indexes", {}), spec.get("no_fit_sources", {})
    if (not isinstance(excluded, dict) or not isinstance(no_fit, dict) or set(excluded) != set(no_fit)
            or set(excluded) & set(spec["source_indexes"])):
        raise ValueError("No-fit sources require disjoint explicit index/evidence bindings")
    for sid, entry in no_fit.items():
        proof = validate_no_fit_source(entry, source_id=sid)
        if proof["source_index_sha256"] != excluded[sid]["sha256"]:
            raise ValueError("No-fit declaration refers to another original source index")
    segments = set()
    for clip in spec["clips"]:
        number, segment = clip.get("rally_index"), clip.get("segment_index", 0)
        if ("contiguous_windows" in clip or type(number) is not int or number < 0
                or type(segment) is not int or segment < 0):
            raise ValueError("Use flat native rally clips with the original rally_index and nonnegative segment_index")
        key = (clip["source_id"], number, segment)
        if key in segments:
            raise ValueError("A native rally segment is declared twice")
        segments.add(key)
    if excluded:
        scope["excluded_no_fit_sources"] = sorted(excluded)
    return scope


def validate_no_fit_source(entry, *, source_id=None):
    """No-fit receipts close real native coverage without inventing an empty clip."""
    check_binding(entry)
    proof = read_json(entry["path"])
    sid = source_id or proof.get("source_id")
    if (proof.get("schema") != "badminton-native-no-fit-needed-v1" or proof.get("status") != "no_fit_needed"
            or not isinstance(sid, str) or proof.get("source_id") != sid
            or type(proof.get("eligible_paired_scan_count")) is not int or proof["eligible_paired_scan_count"] != 0
            or not isinstance(proof.get("reason"), str) or not proof["reason"].strip()):
        raise ValueError("No-fit receipt must explicitly close zero actual eligible scans for its source")
    check_binding(proof["source_index"])
    check_binding(proof["coverage_audit"])
    count = read_json(proof["coverage_audit"]["path"]).get("assigned_unique_scans")
    if (proof["source_index_sha256"] != proof["source_index"]["sha256"]
            or proof.get("coverage_audit_sha256", proof["coverage_audit"]["sha256"]) != proof["coverage_audit"]["sha256"]
            or type(count) is not int or count != 0):
        raise ValueError("No-fit source index/coverage differs from its actual zero-scan evidence")
    index = read_json(proof["source_index"]["path"])
    if index.get("source_kind") != "derived_native_roi_fixed24" or index.get("source_id") != sid:
        raise ValueError("No-fit source needs its actual completed native crop index")
    documents, entries = [], proof.get("evidence_files", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("No-fit source requires explicit native evidence file bindings")
    for item in entries:
        check_binding(item)
        if Path(item["path"]).suffix.lower() == ".json":
            documents.append(read_json(item["path"]))
    for table in documents:
        if not isinstance(table, dict) or table.get("schema") != "badminton-analysis-clips-v1" or table.get("source_id") != sid:
            continue
        for receipt in documents:
            if (isinstance(receipt, dict) and receipt.get("source_id") == sid and receipt.get("status") == "completed"
                    and type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0
                    and receipt.get("source_manifest_sha256") == index["source_manifest"]["sha256"]
                    and receipt.get("analysis_sha256") == table["analysis_sha256"]
                    and receipt.get("planner_sha256") == table.get("planner_sha256")
                    and any(item["sha256"] == table["analysis_sha256"] for item in entries)):
                return proof
    raise ValueError("No-fit source lacks completed actual native analysis/planner evidence")


def _validate_review_rally_spec(spec, schema, *, maximum_seconds):
    if (
        spec.get("schema") != schema
        or spec.get("review_only") is not True
        or spec.get("training_published") is not False
        or not spec.get("camera_label")
    ):
        raise ValueError("Select an explicitly frozen review-only camera pilot")
    if set(spec.get("bundle_sha256", {})) != set(_CONFIG_FILES):
        raise ValueError("Pilot must bind all five frozen calibration files")
    if not isinstance(spec.get("review_authorization"), dict) or not spec["review_authorization"].get("path"):
        raise ValueError("Pilot must bind the existing review-only camera authorization")
    clips = spec.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("Pilot needs actual native-confirmed rally clips")
    intervals, ids = {}, set()
    for clip in clips:
        sid = clip.get("source_id")
        if not isinstance(sid, str) or sid not in spec.get("source_indexes", {}):
            raise ValueError("Pilot clip has no bound source index")
        left, right = clip.get("begin_common_us"), clip.get("end_common_us")
        if type(left) is not int or type(right) is not int or right <= left:
            raise ValueError("Pilot needs exact increasing integer common-clock intervals")
        if clip.get("selection_status") != "native_rally_confirmed" or not clip.get("evidence_files"):
            raise ValueError("Human presence alone does not confirm a native rally")
        values = clip.get("eligible_paired_png_indices")
        if (
            not isinstance(values, list)
            or not values
            or any(type(v) is not int or v < 0 for v in values)
            or any(b <= a for a, b in zip(values, values[1:]))
        ):
            raise ValueError("Pilot needs ordered unique actual paired PNG ordinals")
        local = {(sid, v) for v in values}
        if ids & local:
            raise ValueError("Pilot clips duplicate an actual source scan")
        ids.update(local)
        for old_left, old_right in intervals.setdefault(sid, []):
            if max(left, old_left) < min(right, old_right):
                raise ValueError("Pilot source rally intervals overlap")
        intervals[sid].append((left, right))
    seconds = sum(b - a for ranges in intervals.values() for a, b in ranges) / 1e6
    if maximum_seconds is not None and seconds > maximum_seconds:
        raise ValueError("This short pilot is bounded to at most 180 selected seconds")
    return dict(sources=sorted(intervals), selected_seconds=seconds, selected_scans=len(ids))


def validate_camera_review(spec):
    entry = spec["review_authorization"]
    check_binding(entry)
    review = read_json(entry["path"])
    if (
        review.get("schema") != "badminton-calibration-review-status-v1"
        or review.get("baseline") != spec["camera_label"]
        or review.get("usage") != "review_only"
        or review.get("camera_pilot_authorized") is not True
        or review.get("body_labels_production_admitted") is not False
        or review.get("training_published") is not False
        or review.get("result_files") != spec["bundle_sha256"]
        or not review.get("user_confirmation")
    ):
        raise ValueError("Camera pilot authorization does not bind this exact review-only bundle")
    return review


def validate_native_rally(clip, source_manifest_sha256):
    """Bind selected PNG ordinals to actual native results for this exact crop."""
    files = clip["evidence_files"]
    docs = []
    for item in files:
        check_binding(item)
        if Path(item["path"]).suffix.lower() == ".json":
            document = read_json(item["path"])
            if isinstance(document, dict):
                docs.append(document)
    for table in docs:
        if table.get("schema") != "badminton-analysis-clips-v1" or table.get("source_id") != clip["source_id"]:
            continue
        found = [
            r for r in table["clips"] if r["index"] == clip.get("rally_index") and r.get("kind") == "badminton_rally"
        ]
        if len(found) != 1:
            continue
        native = found[0]
        left, right = native["source_begin_frame"], native["source_begin_frame"] + native["source_frame_count"]
        if any(not left <= i < right for i in clip["eligible_paired_png_indices"]):
            raise ValueError("Pilot selected scan leaves its actual native rally")
        sha = table["analysis_sha256"]
        receipts = [
            d
            for d in docs
            if d.get("source_id") == clip["source_id"]
            and d.get("status") == "completed"
            and type(d.get("exit_code")) is int
            and d["exit_code"] == 0
            and d.get("source_manifest_sha256") == source_manifest_sha256
            and d.get("analysis_sha256") == sha
            and d.get("planner_sha256") == table.get("planner_sha256")
        ]
        if receipts and any(item["sha256"] == sha for item in files):
            return
    raise ValueError("Pilot requires completed native rally/analysis evidence bound to this exact newly cropped source")


def prepare_full_native_rallies(manifest_path, output_root):
    return prepare_pilot(manifest_path, output_root, _full_native=True)


def prepare_pilot(manifest_path, output_root, *, _full_native=False):
    started = time.perf_counter()
    manifest_path, root = Path(manifest_path).resolve(), Path(output_root).resolve()
    if not root.is_relative_to(PROJ_ROOT / "outputs") or root == PROJ_ROOT / "outputs":
        raise ValueError("Use a separate explicit outputs/ pilot root")
    spec = read_json(manifest_path)
    scope = validate_full_native_spec(spec) if _full_native else validate_pilot_spec(spec)
    relocations = load_manifest_relocations(spec)
    validate_camera_review(spec)
    original = binding(manifest_path)
    data_root = Path(spec["data_root"]).resolve()
    cal_dir = Path(spec["calibration_dir"]).resolve()
    for name, digest in spec["bundle_sha256"].items():
        if file_sha(cal_dir / name) != digest:
            raise ValueError("Frozen pilot calibration changed: " + name)
    for clip in spec["clips"]:
        for item in clip["evidence_files"]:
            check_binding(item)
    indexes = {sid: spec["source_indexes"][sid] for sid in scope["sources"]}
    source = root / "source"
    frozen = CalibrationSnapshot.freeze(
        spec["lidar_project_root"],
        data_root / "sessions",
        source / "calibration",
        calibration_dir=cal_dir,
        source_indexes=indexes,
        data_root=data_root,
        review_source_ids=scope["sources"],
        review_authorization=spec["review_authorization"],
        relocations=relocations,
    )
    if frozen.data["configuration_files_sha256"] != spec["bundle_sha256"]:
        raise ValueError("Pilot snapshot and declared camera bundle differ")
    sessions, rows, state = {}, [], {}
    registry = dict(
        schema="badminton-rally-run-source-registry-v1",
        review_only=True,
        training_published=False,
        pilot_manifest=original,
        data_root=str(data_root),
        sources={},
    )
    if _full_native:
        registry.update(execution_scope=spec["execution_scope"], coverage_audit=spec["coverage_audit"])
        if spec.get("excluded_source_indexes"):
            registry.update(excluded_source_indexes=spec["excluded_source_indexes"], no_fit_sources=spec["no_fit_sources"])
    if relocations is not None:
        registry["data_relocations"] = dict(relocations.binding)
    for sid, entry in indexes.items():
        path = _bound_file(data_root, entry, relocations=relocations)
        index = read_json(path)
        if (
            index.get("source_kind") != "derived_native_roi_fixed24"
            or index.get("crop_binding", {}).get("bbox_sha256") != spec["bundle_sha256"]["bbox.json"]
        ):
            raise ValueError("Pilot requires the newly bound crop for this frozen bbox")
        registry["sources"][sid] = dict(
            baseline=False,
            source_index=binding(path),
            source_manifest_sha256=index["source_manifest"]["sha256"],
            common_clock_sha256=index["common_clock"]["sha256"],
            frame_map_sha256=index["frame_map"]["sha256"],
        )
    # Explicit review scope consumes the frozen candidate camera/crop binding;
    # original formal camera/body admission flags remain unchanged.
    for sid, entry in registry["sources"].items():
        from .source_index import SourceIndexedSession

        sessions[sid] = SourceIndexedSession.load_source_index(
            entry["source_index"]["path"],
            data_root,
            expected_sha256=entry["source_index"]["sha256"],
            require_admitted=False,
            relocations=relocations,
        )
    for number, clip in enumerate(spec["clips"]):
        sid = clip["source_id"]
        validate_native_rally(clip, registry["sources"][sid]["source_manifest_sha256"])
        pairs = sessions[sid].pairs()
        chosen = []
        for ordinal in clip["eligible_paired_png_indices"]:
            if ordinal >= len(pairs):
                raise ValueError("Pilot scan ordinal is outside the actual new source")
            pair = pairs[ordinal]
            if pair["status"] != "paired" or not clip["begin_common_us"] <= pair["scan_mid_us"] < clip["end_common_us"]:
                raise ValueError("Pilot scan lacks paired native views or leaves its confirmed rally interval")
            chosen.append(pair)
        rows.append(
            dict(
                source_id=sid,
                source_rally_index=number,
                review_index=number,
                original_rally_index=clip.get("rally_index", number),
                begin_common_us=clip["begin_common_us"],
                end_common_us=clip["end_common_us"],
                first_scan_mid_us=chosen[0]["scan_mid_us"],
                last_scan_mid_us=chosen[-1]["scan_mid_us"],
                source_begin_frame=chosen[0]["lidar_index"],
                source_frame_count=len(chosen),
                eligible_paired_png_indices=[p["lidar_index"] for p in chosen],
                review_only=True,
                evidence_files=clip["evidence_files"],
            )
        )
        if "segment_index" in clip:
            rows[-1]["segment_index"] = clip["segment_index"]
    source.mkdir(parents=True, exist_ok=True)
    ranges_path = source / "pilot_ranges.jsonl"
    ranges_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if ranges_path.exists() and ranges_path.read_text(encoding="utf-8") != ranges_text:
        raise ValueError("Pilot selection changed; use a new output root")
    ranges_path.write_text(ranges_text, encoding="utf-8")
    sources = {
        sid: dict(
            rallies=sum(r["source_id"] == sid for r in rows),
            paired_unique_png_frames=sum(len(r["eligible_paired_png_indices"]) for r in rows if r["source_id"] == sid),
        )
        for sid in scope["sources"]
    }
    intake_path = source / "pilot_intake.json"
    intake = dict(
        schema="badminton-review-rally-intake-v1",
        review_only=True,
        training_published=False,
        pilot_manifest=original,
        prepared_ranges=binding(ranges_path),
        sources=sources,
        rallies=len(rows),
        effective_window_minutes=scope["selected_seconds"] / 60,
    )
    if relocations is not None:
        intake["data_relocations"] = dict(relocations.binding)
    write_frozen(intake_path, intake)
    registry["intake"] = binding(intake_path)
    write_frozen(source / "source_registry.json", registry)
    write_frozen(source / "session_splits.json", dict(review=scope["sources"]))
    state = dict(
        schema=spec["schema"],
        review_only=True,
        training_published=False,
        stage="prepared",
        camera_label=spec["camera_label"],
        manifest=original,
        source_root=str(source),
        calibration_digest=frozen.data["snapshot_digest"],
        selection=scope,
        prepare_seconds=time.perf_counter() - started,
    )
    if relocations is not None:
        state["data_relocations"] = dict(relocations.binding)
    write_frozen(root / "pilot_input.json", dict(state, prepare_seconds=None, stage="frozen_input"))
    plan = prepare_new_sources(source, root / "repair", relocations=relocations)
    state["windows"] = len(plan["jobs"])
    for sid in scope["sources"]:
        state[sid] = dict(state="prepared_review_only", baseline=False)
    write_json(root / "pilot_progress.json", state)
    check_binding(original)
    return state


def require_review_group(splits, sources):
    if (
        not isinstance(splits, dict)
        or set(splits) != {"review"}
        or not isinstance(splits["review"], list)
        or not splits["review"]
        or len(set(splits["review"])) != len(splits["review"])
        or set(splits["review"]) != set(sources)
    ):
        raise ValueError("Pilot bodies require exactly their explicit review-only source group")
    return splits


def review_rally_windows(pairs, ranges):
    """Pilot teachers use each confirmed rally's bounded context, not idle minutes."""
    known = {pair["lidar_index"]: pair for pair in pairs}
    paired = [pair["scan_mid_us"] for pair in pairs if pair["status"] == "paired"]
    if not paired:
        raise ValueError("Review source has no paired scans")
    base = min(paired)
    windows = []
    seen = set()
    for row in ranges:
        left, right = row["begin_common_us"], row["end_common_us"]
        chosen = [known[i] for i in row["eligible_paired_png_indices"]]
        if any(pair["status"] != "paired" or not left <= pair["scan_mid_us"] < right for pair in chosen):
            raise ValueError("Review rally window changed its exact paired coverage")
        ids = {pair["lidar_index"] for pair in chosen}
        if not chosen or seen & ids:
            raise ValueError("Review rallies are empty or duplicate scans")
        seen.update(ids)
        name = f"{left-base:012d}_{right-base:012d}"
        windows.append((name, left, right, chosen))
    return windows


def run_pilot_stage(root, stage, *, gpu_lease=None, threads=1):
    root = Path(root).resolve()
    if not root.is_relative_to(PROJ_ROOT / "outputs") or root == PROJ_ROOT / "outputs":
        raise ValueError("Use the explicit prepared outputs/ pilot root")
    frozen = read_json(root / "pilot_input.json")
    check_binding(frozen["manifest"])
    source, repair = root / "source", root / "repair"
    plan = read_json(repair / "new_source_input.json")
    relocations = load_pilot_run_relocations(source)
    if frozen.get("data_relocations") != (dict(relocations.binding) if relocations is not None else None):
        raise ValueError("Frozen pilot input changed its explicit relocation binding")
    if plan.get("data_relocations") != (dict(relocations.binding) if relocations is not None else None):
        raise ValueError("Prepared pilot plan changed its explicit relocation binding")
    if plan.get("review_only") is not True:
        raise ValueError("Selected run is not a review-only pilot")
    labels = repair / "final_labels"
    if stage in ("teacher", "poses", "fit"):
        if gpu_lease is None:
            raise ValueError("Pilot inference requires an explicit handed-over GPU lease")
        require_gpu_lease(gpu_lease)
    python = [sys.executable, "-u"]
    common = ["--root", str(repair), "--source-root", str(source), "--threads", str(threads)]
    commands = []
    if stage == "teacher":
        commands.append(
            (
                "teacher_cam2",
                python
                + [str(PROJ_ROOT / "tools/run_badminton_initial_teachers.py"), *common, "--gpu-lease", str(gpu_lease)],
            )
        )
        for job in plan["jobs"]:
            commands.append(
                (
                    "teacher_cam1_" + job["source_id"] + "_" + Path(job["source_window"]).name,
                    python
                    + [
                        str(PROJ_ROOT / "tools/prepare_badminton_new_source_repair.py"),
                        *common,
                        "--stage",
                        "teacher",
                        "--source",
                        job["source_id"],
                        "--window=" + Path(job["source_window"]).name,
                        "--camera",
                        "cam1",
                    ],
                )
            )
    elif stage == "poses":
        commands.append(
            ("native_dual_pose", python + [str(PROJ_ROOT / "tools/observe_badminton_repair_poses.py"), *common])
        )
    elif stage == "identity":
        commands.append(
            (
                "identity_inventory",
                python + [str(PROJ_ROOT / "tools/prepare_badminton_identity_review.py"), "--root", str(repair)],
            )
        )
    elif stage == "fit":
        dual = read_json(root / "dual_teacher_receipt.json")
        if dual.get("complete") is not True or dual["source_input_sha256"] != file_sha(
            repair / "new_source_input.json"
        ):
            raise ValueError("Both independently executed camera teachers are required")
        for window in dual["windows"].values():
            for item in window.values():
                check_binding(item)
        decisions = repair / "identity_evidence/identity_decisions.json"
        identity = read_json(decisions)
        if identity.get("review_status") != "reviewed":
            raise ValueError("Pilot identity decisions require actual native review")
        commands.append(
            (
                "native_v2_fit",
                python
                + [
                    str(PROJ_ROOT / "tools/repair_badminton_body_labels.py"),
                    "--union",
                    str(repair / "new_source_input.json"),
                    "--source-root",
                    str(source),
                    "--output-root",
                    str(labels),
                    "--native-poses-root",
                    str(repair),
                    "--joint-ray-filter",
                    "--identity-decisions",
                    str(decisions),
                    "--distal-pose-weight",
                    "0.03",
                    "--lower-limb-reprojection-weight",
                    "2",
                    "--threads",
                    str(threads),
                ],
            )
        )
    elif stage == "verify":
        commands.append(
            (
                "native_member_smpl_verification",
                python
                + [
                    str(PROJ_ROOT / "tools/verify_badminton_native_repair.py"),
                    "--root",
                    str(repair),
                    "--threads",
                    str(threads),
                ],
            )
        )
        commands.append(
            (
                "native_quality_report",
                python + [str(PROJ_ROOT / "tools/report_badminton_pilot.py"), "--root", str(root)],
            )
        )
    else:
        raise ValueError("Unknown bounded pilot stage")
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    timings_path = root / "stage_timings.json"
    timings = read_json(timings_path) if timings_path.exists() else {}
    progress = read_json(root / "pilot_progress.json")
    progress.update(stage=stage + "_running", supervisor_pid=os.getpid(), review_only=True, training_published=False)
    write_json(root / "pilot_progress.json", progress)
    for name, command in commands:
        if stage in ("teacher", "poses", "fit"):
            require_gpu_lease(gpu_lease)
        started = time.perf_counter()
        started_utc = datetime.now(timezone.utc).isoformat()
        with (logs / (name + ".log")).open("ab") as log:
            worker = subprocess.Popen(
                command,
                cwd=PROJ_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            progress.update(active_command=name, worker_pid=worker.pid, log=str(logs / (name + ".log")))
            write_json(root / "pilot_progress.json", progress)
            exit_code = worker.wait()
        timings[name] = dict(
            seconds=time.perf_counter() - started,
            exit_code=exit_code,
            started_utc=started_utc,
            log=str(logs / (name + ".log")),
            completed_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(timings_path, timings)
        if exit_code:
            progress.update(stage="failed", failed_stage=stage, failed_command=name, exit_code=exit_code)
            write_json(root / "pilot_progress.json", progress)
            raise RuntimeError("Pilot stage failed; see " + str(logs / (name + ".log")))
    if stage == "teacher":
        windows = {}
        for job in plan["jobs"]:
            sid, name = job["source_id"], Path(job["source_window"]).name
            primary = Path(job["source_window"]) / "teacher/manifest.json"
            secondary = repair / "secondary_teachers/cam1/sessions" / sid / "windows" / name / "teacher/manifest.json"
            windows[sid + ":" + name] = dict(cam1=binding(secondary), cam2=binding(primary))
        write_json(
            root / "dual_teacher_receipt.json",
            dict(
                complete=True,
                windows=windows,
                source_input_sha256=file_sha(repair / "new_source_input.json"),
                cameras=["cam1", "cam2"],
                fusion_policy="Native v2 fit uses cam2 prior plus both native keypoint views and LiDAR; cam1 prior retained independently for consistency review",
                no_unverified_cross_camera_track_merge=True,
                training_published=False,
            ),
        )
    progress.pop("worker_pid", None)
    progress.pop("active_command", None)
    progress.update(stage=stage + "_complete", training_published=False, review_only=True)
    write_json(root / "pilot_progress.json", progress)
    return progress


def serve_pilot(root, port, *, relocations=None, viewer_output=None):
    from .rally_viewer import prepare_viewer, serve_viewer

    root = Path(root).resolve()
    if not root.is_relative_to(PROJ_ROOT / "outputs"):
        raise ValueError("Select a workspace pilot root")
    labels = root / "repair/final_labels"
    dataset = read_json(labels / "datasets/dataset_manifest.json")
    if dataset.get("review_only") is not True:
        raise ValueError("Pilot viewer requires explicitly review-only fitted bodies")
    data = prepare_viewer(labels, labels, relocations=relocations, viewer_output=viewer_output)
    for sid, entry in data.manifest["sources"].items():
        entry["baseline"] = True
    write_json(
        labels / "build_status.json",
        dict(
            state="review_only",
            review_only=True,
            training_published=False,
            body_qa_accepted=dataset["accepted"],
            sources={
                sid: dict(
                    baseline=True,
                    state="fitted",
                    accepted=dataset["exports"][sid]["accepted"],
                    fitted=sum(
                        read_json(p)["fitted"]
                        for p in (labels / "sessions" / sid / "windows").glob("*/fit_report.json")
                    ),
                    rallies=len(entry["rallies"]),
                    rally_unique_paired_scans=sum(r["source_frame_count"] for r in entry["rallies"]),
                )
                for sid, entry in data.manifest["sources"].items()
            },
        ),
    )
    data.manifest.update(
        review_only=True, training_published=False, camera_label=read_json(root / "pilot_input.json")["camera_label"]
    )
    segmented = any("segment_index" in rally for entry in data.manifest["sources"].values() for rally in entry["rallies"])
    if segmented:
        data.manifest["native_rally_count"] = len({(sid, rally["original_rally_index"])
            for sid, entry in data.manifest["sources"].items() for rally in entry["rallies"]})
    write_json(data.viewer_output / "manifest.json", data.manifest)
    page = data.viewer_output / "index.html"
    html = page.read_text(encoding="utf-8")
    html = html.replace("羽毛球 · 双机 / SMPL 检查", "R11 · 双机 / SMPL 评审")
    html = html.replace(
        "7 来源 · 204 回合", f'{data.manifest["source_count"]} 来源 · {data.manifest["rally_count"]} 回合'
    )
    html = html.replace(
        "7 来源 / 204 回合", f'{data.manifest["source_count"]} 来源 / {data.manifest["rally_count"]} 回合'
    )
    html = html.replace("回合版人帧", "试拟合人帧").replace("回合版数据", "R11 试拟合数据")
    if segmented:
        label = f'{data.manifest["native_rally_count"]} 原生回合 · {data.manifest["rally_count"]} 有效段'
        html = html.replace(f'{data.manifest["rally_count"]} 回合', label)
        html = html.replace('<th>回合</th>', '<th>有效段</th>')
    html = html.replace('<div class="counts">', '<div class="counts"><span>仅所选回合 · 未发布训练</span>')
    html = html.replace(
        "本页展示密封回合子集，合格人帧包含旧场次的回合复用和新增场次，不代表所有历史数据累计量。",
        "本页仅评审冻结 R11 标定下的所选真实回合，候选人体没有发布为训练数据。",
    )
    report_path = root / "quality_report.json"
    if report_path.exists():
        report = read_json(report_path)
        if report["dataset"]["sha256"] != file_sha(labels / "datasets/dataset_manifest.json"):
            raise ValueError("Viewer quality summary belongs to another body dataset")
        cohort = report["cohorts"]["all_fitted"]
        surface = cohort["surface_and_ground_mm"]["surface_mean_m"]["mean"]
        raw = cohort["raw_native_camera_px"]
        summary = (
            f'<p class="notice">全部 {cohort["person_scans"]} 人帧：点面均值 {surface:.2f} mm；'
            f'原始 2D 重投影 CAM1 {raw["cam1"]["mean_px"]["mean"]:.2f} px / '
            f'CAM2 {raw["cam2"]["mean_px"]["mean"]:.2f} px（未做射线剔除）。</p>'
        )
        html = html.replace('<div class="table-scroll">', summary + '<div class="table-scroll">')
    page.write_text(html, encoding="utf-8")
    serve_viewer(data, port=port)
