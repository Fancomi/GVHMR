"""Build fixed-camera body labels for an explicitly sealed multi-source rally intake.

Original calibration/source files are read only. Existing baseline labels remain
in their original run; their hashes and original source bindings are checked
before selecting the exact rally subset. New recordings use SourceIndexedSession.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from hmr4d import PROJ_ROOT
from .calibration import CalibrationSnapshot
from .dual_view import file_sha
from .export import FIELDS, acceptance_reasons, export_joblib
from .pipeline import BadmintonPipeline, write_json
from .relocations import validate_run_relocations
from .sessions import SessionIndex
from .source_index import SourceIndexedSession, _bound_file


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def window_directory(root, dataset, sid, window):
    return Path(dataset.get("window_roots", {}).get(sid, root)) / "sessions" / sid / "windows" / window


def load_run_session(root, calibration, sid, *, relocations=None):
    """Resolve only the explicitly frozen source registry of a rally run."""
    registry_path = Path(root) / "source_registry.json"
    if registry_path.exists():
        registry = read_json(registry_path)
        validate_run_relocations(registry, calibration.data, relocations)
        entry = registry["sources"][sid]
        if entry.get("source_index"):
            review=registry.get('review_only') is True
            if review:
                declared=registry.get('pilot_manifest',{})
                if (calibration.data.get('review_only') is not True or not declared.get('path')
                        or file_sha(declared['path'])!=declared.get('sha256')):
                    raise ValueError('Review source loader requires its original frozen pilot manifest and camera scope')
                pilot=read_json(declared['path'])
                from .pilot import validate_camera_review
                validate_camera_review(pilot)
                expected=pilot.get('source_indexes',{}).get(sid,{})
                if (pilot.get('schema')!='badminton-review-pilot-v1' or pilot.get('review_only') is not True
                        or pilot.get('training_published') is not False
                        or pilot.get('bundle_sha256')!=calibration.data['configuration_files_sha256']
                        or expected.get('sha256')!=entry['source_index']['sha256']
                        or Path(pilot['data_root']).resolve()!=Path(registry['data_root']).resolve()):
                    raise ValueError('Review source registry differs from its explicit pilot input')
            return SourceIndexedSession.load_source_index(
                entry["source_index"]["path"], registry["data_root"], expected_sha256=entry["source_index"]["sha256"],
                require_admitted=not review, relocations=relocations
            )
    elif relocations is not None:
        validate_run_relocations(
            dict(data_root=str(Path(calibration.data["sessions_root"]).parent)), calibration.data, relocations)
    session = SessionIndex.load(Path(calibration.data["sessions_root"]) / sid, calibration.frame_map_path(sid))
    if relocations is not None:
        session.relocation_legacy_audit = audit_legacy_session_paths(session, relocations)
    return session


def audit_legacy_session_paths(session, relocations):
    """Audit paths actually used by the unchanged legacy reader; do not relocate them."""
    paths = [session.lidar_dir / "manifest.json",
             session.meta_dir / f"{session.session_id}_common_clock.json", *session.video_paths.values()]
    for camera in session.video_paths:
        sidecar = session.video_dir / f"{camera}.frames.jsonl"
        if sidecar.exists():
            paths.append(sidecar)
        else:
            name = session._tracks.get(camera, {}).get("raw_index", f"{session.session_id}_{camera}_video_frames.jsonl")
            paths.append(session.meta_dir / name)
    paths.extend(Path(pair["lidar_path"]) for pair in session.pairs())
    old_paths = []
    for path in paths:
        resolved = Path(path).resolve()
        if not resolved.is_relative_to(relocations.root):
            raise ValueError("Legacy source dependency escapes the explicitly selected DATA root")
        logical = resolved.relative_to(relocations.root).as_posix()
        parts = Path(logical).parts
        old_domain = (parts[0] == "upstream" or (
            len(parts) >= 4 and parts[0] == "collections" and parts[2] == "device-time-sources"))
        if relocations.has_mapping(logical) or old_domain:
            old_paths.append(logical)
    if old_paths:
        raise ValueError("Legacy reader still depends on historical relocation paths: " + ", ".join(old_paths[:3]))
    return dict(reader="legacy_session", paths_checked=len(paths), historical_relocation_dependencies=[],
                relocation_applied=False)


def select_window_pairs(pairs, eligible, seconds=60):
    """Deduplicate exact PNG ordinals; accounting scan-start ends never filter mids."""
    selected = [p for p in pairs if p["lidar_index"] in eligible and p["status"] == "paired"]
    if {p["lidar_index"] for p in selected} != set(eligible):
        raise ValueError("Prepared eligible PNG set no longer matches actual paired input")
    if not selected:
        return []
    base = min(p["scan_mid_us"] for p in pairs if p["status"] == "paired")
    width = round(seconds * 1e6)
    groups = {}
    if width <= 0:
        raise ValueError("Positive fitting window required")
    for pair in selected:
        groups.setdefault((pair["scan_mid_us"] - base) // width, []).append(pair)
    return [
        (f"{k*width:012d}_{(k+1)*width:012d}", base + k * width, base + (k + 1) * width, group)
        for k, group in sorted(groups.items())
    ]


def validate_session_splits(splits, sources):
    """Require explicit disjoint, exhaustive whole-source splits for new intakes."""
    if not isinstance(splits, dict) or set(splits) != {"train", "val", "test"}:
        raise ValueError("Session splits require train, val and test")
    if any(not isinstance(v, list) or not v for v in splits.values()):
        raise ValueError("Each session split must be a nonempty list")
    flattened = [sid for values in splits.values() for sid in values]
    if any(not isinstance(sid, str) for sid in flattened):
        raise ValueError("Session split identities must be strings")
    if len(flattened) != len(set(flattened)) or set(flattened) != set(sources):
        raise ValueError("Session splits must cover each source exactly once")
    return {name: list(splits[name]) for name in ("train", "val", "test")}


def validate_split_windows(splits, sources):
    """Prepared paired windows, rather than source names, must populate each split."""
    counts = {name: sum(sources.get(sid, {}).get("fit_windows", 0) for sid in ids)
              for name, ids in splits.items()}
    empty = [name for name, count in counts.items() if count < 1]
    if empty:
        raise ValueError("No usable prepared paired window in split(s): " + ", ".join(empty))
    return counts


class RallyPipeline(BadmintonPipeline):
    def __init__(self, intake, output_root, baseline_root, lidar_project_root, *, calibration_dir=None, session_splits=None):
        self.intake_path = Path(intake).resolve()
        self.intake = read_json(self.intake_path)
        if self.intake.get("schema") != "badminton-rally-intake-verification-v1":
            raise ValueError("A verified sealed rally intake is required")
        self.data_root = Path(self.intake["data_root"]).resolve()
        self.baseline_root = Path(baseline_root).resolve() if baseline_root is not None else None
        self.calibration_dir = calibration_dir
        ranges = self.intake["prepared_ranges"]
        if file_sha(ranges["path"]) != ranges["sha256"]:
            raise ValueError("Prepared rally range hash changed")
        self.rallies = read_rows(ranges["path"])
        if len(self.rallies) != self.intake["rallies"]:
            raise ValueError("Rally range count changed")
        self.eligible = {
            sid: {i for r in self.rallies if r["source_id"] == sid for i in r["eligible_paired_png_indices"]}
            for sid in self.intake["sources"]
        }
        acquisition_path = Path(self.intake["acquisition_manifest"])
        if file_sha(acquisition_path) != self.intake["acquisition_manifest_sha256"]:
            raise ValueError("Sealed acquisition manifest changed")
        self.acquisition = read_json(acquisition_path)
        inventory_path = Path(self.intake["rally_inventory"])
        if file_sha(inventory_path) != self.intake["rally_inventory_sha256"]:
            raise ValueError("Sealed rally inventory changed")
        self.inventory = read_json(inventory_path)
        self.source_entries = {s["source_id"]: s for s in self.inventory["sources"]}
        self.reuse_baseline = any(entry["baseline"] for entry in self.source_entries.values())
        if self.reuse_baseline and self.baseline_root is None:
            raise ValueError("Baseline sources require an explicit baseline root")
        if session_splits is None:
            if calibration_dir is not None:
                raise ValueError("Independent calibration requires explicit whole-source session splits")
            session_splits = dict(
                train=[sid for sid in self.source_entries if sid not in ("20261012_161544", "20261012_154726")],
                val=["20261012_161544"], test=["20261012_154726"])
        self.session_splits = validate_session_splits(session_splits, self.source_entries)
        super().__init__(
            self.data_root / "sessions", lidar_project_root, output_root, preproc_precision="float16", roi_margin_px=20
        )
        self._sessions = {}

    def prepare(self):
        destination = self.output_root / "calibration"
        if self.reuse_baseline and not (destination / "calibration_snapshot.json").exists():
            shutil.copytree(self.baseline_root / "calibration", destination, dirs_exist_ok=True)
        # This validates current canonical inputs against the preserved legacy
        # snapshot's numerical/source equivalence; it never rewrites old bytes.
        self.calibration = CalibrationSnapshot.freeze(
            self.lidar_project_root, self.sessions_root, destination, calibration_dir=self.calibration_dir,
            source_indexes=({sid: entry["source_index"] for sid, entry in self.source_entries.items()}
                            if self.calibration_dir is not None and not self.reuse_baseline else None),
            data_root=self.data_root)
        if self.reuse_baseline:
            baseline = read_json(self.baseline_root / "datasets/dataset_manifest.json")
            if baseline["calibration_digest"] != self.calibration.data["snapshot_digest"]:
                raise ValueError("Baseline fit calibration differs from the frozen run")
        split_path = self.output_root / "session_splits.json"
        if split_path.exists() and read_json(split_path) != self.session_splits:
            raise ValueError("Run session split changed; use a new output root")
        registry = dict(
            schema="badminton-rally-run-source-registry-v1",
            data_root=str(self.data_root),
            intake=dict(path=str(self.intake_path), sha256=file_sha(self.intake_path)),
            acquisition_manifest_sha256=self.intake["acquisition_manifest_sha256"],
            rally_inventory_sha256=self.intake["rally_inventory_sha256"],
            sources={},
        )
        for sid, entry in self.source_entries.items():
            row = dict(
                baseline=entry["baseline"],
                source_manifest_sha256=entry["source_manifest_sha256"],
                common_clock_sha256=entry["common_clock_sha256"],
                frame_map_sha256=entry["frame_map_sha256"],
            )
            if not entry["baseline"]:
                path = _bound_file(self.data_root, entry["source_index"])
                row["source_index"] = dict(path=str(path), sha256=entry["source_index"]["sha256"])
            registry["sources"][sid] = row
        path = self.output_root / "source_registry.json"
        if path.exists() and read_json(path) != registry:
            raise ValueError("Run source registry changed; use a new output root")
        write_json(path, registry)
        self.registry = registry
        state = dict(
            schema="badminton-rally-build-v1",
            state="prepared",
            rallies=len(self.rallies),
            window_minutes=self.intake["effective_window_minutes"],
            sources={},
        )
        for sid, entry in self.source_entries.items():
            session = self._session(sid)
            state["sources"][sid] = dict(
                state="reuse_pending" if entry["baseline"] else "pending",
                baseline=entry["baseline"],
                rallies=self.intake["sources"][sid]["rallies"],
                raw_scans=len(session.pairs()),
                rally_unique_paired_scans=len(self.eligible[sid]),
                fit_windows=len(self._windows(session)),
                fitted=None,
                accepted=None,
                excluded=None,
                failures=[],
            )
        if self.calibration_dir is not None:
            state["split_fit_windows"] = validate_split_windows(self.session_splits, state["sources"])
        # A failed new-source window gate must not freeze an unusable split.
        write_json(split_path, self.session_splits)
        path = self.output_root / "build_status.json"
        if not path.exists():
            write_json(path, state)
        return state

    def _session(self, sid):
        if sid not in self._sessions:
            self._sessions[sid] = load_run_session(self.output_root, self._load(), sid)
            session = self._sessions[sid]
            entry = self.source_entries[sid]
            if not np.array_equal(session.Q, self.calibration.Q):
                raise ValueError("Source M differs from frozen cameras")
            if entry["baseline"]:
                frozen = self.calibration.data["sessions"][sid]
                if (
                    frozen["manifest_sha256"] != entry["source_manifest_sha256"]
                    or frozen["frame_map_sha256"] != entry["frame_map_sha256"]
                    or frozen["source_clock_audit"]["common_index_sha256"] != entry["common_clock_sha256"]
                ):
                    raise ValueError("Baseline source/calibration binding differs from the sealed rally source")
            else:
                index = session.source_index
                if self.calibration_dir is not None:
                    expected = self.calibration.data["configuration_files_sha256"]
                    if (index.get("accepted_calibration_sha256") != expected["calibration.json"]
                            or index.get("accepted_bundle_sha256") != expected):
                        raise ValueError("Source index calibration differs from the independent frozen bundle")
                for name in ("source_manifest", "common_clock", "frame_map"):
                    if index[name]["sha256"] != entry[name + "_sha256"]:
                        raise ValueError("Source index/inventory mismatch")
        return self._sessions[sid]

    def _windows(self, session, start_seconds=0, duration_seconds=None):
        if start_seconds or duration_seconds is not None:
            raise ValueError("Rally builds select exact prepared PNG sets")
        return select_window_pairs(session.pairs(), self.eligible[session.session_id], self.settings["window_seconds"])

    def _fit_digest(self):
        return hashlib.sha256(
            (super()._fit_digest() + file_sha(__file__) + file_sha(self.output_root / "source_registry.json")).encode()
        ).hexdigest()

    def build(self, stage="run"):
        self.prepare()
        path = self.output_root / "build_status.json"
        state = read_json(path)
        state.update(state="running", stage=stage)
        write_json(path, state)
        for sid, entry in self.source_entries.items():
            if entry["baseline"]:
                continue
            current = state["sources"][sid]
            current.update(state="running", failures=[])
            state["current_source"] = sid
            write_json(path, state)
            started = time.perf_counter()
            try:
                result = super().run([sid], stage=stage)
                if stage == "teacher":
                    current.update(state="teacher_complete")
                else:
                    reports = result["windows"]
                    fitted = sum(r["fitted"] for r in reports)
                    current.update(
                        state="fitted",
                        fitted=fitted,
                        accepted=result["accepted"],
                        excluded=fitted - result["accepted"],
                        segmentation_excluded=sum(len(r["segmentation_exclusions"]) for r in reports),
                        fit_rejection_reasons=dict(
                            sum((Counter(r["fit_rejection_reasons"]) for r in reports), Counter())
                        ),
                    )
                current["last_call_seconds"] = time.perf_counter() - started
            except Exception as error:
                current.update(state="failed", failures=[f"{type(error).__name__}: {error}"])
                write_json(path, state)
                # Preserve traceback and valid caches for this source. The
                # supervisor can resume it after the actual cause is resolved.
                raise
            write_json(path, state)
        if stage != "teacher":
            self.aggregate()
        return read_json(path)

    def aggregate(self, session_ids=None):
        import joblib

        baseline = read_json(self.baseline_root / "datasets/dataset_manifest.json") if self.reuse_baseline else {}
        exports = {}
        selected_windows = {}
        window_roots = {}
        bindings = {}
        state = read_json(self.output_root / "build_status.json")
        for sid, entry in self.source_entries.items():
            samples = []
            eligible = self.eligible[sid]
            if entry["baseline"]:
                original = baseline["exports"][sid]
                data = joblib.load(original["path"])
                metadata = read_rows(original["metadata"])
                if len(metadata) != len(data["motion_id"]):
                    raise ValueError("Baseline tensor/metadata count differs")
                # Validate immutable fit reports, actual PNG hashes and all fit
                # artifacts used by the selected windows before baseline reuse.
                selected_windows[sid] = baseline["selected_windows"][sid]
                for window in selected_windows[sid]:
                    directory = self.baseline_root / "sessions" / sid / "windows" / window
                    report = read_json(directory / "fit_report.json")
                    config = read_json(directory / "window.json")
                    if config["calibration_digest"] != self.calibration.data["snapshot_digest"]:
                        raise ValueError("Baseline window camera binding changed")
                    for relative, expected in report["artifacts_sha256"].items():
                        if file_sha(directory / relative) != expected:
                            raise ValueError("Baseline fit artifact changed")
                    for source, expected in report["source_lidar_sha256"].items():
                        if file_sha(source) != expected:
                            raise ValueError("Baseline original PNG changed")
                bindings[sid] = dict(
                    dataset_manifest_sha256=file_sha(self.baseline_root / "datasets/dataset_manifest.json"),
                    tensor_sha256=file_sha(original["path"]),
                    metadata_sha256=file_sha(original["metadata"]),
                )
                for i, row in enumerate(metadata):
                    if row["lidar_index"] not in eligible:
                        continue
                    if data["motion_id"][i] != row["motion_id"]:
                        raise ValueError("Baseline tensor/person order changed")
                    sample = {name: data[name][i].numpy() for name in FIELDS}
                    sample.update(motion_id=row["motion_id"], metadata=row)
                    samples.append(sample)
                window_roots[sid] = str(self.baseline_root)
                all_rows = [
                    r
                    for w in selected_windows[sid]
                    for r in read_rows(self.baseline_root / "sessions" / sid / "windows" / w / "quality.jsonl")
                    if r["lidar_index"] in eligible
                ]
            else:
                selected_windows[sid] = [w[0] for w in self._windows(self._session(sid))]
                all_rows = []
                for window in selected_windows[sid]:
                    directory = self.output_root / "sessions" / sid / "windows" / window
                    if not (directory / "fit_report.json").exists():
                        raise ValueError("Required rally fit window incomplete")
                    all_rows.extend(read_rows(directory / "quality.jsonl"))
                for row in all_rows:
                    if row["lidar_index"] not in eligible:
                        raise ValueError("Fitted scan escapes prepared rally set")
                    if not row["accepted"] or acceptance_reasons(row["qa"], row["source_point_count"]):
                        continue
                    with np.load(row["fit_path"]) as fit:
                        sample = {name: fit[name].copy() for name in FIELDS}
                    sample.update(motion_id=row["motion_id"], metadata=row)
                    samples.append(sample)
                window_roots[sid] = str(self.output_root)
            samples.sort(key=lambda s: (s["metadata"]["scan_mid_us"], s["motion_id"]))
            keys = [(s["metadata"]["lidar_index"], s["motion_id"]) for s in samples]
            if len(set(keys)) != len(keys):
                raise ValueError("Overlapping rally windows duplicated person labels")
            exports[sid] = export_joblib(samples, self.output_root / "datasets" / f"{sid}.pth")
            state["sources"][sid].update(
                state="reused" if entry["baseline"] else "fitted",
                fitted=len(all_rows),
                accepted=len(samples),
                excluded=len(all_rows) - len(samples),
            )
        split_sessions = self.session_splits
        result = dict(
            schema="badminton-ptv3-dataset-v1",
            exports=exports,
            suggested_session_splits={
                name: dict(sessions=sids, accepted=sum(exports[s]["accepted"] for s in sids))
                for name, sids in split_sessions.items()
            },
            selected_windows=selected_windows,
            window_roots=window_roots,
            baseline_reuse_bindings=bindings,
            fit_roots=[str(self.output_root / "sessions")] + ([str(self.baseline_root / "sessions")] if self.reuse_baseline else []),
            source_registry=str(self.output_root / "source_registry.json"),
            source_registry_sha256=file_sha(self.output_root / "source_registry.json"),
            calibration_digest=self.calibration.data["snapshot_digest"],
            accepted=sum(e["accepted"] for e in exports.values()),
            target_body_model="neutral_smpl_24_joints",
            shape_basis="SMPL fitted; never copied from SMPL-X",
            point_count=256,
            coordinate_space="lidar_viewer_xyz_m",
            scope="sealed_seven_source_rally_subset" if self.calibration_dir is None else "sealed_multi_source_rally_subset",
        )
        write_json(self.output_root / "datasets/dataset_manifest.json", result)
        state.update(state="body_labels_complete", body_qa_accepted=result["accepted"])
        write_json(self.output_root / "build_status.json", state)
        return result
