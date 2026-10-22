"""Synthetic byte-bound CPU freezer cases; no production input or inference."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hmr4d.lidar_mocap import formal_baseline as baseline


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(value, indent=1).encode("utf-8") + b"\n")
    return baseline._binding(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bytes_file(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return baseline._sha(path)


@pytest.fixture
def stage(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline, "PROJ_ROOT", tmp_path)
    root = tmp_path / "outputs/formal_synthetic"
    repair = root / "repair"
    folder = root / "gpu_guarded_leg/baseline"
    calibration = "c" * 64
    keys = ["20261019_152542:window_1", "20261020_182854:window_2"]
    jobs = [
        dict(source_id=k.split(":")[0], source_window=str(repair / "initial_teacher" / k.split(":")[1])) for k in keys
    ]
    source = put(
        repair / "new_source_input.json",
        dict(schema="badminton-new-source-repair-input-v1", calibration_digest=calibration, jobs=jobs),
    )
    plan = put(
        repair / "final_labels/repair_plan.json",
        dict(
            jobs=jobs, source_union=source["path"], source_union_sha256=source["sha256"], calibration_digest=calibration
        ),
    )
    qa_values = dict(
        final_p2surface_mean_m=0.025,
        final_p2surface_p90_m=0.05,
        reprojection_mean_px=4.0,
        finite=True,
        floor_penetration_m=0.01,
        reprojection_valid_joints=dict(cam1=9, cam2=8),
    )
    windows, current_dirs, originals, original_qa = (
        {},
        {},
        {},
        dict(schema="badminton-native-absolute-qa-v1", frames={}, windows={}),
    )
    native_windows = {}
    for index, key in enumerate(keys):
        sid, name = key.split(":")
        actual = repair / "final_labels/sessions" / sid / "windows" / name
        old = tmp_path / "parents" / sid / name
        original_rows = [
            dict(
                motion_id="person",
                lidar_index=index,
                accepted=True,
                source_point_count=100,
                qa=qa_values,
                fit_path=str(old / "fits/person.npz"),
            ),
            dict(
                motion_id="person",
                lidar_index=index + 10,
                accepted=False,
                source_point_count=20,
                qa=dict(qa_values, final_p2surface_mean_m=0.2),
                fit_path=str(old / "fits/rejected.npz"),
            ),
        ]
        current_rows = [dict(row, fit_path=str(actual / "fits" / Path(row["fit_path"]).name)) for row in original_rows]
        reports = []
        for path, rows, provenance in ((old, original_rows, "review"), (actual, current_rows, "formal")):
            artifacts = {
                "ptv3.pth": bytes_file(path / "ptv3.pth", b"same synthetic tensor bytes"),
                "fits/person.npz": bytes_file(path / "fits/person.npz", b"same synthetic accepted fit bytes"),
                "fits/rejected.npz": bytes_file(path / "fits/rejected.npz", b"same rejected fit bytes"),
                "quality.jsonl": bytes_file(
                    path / "quality.jsonl", b"".join(json.dumps(r).encode() + b"\n" for r in rows)
                ),
                "ptv3.metadata.jsonl": bytes_file(path / "ptv3.metadata.jsonl", provenance.encode()),
            }
            reports.append(
                put(path / "fit_report.json", dict(accepted=1, artifacts_sha256=artifacts, provenance=provenance))
            )
        snapshot = folder / "reports" / sid / name / "fit_report.json"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes((actual / "fit_report.json").read_bytes())
        windows[key] = dict(original_report=reports[0], current_report=baseline._binding(snapshot))
        current_dirs[key], originals[key] = actual, old
        native_windows[key] = dict(body=dict(report_sha256=reports[1]["sha256"], accepted_checked=1))
        frame_key = f"{sid}/{name}/person/{index}"
        original_qa["frames"][frame_key] = {
            k: qa_values[k] for k in ("final_p2surface_mean_m", "final_p2surface_p90_m", "reprojection_mean_px")
        }
        original_qa["windows"][key] = dict(fit_report_sha256=reports[0]["sha256"], accepted=1)
    native = put(
        repair / "final_labels/native_verification.json",
        dict(
            state="complete",
            passed=True,
            windows=native_windows,
            total_windows=2,
            verified_windows=2,
            accepted_checked=2,
            source_union_sha256=source["sha256"],
            calibration_digest=calibration,
        ),
    )
    body = put(
        repair / "final_labels/repair_receipt.json", dict(state="body_qa_candidate_complete", windows=2, accepted=2)
    )
    config = dict(calibration_digest=calibration, precision="float16")
    poses = put(
        repair / "pose_receipt.json",
        dict(
            state="complete",
            config=config,
            config_digest=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            source_union_sha256=source["sha256"],
        ),
    )
    witness = put(tmp_path / "evidence/review.json", dict(synthetic_native_witness=True))
    identity = put(
        repair / "identity_evidence/identity_decisions.json",
        dict(
            review_status="reviewed",
            windows={k: {} for k in keys},
            source_union_sha256=source["sha256"],
            calibration_digest=calibration,
            evidence_files={poses["path"]: poses["sha256"], witness["path"]: witness["sha256"]},
        ),
    )
    imported = dict(
        schema="badminton-formal-numeric-import-v1",
        state="complete",
        source_union_sha256=source["sha256"],
        calibration_digest=calibration,
        numerical_artifacts_byte_identical=True,
        qa_values_unchanged=True,
        windows=windows,
        original_qa_snapshot=put(root / "numeric_original_qa.json", original_qa),
    )
    import_path = root / "numeric_import.json"
    put(import_path, imported)
    result = SimpleNamespace(
        root=root,
        repair=repair,
        folder=folder,
        keys=keys,
        imported=imported,
        import_path=import_path,
        current_dirs=current_dirs,
        originals=originals,
        native=native,
        body=body,
        poses=poses,
        identity=identity,
        witness=witness,
        source=source,
        plan=plan,
    )
    result.freeze = lambda: baseline.freeze_formal_baseline(root, import_path, baseline._sha(import_path))
    return result


def refresh_report(stage, key):
    actual = stage.current_dirs[key] / "fit_report.json"
    snapshot = Path(stage.imported["windows"][key]["current_report"]["path"])
    snapshot.write_bytes(actual.read_bytes())
    stage.imported["windows"][key]["current_report"] = baseline._binding(snapshot)
    native = read(stage.native["path"])
    native["windows"][key]["body"]["report_sha256"] = baseline._sha(actual)
    put(Path(stage.native["path"]), native)
    put(stage.import_path, stage.imported)


def test_freezes_byte_identical_complete_cpu_evidence_and_can_revalidate(stage):
    result = stage.freeze()
    frozen = read(result["baseline"]["path"])
    assert frozen["schema"] == "gvhmr-formal-preleg-baseline-v1" and frozen["state"] == "complete"
    assert set(frozen["windows"]) == set(stage.keys) and result["accepted"] == 2
    for role in ("native", "body", "pose", "identity"):
        original = result["inputs"][role]["path"]
        snapshot = result["snapshots"][role]["path"]
        assert Path(snapshot).read_bytes() == Path(original).read_bytes()
        assert not os.path.samefile(original, snapshot)
    qa = read(result["snapshots"]["qa"]["path"])
    assert len(qa["frames"]) == 2
    for key in stage.keys:
        assert frozen["windows"][key]["report_snapshot"] == stage.imported["windows"][key]["current_report"]
        assert qa["windows"][key]["accepted"] == 1
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in stage.folder.rglob("*") if p.is_file()}
    assert stage.freeze() == result
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    assert not (stage.folder / "readiness.json").exists()
    assert not (stage.folder.parent / "grant.json").exists()
    assert not (stage.folder.parent / "guarded_leg_stage_lease.json").exists()
    assert result["gpu_started"] is False and result["training_published"] is False


def test_exact_numeric_import_sha_is_required_before_any_snapshot(stage):
    with pytest.raises(ValueError, match="changed"):
        baseline.freeze_formal_baseline(stage.root, stage.import_path, "f" * 64)
    assert not (stage.folder / "baseline.json").exists()


@pytest.mark.parametrize("change", ["state", "source", "calibration", "numeric", "qa", "partial", "extra"])
def test_rejects_partial_or_wrong_formal_import(stage, change):
    value = deepcopy(stage.imported)
    if change == "state":
        value["state"] = "pending"
    if change == "source":
        value["source_union_sha256"] = "d" * 64
    if change == "calibration":
        value["calibration_digest"] = "d" * 64
    if change == "numeric":
        value["numerical_artifacts_byte_identical"] = False
    if change == "qa":
        value["qa_values_unchanged"] = False
    if change == "partial":
        value["windows"].pop(stage.keys[0])
    if change == "extra":
        value["windows"]["20261020_000000:extra"] = {}
    put(stage.import_path, value)
    with pytest.raises(ValueError):
        stage.freeze()
    assert not (stage.folder / "baseline.json").exists()


@pytest.mark.parametrize(
    "role,field,value",
    [
        ("native", "passed", False),
        ("native", "state", "partial"),
        ("native", "verified_windows", 1),
        ("native", "total_windows", 1),
        ("native", "accepted_checked", 1),
        ("native", "calibration_digest", "e" * 64),
        ("native", "source_union_sha256", "e" * 64),
        ("body", "state", "running"),
        ("body", "windows", 1),
        ("body", "accepted", 1),
        ("poses", "state", "partial"),
        ("poses", "source_union_sha256", "e" * 64),
        ("poses", "config_digest", "e" * 64),
        ("identity", "review_status", "pending"),
        ("identity", "calibration_digest", "e" * 64),
        ("identity", "evidence_files", {}),
    ],
)
def test_rejects_missing_actual_native_body_pose_or_identity_completion(stage, role, field, value):
    path = Path(getattr(stage, role)["path"])
    payload = read(path)
    payload[field] = value
    put(path, payload)
    with pytest.raises(ValueError):
        stage.freeze()
    assert not (stage.folder / "baseline.json").exists()


@pytest.mark.parametrize("role", ["native", "identity"])
def test_rejects_missing_window_evidence_despite_matching_global_count(stage, role):
    path = Path(getattr(stage, role)["path"])
    value = read(path)
    value["windows"].pop(stage.keys[0])
    put(path, value)
    with pytest.raises(ValueError):
        stage.freeze()


def test_rejects_changed_identity_witness(stage):
    Path(stage.witness["path"]).write_bytes(b"changed actual witness")
    with pytest.raises(ValueError, match="identity evidence"):
        stage.freeze()


@pytest.mark.parametrize("change", ["snapshot", "actual", "original", "hardlink", "location", "missing"])
def test_report_bindings_must_match_fixed_independent_actual_snapshot(stage, change):
    key = stage.keys[0]
    entry = stage.imported["windows"][key]
    snapshot = Path(entry["current_report"]["path"])
    actual = stage.current_dirs[key] / "fit_report.json"
    if change == "snapshot":
        snapshot.write_bytes(b"{}")
    if change == "actual":
        actual.write_bytes(b"{}")
    if change == "original":
        Path(entry["original_report"]["path"]).write_bytes(b"{}")
    if change == "hardlink":
        snapshot.unlink()
        os.link(actual, snapshot)
    if change == "location":
        entry["current_report"] = baseline._binding(actual)
    if change == "missing":
        snapshot.unlink()
    put(stage.import_path, stage.imported)
    with pytest.raises((ValueError, FileNotFoundError)):
        stage.freeze()


@pytest.mark.parametrize(
    "artifact", ["ptv3.pth", "fits/person.npz", "fits/rejected.npz", "ptv3.metadata.jsonl", "quality.jsonl"]
)
def test_rejects_any_actual_artifact_byte_change(stage, artifact):
    (stage.current_dirs[stage.keys[0]] / artifact).write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        stage.freeze()


@pytest.mark.parametrize("change", ["value", "extra_numeric", "missing_numeric", "escaping"])
def test_rejects_rehashed_numeric_changes_or_escaping_artifacts(stage, change):
    key = stage.keys[0]
    window = stage.current_dirs[key]
    path = window / "fit_report.json"
    report = read(path)
    if change == "value":
        report["artifacts_sha256"]["fits/person.npz"] = bytes_file(window / "fits/person.npz", b"new fit")
    if change == "extra_numeric":
        report["artifacts_sha256"]["fits/extra.npz"] = bytes_file(window / "fits/extra.npz", b"extra fit")
    if change == "missing_numeric":
        report["artifacts_sha256"].pop("fits/rejected.npz")
    if change == "escaping":
        report["artifacts_sha256"]["../outside"] = "a" * 64
    put(path, report)
    refresh_report(stage, key)
    with pytest.raises(ValueError):
        stage.freeze()


def test_rejects_changed_rejected_qa_even_when_rehashed_and_numeric_files_match(stage):
    key = stage.keys[0]
    path = stage.current_dirs[key] / "quality.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["qa"]["reprojection_mean_px"] += 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = read(stage.current_dirs[key] / "fit_report.json")
    report["artifacts_sha256"]["quality.jsonl"] = baseline._sha(path)
    put(stage.current_dirs[key] / "fit_report.json", report)
    refresh_report(stage, key)
    with pytest.raises(ValueError, match="row QA"):
        stage.freeze()


@pytest.mark.parametrize("change", ["values", "frame_membership", "per_window_count", "original_report_sha"])
def test_original_qa_snapshot_is_compared_per_value_membership_window_and_original_report(stage, change):
    path = Path(stage.imported["original_qa_snapshot"]["path"])
    qa = read(path)
    if change == "values":
        next(iter(qa["frames"].values()))["reprojection_mean_px"] += 0.01
    if change == "frame_membership":
        qa["frames"]["invented/person/scan"] = qa["frames"].pop(next(iter(qa["frames"])))
    if change == "per_window_count":
        qa["windows"][stage.keys[0]]["accepted"] = 2
        qa["windows"][stage.keys[1]]["accepted"] = 0
    if change == "original_report_sha":
        qa["windows"][stage.keys[0]]["fit_report_sha256"] = "f" * 64
    stage.imported["original_qa_snapshot"] = put(path, qa)
    put(stage.import_path, stage.imported)
    with pytest.raises(ValueError, match="original QA snapshot"):
        stage.freeze()


def test_actual_qa_gate_is_recomputed_instead_of_trusting_complete_labels(stage):
    key = stage.keys[0]
    for folder in (stage.current_dirs[key], stage.originals[key]):
        path = folder / "quality.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["source_point_count"] = 20
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        report = read(folder / "fit_report.json")
        report["artifacts_sha256"]["quality.jsonl"] = baseline._sha(path)
        put(folder / "fit_report.json", report)
    stage.imported["windows"][key]["original_report"] = baseline._binding(stage.originals[key] / "fit_report.json")
    refresh_report(stage, key)
    with pytest.raises(ValueError, match="unchanged QA gate"):
        stage.freeze()


def test_existing_different_snapshot_is_rejected_without_any_other_publication(stage):
    path = stage.folder / "pose_receipt.json"
    path.write_bytes(b"previous different receipt")
    with pytest.raises(ValueError, match="existing immutable"):
        stage.freeze()
    assert path.read_bytes() == b"previous different receipt"
    assert not (stage.folder / "native_verification.json").exists()
    assert not (stage.folder / "baseline.json").exists()


@pytest.mark.parametrize("marker", ["final_labels/post_refinement.json", "leg_refinement_receipt.json"])
def test_refuses_a_baseline_after_any_leg_refinement_marker(stage, marker):
    put(stage.repair / marker, dict(state="partial"))
    with pytest.raises(ValueError, match="after refinement"):
        stage.freeze()


def test_rejects_wrong_output_target(stage):
    with pytest.raises(ValueError, match="inside project outputs"):
        baseline.freeze_formal_baseline(
            stage.root.parent.parent / "outside", stage.import_path, baseline._sha(stage.import_path)
        )


def test_source_and_body_plan_must_match_exact_jobs(stage):
    path = Path(stage.plan["path"])
    plan = read(path)
    plan["jobs"][0]["source_window"] = str(stage.root / "wrong_teacher" / "window_1")
    put(path, plan)
    with pytest.raises(ValueError, match="exact body plan"):
        stage.freeze()


def test_native_per_window_counts_cannot_hide_behind_an_unchanged_total(stage):
    native = read(stage.native["path"])
    native["windows"][stage.keys[0]]["body"]["accepted_checked"] = 2
    native["windows"][stage.keys[1]]["body"]["accepted_checked"] = 0
    put(Path(stage.native["path"]), native)
    with pytest.raises(ValueError, match="accepted rows were not native verified"):
        stage.freeze()


def test_changed_input_during_absolute_qa_cannot_publish_a_complete_wrapper(stage, monkeypatch):
    from hmr4d.lidar_mocap import new_source_completion

    actual_collector = new_source_completion.collect_absolute_qa

    def changing_collector(root):
        result = actual_collector(root)
        (stage.current_dirs[stage.keys[0]] / "fits/person.npz").write_bytes(b"changed during CPU QA")
        return result

    monkeypatch.setattr(new_source_completion, "collect_absolute_qa", changing_collector)
    with pytest.raises(ValueError, match="during CPU validation"):
        stage.freeze()
    assert not (stage.folder / "baseline.json").exists()
    assert not (stage.folder / "freezer_receipt.json").exists()
