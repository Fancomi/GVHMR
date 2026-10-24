"""Synthetic CPU orchestration; geometry/model/native execution is not claimed."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from hmr4d.lidar_mocap import admitted_native_input, formal_incremental_intake as prep, incremental_release as inc
from hmr4d.lidar_mocap import dual_view_observer as observer, rally_pipeline, relocations
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.fitting import FittingConfig
from hmr4d.lidar_mocap.source_index import SourceIndexedSession
from test_lidar_incremental_release import bind, make_policy, write


@pytest.fixture
def prepared_case(tmp_path, monkeypatch):
    selected = make_policy(tmp_path)
    policy = json.loads(Path(selected["path"]).read_text())
    scope_path = Path(policy["reviewed_scope"]["path"])
    scope = json.loads(scope_path.read_text())
    entry = scope["review_sources"]["fresh"]
    review = Path(entry["review_root"])
    ranges_path = review / "source/pilot_ranges.jsonl"
    ranges_path.write_text(
        json.dumps(dict(source_id="fresh", begin_common_us=100, end_common_us=200, eligible_paired_png_indices=[0]))
        + "\n"
    )
    intake = write(
        review / "source/pilot_intake.json",
        dict(schema="badminton-review-rally-intake-v1", review_only=True, prepared_ranges=bind(ranges_path)),
    )
    registry_path = Path(entry["review_source_registry"]["path"])
    registry = json.loads(registry_path.read_text())
    registry["intake"] = intake
    entry["review_source_registry"] = write(registry_path, registry)
    source_path = Path(entry["review_source_input"]["path"])
    source = json.loads(source_path.read_text())
    source["source_registry_sha256"] = entry["review_source_registry"]["sha256"]
    entry["review_source_input"] = write(source_path, source)
    policy["reviewed_scope"] = write(scope_path, scope)
    selected = write(Path(selected["path"]), policy)
    project = Path(policy["gvhmr_root"])
    monkeypatch.setattr(prep, "PROJ_ROOT", project)
    monkeypatch.setattr(
        CalibrationSnapshot,
        "load",
        staticmethod(lambda path: SimpleNamespace(snapshot_json=Path(path), data=json.loads(Path(path).read_text()))),
    )
    snapshot_calls, source_calls, body_calls = [], [], []

    def snapshot(*args, **kwargs):
        snapshot_calls.append((args, kwargs))
        return dict(schema="synthetic-original-helper-call", actual_source_shape_checked_elsewhere=True)

    monkeypatch.setattr(admitted_native_input, "verify_review_snapshot", snapshot)
    session = SimpleNamespace(
        source_index=json.loads(Path(entry["admitted_source_index"]["path"]).read_text()),
        camera_applicability=dict(
            file=entry["camera_applicability"]["path"], sha256=entry["camera_applicability"]["sha256"]
        ),
        pairs=lambda: [dict(lidar_index=0, scan_mid_us=100, status="paired")],
    )

    def admitted(*args, **kwargs):
        source_calls.append((args, kwargs))
        return session

    monkeypatch.setattr(SourceIndexedSession, "load_source_index", staticmethod(admitted))
    monkeypatch.setattr(rally_pipeline, "load_run_session", lambda *args, **kwargs: session)
    original_resolver = relocations.load_pilot_run_relocations
    monkeypatch.setattr(
        relocations,
        "load_pilot_run_relocations",
        lambda root, calibration=None: (
            original_resolver(root, calibration) if Path(root) != review / "source" else None
        ),
    )

    def body_prepare(command, **kwargs):
        body_calls.append((command, kwargs))
        arg = lambda name: command[command.index(name) + 1]
        source_root, output = Path(arg("--source-root")), Path(arg("--output-root"))
        input_path = Path(arg("--union"))
        source_input = json.loads(input_path.read_text())
        shutil.copytree(source_root / "calibration", output / "calibration")
        shutil.copyfile(source_root / "source_registry.json", output / "source_registry.json")
        write(
            output / "repair_plan.json",
            dict(
                schema="badminton-body-repair-plan-v1",
                source_union=str(input_path),
                source_union_sha256=bind(input_path)["sha256"],
                calibration_digest=source_input["calibration_digest"],
                jobs=source_input["jobs"],
                training_published=False,
                fitting_config=asdict(FittingConfig(distal_pose_prior_weight=0.03, lower_limb_reprojection_weight=2)),
            ),
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(prep.subprocess, "run", body_prepare)
    return SimpleNamespace(
        selected=selected,
        policy=policy,
        entry=entry,
        review=review,
        output=project / "outputs/formal",
        session=session,
        snapshot_calls=snapshot_calls,
        source_calls=source_calls,
        body_calls=body_calls,
    )


def run(case):
    return prep.prepare_formal_increment(case.selected["path"], case.selected["sha256"], case.output)


def test_single_source_intake_uses_original_windows_and_exact_accepted_source(prepared_case):
    case = prepared_case
    before = bind(Path(case.entry["review_source_input"]["path"]))
    result = run(case)
    proof = json.loads(Path(result["path"]).read_text())
    assert proof["state"] == "formal_metadata_prepared_body_import_pending"
    assert proof["model_loaded"] is False and proof["gpu_started"] is False and proof["training_published"] is False
    assert case.snapshot_calls and case.source_calls[0][1]["require_admitted"] is True
    assert case.source_calls[0][1]["expected_sha256"] == case.entry["admitted_source_index"]["sha256"]
    assert len(case.body_calls) == 1
    command, kwargs = case.body_calls[0]
    assert command[command.index("--stage") + 1] == "prepare"
    assert command[command.index("--device") + 1] == "cpu"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
    source = json.loads((case.output / "repair/new_source_input.json").read_text())
    assert source["session_splits"] == dict(train=["fresh"], val=[], test=[])
    assert source["required_nonempty_splits"] == ["train"]
    assert source["release_policy"] == case.selected
    assert source["jobs"][0]["source_id"] == "fresh"
    config = json.loads((Path(source["jobs"][0]["source_window"]) / "window.json").read_text())
    assert config["settings"] == dict(camera="cam2", teacher_stride=1)
    assert config["source_pairs"] == [0] and config["left_us"] == 100 and config["right_us"] == 200
    assert config["calibration_digest"] == "f" * 64
    assert bind(case.output / "source/calibration/calibration_snapshot.json") == {
        "path": str((case.output / "source/calibration/calibration_snapshot.json").resolve()),
        "sha256": case.entry["accepted_parent_snapshot"]["sha256"],
    }
    assert bind(Path(case.entry["review_source_input"]["path"])) == before
    registry = json.loads((case.output / "source/source_registry.json").read_text())
    assert registry["sources"]["fresh"]["source_index"] == case.entry["admitted_source_index"]
    assert registry["release_policy"] == case.selected and registry.get("review_only") is not True


def observer_inputs(case):
    labels = case.output / "repair/final_labels"
    registry = json.loads((labels / "source_registry.json").read_text())
    return (
        labels,
        dict(exports={sid: {} for sid in registry["sources"]}),
        CalibrationSnapshot.load(labels / "calibration/calibration_snapshot.json"),
    )


def rebind_input_equivalence(case, proof_binding):
    """Keep genuine JSON hashes coherent while testing the original observer's checks."""
    input_path = case.output / "repair/new_source_input.json"
    prepared = json.loads(input_path.read_text())
    prepared["formal_derivation"]["input_equivalence"] = proof_binding
    input_binding = write(input_path, prepared)
    plan_path = case.output / "repair/final_labels/repair_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["source_union_sha256"] = input_binding["sha256"]
    write(plan_path, plan)


def test_real_intake_proof_is_accepted_by_the_original_formal_teacher_observer(prepared_case):
    case = prepared_case
    result = run(case)
    prepared_receipt = json.loads(Path(result["path"]).read_text())
    source_input_path = case.output / "repair/new_source_input.json"
    source_input = json.loads(source_input_path.read_text())
    canonical = bind(case.output / "formal_input_equivalence.json")
    assert canonical == prepared_receipt["input_equivalence"]
    assert canonical == source_input["formal_derivation"]["input_equivalence"]
    assert not (case.output / "import_evidence/input_equivalence.json").exists()
    # The producer runs the real input-equivalence audit; only source decoding,
    # snapshot verification and the body-prepare subprocess use the fixture stand-ins.
    assert observer._formal_teacher_proof(*observer_inputs(case)) == dict(
        proof=canonical, source_input=bind(source_input_path)
    )


def test_original_observer_rejects_a_rebound_byte_identical_old_path(prepared_case):
    case = prepared_case
    run(case)
    canonical = case.output / "formal_input_equivalence.json"
    alias = case.output / "import_evidence/input_equivalence.json"
    shutil.copyfile(canonical, alias)
    assert bind(alias)["sha256"] == bind(canonical)["sha256"]
    rebind_input_equivalence(case, bind(alias))
    with pytest.raises(ValueError, match="bound input-equivalence proof"):
        observer._formal_teacher_proof(*observer_inputs(case))


def test_original_observer_rejects_changed_canonical_proof_bytes(prepared_case):
    case = prepared_case
    run(case)
    canonical = case.output / "formal_input_equivalence.json"
    canonical.write_bytes(canonical.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bound input-equivalence proof"):
        observer._formal_teacher_proof(*observer_inputs(case))


@pytest.mark.parametrize("change", ["camera", "physical", "index", "sources", "handoff"])
def test_original_observer_retains_physical_input_checks_after_canonical_binding(prepared_case, change):
    case = prepared_case
    run(case)
    canonical = case.output / "formal_input_equivalence.json"
    proof = json.loads(canonical.read_text())
    if change == "camera":
        proof["evidence"]["formal_calibration"]["sha256"] = "0" * 64
    elif change == "physical":
        proof["sources"]["fresh"]["cameras_court_ground_bbox_clock_exact"] = False
    elif change == "index":
        proof["sources"]["fresh"]["admitted_index"]["sha256"] = "0" * 64
    elif change == "sources":
        proof["sources"] = {}
    elif change == "handoff":
        input_path = case.output / "repair/new_source_input.json"
        prepared = json.loads(input_path.read_text())
        prepared["formal_derivation"]["camera_admission_handoff"]["sha256"] = "0" * 64
        write(input_path, prepared)
    rebind_input_equivalence(case, write(canonical, proof))
    with pytest.raises(ValueError, match="equivalence does not bind|exact physical calibration/input mapping"):
        observer._formal_teacher_proof(*observer_inputs(case))


def test_changed_original_window_rejects_before_any_formal_write(prepared_case):
    case = prepared_case
    source = json.loads(Path(case.entry["review_source_input"]["path"]).read_text())
    (Path(source["jobs"][0]["source_window"]) / "window.json").write_text("{}")
    with pytest.raises(ValueError, match="Original native window changed"):
        run(case)
    assert not case.output.exists() and not case.body_calls


def test_wrong_real_camera_certificate_rejects_before_formal_write(prepared_case):
    case = prepared_case
    case.session.camera_applicability = dict(case.session.camera_applicability, sha256="0" * 64)
    with pytest.raises(ValueError, match="camera certificate"):
        run(case)
    assert not case.output.exists() and not case.body_calls


def test_existing_formal_output_is_preserved(prepared_case):
    case = prepared_case
    case.output.mkdir(parents=True)
    marker = case.output / "do-not-rewrite"
    marker.write_bytes(b"preserved")
    with pytest.raises(ValueError, match="fresh formal root"):
        run(case)
    assert marker.read_bytes() == b"preserved" and not case.body_calls


def test_policy_sha_is_required_before_output_creation(prepared_case):
    case = prepared_case
    with pytest.raises(ValueError, match="Frozen incremental dependency changed"):
        prep.prepare_formal_increment(case.selected["path"], "0" * 64, case.output)
    assert not case.output.exists()


def test_native_time_scope_is_not_interpolated_or_padded(prepared_case):
    case = prepared_case
    case.session.pairs = lambda: [dict(lidar_index=0, scan_mid_us=500, status="paired")]
    with pytest.raises(ValueError, match="exact paired coverage"):
        run(case)
    assert not case.output.exists() and not case.body_calls


def test_real_accepted_bundle_must_equal_original_physical_camera(prepared_case):
    case = prepared_case
    case.session.source_index["accepted_bundle_sha256"] = {"calibration.json": "b" * 64}
    with pytest.raises(ValueError, match="physical camera bundle"):
        run(case)
    assert not case.output.exists() and not case.body_calls
