"""Synthetic CPU CLI contracts; model assets are replaced, not production evidence."""

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap import pipeline as pipeline_module
from hmr4d.lidar_mocap.export import FIELDS, require_nonempty_splits
from hmr4d.lidar_mocap.fitting import FittingConfig
from hmr4d.lidar_mocap.incremental_release import policy_from_binding
from test_lidar_incremental_release import bind, make_policy, write


def read(path):
    return json.loads(Path(path).read_bytes())


def tool(name):
    spec = importlib.util.spec_from_file_location("synthetic_" + name, PROJ_ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def forbidden(*args, **kwargs):
    pytest.fail("A synthetic completed-only CLI reached inference, fitting, or source decoding")


class StoredPipeline(pipeline_module.BadmintonPipeline):
    """Only bypass the checkout output-path constraint for pytest's temporary root."""

    def __init__(self, sessions_root, lidar_project_root, output_root):
        self.output_root = Path(output_root)
        self.calibration = None
        self.settings = {}


@pytest.fixture
def stage(tmp_path, monkeypatch):
    selected = make_policy(tmp_path)
    policy = policy_from_binding(selected)
    project = tmp_path / "project"
    root = project / "outputs/formal/repair"
    labels = root / "final_labels"
    source = project / "outputs/formal/source"
    source_registry = write(source / "source_registry.json", dict(sources=dict(fresh={})))
    write(labels / "source_registry.json", read(source_registry["path"]))
    cal = SimpleNamespace(
        data=dict(snapshot_digest="frozen", sessions_root=str(tmp_path / "unused-media"), sessions=dict(fresh={})),
        camera=lambda name: {},
    )
    write(labels / "calibration/calibration_snapshot.json", {})
    write(source / "calibration/calibration_snapshot.json", {})
    pose = SimpleNamespace(digest="native-poses", policy_sha256="joint-ray-filter")
    name = "000000000000_000000000100"
    original = project / "outputs/review/sessions/fresh/windows" / name
    config = dict(
        window_id=name,
        left_us=100,
        right_us=200,
        calibration_digest="frozen",
        settings=dict(camera="cam2", teacher_stride=1),
        source_pairs=[0],
    )
    original_config = write(original / "window.json", config)
    teacher = dict(
        provenance=dict(camera={}, video=dict(path="unused"), frame_rows=[], options={}),
        provenance_hash="cached-teacher",
        tracks=[],
    )
    write(original / "teacher/manifest.json", teacher)
    output = labels / "sessions/fresh/windows" / name
    write(output / "window.json", config)
    write(output / "teacher/manifest.json", teacher)
    fit = output / "fits/synthetic.npz"
    fit.parent.mkdir(parents=True)
    np.savez(fit, **{field: np.zeros(shape, np.float32) for field, shape in FIELDS.items()})
    qa = dict(
        final_p2surface_mean_m=0.01,
        final_p2surface_p90_m=0.02,
        finite=True,
        floor_penetration_m=0.0,
        reprojection_mean_px=1.0,
        reprojection_valid_joints=dict(cam1=17, cam2=17),
    )
    row = dict(
        accepted=True,
        qa=qa,
        source_point_count=256,
        fit_path=str(fit),
        motion_id="synthetic-motion",
        scan_mid_us=150,
        lidar_index=0,
    )
    (output / "quality.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    cached = dict(
        fit_algorithm_digest=StoredPipeline(None, None, labels)._fit_digest(),
        fitting_config=asdict(FittingConfig(distal_pose_prior_weight=0.03, lower_limb_reprojection_weight=1.0)),
        settings=dict(
            config["settings"],
            share_track_shape=True,
            native_pose_receipt_sha256=pose.digest,
            native_pose_joint_ray_policy_sha256=pose.policy_sha256,
        ),
        teacher_provenance_hash=teacher["provenance_hash"],
        artifacts_sha256={name: bind(output / name)["sha256"] for name in ["fits/synthetic.npz", "quality.jsonl"]},
        source_lidar_sha256={},
    )
    write(output / "fit_report.json", cached)
    union = dict(
        schema="badminton-new-source-repair-input-v1",
        source_root=str(source),
        source_registry_sha256=source_registry["sha256"],
        session_splits=deepcopy(policy.roles),
        suggested_session_splits={role: dict(sessions=ids) for role, ids in policy.roles.items()},
        required_nonempty_splits=policy.required,
        release_policy=selected,
        jobs=[dict(source_id="fresh", source_window=str(original), window_json_sha256=original_config["sha256"])],
    )
    union_path = root / "new_source_input.json"
    write(union_path, union)
    evidence = write(tmp_path / "identity-source.json", dict(synthetic=True))
    identity_path = root / "identity.json"

    def refresh():
        write(union_path, union)
        write(
            identity_path,
            dict(
                review_status="reviewed",
                evidence_files={evidence["path"]: evidence["sha256"]},
                windows={"fresh:" + name: {}},
                source_union_sha256=bind(union_path)["sha256"],
                calibration_digest="frozen",
            ),
        )
        plan = dict(
            schema="badminton-body-repair-plan-v1",
            jobs=union["jobs"],
            source_union=str(union_path),
            source_union_sha256=bind(union_path)["sha256"],
            calibration_digest="frozen",
            fitting_config=asdict(FittingConfig(distal_pose_prior_weight=0.03, lower_limb_reprojection_weight=1.0)),
            training_published=False,
        )
        if union.get("review_only") is True:
            plan.update(review_only=True, purpose=union.get("purpose", "short_native_rally_review"))
        write(labels / "repair_plan.json", plan)

    refresh()
    body = tool("repair_badminton_body_labels")
    apply = tool("apply_badminton_leg_refinement")
    for module in (body, apply):
        monkeypatch.setattr(module.CalibrationSnapshot, "load", lambda path: cal)
        monkeypatch.setattr(module, "BadmintonPipeline", StoredPipeline)
        monkeypatch.setattr(module, "load_pilot_run_relocations", lambda *args: None)
        monkeypatch.setattr(module, "load_run_session", forbidden)
    monkeypatch.setattr(body, "PROJ_ROOT", project)
    monkeypatch.setattr(body, "RayFilteredPoseProvider", lambda *args: pose)
    monkeypatch.setattr(body, "SMPLFitter", forbidden)
    monkeypatch.setattr(body, "run_teacher", lambda *args, **kwargs: teacher)
    write(labels / "repair_receipt.json", dict(state="body_qa_candidate_complete", windows=1))
    write(
        root / "leg_refinement_selection.json",
        dict(
            state="complete",
            windows={"fresh:" + name: dict(selected=[])},
            selected_frames=0,
            refinement_policy_sha256=bind(PROJ_ROOT / "hmr4d/lidar_mocap/leg_refinement.py")["sha256"],
            selection_policy_sha256=bind(PROJ_ROOT / "hmr4d/lidar_mocap/leg_selection.py")["sha256"],
        ),
    )
    return SimpleNamespace(
        project=project,
        root=root,
        labels=labels,
        source=source,
        union=union,
        union_path=union_path,
        original=original,
        output=output,
        identity=identity_path,
        body=body,
        apply=apply,
        refresh=refresh,
    )


def run_body(stage, monkeypatch, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "body",
            "--union",
            str(stage.union_path),
            "--source-root",
            str(stage.source),
            "--output-root",
            str(stage.labels),
            "--stage",
            "fit",
            "--require-completed",
            "--native-poses-root",
            str(stage.root),
            "--joint-ray-filter",
            "--identity-decisions",
            str(stage.identity),
            "--device",
            "cpu",
            "--threads",
            "1",
            *extra,
        ],
    )
    stage.body.main()


def run_apply(stage, monkeypatch, *, review_only=False):
    if not review_only:
        # No real body asset is loaded, and the selected numerical work is empty.
        monkeypatch.setitem(
            sys.modules,
            "smplx",
            SimpleNamespace(create=lambda *args, **kwargs: SimpleNamespace(faces=np.zeros((1, 3), np.int64))),
        )
        monkeypatch.setattr(stage.apply.torch, "as_tensor", lambda *args, **kwargs: None)
        monkeypatch.setattr(stage.apply.torch, "load", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["apply", "--root", str(stage.root), "--threads", "1", *(["--review-sessions-only"] if review_only else [])],
    )
    stage.apply.main()


@pytest.mark.parametrize("entry", ["body", "apply"])
def test_canonical_cli_train_only_uses_real_strict_policy_and_original_aggregate(stage, monkeypatch, entry):
    before = bind(stage.output / "fits/synthetic.npz")
    if entry == "body":
        run_body(stage, monkeypatch)
        receipt = read(stage.labels / "repair_receipt.json")
        assert receipt["state"] == "body_qa_candidate_complete" and receipt["windows"] == 1
        assert receipt["body_qa_only"] is True
    else:
        run_apply(stage, monkeypatch)
        receipt = read(stage.root / "leg_refinement_receipt.json")
        assert receipt["state"] == "complete" and receipt["selected_frames"] == receipt["updated_frames"] == 0
    result = read(stage.labels / "datasets/dataset_manifest.json")
    assert result["schema"] == "badminton-ptv3-dataset-v1"
    assert result["required_nonempty_splits"] == ["train"]
    assert result["accepted"] == 1 and set(result["exports"]) == {"fresh"}
    assert {name: item["sessions"] for name, item in result["suggested_session_splits"].items()} == stage.union[
        "session_splits"
    ]
    assert require_nonempty_splits(result) == dict(train=1)
    assert result["training_published"] is False and before == bind(stage.output / "fits/synthetic.npz")
    assert not (stage.labels / "repair_progress.json").exists()


@pytest.mark.parametrize("entry", ["body", "apply"])
@pytest.mark.parametrize(
    "failure", ["roles", "required", "suggested", "extra_source", "policy_sha", "null_policy", "review"]
)
def test_explicit_bad_contract_stops_before_models_or_writes(stage, monkeypatch, entry, failure):
    if failure == "roles":
        stage.union["session_splits"] = dict(train=[], val=["fresh"], test=[])
    elif failure == "required":
        stage.union["required_nonempty_splits"] = ["train", "val", "test"]
    elif failure == "suggested":
        stage.union["suggested_session_splits"]["val"]["sessions"] = ["fresh"]
    elif failure == "extra_source":
        registry = dict(sources=dict(fresh={}, unknown={}))
        selected = write(stage.source / "source_registry.json", registry)
        write(stage.labels / "source_registry.json", registry)
        stage.union["source_registry_sha256"] = selected["sha256"]
    elif failure == "policy_sha":
        stage.union["release_policy"]["sha256"] = "0" * 64
    elif failure == "null_policy":
        stage.union["release_policy"] = None
    else:
        stage.union["review_only"] = True
    stage.refresh()
    module = getattr(stage, entry)
    monkeypatch.setattr(module.CalibrationSnapshot, "load", forbidden)
    monkeypatch.setattr(module, "write_json", forbidden)
    with pytest.raises(ValueError):
        run_body(stage, monkeypatch) if entry == "body" else run_apply(stage, monkeypatch, review_only=True)
    assert not (stage.labels / "datasets/dataset_manifest.json").exists()
    assert not (stage.root / "leg_refinement_receipt.json").exists()


@pytest.mark.parametrize(
    "failure", ["missing_fit", "missing_teacher", "artifact", "algorithm", "settings", "provenance"]
)
def test_completed_only_does_not_refit_missing_or_stale_window(stage, monkeypatch, failure):
    report = stage.output / "fit_report.json"
    if failure == "missing_fit":
        report.unlink()
    elif failure == "missing_teacher":
        (stage.output / "teacher/manifest.json").unlink()
    elif failure == "artifact":
        with (stage.output / "fits/synthetic.npz").open("ab") as handle:
            handle.write(b"changed")
    else:
        value = read(report)
        value[
            {"algorithm": "fit_algorithm_digest", "settings": "settings", "provenance": "teacher_provenance_hash"}[
                failure
            ]
        ] = "stale"
        write(report, value)
    # Remove the setup baseline receipt; failure must never emit a replacement.
    (stage.labels / "repair_receipt.json").unlink()
    with pytest.raises(ValueError, match="Completed-only"):
        run_body(stage, monkeypatch)
    assert not (stage.labels / "repair_receipt.json").exists()
    assert not (stage.labels / "datasets/dataset_manifest.json").exists()


@pytest.mark.parametrize("entry", ["body", "apply"])
def test_no_policy_keeps_legacy_nonempty_three_roles(stage, monkeypatch, entry):
    stage.union.pop("release_policy")
    stage.refresh()
    (stage.labels / "repair_receipt.json").unlink()
    if entry == "apply":
        write(stage.labels / "repair_receipt.json", dict(windows=1))
    with pytest.raises(ValueError):
        run_body(stage, monkeypatch) if entry == "body" else run_apply(stage, monkeypatch)
    if entry == "body":
        assert not (stage.labels / "repair_receipt.json").exists()
    else:
        assert not (stage.root / "leg_refinement_receipt.json").exists()


def test_body_review_only_path_keeps_review_group_and_no_training_roles(stage, monkeypatch):
    stage.union.pop("release_policy")
    stage.union.update(
        review_only=True,
        session_splits=dict(review=["fresh"]),
        suggested_session_splits=dict(review=dict(sessions=["fresh"])),
        required_nonempty_splits=[],
    )
    stage.refresh()
    run_body(stage, monkeypatch)
    result = read(stage.labels / "datasets/dataset_manifest.json")
    assert result["schema"] == "badminton-review-only-body-labels-v1" and result["review_only"] is True
    assert result["required_nonempty_splits"] == [] and set(result["suggested_session_splits"]) == {"review"}
    assert result["accepted"] == 1 and result["training_published"] is False


def test_apply_review_sessions_entry_validates_policy_without_model_assets(stage, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "smplx", SimpleNamespace(create=forbidden))
    monkeypatch.setattr(stage.apply.torch, "load", forbidden)
    run_apply(stage, monkeypatch, review_only=True)
    assert "review_session_contracts_validated" in capsys.readouterr().out
    assert not (stage.labels / "post_refinement.json").exists()


def test_original_prepare_accepts_the_policy_bound_prepared_input_without_models(stage, monkeypatch, capsys):
    before = bind(stage.labels / "repair_receipt.json")
    monkeypatch.setattr(stage.body, "RayFilteredPoseProvider", forbidden)
    monkeypatch.setattr(stage.body, "run_teacher", forbidden)
    run_body(stage, monkeypatch, "--stage", "prepare")
    assert "Prepared 1 frozen windows; no inference or training publication" in capsys.readouterr().out
    plan = read(stage.labels / "repair_plan.json")
    assert plan["source_union_sha256"] == bind(stage.union_path)["sha256"] and len(plan["jobs"]) == 1
    assert bind(stage.labels / "repair_receipt.json") == before
    assert not (stage.labels / "datasets/dataset_manifest.json").exists()


def test_explicit_train_still_requires_real_accepted_samples(stage, monkeypatch):
    row = read(stage.output / "quality.jsonl")
    row["qa"]["final_p2surface_mean_m"] = 0.2
    (stage.output / "quality.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    cached = read(stage.output / "fit_report.json")
    cached["artifacts_sha256"]["quality.jsonl"] = bind(stage.output / "quality.jsonl")["sha256"]
    write(stage.output / "fit_report.json", cached)
    (stage.labels / "repair_receipt.json").unlink()
    with pytest.raises(ValueError, match="No accepted body samples"):
        run_body(stage, monkeypatch)
    assert not (stage.labels / "repair_receipt.json").exists()
    assert read(stage.labels / "datasets/dataset_manifest.json")["accepted"] == 0
