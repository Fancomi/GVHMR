"""Synthetic CPU adapter mathematics, typed relocation preflight and witnessed resume contracts."""

import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import psutil
import pytest
import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap import formal_completion_policy_compat as compat
from hmr4d.lidar_mocap import formal_native_derive
from hmr4d.lidar_mocap import new_source_completion as consumer
from tools import verify_badminton_formal_dataset as dataset
from test_lidar_formal_baseline import bytes_file, put, read, stage
from test_lidar_formal_completion import finished, pending
from test_lidar_formal_policy_compat import producer_finished, durable_bundle
from test_lidar_relocations import manifest


def test_entire_original_numerical_loop_ast_is_identical_except_explicit_session_lookup():
    original = ast.parse((PROJ_ROOT / "tools/verify_badminton_dataset.py").read_text(encoding="utf-8"))
    adapter = ast.parse(Path(dataset.__file__).read_text(encoding="utf-8"))

    def loop(tree):
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Tuple)
            and [x.id for x in n.target.elts] == ["sid", "entry"]
        )
        node.body = [
            n
            for n in node.body
            if not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "session" for t in n.targets))
        ]
        return ast.dump(node, include_attributes=False)

    assert loop(original) == loop(adapter)
    assert compat._sha(PROJ_ROOT / "tools/verify_badminton_dataset.py") == dataset.ORIGINAL_DATASET_SHA256


@pytest.fixture
def numerical_case(tmp_path, monkeypatch):
    root, data = tmp_path / "run", tmp_path / "DATA"
    data.mkdir()
    resolver = manifest(data, {})
    sids = ["source_a", "source_b"]
    indexes = {sid: put(data / (sid + ".json"), {"synthetic_source_index": True}) for sid in sids}
    handoff = put(
        root / "handoff.json",
        dict(
            schema="badminton-camera-admission-handoff-v1",
            camera_calibration_admitted=True,
            body_labels_production_admitted=False,
            training_published=False,
            source_indexes={
                sid: dict(file=Path(item["path"]).name, sha256=item["sha256"]) for sid, item in indexes.items()
            },
            data_root=str(data),
            accepted_bundle_sha256="b" * 64,
        ),
    )
    intake = put(
        root / "intake.json",
        dict(
            schema="badminton-formal-derived-intake-v1",
            execution_scope="all_effective_native_rallies",
            camera_calibration_admitted=True,
            body_labels_production_admitted=False,
            training_published=False,
            data_root=str(data),
            camera_admission_handoff=handoff,
        ),
    )
    registry = dict(
        schema="badminton-formal-derived-source-registry-v1",
        data_root=str(data),
        intake=intake,
        camera_admission_handoff=handoff,
        data_relocations=dict(resolver.binding),
        sources={sid: dict(source_index=item) for sid, item in indexes.items()},
    )
    put(root / "source_registry.json", registry)
    calibration = SimpleNamespace(
        data=dict(
            snapshot_digest="camera",
            configuration_files_sha256="b" * 64,
            sessions_root=str(data / "sessions"),
            sessions={sid: {} for sid in sids},
            data_relocations=dict(resolver.binding),
        )
    )
    monkeypatch.setattr(dataset.CalibrationSnapshot, "load", lambda path: calibration)
    calls = []
    cloud = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    exports = {}
    for sid in sids:
        fit = root / sid / "fit.npz"
        fit.parent.mkdir(parents=True)
        np.savez(fit, vertices=np.zeros((3, 3), dtype=np.float32))
        rows = [dict(lidar_index=0, source_segment_indices=[1], fit_path=str(fit))]
        metadata = root / sid / "rows.jsonl"
        metadata.write_text(json.dumps(rows[0]) + "\n")
        tensor = root / sid / "data.pth"
        joblib.dump(
            dict(
                pc=torch.tensor(cloud[[1]])[None],
                root_pose=torch.zeros((1, 3)),
                body_pose=torch.zeros((1, 69)),
                shape=torch.zeros((1, 10)),
                transl=torch.zeros((1, 3)),
            ),
            tensor,
        )
        exports[sid] = dict(accepted=1, path=str(tensor), metadata=str(metadata))
    dataset_manifest = root / "datasets/dataset_manifest.json"
    dataset_manifest.parent.mkdir(parents=True, exist_ok=True)
    dataset_manifest.write_text(json.dumps(dict(exports=exports)), encoding="utf-8")

    # The real formal resolver is used; only the synthetic source-index decoder is replaced.
    def source(index, data_root, *, expected_sha256, require_admitted, relocations):
        assert relocations is not None and dict(relocations.binding) == dict(resolver.binding)
        assert require_admitted is True and expected_sha256 == compat._sha(index)
        calls.append(Path(index).stem)
        return SimpleNamespace(pairs=lambda: [0], read_cloud=lambda pair: cloud)

    monkeypatch.setattr("hmr4d.lidar_mocap.rally_pipeline.SourceIndexedSession.load_source_index", source)

    class SyntheticSMPL:
        def eval(self):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(vertices=torch.zeros((1, 3, 3)), joints=torch.zeros((1, 24, 3)))

    def model(*args, **kwargs):
        assert calls == sids  # Every typed session was really checked before model construction.
        calls.append("synthetic_model")
        return SyntheticSMPL()

    monkeypatch.setattr(dataset.smplx, "create", model)
    return SimpleNamespace(
        root=root,
        registry=registry,
        calibration=calibration,
        calls=calls,
        exports=exports,
        args=SimpleNamespace(output_root=root, threads=1, samples_per_session=64, report=None),
    )


def test_real_typed_resolver_and_original_cpu_point_mesh_pelvis_calculations(numerical_case):
    value = numerical_case
    result = dataset.verify_dataset(value.args)
    assert set(result["sessions"]) == {"source_a", "source_b"}
    for row in result["sessions"].values():
        assert row == dict(
            exported=1,
            checked=1,
            max_source_cloud_error_m=0,
            max_rebuilt_mesh_error_m=0,
            max_pelvis_translation_error_m=0,
        )
    assert read(value.root / "datasets/verification.json") == result


@pytest.mark.parametrize("change", ["wrong_binding", "wrong_camera", "changed_intake", "foreign_source"])
def test_every_typed_contract_fails_before_any_model_or_verification_output(numerical_case, change):
    v = numerical_case
    if change == "wrong_binding":
        v.registry["data_relocations"]["sha256"] = "0" * 64
    elif change == "wrong_camera":
        v.calibration.data["configuration_files_sha256"] = "0" * 64
    elif change == "changed_intake":
        Path(v.registry["intake"]["path"]).write_bytes(b"{}")
    else:
        v.registry["sources"]["foreign"] = {}
    put(v.root / "source_registry.json", v.registry)
    with pytest.raises((ValueError, KeyError)):
        dataset.verify_dataset(v.args)
    assert "synthetic_model" not in v.calls
    assert not (v.root / "datasets/verification.json").exists()


@pytest.mark.parametrize("change", ["cloud", "mesh", "metadata_count"])
def test_original_numeric_mismatches_cannot_publish(numerical_case, change):
    v = numerical_case
    entry = v.exports["source_a"]
    if change == "cloud":
        data = joblib.load(entry["path"])
        data["pc"] += 0.001
        joblib.dump(data, entry["path"])
    elif change == "mesh":
        row = json.loads(Path(entry["metadata"]).read_text())
        np.savez(row["fit_path"], vertices=np.ones((3, 3), dtype=np.float32))
    else:
        Path(entry["metadata"]).write_text("")
    with pytest.raises(ValueError):
        dataset.verify_dataset(v.args)
    assert not (v.root / "datasets/verification.json").exists()


def test_later_session_failure_precedes_model_construction(numerical_case, monkeypatch):
    v = numerical_case
    original = dataset.load_run_session

    def load(root, calibration, sid, *, relocations):
        if sid == "source_b":
            raise ValueError("synthetic later session rejected")
        return original(root, calibration, sid, relocations=relocations)

    monkeypatch.setattr(dataset, "load_run_session", load)
    with pytest.raises(ValueError, match="later session"):
        dataset.verify_dataset(v.args)
    assert v.calls == ["source_a"] and not (v.root / "datasets/verification.json").exists()


@pytest.mark.parametrize(
    "flags",
    [
        ["--resume-native-proof", "missing.json"],
        ["--resume-native-proof", "missing.json", "--resume-native-proof-sha256", "0" * 64, "--metadata-only"],
    ],
)
def test_partial_or_ignored_resume_selection_rejected_before_reading_inputs(tmp_path, flags):
    required = [
        "--root",
        "missing",
        "--baseline",
        "missing",
        "--baseline-sha256",
        "0" * 64,
        "--leg-execution",
        "missing",
        "--leg-execution-sha256",
        "0" * 64,
        "--compatibility-implementation",
        "missing",
        "--compatibility-implementation-sha256",
        "0" * 64,
    ]
    result = subprocess.run(
        [sys.executable, "-B", str(PROJ_ROOT / "tools/finalize_badminton_formal_cpu_compat.py"), *required, *flags],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 2
        and "Native resume" in result.stderr
        or (result.returncode == 2 and "Metadata-only" in result.stderr)
    )


@pytest.fixture
def resumable(producer_finished, monkeypatch):
    v = producer_finished
    project = v.root.parent.parent
    bytes_file(project / "tools/verify_badminton_native_bound.py", b"# synthetic prior native entry, not inference\n")
    old_bundle = durable_bundle(v)
    old = read(old_bundle["path"])
    old["code"].pop("dataset_entry")
    old["code_archive"].pop("dataset_entry")
    tested = read(old["test_execution"]["path"])
    tested["code"] = deepcopy(old["code"])
    old["test_execution"] = put(Path(old["test_execution"]["path"]), tested)
    historical = v.root / "gpu_guarded_leg/old_implementation"

    # The real durable checker checks a complete synthetic archive, never a flag-only stand-in.
    def relocate(item, name):
        path = historical / name
        bytes_file(path, Path(item["path"]).read_bytes())
        return compat._binding(path)

    old["code_archive"] = {k: relocate(x, "code/" + k + ".py") for k, x in old["code_archive"].items()}
    old["test_execution"] = relocate(old["test_execution"], "test.json")
    old["test_outputs"] = {k: relocate(x, k + ".bin") for k, x in old["test_outputs"].items()}
    for entry in old["test_sources"]:
        entry["archived"] = relocate(entry["archived"], "test_source.py")
    old["original_checker"]["archived"] = relocate(old["original_checker"]["archived"], "original.py")
    previous = put(historical / "implementation.json", old)
    monkeypatch.setattr(compat, "HISTORICAL_CODE", {k: x["sha256"] for k, x in old["code"].items()})
    checked = compat._baseline(v.root, v.baseline_binding, preleg=False)
    attempt = v.repair / "evaluation/formal_postleg_cpu/attempts" / ("c" * 32)
    attempt.mkdir(parents=True)
    v.worker(checked, "native", attempt, 1)
    native = read(v.native["path"])
    native["actual_native_witnesses"] = [put(v.root / "synthetic_native_witness.json", dict(synthetic_only=True))]
    put(Path(v.native["path"]), native)
    v.native_calls.clear()
    refresh = []
    monkeypatch.setattr(
        formal_native_derive,
        "refresh_native_coverage",
        lambda root, postleg: (refresh.append(postleg) or read(root / "final_labels/native_verification.json")),
    )
    folder = v.root / "gpu_guarded_leg/recovery"
    copies = folder / "actual-predecessor"
    material = compat.inspect_policy_metadata(v.root, v.baseline_binding, v.execution_binding, v.tested)
    policy = dict(
        material,
        compatibility_code=old["code"],
        compatibility_implementation=previous,
        test_execution=old["test_execution"],
    )
    log = compat.CPU_BOOTSTRAP
    commands = []
    for name, args in [
        ("verify_badminton_native_bound.py", ["--root", str(v.repair), "--verified-formal-cache"]),
        (
            "verify_badminton_dataset.py",
            ["--output-root", str(v.repair / "final_labels"), "--samples-per-session", "64"],
        ),
    ]:
        commands.append(["synthetic-python", "-u", "-c", log, str(project / "tools" / name), *args, "--threads", "1"])
    command = [
        "synthetic-python",
        "-u",
        str(compat.compatibility_files()["entry"]),
        "--root",
        str(v.root),
        "--baseline",
        v.baseline_binding["path"],
        "--baseline-sha256",
        v.baseline_binding["sha256"],
        "--leg-execution",
        v.execution_binding["path"],
        "--leg-execution-sha256",
        v.execution_binding["sha256"],
        "--threads",
        "1",
        "--compatibility-implementation",
        previous["path"],
        "--compatibility-implementation-sha256",
        previous["sha256"],
    ]
    commands.insert(0, command)
    actors = [
        dict(pid=os.getpid(), create_time=psutil.Process().create_time() - (i + 1) * 100, command=c)
        for i, c in enumerate(commands)
    ]
    launcher = dict(actors[0], assigned_before_resume=True, affinity=[26, 27])
    start = dict(
        started_utc="2026-10-23T00:00:00+00:00",
        stage="postleg",
        launcher=launcher,
        entry=old["code"]["entry"],
        threads=1,
        affinity=[26, 27],
        cuda_visible_devices="-1",
        priority="BelowNormal",
    )
    stage_log = (
        'workers["dataset"] = _run_cpu_check(checked, "dataset", attempt, threads)\n'
        f'RuntimeError: Formal CPU dataset verification failed (1); see {attempt / "dataset.log"}\n'
    )
    dataset_log = (
        f'{project / "tools/verify_badminton_dataset.py"}\nThis run requires its explicitly bound data relocations\n'
    )
    allocation = dict(
        schema="gvhmr-formal-cpu-worker-v1", affinity=[26, 27], priority="BelowNormal", cuda_visible_devices="-1"
    )
    bindings = {}
    for role, text in [
        ("stage_log", stage_log),
        ("dataset_log", dataset_log),
        ("native_log", json.dumps(allocation) + "\n" + json.dumps(native) + "\n"),
    ]:
        path = copies / (role + ".log")
        bytes_file(path, text.encode())
        bindings[role] = compat._binding(path)
    failure = dict(
        stage="postleg",
        job_empty=True,
        gpu_started=False,
        training_published=False,
        failed_utc="2026-10-23T00:01:00+00:00",
        actual_log=bindings["stage_log"],
        observed_process_identities=actors,
    )
    for role, value in [("failure", failure), ("start", start), ("policy_metadata", policy), ("native_result", native)]:
        bindings[role] = put(copies / (role + ".json"), value)
    exited = put(
        folder / "exit.json",
        dict(
            schema="gvhmr-formal-cpu-predecessor-exit-v1",
            observed_utc="2026-10-23T00:02:00+00:00",
            failure=bindings["failure"],
            job_empty=True,
            gpu_started=False,
            process_identities=[
                dict(pid=x["pid"], create_time=x["create_time"], original_identity_absent=True) for x in actors
            ],
        ),
    )
    proof = dict(
        schema="gvhmr-formal-postleg-native-resume-v1",
        root=str(v.root),
        baseline=v.baseline_binding,
        leg_execution=v.execution_binding,
        failed_attempt=str(attempt),
        previous_implementation=previous,
        evidence=bindings,
        exit_readback=exited,
    )
    v.resume = put(folder / "native-resume.json", proof)
    v.resume_data = proof
    v.refresh = refresh
    v.resume_verify = lambda: compat.verify_formal_postleg_cpu(
        v.root, v.baseline_binding, v.execution_binding, compatibility_test_execution=v.tested, native_resume=v.resume
    )
    return v


def test_explicit_resume_reuses_witness_and_runs_strict_chain_and_new_dataset_once(resumable):
    v = resumable
    receipt = v.resume_verify()
    assert v.native_calls == ["dataset"] and v.refresh == [True]
    worker = receipt["workers"]["native"]
    assert worker["state"] == "reused_completed_native_witness" and "exit_code" not in worker and "pid" not in worker
    assert len(receipt["evidence"]) == 8 and receipt["workers"]["dataset"]["exit_code"] == 0
    reused, _ = consumer._reuse_formal_postleg(
        v.repair,
        read(v.repair / "final_labels/repair_plan.json"),
        v.repair / "evaluation/formal_postleg_cpu_receipt.json",
        compat._sha(v.repair / "evaluation/formal_postleg_cpu_receipt.json"),
        1,
    )
    assert reused == receipt and v.native_calls == ["dataset"] and v.refresh == [True, True]


@pytest.mark.parametrize(
    "change", ["extra", "baseline", "log_drift", "missing_exit", "busy", "wrong_native", "wrong_policy"]
)
def test_resume_cannot_hide_changed_or_incomplete_evidence(resumable, change):
    v = resumable
    proof = v.resume_data
    if change == "extra":
        proof["allow"] = True
    elif change == "baseline":
        proof["baseline"] = dict(proof["baseline"], sha256="0" * 64)
    elif change == "log_drift":
        Path(proof["evidence"]["native_log"]["path"]).write_bytes(b"changed")
    else:
        role = (
            "exit_readback"
            if change in ("missing_exit", "busy")
            else "native_result" if change == "wrong_native" else "policy_metadata"
        )
        item = proof[role] if role == "exit_readback" else proof["evidence"][role]
        doc = read(item["path"])
        if change == "missing_exit":
            doc["process_identities"].pop()
        elif change == "busy":
            doc["job_empty"] = False
        elif change == "wrong_native":
            doc["accepted_checked"] += 1
        else:
            doc["native_qa_complete"] = True
        new = put(Path(item["path"]), doc)
        if role == "exit_readback":
            proof[role] = new
        else:
            proof["evidence"][role] = new
    v.resume = put(Path(v.resume["path"]), proof)
    with pytest.raises(ValueError):
        v.resume_verify()
    assert v.native_calls == [] and not (v.repair / "evaluation/formal_postleg_cpu_receipt.json").exists()


def test_resume_still_rejects_original_strict_full_chain_failure(resumable, monkeypatch):
    v = resumable

    def fail(*args, **kwargs):
        raise ValueError("synthetic original strict chain rejected changed bytes")

    monkeypatch.setattr(formal_native_derive, "refresh_native_coverage", fail)
    with pytest.raises(ValueError, match="strict chain"):
        v.resume_verify()
    assert v.native_calls == []


@pytest.mark.parametrize("mode", ["success", "error", "allocation"])
def test_dataset_launcher_really_starts_cpu_child_and_checks_its_exit_and_allocation(tmp_path, monkeypatch, mode):
    entry = tmp_path / "synthetic_cpu_entry.py"
    entry.write_text(
        "import os, psutil, sys\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'\n"
        "assert psutil.Process().cpu_affinity()==[26,27]\n"
        + ("sys.exit(3)\n" if mode == "error" else "print('synthetic CPU entry finished')\n"),
        encoding="utf-8",
    )
    monkeypatch.setattr(compat, "compatibility_files", lambda: dict(dataset_entry=entry))
    if mode == "allocation":
        monkeypatch.setattr(
            compat, "CPU_BOOTSTRAP", compat.CPU_BOOTSTRAP.replace("affinity=p.cpu_affinity()", "affinity=[]")
        )
    directory = tmp_path / "attempt"
    directory.mkdir()
    if mode == "success":
        worker = compat._run_dataset_check(dict(repair=tmp_path / "repair"), directory, 1)
        assert worker["exit_code"] == 0 and worker["entry"] == compat._binding(entry)
        assert worker["allocation"]["affinity"] == [26, 27]
        assert "synthetic CPU entry finished" in Path(worker["log"]["path"]).read_text()
    else:
        with pytest.raises(RuntimeError):
            compat._run_dataset_check(dict(repair=tmp_path / "repair"), directory, 1)
