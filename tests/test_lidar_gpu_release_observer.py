import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from hmr4d import PROJ_ROOT

spec = importlib.util.spec_from_file_location("release_observer", PROJ_ROOT / "tools/release_badminton_full_gpu.py")
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


def test_transient_reader_lock_retries_exact_bytes(monkeypatch):
    path = Mock()
    path.read_bytes.side_effect = [PermissionError("sharing violation"), b'{"completed":221}']
    monkeypatch.setattr(observer, "Path", lambda value: value)
    pause = Mock()
    monkeypatch.setattr(observer.time, "sleep", pause)
    assert observer.read_state_bytes(path) == b'{"completed":221}'
    pause.assert_called_once_with(0.05)


def test_persistent_reader_lock_remains_bounded(monkeypatch):
    path = Mock()
    path.read_bytes.side_effect = PermissionError("persistent sharing violation")
    monkeypatch.setattr(observer, "Path", lambda value: value)
    pause = Mock()
    monkeypatch.setattr(observer.time, "sleep", pause)
    with pytest.raises(PermissionError):
        observer.read_state_bytes(path, attempts=3)
    assert path.read_bytes.call_count == 3 and pause.call_count == 2


@pytest.fixture
def run_case(tmp_path, monkeypatch):
    root = tmp_path / "outputs/run"
    root.mkdir(parents=True)
    monkeypatch.setattr(observer, "PROJ_ROOT", tmp_path)

    def put(name, value):
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    put("pilot_input.json", dict(manifest=dict(sha256="a" * 64)))
    lease = put(
        "lease.json",
        dict(status="held", owner="GVHMR", manifest_sha256="a" * 64, approved_deadline_utc="2026-10-22T12:32:14Z"),
    )
    progress = put(
        "progress.json",
        dict(state="yielded_before_deadline", safe_to_release_gpu=True, completed_camera_windows=221, exit_code=0),
    )
    receipt = root / "release.json"
    seed = put(
        "seed.json",
        dict(
            root=str(root),
            lease_before=observer.binding(lease),
            observed_process_identities=[dict(pid=111, create_time=5.0)],
        ),
    )
    current = Mock()
    current.create_time.return_value = 10.0

    def process(pid=None):
        if pid in (None, observer.os.getpid()):
            return current
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(observer.psutil, "Process", process)
    monkeypatch.setattr(observer, "observe_owned", lambda root: {})
    monkeypatch.setattr(observer.time, "sleep", lambda seconds: None)

    def invoke(seed_enabled=True):
        args = [
            "release.py",
            "--root",
            str(root),
            "--lease",
            str(lease),
            "--progress",
            str(progress),
            "--receipt",
            str(receipt),
            "--kind",
            "teacher",
        ]
        if seed_enabled:
            args += ["--seed-evidence", str(seed), "--seed-evidence-sha256", observer.binding(seed)["sha256"]]
        monkeypatch.setattr("sys.argv", args)
        observer.main()

    return SimpleNamespace(**locals())


def test_seeded_original_identity_really_exited_before_lease_release(run_case):
    case = run_case
    before = case.progress.read_bytes()
    case.invoke()
    receipt = json.loads(case.receipt.read_bytes())
    assert receipt["observed_process_identities"] == [dict(pid=111, create_time=5.0, exited=True)]
    assert Path(receipt["full_teacher_progress"]["path"]).read_bytes() == before
    journal = json.loads(Path(receipt["process_observations"]["path"]).read_bytes())
    assert journal["alive"] == [] and len(journal["observed_process_identities"]) == 1
    assert json.loads(case.lease.read_bytes())["status"] == "released"


def test_safe_flag_and_empty_process_scan_are_insufficient_without_any_original_identity(run_case, monkeypatch):
    def stop(seconds):
        raise RuntimeError("still observing")

    monkeypatch.setattr(observer.time, "sleep", stop)
    with pytest.raises(RuntimeError, match="still observing"):
        run_case.invoke(seed_enabled=False)
    assert not run_case.receipt.exists()
    assert json.loads(run_case.lease.read_bytes())["status"] == "held"


def test_live_original_identity_blocks_release_without_terminating_it(run_case, monkeypatch):
    actual = observer.psutil.Process
    live = Mock()
    live.create_time.return_value = 5.0
    monkeypatch.setattr(observer.psutil, "Process", lambda pid=None: live if pid == 111 else actual(pid))
    monkeypatch.setattr(observer.time, "sleep", Mock(side_effect=RuntimeError("still alive")))
    with pytest.raises(RuntimeError, match="still alive"):
        run_case.invoke()
    assert not run_case.receipt.exists()
    live.terminate.assert_not_called()
    assert json.loads(run_case.lease.read_bytes())["status"] == "held"


def test_reused_pid_is_distinguished_by_original_create_time(run_case, monkeypatch):
    actual = observer.psutil.Process
    reused = Mock()
    reused.create_time.return_value = 6.0
    monkeypatch.setattr(observer.psutil, "Process", lambda pid=None: reused if pid == 111 else actual(pid))
    run_case.invoke()
    assert run_case.receipt.exists()
    reused.terminate.assert_not_called()


def test_recovery_seed_must_bind_the_actual_held_lease(run_case):
    value = json.loads(run_case.seed.read_bytes())
    value["lease_before"]["sha256"] = "b" * 64
    run_case.seed.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="exact held lease"):
        run_case.invoke()
    assert not run_case.receipt.exists()


def test_prior_journal_restores_original_observations_after_observer_restart(run_case):
    journal = run_case.receipt.with_suffix(".observed.json")
    journal.write_text(
        json.dumps(
            dict(
                root=str(run_case.root),
                lease=str(run_case.lease),
                observed_process_identities=[dict(pid=111, create_time=5.0)],
            )
        ),
        encoding="utf-8",
    )
    run_case.invoke(seed_enabled=False)
    assert json.loads(run_case.receipt.read_bytes())["observed_process_identities"] == [
        dict(pid=111, create_time=5.0, exited=True)
    ]


def test_original_observations_are_persisted_before_a_progress_read_can_fail(run_case, monkeypatch):
    actual = observer.read_state_bytes

    def locked(path, **kwargs):
        if path == run_case.progress:
            raise PermissionError("persistent progress lock")
        return actual(path, **kwargs)

    monkeypatch.setattr(observer, "read_state_bytes", locked)
    with pytest.raises(PermissionError):
        run_case.invoke()
    journal = json.loads(run_case.receipt.with_suffix(".observed.json").read_bytes())
    assert journal["observed_process_identities"] == [dict(pid=111, create_time=5.0)]
    assert not run_case.receipt.exists()
    assert json.loads(run_case.lease.read_bytes())["status"] == "held"
