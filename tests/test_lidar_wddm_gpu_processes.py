"""The real WDDM failure must not weaken foreign-CUDA/PID-reuse exclusion."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from hmr4d.lidar_mocap import guarded_leg_execution as execution
from hmr4d.lidar_mocap.pilot import binding


@pytest.fixture
def desktop(tmp_path, monkeypatch):
    folder = tmp_path / "guard/implementation_wddm_v2"
    folder.mkdir(parents=True)
    request = folder.parent / "guarded_leg_gpu_request.json"
    request.write_text("{}", encoding="utf-8")
    path = folder / "desktop_gpu_allowance.json"
    entry = dict(
        pid=100, create_time=50.0, executable="C:/Windows/explorer.exe", classification="root_observed_desktop_GUI"
    )
    policy = dict(
        schema="gvhmr-guarded-leg-wddm-desktop-identities-v1",
        state="frozen_observed_desktop_only",
        future_or_reused_PID_exempt=False,
        request=binding(request),
        desktop_identities=[entry],
        observed_utc=datetime.now(timezone.utc).isoformat(),
        valid_until_utc=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
    )

    def put():
        path.write_text(json.dumps(policy), encoding="utf-8")
        monkeypatch.setattr(execution, "DESKTOP_ALLOWANCE", binding(path))

    put()
    raw = [100, 200]

    def run(command, **kwargs):
        assert command[1] == "--id=0" and "--query-compute-apps=pid" in command
        assert kwargs["timeout"] == 2 and kwargs["check"] is True
        assert kwargs["creationflags"] & execution.subprocess.CREATE_NO_WINDOW
        return SimpleNamespace(stdout="\n".join(map(str, raw)))

    monkeypatch.setattr(execution.subprocess, "run", run)
    process = SimpleNamespace(create_time=lambda: 50.0, exe=lambda: "C:/Windows/explorer.exe")
    monkeypatch.setattr(execution.psutil, "Process", lambda pid: process)
    return SimpleNamespace(**locals())


def test_mixed_wddm_list_exempts_only_frozen_desktop_and_retains_python_or_unknown_cuda(desktop):
    assert execution.compute_pids() == {200}
    actual = execution.compute_pids.last_observation
    assert actual["raw_wddm_compute_pids"] == [100, 200]
    assert actual["frozen_desktop_exclusions"] == [desktop.entry]
    assert actual["blocking_compute_pids"] == [200]


def test_gui_only_list_is_not_misreported_as_a_training_process(desktop):
    desktop.raw[:] = [100]
    assert execution.compute_pids() == set()
    assert execution.compute_pids.last_observation["raw_wddm_compute_pids"] == [100]


def test_idle_foreign_cuda_still_blocks_without_any_utilization_or_memory_threshold(desktop):
    desktop.raw[:] = [200]
    assert execution.compute_pids() == {200}


def test_same_gui_name_in_a_new_pid_is_not_automatically_exempt(desktop):
    desktop.raw[:] = [101]
    assert execution.compute_pids() == {101}


def test_original_gui_pid_reuse_is_foreign_even_if_program_path_matches(desktop):
    desktop.process.create_time = lambda: 51.0
    assert execution.compute_pids() == {100, 200}


def test_same_pid_and_time_with_changed_executable_does_not_match(desktop):
    desktop.process.exe = lambda: "C:/Other/explorer.exe"
    assert execution.compute_pids() == {100, 200}


@pytest.mark.parametrize("error", [psutil.NoSuchProcess(100), psutil.AccessDenied(100)])
def test_unverifiable_or_stale_gui_identity_remains_blocking(desktop, monkeypatch, error):
    def denied(pid):
        raise error

    monkeypatch.setattr(execution.psutil, "Process", denied)
    assert execution.compute_pids() == {100, 200}


def test_windows_executable_path_case_is_normalized(desktop):
    desktop.process.exe = lambda: "c:/WINDOWS/Explorer.exe"
    assert execution.compute_pids() == {200}


@pytest.mark.parametrize("change", ["python", "reused_allowed", "expired", "wrong_request", "duplicate", "invalid_pid"])
def test_incomplete_or_weakened_desktop_policy_cannot_authorize_an_exemption(desktop, change):
    if change == "python":
        desktop.entry["executable"] = "C:/Python/python.exe"
    if change == "reused_allowed":
        desktop.policy["future_or_reused_PID_exempt"] = True
    if change == "expired":
        desktop.policy["valid_until_utc"] = "2000-01-01T00:00:00+00:00"
    if change == "wrong_request":
        desktop.policy["request"]["sha256"] = "0" * 64
    if change == "duplicate":
        desktop.policy["desktop_identities"].append(dict(desktop.entry))
    if change == "invalid_pid":
        desktop.entry["pid"] = True
    desktop.put()
    with pytest.raises(ValueError):
        execution.compute_pids()


def test_changed_policy_bytes_are_rejected_before_using_desktop_exemptions(desktop):
    desktop.path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        execution.compute_pids()


def test_missing_desktop_proof_fails_closed(desktop):
    desktop.path.unlink()
    with pytest.raises(FileNotFoundError):
        execution.compute_pids()


def test_unparseable_raw_smi_list_is_not_treated_as_free_gpu(desktop):
    desktop.raw[:] = ["N/A"]
    with pytest.raises(RuntimeError):
        execution.compute_pids()


def test_missing_issued_desktop_binding_cannot_probe_gpu_or_claim_exclusivity(monkeypatch):
    monkeypatch.setattr(execution, "DESKTOP_ALLOWANCE", None)

    def forbidden(*args, **kwargs):
        raise AssertionError("GPU query before an explicit binding")

    monkeypatch.setattr(execution.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="explicitly issued"):
        execution.compute_pids()


def test_actual_driver_loads_desktop_binding_from_its_issued_implementation_manifest(desktop):
    path = desktop.folder / "implementation.json"
    path.write_text(
        json.dumps(dict(desktop_allowance=binding(desktop.path), desktop_allowance_fixture_only=False)),
        encoding="utf-8",
    )
    checked = dict(
        request_binding=desktop.policy["request"],
        authorization=dict(
            grant=dict(
                independent_guard=binding(path),
                approved_deadline_utc=(datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
            )
        ),
    )
    execution.configure_desktop_allowance(checked)
    assert execution.DESKTOP_ALLOWANCE == binding(desktop.path)
    assert execution.compute_pids() == {200}


def test_final_desktop_capture_cannot_be_substituted_from_another_request(desktop):
    path = desktop.folder / "implementation.json"
    path.write_text(
        json.dumps(dict(desktop_allowance=binding(desktop.path), desktop_allowance_fixture_only=False)),
        encoding="utf-8",
    )
    checked = dict(
        request_binding=dict(path="other.json", sha256="0" * 64),
        authorization=dict(grant=dict(independent_guard=binding(path))),
    )
    with pytest.raises(ValueError, match="another actual guarded request"):
        execution.configure_desktop_allowance(checked)


@pytest.mark.parametrize("change", ["fixture_only", "stale_capture"])
def test_machine_fixture_or_stale_desktop_snapshot_cannot_start_actual_leg(desktop, change):
    if change == "stale_capture":
        desktop.policy["observed_utc"] = (datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat()
        desktop.put()
    path = desktop.folder / "implementation.json"
    path.write_text(
        json.dumps(
            dict(desktop_allowance=binding(desktop.path), desktop_allowance_fixture_only=change == "fixture_only")
        ),
        encoding="utf-8",
    )
    checked = dict(
        request_binding=desktop.policy["request"], authorization=dict(grant=dict(independent_guard=binding(path)))
    )
    with pytest.raises(ValueError):
        execution.configure_desktop_allowance(checked)


def test_desktop_policy_must_cover_the_complete_approved_leg_window(desktop):
    desktop.policy["valid_until_utc"] = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    desktop.put()
    path = desktop.folder / "implementation.json"
    path.write_text(
        json.dumps(dict(desktop_allowance=binding(desktop.path), desktop_allowance_fixture_only=False)),
        encoding="utf-8",
    )
    checked = dict(
        request_binding=desktop.policy["request"],
        authorization=dict(
            grant=dict(
                independent_guard=binding(path),
                approved_deadline_utc=(datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat(),
            )
        ),
    )
    with pytest.raises(ValueError, match="entire approved"):
        execution.configure_desktop_allowance(checked)
