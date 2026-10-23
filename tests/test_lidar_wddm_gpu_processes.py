"""The real WDDM failure must not weaken foreign-CUDA/PID-reuse exclusion."""

from datetime import datetime, timedelta, timezone
import ctypes
import json
import os
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


@pytest.mark.parametrize("kernel_path", ["C:/Windows/explorer.exe", "C:/Other/explorer.exe", "C:/Python/python.exe"])
def test_access_denied_uses_actual_kernel_image_but_still_requires_exact_frozen_executable(
        desktop, monkeypatch, kernel_path):
    def denied():
        raise psutil.AccessDenied(100)

    desktop.process.exe = denied
    queries = []
    monkeypatch.setattr(execution, "_kernel_process_executable", lambda pid: queries.append(pid) or kernel_path)
    expected = {200} if kernel_path == desktop.entry["executable"] else {100, 200}
    assert execution.compute_pids() == expected and queries == [100]


@pytest.mark.parametrize("error", [OSError("unknown NTSTATUS"), ValueError("unmapped device prefix")])
def test_unverifiable_kernel_image_remains_a_blocking_gpu_identity(desktop, monkeypatch, error):
    def denied():
        raise psutil.AccessDenied(100)

    def failed(pid):
        raise error

    desktop.process.exe = denied
    monkeypatch.setattr(execution, "_kernel_process_executable", failed)
    assert execution.compute_pids() == {100, 200}


def test_readable_process_image_preserves_psutil_path_without_kernel_fallback(monkeypatch):
    process = SimpleNamespace(create_time=lambda: 50.0, exe=lambda: "C:/Windows/explorer.exe")
    monkeypatch.setattr(execution.psutil, "Process", lambda pid: process)

    def forbidden(pid):
        pytest.fail("Readable process paths must keep the ordinary image query")

    monkeypatch.setattr(execution, "_kernel_process_executable", forbidden)
    assert execution.process_executable(100, expected_create_time=50.0) == "C:/Windows/explorer.exe"


@pytest.mark.parametrize("created", [49.0, 51.0])
def test_process_image_rechecks_fresh_pid_creation_time_and_rejects_reuse(monkeypatch, created):
    original = SimpleNamespace(create_time=lambda: 50.0, exe=lambda: "C:/Windows/explorer.exe")
    replacement = SimpleNamespace(create_time=lambda: created)
    processes = iter([original, replacement])
    monkeypatch.setattr(execution.psutil, "Process", lambda pid: next(processes))
    with pytest.raises(psutil.NoSuchProcess, match="during its image query"):
        execution.process_executable(100, expected_create_time=50.0)


def test_wrong_expected_creation_time_is_rejected_before_any_image_read(monkeypatch):
    process = SimpleNamespace(create_time=lambda: 51.0, exe=lambda: pytest.fail("Stale PID must not be read"))
    monkeypatch.setattr(execution.psutil, "Process", lambda pid: process)
    with pytest.raises(psutil.NoSuchProcess, match="before its image query"):
        execution.process_executable(100, expected_create_time=50.0)


@pytest.mark.parametrize("problem", [None, "status", "empty", "prefix", "unmapped", "odd_length"])
def test_kernel_image_requires_actual_successful_query_and_exact_device_mapping(monkeypatch, problem):
    native = r"\Device\Volume7\Windows\explorer.exe"
    if problem == "prefix":
        native = r"\Unknown\Volume7\Windows\explorer.exe"

    class Query:
        def __call__(self, information_class, pointer, size, returned):
            assert information_class == 88 and size == ctypes.sizeof(execution._ProcessImage)
            item = ctypes.cast(pointer, ctypes.POINTER(execution._ProcessImage)).contents
            assert item.pid == 100
            encoded = native.encode("utf-16-le")
            ctypes.memmove(item.image.buffer, encoded, len(encoded))
            item.image.length = 0 if problem == "empty" else len(encoded) - (problem == "odd_length")
            return -1 if problem == "status" else 0

    class Devices:
        def __call__(self, drive, buffer, length):
            if drive == "R:" and problem != "unmapped":
                buffer.value = r"\Device\Volume7"
                return len(buffer.value) + 2
            return 0

    libraries = dict(ntdll=SimpleNamespace(NtQuerySystemInformation=Query()),
                     kernel32=SimpleNamespace(QueryDosDeviceW=Devices()))
    monkeypatch.setattr(execution.ctypes, "WinDLL", lambda name, **kwargs: libraries[name])
    if problem is None:
        assert execution._kernel_process_executable(100) == r"R:\Windows\explorer.exe"
    else:
        with pytest.raises((OSError, ValueError)):
            execution._kernel_process_executable(100)


def test_actual_windows_kernel_image_query_matches_current_real_process_image():
    if os.name != "nt":
        pytest.skip("Actual kernel image query requires Windows")
    process = psutil.Process()
    actual = execution._kernel_process_executable(process.pid)
    assert os.path.normcase(actual) == os.path.normcase(process.exe())
    assert execution.process_executable(process.pid, expected_create_time=process.create_time()) == process.exe()


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


def test_fresh_grant_desktop_binding_can_follow_a_long_cpu_pass_without_rewriting_implementation(desktop):
    path = desktop.folder / "implementation_old_snapshot.json"
    path.write_text(json.dumps(dict(desktop_allowance=dict(path="old", sha256="0" * 64),
                                   desktop_allowance_fixture_only=True)), encoding="utf-8")
    before = path.read_bytes()
    checked = dict(request_binding=desktop.policy["request"], authorization=dict(grant=dict(
        independent_guard=binding(path), desktop_allowance=binding(desktop.path),
        desktop_allowance_fixture_only=False,
        approved_deadline_utc=(datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat())))
    execution.configure_desktop_allowance(checked)
    assert execution.DESKTOP_ALLOWANCE == binding(desktop.path)
    assert path.read_bytes() == before
    assert execution.compute_pids() == {200}


@pytest.mark.parametrize("change", ["fixture", "stale", "short_deadline", "wrong_request"])
def test_independent_grant_desktop_still_requires_real_fresh_exact_full_window_evidence(desktop, change):
    implementation = desktop.folder / "implementation.json"
    implementation.write_text("{}", encoding="utf-8")
    if change == "stale":
        desktop.policy["observed_utc"] = (datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat()
    if change == "short_deadline":
        desktop.policy["valid_until_utc"] = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    if change == "wrong_request":
        desktop.policy["request"] = dict(path="wrong", sha256="0" * 64)
    desktop.put()
    checked = dict(request_binding=binding(desktop.request), authorization=dict(grant=dict(
        independent_guard=binding(implementation), desktop_allowance=binding(desktop.path),
        desktop_allowance_fixture_only=change == "fixture",
        approved_deadline_utc=(datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat())))
    with pytest.raises(ValueError):
        execution.configure_desktop_allowance(checked)
