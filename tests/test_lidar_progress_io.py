"""Real tiny Windows sharing tests; no model, CUDA or production inputs."""

from contextlib import contextmanager
import os
import threading
import time

import pytest

from hmr4d.lidar_mocap import progress_io as io


@contextmanager
def deny_delete(path):
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    handle = create(str(path), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        assert close(handle)


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows sharing semantics")
def test_denied_replace_retries_until_real_reader_releases(tmp_path):
    target, temporary = tmp_path / "progress.json", tmp_path / "progress.json.next"
    target.write_bytes(b"old")
    temporary.write_bytes(b"new exact bytes")
    done, errors = threading.Event(), []

    def writer():
        try:
            io.atomic_replace(temporary, target)
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()

    with deny_delete(target):
        thread = threading.Thread(target=writer)
        thread.start()
        assert not done.wait(0.18)
        assert target.read_bytes() == b"old" and temporary.read_bytes() == b"new exact bytes"
    thread.join(timeout=4)
    assert done.is_set() and not thread.is_alive() and not errors
    assert target.read_bytes() == b"new exact bytes" and not temporary.exists()


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows sharing semantics")
def test_permanent_denial_is_bounded_and_preserves_both_files(tmp_path):
    target, temporary = tmp_path / "progress.json", tmp_path / "progress.json.next"
    target.write_bytes(b"old")
    temporary.write_bytes(b"pending")
    start = time.monotonic()
    with deny_delete(target), pytest.raises(OSError) as failure:
        io.atomic_replace(temporary, target, max_wait_seconds=0.12)
    elapsed = time.monotonic() - start
    assert failure.value.winerror in (5, 32) and 0.1 <= elapsed < 1
    assert target.read_bytes() == b"old" and temporary.read_bytes() == b"pending"


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows sharing semantics")
def test_shared_reader_permits_replace_and_retains_original_snapshot(tmp_path):
    target, temporary = tmp_path / "progress.json", tmp_path / "progress.json.next"
    target.write_bytes(b"old snapshot")
    temporary.write_bytes(b"new snapshot")
    with io.open_read_shared(target) as reader:
        io.atomic_replace(temporary, target, max_wait_seconds=0)
        assert reader.read() == b"old snapshot"
        assert io.read_bytes_shared(target) == b"new snapshot"
    assert reader.closed


def test_other_error_is_immediate_and_preserves_target(tmp_path):
    target = tmp_path / "progress.json"
    target.write_bytes(b"old")
    start = time.monotonic()
    with pytest.raises(FileNotFoundError):
        io.atomic_replace(tmp_path / "missing.next", target)
    assert time.monotonic() - start < 0.5 and target.read_bytes() == b"old"


def test_json_serialization_is_explicit_and_invalid_json_is_not_hidden(tmp_path):
    target = tmp_path / "progress.json"
    io.write_json(target, {"text": "中文", "value": 1}, allow_nan=False)
    assert io.read_bytes_shared(target) == '{\n  "text": "中文",\n  "value": 1\n}\n'.encode()
    assert io.read_json(target) == {"text": "中文", "value": 1}
    temporary = tmp_path / "custom.next"
    io.write_json(target, {"text": "中文"}, temporary=temporary, indent=None, ensure_ascii=True)
    assert io.read_bytes_shared(target) == b'{"text": "\\u4e2d\\u6587"}\n'
    assert not temporary.exists()
    before = target.read_bytes()
    with pytest.raises(ValueError):
        io.write_json(target, {"value": float("nan")}, allow_nan=False)
    assert target.read_bytes() == before
    target.write_bytes(b"invalid")
    with pytest.raises(ValueError):
        io.read_json(target)
