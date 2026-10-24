"""Atomic status writes and Windows readers that permit replacement.

Callers own the paths and serialization policy. Sharing/access denial preserves
the temporary file and previous target; only Windows errors 5/32 retry.
"""

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time


def _replace_once(temporary, target):
    if os.name != "nt" or not target.exists():
        temporary.replace(target)
        return
    import ctypes
    from ctypes import wintypes

    # MoveFileEx/Path.replace can reject an open destination even with share-delete.
    # ReplaceFileW supports replacement while our shared reader keeps its old file.
    replace = ctypes.WinDLL("kernel32", use_last_error=True).ReplaceFileW
    replace.argtypes = (
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    replace.restype = wintypes.BOOL
    if not replace(str(target), str(temporary), None, 0, None, None):
        raise ctypes.WinError(ctypes.get_last_error())


def atomic_replace(temporary, target, *, max_wait_seconds=3.0):
    """Replace once, retrying only WinError 5/32 for at most three seconds."""
    if (
        type(max_wait_seconds) not in (int, float)
        or not math.isfinite(max_wait_seconds)
        or not 0 <= max_wait_seconds <= 3.0
    ):
        raise ValueError("max_wait_seconds must be finite and between zero and three")
    temporary, target = Path(temporary), Path(target)
    deadline = time.monotonic() + max_wait_seconds
    delay = 0.02
    while True:
        try:
            _replace_once(temporary, target)
            return
        except OSError as error:
            remaining = deadline - time.monotonic()
            if os.name != "nt" or getattr(error, "winerror", None) not in (5, 32) or remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 0.1)


@contextmanager
def open_read_shared(path):
    """Yield a binary reader whose Windows handle shares read/write/delete."""
    path = Path(path)
    if os.name != "nt":
        with path.open("rb") as reader:
            yield reader
        return
    import ctypes
    from ctypes import wintypes
    import msvcrt

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
    handle = create(str(path), 0x80000000, 0x1 | 0x2 | 0x4, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close(handle)
        raise
    try:
        reader = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    with reader:
        yield reader


def read_bytes_shared(path):
    """Read actual bytes through a shared-delete handle."""
    with open_read_shared(path) as reader:
        return reader.read()


def read_json(path):
    """Parse actual shared-read bytes; malformed/missing input still raises."""
    return json.loads(read_bytes_shared(path))


def write_json(path, value, *, temporary=None, max_wait_seconds=3.0, indent=2, ensure_ascii=False, **json_options):
    """Write UTF-8 JSON plus newline, then replace; serialization stays explicit."""
    path = Path(path)
    temporary = Path(temporary) if temporary is not None else path.with_name(path.name + ".next")
    if temporary.absolute() == path.absolute():
        raise ValueError("Temporary progress path must differ from target")
    content = json.dumps(value, indent=indent, ensure_ascii=ensure_ascii, **json_options) + "\n"
    temporary.write_bytes(content.encode("utf-8"))
    atomic_replace(temporary, path, max_wait_seconds=max_wait_seconds)
