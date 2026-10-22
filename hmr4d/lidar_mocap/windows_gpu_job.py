"""Private Windows job ownership; children cannot execute before assignment."""

import ctypes
from ctypes import wintypes
from contextlib import contextmanager
import os
import subprocess
import threading
import time

import psutil


class WindowsGPUJob:
    def __init__(self, name, *, create=True):
        if os.name != "nt":
            raise RuntimeError("The guarded-leg executor requires Windows Job Objects")
        self.name, self.handle = name, None
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([wintypes.LPVOID, wintypes.LPCWSTR], wintypes.HANDLE),
            "OpenJobObjectW": ([wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR], wintypes.HANDLE),
            "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD], wintypes.BOOL),
            "QueryInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
                                           ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
            "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            "SetProcessAffinityMask": ([wintypes.HANDLE, ctypes.c_size_t], wintypes.BOOL),
            "SetPriorityClass": ([wintypes.HANDLE, wintypes.DWORD], wintypes.BOOL),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        }
        for key, (args, result) in signatures.items():
            function = getattr(self.kernel, key)
            function.argtypes, function.restype = args, result
        if create:
            ctypes.set_last_error(0)
            self.handle = self.kernel.CreateJobObjectW(None, name)
            error = ctypes.get_last_error()
            if not self.handle or error == 183:
                self.close()
                raise OSError(error, "Refuse an existing or unavailable private job")
            self._kill_on_close()
        else:
            self.handle = self.kernel.OpenJobObjectW(0x0004 | 0x0008, False, name)  # QUERY | TERMINATE
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())

    def _kill_on_close(self):
        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t),
                ("active", wintypes.DWORD), ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

        class Limits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", ctypes.c_ulonglong * 6),
                ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

        limits = Limits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway permission.
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def pids(self):
        capacity = 64
        while capacity <= 4096:
            buffer = ctypes.create_string_buffer(8 + ctypes.sizeof(ctypes.c_size_t) * capacity)
            written = wintypes.DWORD()
            if self.kernel.QueryInformationJobObject(self.handle, 3, buffer, len(buffer), ctypes.byref(written)):
                count = ctypes.c_uint32.from_buffer(buffer, 4).value
                if count > capacity:
                    capacity = count
                    continue
                values = (ctypes.c_size_t * count).from_buffer(buffer, 8)
                return list(values)
            if ctypes.get_last_error() != 234:
                raise ctypes.WinError(ctypes.get_last_error())
            capacity *= 2
        raise RuntimeError("Private job exceeded its bounded process inventory")

    def identities(self):
        values = []
        for pid in self.pids():
            try:
                process = psutil.Process(pid)
                values.append(dict(pid=pid, create_time=process.create_time(), command=process.cmdline()))
            except psutil.NoSuchProcess:
                continue
        return values

    def start(self, command, *, cwd, env, log, affinity):
        """Own and configure the suspended launcher before any Python/model code can run."""
        flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS | 0x00000004
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=flags, close_fds=True)
        # Popen owns the original kernel handle. Never reopen a possibly reused PID
        # to assign, configure, resume or terminate a process.
        handle = wintypes.HANDLE(int(child._handle))
        assigned = False
        try:
            created = psutil.Process(child.pid).create_time()
            if child.poll() is not None:
                raise RuntimeError("Cannot identify the suspended owned launcher")
            if not self.kernel.AssignProcessToJobObject(self.handle, handle):
                raise ctypes.WinError(ctypes.get_last_error())
            assigned = True
            mask = sum(1 << index for index in set(affinity))
            if (not self.kernel.SetProcessAffinityMask(handle, mask)
                    or not self.kernel.SetPriorityClass(handle, subprocess.BELOW_NORMAL_PRIORITY_CLASS)):
                raise ctypes.WinError(ctypes.get_last_error())
            if child.pid not in self.pids():
                raise RuntimeError("Suspended launcher is not in its private job")
            resume = ctypes.WinDLL("ntdll").NtResumeProcess
            resume.argtypes, resume.restype = [wintypes.HANDLE], ctypes.c_long
            status = resume(handle)
            if status != 0:
                raise OSError(status, "Could not resume the owned launcher")
            return child, dict(pid=child.pid, create_time=created, command=list(command),
                               affinity=affinity, assigned_before_resume=True)
        except BaseException:
            if assigned:
                self.terminate()
            else:
                child.terminate()  # Uses the original Popen handle, never a PID lookup.
            child.wait(timeout=5)
            raise

    def wait_empty(self, timeout=6):
        end = time.monotonic() + timeout
        while self.pids():
            if time.monotonic() >= end:
                raise RuntimeError("Private CUDA job did not actually exit")
            time.sleep(.05)

    @contextmanager
    def deadline_guard(self, seconds):
        """Terminate this kernel-owned job even when its controller is blocked in I/O."""
        stop = threading.Event()
        state = dict(fired=False, error=None)

        def enforce():
            if not stop.wait(max(0, seconds)):
                state["fired"] = True
                try:
                    self.terminate()
                except BaseException as error:
                    state["error"] = repr(error)

        timer = threading.Thread(target=enforce, name="owned-job-deadline", daemon=True)
        timer.start()
        try:
            yield state
        finally:
            stop.set()
            timer.join(timeout=2)
            if timer.is_alive():
                raise RuntimeError("Owned job deadline termination did not return")

    def terminate(self):
        if self.handle and not self.kernel.TerminateJobObject(self.handle, 73):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
