"""In-process CPU proof retained by real Windows read handles; never a GPU grant.

The full existing readiness predicate runs once while its inputs are pinned.
Only the live owner can consume the resulting capability. Serialized receipts
are evidence for an issuer, and cannot recreate that capability in another PID.
"""

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import struct
import sys
import threading
import time
import uuid

import psutil

from hmr4d import PROJ_ROOT
from .pilot import binding, check_binding


_AUDIT = threading.local()
_AUDIT_INSTALLED = False
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_REPARSE = 0x400
_DIRECTORY = 0x10


class _FileInfo(ctypes.Structure):
    _fields_ = [("attributes", wintypes.DWORD), ("creation", wintypes.FILETIME),
                ("access", wintypes.FILETIME), ("write", wintypes.FILETIME),
                ("volume", wintypes.DWORD), ("size_high", wintypes.DWORD),
                ("size_low", wintypes.DWORD), ("links", wintypes.DWORD),
                ("index_high", wintypes.DWORD), ("index_low", wintypes.DWORD)]


class _FileIdInfo(ctypes.Structure):
    _fields_ = [("volume", ctypes.c_ulonglong), ("identifier", ctypes.c_ubyte * 16)]


def _kernel():
    if os.name != "nt":
        raise OSError("Verified read passes require actual Windows file sharing, without an unlocked fallback")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                  wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileInformationByHandle.argtypes = [wintypes.HANDLE, ctypes.POINTER(_FileInfo)]
    kernel.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    kernel.DeviceIoControl.restype = wintypes.BOOL
    return kernel


def _absolute(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or ":" in str(path)[2:]:
        raise ValueError("Read-pass inputs require absolute regular paths without traversal or alternate streams")
    return Path(os.path.normpath(path))


class WindowsReadLocks:
    """Leaf read handles deny data write/delete, including hard-link aliases.

    Ancestor handles protect names against directory rename. They do not protect
    children: every declared/stat-only/read file receives its own actual handle.
    VerifiedReadPass always rejects reparse/junction inputs. The optional
    junction branch below exists for synthetic Windows limitation probes.
    Existing directory junctions retain their own reparse handle and resolved
    entity handles. Attribute-only reparse changes remain possible and must be
    rejected by assert_held() at each grant/phase boundary. Other reparse types
    are rejected without an unlocked fallback.
    """

    def __init__(self, *, roots=None, maximum_handles=500000, allow_junctions=True):
        self.kernel = _kernel()
        self.roots = tuple(_absolute(path) for path in roots) if roots is not None else None
        self.maximum_handles = maximum_handles
        self.allow_junctions = allow_junctions
        self.files = {}
        self.directories = {}
        self.protected = set()
        self.closed = False

    def _info(self, handle):
        value = _FileInfo()
        if not self.kernel.GetFileInformationByHandle(handle, ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        identity = _FileIdInfo()
        if not self.kernel.GetFileInformationByHandleEx(handle, 18, ctypes.byref(identity), ctypes.sizeof(identity)):
            raise ctypes.WinError(ctypes.get_last_error())
        return dict(file_id=[identity.volume, bytes(identity.identifier).hex()],
                    bytes=(value.size_high << 32) | value.size_low, attributes=value.attributes)

    def _open(self, path, *, directory):
        if self.closed:
            raise ValueError("Read-pass handles have already been released")
        if len(self.files) + len(self.directories) >= self.maximum_handles:
            raise OSError("Read-pass handle limit reached; no input can remain unlocked")
        flags = 0x00200000 | (0x02000000 if directory else 0x80)
        # Ordinary ancestors must permit child temp.replace() used by the
        # unchanged lease/progress writer. Their own rename remains denied.
        handle = self.kernel.CreateFileW(str(path), 0x80000000, 3 if directory else 1, None, 3, flags, None)
        if handle == _INVALID_HANDLE:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            info = self._info(handle)
            if directory and info["attributes"] & _REPARSE:
                if not self.allow_junctions:
                    raise ValueError(
                        "Verified CPU inputs require canonical entity paths without junction/reparse aliases")
                strict = self.kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, flags, None)
                if strict == _INVALID_HANDLE:
                    raise ctypes.WinError(ctypes.get_last_error())
                self.kernel.CloseHandle(handle)
                handle = strict
                info = self._info(handle)
                if not info["attributes"] & _REPARSE:
                    raise ValueError("An input junction changed during acquisition of its strict read handle")
            if bool(info["attributes"] & _DIRECTORY) != directory:
                raise ValueError("Read-pass input has the wrong file type: " + str(path))
            item = dict(handle=handle, path=str(path), **info)
            if info["attributes"] & _REPARSE:
                if not directory:
                    raise ValueError("A read-pass input leaf must be an ordinary file: " + str(path))
                item["junction_sha256"] = self._junction_sha(handle)
            return item
        except BaseException:
            self.kernel.CloseHandle(handle)
            raise

    def _junction_sha(self, handle):
        payload = ctypes.create_string_buffer(16384)
        returned = wintypes.DWORD()
        if not self.kernel.DeviceIoControl(handle, 0x000900a8, None, 0, payload, len(payload),
                                           ctypes.byref(returned), None):
            raise ctypes.WinError(ctypes.get_last_error())
        data = payload.raw[:returned.value]
        if len(data) < 8 or struct.unpack_from("<I", data)[0] != 0xa0000003:
            raise ValueError("Only an existing ordinary directory junction can join a verified read path")
        return hashlib.sha256(data).hexdigest()

    def pin(self, path):
        path = _absolute(path)
        key = os.path.normcase(str(path))
        if key in self.files:
            return self.files[key]
        if self.roots is None:
            ancestors = list(reversed(path.parents))
        else:
            selected = [root for root in self.roots if path != root and path.is_relative_to(root)]
            if not selected:
                raise ValueError("Read-pass input leaves its explicitly selected lock roots")
            root = max(selected, key=lambda value: len(value.parts))
            ancestors = [root] + [parent for parent in reversed(path.parents)
                                  if parent != root and parent.is_relative_to(root)]
        for parent in ancestors:
            directory_key = os.path.normcase(str(parent))
            if directory_key not in self.directories:
                self.directories[directory_key] = self._open(parent, directory=True)
        item = self._open(path, directory=False)
        self.files[key] = item
        resolved = path.resolve(strict=True)
        if os.path.normcase(str(resolved)) != key:
            canonical = self.pin(resolved)
            if canonical["file_id"] != item["file_id"]:
                raise ValueError("An input alias changed during acquisition of its actual read handles")
        return item

    def assert_held(self):
        if self.closed:
            raise ValueError("A closed read pass cannot authorize execution")
        for item in self.files.values():
            current = self._info(item["handle"])
            if current != {key: item[key] for key in ("file_id", "bytes", "attributes")}:
                raise ValueError("An actual read-pass handle changed identity or metadata")
        for item in self.directories.values():
            current = self._info(item["handle"])
            expected_type = item["attributes"] & (_DIRECTORY | _REPARSE)
            if (current["file_id"] != item["file_id"]
                    or current["attributes"] & (_DIRECTORY | _REPARSE) != expected_type
                    or (item.get("junction_sha256") and self._junction_sha(item["handle"]) != item["junction_sha256"])):
                raise ValueError("An actual read-pass ancestor handle changed identity or file type")

    def release_exact(self, paths):
        """Release selected leaves only after proving no protected file-id alias."""
        keys = {os.path.normcase(str(_absolute(path))) for path in paths}
        if keys & self.protected:
            raise ValueError("An allowed FORMAL write is also an explicitly immutable input role")
        release = {key: value for key, value in self.files.items() if key in keys}
        protected = {tuple(item["file_id"]) for key, item in self.files.items() if key not in keys}
        if any(tuple(item["file_id"]) in protected for item in release.values()):
            raise ValueError("An allowed FORMAL write aliases an immutable or unselected input file-id")
        for key, item in release.items():
            if not self.kernel.CloseHandle(item["handle"]):
                raise ctypes.WinError(ctypes.get_last_error())
            del self.files[key]

    def close(self):
        for collection in (self.files, self.directories):
            for item in reversed(list(collection.values())):
                self.kernel.CloseHandle(item["handle"])
            collection.clear()
        self.closed = True
        self.protected.clear()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()


def _audit_open(event, arguments):
    observer = getattr(_AUDIT, "observer", None)
    if observer is None or getattr(_AUDIT, "busy", False):
        return
    if event == "open":
        path, mode, flags = arguments
        if isinstance(path, int):
            return  # Pipes/descriptors are not filesystem input names.
        if not isinstance(path, (str, bytes)):
            raise ValueError("Unrecognized file input during the verified CPU pass")
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise ValueError("The verified CPU predicate must not write a filesystem input")
        _AUDIT.busy = True
        try:
            observer(Path(os.fsdecode(path)).absolute())
        finally:
            _AUDIT.busy = False


@contextmanager
def observe_reads(observer):
    """Pin Python opens before reading; typed stat/C-extension coverage is separate."""
    global _AUDIT_INSTALLED
    if getattr(_AUDIT, "observer", None) is not None:
        raise ValueError("Nested verified CPU read scopes are not permitted")
    if not _AUDIT_INSTALLED:
        sys.addaudithook(_audit_open)
        _AUDIT_INSTALLED = True
    _AUDIT.observer = observer
    try:
        probe = []
        original = _AUDIT.observer
        _AUDIT.observer = lambda path: probe.append(path)
        sys.audit("open", str(Path(__file__).absolute()), "rb", os.O_RDONLY)
        _AUDIT.observer = original
        if not probe:
            raise RuntimeError("The required Python read observer was not actually installed")
        yield
    finally:
        _AUDIT.observer = None


def _baseline_paths(checked, readiness_binding):
    """Discover names from bound snapshots; content verification is a separate step."""
    from .guarded_leg_gpu import bound_json

    ready = bound_json(readiness_binding)
    baseline = bound_json(ready["evidence"]["formal_complete_baseline"])
    paths = {}
    for key, item in baseline["windows"].items():
        sid, name = key.split(":", 1)
        window = checked["root"] / "final_labels/sessions" / sid / "windows" / name
        report = bound_json(item["report_snapshot"])
        paths[str(window / "fit_report.json")] = item["report_snapshot"]["sha256"]
        for relative, sha in report["artifacts_sha256"].items():
            path = (window / relative).resolve()
            if Path(relative).is_absolute() or not path.is_relative_to(window.resolve()):
                raise ValueError("Baseline artifact leaves its exact window")
            paths[str(path)] = sha
    return paths


def baseline_artifacts(checked, readiness_binding):
    """Rehash current body bytes at each GPU boundary, without rebuilding context."""
    paths = _baseline_paths(checked, readiness_binding)
    for path, sha in paths.items():
        check_binding(dict(path=path, sha256=sha))
    return paths


def _stat_source_path(data_root, reference, resolver):
    """Resolve frozen names for pinning only; the unchanged deep reader checks SHA.

    DataRelocations.resolve deliberately hashes mapped bytes on every read. Using
    its frozen mapping here avoids adding another whole-source read just to find
    the names that must be pinned before that real validation.
    """
    from .relocations import logical_path, require_relocation_root

    require_relocation_root(data_root, resolver)
    logical = logical_path(reference["file"]).as_posix()
    mapped = resolver._entries.get(logical) if resolver is not None else None
    if resolver is not None and mapped is None and logical.casefold() in resolver._source_names:
        raise ValueError("Stat-only lock discovery uses a relocation reference case alias")
    if mapped is not None and (reference.get("sha256") != mapped["sha256"]
                               or ("bytes" in reference and reference["bytes"] != mapped["bytes"])):
        raise ValueError("Stat-only lock discovery differs from its frozen relocation binding")
    path = data_root / (mapped["file"] if mapped is not None else logical)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(data_root.resolve()) or not resolved.is_file():
        raise ValueError("Stat-only lock discovery escapes DATA or is not an ordinary file")
    if reference.get("bytes") is not None and path.stat().st_size != reference["bytes"]:
        raise ValueError("Stat-only lock discovery size differs from the source-index declaration")
    return path


def _stat_only_sources(ready, locks):
    """Expand every original/admitted source-index frame and fullraw member map."""
    from .guarded_leg_gpu import bound_json
    from .relocations import load_pilot_run_relocations

    imported = bound_json(ready["evidence"]["formal_numeric_import"])
    audit = bound_json(imported["audit"])
    equivalence = bound_json(audit["input_equivalence"])
    sources = {}
    for sid, entry in equivalence["sources"].items():
        for registry_binding, index_binding in (
            (entry["original_registry"], entry["original_index"]),
            (equivalence["evidence"]["formal_registry"], entry["admitted_index"]),
        ):
            registry_path = Path(registry_binding["path"])
            registry = bound_json(registry_binding)
            if registry["sources"][sid]["source_index"] != index_binding:
                raise ValueError("Read-pass stat-only expansion differs from its frozen registry")
            resolver = load_pilot_run_relocations(registry_path.parent)
            index = bound_json(index_binding)
            data_root = _absolute(registry["data_root"])
            declared = []
            for frame in index["frames"]:
                cloud = dict(file=frame["file"], bytes=frame["png_bytes"], sha256=frame["png_sha256"])
                declared.append(_stat_source_path(data_root, cloud, resolver))
                if index.get("parent_source_kind") == "original_fullraw_fixed24":
                    declared.append(_stat_source_path(data_root, frame["member_mapping"], resolver))
            for path in declared:
                locks.pin(path)
            key = sid + ":" + index_binding["path"]
            missing = sorted({str(path) for path in declared if os.path.normcase(str(path)) not in locks.files})
            if missing:
                raise ValueError("A declared stat-only source dependency has no actual held read handle")
            sources[key] = dict(index=index_binding, declared=len(declared), locked=len(declared) - len(missing),
                                paths=sorted({str(path) for path in declared}), missing=missing)
    return sources, audit


def _runtime_names():
    """Re-enumerate the exact native inventory names/versions without hashing videos."""
    from .native_witness import CUDA_DLL_MARKERS, SOURCE_FILES

    files, aliases = set(), {}

    def include_path(path):
        path = Path(os.path.abspath(path))
        resolved = str(path.resolve())
        files.add(resolved)
        if os.path.normcase(str(path)) != os.path.normcase(resolved):
            aliases[str(path)] = resolved

    for name in SOURCE_FILES:
        include_path(PROJ_ROOT / name)
    versions = {}
    for name in ("smplx", "torch", "numpy", "joblib", "scipy", "chumpy", "opencv-python", "einops", "opt-einsum"):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            if name != "opt-einsum":
                raise
            versions[name] = "not_installed"
            continue
        versions[name] = distribution.version
        for relative in distribution.files or ():
            declared = Path(distribution.locate_file(relative))
            path = declared.resolve()
            suffix = path.suffix.lower()
            include = path.suffix == ".py" if name == "opt-einsum" else suffix in (".py", ".pyd") or (
                suffix == ".dll" and (name != "torch" or not any(p in path.name.lower() for p in CUDA_DLL_MARKERS)))
            if include and path.is_file():
                include_path(declared)
    customized = sys.modules.get("sitecustomize")
    if customized is not None and getattr(customized, "__file__", None):
        include_path(customized.__file__)
    include_path(psutil.Process().exe())
    python_dll = Path(sys.base_prefix) / "python310.dll"
    if python_dll.exists():
        include_path(python_dll)
    include_path(PROJ_ROOT / "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl")
    return dict(files=sorted(files), versions=dict(versions, python=sys.version), aliases=aliases)


class VerifiedReadPass:
    """A one-use capability; only prepare() may construct a ready live instance."""

    def __init__(self):
        self.locks = None
        self.state = "new"
        self.nonce = uuid.uuid4().hex
        process = psutil.Process()
        self.owner = dict(pid=process.pid, create_time=process.create_time(), command=process.cmdline())
        self.receipt = None
        self.grant_binding = None

    def __getstate__(self):
        raise TypeError("A verified read capability cannot be serialized or restored from a receipt")

    def _live(self):
        process = psutil.Process()
        if self.state != "ready" or self.owner != dict(
                pid=process.pid, create_time=process.create_time(), command=process.cmdline()):
            raise ValueError("The verified read pass does not belong to this same live driver")
        self.locks.assert_held()

    @classmethod
    def prepare(cls, request_path, pregrant_path, pregrant_sha256):
        sys.dont_write_bytecode = True
        from .gpu_handoff import utc_time
        from .guarded_leg_gpu import bound_json, inspect_request, validate_readiness_inputs, verify_implementation
        from .repair_execution import write_progress_json

        result = cls()
        try:
            result.locks = WindowsReadLocks(allow_junctions=False)
            observed_reads = set()

            def pin_read(path):
                result.locks.pin(path)
                observed_reads.add(os.path.normcase(str(_absolute(path))))

            with observe_reads(pin_read):
                pregrant_binding = dict(path=str(_absolute(pregrant_path)), sha256=pregrant_sha256)
                bundle = bound_json(pregrant_binding)
                fields = {"schema", "state", "request", "runtime_contract", "readiness", "independent_guard",
                          "cpu_wait_deadline_utc", "gpu_started", "training_published"}
                if (set(bundle) != fields or bundle["schema"] != "gvhmr-guarded-leg-pregrant-v1"
                        or bundle["state"] != "cpu_inputs_only_not_authorized" or bundle["gpu_started"] is not False
                        or bundle["training_published"] is not False):
                    raise ValueError("An explicit CPU-only pregrant bundle is required")
                checked = inspect_request(request_path)
                if (bundle["request"] != checked["request_binding"]
                        or bundle["runtime_contract"] != checked["runtime_binding"]):
                    raise ValueError("Pregrant CPU inputs belong to another exact request/runtime contract")
                result.wait_deadline = utc_time(bundle["cpu_wait_deadline_utc"])
                if not datetime.now(timezone.utc) < result.wait_deadline <= utc_time(
                        checked["request"]["absolute_hard_deadline_utc"]):
                    raise ValueError("The CPU wait deadline must remain inside the unchanged request deadline")
                verify_implementation(bundle["independent_guard"], checked)
                ready = bound_json(bundle["readiness"])
                result.current_artifacts = _baseline_paths(checked, bundle["readiness"])
                for path in result.current_artifacts:
                    result.locks.pin(path)
                result.stat_sources, result.audit = _stat_only_sources(ready, result.locks)
                result.runtime_names = _runtime_names()
                for path in result.runtime_names.get("aliases", {}):
                    result.locks.pin(path)
                for path in result.runtime_names["files"]:
                    result.locks.pin(path)
                # The original entire deep predicate runs with the real inputs held.
                result.readiness = validate_readiness_inputs(
                    checked, bundle["readiness"], bundle["independent_guard"])
                if datetime.now(timezone.utc) >= result.wait_deadline:
                    raise TimeoutError("CPU verification did not complete before its separate waiting deadline")
                result.inputs = dict(request=bundle["request"], runtime_contract=bundle["runtime_contract"],
                                     readiness=bundle["readiness"], independent_guard=bundle["independent_guard"])
                result.pregrant = pregrant_binding
                result.root = checked["root"]
                result.checked = checked
                mutable = {os.path.normcase(path) for path in result.current_artifacts}
                result.locks.protected = set(result.locks.files) - mutable
                explicit = list(checked["request"]["scope"][0]["existing_input_and_code_bindings"].values())
                explicit += list(ready["evidence"].values())
                for role in ("parent_1019_identity", "parent_1020_identity", "formal_reviewed_identity"):
                    explicit += [dict(path=path, sha256=sha)
                                 for path, sha in bound_json(ready["evidence"][role])["evidence_files"].items()]
                for item in explicit:
                    path = _absolute(item["path"])
                    result.locks.pin(path)
                    result.locks.protected.add(os.path.normcase(str(path)))
                stat_paths = {os.path.normcase(path) for source in result.stat_sources.values()
                              for path in source["paths"]}
                observed_ids = {tuple(result.locks.files[key]["file_id"]) for key in observed_reads}
                result.lock_inventory = [dict(
                    path=item["path"], file_id=item["file_id"], bytes=item["bytes"],
                    python_read_observed=tuple(item["file_id"]) in observed_ids,
                    stat_only_declared=key in stat_paths, write_and_delete_denied=True)
                    for key, item in result.locks.files.items()]
                result.locks.assert_held()
            directory = Path(request_path).resolve().parent / "cpu_read_passes" / result.nonce
            directory.mkdir(parents=True, exist_ok=False)
            inventory_path = directory / "read_set.json"
            write_progress_json(inventory_path, dict(
                schema="gvhmr-guarded-leg-locked-read-set-v1", files=result.lock_inventory,
                stat_only_sources=result.stat_sources, runtime_names=result.runtime_names,
                content_verifier="unchanged_complete_readiness_predicate_under_held_handles",
                additional_content_hashes_computed=False))
            receipt_path = directory / "cpu_ready.json"
            write_progress_json(receipt_path, dict(
                schema="gvhmr-guarded-leg-live-cpu-read-pass-v1", state="cpu_ready_waiting_for_grant",
                driver=result.owner, nonce=result.nonce, pregrant=pregrant_binding, inputs=result.inputs,
                read_set=binding(inventory_path), completed_utc=datetime.now(timezone.utc).isoformat(),
                cpu_wait_deadline_utc=result.wait_deadline.isoformat(), gpu_started=False, training_published=False))
            result.receipt = binding(receipt_path)
            result.locks.pin(inventory_path)
            result.locks.pin(receipt_path)
            result.state = "ready"
            return result
        except BaseException:
            result.close()
            raise

    def validate_grant(self, checked):
        from .gpu_handoff import utc_time
        from .guarded_leg_gpu import bound_json, identity_alive

        self._live()
        grant = checked["authorization"]["grant"]
        check_binding(self.pregrant)
        check_binding(self.receipt)
        if (grant.get("verified_read_pass") != self.receipt
                or checked["request_binding"] != self.inputs["request"]
                or checked["runtime_binding"] != self.inputs["runtime_contract"]
                or any(grant.get(key) != self.inputs[key] for key in ("readiness", "independent_guard"))
                or not grant.get("desktop_allowance") or grant.get("desktop_allowance_fixture_only") is not False):
            raise ValueError("The fresh grant does not match this actual live CPU read-pass instance and inputs")
        receipt = bound_json(self.receipt)
        if receipt["driver"] != self.owner or receipt["nonce"] != self.nonce or not identity_alive(self.owner):
            raise ValueError("A serialized or stale driver receipt cannot authorize a different live instance")
        if checked["authorization"]["start"] < utc_time(receipt["completed_utc"]):
            raise ValueError("Actual GPU authorization must start after the complete CPU read pass")
        if self.grant_binding is None and datetime.now(timezone.utc) >= self.wait_deadline:
            raise TimeoutError("No fresh matching grant arrived before the separate CPU waiting deadline")
        if self.grant_binding is not None and checked["authorization"]["binding"] != self.grant_binding:
            raise ValueError("A live read pass cannot renew, extend or switch its accepted GPU grant")
        if _runtime_names() != self.runtime_names:
            raise ValueError("Native runtime inventory names or versions changed after CPU verification")
        for entry in self.audit["algorithms"].values():
            from .fit_reuse import audit_algorithm_compatibility
            current = audit_algorithm_compatibility(
                entry["original_digest"], entry["current_digest"], entry.get("original_commit"))
            if current != entry:
                raise ValueError("Original/current algorithm compatibility changed after CPU verification")
        exits = bound_json(self.readiness["evidence"]["predecessor_cuda_exit_bundle"])
        if any(identity_alive(item) for item in exits["process_identities"]):
            raise ValueError("A predecessor identity became live after CPU verification")
        self.grant_binding = checked["authorization"]["binding"]
        return self.readiness

    def check_baseline(self, checked):
        self._live()
        current = baseline_artifacts(checked, self.inputs["readiness"])
        if current != self.current_artifacts:
            raise ValueError("Current body baseline membership changed after CPU verification")
        return current

    def before_apply(self, checked):
        """Check real baseline bytes, then release only existing apply write targets."""
        from .guarded_leg_gpu import read_json

        self.validate_grant(checked)
        self.check_baseline(checked)
        release = set()
        with observe_reads(self.locks.pin):
            selection_path = self.root / "leg_refinement_selection.json"
            selection = read_json(selection_path)
            plan = read_json(self.root / "final_labels/repair_plan.json")
            expected = {job["source_id"] + ":" + Path(job["source_window"]).name for job in plan["jobs"]}
            if selection.get("state") != "complete" or set(selection.get("windows", {})) != expected:
                raise ValueError("Only the complete exact leg selection may consume a read pass")
            for key, item in selection["windows"].items():
                if not item["selected"]:
                    continue
                sid, name = key.split(":", 1)
                window = self.root / "final_labels/sessions" / sid / "windows" / name
                rows = [json.loads(line) for line in (window / "quality.jsonl").read_text(encoding="utf-8").splitlines()
                        if line]
                indexed = {(row["motion_id"], row["lidar_index"]): row for row in rows}
                for update in item["selected"]:
                    row = indexed[update["motion_id"], update["lidar_index"]]
                    path = _absolute(update["source_fit_path"])
                    if (path != _absolute(row["fit_path"]) or not path.is_relative_to(window / "fits")
                            or str(path) not in self.current_artifacts):
                        raise ValueError("Selected apply write is not its exact validated current FORMAL fit")
                    check_binding(dict(path=str(path), sha256=update["source_fit_sha256"]))
                    candidate = update["candidate"]
                    candidate_path = _absolute(candidate["arrays_path"])
                    if not candidate_path.is_relative_to(self.root / "leg_refinement_proposals"):
                        raise ValueError("Selected candidate leaves this exact leg proposal directory")
                    check_binding(dict(path=str(candidate_path), sha256=candidate["arrays_sha256"]))
                    release.add(path)
                for relative in ("fit_report.json", "ptv3.pth", "ptv3.metadata.jsonl", "quality.jsonl"):
                    release.add(window / relative)
                # make_review writes the same three available scan positions, including rejected rows.
                available = sorted({row["lidar_index"] for row in rows})
                count = min(3, len(available))
                positions = [int(i * (len(available) - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]
                for position in positions if available else ():
                    for camera in ("cam1", "cam2"):
                        release.add(window / "review" / f"{available[position]:06d}_{camera}.jpg")
                release.update(window / "review" / name for name in ("index.html", "summary.json"))
            original_union = read_json(plan["source_union"])
            sources = sorted({job["source_id"] for job in plan["jobs"]})
            if original_union.get("schema") == "badminton-new-source-repair-input-v1":
                sources = sorted({sid for values in original_union["session_splits"].values() for sid in values})
            for sid in sources:
                if not re.fullmatch(r"[0-9]{8}_[0-9]{6}", sid):
                    raise ValueError("Apply aggregate source leaves its exact session name")
                for suffix in (".pth", ".metadata.jsonl"):
                    release.add(self.root / "final_labels/datasets" / (sid + suffix))
            release.add(self.root / "final_labels/datasets/dataset_manifest.json")
            # Check aliases even for outputs not consumed by the deep native predicate.
            for path in sorted(release):
                if path.exists():
                    self.locks.pin(path)
            self.locks.assert_held()
        self.locks.release_exact(release)
        self.state = "apply_invalidated"
        return sorted(str(path) for path in release)

    def close(self):
        if self.locks is not None:
            self.locks.close()
        self.state = "closed"

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()


def validate_live_receipt(checked, driver):
    """The watchdog binds the one live canonical driver, never restores its pass."""
    from .gpu_handoff import utc_time
    from .guarded_leg_gpu import bound_json, identity_alive

    grant = checked["authorization"]["grant"]
    receipt = bound_json(grant["verified_read_pass"])
    if (receipt.get("schema") != "gvhmr-guarded-leg-live-cpu-read-pass-v1"
            or receipt.get("state") != "cpu_ready_waiting_for_grant"
            or receipt.get("driver") != driver or not identity_alive(driver)
            or not re.fullmatch("[0-9a-f]{32}", str(receipt.get("nonce", "")))
            or receipt.get("gpu_started") is not False or receipt.get("training_published") is not False
            or receipt.get("inputs") != dict(request=checked["request_binding"],
                runtime_contract=checked["runtime_binding"], readiness=grant["readiness"],
                independent_guard=grant["independent_guard"])
            or checked["authorization"]["start"] < utc_time(receipt["completed_utc"])):
        raise ValueError("Fresh GPU authorization does not name its exact actual live verified CPU driver")
    command = driver["command"]
    if (len(command) != 12 or command[8] != "--pregrant" or command[10] != "--pregrant-sha256"
            or str(Path(command[9]).resolve()) != receipt["pregrant"]["path"]
            or command[11] != receipt["pregrant"]["sha256"]):
        raise ValueError("The actual driver command does not bind its CPU-only pregrant bundle")
    check_binding(receipt["pregrant"])
    check_binding(receipt["read_set"])
    return receipt
