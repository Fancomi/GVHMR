"""Explicit byte-bound reads of relocated originals; never discover or move data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate relocation JSON key: {key}")
        result[key] = value
    return result


def logical_path(value):
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or value.startswith("/")
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("Relocation paths must be canonical DATA-relative POSIX paths")
    return PurePosixPath(value)


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Relocation requires a lowercase SHA256")
    return value


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_id(source):
    parts = source.parts
    if (
        len(parts) >= 4
        and parts[0] == "upstream"
        and re.fullmatch(r"\d{4}-\d{2}-\d{2}", parts[1])
        and re.fullmatch(r"\d{8}_\d{6}", parts[2])
        and parts[1].replace("-", "") == parts[2][:8]
    ):
        return parts[2]
    if (
        len(parts) >= 5
        and parts[0] == "collections"
        and parts[2] == "device-time-sources"
        and parts[3] in ("20261010_192753", "20261015_161256", "20261015_162740")
    ):
        return parts[3]
    raise ValueError("Unsupported relocation source domain")


@dataclass(frozen=True, init=False)
class DataRelocations:
    """Frozen per-command manifest; mapped target bytes are verified on every read."""

    root: Path
    binding: object
    _entries: object
    _source_names: object

    def __init__(self, data_root, manifest, expected_sha256):
        root = Path(data_root).resolve(strict=True)
        path = Path(manifest).resolve(strict=True)
        if not root.is_dir() or not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Relocation manifest must be a file inside DATA")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != _digest(expected_sha256):
            raise ValueError("Relocation manifest SHA256 mismatch")
        document = json.loads(payload, object_pairs_hook=_unique_pairs)
        if (
            not isinstance(document, dict)
            or document.get("schema") != "badminton-data-relocations-v1"
            or not isinstance(document.get("entries"), dict)
        ):
            raise ValueError("Unsupported data relocation manifest")
        entries, old_names, destinations = {}, set(), {}
        for old, entry in document["entries"].items():
            source = logical_path(old)
            if not isinstance(entry, dict):
                raise ValueError("Relocation entry must bind file/SHA256/bytes")
            target = logical_path(entry.get("file"))
            size = entry.get("bytes")
            if type(size) is not int or size < 0:
                raise ValueError("Relocation byte size must be a nonnegative integer")
            digest_value = _digest(entry.get("sha256"))
            sid = _source_id(source)
            if len(target.parts) < 3 or target.parts[:2] != ("sessions", sid):
                raise ValueError("Relocation target must belong to the same source ID")
            identity = (sid, digest_value, size)
            target_key = target.as_posix().casefold()
            if old.casefold() in old_names:
                raise ValueError("Duplicate relocation source or case alias")
            if target_key in destinations:
                previous_target, previous_identity = destinations[target_key]
                if previous_target != target.as_posix():
                    raise ValueError("Relocation destination case alias")
                if previous_identity != identity:
                    raise ValueError("Conflicting identities for one relocation destination")
            old_names.add(old.casefold())
            destinations[target_key] = (target.as_posix(), identity)
            entries[old] = MappingProxyType(dict(file=target.as_posix(), sha256=digest_value, bytes=size))
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "_entries", MappingProxyType(entries))
        object.__setattr__(self, "_source_names", MappingProxyType({name.casefold(): name for name in entries}))
        object.__setattr__(
            self,
            "binding",
            MappingProxyType(dict(file=path.relative_to(root).as_posix(), sha256=digest, bytes=len(payload))),
        )

    def resolve(self, reference):
        name = logical_path(reference.get("file")).as_posix()
        entry = self._entries.get(name)
        if entry is None and name.casefold() in self._source_names:
            raise ValueError("Relocation reference case alias")
        if entry is not None:
            if _digest(reference.get("sha256")) != entry["sha256"]:
                raise ValueError("Relocation SHA differs from the immutable input reference")
            if "bytes" in reference and (type(reference["bytes"]) is not int or reference["bytes"] != entry["bytes"]):
                raise ValueError("Relocation size differs from the immutable input reference")
        path = (self.root / (entry["file"] if entry is not None else name)).resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Relocation asset escapes DATA or is not a file")
        if entry is not None:
            session = self.root / "sessions" / _source_id(PurePosixPath(name))
            if not path.is_relative_to(session):
                raise ValueError("Relocation target escapes its canonical session")
            if path.stat().st_size != entry["bytes"] or _file_sha(path) != entry["sha256"]:
                raise ValueError("Relocated asset SHA256/size mismatch")
        return path

    def relative_binding(self, logical_parent, name, sha256):
        parent, child = logical_path(logical_parent), logical_path(name)
        return dict(file=(parent.parent / child).as_posix(), sha256=sha256)

    def relative_reference(self, logical_parent, name, sha256):
        return self.resolve(self.relative_binding(logical_parent, name, sha256))

    def has_mapping(self, logical_name):
        """Whether resolve() already verified this exact mapped input identity."""
        return logical_name in self._entries


def require_relocation_root(data_root, relocations):
    if relocations is not None and Path(data_root).resolve() != relocations.root:
        raise ValueError("Relocation resolver belongs to a different DATA root")


def load_relocations(data_root, manifest=None, expected_sha256=None):
    if (manifest is None) != (expected_sha256 is None):
        raise ValueError("--data-relocations and --data-relocations-sha256 must be supplied together")
    return DataRelocations(data_root, manifest, expected_sha256) if manifest is not None else None


def load_run_relocations(run_root, manifest=None, expected_sha256=None):
    """CLI adapter: only explicitly enabled runs read the frozen registry's DATA root."""
    if manifest is None or expected_sha256 is None:
        return load_relocations(run_root, manifest, expected_sha256)
    registry = json.loads((Path(run_root) / "source_registry.json").read_text(encoding="utf-8-sig"))
    return load_relocations(registry["data_root"], manifest, expected_sha256)


def validate_run_relocations(registry, calibration_data, relocations):
    declared = [document.get("data_relocations") for document in (registry, calibration_data)]
    if relocations is None:
        if any(binding is not None for binding in declared):
            raise ValueError("This run requires its explicitly bound data relocations")
        return
    require_relocation_root(registry["data_root"], relocations)
    require_relocation_root(Path(calibration_data["sessions_root"]).parent, relocations)
    if any(binding is not None and binding != dict(relocations.binding) for binding in declared):
        raise ValueError("Runtime data relocation binding differs from the frozen run")
