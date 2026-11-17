"""Explicit CPU access to the portable current CROP release, without admission.

The original index/manifest bytes certify scientific identity. The portable
view changes declared paths only. Historical payload identities are opaque:
this reader never resolves a parent PNG, reconstructs removed XYZ, or alters
the camera/body/training contracts of SourceIndexedSession.
"""

from __future__ import annotations

import hashlib
import io
import json
import ntpath
from pathlib import Path, PurePosixPath
import re
import zipfile

import cv2
import numpy as np

from .sessions import decode_fixed24


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[0-9]{8}_[0-9]{6}\Z")
_POINT = "decoded_fixed24_valid_returns_before_raw_to_viewer"
_GRID = "zero-based row-major cell in one coordinate band"
_LINEAGE = "lineage_only_not_a_live_payload_dependency"
_SOURCE_V1 = "badminton-final-current-source-index-v1"
_SOURCE_V2 = "badminton-final-current-source-index-v2"
_CURRENT_SCOPE = "current_png_valid_returns_and_grid_cells_only"
_MAP_FIELDS = {"member_mapping": "primary", "immediate_member_mapping": "immediate", "prior_member_mapping": "prior"}
_RETIREMENT_FIELDS = ("point_mapping_available", "point_mapping_status", "point_identity_scope",
                      "point_mapping_retirement_ref")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _equal(a, b):
    # JSON has one finite number type (portable JS helpers serialize 0.0 as 0).
    # Booleans remain a different type; integer frame/time identities are also
    # checked explicitly before use rather than coerced during I/O.
    if type(a) in (int, float) and type(b) in (int, float):
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _json(data):
    def pairs(items):
        result = {}
        for k, v in items:
            _need(k not in result, "Duplicate JSON key")
            result[k] = v
        return result

    def invalid(value):
        raise ValueError("Non-finite JSON number: " + value)

    return json.loads(data.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=invalid)


def _absolute(value):
    return isinstance(value, str) and (value.startswith("/") or ntpath.isabs(value))


def _key(value):
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return ntpath.normpath(value).replace("\\", "/").casefold()
    return str(PurePosixPath(value))


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _identity(value):
    return _digest(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8"))


def _undo_lineage(value):
    if isinstance(value, list):
        return [_undo_lineage(v) for v in value]
    if not isinstance(value, dict):
        return value
    result = {k: _undo_lineage(v) for k, v in value.items()}
    if "reference_scope" in value:
        _need(value["reference_scope"] == _LINEAGE and "file" not in value, "Invalid opaque parent payload identity")
        if "lineage_file" in value:
            _need(isinstance(value["lineage_file"], str) and isinstance(value.get("original_file"), str),
                  "Colliding historical file identities must be retained separately")
            result["file"] = result.pop("lineage_file")
        else:
            _need(isinstance(value.get("original_file"), str), "Missing opaque parent path identity")
            result["file"] = result.pop("original_file")
        del result["reference_scope"]
    return result


class FinalCurrentRelease:
    """A root-confined release handle. Use source(id) for current CROP access."""

    def __init__(self, release_file, data_root, expected_sha256=None):
        self.root = Path(data_root).resolve(strict=True)
        _need(self.root.is_dir(), "Explicit final data root must be a directory")
        self._snapshots = {}
        release_path = Path(release_file).resolve(strict=True)
        _need(release_path.is_relative_to(self.root), "Release must be inside explicit data root")
        raw = self._bytes(release_path.relative_to(self.root).as_posix(), limit=8 * 1024 * 1024)
        self.release_sha256 = _digest(raw)
        _need(expected_sha256 is None or self.release_sha256 == expected_sha256, "Final release SHA256 differs")
        self.release = r = _json(raw)
        _need(isinstance(r, dict) and r.get("schema") == "badminton-final-review-release-v1"
              and r.get("dataset_root") == "." and isinstance(r.get("release_id"), str)
              and bool(r["release_id"]) and _absolute(r.get("origin_recording_root"))
              and isinstance(r.get("sources"), list) and len(r["sources"]) == 35,
              "Expected the explicit portable 35-source final release")
        self.files, self.directories = {}, {}
        mapping = self.document(r["path_map_ref"], limit=16 * 1024 * 1024)
        _need(mapping.get("schema") == "badminton-final-review-path-map-v1"
              and isinstance(mapping.get("files"), list) and len(mapping["files"]) <= 20000
              and isinstance(mapping.get("directories"), list) and len(mapping["directories"]) <= 500,
              "Invalid final path map")
        for entry in mapping["files"]:
            _need(isinstance(entry, dict) and _absolute(entry.get("original_file")), "Invalid original file identity")
            target = {k: entry.get(k) for k in ("file", "sha256", "bytes")}
            self._ref(target)
            key = _key(entry["original_file"])
            _need(key not in self.files, "Duplicate historical file identity")
            self.files[key] = target
        for entry in mapping["directories"]:
            _need(isinstance(entry, dict) and _absolute(entry.get("original_directory")),
                  "Invalid original directory identity")
            self.relative(entry["file"])
            key = _key(entry["original_directory"])
            _need(key not in self.directories, "Duplicate historical directory identity")
            self.directories[key] = entry["file"]
        original = self.document(r["original_manifest_ref"])
        current = self.document(r["canonical_manifest_ref"])
        _need(original.get("schema") == current.get("schema") == "badminton-final-review-manifest-v1"
              and _key(original["recording_root"]) == _key(r["origin_recording_root"])
              and _equal(original.get("counts"), current.get("counts")), "Final master identity/counts differ")
        self.rows = {}
        old_rows = self._source_rows(original.get("sessions"), "id")
        new_rows = self._source_rows(current.get("sessions"), "id")
        for row in r["sources"]:
            sid = row.get("source_id")
            _need(isinstance(sid, str) and _ID.fullmatch(sid) and sid not in self.rows
                  and sid in old_rows and sid in new_rows, "Unknown or duplicate final source")
            for name in ("helper_ref", "source_index_ref"):
                self._ref(row[name])
            canonical = new_rows[sid]
            _need(self.same_ref(canonical["source_manifest_ref"], row["helper_ref"]),
                  "Canonical helper differs from release")
            for field in ("frame_count", "events_present", "events_frame_count"):
                _need(_equal(old_rows[sid].get(field), canonical.get(field)), "Final enumeration changed source counts")
            self.rows[sid] = row
        self.check_unchanged()

    @staticmethod
    def _source_rows(rows, field):
        _need(isinstance(rows, list) and len(rows) == 35, "Final source enumeration must contain 35 sources")
        result = {}
        for row in rows:
            _need(isinstance(row, dict) and isinstance(row.get(field), str) and _ID.fullmatch(row[field])
                  and row[field] not in result, "Invalid or duplicate source enumeration")
            result[row[field]] = row
        return result

    def relative(self, name):
        _need(isinstance(name, str) and bool(name) and "\\" not in name and ":" not in name
              and not name.startswith("/") and all(p not in ("", ".", "..") and p == p.rstrip(" .")
                                                    for p in name.split("/")),
              "Final reference must be a safe root-relative POSIX path")
        result = self.root.joinpath(*name.split("/"))
        _need(result.resolve().is_relative_to(self.root), "Final reference escapes explicit data root")
        return result

    def _ref(self, ref):
        _need(isinstance(ref, dict) and "reference_scope" not in ref and "original_file" not in ref
              and type(ref.get("bytes")) is int and 0 < ref["bytes"] <= 2**53 - 1
              and isinstance(ref.get("sha256"), str) and _HASH.fullmatch(ref["sha256"]),
              "Expected a live file/SHA256/byte-length reference")
        return self.relative(ref.get("file"))

    def _bytes(self, name, ref=None, limit=128 * 1024 * 1024):
        path = self.relative(name)
        actual = path.resolve(strict=True)
        before = actual.stat()
        _need(actual.is_file() and before.st_size <= limit, "Final input exceeds bounded regular-file read")
        if ref is not None:
            self._ref(ref)
            _need(before.st_size == ref["bytes"], "Final input byte length differs")
        raw = actual.read_bytes()
        after = actual.stat()
        signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        _need(signature == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
              and actual == path.resolve(strict=True), "Final input changed while reading")
        if ref is not None:
            _need(_digest(raw) == ref["sha256"], "Final input SHA256 differs")
        old = self._snapshots.setdefault(name, (actual, signature))
        _need(old == (actual, signature), "Final release changed; reload its explicit handle")
        return raw

    def document(self, ref, limit=128 * 1024 * 1024):
        self._ref(ref)
        value = _json(self._bytes(ref["file"], ref, limit))
        _need(isinstance(value, dict), "Expected a final metadata object")
        return value

    def check_unchanged(self):
        for name, (old_path, old_signature) in self._snapshots.items():
            path = self.relative(name).resolve(strict=True)
            stat = path.stat()
            _need(path == old_path and (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) == old_signature,
                  "Final release input changed; reload its explicit handle")

    def same_ref(self, left, right):
        self._ref(left)
        self._ref(right)
        return all(_equal(left[k], right[k]) for k in ("file", "sha256", "bytes"))

    def original_path(self, name, base=None):
        _need(isinstance(name, str), "Invalid historical path")
        if _absolute(name):
            return name
        root = base or self.release["origin_recording_root"]
        return ntpath.join(root, name) if re.match(r"^[A-Za-z]:", root) else str(PurePosixPath(root) / name)

    def original_ref(self, ref, base=None):
        _need(isinstance(ref, dict) and isinstance(ref.get("file"), str), "Missing original scientific reference")
        mapped = self.files.get(_key(self.original_path(ref["file"], base)))
        _need(mapped is not None and mapped["sha256"] == ref.get("sha256")
              and type(ref.get("bytes")) is int and mapped["bytes"] == ref["bytes"],
              "Original scientific identity is absent from final path map")
        return mapped

    def path_only_equal(self, before, after, field="", frame_directory=None, original_base=None):
        if _equal(before, after):
            return True
        if isinstance(before, str) and isinstance(after, str):
            if field.endswith("root") and _key(before) == _key(self.release["origin_recording_root"]) and after == ".":
                return True
            if frame_directory and field == "file" and re.fullmatch(r"frame_[0-9]+\.png", before):
                return after == frame_directory + "/" + before
            # Legacy raw crop indexes mix source-root-relative payload refs
            # with recording-root-relative parent metadata. Both roots are
            # explicit immutable identities; only authorized map matches count.
            historical_keys = {_key(before), _key(self.original_path(before, original_base)),
                               _key(self.original_path(before))}
            targets = set()
            for historical in historical_keys:
                entry = self.files.get(historical)
                if entry:
                    targets.add(entry["file"])
                if historical in self.directories:
                    targets.add(self.directories[historical])
                for directory, target in self.directories.items():
                    prefix = directory.rstrip("/") + "/"
                    if (historical.startswith(prefix)
                            and re.fullmatch(r"frame_[0-9]+\.(png|npz)", historical[len(prefix):])):
                        targets.add(target + "/" + historical[len(prefix):])
            return len(targets) == 1 and after in targets
        if isinstance(before, list) or isinstance(after, list):
            return (isinstance(before, list) and isinstance(after, list) and len(before) == len(after)
                    and all(self.path_only_equal(a, b, field, frame_directory, original_base)
                            for a, b in zip(before, after)))
        return (isinstance(before, dict) and isinstance(after, dict) and before.keys() == after.keys()
                and all(self.path_only_equal(v, after[k], k, frame_directory, original_base)
                        for k, v in before.items()))

    def source(self, source_id):
        _need(source_id in self.rows, "Unknown explicit final source ID")
        return FinalCurrentSource(self, source_id)


def _restore_provenance(comparison, current, original, keys, release):
    p = current.get("provenance")
    _need(isinstance(p, dict), "Portable metadata requires explicit original provenance")
    old = original.get("provenance")
    if "provenance" in original:
        kept = {k: v for k, v in p.items() if k not in keys}
        _need(release.path_only_equal(old, kept) or release.path_only_equal(old, p.get("original_provenance")),
              "Portable metadata lost its original provenance")
        comparison["provenance"] = old
    else:
        _need(set(p) == set(keys), "Unexpected scientific provenance in portable view")
        comparison.pop("provenance", None)


def _without_member_refs(index):
    """Remove only the three explicitly retired descriptor roles for comparison.

    This never fabricates a membership array or modifies an input document. The
    complete old descriptors remain in the hash-bound original/previous proofs.
    """
    return dict(index, frames=[{k: v for k, v in frame.items() if k not in _MAP_FIELDS}
                               for frame in index.get("frames", [])])


def _original_comparison(index, original, release, *, current_only=False):
    comparison = _undo_lineage(index)
    comparison["schema"] = original["schema"]
    del comparison["original_schema"]
    if "data_root" not in original:
        del comparison["data_root"]
    if "original_parent_input_data_root" in comparison:
        comparison["parent_input_data_root"] = comparison.pop("original_parent_input_data_root")
    keys = ["original_source_index_ref", "historical_parent_payload_refs_are_lineage_only"]
    if current_only:
        for field in _RETIREMENT_FIELDS:
            comparison.pop(field)
        keys += ["current_png_points_geometry_clocks_and_frame_values_preserved",
                 "previous_canonical_source_index_ref"]
    else:
        keys += ["scientific_values_and_current_point_members_preserved"]
    _restore_provenance(comparison, index, original, keys, release)
    comparison["source_manifest"] = original["source_manifest"]
    return comparison


def _verify_retired_point_maps(release, index, helper, original, directory, original_base):
    """Validate metadata-only retirement; no NPZ is resolved, opened or replayed."""
    _need(index.get("point_mapping_available") is False
          and index.get("point_mapping_status") == "retired_by_user"
          and index.get("point_identity_scope") == _CURRENT_SCOPE,
          "V2 requires an explicit user-retired current-PNG-only point identity contract")
    for name in _RETIREMENT_FIELDS:
        _need(_equal(helper.get(name), index.get(name)), "Helper and index point-retirement contracts differ")
    retirement_ref = index["point_mapping_retirement_ref"]
    _need(release.same_ref(retirement_ref, release.release.get("point_mapping_retirement_ref")),
          "Point retirement must be bound by the explicit final release")
    retirement = release.document(retirement_ref)
    _need(retirement.get("schema") == "badminton-point-mapping-retirement-v1"
          and retirement.get("complete") is True
          and retirement.get("status") == "excluded_from_current_dataset_by_user"
          and retirement.get("point_mapping_available") is False
          and retirement.get("physical_removal_status") == "execution_recorded_separately",
          "Point-retirement evidence has a different scope or claims unrecorded physical deletion")
    rows = retirement.get("sources")
    _need(isinstance(rows, list) and len(rows) == len(release.rows) and all(isinstance(row, dict) for row in rows)
          and {row.get("source_id") for row in rows} == set(release.rows),
          "Point-retirement evidence does not enumerate the complete final source set")
    records = [row for row in rows if row.get("source_id") == index["source_id"]]
    _need(len(records) == 1, "Point-retirement source is missing or ambiguous")
    record = records[0]
    previous_ref = index["provenance"]["previous_canonical_source_index_ref"]
    _need(release.same_ref(record.get("previous_canonical_source_index_ref"), previous_ref),
          "Point retirement names a different previous canonical index")
    previous = release.document(previous_ref)
    _need(previous.get("schema") == _SOURCE_V1 and previous.get("source_id") == index["source_id"]
          and previous.get("original_schema") == original.get("schema") and previous.get("data_root") == "."
          and previous.get("provenance", {}).get("scientific_values_and_current_point_members_preserved") is True
          and previous.get("provenance", {}).get("historical_parent_payload_refs_are_lineage_only") is True
          and release.same_ref(previous["provenance"]["original_source_index_ref"],
                               index["provenance"]["original_source_index_ref"]),
          "Point retirement requires the exact previous portable-v1 scientific index")
    _need(type(index.get("frame_count")) is int and index["frame_count"] > 0
          and _equal(previous.get("frame_count"), index["frame_count"])
          and all(isinstance(value.get("frames"), list) and all(isinstance(f, dict) for f in value["frames"])
                  and len(value["frames"]) == index["frame_count"] for value in (index, previous, original)),
          "Retired membership metadata must cover every original/current frame")
    before = _without_member_refs(original)
    old_comparison = _without_member_refs(_original_comparison(previous, original, release))
    _need(release.path_only_equal(before, old_comparison, frame_directory=directory, original_base=original_base),
          "Previous canonical proof changed non-membership scientific values")
    counts = {f"{role}_npz_{field}": 0 for role in _MAP_FIELDS.values() for field in ("count", "bytes")}
    for frame, old, raw in zip(index.get("frames", []), previous["frames"], original["frames"]):
        _need(not any(name in frame for name in _MAP_FIELDS), "V2 may not retain active member-map descriptors")
        for name, role in _MAP_FIELDS.items():
            _need((name in old) == (name in raw), "Previous/original point-map roles differ")
            if name not in old:
                continue
            entry, original_entry = old[name], raw[name]
            _need(isinstance(entry, dict) and isinstance(original_entry, dict)
                  and entry.get("schema") == "badminton-native-roi-member-map-v1"
                  and isinstance(entry.get("sha256"), str) and _HASH.fullmatch(entry["sha256"])
                  and type(entry.get("bytes")) is int and entry["bytes"] > 0
                  and isinstance(entry.get("file"), str)
                  and entry["file"].startswith(f"sessions/{index['source_id']}/member-maps/")
                  and _equal({k: v for k, v in entry.items() if k != "file"},
                             {k: v for k, v in original_entry.items() if k != "file"}),
                  "Archived point-map content identity differs from its immutable original metadata")
            counts[f"{role}_npz_count"] += 1
            counts[f"{role}_npz_bytes"] += entry["bytes"]
    counts["total_npz_count"] = sum(counts[f"{role}_npz_count"] for role in _MAP_FIELDS.values())
    counts["total_npz_bytes"] = sum(counts[f"{role}_npz_bytes"] for role in _MAP_FIELDS.values())
    counts["retained_png_count"] = index["frame_count"]
    counts["retained_png_bytes"] = sum(frame["png_bytes"] for frame in index["frames"])
    _need(_equal(record.get("mapping_schema_before_retirement"), previous.get("mapping_schema")),
          "Point retirement changed its original mapping schema")
    for name, expected in counts.items():
        _need(type(record.get(name)) is int and record[name] == expected,
              "Point retirement has a different per-source count/length: " + name)
        _need(isinstance(retirement.get("totals"), dict)
              and type(retirement["totals"].get(name)) is int
              and all(type(row.get(name)) is int and row[name] >= 0 for row in rows)
              and retirement["totals"][name] == sum(row[name] for row in rows),
              "Point-retirement global totals differ: " + name)
    _need(retirement["totals"].get("source_count") == len(release.rows),
          "Point retirement has a different total source count")
    return dict(retirement_ref=retirement_ref, previous_canonical_source_index_ref=previous_ref,
                logical_status="retired_by_user", physical_removal_status="execution_recorded_separately",
                original_membership_replayed=False, previous_member_descriptor_count=counts["total_npz_count"])


class FinalCurrentSource:
    """Read declared v1 members or explicit v2 current clouds; no parent access."""

    def __init__(self, release, source_id):
        self.release, self.source_id = release, source_id
        row = release.rows[source_id]
        self.helper = h = release.document(row["helper_ref"])
        self.index = index = release.document(row["source_index_ref"])
        directory = f"sessions/{source_id}/output-lidar/lidar_{source_id}"
        _need(h.get("schema") == "badminton-final-review-source-v1" and h.get("source_id") == source_id
              and h.get("png_directory") == directory and h.get("canonical_png_directory") == directory
              and release.same_ref(h["source_index_ref"], row["source_index_ref"]),
              "Final helper does not select this canonical current CROP")
        _need(index.get("schema") in (_SOURCE_V1, _SOURCE_V2)
              and index.get("source_id") == source_id and index.get("data_root") == ".",
              "Final-current mode requires the explicit new index schema")
        current_only = index["schema"] == _SOURCE_V2
        provenance = index.get("provenance", {})
        _need(provenance.get("historical_parent_payload_refs_are_lineage_only") is True
              and ((current_only
                    and "scientific_values_and_current_point_members_preserved" not in provenance
                    and provenance.get("current_png_points_geometry_clocks_and_frame_values_preserved") is True)
                   or (not current_only
                       and provenance.get("scientific_values_and_current_point_members_preserved") is True)),
              "Final current index lacks its original scientific boundary")
        self.original = original = release.document(provenance["original_source_index_ref"])
        _need(index.get("original_schema") == original.get("schema")
              and original.get("source_id") == source_id, "Original scientific index names another source/schema")
        original_helper = release.document(h["provenance"]["original_selected_helper_ref"])
        _need(original_helper.get("source_id") == source_id, "Original helper names another source")
        original_base = original_helper.get("source_index_data_root", release.release["origin_recording_root"])
        _need(_absolute(original_base), "Original source root must retain its explicit historical identity")
        raw_ref = original_helper.get("lidar_revision", {}).get(
            "new_source_index_ref", original_helper.get("source_index_ref"))
        _need(release.same_ref(release.original_ref(raw_ref, original_base), provenance["original_source_index_ref"]),
              "Portable view changed the selected original scientific index")
        self.point_mapping_retirement = (_verify_retired_point_maps(
            release, index, h, original, directory, original_base) if current_only else None)
        comparison = _original_comparison(index, original, release, current_only=current_only)
        self.manifest = native = release.document(index["source_manifest"])
        old_native = release.document(release.original_ref(original["source_manifest"], original_base))
        npv = native.get("provenance", {})
        _need(npv.get("path_only_portability_derivation") is True
              and release.same_ref(npv["original_manifest_ref"],
                                   release.original_ref(original["source_manifest"], original_base)),
              "Native manifest lost its original scientific identity")
        restored_native = _undo_lineage(native)
        _restore_provenance(restored_native, native, old_native,
                            ["original_manifest_ref", "path_only_portability_derivation"], release)
        _need(release.path_only_equal(old_native, restored_native,
                                     frame_directory=directory, original_base=original_base),
              "Portable native manifest changed capture arrays/scientific values")
        comparison["source_manifest"] = original["source_manifest"]
        scientific_original = _without_member_refs(original) if current_only else original
        _need(release.path_only_equal(scientific_original, comparison,
                                     frame_directory=directory, original_base=original_base),
              "Portable source index changed non-path scientific values")
        for name in ("frame_count", "frame_fields", "camera_pair_fields", "frames", "fixed24",
                     "png_sha256_and_bytes_by_frame", "camera_calibration_admitted"):
            _need(_equal(h.get(name), original_helper.get(name)),
                  "Portable helper changed frame/grid/camera identities")
        for name in ("geometry", "timing"):
            _need(release.path_only_equal(original_helper.get(name), h.get(name)),
                  "Portable helper changed original geometry or timing")
        self.directory = directory
        self.grid = self._grid()
        self.matrix = np.asarray(index["coordinate_contract"]["application_raw_to_viewer_M"], dtype=np.float64)
        _need(self.matrix.shape == (3, 3) and np.isfinite(self.matrix).all()
              and np.allclose(self.matrix @ self.matrix.T, np.eye(3), atol=2e-4)
              and np.linalg.det(self.matrix) > 0
              and index["coordinate_contract"].get("M_baked_into_stored_xyz") is False,
              "Current CROP requires raw fixed24 and one explicit original application M")
        frames = index.get("frames")
        _need(type(index.get("frame_count")) is int and index["frame_count"] > 0
              and isinstance(frames, list) and len(frames) == index["frame_count"] == h["frame_count"]
              and len(h["frames"]) == len(h["png_sha256_and_bytes_by_frame"]) == len(frames)
              and isinstance(native.get("frames"), list) and len(native["frames"]) == len(frames),
              "Current CROP must retain its complete original frame enumeration")
        for number, frame in enumerate(frames):
            _need(type(frame.get("frame_index")) is int and frame["frame_index"] == number,
                  "Current CROP frame ordinals must preserve original order")
            _need(frame.get("file") == f"{directory}/frame_{number:06d}.png",
                  "Current frame selects a non-canonical variant")
            columns = dict(zip(h["frame_fields"], h["frames"][number]))
            _need(columns["frame_index"] == number and columns["png_basename"] == Path(frame["file"]).name
                  and _equal(columns["point_count"], frame["point_count"])
                  and _equal(h["png_sha256_and_bytes_by_frame"][number], [frame["png_sha256"], frame["png_bytes"]]),
                  "Current helper/PNG identity differs from selected source")
        self._current_png_references(index)
        self._current_png_references(native)
        self._header_identity = _identity({k: v for k, v in index.items() if k != "frames"})
        self._frame_identities = tuple(_identity(frame) for frame in frames)
        self._grid_identity = _identity(self.grid)
        self._matrix_identity = self.matrix.tobytes()
        self._retirement_identity = _identity(self.point_mapping_retirement)
        release.check_unchanged()

    @property
    def point_mapping_available(self):
        return self.index["schema"] == _SOURCE_V1

    def _grid(self):
        grid = dict(self.index["fixed24"])
        shape = grid.get("point_grid_shape")
        width = grid.get("point_width", shape[1] if isinstance(shape, list) and len(shape) == 2 else None)
        height = grid.get("point_height", shape[0] if isinstance(shape, list) and len(shape) == 2 else None)
        _need(type(width) is int and type(height) is int and 0 < width * height <= 10000000
              and grid.get("coordinate_encoding") == "fixed24" and grid.get("stored_xyz_space") == "lidar_sensor_xyz"
              and grid.get("scale") == 1000 and grid.get("center") == 256, "Invalid current raw fixed24 grid")
        grid.update(point_width=width, point_height=height)
        for key, expected in (("point_width", width), ("point_height", height), ("scale", 1000), ("center", 256)):
            _need(_equal(self.manifest.get(key), expected), "Current native manifest grid differs")
        return grid

    def _current_png_references(self, value):
        if isinstance(value, list):
            for item in value:
                self._current_png_references(item)
        elif isinstance(value, dict):
            if isinstance(value.get("file"), str) and value["file"].startswith(self.directory + "/"):
                name = PurePosixPath(value["file"]).name
                if re.fullmatch(r"frame_[0-9]+\.png", name) and "sha256" in value:
                    number = int(name[6:-4])
                    _need(0 <= number < len(self.index["frames"]), "Unknown current PNG reference")
                    frame = self.index["frames"][number]
                    _need(value.get("sha256") == frame["png_sha256"] and value.get("bytes") == frame["png_bytes"],
                          "Historical parent PNG cannot masquerade as a current child")
            for item in value.values():
                self._current_png_references(item)

    def _mapping(self, ref):
        release = self.release
        release._ref(ref)
        _need(ref["file"].startswith(f"sessions/{self.source_id}/member-maps/")
              and ref.get("schema") == "badminton-native-roi-member-map-v1" and ref.get("dtype") == "int64",
              "Current member map must belong to this source's explicit map directory")
        raw = release._bytes(ref["file"], ref, limit=32 * 1024 * 1024)
        capacity = self.grid["point_width"] * self.grid["point_height"]
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            _need(sum(info.file_size for info in archive.infolist()) <= capacity * 24 + 65536,
                  "Member map arrays exceed the original grid capacity")
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            values = {name: data[name] for name in data.files}
        for name, expected in (("schema", "badminton-native-roi-member-map-v1"), ("dtype", "int64"),
                               ("point_index_space", _POINT), ("grid_index_space", _GRID)):
            _need(name in values and values[name].shape == () and values[name].item() == expected,
                  "Member map has a different coordinate/index contract")
        for name in ("parent_valid_to_child", "child_valid_to_parent", "parent_valid_to_grid"):
            _need(name in values and values[name].dtype == np.int64 and values[name].ndim == 1,
                  "Member map identity arrays must be one-dimensional int64")
        inverse, selected, grid = (values[n] for n in
                                   ("parent_valid_to_child", "child_valid_to_parent", "parent_valid_to_grid"))
        _need(len(inverse) == len(grid) == ref["parent_valid_points"] and len(selected) == ref["child_valid_points"]
              and len(grid) <= capacity and np.all((grid >= 0) & (grid < capacity)) and np.all(np.diff(grid) > 0)
              and np.all((selected >= 0) & (selected < len(grid))) and np.all(np.diff(selected) > 0),
              "Member map does not retain ordered original valid/grid identities")
        expected = np.full(len(grid), -1, dtype=np.int64)
        expected[selected] = np.arange(len(selected), dtype=np.int64)
        _need(np.array_equal(expected, inverse), "Original J/current inverse member map differs")
        return values

    def _read_current_payload(self, frame_index):
        """Return actual raw XYZ/grid after the unchanged PNG/context checks."""
        _need(type(frame_index) is int and 0 <= frame_index < self.index["frame_count"],
              "Unknown original frame ordinal")
        frame = self.index["frames"][frame_index]
        _need(_identity({k: v for k, v in self.index.items() if k != "frames"}) == self._header_identity
              and _identity(frame) == self._frame_identities[frame_index]
              and _identity(self.grid) == self._grid_identity and self.matrix.tobytes() == self._matrix_identity
              and _identity(self.point_mapping_retirement) == self._retirement_identity,
              "Verified final source context changed in memory; reload its original release")
        ref = {"file": frame["file"], "sha256": frame["png_sha256"], "bytes": frame["png_bytes"]}
        raw = self.release._bytes(ref["file"], ref, limit=64 * 1024 * 1024)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        width, height = self.grid["point_width"], self.grid["point_height"]
        _need(image is not None and image.dtype == np.uint8 and image.shape == (height * 3, width, 3),
              "Current CROP is not an RGB8 fixed24 grid")
        bands = image.reshape(3, width * height, 3)
        current_grid = np.flatnonzero(np.any(bands != 0, axis=2).all(axis=0))
        _need(type(frame["point_count"]) is int and len(current_grid) == frame["point_count"],
              "Current PNG valid count differs")
        points = decode_fixed24(self.release.relative(ref["file"]), self.grid)
        _need(points.shape == (len(current_grid), 3) and np.isfinite(points).all(),
              "Current fixed24 decode is not finite")
        cells = bands.astype(np.uint32)
        codes = (cells[:, :, 2] << 16) | (cells[:, :, 1] << 8) | cells[:, :, 0]
        expected_points = codes[:, current_grid].T.astype(np.float32) / np.float32(1000) - np.float32(256)
        _need(np.array_equal(points, expected_points), "Decoded XYZ differs from the hash-verified current RGB8 cells")
        self.release.check_unchanged()
        return frame, ref, points, current_grid.astype(np.int64, copy=False)

    def read_current_cloud(self, frame_index, *, viewer=False):
        """Return (XYZ, actual_current_grid_indices, evidence) for explicit v2.

        Grid cells come from the current PNG valid mask; they are not original
        parent valid-return IDs or a fabricated original-to-current identity map.
        """
        _need(not self.point_mapping_available,
              "read_current_cloud requires an explicit v2 current-PNG-only source")
        frame, ref, points, grid = self._read_current_payload(frame_index)
        if viewer:
            points = points @ self.matrix.T
        evidence = dict(frame_index=frame_index, original_frame_id=frame["original_frame_id"],
                        frame_map_lidar_frame=frame["frame_map_lidar_frame"], scan_mid_us=frame["scan_mid_us"],
                        device_first_us=frame.get("device_first_us"), device_last_us=frame.get("device_last_us"),
                        cameras=frame.get("cameras"), current_png_ref=ref, current_valid_points=len(points),
                        point_mapping_available=False, point_mapping_status="retired_by_user",
                        point_identity_scope=_CURRENT_SCOPE,
                        current_grid_indices_sha256=_digest(grid.astype("<i8").tobytes()),
                        current_grid_indices_dtype="int64", current_grid_cells_derived_from_PNG=True,
                        current_encoded_coordinates_hash_verified=True, original_point_identity_available=False,
                        original_valid_grid_verified=False, original_J_to_current_inverse_verified=False,
                        original_point_membership_verified=False, parent_payload_reopened=False,
                        parent_encoded_subset_replayed=False, removed_original_XYZ_reconstructed=False,
                        application_M_count=1 if viewer else 0,
                        camera_exposure_admission_issued=False, body_fitting_started=False,
                        training_admitted=False,
                        point_mapping_retirement=self.point_mapping_retirement)
        return points, grid, evidence

    def read_cloud(self, frame_index, *, viewer=False):
        """Return (points, evidence) only with verified v1 original membership."""
        _need(self.point_mapping_available,
              "Original point-membership maps were retired by the user; use read_current_cloud "
              "for current PNG XYZ/grid only")
        frame, ref, points, current_grid = self._read_current_payload(frame_index)
        mapping = self._mapping(frame["member_mapping"])
        parent_grid, selected = mapping["parent_valid_to_grid"], mapping["child_valid_to_parent"]
        for name, expected in (("parent_scope", self.index["parent_scope"]),
                               ("parent_png_sha256", frame["parent_png"]["sha256"]),
                               ("child_png_sha256", frame["png_sha256"]),
                               ("crop_binding_sha256", frame["crop_binding_sha256"])):
            _need(name in mapping and mapping[name].shape == () and mapping[name].item() == expected,
                  "Member map does not bind the original parent and current encoded child")
        for name, field in (("court_cleanup_binding_sha256", "court_cleanup_binding_sha256"),
                            ("court_cleanup_flat_restore_binding_sha256", "conservative_flat_restore_binding_sha256"),
                            ("floor_bbox_recrop_binding_sha256", "floor_bbox_recrop_binding_sha256")):
            if name in mapping:
                _need(mapping[name].shape == () and mapping[name].item() == frame.get(field),
                      "Current recrop binding differs")
        _need(len(parent_grid) == frame["parent_png"]["valid_points"] and len(selected) == len(current_grid)
              and np.array_equal(parent_grid[selected], current_grid),
              "Current CROP cells do not match selected original J grid")
        if "parent_valid_to_grid_sha256" in frame:
            _need(_digest(parent_grid.astype("<i8").tobytes()) == frame["parent_valid_to_grid_sha256"],
                  "Complete original valid-return grid digest differs")
        auxiliary = "not_present"
        if "prior_member_mapping" in frame and "immediate_member_mapping" in frame:
            prior = self._mapping(frame["prior_member_mapping"])
            immediate = self._mapping(frame["immediate_member_mapping"])
            if (immediate["child_png_sha256"].item() == frame["png_sha256"]
                    and prior["child_png_sha256"].item() == immediate["parent_png_sha256"].item()):
                _need(np.array_equal(parent_grid, prior["parent_valid_to_grid"])
                      and np.array_equal(immediate["parent_valid_to_grid"],
                                         prior["parent_valid_to_grid"][prior["child_valid_to_parent"]])
                      and np.array_equal(selected, prior["child_valid_to_parent"][immediate["child_valid_to_parent"]]),
                      "J to K to current composite member identity differs")
                auxiliary = "same_encoded_current_endpoint_composition_verified"
            else:
                auxiliary = "historical_auxiliary_endpoints_not_current"
        if viewer:
            points = points @ self.matrix.T
        self.release.check_unchanged()
        evidence = dict(frame_index=frame_index, original_frame_id=frame["original_frame_id"],
                        frame_map_lidar_frame=frame["frame_map_lidar_frame"], scan_mid_us=frame["scan_mid_us"],
                        current_png_ref=ref, member_mapping_ref=frame["member_mapping"],
                        current_valid_points=len(points),
                        original_valid_points=len(parent_grid), original_valid_grid_verified=True,
                        original_J_to_current_inverse_verified=True, current_encoded_coordinates_hash_verified=True,
                        auxiliary_mapping_check=auxiliary, parent_payload_reopened=False,
                        parent_encoded_subset_replayed=False, removed_original_XYZ_reconstructed=False,
                        application_M_count=1 if viewer else 0, cameras=frame.get("cameras"))
        return points, evidence


def prepare_final_current(release_file, data_root, source_id, expected_sha256=None):
    """Inspect first/middle/last current scans; does not execute fitting or certify video."""
    release = FinalCurrentRelease(release_file, data_root, expected_sha256)
    source = release.source(source_id)
    samples = []
    for number in sorted({0, source.index["frame_count"] // 2, source.index["frame_count"] - 1}):
        if source.point_mapping_available:
            _, evidence = source.read_cloud(number)
        else:
            _, grid, evidence = source.read_current_cloud(number)
            evidence["current_grid_indices"] = grid.tolist()
        samples.append(evidence)
    report = dict(schema="gvhmr-final-current-cpu-inspection-v1", status="current_crop_readable",
                release_id=release.release["release_id"], release_sha256=release.release_sha256,
                data_root=str(release.root), source_id=source_id, source_count=len(release.rows),
                source_index_ref=release.rows[source_id]["source_index_ref"], frame_count=source.index["frame_count"],
                original_provenance_bytes_verified=True, scientific_path_only_equivalence_verified=True,
                samples=samples, original_source_admissions={k: source.index.get(k) for k in
                ("camera_calibration_admitted", "body_labels_production_admitted", "production_admitted")},
                original_timing=source.helper.get("timing"),
                original_camera_states=source.helper.get("geometry", {}).get("camera_states"),
                camera_exposure_admission_issued=False, body_fitting_started=False, training_admitted=False,
                video_payloads_rehashed=False, all_frame_payloads_verified_by_this_inspection=False,
                manual_annotation_foreground_selected=False)
    if not source.point_mapping_available:
        report.pop("scientific_path_only_equivalence_verified")
        report.update(schema="gvhmr-final-current-cpu-inspection-v2",
                      status="current_png_readable_without_original_point_maps", point_mapping_available=False,
                      point_identity_scope=_CURRENT_SCOPE, original_point_membership_verified=False,
                      original_point_identity_available=False, non_membership_scientific_equivalence_verified=True)
    return report
