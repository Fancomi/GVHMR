"""Serve the exact admitted identity revision through the shared native viewer."""

import argparse
import colorsys
from pathlib import Path
import threading

from hmr4d.lidar_mocap.expanded_dataset import binding, check_binding
from hmr4d.lidar_mocap.rally_pipeline import read_json, read_rows
from hmr4d.lidar_mocap.rally_viewer import ViewerData, prepare_viewer, serve_viewer
from hmr4d.lidar_mocap.track_colors import track_color_registry
from badminton_viewer_contracts import viewer_relocations


class SingleThreadNativePreviewDecoder:
    """Use the shared observer's explicit safe native-PTS decoder."""

    def __init__(self):
        self.guard = threading.Lock()
        self.seek_count = 0
        self.delivered_frames = 0

    def diagnostics(self):
        with self.guard:
            return dict(seeks=self.seek_count, delivered_frames=self.delivered_frames)

    def frames(self, session, camera, indices):
        if not indices:
            return
        from hmr4d.lidar_mocap.dual_view_observer import _camera_frames

        with self.guard:
            self.seek_count += 1
        for index, image in _camera_frames(session, camera, indices, 1):
            with self.guard:
                self.delivered_frames += 1
            yield index, image


def configure_native_decode(data, decode_threads):
    """Select safe1 before any native decode; omitted flags retain the default."""
    if decode_threads is None:
        return
    if type(decode_threads) is not int or decode_threads != 1:
        raise ValueError("Native decode threads must be explicitly 1")
    with data._native_lock:
        if (
            data._native_images
            or any(data.preview_decoder.diagnostics().values())
            or getattr(data.preview_decoder, "states", {})
        ):
            raise ValueError("Configure native decoding before any native frame is decoded or cached")
        data.preview_decoder = SingleThreadNativePreviewDecoder()
    data.manifest["native_decode"] = dict(threads=1, thread_type="SLICE", profile="pyav_single_thread_slice_v1")


class CanonicalViewer(ViewerData):
    def protected_source_members(self, sid, scan, cloud, rows):
        # Canonical actor visibility is a subset. The background display must
        # still protect every original stored fit, including rejected actors.
        full_rows = ViewerData.quality_rows(self, sid).get(scan, [])
        return super().protected_source_members(sid, scan, cloud, full_rows)

    def quality_rows(self, sid):
        raw = super().quality_rows(sid)
        signature = self._color_signatures[sid]
        if sid in self.canonical_frames and self.canonical_frames[sid][0] == signature:
            return self.canonical_frames[sid][1]
        original_colors = dict(self._track_colors)
        mapping = self.canonical_admitted[sid]
        result = {}
        for scan, rows in raw.items():
            kept = []
            for row in rows:
                canonical = mapping.get((row["motion_id"], scan))
                if canonical is not None:
                    kept.append(
                        dict(
                            row,
                            motion_id=canonical["motion_id"],
                            track_id=canonical["track_id"],
                            canonical_identity=canonical.get("canonical_identity"),
                        )
                    )
            if kept:
                result[scan] = kept
        self._track_colors.update(track_color_registry(result))
        for rows in result.values():
            for row in rows:
                revision = row.get("canonical_identity")
                if revision:
                    hue = ((revision["boundary_case"] + 1) * 0.61803398875) % 1
                    color = tuple(round(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.75, 0.95))
                    previous = original_colors[revision["source_motion_id"]]
                    if sum((a - b) ** 2 for a, b in zip(color, previous)) < 10000:
                        color = tuple(round(c * 255) for c in colorsys.hsv_to_rgb((hue + 0.25) % 1, 0.75, 0.95))
                    self._track_colors[row["motion_id"]] = color
        self.canonical_frames[sid] = signature, result
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8774)
    parser.add_argument("--data-relocations", type=Path)
    parser.add_argument("--data-relocations-sha256")
    parser.add_argument("--viewer-output", type=Path)
    parser.add_argument("--display-crop", type=Path, help="Explicit typed formal-body background display sidecar")
    parser.add_argument(
        "--native-decode-threads", type=int, choices=[1], help="Explicit safe single-thread native-PTS RGB decoding"
    )
    args = parser.parse_args()
    if (args.data_relocations is None) != (args.data_relocations_sha256 is None):
        parser.error("--data-relocations and --data-relocations-sha256 must be supplied together")
    manifest = read_json(args.manifest)
    if not manifest.get("canonical_identity_revision"):
        parser.error("Select the explicitly reviewed canonical publication")
    body_path = Path(manifest["source_dataset_manifest"])
    if binding(body_path)["sha256"] != manifest["source_dataset_manifest_sha256"]:
        raise ValueError("Native body source changed")
    root = body_path.parent.parent
    display = None
    if args.display_crop is not None:
        from hmr4d.lidar_mocap.formal_ground_display import read_formal_ground_display

        display = read_formal_ground_display(
            args.display_crop, body_path=body_path, canonical_binding=binding(args.manifest)
        )
    try:
        relocations = viewer_relocations(root, args.data_relocations, args.data_relocations_sha256)
    except ValueError as exc:
        parser.error(str(exc))
    data = prepare_viewer(root, root, relocations=relocations, viewer_output=args.viewer_output)
    configure_native_decode(data, args.native_decode_threads)
    data.__class__ = CanonicalViewer
    data.canonical_admitted = {}
    data.canonical_frames = {}
    for sid, entry in manifest["exports"].items():
        check_binding(dict(path=entry["metadata"], sha256=entry["metadata_sha256"]))
        rows = read_rows(entry["metadata"])
        data.canonical_admitted[sid] = {
            (r.get("canonical_identity", {}).get("source_motion_id", r["motion_id"]), r["lidar_index"]): r for r in rows
        }
    data.manifest["canonical_publication"] = binding(args.manifest)
    data.manifest["display_scope"] = "bilaterally_admitted_canonical_samples_only"
    if display is not None:
        from hmr4d.lidar_mocap.formal_ground_display import configure_formal_ground_display

        configure_formal_ground_display(data, display)
    serve_viewer(data, port=args.port)


if __name__ == "__main__":
    main()
