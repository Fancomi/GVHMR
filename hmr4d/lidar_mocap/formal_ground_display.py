"""Explicit formal-body background preview using the unchanged native renderer.

This does not admit labels or change a calibration/fit. The accepted review
sidecar supplies only the existing ground-mask operation; the formal body,
finish and canonical view must each be bound to their own real bytes.
"""

import copy
import hashlib
import json
from pathlib import Path
import re

from .rally_viewer import GROUND_DISPLAY_SCHEMA, badminton_detection_bbox, load_display_bbox

SCHEMA = "badminton-formal-background-ground-display-v1"
SOURCE = "20261022_161632"
GEOMETRY_FIELDS = (
    "cameras",
    "Q_raw_to_viewer",
    "reference_plane",
    "bbox",
    "court_registration",
    "court",
    "timing_policy",
    "geometry_variant",
    "bbox_rotation_convention",
    "bbox_geometry",
    "coordinate_convention",
    "physical_limits",
    "Q_rounding",
)
COMMON_FIELDS = {
    "sport",
    "scope",
    "source_ids",
    "base_bbox_sha256",
    "ground_ignore_id",
    "ground_positive_local_z_extension_m",
    "preview_only",
    "formal_crop_unchanged",
}


def _need(value, message):
    if not value:
        raise ValueError(message)


def _binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _same(left, right):
    return Path(left).resolve() == Path(right).resolve()


def read_formal_ground_display(path, *, body_path, canonical_binding):
    """Validate small immutable metadata; do not read tensors/models/raw scans."""
    observed = []

    def take(item, expected=None):
        _need(isinstance(item, dict) and set(item) == {"path", "sha256"}, "Malformed formal display binding")
        _need(
            isinstance(item["path"], str)
            and Path(item["path"]).is_absolute()
            and re.fullmatch(r"[0-9a-f]{64}", str(item["sha256"])) is not None,
            "Invalid formal display path/SHA",
        )
        _need(expected is None or _same(item["path"], expected), "Unexpected formal display parent path")
        actual = _binding(item["path"])
        _need(actual["sha256"] == item["sha256"], "Formal display bound bytes changed")
        value = json.loads(Path(item["path"]).read_bytes())
        _need(_binding(item["path"]) == actual, "Formal display input changed while reading")
        observed.append(actual)
        return value

    selected = _binding(path)
    settings = take(selected)
    _need(
        set(settings)
        == COMMON_FIELDS
        | {"schema", "dataset_manifest", "parent_display_crop", "formal_completion", "canonical_publication"}
        and settings["schema"] == SCHEMA,
        "Select the explicit formal ground display schema",
    )
    _need(
        settings["sport"] == "badminton"
        and settings["scope"] == "background_display_only"
        and settings["source_ids"] == [SOURCE]
        and settings["preview_only"] is True
        and settings["formal_crop_unchanged"] is True
        and type(settings["ground_positive_local_z_extension_m"]) in (int, float)
        and settings["ground_positive_local_z_extension_m"] == 0.015,
        "Formal ground display scope mismatch",
    )
    root = Path(body_path).resolve().parent.parent
    body = take(settings["dataset_manifest"], root / "datasets/dataset_manifest.json")
    _need(
        body.get("schema") == "badminton-ptv3-dataset-v1"
        and body.get("training_published") is False
        and body.get("review_only") is not True
        and set(body.get("exports", {})) == {SOURCE},
        "Select an actual unpublished formal body, not the review body",
    )
    canonical = take(settings["canonical_publication"])
    _need(
        settings["canonical_publication"] == canonical_binding
        and canonical.get("source_dataset_manifest_sha256") == settings["dataset_manifest"]["sha256"]
        and _same(canonical["source_dataset_manifest"], body_path)
        and set(canonical.get("exports", {})) == {SOURCE}
        and canonical.get("schema")
        in {"badminton-canonical-bilateral-publication-v1", "badminton-canonical-bilateral-publication-v2"}
        and bool(canonical.get("canonical_identity_revision")),
        "Canonical display uses another formal body",
    )
    finish = take(settings["formal_completion"], root.parent / "new_source_completion_receipt.json")
    _need(
        finish.get("state") == "labels_published_viewer_prepared"
        and finish.get("training_published") is False
        and finish.get("bilateral_labels_published") is True,
        "Formal finish is incomplete or published training",
    )
    evidence = finish.get("evidence", {})
    body_key = next((key for key in evidence if _same(key, body_path)), None)
    _need(
        body_key is not None and evidence[body_key] == settings["dataset_manifest"]["sha256"],
        "Finish does not bind this actual formal body",
    )
    postleg_binding = finish.get("formal_postleg_cpu_receipt")
    postleg = take(postleg_binding, root.parent / "evaluation/formal_postleg_cpu_receipt.json")
    _need(
        postleg.get("schema") == "gvhmr-formal-postleg-cpu-completion-v1"
        and postleg.get("state") == "postleg_native_qa_complete_bilateral_browser_archive_pending"
        and postleg.get("training_published") is False
        and evidence.get(postleg_binding["path"]) == postleg_binding["sha256"],
        "Missing genuine formal postleg binding",
    )
    paired_path = root.parent.with_name(root.parent.name + "_paired") / "datasets/dataset_manifest.json"
    _need(_same(finish["dataset_manifest"], paired_path), "Finish points to another bilateral publication")
    paired = take(dict(path=str(paired_path), sha256=evidence.get(str(paired_path))))
    _need(
        paired.get("source_dataset_manifest_sha256") == settings["dataset_manifest"]["sha256"]
        and _same(paired["source_dataset_manifest"], body_path)
        and all(type(item.get("accepted")) is int for item in (body, finish, paired, canonical))
        and finish["accepted"] > 0
        and body["accepted"] >= finish["accepted"]
        and finish["accepted"] == paired.get("accepted") == canonical.get("accepted"),
        "Formal body/bilateral/canonical completion mismatch",
    )
    parent = take(settings["parent_display_crop"])
    _need(
        set(parent) == COMMON_FIELDS | {"schema", "dataset_manifest"}
        and parent["schema"] == GROUND_DISPLAY_SCHEMA
        and all(parent[key] == settings[key] for key in COMMON_FIELDS),
        "Accepted ground settings or source changed",
    )
    review = take(parent["dataset_manifest"])
    _need(
        review.get("review_only") is True
        and review.get("training_published") is False
        and set(review.get("exports", {})) == {SOURCE},
        "Ground parent is not the actual review body",
    )
    review_root = Path(parent["dataset_manifest"]["path"]).resolve().parent.parent
    calibration = take(_binding(root / "calibration/calibration_snapshot.json"))
    previous = take(_binding(review_root / "calibration/calibration_snapshot.json"))
    _need(
        body.get("calibration_digest") == calibration.get("snapshot_digest")
        and review.get("calibration_digest") == previous.get("snapshot_digest")
        and all(
            key in calibration and key in previous and calibration[key] == previous[key] for key in GEOMETRY_FIELDS
        ),
        "Formal/review camera, court, timing or bbox geometry changed",
    )
    # The real post-leg exporter binds its union, not registry fields on the
    # body. Follow that unchanged input and the genuine CPU completion instead.
    source_binding = postleg.get("source_input")
    source = take(source_binding, root.parent / "new_source_input.json")
    _need(
        source.get("schema") == "badminton-new-source-repair-input-v1"
        and body.get("source_union_manifest_sha256") == source_binding["sha256"]
        and source.get("calibration_digest") == body["calibration_digest"]
        and _same(source["source_root"], root.parent.parent / "source"),
        "Formal body/source input binding mismatch",
    )
    registry_binding = dict(
        path=str(Path(source["source_root"]) / "source_registry.json"), sha256=source["source_registry_sha256"]
    )
    registry = take(registry_binding)
    adjacent = take(dict(path=str(root / "source_registry.json"), sha256=registry_binding["sha256"]))
    _need(adjacent == registry, "Prepared formal registry differs from the original source input")
    old_registry = take(dict(path=review["source_registry"], sha256=review["source_registry_sha256"]))
    _need(
        registry.get("schema") == "badminton-formal-derived-source-registry-v1"
        and set(registry.get("sources", {})) == set(old_registry.get("sources", {})) == {SOURCE}
        and registry["sources"][SOURCE] == old_registry["sources"][SOURCE],
        "Formal/review native source lineage changed",
    )
    return dict(
        path=selected["path"],
        sha256=selected["sha256"],
        settings=settings,
        bindings=observed,
        original_inputs_unchanged=True,
        calibration=calibration,
    )


def configure_formal_ground_display(data, evidence):
    """Configure the existing renderer; preserve its person arrays and matrices."""
    settings = evidence["settings"]
    root = data.root.resolve()
    _need(
        root == data.baseline.resolve()
        and _same(settings["dataset_manifest"]["path"], root / "datasets/dataset_manifest.json")
        and data.viewer_output is not None
        and not data.viewer_output.resolve().is_relative_to(root)
        and set(data.manifest["sources"]) == {SOURCE}
        and data.calibration.data == evidence["calibration"],
        "Prepared formal viewer differs from validated parent or output scope",
    )
    for item in evidence["bindings"]:
        _need(_binding(item["path"]) == item, "Formal display inputs changed before renderer configuration")
    # This is the existing geometry primitive, not the review-parent admission
    # validator. The evidence retained by this viewer remains the new schema.
    adjusted, parent = load_display_bbox(data.calibration, settings["parent_display_crop"]["path"])
    _need(parent["sha256"] == settings["parent_display_crop"]["sha256"], "Accepted ground mask changed")
    data.viewer_bbox = adjusted
    data.base_bbox = badminton_detection_bbox(data.calibration.data["bbox"])
    data.ground_display = {key: copy.deepcopy(value) for key, value in evidence.items() if key != "calibration"}
    data.manifest["point_filter"].update(
        boxes=adjusted["boxes"],
        base_boxes=data.base_bbox["boxes"],
        display_crop=data.ground_display,
        rule="Original Region/net; 15mm background ground preview protecting all formal stored fit members",
        preview_only=True,
        formal_crop_unchanged=True,
    )
    data.manifest.update(
        preview_only=True,
        formal_crop_unchanged=True,
        training_published=False,
        body_dataset_manifest=dict(settings["dataset_manifest"]),
    )
    # These are generated viewer assets only. The body and its review_only field
    # remain untouched; the display preview does not publish training labels.
    output = data.viewer_output
    (output / "manifest.json").write_text(
        json.dumps(data.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    html = output / "index.html"
    text = html.read_text(encoding="utf-8")
    banner = "<span>正式来源 · 15毫米背景显示预览 · 保留原裁剪与人物 · 未发布训练</span>"
    html.write_text(text.replace('<div class="counts">', '<div class="counts">' + banner, 1), encoding="utf-8")
