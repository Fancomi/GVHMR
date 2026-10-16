"""Add a hash-bound delivery receipt without rewriting published training data."""

import argparse
from pathlib import Path

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json
from hmr4d.lidar_mocap.rally_report import report_rally_delivery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJ_ROOT / "outputs/badminton_rallies_v1")
    parser.add_argument("--paired-root", type=Path, default=PROJ_ROOT / "outputs/badminton_rallies_v1_paired")
    args = parser.parse_args()
    source = args.source_root.resolve()
    paired = args.paired_root.resolve()
    manifest_path = paired / "datasets/dataset_manifest.json"
    manifest_sha = file_sha(manifest_path)
    manifest = read_json(manifest_path)
    source_manifest_path = source / "datasets/dataset_manifest.json"
    source_manifest = read_json(source_manifest_path)
    verification_path = paired / "verification.json"
    verification = read_json(verification_path)
    if (
        manifest.get("status") != "complete"
        or source_manifest.get("scope") != "sealed_seven_source_rally_subset"
        or Path(manifest["source_dataset_manifest"]).resolve() != source_manifest_path
        or file_sha(source_manifest_path) != manifest["source_dataset_manifest_sha256"]
        or verification["dataset_manifest_sha256"] != manifest_sha
        or verification["accepted"] != manifest["accepted"]
    ):
        raise ValueError("Delivery receipt requires the unchanged verified sealed rally export")
    quality = report_rally_delivery(source, paired, manifest)
    state = read_json(source / "build_status.json")
    receipt = dict(
        schema="badminton-sealed-rally-delivery-receipt-v1",
        scope="full_sealed_seven_source_rally_subset",
        published_manifest=dict(path=str(manifest_path), sha256=manifest_sha, scope=manifest["scope"]),
        source_manifest=dict(path=str(source_manifest_path), sha256=file_sha(source_manifest_path)),
        verification=dict(path=str(verification_path), sha256=file_sha(verification_path)),
        quality_summary=dict(
            path=str(paired / "quality_summary.json"), sha256=file_sha(paired / "quality_summary.json")
        ),
        final_person_frames=quality["final_paired_person_frames"],
        splits=quality["splits"],
        recorded_resume_calls_seconds={
            sid: entry["last_call_seconds"] for sid, entry in state["sources"].items() if "last_call_seconds" in entry
        },
        timing_limit="Resumed cached production calls only; excludes earlier work and is not an uncached end-to-end benchmark",
        scope_note="Source manifest identifies the sealed rally subset. Published training manifest remains unchanged.",
    )
    if file_sha(manifest_path) != manifest_sha:
        raise ValueError("Published training manifest changed while reporting")
    write_json(paired / "delivery_receipt.json", receipt)
    print(f"Reported {receipt['final_person_frames']} verified person frames: {paired / 'delivery_receipt.json'}")


if __name__ == "__main__":
    main()
