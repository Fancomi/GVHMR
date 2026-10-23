"""Internal CPU adapter: verify formal exports with their explicitly frozen relocation resolver.

The original fixed24/neutral-SMPL numerical loop and report are preserved. This
entry preflights every declared session before opening a model; it never infers a
relocation table or changes the strict default of load_run_session.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import smplx
import torch

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.calibration import CalibrationSnapshot
from hmr4d.lidar_mocap.rally_pipeline import load_run_session
from hmr4d.lidar_mocap.relocations import load_formal_run_relocations
from hmr4d.utils.body_model.contracts import forward_smpl_at_pelvis

ORIGINAL_DATASET_SHA256 = "1d8447e7af1f536a09fda98bee67b201ad1871fc6802d51ed70dd4f580bbbe25"


def verify_dataset(args):
    original = PROJ_ROOT / "tools/verify_badminton_dataset.py"
    if hashlib.sha256(original.read_bytes()).hexdigest() != ORIGINAL_DATASET_SHA256:
        raise ValueError("The original numerical dataset verifier changed")
    torch.set_num_threads(args.threads)
    calibration = CalibrationSnapshot.load(args.output_root / "calibration/calibration_snapshot.json")
    manifest = json.loads((args.output_root / "datasets/dataset_manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((args.output_root / "source_registry.json").read_text(encoding="utf-8-sig"))
    relocations = load_formal_run_relocations(args.output_root, registry, calibration.data)
    if set(manifest["exports"]) != set(registry["sources"]):
        raise ValueError("Formal dataset source scope differs from its explicitly admitted registry")
    sessions = {
        sid: load_run_session(args.output_root, calibration, sid, relocations=relocations)
        for sid in manifest["exports"]
    }
    model = smplx.create(PROJ_ROOT / "inputs/checkpoints/body_models", model_type="smpl", gender="neutral").eval()
    report = {
        "scope": "Original recorded fixed24 returns and native neutral SMPL reconstruction",
        "sessions": {},
        "calibration_digest": calibration.data["snapshot_digest"],
    }
    for sid, entry in manifest["exports"].items():
        if not entry["accepted"]:
            continue
        data = joblib.load(entry["path"])
        rows = [json.loads(line) for line in Path(entry["metadata"]).read_text(encoding="utf-8").splitlines()]
        if len(rows) != len(data["pc"]):
            raise ValueError("Metadata and tensor sample counts differ")
        session = sessions[sid]
        pairs = session.pairs()
        selected = np.unique(np.linspace(0, len(rows) - 1, min(args.samples_per_session, len(rows))).astype(int))
        cloud_error = mesh_error = pelvis_error = 0
        for index in selected:
            row = rows[index]
            cloud = session.read_cloud(pairs[row["lidar_index"]])
            expected = cloud[np.asarray(row["source_segment_indices"], dtype=np.int64)]
            cloud_error = max(cloud_error, float(np.max(np.abs(data["pc"][index].numpy() - expected))))
            with torch.no_grad():
                vertices, joints = forward_smpl_at_pelvis(
                    model,
                    data["root_pose"][index : index + 1],
                    data["body_pose"][index : index + 1],
                    data["shape"][index : index + 1],
                    data["transl"][index : index + 1],
                )
            with np.load(row["fit_path"]) as fit:
                mesh_error = max(mesh_error, float(np.max(np.abs(vertices[0].numpy() - fit["vertices"]))))
            pelvis_error = max(pelvis_error, float((joints[:, 0] - data["transl"][index : index + 1]).abs().max()))
        if cloud_error > 1e-6 or mesh_error > 1e-4 or pelvis_error > 1e-6:
            raise ValueError(f"Actual source/body contract mismatch {sid}: {cloud_error}, {mesh_error}, {pelvis_error}")
        report["sessions"][sid] = {
            "exported": len(rows),
            "checked": len(selected),
            "max_source_cloud_error_m": cloud_error,
            "max_rebuilt_mesh_error_m": mesh_error,
            "max_pelvis_translation_error_m": pelvis_error,
        }
    path = args.report or args.output_root / "datasets/verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples-per-session", type=int, default=64)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if args.samples_per_session < 1:
        parser.error("samples-per-session must be positive")
    verify_dataset(args)


if __name__ == "__main__":
    main()
