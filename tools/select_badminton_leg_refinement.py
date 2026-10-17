"""Select native leg proposals with continuous correction bounds; do not write labels."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.leg_refinement import BODY_JOINTS
from hmr4d.lidar_mocap.leg_selection import select_correction_sequence
from hmr4d.lidar_mocap.pipeline import write_json
from hmr4d.lidar_mocap.rally_pipeline import read_json, read_rows


def select_windows(root, require_complete=True):
    labels = root / "final_labels"
    proposal = root / "leg_refinement_proposals/receipt.json"
    receipt = read_json(proposal)
    if require_complete and receipt["state"] != "complete":
        raise ValueError("Leg proposals are incomplete")
    result = dict(
        state="complete" if receipt["state"] == "complete" else "partial",
        windows={},
        selected_frames=0,
        source_proposals_sha256=file_sha(proposal),
        source_union_sha256=receipt["source_union_sha256"],
        refinement_policy_sha256=receipt["policy_sha256"],
        selection_policy_sha256=file_sha(PROJ_ROOT / "hmr4d/lidar_mocap/leg_selection.py"),
        maximum_correction_step_degrees=20.5,
        training_published=False,
        applied=False,
    )
    identity = np.broadcast_to(np.eye(3), (8, 3, 3))
    for key, info in receipt["windows"].items():
        sid, window = key.split(":")
        source = labels / "sessions" / sid / "windows" / window
        p = read_json(info["path"])
        if file_sha(source / "fit_report.json") != p["source_fit_report_sha256"]:
            raise ValueError("Source fit changed after proposal")
        proposals = {(r["motion_id"], r["lidar_index"]): r for r in p["proposals"]}
        groups = defaultdict(list)
        for row in read_rows(source / "quality.jsonl"):
            candidates = [dict(delta_rotations=identity, gain_px=0, proposal=None, candidate=None)]
            proposal = proposals.get((row["motion_id"], row["lidar_index"]))
            if proposal:
                with np.load(row["fit_path"]) as f:
                    base = Rotation.from_rotvec(f["body_pose"].reshape(23, 3)).as_matrix()[BODY_JOINTS]
                for i, c in enumerate(proposal["candidates"]):
                    if (
                        not c["accepted"]
                        or c["surface_mean_m"] > 0.065
                        or c["surface_p90_m"] > 0.12
                        or c["max_penetration_m"] > 0.12
                    ):
                        continue
                    if file_sha(c["arrays_path"]) != c["arrays_sha256"]:
                        raise ValueError("Leg proposal arrays changed")
                    with np.load(c["arrays_path"]) as f:
                        matrix = Rotation.from_rotvec(f["body_pose"].reshape(23, 3)).as_matrix()[BODY_JOINTS]
                    candidates.append(
                        dict(
                            delta_rotations=matrix @ base.transpose(0, 2, 1),
                            gain_px=c["mean_leg_improvement_px"],
                            proposal=proposal,
                            candidate=i,
                        )
                    )
            groups[int(row["motion_id"].split(":")[-2])].append(
                dict(time_us=row["scan_mid_us"], states=candidates, row=row)
            )
        selected = []
        for frames in groups.values():
            frames.sort(key=lambda r: r["time_us"])
            if len({r["time_us"] for r in frames}) != len(frames):
                raise ValueError("Canonical person has duplicate fitted scans")
            choices = select_correction_sequence(frames)
            for frame, index in zip(frames, choices):
                state = frame["states"][index]
                if state["proposal"] is not None:
                    proposal = state["proposal"]
                    candidate = proposal["candidates"][state["candidate"]]
                    selected.append(
                        dict(
                            motion_id=frame["row"]["motion_id"],
                            lidar_index=frame["row"]["lidar_index"],
                            source_fit_path=proposal["source_fit_path"],
                            source_fit_sha256=proposal["source_fit_sha256"],
                            candidate=candidate,
                            ray_witness=proposal["ray_witness"],
                        )
                    )
        result["windows"][key] = dict(
            source_fit_report_sha256=p["source_fit_report_sha256"],
            proposal_path=info["path"],
            proposal_sha256=info["sha256"],
            candidate_frames=len(proposals),
            selected=selected,
        )
        result["selected_frames"] += len(selected)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJ_ROOT / "outputs/badminton_body_repair_v2")
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    result = select_windows(args.root.resolve(), not args.allow_partial)
    dest = args.root / "leg_refinement_selection.json"
    write_json(dest, result)
    print(
        dict(state=result["state"], windows=len(result["windows"]), selected_frames=result["selected_frames"]),
        flush=True,
    )


if __name__ == "__main__":
    main()
