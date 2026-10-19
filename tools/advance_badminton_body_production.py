"""Continue reviewed fresh identities into isolated body fitting and full QA.

An explicit evidence-bound abstention policy preserves uncertain raw tracklets
separately. Native geometric verdicts and all body/bilateral gates are unchanged.
The controller stops for actual browser acceptance and never releases the GPU.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from hmr4d import PROJ_ROOT
from hmr4d.lidar_mocap.dual_view import file_sha
from hmr4d.lidar_mocap.expanded_dataset import binding, check_binding
from hmr4d.lidar_mocap.rally_pipeline import read_json
from hmr4d.lidar_mocap.repair_execution import require_gpu_lease, write_progress_json as write_json


def execute(args):
    root, source = args.root.resolve(), args.source_root.resolve()
    input_path = root / "new_source_input.json"
    plan = read_json(input_path)
    if (
        not root.is_relative_to(PROJ_ROOT / "outputs")
        or plan.get("schema") != "badminton-new-source-repair-input-v1"
        or Path(plan["source_root"]).resolve() != source
    ):
        raise ValueError("Select a frozen fresh-source production root")
    frozen_input = binding(input_path)
    policy = binding(args.abstention_policy)
    state_path = root / "production_stage.json"
    wait_path = root / "body_production_wait.json"
    while True:
        require_gpu_lease(args.gpu_lease)
        check_binding(frozen_input)
        check_binding(policy)
        state = read_json(state_path)
        if (root / "body_production_inputs.json").exists():
            break
        if state["stage"] == "failed":
            write_json(wait_path, dict(state="predecessor_failed", production_stage=str(state_path)))
            raise ValueError("Identity predecessor failed; body fitting is not admitted")
        if state["stage"] in ("awaiting_identity_case_adjudication", "identity_ready_for_body_prepare"):
            if file_sha(state["identity_decisions"]) != state["identity_decisions_sha256"]:
                raise ValueError("Identity predecessor output changed")
            break
        write_json(
            wait_path,
            dict(
                state="waiting_complete_native_identity_review",
                predecessor_stage=state["stage"],
                policy=policy,
                updated_utc=datetime.now(timezone.utc).isoformat(),
            ),
        )
        time.sleep(5)
    state = dict(
        schema="badminton-fresh-production-stage-v1",
        stage="conservative_identity_seal",
        source_input_sha256=frozen_input["sha256"],
        source_root=str(source),
        training_published=False,
        abstention_policy=policy,
    )
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS=str(args.threads),
        MKL_NUM_THREADS=str(args.threads),
        PYTHONFAULTHANDLER="1",
    )

    def update(**values):
        state.update(values, updated_utc=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)

    def run(stage, script, command):
        require_gpu_lease(args.gpu_lease)
        check_binding(frozen_input)
        check_binding(policy)
        log = root / (stage + ".log")
        update(stage=stage, log=str(log))
        with log.open("ab") as handle:
            result = subprocess.run(
                [sys.executable, "-u", str(PROJ_ROOT / "tools" / script), *map(str, command)],
                cwd=PROJ_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"{stage} failed ({result.returncode}); see {log}")

    try:
        identity_path = root / "identity_evidence/identity_decisions.json"
        freeze_path = root / "body_production_inputs.json"
        if not freeze_path.exists():
            if (root / "final_labels/repair_plan.json").exists():
                raise ValueError("Existing body plan has no frozen identity execution binding")
            run(
                "conservative_identity_seal",
                "build_badminton_identity_decisions.py",
                ["--root", root, "--abstention-policy", args.abstention_policy.resolve()],
            )
            identity = read_json(identity_path)
            if identity.get("review_status") != "reviewed" or identity.get("unresolved_cases"):
                raise ValueError("Identity processing remains unresolved")
            write_json(
                freeze_path,
                dict(
                    schema="badminton-fresh-body-production-inputs-v1",
                    source_input=frozen_input,
                    identity_decisions=binding(identity_path),
                    policy=policy,
                    native_pose_receipt=binding(root / "pose_receipt.json"),
                ),
            )
        frozen = read_json(freeze_path)
        for name in ("source_input", "identity_decisions", "policy", "native_pose_receipt"):
            check_binding(frozen[name])
        if frozen["source_input"] != frozen_input or frozen["policy"] != policy:
            raise ValueError("Body production input contract changed")
        update(identity_decisions=frozen["identity_decisions"], body_inputs=binding(freeze_path))
        run(
            "body_prepare",
            "repair_badminton_body_labels.py",
            [
                "--stage",
                "prepare",
                "--union",
                input_path,
                "--source-root",
                source,
                "--output-root",
                root / "final_labels",
                "--native-poses-root",
                root,
                "--joint-ray-filter",
                "--distal-pose-weight",
                "0.03",
                "--lower-limb-reprojection-weight",
                "2",
                "--identity-decisions",
                identity_path,
                "--threads",
                args.threads,
            ],
        )
        if not (root / "final_labels/post_refinement.json").exists():
            run(
                "body_fitting",
                "run_badminton_repair_windows.py",
                ["--root", root, "--union", input_path, "--source-root", source, "--threads", args.threads],
            )
        run(
            "body_native_leg_and_bilateral_qa",
            "finish_badminton_body_repair.py",
            ["--root", root, "--threads", args.threads, "--observer-max-new-frames", "1000000"],
        )
        complete_path = root / "new_source_completion_receipt.json"
        complete = read_json(complete_path)
        if complete.get("state") != "labels_published_viewer_prepared":
            raise ValueError("Fresh label QA/export is incomplete")
        for path, sha in complete["evidence"].items():
            if file_sha(path) != sha:
                raise ValueError("Completed native/body/bilateral evidence changed")
        update(
            stage="awaiting_browser_playback_verification",
            source_completion=binding(complete_path),
            source_paired_accepted=complete["accepted"],
            browser_playback_verified=False,
            expanded_training_published=False,
        )
        write_json(
            wait_path, dict(state="body_labels_qa_complete_browser_pending", source_completion=binding(complete_path))
        )
    except Exception as error:
        update(stage="failed", failed_stage=state["stage"], error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gpu-lease", type=Path, required=True)
    parser.add_argument("--abstention-policy", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    execute(args)


if __name__ == "__main__":
    main()
