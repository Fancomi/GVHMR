"""Complete fresh-source v2 labels with existing leg, member and bilateral gates."""
import math
import os
from pathlib import Path
import subprocess
import sys

from hmr4d import PROJ_ROOT
from .dual_view import file_sha
from .export import acceptance_reasons
from .pipeline import write_json
from .rally_pipeline import read_json, read_rows

METRICS = ("final_p2surface_mean_m", "final_p2surface_p90_m", "reprojection_mean_px")


def collect_absolute_qa(root):
    root = Path(root)
    labels = root / "final_labels"
    plan = read_json(labels / "repair_plan.json")
    receipt = read_json(labels / "repair_receipt.json")
    if receipt.get("windows") != len(plan["jobs"]):
        raise ValueError("Body baseline is incomplete")
    result = dict(schema="badminton-native-absolute-qa-v1", frames={}, windows={},
                  source_union_sha256=plan["source_union_sha256"], calibration_digest=plan["calibration_digest"])
    for job in plan["jobs"]:
        sid, name = job["source_id"], Path(job["source_window"]).name
        window = labels / "sessions" / sid / "windows" / name
        report = read_json(window / "fit_report.json")
        for relative, expected in report["artifacts_sha256"].items():
            if file_sha(window / relative) != expected:
                raise ValueError("Absolute QA input artifact changed")
        rows = read_rows(window / "quality.jsonl")
        accepted = 0
        for row in rows:
            if not row["accepted"]:
                continue
            if acceptance_reasons(row["qa"], row["source_point_count"]):
                raise ValueError("An accepted body fails the unchanged QA gate")
            key = f"{sid}/{name}/{row['motion_id']}/{row['lidar_index']}"
            if key in result["frames"]:
                raise ValueError("Duplicate accepted person scan")
            values = {metric: float(row["qa"][metric]) for metric in METRICS}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("Non-finite body QA")
            result["frames"][key] = values
            accepted += 1
        if accepted != report["accepted"]:
            raise ValueError("Window accepted count differs from QA rows")
        result["windows"][sid + ":" + name] = dict(fit_report_sha256=file_sha(window / "fit_report.json"), accepted=accepted)
    return result


def compare_same_v2(before, after):
    if any(before[key] != after[key] for key in ("source_union_sha256", "calibration_digest")):
        raise ValueError("Pre/post comparison uses different inputs or cameras")
    if set(before["frames"]) != set(after["frames"]):
        raise ValueError("Leg post-pass changed accepted person-scan membership")
    count = len(before["frames"])
    return dict(scope="same_v2_native_baseline_vs_guarded_leg_postpass", person_scans=count,
                independent_ground_truth=False,
                metrics={key: dict(before_mean=sum(row[key] for row in before["frames"].values())/max(count, 1),
                                   after_mean=sum(row[key] for row in after["frames"].values())/max(count, 1))
                         for key in METRICS})


def finish_new_sources(root, *, threads=1, observer_max_new_frames=500):
    root = Path(root).resolve()
    labels = root / "final_labels"
    plan = read_json(labels / "repair_plan.json")
    if read_json(plan["source_union"]).get("schema") != "badminton-new-source-repair-input-v1":
        raise ValueError("Fresh-source completion requires its explicit prepared input")
    if file_sha(plan["source_union"]) != plan["source_union_sha256"]:
        raise ValueError("Prepared source input changed")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    def run(name, *args):
        log = root / (Path(name).stem + "_fresh_completion.log")
        with log.open("ab") as handle:
            proc = subprocess.run([sys.executable, "-u", str(PROJ_ROOT / "tools" / name), *map(str, args)],
                                  cwd=PROJ_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f"{name} failed ({proc.returncode}); see {log}")
    before_path = root / "evaluation/v2_before_leg.json"
    sealed = labels / "post_refinement.json"
    if not before_path.exists():
        if sealed.exists():
            raise ValueError("Missing genuine pre-leg baseline; cannot reconstruct it after refinement")
        write_json(before_path, collect_absolute_qa(root))
    if not sealed.exists() or read_json(sealed).get("state") != "complete":
        if not sealed.exists():
            run("verify_badminton_native_repair.py", "--root", root, "--threads", threads)
            run("propose_badminton_leg_refinement.py", "--root", root, "--threads", threads)
            run("select_badminton_leg_refinement.py", "--root", root)
        run("apply_badminton_leg_refinement.py", "--root", root, "--threads", threads)
    run("verify_badminton_native_repair.py", "--root", root, "--threads", threads)
    native = read_json(labels / "native_verification.json")
    if not native.get("passed") or native.get("verified_windows") != len(plan["jobs"]):
        raise ValueError("Full native-member and SMPL reconstruction verification is incomplete")
    run("verify_badminton_dataset.py", "--output-root", labels, "--samples-per-session", 64, "--threads", threads)
    after = collect_absolute_qa(root)
    write_json(root / "evaluation/v2_after_leg.json", after)
    write_json(root / "evaluation/v2_leg_comparison.json", compare_same_v2(read_json(before_path), after))
    paired = root.with_name(root.name + "_paired")
    manifest = paired / "datasets/dataset_manifest.json"
    if not manifest.exists():
        observations = paired / "observations/manifest.json"
        if not observations.exists() or not read_json(observations).get("completed"):
            run("observe_badminton_dual_view.py", "--source-root", labels, "--output-root", paired,
                "--threads", threads, "--max-new-frames", observer_max_new_frames)
        if not read_json(observations).get("completed"):
            write_json(root / "completion_progress.json", dict(state="bilateral_observation_pending", training_published=False))
            return
        run("export_badminton_dual_view.py", "finalize", "--source-root", labels, "--output-root", paired,
            "--full-observations", observations)
    run("export_badminton_dual_view.py", "verify", "--output-root", paired)
    from .rally_viewer import prepare_viewer
    prepare_viewer(labels, labels)
    evidence = [labels / "native_verification.json", root / "leg_refinement_receipt.json",
                before_path, root / "evaluation/v2_after_leg.json", root / "evaluation/v2_leg_comparison.json", manifest]
    write_json(root / "new_source_completion_receipt.json", dict(
        state="labels_published_viewer_prepared", training_published=True,
        dataset_manifest=str(manifest), accepted=read_json(manifest)["accepted"],
        evidence={str(path): file_sha(path) for path in evidence},
        viewer_root=str(labels / "viewer"), browser_playback_verified=False,
        note="Native all-member/SMPL and bilateral export gates passed; actual browser playback remains to verify"))
