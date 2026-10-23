"""Complete fresh-source v2 labels with existing leg, member and bilateral gates."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from hmr4d import PROJ_ROOT
from .dual_view import file_sha
from .export import acceptance_reasons
from .pipeline import write_json
from .rally_pipeline import read_json, read_rows
from .relocations import load_pilot_run_relocations

METRICS = ("final_p2surface_mean_m", "final_p2surface_p90_m", "reprojection_mean_px")


def collect_absolute_qa(root):
    root = Path(root)
    labels = root / "final_labels"
    plan = read_json(labels / "repair_plan.json")
    receipt = read_json(labels / "repair_receipt.json")
    if receipt.get("windows") != len(plan["jobs"]):
        raise ValueError("Body baseline is incomplete")
    result = dict(
        schema="badminton-native-absolute-qa-v1",
        frames={},
        windows={},
        source_union_sha256=plan["source_union_sha256"],
        calibration_digest=plan["calibration_digest"],
    )
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
        result["windows"][sid + ":" + name] = dict(
            fit_report_sha256=file_sha(window / "fit_report.json"), accepted=accepted
        )
    return result


def compare_same_v2(before, after):
    if any(before[key] != after[key] for key in ("source_union_sha256", "calibration_digest")):
        raise ValueError("Pre/post comparison uses different inputs or cameras")
    if set(before["frames"]) != set(after["frames"]):
        raise ValueError("Leg post-pass changed accepted person-scan membership")
    count = len(before["frames"])
    return dict(
        scope="same_v2_native_baseline_vs_guarded_leg_postpass",
        person_scans=count,
        independent_ground_truth=False,
        metrics={
            key: dict(
                before_mean=sum(row[key] for row in before["frames"].values()) / max(count, 1),
                after_mean=sum(row[key] for row in after["frames"].values()) / max(count, 1),
            )
            for key in METRICS
        },
    )


def _reuse_formal_postleg(root, plan, receipt_path, expected_sha256, threads):
    """Revalidate an existing CPU completion without starting a missing pass."""
    from .formal_completion import STATE, verify_formal_postleg_cpu

    canonical = root / "evaluation/formal_postleg_cpu_receipt.json"
    path = Path(receipt_path).resolve()
    if (
        root.name != "repair"
        or path != canonical
        or not path.is_file()
        or type(threads) is not int
        or threads not in (1, 2)
        or not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("Select the existing exact formal postleg CPU receipt and its SHA256")
    # The open receipt also prevents deletion/replacement on Windows while the
    # unchanged strict verifier selects its existing-receipt branch.
    with path.open("rb") as handle:
        payload = handle.read()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("Formal postleg CPU receipt SHA256 differs")
        receipt = json.loads(payload)
        if (
            receipt.get("schema") != "gvhmr-formal-postleg-cpu-completion-v1"
            or receipt.get("state") != STATE
            or Path(receipt.get("root", "")).resolve() != root.parent
            or receipt.get("source_input") != dict(path=plan["source_union"], sha256=plan["source_union_sha256"])
            or receipt.get("calibration_digest") != plan["calibration_digest"]
            or receipt.get("windows") != len(plan["jobs"])
            or receipt.get("gpu_started") is not False
            or receipt.get("training_published") is not False
            or set(receipt.get("workers", {})) != {"native", "dataset"}
        ):
            raise ValueError("Existing formal postleg CPU completion belongs to different or incomplete evidence")
        checked = verify_formal_postleg_cpu(root.parent, receipt["baseline"], receipt["leg_execution"], threads=threads)
        handle.seek(0)
        if checked != receipt or handle.read() != payload or file_sha(path) != expected_sha256:
            raise ValueError("Formal postleg CPU completion changed during reuse")
        if receipt.get("accepted") != read_json(root / "final_labels/datasets/dataset_manifest.json")["accepted"]:
            raise ValueError("Formal postleg CPU count differs from its actual body publication")
    return receipt, dict(path=str(path), sha256=expected_sha256)


def _formal_observations(labels, observations, receipt):
    if not observations.exists() or read_json(observations).get("completed") is not True:
        return False
    value = read_json(observations)
    dataset = labels / "datasets/dataset_manifest.json"
    if (
        value.get("source_accepted") != receipt["accepted"]
        or value.get("identity", {}).get("calibration_digest") != receipt["calibration_digest"]
        or value.get("identity", {}).get("source", {}).get("dataset_manifest")
        != dict(path=str(dataset), sha256=file_sha(dataset))
    ):
        raise ValueError("Completed bilateral observations belong to another formal body publication")
    return True


def finish_new_sources(
    root,
    *,
    threads=1,
    observer_max_new_frames=500,
    formal_postleg_cpu_receipt=None,
    formal_postleg_cpu_receipt_sha256=None,
):
    root = Path(root).resolve()
    labels = root / "final_labels"
    plan = read_json(labels / "repair_plan.json")
    if read_json(plan["source_union"]).get("schema") != "badminton-new-source-repair-input-v1":
        raise ValueError("Fresh-source completion requires its explicit prepared input")
    if file_sha(plan["source_union"]) != plan["source_union_sha256"]:
        raise ValueError("Prepared source input changed")
    if (formal_postleg_cpu_receipt is None) != (formal_postleg_cpu_receipt_sha256 is None):
        raise ValueError("Formal postleg CPU receipt path and SHA256 must be supplied together")
    registry_path = labels / "source_registry.json"
    formal = (
        registry_path.exists()
        and read_json(registry_path).get("schema") == "badminton-formal-derived-source-registry-v1"
    )
    if formal != (formal_postleg_cpu_receipt is not None):
        raise ValueError(
            "Formal inputs require their explicit existing postleg CPU receipt; legacy inputs cannot reuse it"
        )
    formal_receipt, formal_binding = None, None
    if formal:
        formal_receipt, formal_binding = _reuse_formal_postleg(
            root, plan, formal_postleg_cpu_receipt, formal_postleg_cpu_receipt_sha256, threads
        )
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))

    def run(name, *args):
        log = root / (Path(name).stem + "_fresh_completion.log")
        with log.open("ab") as handle:
            proc = subprocess.run(
                [sys.executable, "-u", str(PROJ_ROOT / "tools" / name), *map(str, args)],
                cwd=PROJ_ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode:
            raise RuntimeError(f"{name} failed ({proc.returncode}); see {log}")

    before_path = root / "evaluation/v2_before_leg.json"
    sealed = labels / "post_refinement.json"
    if not formal and not before_path.exists():
        if sealed.exists():
            raise ValueError("Missing genuine pre-leg baseline; cannot reconstruct it after refinement")
        write_json(before_path, collect_absolute_qa(root))
    if not formal and (not sealed.exists() or read_json(sealed).get("state") != "complete"):
        if not sealed.exists():
            run("verify_badminton_native_repair.py", "--root", root, "--threads", threads)
            run("propose_badminton_leg_refinement.py", "--root", root, "--threads", threads)
            run("select_badminton_leg_refinement.py", "--root", root)
        run("apply_badminton_leg_refinement.py", "--root", root, "--threads", threads)
    if not formal:
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
    observations = paired / "observations/manifest.json"
    if formal and not _formal_observations(labels, observations, formal_receipt):
        progress = dict(
            state="bilateral_observation_pending",
            training_published=False,
            formal_postleg_cpu_receipt=formal_binding,
            observations=str(observations),
            observer_started=False,
            note="Run the independent native dual-view observer to completion with an explicitly selected device",
        )
        write_json(root / "completion_progress.json", progress)
        return progress
    if not manifest.exists():
        if not observations.exists() or not read_json(observations).get("completed"):
            run(
                "observe_badminton_dual_view.py",
                "--source-root",
                labels,
                "--output-root",
                paired,
                "--threads",
                threads,
                "--max-new-frames",
                observer_max_new_frames,
            )
        if not read_json(observations).get("completed"):
            write_json(
                root / "completion_progress.json", dict(state="bilateral_observation_pending", training_published=False)
            )
            return
        run(
            "export_badminton_dual_view.py",
            "finalize",
            "--source-root",
            labels,
            "--output-root",
            paired,
            "--full-observations",
            observations,
        )
    run("export_badminton_dual_view.py", "verify", "--output-root", paired)
    if formal:
        publication = read_json(manifest)
        if (
            publication.get("source_dataset_manifest") != str(labels / "datasets/dataset_manifest.json")
            or publication.get("source_dataset_manifest_sha256") != file_sha(labels / "datasets/dataset_manifest.json")
            or publication.get("calibration_digest") != formal_receipt["calibration_digest"]
            or publication.get("full_observation_manifest") != str(observations)
            or publication.get("full_observation_manifest_sha256") != file_sha(observations)
        ):
            raise ValueError("Bilateral publication does not bind this formal body and complete observations")
    from .rally_viewer import prepare_viewer

    relocations = load_pilot_run_relocations(labels)
    prepare_viewer(labels, labels, relocations=relocations)
    evidence = [
        labels / "native_verification.json",
        root / "leg_refinement_receipt.json",
        before_path,
        root / "evaluation/v2_after_leg.json",
        root / "evaluation/v2_leg_comparison.json",
        manifest,
    ]
    extra = {}
    if formal:
        evidence.extend(
            [
                Path(formal_binding["path"]),
                observations,
                paired / "verification.json",
                labels / "viewer/manifest.json",
                labels / "datasets/dataset_manifest.json",
            ]
        )
        extra = dict(
            formal_postleg_cpu_receipt=formal_binding,
            bilateral_labels_published=True,
            archive_published=False,
            consumer_admitted=False,
        )
        if file_sha(formal_binding["path"]) != formal_binding["sha256"]:
            raise ValueError("Formal postleg CPU completion changed before bilateral completion")
        for binding in formal_receipt["evidence"].values():
            if file_sha(binding["path"]) != binding["sha256"]:
                raise ValueError("Formal postleg CPU evidence changed before bilateral completion")
    completion = dict(
        state="labels_published_viewer_prepared",
        training_published=not formal,
        dataset_manifest=str(manifest),
        accepted=read_json(manifest)["accepted"],
        evidence={str(path): file_sha(path) for path in evidence},
        viewer_root=str(labels / "viewer"),
        browser_playback_verified=False,
        note="Native all-member/SMPL and bilateral export gates passed; actual browser playback remains to verify",
        **extra,
    )
    write_json(root / "new_source_completion_receipt.json", completion)
    return completion
