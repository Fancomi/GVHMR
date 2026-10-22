"""Audit or import unchanged completed parent bodies on CPU, without training admission."""

import os

os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid

import psutil

from hmr4d import PROJ_ROOT


def main():
    process = psutil.Process()
    process.cpu_affinity([26, 27])
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "apply"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input-equivalence", type=Path)
    parser.add_argument("--input-equivalence-sha256")
    parser.add_argument("--parent-root", type=Path, action="append")
    parser.add_argument("--original-commit", action="append", default=[], metavar="PARENT_ROOT=COMMIT")
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--audit-sha256")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    os.environ.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    import torch

    torch.set_num_threads(args.threads)
    from hmr4d.lidar_mocap.formal_import_audit import audit_formal_numeric_import
    from hmr4d.lidar_mocap.formal_numeric_import import _write_once, import_formal_numeric_bodies
    from hmr4d.lidar_mocap.pilot import binding

    root = args.root.resolve()
    if root == (PROJ_ROOT / "outputs").resolve() or not root.is_relative_to((PROJ_ROOT / "outputs").resolve()):
        parser.error("Use the separate prepared formal output root")
    started = time.perf_counter()
    if args.phase == "audit":
        if not args.input_equivalence or not args.input_equivalence_sha256 or not args.parent_root:
            parser.error("Audit requires exact equivalence binding and explicit complete parent roots")
        commits = {}
        for value in args.original_commit:
            if "=" not in value:
                parser.error("Original source revision uses PARENT_ROOT=COMMIT")
            name, commit = value.rsplit("=", 1)
            if not commit or str(Path(name).resolve()) in commits:
                parser.error("Each original parent revision must be explicit and unique")
            commits[str(Path(name).resolve())] = commit
        result = audit_formal_numeric_import(
            root,
            dict(path=str(args.input_equivalence.resolve()), sha256=args.input_equivalence_sha256),
            args.parent_root,
            original_commits=commits,
        )
        proof = _write_once(root / "import_evidence/numeric_import_audit.json", result)
    else:
        if not args.audit or not args.audit_sha256:
            parser.error("Apply requires the actual completed numerical audit path and SHA256")
        result = import_formal_numeric_bodies(root, dict(path=str(args.audit.resolve()), sha256=args.audit_sha256))
        proof = result["receipt"]
    execution = _write_once(
        root / "import_evidence" / ("execution_" + uuid.uuid4().hex + ".json"),
        dict(
            schema="badminton-formal-numeric-import-execution-v1",
            state="complete",
            phase=args.phase,
            result=proof,
            completed_utc=datetime.now(timezone.utc).isoformat(),
            actual_wall_seconds=time.perf_counter() - started,
            process=dict(
                pid=process.pid,
                create_time=process.create_time(),
                actual_affinity=process.cpu_affinity(),
                actual_priority=int(process.nice()),
                threads=args.threads,
            ),
            gpu_started=False,
            training_published=False,
        ),
    )
    print(
        dict(
            state=result["state"],
            evidence=proof,
            actual_execution=execution,
            gpu_started=False,
            training_published=False,
        )
    )


if __name__ == "__main__":
    main()
