# Repository navigation

Start here. Generated data stays in `outputs/`, model/data inputs in `inputs/`,
and temporary experiments in ignored `.local/`. See [TODO.md](TODO.md) for ownership.

| Workflow | Entry | Core / data flow | Output |
| --- | --- | --- | --- |
| Badminton LiDAR + video → neutral SMPL pairs | `tools/run_badminton_pipeline.py` | `hmr4d/lidar_mocap/`: `calibration.py` + `sessions.py` → `teacher.py` → `segmentation.py` + `fitting.py` → `export.py`, orchestrated by `pipeline.py` | `outputs/<run>/sessions/`, `datasets/` |
| GVHMR video inference | `tools/demo/demo.py` (folder adapter: `demo_folder.py`) | `hmr4d/configs/demo.yaml` → `hmr4d/model/gvhmr/` | `outputs/demo/` |
| GVHMR training / evaluation | `tools/train.py` | `hmr4d/configs/` → `hmr4d/datamodule/` → `hmr4d/model/` | Hydra-configured `outputs/` |
| Inference HTTP service | `scripts/run_service.sh` | `tools/serve/app.py` → `tools/serve/infer_core.py` → GVHMR model | Service jobs; see `scripts/README.md` |

Badminton operations and environment: [docs/INSTALL.md](docs/INSTALL.md).
Training uses Hydra and requires an experiment, for example
`uv run --no-sync python tools/train.py exp=gvhmr/mixed/mixed --help`.
The stage adapter `tools/pair_badminton.py` serves prepare/teacher/fit/export
operations. `tools/verify_badminton_dataset.py` verifies original point members
and native SMPL reconstruction. `tools/render_badminton_reviews.py` makes offline
SMPL previews; `tools/report_badminton_quality.py` reports fit quality/efficiency.

LiDAR owns camera intrinsics/extrinsics, court registration and their joint
projection evaluation. GVHMR consumes declared calibration results and raw
recordings, fits bodies with fixed cameras, and never refines camera parameters.
Touched code imports the existing repository root from `hmr4d.PROJ_ROOT`.
