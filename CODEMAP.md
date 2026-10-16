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
`tools/observe_badminton_dual_view.py` reuses native detections and fills missing
camera observations with YOLO only (`rgb_observations.py`, `dual_view_observer.py`).
`tools/export_badminton_dual_view.py` audits/exports/verifies person-frame subsets
through `dual_view.py`: both corresponding camera detections must exist;
whole-body/pose diagnostics do not add training admission gates.
`tools/prepare_badminton_source.py` / `source_index.py` inspect an explicitly
provided upstream source index on CPU. Candidate calibration applicability
does not admit fitting; default session loading remains unchanged.
`tools/prepare_badminton_rallies.py` / `rally_inventory.py` verify sealed
acquisition/native-analysis/rally contracts and prepare exact PNG/frame-map
ranges on CPU. Window minutes remain separate from qualified body-label minutes.
`tools/build_badminton_rallies.py` / `rally_pipeline.py` fit the explicit sealed
seven-source rally intake, validate baseline reuse, observe both native cameras,
and publish verified SMPL/LiDAR training subsets in a separate paired run.
`tools/serve_badminton_rallies.py` / `rally_viewer.py` + `rally_viewer.html` serve
native dual-camera MP4 playback, synchronized rally seeks and exact paired-frame
stored-fit overlays. `preview_decoder.py` retains PTS-checked decode streams;
the client buffers complete native RGB/mesh/cloud bundles and defaults to the
native badminton Region-minus-Ignore display filter. `tools/verify_badminton_viewer.py`
exercises real Chrome; `tools/benchmark_badminton_playback.py` measures actual
complete commits, buffer waits, decoder seeks and visible RGB identity.
`tools/report_badminton_rally_delivery.py` adds a hash-bound quality receipt to
an already verified publication without rewriting its manifest or tensors.
`tools/build_badminton_reference_union.py` verifies the old full data plus four
new sources, proves old rally overlap by person/scan and exact tensor values,
and publishes a session-disjoint reference-only union without copying data.

LiDAR owns camera intrinsics/extrinsics, court registration and their joint
projection evaluation. GVHMR consumes declared calibration results and raw
recordings, fits bodies with fixed cameras, and never refines camera parameters.
Touched code imports the existing repository root from `hmr4d.PROJ_ROOT`.
