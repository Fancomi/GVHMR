# Repository navigation

Start with the matching workflow below, then open its operating guide or entry.
Do not scan unrelated pipelines. [TODO.md](TODO.md) contains unfinished work only.

| Workflow | Entry | Core and data flow | Outputs / guide |
| --- | --- | --- | --- |
| Calibrated badminton LiDAR + video to neutral SMPL | `tools/run_badminton_full_native.py` for a frozen native rally manifest | `hmr4d/lidar_mocap/`: `full_native.py` + `source_index.py` → `teacher.py` / `sparse_teacher.py` → `identity_repair.py` → `fitting.py` + `body_repair.py` → `dual_view.py` | Independent `outputs/<run>/`; [native production and formal release](docs/BADMINTON_FULL_NATIVE.md) |
| Standard calibrated session pairing | `tools/run_badminton_pipeline.py` | `pipeline.py` → `calibration.py` + `sessions.py` → `teacher.py` → `segmentation.py` + `fitting.py` → `export.py` | `outputs/<run>/sessions/`, `datasets/`; [session route](docs/INSTALL.md#calibrated-session-route) |
| GVHMR video inference | `tools/demo/demo.py` (`demo_folder.py` for folders) | `hmr4d/configs/demo.yaml` → `hmr4d/model/gvhmr/` | `outputs/demo/`; [setup](docs/INSTALL.md) |
| GVHMR training / evaluation | `tools/train.py` | `hmr4d/configs/` → `hmr4d/datamodule/` → `hmr4d/model/` | Hydra-configured `outputs/`; [README](README.md) |
| Inference HTTP service | `scripts/run_service.sh` | `tools/serve/app.py` → `tools/serve/infer_core.py` → GVHMR model | Service jobs; [service guide](scripts/README.md) |

Badminton ownership is explicit: LiDAR supplies camera/court calibration; GVHMR
fits bodies using fixed cameras; the separate rollout repository owns training
and prediction export. Review output, body QA, bilateral qualification, canonical
identity, native pixels, browser acceptance and consumer admission are different
states.

| Badminton operation | Entry / core | Guide |
| --- | --- | --- |
| Inspect source or sealed rally intake on CPU | `tools/prepare_badminton_source.py`, `tools/prepare_badminton_rallies.py` / `source_index.py`, `rally_inventory.py` | [Source and time contracts](docs/INSTALL.md#source-and-time-contracts) |
| Review-only short native pilot | `tools/run_badminton_pilot.py` / `pilot.py` | [Pilot boundary](docs/BADMINTON_REVIEW_PILOT.md) |
| Review identity and refine bodies with the accepted v2 policy | `tools/continue_badminton_native_identity.py`, `tools/repair_badminton_body_labels.py`, `tools/finish_badminton_body_repair.py` | [Repair strategy](docs/BADMINTON_BODY_REPAIR.md), [new-source stages](docs/BADMINTON_FULL_NATIVE.md#new-source-body-production) |
| Import unchanged review bodies into an admitted formal run | `tools/import_badminton_formal_bodies.py` / `formal_import_audit.py`, `formal_numeric_import.py` | [Formal import and native proof](docs/BADMINTON_FULL_NATIVE.md#formal-import-and-native-proof) |
| Prepare a formal increment and execute its bounded leg stage | `tools/prepare_badminton_formal_increment.py`, `tools/run_badminton_incremental_guarded_leg.py` | [Incremental policy](docs/BADMINTON_FULL_NATIVE.md#explicit-incremental-formal-release-policy-v2), [guarded execution](docs/BADMINTON_FULL_NATIVE.md#generic-incremental-guarded-leg) |
| Publish canonical tracks and seal a cumulative archive | `tools/publish_badminton_canonical_tracks.py` or `publish_badminton_canonical_no_change.py` → `assemble_badminton_expansion.py` → `seal_badminton_expansion.py` | [Release gates](docs/BADMINTON_FULL_NATIVE.md#canonical-publication-and-release), [consumer format](docs/BADMINTON_REPAIRED_RELEASE.md) |
| Native dual-camera playback and acceptance | `tools/serve_badminton_canonical_tracks.py`, `tools/verify_badminton_fresh_playback.py` / `rally_viewer.py`, `preview_decoder.py` | [Playback and native pixels](docs/BADMINTON_FULL_NATIVE.md#playback-and-native-pixels) |

Current accepted cumulative archive:
`D:/PROJECT/LIDAR/datas/lidar2smpl_badminton/releases/20261024_expanded_228059/manifest.json`.
It contains train/val/test **170,815 / 23,749 / 33,495** person-frames.
Use one complete version; releases overlap and their counts must not be added.

Shared model conversion and pelvis placement live in
`hmr4d/utils/body_model/contracts.py`; the [consumer contract](docs/BADMINTON_REPAIRED_RELEASE.md)
explains native camera projection and neutral SMPL. Canonical data roots and
historical path aliases are in [DATA_RELOCATIONS.md](docs/DATA_RELOCATIONS.md).
Inputs belong in `inputs/`, generated runs in `outputs/`, disposable experiments
in ignored `.local/`; formal execution must not depend on `.local/`.
