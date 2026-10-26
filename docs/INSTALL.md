# Install

Use [CODEMAP.md](../CODEMAP.md) to choose a workflow. This fork is
`Fancomi/GVHMR`; upstream is `zju3dv/GVHMR`. `pyproject.toml` and `uv.lock`
are the dependency source for both installers.

## Local Windows environment with uv

Run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File tools/setup_uv.ps1
uv sync --extra compat --extra dev --extra review
uv run --no-sync python tools/check_env.py
uv run --no-sync python tools/check_env.py --require-assets
uv run --no-sync python tools/demo/demo.py --video docs/example_video/tennis.mp4 -s --preproc_batch_size 4
```

The local environment is `.venv`: Python 3.10, PyTorch 2.3.0 / torchvision
0.18.0 with CUDA 12.1 and NumPy 1.23.5 for legacy SMPL/chumpy pickles. Setup
installs torch before resolving PyTorch3D, whose source metadata imports torch.
The Windows torch download is checked against the official index's SHA256.

The `dev` extra supplies pytest; `review` supplies Playwright for browser checks.
Add `--extra serve` only for the Flask service. Review checks also need Node and
a separate Chrome for Testing binary; see [playback prerequisites](BADMINTON_FULL_NATIVE.md#playback-and-native-pixels).

Windows builds pinned PyTorch3D 0.7.6 in `.build-tools/pytorch3d`; install Visual
Studio's C++ workload and a CUDA 12.x toolkit first. Setup uses a project-local
MSVC 14.38 toolset, validates its package hashes and loads the existing Windows
SDK without modifying Visual Studio. It defaults to two compiler jobs and
applies the local Pulsar type-alias compatibility patch for newer CUDA headers.
Linux uses the upstream wheel.

`check_env.py` checks actual CUDA KNN forward/backward, mesh rasterization,
imports and Hydra configuration. Missing assets are reported separately;
`--require-assets` returns status 2 when any required asset is absent. The demo's
batch size 4 reduces preprocessing memory; crops move to CUDA per batch.
Video merging falls back to the bundled `imageio-ffmpeg` executable.

Linux uses the same dependency lock:

```bash
bash scripts/install.sh
bash scripts/run_service.sh 0 8666 64
# Existing/custom environment:
# export GVHMR_ENV_DIR=/path/to/gvhmr
```

The `serve` extra contains Flask and `compat` contains legacy runtime imports.
Linux shell syntax has been checked; a complete Linux installation has not
been validated on this Windows host. Service details are in
[scripts/README.md](../scripts/README.md). DPVO is optional; the default moving
camera demo uses SimpleVO. A static camera uses `-s`.

## Assets and original training data

Obtain licensed [SMPL](https://smpl.is.tue.mpg.de/) and
[SMPL-X](https://smpl-x.is.tue.mpg.de/) models yourself. Runtime must not fetch
them from another local workspace.

```text
inputs/checkpoints/
  body_models/smpl/SMPL_NEUTRAL.pkl
  body_models/smplx/SMPLX_NEUTRAL.npz
  gvhmr/gvhmr_siga24_release.ckpt
  hmr2/epoch=10-step=25000.ckpt
  vitpose/vitpose-h-multi-coco.pth
  yolo/yolov8x.pt
```

Other gendered body models are needed for their corresponding evaluation tasks.
The original pretrained assets are available from
[the project download](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link).
If its quota is exhausted, these public mirrors pin the revisions used by the
local setup:

```powershell
$env:HF_HUB_DISABLE_XET = '1'
uv tool run --python 3.10 --from huggingface_hub hf download camenduru/GVHMR --revision 21b32d5389e2e59c0737d4c4095bbc0b8c23f66b gvhmr/gvhmr_siga24_release.ckpt 'hmr2/epoch=10-step=25000.ckpt' vitpose/vitpose-h-multi-coco.pth yolo/yolov8x.pt --local-dir inputs/checkpoints
uv tool run --python 3.10 --from huggingface_hub hf download camenduru/SMPLer-X --revision 9e5548b70b48efe6992faa3b2a418df108542bf7 SMPLX_NEUTRAL.npz --local-dir inputs/checkpoints/body_models/smplx
```

For GVHMR training/evaluation, obtain original datasets under their licenses and
[preprocessed support files](https://drive.google.com/drive/folders/10sEef1V_tULzddFxzCmDUpsIqfv7eP-P?usp=drive_link).
Extract each `<DATASET>_hmr4d_support.tar.gz` under `inputs/`, producing
`AMASS/hmr4d_support`, `BEDLAM/hmr4d_support`, `H36M/hmr4d_support`,
`3DPW/hmr4d_support`, `EMDB/hmr4d_support` and `RICH/hmr4d_support`.
The preprocessed files do not replace the original videos/annotations.

## Calibrated session route

This is the standard session pipeline. For admitted native rally production and
formal cumulative releases use [BADMINTON_FULL_NATIVE.md](BADMINTON_FULL_NATIVE.md).

```powershell
uv run --no-sync python tools/run_badminton_pipeline.py --output-root outputs/badminton_run
uv run --no-sync python tools/pair_badminton.py run --output-root outputs/badminton_smoke --session 20261012_154726 --duration-seconds 6 --fit-batch-size 128 --preproc-precision float16 --roi-margin-px 20
uv run --no-sync python tools/verify_badminton_dataset.py --output-root outputs/badminton_run
```

Default sessions are `D:/PROJECT/LIDAR/datas/badminton/sessions`; the calibration
project is `D:/PROJECT/LIDAR/lidar_project`. Override with `--sessions-root` and
`--lidar-project-root`. The standard route reads the five files under that
project's `config/calibration/badminton-20261012/`: `calibration.json`,
`sources.json`, `sync-policy.json`, `bbox.json`, `roi-config.txt`.
LiDAR owns K/d5/R/t, court registration and joint projection evaluation. GVHMR
fits bodies using these fixed inputs. Different calibration requires a new run.

`--prepare-only` checks source/calibration/pairing and records `prepared`; it
does not fit bodies. The full entry records `job_status.json=complete` only
after export and verification, with up to three validated retries. The internal
`pair_badminton.py` stages are `prepare`, `teacher`, `fit`, `run`, `export`.
Standard full-run defaults are cam2, 60-second windows, teacher stride 2, batch 4,
fit batch 128, 80 iterations, float16 preprocessing and a 20 px ROI margin.

The pipeline freezes original source identities, all timestamp maps, exact
raw-to-viewer SourceM and shared court/plane/bbox geometry. Point transforms use
the true inverse of rounded SourceM; body orientation uses its nearest proper
rotation. Historical v1 bundles remain loadable only with unchanged geometry,
timing, source hashes and maps. Teacher caches bind actual native rows, camera,
model assets, ROI and precision; fit caches additionally bind algorithm/settings
and source PNG/artifact hashes. Existing filenames alone do not validate reuse.

A mapped SMPL-X mesh initializes neutral SMPL. Shape is fitted in SMPL's own basis
and shared per person/window; up to eight observations establish it. Corrections
propagate only across gaps ≤250 ms, using observed torso motion and video rotations.
Point coordinates remain original. The primary RGB weight is 1; a shape-fixed
batch over 8 px maximum residual may try weight 3. Selection requires ≥0.5 px
mean gain, surface mean/P90 degradation ≤1/2 mm, identical shape and retention
of all previously accepted frames.

Body QA requires ≥80 returns, finite parameters, visible point-to-triangle mean
≤65 mm and P90 ≤120 mm, floor penetration ≤120 mm, ≥6 primary keypoints and
primary reprojection ≤50 px. Distances use actual triangles in locally centred
millimetres. Rejected fits remain reviewable and are excluded from export.
Each window retains teachers, fitted NPZ, `quality.jsonl`, `ptv3.pth` and metadata.
Aggregate selection prefers full windows over contained smoke windows and rejects
partial overlaps. Point indices address the decoded valid fixed24 list.

Training qualification is a separate `bilateral_detection` stage: each person
needs a unique actual corresponding detection in both native cameras. Matching
timestamps, interpolated teacher states and body QA alone are insufficient.
Keypoint confidence, occlusion and reprojection diagnostics do not add admission
gates. Use `observe_badminton_dual_view.py` and `export_badminton_dual_view.py`;
formal stage details and explicit CPU decoding are in the
[native production guide](BADMINTON_FULL_NATIVE.md).

## Source and time contracts

```powershell
uv run --no-sync python tools/prepare_badminton_source.py --source-index <index.json> --data-root <badminton-data-root> --source-index-sha256 <digest> --report outputs/source_prepare/report.json
uv run --no-sync python tools/prepare_badminton_rallies.py --acquisition-manifest <acquisition.json> --acquisition-sha256 <digest> --data-root <badminton-data-root> --output-root outputs/rally_intake
```

These CPU adapters inspect immutable inputs; source preparation samples six
spread-out scans and never certifies all body labels. Camera source admission
requires a hash-bound applicability certificate identifying the exact calibration,
application M, native files and review evidence. Candidate applicability is
preparation-only. Explicit dataset-only video contracts retain their original
host receive-axis rejection and require verified device/PTP evidence.

Labels address integer device scan midpoints and nearest native camera records
within 8.5 ms; ties choose the earlier record. Missing clocks/range do not reuse
endpoints. The dense teacher uses actual video PTS and SO(3) interpolation with
recorded brackets and an 18 ms model-frame tolerance. Never infer sensor frame IDs
from rendered videos: the demo writes at 30 fps without resampling source frames.

`prepared_ranges.jsonl` retains original review order, PNG ordinals, clock/map
indices, frame IDs and counted native rally bounds. Its verification binds the
deduplicated ranges. The accounting end is the terminal scan's start time:
process the authoritative PNG range, including the last scan. Overlap counts once
per source; rally-window minutes include buffers and are not qualified label time.

Fullraw indexes use ordered unique `lidar_common_clock_row_indices`, one per
emitted PNG, preserving exact original packet-row objects and integer first/last/mid
times. Only an evidenced final unclosed row may appear under
`unexported_packet_clock_rows` with `unclosed_final_scan_not_emitted`; it creates
no cloud, member array or sample. Legacy outside-camera omissions keep their
separate original checks. Full export/PTP/raw-index/download/parent evidence
remains hash-bound.

Derived crops declare `parent_scope=fullraw_exported_clouds` and
`maps_to_full_pcap_returns=false`. `source_membership.py` checks the original PNG,
member maps, inverse/grid arrays and retained RGB8 coordinate bytes when a cloud
is read. Empty emitted clouds stay empty. Explicit fullraw aliases normalize
dimensions in memory, require stored raw XYZ and `M_baked_into_stored_xyz=false`,
and reject conflicting aliases or applying M twice.

Frozen clock JSON keeps its original bytes. Relative `raw_index` references
resolve through the verified parent's original `common_clock` directory, after
comparing the frozen copy; no basename search or manifest rewrite is allowed.
Original clouds, clocks and member evidence remain required inputs. Canonical
roots and explicit historical resolvers are in [DATA_RELOCATIONS.md](DATA_RELOCATIONS.md).

## Neutral SMPL and the single-frame consumer

GVHMR `hmr4d_results.pt` stores SMPL-X parameters despite the
`smpl_params_incam/global` names: 21 body rotations (63 values), separate root and
10 betas. The consumer needs neutral SMPL, 24 joints and 10 betas.

| Export field | Shape | Meaning |
| --- | --- | --- |
| `root_pose` | `(N,1,3)` | Target-frame root axis angle, radians |
| `body_pose` | `(N,23,3)` | SMPL local axis angles, radians |
| `shape` | `(N,10)` | Fitted neutral SMPL betas |
| `transl` | `(N,3)` | Global shaped-pelvis position, metres |
| `pc` | `(N,256,3)` | Actual points in the same target frame, metres |
| `motion_id` | `(N,)` | Sequence identity |

Map vertices using `hmr4d/utils/body_model/smplx2smpl_sparse.pt`
(10,475 → 6,890), then fit neutral SMPL pose/shape/position. The matrix maps meshes,
not parameters; copying betas and padding hand rotations does not align the two
shape/rest bases. Recompute 24 joints with the consumer model. Shared conversion
and placement live in `hmr4d/utils/body_model/contracts.py`.

Consumer translation is the shaped pelvis, unlike standard SMPL's additive offset.
Export `fitted_output.joints[:,0]`, and reconstruct with
`forward_smpl_at_pelvis`. The target frame is X forward/depth, Y left, Z up. Camera
geometry (right/down/forward) maps with `R=[[0,0,1],[-1,0,0],[0,-1,0]]`;
left-multiply root/global positions and row-multiply points by R transpose, adding
calibrated translation where needed. Local body rotations do not change.
GVHMR world output has arbitrary heading/origin and is not already LiDAR-aligned.

The optional `tools/check_ptv3_contract.py --ptv3-root <consumer-repository>`
loads that explicitly supplied consumer to test conversion conventions on three
generated poses. It is a GPU interface check, not a real-video accuracy test or
a training-data generator. Production does not import the consumer workspace.
The measured setup check is retained under `outputs/setup/`.

For synthetic supervision, sample visible points from the same fitted SMPL mesh
using a sensor viewpoint, then FPS; all-surface sampling has a different distribution.
For actual supervision, preserve synchronized native records and calibrated metric
alignment. Split whole scenes before sampling. Labels remain pseudo-labels:
surface residual and video reprojection do not establish independent motion-capture
accuracy, especially for occluded joints and foot orientation.

The [consumer release contract](BADMINTON_REPAIRED_RELEASE.md) gives exact split,
sequence, native-address and raw-versus-undistorted projection rules.
