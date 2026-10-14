# Install

## Maintained entry points

The working repository is `Fancomi/GVHMR`; `origin` points there and `upstream`
retains `zju3dv/GVHMR`. The integration includes the upstream SimpleVO updates,
the fork's single-image Flask service/exports, and calibrated LiDAR pairing.
Demo, service and fitting reuse `hmr4d/utils/body_model/contracts.py` for mesh
mapping and pelvis placement. Service and teacher reuse the pinhole K validator.
SMPL-X parameter output from the existing service remains SMPL-X; mapping its
mesh to 6890 vertices alone does not turn its parameters into SMPL labels.

`pyproject.toml` and `uv.lock` are the dependency source for both installers.
The `serve` extra contains Flask; `compat` contains the legacy runtime imports.
Windows uses the PowerShell bootstrap below. Linux uses:

```bash
bash scripts/install.sh             # project-local .venv; no forced proxy
bash scripts/run_service.sh 0 8666 64
# Existing/custom environment: export GVHMR_ENV_DIR=/path/to/gvhmr
# The previously supported baidu/aliyun proxy choices remain explicit options.
```

The Linux script's shell syntax has been checked here; the new Linux locked
installation has not been executed on this Windows machine.

## Calibrated badminton training pairs

Run from the repository root in the configured local uv environment:

The complete default Windows job can be run with one command. It prepares a
missing snapshot, runs all sessions, checks complete window coverage, exports
the three datasets, and verifies sampled original returns/rebuilt SMPL meshes:

```powershell
uv run --no-sync python tools/run_badminton_pipeline.py --output-root outputs/badminton_corner_fix_v1
```

`outputs/badminton_corner_fix_v1/job_status.json` is `complete` only after export and
verification succeed. It records errors and retries failed processing up to
three times with validated caches. It checks the active adoption before resuming
and before publishing completion; a changed adoption requires a new output root.
The stage commands remain available below.

```powershell
$pairOutput = 'outputs/badminton_corner_fix_v1'
uv run --no-sync python tools/pair_badminton.py prepare --output-root $pairOutput --preproc-precision float16 --roi-margin-px 20
# Real short clip, before a full run:
uv run --no-sync python tools/pair_badminton.py run --output-root $pairOutput --session 20261012_154726 --duration-seconds 6 --fit-batch-size 8 --preproc-precision float16 --roi-margin-px 20
# All three recordings; native ~59.94fps -> every second video frame:
uv run --no-sync python tools/pair_badminton.py run --output-root $pairOutput --preproc-precision float16 --roi-margin-px 20
uv run --no-sync python tools/pair_badminton.py export --output-root $pairOutput
uv run --no-sync python tools/verify_badminton_dataset.py --output-root $pairOutput
```

Input paths default to `D:/PROJECT/LIDAR/datas/badminton/sessions` and
`D:/PROJECT/LIDAR/lidar_project`. Override them with `--sessions-root` and
`--lidar-project-root`. Output defaults to `outputs/badminton`; source recordings
and the LiDAR project are read only. `--session`, `--start-seconds`,
`--duration-seconds`, `--window-seconds`, `--batch-size`, `--fit-batch-size` and
`--iterations` control processing. The `teacher` and `fit` stages support
separate processing. Cache identity binds original video frame rows, calibrated
K/d, model assets, ROI, precision and source file identity. Completed fitting
also checks algorithm/config identity and artifact/source PNG hashes.
Changing video calibration or time maps requires a new output root; fitting
algorithm changes rebuild the fitted labels while preserving valid teacher
features. Output caches are resumable and never substitute for missing sources.

The pipeline freezes the explicitly adopted
`.local/evidence/BADMINTON-CALIB-1012/multiframe/active-selection.json` and its
hashed dependencies. A newer mtime or `.creating` candidate is not adoption.
`prepare` checks source manifest binding, court reference plane, Q and complete
camera frame maps. Subsequent runs use that frozen snapshot. To process a later
adoption, prepare a separate output root instead of mixing calibration versions.
The current explicitly adopted version is `corner-fix-v1`. Its ground footprint
uses the full `Rz @ Ry @ Rx` bbox rotation and the frozen shared reference plane.
The earlier incomplete `outputs/badminton` run is superseded and preserved for
comparison; it must not be combined with the new calibration's training labels.

LiDAR scans drive labels on their native ~10Hz timestamps. Every source scan
is retained in `sessions/<sid>/pairs.jsonl`, including excluded empty/unmatched
frames. Pairing uses the device scan midpoint and each camera's nearest native
common-clock frame with an 8.5ms limit; frame numbers cannot be copied across
cameras. The teacher retains ~29.97Hz true video PTS, interpolates rotations on
SO(3), and records its separate model-frame error (up to 18ms), interpolation
brackets and nearest real keypoint frame. There is no artificial 10Hz-to-30Hz
frame relabeling or old 50ms shift.

Cam2 is the primary temporal teacher. Its frames are undistorted using adopted
Brown coefficients and original K. All people inside the projected court ROI
are tracked; interrupted or ambiguous identities are separated. Cam1 and Cam2
both use their full adopted raw-LiDAR-to-camera matrices for original-image
overlay review. Exact Q transforms actual points once into viewer coordinates;
root orientation uses the nearest proper rotation to rounded Q.

For each athlete, the mapped SMPL-X mesh first initializes a neutral SMPL fit.
Body, root and pelvis are then optimized against actual segmented LiDAR returns,
video keypoints, pose/temporal priors and soft penetration of the frozen physical
reference plane. A jump is not pulled onto the floor. Training `shape` is fitted
in SMPL's basis and shared within an established person/window; it is not copied
from SMPL-X. Track names include the window because physical identity across
windows is not yet established. Camera-to-LiDAR calibration determines world
placement; GVHMR's gravity/global trajectory does not supply the LiDAR frame.

The quality gate requires at least 80 human returns, finite parameters, mean
point-to-triangle distance <=65mm, P90 <=120mm, maximum floor penetration <=120mm,
at least 6 valid primary keypoints, and primary reprojection error <=50px.
Rejected fits remain reviewable but do not enter training. Missing returns do
not attract invisible body surfaces: the LiDAR loss is one directional.
The final surface metric is measured against actual SMPL triangles, using
locally centred millimetres to avoid PyTorch3D's small-triangle epsilon bias.

Each window contains teacher tracks, fitted `.npz` geometry, `quality.jsonl`,
`ptv3.pth`, source/timestamp/QA metadata and an offline `review/index.html` with
interactive 3D geometry and actual dual-camera overlays. `export` assembles
`datasets/<sid>.pth` and `dataset_manifest.json`, preferring full windows over
contained short smoke runs. Partially overlapping windows are rejected.
Suggested train/val/test partitions hold out whole sessions; athlete identity
overlap between sessions is unverified. Metadata point indices refer to the
decoded valid fixed24 return list, not raw PNG pixel offsets.

The three recordings contain 23,177 source scans and 23,040 nonempty valid
dual-camera pairs. Current `corner-fix-v1` real 6-second checks produced 479 accepted
person samples from 480 fits across all three recordings. Mean visible-surface
distances for the three clips were 19.17, 18.14 and 19.81 mm. Source verification
checked 16 exported samples per session: point coordinates matched the original
returns exactly, and rebuilding SMPL vertices differed by at most `9.54e-7` m.
These checks cover short clips and representative samples, not full-run completion.
These are pseudo-labels: physical exposure
accuracy, per-point scan motion correction, independent surveyed court scale
and accuracy of occluded joints are not established. Fitting distance measures
visible surface overlap and does not prove anatomical ground truth.

The original sibling PTv3 directory is currently absent on this machine.
The previously verified field/pelvis contract below is preserved. The current
verification script checks sampled exported points against original PNGs and
rebuilds stored meshes from their SMPL parameters; it cannot rerun an absent
trainer's loader. Unit checks: `uv run --no-sync python -m unittest discover -s tests`.

## Local Windows environment with uv

From the repository root, run:

```powershell
powershell -ExecutionPolicy Bypass -File tools/setup_uv.ps1
uv run --no-sync python tools/check_env.py
uv run --no-sync python tools/check_env.py --require-assets
uv run --no-sync python tools/demo/demo.py --video docs/example_video/tennis.mp4 -s --preproc_batch_size 4
```

The project uses `.venv`, Python 3.10, PyTorch 2.3.0 / torchvision 0.18.0
with CUDA 12.1, and a committed `uv.lock`. NumPy remains at 1.23.5 for
the legacy SMPL/chumpy pickle loader. The bootstrap script installs torch
before resolving PyTorch3D, whose source metadata imports torch.
The large Windows torch wheel is fetched from the official PyTorch index
using parallel ranges and verified against that index's SHA-256 before
installation with `uv pip`.
Windows builds PyTorch3D 0.7.6 from its pinned upstream source in
`.build-tools/pytorch3d` (a shorter path avoids MSVC object-file path limits); Linux uses
the original upstream wheel. Install Visual Studio's C++ workload and a
CUDA 12.x toolkit before running the Windows script. The build targets
the installed GPU and defaults to two compiler jobs.
The Windows build replaces Pulsar's two `uint`/`ushort` preprocessor
macros with equivalent C++ type aliases, avoiding token expansion inside
CUDA 12.8's tuple headers. This compatibility patch is applied only to
the local checkout by the setup script.

On this machine CUDA Toolkit 12.8 and MSVC 19.51 are installed. The script
fetches a project-local MSVC 14.38 toolset from Microsoft's VS 2022 feed,
verifies the package SHA-256 values, and uses it with the existing Windows
SDK. The installed Visual Studio is not modified.
The local compiler layout also includes the VS `vcvarsall.bat` entrypoint
required by NVCC; builds use the developer environment already loaded
by the setup script.
`check_env.py` checks actual CUDA KNN forward/backward and mesh
rasterization, as well as the demo imports and Hydra configuration.
It reports missing assets separately; the `--require-assets` option exits
with status 2 when assets are missing. Full video inference still requires
all six assets listed below. DPVO is optional and is not installed.
The demo's `--preproc_batch_size 4` option reduces ViTPose/HMR2 activation
memory on this 12 GB GPU. HMR2 image crops are transferred to the GPU per
batch rather than retaining every frame's crop in GPU memory.
Video merging uses the bundled `imageio-ffmpeg` executable when there is
no system `ffmpeg` on PATH.

If Google Drive's shared-download quota is exhausted, the
[demo linked by the project README](https://huggingface.co/spaces/LittleFrog/GVHMR/blob/main/app.py)
uses the following public Hugging Face mirrors. These commands pin the
repository revisions used for this local setup:

```powershell
$env:HF_HUB_DISABLE_XET = '1'
uv tool run --python 3.10 --from huggingface_hub hf download camenduru/GVHMR --revision 21b32d5389e2e59c0737d4c4095bbc0b8c23f66b gvhmr/gvhmr_siga24_release.ckpt 'hmr2/epoch=10-step=25000.ckpt' vitpose/vitpose-h-multi-coco.pth yolo/yolov8x.pt --local-dir inputs/checkpoints
uv tool run --python 3.10 --from huggingface_hub hf download camenduru/SMPLer-X --revision 9e5548b70b48efe6992faa3b2a418df108542bf7 SMPLX_NEUTRAL.npz --local-dir inputs/checkpoints/body_models/smplx
```

This setup reuses the sibling project's existing neutral SMPL file as
`inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl`.

## Using GVHMR results with the sibling single-frame PTv3 project

Code reviewed in `../rollout_lidar_mocap_badminton`:
`src/data/single_frame.py`, `src/models/frame_model_backbone.py`,
`src/trainers/frame_trainer.py`, and
`third_party/vis/vis_tools/smpl/SMPL.py`.

The interface can be connected through model conversion. GVHMR's
`hmr4d_results.pt` is **SMPL-X parameters**, even though its keys are named
`smpl_params_incam` and `smpl_params_global`. Its 63-dimensional body pose
contains 21 local body rotations plus a separate root rotation, and its
betas have 10 dimensions. The downstream trainer uses **neutral SMPL**,
24 joints, and 10 betas. Its network predicts 24 rotation-6D vectors, which
the trainer derives from the axis-angle labels; labels should stay in
axis-angle representation.

| Downstream joblib field | Shape | Meaning |
| --- | --- | --- |
| `root_pose` | `(N, 1, 3)` | Target-frame root axis angle, radians |
| `body_pose` | `(N, 23, 3)` | SMPL local body rotations, radians |
| `shape` | `(N, 10)` | Fitted neutral SMPL betas |
| `transl` | `(N, 3)` | **Global pelvis position**, metres |
| `pc` | `(N, 256, 3)` | Points in the same target frame, metres |
| `motion_id` | `(N,)` | Clip identifier used to split sequences |

The first 22 kinematic joints correspond (pelvis through both wrists).
SMPL adds the two terminal hand joints. Zero padding these two rotations
is an initialization only: **SMPL-X and SMPL shape spaces are different,
so copying betas and padding pose does not produce aligned geometry**.
Use the bundled `hmr4d/utils/body_model/smplx2smpl_sparse.pt` to map
SMPL-X vertices to SMPL topology (10475 -> 6890 vertices), then fit neutral
SMPL pose, shape, and translation to that mesh. The matrix transforms
vertices, not model parameters. Prefer one fitted shape per person/clip,
and recompute all 24 joint targets using the downstream `PySMPL`.
See the [SMPL-X model-transfer documentation](https://github.com/vchoutas/smplx/blob/main/transfer_model/README.md).

There is a second compatibility issue: standard SMPL/SMPL-X `transl` is
an additive mesh offset; the downstream `PySMPL` subtracts the shaped
pelvis before adding its `transl`. After fitting with a standard SMPL
implementation, export `fitted_output.joints[:, 0]` as downstream
`transl`, rather than copying the fitted offset. This preserves the mesh
position under the downstream pelvis-centred forward convention.

The downstream target frame is right-handed: X forward/depth, Y left,
Z up. For GVHMR **camera-frame** geometry (x right, y down, z forward),
the camera-axis remapping is:

```python
R = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
R_root_target = R @ R_root_camera
pelvis_target = R @ pelvis_camera  # add calibrated translation if needed
points_target = points_camera @ R.T
```

Local body rotations stay unchanged under a global frame transform.
For real LiDAR supervision, use synchronized frame IDs, calibrated
camera-to-LiDAR extrinsics, metric alignment, and point-cloud-based
position/shape refinement. GVHMR global output has Y up but an arbitrary
initial heading/origin; it is not already in the LiDAR coordinate frame.
Its monocular metric estimates also need validation against the sensor.
The default demo selects one video track; match that person to the
segmented LiDAR subject before constructing paired labels.

For synthetic supervision, first fit the SMPL labels, then sample visible
points **from that same fitted SMPL mesh** using the sensor viewpoint,
and FPS to 256 points. The sibling project's
`src/tools/data_convert/basic.py::sample_visible_pc` already provides
the visibility/raycasting/FPS route (Open3D is required in that project).
Uniform all-surface sampling has a different input distribution from
visible LiDAR points. Split train/test by person or clip before sampling.
GVHMR labels remain pseudo-labels; transfer/mesh residuals and video
reprojection should be inspected before large-scale data generation.

With both neutral body models available, the interface can be checked
numerically against the actual sibling dataset loader, PySMPL forward,
and shared single-frame decoder:

```powershell
uv run --no-sync python tools/check_ptv3_contract.py
```

This smoke check fits three generated SMPL-X poses and saves
`outputs/setup/ptv3_contract_report.json`. Its point subsets only exercise
the dataset format and are not simulated LiDAR or a training dataset.

Local validation on the RTX 4070 SUPER passed: the three generated poses
had a fitted mean vertex residual of 6.19 mm, and the standard-SMPL to
downstream-PySMPL forward and coordinate transform agreed within
`4.77e-7` metres. This checks conversion conventions, not real-video
accuracy. The downstream loader, 24-joint shared decoder, and pose-6D
round trip also passed. A 60-frame clip of the provided tennis video ran
through tracking, ViTPose, HMR2 features, GVHMR inference, both renderers,
and video merging. Reports are in `outputs/setup/environment_report.json`,
`outputs/setup/ptv3_contract_report.json`, and
`outputs/setup/video_smoke_report.json`.

The demo writes videos at 30 fps without resampling source frames.
Use original timestamps for synchronization, or explicitly resample
both labels and points together. Do not infer sensor frame IDs from the
rendered video's playback timestamps.

## Environment

```bash
git clone https://github.com/zju3dv/GVHMR
cd GVHMR

conda create -y -n gvhmr python=3.10
conda activate gvhmr
pip install -r requirements.txt
pip install -e .
# to install gvhmr in other repo as editable, try adding "python.analysis.extraPaths": ["path/to/your/package"] to settings.json
```

### Optional: DPVO (not recommended if you want fast inference speed)
```bash
cd third-party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip
pip install torch-scatter -f "https://data.pyg.org/whl/torch-2.3.0+cu121.html"
pip install numba pypose
export CUDA_HOME=/usr/local/cuda-12.1/
export PATH=$PATH:/usr/local/cuda-12.1/bin/
pip install -e .
```

## Inputs & Outputs

```bash
mkdir inputs
mkdir outputs
```

**Weights**

```bash
mkdir -p inputs/checkpoints

# 1. You need to sign up for downloading [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/). And the checkpoints should be placed in the following structure:

inputs/checkpoints/
├── body_models/smplx/
│   └── SMPLX_{GENDER}.npz # SMPLX (We predict SMPLX params + evaluation)
└── body_models/smpl/
    └── SMPL_{GENDER}.pkl  # SMPL (rendering and evaluation)

# 2. Download other pretrained models from Google-Drive (By downloading, you agree to the corresponding licences): https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link

inputs/checkpoints/
├── dpvo/
│   └── dpvo.pth
├── gvhmr/
│   └── gvhmr_siga24_release.ckpt
├── hmr2/
│   └── epoch=10-step=25000.ckpt
├── vitpose/
│   └── vitpose-h-multi-coco.pth
└── yolo/
    └── yolov8x.pt
```

**Data**

We provide preprocessed data for training and evaluation.
Note that we do not intend to distribute the original datasets, and you need to download them (annotation, videos, etc.) from the original websites.
*We're unable to provide the original data due to the license restrictions.*
By downloading the preprocessed data, you agree to the original dataset's terms of use and use the data for research purposes only.

You can download them from [Google-Drive](https://drive.google.com/drive/folders/10sEef1V_tULzddFxzCmDUpsIqfv7eP-P?usp=drive_link). Please place them in the "inputs" folder and execute the following commands:

```bash
cd inputs
# Train
tar -xzvf AMASS_hmr4d_support.tar.gz
tar -xzvf BEDLAM_hmr4d_support.tar.gz
tar -xzvf H36M_hmr4d_support.tar.gz
# Test
tar -xzvf 3DPW_hmr4d_support.tar.gz
tar -xzvf EMDB_hmr4d_support.tar.gz
tar -xzvf RICH_hmr4d_support.tar.gz

# The folder structure should be like this:
inputs/
├── AMASS/hmr4d_support/
├── BEDLAM/hmr4d_support/
├── H36M/hmr4d_support/
├── 3DPW/hmr4d_support/
├── EMDB/hmr4d_support/
└── RICH/hmr4d_support/
```
