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

The canonical entry prepares inputs, processes all sessions, checks window
coverage, exports three datasets, and verifies original returns/native SMPL:

```powershell
uv run --no-sync python tools/run_badminton_pipeline.py --output-root outputs/badminton_run
```

Use a new output folder for a new calibration. `job_status.json` is `complete`
only after export and verification; failed processing retries up to three times
using validated caches. `--prepare-only` runs source/calibration/pairing checks
without inference and records `prepared`, which does not mean fitted.

Default inputs are `D:/PROJECT/LIDAR/datas/badminton/sessions` and
`D:/PROJECT/LIDAR/lidar_project`. Override them with `--sessions-root` and
`--lidar-project-root`. The only calibration inputs are the five files in the
latter root's `config/calibration/badminton-20261012/`: `calibration.json`,
`sources.json`, `sync-policy.json`, `bbox.json`, `roi-config.txt`. LiDAR owns
K/d5/R/t, court geometry and their joint projection evaluation; GVHMR consumes
these fixed inputs and fits bodies. No calibration optimizer or experimental
worktree is a runtime dependency.

`prepare` checks both complete cameras, exact raw-to-viewer SourceM, shared
court/plane/bbox geometry, canonical file hashes and source identities. It
rebuilds all timestamp maps from original scan clocks and native camera
sidecars, checks their bytes against the published map SHAs, then freezes its
own bundle. The accepted-source calibration SHA is original provenance; actual
portable-file SHAs are recorded separately. SourceM is rounded: point transforms
use its true inverse, while body orientation uses the nearest proper rotation.
Historical v1 bundles remain independently loadable. Reusing one through
`prepare` requires equivalent geometry, timing, original source hashes and maps;
its bytes/digest are preserved. A changed source requires a new run.

The internal stage adapter supports bounded processing and export operations:

```powershell
uv run --no-sync python tools/pair_badminton.py run --output-root outputs/badminton_smoke --session 20261012_154726 --duration-seconds 6 --fit-batch-size 128 --preproc-precision float16 --roi-margin-px 20
uv run --no-sync python tools/pair_badminton.py export --output-root outputs/badminton_run
uv run --no-sync python tools/verify_badminton_dataset.py --output-root outputs/badminton_run
```

Its stages are `prepare`, `teacher`, `fit`, `run`, `export`. Options include
`--session`, `--start-seconds`, `--duration-seconds`, `--window-seconds`,
`--batch-size`, `--fit-batch-size`, `--iterations`. Full-run settings remain Cam2,
60-second windows, teacher stride 2, batch 4, fit batch 128, 80 fitting iterations,
float16 preprocessing and 20px ROI margin. Teacher cache identity includes original
video rows, camera model, assets, ROI and precision; fit caches additionally bind
algorithm/config and artifact/source PNG hashes. An algorithm change conservatively
invalidates fits while preserving independently validated teacher features.

Every source scan remains in `sessions/<sid>/pairs.jsonl`, including exclusions.
Labels use integer device scan midpoints and nearest native camera frames with
an 8.5ms tolerance; ties choose the earlier frame, and missing clocks/range do
not reuse endpoints. The ~29.97Hz teacher uses true video PTS and SO(3)
interpolation with recorded brackets/model-frame error (up to 18ms). It never
relabels rendered-video frame numbers as sensor timestamps.

The mapped SMPL-X mesh initializes neutral SMPL; shape is fitted in SMPL's own
basis. Pose/pelvis are refined using actual segmented returns, video keypoints,
pose/motion priors and a soft physical-floor constraint. Shape is shared per
person/window. Up to eight observations establish it; accepted corrections carry
across batches only through gaps <=250ms. Continuity follows observed cloud torso
motion and relative video rotations. Original point coordinates are untouched.

The primary RGB weight is 1. A shape-fixed batch with a maximum residual >8px
also tries weight 3. It is selected as a whole only with >=0.5px mean reprojection
gain, <=1/2mm mean surface/P90 increases, identical shape and every previously
accepted frame still accepted. Decisions are retained in frame QA.

Acceptance requires >=80 returns, finite parameters, mean visible point-to-SMPL
triangle distance <=65mm, P90 <=120mm, floor penetration <=120mm, >=6 primary
keypoints and primary reprojection <=50px. Surface distance uses actual triangles
in locally centred millimetres. Rejected fits remain reviewable and are excluded
from training. Each window stores teacher tracks, fitted `.npz`, `quality.jsonl`,
`ptv3.pth`, source/QA metadata and an offline 3D/dual-camera preview. Aggregate
export prefers full windows over contained smoke windows and rejects partial
overlaps. Point indices refer to the decoded valid fixed24 list, not PNG pixels.

The retained complete run `outputs/badminton_corner_fix_v1_continuity` contains
3 sessions, 40 windows, 23,177 original scans / 23,040 valid nonempty dual-camera
pairs and 59,972 accepted person-frame labels (60,847 fitted; 875 rejected).
Keep historical runs read-only. Physical exposure timing, per-point scan motion,
independently surveyed court scale and occluded-joint accuracy remain unverified;
`production_admitted=false`. Visible-surface residual is not anatomical GT.
Suggested train/val/test splits hold out whole sessions; athlete identity overlap
is unverified. The consuming trainer is maintained separately at
`D:/PROJECT/LIDAR/rollout_lidar_mocap_badminton`.

For training, qualify each existing accepted person-frame using corresponding
native detections/tracks in **both** cameras. Missing either view excludes that
person-frame, even when video timestamps match. Full-body keypoint confidence,
net tape, occlusion and reprojection diagnostics are retained as metadata and do
not add admission conditions. Individual visual review is not required.

The delivered `outputs/badminton_dualcam_paired_full_v2` contains 59,408
qualified person-frames: train 14,664 / validation 10,787 / test 33,957.
Its `datasets/dataset_manifest.json` binds tensor/metadata hashes and whole-session
splits. `verification.json` checks every exported value against the original
source subset; `quality_summary.json` reports retained-fit quality and measured
cache/inference scope. A complete manifest is published only after validation.

```powershell
uv run --no-sync python tools/observe_badminton_dual_view.py --output-root outputs/my_paired_run --cache-report outputs/badminton_corner_fix_v1_continuity/sequence_review/cam1_check.json --cache-report outputs/badminton_corner_fix_v1_continuity/sequence_review/cam2_check.json
uv run --no-sync python tools/export_badminton_dual_view.py finalize --output-root outputs/my_paired_run --full-observations outputs/my_paired_run/observations/manifest.json --wait
uv run --no-sync python tools/export_badminton_dual_view.py verify --output-root outputs/my_paired_run
```

Observation reuses exact native-frame teacher detections after checking frozen
calibration, video identity, frame index/PTS/common clock and cache provenance.
Nearest/interpolated teacher frames are not presence evidence. Missing cache
uses YOLO plus one-to-one fixed-mesh bbox association, without new ViTPose,
GVHMR, body fitting or calibration. Per-camera journals resume only with the
same sources/settings/code; an interrupted final line is recoverable. The
second command may run alongside observation and finalizes after all source
labels are checked. Published manifests are immutable. Tensor/metadata hashes,
source indices, independent detection boxes, association and native time/image
evidence are exported with unchanged source point/SMPL values. Original
whole-session splits are preserved; physical identities across sessions remain
unverified. Cached-evidence-only pilot exports are explicitly marked as pilots.

An upstream source index can be checked separately with
`tools/prepare_badminton_source.py --source-index <index.json> --data-root <badminton-data-root> --source-index-sha256 <digest> --report outputs/source_prepare/report.json`.
This CPU-only adapter checks file hashes, explicit raw-XYZ/M semantics, original
device clocks and native camera frame maps, and decodes six spread-out source
scans. It normalizes metadata in memory and never edits original manifests.
Candidate calibration applicability remains preparation-only and cannot admit
body fitting; the default pipeline session list is unchanged.
An admitted camera source requires a hash-bound applicability certificate that
binds its calibration, exact application M, native files and review evidence.
Camera-source admission is separate from body-label production admission.
Explicit dataset-only `video_sidecars` require
`video_index_contract.kind=dataset_only_device_clock`,
`canonical_receive_admitted=false` and verified device/PTP evidence. They retain
the original host receive-axis rejection. Unknown/lost device synchronization,
synthetic frames and video discontinuities cannot provide valid pair evidence.

Receive a sealed native rally collection with:

```powershell
uv run --no-sync python tools/prepare_badminton_rallies.py --acquisition-manifest <acquisition-manifest.json> --acquisition-sha256 <digest> --data-root <badminton-data-root> --output-root outputs/rally_intake
```

This CPU preparation verifies immutable native analysis, counted rally bounds,
the source/camera contracts and all frame-map identities. `prepared_ranges.jsonl`
preserves the declared review order, parent PNG ordinals, original clock/map
indices and frame IDs. Its SHA is bound by `verification.json`, which also
provides deduplicated frame ranges. Camera-bound summaries describe the first
and last **jointly matched** camera rows inside each full parent range.
The accounting end is the last scan's start time; use the authoritative PNG
range for later processing, since filtering scan midpoints by that end would
drop the terminal frame. Overlapping windows count once per source. Window
minutes include pre-roll/cooldown and do not certify qualified training time.
This command starts no model inference or body fitting.

The explicit seven-source body production entry (original three plus four new
sources, 204 accepted native rallies) is:

```powershell
uv run --no-sync python tools/build_badminton_rallies.py --stage prepare
uv run --no-sync python tools/build_badminton_rallies.py
```

Default input is `outputs/upstream_rally_intake_v1/verification.json`, bound to
acquisition SHA `360c4898437c6cb563f607cbdaa5b8cbd790ebd11db34e506c96a4340df60a70`.
The input contains 39.6134098 deduplicated rally-window minutes and 23,946
unique paired PNG scans. These are input counts, not accepted person labels.
`outputs/badminton_rallies_v1/build_status.json` tracks each source's pending,
running, fitted/reused, failed and completed stages, with fitted/QA/exclusion
counts. New sources have 847 / 4,414 / 3,516 / 3,328 selected paired scans,
grouped into 35 bounded fitting windows. Rest frames do not become labels.
Exact PNG membership, including each terminal scan, controls selection.

The command reuses original baseline fits only after checking camera/source,
clock/frame-map and fit-artifact bindings. Original caches remain in their
frozen run. New camera-admitted indexes use the existing teacher and SMPL fitter
with unchanged cameras. The dataset-only device-time source `192753` keeps its
original host-axis rejection and uses only its reviewed device-clock contract.
Matching per-window teacher/fit caches resume after rerunning the same command;
changed inputs/settings fail and require a new output root. Errors remain in
the per-source status and process traceback; they are never counted as success.

After all body windows finish, the same command observes corresponding native
people in both cameras, imports only exact matching previous evidence, applies
the existing bilateral person-detection rule and verifies all tensor values
before publishing `outputs/badminton_rallies_v1_paired/datasets/dataset_manifest.json`.
The body-QA export under `badminton_rallies_v1/datasets` is an intermediate and
must not be confused with the final bilateral-qualified dataset. A separately
started body job can be followed by
`tools/build_badminton_rallies.py --stage publish --wait`.
Keep one body supervisor and one publisher at most; do not publish concurrently
twice. For the newer user-requested split, whole-session train is `163452` plus
the four new sources, validation is `161544`, and test is `154726`. The old
full_v2 dataset and its earlier split/results remain unchanged. This newly
designated test source has appeared in prior training and is not a fresh blind
test. Cross-session physical identities remain unverified.

The native source viewer starts independently while production runs:

```powershell
uv run --no-sync python tools/serve_badminton_rallies.py --display-crop D:/PROJECT/LIDAR/lidar_project/config/calibration/badminton-20261012/display-crop.json
# Optional browser driver, using installed Chrome (no browser download):
uv pip install --python .venv/Scripts/python.exe playwright orjson
uv run --no-sync python tools/verify_badminton_viewer.py
```

Open `http://127.0.0.1:8767/`. HTML and a source/rally manifest are generated in
`outputs/badminton_rallies_v1/viewer/`; original H264 MP4 streams through local
HTTP byte-range requests without copies/transcodes. Source/rally selection,
play/pause, rates, loop and seek use recorded native common-time and media-PTS
anchors. The default paired mode plays original scan transactions using each
camera's PTS-verified decoded native RGB. Preview decoding checks the same
SessionIndex rows as fitting/offline reviews; JPEG85 previews bind record index, PTS, common
clock, original decoded BGR SHA and preview JPEG SHA. Both previews decode
offscreen before one animation-frame commit updates their visible canvases,
the full 3D scene, both camera projections and the public committed scan.
Loading retains the previous complete image/geometry combination. `pause()`
returns a Promise that drains an in-flight transaction and settles the requested
scan. The alternate native-video mode plays the original MP4 timeline.
"Exact paired frame" pauses at a scan. JPEG previews are lossy display assets;
their frame binding and displayed decoded-preview pixels are verified separately.

The preview decoder retains one PyAV stream per source/camera, with two decode
threads. It seeks only on a new stream, backwards requests or gaps over four
seconds, then decodes forward using actual PTS. Playback requests batches of
four consecutive recorded scans, with up to three requests in flight, sixteen
decoded complete bundles buffered, and twelve bundles prepared before starting.
Both camera JPEGs and full mesh normals are prepared ahead of their display
time. Mesh projection UVs use packed Float32 rather than nested JSON arrays;
optional `orjson` accelerates HTTP serialization (stdlib JSON remains supported).
`window.viewer.bufferState()` records actual complete commits and buffer waits;
the status endpoint includes decoder seek/frame counters.

Default point display follows native `badmintonDetectionBboxSet`: the ground
box's top plane defines a five-metre-high Region with the same XY extent and
rotation, then ground/net Ignore boxes are removed. Float32 Z*Y*X quaternion
rotation and closed box boundaries match the native filter. An obsolete third
Region is ignored. The **全部源点** switch restores all source returns. This
filter affects displayed background and human returns in both 3D and camera
projections; it does not alter original inputs, calibration, fits or QA.

The explicit `--display-crop` sidecar additionally extends only the accepted
net Ignore's positive local-X face by 0.19 m. Its schema, sport, display-only
scope and base bbox SHA must match this frozen run. Arithmetic starts with the
original Float64 center, adds the native Float32 rotation's first column times
half the Float64 extension, extends size X, then compiles the Float32 crop.
The viewer manifest records the exact sidecar path/SHA and resulting boxes.
Omit the option to inspect the original accepted bbox without the display fix.
Neither mode rewrites the source bbox or any published training tensor.

The 3D view reuses `review_assets/review.js`, the existing offline WebGL2 player:
all 6,890 SMPL vertices, triangle topology, computed normals, opacity/wireframe,
real human returns, background points and orbit controls. The online adapter
passes the same packed native-frame geometry contract as offline `data.js`.
Large background scans display a deterministic subset of at most 25,000 actual
returns after the selected display filter; fitted meshes remain complete.
Human returns retain all points allowed by that display filter. Both camera overlays project
the same stored mesh and actual points with frozen Brown K/d/R/t. Full projected
triangles form a translucent silhouette union. No geometry is interpolated;
pending/outside-rally scans keep their actual raw cloud and explicitly show
missing body fits. Counts/status refresh during production.

The browser verifier uses a separate headless installed Chrome, decodes both
videos and exercises first/middle/last rally playback/seeking for all seven
sources (21 cases). It also checks native PTS transactions, all-source raw cloud
and both camera projections, full mesh vertex counts, WebGL errors and actual
3D scan progression while playing. Its JSON receipt and screenshot are written
beside the generated page. `/api/geometry?source=<id>&t=<relative-seconds>` is a
read-only original-scan endpoint. `window.viewer` exposes selectSource,
selectRally, seek, play, pause, pairedFrame and commonTime for diagnostics;
await `window.viewer.ready` (or wait for `window.viewerReady === true`) before
automating initial source switches; await each selection/seek Promise and wait
for `!frameBusy` before inspecting the displayed native-frame transaction.
In paired mode use `window.committedPairedFrame`, `lastPairedFrame` and visible
`nativeFrame1/2` canvases. `video.currentTime` belongs to the alternate MP4 mode
and is not proof of the currently displayed paired RGB. The verifier compares
every visible RGB pixel against that scan's bound decoded JPEG, not just PTS
or mesh counts. The focused regression holds the next geometry response while
checking previous RGB/3D retention, then plays/pauses and checks the same-frame
commit again:

```powershell
uv run --no-sync python tools/verify_badminton_viewer.py --atomic-only --output outputs/badminton_rallies_v1/viewer/atomic_loading_verification.json
uv run --no-sync python tools/benchmark_badminton_playback.py --source 20261012_163452 --source 20261009_202713 --seconds 8
```

`iframe#scene.contentWindow.REVIEW_STATE` reports scan/mesh/vertex/point counts.
The playback benchmark measures actual complete frame commits, startup delay,
commit-gap median/P95/max, buffer waits, decoder seeks and visible RGB identity
on the reported browser/GPU. Wall-clock advancement alone is not a frame count.
The server snapshots its HTML/JS/CSS together at startup; restart after backend
or client edits to deploy a consistent version.
Playback controls and the rally-relative progress bar sit above a viewport-sized
3D/dual-camera workspace. Both native camera frames keep their full aspect ratio;
dataset tables sit in a disclosure rather than displacing the inspection views.
Space toggles playback with focus in either the page or embedded 3D canvas;
Left/Right step one actual scan. Preparing/cancelled/buffering/paused states are
visible, and paired playback does not wait for browser MP4 metadata. Each new
rally fits the full body view; automatic framing widens only as necessary until
the user orbits/zooms, and "适配视野" restores automatic framing.

```powershell
uv run --no-sync python tools/verify_badminton_viewer.py --controls-only --output outputs/badminton_rallies_v1/viewer/controls_verification.json
```

This uses actual button/keyboard/slider events, preparation cancellation and
rally looping, checks all three views and controls fit in four viewport sizes,
and projects every current SMPL vertex into the 3D view to detect clipping.
Final delivery `quality_summary.json` distinguishes original input/rally scans,
fit QA rejections, bilateral exclusions and final person frames by source.
For a published run created by an already-loaded older exporter, add the same
quality report and an independent hash-bound receipt without changing its
training manifest or tensors:

```powershell
uv run --no-sync python tools/report_badminton_rally_delivery.py
```

`delivery_receipt.json` binds the published manifest, verification and quality
report, explicitly records the sealed rally scope, and discloses resumed-call
timing limits. A generic scope string in an older published manifest does not
expand the source manifest's sealed rally selection.

The user-requested full-data union references original published tensors without
copying or re-encoding them:

```powershell
uv run --no-sync python tools/build_badminton_reference_union.py
```

`outputs/badminton_full_plus_new_v1/dataset_manifest.json` contains 79,776 unique
person frames: all 59,408 old full-v2 frames plus 20,368 from the four new sources.
The 32,786 old rally-subset frames are checked against their full-v2 person/scan
keys and all five tensor fields, then omitted as duplicates. Each split contains
ordered `parts` with original paths, metadata and hashes; no merged tensor file
is created. Train has 62,414 frames, validation retains full 161544 (10,787), and
test uses new 190709 (6,575), disjoint from train/validation. The manifest records
the requested 100 training epochs; actual training belongs to the consumer queue.
`overlap_verification.json` and `receipt.json` bind the duplicate checks and final
manifest. This is session isolation, not verified unseen-person generalization.

Segmentation evaluates bone scores only on points inside each person's box;
outside scores were always infinite. Three 615k-point native scans matched the
previous implementation's selected points/indices exactly and ran 6.33–7.05x
faster in that isolated stage (`segmentation_benchmark.json`). This is not an
end-to-end uncached throughput claim. The build CLI caps tensor CPU threads to
eight; launch with `OPENBLAS_NUM_THREADS=1` to avoid BLAS oversubscription.

For offline previews and measured fit quality/efficiency:

```powershell
uv run --no-sync python tools/render_badminton_reviews.py --output-root outputs/badminton_run --clips 10 --seconds 5
uv run --no-sync python tools/report_badminton_quality.py --output-root outputs/badminton_run --output-dir outputs/badminton_run/sequence_review
uv run --no-sync python tools/render_badminton_reviews.py --output-root outputs/badminton_run --html-only
uv run --no-sync python tools/render_badminton_reviews.py --output-root outputs/badminton_run --verify-only
```

Open `sequence_review/index.html` and retain its whole folder. It uses native
scans/meshes and actual paired camera images without interpolating geometry.
The quality report counts full-job completed windows including validated cached
windows, and discloses cache reuse in throughput. Reported surface P90 is the
average per-person/frame P90, not a pooled percentile. Optional independent
`tools/check_badminton_cam1.py --output-root outputs/badminton_run` checks detector
agreement on CPU. Its association/occlusion filtering can use fitted geometry;
its pixels are undistorted and do not establish independent anatomical truth.

To inspect an immutable run, supply `--verification-report` to the preview verifier
or `--report` to the dataset verifier, targeting a new output path. The optional
browser checker requires Node with built-in WebSocket and an explicit separately
installed Chrome for Testing binary; it never reuses a personal browser profile:

```powershell
node tools/check_badminton_reviews.mjs --review-dir outputs/badminton_run/sequence_review --browser C:/tools/chrome-for-testing/chrome.exe --cleanup-profile
uv run --no-sync python -m unittest discover -s tests
```

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

Provide the licensed neutral SMPL model at
`inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl`; runtime does not fetch it
from another local project.

## Using GVHMR results with the sibling single-frame PTv3 project

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
