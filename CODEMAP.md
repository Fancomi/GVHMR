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
Canonical dataset roots and explicit historical source relocation:
[docs/DATA_RELOCATIONS.md](docs/DATA_RELOCATIONS.md), `relocations.py`.
Validated identity/foot repair strategy, rejected approaches, run stages and
retention rules: [docs/BADMINTON_BODY_REPAIR.md](docs/BADMINTON_BODY_REPAIR.md).
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
`tools/run_badminton_pilot.py` / `pilot.py` consume an explicitly authorized,
hash-bound review-only camera bundle and native-confirmed short rally ranges.
The separate output freezes actual new crop indexes, runs both camera teachers,
then reuses native pose, identity, v2 fitting, member/SMPL verification and the
existing viewer. Its sole `review` group cannot enter formal training export.
Pilot scope, measured quality/timing and explicit GPU handoff:
[docs/BADMINTON_REVIEW_PILOT.md](docs/BADMINTON_REVIEW_PILOT.md).
`tools/serve_badminton_pilot_review.py` combines verified pilot services into ten
native five-second previews; `tools/verify_badminton_pilot.py` checks actual Chrome playback.
`tools/run_badminton_full_native.py` consumes a separate full-coverage manifest;
`full_native.py` validates exact teacher/native-pose reuse, and
`tools/run_badminton_full_teachers.py` yields at safe GPU lease boundaries.
Full-run contract, commands and real progress: [docs/BADMINTON_FULL_NATIVE.md](docs/BADMINTON_FULL_NATIVE.md).
Explicit fullraw indexes retain cloud ordinals, original packet-row objects and
documented final unclosed scans separately. `source_membership.py` verifies a
derived crop's exact encoded-cell members when its cloud is read. Completion,
PTP, original timestamp indexes and parent lineage stay hash-bound; legacy
camera-coverage omissions keep their existing checks. See docs/INSTALL.md.
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
`tools/audit_badminton_body_labels.py` scans every delivered union label on CPU
for viewer recoloring, short-gap track fragments and lower-limb support review
candidates. It writes separate reports and never rewrites labels or cameras.
`track_colors.py` assigns stable display colors to local detector tracks;
`tools/verify_badminton_track_colors.py` checks the reported opening-seconds
regression, exact-scan links and native RGB/mesh synchronization in Chrome.
`tools/replay_badminton_tracking.py` compares old/new associations on every
frozen native detection. `body_repair.py` shares the exact published-window
selection with `tools/repair_badminton_body_labels.py`, which rebuilds teacher
contexts and bodies into a separate version, retaining unchanged pixel caches.
`tools/serve_badminton_body_repair.py` reuses the native viewer for explicit
repair candidates and defaults to the completed v2 labels.
`tools/prepare_badminton_repair_cases.py` fixes ten continuous old/new clips,
all ankle-review candidates and detector boundaries. Native observations from
`tools/observe_badminton_repair_poses.py` feed fixed-camera epipolar matching
(`multiview.py`), reviewed identity decisions (`identity_repair.py`), dual-view
point ownership and contradictory-ray filtering (`pose_ray_filter.py`).
`tools/render_badminton_repair_comparison.py` exports portable synchronized
six-panel HTML using the same full-mesh 3D and camera projection renderers.
`tools/verify_badminton_native_repair.py` checks every accepted point member,
native teacher crop and SMPL reconstruction; `tools/report_badminton_native_repair.py`
compares both versions against the same raw camera observations and visible
point sets. These repair outputs remain separate from published training data.
`leg_refinement.py` proposes bounded native-SMPL leg corrections when swapping
observed left/right leg labels has a strong fixed-camera geometric witness.
`leg_selection.py` selects continuous corrections over actual scan times.
`tools/run_badminton_repair_windows.py` isolates native processing per window;
`repair_execution.py` retains launch/completion timing in the run itself,
independent of disposable logs. Verified teacher seeds are optional.
`tools/finish_badminton_body_repair.py` waits for complete baseline receipts,
applies guarded corrections with backups, repeats native/body checks and
regenerates the ten old/new clips before sealing the final receipt.

`tools/publish_badminton_repaired_pairs.py` reassociates immutable native
detections to the repaired meshes (`repaired_pairing.py`), then uses the shared
bilateral audit/export verifier. `tools/seal_badminton_repaired_release.py`
proves every aggregate tensor against native-verified windows, seals the unique
62,414-sample training view and exports gap-aware sequence and identity/point/SMPL
lineage indexes. Consumer manifest and native projection contract:
[docs/BADMINTON_REPAIRED_RELEASE.md](docs/BADMINTON_REPAIRED_RELEASE.md).
`tools/audit_badminton_native_pixels.py` checks every test native pixel digest,
reruns bilateral full-person association for affected scans and permits an
independent, hash-bound pixel supplement only when all admissions stay unchanged.
`tools/assemble_badminton_expansion.py` / `expanded_dataset.py` assembles a
provisional complete-old-plus-new union against the consumer's frozen contract.
It preserves all old role rows as exact tensor and metadata-byte prefixes,
keeps every source's own calibration and includes declared zero-sample sources.
Only `assembly_manifest.json` is written: final native pixel and browser
acceptance are still required before consumer publication.
`tools/audit_badminton_fresh_continuity.py` checks all native-verified candidates;
`tools/render_badminton_continuity_cases.py` retains exact dual-camera witnesses.
Reviewed cross-person cuts feed `tools/publish_badminton_canonical_tracks.py` /
`canonical_tracks.py`: only identity keys change, with bit-exact numeric values
and unchanged bilateral membership. `tools/serve_badminton_canonical_tracks.py`
serves the admitted derived view; `tools/verify_badminton_fresh_playback.py`
checks ten real Chrome clips and reviewed native identity boundaries.
`tools/recheck_badminton_expanded_pixels.py` freezes exact native-PNG witnesses
and reruns unchanged full-person admission for affected scans in both cameras.
`tools/seal_badminton_expansion.py` requires all acceptance bindings, archives
independent copies under the canonical data root, verifies them and publishes
formal producer entries pointing to the archive. Existing releases stay frozen;
GPU release is a separate action after the consumer CPU receipt.

Independent adopted days use the same rally entry with `--calibration-dir` and
`--session-splits`; source-only runs freeze admitted source indexes without reading
old baseline fits. `prepare_badminton_new_source_repair.py` freezes their exact
windows and runs bounded initial teachers without legacy fitted labels.
`tools/run_badminton_initial_teachers.py` executes those frozen windows serially
in isolated processes under an explicit GPU lease. The separate
`tools/advance_badminton_new_source.py` supervisor waits for all teacher receipts,
runs native pose observations and stops at a pending identity inventory for
real review; progress and controlled failures stay in the production run.
`screen_badminton_new_identity.py` proposes scope from every actual native endpoint;
`render_badminton_identity_cases.py` renders explicit original RGB witnesses.
Source-filtered geometric reviews reuse only hash-verified complete evidence.
`advance_badminton_identity_review.py` supervises that review; the explicit
abstention policy may preserve unknown raw tracklets separately. The following
`advance_badminton_body_production.py` controller freezes reviewed identity inputs,
fits all windows and runs full QA, then stops for real browser acceptance.
Existing native-observation, body-repair, isolated-window and finish commands then reuse
validated v2 components; `new_source_repair.py` and `new_source_completion.py`
provide the source-only adapters. Completion uses absolute QA and the same v2
pre/post leg comparison, full native verification and standard bilateral export.
See docs/INSTALL.md for input, identity evidence, resume and playback boundaries.

LiDAR owns camera intrinsics/extrinsics, court registration and their joint
projection evaluation. GVHMR consumes declared calibration results and raw
recordings, fits bodies with fixed cameras, and never refines camera parameters.
Touched code imports the existing repository root from `hmr4d.PROJ_ROOT`.
