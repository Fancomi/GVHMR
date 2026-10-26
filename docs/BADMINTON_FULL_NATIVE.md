# Full native rally fitting

This guide covers admitted native rally intake, the accepted v2 body policy and
formal cumulative publication. Use [INSTALL.md](INSTALL.md) for environment,
source/time contracts and neutral-SMPL conventions; use
[BADMINTON_BODY_REPAIR.md](BADMINTON_BODY_REPAIR.md) for identity/leg quality rules.
A review run and a formal archive are separate outputs. LiDAR owns camera/court
calibration; all body stages keep cameras fixed.

## Current accepted data

The cumulative archive is
`D:/PROJECT/LIDAR/datas/lidar2smpl_badminton/releases/20261024_expanded_228059/manifest.json`,
SHA256 `68e0065c20a71d301e3318753fe28bcfce388ae72280a5f884b61762d64ca4a4`.
It contains **228,059** person-frames: train **170,815**, validation **23,749**,
test **33,495**. It preserves all 222,180 parent rows and adds 5,879 samples from
`20261022_161632` (5,887 full-body QA candidates, 26 rallies). New-source val/test
counts are zero; parent val/test tensor and metadata bytes are unchanged.

Archive verification and the final consumer CPU audit passed. Permanent evidence:
`outputs/badminton_1022_formal_v1/release_inputs/tail-v2/final-consumer-contract.json`
(SHA256 `d8715b54c3e7340f3423ba9e3811021ad4cbf6c682254f3a5854204f957a2106`) and
`final-consumer-audit.json`
(`e929668f13f2b3d6fd4bd6c38820a9d3f5f2c9c43f8d11e32c9e40d9d6ea5890`).
Recomputed train/val twelve-frame sequences total 37,364 / 1,846.
The release itself does not start or authorize model training.

Releases are complete overlapping versions, not disjoint batches. Select one
version; never concatenate their counts. Old archives and their default catalog
selection remain immutable. [Consumer format](BADMINTON_REPAIRED_RELEASE.md) and
[data relocation](DATA_RELOCATIONS.md) explain their entry points and aliases.

Historical execution receipts bind the implementation bytes they actually ran.
Replaying an old live-context check requires that original code revision and
its frozen evidence; current source cannot substitute for an old fingerprint.
The pre-cleanup revision is `e6926ae`. New runs freeze the maintained version's
new fingerprints. Never rewrite historical receipts or weaken equality checks
to make a newer implementation appear to be the original execution.

## Native intake and stages

The entry is `tools/run_badminton_full_native.py`, sharing `pilot.py`,
`full_native.py`, native teachers, reviewed identity, v2 fitting and verifiers.
It requires a separate `badminton-native-rally-fit-v1` manifest. The short
[pilot](BADMINTON_REVIEW_PILOT.md) keeps its 180-second limit.

A full manifest binds five camera files, source indexes, native rally evidence,
ordered unique paired scan ordinals, explicit relocation, actual scope approval
and a SHA-bound coverage audit. It declares
`execution_scope=all_effective_native_rallies`. Review output stays
`review_only=true`, `training_published=false`.

```powershell
uv run --no-sync python tools/run_badminton_full_native.py --manifest <full-native-rallies.json> --root <new-review-run> --stage prepare --threads 2
uv run --no-sync python tools/run_badminton_full_native.py --root <new-review-run> --stage teacher --gpu-lease <fresh-held-lease.json> --teacher-sampling-input <frozen-sampling-input.json> --threads 2
```

For an independently accepted camera freeze,
`tools/prepare_badminton_admitted_native.py` derives a separate review-purpose
snapshot from the exact accepted snapshot and freeze proof. Supply
`--manifest`, `--root`, `--accepted-parent`, `--accepted-parent-sha256`,
`--accepted-freeze-proof` and `--accepted-freeze-proof-sha256`. Default inspection
is read-only; `--prepare` runs original CPU source/body-plan preparation.
It retains every geometry/session/clock/artifact field, records explicit
accepted-review lineage and adds no artificial split for a single-source review.

Generic sealed rally intake uses `prepare_badminton_rallies.py`; an independent
source-only run uses `build_badminton_rallies.py --stage prepare --intake <verified>
--output-root <new-run> --calibration-dir <five-file-bundle> --session-splits <json>`.
Sources must occur exactly once in exhaustive whole-source roles. Independent
complete datasets need accepted samples in every required role; individual
zero-rally sources may remain explicit zero-count sources. When all sources have
`baseline:false`, preparation never reads old fitted labels. This intake adapter's
original `run/fit` path does not automatically enable v2 body repair: follow the
new-source body stages below.

Full-source lineage permits original fullraw evidence or strict explicit source
indexes, with original recording/download/clock/PTP bindings. `source_indexes`
and `excluded_source_indexes` use `{file,sha256,bytes}`, relative to `data_root`;
receipt/evidence bindings use `{path,sha256}`. No missing-file search is allowed.

For eligible gaps, publish flat clips retaining original `rally_index` and
zero-based `segment_index`, with actual clock bounds and continuous eligible PNG
ordinals. Source/rally/segment is unique; nested `contiguous_windows` is rejected.
Zero-eligible sources use `excluded_source_indexes` plus `no_fit_sources` evidence.
A `badminton-native-no-fit-needed-v1` marker binds the source index, zero coverage,
and completed native analysis/planner files; no fitted input or SMPL tensors are
created. Overall full-fit scope must still be nonempty.

## Reuse, teachers and owned execution

`seed_badminton_full_teachers.py` and `seed_badminton_full_poses.py` take
`--root <new-run>` and repeated `--reuse-run <verified-parent>`. Teacher reuse
requires identical physical cameras, source indexes, windows/selected scans,
settings, native rows, model/video identity and temporal provenance. Copies are
independent files checked by SHA and the original resume validator. Native pose
reuse preserves boxes, keypoints, confidence and raw pixel hashes, explicitly
rebinding only proven scope metadata. Partial seeding never completes missing
native observations.

`paired_sampling.py` freezes the accepted real-paired-RGB input; `run_teacher`
routes its explicit policy through `sparse_teacher.py` for initial and repaired
contexts. Only actual native RGB anchors provide detection evidence. Inferred
temporal states remain role-marked; they are not independent native detections.
Pass the prepared `--teacher-sampling-input` explicitly; omission must not
silently change an intended 10 Hz policy into dense 30 Hz processing.

Teacher supervision executes one camera/window worker at a time and yields only
at genuine completed subprocess boundaries. The default 300-second launch margin
avoids starting work near lease expiration. `--progress-output <new-progress>`
plus `--resume-progress <bound-complete-checkpoint>` preserves historical batches.
`release_badminton_full_gpu.py` requires real owned process exit before release;
a progress flag alone does not prove CUDA is gone. Full fit uses
`--require-completed --device cpu` to validate existing windows, fails changed
caches rather than refitting silently, and yields before its lease boundary.
`--stage aggregate` is completed-only CPU work after GPU release.

The exact-stage owner `tools/run_badminton_owned_stage.py` wraps one existing
full-native `teacher|poses|fit` invocation. It never dispatches the next stage.
Its closed `gvhmr-owned-full-stage-descriptor-v1` binds absolute root/stage,
pilot input, manifest, source input, explicit teacher sampling input (teacher
only), future lease path, progress/resume paths, hard deadline and the code map
from `owned_gpu_stage.implementation_files()`. Pose progress fields are null;
teacher/fit use distinct progress names within the run. It accepts no arbitrary
shell command.

```powershell
& <python.exe> -u <absolute-repository>/tools/run_badminton_owned_stage.py --inspect --descriptor <descriptor.json> --descriptor-sha256 <actual-SHA>
& <python.exe> -u <absolute-repository>/tools/run_badminton_owned_stage.py --execute --descriptor <descriptor.json> --descriptor-sha256 <actual-SHA> --lease <new-local-lease.json> --lease-sha256 <actual-SHA>
```

The separately issued `gvhmr-owned-full-stage-local-lease-v1` is held, manifest/
descriptor/root/stage-bound, allows GPU0 and one CUDA tree, and has a real window
of at most 120 minutes within the descriptor deadline. Its non-fixture desktop
snapshot must be fresh within fifteen minutes and valid for the entire window.
The CPU controller/watchdog use E26/27, BelowNormal and CUDA disabled; the
suspended launcher receives P0/1, BelowNormal, GPU0 and one math thread before
resume. Every descendant belongs to its private Windows Job. The independent
watchdog must be ready first. Controller/watchdog loss, revoked bindings, foreign
GPU ownership, errors or deadline stop only this Job, with a ten-second margin.

Each lease has a persistent single-attempt claim. Inspect real failure/empty-Job
evidence before clearing a crash lock or issuing new authority. A release receipt
proves original CLI exit zero and owned Job exit; a teacher/fit CLI may have merely
yielded, so also check real stage progress. Completed checkpoints and held
authority files are never rewritten as recovery evidence.

## New-source body production

`prepare_badminton_new_source_repair.py --source-root <prepared-run> --root <repair>`
freezes source windows, roles, native-pose case plan and teacher paths without
manufacturing legacy body labels. Initial teacher work is bounded to one window
with `--stage teacher --source <id> --window <window> --threads 1`.
`run_badminton_initial_teachers.py` supervises those windows.
`advance_badminton_new_source.py` waits for genuine teacher completion, observes
poses and stops at a pending identity inventory.

`continue_badminton_native_identity.py --root <full-run> --request <request>
--grant <continuous-grant>` is the canonical full-native continuation. It waits
for real pose completion/release and the coordinator's identity phase, fills only
missing evidence under that grant, then releases CUDA before CPU review/sealing.
`--verify-complete` validates existing real outcomes without inference. Each
child enters its own suspended Windows Job before execution; errors reclaim only
those owned Jobs and retain issued shorter pose-stage limits.

For explicit evidence review, use:

```powershell
uv run --no-sync python tools/prepare_badminton_identity_review.py --root <repair>
uv run --no-sync python tools/prepare_badminton_identity_review.py --root <repair> --render-endpoints --endpoint-offset 0 --endpoint-limit 20
uv run --no-sync python tools/screen_badminton_new_identity.py --root <repair>
```

The pending inventory records real tracklets/endpoints; screening proposes
candidates and assigns no identity truth. Native RGB witnesses preserve original
records, raw hashes and the exact K-preserved undistortion domain. A reviewed
`badminton-new-source-identity-inventory-v1` binds `source_union_sha256`,
camera digest, evidence files and all selected windows/cases. Then run the
inventory preparation with `--boundary-inventory <reviewed.json>`, observe
`<repair>/identity_evidence` through `observe_badminton_repair_poses.py`, run
`review_badminton_identity_views.py`, and seal via
`build_badminton_identity_decisions.py`.

Unknown relationships remain pending unless an explicit reviewed
`badminton-new-source-identity-abstention-v1` policy selects
`keep_unproven_original_tracklets_separate`. It resets raw joins, joins only
geometrically confirmed pairs, preserves different-person constraints and lists
unknowns as `abstained_relationship_cases`. Evidence/hash checks and graph
contradiction rejection remain required. Historical scene-specific overrides are
never applied to new sources. Source-filtered reviews become complete only after
full evidence coverage is checked.

Prepare and run v2 bodies with fixed cameras:

```powershell
uv run --no-sync python tools/repair_badminton_body_labels.py --stage prepare --union <repair>/new_source_input.json --source-root <prepared-run> --output-root <repair>/final_labels --native-poses-root <repair> --joint-ray-filter --distal-pose-weight 0.03 --lower-limb-reprojection-weight 2 --identity-decisions <repair>/identity_evidence/identity_decisions.json --threads 1
uv run --no-sync python tools/run_badminton_repair_windows.py --root <repair> --union <repair>/new_source_input.json --source-root <prepared-run> --threads 1
```

`advance_badminton_identity_review.py` and
`advance_badminton_body_production.py` are bounded supervisors for those same
stages, with explicit root/source-root/lease. They retain failure progress and
stop for browser acceptance. `finish_badminton_body_repair.py --root <repair>`
performs original native-member/SMPL checks, bounded continuous leg refinement,
same-person pre/post QA and bilateral publication. It binds the actual baseline
before changes. Completion `new_source_completion_receipt.json` can state
`labels_published_viewer_prepared`, but keeps browser/archive/consumer/training
admission false until their separate gates pass.

## Formal import and native proof

A separately admitted run may reuse completed review results only after physical
input, timing, membership, fitting policy, source roles and reviewed identity
equivalence. `audit_badminton_formal_inputs.py` writes the real
`formal_input_equivalence.json`; it does not import bodies. The source input
and preparation receipt must bind that exact path and SHA.

```powershell
uv run --no-sync python tools/import_badminton_formal_bodies.py --phase audit --root <formal> --input-equivalence <formal>/formal_input_equivalence.json --input-equivalence-sha256 <actual-SHA> --parent-root <complete-review-parent> --threads 1
uv run --no-sync python tools/import_badminton_formal_bodies.py --phase apply --root <formal> --audit <actual-audit.json> --audit-sha256 <actual-SHA> --threads 1
```

Audit checks all completed body/native/pose/identity receipts, original reports,
assets, points, teacher bytes and fixed roles. Changed execution digests need
explicit original-revision compatibility proof. Apply copies numeric artifacts
into independent files; teacher values/dtype/shape stay exact while provenance is
rebound. Quality measurements, acceptance and person/scan IDs remain unchanged.
Conflicts and hardlinks to originals fail without replacement. The numeric import
does not itself aggregate or reconstruct native SMPL.

Run the original repair entry in completed-only CPU mode with the formal source
input, identity decisions and unchanged v2 settings to aggregate all windows.
`verify_badminton_native_bound.py` records the actual neutral model opened,
code/CPU runtime and before/after inventory around genuine native reconstruction.
Historical cached results gain no witness retroactively.

`derive_badminton_formal_native.py --root <formal> --numeric-import <receipt>
--numeric-import-sha256 <SHA> --run-required-actual --threads 1` accepts explicit
eligible `--parent-witness` / `--parent-witness-sha256` pairs. Each derived window
needs genuine parent model/runtime execution and full numerical, point, pose,
teacher, identity and input-equivalence proof. Missing or invalid parent evidence
requires actual reconstruction, including the all-actual fallback.
`verification_coverage` must cover every actual/derived window. A pose model SHA
does not prove which model a historical native verifier opened.

`freeze_badminton_formal_baseline.py` freezes independent full pre-leg native,
absolute QA, body/pose and report snapshots under `gpu_guarded_leg/baseline/`.
Before refinement,
`finalize_badminton_formal_cpu.py --stage prepare-leg --root <formal>
--baseline <baseline.json> --baseline-sha256 <SHA>` installs that genuine
before-QA snapshot; it cannot reconstruct it after leg mutation.

After actual guarded exit, use `finalize_badminton_formal_cpu.py --stage post-leg`
for its original fingerprint contract, or the explicit
`finalize_badminton_formal_cpu_compat.py --root <formal> --baseline <json>
--baseline-sha256 <SHA> --leg-execution <completion.json>
--leg-execution-sha256 <SHA> --compatibility-implementation <tested-binding.json>
--compatibility-implementation-sha256 <SHA> --threads 1`.
The compatibility adapter preserves the original numerical verifiers and requires
the exact source/test binding for separate producer fingerprints. Metadata-only
inspection performs no native/QA pass and is not completion.
`verify_badminton_formal_dataset.py` supplies the frozen formal resolver while
retaining original source/cloud/SMPL checks and thresholds.

The resulting `formal_postleg_cpu_receipt.json` still leaves bilateral, pixels,
browser, archive and consumer acceptance pending. Pass its exact path/SHA to
`finish_badminton_body_repair.py --formal-postleg-cpu-receipt <json>
--formal-postleg-cpu-receipt-sha256 <SHA>`. Formal finish verifies that completed
CPU chain without another fit/native/leg pass. It waits for independent bilateral
observations and never silently launches an observer on CUDA.

## Generic incremental guarded leg

`tools/run_badminton_incremental_guarded_leg.py` uses `incremental_guard.py` and
`incremental_guard_execution.py` with
`gvhmr-incremental-guard-parent-map-v1`. Actual parents/sources come from the
policy and numeric audit, never date-based selection. The original
`run_badminton_guarded_leg.py` remains compatible with its existing contracts.
The propose/select/apply numerical CLIs and rules remain unchanged.

The incremental descriptor binds formal root, release policy, source input,
import audit, numeric import, baseline, pre-leg native/QA, parents, exact windows
and source scope. Review parents equal numeric-import `parent_evidence`; they
are distinct from the five preceding sealed-release bindings below.
Request/runtime/readiness/grant must bind this same descriptor/profile.
Readiness includes actual numeric import, complete baseline, reviewed identity,
pre-leg native/QA, real predecessor CUDA exit and independent guard readiness.
A durable implementation package binds exact code and genuine CPU test execution;
fixture tests cannot stand in for production readiness.

```powershell
& <python.exe> -u <absolute-repository>/tools/run_badminton_incremental_guarded_leg.py --inspect-request --request <request.json> --descriptor <descriptor.json> --descriptor-sha256 <SHA>
& <python.exe> -u <absolute-repository>/tools/run_badminton_incremental_guarded_leg.py --preflight --request <request.json> --grant <new-grant.json> --descriptor <descriptor.json> --descriptor-sha256 <SHA>
& <python.exe> -u <absolute-repository>/tools/run_badminton_incremental_guarded_leg.py --execute --request <request.json> --grant <new-grant.json> --descriptor <descriptor.json> --descriptor-sha256 <SHA>
```

The exact `-u` argument order and driver identity are validated. Inspection grants
nothing; preflight requires real authority but starts no CUDA. Grant/local stage
time remains at most thirty minutes within its original absolute deadline.
Startup and pre-apply repeat readiness. Fresh non-fixture desktop identities
(PID, creation time, executable; ≤15 minutes old and valid through the deadline)
distinguish WDDM GUI entries from foreign compute. Unknown, reused or inaccessible
identities block; no process-name exemption or external process termination exists.

CPU controller/selection use E26/27 with CUDA disabled; proposal/application use
0/1, GPU0 and the bound two-thread allocation. Suspended descendants enter the
private Windows Job before executing; the independent watchdog and ten-second
margin enforce deadlines/controller loss/owned-tree exit. Existing locks, leases
and failed outputs prohibit automatic relaunch. Only actual empty-Job proof
permits release. Postleg, native pixels, browser and admission remain separate.

The original guarded entry permits an explicit same-directory
`issuer_runtime_checks_file`, retaining its legacy default. New authorization
needs new immutable request/runtime/schema/implementation/readiness bindings;
selecting a new file never extends expired authority. Its optional `--pregrant
<bundle> --pregrant-sha256 <SHA>` uses `verified_read_pass.py` to do deep CPU checks
before GPU handover. This mode requires a canonical non-junction interpreter and
read-locked exact inputs/runtime; no alias fallback is allowed. A same-process
live `cpu_ready.json` capability binds the read set and driver identity. A fresh
grant binds that capability plus fresh desktop evidence. Before apply, only exact
mutable FORMAL outputs may be released; parent/unselected aliases reject release.
The capability is invalid across mutation or closed handles; restart requires a
new full CPU pass and new authority. Incremental guard has no read-pass mode.

## Explicit incremental formal release policy v2

```powershell
python -B tools/prepare_badminton_formal_increment.py --release-policy <frozen-policy.json> --release-policy-sha256 <SHA> --root <new-formal> --threads 1
```

`badminton-incremental-release-policy-v2` binds reviewed scope and five actual
`parent_release` roles: archive/dataset/training manifests, final consumer contract
and successful final consumer audit. The audit must be
`final_cpu_validation_passed_no_training_started`, stage `final`, binding the
sealed final contract. Its three final-delivery bindings must equal the archive/
dataset/training inputs, whose real seal/verification are checked. Inherit the
actual archive's `provenance/model_bridge.json`; never invent a historical runtime
receipt. The policy binds original accepted camera/source review lineage, not a
future registry or guessed counts.

The policy may declare a train-only increment, while the complete old base/final
union retain all required nonempty roles. Preparation preserves original native
windows/settings/addresses and runs only CPU body planning. Numeric import and
completed-body aggregation retain all original numerical/evidence checks.
Legacy two-parent/all-new-role behavior remains unchanged.

Accepted and review snapshot digests may differ only through recognized
`badminton-admitted-review-calibration-derivation-v1`. Compatibility requires both
self-digests, all original lineage/artifact/source bindings and typed equality
of every other field, plus the actual formal-input equivalence proof. Equal
digests retain their original strict route. This is metadata derivation, not a
camera-geometry exception.

## Canonical publication and release

After full native verification, audit all candidates with
`audit_badminton_fresh_continuity.py` and render original dual-camera witnesses
using `render_badminton_continuity_cases.py`. Pelvis/track flags are review
candidates, not automatic same/different-person truth.

For witnessed cross-person boundaries,
`publish_badminton_canonical_tracks.py --source-manifest <json>
--reviewed-boundaries <json> --output-root <new-root>` changes only identity keys.
Numeric tensors and bilateral membership remain bit-exact. Under explicit v2
policy, `canonical_mode=reviewed_no_changes` instead requires
`publish_badminton_canonical_no_change.py --source-manifest <bilateral.json>
--reviewed-continuity <root-review.json> --output-root <formal>/repair/<new-view>
--release-policy <policy> --release-policy-sha256 <SHA>`.
Its typed v2 proof covers every genuine continuity flag; zero cases are valid
only for a genuinely flag-free audit. Mode strings or reviewer status alone
never admit labels. Full-body QA and bilateral accepted counts remain distinct.

Original full-body exports can omit per-entry SHA fields; then canonical and
consumer checks use the exact body tensor/metadata bindings from the genuine
full continuity audit and matching source/export paths. Present hashes must
agree; bilateral exports still require theirs. No historical labels or receipts
are rewritten for this format compatibility.

`assemble_badminton_expansion.py` writes provisional
`assembly_manifest.json`, preserving all old tensor values and raw metadata
bytes as role prefixes, per-source calibration and zero-sample sources. V2 uses
the exact explicit release policy and sealed
`expanded-source-increment-assembly-input/v2`. Zero-new roles are independent
byte copies with complete parent SHA equality, not reserialized equal tensors.

Only after actual canonical/continuity, native pixels, browser acceptance and
consumer sequence preparation may `seal_badminton_expansion.py --assembly <json>
--browser <json> --continuity <json> --pixels <json> --sequence-report <json>
--data-root <archive-root> --release-id <new-id>` seal independent copies.
V2 stages also require the identical `--release-policy` / SHA pair. The seal
verifies the archive and appends its catalog without changing default selection.
Final independent consumer acceptance is still required; GPU release and any
training authorization are recorded separately.

## Playback and native pixels

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
python -B tools/observe_badminton_dual_view.py --source-root <formal>/repair/final_labels --output-root <formal>/repair_paired --device cpu --threads 1 --decode-threads 1
python -B tools/serve_badminton_canonical_tracks.py --manifest <canonical.json> --viewer-output <new-viewer-runtime> --native-decode-threads 1 --port 8803
python -B tools/verify_badminton_fresh_playback.py --root <formal>/repair --url http://127.0.0.1:8803/ --canonical-manifest <canonical.json> --output-root <new-browser-acceptance>
```

Explicit safe decoding uses PyAV SLICE with one thread and PTS tolerance 2 µs.
Omitting the option keeps the original decoder behavior. Cache reuse still
requires actual pixel SHA, source/camera/native row, calibration/model settings
and bound native-pose file SHA. Cache mismatch runs actual detection; interpolated
or nearby teacher states never count as native presence. Resumption requires the
same decoder profile.

Playback reuses full 6,890-vertex geometry, original point members and native
camera projections. Both RGBs, meshes, cloud and scan identity commit atomically;
loading keeps the prior complete transaction. `preview_decoder.py` retains
PTS-checked streams. Paired mode's visible canvas/frame binding is authoritative,
not alternate MP4 `video.currentTime`. `verify_badminton_viewer.py` covers controls,
viewport/mesh clipping and native transactions; `benchmark_badminton_playback.py`
measures actual commits, waits and decoder seeks rather than wall-clock time.

Fresh playback requires Node and a separate Chrome for Testing binary, selected
by `GVHMR_CHROME` (legacy `LIDAR_CHROME` is accepted) or the local
`build/tools/chrome-for-testing` installation. The maintained lifecycle helper
is `tools/lib/browser_session.mjs`; `--browser-session-helper` is an explicit
override. Owned browser registry/evidence belongs in `outputs/browser_sessions`.
Never attach acceptance to a daily Chrome/profile. Coverage derives from frozen
sources/ranges: require at least two genuine five-second clips per source plus
every reviewed identity boundary. Browser/service teardown and actual owned
process exit are part of acceptance.

Display-only crop sidecars bind the real dataset/source scope and original bbox
SHA. Native Region-minus-Ignore background filtering preserves actual human/mesh
projections; a formal `badminton-formal-background-ground-display-v1` sidecar
may apply the accepted 15 mm background preview while protecting every stored
person member, including noncanonical fits. Use a separate viewer-output root.
No display filter changes source crop, camera/court geometry, body QA or training
membership. The 1022 original browser receipt also binds five actual with/without
display comparisons.

`audit_badminton_expanded_pixels.py` checks every held-out test image with its
source-specific native address/calibration. It does not claim all train/val pixels
were audited. A mismatch requires `recheck_badminton_expanded_pixels.py` lossless
native-PTS captures and unchanged full-person association in both cameras.
Corrections bind original video, record/PTS/common time, PNG and raw BGR hashes.
All person admissions must remain unchanged; otherwise supplement publication
fails. Consumers verify and replay those exact retained captures; future live
decoding is not promised to be byte-reproducible.

The current release's real pixel audit checked 26,124 images with zero unknown
differences. Thirteen historical corrections were recaptured in both cameras;
all 26 affected person admissions stayed unchanged. These immutable captures,
native journals, teacher/fit artifacts, identity evidence, model bridge and
release receipts are dependencies, not disposable caches. The
[release contract](BADMINTON_REPAIRED_RELEASE.md) specifies projection/sequence use.
