# Full native rally fitting

The full-run entry is `tools/run_badminton_full_native.py`. It shares the existing
native teacher, identity, v2 fitting, verification and viewer components with the
short pilot, but requires a separate `badminton-native-rally-fit-v1` manifest.
The short pilot's 180-second limit remains unchanged.

The full manifest retains all five camera-file hashes, source-index bindings,
exact native rally evidence, ordered unique paired scan ordinals and explicit
historical relocation. It additionally records
`execution_scope=all_effective_native_rallies`, the actual `user_confirmation`
and a SHA-bound `coverage_audit`. The original camera `review_authorization`
continues to apply; the new scope does not require another user confirmation.
Fitting remains `review_only=true`, `training_published=false`.

The 1019 manifest is
`D:/PROJECT/LIDAR/datas/badminton/collections/gvhmr-1019-full-20261021/handoff/full-native-rallies.json`
(SHA256 `bb64e2766442558033e5092046e2a89ede3fa03a12a218414ef29ea057dcb8f2`).
Its 44 native windows allocate overlapping buffered ranges once: 4,117 selected
scans / 410.238286 seconds, including the unchanged 14 pilot windows / 1,785 scans
and 2,332 new scans. Three initial scans lack dual-camera pairing. Original rally
lineage and every overlap assignment remain in `coverage-audit.json`.
1020 applicability has since been accepted against the unchanged R11 numerical
bundle, with its new `sources.json` bound separately. Its five recordings total
64.85 raw minutes, which is not qualified rally/body-label duration. Actual
native source contracts remain pending.

The independent run is `outputs/badminton_1019_full_native_v1`. Actual CPU
preparation took 52.63 seconds including command startup; its own measured inner
prepare took 41.32 seconds. The snapshot digest is
`c0151b6c9bc5f529f62bb33ddb184a42ca97e09f286c6ca95073af3aff5a1763`.

```powershell
uv run --no-sync python tools/run_badminton_full_native.py `
  --manifest <full-native-rallies.json> --root outputs/badminton_1019_full_native_v1 `
  --stage prepare --threads 2

uv run --no-sync python tools/seed_badminton_full_teachers.py `
  --root outputs/badminton_1019_full_native_v1 `
  --reuse-run outputs/badminton_1019_r11_pilot_v1 `
  --reuse-run outputs/badminton_1019_r11_pilot_second_v1

uv run --no-sync python tools/seed_badminton_full_poses.py `
  --root outputs/badminton_1019_full_native_v1 `
  --reuse-run outputs/badminton_1019_r11_pilot_v1 `
  --reuse-run outputs/badminton_1019_r11_pilot_second_v1
```

Teacher reuse requires native-verified original runs, identical camera files and
numerical cameras, source indexes, window bounds/selected scans/settings, video
and model identities, original native rows and temporal provenance. It copies
independent files, checks every copied SHA, then invokes the existing teacher
resume validator without inference. The actual 14 windows / 28 camera contexts
passed this check, recorded in `teacher_reuse_receipt.json`.

Native 2D reuse checks the physical camera bundle, source index, original frame
row, all model-file hashes and inference options. Changing the selected source
group changes the snapshot/config digest. A new record explicitly preserves the
old file SHA/config digest while rebinding only this scope metadata; boxes,
keypoints, confidence and original raw pixel SHA remain unchanged. The actual
3,570 records are in `native_pose_reuse_receipt.json`. This is partial seeding:
the native pose stage must compute every remaining frame before a complete
`pose_receipt.json` can admit body fitting.

Teacher GPU execution uses a separate held manifest-bound lease with an actual
UTC deadline. `--stage teacher` dispatches to `run_badminton_full_teachers.py`,
which preserves completed camera-window checkpoints, verifies cached artifacts
and launches one independent native camera worker at a time. The default
300-second safety margin prevents starting another worker near the deadline.
The supervisor yields only after a completed subprocess and records
`safe_to_release_gpu`; the coordinating owner checks actual process exit before
writing the shared lease `released`. Subsequent teacher/pose/fit leases require
another coordinated safe GPU handoff.

```powershell
uv run --no-sync python tools/run_badminton_full_native.py `
  --root outputs/badminton_1019_full_native_v1 --stage teacher `
  --gpu-lease <fresh-held-lease.json> --threads 2
```

Runtime progress is `full_teacher_progress.json`; each active worker's actual
Python PIDs and log are recorded there. A native window can begin before the
source's clock origin; signed window IDs are forwarded as `--window=value` to
retain the original interval. The first such window exposed an argparse-only
startup failure, fixed before CUDA inference; the deadline was not extended.

Full pose, identity review, fitting, native point/SMPL verification, coverage,
outlier/quality reports and the final full-run viewer remain required. Existing
QA acceptance and previous pilot outliers are retained as evidence, not promoted
to independently accurate ground truth or formal training admission.

Batch 01 completed 17 new camera contexts in 1,499.39 seconds: 45 / 88 total,
including 28 reused contexts. It released at 2026-10-22 00:44:18.854099 UTC,
before the 00:48:18 deadline. All observed owned processes exited. Remaining
teacher work is approximately 60–75 GPU minutes excluding queueing, poses and
fitting; subsequent GPU windows require another explicit safe handoff.

The first supervisor retains the historical argparse-only `exit_code=2` field;
its actual final state is `yielded_before_deadline` with 45 valid checkpoints.
That progress file is SHA-bound by its released lease and stays immutable.
Batch 02 uses a new `--progress-output` plus `--resume-progress` pointing to the
first file. The common native worker records actual Python descendants, stays
hidden and runs below normal priority. `tools/release_badminton_full_gpu.py`
checks completed stage state and actual owned CUDA process exit before release,
and preserves an immutable progress snapshot. CPU stages are distinguished from
native inference; no subsequent GPU stage is launched by the release observer.

For batch 02 add these arguments to the full teacher command:

```powershell
--progress-output outputs/badminton_1019_full_native_v1/full_teacher_progress.batch02.json `
--resume-progress outputs/badminton_1019_full_native_v1/full_teacher_progress.json
```

The separate `fit_reuse_audit.json` proves 14 old windows / 3,553 fitted / 3,552
original QA records eligible, including 1,785 actual clouds and 3,570 exact pose
files. Git `e159d1e` reproduces the first pilot's original execution digest:
the nine other execution modules are byte-identical, and calibration AST matches
outside snapshot freezing/input-metadata checks. All numerical calibration
fields and original native session contracts also match. Both original/current
digests remain recorded; this is an eligibility audit, not another fitting run.

Only after current full poses and reviewed identity decisions are complete may
`tools/apply_badminton_full_fit_reuse.py` import these results on CPU with CUDA
masked. It checks unchanged local identity operations and exact teacher parameter
bytes, copies NPZ/joblib artifacts unchanged, and explicitly rebinds scope/path
metadata while retaining original provenance. Original QA and known bad cases
remain unchanged. Full current native verification is still required.

Full `--stage fit` uses a cooperative per-window supervisor. Existing outputs
first pass `--require-completed --device cpu`; changed caches fail without silent
fallback recomputation. New windows use the existing v2 policy and yield before
the lease boundary. GPU release precedes separate CPU-only `--stage aggregate`,
which also requires completed caches and cannot start new fitting/inference.

1020 CPU-only watchers wait for `handoff/<sid>.full-native-rallies.json`, each
source index using the existing `{file, sha256, bytes}` binding with `file`
relative to `data_root`. `{path, sha256}` is reserved for receipt/evidence
bindings and is rejected for `source_indexes` / `excluded_source_indexes`.
Each watcher uses `outputs/badminton_1020_full_native_<sid>_v1`. The joint manifest watcher
remains active as well. Actual launch references are in
`outputs/badminton_1020_full_native_v1/source_intake_watchers.json`; per-run
`intake_watch.json` distinguishes waiting, read-only validation and actual CPU
preparation. `tools/watch_badminton_full_native.py` masks CUDA for every
preparation subprocess. No Windows scheduled tasks are added.

The 1020 bundle binds original recordings because it predates fullraw indexes.
Portable review freezes may validate that explicit fullraw parent against the
original recording manifest, common clock, PTP validation, download receipt and
PCAP bindings, requiring matching file/SHA/bytes and source identity. The four
small originals are hash-checked; PCAP identity retains the original download
SHA and current size contract. The existing explicit source-index branch stays
strict. Neither branch searches for missing files or changes the five-file bundle.

Source `20261020_191011` completed actual CPU intake at
`2026-10-22T03:37:09.128801+00:00`: manifest `93ff37af...`, 23 segments,
2,285 unique eligible scans and 227.736177 seconds. Prepare CLI took 29.481714 s
(19.166396 s internally); the separate complete body-plan prepare took 5.431744 s.
`outputs/badminton_1020_full_native_20261020_191011_v1/cpu_prepare_receipt.json`
binds the prepared registry, input, case plan, calibration and all-window body
plan. Its 46 independent camera-teacher contexts, 4,570 planned native
camera/scan observations and subsequent fitting/QA remain pending.

The explicit two-cohort GPU queue is
`outputs/badminton_full_native_gpu_queue_20261022/queue.json`: 1019 has 43 teacher
contexts left, followed by the first ready 1020 source's 46. Each teacher lease
requests at most 30 minutes and uses its own progress file; subsequent poses,
reviewed identity, fitting, native verification and the full player retain their
existing stage gates. The queue does not grant or launch GPU work automatically;
Sports360's actual explicit handoff is still required.

All five 1020 frozen sources passed the existing consumer's CPU source reader
at `2026-10-22T04:43:12.928298+00:00`, taking 122.102458 s overall. The run
verified original clocks and native camera indexes, plus six encoded point-member
samples per source. All 30 sampled crops were bit-identical subsets of their
emitted parent PNGs; the maximum single-M roundtrip error was 9.46e-7 m.

| Source | Emitted scans | Paired nonempty scans | Empty crop scans |
| --- | ---: | ---: | ---: |
| 20261020_191011 | 3,664 | 3,649 | 11 |
| 20261020_182854 | 10,027 | 10,015 | 6 |
| 20261020_184642 | 6,529 | 6,520 | 0 |
| 20261020_193236 | 11,975 | 11,920 | 49 |
| 20261020_200346 | 6,711 | 6,707 | 1 |
| Total | 38,906 | 38,811 | 67 |

The reader's paired count already excludes empty foreground and validates common
overlap and both native cameras. Of 67 empty crops, 64 have both camera matches
and three lack one or both; another 28 nonempty scans lack camera matches.
The 38,911 original clock rows retain five explicitly unclosed final packet rows.
These source counts precede native rally selection and are not fitted bodies.
Independent reports and their SHA bindings are in
`outputs/badminton_1020_source_cpu_preflight_20261022/completion_receipt.json`
(SHA `7863c8e07168a785c35f721a91ce6d5a4dfff7e1d37a07d930ab5c30108b15bd`)
and `coverage_before_native.json`. The other four source watchers still wait for
actual successful native manifests; no source preflight grants GPU or training admission.

Full-native `prepare` now also runs the all-window body-plan stage with CUDA
masked and `--device cpu`, using the same command/settings as subsequent fitting.
An incomplete or failed body plan cannot make the intake ready. Existing successful
body-plan receipts are checked and retained byte-for-byte. This closes the earlier
manual preparation step for the four subsequent sources and the joint run.
The helper was actually exercised on all 44 prepared 1019 windows (exit 0,
6.128281 s), preserving the old B01 progress, release receipt and full body-plan
SHA. The focused suite has 118 passing CPU checks.

The user's priority instruction on 2026-10-22 superseded waiting for Sports360's
full round. Sports360 saved a complete checkpoint and explicitly paused its GPU
trainers/admission for the whole production. The remaining four native stages
run first under their own explicit grants. The prepared continuous human-production
request is `outputs/badminton_full_native_gpu_queue_20261022/continuous_gpu_request.json`;
it is a request, not a granted lease. It proposes at most 30 hours after the
native CUDA executors exit, actual manifest-bound local stage leases, and 1019
followed by the 1020 joint run. No per-source 1020 GPU fitting has started, so
the joint root avoids duplicate inference. Sports360 resumes only after the
whole production is explicitly returned through the coordinating root.

`tools/watch_badminton_full_gpu.py` is a hidden CPU-only receiver for that final
grant. It verifies the frozen request, both actual manifest hashes, four distinct
native return proofs, the Sports360 stop proof, delegation and the fixed overall
deadline. A sample, future window or incomplete handoff cannot start inference.
On the real window it creates the bounded B02 lease and starts the 1019 teacher
controller plus its independent release observer. It then stays resident as the
deadline monitor while poses, reviewed identity and fitting are dispatched.
An active grant is immutable: replacement, removal or expiry stops only known
CUDA trees in the two authorized outputs, matching PID creation identities.
The receiver's actual PID, heartbeat, child PIDs and release binding are in
`outputs/badminton_full_native_gpu_queue_20261022/continuous_gpu_receiver.json`.
The focused 130-check suite includes real CPU subprocess-tree termination and
proves that a different output root keeps running. No scheduled task is used.

`tools/run_badminton_full_sequence.py` continues teachers, native poses, reviewed
identity, audited fit import, bounded fitting, CPU aggregation/native verification
and the full viewer serially. It emits durable review events and cannot invent a
reviewed identity decision. `live_progress.json` selects the latest teacher batch
and its preceding immutable release snapshot: B03 reports baseline 79 plus its
own nine new camera contexts. Historical timing stays under `previous_attempts`.
A replacement CPU coordinator waits for the existing GPU batch's actual release
before dispatching any following stage; it does not restart the CUDA tree.

At the user's new paired-RGB acceleration request, the CPU sequence was explicitly
held and B02 yielded after its current worker: 79/88 contexts, with 34 new in
2,687.424143 s; actual CUDA release was `2026-10-22T06:26:36.187496+00:00`.
The original local lease was retained before shortening its deadline; the active
continuous grant is unchanged. `execution_hold.json` records the separate
10 Hz frontend trial. The default native 30 Hz contract remains unchanged.

The bounded trial subsequently passed three dual-camera windows, including a
native RGB confirmed jump, and a 50-scan/two-person joint-LiDAR fitting case.
The controlled first window used identical 40-pixel ROI margins in both arms:
122.876 to 70.479 seconds, 1.743x. Two production-20-pixel windows reused exact
historical baselines; their 3.10x/2.06x ratios are not controlled cold A/B estimates.
The same 100 actual person/scan cells fit and passed QA in both backend arms.
Visible-point surface mean was 24.408 to 24.642 mm; all 100 passed the existing
bilateral detection rule. Original point-member and pelvis errors were zero;
rebuilt native SMPL maximum error was 9.54e-7 m in both arms. Camera-1 fitting
support differed in one cell because the old fallback used an adjacent real
exposure; the common 96-cell projection mean was 7.933 to 7.963 px. These are
pseudo-label consistency measurements, not independent anatomical ground truth.
Receipts are under `benchmarks/paired_rgb10/` in the two authorized roots.

The adopted explicit `paired_native_rgb10_latent_v1` input is frozen separately
as each root's `teacher_sampling_input.json`, binding the original source plan
and real backend acceptance. `paired_sampling.py` selects actual native record,
PTS and common-clock rows from LiDAR pairs; integer first-core scan-midpoint
subtraction precedes float seconds. The source MP4 remains 59.94 fps; paired
RGB record steps are five/six, never an index divided by two. No raw recording,
camera, original scan selection or global GPU deadline changes.

`run_teacher` routes this explicit policy through the same initial/repaired
teacher entry. Tracking gaps/minimum context retain their original durations.
Real crops feed the heavy image models at about 10 Hz; role-marked near-30 Hz
inferred states interpolate conditioning features, with virtual 2D confidence
zero. Only actual RGB anchors remain observations. `sample_track` queries those
inferred model states while requiring real RGB evidence within 8,500 us and
preserving its original record/PTS/keypoints; the ordinary 36 ms bracket rule
is unchanged. Identity mapping and exact pixel-cache reuse share that duration
policy, so repaired contexts cannot silently rerun dense preprocessing.

The continuous sequence resumed at 07:29:15 UTC with `--paired-native-rgb`.
1019 B03 finished all nine remaining camera contexts in 415.993 s, reaching
88/88 and releasing actual CUDA at 07:36:16.462899 UTC. Its original 79 dense
contexts remain byte-identical. Native pose, real identity review and fitting
follow; 1020 joint has 247 segments/25,865 scans and its own bound sampling
input. The focused CPU suite has 159 passing checks. Whole-run delivery is
estimated at 15–23 hours from 07:30 UTC, including CPU review/acceptance with
some overlap, within the unchanged 2026-10-23 11:41:39.650687 UTC hard deadline.
No new formal training release has been published.

The first 1019 full review completed GPU fitting at 08:29:40 UTC: 44 windows,
8,124 actual fitted person-scans and 8,115 fit-QA candidates. All 8,115 candidates
passed native point-member/pelvis verification with zero error and rebuilt SMPL
error at most 1.91e-6 m. Ten actual five-second Chrome clips passed at 08:48:50 UTC,
including the tail of rally 19's jump. Review: http://127.0.0.1:8784/ .
The 1020 joint teacher batch actually started at 08:29:43 UTC, overlapping the
first cohort's CPU verification/browser work, and the original sequence adopts
that exact batch only after its real CUDA release.

`tools/audit_badminton_full_coverage.py` accounts for all actual selected scans
before reporting qualification. A nominal two people on 4,117 scans gives 8,234
cells: 112 missing nominal cells offset by two additional fitted hypotheses
produce the net difference of 110. There are two zero-fit scans, 108 one-fit
scans, 4,005 two-fit scans and two three-fit scans. Missing support comprises
92 cells without usable continuous primary teachers and 20 excluded at native
pose/point segmentation. Nine distinct fitted cells fail QA, with potentially
overlapping rejection reasons. Existing cached-native bilateral detection gates
qualify 8,100 of the 8,115 QA candidates; 15 lack a qualifying camera-1 match.
These are presence-rule candidates using original bound native detections, not
a new formal training export or repeated full native-pixel acceptance.

The all-fit mean visible-point residual is 23.367 mm and raw camera reprojection
means are 8.054/4.363 px. Accepted ground outliers remain, including source
152750 scan 3948 at 115.77 mm, source 152542 scan 443 at 110.39 mm and source
152750 scan 5277 at 98.41 mm. Two pelvis-step candidates remain explicitly
reviewable; one has a rejected endpoint and the other is the previously observed
2576-to-2577 fast motion. Full receipts are `coverage_bilateral_audit.json`,
`coverage_bilateral_cells.json`, `quality_report.json`, `outlier_review.json`
and `browser_checks/continuous_verification.json` in the 1019 full root.

The accepted 115.77 mm case is source `20261019_152750`, scan `3948`,
motion `000393173671_000397340410:1:0` (`003948_t000.npz`). Its ground
distance uses the frozen shared physical court plane in viewer metres:
`h = (vertex - origin) dot normalize(normal)`, then
`max_vertex(max(-h, 0))` over all 6,890 SMPL vertices. The plane origin is
`[13.31866272714975, -0.4794938249119665, -2.9825130024854447]` and its
upward normal is `[-0.04387101604378947, -0.10197160568839171, 0.9938194632752052]`.
This is separate from the ground Ignore upper surface. CPU recomputation from
the saved vertices gives 115.76760 mm versus the stored 115.76763 mm; 354
vertices are more than 20 mm below this plane. No plane or camera was changed.

The existing optimizer uses a 20 mm soft tolerance with floor weight 0.5;
this is not an export rejection threshold. `export.acceptance_reasons` rejects
penetration strictly above 120 mm, point-to-surface mean above 65 mm or p90
above 120 mm, fewer than 80 source points, nonfinite fits, combined reprojection
above 50 px, or fewer than six valid joints in every available camera.
This case has 308 points, surface mean/p90 26.25/52.86 mm, finite parameters,
16 valid joints per camera and combined reprojection 40.735 px, so its existing
body QA has no rejection reasons. The combined reprojection averages available
per-camera confidence-weighted errors (57.455/24.014 px here); it does not impose
a separate 50 px limit on each camera. It also passes the unchanged bilateral
detection-presence rule, which does not add an anatomical or 20 mm floor gate.

This person-scan is currently review-only and is absent from formal training.
`dual_view.export_reviewed` rejects a review-only dataset or calibration; the
only present split is `review`. If a separately admitted future formal source,
camera bundle and role assignment retain these tensors and numerical gates,
this person-scan would be eligible under those gates. Its visible defect is
not evidence that the current rules exclude it. The SHA-bound supplement
`ground_acceptance_review.json` preserves the original
`full_native_review_completion.json` bytes and binds the exact fit, QA rows,
plane, presence evidence and rule implementations. No thresholds, sample
membership, cameras or formal releases were changed for this clarification.

Formal-release readiness is recorded separately in
`release_preflight/formal_release_readiness.json`. Existing CPU verification
proves all 8,115 aggregate rows equal their native-verified window tensors;
the current presence-qualified subset has 1,157/6,943 rows and 102 real
continuous segments. The unchanged full continuity screen covers 8,115 rows
and 8,022 adjacent edges with no flags under its existing criteria.
The seven-source role draft preserves the full old 142,567 rows and their
98,011/19,213/25,343 roles. New train sources are both 1019 sources plus
1020 182854/193236/200346; val is 191011 and test is 184642. This draft uses
only frozen eligible-scan counts and still needs the consumer's sealed input
bindings. The original two-source 1019 review cannot independently populate
all three nonempty source-disjoint roles required by the current intake.

Camera admission requires a separate real applicability certificate and new
admitted source indexes, not an edit to the active review inputs. The new
formal calibration package also declares all seven `valid_source_ids` in
`calibration.json`, and matching canonical `frame_maps` in `sources.json`.
Changing source-scope/admission metadata requires an explicit derived-fit
lineage adapter: existing strict review fit-reuse validation is retained.
Final body completion, bilateral/native/pixel checks, version-bound Chrome
acceptance, exact expansion, archive sealing and independent consumer CPU
admission remain separate gates. The expansion loader now accepts a complete
already-expanded base only with the consumer's explicit `old_archive` SHA,
matching archived roles/payloads/source registry and verified per-source
calibration snapshots. The actual 142,567-row base passes this structural and
calibration preflight (13 sources, two calibrations). Actual assembly still
hashes and checks every role payload/row and retains its exact tensor/metadata
prefix. Draft or unsealed joint contracts cannot assemble the expansion.
The expanded-dataset/pixel CPU suite passes 18 checks, including mixed old
calibrations, unchanged prefixes, archive escapes, wrong payload hashes and
substituted calibrations. The browser wrapper still needs the new seven-source
scope; old 1015-specific assertions are not satisfied by relabeling a review
receipt. Pending-runtime ranges in the readiness receipt are planning
estimates, not measured formal-release completion.

The joint publication uses two contract stages to avoid circular hashes.
`expanded-1019-1020-assembly-input/v1` binds the actual seven-source
paired/canonical publication, final per-source admitted counts, old full
archive and source inputs; it must be sealed and complete before assembly.
`producer_joint_manifest` refers to that new source publication. After actual
archive sealing, the separate `expanded-1019-1020-consumer-contract/v1` adds
`final_delivery` archive/dataset/training manifest bindings and typed
`release_evidence` for audit, native verification, continuity, pixels, browser,
seal and sequences. The final receiver rejects assembly-input contracts.
`release_preflight/final_evidence_interface.json` binds historical report
examples and their actual field/status meanings; those examples do not admit
the new data. Drafts and missing/mismatched new counts/publications are rejected.

`tools/audit_badminton_formal_inputs.py` audits a separately prepared formal
source root against explicit `--review-root` inputs and the frozen role draft.
It requires identical physical camera/court/ground/ROI/timing fields, native
session maps/clocks, original full source frames/member declarations and crop
lineage. The formal loader verifies real applicability certificates and their
accepted five-file bundle. Relocation is enabled only by explicit
`--data-relocations` and `--data-relocations-sha256`; default readers and strict
fit reuse remain unchanged. Its input-equivalence receipt grants no fit import,
body completion, repeated full RGB pixel check or training publication.

Actual seven-source formal inputs were frozen and independently audited at
`outputs/badminton_1019_1020_formal_v1`: 46,444 emitted source rows, 29,982
selected scans and 291 effective native rally segments. Preparation took
308.25 CPU seconds; `source_prepare_receipt.json` SHA begins `5513d8bf` and
`formal_input_equivalence.json` begins `0ce0e0f1`. The new snapshot digest is
`5e4c7b0f4e3bf5758ef8e8fe327787c44ffa427519a2542fb4b4de33eed005fb`.
This completes camera/source admission only. Formal repair planning requires
that exact completed audit via `--formal-input-equivalence` plus its SHA;
it retains the original native rally windows and all three nonempty roles.
Existing repair components can resolve the separately typed formal registry's
explicit relocation declaration only through its frozen intake, real camera
handoff, matching source-index identities and camera snapshot bundle. Ordinary
undeclared runs keep their existing relocation checks. The relevant CPU
relocation/formal-planning suite passes 79 checks with one platform skip.

The continuous CPU relay encountered a transient Windows permission error
opening a live teacher journal at 10:10:35 UTC. The running B01 teachers still
completed 101 camera contexts and released real CUDA at 10:25:10. A replacement
relay resumed B02 at 10:32:16 under the same fixed grant/deadline, preserving
all 101 artifacts and the first cohort's completed viewer. Live JSON reads now
retry only bounded permission errors; corrupted JSON and persistent failures
remain errors. Three recovery tests pass; queue recovery receipts preserve the
failed state and replacement process identity.

The formal 291-window body plan also completed on CPU in 125.16 s:
`body_prepare_receipt.json` SHA `efc064d7...` binds the original and formal
window configs, proving exact equality except the admitted calibration digest,
and equal full fitting settings. `repair/new_source_input.json` retains the
existing `badminton-new-source-repair-input-v1` jobs contract and SHA
`81e6409626b6aa7ee44bf78d96763dbbd113f6474cbd88675dc841a8899fa920`.
`preparation_handoff_receipt.json` (`8c8515c9...`) records the completed source
and planning proofs separately from unfinished body/native/identity import.
The consumer independently checked the complete old 142,567 rows and both
calibrations in 116.98 s and the actual new formal camera wrapper in 3.53 s;
its independent receiver has 49 passing CPU checks. These are preparation
receipts, not new body-label or final training admission. Existing guarded-leg
CUDA stages in a new formal output require their own explicit authorized
target/lease; the current immutable grant continues to cover its original two
roots only.

The separate guarded-leg request is
`outputs/badminton_1019_1020_formal_v1/gpu_guarded_leg/guarded_leg_gpu_request.json`
SHA `36799351328d5e7701e953fd5abce79deeadb1588266cfd16b30d2b9dd2c65a6`.
It targets exactly the formal `repair` root and source input `81e64096...`,
with a 5–15 minute planning estimate and 30 minute maximum. Its absolute
deadline is unchanged. Issuer fields, current policy/configuration hashes,
typed grant/readiness definitions and additional runtime checks are adjacent;
actual future baseline/identity/native/exit evidence remains pending. No grant
or live lease was created. The older command matcher does not cover the two
leg CUDA scripts, so an independent guard must capture their exact commands,
target and PID creation identities before launch. Baseline/native evidence
used for authorization must be immutable pre-leg snapshots.

Historical 1015 proposal-loop timing was 50.38 s for 62,895 body-QA candidates,
59 windows and 12 ray-witness cases, producing nine eventual corrections;
that measurement excludes process/model startup. New witness count is unknown.
The library exposes a device parameter, while both complete CLI stages fix
CUDA; a full CPU route has no validated timing. No CPU backend experiment or
algorithm change was made. The root's current CPU allocation is recorded in a
separate request addendum: pure producer CPU tools E26/27, BelowNormal, 1–2
threads; CUDA proposal/application and their inherited scheduling remain 0/1.

The independent executor is `tools/run_badminton_guarded_leg.py`, backed by
`guarded_leg_gpu.py`, `guarded_leg_execution.py` and `windows_gpu_job.py`.
It validates the bound local grant/readiness schemas offline, the unchanged
request/runtime contract and CPU allocation, the complete parent/formal
native/identity/body/import evidence and a tested implementation manifest.
No inspection or preflight creates a lease or launches CUDA. Actual execution
requires the coordinating root's new grant; the old two-root grant is unchanged.

Invoke the future driver with absolute paths in this exact argument order:

```powershell
& D:/PROJECT/LABEL/GVHMR/.venv/Scripts/python.exe -u `
  D:/PROJECT/LABEL/GVHMR/tools/run_badminton_guarded_leg.py --execute `
  --request D:/PROJECT/LABEL/GVHMR/outputs/badminton_1019_1020_formal_v1/gpu_guarded_leg/guarded_leg_gpu_request.json `
  --grant <absolute-path-to-the-actual-new-issued-grant>
```

The controller and independent watchdog remain on E26/27, BelowNormal. A
private Windows Job Object owns every phase and inherited descendant. Launchers
start suspended; their original kernel handles assign job ownership and affinity
before resume. CUDA propose/apply use 0/1; CPU selection uses E26/27 and CUDA=-1.
The watchdog must be live before the first phase. Both controllers enforce the
earlier grant/local-lease deadline with a ten-second shutdown margin, including
an independent timer thread while file checks block. Missing/revoked bindings,
foreign GPU ownership, reused PIDs or vanished controllers stop only the owned
job. Last-handle close also terminates the owned descendants. Existing outputs,
leases and execution locks prohibit silent retries. Only actual empty-job proof
permits release; post-leg native/QA/pixel/Chrome/archive checks remain separate.

The future complete baseline wrapper uses schema
`gvhmr-formal-preleg-baseline-v1`: `state=complete`, exact `source_input`
binding, separate immutable `body_receipt` and `pose_receipt`, and
`windows[key].report_snapshot` bindings. Store all pre-leg native/absolute-QA,
body/pose receipt and fit-report snapshots under the readiness file's baseline
directory; retain byte-identical report copies. The numerical importer uses
`badminton-formal-numeric-import-v1`, exact source/calibration, complete window
scope, `numerical_artifacts_byte_identical=true`, `qa_values_unchanged=true`,
and each window's `original_report` / `current_report` bindings. Current reports
must equal the baseline snapshots; original reports must match the real parent's
native verification. The executor rechecks exact ptv3/fit artifact sets and hashes.
The predecessor exit bundle uses
`gvhmr-guarded-leg-predecessor-exit-bundle-v1`, `state=complete`, unchanged
`predecessor_grant`, actual `process_identities` and `release_receipts` bindings.
Every registered predecessor must really be gone. These interfaces describe
future genuine completion; implementation tests do not fill those pending roles.

For genuine eligible-coverage gaps, publish flat clips retaining original
`rally_index` plus zero-based `segment_index`; source/rally/segment is unique.
Each segment carries actual clock bounds and continuous eligible PNG ordinals.
Nested `contiguous_windows` is rejected. Original native range and evidence stay
bound; the viewer distinguishes original rallies from effective segments.

Zero eligible sources use `excluded_source_indexes` plus `no_fit_sources` evidence
instead of empty clips. Overall full-fit clips must remain nonempty. A standalone
`badminton-native-no-fit-needed-v1` marker binds its actual source index,
zero-count per-source coverage and completed native analysis/planner files. Its
source watcher records `no_fit_needed` and exits without creating fitted input
or SMPL tensors. Joint coverage still accounts for the source explicitly.
