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
1020 is excluded until its own applicability and native source contracts arrive.

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
