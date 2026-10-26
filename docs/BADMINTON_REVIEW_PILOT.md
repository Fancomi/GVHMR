# Native rally review pilot

`tools/run_badminton_pilot.py` consumes a
`badminton-review-pilot-v1` manifest: five original camera files, hash-bound review
authorization, source indexes, explicit historical relocation and native-confirmed
short rally ranges. The pilot is limited to 180 seconds. Outputs remain
`review_only=true`, `training_published=false`; it changes neither camera admission
nor an existing training release. Full coverage uses the
[full-native entry](BADMINTON_FULL_NATIVE.md).

```powershell
uv run --no-sync python tools/run_badminton_pilot.py --manifest <pilot.json> --root <new-run> --stage prepare --threads 2
uv run --no-sync python tools/run_badminton_pilot.py --root <new-run> --stage teacher --gpu-lease <fresh-held-lease.json> --threads 2
```

The two cameras run independent GVHMR teachers. The accepted v2 fitter uses cam2
prior, independent native 2D observations from both cameras and actual LiDAR.
Cam1 prior is retained for review; do not average cross-camera 3D priors without
identity evidence. Local track IDs and unknown relationships remain separate
until native evidence supports a decision.

Each manifest-bound relocation resolver is loaded explicitly and must match the
registry, snapshot, plan and pilot input. It never searches for replacement
files or rewrites frozen manifests. See [data relocation](DATA_RELOCATIONS.md).

Existing review evidence remains at:

| Run | Frozen scope | Evidence |
| --- | --- | --- |
| `outputs/badminton_1019_r11_pilot_v1` | 5 rallies / 525 scans | 1,050 body-QA candidates; `quality_report.json`, `outlier_review.json` |
| `outputs/badminton_1019_r11_pilot_second_v1` | 9 rallies / 1,260 scans | 2,503 fitted / 2,502 QA accepted; `coverage_audit.json`, `coverage_by_scan_track.jsonl`, `coverage_native_review/`, `interior_review/` |
| `outputs/badminton_1019_r11_review` | Ten real five-second clips | Native dual-camera pixels, full 6,890-vertex meshes, synchronized scan commits and actual Chrome receipts |

The second pilot's 17 unfilled track-scan cells exceed the original 18 ms
teacher-time tolerance; no label extrapolation fills them. Native review shows
the athlete remains visible, so this is not proof of absence. Both runs retain
accepted floor-penetration and timed pelvis outliers. Point-to-visible-surface
residual and detector reprojection are internal consistency measures, not
independent anatomical ground truth.

`serve_badminton_pilot_review.py` combines verified pilot services without copying
labels; `verify_badminton_pilot.py` checks actual playback. Original snapshots keep
their own digests. All historical leases are released: viewing existing outputs
requires no inference, and new GPU work needs a fresh stage contract. These
review candidates did not themselves publish training data.
