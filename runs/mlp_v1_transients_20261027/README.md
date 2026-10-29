# Frozen MLP rapid-change diagnostic — 2026-10-27

The residual MLP improves MAE in both diagnostic populations. Rare rapid-change
errors remain important, and a gyro-Y subgroup is worse than LOCF despite the
better overall benchmark. No training, tuning, or correction clipping occurred.

## Definition and scope

Subject 107 only: 102 segments, 285,488 samples, four existing loss masks.
Six frozen MLPs (direct/residual, seeds 0/1/2) plus LOCF: **2,856 replays**.
All original hidden positions were scored, with zero unavailable outputs.

Each channel is flagged at time t when its absolute change from the preceding
recorded sample strictly exceeds the existing training q99 threshold. The first
row of each segment is unflagged; differences never cross segment boundaries.
The two groups are hidden flagged positions and their hidden complement.
“Other” does not mean quiet: it can include fast steady motion and errors after
a rapid change. Labels describe recordings, not diagnosed motion or sensor faults.

These labels use hidden truth for offline scoring only. They cannot directly
trigger an online fallback because the receiver does not know the hidden value.

## Measured results

Counts below use **means across the three seeds**, after pooling error sums and
counts within each seed. There are 24 channel/regime cells per group. They are
descriptive cells, not independent statistical trials or an ensemble model.

| Family | Group | Cells with lower MAE than LOCF | Cells with lower RMSE than LOCF |
|---|---|---:|---:|
| Direct | Rapid recorded change | 22/24 | 22/24 |
| Direct | Other hidden positions | 22/24 | 21/24 |
| Residual | Rapid recorded change | 22/24 | 22/24 |
| Residual | Other hidden positions | 24/24 | 23/24 |

Every residual seed improves MAE in all 24 other-position cells and the same
22 rapid-change cells. Both remaining rapid-change MAE regressions are gyro Y
during five- and ten-sample bursts.

**New weakness:** rapid-change gyro Y during burst 5 contains **49/14,370 hidden
positions (0.341%)**. LOCF RMSE is **0.999938 rad/s**; residual mean-seed RMSE is
**1.192793 rad/s (+19.29%)**. All three seeds regress in this subgroup: +44.15%,
+9.30%, and +4.41%. Residual subgroup MAE rises **7.49%** on average. In the
complementary group, residual mean RMSE improves **12.54%**. The original
predeclared screen concerned overall channel/regime metrics; this new subgroup
finding does not change that criterion retrospectively.

**Error concentration persists:** random-loss acceleration Y has **176/14,176
flagged positions (1.242%)**. They account for **91.26%** of LOCF squared error
and **88.61–89.22%** across residual seeds. Residual models reduce these errors,
but the remaining squared error is still highly concentrated.

**Connection to the previously inspected failure:** residual seed 1/random
acceleration X adds **684.7982** squared-error units over LOCF overall.
The flagged group adds **649.2425 (94.81% of that net increase)**; the other group
adds **35.5557**. This percentage describes a net difference between models,
not the share of the model's total error or a physical cause.

MAE averages absolute misses. RMSE squares misses before averaging, so occasional
large misses have more influence. Better MAE can therefore coexist with worse
RMSE. No per-channel errors with different physical units are combined here.

The residual architecture remains the primary candidate by the original overall
objective. These results justify investigating reliability; they do not establish
a working confidence estimate, abstention policy, INT8 model, or hardware result.

## What the files do

- `../mlp_v1_transients_20261027.py`: replays unchanged checkpoints and groups
  errors using the existing diagnostic functions. Writes the policy before
  opening recordings and refuses an existing output directory.
- `policy.json`: exact group definition, scope and pinned input/source hashes.
- `report.json`: all per-case sums, pooled groups, checks and recording-access log.
- `strata.csv`: 504 physical-unit rows, including each seed, all positions, both
  groups, counts, sums, MAE/RMSE, error shares and changes relative to LOCF.
- `mean_seed_comparison.csv`: 96 rows summarizing per-cell seed means, standard
  deviations, ranges and how many seeds improve. Seed spread does not estimate
  uncertainty across new subjects.
- `../mlp_v1_transients_verify_20261027.py`: independently recomputes saved-sum
  arithmetic and CSV values, checks preserved evidence, and draws the figures.
  It opens no recordings.
- `mae_change.png`, `rmse_change.png`: inspected plots. Negative percentages
  indicate lower error than LOCF; each metric uses a shared color scale.
- `independent_checks.json`: verification scope, environment and evidence hashes.
- `evidence.sha256`: checksums relative to the repository root, including scripts.

## Verification and limits

Every fresh case's overall counts/sums match the saved MLP benchmark at
rtol=atol=1e-12. All three LOCF strata match the frozen Phase-2 archive for every
case and pooled regime. Groups are disjoint, exhaustive, and identical across
predictors; sums and counts reconcile. Scalar change labels match vectorized
labels across every segment. Measurements, truth, masks and model parameters
remain unchanged.

The targeted label-isolation check on segment 82/random changes hidden truth:
labels change, while predictions/status/ages remain identical for all seven
predictors. Full hidden-poisoning, future-prefix, repeated/reset and A-B-A audits
were already run for every case in the original benchmark. This diagnostic
repeats one full replay per case and the targeted isolation check; it does not
repeat all five full audits.

The independent verifier confirms all 2,856 case identities, per-case partitions,
84 pooled strata, and all 504 main CSV rows using separate `math.fsum` formulas.
It also verifies 83 existing evidence/source checksum entries. Saved arithmetic
verification is not a second independent implementation of every predictor.

The main access log contains only canonical subject107.dat (104 opens including
before/after hashing), with zero denied attempts. The Python guard is an
application check, not an OS sandbox. Subjects 108/109 were not accessed.
Frozen inputs, checkpoints, evaluated source and earlier evidence are unchanged.
The root status document still predates these Phase-3 runs; no milestone status,
commit, or push was changed by this diagnostic.

One validation subject and synthetic loss masks do not establish generalization,
statistical significance, real transport behavior, or physical-motion causation.

Verify saved evidence without loading raw recordings:

```bash
sha256sum -c runs/mlp_v1_transients_20261027/evidence.sha256
```

For a fresh replay, copy both scripts to a new version and change their output
directory constants together; preserve this experiment and its frozen policy.
Use `OPENBLAS_NUM_THREADS=1 python3 -B` and the environment recorded in
`independent_checks.json`. Do not overwrite existing output directories.

**Next experiment — pass/fail:** inspect a frozen gyro-Y/burst-5 failure before
designing a reliability rule. PASS requires reproducing its prediction from
received history and reconciling its error with the saved subgroup totals,
without changing checkpoints, masks, thresholds or accessing held-out subjects.
Otherwise FAIL and investigate the discrepancy. This is a proposed next step.
