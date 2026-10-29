# LOCF-on-warning comparison

**Calculation correctness PASS. Usefulness FAIL.** Switching the whole predicted
vector to LOCF when warned increases full-coverage normalized MAE for every
residual seed. Recommendation: keep the original residual predictor; do not
adopt this fixed fallback rule.

## 1. What we tested and why

```text
warned or warning unknown → choose LOCF
known and unwarned        → choose the residual MLP
```

The warning is unchanged from the previous experiment. It uses the latest two
received acceleration vectors, elapsed nominal sample steps and the frozen
training thresholds. It never sees the hidden measurement. Selection applies
to the whole six-channel vector, identically for all three seeds.

The prior experiment established that warned positions are harder. This test
asks a different question: **does LOCF perform better there?** A prediction can
have a large error while still outperforming another available predictor.

## 2. How the software calculates the result

No recording loading, model inference or training is needed. The previous
verified report already holds each model's error sums on exactly the same
warned, unwarned and unknown positions.

```text
hybrid absolute-error sum = MLP.unwarned + LOCF.warned + LOCF.unknown
hybrid squared-error sum  = MLP.unwarned + LOCF.warned + LOCF.unknown
MAE                      = absolute-error sum / scored count
RMSE                     = sqrt(squared-error sum / scored count)
```

The two sum lines use their corresponding fields; absolute and squared errors
are never mixed. Counts are added before computing metrics. Group MAEs/RMSEs
are not averaged directly.

This recombination is valid because both predictors and the warning update
their state using received measurements only. Choosing a different emitted
prediction does not change later predictions. The original real-only history,
model and warning source hashes remain unchanged. This is an evaluation of
the proposed routing rule, not an implemented runtime wrapper.

The script writes its policy before computing results. It evaluates all
**408 segment/regime cases × three residual seeds = 1,224 case calculations**,
including error by loss age. These are saved-sum calculations, not new replays.

## 3. Measured result

All numbers below use subject 107 and the existing four masks. Normalized MAE
is the mean across regimes/channels of pooled physical MAE divided by each
channel's frozen training standard deviation. Lower is better.

| Residual seed | Original MLP | LOCF on warning | Change |
|---|---:|---:|---:|
| 0 | 0.132873 | 0.137456 | +3.449% |
| 1 | 0.131710 | 0.136282 | +3.472% |
| 2 | 0.132516 | 0.137028 | +3.405% |
| Mean across seeds | **0.132366** | **0.136922** | **+3.442%** |

Coverage stays **100%** on the protected-warmup benchmark. Per seed, 1,102 of
57,432 hidden scenario positions use LOCF; 56,330 use the MLP. The four masks
are alternative scenarios, so these are not unique physical lost packets.
There are no unknown hidden positions in this benchmark. The specified unknown
fallback inherits LOCF availability; missing startup before any measurement
would still be unavailable in an actual stream.

The predeclared usefulness criterion required lower normalized MAE for all
three seeds. It fails for all three. The result is preserved without changing
thresholds, selecting favorable channels or substituting a better seed.

| Seed | MAE cells improved / 24 | RMSE cells improved / 24 | Largest RMSE increase |
|---|---:|---:|---:|
| 0 | 4 | 6 | +58.375% |
| 1 | 7 | 8 | +57.083% |
| 2 | 5 | 2 | +57.060% |

The largest regressions occur in **acceleration Y / burst5**. For seed 0,
RMSE rises **3.130330 → 4.957657 m/s²**. There are also improvements: gyro-Y /
burst5 seed-0 RMSE falls **0.188026 → 0.176516 rad/s**. Fixing a selected
gyro-Y weakness therefore does not make the whole-vector rule beneficial.
Every channel/regime and age result is preserved in the CSV.

## 4. Files and checks

- `../mlp_v1_warning_fallback_20261027.py`: verifies prior evidence, declares
  the fixed rule, checks a synthetic routing example, combines saved sums and
  exports full/per-age metrics.
- `../mlp_v1_warning_fallback_verify_20261027.py`: independent arithmetic,
  routing/count checks, CSV verification and figure generation.
- `policy.json`: exact rule, acceptance criterion, assumptions and hashes.
- `synthetic_check.json`: direct pointwise selection on three made-up positions
  matches summed errors in all six channels; mismatched group counts are rejected.
- `report.json`: 1,224 case records, per-age and pooled metrics, seed summaries,
  provenance and separate correctness/usefulness outcomes.
- `comparison.csv`: **450 rows**: three seeds × 25 regime/age levels × six
  channels. Includes original/hybrid/LOCF physical MAE/RMSE and all regressions.
- `fallback_comparison.png`: inspected summary figure. Positive RMSE change
  means worse; per-cell colors summarize seed means, with individual seeds in CSV.
- `independent_checks.json`: exact verification scope and result hashes.
- `evidence.sha256`: all result/script checksums relative to repository root.

The verifier checks all 1,224 case identities, **5,835 per-age case records**,
counts, full coverage, age partitions, independent `math.fsum` pooling, all
normalized objectives and all 450 CSV rows at rtol=atol=1e-12. It also checks the
identity `hybrid - original = LOCF - MLP` on the routed groups, for absolute and
squared sums in every physical channel/regime. Original warning evidence,
frozen inputs, model checkpoints and evaluated source hashes remain unchanged.

Both programs prohibit raw-file opens and record zero attempts. There were zero
model inferences. Subjects 108/109 were not accessed. No production predictor,
warning threshold, dataset, mask, normalization, status document, milestone,
commit or push was changed. Original predictor causality/reset checks are prior
evidence, not fresh runtime hybrid tests. Cross-subject generalization, runtime
hybrid implementation, transport and hardware behavior remain unverified.

Both scripts refuse existing outputs. Recheck saved evidence without recordings:

```bash
sha256sum -c runs/mlp_v1_warning_fallback_20261027/evidence.sha256
```

**Next experiment — pass/fail:** define whether warned predictions may be marked
UNAVAILABLE instead of substituting LOCF. PASS requires an agreed minimum retained
coverage and error criterion before implementing an operational rejection rule;
otherwise FAIL the design and retain the current outputs. Previous conditional
error/coverage measurements are development evidence, not deployed abstention.
