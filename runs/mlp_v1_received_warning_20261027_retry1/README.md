# Causal received-change warning: subject107

**Correctness PASS. Association screen PASS.** The fixed warning identifies
higher-error predictions for all three frozen residual MLP seeds. This is an
exploratory validation result, not a calibrated uncertainty or abstention policy.

## 1. What the warning does

For the last two received six-channel vectors, use acceleration XYZ only:

```text
change_per_step[c] = abs(last_acc[c] - previous_acc[c]) / elapsed_sample_steps
score             = max_c(change_per_step[c] / frozen_training_q99[c])
warning           = score > 1
```

The thresholds are the existing training-only values:
**5.229231 / 6.21074012 / 4.254664448 m/s² per nominal sample step**.
Equality is unwarned. The warning updates on each real measurement and stays
fixed during missing steps. Each segment/regime resets it. Before two received
vectors, the warning is explicitly unknown. Protected warm-up means no validation
hidden position is unknown; startup behavior is covered by synthetic tests.

**Why:** test whether available observations indicate harder predictions, before
changing any predictor. Hidden measurements never enter this warning.
The original MLP/LOCF outputs, masks and normalization remain unchanged.

The previously inspected row 269427/burst5 is warned with score **20.10844**,
using received rows 269422 and 269423. Its hidden value is not needed to warn.

## 2. What was measured

Subject 107 only: 102 segments, 285,488 samples, four frozen masks. Six MLPs and
LOCF give **2,856 algorithm cases**. Every hidden position is scored. The warning
groups are common to every predictor, including the LOCF comparison. Both overall
and loss-age metrics match the original benchmark before splitting into groups.

If known unwarned predictions alone were retained:

| Regime | Hidden positions | Warned | Unwarned retained | Retained fraction |
|---|---:|---:|---:|---:|
| Random | 14,176 | 290 | 13,886 | 97.9543% |
| Burst 2 | 14,226 | 312 | 13,914 | 97.8068% |
| Burst 5 | 14,370 | 240 | 14,130 | 98.3299% |
| Burst 10 | 14,660 | 260 | 14,400 | 98.2265% |

This is **hypothetical retention**, not a deployed rejection rule. Actual model
reconstruction coverage remains 100%. Counts across these four alternative masks
are scenario positions, not unique physical lost packets.

The primary screen was fixed before recording access: warned normalized MAE must
exceed unwarned normalized MAE for **each residual seed**. Normalized MAE is the
equal average over four regimes and six channels of pooled MAE divided by the
frozen training standard deviation. Raw errors remain separate in physical units.

| Residual seed | All normalized MAE | Warned | Unwarned | Warned / unwarned |
|---|---:|---:|---:|---:|
| 0 | 0.132873 | 0.806462 | 0.119544 | 6.746× |
| 1 | 0.131710 | 0.805997 | 0.118347 | 6.810× |
| 2 | 0.132516 | 0.809115 | 0.119108 | 6.793× |

Mean across seeds: all-position normalized MAE **0.132366**, warned **0.807192**,
unwarned **0.119000**. Retaining unwarned positions gives approximately **10.10%**
lower conditional MAE while excluding **1.67–2.19%** of outputs by regime. This
changes which predictions are retained; it does not improve the original forecasts.

**Loss-age control:** warned MAE and RMSE are higher in all **114 comparable
channel/regime/age cells for each residual seed**. There are 19 comparable
regime/age groups × six channels. Random ages 3 and 4 have zero warned positions
and are excluded from this comparison, not counted as wins. These cells are
descriptive comparisons, not independent statistical replicates.

Physical-unit example, gyro Y during burst5, means across residual seeds:

| Group | Positions | MAE (rad/s) | RMSE (rad/s) |
|---|---:|---:|---:|
| Warned | 240 | 0.404331 | 0.649252 |
| Unwarned | 14,130 | 0.092174 | 0.159251 |

The signal identifies difficulty, not necessarily where LOCF wins. For example,
gyro-Y/burst10 warned MAE is **0.461352 rad/s** for residual seed mean versus
**0.520764** for LOCF. A fallback decision therefore needs its own comparison.

## 3. What each file does

- `../mlp_v1_received_warning_20261027.py`: stateful warning and independent
  past-received-index reference. Receives a real six-vector or `None`; holds no
  reconstructed values. Its state includes received acceleration, two indices,
  score, known flag and loss age.
- `../test_mlp_v1_received_warning_20261027.py`: seven synthetic software tests.
  Tests use made-up arrays, not raw recordings or training.
- `../mlp_v1_received_warning_run_20261027.py`: validation-only loading, causal
  warning audits, original model replay, grouping, pooling and metrics export.
- `../mlp_v1_received_warning_verify_20261027.py`: independently checks saved
  arithmetic, warning scores and received indices against existing masks; draws
  the figure. It opens no recordings.
- `policy.json`: exact rule, thresholds, state/reset policy, screen and hashes,
  written before raw-data access. No threshold was fitted or adjusted here.
- `software_tests.txt`: actual seven-test output.
- `warnings.csv`: 57,432 hidden-position records, with warning score, loss age,
  source received rows, their acceleration values and elapsed sample steps.
- `report.json`: per-case and pooled group/age sums, metrics, provenance,
  checkpoints and recording-access log.
- `metrics.csv`: 4,200 physical-unit rows for all predictors, groups and ages.
- `coverage.csv`: 25 regime/age count and retention rows.
- `association_summary.csv`: seven predictor summaries, including seed results
  and same-age comparison counts.
- `warning_summary.png`: inspected retention and error-association figure.
- `independent_checks.json`: verification scope and results.
- `evidence.sha256`: evidence and source checksums, relative to repository root.

Model inputs remain float32 [161] and outputs float32 [6], decoded in float64.
The warning uses float64 received accelerations and score, integer sample indices
and loss age, and Boolean known/warned flags. Scoring is a separate offline step.

## 4. Verification and limits

All seven synthetic tests pass: startup/reset, strict threshold equality,
three-axis and gyro exclusion, consecutive/long gaps, elapsed-step recovery,
ownership/invalid input, hidden-value isolation, all future cutoffs, A-B-A reset
and independent reference parity. These are seven software tests, distinct from
the 2,856 dataset replay cases; the earlier model source/tests remain unchanged.

For every validation case, the new warning matches the independent past-index
formula, hidden-truth NaN poisoning leaves it unchanged, and future value/mask
changes leave its earlier half unchanged. Original arrays remain unchanged.
Every predictor's full and per-age error sums/counts match saved evidence.
Warning partitions and age partitions reconcile. Original full predictor causal
audits remain applicable by unchanged source/checkpoint hashes; this run repeats
single model replays, not all five earlier model audits.

The independent verifier checks all case identities, every warning CSV row,
source indices against the frozen masks, scalar warning scores, `math.fsum`
pooling/partitions, all derived metrics and all 4,200 metric rows at
rtol=atol=1e-12. It is saved arithmetic verification, not a second full raw
implementation of each neural predictor. Access logs contain only canonical
subject107.dat, 104 opens including before/after hashing, and no denied opens.
The verifier opens zero recordings. The guard is an application check, not an
OS sandbox. Frozen inputs, checkpoints, original evaluated source and benchmark
report hashes match before/after. Subjects 108/109 remain untouched.

The q99 thresholds describe adjacent training changes, not a 99% correctness
probability. Average change across a received gap can miss within-gap variation;
the latest-pair rule can forget an earlier unusual value still in MLP history.
The warning does not guarantee that unwarned predictions are accurate or that
warned measurements are faulty. One validation subject and this exploratory
follow-up do not establish statistical significance or new-subject performance.

The first attempt in `../mlp_v1_received_warning_20261027/` stopped on the first
LOCF case because the runner compared lowercase status text with the project's
uppercase enum. It produced no result report. Its policy, partial files and exact
failed runner snapshot are preserved there. The corrected run uses the enum and
this fresh `retry1` output directory; the predictor implementation was unchanged.

No training, predictor modification, operational rejection rule, firmware,
milestone update, Git commit or push occurred. The root status document still
predates the Phase-3 experiments.

Verify saved evidence without reading recordings:

```bash
sha256sum -c runs/mlp_v1_received_warning_20261027_retry1/evidence.sha256
```

**Next experiment — pass/fail:** compare a fixed LOCF-on-warning fallback with
the unchanged MLP, using the saved sums and identical positions. PASS for
usefulness requires lower full-coverage normalized MAE for all three residual
seeds; otherwise FAIL and retain the original predictor. Report every physical
channel/regime MAE/RMSE change, including regressions. This fallback has not been
implemented or selected by the current association experiment.
