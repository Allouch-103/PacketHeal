# Gyro-Y burst-5 failure inspection — 2026-10-27

**Reproduction PASS:** the selected residual forecast matches its saved checkpoint,
received-only input history, network calculations and original validation errors.
This trace demonstrates a large forecast error with the intended data flow.
No training, model changes, clipping, confidence rule or phase acceptance occurred.

## Selection and scope

The prior full diagnostic found rapid-change gyro-Y errors worse than LOCF during
five-sample bursts. Before opening any recording, `policy.json` fixed this selector:

1. Largest residual-seed RMSE increase in that subgroup: **seed 0, +44.15%**.
2. Segment with the largest flagged excess squared-error sum: **segment 92**.
3. Flagged hidden position with the largest excess squared error within that
   segment: **original row 269427**, the fourth missing sample in its burst.

Ties use the earliest seed, segment or sample. Selection deliberately targets a
failure; it does not estimate typical performance. The other two residual seeds
are inspected on the same unchanged burst, rather than substituted for seed 0.
Only subject 107 was opened. No held-out recording or training recording was read.

## What happened

The last received gyro-Y value, at row **269423**, was **0.186639 rad/s**.
The next five samples were hidden by the existing burst mask. The receiver held
that same real vector as its anchor throughout the gap.

| Loss age | Hidden recorded gyro Y | LOCF | Residual seed 0 | Residual seed 1 | Residual seed 2 |
|---|---:|---:|---:|---:|---:|
| 1 | -0.102485 | 0.186639 | -1.420419 | -0.598913 | -0.030318 |
| 2 | 0.349034 | 0.186639 | 1.500930 | 0.132397 | -0.339647 |
| 3 | -0.133842 | 0.186639 | 4.119381 | 2.299024 | 2.316438 |
| **4** | **0.472304** | **0.186639** | **5.278937** | **3.316673** | **3.503581** |
| 5 | 1.698780 | 0.186639 | 4.832566 | 2.024425 | 3.297996 |

All table values except loss age are **rad/s**. Recorded hidden values are used
only for offline scoring. None was supplied to the predictors.

At age 4, seed 0 computes:

```text
normalized gyro-Y correction = 6.179725 (float32 output)
physical correction         = normalized correction × training gyro-Y std
                            = +5.092298 rad/s
prediction                  = last real anchor + physical correction
                            = 0.186639 + 5.092298 = 5.278937 rad/s
correction needed afterward = 0.472304 - 0.186639 = +0.285665 rad/s
MLP absolute error          = 4.806633 rad/s
LOCF absolute error         = 0.285665 rad/s
```

The network predicts a positive correction that is much too large. All three
residual seeds overshoot at this point. Their earlier predictions remain outside
the observation history: each missing step instead holds the last received vector,
marks the slot as missing, shifts the window, and advances loss age.

**Observed input worth investigating:** acceleration Y in the last received
vector jumps from **5.31869 to 130.207 m/s²**. The latter is **25.73 training
standard deviations above the training mean**. At age 4, that same received
vector appears in one received slot and three subsequent held slots, with the
correct received/missing flags. This large cross-channel input is a plausible
contributor to the gyro-Y output, but no feature-ablation experiment has been run.
It is not proof of sensor failure, physical motion, training-range violation,
or the cause of the forecast. Raw data and preprocessing remain unchanged.

The offline gyro-Y rapid-change label at age 4 uses the preceding recorded sample:
`abs(0.472304 - (-0.133842)) = 0.606146 > 0.41480942`. The preceding value was
also hidden. This explains why the label is valid for analysis but unavailable
as an online fallback trigger. It is different from displacement from the anchor.

## Why it matters to RMSE

The entire rapid-change gyro-Y/burst-5 subgroup contains **49 positions**.
LOCF squared-error sum is **48.993916**; seed 0 gives **101.811082**: a net
increase of **52.817166** in (rad/s)². RMSE is `sqrt(sum_squared_error / count)`.

- This one sample contributes **23.022118**, or **43.59%** of that net increase.
- The four flagged positions in this five-sample burst contribute **49.843853**,
  or **94.37%** of the net increase.
- All eight flagged positions in segment 92 contribute **47.450950**: errors
  elsewhere in the segment partly offset this burst's increase.

These percentages concern a **net difference between predictors**, not shares
of the model's total squared error. Improvements and regressions can cancel.
They apply to seed 0, not automatically to the other seeds or the complete model
benchmark. The better overall MAE and the subgroup regression remain preserved.

## Files and data flow

```text
frozen checkpoints + normalization + manifest + mask + subject107
    → diagnostic runner: reset/step(received vector or None)
    → complete-segment audit + comparison with saved benchmark/strata
    → independent history and layer calculations for the selected burst
    → event.json / burst.csv / history.csv / features.npz / trace.csv
    → independent saved-arithmetic verifier
    → independent_checks.json
```

- `../mlp_v1_gyroy_burst5_failure_20261027.py`: targeted replay and plot runner.
  Refuses an existing output directory and writes its selection policy before
  recording access. It uses the existing predictor and audit implementations.
- `policy.json`: exact selection and evidence/source hashes.
- `event.json`: predictions, corrections, errors, checks, subgroup totals,
  environment, checkpoint hashes and recording-access log.
- `burst.csv`: five missing steps, three residual seeds and LOCF in rad/s.
- `history.csv`: 20 slots for each prediction, **100 rows**. Records which
  original received row supplied each held value, all six channels and Z-scores.
- `features.npz`: `features` is **float32 [5,161]**; each row contains 20×8
  history entries plus age/10. It also saves float32 hidden activations [5,32]
  and [5,16], float32 normalized outputs [5,6], float64 physical corrections
  [5,6], and float64 six-channel anchor/normalization constants.
- `trace.csv`, `failure.png`: nearby received and hidden samples, predictions
  and errors. Gray shading marks the selected burst; the red line marks age 4.
- `segment_contributions.csv`: all 102 segments' flagged counts and sums for
  LOCF/primary seed, making the selection and concentration calculations auditable.
- `../mlp_v1_gyroy_burst5_verify_20261027.py`: independent saved-result arithmetic
  and CSV/feature checks. Opens no recordings and refuses existing check output.
- `independent_checks.json`: verification results and precise scope.
- `evidence.sha256`: checksums of this evidence and both scripts, relative to
  the repository root.

## Verified and unverified

All four complete-segment replays match all six channels of saved overall metrics
and all three saved strata. Repeated/reset, A-B-A reset, hidden-truth poisoning,
future-half prefix, input/measurement preservation and loss-age checks pass.
An additional future perturbation after the selected burst leaves its prefix
unchanged. Every one of the **15 model/step combinations** matches independently
rebuilt received-only features, three affine layers, ReLUs and denormalization.
Anchors remain real and fixed across the burst. Seven original source rows match
the loader exactly. Subject107 hashes match the original evidence before/after;
the access log contains four canonical subject107 opens and zero denied attempts.
The guard is an application check, not an OS sandbox.

The separate verifier reconciles 102 segment contributions, subgroup formulas,
five burst rows, 100 history rows, feature dtypes/shapes, correction arithmetic
and trace values with `math.fsum`/scalar calculations at rtol=atol=1e-12.
It verifies saved arithmetic, not another independent full raw predictor replay.
Frozen inputs, prior evidence, sources and checkpoints pass before/after checks.

This is a targeted validation failure under synthetic loss. It does not establish
cross-subject generalization, calibrated uncertainty, feature causation or physical
fault diagnosis. No existing source, frozen artifact, status document, commit or
milestone was changed. The root status document still predates the Phase-3 runs.

Verify saved evidence without loading recordings:

```bash
sha256sum -c runs/mlp_v1_gyroy_burst5_failure_20261027/evidence.sha256
```

**Next experiment — pass/fail:** run a clearly labeled offline input-sensitivity
check on copies of this burst's features, comparing the original input with a
predeclared change to the unusually large received acceleration value. PASS
requires reproducible outputs, an unchanged original replay, preserved masks and
checkpoints, and explicit separation from benchmark accuracy. Otherwise FAIL and
fix the diagnostic. A change in output would demonstrate sensitivity to the
intervention, not establish the physical cause or validate a correction policy.
This follow-up has not been run.
