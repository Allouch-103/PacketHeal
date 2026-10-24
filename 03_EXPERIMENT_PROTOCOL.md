# Experiment Protocol v0.1

## Experiment identifiers and reproducibility
Record git commit, Python environment lock, seed, model config, dataset version/hash, split ID, dropout generator version, clock/sample rate, and evaluation date. Store all measured results in CSV/JSON.

## Candidate benchmark
UCI PAMAP2 (100 Hz IMUs, nine subjects, multiple activities). Select channels conservatively. Verify exact raw-file channel indexing and invalid values from readme, do not assume data are gap-free. Local IMU differs from PAMAP2 Colibri units and calibration.

## Split strategy
Use subject-wise or session-wise disjoint splits. Never mix overlapping windows from one continuous segment across splits. Fit all transformations on training data only. Preserve an untouched test set.

## Loss generation
A: isolated random loss at several rates.
B: contiguous bursts of 2, 5, and 10 samples, with explicit lengths adjustable later.
C: longer gaps beyond designed model horizon.
D: variable/lossy arrival-time traces (later hardware).
Track loss rate, burst length, time since last real observation, and any timestamp irregularity.

## Model baselines
1. Last received value (LOCF).
2. Past-only linear extrapolation (limited slope; stable clipping policy).
3. Causal alpha-beta or Kalman-type filter fitted only on training data.
4. Small causal neural model.
5. Neural model with predictive uncertainty + abstention, if justified.

## Metrics
Per channel and activity/segment: MAE, RMSE, normalized errors, transient-segment errors; overall and burst-conditional performance; prediction interval coverage and width; selective risk vs coverage; runtime p50/p95/worst observed latency; total model Flash/RAM, worst observed missed deadline; transport packet loss, drop detection, buffer overflows.

## Fairness
Compute metrics only for missing positions with known ground truth. Apply identical masks to every model. Report raw baseline performance and confidence intervals across held-out subjects/recordings rather than only one aggregate number.

## Sanity checks
- Predict zero change should be a baseline.
- Verify no future data flows into online model features or normalization.
- Check that imputed samples fed back as inputs do not mask confidence degradation.
- Test reset, startup, first samples, long losses, timestamp wrap, and out-of-order messages.

## Results claims
Write exact measured values only after tests. Distinguish benchmark replay, local real-sensor recordings, and device-on-target metrics.

## Phase 2 v1 — desktop validation acceptance addendum (approved 2026-10-24)

This addendum defines the initial desktop classical-baseline review scope. Broader transport, device, uncertainty, long-outage accuracy, cross-subject generalization and final-test targets remain separate and unverified.

**Phase 2 acceptance (2026-10-24):** The learner explicitly accepted this bounded desktop milestone and marked Phase 2 finished. Evidence: [validation benchmark](benchmarks/phase2_validation_subject107_v1/README.md), 144 software tests, 408 segment/regime cases and 1,224 algorithm cases, preserved in completed-work commit `367ccb8`. The archived evidence and the scope/limitations below remain unchanged. Phase 3 has not started; NN design approval and implementation/training approval remain separate steps.

### Frozen experiment and predictor contract

- Validation **107 only**, all **102 segments / 285,488 samples**, four canonical masks `random_5pct`, `burst_2_5pct`, `burst_5_5pct`, `burst_10_5pct`: **408 segment/regime / 1,224 algorithm cases**. Do not open recordings **108/109**, including for diagnostic selection or checksum scans.
- Preserve frozen split/config, manifest, preparation/normalization and masks/hashes. Use split-aware loading. Keep existing seed-42 split/subject/segment/regime namespaces and saved masks; do not regenerate. Protect the first 20 samples; losses hide whole six-channel vectors.
- Wrist acceleration XYZ (m/s²) and gyro XYZ (rad/s), raw columns `[4,5,6,10,11,12]`, nominal 100 Hz. Predictors advance nominal sample indices; transport timing is not modeled.
- Reset per segment/regime. Pass only the current received vector or `None`; startup is UNAVAILABLE; received outputs are original measurements. Reconstructed outputs never become received measurements or trigger correction. Account for elapsed received-index spacing across gaps.
- Compare LOCF, capped causal linear extrapolation and fixed-gain alpha–beta with frozen train-only settings. Cap hash: `316cc1f1d0e95446329f1f689512703a146f6d07ad00bf463649ad705e13f502`; gain hash: `9eb9a8c44d7870d7d40d31dd6fe0e0a7cac876c38b2f4515fe81b99e690d4395`. All fitting/selection uses subjects **101–106 only**; no refitting or validation retuning in closure.
- Selected `(alpha,beta)=(1,0)` equals LOCF. Report all evaluated family names and the equivalence; do not present three distinct selected behaviors or substitute illustrative gains after validation.

### Overall and diagnostic metrics

- Score only hidden reconstructed positions, reporting unavailable counts/coverage. Pool per-channel absolute/squared sums and counts: MAE=`sum_abs/N`, RMSE=`sqrt(sum_sq/N)`; never average segment means/RMSEs. Preserve per-segment records and physical units; do not silently combine acceleration/gyro units.
- Define pointwise rapid-recorded-change flags by `abs(x[t,c]-x[t-1,c]) > frozen_training_q99[c]`. Equality is not flagged; first row False; no cross-segment differences or new fitting. Reusing cap constants is an explicit diagnostic policy, not proof of physical motion or faults.
- Complete hidden truth may define labels only for post-hoc grouping, never predictor features. Report flagged/complementary other strata with channel-specific hidden/scored/unavailable counts, MAE/RMSE, sums, coverage and fractions of total squared error. Empty strata have null/blank metrics. `Other` can include steady fast motion and post-transient errors.
- Report actual run histograms and errors by loss age within each regime. Current random runs span 1–4; fixed bursts span 2/5/10. Every age within a fixed-length burst uses the same gap cohort. Run length is an offline label, not future information supplied to prediction. Cross-regime masks differ and do not isolate a controlled counterfactual effect of length.
- Preserve unusual high-dynamic readings and distinguish full-population group summaries from targeted failure traces. Per-activity aggregation, transient-event windows, normalized validation metrics, additional seeds and long-outage accuracy remain unperformed.

### Acceptance and reproducibility

- Require unique complete cases and exact original archived replay fingerprints/overall metrics for all three algorithms. Require all hidden positions scored under protected warm-up; test startup unavailability separately.
- Apply identical same-object repeat, A→B→A resets, array/parameter preservation, mask-derived ages, hidden-truth poisoning and future-value/mask prefix checks to every predictor. Synthetic tests cover every cutoff on small traces and nonzero filter gains; full-data future checks compare the earlier half.
- Reconcile transient/complement and age partitions with full per-channel counts exactly and error sums at rtol=atol=1e-12. Require equal fixed-burst gap-cohort counts across ages.
- Independently verify saved metrics with `math.fsum` and reconcile CSV values. Check direct past-received-index formulas/scalar labels on preselected segments **0 and 95 under four regimes**; record that this is eight direct cases, not a second complete predictor implementation.
- Record Git state, Python/NumPy/plot versions, policy/parameter/input/recording/source hashes, test/access logs and minimal pins. Verify inputs/sources before/after; refuse existing outputs and preserve old archives. Permit only canonical subject107 raw opens through the Python guard, explicitly not an OS sandbox.
- Determinism concerns replay values/statuses/ages and numeric results in the recorded environment. Timestamps, paths and Git metadata may differ; whole JSON bytes need not match. Cross-platform bitwise behavior and a complete environment lock remain unverified.
- One validation subject/seed per regime does not support significance, subject-level confidence intervals or final-test generalization claims. Individual samples and channel/regime cells are not independent replicates. Project-wide targets not measured here remain pending.
- Correctness does not require improved accuracy. Preserve negative results. Learner review accepts or revises this bounded desktop milestone; no automatic Phase 3 transition or NN training. A later NN needs a separately approved causal input/training/evaluation design and unchanged comparison masks.
