# Exact proposed document updates

Prepared **2026-10-24 after benchmark revalidation**. Neither original document is edited. This is a proposal, not milestone acceptance or NN authorization.

Base files:

- `05_STATUS_AND_DECISIONS_UPDATED.md`: SHA-256 `a2ab6b22c9ea7dcb4926b7f467eb0bce6415cf8a9f208732c3442a3520d0b7fe`
- `03_EXPERIMENT_PROTOCOL.md`: SHA-256 `3389a40141cf547342b05a5744dd775a68cf071664107d7d38347c8a26a89132`

[proposed_document_updates.patch](proposed_document_updates.patch) contains exact status replacements/inserts and the exact protocol addendum below. Dry-run from repository root:

```bash
git apply --check benchmarks/phase2_validation_subject107_v1/proposed_document_updates.patch
```

Apply only after learner approval. The existing protocol file is untracked; this proposal does not stage/commit it.

## Status changes

1. Keep Phase 2 active, mark the requested desktop analysis verified and awaiting learner acceptance, and date the status 2026-10-24.
2. Record 144 tests, 408 cases / 1,224 algorithm cases, all archived replay matches, uniform causal/reset checks, independent verification and subject107-only access.
3. Add DEC-020 for post-hoc recorded-change/age definitions and DEC-021 for verified evidence and bounded acceptance.
4. Record actual transient and within-burst results, report hash, new evidence links and remaining limitations; correct obsolete pending-analysis text.
5. Make review/acceptance the next task. No phase advancement, new fitting or NN training is authorized.

## Exact protocol addition

The existing protocol remains intact; append this addendum:

## Phase 2 v1 — desktop validation acceptance addendum (proposed 2026-10-24)

This addendum defines the initial desktop classical-baseline review scope. Broader transport, device, uncertainty, long-outage accuracy, cross-subject generalization and final-test targets remain separate and unverified.

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
