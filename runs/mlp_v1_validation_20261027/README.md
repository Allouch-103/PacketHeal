# First PAMAP2 causal MLP benchmark — 2026-10-27

Training and validation completed after the learner explicitly requested
`train`. The previously stopped 2026-10-26 preparation attempt remains separate.
No model was retuned or selected by validation performance.

## Experiment

- Training: subjects 101-106, 834 segments, 3,336 segment/regime cases,
  **388,242 examples** across the four existing frozen masks.
- Six models: direct and LOCF-residual prediction, seeds 0/1/2, fixed 20 epochs,
  Adam learning rate 0.001, batch 256 (64 draws per regime, with replacement).
- Network: 161 -> 32 ReLU -> 16 ReLU -> 6 linear outputs; 5,814 parameters.
  Inputs are 20 observation-only history slots plus current loss age. Static
  normalization is the frozen training-only artifact. Reconstructions never
  become received history. MLP horizon is 10; initialized cases outside scope
  use LOCF. No calibrated uncertainty or abstention is implemented.
- Validation: subject 107, 102 segments / 285,488 samples / 408 segment-regime
  cases, six MLPs plus fresh LOCF = **2,856 algorithm cases**. All hidden positions
  scored: 14,176 / 14,226 / 14,370 / 14,660 across the four alternative regimes.
- Python 3.12.3, NumPy 2.4.4, one OpenBLAS/OMP thread. Source is uncommitted;
  recorded Git HEAD is `d9555f5`, with exact evaluated source hashes and copies
  preserved in the training folder. No commit or push was performed.

## Results

The summary objective is the mean over regimes/channels of pooled physical-unit
MAE divided by the frozen per-channel training standard deviation. It is
dimensionless. Seed standard deviation describes optimization variation, not
uncertainty across unseen subjects.

| Predictor | Validation normalized MAE | Change from LOCF |
|---|---:|---:|
| LOCF = selected alpha-beta | 0.156898 | reference |
| Frozen linear extrapolation (archived) | 0.255905 | +63.10% |
| Direct seed 0 | 0.136465 | -13.02% |
| Direct seed 1 | 0.134995 | -13.96% |
| Direct seed 2 | 0.135137 | -13.87% |
| **Direct mean, 3 seeds** | **0.135532** | **-13.62%** |
| Residual seed 0 | 0.132873 | -15.31% |
| Residual seed 1 | 0.131710 | -16.05% |
| Residual seed 2 | 0.132516 | -15.54% |
| **Residual mean, 3 seeds** | **0.132366** | **-15.64%** |

Sample standard deviations across seeds: direct 0.000811; residual 0.000596.
Every residual seed improved MAE in 24/24 channel/regime cells. Every direct seed
improved MAE in 23/24; random-loss acceleration Z is slightly worse. The largest
RMSE increases relative to LOCF are +3.41% for direct seed 2/random gyro Y and
+0.95% for residual seed 1/random acceleration X. Preserve these regressions.

Both families pass the predeclared practical screen: >=5% mean normalized-MAE
improvement, at least two improving seeds, no >10% RMSE increase in any cell/seed,
and full coverage. This screen is not statistical significance or phase acceptance.

Physical-unit example, burst-10 acceleration Z, means across MLP seeds:

| Predictor | MAE (m/s²) | RMSE (m/s²) |
|---|---:|---:|
| LOCF | 0.781423 | 1.539781 |
| Direct | 0.737742 | 1.443814 |
| Residual | 0.732204 | 1.445614 |

The residual model is the better primary candidate by the predeclared objective.
Errors still grow through consecutive losses. No best-seed substitution, new
normalization, changed mask, or validation hyperparameter search was performed.

## Verification and reproducibility

- Full replay checks passed for every model/case: repeated/reset replay, hidden
  truth poisoned with NaNs, future value/mask prefix independence, A-B-A reset,
  original-array preservation, measured-value preservation, and loss ages.
- Fresh LOCF pooled metrics/counts match the frozen classical CSV. Linear and
  selected alpha-beta use unchanged archived metrics; their raw replays and all
  classical replay fingerprints were not repeated in this experiment.
- Training and validation raw hashes match the relevant frozen calibration/
  benchmark references and remain unchanged before/after their runs. Training
  access is restricted to 101-106; validation to 107. No 108/109 recording was
  opened. The access guard is a Python application check, not an OS sandbox.
- `independent_checks.json` verifies all case identities, saved pooled arithmetic,
  age partitions/cohorts, all 168 exported metric rows, unchanged checkpoints,
  and provenance. It uses independent saved-result arithmetic; it is not a
  second predictor implementation or another raw-data replay.
- Evaluated source/test snapshots, checkpoint hashes, installed-package versions,
  and configuration are in `../mlp_v1_train_20261027/`. The original code tests
  were 174/174 passing before training; source remained unchanged for this run.

Files:

- `validation.json`: complete per-case, per-age and pooled evidence.
- `overall.csv`: physical-unit metrics for all six MLPs and fresh LOCF.
- `frozen_classical_overall.csv`: byte-identical classical-reference table.
- `mean_seed_physical_metrics.csv`: per-channel/regime means across seeds.
- `independent_checks.json`: verification and per-seed summary.
- `mae_comparison.png`, `burst10_by_age.png`: inspected result figures.
- `evidence.sha256`: checksums, relative to the repository root.
- `../mlp_v1_validation_20261027_runner.py`: logging-only replay wrapper.
- `../mlp_v1_verify_20261027.py`: independent verifier and figure source.

Recheck saved evidence without accessing recordings:

```bash
sha256sum -c runs/mlp_v1_train_20261027/evidence.sha256
sha256sum -c runs/mlp_v1_train_20261027/source_snapshot.sha256
sha256sum -c runs/mlp_v1_validation_20261027/evidence.sha256
```

One validation subject and synthetic masks support a development benchmark only.
Cross-subject generalization, rapid-change-stratum diagnostics for the MLP,
uncertainty/abstention, INT8 parity, transport, and device measurements remain
unverified. This result does not complete the broader Phase 3 milestone.

**Next experiment — pass/fail:** review a frozen validation failure trace and the
remaining RMSE regressions before changing the model. PASS requires reconciling
the saved prediction/error with the causal history and preserving existing
settings/evidence. Otherwise FAIL and investigate the discrepancy.
