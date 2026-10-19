# PacketHeal Status & Decisions

**Current phase:** Phase 2 — classical causal baseline evaluation (in progress; LOCF validation benchmark completed)

**Last updated:** 2026-10-19

**Milestone:** Phase 0 and Phase 1 accepted; LOCF evaluated on all 102 eligible subject-107 validation segments under four frozen loss regimes. Other causal baselines and the rest of Phase 2 remain pending.

**Measured reconstruction results:** LOCF per-channel MAE/RMSE on 408 validation segment/regime cases; see [versioned benchmark](benchmarks/locf_validation_subject107_v1/README.md). Single validation subject, synthetic losses only; no final-test or hardware result.

**Hardware validated:** None yet for PacketHeal AI.

**Known assets:** STM32 NUCLEO-U5A5ZJ-Q, ESP32, old Jetson Nano, Ubuntu PC (MX150).

## Decisions

- **DEC-001:** Primary problem is causal online reconstruction, not offline interpolation. **Reason:** Match real-time embedded objective. The estimator must not access future samples.
- **DEC-002:** Start with publicly available 100 Hz IMU time-series benchmark (PAMAP2 Protocol) and artificial loss masks. **Reason:** Known reference measurements without special hardware. PAMAP2 itself is **not** a packet-loss dataset.
- **DEC-003:** Separate software replay milestone from live sensor validation. **Reason:** Avoid unsupported cross-sensor compatibility claims.
- **DEC-004:** Quantized STM32U5 inference comes after strong classical baselines and desktop proof-of-concept. **Reason:** Reduce wasted firmware effort.
- **DEC-005 (2026-10-16):** Use PAMAP2 **Protocol** recordings for subjects 101–109, restricted to wrist 16 g acceleration XYZ (m/s²) and wrist gyroscope XYZ (rad/s), at nominal 100 Hz; original zero-based column indices `[4, 5, 6, 10, 11, 12]`. **Reason:** Bounded six-channel reconstruction benchmark with documented units. Optional recordings are excluded from v1.
- **DEC-006 (2026-10-16):** Accept only finite six-channel measurements in contiguous, activity-consistent segments with timestamp continuity near 10 ms (tolerance 1 ms) and minimum segment length of 256 samples. Keep segments separate. **Reason:** Do not create artificial adjacency across gaps, invalid readings, or activity changes. Retain flagged high-dynamic readings pending explicit quality analysis.
- **DEC-007 (2026-10-16):** Freeze subject-disjoint split: train 101–106, validation 107, primary test 108, supplementary test 109. Keep 109 separate due to its small and atypical Protocol subset. **Reason:** Avoid subject leakage; preserve untouched test sets.
- **DEC-008 (2026-10-16):** Freeze `configs/pamap2_segments_v1.csv`, `configs/pamap2_split_v1.json`, and versioned hashes; do not silently change them. **Reason:** Reproducible benchmark definitions.
- **DEC-009 (2026-10-16):** Fit per-channel Z-score normalization **only on eligible training samples**; save constants, leave raw measurements unchanged. **Reason:** Reduce scale imbalance without test-distribution leakage.
- **DEC-010 (2026-10-16):** Generate deterministic, segment-scoped artificial loss positions using seed 42: random nominal 5%, and bursts of 2, 5, and 10 samples, protecting the first 20 samples of every segment. **Reason:** Reproducible, fair comparison of predictors under varied loss patterns. Report actual rates; burst scenarios need not be exactly 5%.
- **DEC-011 (2026-10-16):** Use a split-aware loading interface for experiments, preserve original timestamps/ground truth, and score **only hidden positions** using strictly causal predictors. **Reason:** Prevent look-ahead and partition misuse. Lower-level `dataset.py` remains callable directly; experiment code must explicitly use the split-aware entry point.
- **DEC-012 (2026-10-16):** Accept PAMAP2 preparation pipeline v1 as **Ready for Baseline Evaluation**, subject to documented synthetic-loss and sensor-quality limitations. **Reason:** Script completed, generated artifacts were reviewed, and segment/sample/loss totals agree with the preparation report. This is **not** validation of model accuracy or hardware performance.
- **DEC-013 (2026-10-19):** Preserve the LOCF subject-107 validation result as benchmark v1, with per-segment and pooled error evidence. LOCF uses only the last received vector, resets every segment/regime, and marks startup losses UNAVAILABLE; no confidence estimate or gap cutoff is claimed. **Reason:** Establish a reproducible reference for later causal predictors without changing frozen splits, masks, or retained high-dynamic measurements. Linear extrapolation still requires design approval before implementation.

## Completed milestones and evidence

### Phase 0 — scope and feasibility
- Causal online reconstruction selected, with classical baselines required before ML and embedded deployment.
- No specific live IMU or board interface/pins assumed verified.

### Phase 1 — PAMAP2 audit and preparation
- Audited 9 PAMAP2 Protocol subject files: **2,872,533 total rows**, of which **2,859,392** contained finite values for the six selected channels.
- Frozen manifest contains **1,187 eligible continuous segments** totaling **2,566,093 samples** (eligibility reflects minimum-length and segmentation rules, not a claim that all excluded data are corrupt).
- Implemented `src/packetheal/dataset.py` (validated segment/timestamp loading, selectable segment index) and `src/packetheal/splits.py` (subject-partition checks). Manual loading/negative-access checks passed.
- Ran `scripts/prepare_pamap2.py` successfully. At the end of that preparation milestone, no predictor or firmware had been executed; subsequent LOCF results are recorded in the Phase 2 section below.

| Partition | Subjects | Eligible segments | Eligible samples |
|---|---|---:|---:|
| Train | 101–106 | 834 | 1,928,693 |
| Validation | 107 | 102 | 285,488 |
| Primary test | 108 | 247 | 343,550 |
| Supplementary test | 109 | 4 | 8,362 |
| **Total** | **101–109** | **1,187** | **2,566,093** |

- Generated `data/processed/pamap2_v1/normalization.json` with six means and **sample standard deviations** from **1,928,693 training samples only**. Formula: `z = (x - mean) / std_sample`; no normalized raw files written.
- Generated `data/processed/pamap2_v1/loss_masks.csv` with **516,878 missing-position records across all regimes and partitions**. This is a count of entries across alternative scenarios, **not** unique physical packets lost in one experiment.
- Generated `data/processed/pamap2_v1/preparation_report.json` with deterministic seed, report totals, and artifact hashes.

| Partition | Random ~5% | Burst 2 | Burst 5 | Burst 10 |
|---|---:|---:|---:|---:|
| Train | 95,600 | 96,002 | 97,260 | 99,380 |
| Validation | 14,176 | 14,226 | 14,370 | 14,660 |
| Primary test | 16,942 | 17,064 | 17,435 | 18,080 |
| Supplementary test | 415 | 418 | 420 | 430 |

**Provenance / frozen artifact references**
- Split config SHA-256: `561577438669aa6f19eb14c898c1cd217c9cd5cff49385239d7c5c9107ba0f00`
- Segment manifest SHA-256: `4ee69d67b248856ea59355b31525db4f4690b2d2a7e5fc5a5f1bd2e959a213ca`
- Normalization JSON SHA-256: `4f2102cf92de523065bbf64532ffb67f8dc4c138db9f4ca48edbcb83eec1b504`
- Loss masks CSV SHA-256: `0c1129cdf75f8eae4f451d5027d32c523d5cf755dde3044a241770d24d5792a3`
- Seed: **42**; warm-up samples per segment: **20**; nominal loss rate: **5%**.

### Phase 2 — LOCF validation reference (2026-10-17; documented 2026-10-19)

- Implemented streaming LOCF, frozen-mask readers, causal replay, hidden-only per-channel metrics, and single-segment/batch runners. Evaluated source files match commit `8c5ca42050aa2fa2108b94687b97982d6154f671`.
- Evaluated validation subject **107** only: **102 segments / 285,488 eligible samples / 408 segment-regime cases**. All cases replayed twice with identical outputs and metrics, received measurements preserved, and frozen artifact/recording hashes unchanged.
- All hidden positions were scored, with **zero unavailable outputs** under the protected warm-up: random **14,176**, burst 2 **14,226**, burst 5 **14,370**, burst 10 **14,660**. These counts are across alternative regimes, not one combined loss trace.
- **29 automated tests passed**, covering causal input isolation, startup/reset, consecutive/long losses, ownership, mask mapping and split enforcement, metric arithmetic, and sample-weighted pooling. Independent checks reconciled pooled sums and the first-segment smoke result.
- Pooled MAE increased on every channel across the four regimes; RMSE was not monotonic for every channel. Masks differ between regimes, and a few high-error recording sections contribute strongly to squared error. This does not establish that longer gaps are easier.
- Inspected two acceleration-Y events against raw rows and the batch replay fingerprints. Segment 86/burst 5: error **186.1849 m/s²** at original row **261463**; segment 77/burst 2: error **180.5997 m/s²** at original row **248184**. Both correctly held a large received value while the recorded signal changed sharply. Physical motion versus sensor/data artifact remains unresolved; no values were removed or clipped.
- Versioned summary, full JSON, pooled CSV, diagnostic figures/traces, hashes, and reproduction instructions: [LOCF validation benchmark v1](benchmarks/locf_validation_subject107_v1/README.md). The historical report retains the pre-commit HEAD and untracked-file state; its six evaluated source hashes were verified against the subsequent LOCF commit.
- Primary and supplementary test subjects **108/109 remain held out**. No other predictor, ML model, firmware, or resource benchmark was run. This records completion of the LOCF validation reference, not completion of Phase 2.

## Known limitations and unresolved decisions

- **Not yet benchmarked:** Causal extrapolation, Kalman-type, and ML predictors; uncertainty calibration, abstention, inference latency, memory, power, and deadline behavior. LOCF evidence is limited to one validation subject and one frozen seed per regime, with no cross-subject or multi-seed confidence intervals.
- **Synthetic vs. real transport:** Current masks represent deliberately hidden sample positions, not UART timing, arrival-order, jitter, corrupt frames, or hardware dropouts.
- **Sensor quality:** Finite and timestamp-contiguous does not guarantee physically plausible IMU values. Unusual high-dynamic readings remain. Two error traces were inspected, but systematic transient, per-activity, normalized-error, and loss-age analyses are pending; do not silently remove values or infer sensor faults from the plots alone.
- **Fairness:** Compare models on identical masks within each regime, and document actual masked fraction; across regimes, masked fractions differ modestly.
- **Data leakage:** Fit any learned preprocessing, hyperparameters, or models on training/validation only. Primary and supplementary test partitions remain held out until the final protocol is frozen. Every prediction must use only information available at prediction time; estimates must never be labeled as measured data.
- **Deployment unresolved:** Actual available live IMU, sensor sampling rate, UART mapping/pins, toolchain/ST AI version, inference deadline, tolerated burst length, uncertainty mechanism, and deterministic fallback policy.
- **Reproducibility limit:** Original benchmark records Python 3.12.3 and NumPy 2.4.4 plus source/data hashes; a full historical environment lock was not captured. Minimal reproduction pins are archived with the benchmark and explicitly distinguished from a full lock.

## Next task — Phase 2: classical causal baselines

Review and approve the **causal linear-extrapolation design** before implementation, including startup fallback, spacing between received samples, consecutive-loss behavior, and a slope-limiting policy defined from training information. Offer the learner the core implementation first unless explicitly delegated. Then compare against the archived LOCF reference using split-aware loading, identical frozen masks, original physical units, and consistent resets. Systematic transient analysis remains pending. Do not train an ML model or develop firmware in this milestone.

**Next experiment — pass/fail:** After design approval, test causal extrapolation on a synthetic constant-speed ramp, startup, consecutive losses, resets, and hidden/future perturbations. **PASS** only if predictions match the approved slope policy, provenance and causal boundaries remain correct, and repeated runs agree; otherwise **FAIL** and fix the implementation before the validation comparison. Linear extrapolation has not been implemented yet.

## Milestone log

| Date | Milestone | Evidence | Decision / next step |
|---|---|---|---|
| 2026-10-14 | Initial project definition | Project charter and initial status document | DEC-001–004; audit public benchmark |
| 2026-10-16 | PAMAP2 audit and subject splits | `audit.json`, `summary.csv`, `segments.csv`, frozen manifest/config | DEC-005–008; enforce eligible segments and subject separation |
| 2026-10-16 | Preparation pipeline v1 | Successful terminal output; `normalization.json`, `loss_masks.csv`, `preparation_report.json` | DEC-009–012; Phase 1 accepted, begin Phase 2 LOCF |
| 2026-10-17 | LOCF subject-107 validation | 102 segments, four regimes, 408 repeated cases; 29 tests; evaluated source committed as `8c5ca42` | LOCF comparison reference established; Phase 2 remains in progress |
| 2026-10-19 | LOCF benchmark archive and second diagnostic | Versioned benchmark JSON/CSV, two error figures/traces, artifact hashes, and summary | DEC-013; design causal linear extrapolation next, pending approval |
