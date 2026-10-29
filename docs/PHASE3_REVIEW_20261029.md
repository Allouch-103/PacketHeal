# Phase 3 desktop review — 2026-10-29

**Final decision: Phase 3 VALIDATED and explicitly accepted by the learner on
2026-10-29, within the desktop prototype scope.** The learner requested final
validation, marking the phase validated, committing/pushing the final work, and
clearer explanations in future chats. Physical/application error limits remain
unset. Known abrupt-change weaknesses and all other limitations below remain.
Phase 4 belongs in the separate conversation; no quantization is continued here.
Final verification and acceptance evidence are in
[the accepted Phase 3 archive](../benchmarks/phase3_validation_accepted_20261029/README.md).

The following is the original pre-acceptance review. Its "pending" statements
record that earlier state and are superseded by the final decision above. Its
exact original bytes are preserved in the acceptance archive, alongside the
unchanged original JSON, CSV, checksums and evaluated source.

**Trained-model/consumer integration PASS. Original development screen PASS.
Formal Phase 3 acceptance PENDING learner review.**

The learner explicitly chose **desktop prototype; keep physical error limits
unset**. This review records completed work and current limits. It does not
authorize INT8 conversion, firmware, a final-test evaluation, further training,
or application use. Phase 2 remains formally accepted.

## 1. Work completed today

The existing work was first committed and pushed to `origin/main` as `8e1f4b0`.
The preservation checkpoint contains the models, evaluated source, original
reports and negative results. Fresh checks passed: **197 software tests**,
**105 checksum entries across 12 manifests**, all six trained checkpoints and
frozen preparation/classical inputs. Actual recording open attempts: zero.

Then `scripts/check_trained_consumer.py` connected **each of the three saved
residual checkpoints** to the existing abstention wrapper and `StatusConsumer`:

```text
artificial received six-vector or None
    -> trained MLP with frozen training normalization
    -> fixed received-change warning / desktop rejection
    -> MEASURED, RECONSTRUCTED, or UNAVAILABLE
    -> consumer preserves provenance and gap indices
```

Each seed passed the same 40-slot controlled example: **23 measurements,
13 estimates, four gaps**. Tests within the integration runner compare accepted
values exactly with the original predictor, preserve real values/loss ages,
check startup and horizon 10/11, preserve real-only anchors, exercise A-B-A reset
of both components, and change future values/missingness at cutoffs 20/24/37.
Saved parameters and prior evidence remain unchanged. The consumer's status-first
handling, unreadable unavailable payload, finite gap placeholders and ownership
tests are also covered by the 197-test preservation checkpoint.

The 40 slots are artificial inputs chosen to exercise software branches, using
real frozen normalization/thresholds and trained weights. They are not new
reconstruction-accuracy evidence or a validation of long-outage accuracy.

Integration evidence: [integration.json](../runs/mlp_v1_trained_consumer_20261029/integration.json).
Review generator: `scripts/review_phase3.py`; it reads pinned saved summaries and
CSVs, opens no raw recordings, and writes the machine-readable review and tables.

## 2. Existing reconstruction evidence

Training used subjects 101–106. Validation used only subject 107: 102 segments,
four unchanged masks, six MLPs plus LOCF, **2,856 algorithm cases**. Three seeds
represent training variation, not independent subjects.

| Predictor | Mean normalized MAE | Change versus full-coverage LOCF |
|---|---:|---:|
| LOCF = selected alpha–beta | 0.156898 | Reference |
| Direct MLP, three seeds | 0.135532 | -13.62% |
| Residual MLP, three seeds | 0.132366 | -15.64% |

The original declared engineering screen passes for both families: at least 5%
mean normalized-MAE reduction, at least two improving seeds, no channel/regime
RMSE increase above 10% in any seed, and full hidden-position coverage. This
screen is a development criterion, not statistical significance or phase acceptance.

Keep the residual family as the primary desktop candidate; retain every seed.
Do not choose seed 1 merely because it has the lowest validation objective.
The direct family remains a useful comparison. Models, warning thresholds and
preprocessing have not been retuned during this review.

## 3. Rejection limits: two choices to review

Coverage below counts **missing positions only**. Received measurements are
always retained. Masks are alternative scenarios, not unique physical packets.

| Regime | Estimates retained | Marked unavailable |
|---|---:|---:|
| Random | 97.9543% | 2.0457% |
| Burst 2 | 97.8068% | 2.1932% |
| Burst 5 | 98.3299% | 1.6701% |
| Burst 10 | 98.2265% | 1.7735% |

- **Strict 98% retained / at most 2% rejected in every regime:** current fixed
  rule fails random and burst-2. Do not describe it as meeting an exact 2% cap.
- **97.5% retained / at most 2.5% rejected in every regime:** current observations
  satisfy this possible desktop target. It was considered after viewing results;
  it is not a predeclared success criterion or a guarantee on new data.

Recommendation: retain the fixed prototype, report the actual 1.67–2.19% rejection
range, and use 97.5% only as a **proposed future desktop review target**. No new
coverage requirement is adopted or enforced by this review. Application minimum
coverage remains unset. Changing the warning threshold to obtain a chosen number
after seeing this validation subject would require a new development protocol.

The accepted estimates have about **10.1% lower normalized MAE** than the original
full set. This is a conditional error on fewer positions, not improved numerical
predictions. On identical accepted positions, residual MAE **and** RMSE are below
LOCF in **72/72 channel/regime/seed cells**. This matched comparison is distinct
from comparing accepted MLP error with full-coverage LOCF error.

The original abstention evidence uses saved sums for all 1,224 residual cases,
plus 24 prior fresh runtime cases on preselected segments 0/92. Today's trained
consumer test does not repeat a full PAMAP2 validation replay.

## 4. Physical error review

Physical-unit limits remain **unset**, as requested. Observed errors are evidence
to inspect, not tolerances inferred from that evidence. The following are the
largest **pooled** accepted-error metrics across four regimes and three seeds
for each channel; all occur in burst-10. They are not maximum sample errors or
bounds, and the MAE/RMSE maxima can belong to different seeds.

| Channel | Unit | Largest pooled MAE | Largest pooled RMSE |
|---|---|---:|---:|
| Acceleration X | m/s² | 0.9741 | 1.9270 |
| Acceleration Y | m/s² | 1.0906 | 3.3330 |
| Acceleration Z | m/s² | 0.7096 | 1.4046 |
| Gyro X | rad/s | 0.2970 | 0.5906 |
| Gyro Y | rad/s | 0.1480 | 0.2615 |
| Gyro Z | rad/s | 0.1985 | 0.4060 |

These values cannot establish whether a later application can tolerate the
estimates. That requires its own per-channel tolerances, coverage requirement
and behavior during gaps. No numerical per-sample guarantee is claimed.

## 5. Problems preserved

- Large individual errors remain possible. On the original rapid-change gyro-Y /
  burst-5 stratum, 49 positions have mean-seed RMSE **1.192793 rad/s**, versus
  LOCF **0.999938 rad/s**: **+19.29%**. All residual seeds regress there. This
  does not contradict the passing overall screen, which uses full-population
  channel/regime aggregates. The specific stratum after rejection has not been
  evaluated here; do not claim rejection solved that weakness.
- The warning is a received-change heuristic, not calibrated confidence. It
  can miss variation hidden inside gaps and can forget earlier changes still
  present in model history. This is one operating point, not a full risk/coverage curve.
- Substituting LOCF when warned increased mean normalized MAE by about **3.44%**;
  that whole-vector fallback remains rejected.
- Only one validation subject has been examined. Subjects 108/109 remain
  untouched. Repeated analysis of 107 creates development-selection risk; these
  diagnostic choices do not provide new independent generalization evidence.
- INT8 numerical parity, device resources/latency, real transport timing and
  application suitability remain unverified. Desktop NaN sentinels require
  explicit validity metadata in any future integer interface.

## 6. Review outcome and continuity

Desktop integration is verified and the original accuracy screen passes.
Prototype scope and unset physical limits are agreed. Phase 3 is **ready for
learner review**, with formal acceptance pending; the broader project is not
complete. No new training, raw recording access, mask generation or hardware
work occurred today. The root status now records the completed Phase 3 work
instead of the stale statement that Phase 3 had not started. The Phase 2
acceptance addendum remains historical; the current status supersedes its old
startup wording.

Machine-readable review and tables: [review.json](../runs/mlp_v1_phase3_review_20261029/review.json),
[coverage.csv](../runs/mlp_v1_phase3_review_20261029/coverage.csv),
[physical-error summary](../runs/mlp_v1_phase3_review_20261029/physical_error_summary.csv).
New integration/review results are local artifacts after the preservation push;
they are not part of commit `8e1f4b0`.

**Next experiment — pass/fail:** learner review of this bounded desktop milestone.
PASS requires explicit acceptance of the prototype-only scope, observed rejection
trade-off and remaining limitations; otherwise FAIL closure and record required
changes. Do not open held-out tests, retrain, quantize or develop firmware merely
because those steps appear on the roadmap.
