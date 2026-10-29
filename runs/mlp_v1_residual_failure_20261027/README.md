# Residual MLP failure reproduction — 2026-10-27

The learner approved inspecting a difficult residual prediction and the remaining
RMSE regression. No retraining, retuning, clipping, mask changes, preprocessing
changes, or milestone acceptance was performed.

## Selection fixed before raw replay

1. Among residual-model cells/seeds, select the largest positive RMSE ratio over
   LOCF: seed 1, random loss, acceleration X, **+0.953768%**.
2. Select the segment with largest excess squared-error sum for that cell:
   **segment 82**. Ties use the earliest segment.
3. Select the hidden position with largest excess squared error within that
   segment: original recording **row 253261**, timestamp **2538.52 s**, loss age 1.

This deliberately selects a failure. It is not representative sampling and does
not replace the complete validation benchmark. The selected segment has 678
samples and 33 hidden positions; the model improves absolute error at 15 of them
and worsens it at 18.

## Reproduced event

All values below concern wrist acceleration X in m/s².

| Quantity | Value |
|---|---:|
| Last received value, row 253260 | 17.733900 |
| MLP predicted correction | +11.614564 |
| MLP prediction | 29.348464 |
| LOCF prediction | 17.733900 |
| Hidden recorded value | 4.667080 |
| Correction required in hindsight | -13.066820 |
| MLP absolute error | 24.681384 |
| LOCF absolute error | 13.066820 |

The MLP made a large positive correction when the recorded next value decreased.
Recent received acceleration-X values were -14.5066, 4.06926, and 17.7339.
Thus its prediction is consistent with continuing a recent rise, but that is an
interpretation of this trace, not proof of which cross-channel features caused
the network output. The history also contains sharp recorded changes, including
47.6314 followed by -17.6462. Their physical origin remains unresolved.

The hidden change magnitude 13.06682 exceeds the existing training q99 threshold
5.229231. That is a post-hoc rapid-recorded-change label; the hidden value was
never available as an input. No sensor-fault diagnosis is established.

The selected sample adds **438.428924** to squared error relative to LOCF, while
the net increase across all 14,176 random-loss positions for this cell is
**684.798201**. The event therefore explains about 64% of the net increase,
with gains/losses elsewhere cancelling; this percentage is not a share of total
model squared error. The segment's net increase is 743.450770, larger than the
full-cell net increase because other segments collectively offset some error.

For the complete seed-1/random-loss/acceleration-X cell:

| Predictor | MAE (m/s²) | RMSE (m/s²) |
|---|---:|---:|
| LOCF | 0.531344 | 1.587578 |
| Residual seed 1 | 0.518211 | 1.602719 |

MAE improves while RMSE worsens. This illustrates how a useful average correction
can coexist with damaging individual corrections. A residual connection supplies
LOCF as a reference; it does not enforce "never worse than LOCF."

## Verification

- Both complete-segment replays match all six channels of saved case metrics,
  including sums/counts/MAE/RMSE, at rtol=atol=1e-12.
- Repeated/reset, hidden-truth poisoning, future-prefix and received-preservation
  audits pass for both predictors.
- The 161 input features were independently rebuilt from past RECEIVED indices
  and match the streaming history exactly. Imputed predictions never fill history.
- Direct evaluation of all three affine layers and the normalization reversal
  reproduces the six streaming outputs exactly at the selected sample.
- Last received, hidden, and next received original recording rows were checked
  directly. Subject107 and checkpoint hashes remain unchanged.
- The raw-access log contains only canonical subject107.dat, with no denied opens.
  No training recording, 108, or 109 was opened.

`policy.json` preserves the selector; `event.json` contains numbers and checks;
`history.csv` records the 20 past slots; `features.npz` preserves actual inputs and
corrections; `trace.csv` and `failure.png` show the event. `artifacts.sha256` uses
paths relative to the repository root. Diagnostic source is
`../mlp_v1_residual_failure_20261027.py`, whose hash is recorded in event.json.

**Reproduction result: PASS.** This confirms a forecasting failure consistent
with the recorded checkpoint and causal inputs, not a newly discovered replay or
denormalization mismatch. It is not a general proof of implementation correctness.

**Next experiment — pass/fail:** quantify rapid-recorded-change versus complementary
errors for the frozen MLPs before selecting a correction limit or confidence rule.
PASS requires exact count partitions and reconciled sums on unchanged masks and
thresholds, with diagnostic labels outside predictor inputs. Otherwise FAIL and
resolve the discrepancy. No new model setting is selected by this trace alone.
