# Desktop abstention prototype

**Prototype correctness PASS. Application acceptance NOT ASSESSED.** The learner
chose “Prototype only: test rejection and report the trade-off.” No application
coverage requirement or physical error limit has been agreed. This implementation
is isolated from the original training/benchmark pipeline.

## 1. Behavior

| Input/condition | Output status | Output values |
|---|---|---|
| Received real six-vector | MEASURED | Original measurement, unchanged |
| Missing; warning known and OFF; loss age <=10 | RECONSTRUCTED | Original predictor estimate |
| Missing; warning on or unknown | UNAVAILABLE | Six NaNs |
| Missing; original predictor unavailable | UNAVAILABLE | Six NaNs |
| Missing; loss age > existing model horizon 10 | UNAVAILABLE | Six NaNs |

The warning being OFF does not prove quiet physical motion or an accurate
prediction. The underlying predictor retains its original
LOCF warm-up fallback before its 20-slot history is ready. The prototype does not
create a new model, adjust thresholds or select a preferred seed.

**Why:** retain useful estimates while making selected gaps explicit. NaN is a
desktop sentinel, not a zero measurement. Downstream code must check `status`
before using `values`. Embedded INT8 cannot carry NaN; a future embedded interface
must preserve validity separately. No firmware or INT8 behavior was implemented.

## 2. Software flow

```text
step(received six-vector or None)
    → stage the fixed received-change warning state
    → advance the original MLP predictor state
    → keep a real measurement or accepted estimate
      OR emit UNAVAILABLE + six NaNs
```

Both components advance once per successful step, including rejected missing
steps. They receive real observations or `None`; rejected estimates never enter
history. A staged shallow warning-state copy prevents a predictor validation or
numerical failure from advancing the warning alone. This relies on the pinned
warning implementation replacing, rather than mutating, its small state arrays.
Synthetic tests verify failure atomicity and history contents.

The prototype deliberately computes the underlying prediction before deciding
whether to expose it. This preserves the existing predictor's state behavior.
Skipping discarded inference is a separate possible optimization; latency and
resource use have not been measured.

Interface: `reset()` and `step(vector_or_None)` return the existing `StepResult`:
float64 values `[6]`, `SampleStatus`, and integer consecutive-loss count. MLP
features remain float32 `[161]`; internal six-output predictions are float32,
decoded using the existing float64 training normalization and real anchor.
The wrapper also records a diagnostic `reason` without changing the shared enum.

## 3. Error versus retained coverage

Full-validation figures below derive from the pinned saved warned/unwarned sums
for **102 subject107 segments × four masks × three residual seeds = 1,224 cases**.
Current frozen gaps have ages <=10 and protected warm-up. Therefore the accepted
set is exactly the saved known-unwarned set; the new >10 cutoff does not change
these figures. Its boundary behavior is tested synthetically, not validated as
an accuracy or timing result on real long outages.

| Regime | Hidden positions | Accepted | UNAVAILABLE | Coverage |
|---|---:|---:|---:|---:|
| Random | 14,176 | 13,886 | 290 | 97.9543% |
| Burst 2 | 14,226 | 13,914 | 312 | 97.8068% |
| Burst 5 | 14,370 | 14,130 | 240 | 98.3299% |
| Burst 10 | 14,660 | 14,400 | 260 | 98.2265% |

Counts are shared by all seeds. The four masks are alternative scenarios, not
distinct real transport events. Coverage concerns hidden positions only; real
measurements always pass through. `hidden = accepted + unavailable`.

MAE/RMSE use accepted positions as their denominator. Rejected errors are neither
treated as zero nor silently removed from coverage counts.

| Residual seed | Original normalized MAE, all positions | Accepted normalized MAE | Conditional change |
|---|---:|---:|---:|
| 0 | 0.132873 | 0.119544 | -10.032% |
| 1 | 0.131710 | 0.118347 | -10.145% |
| 2 | 0.132516 | 0.119108 | -10.118% |

Normalized MAE equally averages physical MAE divided by frozen training standard
deviation over channels/regimes. These are conditional errors on fewer estimates,
not improvements to the original numerical predictions. Individual physical
channel/age metrics and LOCF on the same accepted positions are in `metrics.csv`.
This is a single fixed operating point, not a full risk/coverage curve.

## 4. What was freshly replayed

Before raw access, `policy.json` fixed **segments 0 and 92, all four regimes and
all three residual seeds: 24 fresh runtime cases**. Segment0 checks ordinary
pass-through behavior; segment92 includes the previously investigated failure.
This is a bounded correctness check, not representative sampling or another full
102-segment replay.

Each case checks original-output agreement on accepted positions, NaNs and
UNAVAILABLE on rejected positions, received-value preservation, and loss ages.
Repeated replay, hidden-truth poisoning, intervening A-B-A reset and future-half
prefix audits pass for the wrapper. The complete warning-state fingerprint agrees
with the original saved warning case. Full and per-age fresh error sums/counts
match the saved-sum abstention calculations. Segment0 rejects none; segment92
rejects 36/34/10/10 positions across random/burst2/burst5/burst10 for each seed.

Only canonical subject107.dat is opened: two segment loads and before/after
hashing (four opens), with zero denied attempts. Its hash matches original
evidence. The Python guard is an application check, not an OS sandbox.
Recordings 108/109 and training recordings are untouched. Checkpoints, evaluated
source, frozen dataset artifacts, warning thresholds and prior evidence remain
unchanged. No training, status-document/milestone update, commit or push occurred.

## 5. Files

- `../mlp_v1_abstention_20261027.py`: the isolated `AbstainingMLP` wrapper.
- `../test_mlp_v1_abstention_20261027.py`: ten synthetic tests: real ownership,
  startup/unknown state, accepted forecasts, warning/recovery, horizon boundary,
  real-only history, failure atomicity, reset/causality, all future cutoffs and
  fully rejected scoring. They do not open actual recordings.
- `../mlp_v1_abstention_run_20261027.py`: saved full-validation calculations,
  bounded actual wrapper replay, source/hash checks and export.
- `../mlp_v1_abstention_verify_20261027.py`: independent saved arithmetic,
  count/age partitions and CSV checks; opens no recordings.
- `policy.json`: fixed behavior, learner authorization, acceptance scope,
  preselected runtime cases and source/input hashes.
- `software_tests.txt`: actual ten-test output.
- `report.json`: 1,224 saved-sum case records, pooled/age metrics, 24 runtime
  check records, provenance, raw-access log and explicit unset application criteria.
- `metrics.csv`: 450 physical-unit rows, including unavailable counts, coverage,
  original/accepted errors and matched-position LOCF errors.
- `independent_checks.json`: verification of every case, partition, derived
  metric and CSV row using independent `math.fsum` formulas at rtol=atol=1e-12.
  Runtime record consistency is verified from saved evidence; the independent
  verifier does not repeat raw predictor inference.
- `evidence.sha256`: result and source checksums relative to repository root.

The original complete model suite was 174 passing tests before training; it was
not rerun here because those sources are unchanged. The ten new tests exercise
this wrapper and are distinct from the 24 dataset runtime cases.

Verify saved evidence without accessing recordings:

```bash
sha256sum -c runs/mlp_v1_abstention_20261027/evidence.sha256
```

The warning is a heuristic, not a calibrated confidence probability. Rejection
does not guarantee accurate accepted predictions. No final-test, cross-subject,
real transport, device or application-safety conclusion follows from this prototype.

**Next experiment — pass/fail:** demonstrate a downstream consumer that checks
status before using sensor values. PASS requires real and reconstructed values
to remain distinguishable and UNAVAILABLE samples to be counted/handled without
being treated as measurements; otherwise FAIL the integration. Application
coverage/error requirements remain a separate decision before operational use.
