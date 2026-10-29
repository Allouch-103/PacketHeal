# Phase 3 desktop checkpoint — 2026-10-29

This is a preservation checkpoint requested before further work. Phase 2 remains
formally accepted; Phase 3 is experimental and has not been formally accepted.
Historical run reports, source snapshots, checksums and negative results retain
their original bytes and dates. No new dataset training or validation replay is
performed by this checkpoint.

The completed local artifacts below are explicitly versioned despite the general
`runs/` ignore rule. Other generated runs remain ignored. Statements about being
Git-ignored in historical run READMEs describe their state when originally written.

## Preserved work

- [MLP guide](../../docs/MLP_V1_GUIDE.md): causal representation, mathematics,
  software interfaces and experiment design. Its original proposals are historical;
  the measured outcomes are in the reports below.
- [Training](../../runs/mlp_v1_train_20261027/training.json): direct/residual modes,
  three seeds, six trained checkpoints, source snapshots and training provenance.
- [Validation](../../runs/mlp_v1_validation_20261027/README.md): subject 107 only,
  original comparisons with frozen classical baselines and all-seed regressions.
- [Transient analysis](../../runs/mlp_v1_transients_20261027/README.md),
  [residual failure](../../runs/mlp_v1_residual_failure_20261027/README.md),
  [gyro-Y failure](../../runs/mlp_v1_gyroy_burst5_failure_20261027/README.md), and
  [input sensitivity](../../runs/mlp_v1_accy_sensitivity_20261027/README.md):
  difficult cases and the limitations of overall averages.
- [Received-change warning](../../runs/mlp_v1_received_warning_20261027_retry1/README.md):
  fixed causal rule and full-validation grouping. The failed initial attempt is
  also retained separately; it is not a successful experiment.
- [LOCF fallback](../../runs/mlp_v1_warning_fallback_20261027/README.md): negative
  result, approximately 3.44% higher mean normalized MAE; rule not adopted.
- [Abstention](../../runs/mlp_v1_abstention_20261027/README.md): desktop prototype,
  conditional error/coverage, ten synthetic tests and 24 bounded runtime cases.
- [Consumer](../../runs/mlp_v1_consumer_20261027/README.md): status-first handling,
  six tests and an artificial-model demonstration.

The original validation engineering screen includes per-cell RMSE regressions;
better mean MAE alone does not establish a complete acceptance PASS. The warning
is not calibrated confidence. Application error limits and minimum coverage have
not been agreed. Recordings 108/109 remain protected, and raw dataset files are
excluded from this commit. No INT8, transport or target hardware result is claimed.

## Checkpoint verification

`verification.json` records checks of existing evidence manifests, model/source
identity and frozen preparation/baseline inputs. `software_tests.txt` captures the
current repository and prototype test suites with actual raw recording opens
denied by a Python audit hook. Temporary synthetic test fixtures are permitted.
`test_summary.json` records the result and raw-access attempt count. These tests
include small artificial learning fixtures, not new PAMAP2 model training.

Fresh result: **197 tests passed**, zero failures/errors and zero actual raw-file
open attempts. **105 entries across 12 existing manifests** matched, along with
all six trained checkpoints and frozen preparation/classical inputs. An initial
test launcher omitted `src` from the import path; its error log is retained as
`software_tests_initial.txt`. Correcting the launcher required no model, test or
predictor source changes. CSV CRLF line endings are preserved to retain the
historical hashes; the Git whitespace check uses `cr-at-eol` for these files.

The root status document remains the older Phase 2 closure record at this
checkpoint. Updating that record and reviewing Phase 3 are explicitly the next
authorized work, after committing and pushing this preservation checkpoint.

Next experiment: connect existing trained weights to the unchanged status-aware
consumer. PASS requires unchanged measured values, distinct estimate tags,
preserved gap slots and no use of unavailable payloads; otherwise FAIL.
