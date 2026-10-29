# Phase 3 — validated desktop prototype

**Validated, finished and explicitly accepted by the learner on 2026-10-29.**
The learner requested final validation, recording acceptance, clearer explanations
in future conversations, and committing/pushing the final work.

## What is accepted

The causal floating-point MLP benchmark, fixed-warning desktop rejection,
status-aware consumer and their recorded correctness checks. This is acceptance
of the bounded desktop research prototype. Physical/application error and minimum
coverage requirements remain unset. No hardware or application acceptance follows.

| Result | Verified observation |
|---|---|
| Training | Subjects 101–106; 388,242 examples; six models, 20 epochs each |
| Validation | Subject 107; 102 segments; four masks; 2,856 algorithm cases |
| Direct MLP mean normalized MAE | 0.135532, 13.62% below LOCF |
| Residual MLP mean normalized MAE | 0.132366, 15.64% below LOCF |
| Residual hidden-position coverage with rejection | 97.8068–98.3299% |
| Accepted residual mean normalized MAE | 0.119000; about 10.1% below the original full set |
| Same accepted positions versus LOCF | Lower MAE and RMSE in 72/72 channel/regime/seed cells |
| Original practical engineering screen | Passes for both model families |

Three residual models mean three independently trained copies with the same
architecture/data and different seeds. They are evaluated separately; this is not
an ensemble and no deployment seed has been selected. Normalized MAE is a
dimensionless error score, not a classification-accuracy percentage. Conditional
error decreases when some positions are rejected; original predictions are unchanged.

## Final checks

`scripts/verify_phase3_acceptance.py` independently pools saved absolute/squared
error sums and age partitions, verifies original case identities and accepted
comparisons, checks historical/current hashes, reruns the 197 Phase 3 software
tests and repeats trained-consumer integration for all three residual seeds.
Every actual raw-file open is denied by a Python audit hook; synthetic temporary
test fixtures are allowed. No new PAMAP2 replay or training is performed.

- `acceptance.json`: actual verification outcome, scope, counts and limitations.
- `software_tests.txt`: fresh software test output. Unfinished Phase 4 tests are
  explicitly excluded from the Phase 3 suite.
- `initial_verifier_error.json` and `software_tests_initial_verifier_attempt.txt`:
  the first verifier passed its checks/tests but omitted a JSON-writer import
  when saving its result. The helper import was corrected and verification was
  repeated; no model or original evaluated-source changes were needed.
- `fresh_consumer_integration.json`: three trained seeds, 40 artificial slots each;
  each retains 23 measured values, 13 estimates and four explicit gaps.
- `historical_document_resolution.json` and `historical_documents/`: exact old
  versions of instructions/status/review that prior manifests pinned. Their
  original checksums/manifests are not rewritten. Only these four mutable document
  bindings use archived versions; model, data and evaluated source bindings use
  their original paths and hashes.
- `documents/`: snapshots of the accepted status, final review and new guidance.
  These are the immutable document copies covered by the final archive manifest;
  later changes to live instructions/status do not invalidate this closure.
- `artifacts.sha256`: final archive and newly committed Phase 3 source checksums.

The original trained-consumer/review runner scripts pin all files in the earlier
checkpoint, including then-current instruction documents. Their historical hashes
remain correct for those exact versions. Use the final verifier for closure
checks on the updated checkout; do not weaken model/data bindings to accommodate
mutable instructions. Original JSON records saying acceptance was pending remain
historical; `acceptance.json` and current status now supersede that milestone state.

## Limits retained

The original gyro-Y/burst-5 rapid-change group has 49 positions and mean-seed RMSE
19.29% above LOCF. Large individual errors remain possible. The warning is a
heuristic, not calibrated confidence. The LOCF-on-warning fallback worsened mean
normalized MAE by 3.44% and remains rejected. Only one validation subject and
synthetic losses have been evaluated. Three training seeds are not independent
subjects. Subjects 108/109 remain protected. INT8/device/transport performance
and application suitability remain unmeasured.

## Next conversation

Phase 4 belongs in its separate chat. Read `AGENTS.md`, the accepted status and
`docs/PHASE4_HANDOFF.md`. Local unfinished quantization preparation and its runtime
environment are excluded from this Phase 3 commit/acceptance. Do not continue
quantization in the closure task or retry the rejected install without renewed
authorization.

Next experiment: review the first desktop INT8 plan in the Phase 4 chat. PASS
requires clear tensor interfaces, training-only calibration, fixed comparison
targets and protected held-out subjects; otherwise revise before conversion.
