# Phase 4 handoff

Phase 3 is **validated and accepted as a desktop prototype on 2026-10-29**.
Read `AGENTS.md` and `05_STATUS_AND_DECISIONS_UPDATED.md` before continuing.
The learner wants Phase 4 in the separate chat "Phase 4 — INT8 quantization".

## Explain the next step simply

Use what/why/example in the chat, not only document links. For example:
"We will compare the same residual model before and after INT8 conversion.
This tells us how much error conversion adds. If both versions receive the
same history before missing sample 21, their predictions can be compared fairly."
Explain one step at a time and each file's role before editing it.

## Validated starting point

- Original NumPy models: `runs/mlp_v1_train_20261027/models/`, residual seeds
  0/1/2. Keep all three; no deployment seed was selected and this is no ensemble.
- Network 161→32→16→6, 5,814 parameters. Frozen float32 model features; physical
  float64 six-channel outputs. Received-only history, no estimate feedback.
- Residual mean normalized MAE 0.132366 versus LOCF 0.156898 on validation 107.
  Desktop rejection retains 97.8068–98.3299% of hidden estimates. Physical and
  application coverage limits remain unset. Preserve abrupt-change weaknesses.
- See `benchmarks/phase3_validation_accepted_20261029/` for closure evidence.

## Unfinished preparation: not validated Phase 4 work

These local untracked files were added before the learner clarified that a new
chat was wanted. They are deliberately excluded from the Phase 3 acceptance
commit. Inspect them before any reuse:

- `src/packetheal/quantization.py`: proposed feature extraction and TFLite adapter.
- `scripts/quantize_mlp.py`: proposed calibration/evaluation runner.
- `scripts/export_int8_worker.py`: proposed TensorFlow conversion worker.
- `tests/test_quantization.py`: eight proposed tests, not run during preparation.

No calibration or conversion command ran, no INT8 model was produced, and no raw
training/validation recording was opened by this preparation. The Phase 3 closure
tests intentionally exclude this Phase 4 test file. Untracked preparation may be
absent on another checkout; do not infer that it was committed or verified.

## Runtime state to inspect

- ST Edge AI Core 4.0.1 is installed and its CLI lists `stm32u5`. Bundled Python
  `/mnt/ollama_disk/files/installation st/stedgeai-lin/4.0/Utilities/linux/python`
  reports Python 3.9.13 / TensorFlow 2.18.0 / NumPy 1.26.4. Tool presence does not
  establish target-device compatibility or measurements.
- Project `.venv-int8/` was created and is ignored. LiteRT 2.3.0 and NumPy 2.5.4
  were installed there, then system-package inheritance was disabled. Inspect
  this potentially incomplete environment; do not assume it is ready.
- The later dependency-pinning install was **rejected by the learner** and did
  not run. Do not retry it without renewed authorization. The original experiment
  environment with NumPy 2.4.4 remains unchanged.

## Next experiment

Compare post-training INT8 conversion with retaining the floating-point model;
defer quantization-aware retraining unless the first experiment justifies it.
Proposed development targets were at most 2% normalized-MAE increase and 5%
channel/regime RMSE increase per seed, for full and accepted positions. These
targets are proposals, not application tolerances or measured results.

PASS the plan only after its interfaces, training-only calibration selection,
numerical targets and identical frozen validation masks are clear. Otherwise
revise before converting. Never access subjects 108/109, including hashing.
Never modify frozen artifacts or original evaluated source/checkpoints.
