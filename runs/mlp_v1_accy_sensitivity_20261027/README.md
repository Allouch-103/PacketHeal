# Acceleration-Y input sensitivity

**Result: PASS.** Changing the selected acceleration-Y input strongly changes
gyro-Y predictions in the frozen residual MLPs. This is one copied-input
diagnostic, not a new benchmark or a validated measurement filter.

## 1. Load and verify the original inputs

**What:** load the saved five-step burst and three trained residual checkpoints.
**Why:** use exactly the inputs and weights from the reproduced failure.

Software: the runner checks prior evidence, source, checkpoint and frozen-input
hashes. It loads `features.npz` and `history.csv` from the previous diagnostic.
No raw recording is opened. Files containing prior hidden truth are hashed for
integrity but never parsed as targets or model inputs.

## 2. Change one received measurement in a copy

The last received acceleration Y, row 269423, is **130.207 m/s²**. Replace it
with **5.31869 m/s²**, from the preceding received row 269422.

The previous received value retains local context. Replacing with the training
mean would instead move the feature toward a global average. Neither alternative
is established as the correct physical measurement.

Software: copy the original float32 `[5,161]` feature array. For each prediction,
replace acceleration Y in every slot whose held value comes from row 269423.
There are **1/2/3/4/5** such slots at loss ages 1/2/3/4/5: **15 entries total**.
Use the frozen transformation `z = (replacement - mean_acc_y) / std_acc_y`.
All other channels, received/valid flags and loss ages remain byte-identical.
The copied physical acceleration-Y anchor changes consistently; the gyro-Y anchor
stays **0.186639 rad/s**. Original saved files remain unchanged.

## 3. Run the same fixed networks

Theory: for a fixed function `f`, compare `f(original)` with `f(modified)`.
The difference measures sensitivity to this intervention. It is not a gradient,
training update, confidence estimate or proof about sensor physics.

Software: each input is float32 `[1,161]`; the network produces float32 `[1,6]`.
Decode each residual in float64:

```text
physical_correction[c] = normalized_output[c] × training_std[c]
prediction[c]         = real_anchor[c] + physical_correction[c]
```

Two input cases × three seeds × five ages = **30 distinct forward calculations**.
No hidden target is needed to compare their outputs.

At age 4, original row 269427:

| Residual seed | Original gyro Y | Modified gyro Y | Change |
|---|---:|---:|---:|
| 0 | 5.278937 | 0.901765 | -4.377173 |
| 1 | 3.316673 | 0.418855 | -2.897819 |
| 2 | 3.503581 | 0.285957 | -3.217625 |

All values are rad/s. Holding the other model inputs and gyro-Y anchor fixed,
this particular acceleration-Y intervention lowers the age-4 gyro-Y forecast in
all three seeds. This establishes a computational sensitivity along the chosen
intervention. It does not establish a physical cause, universal feature importance,
or an improvement across other gaps. No error-based threshold or replacement
value was selected, and no accuracy metric is reported for modified inputs.

## 4. Check the experiment

The runner checks exact reproduction of all **15 original six-channel outputs**,
identity-intervention equality, repeated/A-B-A predictions, original input bytes,
weights, sources and prior evidence. Its raw-data deny guard records zero opens.

The separate verifier derives the edit locations from time indices, checks all
other entries and anchors, and independently evaluates the network using Python
float64 arithmetic and `math.fsum`. All 30 calculations agree with the float32
network at predeclared rtol=atol=2e-5. Maximum normalized-output discrepancy is
**3.8005e-6**. Float64 and float32 arithmetic need not be bit-identical.
It also verifies normalization reversal and all 30 prediction / 15 comparison CSV
rows. The verifier opens no raw recordings.

## Files

- `../mlp_v1_accy_sensitivity_20261027.py`: checks inputs, declares the intervention,
  edits copies, runs models, saves outputs and draws the figure.
- `../mlp_v1_accy_sensitivity_verify_20261027.py`: independent edit, scalar-layer,
  denormalization and CSV checks.
- `policy.json`: intervention fixed before inference, alternatives, invariants,
  acceptance checks and pinned hashes.
- `arrays.npz`: original/modified float32 inputs `[5,161]`, Boolean edit mask,
  float32 normalized outputs `[2,3,5,6]`, float64 corrections/predictions with the
  same output shape, physical anchors and normalization constants.
- `edited_positions.csv`: the 15 changed feature positions and their source rows.
- `predictions.csv`: 30 model/input/age records in physical units.
- `comparison.csv`: 15 paired original/modified predictions and differences.
- `report.json`, `independent_checks.json`: provenance and verification results.
- `sensitivity.png`: inspected plot for all three seeds across the five-step gap.
- `evidence.sha256`: evidence and script hashes, relative to the repository root.

Existing model code, datasets, masks, normalization, checkpoints and benchmark
evidence are unchanged. No training, firmware, clipping rule, milestone update,
commit or push occurred. Subjects 108/109 were not accessed. The application
audit guard is not an OS sandbox.

Both scripts refuse existing outputs. Verify saved evidence without recordings:

```bash
sha256sum -c runs/mlp_v1_accy_sensitivity_20261027/evidence.sha256
```

**Next experiment — pass/fail:** evaluate whether large *received* changes can
identify higher-error MLP predictions, using unchanged models/masks and the
existing training-only thresholds. PASS requires strictly causal warning inputs,
reconciled flagged/unflagged error totals and reported coverage; otherwise FAIL
and fix the analysis. Predictive usefulness must be measured separately from
these correctness checks. No online filter or abstention policy is yet validated.
