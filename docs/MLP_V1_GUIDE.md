# Causal MLP v1: implementation and learning guide

## Current state and limits

The implementation was first staged in `/tmp/packetheal_mlp_v1` during a disk
failure. On 2026-10-26 the repository became readable, the staged files survived,
and the implementation was integrated into the repository. Frozen artifacts and
existing baseline source remain unchanged. No actual PAMAP2 recording was opened
for this integration; temporary synthetic recordings exercise the real loader.

The existing loader returns `(timestamps, values)` and expects the PAMAP2_Dataset
root, appending `Protocol/` itself. The access guard and raw-checksum paths now
follow that contract. Tests check the actual archived CSV schema, frozen hashes,
split-aware loader, mask reader, and real baseline protocol. The baseline-protocol
fixture is used only when these tests are copied outside the repository.

Phase 2 is formally accepted in status/DEC-023, committed as `d9555f5`.
The learner explicitly requested the full MLP implementation in this chat and
authorized integration/testing. The training CLI acknowledgement does not record
milestone acceptance or independently authorize a run. PAMAP2 model training and
validation have not been run; uncertainty, quantization, and firmware remain future
work. Implementation acceptance and measured model accuracy are separate.

## What each file does

| File | Responsibility |
|---|---|
| `src/packetheal/mlp.py` | Normalization, observation-only history, dense network, MAE derivatives, Adam, checkpoint files, streaming prediction, supervised examples |
| `src/packetheal/mlp_experiment.py` | Recording guard, frozen-input checks, split-aware loading adapter, balanced training, audited validation, metrics, checkpoint provenance, practical comparison screen |
| `scripts/run_mlp.py` | Command-line entry point: `synthetic`, `train`, `validate` |
| `tests/test_mlp.py` | Hand example, gradient calculations, no feedback, normalization, startup, gaps, resets, future/hidden isolation, checkpoint behavior |
| `tests/test_mlp_experiment.py` | Synthetic access boundaries, pooled arithmetic, runner determinism, loader schema checks, comparison screen |
| `requirements-mlp.txt` | NumPy 2.4.4 used for repository integration |
| `runs/mlp_v1_integration_20261026/` | Integration test evidence and synthetic-run checkpoints/report; Git-ignored outputs |

## Step 1: build features before reading the answer

`CausalHistory.prepare()` returns features for a missing current sample. It reads
only previously committed observations. Each of 20 rows has six normalized values,
one received flag, and one slot-valid flag. The 161st feature is the current age,
clipped to 10 and divided by 10. Padding is zero with slot-valid=0. A missing
historical slot carries the last real value with received=0 and slot-valid=1.

`commit(real_vector)` adds a genuine observation. `commit(None)` adds a missing
slot. Neither function accepts a reconstruction as a special observation type.
The streaming wrapper commits `None` after making a missing prediction.

The MLP activates after 20 elapsed slots, with an existing real observation, and
at loss ages 1-10. Other initialized cases use LOCF. Before the first observation,
loss outputs are UNAVAILABLE. Every segment/regime starts from reset state.

## Step 2: define what the network learns

Direct mode predicts `z(x[t])`. Residual mode predicts
`(x[t] - last_real) / training_std`. Residual denormalization is
`last_real + training_std * output`; the mean cancels.

Both modes use 161 -> 32 ReLU -> 16 ReLU -> 6 linear outputs: 5,814 parameters.
Residual final-layer weights start at zero, giving exact LOCF at initialization.
Received outputs preserve the original physical-unit values.

Normalized MAE is the training loss. `loss_and_gradients` computes its derivatives;
finite-difference tests check every parameter on a small network away from ReLU
kinks. Adam keeps moving averages of gradients and squared gradients. We use
NumPy because PyTorch is not installed; the implementation has no training-time
framework dependency beyond NumPy.

## Step 3: separate supervised answers from inputs

`segment_examples` first builds features from received history, then reads hidden
truth to create the target. Consecutive losses remain filled with the real anchor.
It never fills input windows using clean hidden truth or future values.

Training reuses subjects 101-106, frozen normalization, and frozen masks. Each
batch draws equally from the four regimes with replacement. An epoch consists
of approximately one training-dataset-sized set of draws. Both modes use identical
batch draws for each seed. Final training objectives score every stored example
with equal channel/regime weighting; epoch logs are sampled pre-update losses.

The proposed real run uses 20 epochs, batch 256, Adam learning rate 0.001, seeds
0/1/2, no early stopping, and no subject-107 tuning. No held-out subject is used.
Validation confirms checkpoint hashes and training source identity first.

## Step 4: validate fairly

Validation is restricted to 107 and reuses all 408 frozen segment/regime cases.
Pooled physical-unit MAE/RMSE use sums and counts, not averages of segment RMSEs.
Reports include per-case and pooled metrics, coverage, and loss-age errors.
Each replay checks repeatability, A-B-A resets, hidden-NaN poisoning, changed
future values/masks, original arrays, measured outputs, and loss ages.

Classical source hashes and archived evidence are pinned. LOCF is replayed and
its pooled metrics must match the pinned archive. Linear and selected alpha-beta
use their unchanged archived CSV metrics, copied byte-for-byte into the output.
Selected alpha-beta remains LOCF-equivalent. The staged runner DOES NOT claim to
recheck every classical replay fingerprint or independently rerun linear/filter
replays. Per-channel rapid-change analysis is not yet integrated into this runner;
retain the Phase-2 archive and add that diagnostic before a final scientific claim.

The engineering screen reports every seed, a proposed >=5% mean normalized-MAE
reduction, improvement in at least two of three seeds, no >10% channel/regime
RMSE increase in any seed, and complete coverage. These are practical development
thresholds, not a significance test, calibrated confidence, or deployment approval.

## Commands

From the repository root, the independent synthetic checks are:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python3 -B -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python3 -B scripts/run_mlp.py synthetic --output-dir runs/mlp_v1_synthetic_new
```

The synthetic run deliberately learns on its training fixture. Its improvement
does not establish held-out performance. The direct model can score worse and
that result is retained.

The initial temporary synthetic run used NumPy 1.26.4. Repository tests and the
fresh synthetic run use NumPy 2.4.4, matching the historical archive dependency.
Cross-environment bitwise equality is not claimed. Test logs/JSON record the
environment and an outer guard rejects all non-temporary `.dat` opens, including
attempts that a test catches internally.

The separate dataset commands, after training authorization, are:

```bash
OPENBLAS_NUM_THREADS=1 python3 -B scripts/run_mlp.py train --root /mnt/ollama_disk/files/PacketHeal --dataset-dir /path/to/PAMAP2_Dataset --output-dir runs/mlp_v1_train --phase2-accepted
OPENBLAS_NUM_THREADS=1 python3 -B scripts/run_mlp.py validate --root /mnt/ollama_disk/files/PacketHeal --dataset-dir /path/to/PAMAP2_Dataset --training-dir runs/mlp_v1_train --output-dir runs/mlp_v1_validation
```

`/path/to/PAMAP2_Dataset` is a placeholder, not a verified local path. Outputs
must be new directories outside raw data, frozen artifacts, and source folders.
The script refuses existing outputs. Recording checksums are computed only for
101-106 in training and only 107 in validation. Source/metadata checks never scan
held-out recordings. The Python audit guard is an application check, not a kernel
sandbox. Reports capture installed Python package versions and thread settings;
this is provenance, not a guaranteed portable system environment lock.

**Next experiment — pass/fail:** an explicitly authorized training run on subjects
101-106, followed by the frozen subject-107 evaluation. Software PASS requires
complete case/count reconciliation, frozen-source/input preservation, all leakage
audits, and archived LOCF agreement. Accuracy PASS is a separate practical screen
defined above. Otherwise FAIL and preserve the result without validation retuning.
