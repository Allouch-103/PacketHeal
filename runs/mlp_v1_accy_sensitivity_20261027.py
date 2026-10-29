"""Copied-input sensitivity for one frozen burst; no raw loading or fitting."""

from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from packetheal.mlp_experiment import digest, validate_training_record, verify_frozen, write_json

PRIOR = ROOT / "runs/mlp_v1_gyroy_burst5_failure_20261027"
TRAIN = ROOT / "runs/mlp_v1_train_20261027"
OUT = ROOT / "runs/mlp_v1_accy_sensitivity_20261027"
PRIOR_MANIFEST_HASH = "6931f437ad30df4b84fd4019ecddada61101a27977534a04abb8911ca104cbfd"
RAW = ROOT / "data/raw"


def save_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    raw_attempts = []

    def forbid_recordings(event, args):
        if event == "open" and args and not isinstance(args[0], int):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.suffix == ".dat" or path.is_relative_to(RAW):
                raw_attempts.append(str(path))
                raise PermissionError("This copied-input diagnostic must not open raw data")

    sys.addaudithook(forbid_recordings)
    frozen = verify_frozen(ROOT)
    training, loaded = validate_training_record(ROOT, TRAIN)
    prior_manifest = PRIOR / "evidence.sha256"
    assert digest(prior_manifest) == PRIOR_MANIFEST_HASH
    prior_hashes = {}
    for line in prior_manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and path.suffix != ".dat"
        assert digest(path) == expected
        prior_hashes[name] = expected
    # These files contain received-only inputs and saved network activations.
    # Truth-bearing prior files are only hashed above, never parsed as inputs.
    with np.load(PRIOR / "features.npz", allow_pickle=False) as saved:
        original = saved["features"].copy()
        mean, std = saved["normalization_mean"].copy(), saved["normalization_std"].copy()
        anchor, original_rows = saved["anchor"].copy(), saved["original_rows"].copy()
        saved_z = np.array([saved[f"residual_seed{s}_normalized_corrections"] for s in range(3)])
    with (PRIOR / "history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    assert original.shape == (5, 161) and original.dtype == np.float32 and len(history) == 100
    source_row = int(original_rows[0]) - 1
    preceding_row = source_row - 1
    preceding = [r for r in history if int(r["slot_original_row"]) == preceding_row]
    assert len(preceding) == 5 and all(int(r["received"]) == 1 for r in preceding)
    assert all(int(r["last_real_original_row"]) == preceding_row for r in preceding)
    replacement_values = {float(r["held_acc_y_m_s2"]) for r in preceding}
    assert len(replacement_values) == 1
    replacement = replacement_values.pop()
    original_value = float(anchor[1])
    assert original_value == 130.207 and replacement == 5.31869
    assert original_rows.tolist() == list(range(source_row + 1, source_row + 6))

    edit_mask = np.zeros_like(original, dtype=bool)
    edited_rows = []
    for offset, prediction_row in enumerate(original_rows):
        slots = [r for r in history if int(r["prediction_original_row"]) == int(prediction_row)]
        assert len(slots) == 20 and {int(r["slot"]) for r in slots} == set(range(20))
        for record in slots:
            if int(record["last_real_original_row"]) != source_row:
                continue
            slot = int(record["slot"])
            assert int(record["slot_original_row"]) < prediction_row
            assert float(record["held_acc_y_m_s2"]) == original_value
            column = slot * 8 + 1  # acc_Y within [six values, received, valid]
            edit_mask[offset, column] = True
            edited_rows.append({"loss_age": offset + 1, "prediction_original_row": int(prediction_row),
                                "slot": slot, "slot_original_row": int(record["slot_original_row"]),
                                "received": int(record["received"]), "source_original_row": source_row,
                                "feature_column": column, "original_physical_acc_y": original_value,
                                "replacement_physical_acc_y": replacement})
    np.testing.assert_array_equal(edit_mask.sum(axis=1), [1, 2, 3, 4, 5])
    encoded_original = np.float32((original_value - mean[1]) / std[1])
    encoded_replacement = np.float32((replacement - mean[1]) / std[1])
    np.testing.assert_array_equal(original[edit_mask], np.full(15, encoded_original))
    modified = original.copy()
    modified[edit_mask] = encoded_replacement
    np.testing.assert_array_equal(original != modified, edit_mask)
    np.testing.assert_array_equal(modified[~edit_mask], original[~edit_mask])
    # An identity intervention must be exactly the original input.
    identity = original.copy()
    identity[edit_mask] = encoded_original
    np.testing.assert_array_equal(identity, original)
    anchors = np.array([anchor, anchor.copy()])
    anchors[1, 1] = replacement  # Consistent changed measurement for residual decoding.
    assert anchors[0, 4] == anchors[1, 4]  # Gyro-Y anchor is unchanged.
    original_bytes = original.tobytes()
    original.flags.writeable = False
    cases = ("original", "replace_acc_y_with_previous_received")
    policy = {
        "kind": "copied_input_sensitivity", "subject": 107, "segment": 92,
        "regime": "burst_5_5pct", "original_rows": original_rows.tolist(),
        "source_received_row": source_row, "replacement_source_received_row": preceding_row,
        "feature_channel": "acc_y_m_s2", "output_channel": "gyro_y_rad_s",
        "original_acc_y": original_value, "replacement_acc_y": replacement,
        "reference_choice": "Previous received value; preserves local context, unlike replacing with the global training mean",
        "intervention": "Replace acc-Y in every slot held from the selected received row; change that channel of the copied physical anchor consistently",
        "unchanged": "All other input channels, masks, slot validity, loss ages, gyro-Y anchor, weights, normalization",
        "edit_counts_by_age": edit_mask.sum(axis=1).tolist(), "cases": list(cases), "seeds": [0, 1, 2],
        "theory": "Finite change of a fixed function f(x); measures sensitivity along this intervention",
        "acceptance": "Exact original-output reproduction; only declared edits; repeat/A-B-A determinism; independent layer/denormalization arithmetic; unchanged prior artifacts; zero raw opens",
        "limits": "One selected burst; no targets used; no accuracy benchmark, calibrated filter, physical causation, or sensor-fault diagnosis",
        "prior_manifest_sha256": PRIOR_MANIFEST_HASH, "prior_artifact_hashes": prior_hashes,
        "frozen_inputs": frozen, "training_sha256": digest(TRAIN / "training.json"),
        "runner_sha256": digest(__file__),
    }
    OUT.mkdir()
    write_json(OUT / "policy.json", policy)  # Fixed before running any model.
    models = [(m, model) for m, model in loaded if m["mode"] == "residual"]
    models.sort(key=lambda item: item[0]["seed"])
    assert [m["seed"] for m, _ in models] == [0, 1, 2]
    snapshots = [[p.copy() for p in model.parameters] for _, model in models]
    z = np.empty((2, 3, 5, 6), dtype=np.float32)
    for seed, (_, model) in enumerate(models):
        # Evaluate one row at a time to reproduce the original streaming GEMM.
        for offset in range(5):
            baseline = model.predict(original[offset:offset+1])[0]
            altered = model.predict(modified[offset:offset+1])[0]
            np.testing.assert_array_equal(baseline, saved_z[seed, offset])
            np.testing.assert_array_equal(model.predict(original[offset:offset+1])[0], baseline)
            np.testing.assert_array_equal(model.predict(modified[offset:offset+1])[0], altered)
            np.testing.assert_array_equal(model.predict(identity[offset:offset+1])[0], baseline)
            z[0, seed, offset], z[1, seed, offset] = baseline, altered
    corrections = z.astype(np.float64) * std
    predictions = corrections + anchors[:, None, None, :]
    assert original.tobytes() == original_bytes
    for parameters, (_, model) in zip(snapshots, models):
        for before, after in zip(parameters, model.parameters):
            np.testing.assert_array_equal(before, after)
    save_csv(OUT / "edited_positions.csv", edited_rows)
    records, comparisons = [], []
    for seed in range(3):
        for offset, row in enumerate(original_rows):
            for scenario, name in enumerate(cases):
                records.append({"case": name, "seed": seed, "original_row": int(row), "loss_age": offset + 1,
                                "gyro_y_anchor": float(anchors[scenario, 4]),
                                "normalized_gyro_y_correction": float(z[scenario, seed, offset, 4]),
                                "physical_gyro_y_correction": float(corrections[scenario, seed, offset, 4]),
                                "gyro_y_prediction": float(predictions[scenario, seed, offset, 4])})
            comparisons.append({"seed": seed, "original_row": int(row), "loss_age": offset + 1,
                                "original_gyro_y_prediction": float(predictions[0, seed, offset, 4]),
                                "modified_gyro_y_prediction": float(predictions[1, seed, offset, 4]),
                                "prediction_change": float(predictions[1, seed, offset, 4] - predictions[0, seed, offset, 4])})
    save_csv(OUT / "predictions.csv", records)
    save_csv(OUT / "comparison.csv", comparisons)
    with (OUT / "arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, original_features=original, modified_features=modified,
                            edit_mask=edit_mask, anchors=anchors, normalization_mean=mean,
                            normalization_std=std, normalized_outputs=z, physical_corrections=corrections,
                            predictions=predictions, original_rows=original_rows)
    verify_frozen(ROOT)
    validate_training_record(ROOT, TRAIN)
    for name, expected in prior_hashes.items():
        assert digest(ROOT / name) == expected
    assert not raw_attempts
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "policy": policy,
              "checks_passed": True, "checks": {"original_outputs_exact": 15,
                 "declared_input_edits_only": 15, "repeated_and_A_B_A_outputs_exact": True,
                 "identity_intervention_exact": True, "weights_and_sources_unchanged": True,
                 "prior_evidence_unchanged": True, "raw_recording_open_attempts": raw_attempts},
              "age4_comparison": [r for r in comparisons if r["loss_age"] == 4],
              "checkpoint_hashes": {m["checkpoint"]: m["sha256"] for m, _ in models},
              "source_hashes": training["source_hashes"],
              "environment": {"python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                              "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
              "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()}
    write_json(OUT / "report.json", report)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="constrained")
    for seed, ax in enumerate(axes):
        ax.plot(range(1, 6), predictions[0, seed, :, 4], "o-", color="tab:orange", label="Original received acc Y")
        ax.plot(range(1, 6), predictions[1, seed, :, 4], "s-", color="tab:blue", label="Copied acc Y = previous value")
        ax.axhline(anchor[4], color="gray", linestyle="--", label="LOCF gyro-Y anchor")
        ax.set(title=f"Residual seed {seed}", xlabel="Loss age (samples)", xticks=range(1, 6))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Gyro-Y prediction (rad/s)")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Copied-input sensitivity: acceleration Y 130.207 → 5.31869 m/s²\nOne selected burst; this is not a new accuracy benchmark", fontsize=11)
    fig.savefig(OUT / "sensitivity.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"checks": report["checks"], "age4_comparison": report["age4_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
