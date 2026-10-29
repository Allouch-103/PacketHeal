"""Pinned, validation-only residual failure reproduction; no training/retuning."""

from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from packetheal.baselines import LOCF
from packetheal.mlp import MLP, MLPPredictor
from packetheal.mlp_experiment import (
    audited_replay, digest, error_record, normalization_from_root,
    recording_access_guard, validate_training_record, verify_frozen, write_json,
)
from packetheal.evaluation import load_segment_mask, segment_metadata
from packetheal.splits import load_split_segment

TRAIN = ROOT / "runs/mlp_v1_train_20261027"
REPORT = ROOT / "runs/mlp_v1_validation_20261027/validation.json"
OUT = ROOT / "runs/mlp_v1_residual_failure_20261027"
DATASET = ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset"
RAW = DATASET / "Protocol"
CONFIG = ROOT / "configs/pamap2_split_v1.json"
MANIFEST = ROOT / "configs/pamap2_segments_v1.csv"
MASKS = ROOT / "data/processed/pamap2_v1/loss_masks.csv"
CHANNELS = ("acc_x_m_s2", "acc_y_m_s2", "acc_z_m_s2", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s")


def equal_metrics(first, second):
    for key in ("hidden", "scored", "unavailable"):
        assert first[key] == second[key]
    for key in ("sum_abs", "sum_sq", "mae", "rmse"):
        np.testing.assert_allclose(first[key], second[key], rtol=1e-12, atol=1e-12)


if OUT.exists():
    raise FileExistsError(OUT)
verify_frozen(ROOT)
training, models = validate_training_record(ROOT, TRAIN)
report = json.loads(REPORT.read_text())
verification = json.loads((REPORT.parent / "independent_checks.json").read_text())
assert verification["verification_pass"]
assert report["training_report_sha256"] == digest(TRAIN / "training.json")
report_entries = (REPORT.parent / "evidence.sha256").read_text().splitlines()
expected_report = next(line.split()[0] for line in report_entries if line.split(maxsplit=1)[1] == str(REPORT.relative_to(ROOT)))
assert digest(REPORT) == expected_report

# Select the worst residual cell using only previously saved aggregate metrics.
choices = []
for seed in range(3):
    for regime, metrics in report["pooled"][f"residual_seed{seed}"].items():
        base = report["pooled"]["locf"][regime]
        choices.extend((metrics["rmse"][c] / base["rmse"][c] - 1, seed, regime, c) for c in range(6))
change, seed, regime, channel = max(choices, key=lambda x: (x[0], -x[1], x[2], -x[3]))
base_cases = {case["segment"]: case for case in report["cases"] if case["mode"] == "locf" and case["regime"] == regime}
selected_cases = [case for case in report["cases"] if (case["mode"], case["seed"], case["regime"]) == ("residual", seed, regime)]
chosen = max(selected_cases, key=lambda case: (case["metrics"]["sum_sq"][channel] - base_cases[case["segment"]]["metrics"]["sum_sq"][channel], -case["segment"]))
segment = chosen["segment"]
OUT.mkdir()
policy = {
    "mode": "residual", "seed": seed, "subject": 107, "regime": regime,
    "channel": CHANNELS[channel], "channel_index": channel, "segment": segment,
    "cell_selection": "Largest positive pooled RMSE ratio over LOCF among residual cells/seeds",
    "segment_selection": "Largest excess squared-error sum over LOCF; earliest segment breaks ties",
    "point_selection": "Largest hidden-position excess squared error in selected segment; earliest breaks ties",
    "scope": "Targeted post-hoc failure, not representative sampling; no retuning",
    "validation_sha256": expected_report, "pooled_rmse_change_percent": 100 * change,
}
write_json(OUT / "policy.json", policy)
metadata, model = next((metadata, model) for metadata, model in models if (metadata["mode"], metadata["seed"]) == ("residual", seed))
norm = normalization_from_root(ROOT)
model_parameters = [p.copy() for p in model.parameters]

with recording_access_guard(RAW, [107]) as access:
    before = digest(RAW / "subject107.dat")
    assert before == report["recording_hash"]
    timestamps, truth = load_split_segment(DATASET, MANIFEST, CONFIG, "validation", 107, segment)
    segment_info = segment_metadata(MANIFEST, 107, segment)
    missing = load_segment_mask(MASKS, MANIFEST, CONFIG, "validation", 107, segment, regime)
    values, statuses, ages = audited_replay(MLPPredictor(model, norm), truth, missing)
    locf_values, locf_statuses, locf_ages = audited_replay(LOCF(6), truth, missing)
    equal_metrics(error_record(truth, values, missing), chosen["metrics"])
    equal_metrics(error_record(truth, locf_values, missing), base_cases[segment]["metrics"])
    np.testing.assert_array_equal(ages, locf_ages)

    hidden = np.flatnonzero(missing)
    errors = values[hidden, channel] - truth[hidden, channel]
    base_errors = locf_values[hidden, channel] - truth[hidden, channel]
    excess = errors ** 2 - base_errors ** 2
    selected = int(np.argmax(excess))
    event = int(hidden[selected])
    last = int(np.flatnonzero(~missing[:event])[-1])
    original_start = int(segment_info["start_row"])

    # Reconstruct the network input independently from past RECEIVED indices.
    # Do not read hidden truth or future values to fill any feature row.
    independent_rows = np.zeros((20, 8), dtype=np.float64)
    history = []
    for slot, t in enumerate(range(event - 20, event)):
        assert t >= 0
        last_real = int(np.flatnonzero(~missing[:t + 1])[-1])
        held = truth[last_real].copy()
        independent_rows[slot, :6] = (held - norm.mean) / norm.std
        independent_rows[slot, 6:] = [float(not missing[t]), 1.0]
        history.append({"local_index": t, "original_row": original_start + t,
                        "last_real_original_row": original_start + last_real,
                        "received": int(not missing[t]),
                        **{f"held_{name}": float(held[c]) for c, name in enumerate(CHANNELS)},
                        **{f"z_{name}": float(independent_rows[slot, c]) for c, name in enumerate(CHANNELS)}})
    manual_features = np.r_[independent_rows.reshape(-1), min(event - last, 10) / 10].astype(np.float32)
    predictor = MLPPredictor(model, norm)
    for t in range(event):
        predictor.step(None if missing[t] else truth[t])
    features, eligible = predictor.state.prepare()
    assert eligible
    np.testing.assert_array_equal(features, manual_features)
    np.testing.assert_array_equal(predictor.state.anchor, truth[last])
    assert predictor.state.age + 1 == event - last == int(ages[event])

    # Independent expression of the checkpoint's three affine layers.
    w1, b1, w2, b2, w3, b3 = model.parameters
    h1 = np.maximum(manual_features[None] @ w1 + b1, 0)
    h2 = np.maximum(h1 @ w2 + b2, 0)
    normalized_correction = (h2 @ w3 + b3)[0]
    np.testing.assert_array_equal(normalized_correction, model.predict(features[None])[0])
    physical_correction = normalized_correction.astype(np.float64) * norm.std
    manual_prediction = truth[last].copy()
    changed = normalized_correction != 0
    manual_prediction[changed] += physical_correction[changed]
    np.testing.assert_array_equal(manual_prediction, values[event])

    # Read three selected original rows directly as a source-lineage check.
    next_real = event + 1
    while next_real < len(truth) and missing[next_real]:
        next_real += 1
    source_indices = [last, event] + ([next_real] if next_real < len(truth) else [])
    requested = {original_start + t: t for t in source_indices}
    raw_checked = []
    with (RAW / "subject107.dat").open() as stream:
        for row_number, line in enumerate(stream, start=1):
            if row_number in requested:
                columns = line.split()
                local = requested[row_number]
                observed = np.array([float(columns[c]) for c in [4, 5, 6, 10, 11, 12]])
                np.testing.assert_array_equal(observed, truth[local])
                assert float(columns[0]) == timestamps[local]
                raw_checked.append(row_number)
            if row_number >= max(requested):
                break
    assert len(raw_checked) == len(requested)
    assert digest(RAW / "subject107.dat") == before

for before_parameter, after_parameter in zip(model_parameters, model.parameters):
    np.testing.assert_array_equal(before_parameter, after_parameter)
assert digest(TRAIN / "models" / metadata["checkpoint"]) == metadata["sha256"]
verify_frozen(ROOT)

low, high = max(0, event - 30), min(len(truth), event + 21)
trace = []
for t in range(low, high):
    trace.append({"local_index": t, "original_row": original_start + t,
                  "time_relative_event_s": float(timestamps[t] - timestamps[event]),
                  "timestamp_s": float(timestamps[t]), "received": int(not missing[t]),
                  "loss_age": int(ages[t]), "recorded_truth": float(truth[t, channel]),
                  "residual_prediction": float(values[t, channel]), "locf_prediction": float(locf_values[t, channel]),
                  "residual_signed_error": float(values[t, channel] - truth[t, channel]) if missing[t] else "",
                  "locf_signed_error": float(locf_values[t, channel] - truth[t, channel]) if missing[t] else ""})
for name, rows in (("trace.csv", trace), ("history.csv", history)):
    with (OUT / name).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
with (OUT / "features.npz").open("xb") as stream:
    np.savez_compressed(stream, features=features, normalized_correction=normalized_correction,
                        physical_correction=physical_correction, manual_prediction=manual_prediction)

total_excess = report["pooled"][f"residual_seed{seed}"][regime]["sum_sq"][channel] - report["pooled"]["locf"][regime]["sum_sq"][channel]
summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(), **policy,
           "segment_metadata": segment_info, "event_local_index": event,
           "event_original_row": original_start + event, "event_timestamp_s": float(timestamps[event]),
           "loss_age": int(ages[event]), "last_received_original_row": original_start + last,
           "last_received_value": float(truth[last, channel]), "recorded_hidden_value": float(truth[event, channel]),
           "residual_prediction": float(values[event, channel]), "locf_prediction": float(locf_values[event, channel]),
           "predicted_correction": float(physical_correction[channel]),
           "required_correction": float(truth[event, channel] - truth[last, channel]),
           "normalized_predicted_correction": float(normalized_correction[channel]),
           "residual_absolute_error": float(abs(errors[selected])), "locf_absolute_error": float(abs(base_errors[selected])),
           "event_excess_squared_error": float(excess[selected]),
           "segment_excess_squared_error": chosen["metrics"]["sum_sq"][channel] - base_cases[segment]["metrics"]["sum_sq"][channel],
           "cell_net_excess_squared_error": total_excess,
           "segment_metrics": {"residual": chosen["metrics"], "locf": base_cases[segment]["metrics"]},
           "segment_missing_count": len(hidden),
           "segment_positions_with_lower_absolute_error": int(np.sum(np.abs(errors) < np.abs(base_errors))),
           "segment_positions_with_higher_absolute_error": int(np.sum(np.abs(errors) > np.abs(base_errors))),
           "segment_positions_with_equal_absolute_error": int(np.sum(np.abs(errors) == np.abs(base_errors))),
           "checks": {"saved_case_metrics_match": True, "independent_history_features_match": True,
                      "manual_network_and_denormalization_match": True, "raw_rows_match": raw_checked,
                      "checkpoint_parameters_unchanged": True, "recording_hash_unchanged": True},
           "recording_access": access, "recording_sha256": before,
           "checkpoint": metadata["checkpoint"], "checkpoint_sha256": metadata["sha256"],
           "diagnostic_source_sha256": digest(__file__),
           "interpretation_limit": "Targeted worst-excess event; not representative accuracy or physical-fault diagnosis"}
write_json(OUT / "event.json", summary)

window = np.arange(low, high)
x = timestamps[window] - timestamps[event]
received = ~missing[window]
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
axes[0].plot(x, truth[window, channel], "--", color="gray", label="Recorded truth (post-hoc)")
axes[0].scatter(x[received], truth[window[received], channel], color="black", s=22, label="Received samples", zorder=4)
axes[0].scatter(x[~received], values[window[~received], channel], color="tab:orange", marker="D", s=35, label="Residual MLP")
axes[0].scatter(x[~received], locf_values[window[~received], channel], color="tab:blue", marker="x", s=45, label="LOCF")
axes[0].set_ylabel("Acceleration X (m/s²)")
axes[0].legend(loc="best", fontsize=9)
axes[1].plot(x[~received], np.abs(values[window[~received], channel] - truth[window[~received], channel]), "o", color="tab:orange", label="Residual absolute error")
axes[1].plot(x[~received], np.abs(locf_values[window[~received], channel] - truth[window[~received], channel]), "x", color="tab:blue", label="LOCF absolute error")
axes[1].set(xlabel="Time relative to selected hidden sample (s)", ylabel="Absolute error (m/s²)")
axes[1].legend(loc="best", fontsize=9)
for ax in axes:
    ax.axvline(0, color="red", linestyle=":", label="Selected event")
    ax.axvspan(-.005, .005, color="red", alpha=.08)
    ax.grid(alpha=.2)
fig.suptitle(f"Targeted residual failure: seed {seed}, segment {segment}, row {original_start + event}, age {ages[event]}")
fig.savefig(OUT / "failure.png", dpi=160)
plt.close(fig)
with (OUT / "artifacts.sha256").open("x") as stream:
    for name in ("policy.json", "event.json", "trace.csv", "history.csv", "features.npz", "failure.png"):
        stream.write(digest(OUT / name) + "  " + str((OUT / name).relative_to(ROOT)) + "\n")
print(json.dumps({key: value for key, value in summary.items() if key not in ("segment_metrics", "recording_access")}, indent=2))
