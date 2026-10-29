"""Targeted frozen gyro-Y/burst-5 failure reproduction; no training or tuning."""

from datetime import datetime, timezone
import csv
import json
import math
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
from packetheal.baselines import LOCF
from packetheal.diagnostics import large_change_flags, reconcile_partition, stratified_sums
from packetheal.evaluation import ReplayResult, load_segment_mask, segment_metadata
from packetheal.mlp import MLPPredictor
from packetheal.mlp_experiment import (
    CHANNELS, audited_replay, digest, error_record, normalization_from_root,
    recording_access_guard, streaming_replay, validate_training_record, verify_frozen, write_json,
)
from packetheal.splits import load_split_segment

TRAIN = ROOT / "runs/mlp_v1_train_20261027"
VALIDATION = ROOT / "runs/mlp_v1_validation_20261027/validation.json"
STRATA = ROOT / "runs/mlp_v1_transients_20261027/report.json"
OUT = ROOT / "runs/mlp_v1_gyroy_burst5_failure_20261027"
DATASET = ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset"
RAW = DATASET / "Protocol"
CONFIG = ROOT / "configs/pamap2_split_v1.json"
MANIFEST = ROOT / "configs/pamap2_segments_v1.csv"
MASKS = ROOT / "data/processed/pamap2_v1/loss_masks.csv"
REGIME, CHANNEL = "burst_5_5pct", 4
GROUPS = ("all", "large_change", "other")


def equal_strata(first, second):
    for key in ("hidden_count", "scored_count", "unavailable_count"):
        np.testing.assert_array_equal(first[key], second[key])
    for key in ("absolute_error_sum", "squared_error_sum"):
        np.testing.assert_allclose(first[key], second[key], rtol=1e-12, atol=1e-12)


def equal_metrics(first, second):
    for key in ("hidden", "scored", "unavailable"):
        assert first[key] == second[key]
    for key in ("sum_abs", "sum_sq", "mae", "rmse"):
        np.testing.assert_allclose(first[key], second[key], rtol=1e-12, atol=1e-12)


def save_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    verify_frozen(ROOT)
    training, all_models = validate_training_record(ROOT, TRAIN)
    assert digest(STRATA) == "81346ab14964188a05ece244cf375fbd5e31263e3ac076c60bfb3002552b3402"
    assert digest(VALIDATION) == "3f1dde9e7e02b3dc4e73093899252de567229f4e4be2c8a83ac84fe066464604"
    strata, validation = json.loads(STRATA.read_text()), json.loads(VALIDATION.read_text())
    assert strata["checks_passed"]
    assert validation["training_report_sha256"] == digest(TRAIN / "training.json")
    checked = {}
    for name in ("mlp_v1_train_20261027/evidence.sha256", "mlp_v1_validation_20261027/evidence.sha256",
                 "mlp_v1_transients_20261027/evidence.sha256"):
        manifest = ROOT / "runs" / name
        lines = manifest.read_text().splitlines()
        for line in lines:
            expected, path = line.split(maxsplit=1)
            resolved = (ROOT / path).resolve()
            assert resolved.is_relative_to(ROOT) and resolved.suffix != ".dat"
            assert digest(resolved) == expected
        checked[name] = {"entries": len(lines), "sha256": digest(manifest)}

    base_pooled = strata["pooled"]["locf"][REGIME]["large_change"]
    choices = [(strata["pooled"][f"residual_seed{s}"][REGIME]["large_change"]["rmse"][CHANNEL]
                / base_pooled["rmse"][CHANNEL] - 1, s) for s in range(3)]
    change, seed = max(choices, key=lambda item: (item[0], -item[1]))
    primary = f"residual_seed{seed}"
    indexed = {(r["algorithm"], r["segment"], r["regime"]): r for r in strata["cases"]}
    excess_by_segment = {
        i: indexed[primary, i, REGIME]["strata"]["large_change"]["squared_error_sum"][CHANNEL]
        - indexed["locf", i, REGIME]["strata"]["large_change"]["squared_error_sum"][CHANNEL]
        for i in range(102)
    }
    segment = max(excess_by_segment, key=lambda i: (excess_by_segment[i], -i))
    policy = {
        "subject": 107, "regime": REGIME, "channel": CHANNELS[CHANNEL], "channel_index": CHANNEL,
        "mode": "residual", "seed": seed, "segment": segment,
        "subgroup": "large_change", "subgroup_selection": "Gyro-Y/burst-5 weakness identified by prior full diagnostic",
        "seed_selection": "Largest flagged-subgroup RMSE ratio over LOCF; earliest seed breaks ties",
        "segment_selection": "Largest flagged-subgroup excess squared-error sum; earliest segment breaks ties",
        "point_selection": "Largest excess squared error among FLAGGED hidden positions in selected segment; earliest position breaks ties",
        "burst_selection": "The complete existing frozen burst containing that point; inspect all three residual seeds",
        "scope": "Targeted post-hoc failure, not representative sampling or hyperparameter selection",
        "checks": "Full-segment replay audits and saved sums; independent received-only inputs and three affine layers for every missing position in chosen burst",
        "limits": "Hidden truth defines scoring labels only; no motion/fault attribution or causal feature attribution",
        "validation_sha256": digest(VALIDATION), "strata_sha256": digest(STRATA),
        "training_sha256": digest(TRAIN / "training.json"), "diagnostic_script_sha256": digest(__file__),
        "primary_subgroup_rmse_change_percent": 100 * change, "prior_manifests": checked,
    }
    OUT.mkdir()
    write_json(OUT / "policy.json", policy)  # Pin selector before any raw recording access.
    models = {f"residual_seed{m['seed']}": (m, model) for m, model in all_models if m["mode"] == "residual"}
    original_parameters = {name: [p.copy() for p in model.parameters] for name, (_, model) in models.items()}
    norm = normalization_from_root(ROOT)
    thresholds = strata["policy"]["thresholds"]
    validation_cases = {(r["mode"] if r["mode"] == "locf" else f"{r['mode']}_seed{r['seed']}", r["segment"], r["regime"]): r
                        for r in validation["cases"]}

    with recording_access_guard(RAW, [107]) as access:
        recording_hash = digest(RAW / "subject107.dat")
        assert recording_hash == validation["recording_hash"] == strata["recording_sha256"]
        timestamps, truth = load_split_segment(DATASET, MANIFEST, CONFIG, "validation", 107, segment)
        metadata = segment_metadata(MANIFEST, 107, segment)
        missing = load_segment_mask(MASKS, MANIFEST, CONFIG, "validation", 107, segment, REGIME)
        original_start = int(metadata["start_row"])
        flags = large_change_flags(truth, thresholds)
        hidden = np.broadcast_to(missing[:, None], truth.shape)
        selections = {"all": hidden, "large_change": hidden & flags, "other": hidden & ~flags}
        predictors = {"locf": LOCF(6), **{name: MLPPredictor(model, norm) for name, (_, model) in models.items()}}
        results, groups = {}, {}
        for name, predictor in predictors.items():
            results[name] = ReplayResult(*audited_replay(predictor, truth, missing))
            equal_metrics(error_record(truth, results[name].values, missing), validation_cases[name, segment, REGIME]["metrics"])
            groups[name] = {g: stratified_sums(truth, missing, results[name], selections[g]) for g in GROUPS}
            reconcile_partition([groups[name]["large_change"], groups[name]["other"]], groups[name]["all"])
            for g in GROUPS:
                equal_strata(groups[name][g], indexed[name, segment, REGIME]["strata"][g])
            # Check future isolation through the end of this segment as well as
            # the original full-case half-prefix audit inside audited_replay.
            print(f"{name}: full-segment audits and all saved strata match", flush=True)
        eligible = np.flatnonzero(missing & flags[:, CHANNEL])
        selected_error = results[primary].values[eligible, CHANNEL] - truth[eligible, CHANNEL]
        base_error = results["locf"].values[eligible, CHANNEL] - truth[eligible, CHANNEL]
        event = int(eligible[int(np.argmax(selected_error ** 2 - base_error ** 2))])
        first, end = event, event + 1
        while first > 0 and missing[first - 1]:
            first -= 1
        while end < len(missing) and missing[end]:
            end += 1
        assert end - first == 5 and first >= 20 and first > 0 and not missing[first - 1]
        anchor = first - 1
        cutoff = end
        for name, predictor in predictors.items():
            changed, changed_mask = truth.copy(), missing.copy()
            changed[cutoff:] += .25
            changed_mask[cutoff:] = ~changed_mask[cutoff:]
            future = streaming_replay(predictor, changed, changed_mask)
            for a, b in zip((results[name].values, results[name].status, results[name].missing_run), future):
                np.testing.assert_array_equal(a[:cutoff], b[:cutoff])

        # Independently rebuild every slot using only indices received BEFORE
        # each prediction. Hidden samples/predictions never supply held values.
        manual_features, history_rows = [], []
        for t in range(first, end):
            rows = np.zeros((20, 8), dtype=np.float64)
            for slot, j in enumerate(range(t - 20, t)):
                last_real = int(np.flatnonzero(~missing[:j + 1])[-1])
                assert last_real <= j < t and not missing[last_real]
                rows[slot, :6] = (truth[last_real] - norm.mean) / norm.std
                rows[slot, 6:] = [float(not missing[j]), 1.0]
                history_rows.append({"prediction_original_row": original_start + t, "slot": slot,
                                     "slot_original_row": original_start + j, "received": int(not missing[j]),
                                     "last_real_original_row": original_start + last_real,
                                     **{f"held_{c}": float(truth[last_real, k]) for k, c in enumerate(CHANNELS)},
                                     **{f"z_{c}": float(rows[slot, k]) for k, c in enumerate(CHANNELS)}})
            manual_features.append(np.r_[rows.reshape(-1), (t - anchor) / 10].astype(np.float32))
        manual_features = np.array(manual_features, dtype=np.float32)
        corrections, normalized_corrections, first_layers, second_layers = {}, {}, {}, {}
        for name, (_, model) in models.items():
            predictor = MLPPredictor(model, norm)
            for t in range(first):
                predictor.step(None if missing[t] else truth[t])
            corrections[name], normalized_corrections[name], first_layers[name], second_layers[name] = [], [], [], []
            for offset, t in enumerate(range(first, end)):
                features, ready = predictor.state.prepare()
                assert ready
                np.testing.assert_array_equal(features, manual_features[offset])
                np.testing.assert_array_equal(predictor.state.anchor, truth[anchor])
                assert predictor.state.age + 1 == t - anchor == int(results[name].missing_run[t])
                w1, b1, w2, b2, w3, b3 = model.parameters
                h1 = np.maximum(features[None] @ w1 + b1, 0)
                h2 = np.maximum(h1 @ w2 + b2, 0)
                z = (h2 @ w3 + b3)[0]
                np.testing.assert_array_equal(z, model.predict(features[None])[0])
                correction = z.astype(np.float64) * norm.std
                manual = truth[anchor].copy()
                nonzero = z != 0
                manual[nonzero] += correction[nonzero]
                np.testing.assert_array_equal(manual, results[name].values[t])
                np.testing.assert_array_equal(predictor.step(None).values, manual)
                corrections[name].append(correction)
                normalized_corrections[name].append(z)
                first_layers[name].append(h1[0])
                second_layers[name].append(h2[0])
        # Direct raw-row lineage check: last received, five hidden, next received.
        requested = [anchor, *range(first, end)] + ([end] if end < len(truth) else [])
        row_map = {original_start + t: t for t in requested}
        raw_rows = []
        with (RAW / "subject107.dat").open() as stream:
            for raw_row, line in enumerate(stream, 1):
                if raw_row in row_map:
                    columns = line.split()
                    t = row_map[raw_row]
                    np.testing.assert_array_equal(np.array([float(columns[c]) for c in (4, 5, 6, 10, 11, 12)]), truth[t])
                    assert float(columns[0]) == timestamps[t]
                    raw_rows.append(raw_row)
                if raw_row >= max(row_map):
                    break
        assert raw_rows == sorted(row_map)
        assert digest(RAW / "subject107.dat") == recording_hash

    verify_frozen(ROOT)
    validate_training_record(ROOT, TRAIN)
    assert digest(STRATA) == policy["strata_sha256"] and digest(VALIDATION) == policy["validation_sha256"]
    for name, (_, model) in models.items():
        for a, b in zip(original_parameters[name], model.parameters):
            np.testing.assert_array_equal(a, b)
    burst_rows = []
    for t in range(first, end):
        row = {"original_row": original_start + t, "local_index": t, "timestamp_s": float(timestamps[t]),
               "loss_age": int(results[primary].missing_run[t]), "rapid_change_gyro_y": int(flags[t, CHANNEL]),
               "recorded_truth": float(truth[t, CHANNEL]), "anchor": float(truth[anchor, CHANNEL]),
               "required_correction": float(truth[t, CHANNEL] - truth[anchor, CHANNEL])}
        for name in predictors:
            predicted = float(results[name].values[t, CHANNEL])
            row.update({f"{name}_prediction": predicted, f"{name}_signed_error": predicted - float(truth[t, CHANNEL]),
                        f"{name}_correction": predicted - float(truth[anchor, CHANNEL])})
        burst_rows.append(row)
    low, high = max(0, first - 25), min(len(truth), end + 21)
    trace = [{"original_row": original_start + t, "local_index": t, "timestamp_s": float(timestamps[t]),
              "relative_event_time_s": float(timestamps[t] - timestamps[event]), "received": int(not missing[t]),
              "loss_age": int(results[primary].missing_run[t]), "rapid_change_gyro_y": int(flags[t, CHANNEL]),
              "recorded_truth": float(truth[t, CHANNEL]),
              **{f"{name}_prediction": float(result.values[t, CHANNEL]) for name, result in results.items()}}
             for t in range(low, high)]
    contributions = []
    for i in range(102):
        base = indexed["locf", i, REGIME]["strata"]["large_change"]
        a = indexed[primary, i, REGIME]["strata"]["large_change"]
        contributions.append({"segment": i, "flagged_count": base["scored_count"][CHANNEL],
                              "locf_squared_error_sum": base["squared_error_sum"][CHANNEL],
                              "primary_squared_error_sum": a["squared_error_sum"][CHANNEL],
                              "excess_squared_error_sum": excess_by_segment[i]})
    save_csv(OUT / "burst.csv", burst_rows)
    save_csv(OUT / "history.csv", history_rows)
    save_csv(OUT / "trace.csv", trace)
    save_csv(OUT / "segment_contributions.csv", contributions)
    with (OUT / "features.npz").open("xb") as stream:
        arrays = {"features": manual_features, "normalization_mean": norm.mean, "normalization_std": norm.std,
                  "anchor": truth[anchor], "original_rows": np.arange(first, end) + original_start}
        for name in models:
            arrays.update({f"{name}_physical_corrections": np.array(corrections[name]),
                           f"{name}_normalized_corrections": np.array(normalized_corrections[name]),
                           f"{name}_h1": np.array(first_layers[name]), f"{name}_h2": np.array(second_layers[name])})
        np.savez_compressed(stream, **arrays)
    subgroup = {name: strata["pooled"][name][REGIME]["large_change"] for name in predictors}
    event_values = {name: float(results[name].values[event, CHANNEL]) for name in predictors}
    event_excess = (event_values[primary] - truth[event, CHANNEL]) ** 2 - (event_values["locf"] - truth[event, CHANNEL]) ** 2
    cell_excess = subgroup[primary]["squared_error_sum"][CHANNEL] - subgroup["locf"]["squared_error_sum"][CHANNEL]
    event_record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "policy": policy,
        "segment_metadata": metadata, "event_original_row": original_start + event,
        "event_local_index": event, "event_timestamp_s": float(timestamps[event]),
        "loss_age": int(results[primary].missing_run[event]), "last_received_original_row": original_start + anchor,
        "anchor_value": float(truth[anchor, CHANNEL]), "recorded_hidden_value": float(truth[event, CHANNEL]),
        "required_correction": float(truth[event, CHANNEL] - truth[anchor, CHANNEL]),
        "predictions": event_values, "physical_corrections": {name: event_values[name] - float(truth[anchor, CHANNEL]) for name in predictors},
        "absolute_errors": {name: abs(value - float(truth[event, CHANNEL])) for name, value in event_values.items()},
        "burst_original_rows": list(range(original_start + first, original_start + end)),
        "event_excess_squared_error": float(event_excess), "segment_flagged_excess_squared_error": excess_by_segment[segment],
        "subgroup_net_excess_squared_error": cell_excess,
        "event_fraction_of_subgroup_net_increase": float(event_excess / cell_excess),
        "subgroup_metrics": subgroup, "segment_strata": groups,
        "checks": {"all_saved_segment_metrics_and_strata_match": True, "full_segment_causal_audits_pass": True,
                   "future_after_burst_prefix_isolation": True, "independent_feature_matches": 15,
                   "independent_network_and_denormalization_matches": 15, "constant_real_anchor_across_burst": True,
                   "raw_rows_match": raw_rows, "parameters_checkpoints_sources_frozen_inputs_unchanged": True,
                   "raw_hash_unchanged": True},
        "recording_sha256": recording_hash, "recording_access": access,
        "source_hashes": training["source_hashes"],
        "checkpoint_hashes": {m["checkpoint"]: m["sha256"] for m, _ in models.values()},
        "environment": {"python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip(),
        "limits": ["Worst-excess flagged sample in a deliberately selected failure segment", "No causal attribution to individual input features",
                   "No confidence estimate, clipping, retuning, or phase acceptance", "Recordings 108/109 remain untouched"],
    }
    write_json(OUT / "event.json", event_record)
    window = np.arange(low, high)
    x = timestamps[window] - timestamps[event]
    received = ~missing[window]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    axes[0].plot(x, truth[window, CHANNEL], "--", color="gray", label="Recorded truth (post-hoc)")
    axes[0].scatter(x[received], truth[window[received], CHANNEL], s=23, color="black", label="Received samples", zorder=5)
    style = {"locf": ("tab:blue", "x"), "residual_seed0": ("tab:orange", "D"),
             "residual_seed1": ("tab:green", "+"), "residual_seed2": ("tab:purple", "1")}
    for name, result in results.items():
        color, marker = style[name]
        ix = window[~received]
        axes[0].scatter(x[~received], result.values[ix, CHANNEL], color=color, marker=marker, s=37, label=name)
        axes[1].scatter(x[~received], np.abs(result.values[ix, CHANNEL] - truth[ix, CHANNEL]), color=color, marker=marker, s=37, label=name)
    axes[0].set_ylabel("Gyro Y (rad/s)")
    axes[1].set(xlabel="Time relative to selected hidden sample (s)", ylabel="Absolute error (rad/s)")
    for ax in axes:
        ax.axvspan(float(timestamps[first] - timestamps[event]) - .005,
                   float(timestamps[end-1] - timestamps[event]) + .005, color="gray", alpha=.12)
        ax.axvline(0, color="red", linestyle=":")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle(f"Gyro-Y burst-5 failure: primary residual seed {seed}, segment {segment}, row {original_start + event}, age {event-anchor}")
    fig.savefig(OUT / "failure.png", dpi=160)
    plt.close(fig)
    print(json.dumps({key: event_record[key] for key in ("event_original_row", "loss_age", "anchor_value", "recorded_hidden_value", "required_correction", "predictions", "absolute_errors", "physical_corrections", "event_excess_squared_error", "segment_flagged_excess_squared_error", "subgroup_net_excess_squared_error", "event_fraction_of_subgroup_net_increase", "checks")}, indent=2))


if __name__ == "__main__":
    main()
