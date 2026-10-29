"""Independent saved-result arithmetic and figures; no recording access."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "runs/mlp_v1_train_20261027"
VAL = ROOT / "runs/mlp_v1_validation_20261027"
REGIMES = ("random_5pct", "burst_2_5pct", "burst_5_5pct", "burst_10_5pct")
CHANNELS = ("acc_x_m_s2", "acc_y_m_s2", "acc_z_m_s2", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(first, second):
    assert math.isclose(float(first), float(second), rel_tol=1e-12, abs_tol=1e-12), (first, second)


def pooled(records):
    count = sum(r["scored"] for r in records)
    absolute = [math.fsum(r["sum_abs"][c] for r in records) for c in range(6)]
    squared = [math.fsum(r["sum_sq"][c] for r in records) for c in range(6)]
    return {"hidden": sum(r["hidden"] for r in records), "scored": count,
            "unavailable": sum(r["unavailable"] for r in records),
            "sum_abs": absolute, "sum_sq": squared,
            "mae": [a / count if count else None for a in absolute],
            "rmse": [math.sqrt(s / count) if count else None for s in squared]}


def compare(first, second):
    for field in ("hidden", "scored", "unavailable"):
        assert first[field] == second[field], (field, first[field], second[field])
    for field in ("sum_abs", "sum_sq", "mae", "rmse"):
        for a, b in zip(first[field], second[field]):
            if a is None or b is None:
                assert a is b
            else:
                close(a, b)


training = json.loads((TRAIN / "training.json").read_text())
validation = json.loads((VAL / "validation.json").read_text())
std = np.array(json.loads((ROOT / "data/processed/pamap2_v1/normalization.json").read_text())["std_sample"])
assert validation["training_report_sha256"] == sha(TRAIN / "training.json")
for model in training["models"]:
    assert sha(TRAIN / "models" / model["checkpoint"]) == model["sha256"]
for name, expected in training["source_hashes"].items():
    assert sha(ROOT / name) == expected
assert training["recording_access"]["allowed_subjects"] == list(range(101, 107))
assert validation["recording_access"]["allowed_subjects"] == [107]
assert not training["recording_access"]["denied_recordings"]
assert not validation["recording_access"]["denied_recordings"]
assert {Path(p).name for p in validation["recording_access"]["opened_recordings"]} == {"subject107.dat"}

cases = validation["cases"]
assert len(cases) == 2856
identities = {(c["mode"], c["seed"], c["subject"], c["segment"], c["regime"]) for c in cases}
expected = {(mode, seed, 107, segment, regime)
            for mode, seed in [("locf", None)] + [(m, s) for m in ("direct", "residual") for s in range(3)]
            for segment in range(102) for regime in REGIMES}
assert identities == expected and len(identities) == len(cases)
for case in cases:
    assert set(case["by_age"]) == {str(i) for i in range(1, 11)}
    compare(pooled(list(case["by_age"].values())), case["metrics"])
    if case["regime"] != "random_5pct":
        length = int(case["regime"].split("_")[1])
        counts = [case["by_age"][str(i)]["hidden"] for i in range(1, length + 1)]
        assert len(set(counts)) == 1
    assert case["metrics"]["unavailable"] == 0

for identity, regimes in validation["pooled"].items():
    mode = "locf" if identity == "locf" else identity.split("_seed")[0]
    seed = None if identity == "locf" else int(identity.split("_seed")[1])
    for regime in REGIMES:
        selected = [c["metrics"] for c in cases if (c["mode"], c["seed"], c["regime"]) == (mode, seed, regime)]
        assert len(selected) == 102
        compare(pooled(selected), regimes[regime])

with (VAL / "overall.csv").open(newline="") as stream:
    exported = list(csv.DictReader(stream))
assert len(exported) == 168
assert len({(r["algorithm"], r["regime"], r["channel"]) for r in exported}) == 168
for row in exported:
    c = CHANNELS.index(row["channel"])
    metric = validation["pooled"][row["algorithm"]][row["regime"]]
    for field in ("hidden", "scored", "unavailable"):
        assert int(row[field]) == metric[field]
    for field in ("mae", "rmse"):
        close(row[field], metric[field][c])
assert sha(VAL / "frozen_classical_overall.csv") == validation["baseline_reference"]["sha256"]


def objective(key):
    return float(np.mean([np.mean(np.asarray(validation["pooled"][key][r]["mae"]) / std) for r in REGIMES]))


baseline = objective("locf")
models = []
for metadata in training["models"]:
    mode, seed = metadata["mode"], metadata["seed"]
    key = f"{mode}_seed{seed}"
    bad_rmse = []
    mae_wins = rmse_wins = 0
    for regime in REGIMES:
        candidate = validation["pooled"][key][regime]
        reference = validation["pooled"]["locf"][regime]
        for c, channel in enumerate(CHANNELS):
            mae_wins += candidate["mae"][c] < reference["mae"][c]
            rmse_wins += candidate["rmse"][c] < reference["rmse"][c]
            change = 100 * (candidate["rmse"][c] / reference["rmse"][c] - 1)
            if change > 10:
                bad_rmse.append({"regime": regime, "channel": channel, "rmse_change_percent": change})
    score = objective(key)
    models.append({"mode": mode, "seed": seed, "normalized_mae": score,
                   "mae_change_percent": 100 * (score / baseline - 1),
                   "mae_improved_cells_of_24": int(mae_wins), "rmse_improved_cells_of_24": int(rmse_wins),
                   "rmse_regressions_above_10pct": bad_rmse})

means = {}
for mode in ("direct", "residual"):
    selected = [m for m in models if m["mode"] == mode]
    scores = [m["normalized_mae"] for m in selected]
    score = float(np.mean(scores))
    screen = validation["engineering_screen"]["modes"][mode]
    close(screen["mean_objective"], score)
    improving = sum(s < baseline for s in scores)
    rmse_guard = all(not m["rmse_regressions_above_10pct"] for m in selected)
    assert screen["improving_seeds"] == improving
    assert screen["rmse_guard_pass"] == rmse_guard
    assert screen["screen_pass"] == (score <= .95 * baseline and improving >= 2 and rmse_guard)
    means[mode] = {"normalized_mae_mean": score, "normalized_mae_seed_std_sample": float(np.std(scores, ddof=1)),
                   "mae_change_percent": 100 * (score / baseline - 1), "engineering_screen_pass": screen["screen_pass"]}

summary = {"verification_pass": True, "algorithm_cases": len(cases), "segment_regime_cases": 408,
           "saved_csv_rows_reconciled": len(exported), "loss_age_partitions_reconcile": True,
           "fixed_burst_age_cohorts_equal": True, "zero_unavailable_hidden_outputs": True,
           "checkpoint_hashes_unchanged": True, "held_out_recordings_opened": False,
           "locf_normalized_mae": baseline, "models": models, "mean_across_seeds": means,
           "scope": "Independent saved arithmetic, identities and provenance; no raw replay",
           "verifier_sha256": sha(__file__)}
with (VAL / "independent_checks.json").open("x") as stream:
    json.dump(summary, stream, indent=2, allow_nan=False)

matrices = {}
for mode in ("direct", "residual"):
    matrices[mode] = np.array([[100 * (np.mean([validation["pooled"][f"{mode}_seed{s}"][r]["mae"][c] for s in range(3)]) / validation["pooled"]["locf"][r]["mae"][c] - 1)
                               for r in REGIMES] for c in range(6)])
bound = max(1., max(float(np.abs(a).max()) for a in matrices.values()))
fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
for ax, mode in zip(axes, ("direct", "residual")):
    artist = ax.imshow(matrices[mode], cmap="RdBu_r", norm=TwoSlopeNorm(0, -bound, bound), aspect="auto")
    ax.set(title=mode.capitalize(), xticks=range(4), xticklabels=["Random", "Burst 2", "Burst 5", "Burst 10"],
           yticks=range(6), yticklabels=["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"])
    for c in range(6):
        for r in range(4):
            ax.text(r, c, f"{matrices[mode][c,r]:+.1f}%", ha="center", va="center", fontsize=9)
fig.colorbar(artist, ax=axes, label="MAE change relative to LOCF (%)")
fig.suptitle("Subject 107: mean MAE across 3 seeds; negative means lower error")
fig.savefig(VAL / "mae_comparison.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
for c, ax in enumerate(axes.flat):
    for mode, color in (("locf", "black"), ("direct", "tab:blue"), ("residual", "tab:orange")):
        by_seed = []
        for seed in ([None] if mode == "locf" else range(3)):
            selected = [case for case in cases if (case["mode"], case["seed"], case["regime"]) == (mode, seed, "burst_10_5pct")]
            by_seed.append([pooled([case["by_age"][str(age)] for case in selected])["mae"][c] for age in range(1, 11)])
        ax.plot(range(1, 11), np.mean(by_seed, axis=0), label=mode, color=color, marker="o", markersize=3)
    ax.set(title=CHANNELS[c], xlabel="Loss age (samples)", ylabel="MAE (m/s²)" if c < 3 else "MAE (rad/s)")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
fig.suptitle("Subject 107 / burst 10: same gap cohort at every age; MLP curves average 3 seeds")
fig.savefig(VAL / "burst10_by_age.png", dpi=160)
plt.close(fig)

with (VAL / "mean_seed_physical_metrics.csv").open("x", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["mode", "regime", "channel", "unit", "locf_mae", "mean_seed_mae", "locf_rmse", "mean_seed_rmse"])
    writer.writeheader()
    for mode in ("direct", "residual"):
        for regime in REGIMES:
            for c, channel in enumerate(CHANNELS):
                reference = validation["pooled"]["locf"][regime]
                writer.writerow({"mode": mode, "regime": regime, "channel": channel, "unit": "m/s²" if c < 3 else "rad/s",
                                 "locf_mae": reference["mae"][c], "locf_rmse": reference["rmse"][c],
                                 "mean_seed_mae": np.mean([validation["pooled"][f"{mode}_seed{s}"][regime]["mae"][c] for s in range(3)]),
                                 "mean_seed_rmse": np.mean([validation["pooled"][f"{mode}_seed{s}"][regime]["rmse"][c] for s in range(3)])})

outputs = ("validation.json", "overall.csv", "frozen_classical_overall.csv", "independent_checks.json", "mae_comparison.png", "burst10_by_age.png", "mean_seed_physical_metrics.csv")
with (VAL / "evidence.sha256").open("x") as stream:
    for name in outputs:
        path = VAL / name
        stream.write(sha(path) + "  " + str(path.relative_to(ROOT)) + "\n")
print(json.dumps(summary, indent=2))
