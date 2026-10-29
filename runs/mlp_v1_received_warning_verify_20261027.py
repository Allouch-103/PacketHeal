"""Independent saved arithmetic, mask/causal-index checks and warning plots."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from packetheal.evaluation import load_subject_masks

OUT = ROOT / "runs/mlp_v1_received_warning_20261027_retry1"
GROUPS = ("all", "warned", "unwarned", "unknown")
REGIMES = ("random_5pct", "burst_2_5pct", "burst_5_5pct", "burst_10_5pct")
CHANNELS = ("acc_x_m_s2", "acc_y_m_s2", "acc_z_m_s2", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s")
COUNTS, SUMS = ("hidden", "scored", "unavailable"), ("sum_abs", "sum_sq")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def close(a, b):
    if a is None or b is None:
        assert a is None and b is None
    else:
        assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)


def equal_record(a, b):
    for k in COUNTS:
        assert a[k] == b[k]
    for k in SUMS:
        for c in range(6): close(a[k][c], b[k][c])


def pooled(records):
    out = {k: sum(r[k] for r in records) for k in COUNTS}
    for k in SUMS:
        out[k] = [math.fsum(r[k][c] for r in records) for c in range(6)]
    n = out["scored"]
    out["mae"] = [v/n if n else None for v in out["sum_abs"]]
    out["rmse"] = [math.sqrt(v/n) if n else None for v in out["sum_sq"]]
    return out


def save_csv(name, rows):
    with (OUT / name).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    if (OUT / "independent_checks.json").exists():
        raise FileExistsError("Saved verification already exists")
    raw_attempts = []
    def deny_raw(event, args):
        if event == "open" and args and not isinstance(args[0], int):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.suffix == ".dat" or path.is_relative_to(ROOT / "data/raw"):
                raw_attempts.append(str(path)); raise PermissionError("Saved verifier must not open recordings")
    sys.addaudithook(deny_raw)
    report = json.loads((OUT / "report.json").read_text()); policy = json.loads((OUT / "policy.json").read_text())
    assert report["correctness_pass"] and policy == report["policy"]
    for path, expected in {**policy["frozen_inputs"], **policy["diagnostic_sources"], **report["source_hashes"]}.items():
        assert digest(ROOT / path) == expected
    for name, expected in report["model_hashes"].items():
        assert digest(ROOT / "runs/mlp_v1_train_20261027/models" / name) == expected
    previous_path = ROOT / "runs/mlp_v1_validation_20261027/validation.json"
    assert digest(previous_path) == policy["validation_sha256"]
    previous = json.loads(previous_path.read_text())
    previous_cases = {(r["mode"] if r["mode"] == "locf" else f"{r['mode']}_seed{r['seed']}", r["segment"], r["regime"]): r for r in previous["cases"]}
    names = ["locf"] + [f"{mode}_seed{s}" for mode in ("direct", "residual") for s in range(3)]
    cases = report["cases"]; indexed = {(r["algorithm"], r["segment"], r["regime"]): r for r in cases}
    assert len(cases) == len(indexed) == 2856
    assert set(indexed) == {(n, i, r) for n in names for i in range(102) for r in REGIMES}
    warning_cases = {(r["segment"], r["regime"]): r for r in report["warning_cases"]}
    assert len(warning_cases) == 408
    segments, masks = load_subject_masks(ROOT / "data/processed/pamap2_v1/loss_masks.csv", ROOT / "configs/pamap2_segments_v1.csv",
                                         ROOT / "configs/pamap2_split_v1.json", "validation", 107)
    received = {key: np.flatnonzero(~mask) for key, mask in masks.items()}
    seen, signal_counts, age_counts = set(), {}, {}
    caps = policy["thresholds_per_nominal_step"]
    with (OUT / "warnings.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            segment, regime = int(row["segment"]), row["regime"]
            raw_row, age = int(row["original_row"]), int(row["loss_age"])
            key = segment, regime
            start = int(segments[segment]["start_row"]); t = raw_row - start
            assert masks[key][t] and (key, t) not in seen; seen.add((key, t))
            pos = int(np.searchsorted(received[key], t, side="left")) - 1
            assert pos >= 1
            last, prev = int(received[key][pos]), int(received[key][pos-1])
            assert prev < last < t and raw_row - int(row["last_received_row"]) == age
            assert int(row["last_received_row"]) == start + last and int(row["previous_received_row"]) == start + prev
            assert int(row["elapsed_steps"]) == last-prev and int(row["known"]) == 1
            score = max(abs(float(row[f"last_acc_{axis}"]) - float(row[f"previous_acc_{axis}"])) / (last-prev) / caps[c] for c, axis in enumerate("xyz"))
            close(score, row["score"])
            assert bool(int(row["warned"])) == (score > 1)
            group = "warned" if score > 1 else "unwarned"
            for g in ("all", group):
                signal_counts[key, g] = signal_counts.get((key, g), 0) + 1
                age_counts[key, age, g] = age_counts.get((key, age, g), 0) + 1
    assert len(seen) == sum(int(m.sum()) for m in masks.values()) == 57432
    empty = {"hidden": 0, "scored": 0, "unavailable": 0, "sum_abs": [0.]*6, "sum_sq": [0.]*6}
    for row in cases:
        assert row["subject"] == 107
        key = row["segment"], row["regime"]
        original = previous_cases[row["algorithm"], *key]
        equal_record(row["groups"]["all"], original["metrics"])
        equal_record(pooled([row["groups"][g] for g in GROUPS[1:]]), row["groups"]["all"])
        for g in GROUPS:
            assert row["groups"][g]["hidden"] == signal_counts.get((key, g), 0)
            assert row["groups"][g]["hidden"] == row["groups"][g]["scored"] and row["groups"][g]["unavailable"] == 0
            assert warning_cases[key]["counts"][g] == signal_counts.get((key, g), 0)
            equal_record(pooled([groups[g] for groups in row["by_age"].values()]), row["groups"][g])
        for a, groups in row["by_age"].items():
            equal_record(groups["all"], original["by_age"][a])
            equal_record(pooled([groups[g] for g in GROUPS[1:]]), groups["all"])
            for g in GROUPS:
                assert groups[g]["hidden"] == age_counts.get((key, int(a), g), 0)
    for name in names:
        for regime in REGIMES:
            selected = [indexed[name, i, regime] for i in range(102)]
            for a, groups in {"all": report["pooled"][name][regime], **report["by_age"][name][regime]}.items():
                for g in GROUPS:
                    expected = pooled([r["groups"][g] if a == "all" else r["by_age"].get(a, {}).get(g, empty) for r in selected])
                    equal_record(expected, groups[g])
                    for metric in ("mae", "rmse"):
                        for c in range(6): close(expected[metric][c], groups[g][metric][c])
    with (OUT / "metrics.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4200
    seen_csv = set()
    for row in rows:
        name, regime, age, g = (row[k] for k in ("algorithm", "regime", "age", "group"))
        c = CHANNELS.index(row["channel"])
        identity = name, regime, age, g, c
        assert identity not in seen_csv; seen_csv.add(identity)
        groups = report["pooled"][name][regime] if age == "all" else report["by_age"][name][regime][age]
        m = groups[g]; assert row["unit"] == ("m/s^2" if c < 3 else "rad/s")
        for k in COUNTS: assert int(row[k]) == m[k]
        for k in (*SUMS, "mae", "rmse"): close(None if row[k] == "" else row[k], m[k][c])
        close(row["group_fraction_of_all_hidden"], m["hidden"] / groups["all"]["hidden"])
        close(row["hypothetical_unwarned_retention"], groups["unwarned"]["hidden"] / groups["all"]["hidden"])
    norm = json.loads((ROOT / "data/processed/pamap2_v1/normalization.json").read_text())["std_sample"]
    summaries, coverage_rows = [], []
    for name in names:
        summary = {"algorithm": name}
        for g in ("all", "warned", "unwarned"):
            values = [report["pooled"][name][r][g]["mae"][c] / norm[c] for r in REGIMES for c in range(6)]
            summary[f"normalized_mae_{g}"] = math.fsum(values) / 24
        summary["warned_to_unwarned_mae_ratio"] = summary["normalized_mae_warned"] / summary["normalized_mae_unwarned"]
        summary["retained_mae_change_percent_vs_all"] = 100 * (summary["normalized_mae_unwarned"] / summary["normalized_mae_all"] - 1)
        for metric in ("mae", "rmse"):
            total = higher = 0
            for regime in REGIMES:
                for groups in report["by_age"][name][regime].values():
                    if groups["warned"]["scored"] and groups["unwarned"]["scored"]:
                        total += 6
                        higher += sum(a > b for a, b in zip(groups["warned"][metric], groups["unwarned"][metric]))
            summary[f"same_age_warned_higher_{metric}_cells"] = higher
            summary["same_age_comparable_cells"] = total
        summaries.append(summary)
    for regime in REGIMES:
        for age, groups in {"all": report["pooled"]["locf"][regime], **report["by_age"]["locf"][regime]}.items():
            coverage_rows.append({"regime": regime, "age": age, **{g: groups[g]["hidden"] for g in GROUPS},
                                  "hypothetical_unwarned_retention": groups["unwarned"]["hidden"] / groups["all"]["hidden"]})
    save_csv("association_summary.csv", summaries); save_csv("coverage.csv", coverage_rows)
    residual = [r for r in summaries if r["algorithm"].startswith("residual_")]
    screen = all(r["normalized_mae_warned"] > r["normalized_mae_unwarned"] for r in residual)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    overall_coverage = [r for r in coverage_rows if r["age"] == "all"]
    positions = np.arange(4)
    axes[0].bar(positions, [100*r["hypothetical_unwarned_retention"] for r in overall_coverage], color="tab:blue")
    for i, r in enumerate(overall_coverage): axes[0].text(i, 100*r["hypothetical_unwarned_retention"]+1, f"{100*r['hypothetical_unwarned_retention']:.2f}%", ha="center")
    axes[0].set(ylim=(0, 110), ylabel="Hidden positions retained (%)", title="Hypothetical retention: keep unwarned")
    for seed in range(3):
        name = f"residual_seed{seed}"
        ratios = [math.fsum(report["pooled"][name][r]["warned"]["mae"][c]/norm[c] for c in range(6)) /
                  math.fsum(report["pooled"][name][r]["unwarned"]["mae"][c]/norm[c] for c in range(6)) for r in REGIMES]
        axes[1].bar(positions+(seed-1)*.23, ratios, width=.23, label=f"Residual seed {seed}")
    axes[1].axhline(1, color="black", linestyle="--"); axes[1].set(ylabel="Warned / unwarned normalized MAE", title="Error association (larger = greater separation)", yscale="log")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(positions, ["Random", "Burst 2", "Burst 5", "Burst 10"]); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Subject 107: fixed received-change warning; unchanged predictions", fontsize=12)
    fig.savefig(OUT / "warning_summary.png", dpi=160); plt.close(fig)
    access = report["recording_access"]
    assert access["allowed_subjects"] == [107] and not access["denied_recordings"]
    assert {Path(p).name for p in access["opened_recordings"]} == {"subject107.dat"}
    assert not raw_attempts
    checks = {"verification_pass": True, "case_identities": 2856, "warning_cases": 408,
              "warning_csv_rows": len(seen), "metrics_csv_rows": len(rows), "association_screen_pass": screen,
              "scope": "Independent saved math.fsum pooling, partitions, formulas, all metric CSV fields, all warning scores and past received indices against frozen masks",
              "raw_recordings_opened_by_verifier": 0, "association_summary": summaries,
              "coverage": overall_coverage, "report_sha256": digest(OUT / "report.json"), "verifier_sha256": digest(__file__),
              "limits": "Association screen is exploratory, not significance or calibrated abstention; scalar average received changes can hide within-gap motion"}
    with (OUT / "independent_checks.json").open("x") as stream: json.dump(checks, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"verification_pass": True, "association_screen_pass": screen, "coverage": overall_coverage, "residual_summaries": residual}, indent=2))


if __name__ == "__main__": main()
