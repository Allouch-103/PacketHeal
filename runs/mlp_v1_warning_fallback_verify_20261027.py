"""Independent saved-sum fallback verification and plots; no raw data loading."""

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
OUT = ROOT / "runs/mlp_v1_warning_fallback_20261027"
PARENT = ROOT / "runs/mlp_v1_received_warning_20261027_retry1"
REGIMES = ("random_5pct", "burst_2_5pct", "burst_5_5pct", "burst_10_5pct")
CHANNELS = ("acc_x_m_s2", "acc_y_m_s2", "acc_z_m_s2", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s")
NAMES = tuple(f"residual_seed{s}" for s in range(3))


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


def sum_records(records):
    totals = {k: sum(r[k] for r in records) for k in ("hidden", "scored", "unavailable")}
    for k in ("sum_abs", "sum_sq"):
        totals[k] = [math.fsum(r[k][c] for r in records) for c in range(6)]
    n = totals["scored"]
    totals["mae"] = [v/n if n else None for v in totals["sum_abs"]]
    totals["rmse"] = [math.sqrt(v/n) if n else None for v in totals["sum_sq"]]
    return totals


def check_record(actual, expected):
    for k in ("hidden", "scored", "unavailable"):
        assert actual[k] == expected[k]
    for k in ("sum_abs", "sum_sq", "mae", "rmse"):
        for c in range(6): close(actual[k][c], expected[k][c])


def main():
    if (OUT / "independent_checks.json").exists():
        raise FileExistsError("Verification outputs already exist")
    raw_attempts = []
    def deny_raw(event, args):
        if event == "open" and args and not isinstance(args[0], int):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.suffix == ".dat" or path.is_relative_to(ROOT / "data/raw"):
                raw_attempts.append(str(path)); raise PermissionError("Raw recordings prohibited in verifier")
    sys.addaudithook(deny_raw)
    report = json.loads((OUT / "report.json").read_text()); policy = json.loads((OUT / "policy.json").read_text())
    assert report["policy"] == policy and report["correctness_pass"]
    assert digest(ROOT / "runs/mlp_v1_warning_fallback_20261027.py") == policy["runner_sha256"]
    assert digest(PARENT / "evidence.sha256") == policy["parent_manifest_sha256"]
    for name, expected in {**policy["parent_hashes"], **policy["frozen_inputs"], **report["original_source_hashes"]}.items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and path.suffix != ".dat" and digest(path) == expected
    for name, expected in report["model_hashes"].items():
        assert digest(ROOT / "runs/mlp_v1_train_20261027/models" / name) == expected
    parent = json.loads((PARENT / "report.json").read_text())
    parent_cases = {(r["algorithm"], r["segment"], r["regime"]): r for r in parent["cases"]}
    cases = report["cases"]
    indexed = {(r["algorithm"], r["segment"], r["regime"]): r for r in cases}
    assert len(cases) == len(indexed) == report["case_count"] == 1224
    assert set(indexed) == {(n, i, r) for n in NAMES for i in range(102) for r in REGIMES}
    age_case_checks = 0
    for row in cases:
        assert row["subject"] == 107
        name, segment, regime = row["algorithm"], row["segment"], row["regime"]
        original, base = parent_cases[name, segment, regime], parent_cases["locf", segment, regime]
        assert row["routed_to_locf"] == base["groups"]["warned"]["hidden"] + base["groups"]["unknown"]["hidden"]
        assert row["routed_to_mlp"] == original["groups"]["unwarned"]["hidden"]
        assert row["routed_to_locf"] + row["routed_to_mlp"] == original["groups"]["all"]["hidden"]
        levels = {"all": row["metrics"], **row["by_age"]}
        for age, actual in levels.items():
            m = original["groups"] if age == "all" else original["by_age"][age]
            b = base["groups"] if age == "all" else base["by_age"][age]
            expected = sum_records([m["unwarned"], b["warned"], b["unknown"]])
            check_record(actual, expected)
            assert actual["scored"] == actual["hidden"] and actual["unavailable"] == 0
            if age != "all": age_case_checks += 1
        check_record(row["metrics"], sum_records(list(row["by_age"].values())))
    std = json.loads((ROOT / "data/processed/pamap2_v1/normalization.json").read_text())["std_sample"]
    summary = []
    for name in NAMES:
        for regime in REGIMES:
            selected = [indexed[name, i, regime] for i in range(102)]
            check_record(report["pooled"][name][regime], sum_records([r["metrics"] for r in selected]))
            for age, actual in report["by_age"][name][regime].items():
                check_record(actual, sum_records([r["by_age"][age] for r in selected if age in r["by_age"]]))
            original = parent["pooled"][name][regime]
            base = parent["pooled"]["locf"][regime]
            # Independent identity: hybrid-original = LOCF-MLP on routed groups.
            for field in ("sum_abs", "sum_sq"):
                for c in range(6):
                    delta = math.fsum(base[g][field][c] - original[g][field][c] for g in ("warned", "unknown"))
                    close(report["pooled"][name][regime][field][c] - original["all"][field][c], delta)
        original = math.fsum(parent["pooled"][name][r]["all"]["mae"][c] / std[c] for r in REGIMES for c in range(6)) / 24
        hybrid = math.fsum(report["pooled"][name][r]["mae"][c] / std[c] for r in REGIMES for c in range(6)) / 24
        saved_summary = next(s for s in report["summaries"] if s["algorithm"] == name)
        close(original, saved_summary["original_normalized_mae"]); close(hybrid, saved_summary["hybrid_normalized_mae"])
        close(100 * (hybrid/original - 1), saved_summary["change_percent"])
        assert saved_summary["improves"] == (hybrid < original)
        for metric in ("mae", "rmse"):
            assert saved_summary[f"{metric}_cells_improved"] == sum(report["pooled"][name][r][metric][c] < parent["pooled"][name][r]["all"][metric][c] for r in REGIMES for c in range(6))
        close(saved_summary["largest_rmse_increase_percent"], max(100*(report["pooled"][name][r]["rmse"][c]/parent["pooled"][name][r]["all"]["rmse"][c]-1) for r in REGIMES for c in range(6)))
        summary.append(saved_summary)
    assert report["primary_usefulness_pass"] == all(s["improves"] for s in summary)
    with (OUT / "comparison.csv").open(newline="") as stream: rows = list(csv.DictReader(stream))
    assert len(rows) == 450
    identities = set()
    for row in rows:
        name, regime, age = row["algorithm"], row["regime"], row["age"]
        c = CHANNELS.index(row["channel"]); identity = name, regime, age, c
        assert identity not in identities; identities.add(identity)
        actual = report["pooled"][name][regime] if age == "all" else report["by_age"][name][regime][age]
        original = parent["pooled"][name][regime]["all"] if age == "all" else parent["by_age"][name][regime][age]["all"]
        base = parent["pooled"]["locf"][regime]["all"] if age == "all" else parent["by_age"]["locf"][regime][age]["all"]
        assert row["unit"] == ("m/s^2" if c < 3 else "rad/s")
        for k in ("hidden", "scored", "unavailable"): assert int(row[k]) == actual[k]
        close(row["coverage"], actual["scored"]/actual["hidden"])
        for metric in ("mae", "rmse"):
            close(row[f"hybrid_{metric}"], actual[metric][c]); close(row[f"original_{metric}"], original[metric][c]); close(row[f"locf_{metric}"], base[metric][c])
            close(row[f"{metric}_change_percent"], 100*(actual[metric][c]/original[metric][c]-1))
        for field in ("sum_abs", "sum_sq"):
            close(row[f"original_{field}"], original[field][c]); close(row[f"hybrid_{field}"], actual[field][c])
    assert not raw_attempts and not report["raw_recording_open_attempts"]
    # Charts are summaries only; every seed's physical-cell values remain in CSV.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    x = np.arange(3)
    axes[0].bar(x-.18, [s["original_normalized_mae"] for s in summary], .36, label="Original residual MLP")
    axes[0].bar(x+.18, [s["hybrid_normalized_mae"] for s in summary], .36, label="LOCF on warning")
    locf = math.fsum(parent["pooled"]["locf"][r]["all"]["mae"][c]/std[c] for r in REGIMES for c in range(6))/24
    axes[0].axhline(locf, color="gray", linestyle="--", label="Always LOCF")
    axes[0].set(xticks=x, xticklabels=["Seed 0", "Seed 1", "Seed 2"], ylabel="Normalized MAE (lower is better)", title="Full coverage, same hidden positions")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=.2)
    changes = np.array([[100*(math.fsum(report["pooled"][n][r]["rmse"][c] for n in NAMES)/math.fsum(parent["pooled"][n][r]["all"]["rmse"][c] for n in NAMES)-1) for c in range(6)] for r in REGIMES])
    limit = max(10, float(np.max(np.abs(changes))))
    im = axes[1].imshow(changes, cmap="RdYlGn_r", vmin=-limit, vmax=limit, aspect="auto")
    axes[1].set_xticks(range(6), ["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"])
    axes[1].set_yticks(range(4), ["Random", "Burst 2", "Burst 5", "Burst 10"])
    axes[1].set_title("RMSE change vs original MLP (mean of seeds)")
    for i in range(4):
        for j in range(6): axes[1].text(j, i, f"{changes[i,j]:+.1f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes[1], label="Change (%)")
    fig.suptitle("Subject107: fixed warning fallback; positive change = higher error", fontsize=12)
    fig.savefig(OUT / "fallback_comparison.png", dpi=160); plt.close(fig)
    checks = {"verification_pass": True, "case_identities": 1224, "age_case_records": age_case_checks,
              "comparison_csv_rows": len(rows), "primary_usefulness_pass": report["primary_usefulness_pass"],
              "verification": "Independent per-case/age routing formulas, math.fsum pooling, count/age partitions, routed-group delta identity, normalized objectives and all CSV values",
              "raw_recordings_opened_by_verifier": 0, "model_inferences": 0,
              "mean_original_normalized_mae": math.fsum(s["original_normalized_mae"] for s in summary)/3,
              "mean_hybrid_normalized_mae": math.fsum(s["hybrid_normalized_mae"] for s in summary)/3,
              "report_sha256": digest(OUT / "report.json"), "verifier_sha256": digest(__file__),
              "limits": "Saved-sum policy evaluation; runtime hybrid streaming and hardware behavior not tested"}
    with (OUT / "independent_checks.json").open("x") as stream: json.dump(checks, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__": main()
