"""Independent saved-sum/CSV verification and physical-unit diagnostic plots.

Reads saved evidence only. No recording loading, fitting, or predictor replay.
"""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/mlp_v1_transients_20261027"
REGIMES = ("random_5pct", "burst_2_5pct", "burst_5_5pct", "burst_10_5pct")
CHANNELS = ("acc_x_m_s2", "acc_y_m_s2", "acc_z_m_s2", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s")
COUNTS = ("hidden_count", "scored_count", "unavailable_count")
SUMS = ("absolute_error_sum", "squared_error_sum")
GROUPS = ("all", "large_change", "other")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)


def save_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    if (OUT / "independent_checks.json").exists():
        raise FileExistsError("Verification outputs already exist")
    report = json.loads((OUT / "report.json").read_text())
    policy = json.loads((OUT / "policy.json").read_text())
    assert report["policy"] == policy and report["checks_passed"]
    assert digest(ROOT / "runs/mlp_v1_transients_20261027.py") == policy["script_sha256"]
    preserved = {}
    for manifest, base in (
        (ROOT / "runs/mlp_v1_train_20261027/evidence.sha256", ROOT),
        (ROOT / "runs/mlp_v1_train_20261027/source_snapshot.sha256", ROOT),
        (ROOT / "runs/mlp_v1_validation_20261027/evidence.sha256", ROOT),
        (ROOT / "runs/mlp_v1_residual_failure_20261027/artifacts.sha256", ROOT),
        (ROOT / "benchmarks/phase2_validation_subject107_v1/artifacts.sha256", ROOT / "benchmarks/phase2_validation_subject107_v1"),
        (ROOT / "benchmarks/phase2_validation_subject107_v1/evaluated_sources.sha256", ROOT),
    ):
        lines = manifest.read_text().splitlines()
        for line in lines:
            expected_hash, name = line.split(maxsplit=1)
            path = (base / name.strip()).resolve()
            assert path.is_relative_to(ROOT) and path.suffix != ".dat"
            assert digest(path) == expected_hash, path
        preserved[str(manifest.relative_to(ROOT))] = len(lines)
    for path, expected in {**policy["frozen_inputs"], **report["source_hashes"]}.items():
        assert digest(ROOT / path) == expected
    for name, expected in report["checkpoint_hashes"].items():
        assert digest(ROOT / "runs/mlp_v1_train_20261027/models" / name) == expected
    validation_path = ROOT / "runs/mlp_v1_validation_20261027/validation.json"
    assert digest(validation_path) == policy["validation_sha256"]
    validation = json.loads(validation_path.read_text())
    classical = json.loads((ROOT / "benchmarks/phase2_validation_subject107_v1/benchmark.json").read_text())
    access = report["recording_access"]
    assert access["allowed_subjects"] == [107] and not access["denied_recordings"]
    assert set(access["opened_recordings"]) == {str(ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset/Protocol/subject107.dat")}
    assert report["recording_sha256"] == validation["recording_hash"]
    names = ["locf"] + [f"{mode}_seed{seed}" for mode in ("direct", "residual") for seed in range(3)]
    cases = report["cases"]
    indexed = {(r["algorithm"], r["segment"], r["regime"]): r for r in cases}
    expected = {(name, segment, regime) for name in names for segment in range(102) for regime in REGIMES}
    assert len(cases) == len(indexed) == 2856 and set(indexed) == expected
    assert report["samples"] == 285488
    for row in cases:
        assert row["subject"] == 107
        groups = row["strata"]
        for c in range(6):
            for field in COUNTS:
                assert groups["large_change"][field][c] + groups["other"][field][c] == groups["all"][field][c]
            for field in SUMS:
                close(math.fsum(groups[g][field][c] for g in ("large_change", "other")), groups["all"][field][c])
            for group in GROUPS:
                metrics = groups[group]
                assert metrics["hidden_count"][c] == metrics["scored_count"][c]
                assert metrics["unavailable_count"][c] == 0
    for name in names:
        for regime in REGIMES:
            for group in GROUPS:
                saved = report["pooled"][name][regime][group]
                selected = [indexed[name, segment, regime]["strata"][group] for segment in range(102)]
                for c in range(6):
                    for field in COUNTS:
                        assert sum(r[field][c] for r in selected) == saved[field][c]
                    for field in SUMS:
                        close(math.fsum(r[field][c] for r in selected), saved[field][c])
                    n = saved["scored_count"][c]
                    close(saved["mae"][c], saved["absolute_error_sum"][c] / n)
                    close(saved["rmse"][c], math.sqrt(saved["squared_error_sum"][c] / n))
                    close(saved["coverage"][c], n / saved["hidden_count"][c])
                    assert n == report["pooled"]["locf"][regime][group]["scored_count"][c]
                    if name == "locf":
                        reference = classical["pooled_by_regime"][regime]["algorithms"]["locf"]["strata"][group]
                        for field in (*COUNTS, *SUMS, "mae", "rmse", "coverage"):
                            close(saved[field][c], reference[field][c])
                if group == "all":
                    reference = validation["pooled"][name][regime]
                    for new, old in zip(SUMS, ("sum_abs", "sum_sq")):
                        for c in range(6):
                            close(saved[new][c], reference[old][c])
    with (OUT / "strata.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 504
    assert {(r["algorithm"], r["regime"], r["channel"], r["group"]) for r in rows} == {
        (name, regime, channel, group) for name in names for regime in REGIMES for channel in CHANNELS for group in GROUPS}
    for row in rows:
        name, regime, group = (row[k] for k in ("algorithm", "regime", "group"))
        c = CHANNELS.index(row["channel"])
        saved = report["pooled"][name][regime][group]
        base = report["pooled"]["locf"][regime][group]
        whole = report["pooled"][name][regime]["all"]
        assert row["unit"] == ("m/s^2" if c < 3 else "rad/s")
        for field in (*COUNTS, *SUMS, "mae", "rmse", "coverage"):
            close(row[field], saved[field][c])
        close(row["hidden_fraction_of_all"], saved["hidden_count"][c] / whole["hidden_count"][c])
        close(row["squared_error_fraction_of_all"], saved["squared_error_sum"][c] / whole["squared_error_sum"][c])
        for metric in ("mae", "rmse"):
            close(row[f"{metric}_change_percent_vs_locf"], 100 * (saved[metric][c] / base[metric][c] - 1))
    summary = []
    for mode in ("direct", "residual"):
        for group in ("large_change", "other"):
            for regime in REGIMES:
                for c, channel in enumerate(CHANNELS):
                    base = report["pooled"]["locf"][regime][group]
                    row = {"mode": mode, "group": group, "regime": regime, "channel": channel,
                           "unit": "m/s^2" if c < 3 else "rad/s", "count": base["scored_count"][c]}
                    for metric in ("mae", "rmse"):
                        values = [report["pooled"][f"{mode}_seed{seed}"][regime][group][metric][c] for seed in range(3)]
                        row.update({f"locf_{metric}": base[metric][c], f"mean_seed_{metric}": statistics.mean(values),
                                    f"seed_sd_{metric}": statistics.stdev(values),
                                    f"mean_seed_{metric}_change_percent": 100 * (statistics.mean(values) / base[metric][c] - 1),
                                    f"seeds_improving_{metric}": sum(v < base[metric][c] for v in values),
                                    f"min_seed_{metric}_change_percent": 100 * (min(values) / base[metric][c] - 1),
                                    f"max_seed_{metric}_change_percent": 100 * (max(values) / base[metric][c] - 1)})
                    summary.append(row)
    with (OUT / "mean_seed_comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for metric in ("mae", "rmse"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 6.7), layout="constrained")
        limit = max(10, max(abs(r[f"mean_seed_{metric}_change_percent"]) for r in summary))
        for i, mode in enumerate(("direct", "residual")):
            for j, group in enumerate(("large_change", "other")):
                ax = axes[i, j]
                data = np.array([[next(r[f"mean_seed_{metric}_change_percent"] for r in summary
                                       if (r["mode"], r["group"], r["regime"], r["channel"]) == (mode, group, regime, channel))
                                  for channel in CHANNELS] for regime in REGIMES])
                im = ax.imshow(data, cmap="RdYlGn_r", vmin=-limit, vmax=limit, aspect="auto")
                ax.set_xticks(range(6), ["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"])
                ax.set_yticks(range(4), ["Random", "Burst 2", "Burst 5", "Burst 10"])
                ax.set_title(f"{mode.capitalize()} | {'Rapid recorded change' if group == 'large_change' else 'Other hidden positions'}")
                for r in range(4):
                    for c in range(6):
                        ax.text(c, r, f"{data[r,c]:+.1f}%", ha="center", va="center", fontsize=9)
                fig.colorbar(im, ax=ax, label="Change vs LOCF (%)", shrink=.85)
        fig.suptitle(f"Subject 107: {metric.upper()} change, mean of 3 frozen seeds\nNegative = lower error; each cell compares errors in the same channel's physical units", fontsize=12)
        fig.savefig(OUT / f"{metric}_change.png", dpi=160)
        plt.close(fig)
    checks = {"verification_pass": True, "case_identities": len(cases), "csv_rows": len(rows),
              "pooled_strata": 84, "mean_seed_rows": len(summary),
              "arithmetic": "Independent math.fsum pooling, exact counts, per-case partitions, derived metrics and all CSV fields; rtol=atol=1e-12",
              "reference_checks": "Every pooled original metric and frozen LOCF stratum reconciled",
              "preserved_evidence_manifests": preserved,
              "raw_access": "Verifier opens no recordings; main report allows subject107 only",
              "scope": "Saved arithmetic verification, not an independent full predictor implementation",
              "environment": {"python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                              "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")},
              "script_sha256": digest(__file__), "report_sha256": digest(OUT / "report.json")}
    save_json(OUT / "independent_checks.json", checks)
    for mode in ("direct", "residual"):
        for group in ("large_change", "other"):
            selected = [r for r in summary if (r["mode"], r["group"]) == (mode, group)]
            print(mode, group, {metric: {"improved_mean_cells": sum(r[f"mean_seed_{metric}_change_percent"] < 0 for r in selected),
                                        "all_three_seeds_improve_cells": sum(r[f"seeds_improving_{metric}"] == 3 for r in selected),
                                        "worst_mean_change_percent": max(r[f"mean_seed_{metric}_change_percent"] for r in selected)} for metric in ("mae", "rmse")})
    print("PASS", checks)


if __name__ == "__main__":
    main()
