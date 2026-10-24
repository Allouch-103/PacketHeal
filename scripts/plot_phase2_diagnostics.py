#!/usr/bin/env python3
"""Plot pinned Phase-2 diagnostics without accessing any raw recording."""

import argparse
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHANNELS = ("Acceleration X", "Acceleration Y", "Acceleration Z", "Gyroscope X", "Gyroscope Y", "Gyroscope Z")
REGIMES = ("random_5pct", "burst_2_5pct", "burst_5_5pct", "burst_10_5pct")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    content = args.report.read_bytes()
    if hashlib.sha256(content).hexdigest() != args.report_sha256:
        raise ValueError("Diagnostic report checksum mismatch")
    report = json.loads(content)
    if report["subjects"] != [107] or (report["alpha"], report["beta"]) != (1, 0):
        raise ValueError("Plot definitions require the fixed subject107 / LOCF-endpoint experiment")
    # Prove equivalence before sharing the LOCF/alpha-beta curve in each panel.
    for entry in report["pooled_by_regime"].values():
        if entry["algorithms"]["locf"] != entry["algorithms"]["alpha_beta"]:
            raise ValueError("Cannot share curves for predictors with different results")
    args.output_dir.mkdir(parents=True)
    colors = {"locf": "#1464a0", "linear": "#b43a30"}
    names = {"locf": "LOCF = selected alpha-beta", "linear": "Capped linear"}
    for metric in ("mae", "rmse"):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        handles = None
        for c, ax in enumerate(axes.flat):
            for name in ("locf", "linear"):
                for group, style, marker in (("other", "-", "o"), ("large_change", "--", "s")):
                    values = [report["pooled_by_regime"][r]["algorithms"][name]["strata"][group][metric][c]
                              for r in REGIMES]
                    y = [np.nan if v is None or v == 0 else v for v in values]
                    ax.plot(range(4), y, style, marker=marker, color=colors[name],
                            label=f"{names[name]}: {group.replace('_', ' ')}")
            ax.set_yscale("log")
            ax.set_title(CHANNELS[c])
            ax.set_xticks(range(4), ("Random", "Burst 2", "Burst 5", "Burst 10"))
            ax.set_ylabel(f"{metric.upper()} ({report['channel_units'][c]}); log scale")
            ax.grid(alpha=.25, which="both")
            if handles is None:
                handles = ax.get_legend_handles_labels()
        fig.suptitle(f"Subject107: {metric.upper()} during large recorded changes versus other hidden samples\n"
                     "Large change = per-channel adjacent difference > frozen training q99 threshold", fontsize=14)
        fig.legend(*handles, loc="outside lower center", ncol=2, fontsize=10)
        fig.savefig(args.output_dir / f"transient_{metric}.png", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        entry = report["pooled_by_regime"]["burst_10_5pct"]
        for c, ax in enumerate(axes.flat):
            for name in ("locf", "linear"):
                values = [entry["algorithms"][name]["strata"][f"age_{age}"][metric][c] for age in range(1, 11)]
                ax.plot(range(1, 11), values, marker="o", color=colors[name], label=names[name])
            ax.set_title(CHANNELS[c])
            ax.set_xlabel("Position inside the missing run (nominal sample steps)")
            ax.set_ylabel(f"{metric.upper()} ({report['channel_units'][c]})")
            ax.set_xticks(range(1, 11))
            ax.grid(alpha=.25)
        fig.suptitle(f"Subject107, burst 10: {metric.upper()} as the outage progresses\n"
                     "Same 1,466 gap cohort at every age; 10 ms per step. Other regimes use different hidden positions.",
                     fontsize=14)
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
        fig.savefig(args.output_dir / f"burst10_{metric}.png", dpi=150)
        plt.close(fig)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "figures": 4,
                      "matplotlib_version": matplotlib.__version__,
                      "zero_or_empty_transient_points": "not plotted; counts and original values remain in CSV/JSON"}, indent=2))


if __name__ == "__main__":
    main()
