#!/usr/bin/env python3
"""Export the frozen gain search and validation comparison without loading data.

MPLCONFIGDIR=/tmp/packetheal-matplotlib python3 -B scripts/plot_alpha_beta_comparison.py
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT_SHA256 = "074ef6634fdc2e372d851a534297d285feaafe3c89224a55ba2e48127048643b"
GAINS_SHA256 = "9eb9a8c44d7870d7d40d31dd6fe0e0a7cac876c38b2f4515fe81b99e690d4395"


def pinned_json(path, digest):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError(f"Frozen evidence checksum mismatch: {path}")
    return json.loads(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        default=ROOT / "runs/alpha_beta_validation_subject107_all_regimes.json")
    parser.add_argument("--gains", type=Path, default=ROOT / "data/processed/alpha_beta_v1/gains.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/alpha_beta_summary")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    report = pinned_json(args.report, REPORT_SHA256)
    gains = pinned_json(args.gains, GAINS_SHA256)
    assert report["gains_sha256"] == GAINS_SHA256
    assert (report["alpha"], report["beta"]) == (gains["alpha"], gains["beta"]) == (1, 0)
    regimes = gains["search"]["regimes"]
    labels = ["Random ~5%", "Burst 2", "Burst 5", "Burst 10"]
    channels = ["Accel X", "Accel Y", "Accel Z", "Gyro X", "Gyro Y", "Gyro Z"]
    rows = []
    for regime in regimes:
        for record in report["comparison_by_regime"][regime]:
            row = {"regime": regime, **record}
            for metric in ("mae", "rmse"):
                for baseline in ("locf", "linear"):
                    row[f"{metric}_change_vs_{baseline}_pct"] = 100 * (
                        row[f"alpha_beta_{metric}"] / row[f"{baseline}_{metric}"] - 1)
            rows.append(row)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.3), layout="constrained",
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})
    search = gains["search"]
    scores = np.array([c["objective"] for c in gains["candidate_results"]]).reshape(
        len(search["alpha_values"]), len(search["beta_values"]))
    image = axes[0].imshow(scores, cmap="YlOrRd", aspect="auto")
    axes[0].set_xticks(range(scores.shape[1]), [str(v) for v in search["beta_values"]])
    axes[0].set_yticks(range(scores.shape[0]), [str(v) for v in search["alpha_values"]])
    axes[0].set_xlabel("beta: trend correction gain")
    axes[0].set_ylabel("alpha: value correction gain")
    axes[0].set_title("TRAIN 101–106: choose gains\nMean normalized MAE (lower is better)", pad=14)
    selected = np.unravel_index(gains["selected_candidate_index"], scores.shape)
    axes[0].add_patch(Rectangle((selected[1] - .5, selected[0] - .5), 1, 1,
                                fill=False, edgecolor="#145a32", linewidth=4))
    for (y, x), score in np.ndenumerate(scores):
        axes[0].text(x, y, f"{score:.3f}", ha="center", va="center",
                     color="white" if score > .30 else "black", fontsize=11)
    fig.colorbar(image, ax=axes[0], shrink=.7, label="dimensionless training objective")
    for ax, metric in zip(axes[1:], ("mae", "rmse")):
        changes = np.array([[report["comparison_by_regime"][r][c][f"alpha_beta_{metric}"] /
                             report["comparison_by_regime"][r][c][f"linear_{metric}"] * 100 - 100
                             for r in regimes] for c in range(6)])
        image = ax.imshow(changes, cmap="RdBu_r", vmin=-100, vmax=100, aspect="auto")
        ax.set_xticks(range(4), labels, rotation=25, ha="right")
        ax.set_yticks(range(6), channels)
        ax.set_title(f"VALIDATION 107: {metric.upper()} change vs linear\nNegative = lower error", pad=14)
        for (y, x), change in np.ndenumerate(changes):
            ax.text(x, y, f"{change:+.1f}%", ha="center", va="center",
                    color="white" if abs(change) > 55 else "black", fontsize=11)
        fig.colorbar(image, ax=ax, shrink=.7, label="change (%) within channel/regime")
    fig.suptitle("Alpha–beta benchmark: training selects alpha = 1, beta = 0 → LOCF\n"
                 "Validation: 102 segments × 4 masks = 408 cases; exact LOCF agreement in every case",
                 fontsize=16)
    fig.supxlabel("One global pair from 20 candidates; fixed training q99 slope caps. "
                  "One validation subject / one frozen seed per regime; no test or hardware result.", fontsize=11)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig.savefig(args.output_dir / "comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "rows": len(rows),
                      "matplotlib_version": matplotlib.__version__}, indent=2))


if __name__ == "__main__":
    main()
