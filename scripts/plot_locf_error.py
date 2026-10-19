#!/usr/bin/env python3
"""Locate and plot the largest hidden acceleration-Y error in selected cases.

Outputs a PNG, sample trace CSV, and JSON explaining the event. Original
measurements and frozen masks are preserved, including unusual signal values.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_locf import git_output, sha256
from evaluate_locf_batch import replay_digest
from packetheal.baselines import LOCF
from packetheal.evaluation import REGIMES, load_subject_masks, replay_segment
from packetheal.splits import load_split_segment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "runs/locf_validation_subject107_all_regimes.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/locf_validation_diagnostic")
    parser.add_argument("--segment-index", type=int, help="Search one segment instead of the entire subject")
    parser.add_argument("--regime", choices=REGIMES, help="Search one regime instead of all four")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    report = json.loads(args.report.read_text())
    if report["split"] != "validation" or report["subject"] != 107:
        raise ValueError("This diagnostic is restricted to validation subject 107")
    for name, expected in report["provenance"]["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Source differs from the evaluated version: {name}")
    manifest = ROOT / "configs/pamap2_segments_v1.csv"
    config = ROOT / "configs/pamap2_split_v1.json"
    processed = ROOT / "data/processed/pamap2_v1"
    artifacts = {
        "segment_manifest": manifest, "split_config": config,
        "loss_masks": processed / "loss_masks.csv",
        "normalization": processed / "normalization.json",
        "preparation_report": processed / "preparation_report.json",
    }
    for name, path in artifacts.items():
        if sha256(path) != report["provenance"]["artifact_sha256"][name]:
            raise ValueError(f"Input differs from the evaluated version: {path}")
    dataset_dir = ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset"
    recording = dataset_dir / "Protocol/subject107.dat"
    if sha256(recording) != report["provenance"]["recording_sha256"]:
        raise ValueError("Recording differs from the evaluated version")
    segments, masks = load_subject_masks(
        processed / "loss_masks.csv", manifest, config, "validation", 107,
        history_preserved_samples=report["history_preserved_samples"],
    )
    cases = {(case["segment_index"], case["regime"]): case for case in report["per_segment"]}
    if len(cases) != len(segments) * len(REGIMES) or report["cases"] != len(cases):
        raise ValueError("Batch report has incomplete or duplicate case identities")
    if args.segment_index is not None and not 0 <= args.segment_index < len(segments):
        raise ValueError("segment-index is outside the subject's eligible segments")
    selected_indices = list(range(len(segments))) if args.segment_index is None else [args.segment_index]
    selected_regimes = list(REGIMES) if args.regime is None else [args.regime]
    best = None
    predictor = LOCF(6)
    checked_cases = 0
    for completed, index in enumerate(selected_indices, start=1):
        segment = segments[index]
        timestamps, truth = load_split_segment(
            dataset_dir, manifest, config, "validation", 107, index,
        )
        for regime in selected_regimes:
            missing = masks[index, regime]
            result = replay_segment(predictor, truth, missing)
            if replay_digest(result) != cases[index, regime]["replay_sha256"]:
                raise ValueError("Diagnostic replay differs from the batch report")
            checked_cases += 1
            hidden = np.flatnonzero(missing)
            errors = np.abs(result.values[hidden, 1] - truth[hidden, 1])
            peak = int(np.argmax(errors))
            if best is None or float(errors[peak]) > best["absolute_error"]:
                best = {
                    "absolute_error": float(errors[peak]), "segment_index": index,
                    "regime": regime, "event_index": int(hidden[peak]),
                    "timestamps": timestamps, "truth": truth, "missing": missing,
                    "result": result, "segment": segment,
                }
        if completed % 10 == 0 or completed == len(selected_indices):
            print(f"Checked diagnostic replay {completed}/{len(selected_indices)} selected segments", flush=True)
    event = best["event_index"]
    truth, missing, result = best["truth"], best["missing"], best["result"]
    timestamps = best["timestamps"]
    previous = int(np.flatnonzero(~missing[:event])[-1])
    np.testing.assert_array_equal(result.values[event], truth[previous])
    start = int(best["segment"]["start_row"])
    original_rows = {start + previous: previous, start + event: event}
    with recording.open() as stream:
        for row_number, line in enumerate(stream, start=1):
            if row_number in original_rows:
                np.testing.assert_equal(float(line.split()[5]), truth[original_rows[row_number], 1])
            if row_number >= max(original_rows):
                break

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    step = 0.01
    edges = np.diff(np.r_[False, missing, False].astype(int))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    for ax, radius, title in zip(axes, (200, 15), ("Surrounding recording", "Zoom at the largest hidden error in selected cases")):
        left, right = max(0, event - radius), min(len(truth), event + radius + 1)
        indices = np.arange(left, right)
        hidden = indices[missing[indices]]
        real = indices[~missing[indices]]
        ax.plot(timestamps[indices], truth[indices, 1], color="#2166ac", lw=1.5,
                marker="o" if radius == 15 else None, markersize=4, label="Original recorded signal")
        ax.scatter(timestamps[real], truth[real, 1], s=18, facecolors="white", edgecolors="#2166ac",
                   zorder=3, label="Received samples")
        ax.plot(timestamps[hidden], result.values[hidden, 1], linestyle="none", marker="x",
                color="#d95f02", markersize=7, markeredgewidth=2, label="LOCF estimates at hidden samples")
        for first, end in runs:
            if first < right and end > left:
                ax.axvspan(timestamps[first] - step / 2, timestamps[end - 1] + step / 2,
                           color="#fdb863", alpha=0.22)
        ax.axvline(timestamps[event], color="#555555", linestyle=":", lw=1)
        ax.set(xlim=(timestamps[left] - step / 2, timestamps[right - 1] + step / 2),
               xlabel="Original recording timestamp (s)", ylabel="Acceleration Y (m/s²)", title=title)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="best", fontsize=9)
    axes[1].annotate(
        f"LOCF: {result.values[event, 1]:.3f}\nRecorded: {truth[event, 1]:.3f}\n"
        f"Absolute error: {best['absolute_error']:.3f} m/s²",
        xy=(timestamps[event], result.values[event, 1]), xycoords="data",
        xytext=(0.02, 0.96), textcoords="axes fraction", va="top", fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    figure.suptitle(
        f"LOCF diagnostic — subject 107, segment {best['segment_index']}, {best['regime']}\n"
        "Orange shading = synthetically hidden samples; unusual recorded values are retained",
        fontsize=13,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    image_path = args.output_dir / "largest_accel_y_error.png"
    figure.savefig(image_path, dpi=160)
    plt.close(figure)
    left, right = max(0, event - 200), min(len(truth), event + 201)
    with (args.output_dir / "sample_trace.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["segment_index", "local_index", "original_row", "timestamp_s",
                         "acc_y_recorded_m_s2", "acc_y_output_m_s2", "missing", "status", "missing_run"])
        for index in range(left, right):
            writer.writerow([best["segment_index"], index, start + index, timestamps[index],
                             truth[index, 1], result.values[index, 1], bool(missing[index]),
                             result.status[index], int(result.missing_run[index])])
    summary = {
        "subject": 107, "split": "validation", "segment_index": best["segment_index"],
        "segment": best["segment"], "regime": best["regime"], "channel": "acc_y_m_s2",
        "event_local_index": event, "event_original_row": start + event,
        "event_timestamp_s": float(timestamps[event]),
        "last_received_local_index": previous, "last_received_original_row": start + previous,
        "recorded_value": float(truth[event, 1]), "last_received_value": float(truth[previous, 1]),
        "reconstructed_value": float(result.values[event, 1]),
        "absolute_error_m_s2": best["absolute_error"],
        "missing_run_at_event": int(result.missing_run[event]),
        "search_scope": {
            "segment_indices": selected_indices, "regimes": selected_regimes,
            "cases_checked": checked_cases,
        },
        "verification": {
            "all_selected_replay_fingerprints_match_batch_report": True,
            "reconstruction_equals_last_received_vector": True,
            "event_and_last_received_values_match_raw_recording": True,
        },
        "source_report": str(args.report.resolve()), "source_report_sha256": sha256(args.report),
        "git_commit": git_output("rev-parse", "HEAD"), "diagnostic_source_sha256": sha256(__file__),
        "recording_sha256": report["provenance"]["recording_sha256"],
        "interpretation_limit": "The plot alone cannot establish whether unusual recorded values are physical motion or sensor/data artifacts.",
    }
    with (args.output_dir / "event.json").open("x") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"plot": str(image_path.resolve()), "event": summary}, indent=2))


if __name__ == "__main__":
    main()
