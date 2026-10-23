#!/usr/bin/env python3
"""Reproduce and explain a targeted acceleration-Z / burst-10 failure.

Choose the segment with the largest excess squared-error sum over LOCF, then
the hidden sample with the largest excess squared error inside that segment.
This is a targeted diagnostic, not a representative performance estimate.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_linear_extrapolation import CAP_SHA256, checked_linear_replay, load_frozen_caps
from evaluate_locf import git_output, sha256
from evaluate_locf_batch import replay_digest
from packetheal.baselines import LOCF
from packetheal.evaluation import load_segment_mask, replay_segment, score_hidden, segment_metadata
from packetheal.splits import load_split_segment

REPORT_SHA256 = "04817522032f98a19851f2bec80567501f247ecb99a2e6f13894efaf5c453955"
CHANNEL = 2
REGIME = "burst_10_5pct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "runs/linear_validation_subject107_all_regimes.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/linear_validation_failure_diagnostic")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    if sha256(args.report) != REPORT_SHA256:
        raise ValueError("Diagnostic requires the verified full-subject comparison report")
    report = json.loads(args.report.read_text())
    if (report["split"], report["subject"], report["predictor"]) != (
        "validation", 107, "LinearExtrapolator",
    ):
        raise ValueError("Wrong diagnostic partition/predictor")
    case = max(
        (case for case in report["per_segment"] if case["regime"] == REGIME),
        key=lambda case: (case["metrics"]["squared_error_sum"][CHANNEL]
                          - case["locf_metrics"]["squared_error_sum"][CHANNEL],
                          -case["segment_index"]),
    )
    config_path = ROOT / "configs/pamap2_split_v1.json"
    manifest = ROOT / "configs/pamap2_segments_v1.csv"
    processed = ROOT / "data/processed/pamap2_v1"
    cap_path = ROOT / "data/processed/linear_extrapolation_v1/slope_limits.json"
    paths = {
        "split_config": config_path, "segment_manifest": manifest,
        "loss_masks": processed / "loss_masks.csv",
        "normalization": processed / "normalization.json",
        "preparation_report": processed / "preparation_report.json",
        "audit_report": ROOT / "data/reports/pamap2_audit/audit.json",
    }
    checks = {paths[name]: value for name, value in report["provenance"]["artifact_sha256"].items()}
    checks.update({ROOT / name: value for name, value in report["provenance"]["source_sha256"].items()})
    dataset = ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset"
    recording = dataset / "Protocol/subject107.dat"
    checks.update({recording: report["provenance"]["recording_sha256"],
                   cap_path: report["provenance"]["slope_caps_sha256"],
                   cap_path.with_suffix(".json.sha256"): report["provenance"]["slope_caps_sidecar_sha256"],
                   args.report: REPORT_SHA256, Path(__file__).resolve(): sha256(__file__)})
    for path, expected in checks.items():
        if sha256(path) != expected:
            raise ValueError(f"Input/source differs from evaluated version: {path}")
    config = json.loads(config_path.read_text())
    preparation = json.loads(paths["preparation_report"].read_text())
    caps = load_frozen_caps(cap_path, config, preparation, expected_sha256=CAP_SHA256)
    if caps["max_delta_per_step"] != report["slope_caps"]:
        raise ValueError("Report caps differ from the frozen artifact")
    index = case["segment_index"]
    timestamps, truth = load_split_segment(dataset, manifest, config_path, "validation", 107, index)
    segment = segment_metadata(manifest, 107, index)
    if segment != case["segment"]:
        raise ValueError("Segment differs from the saved comparison")
    missing = load_segment_mask(
        paths["loss_masks"], manifest, config_path, "validation", 107, index, REGIME,
        history_preserved_samples=report["history_preserved_samples"],
    )
    linear, linear_metrics = checked_linear_replay(caps["max_delta_per_step"], truth, missing)
    locf = replay_segment(LOCF(), truth, missing)
    if (replay_digest(linear) != case["replay_sha256"]
            or replay_digest(locf) != case["locf_replay_sha256"]
            or linear_metrics != case["metrics"]
            or score_hidden(truth, missing, locf) != case["locf_metrics"]):
        raise ValueError("Diagnostic replay fingerprints/metrics differ from the saved case")

    hidden = np.flatnonzero(missing)
    linear_errors = linear.values[hidden, CHANNEL] - truth[hidden, CHANNEL]
    locf_errors = locf.values[hidden, CHANNEL] - truth[hidden, CHANNEL]
    excess = linear_errors ** 2 - locf_errors ** 2
    event = int(hidden[np.argmax(excess)])
    previous, last = map(int, np.flatnonzero(~missing[:event])[-2:])
    spacing, horizon = last - previous, event - last
    uncapped_slope = (truth[last] - truth[previous]) / spacing
    clipped_slope = np.clip(uncapped_slope, -np.array(caps["max_delta_per_step"]), caps["max_delta_per_step"])
    manual = truth[last] + horizon * clipped_slope
    np.testing.assert_array_equal(manual, linear.values[event])
    np.testing.assert_array_equal(truth[last], locf.values[event])
    burst_start = event
    while burst_start > 0 and missing[burst_start - 1]:
        burst_start -= 1
    burst_end = event + 1
    while burst_end < len(missing) and missing[burst_end]:
        burst_end += 1
    if burst_end - burst_start != 10 or horizon != int(linear.missing_run[event]):
        raise ValueError("Selected event is not consistent with the ten-sample loss age")

    # Verify all six physical channels and timestamps directly at original rows.
    start_row = int(segment["start_row"])
    raw_rows = {start_row + i: i for i in (previous, last, event)}
    checked_rows = []
    with recording.open() as stream:
        for row, line in enumerate(stream, start=1):
            if row in raw_rows:
                fields = line.split()
                i = raw_rows[row]
                np.testing.assert_array_equal([float(fields[c]) for c in (4, 5, 6, 10, 11, 12)], truth[i])
                assert float(fields[0]) == timestamps[i]
                checked_rows.append(row)
            if row >= max(raw_rows):
                break
    if set(checked_rows) != set(raw_rows):
        raise ValueError("Not all original recording rows were checked")

    origin = timestamps[burst_start]
    time_ms = (timestamps - origin) * 1000
    left, right = max(0, event - 150), min(len(truth), event + 151)
    zoom_left, zoom_right = max(0, previous - 5), min(len(truth), burst_end + 8)
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    for ax, low, high, title in (
        (axes[0], left, right, "Surrounding recording"),
        (axes[1], zoom_left, zoom_right, "Selected gap: predictions use the two blue diamonds only"),
    ):
        indices = np.arange(low, high)
        real = indices[~missing[indices]]
        hidden_window = indices[missing[indices]]
        ax.plot(time_ms[indices], truth[indices, CHANNEL], color="#262626", lw=1.5,
                marker="o" if ax is axes[1] else None, markersize=3,
                label="Recorded signal (hidden truth used only for scoring)")
        ax.scatter(time_ms[real], truth[real, CHANNEL], color="#2166ac", s=18, zorder=4, label="Received samples")
        if ax is axes[0]:
            for result, color, label in ((locf, "#d99000", "LOCF estimate"), (linear, "#c43c39", "Linear estimate")):
                ax.scatter(time_ms[hidden_window], result.values[hidden_window, CHANNEL],
                           marker="x", color=color, s=25, label=label)
        else:
            forecast = np.arange(last, burst_end)
            for result, color, label in ((locf, "#d99000", "LOCF estimate"), (linear, "#c43c39", "Linear estimate")):
                ax.plot(time_ms[forecast], result.values[forecast, CHANNEL], marker="x", lw=2,
                        color=color, markersize=5, label=label)
        ax.axvspan(time_ms[burst_start] - 5, time_ms[burst_end - 1] + 5, color="#f5d48a", alpha=.35)
        ax.scatter(time_ms[[previous, last]], truth[[previous, last], CHANNEL], marker="D",
                   s=70, color="#2166ac", edgecolors="white", zorder=6, label="Two causal history points")
        ax.axvline(time_ms[event], color="#777777", linestyle=":", lw=1)
        ax.set(xlim=(time_ms[low] - 5, time_ms[high - 1] + 5), xlabel="Time from selected gap start (ms)",
               ylabel="Acceleration Z (m/s²)", title=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=2, loc="best")
    axes[1].annotate(
        f"At missing sample {horizon}/10:\nRecorded = {truth[event, CHANNEL]:.3f}\n"
        f"LOCF = {locf.values[event, CHANNEL]:.3f}\nLinear = {linear.values[event, CHANNEL]:.3f} m/s²",
        xy=(time_ms[event], linear.values[event, CHANNEL]), xytext=(.02, .96),
        textcoords="axes fraction", va="top", fontsize=10,
        bbox={"facecolor": "white", "alpha": .92, "edgecolor": "#cccccc"},
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    figure.suptitle(f"Targeted linear extrapolation failure — subject 107, segment {index}\n"
                   "Shading = selected synthetic ten-sample gap; no recorded values removed", fontsize=13)
    # All read-only scientific checks must pass before creating the output folder.
    for path, expected in checks.items():
        if sha256(path) != expected:
            raise ValueError(f"Input/source changed during diagnostic: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    image_path, trace_path = args.output_dir / "accel_z_burst10_failure.png", args.output_dir / "sample_trace.csv"
    figure.savefig(image_path, dpi=170)
    plt.close(figure)
    with trace_path.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["local_index", "original_row", "timestamp_s", "time_from_gap_start_ms",
                         "acc_z_recorded_m_s2", "missing", "locf_output_m_s2", "locf_status",
                         "linear_output_m_s2", "linear_status", "missing_run"])
        for i in range(left, right):
            writer.writerow([i, start_row + i, timestamps[i], time_ms[i], truth[i, CHANNEL], bool(missing[i]),
                             locf.values[i, CHANNEL], locf.status[i], linear.values[i, CHANNEL],
                             linear.status[i], int(linear.missing_run[i])])
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject": 107, "split": "validation", "segment_index": index, "segment": segment,
        "regime": REGIME, "channel": "acc_z_m_s2", "unit": "m/s²",
        "selection": "Largest segment excess acceleration-Z squared-error sum, then largest hidden-sample excess squared error; ties use earliest segment/sample",
        "segment_excess_squared_error_sum": case["metrics"]["squared_error_sum"][CHANNEL] - case["locf_metrics"]["squared_error_sum"][CHANNEL],
        "event_local_index": event, "event_original_row": start_row + event,
        "event_timestamp_s": float(timestamps[event]),
        "burst_start_local_index": burst_start, "burst_end_local_index_exclusive": burst_end,
        "previous_received_local_index": previous, "last_received_local_index": last,
        "previous_received_original_row": start_row + previous, "last_received_original_row": start_row + last,
        "previous_received_value": float(truth[previous, CHANNEL]), "last_received_value": float(truth[last, CHANNEL]),
        "received_spacing_steps": spacing, "horizon_steps": horizon,
        "uncapped_slope_per_step": float(uncapped_slope[CHANNEL]),
        "slope_cap_per_step": caps["max_delta_per_step"][CHANNEL],
        "clipped_slope_per_step": float(clipped_slope[CHANNEL]),
        "slope_was_clipped": bool(uncapped_slope[CHANNEL] != clipped_slope[CHANNEL]),
        "recorded_value": float(truth[event, CHANNEL]), "locf_value": float(locf.values[event, CHANNEL]),
        "linear_value": float(linear.values[event, CHANNEL]), "manual_prediction": float(manual[CHANNEL]),
        "locf_absolute_error": float(abs(locf.values[event, CHANNEL] - truth[event, CHANNEL])),
        "linear_absolute_error": float(abs(linear.values[event, CHANNEL] - truth[event, CHANNEL])),
        "verification": {"both_replay_fingerprints_and_metrics_match_batch": True,
                         "manual_six_channel_prediction_matches": True, "locf_equals_last_received": True,
                         "three_original_rows_checked_all_six_channels_and_timestamps": True,
                         "ten_sample_gap_and_loss_age_match": True, "inputs_and_sources_unchanged": True},
        "source_report": str(args.report.resolve()), "source_report_sha256": REPORT_SHA256,
        "slope_caps_sha256": CAP_SHA256, "diagnostic_source_sha256": sha256(__file__),
        "recording_sha256": report["provenance"]["recording_sha256"],
        "git_commit": git_output("rev-parse", "HEAD"),
        "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
        "output_sha256": {image_path.name: sha256(image_path), trace_path.name: sha256(trace_path)},
        "interpretation_limit": "Targeted failure only; does not establish typical behavior or distinguish physical motion from sensor/data artifacts.",
    }
    with (args.output_dir / "event.json").open("x") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"plot": str(image_path.resolve()), "event": summary}, indent=2))


if __name__ == "__main__":
    main()
