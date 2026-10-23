#!/usr/bin/env python3
"""Compare calibrated linear extrapolation with archived LOCF on one segment.

    python3 -B scripts/evaluate_linear_extrapolation.py

Validation subject 107 only. Caps and synthetic masks are read, never fitted.
The default case is segment 0 / random_5pct; no batch evaluation is performed.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_locf import git_output, sha256
from evaluate_locf_batch import replay_digest
from packetheal.baselines import LOCF
from packetheal.evaluation import (
    REGIMES, load_segment_mask, replay_segment, score_hidden, segment_metadata,
)
from packetheal.linear_extrapolation import LinearExtrapolator
from packetheal.splits import load_split_segment


# Explicitly bind this experiment to the settings frozen before validation.
CAP_SHA256 = "316cc1f1d0e95446329f1f689512703a146f6d07ad00bf463649ad705e13f502"
LOCF_SHA256 = "46946e2169aed8f46cbb7628443bc2406048a275671639086497559ad73ecea8"


def load_frozen_caps(path, config, preparation, *, expected_sha256):
    """Verify bytes and the training-only calibration contract; return metadata.

    The expected hash is pinned by the experiment, independently of the sidecar.
    Updating both a JSON and its sidecar therefore cannot silently retune it.
    """
    path = Path(path)
    checksum = path.with_suffix(path.suffix + ".sha256").read_text().split()
    if checksum != [expected_sha256, path.name] or sha256(path) != expected_sha256:
        raise ValueError("Frozen slope-cap checksum mismatch")
    caps = json.loads(path.read_text())
    expected = {
        "schema_version": "1", "purpose": "linear_extrapolation_slope_caps",
        "fit_split": "train", "subjects": list(range(101, 107)),
        "quantile": 0.99, "quantile_method": "linear", "n_channels": 6,
        "channel_order": config["channels"], "sampling_frequency_hz": 100,
        "normalization_applied": False, "loss_masks_applied": False,
        "cap_units": ["m/s² per sample step"] * 3 + ["rad/s per sample step"] * 3,
    }
    for key, value in expected.items():
        if caps.get(key) != value:
            raise ValueError(f"Incompatible slope-cap metadata: {key}")
    train = preparation["splits"]["train"]
    if (config["train_subjects"] != expected["subjects"]
            or train["subjects"] != expected["subjects"]
            or config["sampling_frequency_hz"] != 100
            or len(config["channels"]) != 6):
        raise ValueError("Calibration/configuration training contract mismatch")
    for key in ("segments", "samples"):
        if caps[key] != train[key]:
            raise ValueError(f"Calibration count mismatch: {key}")
    if caps["pairs_per_channel"] != train["samples"] - train["segments"]:
        raise ValueError("Calibration adjacent-pair count mismatch")
    if set(caps["per_subject"]) != {str(s) for s in expected["subjects"]}:
        raise ValueError("Calibration per-subject membership mismatch")
    for key in ("segments", "samples", "pairs_per_channel"):
        if sum(row[key] for row in caps["per_subject"].values()) != caps[key]:
            raise ValueError(f"Calibration per-subject total mismatch: {key}")
    if (not caps["repeat_quantile_identical"]
            or not all(caps["verification"].values())):
        raise ValueError("Calibration verification was not successful")
    # Reuse the predictor's shape, finite-value and nonnegative validation.
    LinearExtrapolator(caps["max_delta_per_step"])
    return caps


def check_locf_reference(archive, segment_index, regime, segment, result, metrics):
    """Require the exact archived case, including full output provenance.

    The replay digest covers values, every status, and loss ages. In particular,
    matching only the *number* of hidden positions is insufficient.
    """
    if (archive["predictor"], archive["split"], archive["subject"]) != (
        "LOCF", "validation", 107,
    ):
        raise ValueError("Wrong LOCF reference partition/predictor")
    cases = [case for case in archive["per_segment"]
             if case["segment_index"] == segment_index and case["regime"] == regime]
    if len(cases) != 1:
        raise ValueError("Expected exactly one archived LOCF case")
    case = cases[0]
    if (case["segment"] != segment or case["replay_sha256"] != replay_digest(result)
            or case["metrics"] != metrics):
        raise ValueError("Current LOCF replay differs from the archived reference")
    return case


def checked_linear_replay(caps, truth, missing):
    """Check reset/repeatability, preservation, and causal input isolation."""
    original, original_mask = truth.copy(), missing.copy()
    predictor = LinearExtrapolator(caps)
    first = replay_segment(predictor, truth, missing)
    second = replay_segment(predictor, truth, missing)  # Same object must reset.
    for field in ("values", "status", "missing_run"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    metrics = score_hidden(truth, missing, first)
    if metrics != score_hidden(truth, missing, second):
        raise ValueError("Repeated linear metrics differ")
    if metrics["unavailable_samples"] or metrics["scored_samples"] != int(missing.sum()):
        raise ValueError("Not all hidden positions were reconstructed")
    np.testing.assert_array_equal(first.values[~missing], truth[~missing])

    poisoned = truth.copy()
    poisoned[missing] = np.nan  # Hidden truth cannot reach the predictor.
    isolated = replay_segment(predictor, poisoned, missing)
    future = truth.copy()
    cutoff = len(truth) // 2
    future[cutoff:] += 10.0
    altered = replay_segment(predictor, future, missing)
    for field in ("values", "status", "missing_run"):
        np.testing.assert_array_equal(getattr(first, field), getattr(isolated, field))
        np.testing.assert_array_equal(
            getattr(first, field)[:cutoff], getattr(altered, field)[:cutoff],
        )
    np.testing.assert_array_equal(truth, original)
    np.testing.assert_array_equal(missing, original_mask)
    np.testing.assert_array_equal(predictor.max_delta_per_step, caps)
    return first, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment-index", type=int, default=0)
    parser.add_argument("--regime", choices=REGIMES, default="random_5pct")
    parser.add_argument("--output", type=Path, help="New JSON file; overwrite refused")
    args = parser.parse_args(argv)
    output = args.output or ROOT / "runs" / (
        f"linear_validation_subject107_segment{args.segment_index}_{args.regime}.json"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    config_path = ROOT / "configs/pamap2_split_v1.json"
    manifest = ROOT / "configs/pamap2_segments_v1.csv"
    processed = ROOT / "data/processed/pamap2_v1"
    cap_path = ROOT / "data/processed/linear_extrapolation_v1/slope_limits.json"
    archive_path = ROOT / "benchmarks/locf_validation_subject107_v1/benchmark.json"
    preparation_path = processed / "preparation_report.json"
    config = json.loads(config_path.read_text())
    preparation = json.loads(preparation_path.read_text())
    caps = load_frozen_caps(cap_path, config, preparation, expected_sha256=CAP_SHA256)
    if config["validation_subjects"] != [107]:
        raise ValueError("This experiment requires validation subject 107 only")
    if sha256(archive_path) != LOCF_SHA256:
        raise ValueError("Archived LOCF benchmark checksum mismatch")
    archive = json.loads(archive_path.read_text())

    artifact_paths = {
        "split_config": config_path, "segment_manifest": manifest,
        "loss_masks": processed / "loss_masks.csv",
        "normalization": processed / "normalization.json",
        "preparation_report": preparation_path,
        "audit_report": ROOT / "data/reports/pamap2_audit/audit.json",
    }
    hashes = {name: sha256(path) for name, path in artifact_paths.items()}
    if hashes != caps["provenance"]["artifact_sha256"]:
        raise ValueError("Calibration data provenance no longer matches current artifacts")
    for name, value in archive["provenance"]["artifact_sha256"].items():
        if hashes[name] != value:
            raise ValueError(f"LOCF and linear artifact mismatch: {name}")
    sources = dict(caps["provenance"]["source_sha256"])
    for name, value in archive["provenance"]["source_sha256"].items():
        if name in sources and sources[name] != value:
            raise ValueError(f"Calibration and LOCF source mismatch: {name}")
        sources[name] = value
    for name, value in sources.items():
        if sha256(ROOT / name) != value:
            raise ValueError(f"Previously evaluated source changed: {name}")
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = sha256(__file__)

    dataset = ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset"
    recording = dataset / "Protocol/subject107.dat"
    recording_hash = sha256(recording)
    if recording_hash != archive["provenance"]["recording_sha256"]:
        raise ValueError("Validation recording differs from archived LOCF input")
    timestamps, truth = load_split_segment(
        dataset, manifest, config_path, "validation", 107, args.segment_index,
    )
    segment = segment_metadata(manifest, 107, args.segment_index)
    missing = load_segment_mask(
        artifact_paths["loss_masks"], manifest, config_path, "validation", 107,
        args.segment_index, args.regime,
        history_preserved_samples=preparation["history_preserved_samples"],
    )
    locf = replay_segment(LOCF(), truth, missing)
    locf_metrics = score_hidden(truth, missing, locf)
    check_locf_reference(archive, args.segment_index, args.regime, segment, locf, locf_metrics)
    result, metrics = checked_linear_replay(caps["max_delta_per_step"], truth, missing)
    comparison = [
        {"channel": name, "unit": "m/s²" if i < 3 else "rad/s",
         "locf_mae": locf_metrics["mae"][i], "linear_mae": metrics["mae"][i],
         "locf_rmse": locf_metrics["rmse"][i], "linear_rmse": metrics["rmse"][i]}
        for i, name in enumerate(config["channels"])
    ]

    # Detect input or source changes during the experiment, before saving evidence.
    for name, path in artifact_paths.items():
        if sha256(path) != hashes[name]:
            raise ValueError(f"Input changed during replay: {name}")
    for path, expected in ((cap_path, CAP_SHA256), (archive_path, LOCF_SHA256),
                           (recording, recording_hash)):
        if sha256(path) != expected:
            raise ValueError(f"Input changed during replay: {path}")
    for name, value in sources.items():
        if sha256(ROOT / name) != value:
            raise ValueError(f"Source changed during replay: {name}")

    report = {
        "schema_version": "1", "predictor": "LinearExtrapolator",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "validation", "subject": 107, "segment_index": args.segment_index,
        "segment": segment, "regime": args.regime, "samples": len(truth),
        "timestamp_start_s": float(timestamps[0]), "timestamp_end_s": float(timestamps[-1]),
        "sampling_frequency_hz": 100, "channel_order": config["channels"],
        "channel_units": ["m/s²"] * 3 + ["rad/s"] * 3,
        "normalization_applied": False, "mask_seed": preparation["seed"],
        "history_preserved_samples": preparation["history_preserved_samples"],
        "actual_missing_fraction": float(missing.mean()),
        "max_missing_run": int(result.missing_run.max()),
        "loss_source": "frozen synthetic sample losses; no transport timing simulation",
        "slope_caps": caps["max_delta_per_step"],
        "slope_cap_units": caps["cap_units"],
        "metrics": metrics, "locf_metrics": locf_metrics, "comparison": comparison,
        "replay_sha256": replay_digest(result), "locf_replay_sha256": replay_digest(locf),
        "verification": {
            "locf_matches_archived_case": True, "all_hidden_positions_scored": True,
            "repeat_and_reset_identical": True, "received_samples_preserved": True,
            "hidden_truth_nan_replay_identical": True, "future_change_prefix_identical": True,
            "ground_truth_and_mask_preserved": True, "caps_data_and_sources_unchanged": True,
        },
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status_porcelain": git_output("status", "--porcelain"),
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "runtime_dependency_pins": [f"numpy=={np.__version__}"],
            "artifact_sha256": hashes, "recording_sha256": recording_hash,
            "source_sha256": sources, "slope_caps_sha256": CAP_SHA256,
            "locf_benchmark_sha256": LOCF_SHA256,
            "training_recording_check": "Calibration-time hashes retained in pinned cap artifact; training raw files not reopened",
        },
        "interpretation": "Single-segment pipeline check; does not establish full-subject superiority or uncertainty coverage.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(output.resolve()), "hidden_samples": metrics["hidden_samples"],
                      "comparison": comparison, "verification": report["verification"]}, indent=2))


if __name__ == "__main__":
    main()
