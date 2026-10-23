#!/usr/bin/env python3
"""Compare fixed-cap linear extrapolation with LOCF on all validation cases.

    python3 -B scripts/evaluate_linear_extrapolation_batch.py

Only subject 107 is loaded: 102 segments under four frozen synthetic masks.
No calibration, parameter selection, held-out testing, or hardware measurement.
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

from evaluate_linear_extrapolation import (
    CAP_SHA256, LOCF_SHA256, check_locf_reference, checked_linear_replay, load_frozen_caps,
)
from evaluate_locf import git_output, sha256
from evaluate_locf_batch import replay_digest
from packetheal.baselines import LOCF
from packetheal.evaluation import (
    REGIMES, load_subject_masks, pool_metrics, replay_segment, score_hidden,
)
from packetheal.splits import load_split_segment


def summarize_cases(cases, expected, archived_pooled):
    """Reject missing/duplicate cases; pool error sums, never segment MAE/RMSE.

    Every segment must occur once under every regime, with consistent lengths.
    LOCF pooling must reproduce the archived totals, independently of linear's
    errors. Both predictors must score all positions required by preparation.
    """
    wanted = {(i, regime) for i in range(expected["segments"]) for regime in REGIMES}
    keys = [(case["segment_index"], case["regime"]) for case in cases]
    if len(keys) != len(wanted) or set(keys) != wanted:
        raise ValueError("Missing, duplicate, or unexpected segment/regime cases")
    lengths = {}
    for case in cases:
        index, count = case["segment_index"], case["samples"]
        if count <= 0 or lengths.setdefault(index, count) != count:
            raise ValueError("Inconsistent segment sample counts across regimes")
    sample_count = sum(lengths.values())
    if sample_count != expected["samples"]:
        raise ValueError("Eligible sample count differs from preparation")

    pooled = {}
    for regime in REGIMES:
        selected = sorted((case for case in cases if case["regime"] == regime),
                          key=lambda case: case["segment_index"])
        for case in selected:
            for name in ("metrics", "locf_metrics"):
                metrics = case[name]
                if (metrics["unavailable_samples"]
                        or metrics["scored_samples"] != metrics["hidden_samples"]):
                    raise ValueError("Every case must score all hidden positions")
            if case["metrics"]["hidden_samples"] != case["locf_metrics"]["hidden_samples"]:
                raise ValueError("Predictor hidden counts differ within a case")
        metrics = pool_metrics([case["metrics"] for case in selected])
        locf_metrics = pool_metrics([case["locf_metrics"] for case in selected])
        if metrics["hidden_samples"] != expected["lost_by_regime"][regime]:
            raise ValueError(f"Frozen hidden-position total differs: {regime}")
        if locf_metrics != archived_pooled[regime]["metrics"]:
            raise ValueError(f"Pooled LOCF metrics differ from archive: {regime}")
        pooled[regime] = {
            "segments": len(selected), "samples": sample_count,
            "actual_missing_fraction": metrics["hidden_samples"] / sample_count,
            "max_missing_run": max(case["max_missing_run"] for case in selected),
            "metrics": metrics, "locf_metrics": locf_metrics,
        }
    return pooled


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/linear_validation_subject107_all_regimes.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    config_path = ROOT / "configs/pamap2_split_v1.json"
    manifest = ROOT / "configs/pamap2_segments_v1.csv"
    processed = ROOT / "data/processed/pamap2_v1"
    cap_path = ROOT / "data/processed/linear_extrapolation_v1/slope_limits.json"
    archive_path = ROOT / "benchmarks/locf_validation_subject107_v1/benchmark.json"
    preparation_path = processed / "preparation_report.json"
    config = json.loads(config_path.read_text())
    preparation = json.loads(preparation_path.read_text())
    caps = load_frozen_caps(cap_path, config, preparation, expected_sha256=CAP_SHA256)
    expected = preparation["splits"]["validation"]
    if config["validation_subjects"] != [107] or expected["subjects"] != [107]:
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
    for path in (Path(__file__).resolve(), ROOT / "scripts/evaluate_linear_extrapolation.py"):
        sources[str(path.relative_to(ROOT))] = sha256(path)
    cap_sidecar = cap_path.with_suffix(".json.sha256")
    sidecar_hash = sha256(cap_sidecar)
    dataset = ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset"
    recording = dataset / "Protocol/subject107.dat"
    recording_hash = sha256(recording)
    if recording_hash != archive["provenance"]["recording_sha256"]:
        raise ValueError("Validation recording differs from archived LOCF input")

    segments, masks = load_subject_masks(
        artifact_paths["loss_masks"], manifest, config_path, "validation", 107,
        history_preserved_samples=preparation["history_preserved_samples"],
    )
    if (len(segments) != expected["segments"]
            or sum(int(segment["samples"]) for segment in segments) != expected["samples"]):
        raise ValueError("Manifest totals differ from preparation")
    cases = []
    for index, segment in enumerate(segments):
        timestamps, truth = load_split_segment(
            dataset, manifest, config_path, "validation", 107, index,
        )
        for regime in REGIMES:
            missing = masks[index, regime]
            locf = replay_segment(LOCF(), truth, missing)
            locf_metrics = score_hidden(truth, missing, locf)
            check_locf_reference(archive, index, regime, segment, locf, locf_metrics)
            result, metrics = checked_linear_replay(caps["max_delta_per_step"], truth, missing)
            cases.append({
                "segment_index": index, "segment": segment, "regime": regime,
                "samples": len(truth), "actual_missing_fraction": float(missing.mean()),
                "timestamp_start_s": float(timestamps[0]), "timestamp_end_s": float(timestamps[-1]),
                "max_missing_run": int(result.missing_run.max()),
                "replay_sha256": replay_digest(result), "locf_replay_sha256": replay_digest(locf),
                "metrics": metrics, "locf_metrics": locf_metrics,
            })
        print(f"Verified segment {index + 1}/{len(segments)} under all four regimes", flush=True)
    pooled = summarize_cases(cases, expected, archive["pooled_by_regime"])
    comparison = {
        regime: [
            {"channel": name, "unit": "m/s²" if i < 3 else "rad/s",
             "locf_mae": entry["locf_metrics"]["mae"][i],
             "linear_mae": entry["metrics"]["mae"][i],
             "locf_rmse": entry["locf_metrics"]["rmse"][i],
             "linear_rmse": entry["metrics"]["rmse"][i]}
            for i, name in enumerate(config["channels"])
        ] for regime, entry in pooled.items()
    }

    for name, path in artifact_paths.items():
        if sha256(path) != hashes[name]:
            raise ValueError(f"Input changed during evaluation: {name}")
    for path, value in ((cap_path, CAP_SHA256), (cap_sidecar, sidecar_hash),
                        (archive_path, LOCF_SHA256), (recording, recording_hash)):
        if sha256(path) != value:
            raise ValueError(f"Input changed during evaluation: {path}")
    for name, value in sources.items():
        if sha256(ROOT / name) != value:
            raise ValueError(f"Source changed during evaluation: {name}")
    report = {
        "schema_version": "1", "predictor": "LinearExtrapolator",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "validation", "subject": 107, "segments": len(segments),
        "samples": expected["samples"], "cases": len(cases),
        "channel_order": config["channels"], "channel_units": ["m/s²"] * 3 + ["rad/s"] * 3,
        "sampling_frequency_hz": 100, "normalization_applied": False,
        "mask_seed": preparation["seed"],
        "history_preserved_samples": preparation["history_preserved_samples"],
        "loss_source": "frozen synthetic sample losses; no transport timing simulation",
        "slope_caps": caps["max_delta_per_step"], "slope_cap_units": caps["cap_units"],
        "aggregation": "sum absolute/squared errors, divide by scored samples per regime/channel",
        "verification": {
            "locf_matches_archived_case_all_cases": True,
            "identical_repeated_replays_and_reset_all_cases": True,
            "received_samples_and_ground_truth_preserved": True,
            "hidden_truth_nan_replay_identical_all_cases": True,
            "future_change_prefix_identical_all_cases": True,
            "all_hidden_positions_scored": True, "case_and_sample_totals_reconciled": True,
            "pooled_locf_matches_archive": True, "caps_data_and_sources_unchanged": True,
        },
        "pooled_by_regime": pooled, "comparison_by_regime": comparison, "per_segment": cases,
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status_porcelain": git_output("status", "--porcelain"),
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "runtime_dependency_pins": [f"numpy=={np.__version__}"],
            "artifact_sha256": hashes, "recording_sha256": recording_hash,
            "source_sha256": sources, "slope_caps_sha256": CAP_SHA256,
            "slope_caps_sidecar_sha256": sidecar_hash, "locf_benchmark_sha256": LOCF_SHA256,
            "training_recording_check": "Calibration-time hashes retained in pinned cap artifact; training raw files not reopened",
        },
        "interpretation": "One validation subject and one frozen seed per regime; no final-test, uncertainty, or device performance claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(args.output.resolve()), "cases": len(cases),
                      "comparison_by_regime": comparison}, indent=2))


if __name__ == "__main__":
    main()
