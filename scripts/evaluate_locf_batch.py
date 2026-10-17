#!/usr/bin/env python3
"""Evaluate all validation subject-107 segments under all four frozen regimes.

    python3 scripts/evaluate_locf_batch.py --output /path/to/new-report.json

Every case is replayed twice to verify exact repeatability. This script has no
held-out subject option and does not generate or modify experimental masks.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_locf import git_output, sha256
from packetheal.baselines import LOCF
from packetheal.evaluation import (
    REGIMES, load_subject_masks, pool_metrics, replay_segment, score_hidden,
)
from packetheal.splits import load_split_segment


def replay_digest(result):
    digest = hashlib.sha256()
    for values in (result.values, result.status, result.missing_run):
        digest.update(values.tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/locf_validation_subject107_all_regimes.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    config_path = ROOT / "configs/pamap2_split_v1.json"
    manifest_path = ROOT / "configs/pamap2_segments_v1.csv"
    processed = ROOT / "data/processed/pamap2_v1"
    preparation_path = processed / "preparation_report.json"
    preparation = json.loads(preparation_path.read_text())
    artifacts = {
        "split_config": (config_path, "split_config_sha256"),
        "segment_manifest": (manifest_path, "segment_manifest_sha256"),
        "loss_masks": (processed / "loss_masks.csv", "mask_sha256"),
        "normalization": (processed / "normalization.json", "normalization_sha256"),
    }
    hashes = {}
    for name, (path, key) in artifacts.items():
        hashes[name] = sha256(path)
        if hashes[name] != preparation[key]:
            raise ValueError(f"Frozen artifact hash mismatch: {path}")
    hashes["preparation_report"] = sha256(preparation_path)
    config = json.loads(config_path.read_text())
    expected = preparation["splits"]["validation"]
    if expected["subjects"] != [107]:
        raise ValueError("This runner expects validation subject 107 only")
    segments, masks = load_subject_masks(
        processed / "loss_masks.csv", manifest_path, config_path, "validation", 107,
        history_preserved_samples=preparation["history_preserved_samples"],
    )
    if len(segments) != expected["segments"]:
        raise ValueError("Segment count disagrees with preparation report")
    dataset_dir = ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset"
    recording = dataset_dir / "Protocol/subject107.dat"
    recording_hash = sha256(recording)
    cases = []
    sample_count = 0
    predictor = LOCF(6)
    for index, segment in enumerate(segments):
        timestamps, truth = load_split_segment(
            dataset_dir, manifest_path, config_path, "validation", 107, index,
        )
        original = truth.copy()
        sample_count += len(truth)
        for regime in REGIMES:
            missing = masks[index, regime]
            first = replay_segment(predictor, truth, missing)
            second = replay_segment(predictor, truth, missing)
            for field in ("values", "status", "missing_run"):
                np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
            np.testing.assert_array_equal(truth, original)
            np.testing.assert_array_equal(first.values[~missing], truth[~missing])
            metrics = score_hidden(truth, missing, first)
            if metrics != score_hidden(truth, missing, second):
                raise ValueError("Repeated replay metrics differ")
            if metrics["unavailable_samples"] or metrics["scored_samples"] != int(missing.sum()):
                raise ValueError("Hidden positions were not all reconstructed")
            cases.append({
                "segment_index": index, "segment": segment, "regime": regime,
                "samples": len(truth), "actual_missing_fraction": float(missing.mean()),
                "timestamp_start_s": float(timestamps[0]), "timestamp_end_s": float(timestamps[-1]),
                "max_missing_run": int(first.missing_run.max()),
                "replay_sha256": replay_digest(first), "metrics": metrics,
            })
        print(f"Verified segment {index + 1}/{len(segments)} under all four regimes", flush=True)
    if sample_count != expected["samples"] or len(cases) != len(segments) * len(REGIMES):
        raise ValueError("Sample/case totals disagree with preparation report")
    pooled = {}
    for regime in REGIMES:
        selected = [case for case in cases if case["regime"] == regime]
        metrics = pool_metrics([case["metrics"] for case in selected])
        if metrics["hidden_samples"] != expected["lost_by_regime"][regime]:
            raise ValueError(f"Hidden-position count mismatch for {regime}")
        pooled[regime] = {
            "segments": len(selected), "samples": sample_count,
            "actual_missing_fraction": metrics["hidden_samples"] / sample_count,
            "max_missing_run": max(case["max_missing_run"] for case in selected),
            "metrics": metrics,
        }
    # Check inputs again after execution to detect accidental modification.
    for name, (path, _) in artifacts.items():
        if sha256(path) != hashes[name]:
            raise ValueError(f"Artifact changed during evaluation: {path}")
    if sha256(preparation_path) != hashes["preparation_report"] or sha256(recording) != recording_hash:
        raise ValueError("Preparation report or recording changed during evaluation")
    source_paths = [
        ROOT / "src/packetheal" / name
        for name in ("baselines.py", "evaluation.py", "dataset.py", "splits.py")
    ] + [Path(__file__).resolve(), ROOT / "scripts/evaluate_locf.py"]
    report = {
        "schema_version": "1", "predictor": "LOCF", "split": "validation", "subject": 107,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "segments": len(segments), "samples": sample_count, "cases": len(cases),
        "channel_order": config["channels"], "channel_units": ["m/s²"] * 3 + ["rad/s"] * 3,
        "sampling_frequency_hz": config["sampling_frequency_hz"],
        "normalization_applied": False, "mask_seed": preparation["seed"],
        "history_preserved_samples": preparation["history_preserved_samples"],
        "loss_source": "frozen synthetic sample losses; no transport timing simulation",
        "aggregation": "sum absolute/squared errors, divide by scored samples per regime/channel",
        "verification": {
            "identical_repeated_replays_all_cases": True,
            "received_samples_and_ground_truth_preserved": True,
            "all_hidden_positions_scored": True, "input_hashes_unchanged": True,
        },
        "pooled_by_regime": pooled, "per_segment": cases,
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status_porcelain": git_output("status", "--porcelain"),
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "artifact_sha256": hashes, "recording_sha256": recording_hash,
            "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(args.output.resolve()), "pooled_by_regime": pooled}, indent=2))


if __name__ == "__main__":
    main()
