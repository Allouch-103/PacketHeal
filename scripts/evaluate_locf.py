#!/usr/bin/env python3
"""Replay one train/validation segment using frozen loss positions and LOCF.

Run from any directory: python3 /path/to/PacketHeal/scripts/evaluate_locf.py
Held-out test partitions are deliberately excluded from this development CLI.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from packetheal.baselines import LOCF
from packetheal.evaluation import (
    REGIMES, load_segment_mask, replay_segment, score_hidden, segment_metadata,
)
from packetheal.splits import load_split_segment


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--subject", type=int, default=107)
    parser.add_argument("--segment-index", type=int, default=0)
    parser.add_argument("--regime", choices=REGIMES, default="random_5pct")
    parser.add_argument("--output", type=Path, help="New JSON file; existing files are refused")
    args = parser.parse_args()

    config_path = ROOT / "configs/pamap2_split_v1.json"
    manifest_path = ROOT / "configs/pamap2_segments_v1.csv"
    processed = ROOT / "data/processed/pamap2_v1"
    mask_path = processed / "loss_masks.csv"
    preparation_path = processed / "preparation_report.json"
    preparation = json.loads(preparation_path.read_text())
    artifacts = {
        "split_config": (config_path, "split_config_sha256"),
        "segment_manifest": (manifest_path, "segment_manifest_sha256"),
        "loss_masks": (mask_path, "mask_sha256"),
        "normalization": (processed / "normalization.json", "normalization_sha256"),
    }
    hashes = {}
    for name, (path, key) in artifacts.items():
        hashes[name] = sha256(path)
        if hashes[name] != preparation[key]:
            raise ValueError(f"Frozen artifact hash mismatch: {path}")
    hashes["preparation_report"] = sha256(preparation_path)

    timestamps, truth = load_split_segment(
        ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset",
        manifest_path, config_path, args.split, args.subject, args.segment_index,
    )
    segment = segment_metadata(manifest_path, args.subject, args.segment_index)
    missing = load_segment_mask(
        mask_path, manifest_path, config_path, args.split, args.subject,
        args.segment_index, args.regime,
        history_preserved_samples=preparation["history_preserved_samples"],
    )
    result = replay_segment(LOCF(truth.shape[1]), truth, missing)
    metrics = score_hidden(truth, missing, result)
    if metrics["unavailable_samples"]:
        raise ValueError("Unexpected unavailable samples despite the protected warm-up")
    config = json.loads(config_path.read_text())
    channel_metrics = [
        {"channel": name, "unit": "m/s²" if index < 3 else "rad/s",
         "mae": metrics["mae"][index], "rmse": metrics["rmse"][index]}
        for index, name in enumerate(config["channels"])
    ]
    metrics.pop("mae")
    metrics.pop("rmse")
    metrics["per_channel"] = channel_metrics

    raw_path = (
        ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset/Protocol"
        / segment["source_file"]
    )
    source_paths = [
        ROOT / "src/packetheal" / name
        for name in ("baselines.py", "evaluation.py", "dataset.py", "splits.py")
    ] + [Path(__file__).resolve()]
    report = {
        "schema_version": "1",
        "predictor": "LOCF",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split, "subject": args.subject,
        "segment_index": args.segment_index, "segment": segment,
        "regime": args.regime, "samples": len(truth),
        "timestamp_start_s": float(timestamps[0]),
        "timestamp_end_s": float(timestamps[-1]),
        "sampling_frequency_hz": config["sampling_frequency_hz"],
        "actual_missing_fraction": float(missing.mean()),
        "max_missing_run": int(result.missing_run.max()),
        "normalization_applied": False,
        "loss_source": "frozen synthetic sample losses; no transport timing simulation",
        "mask_seed": preparation["seed"],
        "history_preserved_samples": preparation["history_preserved_samples"],
        "metrics": metrics,
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status_porcelain": git_output("status", "--porcelain"),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "artifact_sha256": hashes,
            "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
            "recording_sha256": sha256(raw_path),
        },
    }
    output = args.output or ROOT / "runs" / (
        f"locf_{args.split}_subject{args.subject}_segment{args.segment_index}_{args.regime}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"report": str(output.resolve()), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
