#!/usr/bin/env python3
"""Fit and freeze six 99th-percentile slope caps using training subjects only.

No validation/test recording, loss-mask simulation, or accuracy evaluation is
used. Existing output files are refused. Run from any directory.
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
from packetheal.calibration import QUANTILE, calibrate_training_slope_caps
from packetheal.linear_extrapolation import LinearExtrapolator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/linear_extrapolation_v1/slope_limits.json")
    args = parser.parse_args()
    checksum_path = Path(str(args.output) + ".sha256")
    if args.output.exists() or checksum_path.exists():
        raise FileExistsError(f"Refusing to overwrite calibration output: {args.output}")
    config_path = ROOT / "configs/pamap2_split_v1.json"
    manifest_path = ROOT / "configs/pamap2_segments_v1.csv"
    processed = ROOT / "data/processed/pamap2_v1"
    preparation_path = processed / "preparation_report.json"
    audit_path = ROOT / "data/reports/pamap2_audit/audit.json"
    dataset_dir = ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset"
    preparation = json.loads(preparation_path.read_text())
    config = json.loads(config_path.read_text())
    artifacts = {
        "split_config": (config_path, "split_config_sha256"),
        "segment_manifest": (manifest_path, "segment_manifest_sha256"),
        "loss_masks": (processed / "loss_masks.csv", "mask_sha256"),
        "normalization": (processed / "normalization.json", "normalization_sha256"),
    }
    input_hashes = {}
    for name, (path, key) in artifacts.items():
        input_hashes[name] = sha256(path)
        if input_hashes[name] != preparation[key]:
            raise ValueError(f"Frozen artifact hash mismatch: {path}")
    input_hashes["preparation_report"] = sha256(preparation_path)
    input_hashes["audit_report"] = sha256(audit_path)
    expected = preparation["splits"]["train"]
    if expected["subjects"] != list(range(101, 107)):
        raise ValueError("Preparation report must identify training subjects 101–106")
    audit = {row["file"]: row["sha256"] for row in json.loads(audit_path.read_text())["subjects"]}
    recordings = {}
    for subject in expected["subjects"]:
        path = dataset_dir / "Protocol" / f"subject{subject}.dat"
        digest = sha256(path)
        if digest != audit[path.name]:
            raise ValueError(f"Training recording differs from the audited input: {path}")
        recordings[str(subject)] = {"source_file": path.name, "sha256": digest}
    source_paths = [ROOT / "src/packetheal" / name for name in (
        "calibration.py", "linear_extrapolation.py", "dataset.py", "splits.py",
        "baselines.py", "evaluation.py",
    )] + [Path(__file__).resolve(), ROOT / "scripts/evaluate_locf.py"]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}

    def progress(subject, completed, total):
        if completed % 25 == 0 or completed == total:
            print(f"Training subject {subject}: {completed}/{total} segments loaded", flush=True)

    result = calibrate_training_slope_caps(
        dataset_dir, manifest_path, config_path, quantile=QUANTILE, progress=progress,
    )
    if (result["segments"] != expected["segments"] or result["samples"] != expected["samples"]
            or result["pairs_per_channel"] != expected["samples"] - expected["segments"]):
        raise ValueError("Calibration counts disagree with the preparation report")
    LinearExtrapolator(result["max_delta_per_step"])
    for name, (path, _) in artifacts.items():
        if sha256(path) != input_hashes[name]:
            raise ValueError(f"Artifact changed during calibration: {path}")
    for name, path in (("preparation_report", preparation_path), ("audit_report", audit_path)):
        if sha256(path) != input_hashes[name]:
            raise ValueError(f"Metadata changed during calibration: {path}")
    for subject, info in recordings.items():
        if sha256(dataset_dir / "Protocol" / info["source_file"]) != info["sha256"]:
            raise ValueError(f"Training recording {subject} changed during calibration")
    for relative, digest in source_hashes.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"Source changed during calibration: {relative}")
    artifact = {
        "schema_version": "1", "purpose": "linear_extrapolation_slope_caps",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **result,
        "sampling_frequency_hz": config["sampling_frequency_hz"],
        "channel_order": config["channels"],
        "cap_units": ["m/s² per sample step"] * 3 + ["rad/s per sample step"] * 3,
        "pair_definition": "abs(x[i] - x[i-1]) inside each eligible segment; no cross-boundary pairs",
        "normalization_applied": False, "loss_masks_applied": False,
        "verification": {
            "counts_match_preparation": True,
            "repeat_quantile_identical": result["repeat_quantile_identical"],
            "limits_accepted_by_predictor": True, "inputs_and_sources_unchanged": True,
        },
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_status_porcelain": git_output("status", "--porcelain"),
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "runtime_dependency_pins": [f"numpy=={np.__version__}"],
            "randomness": "none", "artifact_sha256": input_hashes,
            "training_recordings": recordings, "source_sha256": source_hashes,
        },
        "interpretation": "Training-change quantiles; not physical validity bounds, confidence intervals, or measured prediction accuracy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(artifact, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    digest = sha256(args.output)
    with checksum_path.open("x") as stream:
        stream.write(f"{digest}  {args.output.name}\n")
    print(json.dumps({
        "artifact": str(args.output.resolve()), "sha256": digest,
        "subjects": result["subjects"], "segments": result["segments"],
        "samples": result["samples"], "pairs_per_channel": result["pairs_per_channel"],
        "channel_order": config["channels"], "max_delta_per_step": result["max_delta_per_step"],
    }, indent=2))


if __name__ == "__main__":
    main()
