"""Frozen-input verification shared by alpha-beta selection and evaluation."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_linear_extrapolation import CAP_SHA256, load_frozen_caps
from evaluate_locf import sha256


def load_context(extra_sources):
    processed = ROOT / "data/processed/pamap2_v1"
    paths = {
        "split_config": ROOT / "configs/pamap2_split_v1.json",
        "segment_manifest": ROOT / "configs/pamap2_segments_v1.csv",
        "loss_masks": processed / "loss_masks.csv",
        "normalization": processed / "normalization.json",
        "preparation_report": processed / "preparation_report.json",
        "audit_report": ROOT / "data/reports/pamap2_audit/audit.json",
    }
    config = json.loads(paths["split_config"].read_text())
    preparation = json.loads(paths["preparation_report"].read_text())
    cap_path = ROOT / "data/processed/linear_extrapolation_v1/slope_limits.json"
    caps = load_frozen_caps(cap_path, config, preparation, expected_sha256=CAP_SHA256)
    hashes = {name: sha256(path) for name, path in paths.items()}
    if hashes != caps["provenance"]["artifact_sha256"]:
        raise ValueError("Current artifacts differ from the frozen calibration inputs")
    normalization = json.loads(paths["normalization"].read_text())
    scales = np.asarray(normalization["std_sample"], dtype=float)
    if (normalization["fit_split"] != "train"
            or normalization["channel_order"] != config["channels"]
            or normalization["count_training_samples"] != preparation["splits"]["train"]["samples"]
            or scales.shape != (6,) or not np.isfinite(scales).all() or (scales <= 0).any()):
        raise ValueError("Normalization does not match the training-only six-channel contract")
    sources = dict(caps["provenance"]["source_sha256"])
    for name, expected in sources.items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Frozen calibration source changed: {name}")
    for name in {"scripts/alpha_beta_common.py", "scripts/evaluate_linear_extrapolation.py",
                 "scripts/evaluate_locf_batch.py", "src/packetheal/alpha_beta.py", *extra_sources}:
        sources[name] = sha256(ROOT / name)
    paths["slope_caps"] = cap_path
    paths["slope_caps_sidecar"] = cap_path.with_suffix(".json.sha256")
    hashes.update({name: sha256(paths[name]) for name in ("slope_caps", "slope_caps_sidecar")})
    return {"paths": paths, "hashes": hashes, "sources": sources, "config": config,
            "preparation": preparation, "caps": caps, "scales": scales,
            "dataset": ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset"}


def verify_unchanged(context):
    for name, path in context["paths"].items():
        if sha256(path) != context["hashes"][name]:
            raise ValueError(f"Input changed during experiment: {path}")
    for name, expected in context["sources"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Source changed during experiment: {name}")
