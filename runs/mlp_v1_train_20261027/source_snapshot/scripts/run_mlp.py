#!/usr/bin/env python3
"""Train or validate the small MLP. Default synthetic smoke touches no dataset."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from packetheal.mlp import Normalization, segment_examples
from packetheal.mlp_experiment import train_models, train_experiment, validation_experiment, write_json


def synthetic_experiment(output_dir):
    """Check learning end-to-end using causally hidden artificial ramps only."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    normalization = Normalization(np.zeros(6), np.ones(6))
    x, direct, residual, ids = [], [], [], []
    slopes = np.array([.002, -.003, .004, -.002, .001, -.001])
    truth = np.arange(120)[:, None] * slopes + np.array([.1, .2, .3, .4, .5, .6])
    for regime, length in enumerate((1, 2, 5, 10)):
        missing = np.zeros(len(truth), dtype=bool)
        for start in (25, 50, 75, 100):
            missing[start:start + length] = True
        features, target, correction, _ = segment_examples(truth, missing, normalization)
        x.append(features)
        direct.append(target)
        residual.append(correction)
        ids.append(np.full(len(features), regime))
    targets = {"direct": np.concatenate(direct), "residual": np.concatenate(residual)}
    models = train_models(np.concatenate(x), targets, np.concatenate(ids), output_dir / "models",
                          epochs=200, batch_size=64, seeds=(0,), progress=lambda message: None)
    locf = float(np.mean([np.abs(targets["residual"][np.concatenate(ids) == i]).mean() for i in range(4)]))
    residual_result = next(model for model in models if model["mode"] == "residual")
    report = {"synthetic_only": True, "dataset_recordings_opened": 0,
              "locf_equal_regime_normalized_mae": locf, "models": models,
              "residual_improves_over_locf": residual_result["final_equal_regime_normalized_mae"] < locf,
              "note": "Training-fixture fit only; not evidence of PAMAP2 generalization"}
    write_json(output_dir / "synthetic.json", report)
    if not report["residual_improves_over_locf"]:
        raise AssertionError("Synthetic residual learning check failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("synthetic", "train", "validate"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-dir", type=Path, help="PAMAP2_Dataset root containing Protocol/")
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--phase2-accepted", action="store_true", help="Acknowledge recorded Phase 2 acceptance for dataset training")
    args = parser.parse_args()
    if args.command == "synthetic":
        report = synthetic_experiment(args.output_dir)
        print(json.dumps({"synthetic_only": True, "residual_improves_over_locf": report["residual_improves_over_locf"]}))
        return
    if args.dataset_dir is None:
        parser.error("--dataset-dir must identify the PAMAP2_Dataset root containing Protocol/")
    if args.command == "train":
        if not args.phase2_accepted:
            parser.error("Acknowledge Phase 2 acceptance with --phase2-accepted before dataset training")
        train_experiment(args.root, args.dataset_dir, args.output_dir)
    else:
        if args.training_dir is None:
            parser.error("--training-dir is required for validation")
        validation_experiment(args.root, args.dataset_dir, args.training_dir, args.output_dir)
    print(f"Saved {args.command} evidence to {args.output_dir}")


if __name__ == "__main__":
    main()
