#!/usr/bin/env python3
"""Connect saved residual MLPs to the desktop consumer using artificial inputs."""

from dataclasses import asdict
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "runs")]
from packetheal.baselines import SampleStatus
from packetheal.mlp import MLPPredictor
from packetheal.mlp_experiment import (
    digest, normalization_from_root, validate_training_record, verify_frozen, write_json,
)
from mlp_v1_abstention_20261027 import AbstainingMLP
from mlp_v1_consumer_20261027 import StatusConsumer


def replay(predictor, consumer, observations):
    predictor.reset()
    consumer.reset()
    reasons = []
    for observation in observations:
        result = predictor.step(None if observation is None else observation.copy())
        consumer.consume(result)
        reasons.append(predictor.reason)
    return [asdict(record) for record in consumer.records], reasons


def exercise_model(model, norm, caps):
    # Artificial, controlled input; these slots do not measure model accuracy.
    base = norm.mean.copy()
    spike = base.copy()
    spike[0] += 4 * caps[0]  # Across three elapsed slots: warning score 4/3.
    observations = ([None] + [base.copy() for _ in range(20)] + [None, None]
                    + [spike.copy(), None, None, spike.copy()]
                    + [None] * 11 + [base.copy(), None])
    predictor, consumer = AbstainingMLP(model, norm, caps), StatusConsumer()
    records, reasons = replay(predictor, consumer, observations)
    counts = {status.value: sum(r["status"] == status.value for r in records)
              for status in SampleStatus}
    assert counts == {"MEASURED": 23, "RECONSTRUCTED": 13, "UNAVAILABLE": 4}
    assert [r["sample_index"] for r in records] == list(range(40))
    assert [r["sample_index"] for r in records if r["values"] is None] == [0, 24, 25, 37]
    assert [reasons[i] for i in [0, 24, 25, 37]] == [
        "INSUFFICIENT_HISTORY", "RECEIVED_CHANGE", "RECEIVED_CHANGE", "HORIZON_EXCEEDED"]

    # Compare exposed estimates against the original saved model's streaming path.
    original = MLPPredictor(model, norm)
    last_real, age = None, 0
    for i, observation in enumerate(observations):
        result = original.step(None if observation is None else observation.copy())
        record = records[i]
        age = age + 1 if observation is None else 0
        assert record["missing_run"] == age
        if observation is not None:
            last_real = observation
            assert record["status"] == SampleStatus.MEASURED.value
            np.testing.assert_array_equal(record["values"], observation)
        elif record["status"] == SampleStatus.RECONSTRUCTED.value:
            np.testing.assert_array_equal(record["values"], result.values)
    np.testing.assert_array_equal(predictor.predictor.state.anchor, last_real)
    assert np.isfinite(predictor.predictor.state.rows).all()

    # A -> other stream -> A: reset both components together.
    replay(predictor, consumer, [base + norm.std] * 23 + [None] * 12)
    assert replay(predictor, consumer, observations) == (records, reasons)
    for cutoff in (20, 24, 37):
        changed = [None if x is None else x.copy() for x in observations]
        for i in range(cutoff, len(changed)):
            changed[i] = base + norm.std if changed[i] is None else None
        altered, _ = replay(predictor, consumer, changed)
        assert altered[:cutoff] == records[:cutoff]
    json.dumps(records, allow_nan=False)
    return {"counts": counts, "records": records, "reasons": reasons,
            "measured_values_unchanged": True, "accepted_values_match_original": True,
            "gap_indices_preserved": True, "reset_a_b_a_pass": True,
            "future_prefix_cutoffs_checked": [20, 24, 37],
            "real_anchor_preserved": True, "consumer_json_has_no_nan": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "runs") or output == ROOT / "runs":
        raise ValueError("Use a new output directory under runs/")
    if output.exists():
        raise FileExistsError(output)

    raw_attempts = []
    def deny_raw(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(arguments[0])).absolute()
            if path.is_relative_to(ROOT / "data/raw") or path.resolve().is_relative_to(ROOT / "data/raw"):
                raw_attempts.append(str(path))
                raise PermissionError("This integration check must not open raw recordings")
    sys.addaudithook(deny_raw)

    # Check the committed preservation checkpoint, including prior reports/source.
    manifest = ROOT / "benchmarks/phase3_checkpoint_20261029/evidence.sha256"
    evidence = {}
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or path.is_relative_to(ROOT / "data/raw"):
            raise ValueError("Unsafe evidence path")
        if digest(path) != expected:
            raise ValueError(f"Preserved evidence changed: {name}")
        evidence[name] = expected
    frozen = verify_frozen(ROOT)
    training, models = validate_training_record(ROOT, ROOT / "runs/mlp_v1_train_20261027")
    norm = normalization_from_root(ROOT)
    # Read the original warning thresholds, not new fitted or synthetic limits.
    warning_policy = json.loads((ROOT / "runs/mlp_v1_received_warning_20261027_retry1/policy.json").read_text())
    caps = np.asarray(warning_policy["thresholds_per_nominal_step"], dtype=np.float64)
    checked = []
    for metadata, model in models:
        if metadata["mode"] != "residual":
            continue
        parameters = [value.copy() for value in model.parameters]
        result = exercise_model(model, norm, caps)
        for before, after in zip(parameters, model.parameters):
            np.testing.assert_array_equal(before, after)
        checked.append({"seed": metadata["seed"], "checkpoint_sha256": metadata["sha256"], **result})
    assert {row["seed"] for row in checked} == {0, 1, 2}
    for name, expected in evidence.items():
        assert digest(ROOT / name) == expected, name
    assert verify_frozen(ROOT) == frozen
    validate_training_record(ROOT, ROOT / "runs/mlp_v1_train_20261027")
    assert not raw_attempts
    output.mkdir()
    write_json(output / "integration.json", {
        "date": "2026-10-29", "kind": "trained_model_consumer_integration",
        "scope": "Three trained residual seeds; artificial 40-slot inputs; software checks only, not new accuracy evidence",
        "software_correctness_pass": True, "application_acceptance": "NOT_ASSESSED",
        "trained_model_shape": "float32 [161] -> [32] -> [16] -> [6]; physical float64 output [6]",
        "normalization": "Unchanged frozen training means/sample standard deviations",
        "warning_thresholds": caps.tolist(), "model_results": checked,
        "raw_recording_open_attempts": raw_attempts, "frozen_inputs": frozen,
        "preserved_evidence_entries_checked_before_after": len(evidence),
        "training_report_sha256": digest(ROOT / "runs/mlp_v1_train_20261027/training.json"),
        "script_sha256": digest(__file__), "python": sys.version, "numpy": np.__version__,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
    })
    print("PASS: all 3 trained residual models -> abstention -> consumer; 23 measured / 13 reconstructed / 4 gaps each; zero raw opens")


if __name__ == "__main__":
    main()
