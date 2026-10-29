"""Independent scalar forward calculation and edit checks; no raw data opens."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/mlp_v1_accy_sensitivity_20261027"
PRIOR = ROOT / "runs/mlp_v1_gyroy_burst5_failure_20261027"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)


def affine(values, weights, bias):
    # Python floats and math.fsum give a float64 reference, independent of
    # the predictor's float32 NumPy matrix multiplications.
    return [math.fsum(float(v) * float(weights[i, j]) for i, v in enumerate(values)) + float(bias[j])
            for j in range(len(bias))]


def main():
    output = OUT / "independent_checks.json"
    if output.exists():
        raise FileExistsError(output)
    raw_attempts = []

    def deny_raw(event, args):
        if event == "open" and args and not isinstance(args[0], int):
            p = Path(os.fsdecode(args[0])).resolve()
            if p.suffix == ".dat" or p.is_relative_to(ROOT / "data/raw"):
                raw_attempts.append(str(p))
                raise PermissionError("Raw data prohibited in saved-input verifier")

    sys.addaudithook(deny_raw)
    policy = json.loads((OUT / "policy.json").read_text())
    report = json.loads((OUT / "report.json").read_text())
    assert report["policy"] == policy and report["checks_passed"]
    assert digest(ROOT / "runs/mlp_v1_accy_sensitivity_20261027.py") == policy["runner_sha256"]
    assert digest(PRIOR / "evidence.sha256") == policy["prior_manifest_sha256"]
    for relative, expected in {**policy["prior_artifact_hashes"], **policy["frozen_inputs"], **report["source_hashes"]}.items():
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT) and path.suffix != ".dat"
        assert digest(path) == expected
    with np.load(OUT / "arrays.npz", allow_pickle=False) as stream:
        arrays = {name: stream[name].copy() for name in stream.files}
    x, changed = arrays["original_features"], arrays["modified_features"]
    mask, anchors = arrays["edit_mask"], arrays["anchors"]
    mean, std = arrays["normalization_mean"], arrays["normalization_std"]
    rows, z = arrays["original_rows"], arrays["normalized_outputs"]
    assert x.shape == changed.shape == mask.shape == (5, 161)
    assert x.dtype == changed.dtype == z.dtype == np.float32 and mask.dtype == np.bool_
    assert z.shape == arrays["predictions"].shape == (2, 3, 5, 6)
    assert arrays["predictions"].dtype == np.float64
    expected_mask = np.zeros_like(mask)
    source = policy["source_received_row"]
    # Derive locations from causal time indices, not the runner's source-ID loop.
    for t, current in enumerate(rows):
        for slot in range(20):
            slot_row = int(current) - 20 + slot
            if source <= slot_row < int(current):
                expected_mask[t, slot * 8 + 1] = True
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(x != changed, mask)
    np.testing.assert_array_equal(x[~mask], changed[~mask])
    np.testing.assert_array_equal(changed[mask], np.full(15, np.float32((policy["replacement_acc_y"] - mean[1]) / std[1])))
    np.testing.assert_array_equal(x[:, -1], np.array([.1, .2, .3, .4, .5], dtype=np.float32))
    np.testing.assert_array_equal(changed[:, -1], x[:, -1])
    np.testing.assert_array_equal(anchors[0, [0, 2, 3, 4, 5]], anchors[1, [0, 2, 3, 4, 5]])
    close(anchors[1, 1], policy["replacement_acc_y"])
    with np.load(PRIOR / "features.npz", allow_pickle=False) as previous:
        np.testing.assert_array_equal(previous["features"], x)
        np.testing.assert_array_equal(previous["anchor"], anchors[0])
        for seed in range(3):
            np.testing.assert_array_equal(z[0, seed], previous[f"residual_seed{seed}_normalized_corrections"])
    # Tolerance fixed before reference computations: float64 scalar arithmetic
    # need not be bit-identical to the recorded float32 BLAS calculations.
    max_difference = 0.0
    for seed in range(3):
        name = f"residual_seed{seed}.npz"
        checkpoint = ROOT / "runs/mlp_v1_train_20261027/models" / name
        assert digest(checkpoint) == report["checkpoint_hashes"][name]
        with np.load(checkpoint, allow_pickle=False) as saved:
            # Discover serialization names from the checkpoint's explicit
            # architecture metadata and parameter arrays; no MLP code imported.
            weights = [saved[f"p{i}"].copy() for i in range(6)]
        for scenario, features in enumerate((x, changed)):
            for t in range(5):
                h1 = [max(v, 0.0) for v in affine(features[t], weights[0], weights[1])]
                h2 = [max(v, 0.0) for v in affine(h1, weights[2], weights[3])]
                reference = np.array(affine(h2, weights[4], weights[5]))
                np.testing.assert_allclose(reference, z[scenario, seed, t], rtol=2e-5, atol=2e-5)
                max_difference = max(max_difference, float(np.max(np.abs(reference - z[scenario, seed, t]))))
    np.testing.assert_array_equal(z.astype(np.float64) * std, arrays["physical_corrections"])
    np.testing.assert_array_equal(arrays["physical_corrections"] + anchors[:, None, None, :], arrays["predictions"])
    with (OUT / "predictions.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    with (OUT / "comparison.csv").open(newline="") as stream:
        comparisons = list(csv.DictReader(stream))
    assert len(records) == 30 and len(comparisons) == 15
    assert {(r["case"], int(r["seed"]), int(r["loss_age"])) for r in records} == {
        (name, s, t) for name in policy["cases"] for s in range(3) for t in range(1, 6)}
    for row in records:
        scenario = policy["cases"].index(row["case"])
        seed, t = int(row["seed"]), int(row["loss_age"]) - 1
        assert int(row["original_row"]) == rows[t]
        close(row["gyro_y_anchor"], anchors[scenario, 4])
        close(row["normalized_gyro_y_correction"], z[scenario, seed, t, 4])
        close(row["physical_gyro_y_correction"], arrays["physical_corrections"][scenario, seed, t, 4])
        close(row["gyro_y_prediction"], arrays["predictions"][scenario, seed, t, 4])
    for row in comparisons:
        seed, t = int(row["seed"]), int(row["loss_age"]) - 1
        close(row["original_gyro_y_prediction"], arrays["predictions"][0, seed, t, 4])
        close(row["modified_gyro_y_prediction"], arrays["predictions"][1, seed, t, 4])
        close(row["prediction_change"], float(row["modified_gyro_y_prediction"]) - float(row["original_gyro_y_prediction"]))
    with (OUT / "edited_positions.csv").open(newline="") as stream:
        edits = list(csv.DictReader(stream))
    assert len(edits) == 15
    assert {(int(r["loss_age"])-1, int(r["feature_column"])) for r in edits} == set(zip(*np.nonzero(expected_mask)))
    assert not raw_attempts and not report["checks"]["raw_recording_open_attempts"]
    checks = {"verification_pass": True, "original_predictions_exact": 15,
              "declared_acc_y_input_edits": 15, "independent_scalar_network_outputs": 30,
              "scalar_float64_vs_float32_tolerance": {"rtol": 2e-5, "atol": 2e-5},
              "maximum_normalized_output_difference": max_difference,
              "prediction_csv_rows_verified": 30, "comparison_csv_rows_verified": 15,
              "raw_recording_open_attempts": raw_attempts,
              "report_sha256": digest(OUT / "report.json"), "verifier_sha256": digest(__file__)}
    with output.open("x") as stream:
        json.dump(checks, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
