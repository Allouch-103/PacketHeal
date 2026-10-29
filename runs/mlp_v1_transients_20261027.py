"""Post-hoc validation strata for six frozen MLPs; no fitting or retuning."""

from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from packetheal.baselines import LOCF
from packetheal.diagnostics import large_change_flags, pool_strata, reconcile_partition, stratified_sums
from packetheal.evaluation import ReplayResult
from packetheal.mlp import MLPPredictor
from packetheal.mlp_experiment import (
    CHANNELS, REGIMES, digest, iter_segments, normalization_from_root,
    recording_access_guard, streaming_replay, validate_training_record,
    verify_frozen, write_json,
)

TRAIN = ROOT / "runs/mlp_v1_train_20261027"
VALIDATION = ROOT / "runs/mlp_v1_validation_20261027/validation.json"
CLASSICAL = ROOT / "benchmarks/phase2_validation_subject107_v1/benchmark.json"
OUT = ROOT / "runs/mlp_v1_transients_20261027"
DATASET = ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset"
GROUPS = ("all", "large_change", "other")
COUNTS = ("hidden_count", "scored_count", "unavailable_count")
SUMS = ("absolute_error_sum", "squared_error_sum")


def equal_strata(a, b):
    for key in COUNTS:
        np.testing.assert_array_equal(a[key], b[key])
    for key in SUMS:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-12)


def equal_previous(a, b):
    for new, old in zip(COUNTS, ("hidden", "scored", "unavailable")):
        np.testing.assert_array_equal(a[new], [b[old]] * 6)
    for new, old in zip(SUMS, ("sum_abs", "sum_sq")):
        np.testing.assert_allclose(a[new], b[old], rtol=1e-12, atol=1e-12)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    frozen = verify_frozen(ROOT)
    training, models = validate_training_record(ROOT, TRAIN)
    validation = json.loads(VALIDATION.read_text())
    assert digest(VALIDATION) == "3f1dde9e7e02b3dc4e73093899252de567229f4e4be2c8a83ac84fe066464604"
    assert validation["training_report_sha256"] == digest(TRAIN / "training.json")
    assert json.loads((VALIDATION.parent / "independent_checks.json").read_text())["verification_pass"]
    classical = json.loads(CLASSICAL.read_text())
    thresholds = json.loads((ROOT / "data/processed/linear_extrapolation_v1/slope_limits.json").read_text())["max_delta_per_step"]
    norm = normalization_from_root(ROOT)
    predictors = {"locf": LOCF(6)}
    predictors.update({f"{meta['mode']}_seed{meta['seed']}": MLPPredictor(model, norm) for meta, model in models})
    originals = {name: [p.copy() for p in predictor.model.parameters]
                 for name, predictor in predictors.items() if name != "locf"}
    saved_cases = {(r["mode"] if r["mode"] == "locf" else f"{r['mode']}_seed{r['seed']}", r["segment"], r["regime"]): r["metrics"]
                   for r in validation["cases"]}
    classical_cases = {(r["segment_index"], r["regime"]): r for r in classical["per_segment"]}
    policy = {
        "kind": "post_hoc_error_stratification", "subject": 107,
        "thresholds": thresholds, "threshold_source": "frozen training q99 per-channel sample changes",
        "flag": "abs(x[t,c] - x[t-1,c]) > threshold[c]; equality unflagged; first segment row False",
        "groups": {"large_change": "hidden AND flag", "other": "hidden AND NOT flag"},
        "causality": "Labels use truth only for offline scoring; never supplied to predictors",
        "scope": "All 102 validation segments and four frozen regimes; six unchanged MLPs and LOCF",
        "interpretation": "Other is the complementary group, not necessarily quiet motion; labels do not diagnose motion or faults",
        "replay_checks": "Single full replay per case, match all saved overall sums/counts; previous full causal audits remain applicable because sources/checkpoints are unchanged",
        "additional_checks": "All flags checked by scalar differences; hidden-label mutation isolation on segment82/random for every predictor",
        "no_fitting": True, "validation_sha256": digest(VALIDATION),
        "training_sha256": digest(TRAIN / "training.json"), "classical_sha256": digest(CLASSICAL),
        "script_sha256": digest(__file__), "frozen_inputs": frozen,
    }
    OUT.mkdir()
    write_json(OUT / "policy.json", policy)  # Freeze policy BEFORE opening any recording.
    cases, isolation = [], []
    samples = 0
    start = time.monotonic()
    with recording_access_guard(DATASET / "Protocol", [107]) as access:
        recording_before = digest(DATASET / "Protocol/subject107.dat")
        assert recording_before == validation["recording_hash"]
        for subject, index, metadata, truth, masks in iter_segments(ROOT, DATASET, "validation"):
            assert subject == 107
            samples += len(truth)
            original_truth = truth.copy()
            flags = large_change_flags(truth, thresholds)
            # Independent scalar label definition, including segment-boundary reset.
            assert not flags[0].any()
            for c in range(6):
                scalar = [False] + [abs(float(truth[t, c]) - float(truth[t-1, c])) > thresholds[c]
                                    for t in range(1, len(truth))]
                np.testing.assert_array_equal(flags[:, c], scalar)
            for regime in REGIMES:
                missing = masks[regime]
                original_mask = missing.copy()
                hidden = np.broadcast_to(missing[:, None], truth.shape)
                selections = {"all": hidden, "large_change": hidden & flags, "other": hidden & ~flags}
                assert not (selections["large_change"] & selections["other"]).any()
                np.testing.assert_array_equal(selections["large_change"] | selections["other"], hidden)
                for name, predictor in predictors.items():
                    result = ReplayResult(*streaming_replay(predictor, truth, missing))
                    np.testing.assert_array_equal(result.values[~missing], truth[~missing])
                    groups = {group: stratified_sums(truth, missing, result, selection)
                              for group, selection in selections.items()}
                    reconcile_partition([groups["large_change"], groups["other"]], groups["all"])
                    equal_previous(groups["all"], saved_cases[name, index, regime])
                    if name == "locf":
                        reference = classical_cases[index, regime]["algorithms"]["locf"]["strata"]
                        for group in GROUPS:
                            equal_strata(groups[group], reference[group])
                    if index == 82 and regime == "random_5pct":
                        changed = truth.copy()
                        changed[missing] += 1000
                        new_flags = large_change_flags(changed, thresholds)
                        assert (new_flags[missing] != flags[missing]).any()
                        repeated = streaming_replay(predictor, changed, missing)
                        for a, b in zip((result.values, result.status, result.missing_run), repeated):
                            np.testing.assert_array_equal(a, b)
                        isolation.append({"algorithm": name, "segment": index, "regime": regime,
                                          "labels_changed": True, "predictions_status_ages_unchanged": True})
                    np.testing.assert_array_equal(truth, original_truth)
                    np.testing.assert_array_equal(missing, original_mask)
                    cases.append({"algorithm": name, "subject": subject, "segment": index,
                                  "regime": regime, "strata": groups})
                print(f"segment={index:03d} regime={regime} complete; cases={len(cases)} elapsed={time.monotonic()-start:.1f}s", flush=True)
        assert digest(DATASET / "Protocol/subject107.dat") == recording_before
    assert samples == 285488 and len(cases) == 2856 and len(isolation) == 7
    pooled = {name: {regime: {group: pool_strata([r["strata"][group] for r in cases
                                                if r["algorithm"] == name and r["regime"] == regime])
                             for group in GROUPS} for regime in REGIMES} for name in predictors}
    for name in predictors:
        for regime in REGIMES:
            groups = pooled[name][regime]
            reconcile_partition([groups["large_change"], groups["other"]], groups["all"])
            equal_previous(groups["all"], validation["pooled"][name][regime])
            for group in GROUPS:
                np.testing.assert_array_equal(groups[group]["hidden_count"], pooled["locf"][regime][group]["hidden_count"])
                assert not any(groups[group]["unavailable_count"])
                if name == "locf":
                    equal_strata(groups[group], classical["pooled_by_regime"][regime]["algorithms"]["locf"]["strata"][group])
    verify_frozen(ROOT)
    validate_training_record(ROOT, TRAIN)
    assert digest(VALIDATION) == policy["validation_sha256"]
    for name, parameters in originals.items():
        for a, b in zip(parameters, predictors[name].model.parameters):
            np.testing.assert_array_equal(a, b)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "policy": policy,
              "segments": 102, "samples": samples, "cases": cases, "pooled": pooled,
              "recording_sha256": recording_before, "recording_access": access,
              "label_isolation": isolation, "source_hashes": training["source_hashes"],
              "checkpoint_hashes": {m["checkpoint"]: m["sha256"] for m, _ in models},
              "checks_passed": True,
              "limits": ["One validation subject with synthetic losses", "No tuning, uncertainty calibration, or INT8 deployment",
                         "Post-hoc truth labels cannot serve as online abstention inputs", "Seed spread is not cross-subject uncertainty"]}
    write_json(OUT / "report.json", report)
    fields = ["algorithm", "regime", "channel", "unit", "group", *COUNTS, *SUMS,
              "mae", "rmse", "coverage", "hidden_fraction_of_all", "squared_error_fraction_of_all",
              "mae_change_percent_vs_locf", "rmse_change_percent_vs_locf"]
    with (OUT / "strata.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in predictors:
            for regime in REGIMES:
                for group in GROUPS:
                    metrics, base = pooled[name][regime][group], pooled["locf"][regime][group]
                    whole = pooled[name][regime]["all"]
                    for c, channel in enumerate(CHANNELS):
                        row = {"algorithm": name, "regime": regime, "channel": channel,
                               "unit": "m/s^2" if c < 3 else "rad/s", "group": group,
                               **{key: metrics[key][c] for key in (*COUNTS, *SUMS, "mae", "rmse", "coverage")},
                               "hidden_fraction_of_all": metrics["hidden_count"][c] / whole["hidden_count"][c],
                               "squared_error_fraction_of_all": metrics["squared_error_sum"][c] / whole["squared_error_sum"][c]}
                        for metric in ("mae", "rmse"):
                            row[f"{metric}_change_percent_vs_locf"] = 100 * (metrics[metric][c] / base[metric][c] - 1)
                        writer.writerow(row)
    print(f"PASS: {len(cases)} replays, partitions and archived totals reconcile. Output: {OUT}", flush=True)


if __name__ == "__main__":
    main()
