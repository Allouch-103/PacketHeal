"""Frozen subject107 evaluation of a causal received-change warning."""

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from packetheal.baselines import LOCF
from packetheal.mlp import MLPPredictor
from packetheal.mlp_experiment import (
    CHANNELS, REGIMES, digest, error_record, iter_segments, normalization_from_root,
    pool_records, provenance, recording_access_guard, streaming_replay,
    validate_training_record, verify_frozen, write_json,
)
from mlp_v1_received_warning_20261027 import ReceivedChangeWarning, warning_replay, reference_warning

OUT = ROOT / "runs/mlp_v1_received_warning_20261027"
TRAIN = ROOT / "runs/mlp_v1_train_20261027"
VALIDATION = ROOT / "runs/mlp_v1_validation_20261027/validation.json"
DATASET = ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset"
GROUPS = ("all", "warned", "unwarned", "unknown")
COUNT_FIELDS = ("hidden", "scored", "unavailable")
SUM_FIELDS = ("sum_abs", "sum_sq")
SOURCES = ("runs/mlp_v1_received_warning_20261027.py", "runs/mlp_v1_received_warning_run_20261027.py",
           "runs/test_mlp_v1_received_warning_20261027.py")


def zero_record():
    return {"hidden": 0, "scored": 0, "unavailable": 0, "sum_abs": [0.] * 6, "sum_sq": [0.] * 6}


def compact(truth, values, selected):
    result = error_record(truth, values, selected)
    return {k: result[k] for k in (*COUNT_FIELDS, *SUM_FIELDS)}


def equal_sums(a, b):
    for key in COUNT_FIELDS:
        assert a[key] == b[key], key
    for key in SUM_FIELDS:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-12)


def partition(parts, whole):
    equal_sums(pool_records(parts), whole)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    frozen = verify_frozen(ROOT)
    training, models = validate_training_record(ROOT, TRAIN)
    assert digest(VALIDATION) == "3f1dde9e7e02b3dc4e73093899252de567229f4e4be2c8a83ac84fe066464604"
    reference = json.loads(VALIDATION.read_text())
    assert reference["training_report_sha256"] == digest(TRAIN / "training.json")
    assert json.loads((VALIDATION.parent / "independent_checks.json").read_text())["verification_pass"]
    caps = json.loads((ROOT / "data/processed/linear_extrapolation_v1/slope_limits.json").read_text())["max_delta_per_step"][:3]
    norm = normalization_from_root(ROOT)
    predictors = {"locf": LOCF(6), **{f"{m['mode']}_seed{m['seed']}": MLPPredictor(model, norm) for m, model in models}}
    parameters = {name: [p.copy() for p in predictor.model.parameters] for name, predictor in predictors.items() if name != "locf"}
    saved = {(r["mode"] if r["mode"] == "locf" else f"{r['mode']}_seed{r['seed']}", r["segment"], r["regime"]): r for r in reference["cases"]}
    policy = {
        "kind": "causal_warning_association", "subject": 107, "segments": 102,
        "rule": "score=max over acceleration XYZ of abs(last_received-previous_received)/(elapsed nominal steps*training q99); warn iff score>1",
        "thresholds_per_nominal_step": caps, "strict_equality_unwarned": True,
        "state": "Latest received pair only; hold score through missing steps; refresh at next real sample; reset every segment/regime",
        "startup": "Unknown until two received vectors; unknown separate from unwarned",
        "comparison": "warned/unwarned/unknown hidden positions, common to all predictors; no predictor or measurement changes",
        "age_control": "Report within each regime and loss age as well as overall",
        "hypothetical_retention": "Retain known unwarned hidden positions; coverage=unwarned count/all hidden count; models themselves still reconstruct all positions",
        "primary_association_screen": "For EACH of 3 residual seeds, equal-regime/channel training-std-normalized MAE in warned group exceeds unwarned; all 24 group cells must be nonempty",
        "secondary": "RMSE, per-cell/age exceptions, retained errors versus all-position MLP and matched-position LOCF, coverage; no minimum coverage chosen after results",
        "limits": "Exploratory validation after a selected failure; single subject and frozen synthetic masks; warning is not calibrated uncertainty or proof of faulty sensors",
        "normalization": "Frozen training std used only in dimensionless summary; physical-unit metrics kept per channel",
        "validation_sha256": digest(VALIDATION), "training_sha256": digest(TRAIN / "training.json"),
        "frozen_inputs": frozen, "diagnostic_sources": {name: digest(ROOT / name) for name in SOURCES},
    }
    OUT.mkdir()
    write_json(OUT / "policy.json", policy)  # Fixed before any raw opens or warning outcomes.
    tested = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT / "runs"),
                             "-p", "test_mlp_v1_received_warning_20261027.py", "-v"], cwd=ROOT, text=True, capture_output=True)
    with (OUT / "software_tests.txt").open("x") as stream:
        stream.write(tested.stdout + tested.stderr)
    assert tested.returncode == 0
    cases, warning_cases = [], []
    count_samples = 0
    warning = ReceivedChangeWarning(caps)
    start = time.monotonic()
    warning_fields = ["segment", "regime", "original_row", "loss_age", "known", "warned", "score",
                      "last_received_row", "previous_received_row", "elapsed_steps",
                      "last_acc_x", "last_acc_y", "last_acc_z", "previous_acc_x", "previous_acc_y", "previous_acc_z"]
    with (OUT / "warnings.csv").open("x", newline="") as warning_stream:
        writer = csv.DictWriter(warning_stream, fieldnames=warning_fields); writer.writeheader()
        with recording_access_guard(DATASET / "Protocol", [107]) as access:
            recording_hash = digest(DATASET / "Protocol/subject107.dat")
            assert recording_hash == reference["recording_hash"]
            for subject, index, metadata, truth, masks in iter_segments(ROOT, DATASET, "validation"):
                assert subject == 107
                count_samples += len(truth)
                original_truth = truth.copy()
                for regime in REGIMES:
                    missing = masks[regime]; original_mask = missing.copy()
                    state = warning_replay(warning, truth, missing)
                    for a, b in zip(state, reference_warning(truth, missing, caps)):
                        np.testing.assert_array_equal(a, b)
                    poisoned = truth.copy(); poisoned[missing] = np.nan
                    for a, b in zip(state, warning_replay(warning, poisoned, missing)):
                        np.testing.assert_array_equal(a, b)
                    cutoff = len(truth) // 2
                    future_truth, future_mask = truth.copy(), missing.copy()
                    future_truth[cutoff:] += 1; future_mask[cutoff:] = ~future_mask[cutoff:]
                    for a, b in zip(state, warning_replay(warning, future_truth, future_mask)):
                        np.testing.assert_array_equal(a[:cutoff], b[:cutoff])
                    warned, known, scores, ages, last, previous = state
                    assert known[missing].all() and (ages[missing] <= 10).all()
                    selections = {"all": missing, "warned": missing & known & warned,
                                  "unwarned": missing & known & ~warned, "unknown": missing & ~known}
                    np.testing.assert_array_equal(sum(selections[g].astype(int) for g in GROUPS[1:]), missing.astype(int))
                    fingerprint = hashlib.sha256()
                    for array in state:
                        fingerprint.update(array.tobytes())
                    counts = {g: int(selections[g].sum()) for g in GROUPS}
                    warning_cases.append({"subject": subject, "segment": index, "regime": regime,
                                          "counts": counts, "warning_state_sha256": fingerprint.hexdigest()})
                    original_start = int(metadata["start_row"])
                    for t in np.flatnonzero(missing):
                        assert 0 <= previous[t] < last[t] < t
                        assert not missing[last[t]] and not missing[previous[t]]
                        writer.writerow({"segment": index, "regime": regime, "original_row": original_start + int(t),
                                         "loss_age": int(ages[t]), "known": int(known[t]), "warned": int(warned[t]), "score": float(scores[t]),
                                         "last_received_row": original_start + int(last[t]), "previous_received_row": original_start + int(previous[t]),
                                         "elapsed_steps": int(last[t] - previous[t]),
                                         **{f"last_acc_{axis}": float(truth[last[t], c]) for c, axis in enumerate("xyz")},
                                         **{f"previous_acc_{axis}": float(truth[previous[t], c]) for c, axis in enumerate("xyz")}})
                    for name, predictor in predictors.items():
                        values, statuses, predicted_ages = streaming_replay(predictor, truth, missing)
                        np.testing.assert_array_equal(predicted_ages, ages)
                        np.testing.assert_array_equal(values[~missing], truth[~missing])
                        assert (statuses[missing] == "reconstructed").all()
                        grouped = {g: compact(truth, values, selection) for g, selection in selections.items()}
                        equal_sums(grouped["all"], saved[name, index, regime]["metrics"])
                        partition([grouped[g] for g in GROUPS[1:]], grouped["all"])
                        by_age = {}
                        for age in range(1, int(ages[missing].max()) + 1):
                            if not (missing & (ages == age)).any():
                                continue
                            by_age[str(age)] = {g: compact(truth, values, selection & (ages == age)) for g, selection in selections.items()}
                            partition([by_age[str(age)][g] for g in GROUPS[1:]], by_age[str(age)]["all"])
                            equal_sums(by_age[str(age)]["all"], saved[name, index, regime]["by_age"][str(age)])
                        for g in GROUPS:
                            partition([groups[g] for groups in by_age.values()], grouped[g])
                        cases.append({"algorithm": name, "subject": subject, "segment": index,
                                      "regime": regime, "groups": grouped, "by_age": by_age})
                    np.testing.assert_array_equal(truth, original_truth)
                    np.testing.assert_array_equal(missing, original_mask)
                print(f"segment={index:03d}/101 cases={len(cases)} elapsed={time.monotonic()-start:.1f}s", flush=True)
            assert digest(DATASET / "Protocol/subject107.dat") == recording_hash
    assert count_samples == 285488 and len(cases) == 2856 and len(warning_cases) == 408
    pooled, by_age = {}, {}
    for name in predictors:
        pooled[name], by_age[name] = {}, {}
        for regime in REGIMES:
            selected = [r for r in cases if r["algorithm"] == name and r["regime"] == regime]
            pooled[name][regime] = {g: pool_records([r["groups"][g] for r in selected]) for g in GROUPS}
            equal_sums(pooled[name][regime]["all"], reference["pooled"][name][regime])
            partition([pooled[name][regime][g] for g in GROUPS[1:]], pooled[name][regime]["all"])
            ages = sorted({int(a) for r in selected for a in r["by_age"]})
            by_age[name][regime] = {str(a): {g: pool_records([r["by_age"].get(str(a), {}).get(g, zero_record()) for r in selected]) for g in GROUPS} for a in ages}
            for g in GROUPS:
                partition([groups[g] for groups in by_age[name][regime].values()], pooled[name][regime][g])
    verify_frozen(ROOT); validate_training_record(ROOT, TRAIN)
    assert digest(VALIDATION) == policy["validation_sha256"]
    for source, expected in policy["diagnostic_sources"].items():
        assert digest(ROOT / source) == expected
    for name, before in parameters.items():
        for a, b in zip(before, predictors[name].model.parameters):
            np.testing.assert_array_equal(a, b)
    report = {**provenance(ROOT), "created_at_utc": datetime.now(timezone.utc).isoformat(), "policy": policy,
              "segments": 102, "samples": count_samples, "warning_cases": warning_cases,
              "cases": cases, "pooled": pooled, "by_age": by_age,
              "recording_hash": recording_hash, "recording_access": access,
              "model_hashes": {m["checkpoint"]: m["sha256"] for m, _ in models}, "correctness_pass": True}
    write_json(OUT / "report.json", report)
    fields = ["algorithm", "regime", "age", "group", "channel", "unit", *COUNT_FIELDS, *SUM_FIELDS,
              "mae", "rmse", "group_fraction_of_all_hidden", "hypothetical_unwarned_retention"]
    with (OUT / "metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for name in predictors:
            for regime in REGIMES:
                levels = {"all": pooled[name][regime], **by_age[name][regime]}
                for age, groups in levels.items():
                    total = groups["all"]["hidden"]
                    for g, metrics in groups.items():
                        for c, channel in enumerate(CHANNELS):
                            writer.writerow({"algorithm": name, "regime": regime, "age": age, "group": g,
                                             "channel": channel, "unit": "m/s^2" if c < 3 else "rad/s",
                                             **{k: metrics[k] for k in COUNT_FIELDS},
                                             **{k: metrics[k][c] for k in (*SUM_FIELDS, "mae", "rmse")},
                                             "group_fraction_of_all_hidden": metrics["hidden"] / total if total else None,
                                             "hypothetical_unwarned_retention": groups["unwarned"]["hidden"] / total if total else None})
    print("PASS: original overall/age metrics preserved; warning partitions reconcile.", flush=True)


if __name__ == "__main__":
    main()
