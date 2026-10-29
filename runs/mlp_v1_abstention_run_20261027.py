"""Prototype abstention: full saved-sum trade-off plus bounded raw replay checks."""

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from packetheal.baselines import SampleStatus
from packetheal.mlp import MLPPredictor
from packetheal.mlp_experiment import (
    CHANNELS, REGIMES, audited_replay, digest, error_record, normalization_from_root,
    pool_records, recording_access_guard, streaming_replay, unpack_loaded_segment,
    validate_training_record, verify_frozen, write_json,
)
from packetheal.evaluation import load_subject_masks
from packetheal.splits import load_split_segment
from mlp_v1_abstention_20261027 import AbstainingMLP
from mlp_v1_received_warning_20261027 import ReceivedChangeWarning, warning_replay

PARENT = ROOT / "runs/mlp_v1_received_warning_20261027_retry1"
TRAIN = ROOT / "runs/mlp_v1_train_20261027"
OUT = ROOT / "runs/mlp_v1_abstention_20261027"
DATASET = ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset"
MANIFEST = ROOT / "configs/pamap2_segments_v1.csv"
CONFIG = ROOT / "configs/pamap2_split_v1.json"
SOURCES = ("runs/mlp_v1_abstention_20261027.py", "runs/mlp_v1_abstention_run_20261027.py", "runs/test_mlp_v1_abstention_20261027.py")


def selective_record(all_positions, accepted):
    hidden, scored = all_positions["hidden"], accepted["scored"]
    assert 0 <= scored <= hidden
    return {"hidden": hidden, "scored": scored, "unavailable": hidden-scored,
            "sum_abs": accepted["sum_abs"].copy(), "sum_sq": accepted["sum_sq"].copy(),
            "mae": [v/scored if scored else None for v in accepted["sum_abs"]],
            "rmse": [float(np.sqrt(v/scored)) if scored else None for v in accepted["sum_sq"]]}


def equal_record(a, b):
    for key in ("hidden", "scored", "unavailable"):
        assert a[key] == b[key]
    for key in ("sum_abs", "sum_sq"):
        np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-12)
    for key in ("mae", "rmse"):
        for first, second in zip(a[key], b[key]):
            if first is None or second is None:
                assert first is None and second is None
            else:
                np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    frozen = verify_frozen(ROOT)
    training, loaded = validate_training_record(ROOT, TRAIN)
    parent_manifest = PARENT / "evidence.sha256"
    assert digest(parent_manifest) == "cf757178a3e186ca837603fe34354ca15636fdc3dfa1f3e1b1ca24f3981c4f92"
    parent_hashes = {}
    for line in parent_manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and path.suffix != ".dat" and digest(path) == expected
        parent_hashes[name] = expected
    parent = json.loads((PARENT / "report.json").read_text())
    assert parent["correctness_pass"] and json.loads((PARENT / "independent_checks.json").read_text())["verification_pass"]
    caps = parent["policy"]["thresholds_per_nominal_step"]
    norm = normalization_from_root(ROOT)
    names = [f"residual_seed{s}" for s in range(3)]
    models = {f"residual_seed{meta['seed']}": (meta, model) for meta, model in loaded if meta["mode"] == "residual"}
    snapshots = {name: [p.copy() for p in model.parameters] for name, (_, model) in models.items()}
    parent_cases = {(r["algorithm"], r["segment"], r["regime"]): r for r in parent["cases"]}
    warning_cases = {(r["segment"], r["regime"]): r for r in parent["warning_cases"]}
    policy = {
        "kind": "desktop_abstention_prototype", "authorization": "Learner chose prototype only: test rejection and report trade-off",
        "subject": 107, "missing_rule": "UNAVAILABLE if warning on/unknown, original predictor unavailable, or loss age > existing model horizon 10; otherwise original output",
        "real_rule": "Always original MEASURED vector; warning never hides a received measurement",
        "rejected_values": "Six float64 NaNs; original missing_run retained",
        "state": "Advance original MLP and warning on every successful step using only received vector or None; rejected outputs never fed back",
        "prototype_limit": "Original inference still computed before rejection; not an inference-time optimization",
        "startup": "Original LOCF warm-up fallback remains; unknown warning rejects missing positions until two real observations",
        "full_validation": "Saved sums on all 102 segments/four frozen masks, 3 residual seeds; gaps <=10 so same accepted set as saved unwarned group",
        "fresh_runtime_segments": [0, 92], "fresh_runtime_regimes": list(REGIMES), "fresh_runtime_seeds": [0, 1, 2],
        "selection_limit": "Segment0 ordinary correctness check; segment92 previously inspected failure; bounded correctness checks, not representative sampling",
        "correctness_acceptance": "10 synthetic tests; accepted values unchanged; rejected values NaN+UNAVAILABLE; all saved counts/sums reconcile; full replay audits on 24 fresh cases",
        "application_acceptance": "NOT_ASSESSED", "minimum_application_coverage": None, "application_error_limits": None,
        "acceptance_limit": "No application-specific error/coverage requirements agreed; no operational enablement or calibrated confidence claim",
        "no_tuning": "Frozen warning thresholds, checkpoints, masks and normalization unchanged",
        "parent_manifest_sha256": digest(parent_manifest), "parent_hashes": parent_hashes, "frozen_inputs": frozen,
        "sources": {name: digest(ROOT / name) for name in SOURCES},
    }
    OUT.mkdir(); write_json(OUT / "policy.json", policy)
    tests = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(ROOT / "runs"),
                            "-p", "test_mlp_v1_abstention_20261027.py", "-v"], cwd=ROOT, text=True, capture_output=True)
    with (OUT / "software_tests.txt").open("x") as stream: stream.write(tests.stdout+tests.stderr)
    assert tests.returncode == 0
    cases, pooled, by_age = [], {}, {}
    for name in names:
        pooled[name], by_age[name] = {}, {}
        for regime in REGIMES:
            for segment in range(102):
                original = parent_cases[name, segment, regime]
                metrics = selective_record(original["groups"]["all"], original["groups"]["unwarned"])
                ages = {a: selective_record(groups["all"], groups["unwarned"]) for a, groups in original["by_age"].items()}
                equal_record(pool_records(list(ages.values())), metrics)
                cases.append({"algorithm": name, "subject": 107, "segment": segment, "regime": regime, "metrics": metrics, "by_age": ages})
            selected = [r for r in cases if r["algorithm"] == name and r["regime"] == regime]
            pooled[name][regime] = pool_records([r["metrics"] for r in selected])
            equal_record(pooled[name][regime], selective_record(parent["pooled"][name][regime]["all"], parent["pooled"][name][regime]["unwarned"]))
            by_age[name][regime] = {a: selective_record(groups["all"], groups["unwarned"]) for a, groups in parent["by_age"][name][regime].items()}
    assert len(cases) == 1224
    indexed = {(r["algorithm"], r["segment"], r["regime"]): r for r in cases}
    segments, masks = load_subject_masks(ROOT / "data/processed/pamap2_v1/loss_masks.csv", MANIFEST, CONFIG, "validation", 107)
    runtime = []
    with recording_access_guard(DATASET / "Protocol", [107]) as access:
        raw_hash = digest(DATASET / "Protocol/subject107.dat")
        assert raw_hash == parent["recording_hash"]
        for segment in policy["fresh_runtime_segments"]:
            truth = unpack_loaded_segment(load_split_segment(DATASET, MANIFEST, CONFIG, "validation", 107, segment), int(segments[segment]["samples"]))
            for regime in REGIMES:
                missing = masks[segment, regime]
                state = warning_replay(ReceivedChangeWarning(caps), truth, missing)
                fingerprint = hashlib.sha256()
                for array in state: fingerprint.update(array.tobytes())
                assert fingerprint.hexdigest() == warning_cases[segment, regime]["warning_state_sha256"]
                warned, known, _, ages, _, _ = state
                reject = missing & (warned | ~known | (ages > 10))
                keep = missing & ~reject
                for name in names:
                    _, model = models[name]
                    original = streaming_replay(MLPPredictor(model, norm), truth, missing)
                    equal_record(error_record(truth, original[0], missing), pool_records([parent_cases[name, segment, regime]["groups"]["all"]]))
                    values, statuses, observed_ages = audited_replay(AbstainingMLP(model, norm, caps), truth, missing)
                    np.testing.assert_array_equal(values[~missing], truth[~missing])
                    np.testing.assert_array_equal(values[keep], original[0][keep])
                    assert np.isnan(values[reject]).all()
                    assert (statuses[reject] == SampleStatus.UNAVAILABLE.value).all()
                    assert (statuses[keep] == SampleStatus.RECONSTRUCTED.value).all()
                    np.testing.assert_array_equal(observed_ages, ages)
                    actual = error_record(truth, values, missing)
                    equal_record(actual, indexed[name, segment, regime]["metrics"])
                    for age, expected in indexed[name, segment, regime]["by_age"].items():
                        equal_record(error_record(truth, values, missing & (ages == int(age))), expected)
                    runtime.append({"algorithm": name, "segment": segment, "regime": regime, "metrics": actual,
                                    "accepted_values_exact": True, "rejected_values_all_nan": True, "replay_audits_pass": True})
                    print(f"runtime {name} segment={segment} {regime}: accepted={actual['scored']} rejected={actual['unavailable']} PASS", flush=True)
        assert digest(DATASET / "Protocol/subject107.dat") == raw_hash
    assert len(runtime) == 24
    verify_frozen(ROOT); validate_training_record(ROOT, TRAIN)
    for source, expected in policy["sources"].items(): assert digest(ROOT / source) == expected
    for path, expected in parent_hashes.items(): assert digest(ROOT / path) == expected
    for name, before in snapshots.items():
        for a, b in zip(before, models[name][1].parameters): np.testing.assert_array_equal(a, b)
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "policy": policy, "cases": cases,
              "pooled": pooled, "by_age": by_age, "runtime_checks": runtime, "correctness_pass": True,
              "application_acceptance": "NOT_ASSESSED", "recording_access": access, "recording_hash": raw_hash,
              "checkpoint_hashes": {m["checkpoint"]: m["sha256"] for m, _ in models.values()},
              "original_source_hashes": training["source_hashes"], "python": sys.version, "numpy": np.__version__}
    write_json(OUT / "report.json", report)
    fields = ["algorithm", "regime", "age", "channel", "unit", "hidden", "accepted", "unavailable", "coverage",
              "accepted_mae", "accepted_rmse", "original_all_mae", "original_all_rmse", "matched_locf_mae", "matched_locf_rmse"]
    with (OUT / "metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for name in names:
            for regime in REGIMES:
                for age, metrics in {"all": pooled[name][regime], **by_age[name][regime]}.items():
                    groups = parent["pooled"][name][regime] if age == "all" else parent["by_age"][name][regime][age]
                    locf = parent["pooled"]["locf"][regime]["unwarned"] if age == "all" else parent["by_age"]["locf"][regime][age]["unwarned"]
                    for c, channel in enumerate(CHANNELS):
                        writer.writerow({"algorithm": name, "regime": regime, "age": age, "channel": channel,
                                         "unit": "m/s^2" if c < 3 else "rad/s", "hidden": metrics["hidden"], "accepted": metrics["scored"],
                                         "unavailable": metrics["unavailable"], "coverage": metrics["scored"]/metrics["hidden"],
                                         "accepted_mae": metrics["mae"][c], "accepted_rmse": metrics["rmse"][c],
                                         "original_all_mae": groups["all"]["mae"][c], "original_all_rmse": groups["all"]["rmse"][c],
                                         "matched_locf_mae": locf["mae"][c], "matched_locf_rmse": locf["rmse"][c]})
    print("PASS: desktop behavior and saved full-validation trade-off reconcile; application acceptance NOT ASSESSED", flush=True)


if __name__ == "__main__": main()
