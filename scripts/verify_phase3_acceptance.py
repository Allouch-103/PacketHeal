#!/usr/bin/env python3
"""Final Phase 3 verification: saved arithmetic, software tests, synthetic integration.

No actual recording may be opened. Mutable instruction/status documents pinned
by older reports are verified against their exact archived historical versions.
All model, evaluated source and dataset bindings still use their original paths.
"""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/phase3_validation_accepted_20261029"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "runs"), str(ROOT / "tests")]
from packetheal.mlp_experiment import REGIMES, digest, normalization_from_root, validate_training_record, verify_frozen, write_json
from check_trained_consumer import exercise_model


def compare_metrics(actual, expected):
    for key in ("hidden", "scored", "unavailable"):
        assert actual[key] == expected[key], (key, actual[key], expected[key])
    for key in ("sum_abs", "sum_sq", "mae", "rmse"):
        for a, b in zip(actual[key], expected[key]):
            if a is None or b is None:
                assert a is b
            else:
                assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (key, a, b)


def independent_pool(records):
    count = sum(r["scored"] for r in records)
    absolute = [math.fsum(r["sum_abs"][c] for r in records) for c in range(6)]
    squared = [math.fsum(r["sum_sq"][c] for r in records) for c in range(6)]
    return {"hidden": sum(r["hidden"] for r in records), "scored": count,
            "unavailable": sum(r["unavailable"] for r in records),
            "sum_abs": absolute, "sum_sq": squared,
            "mae": [a/count if count else None for a in absolute],
            "rmse": [math.sqrt(s/count) if count else None for s in squared]}


def verify_manifests():
    resolution = json.loads((OUT / "historical_document_resolution.json").read_text())
    allowed_documents = {"AGENTS.md", "01_PROJECT_INSTRUCTIONS.md",
                         "05_STATUS_AND_DECISIONS_UPDATED.md", "docs/PHASE3_REVIEW_20261029.md"}
    assert set(resolution) == allowed_documents
    manifests = [ROOT / "benchmarks/phase3_checkpoint_20261029/evidence.sha256",
                 ROOT / "runs/mlp_v1_trained_consumer_20261029/evidence.sha256",
                 ROOT / "runs/mlp_v1_phase3_review_20261029/evidence.sha256"]
    names = ["integration_20261026", "train_20261027", "validation_20261027", "transients_20261027",
             "residual_failure_20261027", "gyroy_burst5_failure_20261027", "accy_sensitivity_20261027",
             "received_warning_20261027_retry1", "warning_fallback_20261027", "abstention_20261027", "consumer_20261027"]
    for name in names:
        manifests.extend(sorted((ROOT / "runs" / ("mlp_v1_" + name)).glob("*.sha256")))
    results = []
    for manifest in manifests:
        historical, count = [], 0
        for line in manifest.read_text().splitlines():
            if not line.strip(): continue
            expected, name = line.split(maxsplit=1)
            path = (ROOT / name).resolve()
            assert path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / "data/raw")
            if digest(path) != expected:
                assert name in allowed_documents and resolution[name]["sha256"] == expected, name
                path = (ROOT / resolution[name]["historical_snapshot"]).resolve()
                assert path.is_relative_to(OUT / "historical_documents")
                assert digest(path) == expected, name
                historical.append(name)
            count += 1
        results.append({"manifest": str(manifest.relative_to(ROOT)), "entries_verified": count,
                        "historical_document_bindings": historical})
    return results


def main():
    if (OUT / "acceptance.json").exists() or (OUT / "software_tests.txt").exists():
        raise FileExistsError("Final acceptance outputs already exist")
    attempts = []
    def deny_actual_raw(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(arguments[0])).absolute()
            if path.is_relative_to(ROOT / "data/raw") or path.resolve().is_relative_to(ROOT / "data/raw"):
                attempts.append(str(path))
                raise PermissionError("Phase 3 closure must not open actual recordings")
    sys.addaudithook(deny_actual_raw)
    manifests = verify_manifests()
    frozen = verify_frozen(ROOT)
    training, models = validate_training_record(ROOT, ROOT / "runs/mlp_v1_train_20261027")
    validation = json.loads((ROOT / "runs/mlp_v1_validation_20261027/validation.json").read_text())
    cases = validation["cases"]
    identities = {(r["mode"], r["seed"], r["subject"], r["segment"], r["regime"]) for r in cases}
    expected = {(mode, seed, 107, segment, regime)
                for mode, seed in [("locf", None)] + [(m, s) for m in ("direct", "residual") for s in range(3)]
                for segment in range(102) for regime in REGIMES}
    assert len(cases) == 2856 and identities == expected
    for case in cases:
        compare_metrics(independent_pool(list(case["by_age"].values())), case["metrics"])
    for key, regimes in validation["pooled"].items():
        mode, seed = ("locf", None) if key == "locf" else (key.split("_seed")[0], int(key.split("_seed")[1]))
        for regime, metrics in regimes.items():
            selected = [r["metrics"] for r in cases if (r["mode"], r["seed"], r["regime"]) == (mode, seed, regime)]
            assert len(selected) == 102
            compare_metrics(independent_pool(selected), metrics)

    abstention = json.loads((ROOT / "runs/mlp_v1_abstention_20261027/report.json").read_text())
    assert len(abstention["cases"]) == 1224 and len(abstention["runtime_checks"]) == 24
    for seed in range(3):
        for regime in REGIMES:
            selected = [r["metrics"] for r in abstention["cases"] if r["algorithm"] == f"residual_seed{seed}" and r["regime"] == regime]
            assert len(selected) == 102
            compare_metrics(independent_pool(selected), abstention["pooled"][f"residual_seed{seed}"][regime])
    with (ROOT / "runs/mlp_v1_abstention_20261027/metrics.csv").open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["age"] == "all"]
    assert len(rows) == 72
    assert all(float(r["accepted_mae"]) < float(r["matched_locf_mae"]) and
               float(r["accepted_rmse"]) < float(r["matched_locf_rmse"]) for r in rows)

    # Exactly the validated Phase 3 suites. Do not run unfinished Phase 4 tests.
    loader = unittest.TestLoader()
    suites = [loader.loadTestsFromName(p.stem) for p in sorted((ROOT / "tests").glob("test_*.py"))
              if p.name != "test_quantization.py"]
    suites.extend(loader.loadTestsFromName(p.stem) for p in sorted((ROOT / "runs").glob("test_mlp_v1*.py")))
    with (OUT / "software_tests.txt").open("x") as stream:
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.TestSuite(suites))
    assert tests.wasSuccessful() and tests.testsRun == 197
    norm = normalization_from_root(ROOT)
    caps = json.loads((ROOT / "runs/mlp_v1_received_warning_20261027_retry1/policy.json").read_text())["thresholds_per_nominal_step"]
    integration = []
    for metadata, model in models:
        if metadata["mode"] != "residual": continue
        parameters = [p.copy() for p in model.parameters]
        integration.append({"seed": metadata["seed"], **exercise_model(model, norm, np.asarray(caps))})
        for before, after in zip(parameters, model.parameters): np.testing.assert_array_equal(before, after)
    assert len(integration) == 3
    assert verify_frozen(ROOT) == frozen
    validate_training_record(ROOT, ROOT / "runs/mlp_v1_train_20261027")
    assert verify_manifests() == manifests and not attempts
    summary = {
        "date": "2026-10-29", "phase": 3, "status": "VALIDATED_AND_ACCEPTED",
        "authorization": "Learner explicitly requested final Phase 3 validation, updated explanation instructions, commit/push, and marking Phase 3 validated",
        "scope": "Desktop causal MLP benchmark, heuristic abstention and status-aware consumer; no application or hardware acceptance",
        "tests": {"run": tests.testsRun, "failures": len(tests.failures), "errors": len(tests.errors), "pass": tests.wasSuccessful(), "excluded": "Unfinished Phase 4 test_quantization.py"},
        "manifests_verified": manifests, "original_validation_cases_reconciled": len(cases),
        "abstention_saved_cases_reconciled": len(abstention["cases"]), "prior_fresh_abstention_runtime_cases": 24,
        "fresh_trained_consumer_cases": 3, "fresh_artificial_slots": 120,
        "fresh_checkpoint_verification_count": len(models),
        "matched_accepted_locf_mae_rmse_wins": {"cells": 72, "mae": 72, "rmse": 72},
        "original_model_screen": validation["engineering_screen"],
        "frozen_inputs": frozen, "source_hashes": training["source_hashes"],
        "actual_raw_open_attempts": attempts, "new_training": False, "new_accuracy_replay": False,
        "quantization_run": False, "firmware_work": False,
        "application_error_limits": None, "application_minimum_coverage": None,
        "limitations": ["Validation subject107 only; subjects108/109 untouched", "Synthetic losses only", "Warning is not calibrated confidence", "Abrupt-change regressions and large individual errors remain", "No INT8 or device evidence"],
        "communication_update": "Short what/why explanations, PacketHeal examples, file roles and test meaning in AGENTS.md and startup instructions",
        "verifier_sha256": digest(__file__), "python": sys.version, "numpy": np.__version__,
    }
    write_json(OUT / "acceptance.json", summary)
    write_json(OUT / "fresh_consumer_integration.json", integration)
    print(json.dumps({"phase3": summary["status"], "software_tests_passed": tests.testsRun,
                      "saved_validation_cases": len(cases), "abstention_cases": len(abstention["cases"]),
                      "fresh_consumer_cases": len(integration), "actual_raw_open_attempts": attempts}))


if __name__ == "__main__": main()
