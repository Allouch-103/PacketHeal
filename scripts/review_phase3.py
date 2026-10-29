#!/usr/bin/env python3
"""Summarize pinned Phase 3 evidence and coverage choices without raw data."""

import csv
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from packetheal.mlp_experiment import CHANNELS, REGIMES, digest, verify_frozen, write_json

OUT = ROOT / "runs/mlp_v1_phase3_review_20261029"
INTEGRATION = ROOT / "runs/mlp_v1_trained_consumer_20261029/integration.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    attempts = []
    def deny_raw(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(arguments[0])).absolute()
            if path.is_relative_to(ROOT / "data/raw") or path.resolve().is_relative_to(ROOT / "data/raw"):
                attempts.append(str(path))
                raise PermissionError("Saved-result review must not open raw recordings")
    sys.addaudithook(deny_raw)
    frozen = verify_frozen(ROOT)
    # Pin saved inputs to the freshly checked preservation checkpoint.
    manifest = ROOT / "benchmarks/phase3_checkpoint_20261029/evidence.sha256"
    evidence = {}
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / "data/raw")
        assert digest(path) == expected, name
        evidence[name] = expected
    integration = json.loads(INTEGRATION.read_text())
    assert integration["script_sha256"] == digest(ROOT / "scripts/check_trained_consumer.py")
    assert integration["software_correctness_pass"] and not integration["raw_recording_open_attempts"]
    assert {r["seed"] for r in integration["model_results"]} == {0, 1, 2}
    original = json.loads((ROOT / "runs/mlp_v1_validation_20261027/independent_checks.json").read_text())
    abstention = json.loads((ROOT / "runs/mlp_v1_abstention_20261027/independent_checks.json").read_text())
    assert original["verification_pass"] and abstention["verification_pass"]
    with (ROOT / "runs/mlp_v1_abstention_20261027/metrics.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["age"] == "all"]
    assert len(rows) == 72
    assert {(r["algorithm"], r["regime"], r["channel"]) for r in rows} == {
        (f"residual_seed{s}", regime, channel) for s in range(3) for regime in REGIMES for channel in CHANNELS}
    coverage = []
    for regime in REGIMES:
        selected = [r for r in rows if r["regime"] == regime]
        counts = {(int(r["hidden"]), int(r["accepted"]), int(r["unavailable"])) for r in selected}
        assert len(counts) == 1
        hidden, accepted, unavailable = counts.pop()
        assert hidden == accepted + unavailable
        fraction = accepted / hidden
        assert all(math.isclose(float(r["coverage"]), fraction, abs_tol=1e-12) for r in selected)
        prior = next(r for r in abstention["coverage"] if r["regime"] == regime)
        assert (hidden, accepted, unavailable) == (prior["hidden"], prior["accepted"], prior["unavailable"])
        coverage.append({"regime": regime, "hidden": hidden, "accepted": accepted,
                         "unavailable": unavailable, "retained_fraction": fraction,
                         "rejected_fraction": unavailable / hidden})

    physical = []
    for channel in CHANNELS:
        selected = [r for r in rows if r["channel"] == channel]
        record = {"channel": channel, "unit": selected[0]["unit"]}
        for metric in ("mae", "rmse"):
            worst = max(selected, key=lambda r: float(r[f"accepted_{metric}"]))
            record[f"largest_pooled_accepted_{metric}"] = float(worst[f"accepted_{metric}"])
            record[f"{metric}_seed"] = worst["algorithm"]
            record[f"{metric}_regime"] = worst["regime"]
        physical.append(record)
    matched = {metric: sum(float(r[f"accepted_{metric}"]) < float(r[f"matched_locf_{metric}"]) for r in rows)
               for metric in ("mae", "rmse")}
    with (ROOT / "runs/mlp_v1_transients_20261027/mean_seed_comparison.csv").open(newline="") as stream:
        weakness = next(r for r in csv.DictReader(stream)
                        if (r["mode"], r["group"], r["regime"], r["channel"]) ==
                        ("residual", "large_change", "burst_5_5pct", "gyro_y_rad_s"))
    alternatives = [{"minimum_retained_fraction": minimum,
                     "maximum_rejected_fraction": 1 - minimum,
                     "passes_observed_regimes": [r["regime"] for r in coverage if r["retained_fraction"] >= minimum],
                     "fails_observed_regimes": [r["regime"] for r in coverage if r["retained_fraction"] < minimum]}
                    for minimum in (.98, .975)]
    result = {
        "date": "2026-10-29", "kind": "phase3_desktop_review", "subject": 107,
        "learner_scope": "Desktop prototype; keep physical error limits unset",
        "phase3_formal_acceptance": "PENDING_LEARNER_REVIEW",
        "original_engineering_screen": original["mean_across_seeds"],
        "locf_normalized_mae": original["locf_normalized_mae"],
        "abstention_per_seed": abstention["summaries"], "coverage": coverage,
        "coverage_alternatives": alternatives,
        "coverage_recommendation": "97.5% is a possible future desktop review target; it was chosen after observing results and is not a predeclared validation success criterion or application guarantee. No new limit has been adopted or enforced.",
        "matched_locf_comparison": {"channel_regime_seed_cells": len(rows), "lower_mae_cells": matched["mae"], "lower_rmse_cells": matched["rmse"], "positions": "Identical accepted positions only; not comparison with full-coverage LOCF errors"},
        "physical_error_summary": physical,
        "physical_summary_limit": "Largest pooled channel/regime/seed aggregate; not maximum individual sample error or a bound. Physical error limits remain null.",
        "application_mae_limits": None, "application_rmse_limits": None,
        "application_pointwise_limits": None, "application_minimum_coverage": None,
        "known_original_transient_weakness": weakness,
        "warning_limit": "A fixed received-change heuristic, not calibrated confidence; current results are one operating point, not a full risk/coverage curve",
        "software_integration_pass": integration["software_correctness_pass"],
        "trained_seeds_tested": [r["seed"] for r in integration["model_results"]],
        "checkpoint_commit_pushed_before_work": "8e1f4b0",
        "recordings_opened_in_this_review": 0, "raw_recording_open_attempts": attempts,
        "frozen_inputs": frozen, "integration_report_sha256": digest(INTEGRATION),
        "saved_evidence_entries_verified": len(evidence), "script_sha256": digest(__file__),
        "no_changes": "No training, model/threshold tuning, raw recording access, new masks, normalization changes, INT8 work, firmware or new accuracy replay",
    }
    assert matched == {"mae": 72, "rmse": 72}
    assert not attempts and verify_frozen(ROOT) == frozen
    for name, expected in evidence.items():
        assert digest(ROOT / name) == expected, name
    OUT.mkdir()
    write_json(OUT / "review.json", result)
    for name, records in (("coverage.csv", coverage), ("physical_error_summary.csv", physical)):
        with (OUT / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    print("Review recorded: original development screen PASS; trained-consumer integration PASS; application limits unset; Phase 3 formal acceptance pending")


if __name__ == "__main__":
    main()
