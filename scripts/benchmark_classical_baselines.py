#!/usr/bin/env python3
"""Phase-2 validation closure: unchanged predictors, new post-hoc diagnostics.

    python3 -B scripts/benchmark_classical_baselines.py

Only subject107 raw data may be opened. No fitting, new masks, or NN training.
Existing outputs are refused. Original benchmark/source identities are checked.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

from alpha_beta_common import ROOT, load_context, verify_unchanged
from evaluate_alpha_beta import load_selected_gains
from evaluate_locf import git_output, sha256
from evaluate_locf_batch import replay_digest
from packetheal.alpha_beta import AlphaBetaPredictor
from packetheal.baselines import LOCF
from packetheal.diagnostics import large_change_flags, loss_run_metadata, pool_strata, reconcile_partition, stratified_sums
from packetheal.evaluation import REGIMES, load_subject_masks, pool_metrics, replay_segment, score_hidden
from packetheal.linear_extrapolation import LinearExtrapolator
from packetheal.splits import load_split_segment

ALGORITHMS = ("locf", "linear", "alpha_beta")
REFERENCE_SHA256 = "074ef6634fdc2e372d851a534297d285feaafe3c89224a55ba2e48127048643b"
GAINS_SHA256 = "9eb9a8c44d7870d7d40d31dd6fe0e0a7cac876c38b2f4515fe81b99e690d4395"
CAP_SHA256 = "316cc1f1d0e95446329f1f689512703a146f6d07ad00bf463649ad705e13f502"
METRIC_KEYS = {"locf": "locf_metrics", "linear": "linear_metrics", "alpha_beta": "metrics"}
DIGEST_KEYS = {"locf": "locf_replay_sha256", "linear": "linear_replay_sha256", "alpha_beta": "replay_sha256"}


@contextmanager
def validation_access_guard(raw_root, recording):
    """Audit path-based Python opens; allow only the canonical subject107 file.

    The callback is disabled on exit (Python audit hooks cannot be removed).
    This is an additional executable access check, not an OS security sandbox.
    """
    raw = os.path.abspath(raw_root) + os.sep
    allowed = os.path.abspath(recording)
    if not allowed.startswith(raw) or Path(allowed).name != "subject107.dat":
        raise ValueError("Only a subject107 recording under raw_root can be allowed")
    active = False
    access = {"opened_recordings": [], "denied_open_attempts": 0}

    def audit(event, args):
        if active and event == "open" and isinstance(args[0], (str, bytes)):
            lexical = os.path.abspath(os.fsdecode(args[0]))
            real = os.path.realpath(lexical)
            if lexical.startswith(raw) or real.startswith(raw):
                if lexical != allowed or real != allowed:
                    access["denied_open_attempts"] += 1
                    raise PermissionError("Only canonical subject107 raw recording access is permitted")
                if allowed not in access["opened_recordings"]:
                    access["opened_recordings"].append(allowed)

    sys.addaudithook(audit)
    active = True
    try:
        yield access
    finally:
        active = False


def validate_policy(policy):
    expected = {"schema_version": "1", "split": "validation", "subjects": [107],
        "algorithms": list(ALGORITHMS), "regimes": list(REGIMES),
        "gains_sha256": GAINS_SHA256, "slope_caps_sha256": CAP_SHA256,
        "reference_sha256": REFERENCE_SHA256,
        "transient_rule": "per-channel abs(x[t]-x[t-1]) strictly greater than frozen training q99 slope cap",
        "first_row_transient": False, "cross_segment_differences": False,
        "diagnostic_labels_are_predictor_inputs": False, "new_fitting_or_masks": False,
        "partition_sum_rtol": 1e-12, "partition_sum_atol": 1e-12,
        "loss_ages": {r: list(range(1, n + 1)) for r, n in zip(REGIMES, (4, 2, 5, 10))}}
    for key, value in expected.items():
        if policy.get(key) != value:
            raise ValueError(f"Analysis policy differs from the approved definition: {key}")


def _same(first, second, stop=None):
    for field in ("values", "status", "missing_run"):
        np.testing.assert_array_equal(getattr(first, field)[:stop], getattr(second, field)[:stop])


def checked_replay(predictor, truth, missing):
    """Apply the same reset/causality/provenance checks to EVERY predictor."""
    original, original_mask = truth.copy(), missing.copy()
    settings = {key: getattr(predictor, key) for key in ("alpha", "beta", "max_delta_per_step")
                if hasattr(predictor, key)}
    first = replay_segment(predictor, truth, missing)
    repeated = replay_segment(predictor, truth, missing)
    _same(first, repeated)
    metrics = score_hidden(truth, missing, first)
    if score_hidden(truth, missing, repeated) != metrics:
        raise ValueError("Repeated metrics differ")
    poisoned = truth.copy()
    poisoned[missing] = np.nan
    _same(first, replay_segment(predictor, poisoned, missing))
    future, future_mask = truth.copy(), missing.copy()
    cutoff = len(truth) // 2
    future[cutoff:] += 10
    future_mask[cutoff:] = ~future_mask[cutoff:]
    _same(first, replay_segment(predictor, future, future_mask), cutoff)
    # Different intervening segment, then A again using the SAME object.
    replay_segment(predictor, truth[:32] + 1, missing[:32])
    _same(first, replay_segment(predictor, truth, missing))
    np.testing.assert_array_equal(first.values[~missing], truth[~missing])
    np.testing.assert_array_equal(first.missing_run, loss_run_metadata(missing)[0])
    np.testing.assert_array_equal(truth, original)
    np.testing.assert_array_equal(missing, original_mask)
    for key, value in settings.items():
        np.testing.assert_array_equal(getattr(predictor, key), value)
    return first, metrics


def case_diagnostics(truth, missing, result, thresholds, ages_to_report):
    """Construct disjoint partitions after prediction; reconcile with full errors."""
    ages, lengths = loss_run_metadata(missing)
    if set(ages[missing]) - set(ages_to_report):
        raise ValueError("Unexpected loss age outside the frozen mask definition")
    hidden = np.broadcast_to(missing[:, None], truth.shape)
    flags = large_change_flags(truth, thresholds)
    masks = {"all": hidden, "large_change": hidden & flags, "other": hidden & ~flags,
             **{f"age_{age}": np.broadcast_to((ages == age)[:, None], truth.shape)
                for age in ages_to_report}}
    groups = {key: stratified_sums(truth, missing, result, selection) for key, selection in masks.items()}
    reconcile_partition([groups["large_change"], groups["other"]], groups["all"])
    reconcile_partition([groups[f"age_{age}"] for age in ages_to_report], groups["all"])
    run_starts = ages == 1
    return groups, Counter(map(int, lengths[run_starts]))


def _csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_tables(output, report):
    overall, transients, loss_age, runs = [], [], [], []
    for regime, entry in report["pooled_by_regime"].items():
        for length, count in entry["run_length_histogram"].items():
            runs.append({"regime": regime, "run_length": int(length), "runs": count,
                         "hidden_vectors": int(length) * count})
        for algorithm in ALGORITHMS:
            full = entry["algorithms"][algorithm]["metrics"]
            strata = entry["algorithms"][algorithm]["strata"]
            for c, channel in enumerate(report["channel_order"]):
                base = {"regime": regime, "algorithm": algorithm, "channel": channel,
                        "unit": report["channel_units"][c]}
                overall.append({**base, "hidden_count": full["hidden_samples"],
                    "scored_count": full["scored_samples"], "unavailable_count": full["unavailable_samples"],
                    "mae": full["mae"][c], "rmse": full["rmse"][c]})
                for group, values in strata.items():
                    if group == "all":
                        continue
                    sse = full["squared_error_sum"][c]
                    row = {**base, "group": group,
                           **{key: values[key][c] for key in ("hidden_count", "scored_count", "unavailable_count",
                               "absolute_error_sum", "squared_error_sum", "mae", "rmse", "coverage")},
                           "squared_error_fraction_of_all": values["squared_error_sum"][c] / sse if sse else None}
                    (loss_age if group.startswith("age_") else transients).append(row)
    for name, rows in (("overall.csv", overall), ("transients.csv", transients),
                       ("loss_age.csv", loss_age), ("loss_runs.csv", runs)):
        _csv(output / name, rows)


def run_benchmark(output, policy_path, access):
    policy = json.loads(policy_path.read_text())
    validate_policy(policy)
    context = load_context({"scripts/benchmark_classical_baselines.py", "scripts/plot_phase2_diagnostics.py",
        "src/packetheal/diagnostics.py", "tests/test_diagnostics.py", "tests/test_phase2_benchmark.py",
        "scripts/evaluate_alpha_beta.py", "scripts/select_alpha_beta_gains.py", "src/packetheal/gain_selection.py"})
    reference_path = ROOT / "benchmarks/alpha_beta_validation_subject107_v1/benchmark.json"
    if sha256(reference_path) != REFERENCE_SHA256:
        raise ValueError("Frozen combined reference checksum mismatch")
    reference = json.loads(reference_path.read_text())
    paths, hashes = context["paths"], context["hashes"]
    paths.update({"analysis_policy": policy_path, "reference_benchmark": reference_path,
        "gain_search": ROOT / "configs/alpha_beta_gain_search_v1.json",
        "gains": ROOT / "data/processed/alpha_beta_v1/gains.json",
        "gains_sidecar": ROOT / "data/processed/alpha_beta_v1/gains.json.sha256",
        "locf_benchmark": ROOT / "benchmarks/locf_validation_subject107_v1/benchmark.json",
        "linear_benchmark": ROOT / "benchmarks/linear_validation_subject107_v1/benchmark.json",
        "validation_recording": context["dataset"] / "Protocol/subject107.dat"})
    hashes.update({key: sha256(path) for key, path in paths.items()})
    for key, expected in reference["provenance"]["artifact_sha256"].items():
        if hashes.get(key) != expected:
            raise ValueError(f"Frozen comparison input changed: {key}")
    for source, expected in reference["provenance"]["source_sha256"].items():
        if sha256(ROOT / source) != expected:
            raise ValueError(f"Frozen comparison source changed: {source}")
        context["sources"][source] = expected
    gains = load_selected_gains(paths["gains"], GAINS_SHA256, context, json.loads(paths["gain_search"].read_text()))
    if context["config"]["validation_subjects"] != [107]:
        raise ValueError("Only subject107 validation is allowed")
    expected = context["preparation"]["splits"]["validation"]
    segments, masks = load_subject_masks(paths["loss_masks"], paths["segment_manifest"], paths["split_config"],
        "validation", 107, history_preserved_samples=context["preparation"]["history_preserved_samples"])
    lookup = {(c["segment_index"], c["regime"]): c for c in reference["per_segment"]}
    case_keys = {(i, r) for i in range(expected["segments"]) for r in REGIMES}
    if len(segments) != expected["segments"] or len(lookup) != len(reference["per_segment"]) or set(lookup) != case_keys:
        raise ValueError("Incomplete/duplicate reference cases or validation segments")
    limits = context["caps"]["max_delta_per_step"]
    predictors = {"locf": LOCF(), "linear": LinearExtrapolator(limits),
                  "alpha_beta": AlphaBetaPredictor(gains["alpha"], gains["beta"], limits)}
    cases, sample_count = [], 0
    for index, segment in enumerate(segments):
        if segment["source_file"] != "subject107.dat":
            raise ValueError("Validation manifest names another raw recording")
        timestamps, truth = load_split_segment(context["dataset"], paths["segment_manifest"],
            paths["split_config"], "validation", 107, index)
        sample_count += len(truth)
        for regime in REGIMES:
            missing, saved = masks[index, regime], lookup[index, regime]
            if (saved["segment"] != segment or saved["samples"] != len(truth)
                    or saved["timestamp_start_s"] != float(timestamps[0])
                    or saved["timestamp_end_s"] != float(timestamps[-1])):
                raise ValueError("Case identity/timestamps changed")
            algorithms, histogram = {}, None
            for name, predictor in predictors.items():
                result, metrics = checked_replay(predictor, truth, missing)
                digest = replay_digest(result)
                if metrics != saved[METRIC_KEYS[name]] or digest != saved[DIGEST_KEYS[name]]:
                    raise ValueError(f"Full replay differs from frozen reference: {index}/{regime}/{name}")
                if metrics["unavailable_samples"] or metrics["scored_samples"] != int(missing.sum()):
                    raise ValueError("Incomplete hidden-position coverage")
                groups, histogram = case_diagnostics(truth, missing, result, limits, policy["loss_ages"][regime])
                all_groups = pool_strata([groups["all"]])
                for key in ("mae", "rmse", "absolute_error_sum", "squared_error_sum"):
                    np.testing.assert_allclose(all_groups[key], metrics[key], rtol=1e-12, atol=1e-12)
                algorithms[name] = {"metrics": metrics, "replay_sha256": digest, "strata": groups}
            if regime != "random_5pct" and set(histogram) != {int(regime.split('_')[1])}:
                raise ValueError("Fixed-burst regime contains a different run length")
            if algorithms["locf"] != algorithms["alpha_beta"]:
                raise ValueError("Selected (1,0) alpha-beta differs from LOCF")
            cases.append({"segment_index": index, "segment": segment, "regime": regime,
                          "samples": len(truth), "run_length_histogram": dict(sorted(histogram.items())),
                          "algorithms": algorithms})
        print(f"Phase-2 audit: segment {index + 1}/{len(segments)}; all three predictors/four masks", flush=True)
    if (sample_count != expected["samples"] or len(cases) != len(case_keys)
            or {(c["segment_index"], c["regime"]) for c in cases} != case_keys):
        raise ValueError("Validation case/sample totals differ")
    pooled = {}
    for regime in REGIMES:
        selected = [c for c in cases if c["regime"] == regime]
        histogram = Counter()
        for case in selected:
            histogram.update(case["run_length_histogram"])
        if sum(k * n for k, n in histogram.items()) != expected["lost_by_regime"][regime]:
            raise ValueError("Run-length histogram does not reconcile with hidden count")
        algorithms = {}
        for name in ALGORITHMS:
            metrics = pool_metrics([c["algorithms"][name]["metrics"] for c in selected])
            if metrics != reference["pooled_by_regime"][regime][METRIC_KEYS[name]]:
                raise ValueError("Pooled metrics differ from frozen reference")
            groups = selected[0]["algorithms"][name]["strata"]
            strata = {key: pool_strata([c["algorithms"][name]["strata"][key] for c in selected]) for key in groups}
            reconcile_partition([strata["large_change"], strata["other"]], strata["all"])
            reconcile_partition([strata[f"age_{age}"] for age in policy["loss_ages"][regime]], strata["all"])
            if regime != "random_5pct":
                for age in policy["loss_ages"][regime]:
                    np.testing.assert_array_equal(strata[f"age_{age}"]["hidden_count"], [sum(histogram.values())] * 6)
            algorithms[name] = {"metrics": metrics, "strata": strata}
        pooled[regime] = {"segments": len(selected), "samples": sample_count,
                         "run_length_histogram": dict(sorted(histogram.items())), "algorithms": algorithms}
    verify_unchanged(context)
    if access["denied_open_attempts"] or len(access["opened_recordings"]) != 1:
        raise ValueError("Unexpected raw recording access")
    report = {"schema_version": "1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "validation", "subjects": [107], "segments": len(segments), "samples": sample_count,
        "cases": len(cases), "algorithm_cases": len(cases) * len(ALGORITHMS),
        "algorithms": list(ALGORITHMS), "channel_order": context["config"]["channels"],
        "channel_units": ["m/s²"] * 3 + ["rad/s"] * 3, "sampling_frequency_hz": 100,
        "mask_seed": context["preparation"]["seed"], "history_preserved_samples": 20,
        "alpha": gains["alpha"], "beta": gains["beta"], "large_change_thresholds": limits,
        "policy": policy, "pooled_by_regime": pooled, "per_segment": cases,
        "verification": {"all_1224_algorithm_cases_match_archived_replays_and_metrics": True,
            "same_object_repeat_and_A_B_A_resets_all_algorithms": True,
            "hidden_truth_poison_and_future_measurement_mask_prefix_checks_all_algorithms": True,
            "received_original_arrays_parameters_and_loss_ages_preserved": True,
            "all_hidden_positions_scored": True, "all_strata_and_run_counts_reconciled": True,
            "all_fixed_burst_age_cohorts_equal_in_size": True,
            "inputs_sources_and_parameters_unchanged": True, "no_fitting_or_masks_generated": True},
        "provenance": {"git_commit": git_output("rev-parse", "HEAD"),
            "git_status_porcelain": git_output("status", "--porcelain"), "python_version": platform.python_version(),
            "numpy_version": np.__version__, "artifact_sha256": hashes, "source_sha256": context["sources"],
            "raw_access": access},
        "limits": ["One validation subject and one frozen seed per regime; no test/generalization/significance claim.",
            "Large-change labels use hidden truth for post-hoc analysis only and may include sensor/data anomalies.",
            "Within-burst ages share gap cohorts; regimes use different hidden positions, not a controlled causal length effect.",
            "Synthetic nominal-step losses only; no transport, long-outage accuracy, confidence, NN or device measurements."]}
    output.mkdir(parents=True, exist_ok=False)
    with (output / "benchmark.json").open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    export_tables(output, report)
    print(json.dumps({"output": str(output.resolve()), "cases": len(cases),
        "benchmark_sha256": sha256(output / "benchmark.json"), "raw_access": access}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/phase2_validation_subject107_v1")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    recording = ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset/Protocol/subject107.dat"
    with validation_access_guard(ROOT / "data/raw", recording) as access:
        run_benchmark(args.output_dir, ROOT / "configs/phase2_analysis_v1.json", access)


if __name__ == "__main__":
    main()
