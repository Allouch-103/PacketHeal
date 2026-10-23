#!/usr/bin/env python3
"""Compare frozen training-selected alpha-beta with LOCF/linear on subject 107.

    python3 -B scripts/evaluate_alpha_beta.py --gains-sha256 HASH_FROM_SELECTION

Evaluates all 102 segments/four masks. The independently supplied gain checksum
binds this run to settings frozen before opening validation measurements.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np

from alpha_beta_common import ROOT, load_context, verify_unchanged
from evaluate_linear_extrapolation import LOCF_SHA256, check_locf_reference
from evaluate_locf import git_output, sha256
from evaluate_locf_batch import replay_digest
from packetheal.alpha_beta import AlphaBetaPredictor
from packetheal.baselines import LOCF
from packetheal.evaluation import REGIMES, load_subject_masks, pool_metrics, replay_segment, score_hidden
from packetheal.gain_selection import choose_gain_candidate
from packetheal.linear_extrapolation import LinearExtrapolator
from packetheal.splits import load_split_segment

LINEAR_SHA256 = "04817522032f98a19851f2bec80567501f247ecb99a2e6f13894efaf5c453955"


def load_selected_gains(path, expected_sha256, context, search):
    """Verify the pinned bytes, training contract and minimum-score choice."""
    path = Path(path)
    if (path.with_suffix(path.suffix + ".sha256").read_text().split() != [expected_sha256, path.name]
            or sha256(path) != expected_sha256):
        raise ValueError("Frozen gain checksum mismatch")
    fitted = json.loads(path.read_text())
    expected = context["preparation"]["splits"]["train"]
    if (fitted["schema_version"] != "1" or fitted["purpose"] != "alpha_beta_training_gain_selection"
            or fitted["fit_split"] != "train" or fitted["subjects"] != list(range(101, 107))
            or fitted["subjects"] != expected["subjects"]
            or fitted["segments"] != expected["segments"] or fitted["samples"] != expected["samples"]
            or fitted["cases"] != expected["segments"] * len(REGIMES)
            or fitted["search"] != search or search["fit_split"] != "train"
            or search["subjects"] != expected["subjects"] or search["regimes"] != list(REGIMES)
            or fitted["channel_order"] != context["config"]["channels"]
            or fitted["sampling_frequency_hz"] != 100
            or fitted["normalization_applied_to_predictor"] is not False
            or fitted["slope_caps_sha256"] != context["hashes"]["slope_caps"]
            or fitted["slope_caps"] != context["caps"]["max_delta_per_step"]):
        raise ValueError("Gain artifact violates the frozen training/experiment contract")
    np.testing.assert_array_equal(fitted["objective_training_std_sample"], context["scales"])
    pairs = [(a, b) for a in search["alpha_values"] for b in search["beta_values"]]
    results = fitted["candidate_results"]
    if len(results) != len(pairs) or not pairs:
        raise ValueError("Incomplete gain search")
    sums = np.empty((len(REGIMES), len(pairs), 6))
    counts = np.array([expected["lost_by_regime"][regime] for regime in REGIMES])
    for k, pair in enumerate(pairs):
        candidate = results[k]
        if (candidate["alpha"], candidate["beta"]) != pair:
            raise ValueError("Gain candidates are not in predeclared order")
        for r, regime in enumerate(REGIMES):
            metrics = candidate["by_regime"][regime]
            if metrics["hidden_samples"] != counts[r]:
                raise ValueError("Gain selection hidden count mismatch")
            sums[r, k] = metrics["absolute_error_sum"]
    selected, objectives = choose_gain_candidate(sums, counts, context["scales"])
    np.testing.assert_array_equal(objectives, [candidate["objective"] for candidate in results])
    if (selected != fitted["selected_candidate_index"]
            or (fitted["alpha"], fitted["beta"]) != pairs[selected]
            or fitted["objective"] != float(objectives[selected])):
        raise ValueError("Selected gains are not the minimum predeclared training objective")
    if not all(fitted["verification"].values()) or fitted["verification"]["streaming_parity_prefixes"] != 24:
        raise ValueError("Gain-selection verification is incomplete")
    AlphaBetaPredictor(fitted["alpha"], fitted["beta"], fitted["slope_caps"])
    return fitted


def checked_alpha_replay(predictor, truth, missing):
    original, original_mask = truth.copy(), missing.copy()
    settings = (predictor.alpha, predictor.beta, predictor.max_delta_per_step)
    first = replay_segment(predictor, truth, missing)
    second = replay_segment(predictor, truth, missing)
    metrics = score_hidden(truth, missing, first)
    if metrics != score_hidden(truth, missing, second):
        raise ValueError("Repeated metrics differ")
    if metrics["unavailable_samples"] or metrics["scored_samples"] != int(missing.sum()):
        raise ValueError("Not all hidden positions were reconstructed")
    poisoned = truth.copy()
    poisoned[missing] = np.nan
    isolated = replay_segment(predictor, poisoned, missing)
    future = truth.copy()
    cutoff = len(truth) // 2
    future[cutoff:] += 10
    altered = replay_segment(predictor, future, missing)
    for field in ("values", "status", "missing_run"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
        np.testing.assert_array_equal(getattr(first, field), getattr(isolated, field))
        np.testing.assert_array_equal(getattr(first, field)[:cutoff], getattr(altered, field)[:cutoff])
    np.testing.assert_array_equal(first.values[~missing], truth[~missing])
    np.testing.assert_array_equal(truth, original)
    np.testing.assert_array_equal(missing, original_mask)
    assert settings[:2] == (predictor.alpha, predictor.beta)
    np.testing.assert_array_equal(settings[2], predictor.max_delta_per_step)
    return first, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gains-sha256", required=True, help="Checksum printed by training-only selection")
    parser.add_argument("--gains", type=Path, default=ROOT / "data/processed/alpha_beta_v1/gains.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/alpha_beta_validation_subject107_all_regimes.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    context = load_context({"src/packetheal/gain_selection.py", "scripts/select_alpha_beta_gains.py",
                            "scripts/evaluate_alpha_beta.py"})
    search_path = ROOT / "configs/alpha_beta_gain_search_v1.json"
    search = json.loads(search_path.read_text())
    fitted = load_selected_gains(args.gains, args.gains_sha256, context, search)
    # Validate selection provenance without reopening any training recording.
    paths, hashes = context["paths"], context["hashes"]
    paths["gain_search"], hashes["gain_search"] = search_path, sha256(search_path)
    for name, expected in fitted["provenance"]["artifact_sha256"].items():
        if name.startswith("training_recording_"):
            subject = name.removeprefix("training_recording_")
            if expected != context["caps"]["provenance"]["training_recordings"][subject]["sha256"]:
                raise ValueError("Training recording provenance differs between gain and cap artifacts")
        elif hashes.get(name) != expected:
            raise ValueError(f"Gain input provenance changed: {name}")
    for name, expected in fitted["provenance"]["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Gain selection source changed: {name}")
        context["sources"][name] = expected
    paths["gains"], hashes["gains"] = args.gains, args.gains_sha256
    paths["gains_sidecar"] = args.gains.with_suffix(args.gains.suffix + ".sha256")
    hashes["gains_sidecar"] = sha256(paths["gains_sidecar"])
    archives = {}
    for name, filename, expected in (
        ("locf", "locf_validation_subject107_v1", LOCF_SHA256),
        ("linear", "linear_validation_subject107_v1", LINEAR_SHA256),
    ):
        path = ROOT / "benchmarks" / filename / "benchmark.json"
        if sha256(path) != expected:
            raise ValueError(f"Archived baseline changed: {name}")
        archives[name] = json.loads(path.read_text())
        paths[f"{name}_benchmark"], hashes[f"{name}_benchmark"] = path, expected
    expected = context["preparation"]["splits"]["validation"]
    if expected["subjects"] != [107] or context["config"]["validation_subjects"] != [107]:
        raise ValueError("Only validation subject 107 is allowed")
    recording = context["dataset"] / "Protocol/subject107.dat"
    recording_hash = sha256(recording)
    if any(a["provenance"]["recording_sha256"] != recording_hash for a in archives.values()):
        raise ValueError("Validation recording differs from baseline inputs")
    paths["validation_recording"], hashes["validation_recording"] = recording, recording_hash
    segments, masks = load_subject_masks(paths["loss_masks"], paths["segment_manifest"],
        paths["split_config"], "validation", 107,
        history_preserved_samples=context["preparation"]["history_preserved_samples"])
    if len(segments) != expected["segments"]:
        raise ValueError("Validation segment count mismatch")
    linear_cases = {(c["segment_index"], c["regime"]): c for c in archives["linear"]["per_segment"]}
    predictor = AlphaBetaPredictor(fitted["alpha"], fitted["beta"], fitted["slope_caps"])
    cases, samples = [], 0
    for index, segment in enumerate(segments):
        timestamps, truth = load_split_segment(context["dataset"], paths["segment_manifest"],
                                               paths["split_config"], "validation", 107, index)
        samples += len(truth)
        for regime in REGIMES:
            missing = masks[index, regime]
            locf = replay_segment(LOCF(), truth, missing)
            locf_metrics = score_hidden(truth, missing, locf)
            check_locf_reference(archives["locf"], index, regime, segment, locf, locf_metrics)
            linear = replay_segment(LinearExtrapolator(fitted["slope_caps"]), truth, missing)
            linear_metrics = score_hidden(truth, missing, linear)
            saved = linear_cases[index, regime]
            if (saved["segment"] != segment or saved["replay_sha256"] != replay_digest(linear)
                    or saved["metrics"] != linear_metrics):
                raise ValueError("Linear replay differs from archive")
            result, metrics = checked_alpha_replay(predictor, truth, missing)
            cases.append({"segment_index": index, "segment": segment, "regime": regime,
                "samples": len(truth), "timestamp_start_s": float(timestamps[0]),
                "timestamp_end_s": float(timestamps[-1]), "actual_missing_fraction": float(missing.mean()),
                "max_missing_run": int(result.missing_run.max()), "replay_sha256": replay_digest(result),
                "locf_replay_sha256": replay_digest(locf), "linear_replay_sha256": replay_digest(linear),
                "metrics": metrics, "locf_metrics": locf_metrics, "linear_metrics": linear_metrics})
        print(f"Verified validation segment {index + 1}/{len(segments)} under four regimes", flush=True)
    if samples != expected["samples"] or len(cases) != len(segments) * len(REGIMES):
        raise ValueError("Validation sample/case total mismatch")
    pooled, comparison = {}, {}
    for regime in REGIMES:
        selected = [case for case in cases if case["regime"] == regime]
        entry = {key: pool_metrics([case[key] for case in selected])
                 for key in ("metrics", "locf_metrics", "linear_metrics")}
        if entry["metrics"]["hidden_samples"] != expected["lost_by_regime"][regime]:
            raise ValueError("Validation hidden total mismatch")
        for name in ("locf", "linear"):
            if entry[f"{name}_metrics"] != archives[name]["pooled_by_regime"][regime]["metrics"]:
                raise ValueError("Pooled baseline metrics differ from archive")
        pooled[regime] = {"segments": len(selected), "samples": samples,
            "max_missing_run": max(case["max_missing_run"] for case in selected), **entry}
        comparison[regime] = [{"channel": channel, "unit": "m/s²" if c < 3 else "rad/s",
            **{f"{name}_{metric}": entry[key][metric][c]
               for name, key in (("alpha_beta", "metrics"), ("locf", "locf_metrics"), ("linear", "linear_metrics"))
               for metric in ("mae", "rmse")}}
            for c, channel in enumerate(context["config"]["channels"])]
    verify_unchanged(context)
    report = {"schema_version": "1", "predictor": "AlphaBetaPredictor",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "split": "validation", "subject": 107,
        "segments": len(segments), "samples": samples, "cases": len(cases),
        "alpha": fitted["alpha"], "beta": fitted["beta"], "slope_caps": fitted["slope_caps"],
        "gains_sha256": args.gains_sha256, "channel_order": context["config"]["channels"],
        "channel_units": ["m/s²"] * 3 + ["rad/s"] * 3, "sampling_frequency_hz": 100,
        "normalization_applied": False, "mask_seed": context["preparation"]["seed"],
        "history_preserved_samples": context["preparation"]["history_preserved_samples"],
        "loss_source": "frozen synthetic sample losses; no transport timing simulation",
        "aggregation": "pool absolute/squared error sums and scored counts per channel/regime",
        "verification": {"both_baselines_match_archives_all_cases": True,
            "repeat_and_segment_reset_identical_all_cases": True, "received_and_original_arrays_preserved": True,
            "hidden_truth_nan_replay_identical_all_cases": True, "future_change_prefix_identical_all_cases": True,
            "all_hidden_positions_scored": True, "all_totals_and_baseline_pooled_metrics_reconciled": True,
            "gains_frozen_before_validation": True, "inputs_and_sources_unchanged": True},
        "pooled_by_regime": pooled, "comparison_by_regime": comparison, "per_segment": cases,
        "provenance": {"git_commit": git_output("rev-parse", "HEAD"),
            "git_status_porcelain": git_output("status", "--porcelain"),
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "runtime_dependency_pins": [f"numpy=={np.__version__}"],
            "recording_sha256": recording_hash,
            "artifact_sha256": hashes, "source_sha256": context["sources"],
            "training_selection": "all eligible training subjects only; no validation retuning"},
        "interpretation": "One validation subject/one frozen seed per regime; no test, uncertainty, transport or device result."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(args.output.resolve()), "alpha": fitted["alpha"], "beta": fitted["beta"],
                      "cases": len(cases), "comparison_by_regime": comparison}, indent=2))


if __name__ == "__main__":
    main()
