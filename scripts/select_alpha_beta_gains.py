#!/usr/bin/env python3
"""Select one alpha-beta gain pair from the predeclared training-only grid.

    python3 -B scripts/select_alpha_beta_gains.py

Uses all eligible subjects 101-106 and four frozen TRAIN masks. No validation
or test recording is opened. Physical inputs remain raw; frozen training
standard deviations scale only the selection objective across channel units.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np

from alpha_beta_common import ROOT, load_context, verify_unchanged
from evaluate_locf import git_output, sha256
from packetheal.alpha_beta import AlphaBetaPredictor
from packetheal.evaluation import REGIMES, load_subject_masks, replay_segment, score_hidden
from packetheal.gain_selection import choose_gain_candidate, score_gain_candidates
from packetheal.splits import load_split_config, load_split_segment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/alpha_beta_v1/gains.json")
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite gain selection: {args.output}")
    context = load_context({"src/packetheal/gain_selection.py", "scripts/select_alpha_beta_gains.py"})
    search_path = ROOT / "configs/alpha_beta_gain_search_v1.json"
    context["paths"]["gain_search"] = search_path
    context["hashes"]["gain_search"] = sha256(search_path)
    search = json.loads(search_path.read_text())
    expected = context["preparation"]["splits"]["train"]
    subjects = list(range(101, 107))
    groups = load_split_config(context["paths"]["split_config"])
    if (search["fit_split"] != "train" or search["subjects"] != subjects
            or groups["train"] != subjects or expected["subjects"] != subjects
            or search["regimes"] != list(REGIMES)
            or search["alpha_values"] != [.25, .5, .75, 1.]
            or search["beta_values"] != [0., .1, .25, .5, 1.]):
        raise ValueError("Selection must use the predeclared training subjects, regimes, and 20-pair grid")
    pairs = [(a, b) for a in search["alpha_values"] for b in search["beta_values"]]
    limits = context["caps"]["max_delta_per_step"]
    recordings = {}
    for subject in subjects:
        name = f"subject{subject}.dat"
        path = context["dataset"] / "Protocol" / name
        digest = sha256(path)
        if digest != context["caps"]["provenance"]["training_recordings"][str(subject)]["sha256"]:
            raise ValueError(f"Training recording differs from calibration: {name}")
        recordings[str(subject)] = {"source_file": name, "sha256": digest}
        context["paths"][f"training_recording_{subject}"] = path
        context["hashes"][f"training_recording_{subject}"] = digest
    absolute = np.zeros((len(REGIMES), len(pairs), 6))
    squared = np.zeros_like(absolute)
    counts = np.zeros(len(REGIMES), dtype=np.int64)
    segments_done = samples_done = parity_prefixes = 0
    per_subject = {}
    print(f"Frozen search: {len(pairs)} pairs; all {expected['segments']} training segments; no validation", flush=True)
    for subject in subjects:
        segments, masks = load_subject_masks(
            context["paths"]["loss_masks"], context["paths"]["segment_manifest"],
            context["paths"]["split_config"], "train", subject,
            history_preserved_samples=context["preparation"]["history_preserved_samples"],
        )
        subject_samples = 0
        for index, segment in enumerate(segments):
            if segment["source_file"] != f"subject{subject}.dat":
                raise ValueError("Unexpected training source identity")
            _, truth = load_split_segment(
                context["dataset"], context["paths"]["segment_manifest"],
                context["paths"]["split_config"], "train", subject, index,
            )
            if len(truth) != int(segment["samples"]):
                raise ValueError("Loaded training length disagrees with manifest")
            subject_samples += len(truth)
            for r, regime in enumerate(REGIMES):
                mask = masks[index, regime]
                scores = score_gain_candidates(truth, mask, pairs, limits)
                absolute[r] += scores["absolute_error_sum"]
                squared[r] += scores["squared_error_sum"]
                counts[r] += scores["hidden_samples"]
                if index == 0:
                    # All 20 candidates on a fixed, bounded prefix per subject
                    # and regime: optimized training math must match real replay.
                    prefix, prefix_mask = truth[:128], mask[:128]
                    batched = score_gain_candidates(prefix, prefix_mask, pairs, limits)
                    for k, (alpha, beta) in enumerate(pairs):
                        scalar = score_hidden(prefix, prefix_mask, replay_segment(
                            AlphaBetaPredictor(alpha, beta, limits), prefix, prefix_mask,
                        ))
                        for key in ("absolute_error_sum", "squared_error_sum"):
                            np.testing.assert_allclose(batched[key][k], scalar[key], rtol=1e-12, atol=1e-12)
                    parity_prefixes += 1
            segments_done += 1
            if segments_done % 10 == 0 or index + 1 == len(segments):
                print(f"Training selection: {segments_done}/{expected['segments']} segments; subject {subject}", flush=True)
        samples_done += subject_samples
        per_subject[str(subject)] = {"segments": len(segments), "samples": subject_samples}
        recorded = context["caps"]["per_subject"][str(subject)]
        if len(segments) != recorded["segments"] or subject_samples != recorded["samples"]:
            raise ValueError("Subject training totals differ from frozen calibration")
    if (segments_done != expected["segments"] or samples_done != expected["samples"]
            or counts.tolist() != [expected["lost_by_regime"][r] for r in REGIMES]):
        raise ValueError("Selection segment/sample/hidden counts differ from preparation")
    selected, objectives = choose_gain_candidate(absolute, counts, context["scales"])
    repeated, repeat_scores = choose_gain_candidate(absolute.copy(), counts.copy(), context["scales"].copy())
    np.testing.assert_array_equal(objectives, repeat_scores)
    if selected != repeated or parity_prefixes != 24:
        raise ValueError("Selection determinism or streaming parity incomplete")
    candidates = []
    for k, (alpha, beta) in enumerate(pairs):
        by_regime = {}
        for r, regime in enumerate(REGIMES):
            by_regime[regime] = {"hidden_samples": int(counts[r]),
                "absolute_error_sum": absolute[r, k].tolist(), "squared_error_sum": squared[r, k].tolist(),
                "mae": (absolute[r, k] / counts[r]).tolist(),
                "rmse": np.sqrt(squared[r, k] / counts[r]).tolist()}
        candidates.append({"alpha": alpha, "beta": beta, "objective": float(objectives[k]), "by_regime": by_regime})
    verify_unchanged(context)
    artifact = {
        "schema_version": "1", "purpose": "alpha_beta_training_gain_selection",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "fit_split": "train", "subjects": subjects,
        "segments": segments_done, "samples": samples_done, "cases": segments_done * len(REGIMES),
        "per_subject": per_subject, "search": search, "selected_candidate_index": selected,
        "alpha": pairs[selected][0], "beta": pairs[selected][1], "objective": float(objectives[selected]),
        "candidate_results": candidates, "slope_caps": limits,
        "slope_caps_sha256": context["hashes"]["slope_caps"],
        "channel_order": context["config"]["channels"], "sampling_frequency_hz": 100,
        "objective_training_std_sample": context["scales"].tolist(), "normalization_applied_to_predictor": False,
        "verification": {"all_training_counts_reconciled": True, "streaming_parity_prefixes": parity_prefixes,
                         "repeat_objective_and_selection_identical": True, "inputs_and_sources_unchanged": True},
        "provenance": {"git_commit": git_output("rev-parse", "HEAD"),
            "git_status_porcelain": git_output("status", "--porcelain"),
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "artifact_sha256": context["hashes"], "source_sha256": context["sources"],
            "training_recordings": recordings, "randomness": "none; frozen training masks"},
        "interpretation": "Minimum training objective among 20 fixed candidates; not validation/test performance or confidence calibration.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write("\n")
    digest = sha256(args.output)
    with sidecar.open("x") as stream:
        stream.write(f"{digest}  {args.output.name}\n")
    print(json.dumps({"artifact": str(args.output.resolve()), "sha256": digest,
                      "selected_alpha": artifact["alpha"], "selected_beta": artifact["beta"],
                      "objective": artifact["objective"], "segments": segments_done, "samples": samples_done}, indent=2))


if __name__ == "__main__":
    main()
