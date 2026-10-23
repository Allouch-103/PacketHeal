"""Training-only slope limits for the nominal uniformly sampled benchmark."""

import csv
from pathlib import Path

import numpy as np

from .splits import load_split_config, load_split_segment


QUANTILE = 0.99
QUANTILE_METHOD = "linear"


def fit_slope_caps(segments, quantile=QUANTILE, n_channels=6):
    """Pool absolute within-segment changes and take a quantile per channel.

    ``segments`` yields original finite arrays of shape (N, channels), N >= 2.
    No normalization, masks, measurement clipping, or boundary-crossing pair
    is used. Each consecutive pair has equal weight. This pure calculation
    does not know subject identities; real experiments must enter through
    ``calibrate_training_slope_caps`` below to enforce the training partition.
    """
    if isinstance(n_channels, bool) or not isinstance(n_channels, int):
        raise TypeError("n_channels must be an integer")
    if n_channels < 1:
        raise ValueError("n_channels must be positive")
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float, np.integer, np.floating)):
        raise ValueError("quantile must be a finite number in [0, 1]")
    if not np.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("quantile must be a finite number in [0, 1]")
    changes = []
    samples = 0
    for segment in segments:
        if np.iscomplexobj(segment):
            raise ValueError("Calibration segments must contain real values")
        values = np.asarray(segment, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != n_channels or len(values) < 2:
            raise ValueError("Each segment must have shape (N >= 2, n_channels)")
        if not np.isfinite(values).all():
            raise ValueError("Calibration segments must contain only finite values")
        with np.errstate(over="raise", invalid="raise"):
            changes.append(np.abs(np.diff(values, axis=0)))
        samples += len(values)
    if not changes:
        raise ValueError("At least one calibration segment is required")
    segment_count = len(changes)
    pooled = np.concatenate(changes, axis=0)
    del changes
    limits = np.quantile(pooled, quantile, axis=0, method=QUANTILE_METHOD)
    # Pair order must not affect a pooled percentile. Repeat the derivation
    # using the same training changes in reverse order, without reloading data.
    repeated = np.quantile(pooled[::-1], quantile, axis=0, method=QUANTILE_METHOD)
    if not np.array_equal(limits, repeated):
        raise ValueError("Repeated quantile calculation produced different caps")
    if not np.isfinite(limits).all() or (limits < 0).any():
        raise ValueError("Slope caps must be finite and nonnegative")
    return {
        "quantile": float(quantile), "quantile_method": QUANTILE_METHOD,
        "n_channels": n_channels, "segments": segment_count, "samples": samples,
        "pairs_per_channel": len(pooled), "max_delta_per_step": limits.tolist(),
        "strictly_above_cap_counts": (pooled > limits).sum(axis=0).tolist(),
        "repeat_quantile_identical": True,
    }


def calibrate_training_slope_caps(
    dataset_dir, manifest_path, config_path, *, quantile=QUANTILE, progress=None,
):
    """Load only frozen training subjects 101–106 via the split-aware loader.

    Return caps, actual counts, and subject-level input counts. Optional progress
    receives (subject, completed_segments_for_subject, subject_segment_count).
    Artifact hash verification and persistence belong to the CLI runner.
    """
    groups = load_split_config(config_path)
    subjects = groups["train"]
    if subjects != list(range(101, 107)):
        raise ValueError("Calibration v1 requires training subjects 101–106 only")
    by_subject = {subject: [] for subject in subjects}
    with Path(manifest_path).open(newline="") as stream:
        for record in csv.DictReader(stream):
            subject = int(record["subject"].removeprefix("subject"))
            if subject not in by_subject:
                continue
            start, end, count = (int(record[k]) for k in ("start_row", "end_row", "samples"))
            if (record["source_file"] != f"subject{subject}.dat" or start < 1
                    or end < start or count != end - start + 1 or count < 2):
                raise ValueError("Invalid training segment identity or boundaries")
            by_subject[subject].append(record)
    subject_counts = {}
    for subject, records in by_subject.items():
        if not records:
            raise ValueError(f"No training segments found for subject {subject}")
        ordered = sorted(records, key=lambda r: int(r["start_row"]))
        if any(int(a["end_row"]) >= int(b["start_row"]) for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Training manifest contains overlapping segments")
        subject_counts[str(subject)] = {
            "segments": len(records),
            "samples": sum(int(row["samples"]) for row in records),
            "pairs_per_channel": sum(int(row["samples"]) - 1 for row in records),
        }

    def training_segments():
        for subject, records in by_subject.items():
            # Preserve manifest order: segment_index has the same meaning as
            # in dataset.load_segment, even if manifest rows are not sorted.
            for index, record in enumerate(records):
                _, values = load_split_segment(
                    dataset_dir, manifest_path, config_path, split="train",
                    subject=subject, segment_index=index,
                )
                if len(values) != int(record["samples"]):
                    raise ValueError("Loaded training length differs from the manifest")
                if progress is not None:
                    progress(subject, index + 1, len(records))
                yield values

    result = fit_slope_caps(training_segments(), quantile=quantile, n_channels=6)
    for key in ("segments", "samples", "pairs_per_channel"):
        if result[key] != sum(counts[key] for counts in subject_counts.values()):
            raise ValueError(f"Loaded {key} disagrees with the training manifest")
    return {**result, "fit_split": "train", "subjects": subjects, "per_subject": subject_counts}
