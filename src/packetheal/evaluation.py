"""Frozen-mask loading, causal segment replay, and physical-unit error metrics."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .baselines import SampleStatus
from .splits import load_split_config


REGIMES = ("random_5pct", "burst_2_5pct", "burst_5_5pct", "burst_10_5pct")
MASK_FIELDS = [
    "split", "subject", "segment_start_row", "regime", "missing_original_row"
]


def segment_metadata(manifest_path, subject, segment_index):
    """Select in manifest order, exactly as the existing dataset loader does."""
    if isinstance(segment_index, bool) or not isinstance(segment_index, int):
        raise TypeError("segment_index must be an integer")
    with Path(manifest_path).open(newline="") as stream:
        segments = [
            row for row in csv.DictReader(stream)
            if row["subject"] == f"subject{subject}"
        ]
    if not segments:
        raise ValueError(f"No validated segments found for subject {subject}")
    if not 0 <= segment_index < len(segments):
        raise IndexError(f"Invalid segment index {segment_index} for subject {subject}")
    segment = segments[segment_index]
    start, end, count = (int(segment[k]) for k in ("start_row", "end_row", "samples"))
    if start < 1 or end < start or count != end - start + 1:
        raise ValueError("Manifest has inconsistent segment boundaries")
    return segment


def load_segment_mask(
    mask_path, manifest_path, config_path, split, subject, segment_index, regime,
    *, history_preserved_samples=20,
):
    """Read existing positions only; True means the entire sample is hidden.

    CSV positions are one-based original file rows, not local array indices.
    No RNG is used and no mask is regenerated. The v1 CSV has nonempty masks
    for every supported segment/regime; an absent selection is an error.
    """
    groups = load_split_config(config_path)
    if split not in groups or subject not in groups[split]:
        raise ValueError(f"Subject {subject} is not permitted in '{split}'")
    if regime not in REGIMES:
        raise ValueError(f"Unknown loss regime: {regime}")
    if history_preserved_samples < 0:
        raise ValueError("history_preserved_samples must be nonnegative")
    segment = segment_metadata(manifest_path, subject, segment_index)
    start = int(segment["start_row"])
    missing = np.zeros(int(segment["samples"]), dtype=bool)
    with Path(mask_path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != MASK_FIELDS:
            raise ValueError("Unexpected frozen-mask CSV fields")
        for row in reader:
            if (
                row["split"] != split or row["subject"] != str(subject)
                or row["segment_start_row"] != str(start)
                or row["regime"] != regime
            ):
                continue
            index = int(row["missing_original_row"]) - start
            if not 0 <= index < len(missing):
                raise ValueError("Frozen-mask row is outside the selected segment")
            if index < history_preserved_samples:
                raise ValueError("Frozen mask hides a protected warm-up sample")
            if missing[index]:
                raise ValueError("Duplicate frozen-mask position")
            missing[index] = True
    if not missing.any():
        raise ValueError("No frozen-mask records found for this selection")
    return missing


def load_subject_masks(
    mask_path, manifest_path, config_path, split, subject,
    *, history_preserved_samples=20,
):
    """Read all frozen regimes for one permitted subject in one CSV pass.

    Return manifest records in loader order and masks keyed by
    (segment_index, regime). Unknown, duplicate, absent, or invalid positions
    for this subject are errors; other subjects' measurements are never read.
    """
    groups = load_split_config(config_path)
    if split not in groups or subject not in groups[split]:
        raise ValueError(f"Subject {subject} is not permitted in '{split}'")
    if history_preserved_samples < 0:
        raise ValueError("history_preserved_samples must be nonnegative")
    with Path(manifest_path).open(newline="") as stream:
        segments = [row for row in csv.DictReader(stream)
                    if row["subject"] == f"subject{subject}"]
    if not segments:
        raise ValueError(f"No validated segments found for subject {subject}")
    by_start = {}
    masks = {}
    for index, segment in enumerate(segments):
        start, end, count = (int(segment[k]) for k in ("start_row", "end_row", "samples"))
        if start < 1 or end < start or count != end - start + 1 or start in by_start:
            raise ValueError("Manifest has inconsistent or duplicate segment boundaries")
        by_start[start] = index
        for regime in REGIMES:
            masks[index, regime] = np.zeros(count, dtype=bool)
    with Path(mask_path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != MASK_FIELDS:
            raise ValueError("Unexpected frozen-mask CSV fields")
        for row in reader:
            if row["subject"] != str(subject):
                continue
            start = int(row["segment_start_row"])
            regime = row["regime"]
            if row["split"] != split or start not in by_start or regime not in REGIMES:
                raise ValueError("Frozen mask has an unknown subject/segment/regime selection")
            mask = masks[by_start[start], regime]
            index = int(row["missing_original_row"]) - start
            if not 0 <= index < len(mask):
                raise ValueError("Frozen-mask row is outside the selected segment")
            if index < history_preserved_samples:
                raise ValueError("Frozen mask hides a protected warm-up sample")
            if mask[index]:
                raise ValueError("Duplicate frozen-mask position")
            mask[index] = True
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("No frozen-mask records found for a segment/regime selection")
    return segments, masks


@dataclass(frozen=True)
class ReplayResult:
    values: np.ndarray
    status: np.ndarray
    missing_run: np.ndarray


def _validate_arrays(ground_truth, missing):
    truth = np.asarray(ground_truth, dtype=np.float64)
    mask = np.asarray(missing)
    if truth.ndim != 2 or truth.shape[1] < 1:
        raise ValueError("ground_truth must have shape (N, channels)")
    if mask.dtype != np.bool_ or mask.shape != (len(truth),):
        raise ValueError("missing must be a Boolean array of shape (N,)")
    return truth, mask


def replay_segment(predictor, ground_truth, missing):
    """Reset and stream one segment; hidden rows never reach the predictor.

    Only a current received vector or None is passed to step(). Hidden values
    need not even be finite for replay; scoring separately requires finite truth.
    """
    truth, mask = _validate_arrays(ground_truth, missing)
    if truth.shape[1] != predictor.n_channels:
        raise ValueError("Predictor channel count does not match the segment")
    predictor.reset()
    values = np.empty_like(truth)
    status = np.empty(len(truth), dtype="U13")
    missing_run = np.empty(len(truth), dtype=np.int64)
    for index in range(len(truth)):
        observation = None if mask[index] else truth[index].copy()
        output = predictor.step(observation)
        values[index] = output.values
        status[index] = output.status.value
        missing_run[index] = output.missing_run
    return ReplayResult(values, status, missing_run)


def score_hidden(ground_truth, missing, result):
    """Per-channel errors on hidden reconstructed rows, in original units.

    Unavailable rows are counted explicitly and excluded from error metrics.
    If none can be scored, MAE/RMSE are None rather than misleading zeros.
    """
    truth, mask = _validate_arrays(ground_truth, missing)
    if result.values.shape != truth.shape or result.status.shape != mask.shape:
        raise ValueError("Replay result dimensions do not match ground truth")
    reconstructed = result.status == SampleStatus.RECONSTRUCTED.value
    unavailable = result.status == SampleStatus.UNAVAILABLE.value
    measured = result.status == SampleStatus.MEASURED.value
    if not np.all(measured[~mask]) or not np.all((reconstructed | unavailable)[mask]):
        raise ValueError("Replay provenance does not agree with the loss mask")
    if not np.isfinite(truth[mask]).all():
        raise ValueError("Hidden ground truth must be finite for evaluation")
    selected = mask & reconstructed
    hidden_count = int(mask.sum())
    scored_count = int(selected.sum())
    absolute_sum = np.zeros(truth.shape[1], dtype=np.float64)
    squared_sum = np.zeros(truth.shape[1], dtype=np.float64)
    if scored_count:
        if not np.isfinite(result.values[selected]).all():
            raise ValueError("Reconstructed values must be finite")
        with np.errstate(over="raise", invalid="raise"):
            errors = result.values[selected] - truth[selected]
            absolute_sum = np.sum(np.abs(errors), axis=0)
            squared_sum = np.sum(errors ** 2, axis=0)
    return _metrics_from_sums(
        hidden_count, scored_count, int((mask & unavailable).sum()),
        absolute_sum, squared_sum,
    )


def _metrics_from_sums(hidden_count, scored_count, unavailable_count, absolute_sum, squared_sum):
    return {
        "hidden_samples": hidden_count,
        "scored_samples": scored_count,
        "unavailable_samples": unavailable_count,
        "reconstruction_coverage": scored_count / hidden_count if hidden_count else None,
        "absolute_error_sum": absolute_sum.tolist(),
        "squared_error_sum": squared_sum.tolist(),
        "mae": (absolute_sum / scored_count).tolist() if scored_count else None,
        "rmse": np.sqrt(squared_sum / scored_count).tolist() if scored_count else None,
    }


def pool_metrics(segment_metrics):
    """Pool hidden-position errors, weighting each reconstructed sample equally.

    RMSE = sqrt(total squared error / total scored samples), never an average
    of segment RMSEs. Unavailable positions remain explicit in pooled coverage.
    """
    if not segment_metrics:
        raise ValueError("At least one segment metric record is required")
    absolute_sum = np.zeros(len(segment_metrics[0]["absolute_error_sum"]))
    squared_sum = np.zeros_like(absolute_sum)
    hidden = scored = unavailable = 0
    for metrics in segment_metrics:
        absolute = np.asarray(metrics["absolute_error_sum"], dtype=np.float64)
        squared = np.asarray(metrics["squared_error_sum"], dtype=np.float64)
        if absolute.shape != absolute_sum.shape or squared.shape != squared_sum.shape:
            raise ValueError("Cannot pool different channel counts")
        if (not np.isfinite(absolute).all() or not np.isfinite(squared).all()
                or (absolute < 0).any() or (squared < 0).any()):
            raise ValueError("Error sums must be finite and nonnegative")
        counts = [metrics[k] for k in ("hidden_samples", "scored_samples", "unavailable_samples")]
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts):
            raise ValueError("Sample counts must be nonnegative integers")
        if counts[0] != counts[1] + counts[2]:
            raise ValueError("Hidden sample counts do not reconcile")
        hidden += counts[0]
        scored += counts[1]
        unavailable += counts[2]
        with np.errstate(over="raise", invalid="raise"):
            absolute_sum += absolute
            squared_sum += squared
    return _metrics_from_sums(hidden, scored, unavailable, absolute_sum, squared_sum)
