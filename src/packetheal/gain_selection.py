"""Batched training scores for an explicit alpha-beta gain grid.

This optimization carries candidate states together, never across segments.
Its error sums are checked against the public streaming predictor in tests
and on training prefixes by the selection runner. It does not load data or
choose a partition; the runner must enforce training-only access.
"""

import numpy as np

from .alpha_beta import AlphaBetaPredictor


def score_gain_candidates(ground_truth, missing, candidates, max_delta_per_step):
    """Return per-candidate absolute/squared hidden-error sums in raw units.

    Complete finite segments and a received first sample are required; frozen
    training masks protect the first 20. A zero-loss prefix is allowed for
    parity checks. Hidden values enter scoring only, never state correction.
    Memory is O(candidates * channels), not O(samples * candidates * channels).
    """
    if np.iscomplexobj(ground_truth):
        raise ValueError("Ground truth must be real")
    truth = np.asarray(ground_truth, dtype=np.float64)
    mask = np.asarray(missing)
    if truth.ndim != 2 or not len(truth) or not truth.shape[1] or not np.isfinite(truth).all():
        raise ValueError("Ground truth must be a nonempty finite (N, C) array")
    if mask.dtype != np.bool_ or mask.shape != (len(truth),) or mask[0]:
        raise ValueError("Mask must be Boolean, match the segment, and preserve its first sample")
    pairs = list(candidates)
    if not pairs:
        raise ValueError("At least one gain candidate is required")
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("Each candidate must be an alpha/beta pair")
        AlphaBetaPredictor._gain(pair[0], "alpha", exclude_zero=True)
        AlphaBetaPredictor._gain(pair[1], "beta")
    validated = AlphaBetaPredictor(*pairs[0], max_delta_per_step, n_channels=truth.shape[1])
    caps = validated.max_delta_per_step
    gains = np.asarray(pairs, dtype=np.float64)
    alpha, beta = gains[:, :1], gains[:, 1:]
    smooth, moving = np.flatnonzero(alpha[:, 0] != 1), np.flatnonzero(beta[:, 0] != 0)
    exact = np.flatnonzero(alpha[:, 0] == 1)
    values = np.broadcast_to(truth[0], (len(pairs), truth.shape[1])).copy()
    trends = np.zeros_like(values)
    absolute, squared = np.zeros_like(values), np.zeros_like(values)
    last_update = 0
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for index in range(1, len(truth)):
            elapsed = index - last_update
            observation = truth[index]
            if mask[index]:
                errors = values + elapsed * trends - observation
                absolute += np.abs(errors)
                squared += errors ** 2
                continue
            # Both corrections use OLD values/trends, not each other's result.
            if len(smooth):
                predicted = values[smooth] + elapsed * trends[smooth]
                corrected = (1 - alpha[smooth]) * predicted + alpha[smooth] * observation
            if len(moving):
                target = (observation - values[moving]) / elapsed
                trends[moving] = (1 - beta[moving]) * trends[moving] + beta[moving] * target
                np.clip(trends, -caps, caps, out=trends)
            if len(smooth):
                values[smooth] = corrected
            if len(exact):
                values[exact] = observation
            last_update = index
    return {"hidden_samples": int(mask.sum()), "absolute_error_sum": absolute,
            "squared_error_sum": squared}


def choose_gain_candidate(absolute_error_sums, hidden_counts, training_std):
    """Minimize mean_r,c(MAE[r,k,c]/train_std[c]); ties use candidate order."""
    sums = np.asarray(absolute_error_sums, dtype=np.float64)
    counts = np.asarray(hidden_counts)
    scales = np.asarray(training_std, dtype=np.float64)
    if (sums.ndim != 3 or not all(sums.shape) or counts.shape != (sums.shape[0],)
            or scales.shape != (sums.shape[2],)):
        raise ValueError("Expected sums (regimes, candidates, channels), counts (regimes), std (channels)")
    if (not np.isfinite(sums).all() or (sums < 0).any()
            or not np.issubdtype(counts.dtype, np.integer) or (counts <= 0).any()
            or not np.isfinite(scales).all() or (scales <= 0).any()):
        raise ValueError("Require finite nonnegative sums, positive integer counts, positive finite scales")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scores = (sums / counts[:, None, None] / scales[None, None, :]).mean(axis=(0, 2))
    return int(np.argmin(scores)), scores
