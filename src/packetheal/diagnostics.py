"""Post-hoc error strata; none of these labels are predictor inputs."""

import numpy as np

from .baselines import SampleStatus
from .evaluation import score_hidden


def large_change_flags(ground_truth, thresholds):
    """Flag abs(x[t,c]-x[t-1,c]) > frozen training threshold[c].

    Call separately per segment. The first row is always False. This detects
    large recorded changes, not verified physical motion or sensor faults.
    Hidden truth is legitimate for this offline diagnostic only.
    """
    if np.iscomplexobj(ground_truth) or np.iscomplexobj(thresholds):
        raise ValueError("Diagnostic inputs must be real")
    truth, limits = np.asarray(ground_truth, dtype=float), np.asarray(thresholds, dtype=float)
    if (truth.ndim != 2 or not all(truth.shape) or limits.shape != (truth.shape[1],)
            or not np.isfinite(truth).all() or not np.isfinite(limits).all()
            or (limits < 0).any()):
        raise ValueError("Require finite nonempty (N,C) truth and nonnegative (C,) thresholds")
    flags = np.zeros(truth.shape, dtype=bool)
    with np.errstate(over="raise", invalid="raise"):
        flags[1:] = np.abs(np.diff(truth, axis=0)) > limits
    return flags


def loss_run_metadata(missing):
    """Return independent loss ages and complete run lengths (zero if received).

    Complete run length is an offline label that can use the rest of the mask;
    it must never be supplied to the causal predictor.
    """
    mask = np.asarray(missing)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("missing must be a one-dimensional Boolean mask")
    ages, lengths = np.zeros(len(mask), dtype=np.int64), np.zeros(len(mask), dtype=np.int64)
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        ages[start:end] = np.arange(1, end - start + 1)
        lengths[start:end] = end - start
    return ages, lengths


def stratified_sums(ground_truth, missing, result, selection):
    """Score a Boolean (N,C) subset of the ORIGINAL hidden positions.

    Keep the original mask for provenance checks; replacing it with a subset
    would incorrectly label other reconstructed positions as measured.
    Counts vary per channel. Unavailable selected positions remain explicit.
    """
    score_hidden(ground_truth, missing, result)  # Shared full-mask validation.
    truth, mask, selected = (np.asarray(ground_truth, dtype=float),
                             np.asarray(missing), np.asarray(selection))
    if (selected.shape != truth.shape or selected.dtype != np.bool_
            or (selected & ~mask[:, None]).any()):
        raise ValueError("selection must be Boolean (N,C) and contain hidden positions only")
    scored = selected & (result.status == SampleStatus.RECONSTRUCTED.value)[:, None]
    unavailable = selected & (result.status == SampleStatus.UNAVAILABLE.value)[:, None]
    errors = np.zeros_like(truth)
    with np.errstate(over="raise", invalid="raise"):
        np.subtract(result.values, truth, out=errors, where=scored)
        absolute, squared = np.abs(errors).sum(axis=0), (errors ** 2).sum(axis=0)
    return {"hidden_count": selected.sum(axis=0).tolist(),
            "scored_count": scored.sum(axis=0).tolist(),
            "unavailable_count": unavailable.sum(axis=0).tolist(),
            "absolute_error_sum": absolute.tolist(), "squared_error_sum": squared.tolist()}


def pool_strata(records):
    """Pool per-channel sums/counts; empty channel strata have None metrics."""
    if not records:
        raise ValueError("At least one stratum record is required")
    shape = np.asarray(records[0]["hidden_count"]).shape
    if len(shape) != 1 or not shape[0]:
        raise ValueError("Stratum fields must be nonempty channel vectors")
    count_fields = ("hidden_count", "scored_count", "unavailable_count")
    sum_fields = ("absolute_error_sum", "squared_error_sum")
    totals = {key: np.zeros(shape, dtype=np.int64 if key in count_fields else float)
              for key in (*count_fields, *sum_fields)}
    for record in records:
        arrays = {key: np.asarray(record[key]) for key in totals}
        if any(a.shape != shape for a in arrays.values()):
            raise ValueError("Stratum channel dimensions differ")
        if any(not np.issubdtype(arrays[k].dtype, np.integer) or (arrays[k] < 0).any()
               for k in count_fields):
            raise ValueError("Counts must be nonnegative integer channel vectors")
        if not np.array_equal(arrays["hidden_count"], arrays["scored_count"] + arrays["unavailable_count"]):
            raise ValueError("Hidden/scored/unavailable counts do not reconcile")
        if any(np.iscomplexobj(arrays[k]) or not np.isfinite(arrays[k]).all()
               or (arrays[k] < 0).any() for k in sum_fields):
            raise ValueError("Error sums must be finite and nonnegative")
        empty = arrays["scored_count"] == 0
        if any((arrays[k][empty] != 0).any() for k in sum_fields):
            raise ValueError("Unscored channels must have zero error sums")
        with np.errstate(over="raise", invalid="raise"):
            for key in totals:
                totals[key] += arrays[key]
    hidden, scored = totals["hidden_count"], totals["scored_count"]
    return {**{k: v.tolist() for k, v in totals.items()},
            "mae": [float(a / n) if n else None for a, n in zip(totals["absolute_error_sum"], scored)],
            "rmse": [float(np.sqrt(s / n)) if n else None for s, n in zip(totals["squared_error_sum"], scored)],
            "coverage": [float(n / h) if h else None for n, h in zip(scored, hidden)]}


def reconcile_partition(parts, whole):
    """Require disjoint exhaustive groups' counts/sums to reconcile numerically.

    This arithmetic check complements tests of how group masks are constructed;
    totals alone cannot prove disjointness of arbitrary caller-provided groups.
    """
    pooled = pool_strata(parts)
    for key in ("hidden_count", "scored_count", "unavailable_count"):
        np.testing.assert_array_equal(pooled[key], whole[key])
    for key in ("absolute_error_sum", "squared_error_sum"):
        np.testing.assert_allclose(pooled[key], whole[key], rtol=1e-12, atol=1e-12)
