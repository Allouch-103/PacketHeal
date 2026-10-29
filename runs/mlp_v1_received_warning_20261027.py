"""Causal received-acceleration warning, separate from frozen predictors."""

import numpy as np


class ReceivedChangeWarning:
    """Track the latest two received accelerations and nominal sample indices.

    Score = max_c(abs(a_last[c]-a_previous[c]) / elapsed_steps / q99[c]).
    Known only after two real samples. Score > 1 warns the whole vector.
    Missing steps hold the score; reconstructed samples are never observations.
    """

    def __init__(self, thresholds):
        self.thresholds = np.array(thresholds, dtype=np.float64, copy=True)
        if self.thresholds.shape != (3,) or not np.isfinite(self.thresholds).all() or (self.thresholds <= 0).any():
            raise ValueError("Three positive finite acceleration thresholds required")
        self.thresholds.flags.writeable = False
        self.reset()

    def reset(self):
        self.index = 0
        self.last_index = self.previous_index = -1
        self.last = None
        self.score = float("nan")
        self.known = False
        self.age = 0

    def step(self, observation):
        values = None
        if observation is not None:
            values = np.array(observation, dtype=np.float64, copy=True)
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError("A received observation must be a finite six-vector")
        if values is None:
            self.age += 1
        else:
            if self.last is not None:
                elapsed = self.index - self.last_index
                with np.errstate(over="raise", invalid="raise"):
                    score = float(np.max(np.abs(values[:3] - self.last) / elapsed / self.thresholds))
                self.score, self.known = score, True
            self.previous_index, self.last_index = self.last_index, self.index
            self.last = values[:3].copy()
            self.age = 0
        self.index += 1
        return (self.known and self.score > 1, self.known, self.score, self.age,
                self.last_index, self.previous_index)


def warning_replay(warning, truth, missing):
    """Only pass received vectors or None, exactly as for the predictors."""
    warning.reset()
    rows = [warning.step(None if hidden else truth[t].copy()) for t, hidden in enumerate(missing)]
    return tuple(np.array(column, dtype=dtype) for column, dtype in
                 zip(zip(*rows), (bool, bool, float, np.int64, np.int64, np.int64)))


def reference_warning(truth, missing, thresholds):
    """Independent past-index formula for verification, not the streaming path."""
    received = np.flatnonzero(~missing)
    n = len(missing)
    positions = np.searchsorted(received, np.arange(n), side="right") - 1
    known = positions >= 1
    score = np.full(n, np.nan)
    last, previous = np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64)
    has_one = positions >= 0
    last[has_one] = received[positions[has_one]]
    previous[known] = received[positions[known] - 1]
    if known.any():
        changes = np.abs(truth[received[1:], :3] - truth[received[:-1], :3])
        pair_scores = np.max(changes / np.diff(received)[:, None] / np.asarray(thresholds), axis=1)
        score[known] = pair_scores[positions[known] - 1]
    ages = np.zeros(n, dtype=np.int64)
    for t in range(n):
        ages[t] = (ages[t-1] + 1 if t else 1) if missing[t] else 0
    return known & (score > 1), known, score, ages, last, previous
