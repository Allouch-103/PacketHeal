"""Causal streaming baselines; predictors never receive hidden ground truth."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SampleStatus(str, Enum):
    MEASURED = "MEASURED"
    RECONSTRUCTED = "RECONSTRUCTED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class StepResult:
    values: np.ndarray
    status: SampleStatus
    missing_run: int


class LOCF:
    """Hold the last received vector, without feeding estimates back as real data.

    Call ``step(vector)`` for a received sample and ``step(None)`` for a loss.
    Before the first received sample, losses produce NaNs and UNAVAILABLE.
    This baseline has no gap-length cutoff or calibrated confidence estimate.
    """

    def __init__(self, n_channels=6):
        if isinstance(n_channels, bool) or not isinstance(n_channels, int):
            raise TypeError("n_channels must be an integer")
        if n_channels < 1:
            raise ValueError("n_channels must be positive")
        self.n_channels = n_channels
        self.reset()

    def reset(self):
        """Forget all history at a segment boundary."""
        self._last_received = None
        self._missing_run = 0

    def step(self, observation):
        """Return an owned output vector, provenance, and consecutive-loss count."""
        if observation is not None:
            values = np.asarray(observation, dtype=np.float64)
            if values.shape != (self.n_channels,):
                raise ValueError("Received sample has the wrong channel shape")
            if not np.isfinite(values).all():
                raise ValueError("Received samples must contain only finite values")
            self._last_received = values.copy()
            self._missing_run = 0
            return StepResult(values.copy(), SampleStatus.MEASURED, 0)

        self._missing_run += 1
        if self._last_received is None:
            return StepResult(
                np.full(self.n_channels, np.nan),
                SampleStatus.UNAVAILABLE,
                self._missing_run,
            )
        return StepResult(
            self._last_received.copy(),
            SampleStatus.RECONSTRUCTED,
            self._missing_run,
        )
