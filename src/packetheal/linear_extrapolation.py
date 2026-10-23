"""Two-received-point extrapolation for uniformly spaced sample steps."""

import numpy as np

from .baselines import SampleStatus, StepResult


class LinearExtrapolator:
    """Extrapolate a capped per-channel slope from the last two real samples.

    ``step(vector)`` consumes a received sample; ``step(None)`` consumes a loss.
    Every successful call advances one nominal sample period (10 ms in the
    current benchmark). This interface does not model irregular arrival times.

    ``max_delta_per_step`` is a required finite, nonnegative vector, in each
    channel's physical units per sample step. Only the slope is capped;
    received measurements pass through unchanged. Production caps must be
    determined from training information, not selected using validation/test.

    Startup losses are UNAVAILABLE. With one received sample, losses use LOCF.
    Estimates never update real history. Reset at every segment boundary.
    There is no outage cutoff, output-magnitude bound, or confidence estimate.
    Invalid input and numerical overflow raise without advancing state.
    """

    def __init__(self, max_delta_per_step, n_channels=6):
        if isinstance(n_channels, bool) or not isinstance(n_channels, int):
            raise TypeError("n_channels must be an integer")
        if n_channels < 1:
            raise ValueError("n_channels must be positive")
        self.n_channels = n_channels
        limits = self._finite_vector(max_delta_per_step, "max_delta_per_step")
        if (limits < 0).any():
            raise ValueError("max_delta_per_step must be nonnegative")
        self._max_delta_per_step = limits.copy()
        self.reset()

    @property
    def max_delta_per_step(self):
        """Return an owned copy so callers cannot silently change the policy."""
        return self._max_delta_per_step.copy()

    def _finite_vector(self, value, name):
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must contain real values")
        values = np.asarray(value, dtype=np.float64)
        if values.shape != (self.n_channels,):
            raise ValueError(f"{name} must have shape ({self.n_channels},)")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite values")
        return values

    def reset(self):
        """Forget received history, slope, loss age, and sample position."""
        self._index = -1
        self._last_received = None
        self._last_received_index = None
        self._slope = None
        self._missing_run = 0

    def step(self, observation):
        """Return values/provenance/loss age, using real history only."""
        index = self._index + 1
        if observation is not None:
            values = self._finite_vector(observation, "Received sample")
            slope = None
            if self._last_received is not None:
                spacing = index - self._last_received_index
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        slope = (values - self._last_received) / spacing
                        slope = np.clip(slope, -self._max_delta_per_step, self._max_delta_per_step)
                except FloatingPointError as error:
                    raise FloatingPointError("Non-finite slope calculation") from error
            # Commit history only after validation and arithmetic succeed.
            self._last_received = values.copy()
            self._last_received_index = index
            self._slope = slope
            self._missing_run = 0
            self._index = index
            return StepResult(values.copy(), SampleStatus.MEASURED, 0)

        missing_run = self._missing_run + 1
        if self._last_received is None:
            values = np.full(self.n_channels, np.nan)
            status = SampleStatus.UNAVAILABLE
        elif self._slope is None:
            values = self._last_received.copy()
            status = SampleStatus.RECONSTRUCTED
        else:
            horizon = index - self._last_received_index
            try:
                with np.errstate(over="raise", invalid="raise"):
                    values = self._last_received + horizon * self._slope
            except FloatingPointError as error:
                raise FloatingPointError("Non-finite extrapolation forecast") from error
            status = SampleStatus.RECONSTRUCTED
        self._index = index
        self._missing_run = missing_run
        return StepResult(values, status, missing_run)
