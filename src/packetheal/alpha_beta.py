"""Fixed-gain causal value/trend prediction at nominal sample-step indices."""

from numbers import Real

import numpy as np

from .baselines import SampleStatus, StepResult


class AlphaBetaPredictor:
    """Correct a value/trend state only when a real observation arrives.

    At a received index, with ``elapsed`` steps since the last correction:
    ``predicted = value + elapsed * trend``;
    ``value_new = predicted + alpha * (observation - predicted)``;
    ``trend_new = trend + beta * (observation - predicted) / elapsed``.
    The trend is then clipped to the supplied per-channel limits. Equivalent
    convex combinations are used below to avoid unnecessary cancellation.

    Scalar gains are explicit, dimensionless, and shared by all channels:
    0 < alpha <= 1 and 0 <= beta <= 1. These ranges define this implementation's
    policy; they are not a general stability/accuracy guarantee for any signal.
    Limits use physical channel units per nominal sample step, not per second.

    The first observation initializes value to the received vector and trend
    to zero. Before initialization, losses are UNAVAILABLE; with one real
    sample, the zero trend gives LOCF fallback. Losses advance only time/age:
    they do not perform a measurement correction or become received history.
    Received outputs are always the original measurement, even when alpha < 1
    makes the internal value differ from it. Reset every segment/regime.

    There is no outage cutoff, output bound, or confidence estimate. Invalid
    input or non-finite arithmetic raises without advancing any state.
    """

    def __init__(self, alpha, beta, max_delta_per_step, n_channels=6):
        if isinstance(n_channels, bool) or not isinstance(n_channels, int):
            raise TypeError("n_channels must be an integer")
        if n_channels < 1:
            raise ValueError("n_channels must be positive")
        self.n_channels = n_channels
        self._alpha = self._gain(alpha, "alpha", exclude_zero=True)
        self._beta = self._gain(beta, "beta")
        limits = self._finite_vector(max_delta_per_step, "max_delta_per_step")
        if (limits < 0).any():
            raise ValueError("max_delta_per_step must be nonnegative")
        self._limits = limits.copy()
        self.reset()

    @staticmethod
    def _gain(value, name, *, exclude_zero=False):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a real scalar gain")
        if not np.isfinite(value) or not 0 <= value <= 1 or (exclude_zero and value == 0):
            interval = "(0, 1]" if exclude_zero else "[0, 1]"
            raise ValueError(f"{name} must be finite and in {interval}")
        return float(value)

    def _finite_vector(self, value, name):
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must contain real values")
        values = np.asarray(value, dtype=np.float64)
        if values.shape != (self.n_channels,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a finite vector of shape ({self.n_channels},)")
        return values

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def max_delta_per_step(self):
        return self._limits.copy()

    def reset(self):
        self._index = -1
        self._last_update_index = None
        self._value = None
        self._trend = None
        self._missing_run = 0

    def step(self, observation):
        index = self._index + 1
        if observation is not None:
            values = self._finite_vector(observation, "Received sample")
            if self._value is None:
                corrected, trend = values.copy(), np.zeros(self.n_channels)
            else:
                elapsed = index - self._last_update_index
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        # Endpoints avoid unused forecasts/differences and make
                        # alpha=1,beta=0 LOCF; alpha=beta=1 capped linear.
                        if self.alpha == 1:
                            corrected = values.copy()
                        else:
                            predicted = self._value + elapsed * self._trend
                            corrected = (1 - self.alpha) * predicted + self.alpha * values
                        if self.beta == 0:
                            trend = self._trend.copy()
                        else:
                            target = (values - self._value) / elapsed
                            trend = (1 - self.beta) * self._trend + self.beta * target
                            trend = np.clip(trend, -self._limits, self._limits)
                        if not np.isfinite(corrected).all() or not np.isfinite(trend).all():
                            raise FloatingPointError("Non-finite correction")
                except FloatingPointError as error:
                    raise FloatingPointError("Non-finite alpha-beta update") from error
            self._value, self._trend = corrected, trend
            self._last_update_index = index
            self._index, self._missing_run = index, 0
            return StepResult(values.copy(), SampleStatus.MEASURED, 0)

        missing_run = self._missing_run + 1
        if self._value is None:
            forecast = np.full(self.n_channels, np.nan)
            status = SampleStatus.UNAVAILABLE
        else:
            try:
                with np.errstate(over="raise", invalid="raise"):
                    forecast = self._value + (index - self._last_update_index) * self._trend
                    if not np.isfinite(forecast).all():
                        raise FloatingPointError("Non-finite forecast")
            except FloatingPointError as error:
                raise FloatingPointError("Non-finite alpha-beta forecast") from error
            status = SampleStatus.RECONSTRUCTED
        self._index, self._missing_run = index, missing_run
        return StepResult(forecast, status, missing_run)
