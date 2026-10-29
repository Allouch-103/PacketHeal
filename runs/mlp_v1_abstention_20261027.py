"""Desktop-only abstention wrapper; frozen MLP and warning implementations unchanged."""

from copy import copy
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from packetheal.baselines import SampleStatus, StepResult
from packetheal.mlp import MLPPredictor
from mlp_v1_received_warning_20261027 import ReceivedChangeWarning


class AbstainingMLP:
    """step(real six-vector or None) -> existing StepResult; rejection is NaN.

    Real measurements always pass through. Missing samples are unavailable if
    the warning is unknown/on, the original predictor is unavailable, or the
    original model's supported loss horizon is exceeded. Both states advance on
    every successful step. Rejected predictions never become observations.

    This prototype computes the original prediction even when rejecting it;
    no inference-time optimization or calibrated confidence is claimed.
    """

    n_channels = 6

    def __init__(self, model, normalization, thresholds, history=20, horizon=10):
        self.predictor = MLPPredictor(model, normalization, history, horizon)
        self.warning = ReceivedChangeWarning(thresholds)
        self.horizon = horizon
        self.reason = "RESET"

    def reset(self):
        self.predictor.reset()
        self.warning.reset()
        self.reason = "RESET"

    def step(self, observation):
        # Warning.step replaces its small arrays instead of mutating them.
        # Stage a shallow state copy so either component's validation failure
        # cannot leave the warning ahead of the original predictor.
        candidate = copy(self.warning)
        warned, known, _, age, _, _ = candidate.step(observation)
        original = self.predictor.step(observation)
        assert age == original.missing_run
        if observation is not None:
            result, reason = original, "MEASURED"
        elif original.status == SampleStatus.UNAVAILABLE or not known:
            result, reason = StepResult(np.full(6, np.nan), SampleStatus.UNAVAILABLE, age), "INSUFFICIENT_HISTORY"
        elif age > self.horizon:
            result, reason = StepResult(np.full(6, np.nan), SampleStatus.UNAVAILABLE, age), "HORIZON_EXCEEDED"
        elif warned:
            result, reason = StepResult(np.full(6, np.nan), SampleStatus.UNAVAILABLE, age), "RECEIVED_CHANGE"
        else:
            result, reason = original, "ACCEPTED"
        self.warning, self.reason = candidate, reason
        return result
