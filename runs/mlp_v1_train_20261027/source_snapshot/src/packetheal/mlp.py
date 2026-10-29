"""Small causal MLP: observation history, NumPy regression, and streaming wrapper.

Core module has no dataset I/O. Missing truth is only a supervised target;
neither predictions nor hidden measurements enter observation history.
"""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np


@dataclass(frozen=True)
class Normalization:
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self):
        mean = np.array(self.mean, dtype=np.float64, copy=True)
        std = np.array(self.std, dtype=np.float64, copy=True)
        if mean.shape != (6,) or std.shape != (6,):
            raise ValueError("Expected six normalization means and standard deviations")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Normalization must be finite with positive standard deviations")
        mean.flags.writeable = std.flags.writeable = False
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    def encode(self, values):
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std


class CausalHistory:
    """20 past slots x (six normalized values, received, slot-valid), plus age.

    prepare() only reads state. commit() advances it. Thus the current hidden
    value cannot enter its own input, and a failed prediction need not mutate
    state. Only real observations update the LOCF anchor.
    """

    def __init__(self, normalization, history=20, horizon=10):
        if type(history) is not int or history < 1 or type(horizon) is not int or horizon < 1:
            raise ValueError("history and horizon must be positive integers")
        self.normalization = normalization
        self.history = history
        self.horizon = horizon
        self.input_size = history * 8 + 1
        self.reset()

    def reset(self):
        self.rows = np.zeros((self.history, 8), dtype=np.float64)
        self.anchor = None
        self.elapsed = 0
        self.age = 0

    def validate(self, observation):
        if observation is None:
            return None
        values = np.array(observation, dtype=np.float64, copy=True)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("Observation must contain six finite physical-unit values")
        encoded = self.normalization.encode(values)
        if not np.isfinite(encoded).all() or np.max(np.abs(encoded)) > np.finfo(np.float32).max:
            raise ValueError("Observation cannot be represented in model features")
        return values

    def prepare(self):
        """Features for a missing current sample; age 1 is the first loss."""
        age = self.age + 1
        features = np.concatenate((self.rows.reshape(-1), [min(age, self.horizon) / self.horizon]))
        eligible = self.anchor is not None and self.elapsed >= self.history and age <= self.horizon
        return features.astype(np.float32), eligible

    def commit(self, observation):
        values = self.validate(observation)
        if values is None:
            anchor, age = self.anchor, self.age + 1
        else:
            anchor, age = values, 0
        row = np.zeros(8, dtype=np.float64)
        if anchor is not None:
            row[:6] = self.normalization.encode(anchor)
        row[6] = float(values is not None)
        row[7] = 1.0
        self.rows[:-1] = self.rows[1:].copy()
        self.rows[-1] = row
        self.anchor = None if anchor is None else anchor.copy()
        self.age, self.elapsed = age, self.elapsed + 1


class MLP:
    """Two ReLU hidden layers, linear six-output regression, float32 weights."""

    def __init__(self, input_size=161, widths=(32, 16), mode="residual", seed=0):
        if mode not in ("direct", "residual"):
            raise ValueError("mode must be direct or residual")
        if type(input_size) is not int or input_size < 1 or len(widths) != 2:
            raise ValueError("Expected input size and two hidden widths")
        if any(type(n) is not int or n < 1 for n in widths):
            raise ValueError("Hidden widths must be positive integers")
        self.mode, self.input_size, self.widths = mode, input_size, tuple(widths)
        rng = np.random.default_rng(seed)
        sizes = (input_size, *widths, 6)
        self.parameters = []
        for left, right in zip(sizes[:-1], sizes[1:]):
            self.parameters.extend([
                (rng.standard_normal((left, right)) * np.sqrt(2 / left)).astype(np.float32),
                np.zeros(right, dtype=np.float32),
            ])
        if mode == "residual":
            self.parameters[-2].fill(0)

    def _forward(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.input_size or not np.isfinite(x).all():
            raise ValueError("Expected a finite [batch, input_size] feature array")
        w1, b1, w2, b2, w3, b3 = self.parameters
        with np.errstate(over="raise", invalid="raise"):
            h1 = np.maximum(x @ w1 + b1, 0)
            h2 = np.maximum(h1 @ w2 + b2, 0)
            output = h2 @ w3 + b3
        return output, (x, h1, h2)

    def predict(self, features):
        return self._forward(features)[0]

    def loss_and_gradients(self, features, targets, weights=None):
        """Weighted normalized MAE. Explicit derivatives permit gradient tests."""
        output, (x, h1, h2) = self._forward(features)
        target = np.asarray(targets, dtype=np.float32)
        if target.shape != output.shape or len(target) == 0 or not np.isfinite(target).all():
            raise ValueError("Expected finite, nonempty [batch, 6] targets")
        weights = np.ones(len(x), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
        if weights.shape != (len(x),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("Invalid example weights")
        weights = weights / weights.sum()
        error = output.astype(np.float64) - target
        loss = float(np.sum(np.abs(error) * weights[:, None]) / 6)
        delta = (np.sign(error) * weights[:, None] / 6).astype(np.float32)
        w1, b1, w2, b2, w3, b3 = self.parameters
        dw3, db3 = h2.T @ delta, delta.sum(axis=0)
        delta = (delta @ w3.T) * (h2 > 0)
        dw2, db2 = h1.T @ delta, delta.sum(axis=0)
        delta = (delta @ w2.T) * (h1 > 0)
        dw1, db1 = x.T @ delta, delta.sum(axis=0)
        gradients = [dw1, db1, dw2, db2, dw3, db3]
        if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Non-finite loss or gradient")
        return loss, gradients

    def save(self, path):
        metadata = json.dumps({"mode": self.mode, "input_size": self.input_size, "widths": self.widths})
        with Path(path).open("xb") as stream:
            np.savez_compressed(stream, metadata=np.array(metadata), **{f"p{i}": p for i, p in enumerate(self.parameters)})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            result = cls(**metadata)
            for i, parameter in enumerate(result.parameters):
                value = archive[f"p{i}"]
                if value.shape != parameter.shape or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError("Invalid checkpoint tensor")
                parameter[:] = value
        return result


class Adam:
    """Adam optimizer; moments are reset for every independent model run."""

    def __init__(self, parameters, learning_rate=1e-3):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        self.parameters, self.learning_rate = parameters, learning_rate
        self.m = [np.zeros_like(p) for p in parameters]
        self.v = [np.zeros_like(p) for p in parameters]
        self.steps = 0

    def step(self, gradients):
        if len(gradients) != len(self.parameters) or any(g.shape != p.shape or not np.isfinite(g).all() for g, p in zip(gradients, self.parameters)):
            raise ValueError("Invalid gradients")
        step = self.steps + 1
        moments, variances, updated = [], [], []
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for p, m, v, g in zip(self.parameters, self.m, self.v, gradients):
                new_m = .9 * m + .1 * g
                new_v = .999 * v + .001 * g * g
                update = self.learning_rate * (new_m / (1 - .9 ** step)) / (np.sqrt(new_v / (1 - .999 ** step)) + 1e-8)
                value = p - update
                if not np.isfinite(value).all():
                    raise FloatingPointError("Non-finite Adam update")
                moments.append(new_m)
                variances.append(new_v)
                updated.append(value)
        for i, p in enumerate(self.parameters):
            p[:] = updated[i]
            self.m[i][:], self.v[i][:] = moments[i], variances[i]
        self.steps = step


class MLPPredictor:
    """Compatible with reset()/step(vector_or_None) and existing StepResult."""

    def __init__(self, model, normalization, history=20, horizon=10):
        self.model = model
        self.state = CausalHistory(normalization, history, horizon)
        if model.input_size != self.state.input_size:
            raise ValueError("Network input size does not match feature contract")

    def reset(self):
        self.state.reset()

    def step(self, observation):
        # Import only for streaming use: the math and feature builder are standalone.
        from .baselines import SampleStatus, StepResult
        received = self.state.validate(observation)
        if received is not None:
            result = StepResult(received.copy(), SampleStatus.MEASURED, 0)
        elif self.state.anchor is None:
            result = StepResult(np.full(6, np.nan), SampleStatus.UNAVAILABLE, self.state.age + 1)
        else:
            features, eligible = self.state.prepare()
            values = self.state.anchor.copy()
            if eligible:
                output = self.model.predict(features[None])[0].astype(np.float64)
                if self.model.mode == "residual":
                    # Preserve the anchor's exact bytes, including signed zero,
                    # on channels where the model supplies no correction.
                    changed = output != 0
                    values[changed] += output[changed] * self.state.normalization.std[changed]
                else:
                    values = self.state.normalization.mean + output * self.state.normalization.std
            if not np.isfinite(values).all():
                raise FloatingPointError("Non-finite reconstructed output")
            result = StepResult(values, SampleStatus.RECONSTRUCTED, self.state.age + 1)
        self.state.commit(received)
        return result


def segment_examples(truth, missing, normalization, history=20, horizon=10):
    """Supervised examples: hidden truth is read only AFTER causal features exist."""
    truth = np.asarray(truth, dtype=np.float64)
    missing = np.asarray(missing)
    if truth.ndim != 2 or truth.shape[1] != 6 or missing.dtype != np.bool_ or missing.shape != (len(truth),):
        raise ValueError("Expected [time, 6] truth and boolean [time] missing mask")
    if not np.isfinite(truth).all():
        raise ValueError("Supervised targets must be finite")
    state = CausalHistory(normalization, history, horizon)
    features, direct, residual, indices = [], [], [], []
    for t in range(len(truth)):
        if missing[t]:
            inputs, eligible = state.prepare()
            if eligible:
                features.append(inputs)
                direct.append(normalization.encode(truth[t]))
                residual.append((truth[t] - state.anchor) / normalization.std)
                indices.append(t)
            state.commit(None)
        else:
            state.commit(truth[t])
    return (
        np.asarray(features, dtype=np.float32).reshape(-1, state.input_size),
        np.asarray(direct, dtype=np.float32).reshape(-1, 6),
        np.asarray(residual, dtype=np.float32).reshape(-1, 6),
        np.asarray(indices, dtype=np.int64),
    )
