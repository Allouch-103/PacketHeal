"""Synthetic-only tests. Baseline protocol fixture substitutes the unreadable repo.

The fixture tests public interface expectations; integration with actual repository
baselines and dataset loader MUST be repeated when the filesystem is readable.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from packetheal.mlp import Adam, CausalHistory, MLP, MLPPredictor, Normalization, segment_examples


class SampleStatus(str, Enum):
    MEASURED = "MEASURED"
    RECONSTRUCTED = "RECONSTRUCTED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class StepResult:
    values: np.ndarray
    status: SampleStatus
    missing_run: int


try:
    from packetheal.baselines import SampleStatus, StepResult
    PROTOCOL = sys.modules["packetheal.baselines"]
except ModuleNotFoundError as error:
    if error.name != "packetheal.baselines":
        raise
    PROTOCOL = types.ModuleType("packetheal.baselines")
    PROTOCOL.SampleStatus, PROTOCOL.StepResult = SampleStatus, StepResult


def replay(predictor, truth, missing):
    predictor.reset()
    values, statuses, ages = [], [], []
    for row, hidden in zip(truth, missing):
        result = predictor.step(None if hidden else row)
        values.append(result.values)
        statuses.append(result.status)
        ages.append(result.missing_run)
    return np.asarray(values), statuses, ages


class MLPTests(unittest.TestCase):
    def setUp(self):
        self.normalization = Normalization(np.zeros(6), np.ones(6))
        self.protocol_patch = patch.dict(sys.modules, {"packetheal.baselines": PROTOCOL})
        self.protocol_patch.start()
        self.addCleanup(self.protocol_patch.stop)

    def test_manual_example_history_targets_and_loss_age(self):
        truth = np.ones((25, 6))
        truth[21:25, 0] = [1.2, 1.3, 1.35, 1.5]
        missing = np.zeros(25, dtype=bool)
        missing[22:24] = True
        x, direct, residual, indices = segment_examples(truth, missing, self.normalization)
        np.testing.assert_array_equal(indices, [22, 23])
        self.assertEqual(x.shape, (2, 161))
        np.testing.assert_allclose(x[:, -1], [.1, .2])
        rows = x[:, :-1].reshape(2, 20, 8)
        np.testing.assert_allclose(rows[0, -1, :6], truth[21])
        np.testing.assert_allclose(rows[1, -1, :6], truth[21])
        np.testing.assert_array_equal(rows[:, -1, 6], [1, 0])
        np.testing.assert_array_equal(rows[:, :, 7], 1)
        np.testing.assert_allclose(residual[:, 0], [.1, .15], atol=1e-7)
        np.testing.assert_allclose(direct[:, 0], [1.3, 1.35])

    def test_hidden_truth_changes_targets_but_never_features(self):
        truth = np.arange(180, dtype=float).reshape(30, 6)
        mask = np.zeros(30, dtype=bool)
        mask[21:25] = True
        before = segment_examples(truth, mask, self.normalization)
        truth[mask] = -9999
        after = segment_examples(truth, mask, self.normalization)
        np.testing.assert_array_equal(before[0], after[0])
        self.assertFalse(np.array_equal(before[2], after[2]))

    def test_prepare_does_not_advance_and_padding_is_explicit(self):
        state = CausalHistory(self.normalization)
        first, eligible = state.prepare()
        np.testing.assert_array_equal(first[:-1], 0)
        self.assertFalse(eligible)
        self.assertEqual(state.elapsed, 0)
        state.commit(np.ones(6))
        second, eligible = state.prepare()
        np.testing.assert_array_equal(second[:-1].reshape(20, 8)[-1], np.ones(8))
        np.testing.assert_array_equal(second[:-1].reshape(20, 8)[:-1], 0)
        self.assertFalse(eligible)

    def test_future_values_masks_and_every_prefix_are_isolated(self):
        rng = np.random.default_rng(7)
        truth = rng.normal(size=(32, 6))
        mask = rng.random(32) < .3
        model = MLP(mode="direct", seed=3)
        predictor = MLPPredictor(model, self.normalization)
        original = replay(predictor, truth, mask)
        for cutoff in range(33):
            changed, changed_mask = truth.copy(), mask.copy()
            changed[cutoff:] += 1000
            changed_mask[cutoff:] = ~changed_mask[cutoff:]
            actual = replay(predictor, changed, changed_mask)
            np.testing.assert_array_equal(original[0][:cutoff], actual[0][:cutoff])
            self.assertEqual(original[1][:cutoff], actual[1][:cutoff])
            self.assertEqual(original[2][:cutoff], actual[2][:cutoff])

    def test_zero_residual_exactly_equals_locf_including_startup(self):
        truth = np.arange(240, dtype=float).reshape(40, 6) / 10
        mask = np.zeros(40, dtype=bool)
        mask[:2] = True
        mask[22:37] = True
        output, statuses, ages = replay(MLPPredictor(MLP(), self.normalization), truth, mask)
        anchor = None
        age = 0
        for t in range(len(truth)):
            if mask[t]:
                age += 1
                self.assertEqual(ages[t], age)
                if anchor is None:
                    self.assertTrue(np.isnan(output[t]).all())
                    self.assertEqual(statuses[t], SampleStatus.UNAVAILABLE)
                else:
                    np.testing.assert_array_equal(output[t], anchor)
                    self.assertEqual(statuses[t], SampleStatus.RECONSTRUCTED)
            else:
                anchor, age = truth[t], 0
                np.testing.assert_array_equal(output[t], anchor)

    def test_residual_denormalization_and_no_recursive_feedback(self):
        norm = Normalization(np.arange(6) + 10, np.arange(6) + 2)
        model = MLP()
        model.parameters[-1][:] = .25
        predictor = MLPPredictor(model, norm)
        anchor = np.arange(6) + 1.123456789
        for _ in range(20):
            predictor.step(anchor)
        expected = anchor + .25 * norm.std
        np.testing.assert_array_equal(predictor.step(None).values, expected)
        np.testing.assert_array_equal(predictor.step(None).values, expected)
        np.testing.assert_array_equal(predictor.state.anchor, anchor)
        np.testing.assert_allclose(predictor.state.rows[-1, :6], norm.encode(anchor))
        returned = predictor.step(anchor)
        returned.values[:] = 999
        np.testing.assert_array_equal(predictor.state.anchor, anchor)

    def test_zero_residual_preserves_locf_bytes_including_signed_zero(self):
        anchor = np.array([-0., 0., -0., 1., -2., 3.], dtype=np.float64)
        predictor = MLPPredictor(MLP(), self.normalization)
        for _ in range(20):
            predictor.step(anchor)
        self.assertEqual(predictor.step(None).values.tobytes(), anchor.tobytes())

    def test_direct_denormalization_and_long_loss_fallback(self):
        norm = Normalization(np.arange(6), np.arange(6) + 2)
        model = MLP(mode="direct")
        for p in model.parameters:
            p.fill(0)
        model.parameters[-1][:] = .5
        predictor = MLPPredictor(model, norm)
        for _ in range(20):
            predictor.step(np.ones(6) * 7)
        for age in range(1, 13):
            output = predictor.step(None)
            np.testing.assert_array_equal(output.values, norm.mean + .5 * norm.std if age <= 10 else np.ones(6) * 7)
            self.assertEqual(output.missing_run, age)

    def test_reset_a_b_a_and_hidden_nan_poisoning(self):
        truth = np.arange(210.).reshape(35, 6)
        mask = np.zeros(35, dtype=bool)
        mask[22:27] = True
        predictor = MLPPredictor(MLP(mode="direct"), self.normalization)
        first = replay(predictor, truth, mask)
        replay(predictor, truth + 50, ~mask)
        second = replay(predictor, truth, mask)
        poisoned = truth.copy()
        poisoned[mask] = np.nan
        third = replay(predictor, poisoned, mask)
        for actual in (second, third):
            np.testing.assert_array_equal(first[0], actual[0])
            self.assertEqual(first[1:], actual[1:])

    def test_invalid_observation_is_atomic(self):
        predictor = MLPPredictor(MLP(), self.normalization)
        predictor.step(np.ones(6))
        before = predictor.state.rows.copy()
        for invalid in ([1, 2], [np.nan] * 6, [np.inf] * 6):
            with self.assertRaises(ValueError):
                predictor.step(invalid)
            self.assertEqual(predictor.state.elapsed, 1)
            np.testing.assert_array_equal(before, predictor.state.rows)

    def test_gradients_match_finite_differences_for_every_parameter(self):
        model = MLP(input_size=3, widths=(3, 2), mode="direct", seed=9)
        model.parameters[1][:] = 2
        model.parameters[3][:] = 2
        x = np.array([[.3, .5, .7], [.2, -.1, .8]], dtype=np.float32)
        y = np.ones((2, 6), dtype=np.float32) * 5
        weights = np.array([1., 3.])
        _, gradients = model.loss_and_gradients(x, y, weights)
        for p, gradient in zip(model.parameters, gradients):
            for index in np.ndindex(p.shape):
                original, epsilon = float(p[index]), .002
                p[index] = original + epsilon
                plus = model.loss_and_gradients(x, y, weights)[0]
                p[index] = original - epsilon
                minus = model.loss_and_gradients(x, y, weights)[0]
                p[index] = original
                self.assertAlmostEqual(float(gradient[index]), (plus - minus) / (2 * epsilon), delta=.002)

    def test_adam_first_update_matches_formula(self):
        parameter = np.array([1., 2.], dtype=np.float32)
        optimizer = Adam([parameter], learning_rate=.01)
        optimizer.step([np.array([2., -3.], dtype=np.float32)])
        np.testing.assert_allclose(parameter, [.99, 2.01], atol=1e-6)
        self.assertEqual(optimizer.steps, 1)

    def test_weighted_mae_hand_calculation(self):
        model = MLP(input_size=2)
        loss, _ = model.loss_and_gradients(np.zeros((2, 2)), np.array([[1] * 6, [3] * 6]), [1, 3])
        self.assertEqual(loss, 2.5)

    def test_tiny_fixture_can_be_learned(self):
        model = MLP(input_size=3, widths=(8, 4), seed=5)
        x = np.random.default_rng(3).normal(size=(16, 3)).astype(np.float32)
        y = np.broadcast_to(np.array([.1, -.2, .3, -.4, .5, -.6]), (16, 6)).astype(np.float32)
        optimizer = Adam(model.parameters, .01)
        before = model.loss_and_gradients(x, y)[0]
        for _ in range(400):
            _, gradients = model.loss_and_gradients(x, y)
            optimizer.step(gradients)
        after = model.loss_and_gradients(x, y)[0]
        self.assertLess(after, before * .05)

    def test_checkpoint_roundtrip_and_overwrite_refusal(self):
        model = MLP(mode="direct", seed=12)
        x = np.random.default_rng(3).normal(size=(4, 161))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.npz"
            model.save(path)
            np.testing.assert_array_equal(model.predict(x), MLP.load(path).predict(x))
            with self.assertRaises(FileExistsError):
                model.save(path)

    def test_training_batch_predictions_match_streaming(self):
        truth = np.random.default_rng(5).normal(size=(40, 6))
        mask = np.zeros(40, dtype=bool)
        mask[22:27] = True
        features, _, _, indices = segment_examples(truth, mask, self.normalization)
        model = MLP(mode="direct")
        actual = replay(MLPPredictor(model, self.normalization), truth, mask)[0]
        np.testing.assert_allclose(actual[indices], model.predict(features), rtol=1e-5, atol=1e-6)

    def test_parameter_count_and_determinism(self):
        model, repeated = MLP(seed=7), MLP(seed=7)
        self.assertEqual(sum(p.size for p in model.parameters), 5814)
        for p, q in zip(model.parameters, repeated.parameters):
            np.testing.assert_array_equal(p, q)

    def test_normalization_validation_and_ownership(self):
        mean, std = np.zeros(6), np.ones(6)
        norm = Normalization(mean, std)
        mean[:] = 99
        std[:] = 0
        np.testing.assert_array_equal(norm.mean, 0)
        np.testing.assert_array_equal(norm.std, 1)
        with self.assertRaises(ValueError):
            Normalization(np.zeros(6), np.zeros(6))


if __name__ == "__main__":
    unittest.main()
