"""Synthetic correctness, baseline reductions, and causal replay contracts."""

import unittest

import numpy as np

from packetheal.alpha_beta import AlphaBetaPredictor
from packetheal.baselines import LOCF, SampleStatus
from packetheal.evaluation import replay_segment, score_hidden
from packetheal.linear_extrapolation import LinearExtrapolator


class AlphaBetaTests(unittest.TestCase):
    def predictor(self, alpha=.5, beta=.25, caps=None, channels=1):
        return AlphaBetaPredictor(alpha, beta, [10] * channels if caps is None else caps,
                                  n_channels=channels)

    def test_startup_and_single_real_sample_fallback(self):
        predictor = self.predictor()
        results = [predictor.step(v) for v in (None, None, [10], None, None)]
        self.assertEqual([r.status for r in results], [SampleStatus.UNAVAILABLE] * 2 +
                         [SampleStatus.MEASURED] + [SampleStatus.RECONSTRUCTED] * 2)
        self.assertTrue(np.isnan(results[0].values).all())
        self.assertTrue(np.isnan(results[1].values).all())
        np.testing.assert_array_equal([r.values for r in results[2:]], [[10], [10], [10]])
        self.assertEqual([r.missing_run for r in results], [1, 2, 0, 1, 2])

    def test_known_correction_preserves_received_output(self):
        predictor = self.predictor()
        predictor.step([0])
        received = predictor.step([4])
        # Predicted=0, error=4: corrected value=2, trend=1.
        np.testing.assert_array_equal(received.values, [4])
        self.assertEqual(received.status, SampleStatus.MEASURED)
        np.testing.assert_array_equal(predictor.step(None).values, [3])
        np.testing.assert_array_equal(predictor.step(None).values, [4])

    def test_gap_recovery_uses_elapsed_steps_and_last_corrected_anchor(self):
        predictor = self.predictor()
        for observation in ([0], [4], None, None):
            predictor.step(observation)
        predictor.step([8])
        # elapsed=3, predicted=5, residual=3: value=6.5, trend=1.25.
        np.testing.assert_array_equal(predictor.step(None).values, [7.75])
        np.testing.assert_array_equal(predictor.step(None).values, [9])

    def test_gain_one_zero_reduces_to_locf_including_startup(self):
        truth = np.arange(30, dtype=float).reshape(10, 3)
        missing = np.array([True, False, True, True, False, True, False, False, True, True])
        actual = replay_segment(self.predictor(1, 0, channels=3), truth, missing)
        expected = replay_segment(LOCF(3), truth, missing)
        for field in ("values", "status", "missing_run"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))

    def test_gain_one_one_reduces_to_capped_linear_through_recovery(self):
        truth = np.array([[0, 5], [2, -2], [9, 4], [1, -10], [12, 4],
                          [18, -20], [21, 10], [100, -100]], dtype=float)
        missing = np.array([False, False, True, True, False, True, False, True])
        actual = replay_segment(self.predictor(1, 1, [2, 3], 2), truth, missing)
        expected = replay_segment(LinearExtrapolator([2, 3], 2), truth, missing)
        np.testing.assert_allclose(actual.values, expected.values, rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(actual.status, expected.status)
        np.testing.assert_array_equal(actual.missing_run, expected.missing_run)

    def test_constant_and_signed_ramps_with_exact_gain_endpoints(self):
        for delta in (0., 2., -2.):
            truth = 7 + np.arange(12)[:, None] * delta
            mask = np.array([False, False, True, True, False, True] * 2)
            result = replay_segment(self.predictor(1, 1), truth, mask)
            np.testing.assert_array_equal(result.values, truth)
            self.assertEqual(score_hidden(truth, mask, result)["mae"], [0])

    def test_trend_caps_are_per_channel_and_do_not_clip_received_values(self):
        predictor = self.predictor(.5, .5, [1, 2, 0], 3)
        predictor.step([0, 0, 0])
        np.testing.assert_array_equal(predictor.step([100, -100, 100]).values, [100, -100, 100])
        # Corrected state=[50,-50,50], clipped trend=[1,-2,0].
        np.testing.assert_array_equal(predictor.step(None).values, [51, -52, 50])

    def test_zero_caps_and_beta_zero_have_distinct_value_smoothing(self):
        for caps, beta in (([0], .5), ([10], 0)):
            predictor = self.predictor(.5, beta, caps)
            predictor.step([0])
            predictor.step([8])
            np.testing.assert_array_equal(predictor.step(None).values, [4])

    def test_long_outage_keeps_anchor_and_trend_without_measurement_feedback(self):
        predictor = self.predictor()
        predictor.step([0])
        predictor.step([4])
        for horizon in range(1, 101):
            result = predictor.step(None)
            np.testing.assert_array_equal(result.values, [2 + horizon])
            self.assertEqual(result.missing_run, horizon)

    def test_reset_forgets_filtered_state_trend_and_clock(self):
        predictor = self.predictor()
        for value in ([0], [4], None):
            predictor.step(value)
        predictor.reset()
        result = predictor.step(None)
        self.assertEqual(result.status, SampleStatus.UNAVAILABLE)
        self.assertEqual(result.missing_run, 1)
        predictor.step([99])
        np.testing.assert_array_equal(predictor.step(None).values, [99])

    def test_array_ownership_and_read_only_gain_properties(self):
        caps = np.array([10.])
        predictor = self.predictor(caps=caps)
        first = np.array([0.])
        output = predictor.step(first)
        first[:] = 20
        output.values[:] = 30
        second = np.array([4.])
        measured = predictor.step(second)
        second[:] = 40
        measured.values[:] = 50
        caps[:] = 0
        predictor.max_delta_per_step[:] = 0
        estimated = predictor.step(None)
        np.testing.assert_array_equal(estimated.values, [3])
        estimated.values[:] = 100
        np.testing.assert_array_equal(predictor.step(None).values, [4])
        with self.assertRaises(AttributeError):
            predictor.alpha = 1
        with self.assertRaises(AttributeError):
            predictor.beta = 1

    def test_bad_gains_limits_and_channel_counts(self):
        for alpha in (0, -1, 1.1, np.nan, np.inf, True, [0.5], "0.5", 1j):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                self.predictor(alpha=alpha)
        for beta in (-1, 1.1, np.nan, np.inf, True, [0.5], "0.5", 1j):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                self.predictor(beta=beta)
        for limits in (1, [], [1, 2], [[1]], [-1], [np.nan], [np.inf], [1j]):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                self.predictor(caps=limits)
        for channels in (True, 1.5):
            with self.assertRaises(TypeError):
                self.predictor(channels=channels)
        for channels in (0, -1):
            with self.assertRaises(ValueError):
                self.predictor(channels=channels)

    def test_bad_observations_fail_without_advancing_state(self):
        predictor = self.predictor()
        predictor.step([0])
        predictor.step([4])
        for observation in (1, [], [1, 2], [[1]], [np.nan], [np.inf], [1j]):
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                predictor.step(observation)
        np.testing.assert_array_equal(predictor.step(None).values, [3])
        predictor.step([6])
        # elapsed=2, predicted=4, residual=2 => state=5, trend=1.25.
        np.testing.assert_array_equal(predictor.step(None).values, [6.25])

    def test_update_overflow_fails_without_advancing_or_replacing_state(self):
        maximum = np.finfo(float).max
        predictor = self.predictor(1, 1, [maximum])
        predictor.step([-maximum])
        with self.assertRaisesRegex(FloatingPointError, "update"):
            predictor.step([maximum])
        np.testing.assert_array_equal(predictor.step(None).values, [-maximum])
        predictor.step([0])
        np.testing.assert_allclose(predictor.step(None).values, [maximum / 2])

    def test_forecast_overflow_fails_without_advancing_time(self):
        predictor = self.predictor(1, 1, [1e308])
        predictor.step([1e308])
        predictor.step([1.5e308])
        with self.assertRaisesRegex(FloatingPointError, "forecast"):
            predictor.step(None)
        predictor.step([1e308])
        np.testing.assert_allclose(predictor.step(None).values, [5e307])

    def test_locf_endpoint_does_not_compute_unused_overflowing_difference(self):
        maximum = np.finfo(float).max
        predictor = self.predictor(1, 0, [maximum])
        predictor.step([-maximum])
        predictor.step([maximum])
        np.testing.assert_array_equal(predictor.step(None).values, [maximum])


class AlphaBetaReplayTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.arange(72, dtype=float).reshape(12, 6) / 7
        self.mask = np.array([False, False, True, True, True, False, True, False, True, True, False, True])

    def predictor(self):
        return AlphaBetaPredictor(.5, .25, [2] * 6)

    def test_hidden_truth_and_future_independence(self):
        first = replay_segment(self.predictor(), self.truth, self.mask)
        changed = self.truth.copy()
        changed[self.mask] = np.nan
        second = replay_segment(self.predictor(), changed, self.mask)
        for field in ("values", "status", "missing_run"):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
        future = self.truth.copy()
        future[6:] += 100
        altered = replay_segment(self.predictor(), future, self.mask)
        np.testing.assert_array_equal(first.values[:6], altered.values[:6])

    def test_received_preservation_repeatability_and_segment_reset(self):
        predictor = self.predictor()
        original = self.truth.copy()
        first = replay_segment(predictor, self.truth, self.mask)
        second = replay_segment(predictor, self.truth, self.mask)
        for field in ("values", "status", "missing_run"):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
        np.testing.assert_array_equal(self.truth, original)
        np.testing.assert_array_equal(first.values[~self.mask], self.truth[~self.mask])
        metrics = score_hidden(self.truth, self.mask, first)
        self.assertEqual(metrics["scored_samples"], int(self.mask.sum()))
        self.assertEqual(metrics["unavailable_samples"], 0)
        result = replay_segment(predictor, self.truth[:2], np.array([True, False]))
        self.assertEqual(result.status[0], "UNAVAILABLE")
        np.testing.assert_array_equal(result.values[1], self.truth[1])


if __name__ == "__main__":
    unittest.main()
