"""Synthetic acceptance tests; caps are explicit and no PAMAP2 data is read."""

import unittest

import numpy as np

from packetheal.baselines import SampleStatus
from packetheal.evaluation import replay_segment, score_hidden
from packetheal.linear_extrapolation import LinearExtrapolator


class LinearExtrapolatorTests(unittest.TestCase):
    def test_startup_and_single_measurement_fallback(self):
        predictor = LinearExtrapolator([5], n_channels=1)
        outputs = [predictor.step(value) for value in (None, None, [10], None, None)]
        self.assertTrue(np.isnan(outputs[0].values).all())
        self.assertTrue(np.isnan(outputs[1].values).all())
        np.testing.assert_array_equal([out.values for out in outputs[2:]], [[10], [10], [10]])
        self.assertEqual([out.status for out in outputs], [
            SampleStatus.UNAVAILABLE, SampleStatus.UNAVAILABLE, SampleStatus.MEASURED,
            SampleStatus.RECONSTRUCTED, SampleStatus.RECONSTRUCTED,
        ])
        self.assertEqual([out.missing_run for out in outputs], [1, 2, 0, 1, 2])

    def test_constant_rising_and_falling_signals_through_a_burst(self):
        for initial, delta in ((7, 0), (10, 2), (10, -2)):
            with self.subTest(initial=initial, delta=delta):
                predictor = LinearExtrapolator([3], n_channels=1)
                inputs = ([initial], [initial + delta], None, None, [initial + 4 * delta], None)
                outputs = [predictor.step(value) for value in inputs]
                np.testing.assert_array_equal(
                    [out.values for out in outputs], [[initial + index * delta] for index in range(6)]
                )
                self.assertEqual([out.missing_run for out in outputs], [0, 0, 1, 2, 0, 1])

    def test_two_received_samples_separated_by_a_loss_use_their_spacing(self):
        predictor = LinearExtrapolator([10], n_channels=1)
        outputs = [predictor.step(value) for value in ([10], None, [16], None)]
        np.testing.assert_array_equal([out.values for out in outputs], [[10], [10], [16], [19]])

    def test_recovery_uses_real_measurements_instead_of_previous_estimates(self):
        predictor = LinearExtrapolator([20], n_channels=1)
        outputs = [predictor.step(value) for value in ([0], [2], None, None, [12], None)]
        # Last real points are 2 at index 1 and 12 at index 4: slope = 10/3.
        np.testing.assert_allclose([out.values for out in outputs], [[0], [2], [4], [6], [12], [12 + 10 / 3]])

    def test_per_channel_slope_caps_include_negative_and_zero_slopes(self):
        predictor = LinearExtrapolator([2, 3, 0], n_channels=3)
        predictor.step([0, 0, 1])
        received = predictor.step([100, -100, 100])
        np.testing.assert_array_equal(received.values, [100, -100, 100])
        self.assertEqual(received.status, SampleStatus.MEASURED)
        np.testing.assert_array_equal(predictor.step(None).values, [102, -103, 100])
        np.testing.assert_array_equal(predictor.step(None).values, [104, -106, 100])

    def test_clipped_forecasts_do_not_replace_real_history(self):
        predictor = LinearExtrapolator([2], n_channels=1)
        outputs = [predictor.step(value) for value in ([0], [100], None, [101], None)]
        np.testing.assert_array_equal([out.values for out in outputs], [[0], [100], [102], [101], [101.5]])

    def test_reset_clears_slope_history_and_loss_age(self):
        predictor = LinearExtrapolator([5], n_channels=1)
        for value in ([1], [3], None, None):
            predictor.step(value)
        predictor.reset()
        startup = predictor.step(None)
        self.assertTrue(np.isnan(startup.values).all())
        self.assertEqual(startup.status, SampleStatus.UNAVAILABLE)
        self.assertEqual(startup.missing_run, 1)
        predictor.step([100])
        fallback = predictor.step(None)
        np.testing.assert_array_equal(fallback.values, [100])
        self.assertEqual(fallback.missing_run, 1)

    def test_long_outage_advances_horizon_without_silent_cutoff(self):
        predictor = LinearExtrapolator([2], n_channels=1)
        predictor.step([1])
        predictor.step([1.5])
        for count in range(1, 101):
            output = predictor.step(None)
            np.testing.assert_array_equal(output.values, [1.5 + count * 0.5])
            self.assertEqual(output.missing_run, count)
            self.assertEqual(output.status, SampleStatus.RECONSTRUCTED)

    def test_inputs_and_returned_values_do_not_alias_history(self):
        predictor = LinearExtrapolator([5, 5], n_channels=2)
        first = np.array([1., 2.])
        output = predictor.step(first)
        np.testing.assert_array_equal(first, [1, 2])
        first[:] = 99
        output.values[:] = 88
        second = np.array([3., 5.])
        measured = predictor.step(second)
        second[:] = 77
        measured.values[:] = 66
        reconstructed = predictor.step(None)
        np.testing.assert_array_equal(reconstructed.values, [5, 8])
        reconstructed.values[:] = 55
        np.testing.assert_array_equal(predictor.step(None).values, [7, 11])

    def test_limit_input_and_property_mutations_cannot_change_policy(self):
        limits = np.array([2.])
        predictor = LinearExtrapolator(limits, n_channels=1)
        limits[:] = 0
        predictor.max_delta_per_step[:] = 0
        predictor.step([0])
        predictor.step([100])
        np.testing.assert_array_equal(predictor.step(None).values, [102])
        np.testing.assert_array_equal(predictor.max_delta_per_step, [2])

    def test_invalid_limits_and_channel_counts(self):
        for count in (0, -1):
            with self.subTest(count=count), self.assertRaises(ValueError):
                LinearExtrapolator([1], n_channels=count)
        for count in (True, 1.5):
            with self.subTest(count=count), self.assertRaises(TypeError):
                LinearExtrapolator([1], n_channels=count)
        for limits in (1, [], [1, 2], [[1]], [-1], [np.nan], [np.inf], [1 + 1j]):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                LinearExtrapolator(limits, n_channels=1)

    def test_bad_measurements_do_not_advance_time_or_replace_history(self):
        predictor = LinearExtrapolator([10], n_channels=1)
        predictor.step([10])
        predictor.step([12])
        for value in (1, [], [1, 2], [[1]], [np.nan], [np.inf], [1 + 1j]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                predictor.step(value)
        np.testing.assert_array_equal(predictor.step(None).values, [14])
        predictor.step([18])
        np.testing.assert_array_equal(predictor.step(None).values, [21])

    def test_slope_overflow_fails_without_replacing_history_or_advancing(self):
        maximum = np.finfo(np.float64).max
        predictor = LinearExtrapolator([maximum], n_channels=1)
        predictor.step([-maximum])
        with self.assertRaisesRegex(FloatingPointError, "slope"):
            predictor.step([maximum])
        predictor.step([0])
        np.testing.assert_array_equal(predictor.step(None).values, [maximum])

    def test_forecast_overflow_fails_without_advancing_time(self):
        predictor = LinearExtrapolator([1e308], n_channels=1)
        predictor.step([1e308])
        predictor.step([1.5e308])
        with self.assertRaisesRegex(FloatingPointError, "forecast"):
            predictor.step(None)
        predictor.step([1e308])
        np.testing.assert_allclose(predictor.step(None).values, [5e307])


class LinearReplayTests(unittest.TestCase):
    def setUp(self):
        self.slopes = np.array([0.5, -0.25, 0, 1, -1, 2])
        self.truth = np.array([10, 1, 5, -20, 10, 0]) + np.arange(12)[:, None] * self.slopes
        self.missing = np.array([False, False, True, True, True, False, True, False, True, True, False, True])

    def predictor(self):
        return LinearExtrapolator([2] * 6)

    def test_six_channel_ramps_reconstruct_exactly_with_existing_replay(self):
        original = self.truth.copy()
        result = replay_segment(self.predictor(), self.truth, self.missing)
        np.testing.assert_array_equal(self.truth, original)
        np.testing.assert_array_equal(result.values, self.truth)
        self.assertTrue((result.status[self.missing] == "RECONSTRUCTED").all())
        self.assertTrue((result.status[~self.missing] == "MEASURED").all())
        metrics = score_hidden(self.truth, self.missing, result)
        self.assertEqual(metrics["mae"], [0] * 6)
        self.assertEqual(metrics["rmse"], [0] * 6)
        self.assertEqual(metrics["scored_samples"], int(self.missing.sum()))
        self.assertEqual(metrics["unavailable_samples"], 0)

    def test_hidden_measurements_cannot_change_predictions(self):
        reference = replay_segment(self.predictor(), self.truth, self.missing)
        poisoned = self.truth.copy()
        poisoned[self.missing] = np.nan
        actual = replay_segment(self.predictor(), poisoned, self.missing)
        for field in ("values", "status", "missing_run"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(reference, field))

    def test_future_measurements_cannot_change_earlier_predictions(self):
        reference = replay_segment(self.predictor(), self.truth, self.missing)
        future = self.truth.copy()
        future[5:] += 100
        actual = replay_segment(self.predictor(), future, self.missing)
        np.testing.assert_array_equal(actual.values[:5], reference.values[:5])
        np.testing.assert_array_equal(actual.values[5], future[5])

    def test_repeated_replay_is_identical_and_each_segment_resets(self):
        predictor = self.predictor()
        first = replay_segment(predictor, self.truth, self.missing)
        second = replay_segment(predictor, self.truth, self.missing)
        for field in ("values", "status", "missing_run"):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
        next_segment = replay_segment(predictor, self.truth[:2], np.array([True, False]))
        self.assertTrue(np.isnan(next_segment.values[0]).all())
        self.assertEqual(next_segment.status[0], "UNAVAILABLE")
        np.testing.assert_array_equal(next_segment.values[1], self.truth[1])


if __name__ == "__main__":
    unittest.main()
