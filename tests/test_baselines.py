"""Small, inspectable streaming examples; no dataset files are needed."""

import unittest

import numpy as np

from packetheal.baselines import LOCF, SampleStatus


class LOCFTests(unittest.TestCase):
    def test_received_isolated_burst_and_recovery(self):
        predictor = LOCF(2)
        observations = ([10, 100], None, [20, 200], None, None, [30, 300])
        results = [predictor.step(value) for value in observations]
        np.testing.assert_array_equal(
            [r.values for r in results],
            [[10, 100], [10, 100], [20, 200], [20, 200], [20, 200], [30, 300]],
        )
        self.assertEqual([r.missing_run for r in results], [0, 1, 0, 1, 2, 0])
        self.assertEqual(
            [r.status for r in results],
            [SampleStatus.MEASURED, SampleStatus.RECONSTRUCTED,
             SampleStatus.MEASURED, SampleStatus.RECONSTRUCTED,
             SampleStatus.RECONSTRUCTED, SampleStatus.MEASURED],
        )

    def test_startup_and_reset_have_no_measurement(self):
        predictor = LOCF(2)
        for count in (1, 2):
            result = predictor.step(None)
            self.assertTrue(np.isnan(result.values).all())
            self.assertEqual(result.status, SampleStatus.UNAVAILABLE)
            self.assertEqual(result.missing_run, count)
        predictor.step([7, 8])
        predictor.reset()
        result = predictor.step(None)
        self.assertEqual(result.status, SampleStatus.UNAVAILABLE)
        self.assertEqual(result.missing_run, 1)

    def test_long_outage_keeps_last_real_value_and_loss_age(self):
        predictor = LOCF(1)
        predictor.step([8])
        for count in range(1, 101):
            result = predictor.step(None)
            np.testing.assert_array_equal(result.values, [8])
            self.assertEqual(result.missing_run, count)
            self.assertEqual(result.status, SampleStatus.RECONSTRUCTED)

    def test_input_and_output_mutations_do_not_change_stored_history(self):
        predictor = LOCF(2)
        received = np.array([1.0, 2.0])
        measured = predictor.step(received)
        np.testing.assert_array_equal(received, [1, 2])
        received[:] = 99
        measured.values[:] = 88
        reconstructed = predictor.step(None)
        np.testing.assert_array_equal(reconstructed.values, [1, 2])
        reconstructed.values[:] = 77
        np.testing.assert_array_equal(predictor.step(None).values, [1, 2])

    def test_bad_received_values_are_rejected_without_replacing_history(self):
        predictor = LOCF(2)
        predictor.step([1, 2])
        for value in ([1], [[1, 2]], [np.nan, 1], [1, np.inf]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                predictor.step(value)
        np.testing.assert_array_equal(predictor.step(None).values, [1, 2])

    def test_invalid_channel_counts(self):
        for count in (0, -1):
            with self.subTest(count=count), self.assertRaises(ValueError):
                LOCF(count)
        for count in (True, 1.5):
            with self.subTest(count=count), self.assertRaises(TypeError):
                LOCF(count)


if __name__ == "__main__":
    unittest.main()
