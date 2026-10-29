"""Synthetic tests only; never opens dataset recordings."""

import unittest
import numpy as np
from mlp_v1_received_warning_20261027 import ReceivedChangeWarning, warning_replay, reference_warning


class WarningTests(unittest.TestCase):
    def test_startup_unknown_and_reset(self):
        w = ReceivedChangeWarning([1, 2, 3])
        self.assertEqual(w.step(None)[:2], (False, False))
        self.assertEqual(w.step(np.zeros(6))[:2], (False, False))
        w.step(np.ones(6) * 5)
        w.reset()
        self.assertEqual(w.step(None)[:2], (False, False))
        self.assertEqual(w.last_index, -1)

    def test_manual_hold_elapsed_equality_and_recovery(self):
        w = ReceivedChangeWarning([1, 2, 3])
        w.step(np.zeros(6))
        self.assertEqual(w.step([4, 0, 0, 0, 0, 0])[:4], (True, True, 4, 0))
        self.assertEqual(w.step(None)[:4], (True, True, 4, 1))
        self.assertEqual(w.step(None)[:4], (True, True, 4, 2))
        # Change 3 over three elapsed steps = exactly 1: strict > is false.
        self.assertEqual(w.step([7, 0, 0, 0, 0, 0])[:4], (False, True, 1, 0))
        self.assertEqual(w.step(None)[:4], (False, True, 1, 1))

    def test_all_acceleration_axes_and_gyro_exclusion(self):
        for c, threshold in enumerate([1, 2, 3]):
            w = ReceivedChangeWarning([1, 2, 3])
            w.step(np.zeros(6))
            value = np.zeros(6); value[c] = -2 * threshold
            self.assertTrue(w.step(value)[0])
        w = ReceivedChangeWarning([1, 2, 3]); w.step(np.zeros(6))
        self.assertFalse(w.step([0, 0, 0, 1000, -1000, 500])[0])

    def test_input_ownership_and_invalid_state(self):
        thresholds = np.ones(3)
        w = ReceivedChangeWarning(thresholds); thresholds[:] = 100
        a = np.zeros(6); w.step(a); a[:] = 100
        state = (w.index, w.last_index, w.age)
        for invalid in ([1, 2], [np.nan]*6):
            with self.assertRaises(ValueError): w.step(invalid)
            self.assertEqual((w.index, w.last_index, w.age), state)
        self.assertTrue(w.step([2, 0, 0, 0, 0, 0])[0])
        for invalid in ([0, 1, 1], [np.inf, 1, 1], [1, 2]):
            with self.assertRaises(ValueError): ReceivedChangeWarning(invalid)

    def test_reference_and_hidden_isolation(self):
        truth = np.arange(72).reshape(12, 6).astype(float)
        mask = np.array([True, False, True, False, False, True, True, True, False, False, True, False])
        w = ReceivedChangeWarning([2, 3, 4])
        before = truth.copy(); original_mask = mask.copy()
        expected = warning_replay(w, truth, mask)
        for a, b in zip(expected, reference_warning(truth, mask, w.thresholds)):
            np.testing.assert_array_equal(a, b)
        poisoned = truth.copy(); poisoned[mask] = np.nan
        for a, b in zip(expected, warning_replay(w, poisoned, mask)):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(before, truth)
        np.testing.assert_array_equal(original_mask, mask)

    def test_every_future_cutoff(self):
        truth = np.arange(60).reshape(10, 6).astype(float)
        mask = np.array([False, False, True, True, False, True, False, False, True, True])
        w = ReceivedChangeWarning([4, 5, 6]); expected = warning_replay(w, truth, mask)
        for cutoff in range(11):
            changed, altered_mask = truth.copy(), mask.copy()
            changed[cutoff:] += 123; altered_mask[cutoff:] = ~altered_mask[cutoff:]
            for a, b in zip(expected, warning_replay(w, changed, altered_mask)):
                np.testing.assert_array_equal(a[:cutoff], b[:cutoff])

    def test_a_b_a_reset_and_consecutive_gaps(self):
        w = ReceivedChangeWarning([1, 1, 1])
        a = np.arange(90).reshape(15, 6).astype(float)
        mask = np.array([False, False]+[True]*10+[False, False, True])
        original = warning_replay(w, a, mask)
        warning_replay(w, np.zeros((4, 6)), np.array([True, False, True, True]))
        for x, y in zip(original, warning_replay(w, a, mask)):
            np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(original[3][2:12], np.arange(1, 11))
        self.assertTrue(original[0][2:12].all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
