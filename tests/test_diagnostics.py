"""Hand-calculated post-hoc labels, counts, errors, and partition checks."""

import unittest
import numpy as np

from packetheal.baselines import LOCF
from packetheal.diagnostics import large_change_flags, loss_run_metadata, pool_strata, reconcile_partition, stratified_sums
from packetheal.evaluation import replay_segment


class DiagnosticLabelsTests(unittest.TestCase):
    def test_channel_specific_changes_use_strict_threshold(self):
        truth = np.array([[0., 0], [5, -2], [11, -5], [11, -5]])
        np.testing.assert_array_equal(large_change_flags(truth, [5, 2]),
            [[False, False], [False, False], [True, True], [False, False]])

    def test_segment_boundaries_never_create_a_change(self):
        for truth in (np.array([[0.], [0.]]), np.array([[100.], [100.]]), np.array([[999.]])):
            self.assertFalse(large_change_flags(truth, [1]).any())

    def test_constant_ramp_and_step_flags(self):
        np.testing.assert_array_equal(large_change_flags([[0], [0], [10], [10]], [5])[:, 0],
                                      [False, False, True, False])
        np.testing.assert_array_equal(large_change_flags([[0], [2], [4]], [1])[:, 0], [False, True, True])

    def test_bad_truth_and_thresholds_rejected(self):
        for truth, caps in (([], [1]), ([1, 2], [1]), ([[1j]], [1]), ([[np.nan]], [1]),
                            ([[1]], [-1]), ([[1]], [np.inf]), ([[1]], [1, 2])):
            with self.subTest(truth=truth, caps=caps), self.assertRaises(ValueError):
                large_change_flags(truth, caps)

    def test_difference_overflow_fails(self):
        with self.assertRaises(FloatingPointError):
            large_change_flags([[-1e308], [1e308]], [1])

    def test_loss_ages_and_full_lengths_are_independent(self):
        ages, lengths = loss_run_metadata(np.array([False, True, True, False, True, True, True, False]))
        np.testing.assert_array_equal(ages, [0, 1, 2, 0, 1, 2, 3, 0])
        np.testing.assert_array_equal(lengths, [0, 2, 2, 0, 3, 3, 3, 0])

    def test_loss_runs_at_both_boundaries_and_empty_mask(self):
        ages, lengths = loss_run_metadata(np.array([True, True, False, True]))
        np.testing.assert_array_equal(ages, [1, 2, 0, 1])
        np.testing.assert_array_equal(lengths, [2, 2, 0, 1])
        for mask in (np.array([], dtype=bool), np.zeros(3, dtype=bool)):
            self.assertFalse(loss_run_metadata(mask)[0].any())

    def test_loss_mask_must_be_boolean_vector(self):
        for mask in ([0, 1], [[False, True]], True):
            with self.assertRaises(ValueError):
                loss_run_metadata(mask)


class StratifiedScoreTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.array([[0., 0], [4, 40], [6, 60], [8, 80], [12, 120]])
        self.mask = np.array([False, True, True, False, True])
        self.result = replay_segment(LOCF(2), self.truth, self.mask)
        self.hidden = np.broadcast_to(self.mask[:, None], self.truth.shape)

    def score(self, selection):
        return stratified_sums(self.truth, self.mask, self.result, selection)

    def test_different_channel_counts_and_known_errors(self):
        selection = np.zeros_like(self.truth, dtype=bool)
        selection[[1, 4], 0], selection[2, 1] = True, True
        result = pool_strata([self.score(selection)])
        self.assertEqual(result['scored_count'], [2, 1])
        self.assertEqual(result['absolute_error_sum'], [8, 60])
        self.assertEqual(result['squared_error_sum'], [32, 3600])
        self.assertEqual(result['mae'], [4, 60])
        self.assertEqual(result['rmse'], [4, 60])

    def test_other_hidden_positions_keep_original_provenance(self):
        selection = self.hidden.copy()
        selection[2:] = False
        self.assertEqual(self.score(selection)['scored_count'], [1, 1])

    def test_empty_groups_and_unavailable_outputs_have_no_zero_error_claim(self):
        empty = pool_strata([self.score(np.zeros_like(self.hidden))])
        self.assertEqual(empty['mae'], [None, None])
        self.assertEqual(empty['coverage'], [None, None])
        mask = np.array([True, False])
        truth = np.ones((2, 2))
        unavailable = pool_strata([stratified_sums(truth, mask, replay_segment(LOCF(2), truth, mask),
                                                   np.broadcast_to(mask[:, None], truth.shape))])
        self.assertEqual(unavailable['unavailable_count'], [1, 1])
        self.assertEqual(unavailable['coverage'], [0., 0.])
        self.assertEqual(unavailable['rmse'], [None, None])

    def test_selected_received_positions_bad_shape_or_dtype_rejected(self):
        for selection in (np.ones_like(self.hidden), self.mask, self.hidden.astype(int)):
            with self.assertRaises(ValueError):
                self.score(selection)

    def test_partition_conserves_every_channel_count_and_error_sum(self):
        flags = large_change_flags(self.truth, [3, 50])
        a, b, full = self.score(self.hidden & flags), self.score(self.hidden & ~flags), self.score(self.hidden)
        reconcile_partition([a, b], full)
        with self.assertRaises(AssertionError):
            reconcile_partition([a, a], full)

    def test_pooling_uses_sums_and_counts_not_means(self):
        a = {'hidden_count': [1], 'scored_count': [1], 'unavailable_count': [0],
             'absolute_error_sum': [1.], 'squared_error_sum': [1.]}
        b = {'hidden_count': [3], 'scored_count': [3], 'unavailable_count': [0],
             'absolute_error_sum': [9.], 'squared_error_sum': [27.]}
        result = pool_strata([a, b])
        self.assertEqual(result['mae'], [2.5])
        np.testing.assert_allclose(result['rmse'], [np.sqrt(7)])
        self.assertEqual(pool_strata([b, a]), result)

    def test_bad_counts_sums_or_empty_pools_fail(self):
        record = self.score(self.hidden)
        for key, bad in (('hidden_count', [1, 1]), ('scored_count', [1.5, 1]),
                         ('scored_count', [True, False]), ('absolute_error_sum', [-1, 0]),
                         ('squared_error_sum', [np.inf, 0]), ('unavailable_count', [0])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                pool_strata([{**record, key: bad}])
        with self.assertRaises(ValueError):
            pool_strata([])
        with self.assertRaises(ValueError):
            pool_strata([{'hidden_count': [0], 'scored_count': [0], 'unavailable_count': [0],
                          'absolute_error_sum': [1.], 'squared_error_sum': [1.]}])

    def test_scoring_overflow_rejected(self):
        truth, mask = np.array([[-1e308], [1e308]]), np.array([False, True])
        with self.assertRaises(FloatingPointError):
            stratified_sums(truth, mask, replay_segment(LOCF(1), truth, mask), mask[:, None])


if __name__ == '__main__':
    unittest.main()
