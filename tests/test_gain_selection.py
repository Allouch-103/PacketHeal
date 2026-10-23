"""Independent streaming parity and the predeclared training-score objective."""

import unittest

import numpy as np

from packetheal.alpha_beta import AlphaBetaPredictor
from packetheal.evaluation import replay_segment, score_hidden
from packetheal.gain_selection import choose_gain_candidate, score_gain_candidates


class CandidateReplayTests(unittest.TestCase):
    def setUp(self):
        self.pairs = [(a, b) for a in (.25, .5, .75, 1) for b in (0, .1, .25, .5, 1)]

    def assert_parity(self, truth, mask, caps):
        original, original_mask = truth.copy(), mask.copy()
        batched = score_gain_candidates(truth, mask, self.pairs, caps)
        for i, (alpha, beta) in enumerate(self.pairs):
            scalar = score_hidden(truth, mask, replay_segment(
                AlphaBetaPredictor(alpha, beta, caps, truth.shape[1]), truth, mask,
            ))
            self.assertEqual(batched["hidden_samples"], scalar["hidden_samples"])
            for key in ("absolute_error_sum", "squared_error_sum"):
                np.testing.assert_allclose(batched[key][i], scalar[key], rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(truth, original)
        np.testing.assert_array_equal(mask, original_mask)

    def test_all_candidates_match_streaming_on_random_mask_and_signed_transients(self):
        rng = np.random.default_rng(741)
        truth = rng.normal(size=(97, 6)) * [1, 10, .1, 20, 2, 3]
        mask = rng.random(97) < .3
        mask[0] = False
        self.assert_parity(truth, mask, [0, .5, 3, 10, 1, 1])

    def test_bursts_and_gap_recovery_match_streaming(self):
        truth = np.array([[0., 0], [4, -4], [80, -80], [40, -40], [8, -8],
                          [16, -16], [3, -3], [2, -2], [100, -100]])
        mask = np.array([False, False, True, True, False, True, True, False, True])
        self.assert_parity(truth, mask, [10, 2])

    def test_constant_channels_and_no_loss_prefix_have_zero_sums(self):
        self.assert_parity(np.ones((20, 2)), np.array([False] + [True] * 19), [1, 2])
        self.assert_parity(np.arange(20.).reshape(10, 2), np.zeros(10, dtype=bool), [1, 2])

    def test_calls_reset_candidate_states(self):
        first = np.array([[0.], [4.], [100.]])
        mask = np.array([False, False, True])
        a = score_gain_candidates(first, mask, self.pairs, [2])
        score_gain_candidates(first + 200, mask, self.pairs, [2])
        b = score_gain_candidates(first, mask, self.pairs, [2])
        np.testing.assert_array_equal(a["absolute_error_sum"], b["absolute_error_sum"])

    def test_bad_arrays_or_uninitialized_masks_are_rejected(self):
        for truth in ([], [1, 2], [[1j]], [[np.nan]], [[np.inf]]):
            with self.subTest(truth=truth), self.assertRaises(ValueError):
                score_gain_candidates(truth, np.array([False]), self.pairs, [1])
        truth = np.ones((2, 1))
        for mask in ([False], [0, 1], [True, False]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                score_gain_candidates(truth, np.array(mask), self.pairs, [1])

    def test_empty_grid_bad_gains_and_bad_caps_are_rejected(self):
        for pairs in ([], [(0, .5)], [(.5, 2)], [(.5,)], [(True, .5)]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                score_gain_candidates(np.ones((2, 1)), np.array([False, True]), pairs, [1])
        with self.assertRaises(ValueError):
            score_gain_candidates(np.ones((2, 1)), np.array([False, True]), self.pairs, [-1])

    def test_overflow_is_rejected_instead_of_silently_affecting_ranking(self):
        truth = np.array([[-1e308], [1e308], [0.]])
        with self.assertRaises(FloatingPointError):
            score_gain_candidates(truth, np.array([False, False, True]), self.pairs, [1])


class ObjectiveTests(unittest.TestCase):
    def test_known_equal_regime_and_channel_weighting(self):
        # Candidate 0: normalized MAEs [.5,.5], [1,1] -> .75.
        # Candidate 1: normalized MAEs [1,1], [.1,.1] -> .55.
        sums = np.array([[[2, 20], [4, 40]], [[20, 200], [2, 20]]], dtype=float)
        selected, scores = choose_gain_candidate(sums, [2, 10], [2, 20])
        np.testing.assert_allclose(scores, [.75, .55])
        self.assertEqual(selected, 1)

    def test_channel_units_rescale_without_changing_selection(self):
        sums = np.array([[[2., 20], [3, 10]]])
        selected, scores = choose_gain_candidate(sums, [2], [2, 20])
        sums[:, :, 1] *= 1000
        again, scaled = choose_gain_candidate(sums, [2], [2, 20000])
        np.testing.assert_array_equal(scores, scaled)
        self.assertEqual(selected, again)

    def test_ties_use_first_predeclared_candidate(self):
        selected, scores = choose_gain_candidate(np.ones((4, 20, 6)), [1] * 4, [1] * 6)
        self.assertEqual(selected, 0)
        np.testing.assert_array_equal(scores, np.ones(20))

    def test_invalid_objective_inputs_are_rejected(self):
        for sums, counts, scales in (([], [1], [1]), ([[[1]]], [0], [1]),
                                    ([[[1]]], [1.5], [1]), ([[[1]]], [1], [0]),
                                    ([[[1]]], [1], [np.nan]), ([[[-1]]], [1], [1]),
                                    ([[[np.inf]]], [1], [1]), ([[[1]]], [1, 2], [1])):
            with self.subTest(sums=sums, counts=counts, scales=scales), self.assertRaises(ValueError):
                choose_gain_candidate(sums, counts, scales)


if __name__ == "__main__":
    unittest.main()
