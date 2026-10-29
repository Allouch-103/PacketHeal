"""Synthetic abstention tests; no actual recording access."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from packetheal.baselines import SampleStatus
from packetheal.evaluation import ReplayResult, score_hidden
from packetheal.mlp import MLP, MLPPredictor, Normalization
from packetheal.mlp_experiment import audited_replay, streaming_replay
from mlp_v1_abstention_20261027 import AbstainingMLP


class AbstentionTests(unittest.TestCase):
    def make(self):
        norm = Normalization(np.zeros(6), np.ones(6))
        model = MLP(mode="residual", seed=0)
        model.parameters[-1].fill(2)
        return AbstainingMLP(model, norm, [1, 1, 1])

    def warm(self, predictor):
        for _ in range(20): predictor.step(np.zeros(6))

    def test_real_preservation_ownership(self):
        p = self.make(); value = np.arange(6, dtype=float)
        result = p.step(value)
        self.assertEqual(result.status, SampleStatus.MEASURED)
        np.testing.assert_array_equal(result.values, value)
        result.values[:] = 100; value[:] = 200
        np.testing.assert_array_equal(p.predictor.state.anchor, np.arange(6))

    def test_startup_unknown_and_history_fallback(self):
        p = self.make()
        self.assertEqual(p.step(None).status, SampleStatus.UNAVAILABLE)
        p.step(np.zeros(6))
        self.assertEqual(p.step(None).status, SampleStatus.UNAVAILABLE)
        p.step(np.zeros(6))
        # Original predictor uses LOCF before its 20-slot history is ready.
        result = p.step(None)
        self.assertEqual(result.status, SampleStatus.RECONSTRUCTED)
        np.testing.assert_array_equal(result.values, np.zeros(6))

    def test_unwarned_matches_original_forecast(self):
        p = self.make(); self.warm(p)
        result = p.step(None)
        self.assertEqual(result.status, SampleStatus.RECONSTRUCTED)
        self.assertEqual(p.reason, "ACCEPTED")
        np.testing.assert_array_equal(result.values, np.full(6, 2.))

    def test_warning_consecutive_rejections_and_recovery(self):
        p = self.make(); self.warm(p)
        p.step([2, 0, 0, 0, 0, 0])
        for age in range(1, 4):
            result = p.step(None)
            self.assertEqual(result.status, SampleStatus.UNAVAILABLE)
            self.assertEqual(result.missing_run, age)
            self.assertTrue(np.isnan(result.values).all())
        # Return change 2 over FOUR elapsed steps = 0.5, below threshold.
        p.step(np.zeros(6)); self.assertEqual(p.warning.score, .5)
        self.assertEqual(p.step(None).status, SampleStatus.RECONSTRUCTED)
        p.step([4, 0, 0, 0, 0, 0])
        self.assertEqual(p.step(None).status, SampleStatus.UNAVAILABLE)
        p.step([4, 0, 0, 0, 0, 0])
        self.assertEqual(p.step(None).status, SampleStatus.RECONSTRUCTED)

    def test_horizon_boundary_and_recovery(self):
        p = self.make(); self.warm(p)
        for age in range(1, 11):
            self.assertEqual(p.step(None).status, SampleStatus.RECONSTRUCTED)
        for age in (11, 12):
            result = p.step(None)
            self.assertEqual(result.missing_run, age)
            self.assertEqual(result.status, SampleStatus.UNAVAILABLE)
            self.assertEqual(p.reason, "HORIZON_EXCEEDED")
        p.step(np.zeros(6)); self.assertEqual(p.step(None).status, SampleStatus.RECONSTRUCTED)

    def test_rejected_output_never_enters_history(self):
        p = self.make(); self.warm(p)
        p.step([2, 0, 0, 0, 0, 0]); p.step(None); p.step(None)
        np.testing.assert_array_equal(p.predictor.state.anchor, [2, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(p.predictor.state.rows[-2:, :6], [[2, 0, 0, 0, 0, 0]] * 2)
        np.testing.assert_array_equal(p.predictor.state.rows[-2:, 6], [0, 0])
        self.assertTrue(np.isfinite(p.predictor.state.rows).all())

    def test_failures_do_not_advance_either_state(self):
        p = self.make(); self.warm(p)
        for bad in ([1, 2], [np.nan] * 6, [1e100] * 6):
            before = p.predictor.state.rows.copy(); index, elapsed = p.warning.index, p.predictor.state.elapsed
            with self.assertRaises(ValueError): p.step(bad)
            np.testing.assert_array_equal(p.predictor.state.rows, before)
            self.assertEqual((p.warning.index, p.predictor.state.elapsed), (index, elapsed))
        with patch.object(p.predictor.model, "predict", side_effect=FloatingPointError("synthetic failure")):
            index, elapsed = p.warning.index, p.predictor.state.elapsed
            with self.assertRaises(FloatingPointError): p.step(None)
            self.assertEqual((p.warning.index, p.predictor.state.elapsed), (index, elapsed))

    def test_reset_and_full_causal_audits(self):
        p = self.make()
        truth = np.zeros((35, 6)); truth[20, 0] = 5
        missing = np.zeros(35, dtype=bool); missing[21:25] = True; missing[28:30] = True
        result = audited_replay(p, truth, missing)
        self.assertTrue((result[1][21:25] == SampleStatus.UNAVAILABLE.value).all())
        p.reset()
        self.assertEqual(p.warning.index, 0)
        self.assertEqual(p.reason, "RESET")
        self.assertEqual(p.step(None).status, SampleStatus.UNAVAILABLE)

    def test_every_future_cutoff(self):
        p = self.make(); truth = np.zeros((12, 6)); truth[3, 1] = 4
        missing = np.array([False, False, True, False, True, True, False, False, True, True, False, True])
        original = streaming_replay(p, truth, missing)
        for cutoff in range(13):
            altered, mask = truth.copy(), missing.copy()
            altered[cutoff:] += 5; mask[cutoff:] = ~mask[cutoff:]
            for a, b in zip(original, streaming_replay(p, altered, mask)):
                np.testing.assert_array_equal(a[:cutoff], b[:cutoff])

    def test_fully_rejected_scoring(self):
        p = self.make(); truth = np.zeros((3, 6)); missing = np.array([False, True, True])
        result = ReplayResult(*streaming_replay(p, truth, missing))
        metrics = score_hidden(truth, missing, result)
        self.assertEqual(metrics["hidden_samples"], 2)
        self.assertEqual(metrics["unavailable_samples"], 2)
        self.assertEqual(metrics["reconstruction_coverage"], 0)
        self.assertIsNone(metrics["mae"])
        self.assertIsNone(metrics["rmse"])


if __name__ == "__main__": unittest.main(verbosity=2)
