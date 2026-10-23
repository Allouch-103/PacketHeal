"""Gain freezing, minimum-score selection and actual replay checks."""

from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_alpha_beta import checked_alpha_replay, load_selected_gains, main
from evaluate_locf import sha256
from packetheal.alpha_beta import AlphaBetaPredictor
from packetheal.evaluation import REGIMES


class FrozenGainTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "gains.json"
        subjects = list(range(101, 107))
        self.search = {"fit_split": "train", "subjects": subjects, "regimes": list(REGIMES),
                       "alpha_values": [.5, 1], "beta_values": [0, .25]}
        expected = {"subjects": subjects, "segments": 6, "samples": 12,
                    "lost_by_regime": {r: 6 for r in REGIMES}}
        self.context = {"preparation": {"splits": {"train": expected}},
                        "config": {"channels": ["ax", "ay", "az", "gx", "gy", "gz"]},
                        "hashes": {"slope_caps": "a" * 64}, "caps": {"max_delta_per_step": [2] * 6},
                        "scales": np.ones(6)}
        results = []
        for k, (alpha, beta) in enumerate((a, b) for a in [.5, 1] for b in [0, .25]):
            results.append({"alpha": alpha, "beta": beta, "objective": k + 1.,
                            "by_regime": {r: {"hidden_samples": 6,
                                "absolute_error_sum": [6 * (k + 1)] * 6} for r in REGIMES}})
        self.fitted = {"schema_version": "1", "purpose": "alpha_beta_training_gain_selection",
            "fit_split": "train", "subjects": subjects, "segments": 6, "samples": 12, "cases": 24,
            "search": self.search, "channel_order": self.context["config"]["channels"],
            "sampling_frequency_hz": 100, "normalization_applied_to_predictor": False,
            "slope_caps_sha256": "a" * 64, "slope_caps": [2] * 6,
            "objective_training_std_sample": [1] * 6, "candidate_results": results,
            "selected_candidate_index": 0, "alpha": .5, "beta": 0, "objective": 1.,
            "verification": {"all_training_counts_reconciled": True, "streaming_parity_prefixes": 24,
                             "repeat_objective_and_selection_identical": True, "inputs_and_sources_unchanged": True}}

    def write(self, data=None):
        self.path.write_text(json.dumps(self.fitted if data is None else data))
        digest = sha256(self.path)
        self.path.with_suffix(".json.sha256").write_text(f"{digest}  {self.path.name}\n")
        return digest

    def load(self, digest):
        return load_selected_gains(self.path, digest, self.context, self.search)

    def test_valid_checksum_contract_and_minimum_selection(self):
        self.assertEqual(self.load(self.write()), self.fitted)

    def test_rewriting_json_and_sidecar_cannot_bypass_independent_pin(self):
        original = self.write()
        self.fitted["alpha"] = 1
        self.write()
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.load(original)

    def test_wrong_training_membership_counts_caps_or_units_are_rejected(self):
        for key, value in (("fit_split", "validation"), ("subjects", [107]),
                           ("subjects", [108]), ("samples", 13), ("cases", 23),
                           ("slope_caps", [3] * 6), ("sampling_frequency_hz", 50),
                           ("normalization_applied_to_predictor", True),
                           ("channel_order", list(reversed(self.fitted["channel_order"])))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(self.write({**self.fitted, key: value}))

    def test_higher_score_candidate_cannot_be_claimed_as_selected(self):
        changed = {**self.fitted, "selected_candidate_index": 3, "alpha": 1, "beta": .25, "objective": 4.}
        with self.assertRaisesRegex(ValueError, "minimum"):
            self.load(self.write(changed))

    def test_missing_candidate_wrong_order_hidden_count_and_parity_rejected(self):
        for mutation in ("missing", "order", "hidden", "parity"):
            changed = deepcopy(self.fitted)
            if mutation == "missing":
                changed["candidate_results"].pop()
            elif mutation == "order":
                changed["candidate_results"].reverse()
            elif mutation == "hidden":
                changed["candidate_results"][0]["by_regime"][REGIMES[0]]["hidden_samples"] += 1
            else:
                changed["verification"]["streaming_parity_prefixes"] = 23
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.load(self.write(changed))


class ValidationReplayTests(unittest.TestCase):
    def test_checked_replay_scores_only_hidden_and_preserves_arrays(self):
        truth = np.repeat(np.array([[0.], [4.], [6.], [8.], [12.], [10.]]), 6, axis=1)
        missing = np.array([False, False, True, True, False, True])
        original = truth.copy()
        result, metrics = checked_alpha_replay(AlphaBetaPredictor(.5, .25, [10] * 6), truth, missing)
        np.testing.assert_allclose(result.values[:, 0], [0, 4, 3, 4, 12, 10 + 1 / 12])
        np.testing.assert_array_equal(truth, original)
        self.assertEqual(metrics["scored_samples"], 3)
        self.assertEqual(metrics["unavailable_samples"], 0)

    def test_startup_unavailable_positions_fail_protected_mask_acceptance(self):
        with self.assertRaisesRegex(ValueError, "Not all hidden"):
            checked_alpha_replay(AlphaBetaPredictor(.5, .25, [1] * 6),
                                 np.ones((3, 6)), np.array([True, False, True]))

    def test_existing_output_refused_before_inputs_are_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text("keep")
            with self.assertRaises(FileExistsError):
                main(["--gains-sha256", "a" * 64, "--output", str(path)])
            self.assertEqual(path.read_text(), "keep")

    def test_cli_excludes_held_out_subjects_and_splits(self):
        for options in (["--subject", "108"], ["--split", "test"]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--gains-sha256", "a" * 64, *options])


if __name__ == "__main__":
    unittest.main()
