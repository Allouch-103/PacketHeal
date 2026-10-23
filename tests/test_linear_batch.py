"""Coverage and sample-weighted aggregation contracts for validation batches."""

from contextlib import redirect_stderr
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_linear_extrapolation_batch import main, summarize_cases
from packetheal.evaluation import REGIMES


def metrics(hidden, absolute, squared):
    return {
        "hidden_samples": hidden, "scored_samples": hidden, "unavailable_samples": 0,
        "reconstruction_coverage": 1.0, "absolute_error_sum": [absolute] * 6,
        "squared_error_sum": [squared] * 6, "mae": [absolute / hidden] * 6,
        "rmse": [float(np.sqrt(squared / hidden))] * 6,
    }


class LinearBatchTests(unittest.TestCase):
    def setUp(self):
        self.expected = {
            "segments": 2, "samples": 7,
            "lost_by_regime": {regime: 4 for regime in REGIMES},
        }
        self.cases = []
        for index, samples, count, error in ((0, 2, 1, 1), (1, 5, 3, 3)):
            for regime in REGIMES:
                self.cases.append({
                    "segment_index": index, "regime": regime, "samples": samples,
                    "max_missing_run": count,
                    "metrics": metrics(count, count * error, count * error ** 2),
                    "locf_metrics": metrics(count, count * (error + 1), count * (error + 1) ** 2),
                })
        self.archived = {regime: {"metrics": metrics(4, 14, 52)} for regime in REGIMES}

    def summarize(self, cases=None):
        return summarize_cases(self.cases if cases is None else cases,
                               self.expected, self.archived)

    def test_known_unequal_segment_errors_use_sample_weighting(self):
        # Hidden errors are [1] and [3, 3, 3]: MAE=2.5, RMSE=sqrt(7).
        pooled = self.summarize()
        for regime in REGIMES:
            entry = pooled[regime]
            self.assertEqual(entry["metrics"], metrics(4, 10, 28))
            self.assertEqual(entry["locf_metrics"], metrics(4, 14, 52))
            self.assertEqual(entry["samples"], 7)
            self.assertEqual(entry["segments"], 2)
            self.assertEqual(entry["actual_missing_fraction"], 4 / 7)
            self.assertNotEqual(entry["metrics"]["mae"][0], (1 + 3) / 2)

    def test_case_order_does_not_change_pooled_results(self):
        self.assertEqual(self.summarize(), self.summarize(list(reversed(self.cases))))

    def test_missing_duplicate_and_unknown_cases_are_rejected(self):
        duplicate = self.cases[:-1] + [self.cases[0]]
        unknown = deepcopy(self.cases)
        unknown[0]["segment_index"] = 2
        for cases in (self.cases[:-1], duplicate, unknown):
            with self.subTest(cases=cases), self.assertRaisesRegex(ValueError, "cases"):
                self.summarize(cases)

    def test_sample_total_or_per_regime_length_mismatch_is_rejected(self):
        changed = deepcopy(self.cases)
        changed[0]["samples"] += 1
        with self.assertRaisesRegex(ValueError, "sample counts"):
            self.summarize(changed)
        self.expected["samples"] += 1
        with self.assertRaisesRegex(ValueError, "Eligible sample"):
            self.summarize()

    def test_incomplete_scoring_and_predictor_count_mismatch_are_rejected(self):
        changed = deepcopy(self.cases)
        changed[0]["metrics"]["scored_samples"] = 0
        changed[0]["metrics"]["unavailable_samples"] = 1
        with self.assertRaisesRegex(ValueError, "all hidden"):
            self.summarize(changed)
        changed = deepcopy(self.cases)
        changed[0]["locf_metrics"] = metrics(2, 4, 8)
        with self.assertRaisesRegex(ValueError, "hidden counts differ"):
            self.summarize(changed)

    def test_frozen_hidden_total_and_archived_locf_must_match(self):
        self.expected["lost_by_regime"][REGIMES[0]] = 5
        with self.assertRaisesRegex(ValueError, "hidden-position total"):
            self.summarize()
        self.expected["lost_by_regime"][REGIMES[0]] = 4
        self.archived[REGIMES[0]]["metrics"]["absolute_error_sum"][0] += 1
        with self.assertRaisesRegex(ValueError, "LOCF metrics differ"):
            self.summarize()

    def test_existing_report_is_refused_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("preserve me")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                main(["--output", str(output)])
            self.assertEqual(output.read_text(), "preserve me")

    def test_cli_does_not_offer_held_out_subjects_or_splits(self):
        for options in (["--subject", "108"], ["--split", "test"]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(options)
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
