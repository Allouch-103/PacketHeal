"""Uniform replay audits, diagnostic partitions, and access boundaries."""

from contextlib import redirect_stderr
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_classical_baselines import case_diagnostics, checked_replay, main, validate_policy, validation_access_guard
from packetheal.alpha_beta import AlphaBetaPredictor
from packetheal.baselines import LOCF
from packetheal.evaluation import replay_segment
from packetheal.linear_extrapolation import LinearExtrapolator

ROOT = Path(__file__).resolve().parents[1]


class Phase2ReplayTests(unittest.TestCase):
    def constructors(self):
        return [lambda: LOCF(2), lambda: LinearExtrapolator([2, 3], 2),
                lambda: AlphaBetaPredictor(1, 0, [2, 3], 2),
                lambda: AlphaBetaPredictor(.5, .25, [2, 3], 2)]

    def test_uniform_checks_cover_all_predictors_including_nonzero_filter(self):
        truth = np.array([[0., 0], [4, -4], [6, -6], [8, -8], [12, -12], [10, -10]])
        mask = np.array([False, False, True, True, False, True])
        for create in self.constructors():
            result, metrics = checked_replay(create(), truth, mask)
            self.assertEqual(metrics['scored_samples'], 3)
            np.testing.assert_array_equal(result.missing_run, [0, 0, 1, 2, 0, 1])

    def test_all_synthetic_prefixes_survive_future_values_and_future_mask_changes(self):
        truth = np.arange(24.).reshape(12, 2)
        mask = np.array([True, False, False, True, True, False, True, False, True, True, False, True])
        for create in self.constructors():
            full = replay_segment(create(), truth, mask)
            for cutoff in range(len(truth) + 1):
                changed, future_mask = truth.copy(), mask.copy()
                changed[cutoff:] += 100
                future_mask[cutoff:] = ~future_mask[cutoff:]
                actual = replay_segment(create(), changed, future_mask)
                prefix = replay_segment(create(), truth[:cutoff], mask[:cutoff])
                for field in ('values', 'status', 'missing_run'):
                    np.testing.assert_array_equal(getattr(full, field)[:cutoff], getattr(actual, field)[:cutoff])
                    np.testing.assert_array_equal(getattr(full, field)[:cutoff], getattr(prefix, field))

    def test_broken_reset_is_detected(self):
        class BrokenReset(LOCF):
            def reset(self):
                if not hasattr(self, '_last_received'):
                    super().reset()
        with self.assertRaises(AssertionError):
            checked_replay(BrokenReset(1), np.array([[1.], [2.], [3.]]), np.array([True, False, True]))

    def test_posthoc_labels_can_change_without_changing_predictions(self):
        truth = np.array([[0., 0], [0, 0], [10, 0], [10, 0]])
        missing = np.array([False, False, True, False])
        first = replay_segment(LOCF(2), truth, missing)
        a, runs = case_diagnostics(truth, missing, first, [5, 5], [1])
        changed = truth.copy()
        changed[missing] = 0
        second = replay_segment(LOCF(2), changed, missing)
        b, _ = case_diagnostics(changed, missing, second, [5, 5], [1])
        np.testing.assert_array_equal(first.values, second.values)
        self.assertEqual(a['large_change']['hidden_count'], [1, 0])
        self.assertEqual(b['large_change']['hidden_count'], [0, 0])
        self.assertEqual(runs, {1: 1})

    def test_burst_strata_and_unexpected_ages(self):
        truth = np.arange(12.).reshape(6, 2)
        missing = np.array([False, True, True, False, True, True])
        result = replay_segment(LOCF(2), truth, missing)
        groups, runs = case_diagnostics(truth, missing, result, [1, 1], [1, 2])
        self.assertEqual(groups['age_1']['hidden_count'], [2, 2])
        self.assertEqual(groups['age_2']['hidden_count'], [2, 2])
        self.assertEqual(runs, {2: 2})
        with self.assertRaises(ValueError):
            case_diagnostics(truth, missing, result, [1, 1], [1])


class Phase2BoundaryTests(unittest.TestCase):
    def test_policy_rejects_partition_parameter_or_diagnostic_changes(self):
        policy = json.loads((ROOT / 'configs/phase2_analysis_v1.json').read_text())
        validate_policy(policy)
        for key, value in [('subjects', [108]), ('split', 'train'), ('gains_sha256', '0' * 64),
                           ('new_fitting_or_masks', True), ('diagnostic_labels_are_predictor_inputs', True),
                           ('first_row_transient', True), ('loss_ages', {})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_policy({**policy, key: value})

    def test_only_canonical_validation_recording_can_be_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            allowed, held_out = raw / 'subject107.dat', raw / 'subject108.dat'
            allowed.write_text('synthetic validation fixture')
            held_out.write_text('synthetic held-out fixture')
            alias = raw / 'alias.dat'
            alias.symlink_to(held_out)
            with validation_access_guard(raw, allowed) as access:
                self.assertEqual(allowed.read_text(), 'synthetic validation fixture')
                for path in (held_out, alias):
                    with self.assertRaises(PermissionError):
                        path.read_text()
            self.assertEqual(access['opened_recordings'], [str(allowed)])
            self.assertEqual(access['denied_open_attempts'], 2)

    def test_guard_cannot_allow_another_subject(self):
        with self.assertRaises(ValueError):
            with validation_access_guard('/tmp', '/tmp/subject109.dat'):
                pass

    def test_cli_excludes_held_out_subject_and_split_options(self):
        for args in (['--subject', '108'], ['--split', 'test']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)

    def test_existing_output_refused_without_raw_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'preserve'
            path.mkdir()
            with self.assertRaises(FileExistsError):
                main(['--output-dir', str(path)])
            self.assertEqual(list(path.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
