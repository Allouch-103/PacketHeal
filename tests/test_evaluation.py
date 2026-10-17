"""Causality, metric arithmetic, and frozen-mask identity/row conversion."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from packetheal.baselines import LOCF
from packetheal.evaluation import (
    MASK_FIELDS, REGIMES, load_segment_mask, load_subject_masks, pool_metrics,
    replay_segment, score_hidden,
)


class ReplayAndMetricTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.array([[10, 100], [13, 130], [17, 170], [20, 200], [22, 220.]])
        self.missing = np.array([False, True, True, False, False])

    def test_replay_preserves_received_samples_and_inputs(self):
        before = self.truth.copy()
        result = replay_segment(LOCF(2), self.truth, self.missing)
        np.testing.assert_array_equal(self.truth, before)
        np.testing.assert_array_equal(
            result.values[~self.missing], self.truth[~self.missing]
        )
        np.testing.assert_array_equal(result.values[self.missing], [[10, 100], [10, 100]])
        self.assertEqual(result.status.tolist(), [
            "MEASURED", "RECONSTRUCTED", "RECONSTRUCTED", "MEASURED", "MEASURED"
        ])
        np.testing.assert_array_equal(result.missing_run, [0, 1, 2, 0, 0])

    def test_predictor_receives_none_for_each_hidden_sample(self):
        class RecordingLOCF(LOCF):
            def step(self, observation):
                received.append(None if observation is None else observation.copy())
                return super().step(observation)

        received = []
        replay_segment(RecordingLOCF(2), self.truth, self.missing)
        self.assertEqual(len(received), len(self.truth))
        self.assertIsNone(received[1])
        self.assertIsNone(received[2])
        np.testing.assert_array_equal(received[3], self.truth[3])

    def test_hidden_ground_truth_cannot_change_predictions(self):
        reference = replay_segment(LOCF(2), self.truth, self.missing)
        changed = self.truth.copy()
        changed[self.missing] = np.nan
        actual = replay_segment(LOCF(2), changed, self.missing)
        np.testing.assert_array_equal(actual.values, reference.values)
        np.testing.assert_array_equal(actual.status, reference.status)

    def test_future_measurement_cannot_change_earlier_outputs(self):
        reference = replay_segment(LOCF(2), self.truth, self.missing)
        changed = self.truth.copy()
        changed[3] = [500, 600]
        actual = replay_segment(LOCF(2), changed, self.missing)
        np.testing.assert_array_equal(actual.values[:3], reference.values[:3])
        np.testing.assert_array_equal(actual.values[3], [500, 600])

    def test_each_segment_resets_and_repeat_is_identical(self):
        predictor = LOCF(2)
        first = replay_segment(predictor, self.truth, self.missing)
        second = replay_segment(predictor, self.truth, self.missing)
        for name in ("values", "status", "missing_run"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        next_segment = replay_segment(
            predictor, np.array([[999, 999], [1, 2]]), np.array([True, False])
        )
        self.assertTrue(np.isnan(next_segment.values[0]).all())
        self.assertEqual(next_segment.status[0], "UNAVAILABLE")

    def test_metrics_match_hand_calculation_per_channel(self):
        result = replay_segment(LOCF(2), self.truth, self.missing)
        metrics = score_hidden(self.truth, self.missing, result)
        np.testing.assert_allclose(metrics["mae"], [5, 50])
        np.testing.assert_allclose(metrics["rmse"], np.sqrt(29) * np.array([1, 10]))
        self.assertEqual(metrics["scored_samples"], 2)
        self.assertEqual(metrics["hidden_samples"], 2)
        self.assertEqual(metrics["unavailable_samples"], 0)
        self.assertEqual(metrics["reconstruction_coverage"], 1)
        # Deliberately wrong outputs at received rows must not enter the score.
        result.values[~self.missing] = -100000
        self.assertEqual(score_hidden(self.truth, self.missing, result), metrics)

    def test_unavailable_rows_report_coverage_instead_of_zero_error(self):
        truth = np.array([[1.], [4.], [7.]])
        missing = np.array([True, False, True])
        result = replay_segment(LOCF(1), truth, missing)
        metrics = score_hidden(truth, missing, result)
        self.assertEqual(metrics["hidden_samples"], 2)
        self.assertEqual(metrics["scored_samples"], 1)
        self.assertEqual(metrics["unavailable_samples"], 1)
        self.assertEqual(metrics["reconstruction_coverage"], 0.5)
        self.assertEqual(metrics["mae"], [3])
        self.assertEqual(metrics["rmse"], [3])

    def test_no_reconstructions_or_no_losses_have_explicit_empty_metrics(self):
        for missing in (np.ones(5, dtype=bool), np.zeros(5, dtype=bool)):
            result = replay_segment(LOCF(2), self.truth, missing)
            metrics = score_hidden(self.truth, missing, result)
            self.assertEqual(metrics["scored_samples"], 0)
            self.assertIsNone(metrics["mae"])
            self.assertIsNone(metrics["rmse"])
            self.assertEqual(metrics["reconstruction_coverage"], 0 if missing.any() else None)

    def test_invalid_shapes_mask_and_received_nan_are_rejected(self):
        for missing in (np.array([0, 1, 1, 0, 0]), np.array([True]), np.zeros((5, 1), bool)):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                replay_segment(LOCF(2), self.truth, missing)
        with self.assertRaises(ValueError):
            replay_segment(LOCF(1), self.truth, self.missing)
        invalid = self.truth.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            replay_segment(LOCF(2), invalid, self.missing)

    def test_scoring_rejects_nonfinite_hidden_truth_and_bad_provenance(self):
        result = replay_segment(LOCF(2), self.truth, self.missing)
        changed = self.truth.copy()
        changed[1, 0] = np.nan
        with self.assertRaises(ValueError):
            score_hidden(changed, self.missing, result)
        result.status[1] = "MEASURED"
        with self.assertRaises(ValueError):
            score_hidden(self.truth, self.missing, result)


class MaskFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.config = root / "split.json"
        self.config.write_text(json.dumps({
            "train_subjects": list(range(101, 107)), "validation_subjects": [107],
            "test_subjects": [108], "supplementary_test_subjects": [109],
        }))
        self.manifest = root / "segments.csv"
        with self.manifest.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["subject", "start_row", "end_row", "samples"])
            writer.writerow(["subject107", 1001, 1030, 30])
            writer.writerow(["subject107", 2001, 2030, 30])
        self.mask_path = root / "masks.csv"
        self.rows = [
            ["validation", 107, 1001, "random_5pct", 1023],
            ["validation", 107, 1001, "random_5pct", 1026],
            ["validation", 107, 2001, "random_5pct", 2025],
            ["validation", 107, 1001, "burst_2_5pct", 1027],
            ["validation", 107, 1001, "burst_2_5pct", 1028],
        ]
        self.write_masks()

    def write_masks(self):
        with self.mask_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(MASK_FIELDS)
            writer.writerows(self.rows)

    def load(self, **changes):
        kwargs = dict(split="validation", subject=107, segment_index=0, regime="random_5pct")
        kwargs.update(changes)
        return load_segment_mask(self.mask_path, self.manifest, self.config, **kwargs)


class FrozenMaskTests(MaskFixture):
    def test_nonzero_start_row_and_exact_segment_regime_selection(self):
        original = self.mask_path.read_bytes()
        np.testing.assert_array_equal(np.flatnonzero(self.load()), [22, 25])
        np.testing.assert_array_equal(np.flatnonzero(self.load(segment_index=1)), [24])
        np.testing.assert_array_equal(
            np.flatnonzero(self.load(regime="burst_2_5pct")), [26, 27]
        )
        self.assertEqual(self.mask_path.read_bytes(), original)

    def test_duplicate_selected_row_is_rejected(self):
        self.rows.append(self.rows[0])
        self.write_masks()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load()

    def test_out_of_bounds_and_protected_positions_are_rejected(self):
        for row in (1000, 1031, 1001, 1020):
            with self.subTest(row=row):
                self.rows[0][-1] = row
                self.write_masks()
                with self.assertRaises(ValueError):
                    self.load()

    def test_unknown_or_absent_selection_is_rejected(self):
        for changes in (
            {"split": "unknown"}, {"subject": 108}, {"regime": "unknown"},
            {"regime": "burst_10_5pct"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load(**changes)
        with self.assertRaises(IndexError):
            self.load(segment_index=2)

    def test_incompatible_csv_schema_is_rejected(self):
        self.mask_path.write_text("local_index\n22\n")
        with self.assertRaisesRegex(ValueError, "fields"):
            self.load()


class SubjectMaskTests(MaskFixture):
    def complete_rows(self):
        self.rows = [
            ["validation", 107, start, regime, start + offset]
            for start in (1001, 2001) for regime in REGIMES for offset in (22, 25)
        ]
        self.write_masks()

    def load_subject(self, **changes):
        kwargs = dict(split="validation", subject=107)
        kwargs.update(changes)
        return load_subject_masks(self.mask_path, self.manifest, self.config, **kwargs)

    def test_batch_masks_equal_single_segment_reader_for_every_case(self):
        self.complete_rows()
        segments, masks = self.load_subject()
        self.assertEqual([int(row["start_row"]) for row in segments], [1001, 2001])
        self.assertEqual(len(masks), 8)
        for index in range(2):
            for regime in REGIMES:
                np.testing.assert_array_equal(
                    masks[index, regime], self.load(segment_index=index, regime=regime)
                )

    def test_incomplete_subject_masks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "No frozen-mask records"):
            self.load_subject()

    def test_bad_subject_mask_entries_are_rejected(self):
        for field, value in ((0, "train"), (2, 9999), (3, "unknown"), (4, 1001), (4, 1031)):
            with self.subTest(field=field, value=value):
                self.complete_rows()
                self.rows[0][field] = value
                self.write_masks()
                with self.assertRaises(ValueError):
                    self.load_subject()
        self.complete_rows()
        self.rows.append(self.rows[0])
        self.write_masks()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load_subject()

    def test_wrong_partition_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load_subject(subject=108)


class PooledMetricTests(unittest.TestCase):
    @staticmethod
    def metrics(values, missing):
        truth = np.asarray(values, dtype=float).reshape(-1, 1)
        mask = np.asarray(missing, dtype=bool)
        return score_hidden(truth, mask, replay_segment(LOCF(1), truth, mask))

    def test_unequal_segment_sizes_match_direct_pooled_errors(self):
        # Segment A has one error of magnitude 2; B has three errors of magnitude 4.
        first = self.metrics([0, 2], [False, True])
        second = self.metrics([0, 4, 4, 4], [False, True, True, True])
        pooled = pool_metrics([first, second])
        self.assertEqual(pooled["hidden_samples"], 4)
        self.assertEqual(pooled["absolute_error_sum"], [14])
        self.assertEqual(pooled["squared_error_sum"], [52])
        self.assertEqual(pooled["mae"], [3.5])
        np.testing.assert_allclose(pooled["rmse"], [np.sqrt(13)])
        self.assertNotEqual(pooled["mae"], [(first["mae"][0] + second["mae"][0]) / 2])
        self.assertNotEqual(pooled["rmse"], [(first["rmse"][0] + second["rmse"][0]) / 2])

    def test_unavailable_samples_remain_explicit_in_pooled_coverage(self):
        unavailable = self.metrics([5, 6], [True, True])
        available = self.metrics([0, 2], [False, True])
        pooled = pool_metrics([unavailable, available])
        self.assertEqual(pooled["hidden_samples"], 3)
        self.assertEqual(pooled["scored_samples"], 1)
        self.assertEqual(pooled["unavailable_samples"], 2)
        self.assertEqual(pooled["reconstruction_coverage"], 1 / 3)
        self.assertEqual(pooled["mae"], [2])
        self.assertEqual(pooled["rmse"], [2])

    def test_empty_pool_and_inconsistent_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            pool_metrics([])
        bad = self.metrics([0, 2], [False, True])
        bad["scored_samples"] = 0
        with self.assertRaises(ValueError):
            pool_metrics([bad])

    def test_incompatible_channels_and_nonfinite_sums_are_rejected(self):
        good = self.metrics([0, 2], [False, True])
        bad = dict(good, absolute_error_sum=[1, 2], squared_error_sum=[1, 4])
        with self.assertRaises(ValueError):
            pool_metrics([good, bad])
        for value in (float("inf"), float("nan"), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pool_metrics([dict(good, absolute_error_sum=[value])])


if __name__ == "__main__":
    unittest.main()
