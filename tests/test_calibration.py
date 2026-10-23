"""Known quantiles and split-aware calibration on tiny synthetic recordings."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from packetheal.calibration import calibrate_training_slope_caps, fit_slope_caps
from packetheal.splits import load_split_segment


class SlopeCapTests(unittest.TestCase):
    def test_known_99th_percentile_excludes_segment_boundary(self):
        first = np.array([[0.], [1.], [3.]])
        second = np.array([[100.], [103.], [107.]])
        result = fit_slope_caps([first, second], n_channels=1)
        # Sorted within-segment changes: 1, 2, 3, 4. Linear q99 = 3.97.
        np.testing.assert_allclose(result["max_delta_per_step"], [3.97])
        self.assertEqual(result["quantile_method"], "linear")
        self.assertEqual(result["quantile"], 0.99)
        self.assertEqual(result["samples"], 6)
        self.assertEqual(result["segments"], 2)
        self.assertEqual(result["pairs_per_channel"], 4)
        self.assertEqual(result["strictly_above_cap_counts"], [1])
        self.assertTrue(result["repeat_quantile_identical"])
        np.testing.assert_array_equal(first, [[0], [1], [3]])
        np.testing.assert_array_equal(second, [[100], [103], [107]])

    def test_absolute_changes_and_channel_specific_quantiles(self):
        values = np.array([[10., 100.], [8., 104.], [5., 105.]])
        result = fit_slope_caps([values], quantile=0.5, n_channels=2)
        np.testing.assert_array_equal(result["max_delta_per_step"], [2.5, 2.5])
        self.assertEqual(result["pairs_per_channel"], 2)

    def test_all_pairs_are_pooled_instead_of_averaging_segment_quantiles(self):
        result = fit_slope_caps([[[0], [1]], [[0], [10], [20], [30]]], n_channels=1)
        self.assertEqual(result["max_delta_per_step"], [10])
        self.assertEqual(result["pairs_per_channel"], 4)

    def test_constant_channels_produce_valid_zero_caps(self):
        result = fit_slope_caps([np.ones((4, 6))])
        self.assertEqual(result["max_delta_per_step"], [0] * 6)
        self.assertEqual(result["strictly_above_cap_counts"], [0] * 6)
        self.assertEqual(result["pairs_per_channel"], 3)

    def test_empty_nonfinite_complex_or_malformed_segments_are_rejected(self):
        with self.assertRaises(ValueError):
            fit_slope_caps([], n_channels=1)
        for values in ([1, 2], [[1]], np.zeros((2, 0)), [[1, 2], [3, 4]],
                       [[0], [np.nan]], [[0], [np.inf]], [[0], [1 + 1j]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                fit_slope_caps([values], n_channels=1)

    def test_invalid_quantiles_and_channel_counts_are_rejected(self):
        for quantile in (-0.1, 1.1, np.nan, np.inf, True, "0.99"):
            with self.subTest(quantile=quantile), self.assertRaises(ValueError):
                fit_slope_caps([[[0], [1]]], quantile=quantile, n_channels=1)
        for count in (0, -1):
            with self.subTest(count=count), self.assertRaises(ValueError):
                fit_slope_caps([], n_channels=count)
        for count in (True, 1.5):
            with self.subTest(count=count), self.assertRaises(TypeError):
                fit_slope_caps([], n_channels=count)

    def test_difference_overflow_is_not_silently_removed(self):
        largest = np.finfo(np.float64).max
        with self.assertRaises(FloatingPointError):
            fit_slope_caps([[[-largest], [largest]]], n_channels=1)


class TrainingCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.dataset = root / "dataset"
        protocol = self.dataset / "Protocol"
        protocol.mkdir(parents=True)
        self.config = root / "split.json"
        self.groups = {
            "train_subjects": list(range(101, 107)), "validation_subjects": [107],
            "test_subjects": [108], "supplementary_test_subjects": [109],
        }
        self.config.write_text(json.dumps(self.groups))
        self.manifest = root / "segments.csv"
        self.rows = []
        for subject in range(101, 110):
            self.rows.append(self.record(subject, 1, 3))
            if subject > 106:
                continue  # No validation/test raw files exist in this fixture.
            with (protocol / f"subject{subject}.dat").open("w") as stream:
                for original_row in range(1, 8 if subject == 101 else 4):
                    fields = ["0"] * 54
                    fields[0] = str((original_row - 1) * 0.01)
                    for channel, column in enumerate((4, 5, 6, 10, 11, 12)):
                        if original_row == 4:
                            fields[column] = "nan"  # Outside eligible segments.
                        else:
                            offset = original_row - 1 if original_row < 4 else original_row - 5
                            base = subject if original_row < 4 else 1000
                            fields[column] = str(base + offset * (channel + 1))
                    stream.write(" ".join(fields) + "\n")
        self.rows.append(self.record(101, 5, 7))
        self.write_manifest()

    @staticmethod
    def record(subject, start, end):
        return {
            "subject": f"subject{subject}", "source_file": f"subject{subject}.dat",
            "start_row": start, "end_row": end, "samples": end - start + 1,
            "start_timestamp_s": (start - 1) * 0.01, "end_timestamp_s": (end - 1) * 0.01,
        }

    def write_manifest(self):
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(self.rows)

    def calibrate(self, **kwargs):
        return calibrate_training_slope_caps(self.dataset, self.manifest, self.config, **kwargs)

    def test_real_split_aware_loader_reads_only_training_segments(self):
        with patch("packetheal.calibration.load_split_segment", wraps=load_split_segment) as loader:
            result = self.calibrate()
        self.assertEqual(result["fit_split"], "train")
        self.assertEqual(result["subjects"], list(range(101, 107)))
        self.assertEqual(result["segments"], 7)
        self.assertEqual(result["samples"], 21)
        self.assertEqual(result["pairs_per_channel"], 14)
        self.assertEqual(result["max_delta_per_step"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(result["per_subject"]["101"], {"segments": 2, "samples": 6, "pairs_per_channel": 4})
        self.assertEqual(len(loader.call_args_list), 7)
        self.assertTrue(all(call.kwargs["split"] == "train" for call in loader.call_args_list))
        self.assertEqual([call.kwargs["subject"] for call in loader.call_args_list], [101, 101, 102, 103, 104, 105, 106])
        self.assertEqual([call.kwargs["segment_index"] for call in loader.call_args_list[:2]], [0, 1])

    def test_unsorted_manifest_preserves_loader_indices_and_caps(self):
        self.rows.reverse()
        self.write_manifest()
        self.assertEqual(self.calibrate()["max_delta_per_step"], [1, 2, 3, 4, 5, 6])

    def test_changed_training_partition_is_rejected_before_raw_loading(self):
        self.groups["train_subjects"] = [101, 102, 103, 104, 105, 107]
        self.groups["validation_subjects"] = [106]
        self.config.write_text(json.dumps(self.groups))
        with patch("packetheal.calibration.load_split_segment") as loader:
            with self.assertRaisesRegex(ValueError, "101–106"):
                self.calibrate()
            loader.assert_not_called()

    def test_missing_training_subject_is_rejected(self):
        self.rows = [row for row in self.rows if row["subject"] != "subject106"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "subject 106"):
            self.calibrate()

    def test_overlapping_segments_and_wrong_source_identity_are_rejected(self):
        self.rows[-1] = self.record(101, 1, 3)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.calibrate()
        self.rows[-1] = self.record(101, 5, 7)
        self.rows[0]["source_file"] = "subject107.dat"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "identity"):
            self.calibrate()

    def test_unexpected_loaded_length_is_rejected(self):
        with patch("packetheal.calibration.load_split_segment", return_value=(np.zeros(2), np.zeros((2, 6)))):
            with self.assertRaisesRegex(ValueError, "length"):
                self.calibrate()


if __name__ == "__main__":
    unittest.main()
