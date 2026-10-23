"""Scientific contract checks for the single-segment comparison runner."""

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
from evaluate_linear_extrapolation import (
    check_locf_reference, checked_linear_replay, load_frozen_caps, main,
)
from evaluate_locf import sha256
from evaluate_locf_batch import replay_digest
from packetheal.baselines import LOCF
from packetheal.evaluation import replay_segment, score_hidden


class FrozenCapContractTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "caps.json"
        self.config = {
            "channels": ["ax", "ay", "az", "gx", "gy", "gz"],
            "train_subjects": list(range(101, 107)), "sampling_frequency_hz": 100,
        }
        self.preparation = {"splits": {"train": {
            "subjects": list(range(101, 107)), "segments": 6, "samples": 12,
        }}}
        self.caps = {
            "schema_version": "1", "purpose": "linear_extrapolation_slope_caps",
            "fit_split": "train", "subjects": list(range(101, 107)),
            "quantile": .99, "quantile_method": "linear", "n_channels": 6,
            "channel_order": self.config["channels"], "sampling_frequency_hz": 100,
            "cap_units": ["m/s² per sample step"] * 3 + ["rad/s per sample step"] * 3,
            "normalization_applied": False, "loss_masks_applied": False,
            "segments": 6, "samples": 12, "pairs_per_channel": 6,
            "per_subject": {str(s): {"segments": 1, "samples": 2, "pairs_per_channel": 1}
                            for s in range(101, 107)},
            "repeat_quantile_identical": True, "verification": {"counts_match_preparation": True},
            "max_delta_per_step": [1, 2, 3, 4, 5, 6],
        }

    def write_artifact(self, caps):
        self.path.write_text(json.dumps(caps))
        checksum = sha256(self.path)
        self.path.with_suffix(".json.sha256").write_text(f"{checksum}  {self.path.name}\n")
        return checksum

    def load(self, checksum):
        return load_frozen_caps(self.path, self.config, self.preparation,
                                expected_sha256=checksum)

    def test_valid_training_contract_preserves_caps(self):
        checksum = self.write_artifact(self.caps)
        self.assertEqual(self.load(checksum), self.caps)
        self.assertEqual(sha256(self.path), checksum)

    def test_editing_both_caps_and_sidecar_cannot_bypass_pinned_hash(self):
        original = self.write_artifact(self.caps)
        self.caps["max_delta_per_step"][0] += 1
        self.write_artifact(self.caps)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.load(original)

    def test_wrong_sidecar_filename_is_rejected(self):
        checksum = self.write_artifact(self.caps)
        self.path.with_suffix(".json.sha256").write_text(f"{checksum}  other.json\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.load(checksum)

    def test_validation_or_test_calibration_and_changed_policy_are_rejected(self):
        for key, value in (("fit_split", "validation"), ("subjects", [107]),
                           ("subjects", [108]), ("quantile", .95),
                           ("quantile_method", "nearest"), ("normalization_applied", True),
                           ("loss_masks_applied", True), ("sampling_frequency_hz", 50),
                           ("channel_order", list(reversed(self.config["channels"])) )):
            with self.subTest(key=key, value=value):
                changed = {**self.caps, key: value}
                with self.assertRaises(ValueError):
                    self.load(self.write_artifact(changed))

    def test_count_and_per_subject_inconsistencies_are_rejected(self):
        for key in ("segments", "samples", "pairs_per_channel"):
            with self.subTest(key=key):
                changed = {**self.caps, key: self.caps[key] + 1}
                with self.assertRaises(ValueError):
                    self.load(self.write_artifact(changed))
        changed = deepcopy(self.caps)
        changed["per_subject"]["101"]["samples"] += 1
        with self.assertRaisesRegex(ValueError, "per-subject total"):
            self.load(self.write_artifact(changed))

    def test_invalid_cap_vectors_are_rejected(self):
        for limits in ([1] * 5, [-1] * 6, [float("nan")] * 6, [float("inf")] * 6):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                self.load(self.write_artifact({**self.caps, "max_delta_per_step": limits}))


class LinearComparisonTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.repeat(np.array([[0.], [1.], [4.], [9.]]), 6, axis=1)
        self.missing = np.array([False, False, True, True])
        self.segment = {"subject": "subject107", "start_row": "1", "end_row": "4"}
        self.locf = replay_segment(LOCF(), self.truth, self.missing)
        self.metrics = score_hidden(self.truth, self.missing, self.locf)
        self.case = {
            "segment_index": 0, "regime": "random_5pct", "segment": self.segment,
            "replay_sha256": replay_digest(self.locf), "metrics": self.metrics,
        }
        self.archive = {"predictor": "LOCF", "split": "validation", "subject": 107,
                        "per_segment": [self.case]}

    def check_reference(self, result=None, metrics=None):
        return check_locf_reference(self.archive, 0, "random_5pct", self.segment,
                                    self.locf if result is None else result,
                                    self.metrics if metrics is None else metrics)

    def test_checked_replay_has_known_errors_and_preserves_inputs(self):
        original = self.truth.copy()
        result, metrics = checked_linear_replay([10] * 6, self.truth, self.missing)
        np.testing.assert_array_equal(result.values[:, 0], [0, 1, 2, 3])
        np.testing.assert_array_equal(metrics["mae"], [4] * 6)
        np.testing.assert_allclose(metrics["rmse"], [np.sqrt(20)] * 6)
        self.assertEqual(metrics["scored_samples"], 2)
        self.assertEqual(metrics["unavailable_samples"], 0)
        np.testing.assert_array_equal(self.truth, original)
        self.assertIs(self.check_reference(), self.case)

    def test_same_loss_count_at_different_positions_does_not_match_archive(self):
        shifted = np.array([False, True, False, True])
        replay = replay_segment(LOCF(), self.truth, shifted)
        self.assertEqual(shifted.sum(), self.missing.sum())
        with self.assertRaisesRegex(ValueError, "differs"):
            self.check_reference(replay, score_hidden(self.truth, shifted, replay))

    def test_wrong_partition_missing_duplicate_or_changed_reference_rejected(self):
        variants = [
            {**self.archive, "subject": 108},
            {**self.archive, "per_segment": []},
            {**self.archive, "per_segment": [self.case, self.case]},
            {**self.archive, "per_segment": [{**self.case, "metrics": {}}]},
            {**self.archive, "per_segment": [{**self.case, "segment": {}}]},
        ]
        for archive in variants:
            with self.subTest(archive=archive), self.assertRaises(ValueError):
                check_locf_reference(archive, 0, "random_5pct", self.segment,
                                     self.locf, self.metrics)

    def test_cli_does_not_allow_subject_or_test_partition_selection(self):
        for options in (["--subject", "108"], ["--split", "test"]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(options)
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
