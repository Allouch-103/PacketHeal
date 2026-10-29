"""Runner tests use temporary synthetic recordings and arrays only."""

from pathlib import Path
import csv
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from packetheal.mlp import Normalization, MLP, MLPPredictor
from packetheal.mlp_experiment import (
    audited_replay, error_record, pool_records, recording_access_guard,
    train_models, unpack_loaded_segment,
    check_output_location, engineering_screen, REGIMES,
    iter_segments, archive_rows, verify_frozen, check_locf_reference, normalization_from_root,
)
from test_mlp import PROTOCOL
from unittest.mock import patch


class ExperimentTests(unittest.TestCase):
    def test_outputs_cannot_be_placed_inside_frozen_data_or_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for part in ("data", "configs", "benchmarks", "src", "tests", "scripts", "raw"):
                with self.assertRaises(ValueError):
                    check_output_location(root, root / "raw", root / part / "new")
            check_output_location(root, root / "raw", root / "runs" / "new")

    def test_screen_retains_bad_seed_and_rmse_regressions(self):
        def regimes(mae=1., rmse=2.):
            return {r: {"mae": [mae] * 6, "rmse": [rmse] * 6, "unavailable": 0} for r in REGIMES}
        pooled = {"locf": regimes()}
        for mode in ("direct", "residual"):
            for seed in range(3):
                pooled[f"{mode}_seed{seed}"] = regimes(.8)
        norm = Normalization(np.zeros(6), np.ones(6))
        first = engineering_screen(pooled, norm, (0, 1, 2))
        self.assertTrue(first["modes"]["residual"]["screen_pass"])
        pooled["residual_seed2"][REGIMES[0]]["rmse"][0] = 3.
        second = engineering_screen(pooled, norm, (0, 1, 2))
        self.assertFalse(second["modes"]["residual"]["screen_pass"])
        self.assertEqual(len(second["modes"]["residual"]["seed_objectives"]), 3)

    def test_access_guard_denies_other_subjects_symlinks_and_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allowed = root / "subject101.dat"
            heldout = root / "subject108.dat"
            allowed.write_text("synthetic training fixture")
            heldout.write_text("synthetic test fixture")
            alias = root / "alias.txt"
            alias.symlink_to(heldout)
            with recording_access_guard(root, [101]) as log:
                self.assertEqual(allowed.read_text(), "synthetic training fixture")
                for forbidden in (heldout, alias):
                    with self.assertRaises(PermissionError):
                        forbidden.read_text()
                with self.assertRaises(PermissionError):
                    allowed.write_text("overwrite")
            self.assertEqual(len(log["opened_recordings"]), 1)
            self.assertEqual(len(log["denied_recordings"]), 2)
            self.assertEqual(allowed.read_text(), "synthetic training fixture")

    def test_validation_guard_denies_training_and_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subject in (101, 107, 108, 109):
                (root / f"subject{subject}.dat").write_text("synthetic")
            with recording_access_guard(root, [107]):
                self.assertEqual((root / "subject107.dat").read_text(), "synthetic")
                for subject in (101, 108, 109):
                    with self.assertRaises(PermissionError):
                        (root / f"subject{subject}.dat").read_text()
        with self.assertRaises(ValueError):
            with recording_access_guard("/tmp", [108]):
                pass

    def test_metric_pooling_uses_sums_instead_of_average_rmse(self):
        first = error_record(np.zeros((1, 6)), np.ones((1, 6)) * 4, np.array([True]))
        second = error_record(np.zeros((3, 6)), np.ones((3, 6)) * 2, np.ones(3, dtype=bool))
        pooled = pool_records([first, second])
        self.assertEqual(pooled["hidden"], 4)
        np.testing.assert_array_equal(pooled["mae"], [2.5] * 6)
        np.testing.assert_allclose(pooled["rmse"], [np.sqrt(7)] * 6)

    def test_metric_empty_unavailable_and_received_exclusion(self):
        truth = np.zeros((3, 6))
        predicted = np.array([[np.nan] * 6, [2] * 6, [100] * 6])
        row = error_record(truth, predicted, np.array([True, True, False]))
        self.assertEqual((row["hidden"], row["scored"], row["unavailable"]), (2, 1, 1))
        self.assertEqual(row["mae"], [2] * 6)
        self.assertEqual(error_record(truth, predicted, np.zeros(3, dtype=bool))["mae"], [None] * 6)

    def test_loader_boundary_requires_verified_shapes_and_timestamps(self):
        times, values = np.arange(5) * .01, np.ones((5, 6))
        np.testing.assert_array_equal(unpack_loaded_segment((times, values), 5), values)
        with self.assertRaises(TypeError):
            unpack_loaded_segment({"timestamps": times, "data": values}, 5)
        with self.assertRaises(TypeError):
            unpack_loaded_segment(values, 5)
        with self.assertRaises(ValueError):
            unpack_loaded_segment((times * 2, values), 5)
        with self.assertRaises(ValueError):
            unpack_loaded_segment((times, values[:, :3]), 5)

    def test_existing_loader_and_masks_with_temporary_recordings(self):
        """Exercise actual split-aware loader, CSV parser, and Protocol path."""
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Protocol").mkdir()
            (root / "configs").mkdir()
            (root / "data/processed/pamap2_v1").mkdir(parents=True)
            config = json.loads((repository / "configs/pamap2_split_v1.json").read_text())
            (root / "configs/pamap2_split_v1.json").write_text(json.dumps(config))
            manifest_rows, mask_rows = [], []
            expected_truth = np.arange(36 * 6, dtype=float).reshape(36, 6) / 100
            for subject, split in ((101, "train"), (107, "validation")):
                rows = np.zeros((36, 54))
                rows[:, 0] = np.arange(36) * .01
                rows[:, [4, 5, 6, 10, 11, 12]] = expected_truth
                np.savetxt(root / "Protocol" / f"subject{subject}.dat", rows)
                manifest_rows.append({"subject": f"subject{subject}", "source_file": f"subject{subject}.dat",
                                      "start_row": 1, "end_row": 36, "samples": 36,
                                      "start_timestamp_s": 0, "end_timestamp_s": .35})
                for regime, length in zip(REGIMES, (1, 2, 5, 10)):
                    mask_rows.extend({"split": split, "subject": subject, "segment_start_row": 1,
                                      "regime": regime, "missing_original_row": row} for row in range(23, 23 + length))
            with (root / "configs/pamap2_segments_v1.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
                writer.writeheader()
                writer.writerows(manifest_rows)
            with (root / "data/processed/pamap2_v1/loss_masks.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(mask_rows[0]))
                writer.writeheader()
                writer.writerows(mask_rows)
            for subject, split in ((101, "train"), (107, "validation")):
                with recording_access_guard(root / "Protocol", [subject]) as access:
                    iterator = iter_segments(root, root, split)
                    actual_subject, index, metadata, truth, masks = next(iterator)
                    iterator.close()
                self.assertEqual((actual_subject, index), (subject, 0))
                np.testing.assert_array_equal(truth, expected_truth)
                self.assertEqual([int(masks[r].sum()) for r in REGIMES], [1, 2, 5, 10])
                self.assertEqual(access["denied_recordings"], [])
                self.assertEqual(access["opened_recordings"], [str(root / "Protocol" / f"subject{subject}.dat")])

    def test_actual_frozen_metadata_and_archived_csv_without_raw_opens(self):
        root = Path(__file__).resolve().parents[1]
        frozen = verify_frozen(root)
        self.assertEqual(len(frozen), 7)
        self.assertEqual(len(archive_rows(root)), 72)
        self.assertEqual(normalization_from_root(root).std.shape, (6,))
        pooled = {}
        for regime in REGIMES:
            rows = [row for row in archive_rows(root) if row["algorithm"] == "locf" and row["regime"] == regime]
            pooled[regime] = {"mae": [float(row["mae"]) for row in rows],
                              "rmse": [float(row["rmse"]) for row in rows],
                              "hidden": int(rows[0]["hidden_count"]), "scored": int(rows[0]["scored_count"]),
                              "unavailable": int(rows[0]["unavailable_count"])}
        check_locf_reference(root, pooled)
        pooled[REGIMES[0]]["hidden"] += 1
        with self.assertRaises(AssertionError):
            check_locf_reference(root, pooled)

    def test_training_seeds_modes_are_complete_and_repeatable(self):
        rng = np.random.default_rng(12)
        features = rng.normal(size=(32, 5)).astype(np.float32)
        targets = {mode: np.full((32, 6), .2, dtype=np.float32) for mode in ("direct", "residual")}
        ids = np.repeat(np.arange(4), 8)
        with tempfile.TemporaryDirectory() as directory:
            first = train_models(features, targets, ids, Path(directory) / "a", epochs=3, batch_size=16, seeds=(0, 1), progress=lambda s: None)
            second = train_models(features, targets, ids, Path(directory) / "b", epochs=3, batch_size=16, seeds=(0, 1), progress=lambda s: None)
            self.assertEqual([(r["mode"], r["seed"]) for r in first], [("direct", 0), ("direct", 1), ("residual", 0), ("residual", 1)])
            for a, b in zip(first, second):
                self.assertEqual(a["sampled_epoch_mae"], b["sampled_epoch_mae"])
                self.assertEqual(a["final_equal_regime_normalized_mae"], b["final_equal_regime_normalized_mae"])
                left, right = MLP.load(Path(directory) / "a" / a["checkpoint"]), MLP.load(Path(directory) / "b" / b["checkpoint"])
                for p, q in zip(left.parameters, right.parameters):
                    np.testing.assert_array_equal(p, q)
            with self.assertRaises(FileExistsError):
                train_models(features, targets, ids, Path(directory) / "a", epochs=1, seeds=(0,))

    def test_training_rejects_missing_regime_and_invalid_targets(self):
        x = np.ones((8, 3))
        y = {mode: np.ones((8, 6)) for mode in ("direct", "residual")}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                train_models(x, y, np.zeros(8), Path(directory) / "empty")
            with self.assertRaises(ValueError):
                train_models(x, y, np.repeat(np.arange(4), 2), Path(directory) / "badbatch", batch_size=5)

    def test_full_replay_audits_on_nonzero_corrections(self):
        truth = np.arange(240, dtype=float).reshape(40, 6)
        missing = np.zeros(40, dtype=bool)
        missing[22:28] = True
        norm = Normalization(np.zeros(6), np.ones(6))
        model = MLP()
        model.parameters[-1][:] = .2
        with patch.dict(sys.modules, {"packetheal.baselines": PROTOCOL}):
            values, statuses, ages = audited_replay(MLPPredictor(model, norm), truth, missing)
        np.testing.assert_array_equal(ages[22:28], np.arange(1, 7))
        np.testing.assert_array_equal(statuses[missing], "RECONSTRUCTED")
        np.testing.assert_allclose(values[22:28], np.broadcast_to(truth[21] + np.float32(.2), (6, 6)))


if __name__ == "__main__":
    unittest.main()
