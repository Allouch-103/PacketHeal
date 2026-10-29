"""Consumer contract tests; no raw recordings or fitted checkpoints."""

import json
import unittest
from types import SimpleNamespace

import numpy as np

from mlp_v1_consumer_20261027 import StatusConsumer, synthetic_demo
from packetheal.baselines import SampleStatus, StepResult


class ConsumerTests(unittest.TestCase):
    def test_unavailable_payload_is_never_read(self):
        class ForbiddenPayload:
            status = SampleStatus.UNAVAILABLE
            missing_run = 1

            @property
            def values(self):
                raise AssertionError("UNAVAILABLE values were accessed")

        consumer = StatusConsumer()
        self.assertIsNone(consumer.consume(ForbiddenPayload()).values)
        # Also ignore finite placeholders: status, not NaN detection, decides.
        result = StepResult(np.full(6, 999.), SampleStatus.UNAVAILABLE, 2)
        self.assertIsNone(consumer.consume(result).values)

    def test_routes_keep_zero_measurements_and_estimates_distinct(self):
        consumer = StatusConsumer()
        consumer.consume(StepResult(np.zeros(6), SampleStatus.MEASURED, 0))
        consumer.consume(StepResult(np.ones(6), SampleStatus.RECONSTRUCTED, 1))
        consumer.consume(StepResult(np.full(6, np.nan), SampleStatus.UNAVAILABLE, 2))
        groups = consumer.grouped()
        self.assertEqual([r.sample_index for r in groups["MEASURED"]], [0])
        self.assertEqual(groups["MEASURED"][0].values, (0.,) * 6)
        self.assertEqual([r.sample_index for r in groups["RECONSTRUCTED"]], [1])
        self.assertEqual([r.sample_index for r in groups["UNAVAILABLE"]], [2])

    def test_bad_input_does_not_append_or_advance(self):
        consumer = StatusConsumer()
        bad = [SimpleNamespace(status="MEASURED", missing_run=0, values=np.zeros(6)),
               StepResult(np.zeros(6), SampleStatus.MEASURED, 1),
               StepResult(np.zeros(6), SampleStatus.UNAVAILABLE, 0),
               StepResult(np.zeros(6), SampleStatus.UNAVAILABLE, True),
               StepResult(np.zeros(6), SampleStatus.UNAVAILABLE, -1),
               StepResult(np.zeros(5), SampleStatus.MEASURED, 0),
               StepResult(np.full(6, np.nan), SampleStatus.RECONSTRUCTED, 1)]
        for result in bad:
            with self.assertRaises(ValueError):
                consumer.consume(result)
            self.assertEqual(consumer.records, [])

    def test_output_ownership_and_reset(self):
        consumer = StatusConsumer()
        original = np.arange(6, dtype=float)
        record = consumer.consume(StepResult(original, SampleStatus.MEASURED, 0))
        original[:] = 99
        self.assertEqual(record.values, tuple(range(6)))
        consumer.reset()
        result = consumer.consume(StepResult(np.zeros(6), SampleStatus.MEASURED, 0))
        self.assertEqual(result.sample_index, 0)

    def test_all_gaps_are_explicit_without_fake_zero_values(self):
        consumer = StatusConsumer()
        for age in range(1, 4):
            consumer.consume(StepResult(np.full(6, np.nan), SampleStatus.UNAVAILABLE, age))
        groups = consumer.grouped()
        self.assertEqual(groups["MEASURED"], [])
        self.assertEqual(groups["RECONSTRUCTED"], [])
        self.assertEqual([r.values for r in groups["UNAVAILABLE"]], [None] * 3)

    def test_end_to_end_synthetic_stream(self):
        report = synthetic_demo()
        rows = report["records"]
        self.assertEqual(report["counts"], {"MEASURED": 23, "RECONSTRUCTED": 13, "UNAVAILABLE": 4})
        self.assertEqual([r["sample_index"] for r in rows], list(range(40)))
        for i in [0, 24, 25, 37]:
            self.assertIsNone(rows[i]["values"])
        self.assertEqual(rows[23]["status"], "MEASURED")  # spike preserved
        self.assertEqual(rows[23]["values"][0], 4.)
        self.assertEqual(rows[21]["values"][0], .25)  # visible estimate
        self.assertEqual(rows[36]["missing_run"], 10)
        self.assertEqual(rows[37]["missing_run"], 11)
        self.assertEqual(rows[39]["status"], "RECONSTRUCTED")  # recovery
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
