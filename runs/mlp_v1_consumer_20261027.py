"""Small desktop consumer demo. Synthetic inputs/model only; no training."""

from dataclasses import asdict, dataclass
import json
from numbers import Integral
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from packetheal.baselines import SampleStatus
from packetheal.mlp import MLP, Normalization
from mlp_v1_abstention_20261027 import AbstainingMLP


@dataclass(frozen=True)
class ConsumerRecord:
    sample_index: int
    status: str
    missing_run: int
    values: tuple | None


class StatusConsumer:
    """Preserve sample positions and provenance; gaps contain no numeric value.

    This is a log/display consumer, not a control or filtering policy. Each
    consume call represents one nominal sample slot; reset at segment boundaries.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.records = []

    def consume(self, result):
        status = result.status
        if not isinstance(status, SampleStatus):
            raise ValueError("Expected an explicit SampleStatus")
        age = result.missing_run
        if isinstance(age, bool) or not isinstance(age, Integral) or age < 0:
            raise ValueError("Expected a nonnegative integer loss age")
        if (status == SampleStatus.MEASURED) != (age == 0):
            raise ValueError("Only a measured sample may have loss age zero")

        # Branch BEFORE reading values: unavailable payloads are never consumed.
        if status == SampleStatus.UNAVAILABLE:
            values = None
        else:
            array = np.asarray(result.values, dtype=np.float64)
            if array.shape != (6,) or not np.isfinite(array).all():
                raise ValueError("Measured/estimated values must be finite [6]")
            values = tuple(float(value) for value in array)

        record = ConsumerRecord(len(self.records), status.value, int(age), values)
        self.records.append(record)
        return record

    def grouped(self):
        """Separate routes for measurements, estimates, and gaps."""
        return {status.value: [r for r in self.records if r.status == status.value]
                for status in SampleStatus}


def synthetic_demo():
    # Deliberately artificial model: output a fixed normalized residual +0.25.
    # This makes estimates visibly different from measurements. No fitted model
    # or benchmark accuracy is represented by this demonstration.
    model = MLP(mode="residual", seed=0)
    model.parameters[-1].fill(.25)
    predictor = AbstainingMLP(model, Normalization(np.zeros(6), np.ones(6)), [1, 1, 1])
    consumer = StatusConsumer()
    base = np.array([0., 0., 9.81, 0., 0., 0.])
    spike = base.copy()
    # Previous real sample is three slots earlier: 4/3 > warning threshold 1.
    spike[0] = 4.
    observations = ([None] + [base.copy() for _ in range(20)] + [None, None]
                    + [spike.copy(), None, None, spike.copy()]
                    + [None] * 11 + [base.copy(), None])
    for observation in observations:
        consumer.consume(predictor.step(observation))
    groups = consumer.grouped()
    counts = {status: len(records) for status, records in groups.items()}
    expected = {"MEASURED": 23, "RECONSTRUCTED": 13, "UNAVAILABLE": 4}
    if counts != expected:
        raise AssertionError((counts, expected))
    if [r.sample_index for r in groups["UNAVAILABLE"]] != [0, 24, 25, 37]:
        raise AssertionError("Gap positions changed")
    return {
        "scope": "synthetic software integration; artificial fixed-output MLP; no training",
        "channel_order": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "units": ["m/s^2"] * 3 + ["rad/s"] * 3,
        "thresholds": "synthetic [1,1,1]; not changed in any real experiment",
        "counts": counts,
        "records": [asdict(record) for record in consumer.records],
    }


if __name__ == "__main__":
    def deny_raw_open(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, Path)):
            path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
            if path.is_relative_to(ROOT / "data/raw"):
                raise PermissionError("This synthetic demo must not open raw data")
    sys.addaudithook(deny_raw_open)
    report = synthetic_demo()
    output = ROOT / "runs/mlp_v1_consumer_20261027"
    output.mkdir(exist_ok=False)
    (output / "demo.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["counts"]))
