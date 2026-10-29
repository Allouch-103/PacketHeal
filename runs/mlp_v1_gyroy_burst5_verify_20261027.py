"""Independent saved-evidence arithmetic for the targeted gyro-Y failure.

No recording access, model training, or parameter changes.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/mlp_v1_gyroy_burst5_failure_20261027"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)


def main():
    path = OUT / "independent_checks.json"
    if path.exists():
        raise FileExistsError(path)
    event = json.loads((OUT / "event.json").read_text())
    policy = json.loads((OUT / "policy.json").read_text())
    assert event["policy"] == policy
    assert digest(ROOT / "runs/mlp_v1_gyroy_burst5_failure_20261027.py") == policy["diagnostic_script_sha256"]
    assert policy["subject"] == 107 and policy["regime"] == "burst_5_5pct" and policy["channel_index"] == 4
    for name, meta in policy["prior_manifests"].items():
        manifest = ROOT / "runs" / name
        assert digest(manifest) == meta["sha256"]
        for line in manifest.read_text().splitlines():
            expected, relative = line.split(maxsplit=1)
            referenced = (ROOT / relative).resolve()
            assert referenced.is_relative_to(ROOT) and referenced.suffix != ".dat"
            assert digest(referenced) == expected
    for source, expected in event["source_hashes"].items():
        assert digest(ROOT / source) == expected
    for checkpoint, expected in event["checkpoint_hashes"].items():
        assert digest(ROOT / "runs/mlp_v1_train_20261027/models" / checkpoint) == expected
    strata_path = ROOT / "runs/mlp_v1_transients_20261027/report.json"
    assert digest(strata_path) == policy["strata_sha256"]
    report = json.loads(strata_path.read_text())
    primary = f"residual_seed{policy['seed']}"
    c, regime, segment = 4, policy["regime"], policy["segment"]
    base = report["pooled"]["locf"][regime]["large_change"]
    ratios = {seed: report["pooled"][f"residual_seed{seed}"][regime]["large_change"]["rmse"][c] / base["rmse"][c]
              for seed in range(3)}
    assert policy["seed"] == max(ratios, key=lambda s: (ratios[s], -s))
    with (OUT / "segment_contributions.csv").open(newline="") as stream:
        contributions = list(csv.DictReader(stream))
    assert len(contributions) == 102 and {int(r["segment"]) for r in contributions} == set(range(102))
    assert sum(int(r["flagged_count"]) for r in contributions) == 49
    assert int(max(contributions, key=lambda r: (float(r["excess_squared_error_sum"]), -int(r["segment"])))["segment"]) == segment
    for row in contributions:
        i = int(row["segment"])
        for name, key in (("locf", "locf_squared_error_sum"), (primary, "primary_squared_error_sum")):
            case = next(r for r in report["cases"] if (r["algorithm"], r["regime"], r["segment"]) == (name, regime, i))
            close(row[key], case["strata"]["large_change"]["squared_error_sum"][c])
            assert int(row["flagged_count"]) == case["strata"]["large_change"]["scored_count"][c]
        close(row["excess_squared_error_sum"], float(row["primary_squared_error_sum"]) - float(row["locf_squared_error_sum"]))
    close(math.fsum(float(r["excess_squared_error_sum"]) for r in contributions), event["subgroup_net_excess_squared_error"])
    for name in event["subgroup_metrics"]:
        metric = event["subgroup_metrics"][name]
        n = metric["scored_count"][c]
        assert n == 49
        close(metric["mae"][c], metric["absolute_error_sum"][c] / n)
        close(metric["rmse"][c], math.sqrt(metric["squared_error_sum"][c] / n))
        if name in ("locf", primary):
            field = "locf_squared_error_sum" if name == "locf" else "primary_squared_error_sum"
            close(math.fsum(float(r[field]) for r in contributions), metric["squared_error_sum"][c])
    with (OUT / "burst.csv").open(newline="") as stream:
        burst = list(csv.DictReader(stream))
    with (OUT / "history.csv").open(newline="") as stream:
        history = list(csv.DictReader(stream))
    with (OUT / "trace.csv").open(newline="") as stream:
        trace = list(csv.DictReader(stream))
    assert len(burst) == 5 and [int(r["loss_age"]) for r in burst] == [1, 2, 3, 4, 5]
    assert [int(r["original_row"]) for r in burst] == event["burst_original_rows"]
    assert len(history) == 100
    trace_by_row = {int(r["original_row"]): r for r in trace}
    selected = next(r for r in burst if int(r["original_row"]) == event["event_original_row"])
    assert int(selected["rapid_change_gyro_y"]) == 1
    for row in burst:
        close(row["anchor"], event["anchor_value"])
        close(row["required_correction"], float(row["recorded_truth"]) - float(row["anchor"]))
        raw_row = int(row["original_row"])
        if int(row["loss_age"]) > 1:
            previous = burst[int(row["loss_age"]) - 2]
            flagged = abs(float(row["recorded_truth"]) - float(previous["recorded_truth"])) > report["policy"]["thresholds"][4]
        else:
            flagged = abs(float(row["recorded_truth"]) - event["anchor_value"]) > report["policy"]["thresholds"][4]
        assert bool(int(row["rapid_change_gyro_y"])) == flagged
        for name in event["predictions"]:
            close(row[f"{name}_signed_error"], float(row[f"{name}_prediction"]) - float(row["recorded_truth"]))
            close(row[f"{name}_correction"], float(row[f"{name}_prediction"]) - event["anchor_value"])
            close(row[f"{name}_prediction"], trace_by_row[raw_row][f"{name}_prediction"])
            if row is selected:
                close(row[f"{name}_prediction"], event["predictions"][name])
                close(abs(float(row[f"{name}_signed_error"])), event["absolute_errors"][name])
    close(event["event_excess_squared_error"], float(selected[f"{primary}_signed_error"]) ** 2 - float(selected["locf_signed_error"]) ** 2)
    close(event["event_fraction_of_subgroup_net_increase"], event["event_excess_squared_error"] / event["subgroup_net_excess_squared_error"])
    flagged_burst_excess = math.fsum(float(r[f"{primary}_signed_error"]) ** 2 - float(r["locf_signed_error"]) ** 2
                                    for r in burst if int(r["rapid_change_gyro_y"]))
    with np.load(OUT / "features.npz", allow_pickle=False) as arrays:
        features = arrays["features"]
        assert features.shape == (5, 161) and features.dtype == np.float32
        np.testing.assert_array_equal(features[:, -1], np.array([.1, .2, .3, .4, .5], dtype=np.float32))
        for offset, row in enumerate(burst):
            slots = sorted((h for h in history if int(h["prediction_original_row"]) == int(row["original_row"])), key=lambda h: int(h["slot"]))
            assert len(slots) == 20
            for slot, entry in enumerate(slots):
                assert int(entry["last_real_original_row"]) <= int(entry["slot_original_row"]) < int(row["original_row"])
                assert int(entry["slot_original_row"]) == int(row["original_row"]) - 20 + slot
                for channel, name in enumerate(("acc_x_m_s2", "acc_y_m_s2", "acc_z_m_s2", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s")):
                    close(entry[f"z_{name}"], (float(entry[f"held_{name}"]) - arrays["normalization_mean"][channel]) / arrays["normalization_std"][channel])
                    assert features[offset, slot * 8 + channel] == np.float32(entry[f"z_{name}"])
                assert features[offset, slot * 8 + 6] == float(entry["received"])
                assert features[offset, slot * 8 + 7] == 1
        for name in ("residual_seed0", "residual_seed1", "residual_seed2"):
            for offset, row in enumerate(burst):
                z = arrays[f"{name}_normalized_corrections"][offset]
                correction = arrays[f"{name}_physical_corrections"][offset]
                np.testing.assert_array_equal(z.astype(np.float64) * arrays["normalization_std"], correction)
                close(arrays["anchor"][4] + correction[4], row[f"{name}_prediction"])
    access = event["recording_access"]
    assert access["allowed_subjects"] == [107] and not access["denied_recordings"]
    assert {Path(p).name for p in access["opened_recordings"]} == {"subject107.dat"}
    checks = {"verification_pass": True, "raw_recordings_opened_by_verifier": 0,
              "segment_contribution_rows": 102, "burst_rows": 5, "history_rows": 100,
              "flagged_burst_excess_squared_error": flagged_burst_excess,
              "flagged_burst_fraction_of_subgroup_net_increase": flagged_burst_excess / event["subgroup_net_excess_squared_error"],
              "saved_checks": "Selection, subgroup sums and formulas, all burst predictions/errors, history/feature CSV parity, normalization and correction arithmetic",
              "limits": "Independent saved arithmetic; full raw replay and layer checks are in event.json, not duplicated here",
              "event_sha256": digest(OUT / "event.json"), "verification_script_sha256": digest(__file__)}
    with path.open("x") as stream:
        json.dump(checks, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
