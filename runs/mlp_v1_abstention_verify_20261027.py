"""Independent abstention saved arithmetic and coverage/CSV verification."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/mlp_v1_abstention_20261027"
PARENT = ROOT / "runs/mlp_v1_received_warning_20261027_retry1"
REGIMES = ("random_5pct", "burst_2_5pct", "burst_5_5pct", "burst_10_5pct")
CHANNELS = ("acc_x_m_s2", "acc_y_m_s2", "acc_z_m_s2", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s")
NAMES = tuple(f"residual_seed{s}" for s in range(3))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def close(a, b):
    if a is None or b is None:
        assert a is None and b is None
    else:
        assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)


def check_selective(actual, all_positions, accepted):
    assert actual["hidden"] == all_positions["hidden"]
    assert actual["scored"] == accepted["scored"]
    assert actual["unavailable"] == all_positions["hidden"] - accepted["scored"]
    n = actual["scored"]
    for c in range(6):
        close(actual["sum_abs"][c], accepted["sum_abs"][c]); close(actual["sum_sq"][c], accepted["sum_sq"][c])
        close(actual["mae"][c], accepted["sum_abs"][c]/n if n else None)
        close(actual["rmse"][c], math.sqrt(accepted["sum_sq"][c]/n) if n else None)


def check_pool(records, actual):
    for key in ("hidden", "scored", "unavailable"):
        assert sum(r[key] for r in records) == actual[key]
    for key in ("sum_abs", "sum_sq"):
        for c in range(6): close(math.fsum(r[key][c] for r in records), actual[key][c])


def main():
    output = OUT / "independent_checks.json"
    if output.exists(): raise FileExistsError(output)
    raw_attempts = []
    def deny_raw(event, args):
        if event == "open" and args and not isinstance(args[0], int):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.suffix == ".dat" or path.is_relative_to(ROOT / "data/raw"):
                raw_attempts.append(str(path)); raise PermissionError("Saved verifier cannot open recordings")
    sys.addaudithook(deny_raw)
    report = json.loads((OUT / "report.json").read_text()); policy = json.loads((OUT / "policy.json").read_text())
    assert report["policy"] == policy and report["correctness_pass"]
    assert report["application_acceptance"] == policy["application_acceptance"] == "NOT_ASSESSED"
    assert policy["minimum_application_coverage"] is None and policy["application_error_limits"] is None
    for relative, expected in {**policy["sources"], **policy["parent_hashes"], **policy["frozen_inputs"], **report["original_source_hashes"]}.items():
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT) and path.suffix != ".dat" and digest(path) == expected
    for name, expected in report["checkpoint_hashes"].items():
        assert digest(ROOT / "runs/mlp_v1_train_20261027/models" / name) == expected
    assert digest(PARENT / "evidence.sha256") == policy["parent_manifest_sha256"]
    parent = json.loads((PARENT / "report.json").read_text())
    parent_cases = {(r["algorithm"], r["segment"], r["regime"]): r for r in parent["cases"]}
    cases = report["cases"]
    indexed = {(r["algorithm"], r["segment"], r["regime"]): r for r in cases}
    assert len(cases) == len(indexed) == 1224
    assert set(indexed) == {(n, i, r) for n in NAMES for i in range(102) for r in REGIMES}
    for row in cases:
        assert row["subject"] == 107
        original = parent_cases[row["algorithm"], row["segment"], row["regime"]]
        check_selective(row["metrics"], original["groups"]["all"], original["groups"]["unwarned"])
        check_pool(list(row["by_age"].values()), row["metrics"])
        for age, actual in row["by_age"].items():
            check_selective(actual, original["by_age"][age]["all"], original["by_age"][age]["unwarned"])
    coverage, summaries = [], []
    std = json.loads((ROOT / "data/processed/pamap2_v1/normalization.json").read_text())["std_sample"]
    for name in NAMES:
        for regime in REGIMES:
            actual = report["pooled"][name][regime]; groups = parent["pooled"][name][regime]
            check_selective(actual, groups["all"], groups["unwarned"])
            check_pool([indexed[name, i, regime]["metrics"] for i in range(102)], actual)
            check_pool(list(report["by_age"][name][regime].values()), actual)
            for age, by_age in report["by_age"][name][regime].items():
                check_selective(by_age, parent["by_age"][name][regime][age]["all"], parent["by_age"][name][regime][age]["unwarned"])
                check_pool([indexed[name, i, regime]["by_age"][age] for i in range(102) if age in indexed[name, i, regime]["by_age"]], by_age)
            if name == NAMES[0]:
                coverage.append({"regime": regime, "hidden": actual["hidden"], "accepted": actual["scored"],
                                 "unavailable": actual["unavailable"], "coverage": actual["scored"]/actual["hidden"]})
        original_mae = math.fsum(parent["pooled"][name][r]["all"]["mae"][c]/std[c] for r in REGIMES for c in range(6))/24
        accepted_mae = math.fsum(report["pooled"][name][r]["mae"][c]/std[c] for r in REGIMES for c in range(6))/24
        summaries.append({"algorithm": name, "original_normalized_mae": original_mae,
                          "accepted_normalized_mae": accepted_mae,
                          "conditional_change_percent": 100*(accepted_mae/original_mae-1)})
    runtime = report["runtime_checks"]
    assert len(runtime) == 24
    assert {(r["algorithm"], r["segment"], r["regime"]) for r in runtime} == {(n, i, r) for n in NAMES for i in (0, 92) for r in REGIMES}
    for row in runtime:
        expected = indexed[row["algorithm"], row["segment"], row["regime"]]["metrics"]
        for k in ("hidden", "scored", "unavailable"): assert row["metrics"][k] == expected[k]
        for k in ("sum_abs", "sum_sq", "mae", "rmse"):
            for c in range(6): close(row["metrics"][k][c], expected[k][c])
        assert row["accepted_values_exact"] and row["rejected_values_all_nan"] and row["replay_audits_pass"]
    with (OUT / "metrics.csv").open(newline="") as stream: rows = list(csv.DictReader(stream))
    assert len(rows) == 450
    identities = set()
    for row in rows:
        name, regime, age = row["algorithm"], row["regime"], row["age"]
        c = CHANNELS.index(row["channel"]); identity = name, regime, age, c
        assert identity not in identities; identities.add(identity)
        actual = report["pooled"][name][regime] if age == "all" else report["by_age"][name][regime][age]
        groups = parent["pooled"][name][regime] if age == "all" else parent["by_age"][name][regime][age]
        locf = parent["pooled"]["locf"][regime]["unwarned"] if age == "all" else parent["by_age"]["locf"][regime][age]["unwarned"]
        assert int(row["hidden"]) == actual["hidden"] and int(row["accepted"]) == actual["scored"] and int(row["unavailable"]) == actual["unavailable"]
        close(row["coverage"], actual["scored"]/actual["hidden"])
        assert row["unit"] == ("m/s^2" if c<3 else "rad/s")
        for metric in ("mae", "rmse"):
            close(row[f"accepted_{metric}"], actual[metric][c]); close(row[f"original_all_{metric}"], groups["all"][metric][c]); close(row[f"matched_locf_{metric}"], locf[metric][c])
    access = report["recording_access"]
    assert access["allowed_subjects"] == [107] and not access["denied_recordings"]
    assert {Path(p).name for p in access["opened_recordings"]} == {"subject107.dat"}
    assert report["recording_hash"] == parent["recording_hash"] and not raw_attempts
    checks = {"verification_pass": True, "full_validation_saved_cases": 1224, "fresh_runtime_cases": 24,
              "metrics_csv_rows": len(rows), "raw_recordings_opened_by_verifier": 0,
              "scope": "Independent hidden/accepted/unavailable counts, math.fsum pooling, conditional metrics, case/age partitions, CSV and bounded-runtime record consistency",
              "application_acceptance": "NOT_ASSESSED", "coverage": coverage, "summaries": summaries,
              "report_sha256": digest(OUT / "report.json"), "verifier_sha256": digest(__file__),
              "limits": "Raw runtime checks are preselected segments0/92 only; other full-validation results derive from pinned saved sums"}
    with output.open("x") as stream: json.dump(checks, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__": main()
