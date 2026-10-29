"""Saved-sum evaluation of fixed LOCF-on-warning routing; no raw data/inference."""

from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from packetheal.mlp_experiment import CHANNELS, REGIMES, digest, pool_records, verify_frozen, write_json

PARENT = ROOT / "runs/mlp_v1_received_warning_20261027_retry1"
OUT = ROOT / "runs/mlp_v1_warning_fallback_20261027"
PARENT_MANIFEST = "cf757178a3e186ca837603fe34354ca15636fdc3dfa1f3e1b1ca24f3981c4f92"
GROUPS = ("warned", "unwarned", "unknown")
FIELDS = ("hidden", "scored", "unavailable", "sum_abs", "sum_sq")


def equal_record(a, b):
    for key in FIELDS[:3]:
        assert a[key] == b[key]
    for key in FIELDS[3:]:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-12)


def route_sums(mlp, locf):
    """Groups must refer to the SAME hidden positions; never feed outputs back."""
    for group in GROUPS:
        for key in FIELDS[:3]:
            assert mlp[group][key] == locf[group][key]
    return pool_records([mlp["unwarned"], locf["warned"], locf["unknown"]])


def synthetic_route_check():
    """Compare pooled routing with direct errors on three synthetic positions."""
    scale = np.arange(1., 7.)
    mlp_errors = np.array([1., 10., 5.])[:, None] * scale
    locf_errors = np.array([2., 4., 3.])[:, None] * scale
    selected = {"unwarned": [0], "warned": [1], "unknown": [2]}
    def record(errors, ix):
        return {"hidden": len(ix), "scored": len(ix), "unavailable": 0,
                "sum_abs": np.abs(errors[ix]).sum(axis=0).tolist(),
                "sum_sq": (errors[ix] ** 2).sum(axis=0).tolist()}
    mlp = {g: record(mlp_errors, ix) for g, ix in selected.items()}
    locf = {g: record(locf_errors, ix) for g, ix in selected.items()}
    routed = route_sums(mlp, locf)
    direct = np.array([mlp_errors[0], locf_errors[1], locf_errors[2]])
    equal_record(routed, record(direct, [0, 1, 2]))
    np.testing.assert_array_equal(routed["sum_abs"], 8 * scale)
    np.testing.assert_array_equal(routed["sum_sq"], 26 * scale ** 2)
    invalid = {g: dict(v) for g, v in locf.items()}
    invalid["warned"]["hidden"] += 1
    try:
        route_sums(mlp, invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("Mismatched routing groups were accepted")
    return {"direct_pointwise_routing_matches_sums": True, "mismatched_counts_rejected": True,
            "absolute_error_sum_per_channel": routed["sum_abs"], "squared_error_sum_per_channel": routed["sum_sq"]}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    raw_attempts = []
    def forbid_raw(event, args):
        if event == "open" and args and not isinstance(args[0], int):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.suffix == ".dat" or path.is_relative_to(ROOT / "data/raw"):
                raw_attempts.append(str(path)); raise PermissionError("Saved-sum analysis must not open raw data")
    sys.addaudithook(forbid_raw)
    frozen = verify_frozen(ROOT)
    assert digest(PARENT / "evidence.sha256") == PARENT_MANIFEST
    parent_hashes = {}
    for line in (PARENT / "evidence.sha256").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT) and path.suffix != ".dat" and digest(path) == expected
        parent_hashes[name] = expected
    assert json.loads((PARENT / "independent_checks.json").read_text())["verification_pass"]
    parent = json.loads((PARENT / "report.json").read_text())
    assert parent["correctness_pass"]
    indexed = {(r["algorithm"], r["segment"], r["regime"]): r for r in parent["cases"]}
    assert len(indexed) == len(parent["cases"]) == 2856
    names = [f"residual_seed{s}" for s in range(3)]
    std = np.asarray(json.loads((ROOT / "data/processed/pamap2_v1/normalization.json").read_text())["std_sample"])
    policy = {
        "kind": "saved_sum_LOCF_on_warning_comparison", "subject": 107,
        "rule": {"warned": "LOCF", "unknown": "LOCF", "unwarned": "residual MLP"},
        "selection_unit": "Whole six-channel vector; same frozen warning for all seeds",
        "scope": "All 408 subject107 segment/regime cases for 3 residual seeds; overall and by loss age",
        "assumption": "MLP, LOCF and warning states depend only on received observations; emitted reconstructions never feed back. Source errors can therefore be recombined on disjoint groups.",
        "formula": "sum_abs/sum_sq = MLP.unwarned + LOCF.warned + LOCF.unknown; add counts before MAE and sqrt(mean squared error)",
        "primary_usefulness_pass": "Strictly lower full-coverage equal-regime/channel training-std-normalized MAE than unchanged MLP for EACH of 3 residual seeds",
        "secondary": "All channel/regime and loss-age MAE/RMSE changes, including regressions; compare full LOCF too",
        "no_tuning": "No changed warning thresholds, channel-specific routing, seed selection, training or new masks",
        "limits": "Exploratory saved-validation-sum analysis, not a new streaming hybrid implementation or hardware result; no cross-subject claim",
        "startup": "Unknown uses original LOCF availability; current protected-warmup benchmark has no unknown hidden positions",
        "parent_manifest_sha256": PARENT_MANIFEST, "parent_hashes": parent_hashes,
        "parent_report_sha256": digest(PARENT / "report.json"), "frozen_inputs": frozen,
        "runner_sha256": digest(__file__),
    }
    OUT.mkdir()
    write_json(OUT / "policy.json", policy)
    write_json(OUT / "synthetic_check.json", synthetic_route_check())
    cases = []
    for name in names:
        for segment in range(102):
            for regime in REGIMES:
                mlp, locf = indexed[name, segment, regime], indexed["locf", segment, regime]
                for original in (mlp, locf):
                    equal_record(pool_records([original["groups"][g] for g in GROUPS]), original["groups"]["all"])
                metrics = route_sums(mlp["groups"], locf["groups"])
                assert all(metrics[k] == mlp["groups"]["all"][k] for k in FIELDS[:3])
                assert set(mlp["by_age"]) == set(locf["by_age"])
                by_age = {age: route_sums(mlp["by_age"][age], locf["by_age"][age]) for age in mlp["by_age"]}
                equal_record(pool_records(list(by_age.values())), metrics)
                cases.append({"algorithm": name, "subject": 107, "segment": segment, "regime": regime,
                              "metrics": metrics, "by_age": by_age,
                              "routed_to_locf": locf["groups"]["warned"]["hidden"] + locf["groups"]["unknown"]["hidden"],
                              "routed_to_mlp": mlp["groups"]["unwarned"]["hidden"]})
    assert len(cases) == 1224
    pooled, by_age, summaries = {}, {}, []
    for name in names:
        pooled[name], by_age[name] = {}, {}
        for regime in REGIMES:
            selected = [r for r in cases if r["algorithm"] == name and r["regime"] == regime]
            pooled[name][regime] = pool_records([r["metrics"] for r in selected])
            ages = sorted({int(a) for r in selected for a in r["by_age"]})
            by_age[name][regime] = {str(a): pool_records([r["by_age"][str(a)] for r in selected if str(a) in r["by_age"]]) for a in ages}
            equal_record(route_sums(parent["pooled"][name][regime], parent["pooled"]["locf"][regime]), pooled[name][regime])
        original = math.fsum(parent["pooled"][name][r]["all"]["mae"][c] / std[c] for r in REGIMES for c in range(6)) / 24
        hybrid = math.fsum(pooled[name][r]["mae"][c] / std[c] for r in REGIMES for c in range(6)) / 24
        summaries.append({"algorithm": name, "original_normalized_mae": original, "hybrid_normalized_mae": hybrid,
                          "change_percent": 100 * (hybrid / original - 1), "improves": hybrid < original,
                          "mae_cells_improved": sum(pooled[name][r]["mae"][c] < parent["pooled"][name][r]["all"]["mae"][c] for r in REGIMES for c in range(6)),
                          "rmse_cells_improved": sum(pooled[name][r]["rmse"][c] < parent["pooled"][name][r]["all"]["rmse"][c] for r in REGIMES for c in range(6)),
                          "largest_rmse_increase_percent": max(100 * (pooled[name][r]["rmse"][c] / parent["pooled"][name][r]["all"]["rmse"][c] - 1) for r in REGIMES for c in range(6))})
    verify_frozen(ROOT)
    for name, expected in parent_hashes.items():
        assert digest(ROOT / name) == expected
    for name, expected in parent["model_hashes"].items():
        assert digest(ROOT / "runs/mlp_v1_train_20261027/models" / name) == expected
    assert not raw_attempts
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "policy": policy,
              "case_count": len(cases), "cases": cases, "pooled": pooled, "by_age": by_age,
              "summaries": summaries, "correctness_pass": True, "primary_usefulness_pass": all(s["improves"] for s in summaries),
              "raw_recording_open_attempts": raw_attempts, "model_hashes": parent["model_hashes"],
              "original_source_hashes": parent["source_hashes"], "python": sys.version, "numpy": np.__version__,
              "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()}
    write_json(OUT / "report.json", report)
    fields = ["algorithm", "regime", "age", "channel", "unit", "hidden", "scored", "unavailable", "coverage",
              "original_mae", "hybrid_mae", "locf_mae", "mae_change_percent", "original_rmse", "hybrid_rmse", "locf_rmse", "rmse_change_percent",
              "original_sum_abs", "hybrid_sum_abs", "original_sum_sq", "hybrid_sum_sq"]
    with (OUT / "comparison.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for name in names:
            for regime in REGIMES:
                for age, metrics in {"all": pooled[name][regime], **by_age[name][regime]}.items():
                    original = parent["pooled"][name][regime]["all"] if age == "all" else parent["by_age"][name][regime][age]["all"]
                    locf = parent["pooled"]["locf"][regime]["all"] if age == "all" else parent["by_age"]["locf"][regime][age]["all"]
                    for c, channel in enumerate(CHANNELS):
                        row = {"algorithm": name, "regime": regime, "age": age, "channel": channel,
                               "unit": "m/s^2" if c < 3 else "rad/s", **{k: metrics[k] for k in FIELDS[:3]},
                               "coverage": metrics["scored"] / metrics["hidden"]}
                        for metric in ("mae", "rmse"):
                            row.update({f"original_{metric}": original[metric][c], f"hybrid_{metric}": metrics[metric][c],
                                        f"locf_{metric}": locf[metric][c], f"{metric}_change_percent": 100 * (metrics[metric][c] / original[metric][c] - 1)})
                        for field in ("sum_abs", "sum_sq"):
                            row[f"original_{field}"] = original[field][c]; row[f"hybrid_{field}"] = metrics[field][c]
                        writer.writerow(row)
    print(json.dumps({"correctness_pass": True, "primary_usefulness_pass": report["primary_usefulness_pass"], "summaries": summaries}, indent=2))


if __name__ == "__main__": main()
