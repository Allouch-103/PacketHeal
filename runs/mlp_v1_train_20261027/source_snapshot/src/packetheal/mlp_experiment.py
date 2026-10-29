"""Experiment orchestration for the first direct/residual causal MLP comparison.

No raw recording is opened at import. Training and validation have separate
access allowlists. Dataset paths follow the existing loader: dataset_dir is the
PAMAP2_Dataset root; recordings live in its Protocol subdirectory.
"""

from contextlib import contextmanager
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from .mlp import Adam, MLP, MLPPredictor, Normalization, segment_examples


REGIMES = ("random_5pct", "burst_2_5pct", "burst_5_5pct", "burst_10_5pct")
CHANNELS = ("acc_x_m_s2", "acc_y_m_s2", "acc_z_m_s2", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s")
FROZEN = {
    "configs/pamap2_split_v1.json": "561577438669aa6f19eb14c898c1cd217c9cd5cff49385239d7c5c9107ba0f00",
    "configs/pamap2_segments_v1.csv": "4ee69d67b248856ea59355b31525db4f4690b2d2a7e5fc5a5f1bd2e959a213ca",
    "data/processed/pamap2_v1/normalization.json": "4f2102cf92de523065bbf64532ffb67f8dc4c138db9f4ca48edbcb83eec1b504",
    "data/processed/pamap2_v1/loss_masks.csv": "0c1129cdf75f8eae4f451d5027d32c523d5cf755dde3044a241770d24d5792a3",
    "data/processed/linear_extrapolation_v1/slope_limits.json": "316cc1f1d0e95446329f1f689512703a146f6d07ad00bf463649ad705e13f502",
    "data/processed/alpha_beta_v1/gains.json": "9eb9a8c44d7870d7d40d31dd6fe0e0a7cac876c38b2f4515fe81b99e690d4395",
    "benchmarks/phase2_validation_subject107_v1/benchmark.json": "6452b308976483d00d976aef71cf53e0688c73fba56f21c0448c14ce9f467e8b",
}
SOURCE_MANIFEST_HASH = "b597d719b4baed615f96784bc45e43d1b54c4de18f87e88e84d2b7b32e40dc1c"
ARCHIVED_CSV_HASH = "26cbdd813f7fca6e1425a30a8764b0349ea667175605dc364251f2385badaf38"
MLP_SOURCES = (
    "src/packetheal/mlp.py", "src/packetheal/mlp_experiment.py",
    "scripts/run_mlp.py", "tests/test_mlp.py", "tests/test_mlp_experiment.py",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check_output_location(root, dataset_dir, output_dir):
    output = Path(output_dir).resolve()
    protected = [Path(root).resolve() / part for part in ("data", "configs", "benchmarks", "src", "tests", "scripts")]
    protected.append(Path(dataset_dir).resolve())
    if any(output == path or output.is_relative_to(path) for path in protected):
        raise ValueError("Outputs must be outside raw data, frozen artifacts, and source directories")


def verify_frozen(root):
    """Explicit files only. Never scan/hash raw recordings or discover subjects."""
    root = Path(root).resolve()
    for name, expected in FROZEN.items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    manifest = root / "benchmarks/phase2_validation_subject107_v1/evaluated_sources.sha256"
    if digest(manifest) != SOURCE_MANIFEST_HASH:
        raise ValueError("Evaluated source manifest changed")
    for line in manifest.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = (root / name.strip()).resolve()
        if not path.is_relative_to(root) or path.suffix == ".dat":
            raise ValueError("Unsafe source manifest entry")
        if digest(path) != expected:
            raise ValueError(f"Baseline source changed: {name}")
    if digest(root / "benchmarks/phase2_validation_subject107_v1/overall.csv") != ARCHIVED_CSV_HASH:
        raise ValueError("Archived baseline CSV changed")
    return dict(FROZEN)


@contextmanager
def recording_access_guard(dataset_dir, subjects):
    """Python audit-hook guard; an application check, not an OS sandbox.

    A permanent hook is made inactive on exit; no dataset is opened to build the
    allowlist. Symlink aliases resolve to their actual path before permission.
    """
    subjects = tuple(subjects)
    if not subjects or any(s not in range(101, 108) for s in subjects):
        raise ValueError("Only subjects 101 through 107 are permitted")
    raw = Path(dataset_dir).resolve()
    allowed = {raw / f"subject{s}.dat" for s in subjects}
    log = {"allowed_subjects": list(subjects), "opened_recordings": [], "denied_recordings": []}
    active = [True]

    def audit(event, arguments):
        if event != "open" or not active[0] or not arguments or isinstance(arguments[0], int):
            return
        candidate = Path(os.fsdecode(arguments[0])).absolute()
        resolved = candidate.resolve()
        if candidate.suffix != ".dat" and resolved.suffix != ".dat" and not resolved.is_relative_to(raw):
            return
        if resolved not in allowed:
            log["denied_recordings"].append(str(resolved))
            raise PermissionError(f"Recording access denied: {resolved}")
        mode, flags = arguments[1:3]
        if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
            raise PermissionError("Raw recordings are read-only")
        log["opened_recordings"].append(str(resolved))

    sys.addaudithook(audit)
    try:
        yield log
    finally:
        active[0] = False


def unpack_loaded_segment(result, count):
    """Validate the existing load_split_segment (timestamps, values) interface."""
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("Expected the loader's (timestamps, values) tuple")
    times, values = result
    times, values = np.asarray(times), np.asarray(values)
    if times.shape != (count,) or values.shape != (count, 6):
        raise ValueError("Loader must return [samples] timestamps and [samples,6] wrist channels")
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise ValueError("Segment contains non-finite measurements")
    if count > 1 and not np.all(np.abs(np.diff(times) - .01) <= .001 + 1e-10):
        raise ValueError("Timestamp continuity violated")
    return values.astype(np.float64, copy=False)


def normalization_from_root(root):
    value = json.loads((Path(root) / "data/processed/pamap2_v1/normalization.json").read_text())
    if value["fit_split"] != "train" or tuple(value["channel_order"]) != CHANNELS:
        raise ValueError("Wrong normalization provenance or channel order")
    return Normalization(value["mean"], value["std_sample"])


def iter_segments(root, dataset_dir, split):
    """Use the existing split-aware entry point and frozen-mask reader."""
    from .splits import load_split_segment
    from .evaluation import load_subject_masks
    if split not in ("train", "validation"):
        raise ValueError("Test partitions are prohibited")
    root = Path(root)
    config = root / "configs/pamap2_split_v1.json"
    groups = json.loads(config.read_text())
    expected = list(range(101, 107)) if split == "train" else [107]
    if groups[f"{split}_subjects"] != expected:
        raise ValueError("Split differs from the frozen protocol")
    manifest = root / "configs/pamap2_segments_v1.csv"
    for subject in expected:
        segments, masks = load_subject_masks(
            root / "data/processed/pamap2_v1/loss_masks.csv", manifest,
            config, split, subject,
        )
        for index, metadata in enumerate(segments):
            loaded = load_split_segment(dataset_dir, manifest, config, split, subject, index)
            values = unpack_loaded_segment(loaded, int(metadata["samples"]))
            yield subject, index, metadata, values, {r: masks[index, r] for r in REGIMES}


def train_models(features, targets, regime_ids, output_dir, *, epochs=20, batch_size=256,
                 learning_rate=1e-3, seeds=(0, 1, 2), widths=(32, 16), progress=print):
    """Balanced regime sampling gives each channel/regime equal expected weight.

    One epoch draws approximately N examples WITH replacement, equally from
    each regime. Reported final objectives pool every stored training example.
    Both modes use identical seeded batch draws and training budgets.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("Invalid training features")
    if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 4 or batch_size % 4:
        raise ValueError("Positive epochs and a batch size divisible by four are required")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    ids = np.asarray(regime_ids)
    if ids.shape != (len(features),) or not np.isin(ids, range(4)).all():
        raise ValueError("Invalid regime IDs")
    groups = [np.flatnonzero(ids == i) for i in range(4)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Every frozen regime needs training examples")
    for mode in ("direct", "residual"):
        if np.asarray(targets[mode]).shape != (len(features), 6) or not np.isfinite(targets[mode]).all():
            raise ValueError("Invalid targets")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for mode in ("direct", "residual"):
        for seed in seeds:
            rng = np.random.default_rng(seed)
            model = MLP(features.shape[1], widths=widths, mode=mode, seed=seed)
            optimizer = Adam(model.parameters, learning_rate)
            curve = []
            for epoch in range(epochs):
                losses = []
                for _ in range(math.ceil(len(features) / batch_size)):
                    selected = np.concatenate([rng.choice(group, batch_size // 4, replace=True) for group in groups])
                    loss, gradient = model.loss_and_gradients(features[selected], targets[mode][selected])
                    optimizer.step(gradient)
                    losses.append(loss)
                curve.append(float(np.mean(losses)))
                progress(f"{mode} seed={seed} epoch={epoch + 1}/{epochs} sampled MAE={curve[-1]:.6f}")
            regime_losses = []
            for group in groups:
                total = 0.0
                for start in range(0, len(group), 1024):
                    selected = group[start:start + 1024]
                    total += float(np.abs(model.predict(features[selected]).astype(np.float64) - targets[mode][selected]).sum())
                regime_losses.append(total / (6 * len(group)))
            checkpoint = output_dir / f"{mode}_seed{seed}.npz"
            model.save(checkpoint)
            results.append({"mode": mode, "seed": seed, "checkpoint": checkpoint.name,
                            "sha256": digest(checkpoint), "sampled_epoch_mae": curve,
                            "final_normalized_mae_by_regime": regime_losses,
                            "final_equal_regime_normalized_mae": float(np.mean(regime_losses))})
    return results


def error_record(truth, predicted, selected):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    selected = np.asarray(selected, dtype=bool)
    if truth.shape != predicted.shape or truth.ndim != 2 or truth.shape[1] != 6 or selected.shape != (len(truth),):
        raise ValueError("Invalid metric inputs")
    available = np.isfinite(predicted).all(axis=1)
    if not np.isfinite(truth[selected]).all():
        raise ValueError("Non-finite scoring truth")
    error = predicted[selected & available] - truth[selected & available]
    n = len(error)
    absolute = [math.fsum(abs(float(e)) for e in error[:, c]) for c in range(6)]
    squared = [math.fsum(float(e) ** 2 for e in error[:, c]) for c in range(6)]
    return {"hidden": int(selected.sum()), "scored": n, "unavailable": int(selected.sum()) - n,
            "sum_abs": absolute, "sum_sq": squared,
            "mae": [v / n for v in absolute] if n else [None] * 6,
            "rmse": [math.sqrt(v / n) for v in squared] if n else [None] * 6}


def pool_records(records):
    n = sum(row["scored"] for row in records)
    absolute = [math.fsum(row["sum_abs"][c] for row in records) for c in range(6)]
    squared = [math.fsum(row["sum_sq"][c] for row in records) for c in range(6)]
    return {"hidden": sum(row["hidden"] for row in records), "scored": n,
            "unavailable": sum(row["unavailable"] for row in records),
            "sum_abs": absolute, "sum_sq": squared,
            "mae": [v / n for v in absolute] if n else [None] * 6,
            "rmse": [math.sqrt(v / n) for v in squared] if n else [None] * 6}


def streaming_replay(predictor, truth, missing):
    predictor.reset()
    values, statuses, ages = [], [], []
    for t, hidden in enumerate(missing):
        result = predictor.step(None if hidden else truth[t].copy())
        values.append(result.values)
        statuses.append(str(result.status.value))
        ages.append(result.missing_run)
    return np.asarray(values), np.asarray(statuses), np.asarray(ages)


def audited_replay(predictor, truth, missing):
    """Repeat/reset, hidden poisoning, future value/mask checks for every case."""
    original_truth, original_mask = truth.copy(), missing.copy()
    original = streaming_replay(predictor, truth, missing)
    checks = [streaming_replay(predictor, truth, missing)]
    poisoned = truth.copy()
    poisoned[missing] = np.nan
    checks.append(streaming_replay(predictor, poisoned, missing))
    streaming_replay(predictor, np.ones((3, 6)), np.array([False, True, False]))
    checks.append(streaming_replay(predictor, truth, missing))
    for check in checks:
        for a, b in zip(original, check):
            np.testing.assert_array_equal(a, b)
    cutoff = len(truth) // 2
    changed, changed_mask = truth.copy(), missing.copy()
    changed[cutoff:] += .25
    changed_mask[cutoff:] = ~changed_mask[cutoff:]
    future = streaming_replay(predictor, changed, changed_mask)
    for a, b in zip(original, future):
        np.testing.assert_array_equal(a[:cutoff], b[:cutoff])
    np.testing.assert_array_equal(truth, original_truth)
    np.testing.assert_array_equal(missing, original_mask)
    np.testing.assert_array_equal(original[0][~missing], truth[~missing])
    expected_age, count = [], 0
    for hidden in missing:
        count = count + 1 if hidden else 0
        expected_age.append(count)
    np.testing.assert_array_equal(original[2], expected_age)
    return original


def provenance(root):
    root = Path(root)
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], text=True, capture_output=True, check=True)
        return result.stdout.strip()
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
            "installed_python_packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions() if d.metadata.get("Name")),
            "thread_environment": {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "source_hashes": {name: digest(root / name) for name in MLP_SOURCES},
            "sample_rate_hz": 100, "history": 20, "horizon": 10,
            "objective": "equal regime/channel normalized MAE", "frozen_inputs": dict(FROZEN)}


def train_experiment(root, dataset_dir, output_dir, *, epochs=20, seeds=(0, 1, 2)):
    root, output_dir = Path(root), Path(output_dir)
    check_output_location(root, dataset_dir, output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    verify_frozen(root)
    normalization = normalization_from_root(root)
    features, direct, residual, ids, cases = [], [], [], [], []
    raw = Path(dataset_dir) / "Protocol"
    with recording_access_guard(raw, range(101, 107)) as access:
        recordings_before = {str(s): digest(raw / f"subject{s}.dat") for s in range(101, 107)}
        for subject, index, metadata, truth, masks in iter_segments(root, dataset_dir, "train"):
            for regime_id, regime in enumerate(REGIMES):
                x, y, r, positions = segment_examples(truth, masks[regime], normalization)
                if len(x) != int(masks[regime].sum()):
                    raise AssertionError("Frozen protected-warmup examples must cover every hidden position")
                features.append(x)
                direct.append(y)
                residual.append(r)
                ids.append(np.full(len(x), regime_id, dtype=np.int8))
                cases.append({"subject": subject, "segment": index, "regime": regime, "examples": len(x)})
        if len(cases) != 3336:
            raise AssertionError("Expected 834 training segments under four regimes")
        x = np.concatenate(features)
        targets = {"direct": np.concatenate(direct), "residual": np.concatenate(residual)}
        regime_ids = np.concatenate(ids)
        counts = [int((regime_ids == i).sum()) for i in range(4)]
        if counts != [95600, 96002, 97260, 99380]:
            raise AssertionError("Training hidden counts do not match frozen preparation")
        models = train_models(x, targets, regime_ids, output_dir / "models", epochs=epochs, seeds=seeds)
        for subject, before in recordings_before.items():
            if digest(raw / f"subject{subject}.dat") != before:
                raise AssertionError("Training recording changed")
    verify_frozen(root)
    record = provenance(root)
    record.update({"kind": "training", "epochs": epochs, "batch_size": 256, "learning_rate": .001,
                   "seeds": list(seeds), "widths": [32, 16], "sampling": "balanced regimes with replacement",
                   "cases": cases, "counts": counts, "models": models,
                   "recording_hashes": recordings_before, "recording_access": access})
    write_json(output_dir / "training.json", record)
    return record


def validate_training_record(root, training_dir):
    training_dir = Path(training_dir)
    record = json.loads((training_dir / "training.json").read_text())
    if record.get("kind") != "training" or record.get("frozen_inputs") != FROZEN:
        raise ValueError("Checkpoint provenance does not match this experiment")
    if record["recording_access"]["allowed_subjects"] != list(range(101, 107)) or record["recording_access"]["denied_recordings"]:
        raise ValueError("Training access log is invalid")
    if record["history"] != 20 or record["horizon"] != 10 or record["widths"] != [32, 16]:
        raise ValueError("Unexpected model configuration")
    if set(record.get("source_hashes", {})) != set(MLP_SOURCES):
        raise ValueError("Unexpected training source paths")
    for name, expected in record["source_hashes"].items():
        if digest(Path(root) / name) != expected:
            raise ValueError("Training source changed before validation")
    models = []
    identities = set()
    for metadata in record["models"]:
        identity = (metadata["mode"], metadata["seed"])
        if identity in identities:
            raise ValueError("Duplicate model")
        identities.add(identity)
        path = (training_dir / "models" / metadata["checkpoint"]).resolve()
        if not path.is_relative_to((training_dir / "models").resolve()) or digest(path) != metadata["sha256"]:
            raise ValueError("Checkpoint hash/path mismatch")
        model = MLP.load(path)
        if model.mode != metadata["mode"] or model.input_size != 161 or model.widths != (32, 16):
            raise ValueError("Checkpoint configuration mismatch")
        models.append((metadata, model))
    expected = {(mode, seed) for mode in ("direct", "residual") for seed in record["seeds"]}
    if not expected or identities != expected:
        raise ValueError("Incomplete model/seed comparison")
    return record, models


def archive_rows(root):
    """Read only pinned saved metrics; do not load recordings for old baselines."""
    path = Path(root) / "benchmarks/phase2_validation_subject107_v1/overall.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"algorithm", "regime", "channel", "unit", "hidden_count", "scored_count", "unavailable_count", "mae", "rmse"}
    if len(rows) != 72 or any(not required.issubset(row) for row in rows):
        raise ValueError("Archived baseline CSV schema changed")
    identities = {(row["algorithm"], row["regime"], row["channel"]) for row in rows}
    expected = {(algorithm, regime, channel) for algorithm in ("locf", "linear", "alpha_beta")
                for regime in REGIMES for channel in CHANNELS}
    if identities != expected:
        raise ValueError("Archived baseline CSV case identities changed")
    return rows


def check_locf_reference(root, pooled):
    rows = [row for row in archive_rows(root) if row["algorithm"].lower() == "locf"]
    if len(rows) != 24:
        raise ValueError("Expected 24 archived LOCF channel/regime cells")
    for regime in REGIMES:
        selected = [row for row in rows if row["regime"] == regime]
        if len(selected) != 6 or tuple(row["channel"] for row in selected) != CHANNELS:
            raise ValueError("Archived channel naming/order changed")
        for c, row in enumerate(selected):
            for field, metric_field in (("hidden_count", "hidden"), ("scored_count", "scored"), ("unavailable_count", "unavailable")):
                if int(row[field]) != pooled[regime][metric_field]:
                    raise AssertionError("Fresh LOCF counts differ from frozen archive")
            for metric in ("mae", "rmse"):
                if not math.isclose(pooled[regime][metric][c], float(row[metric]), rel_tol=1e-12, abs_tol=1e-12):
                    raise AssertionError("Fresh LOCF metrics differ from frozen archive")


def engineering_screen(pooled, normalization, seeds):
    """Proposed practical screen, not significance or subject-level confidence."""
    def objective(regimes):
        return float(np.mean([np.mean(np.asarray(regimes[r]["mae"]) / normalization.std) for r in REGIMES]))
    baseline = objective(pooled["locf"])
    result = {"baseline_equal_regime_normalized_mae": baseline, "thresholds": {
        "mean_mae_reduction_fraction": .05, "maximum_cell_rmse_increase_fraction": .10,
        "required_improving_seeds": math.ceil(2 * len(seeds) / 3)}, "modes": {}}
    for mode in ("direct", "residual"):
        scores = [objective(pooled[f"{mode}_seed{seed}"]) for seed in seeds]
        improving = sum(score < baseline for score in scores)
        rmse_ok = all(np.all(np.asarray(pooled[f"{mode}_seed{seed}"][r]["rmse"]) <= 1.1 * np.asarray(pooled["locf"][r]["rmse"])) for seed in seeds for r in REGIMES)
        coverage_ok = all(pooled[f"{mode}_seed{seed}"][r]["unavailable"] == 0 for seed in seeds for r in REGIMES)
        mean = float(np.mean(scores))
        result["modes"][mode] = {"seed_objectives": scores, "mean_objective": mean,
                                  "improving_seeds": improving, "rmse_guard_pass": bool(rmse_ok),
                                  "coverage_pass": coverage_ok,
                                  "screen_pass": bool(baseline > 0 and mean <= .95 * baseline and improving >= math.ceil(2 * len(seeds) / 3) and rmse_ok and coverage_ok)}
    return result


def validation_experiment(root, dataset_dir, training_dir, output_dir):
    from .baselines import LOCF
    root, output_dir = Path(root), Path(output_dir)
    check_output_location(root, dataset_dir, output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    verify_frozen(root)
    training, models = validate_training_record(root, training_dir)
    normalization = normalization_from_root(root)
    cases = []
    raw = Path(dataset_dir) / "Protocol"
    with recording_access_guard(raw, [107]) as access:
        recording_hash = digest(raw / "subject107.dat")
        for subject, index, metadata, truth, masks in iter_segments(root, dataset_dir, "validation"):
            for regime in REGIMES:
                missing = masks[regime]
                for identity, predictor in [({"mode": "locf", "seed": None}, LOCF(6))] + [
                    ({"mode": m["mode"], "seed": m["seed"]}, MLPPredictor(model, normalization)) for m, model in models
                ]:
                    values, statuses, ages = audited_replay(predictor, truth, missing)
                    metrics = error_record(truth, values, missing)
                    if metrics["unavailable"] or not np.all(statuses[missing] == "RECONSTRUCTED"):
                        raise AssertionError("Frozen hidden positions must all be reconstructed")
                    cases.append({**identity, "subject": subject, "segment": index, "regime": regime,
                                  "metrics": metrics,
                                  "by_age": {str(age): error_record(truth, values, missing & (ages == age)) for age in range(1, 11)}})
        if digest(raw / "subject107.dat") != recording_hash:
            raise AssertionError("Validation recording changed")
    pooled = {}
    identities = [("locf", None)] + [(m["mode"], m["seed"]) for m, _ in models]
    for mode, seed in identities:
        key = mode if seed is None else f"{mode}_seed{seed}"
        pooled[key] = {}
        for regime, count in zip(REGIMES, [14176, 14226, 14370, 14660]):
            selected = [case for case in cases if (case["mode"], case["seed"], case["regime"]) == (mode, seed, regime)]
            if len(selected) != 102:
                raise AssertionError("Expected 102 distinct segments per model/regime")
            pooled[key][regime] = pool_records([case["metrics"] for case in selected])
            if pooled[key][regime]["hidden"] != count:
                raise AssertionError("Validation hidden counts mismatch")
    check_locf_reference(root, pooled["locf"])
    verify_frozen(root)
    record = provenance(root)
    record.update({"kind": "validation", "training_report_sha256": digest(Path(training_dir) / "training.json"),
                   "models": training["models"], "recording_hash": recording_hash,
                   "recording_access": access, "cases": cases, "pooled": pooled,
                   "engineering_screen": engineering_screen(pooled, normalization, training["seeds"]),
                   "baseline_reference": {"path": "benchmarks/phase2_validation_subject107_v1/overall.csv",
                                          "sha256": ARCHIVED_CSV_HASH, "locf_metrics_replayed_and_matched": True,
                                          "linear_alpha_beta_metrics": "unchanged archived metrics; no fresh replay",
                                          "archive_replay_fingerprints_rechecked": False},
                   "limitations": ["one validation subject", "synthetic masks", "no uncertainty calibration",
                                   "no quantization/hardware measurements", "no held-out tests"]})
    write_json(output_dir / "validation.json", record)
    with (output_dir / "overall.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["algorithm", "regime", "channel", "unit", "hidden", "scored", "unavailable", "mae", "rmse"])
        writer.writeheader()
        for algorithm, regimes in pooled.items():
            for regime, row in regimes.items():
                for c, channel in enumerate(CHANNELS):
                    writer.writerow({"algorithm": algorithm, "regime": regime, "channel": channel,
                                     "unit": "m/s^2" if c < 3 else "rad/s",
                                     "hidden": row["hidden"], "scored": row["scored"], "unavailable": row["unavailable"],
                                     "mae": row["mae"][c], "rmse": row["rmse"][c]})
    with (root / "benchmarks/phase2_validation_subject107_v1/overall.csv").open("rb") as source:
        with (output_dir / "frozen_classical_overall.csv").open("xb") as destination:
            destination.write(source.read())
    return record
