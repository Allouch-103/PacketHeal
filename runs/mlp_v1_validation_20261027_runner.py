"""Logging-only wrapper around the unchanged frozen MLP validation runner."""

from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import packetheal.mlp_experiment as experiment

RAW = ROOT / "data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset/Protocol"
TRAINING = ROOT / "runs/mlp_v1_train_20261027"
OUTPUT = ROOT / "runs/mlp_v1_validation_20261027"
PROGRESS = ROOT / "runs/mlp_v1_validation_20261027.progress.json"

experiment.verify_frozen(ROOT)
if OUTPUT.exists() or PROGRESS.exists():
    raise FileExistsError("Validation output/progress already exists")
reference = json.loads((ROOT / "benchmarks/phase2_validation_subject107_v1/benchmark.json").read_text())
expected = reference["provenance"]["artifact_sha256"]["validation_recording"]
with experiment.recording_access_guard(RAW, [107]) as preflight:
    actual = experiment.digest(RAW / "subject107.dat")
    if actual != expected:
        raise ValueError("Subject107 differs from the frozen classical benchmark")
del reference
experiment.write_json(ROOT / "runs/mlp_v1_validation_20261027.launch.json", {
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "wrapper_sha256": experiment.digest(__file__),
    "training_report_sha256": experiment.digest(TRAINING / "training.json"),
    "recording_hash": actual,
    "preflight_access": preflight,
    "wrapper_behavior": "Calls original audited_replay unchanged; logs completed case groups only",
})

original_replay = experiment.audited_replay
completed = 0
started = time.monotonic()


def logged_replay(predictor, truth, missing):
    global completed
    result = original_replay(predictor, truth, missing)
    completed += 1
    if completed % 7 == 0:
        group = completed // 7
        progress = {
            "completed_algorithm_cases": completed,
            "expected_algorithm_cases": 2856,
            "completed_segment_regime_cases": group,
            "expected_segment_regime_cases": 408,
            "elapsed_seconds": time.monotonic() - started,
        }
        PROGRESS.write_text(json.dumps(progress, indent=2) + "\n")
        print(f"Completed segment/regime {group}/408; model cases {completed}/2856", flush=True)
    return result


experiment.audited_replay = logged_replay
print("Subject107 hash matches frozen benchmark. Starting audited validation.", flush=True)
report = experiment.validation_experiment(ROOT, RAW.parent, TRAINING, OUTPUT)
print(json.dumps({"completed": completed, "screen": report["engineering_screen"]}, indent=2), flush=True)
