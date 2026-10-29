# Status-aware desktop consumer

Software integration PASS: six consumer tests pass. No training, dataset replay,
firmware, model acceptance, or new accuracy result is implied.

## What happens and why

```text
artificial observation [6] or None
    -> existing abstaining predictor
    -> StepResult(values, status, missing_run)
    -> StatusConsumer.consume
    -> record measurement, estimate, or explicit gap
```

The consumer reads status first. MEASURED and RECONSTRUCTED require a finite
float64 six-vector and retain different tags. UNAVAILABLE becomes `values=null`
in JSON without reading the numeric payload. It never substitutes zeros or holds
the previous output. Sample indices remain consecutive, preserving gap positions.
This is logging/display plumbing; no downstream filter or control policy is chosen.

Each record has an integer sample index, string status, integer consecutive-loss
age, and immutable six-value tuple or None. The tuple owns its values, so later
changes to predictor arrays cannot alter saved records. Invalid input raises an
exception without appending a record. Reset the predictor and consumer together
when starting a new independent segment. `grouped()` supplies separate lists for
measurements, estimates, and gaps; it does not combine them into a sensor average.
The consumer does not infer transport timestamps or detect missing packets.

## Demonstration

The model is deliberately artificial: an untrained residual MLP emits a fixed
normalized correction +0.25, using identity normalization and synthetic warning
thresholds [1,1,1]. This makes estimates distinguishable in the example. It does
not load or modify trained checkpoints or real normalization/threshold artifacts.

The 40 nominal slots produce **23 MEASURED, 13 RECONSTRUCTED, 4 UNAVAILABLE**.
Gaps are at indices **0, 24, 25, 37**. Index 0 tests startup, indices 24–25 test
warning rejection, and index 37 tests loss age 11 beyond the existing horizon 10.
Index 39 demonstrates recovery. A received acceleration-X value of 4 is preserved
at index 23. Its change is divided by three elapsed slots: 4/3 > threshold 1.
An initial development example used 2, which correctly did not trigger a warning
(2/3 < 1); the artificial example was corrected, not the warning algorithm.

## Files and reproduction

- `../mlp_v1_consumer_20261027.py`: consumer plus artificial end-to-end demo.
- `../test_mlp_v1_consumer_20261027.py`: six software tests. They cover status-first
  access with an unreadable unavailable payload, ignored finite gap placeholders,
  separate routes and real zeros, invalid-input atomicity, ownership/reset,
  entirely unavailable streams, and the complete 40-slot example.
- `demo.json`: every output slot, including explicit null gaps.
- `software_tests.txt`: captured passing unittest output.
- `evidence.sha256`: checksums relative to repository root.

```bash
python3 -B -m unittest discover -s runs -p test_mlp_v1_consumer_20261027.py -v
sha256sum -c runs/mlp_v1_consumer_20261027/evidence.sha256
```

The demo script refuses an existing output directory. Its CLI installs a Python
audit hook denying opens under data/raw; the demo needs no actual recording.
Tests use artificial arrays and no checkpoints. Frozen artifacts and earlier
evaluated source are unchanged. Subjects 108/109 are not opened. Files under
runs/ remain Git-ignored local prototypes. The root status document still records
Phase 2 closure; this task does not rewrite it or declare Phase 3 accepted.

Next experiment: exercise this consumer with an existing trained checkpoint on
the same artificial stream. PASS requires unchanged measurements, separate
estimate tags, preserved gap indices and no use of unavailable payloads; otherwise
FAIL. This would check plumbing with trained weights, not reconstruction accuracy.
