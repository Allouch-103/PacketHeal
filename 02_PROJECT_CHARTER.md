# PacketHeal AI — Project Charter v0.1

## Problem
Periodic sensors transmit samples to a receiver. Packet losses, late arrivals, or damaged messages create gaps. A causal estimator may reconstruct short gaps, but can be misleading during abrupt signal changes or extended outages. We want better-than-trivial estimates when possible and honest abstention when not.

## Deliverable
A reproducible ML benchmark plus a running ESP32 → STM32U5 embedded demonstrator that:
1. Detects missing sensor packets via monotonically increasing sequence numbers.
2. Predicts missing sample values from past samples only.
3. Estimates reliability and can return INVALID/ABSTAIN.
4. Reports whether every sample is REAL, RECONSTRUCTED, or UNAVAILABLE.
5. Meets measured device-specific inference time and memory budgets.

## Initial scope
- Stream: 3-axis acceleration + 3-axis angular rate at nominal 100 Hz; start with fewer channels if necessary.
- Benchmark: PAMAP2 wrist IMU (candidate), verify units, sampling and missingness. Dataset does not contain packet-loss annotations; inject independent controlled drop patterns.
- Live data: ESP32 plus a compatible IMU *if available*. Otherwise start with deterministic playback of stored sequences over UART.
- Main mode: strictly causal, fixed model and static normalization at inference time.
- Models: hold-last, causal extrapolation, simple state estimator, small MLP/1D-CNN or TCN. More complex models only after baseline failure analysis.

## Not in first release
- Certified industrial protocol conformance or IEC 61850 implementation.
- Guaranteed recovery from arbitrarily long losses.
- Safety-critical control based on imputed data.
- Custom PCB, cloud infrastructure, large foundation models.
- On-device neural-network training.

## Dataset validity checks
- Inspect PAMAP2 sensor sampling, channel names/units, timestamps, known invalid fields, labeled activity segments, missing samples, and transitions.
- Split *before* constructing windows; hold out at least one participant/session when feasible.
- Build an offline clean-data mask and define valid ground truth only where original measurements are present and trustworthy.
- Train on simulated losses generated from training partitions; evaluate on independently seeded holdout masks and distinct patterns.

## Correct comparison
At time t, estimator may use only packets received no later than t. A centered interpolator using future samples is an *offline upper/reference*, not a valid real-time baseline.

## Acceptance criteria (targets, not measured achievements)
- A reproducible evaluation script and held-out split manifest.
- Model tested against >=3 causal baselines under identical loss masks.
- Improvements reported with uncertainty across seeds/sessions rather than a cherry-picked sequence.
- End-to-end UART replay: missing packet identification and telemetry with correct provenance tags.
- Working quantized model on STM32U5 if deployment tooling and memory constraints permit; otherwise document the blocker precisely and benchmark closest viable implementation.
- Evidence that models refrain from low-confidence reconstructions for long gaps/out-of-distribution cases.

## Hardware
- NUCLEO-U5A5ZJ-Q, ESP32, Ubuntu PC. Jetson optional; no expensive GPU required.
- Do NOT assume UART DMA behavior, pin mapping, debugger transport, RAM budget, or tool compatibility without verifying exact versions and board reference manuals.

## Project question
Can a compact quantized causal model outperform low-cost traditional estimators during short sensor-sample losses while expressing uncertainty, under STM32U5 real-time resource constraints?

## Scientific caution
Successful artificial-dropout reconstruction is evidence for the reconstruction algorithm, not proof that real radio/fieldbus faults or all industrial conditions are solved.
