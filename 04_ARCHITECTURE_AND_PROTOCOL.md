# Architecture & Wire Protocol v0.1 — Proposed, Not Frozen

ESP32 sensor/playback producer -> UART -> STM32U5 timestamped receiver -> gap detector -> causal estimator -> confidence/abstention -> tagged output -> USB/UART PC visualizer.

## Frame concept
Version | sequence_number | source_timestamp_us | sample_count | accel_xyz | gyro_xyz | flags | CRC

Define explicit field widths, byte order, framing (COBS/SLIP or fixed length), CRC polynomial, reset behavior, and timestamp wrap before firmware implementation. UART is a serial stream, not packetized: the application must delimit and validate frames. Distinguish missing frames from CRC-failed frames and from delayed/out-of-order frames.

## Timing
At 100 Hz nominal period is 10 ms. Sequence continuity proves packet omission only after a later frame arrives or a well-defined timeout fires; lost samples cannot always be recognized exactly at their scheduled timestamp without synchronized clocks or an agreed receive schedule.

## Buffering
Single producer/single consumer bounded ring buffers where appropriate. Explicit overflow counter and sample-status enum: MEASURED, RECONSTRUCTED, UNAVAILABLE. Every record also includes source timestamp and estimation timestamp.

## STM32 tasks (tentative)
- UART RX ISR/DMA handles transfer only; no inference inside ISR.
- Frame parser and sequence/timestamp validation.
- Optional estimator task, priority below critical sampling/communication duties.
- Telemetry streaming task.
- Watchdog and deterministic fallback; no inference blocking a safety-critical operation.

## Validation
First use deterministic stored playback to establish numeric parity between Python and STM32. Add actual sensor only after protocol and inference pass. Measure data rate, loss observability, deadlines, memory, and reconstruction quality separately.
