"""
PacketHeal AI — PAMAP2 Dataset Loader
Phase 1: Dataset Preparation

Loads validated continuous IMU segments from PAMAP2.

Selected channels:
    Accelerometer X, Y, Z (m/s²)
    Gyroscope X, Y, Z (rad/s)

Features:
    - Preserves original timestamps
    - Loads original sensor measurements
    - Supports selecting any qualifying segment
    - Validates column count
    - Rejects NaN and infinite measurements
    - Checks timestamp continuity
    - Verifies segment boundaries and length
    - Does not modify raw data
"""

import csv
from pathlib import Path

import numpy as np


# PAMAP2 column indices (0-based)
CHANNEL_INDICES = [4, 5, 6, 10, 11, 12]

# Nominal sampling frequency
SAMPLE_RATE_HZ = 100

# Expected sampling interval
EXPECTED_DT = 1.0 / SAMPLE_RATE_HZ

# Timestamp tolerance: 1 ms
DT_TOLERANCE = 0.001


def load_segment(
    dataset_dir,
    manifest_path,
    subject=101,
    segment_index=0
):
    """
    Load a validated continuous IMU segment for a subject.

    Parameters
    ----------
    dataset_dir : str or Path
        Path to PAMAP2_Dataset directory.

    manifest_path : str or Path
        Path to pamap2_segments_v1.csv.

    subject : int
        PAMAP2 subject identifier.

    segment_index : int
        Index of the qualifying segment for this subject.
        Starts at 0.

    Returns
    -------
    timestamps : numpy.ndarray
        Shape (N,), original timestamps in seconds.

    data : numpy.ndarray
        Shape (N, 6), IMU measurements in original units.

    Channel order:
        [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    """

    dataset_dir = Path(dataset_dir)
    manifest_path = Path(manifest_path)

    # --------------------------------------------------
    # 1. Read validated segment manifest
    # --------------------------------------------------

    with manifest_path.open("r", newline="") as f:
        segments = [
            row
            for row in csv.DictReader(f)
            if row["subject"] == f"subject{subject}"
        ]

    if not segments:
        raise ValueError(
            f"No validated segments found for subject {subject}"
        )

    # --------------------------------------------------
    # 2. Select requested segment
    # --------------------------------------------------

    if not isinstance(segment_index, int):
        raise TypeError(
            "segment_index must be an integer"
        )

    if not 0 <= segment_index < len(segments):
        raise IndexError(
            f"Segment index {segment_index} is invalid. "
            f"Subject {subject} has {len(segments)} qualifying segments."
        )

    segment = segments[segment_index]

    # --------------------------------------------------
    # 3. Locate original recording
    # --------------------------------------------------

    file_path = (
        dataset_dir
        / "Protocol"
        / segment["source_file"]
    )

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Recording not found: {file_path}"
        )

    start_row = int(segment["start_row"])
    end_row = int(segment["end_row"])
    expected_samples = int(segment["samples"])

    if start_row < 1 or end_row < start_row:
        raise ValueError(
            "Invalid segment row boundaries"
        )

    if expected_samples != end_row - start_row + 1:
        raise ValueError(
            "Manifest sample count does not match row boundaries"
        )

    # --------------------------------------------------
    # 4. Extract timestamps and sensor measurements
    # --------------------------------------------------

    selected_rows = []
    timestamps = []

    with file_path.open("r") as f:

        for row_number, line in enumerate(f, start=1):

            if row_number < start_row:
                continue

            if row_number > end_row:
                break

            values = line.split()

            if len(values) != 54:
                raise ValueError(
                    f"Invalid column count at row {row_number}"
                )

            timestamp = float(values[0])

            imu_values = [
                float(values[i])
                for i in CHANNEL_INDICES
            ]

            timestamps.append(timestamp)
            selected_rows.append(imu_values)

    # --------------------------------------------------
    # 5. Convert to NumPy arrays
    # --------------------------------------------------

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64
    )

    data = np.asarray(
        selected_rows,
        dtype=np.float64
    )

    # --------------------------------------------------
    # 6. Validate dimensions
    # --------------------------------------------------

    if timestamps.shape != (expected_samples,):
        raise ValueError(
            "Timestamp length does not match manifest"
        )

    if data.shape != (expected_samples, 6):
        raise ValueError(
            "IMU data shape does not match manifest"
        )

    # --------------------------------------------------
    # 7. Validate numerical values
    # --------------------------------------------------

    if not np.isfinite(timestamps).all():
        raise ValueError(
            "Segment contains invalid timestamps"
        )

    if not np.isfinite(data).all():
        raise ValueError(
            "Segment contains NaN or infinite IMU values"
        )

    # --------------------------------------------------
    # 8. Validate timestamp continuity
    # --------------------------------------------------

    dt = np.diff(timestamps)

    if not np.all(dt > 0):
        raise ValueError(
            "Timestamps are not strictly increasing"
        )

    if not np.all(
        np.abs(dt - EXPECTED_DT) <= DT_TOLERANCE + 1e-10
    ):
        raise ValueError(
            "Irregular sampling intervals detected"
        )

    # --------------------------------------------------
    # 9. Verify manifest timestamps
    # --------------------------------------------------

    expected_start = float(
        segment["start_timestamp_s"]
    )

    expected_end = float(
        segment["end_timestamp_s"]
    )

    if not np.isclose(
        timestamps[0],
        expected_start,
        atol=1e-8,
        rtol=0
    ):
        raise ValueError(
            "Start timestamp does not match manifest"
        )

    if not np.isclose(
        timestamps[-1],
        expected_end,
        atol=1e-8,
        rtol=0
    ):
        raise ValueError(
            "End timestamp does not match manifest"
        )

    return timestamps, data