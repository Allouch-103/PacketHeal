"""
PacketHeal AI — Subject-wise dataset split enforcement.

Reads the approved split configuration and prevents
loading subjects outside the requested partition.
"""

import json
from pathlib import Path

from .dataset import load_segment


SPLIT_KEYS = {
    "train": "train_subjects",
    "validation": "validation_subjects",
    "test": "test_subjects",
    "supplementary_test": "supplementary_test_subjects",
}


def load_split_config(config_path):
    """Load and validate the approved subject-wise split."""

    config = json.loads(Path(config_path).read_text())

    groups = {
        name: config[key]
        for name, key in SPLIT_KEYS.items()
    }

    subjects = [
        subject
        for group in groups.values()
        for subject in group
    ]

    if len(subjects) != len(set(subjects)):
        raise ValueError("Subjects overlap between partitions")

    if set(subjects) != set(range(101, 110)):
        raise ValueError("Missing or unexpected subjects")

    return groups


def load_split_segment(
    dataset_dir,
    manifest_path,
    config_path,
    split,
    subject,
    segment_index=0,
):
    """
    Load a segment only if the subject belongs
    to the requested partition.
    """

    groups = load_split_config(config_path)

    if split not in groups:
        raise ValueError(f"Unknown split: {split}")

    if subject not in groups[split]:
        raise ValueError(
            f"Subject {subject} is not permitted in '{split}'"
        )

    return load_segment(
        dataset_dir,
        manifest_path,
        subject=subject,
        segment_index=segment_index,
    )