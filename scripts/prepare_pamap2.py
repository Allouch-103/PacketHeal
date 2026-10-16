#!/usr/bin/env python3
"""PacketHeal AI: phase-1 reproducible preparation (no ML training).

Reads frozen PAMAP2 Protocol manifest and split, validates eligible rows,
fits normalization on TRAIN eligible rows only, and writes seeded masks.
No raw data is modified. Run at the project root.
"""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path.cwd()
RAW = ROOT / 'data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset/Protocol'
CONFIG = ROOT / 'configs/pamap2_split_v1.json'
MANIFEST = ROOT / 'configs/pamap2_segments_v1.csv'
OUT = ROOT / 'data/processed/pamap2_v1'
COLS = (4, 5, 6, 10, 11, 12)
NAMES = ('acc_x_m_s2', 'acc_y_m_s2', 'acc_z_m_s2', 'gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s')
SPLIT_KEYS = {'train':'train_subjects', 'validation':'validation_subjects', 'test':'test_subjects', 'supplementary_test':'supplementary_test_subjects'}
HISTORY = 20  # protect first 200 ms as received context
LOSS_RATE = 0.05
REGIMES = ('random_5pct', 'burst_2_5pct', 'burst_5_5pct', 'burst_10_5pct')


def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def seeded_rng(seed, split, subject, start, regime):
    text = f'{seed}:{split}:{subject}:{start}:{regime}'
    digest = hashlib.sha256(text.encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], 'little'))


def make_mask(n, regime, rng):
    """Generate offline missing-position mask; never treat it as receiver observability."""
    assert n >= 256
    mask = np.zeros(n, dtype=bool)
    allowed = np.arange(HISTORY, n)
    if regime == 'random_5pct':
        k = max(1, round(len(allowed) * LOSS_RATE))
        mask[rng.choice(allowed, size=k, replace=False)] = True
    else:
        burst = int(regime.split('_')[1])
        target = max(burst, round(len(allowed) * LOSS_RATE))
        # Nonoverlapping fixed-length bursts, including their boundaries.
        starts = rng.permutation(np.arange(HISTORY, n - burst + 1))
        selected = 0
        for start in starts:
            if selected >= target:
                break
            if not mask[max(HISTORY, start-1):min(n, start+burst+1)].any():
                mask[start:start + burst] = True
                selected += burst
    assert not mask[:HISTORY].any()
    return np.flatnonzero(mask)


def merge_stats(n1, m1, s1, batch):
    """Numerically stable parallel-variance update on a finite training batch."""
    n2 = len(batch)
    if n2 == 0:
        return n1, m1, s1
    mean2 = batch.mean(axis=0)
    m2 = ((batch - mean2) ** 2).sum(axis=0)
    if n1 == 0:
        return n2, mean2, m2
    delta = mean2 - m1
    n = n1 + n2
    return n, m1 + delta * n2 / n, s1 + m2 + delta**2 * n1 * n2 / n


def main():
    config = json.loads(CONFIG.read_text())
    groups = {name: config[key] for name, key in SPLIT_KEYS.items()}
    assigned = [s for subjects in groups.values() for s in subjects]
    assert len(assigned) == len(set(assigned)), 'Leakage: overlapping subject IDs'
    assert set(assigned) == set(range(101, 110)), 'Unexpected or unassigned subjects'
    subject_split = {s: name for name, subjects in groups.items() for s in subjects}
    assert config['sampling_frequency_hz'] == 100
    seed = config['random_seed']

    by_subject = defaultdict(list)
    with MANIFEST.open(newline='') as f:
        for rec in csv.DictReader(f):
            s = int(rec['subject'].removeprefix('subject'))
            assert s in subject_split
            start, end, n = (int(rec[key]) for key in ('start_row','end_row','samples'))
            assert 1 <= start <= end and n == end - start + 1 and n >= 256
            assert rec['source_file'] == f'subject{s}.dat'
            by_subject[s].append((start, end, rec))
    assert set(by_subject) == set(subject_split), 'Missing subject in segment manifest'
    for s, segments in by_subject.items():
        segments.sort(key=lambda tup: tup[0])
        assert all(a[1] < b[0] for a, b in zip(segments, segments[1:])), f'Overlapping segments in subject {s}'

    OUT.mkdir(parents=True, exist_ok=True)
    n_total = 0
    mean = np.zeros(6, dtype=np.float64)
    m2 = np.zeros(6, dtype=np.float64)
    report = {k: {'subjects': v, 'segments':0, 'samples':0, 'lost_by_regime':Counter()} for k,v in groups.items()}
    mask_file = OUT / 'loss_masks.csv'
    with mask_file.open('w', newline='') as fmask:
        writer = csv.writer(fmask)
        writer.writerow(['split','subject','segment_start_row','regime','missing_original_row'])
        for subject in sorted(by_subject):
            split = subject_split[subject]
            path = RAW / f'subject{subject}.dat'
            if not path.exists():
                raise FileNotFoundError(path)
            segments = by_subject[subject]
            index, batch = 0, []
            seg = segments[index]
            rows_checked = 0
            with path.open('r') as f:
                for row_num, line in enumerate(f, start=1):
                    if index >= len(segments):
                        break
                    if row_num < seg[0]:
                        continue
                    if row_num > seg[1]:
                        if batch:
                            n_total, mean, m2 = merge_stats(n_total, mean, m2, np.asarray(batch))
                            batch.clear()
                        index += 1
                        if index >= len(segments):
                            break
                        seg = segments[index]
                        if row_num < seg[0]:
                            continue
                    assert seg[0] <= row_num <= seg[1], f'Invalid segment gap, subject {subject}'
                    fields = line.split()
                    assert len(fields) == 54, (subject, row_num, 'columns')
                    stamp = float(fields[0]); activity = int(fields[1])
                    assert activity == int(seg[2]['activity_id']), (subject, row_num, 'activity')
                    expected_time = float(seg[2]['start_timestamp_s']) + (row_num - seg[0]) * 0.01
                    assert abs(stamp-expected_time) < 0.0010000001, (subject, row_num, 'timestamp')
                    vals = np.asarray([float(fields[i]) for i in COLS], dtype=np.float64)
                    assert np.isfinite(vals).all(), (subject, row_num, 'invalid signal')
                    rows_checked += 1
                    if split == 'train':
                        batch.append(vals)
                        if len(batch) == 10000:
                            n_total, mean, m2 = merge_stats(n_total, mean, m2, np.asarray(batch))
                            batch.clear()
            if batch:
                n_total, mean, m2 = merge_stats(n_total, mean, m2, np.asarray(batch))
            assert rows_checked == sum(end-start+1 for start,end,_ in segments), (subject, 'row count')
            for start, end, rec in segments:
                n = end - start + 1
                report[split]['segments'] += 1
                report[split]['samples'] += n
                for regime in REGIMES:
                    rng = seeded_rng(seed, split, subject, start, regime)
                    missing_idx = make_mask(n, regime, rng)
                    report[split]['lost_by_regime'][regime] += len(missing_idx)
                    for i in missing_idx:
                        writer.writerow([split, subject, start, regime, start + int(i)])
            print(f'Verified subject {subject}: {rows_checked:,} eligible samples ({split})', flush=True)

    assert n_total == report['train']['samples']
    assert n_total > 1
    sd = np.sqrt(m2 / (n_total - 1))
    assert (sd > 0).all() and np.isfinite(sd).all()
    norm = {'method':'per_channel_train_only_standardization', 'count_training_samples':n_total,
            'channel_order':NAMES, 'mean':mean.tolist(), 'std_sample':sd.tolist(),
            'formula':'z=(x-mean)/std_sample', 'fit_split':'train', 'no_normalized_raw_files_written':True}
    (OUT / 'normalization.json').write_text(json.dumps(norm, indent=2) + '\n')
    for stats in report.values():
        stats['lost_by_regime'] = dict(stats['lost_by_regime'])
    metadata = {'schema_version':'1', 'seed':seed, 'history_preserved_samples':HISTORY,
                'nominal_loss_rate':LOSS_RATE, 'regimes':REGIMES,
                'timestamp_tolerance_s':0.001, 'split_config_sha256':hash_file(CONFIG),
                'segment_manifest_sha256':hash_file(MANIFEST),
                'normalization_sha256':hash_file(OUT / 'normalization.json'),
                'mask_sha256':hash_file(mask_file), 'splits':report,
                'warning':'Only artificial missing-position masks; no predictors, leakage from future samples, or arrival-time simulation.'}
    (OUT / 'preparation_report.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('\nPASS: training-only normalization + deterministic loss masks created')
    for split, stats in report.items():
        print(f"{split}: {stats['segments']} segments / {stats['samples']:,} samples; masks: {stats['lost_by_regime']}")
    print('Outputs:', OUT)
    print('No model training or raw-data modification performed.')

if __name__ == '__main__':
    main()
