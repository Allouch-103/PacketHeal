#!/usr/bin/env python3
"""Independently reconcile saved Phase-2 arithmetic and two raw-data cases.

Uses math.fsum for every saved stratum and direct past-index formulas for
segments 0 and 95 under all masks. Does not call predictor.step() or the new
diagnostic scorer for the independent reconstruction check.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from benchmark_classical_baselines import ALGORITHMS, ROOT, validation_access_guard
from evaluate_locf import sha256
from packetheal.evaluation import load_subject_masks
from packetheal.splits import load_split_segment


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"Independent arithmetic mismatch: {actual} != {expected}")


def direct_forecasts(truth, missing, caps):
    """Direct formulas indexed by preceding RECEIVED positions, without replay."""
    locf, linear = truth.copy(), truth.copy()
    received = np.flatnonzero(~missing)
    for i in np.flatnonzero(missing):
        pos = int(np.searchsorted(received, i)) - 1
        if pos < 0:
            locf[i], linear[i] = np.nan, np.nan
            continue
        last = int(received[pos])
        locf[i] = truth[last]
        linear[i] = truth[last]
        if pos:
            previous = int(received[pos - 1])
            for c, cap in enumerate(caps):
                delta = float(truth[last, c] - truth[previous, c]) / (last - previous)
                limited = min(max(delta, -cap), cap)
                linear[i, c] = truth[last, c] + (i - last) * limited
    return {"locf": locf, "linear": linear, "alpha_beta": locf.copy()}


def verify_saved(report):
    if (report['subjects'], report['segments'], report['samples'], report['cases'], report['algorithm_cases']) != (
            [107], 102, 285488, 408, 1224):
        raise ValueError('Unexpected validation coverage')
    cases = report['per_segment']
    regimes = report['policy']['regimes']
    if len(cases) != 408 or {(c['segment_index'],c['regime']) for c in cases} != {
            (i,r) for i in range(102) for r in regimes}:
        raise ValueError('Case identities are incomplete or duplicated')
    for regime, entry in report['pooled_by_regime'].items():
        selected = [c for c in cases if c['regime'] == regime]
        for name in ALGORITHMS:
            pooled = entry['algorithms'][name]
            records = [c['algorithms'][name] for c in selected]
            metrics = pooled['metrics']
            count = sum(r['metrics']['scored_samples'] for r in records)
            if count != metrics['hidden_samples'] or metrics['unavailable_samples']:
                raise ValueError('Unexpected overall coverage')
            for channel in range(6):
                absolute = math.fsum(r['metrics']['absolute_error_sum'][channel] for r in records)
                squared = math.fsum(r['metrics']['squared_error_sum'][channel] for r in records)
                for field,value in [('absolute_error_sum',absolute), ('squared_error_sum',squared),
                                    ('mae',absolute/count), ('rmse',math.sqrt(squared/count))]:
                    close(metrics[field][channel],value)
            for group, values in pooled['strata'].items():
                for channel in range(6):
                    parts = [r['strata'][group] for r in records]
                    for field in ('hidden_count','scored_count','unavailable_count'):
                        if values[field][channel] != sum(p[field][channel] for p in parts):
                            raise ValueError('Per-channel stratum count mismatch')
                    n = values['scored_count'][channel]
                    a = math.fsum(p['absolute_error_sum'][channel] for p in parts)
                    s = math.fsum(p['squared_error_sum'][channel] for p in parts)
                    close(values['absolute_error_sum'][channel],a)
                    close(values['squared_error_sum'][channel],s)
                    if n:
                        close(values['mae'][channel],a/n)
                        close(values['rmse'][channel],math.sqrt(s/n))
                    elif values['mae'][channel] is not None or values['rmse'][channel] is not None:
                        raise ValueError('Empty stratum claims an accuracy value')
            for parts in (['large_change','other'],[f'age_{a}' for a in report['policy']['loss_ages'][regime]]):
                for channel in range(6):
                    for field in ('hidden_count','scored_count','unavailable_count','absolute_error_sum','squared_error_sum'):
                        close(math.fsum(pooled['strata'][p][field][channel] for p in parts),
                              pooled['strata']['all'][field][channel])
        if entry['algorithms']['locf'] != entry['algorithms']['alpha_beta']:
            raise ValueError('Selected alpha-beta is not identical to LOCF')


def verify_csv(directory, report):
    expected_sizes = {'overall.csv':72, 'transients.csv':144, 'loss_age.csv':378, 'loss_runs.csv':7}
    for name,size in expected_sizes.items():
        with (directory/name).open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        if len(rows) != size:
            raise ValueError(f'Incomplete CSV: {name}')
        seen=set()
        for row in rows:
            entry=report['pooled_by_regime'][row['regime']]
            if name=='loss_runs.csv':
                length=int(row['run_length']); runs=entry['run_length_histogram'][str(length)]
                if int(row['runs'])!=runs or int(row['hidden_vectors'])!=length*runs:
                    raise ValueError('Run CSV mismatch')
                key=(row['regime'],length)
            else:
                channel=report['channel_order'].index(row['channel'])
                if row['unit']!=report['channel_units'][channel]:
                    raise ValueError('Physical units differ')
                algorithm=entry['algorithms'][row['algorithm']]
                group=row.get('group','all'); key=(row['regime'],row['algorithm'],channel,group)
                values=algorithm['strata'][group]
                for field in ('hidden_count','scored_count','unavailable_count','mae','rmse'):
                    value=values[field][channel]
                    if value is None:
                        if row[field]!='': raise ValueError('Empty CSV metric must be blank')
                    else: close(float(row[field]),value)
                if name!='overall.csv':
                    for field in ('absolute_error_sum','squared_error_sum','coverage'):
                        value=values[field][channel]
                        if value is None:
                            if row[field]!='': raise ValueError('Empty CSV coverage must be blank')
                        else: close(float(row[field]),value)
                    total=algorithm['metrics']['squared_error_sum'][channel]
                    if total: close(float(row['squared_error_fraction_of_all']),values['squared_error_sum'][channel]/total)
                    elif row['squared_error_fraction_of_all']!='': raise ValueError('Undefined fraction must be blank')
            if key in seen: raise ValueError('Duplicate CSV selection')
            seen.add(key)


def verify_raw_cases(report):
    paths = ROOT/'data/processed/pamap2_v1'
    dataset = ROOT/'data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset'
    config,manifest = ROOT/'configs/pamap2_split_v1.json',ROOT/'configs/pamap2_segments_v1.csv'
    segments,masks = load_subject_masks(paths/'loss_masks.csv',manifest,config,'validation',107)
    lookup={(c['segment_index'],c['regime']):c for c in report['per_segment']}
    if (report['alpha'],report['beta'])!=(1,0):
        raise ValueError('Direct alpha-beta equivalence check requires frozen (1,0) gains')
    checked=[]
    for index in (0,95):
        _,truth=load_split_segment(dataset,manifest,config,'validation',107,index)
        for regime in report['policy']['regimes']:
            mask=masks[index,regime]
            direct=direct_forecasts(truth,mask,report['large_change_thresholds'])
            ages=[]; age=0
            for hidden in mask:
                age=age+1 if hidden else 0
                ages.append(age)
            for name,forecast in direct.items():
                case=lookup[index,regime]['algorithms'][name]
                for channel,threshold in enumerate(report['large_change_thresholds']):
                    errors={group:[] for group in case['strata']}
                    for i in np.flatnonzero(mask):
                        error=float(forecast[i,channel]-truth[i,channel])
                        change=i>0 and abs(float(truth[i,channel]-truth[i-1,channel]))>threshold
                        for group in ('all','large_change' if change else 'other',f'age_{ages[i]}'):
                            errors[group].append(error)
                    for group,values in errors.items():
                        saved=case['strata'][group]
                        if saved['scored_count'][channel]!=len(values) or saved['unavailable_count'][channel]:
                            raise ValueError('Direct raw-case group membership/count differs')
                        close(saved['absolute_error_sum'][channel],math.fsum(abs(e) for e in values))
                        close(saved['squared_error_sum'][channel],math.fsum(e*e for e in values))
            checked.append({'segment_index':index,'regime':regime,'hidden_samples':int(mask.sum())})
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--report-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(f'Refusing to overwrite {args.output}')
    if sha256(args.report)!=args.report_sha256: raise ValueError('Report checksum mismatch')
    report=json.loads(args.report.read_text())
    verify_saved(report)
    verify_csv(args.report.parent,report)
    recording=ROOT/'data/raw/pamap2+physical+activity+monitoring/PAMAP2_Dataset/Protocol/subject107.dat'
    with validation_access_guard(ROOT/'data/raw',recording) as access:
        if sha256(recording)!=report['provenance']['artifact_sha256']['validation_recording']:
            raise ValueError('Raw subject107 hash differs')
        checked=verify_raw_cases(report)
        for source,expected in report['provenance']['source_sha256'].items():
            if sha256(ROOT/source)!=expected: raise ValueError(f'Evaluated source changed: {source}')
    if sha256(args.report)!=args.report_sha256: raise ValueError('Report changed during verification')
    result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'report_sha256':args.report_sha256,
        'independent_math_fsum_all_saved_overall_and_stratum_metrics':True,
        'csv_rows_and_values_reconciled':{'overall':72,'transients':144,'loss_age':378,'loss_runs':7},
        'direct_past_index_predictions_and_scalar_change_labels_match_raw_cases':checked,
        'raw_access':access,'evaluated_source_hashes_unchanged':True,
        'verifier_source_sha256':sha256(Path(__file__)),
        'scope':'Independent saved arithmetic for all cases; direct formulas/labels on two segments under four regimes, not an independent full 408-case predictor implementation.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2); stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
