#!/usr/bin/env python3
"""PacketHeal AI Phase 0: read-only PAMAP2 Protocol audit (stdlib only)."""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

CHANNELS = {4:'acc_x_m_s2',5:'acc_y_m_s2',6:'acc_z_m_s2',10:'gyro_x_rad_s',11:'gyro_y_rad_s',12:'gyro_z_rad_s'}
NUMERIC = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')


def percentile(vals, p):
    if not vals: return None
    vals.sort()
    x=(len(vals)-1)*p
    lo=int(x); hi=min(lo+1,len(vals)-1)
    return vals[lo] + (vals[hi]-vals[lo])*(x-lo)


def parse_number(s):
    low=s.lower()
    if low in ('nan','+nan','-nan'): return float('nan'), 'nan'
    if low in ('inf','+inf','-inf','infinity','+infinity','-infinity'): return float('nan'), 'infinity'
    if not NUMERIC.fullmatch(s): return float('nan'), 'bad_token'
    try:
        v=float(s)
    except (ValueError,OverflowError): return float('nan'), 'bad_token'
    if not math.isfinite(v): return float('nan'), 'infinity'
    return v,'valid'


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def audit_file(path, tol, min_len, segment_writer):
    issues=Counter(); missing_by_channel=Counter(); activities=Counter(); valid_activities=Counter()
    distributions={name:[] for name in CHANNELS.values()}
    row_count=0; valid_count=0; prev_t=None; dt_min=float('inf'); dt_max=float('-inf')
    dt_irregular=0; dt_non_increasing=0; delta_count=0
    seg_start=None; seg_start_t=None; seg_activity=None; seg_last_t=None; seg_last_row=None; seg_len=0
    seg_count=0; eligible=0; eligible_samples=0; longest=0; gap_lengths=[]; missing_run=0
    first_samples=[]; invalid_examples=[]
    def finish_segment():
        nonlocal seg_len, seg_start, seg_start_t, seg_activity, seg_last_t, seg_last_row, seg_count, eligible, eligible_samples, longest
        if seg_len:
            seg_count+=1
            longest=max(longest,seg_len)
            if seg_len>=min_len:
                eligible+=1; eligible_samples+=seg_len
                segment_writer.writerow({'subject':path.stem,'source_file':path.name,'start_row':seg_start,'end_row':seg_last_row,'start_timestamp_s':seg_start_t,'end_timestamp_s':seg_last_t,'activity_id':seg_activity,'samples':seg_len,'duration_s':round(seg_last_t-seg_start_t+0.01,6)})
            seg_len=0; seg_start=None
    with path.open('r',encoding='utf-8',errors='replace') as f:
        for row_count,line in enumerate(f,1):
            tokens=line.split(); row_valid=True
            if len(tokens)!=54:
                issues['wrong_column_count_rows']+=1
                if len(invalid_examples)<5: invalid_examples.append(f'row {row_count}: {len(tokens)} columns')
                finish_segment(); prev_t=None
                continue
            timestamp,t_state=parse_number(tokens[0]); act,act_state=parse_number(tokens[1])
            if t_state!='valid': issues['invalid_timestamps']+=1; row_valid=False
            if act_state!='valid' or not act.is_integer(): issues['invalid_activity_ids']+=1; row_valid=False
            activity=int(act) if act_state=='valid' and act.is_integer() else None
            if activity is not None: activities[str(activity)]+=1
            dt_ok=True
            if math.isfinite(timestamp):
                if prev_t is not None:
                    dt=timestamp-prev_t; delta_count+=1
                    dt_min=min(dt_min,dt);dt_max=max(dt_max,dt)
                    if dt<=0: dt_non_increasing+=1
                    if abs(dt-0.01)>tol+1e-10: dt_irregular+=1;dt_ok=False
                prev_t=timestamp
            else:
                prev_t=None;dt_ok=False
            for idx,name in CHANNELS.items():
                value,state=parse_number(tokens[idx])
                if state!='valid':
                    missing_by_channel[name]+=1;issues[f'channel_values_{state}']+=1;row_valid=False
                    if state!='nan' and len(invalid_examples)<5: invalid_examples.append(f'row {row_count}, {name}: {tokens[idx]}')
                else: distributions[name].append(value)
            if row_valid:
                valid_count+=1;valid_activities[str(activity)]+=1
                if missing_run: gap_lengths.append(missing_run);missing_run=0
            else:
                missing_run+=1
            if not row_valid or not dt_ok or (seg_len and activity!=seg_activity): finish_segment()
            if row_valid:
                if not seg_len:
                    seg_start=row_count;seg_start_t=timestamp;seg_activity=activity
                seg_len+=1;seg_last_t=timestamp;seg_last_row=row_count
    finish_segment()
    if missing_run:gap_lengths.append(missing_run)
    distribution_stats={name:{'min':min(v) if v else None,'p01':percentile(v,0.01),'median':percentile(v,0.5),'p99':percentile(v,0.99),'max':max(v) if v else None} for name,v in distributions.items()}
    return {
        'file':path.name,'sha256':sha256(path),'rows':row_count,'valid_six_channel_rows':valid_count,
        'invalid_six_channel_or_metadata_rows':row_count-valid_count,
        'row_valid_fraction':round(valid_count/row_count,6) if row_count else None,
        'issues':dict(issues),'missing_by_channel':dict(missing_by_channel),
        'timestamps':{'intervals_checked':delta_count,'irregular_intervals':dt_irregular,'non_increasing_intervals':dt_non_increasing,'min_dt_s':None if not delta_count else dt_min,'max_dt_s':None if not delta_count else dt_max},
        'activities_all_rows':dict(sorted(activities.items(),key=lambda x:int(x[0]))),
        'activities_valid_rows':dict(sorted(valid_activities.items(),key=lambda x:int(x[0]))),
        'segments':{'total':seg_count,'qualifying':eligible,'qualifying_samples':eligible_samples,'longest_samples':longest},
        'invalid_row_runs':{'count':len(gap_lengths),'max_samples':max(gap_lengths,default=0)},
        'channels':distribution_stats,'examples':invalid_examples,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path,help='path to PAMAP2_Dataset folder containing Protocol')
    p.add_argument('--output',type=Path,default=Path('data/reports/pamap2_audit'))
    p.add_argument('--dt-tolerance-ms',type=float,default=1.0)
    p.add_argument('--min-segment',type=int,default=256)
    a=p.parse_args()
    protocol=a.dataset/'Protocol'
    files=sorted(protocol.glob('subject*.dat'))
    if not files: p.error(f'No subject*.dat found in {protocol}')
    a.output.mkdir(parents=True,exist_ok=True)
    summary=[]
    with (a.output/'segments.csv').open('w',newline='') as g:
        fields=['subject','source_file','start_row','end_row','start_timestamp_s','end_timestamp_s','activity_id','samples','duration_s']
        writer=csv.DictWriter(g,fieldnames=fields);writer.writeheader()
        for file in files:
            print(f'Auditing {file.name} ...',flush=True)
            result=audit_file(file,a.dt_tolerance_ms/1000,a.min_segment,writer)
            summary.append(result)
    with (a.output/'audit.json').open('w') as f:
        json.dump({'config':{'timestamp_tolerance_ms':a.dt_tolerance_ms,'minimum_segment_samples':a.min_segment,'channels':CHANNELS,'segment_policy':'split at invalid six-channel/metadata record, timestamp irregularity, or activity ID change'},'subjects':summary},f,indent=2)
    with (a.output/'summary.csv').open('w',newline='') as f:
        fields=['file','rows','valid_six_channel_rows','invalid_six_channel_or_metadata_rows','irregular_intervals','non_increasing_intervals','qualifying_segments','qualifying_samples','longest_segment_samples']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in summary:
            w.writerow({'file':r['file'],'rows':r['rows'],'valid_six_channel_rows':r['valid_six_channel_rows'],'invalid_six_channel_or_metadata_rows':r['invalid_six_channel_or_metadata_rows'],'irregular_intervals':r['timestamps']['irregular_intervals'],'non_increasing_intervals':r['timestamps']['non_increasing_intervals'],'qualifying_segments':r['segments']['qualifying'],'qualifying_samples':r['segments']['qualifying_samples'],'longest_segment_samples':r['segments']['longest_samples']})
    print('\nFILE          ROWS       VALID      IRREG_DT  SEGMENTS>=256  LONGEST')
    for r in summary:
        print(f"{r['file']:<14}{r['rows']:<11}{r['valid_six_channel_rows']:<11}{r['timestamps']['irregular_intervals']:<10}{r['segments']['qualifying']:<15}{r['segments']['longest_samples']}")
    print(f'\nSaved {a.output}/audit.json, summary.csv, segments.csv')
    print('CAUTION: min/max/percentiles are diagnostic, not validated clipping thresholds. No raw files modified.')

if __name__=='__main__': main()
