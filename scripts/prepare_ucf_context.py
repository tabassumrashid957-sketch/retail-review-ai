"""Preserve complete annotated incidents with context; no shelf/pocket stage labels."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_ucf_crime import parse_annotations, video_details, file_sha256, write_clip, complement_intervals, atomic_json
BASE = ROOT / 'data/real_retail/ucf_crime'


def plan_pairs(events, frames, fps, context_seconds=2, guard_seconds=1):
    """Never shorten an incident merely to find a matching negative."""
    available = complement_intervals(events, frames, round(guard_seconds * fps))
    plans, skipped = [], []
    pad = round(context_seconds * fps)
    for index, (event_start, event_end) in enumerate(events):
        positive = (max(0, event_start-pad), min(frames, event_end+pad))
        width = positive[1]-positive[0]
        options = [(end-start, start, end, n) for n,(start,end) in enumerate(available) if end-start >= width]
        if not options:
            skipped.append({'event_index': index, 'reason': 'no duration-matched non-event interval; complete positive was not truncated'})
            continue
        _, start, end, position = max(options)
        begin = start+(end-start-width)//2
        normal = (begin, begin+width)
        remaining = [(start,begin),(begin+width,end)]
        available = available[:position]+[(a,b) for a,b in remaining if b>a]+available[position+1:]
        plans.append({'event_index': index, 'positive': positive, 'normal': normal, 'event': (event_start,event_end)})
    return plans, skipped


def prepare(videos=BASE/'source_videos', annotations=BASE/'annotations.txt', output=BASE/'context', splits=None):
    if splits is None:
        saved = Path(output)/'source_splits.json'
        splits = saved if saved.is_file() else ROOT/'artifacts/model/split_manifest.json'
    videos, annotations, output, splits = map(lambda p:Path(p).resolve(), (videos,annotations,output,splits))
    if videos == output or videos in output.parents:
        raise ValueError('Output must be separate from source videos')
    groups = {}
    for row in json.loads(splits.read_text())['records']:
        if row['group'] in groups and groups[row['group']] != row['split']:
            raise ValueError('Source group crosses baseline splits')
        groups[row['group']] = row['split']
    old = {}
    meta_path = output/'metadata.json'
    if meta_path.is_file():
        old = {r['id']:r for r in json.loads(meta_path.read_text()).get('records',[])}
    rows, skipped = [], []
    for name, events in sorted(parse_annotations(annotations).items()):
        source = videos/name
        details = video_details(source)
        if any(end>details['frames'] for _,end in events):raise ValueError(f'Annotation exceeds source {name}')
        sha = file_sha256(source)
        group = 'ucf:'+name
        if group not in groups:raise ValueError(f'Missing original split for {group}')
        pairs, omissions = plan_pairs(events, details['frames'], details['fps'])
        skipped.extend([{'source_video':name, **r} for r in omissions])
        for pair in pairs:
            for label, kind, category in [(0,'normal','Normal'),(1,'positive','Shoplifting')]:
                start,end = pair[kind]
                identifier = f'{sha[:16]}-context-{kind}-{start}-{end}'
                path = output/'clips'/category/(identifier+'.mp4')
                cached = old.get(identifier)
                if not (cached and path.is_file() and cached['source_sha256']==sha and cached['clip_sha256']==file_sha256(path)):
                    write_clip(source,path,start,end,details)
                row = {'id':identifier,'path':str(path),'label':label,'group':group,'split':groups[group],
                       'duration':(end-start)/details['fps'],'fps':details['fps'],'source_video':name,
                       'start_frame':start,'end_frame':end,'source_sha256':sha,'clip_sha256':file_sha256(path),
                       'annotation_event_start_frame':pair['event'][0], 'annotation_event_end_frame':pair['event'][1],
                       'complete_annotation_included': bool(label and start<=pair['event'][0] and end>=pair['event'][1])}
                if label and not row['complete_annotation_included']:raise ValueError('Truncated positive')
                rows.append(row)
                print(f'[{len(rows)}] {identifier}: {row["duration"]:.2f}s',flush=True)
    if not rows or {r['label'] for r in rows}!={0,1}:raise ValueError('No balanced context dataset')
    for split in ['train','val','test']:
        if {r['label'] for r in rows if r['split']==split}!={0,1}:raise ValueError(f'Missing split coverage: {split}')
    output.mkdir(parents=True,exist_ok=True)
    if splits != output/'source_splits.json':
        atomic_json(output/'source_splits.json',json.loads(splits.read_text()))
    with (output/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    metadata={'schema_version':1,'context_seconds':2,'negative_guard_seconds':1,'duration_matched_pairs':True,
              'clip_count':len(rows),'source_count':len({r['group'] for r in rows}),'records':rows,'skipped':skipped,
              'split_rule':'Exact source assignments preserved from previous short-clip experiment.',
              'evaluation_caveat':'Exploratory follow-up reuses previous test sources; not a pristine new holdout or official UCF benchmark.',
              'label_scope':'Complete original incident annotation plus context, NOT independent labels for shelf pickup, object trajectory or pocket placement.'}
    atomic_json(meta_path,metadata)
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos',type=Path,default=BASE/'source_videos');p.add_argument('--annotations',type=Path,default=BASE/'annotations.txt')
    p.add_argument('--output',type=Path,default=BASE/'context');p.add_argument('--splits',type=Path)
    a=p.parse_args();m=prepare(a.videos,a.annotations,a.output,a.splits)
    print(f'Prepared {m["clip_count"]} context clips from {m["source_count"]} original recordings.')
