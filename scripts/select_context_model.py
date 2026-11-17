"""Compare a frozen previous head with context-trained models on validation only.

The previous model may only have trained on source groups assigned to training
in this experiment. This is exploratory reuse of previously evaluated sources.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from shoplifting.train import load_feature_index,load_bags,_mil_scores,_write_json
from shoplifting.mil import head_from_checkpoint
from shoplifting.metrics import binary_metrics,select_threshold
from shoplifting.decision import calibrate_policy,classify_decision


def prefer_frozen(frozen_validation, current_validation):
    return (frozen_validation['pr_auc'],frozen_validation['f1'])>(current_validation['pr_auc'],current_validation['f1'])


def validate_frozen_groups(old_splits, records):
    old_train={r['group'] for r in old_splits if r['split']=='train'}
    nontraining={r['group'] for r in records if r['split']!='train'}
    if old_train & nontraining:
        raise ValueError('Frozen head training sources overlap context validation/test sources')


def adapt(candidate,baseline,features):
    candidate,baseline,features=map(lambda p:Path(p).resolve(),(candidate,baseline,features))
    records,config=load_feature_index(features)
    old_splits=json.loads((baseline.parent/'split_manifest.json').read_text())['records']
    validate_frozen_groups(old_splits,records)
    checkpoint=torch.load(baseline,map_location='cpu',weights_only=True)
    for key in ['backbone','weights','num_frames','window_seconds','feature_dim']:
        if checkpoint['feature_config'].get(key)!=config.get(key):raise ValueError(f'Incompatible frozen encoder feature setting: {key}')
    bags=load_bags(features,records)
    mean=torch.as_tensor(checkpoint['feature_mean']);scale=torch.as_tensor(checkpoint['feature_scale']).clamp_min(1e-6)
    normalized={i:(v-mean)/scale for i,v in bags.items()}
    head=head_from_checkpoint(checkpoint).eval()
    splits={s:[r for r in records if r['split']==s] for s in ['train','val','test']}
    labels={s:np.array([r['label'] for r in rows]) for s,rows in splits.items()}
    # Make the selection using validation BEFORE computing frozen test scores.
    val=_mil_scores(head,splits['val'],normalized,checkpoint['topk_fraction'],'cpu')
    threshold=select_threshold(labels['val'],val)
    val_metrics=binary_metrics(labels['val'],val,threshold)
    metrics=json.loads((candidate/'metrics.json').read_text())
    current=metrics['selected_model']
    selected=prefer_frozen(val_metrics,metrics['models'][current]['val'])
    scores={'val':val}
    for split in ['train','test']:
        scores[split]=_mil_scores(head,splits[split],normalized,checkpoint['topk_fraction'],'cpu')
    metrics['models']['frozen_transfer']={'threshold':threshold,**{s:binary_metrics(labels[s],scores[s],threshold) for s in splits}}
    rule='Compare frozen previous head and context-trained winner by validation AP, then validation F1; exact tie retains context-trained model. No test values select deployment.'
    metrics['selection']['context_transfer_rule']=rule
    metrics['selection']['frozen_transfer_baseline']=str(baseline.relative_to(ROOT)) if baseline.is_relative_to(ROOT) else str(baseline)
    metrics['limitations'].append('The frozen transferred head was trained on short incident clips; longer inference context does not turn it into a shelf-to-pocket stage detector.')
    if selected:
        checkpoint['feature_config']=config
        checkpoint['threshold']=threshold
        checkpoint['decision_policy']=calibrate_policy(labels['val'],val,threshold)
        checkpoint['selected_model']='frozen_transfer'
        checkpoint['context_adaptation']='Frozen head from source-disjoint short-clip training; denser whole-incident inference windows selected using context validation.'
        checkpoint['deployment_selection_rule']=rule
        torch.save(checkpoint,candidate/'model.pt')
        torch.save(checkpoint,candidate/'frozen_transfer_model.pt')
        _write_json(candidate/'decision_policy.json',checkpoint['decision_policy'])
        metrics['selected_model']='frozen_transfer'
        metrics['selection']['deployed_model']='frozen_transfer'
        policy=checkpoint['decision_policy'];selective={}
        for split in splits:
            statuses=[classify_decision(float(v),float(r['duration']),policy)['status'] for r,v in zip(splits[split],scores[split])]
            decided=np.array([s!='insufficient_evidence' for s in statuses]);flagged=np.array([s=='needs_review' for s in statuses]);y=labels[split]
            selective[split]={'sample_count':len(y),'coverage':float(decided.mean()),'insufficient_evidence_count':int((~decided).sum()),
              'review_count':int(flagged.sum()),'review_precision':float(y[flagged].mean()) if flagged.any() else None,
              'positive_review_recall':float((flagged & (y==1)).sum()/(y==1).sum()),
              'accuracy_among_decided':float((flagged[decided]==y[decided]).mean()) if decided.any() else None}
        metrics['selective_decision']['counts_before_technical_visibility_checks']=selective
    for split,filename in [('val','validation_predictions.csv'),('test','test_predictions.csv')]:
        path=candidate/filename
        with path.open(newline='') as f:reader=csv.DictReader(f);fields=list(reader.fieldnames);rows=list(reader)
        byid={r['id']:float(v) for r,v in zip(splits[split],scores[split])}
        for row in rows:
            row['frozen_transfer_score']=byid[row['id']];row['frozen_transfer_prediction']=int(byid[row['id']]>=threshold)
        for field in ['frozen_transfer_score','frozen_transfer_prediction']:
            if field not in fields:fields.append(field)
        with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    _write_json(candidate/'metrics.json',metrics)
    print('Context deployment:',metrics['selected_model'],'frozen validation AP:',val_metrics['pr_auc'],'context-trained validation AP:',metrics['models'][current]['val']['pr_auc'])
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--features',type=Path,required=True)
    a=p.parse_args();adapt(a.candidate,a.baseline,a.features)
