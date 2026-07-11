"""Adaptive Hungarian control for the compact GRSL headline slice.

The comparison is source-conditional: two frozen VisDrone configurations,
four UAVDT configurations and one AI-TOD configuration.  It is not a
cross-dataset detector leaderboard and it does not claim a fourth prediction
source for structural-only DOTA/xView.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / 'scripts'))
from evaluate_cached_vehicle_grid import CANDIDATES, gt_map, iou_matrix, match

POLICIES=('include_all','exclude_without_ignore_protection','exclude_with_ignore_protection')
HEADLINES=(('absolute',24.0),('normalized',.015))
IOUS=(.25,.50); CONF=.25

def evaluate(candidate, mode, threshold, policy, iou_threshold):
    source,path,classes=CANDIDATES[candidate]; gt=gt_map(source)
    raw=pd.read_parquet(path); covered_names=set(raw.img_name.unique()) & set(gt)
    pred=raw[raw.cls.isin(classes)]
    groups={name:g for name,g in pred.groupby('img_name',sort=False)}
    tp=fp=fn=0; names=sorted(covered_names)
    for name in names:
        record=gt[name]; denom=max(record['W'],record['H'])
        boxes=record['boxes'] if policy=='include_all' else [b for b in record['boxes'] if (b[4] if mode=='absolute' else b[4]/denom)>=threshold]
        predictions=[(r.x1,r.y1,r.x2,r.y2,float(r.score)) for r in groups.get(name,pd.DataFrame()).itertuples() if float(r.score)>=CONF]
        if policy=='exclude_with_ignore_protection' and record['ignore']:
            predictions=[p for p in predictions if float(iou_matrix([p],record['ignore']).max())<.5]
        if not predictions: fn+=len(boxes); continue
        if not boxes: fp+=len(predictions); continue
        matrix=iou_matrix(predictions,boxes)
        # Lexicographic assignment: first maximize the number of threshold-valid
        # edges, then maximize their total IoU.  Padding with dummies lets an
        # unmatched prediction/GT carry zero cost; invalid edges are never given
        # an advantage over a valid edge.
        npred,ngt=matrix.shape; n=npred+ngt; big=float(n+1)
        cost=np.zeros((n,n),dtype=float)
        valid=matrix>=iou_threshold
        cost[:npred,:ngt][valid]=-(big+matrix[valid])
        rr,cc=linear_sum_assignment(cost)
        matched=sum(i<npred and j<ngt and valid[i,j] for i,j in zip(rr,cc)); tp+=matched; fp+=len(predictions)-matched; fn+=len(boxes)-matched
    precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.
    return {'source':source,'candidate':candidate,'mode':mode,'threshold':threshold,'small_object_policy':policy,
            'matching':'hungarian_lexicographic_iou','confidence':CONF,'iou':iou_threshold,'tp':tp,'fp':fp,'fn':fn,'images':len(names),
            'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}

def evaluate_greedy(candidate, mode, threshold, policy, iou_threshold):
    source,path,classes=CANDIDATES[candidate]; gt=gt_map(source)
    raw=pd.read_parquet(path); covered_names=set(raw.img_name.unique()) & set(gt)
    pred=raw[raw.cls.isin(classes)]; groups={name:g for name,g in pred.groupby('img_name',sort=False)}
    tp=fp=fn=0; names=sorted(covered_names)
    for name in names:
        record=gt[name]; denom=max(record['W'],record['H'])
        boxes=record['boxes'] if policy=='include_all' else [b for b in record['boxes'] if (b[4] if mode=='absolute' else b[4]/denom)>=threshold]
        predictions=[(r.x1,r.y1,r.x2,r.y2,float(r.score)) for r in groups.get(name,pd.DataFrame()).itertuples() if float(r.score)>=CONF]
        if policy=='exclude_with_ignore_protection' and record['ignore']:
            predictions=[p for p in predictions if float(iou_matrix([p],record['ignore']).max())<.5]
        a,b,c,_=match(predictions,boxes,iou_threshold);tp+=a;fp+=b;fn+=c
    precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.
    return {'source':source,'candidate':candidate,'mode':mode,'threshold':threshold,'small_object_policy':policy,'matching':'greedy_iou','confidence':CONF,'iou':iou_threshold,'tp':tp,'fp':fp,'fn':fn,'images':len(names),'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}

def main():
    out=ROOT/'outputs'/'matching_headline'; out.mkdir(parents=True,exist_ok=True)
    rows=[]; greedy_rows=[]
    for candidate in CANDIDATES:
        for mode,threshold in HEADLINES:
            for policy in POLICIES:
                for iou in IOUS:
                    row=evaluate(candidate,mode,threshold,policy,iou);rows.append(row);greedy_rows.append(evaluate_greedy(candidate,mode,threshold,policy,iou));print(row,flush=True)
    h=pd.DataFrame(rows); h.to_parquet(out/'hungarian_headline_subset.parquet',index=False);h.to_csv(out/'hungarian_headline_subset.csv',index=False)
    g=pd.DataFrame(greedy_rows)
    g.to_parquet(out/'greedy_headline_subset_coverage_corrected.parquet',index=False)
    merged=g.merge(h,on=['source','candidate','mode','threshold','small_object_policy','confidence','iou','images'],suffixes=('_greedy','_hungarian'),validate='one_to_one')
    merged['f1_delta_hungarian_minus_greedy']=merged.f1_hungarian-merged.f1_greedy
    merged.to_csv(out/'greedy_hungarian_agreement.csv',index=False)
    ranks=[]
    keys=['source','mode','threshold','small_object_policy','iou']
    for key,a in merged.groupby(keys):
        if a.candidate.nunique()<2: continue
        gw=a.sort_values('f1_greedy',ascending=False).iloc[0]; hw=a.sort_values('f1_hungarian',ascending=False).iloc[0]
        row=dict(zip(keys,key)); row.update(greedy_winner=gw.candidate,hungarian_winner=hw.candidate,winner_agree=gw.candidate==hw.candidate,
                     max_abs_f1_delta=float(a.f1_delta_hungarian_minus_greedy.abs().max()),n_candidates=a.candidate.nunique())
        ranks.append(row)
    ranks=pd.DataFrame(ranks);ranks.to_csv(out/'matching_rank_robustness.csv',index=False)
    summary=pd.DataFrame([{'settings':len(merged),'ranked_settings':len(ranks),'winner_agreement_rate':float(ranks.winner_agree.mean()) if len(ranks) else np.nan,
                           'max_abs_f1_delta':float(merged.f1_delta_hungarian_minus_greedy.abs().max()),
                           'mean_abs_f1_delta':float(merged.f1_delta_hungarian_minus_greedy.abs().mean()),
                           'criterion_winner_agreement_ge_0_95':bool(ranks.winner_agree.mean()>=.95) if len(ranks) else False,
                           'criterion_max_abs_f1_delta_lt_0_005':bool(merged.f1_delta_hungarian_minus_greedy.abs().max()<.005)}])
    summary.to_csv(out/'matching_delta_summary.csv',index=False)
    (out/'README.md').write_text('84 source-conditional headline settings: seven frozen candidates x two scale rules x three policies x two IoU thresholds at confidence .25. Greedy rows are selected from the completed frozen grid. Hungarian uses a threshold-first lexicographic assignment (maximum valid-match cardinality, then total IoU) under the same support and source-ignore contract.\n')
    print(json.dumps(summary.iloc[0].to_dict(),indent=2))
if __name__=='__main__': main()
