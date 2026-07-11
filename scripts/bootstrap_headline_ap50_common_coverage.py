"""Paired image/sequence bootstrap for AP50 differences in declared winner pairs.

This implements the AP50 one-category, maxDets=100 slice used by the headline
COCO control.  Per-image score-ordered TP flags are precomputed once; bootstrap
replicates only reweight image contributions, preserving paired resampling.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_cached_vehicle_grid import CANDIDATES,gt_map,iou_matrix

PAIRS={'visdrone':('visdrone_sahi640','visdrone_baseline640'),'uavdt':('uavdt_tiling','uavdt_baseline640')}
HEAD=[('absolute',24.),('normalized',.015)]
POL={'visdrone':['include_all','exclude_without_ignore_protection','exclude_with_ignore_protection'],
     'uavdt':['include_all','exclude_without_ignore_protection']}
IOU=.5

def records(candidate,mode,threshold,policy,names,gt,max_dets=100):
    source,path,classes=CANDIDATES[candidate];raw=pd.read_parquet(path);pred=raw[raw.cls.isin(classes)]
    groups={n:g for n,g in pred.groupby('img_name',sort=False)}; scores=[]; flags=[]; image_ids=[]; gt_counts=[]
    for ii,n in enumerate(names):
        r=gt[n];den=max(r['W'],r['H'])
        boxes=r['boxes'] if policy=='include_all' else [b for b in r['boxes'] if (b[4] if mode=='absolute' else b[4]/den)>=threshold]
        pp=[(x.x1,x.y1,x.x2,x.y2,float(x.score)) for x in groups.get(n,pd.DataFrame()).itertuples()]
        if policy=='exclude_with_ignore_protection' and r['ignore']:
            pp=[p for p in pp if float(iou_matrix([p],r['ignore']).max())<.5]
        pp.sort(key=lambda x:-x[4]);pp=pp[:max_dets];gt_counts.append(len(boxes))
        if not pp: continue
        if not boxes:
            scores.extend([x[4] for x in pp]);flags.extend([0]*len(pp));image_ids.extend([ii]*len(pp));continue
        mat=iou_matrix(pp,boxes);unused=np.ones(len(boxes),dtype=bool)
        for pi,p in enumerate(pp):
            v=np.where(unused,mat[pi],-1.);gi=int(np.argmax(v));hit=int(v[gi]>=IOU)
            if hit:unused[gi]=False
            scores.append(p[4]);flags.append(hit);image_ids.append(ii)
    scores=np.asarray(scores,float);flags=np.asarray(flags,np.int8);image_ids=np.asarray(image_ids,np.int32);gt_counts=np.asarray(gt_counts,np.int32)
    # Stable ordering makes tie handling deterministic and common across reps.
    order=np.argsort(-scores,kind='mergesort')
    return {'score':scores[order],'tp':flags[order],'img':image_ids[order],'gt':gt_counts}

def ap50(rec,weights):
    denom=float(np.dot(weights,rec['gt']))
    if denom<=0:return np.nan
    w=weights[rec['img']];tp=np.cumsum(w*rec['tp']);fp=np.cumsum(w*(1-rec['tp']))
    recall=tp/denom;precision=tp/np.maximum(tp+fp,1e-12)
    # COCO's 101 recall thresholds after the monotone precision envelope.
    env=np.maximum.accumulate(precision[::-1])[::-1];idx=np.searchsorted(recall,np.linspace(0,1,101),side='left')
    vals=np.zeros(101);valid=idx<len(env);vals[valid]=env[idx[valid]]
    return float(vals.mean())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--replicates',type=int,default=1000);ap.add_argument('--seed',type=int,default=20260712);ap.add_argument('--max-dets',type=int,default=100);ap.add_argument('--out',type=Path,default=None);args=ap.parse_args();rng=np.random.default_rng(args.seed);rows=[]
    for source,(winner,runner) in PAIRS.items():
        gt=gt_map(source);covered=[set(pd.read_parquet(CANDIDATES[c][1]).img_name.unique())&set(gt) for c in (winner,runner)];names=sorted(set.intersection(*covered));units=[n.split('_')[0] if source=='uavdt' else n for n in names];uindex={u:i for i,u in enumerate(sorted(set(units)))};img_unit=np.array([uindex[u] for u in units])
        for mode,threshold in HEAD:
            for policy in POL[source]:
                a=records(winner,mode,threshold,policy,names,gt,args.max_dets);b=records(runner,mode,threshold,policy,names,gt,args.max_dets)
                one=np.ones(len(names),dtype=int);point_a=ap50(a,one);point_b=ap50(b,one);diff=[];wins=[]
                for _ in range(args.replicates):
                    sampled=rng.integers(0,len(uindex),len(uindex));unit_w=np.bincount(sampled,minlength=len(uindex));weights=unit_w[img_unit]
                    da=ap50(a,weights);db=ap50(b,weights);diff.append(da-db);wins.append(da>db)
                rows.append({'source':source,'mode':mode,'threshold':threshold,'policy':policy,'iou':IOU,'max_dets':args.max_dets,'winner_candidate':winner,'runner_up_candidate':runner,'common_images':len(names),'common_units':len(uindex),'point_ap50_winner':point_a,'point_ap50_runner_up':point_b,'point_ap50_difference':point_a-point_b,'ci95_low':float(np.quantile(diff,.025)),'ci95_high':float(np.quantile(diff,.975)),'winner_probability':float(np.mean(wins)),'replicates':args.replicates,'resampling_unit':'sequence_proxy' if source=='uavdt' else 'image'})
                print(rows[-1],flush=True)
    out=ROOT/'outputs/statistics';out.mkdir(exist_ok=True);target=args.out or out/('bootstrap_headline_ap50_common_coverage.csv' if args.max_dets==100 else f'ap50_maxdets_{args.max_dets}_sensitivity.csv');pd.DataFrame(rows).to_csv(target,index=False)
if __name__=='__main__':main()
