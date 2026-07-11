"""Materialize coverage-corrected full-grid artifacts and conservative rank summaries.

The raw prediction artifact, rather than class-filtered predictions, defines
whether an image is evaluated. Thus an image with no retained vehicle-class
prediction remains an empty-prediction evaluation unit.  This script never
mixes the legacy grid with the corrected grid.
"""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'outputs'/'candidate_grid_coverage_corrected'
OUT=ROOT/'outputs'/'coverage_corrected_grid'

HEAD={'absolute':('scale_threshold_px',24.0),'normalized':('scale_threshold_norm',.015)}
POLICIES=['include_all','exclude_without_ignore_protection','exclude_with_ignore_protection']

def main():
    all_modes=[]
    for mode in HEAD:
        paths=sorted((IN/mode).glob('*/metrics_long.parquet'))
        if len(paths)!=7:
            raise RuntimeError(f'{mode}: expected seven candidate artifacts, found {len(paths)}')
        d=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
        if d.candidate.nunique()!=7:
            raise RuntimeError(f'{mode}: expected seven candidates')
        d['scale_mode']=mode
        target=OUT/mode; target.mkdir(parents=True,exist_ok=True)
        d.drop(columns='scale_mode').to_parquet(target/'metrics_long.parquet',index=False)
        d.drop(columns='scale_mode').to_csv(target/'metrics_long.csv',index=False)
        all_modes.append(d)
    full=pd.concat(all_modes,ignore_index=True)
    OUT.mkdir(parents=True,exist_ok=True)
    full.to_parquet(OUT/'metrics_long.parquet',index=False)

    headline=[]; winners=[]; independent=[]
    for mode,(threshold_col,threshold) in HEAD.items():
        h=full[(full.scale_mode==mode)&(full.confidence==.25)&(full.iou==.25)&(full[threshold_col]==threshold)].copy()
        for (source,candidate),g in h.groupby(['source','candidate'],sort=True):
            lo,hi=float(g.f1.min()),float(g.f1.max())
            headline.append({'source':source,'candidate':candidate,'mode':mode,'threshold':threshold,
                             'f1_min':lo,'f1_max':hi,'f1_policy_band':hi-lo,'images':int(g.images.iloc[0]),
                             'policies_evaluated':int(g.small_object_policy.nunique())})
        # Winner checks only have inferential content for multi-candidate pools.
        for (source,policy),g in h.groupby(['source','small_object_policy'],sort=True):
            n=g.candidate.nunique()
            row={'source':source,'mode':mode,'threshold':threshold,'small_object_policy':policy,
                 'candidate_count':int(n),'images':int(g.images.iloc[0])}
            if n>=2:
                ranked=g.sort_values(['f1','candidate'],ascending=[False,True]).reset_index(drop=True)
                row.update(winner=ranked.candidate.iloc[0],winner_f1=float(ranked.f1.iloc[0]),
                           runner_up=ranked.candidate.iloc[1],runner_up_f1=float(ranked.f1.iloc[1]),
                           winner_margin=float(ranked.f1.iloc[0]-ranked.f1.iloc[1]),rank_stability_testable=True)
            else:
                row.update(winner=None,winner_f1=None,runner_up=None,runner_up_f1=None,winner_margin=None,rank_stability_testable=False)
            winners.append(row)
        # Determine duplicate policy *outcomes* at this headline, separately by
        # source.  Exact metric tuples make the count auditable rather than
        # assuming that source-ignore behavior is identical.
        for source,g in h.groupby('source',sort=True):
            sig=[]
            for policy,p in g.groupby('small_object_policy',sort=True):
                q=p.sort_values('candidate')[['candidate','tp','fp','fn','f1']]
                sig.append((policy,tuple(map(tuple,q.to_numpy()))))
            n_unique=len({x[1] for x in sig})
            independent.append({'source':source,'mode':mode,'threshold':threshold,
                                'policies_declared':len(sig),'independent_policy_outcomes':n_unique,
                                'duplicate_policy_groups':json.dumps([[a for a,b in sig if b==signature]
                                    for signature in dict.fromkeys(b for a,b in sig)],default=str)})
    pd.DataFrame(headline).to_csv(OUT/'headline_f1_policy_bands.csv',index=False)
    pd.DataFrame(winners).to_csv(OUT/'headline_f1_winners.csv',index=False)
    independent=pd.DataFrame(independent); independent.to_csv(OUT/'headline_policy_independence.csv',index=False)

    ranked=pd.DataFrame(winners).query('rank_stability_testable')
    # The manuscript's matcher headline also includes IoU=.50.  Compute the
    # analogous independent-policy count directly instead of multiplying a
    # .25-only count by assumption.
    f1_raw=f1_effective=0
    for mode,(threshold_col,threshold) in HEAD.items():
        h=full[(full.scale_mode==mode)&(full.confidence==.25)&(full.iou.isin([.25,.50]))&(full[threshold_col]==threshold)]
        for (source,iou),g in h.groupby(['source','iou'],sort=True):
            if g.candidate.nunique()<2: continue
            f1_raw += int(g.small_object_policy.nunique())
            signatures=[]
            for _,p in g.groupby('small_object_policy',sort=True):
                q=p.sort_values('candidate')[['candidate','tp','fp','fn','f1']]
                signatures.append(tuple(map(tuple,q.to_numpy())))
            f1_effective += len(set(signatures))
    counts={
        'raw_multi_candidate_f1_headline_settings_iou025':int(len(ranked)),
        'effective_multi_candidate_f1_policy_outcomes_iou025':int(independent[independent.source.isin(['visdrone','uavdt'])].independent_policy_outcomes.sum()),
        'raw_multi_candidate_f1_headline_settings_iou025_iou050':f1_raw,
        'effective_multi_candidate_f1_policy_outcomes_iou025_iou050':f1_effective,
        'note':'AI-TOD has one candidate and is excluded from rank-stability counts. Effective counts collapse exactly duplicate source-ignore policy outcomes.'
    }
    (OUT/'README.md').write_text(
        'Coverage-corrected full greedy grid. Image coverage is determined from raw prediction rows before semantic class filtering; therefore empty vehicle predictions are evaluated. `headline_f1_winners.csv` reports only multi-candidate rank checks as testable. `headline_policy_independence.csv` identifies duplicate policy outcomes.\n')
    (OUT/'summary.json').write_text(json.dumps(counts,indent=2)+'\n')
    print(json.dumps({'rows':len(full),'headline_rows':len(headline),'ranked_headline_rows':len(ranked),**counts},indent=2))

if __name__=='__main__':
    main()
