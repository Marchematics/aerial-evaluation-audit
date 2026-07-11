"""Paired cluster bootstrap for greedy F1 winner-vs-runner-up headline controls."""
from pathlib import Path
import sys,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_cached_vehicle_grid import CANDIDATES,gt_map,iou_matrix,match
PAIRS={'visdrone':('visdrone_sahi640','visdrone_baseline640'),'uavdt':('uavdt_tiling','uavdt_baseline640')}
HEAD=[('absolute',24.),('normalized',.015)];POL=['include_all','exclude_without_ignore_protection','exclude_with_ignore_protection'];IOUS=[.25,.5];CONF=.25;RNG=np.random.default_rng(20260712)
def units(candidate,mode,thr,policy,iou,names,gt):
 source,path,classes=CANDIDATES[candidate]; raw=pd.read_parquet(path);pred=raw[raw.cls.isin(classes)];groups={n:g for n,g in pred.groupby('img_name',sort=False)};rows=[]
 for n in names:
  r=gt[n];den=max(r['W'],r['H']);boxes=r['boxes'] if policy=='include_all' else [b for b in r['boxes'] if (b[4] if mode=='absolute' else b[4]/den)>=thr]
  pp=[(x.x1,x.y1,x.x2,x.y2,float(x.score)) for x in groups.get(n,pd.DataFrame()).itertuples() if float(x.score)>=CONF]
  if policy=='exclude_with_ignore_protection' and r['ignore']: pp=[p for p in pp if float(iou_matrix([p],r['ignore']).max())<.5]
  tp,fp,fn,_=match(pp,boxes,iou);rows.append((n.split('_')[0] if source=='uavdt' else n,tp,fp,fn))
 return pd.DataFrame(rows,columns=['unit','tp','fp','fn']).groupby('unit',as_index=False).sum()
def f1(x):
 tp,fp,fn=x;return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
def main():
 out=[]
 for source,(winner,runner) in PAIRS.items():
  gt=gt_map(source);rawsets=[]
  for cand in (winner,runner): rawsets.append(set(pd.read_parquet(CANDIDATES[cand][1]).img_name.unique())&set(gt))
  names=sorted(set.intersection(*rawsets))
  for mode,thr in HEAD:
   for policy in POL:
    for iou in IOUS:
     a=units(winner,mode,thr,policy,iou,names,gt).set_index('unit');b=units(runner,mode,thr,policy,iou,names,gt).set_index('unit');common=sorted(set(a.index)&set(b.index));a=a.loc[common];b=b.loc[common]
     dif=[];win=[]
     for _ in range(1000):
      idx=RNG.integers(0,len(common),len(common));fa=f1(a.iloc[idx][['tp','fp','fn']].sum().to_numpy());fb=f1(b.iloc[idx][['tp','fp','fn']].sum().to_numpy());dif.append(fa-fb);win.append(fa>fb)
     point=f1(a[['tp','fp','fn']].sum().to_numpy())-f1(b[['tp','fp','fn']].sum().to_numpy())
     out.append({'source':source,'mode':mode,'threshold':thr,'policy':policy,'iou':iou,'winner_candidate':winner,'runner_up_candidate':runner,'common_images':len(names),'common_units':len(common),'point_f1_difference':point,'ci95_low':float(np.quantile(dif,.025)),'ci95_high':float(np.quantile(dif,.975)),'winner_probability':float(np.mean(win)),'replicates':1000,'resampling_unit':'sequence_proxy' if source=='uavdt' else 'image'})
 path=ROOT/'outputs'/'statistics';path.mkdir(exist_ok=True);pd.DataFrame(out).to_csv(path/'bootstrap_headline_f1_common_coverage.csv',index=False)
if __name__=='__main__':main()
