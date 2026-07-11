"""Greedy headline control: excluded valid boxes act as ignore regions."""
from pathlib import Path
import sys,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_cached_vehicle_grid import CANDIDATES,gt_map,iou_matrix,match
CONF=.25;IOU=.25;HEAD=[('absolute',24.),('normalized',.015)]
def main():
 rows=[]
 for candidate,(source,path,classes) in CANDIDATES.items():
  gt=gt_map(source);raw=pd.read_parquet(path);names=sorted(set(raw.img_name.unique())&set(gt));pred=raw[raw.cls.isin(classes)];groups={n:g for n,g in pred.groupby('img_name',sort=False)}
  for mode,threshold in HEAD:
   tp=fp=fn=0
   for n in names:
    r=gt[n];den=max(r['W'],r['H']); kept=[];removed=[]
    for b in r['boxes']:
     size=b[4] if mode=='absolute' else b[4]/den
     (kept if size>=threshold else removed).append(b)
    pp=[(x.x1,x.y1,x.x2,x.y2,float(x.score)) for x in groups.get(n,pd.DataFrame()).itertuples() if float(x.score)>=CONF]
    ignores=removed+r['ignore']
    pp=[p for p in pp if not ignores or float(iou_matrix([p],ignores).max())<.5]
    a,b,c,_=match(pp,kept,IOU);tp+=a;fp+=b;fn+=c
   pr=tp/(tp+fp) if tp+fp else 0;rc=tp/(tp+fn) if tp+fn else 0
   rows.append({'source':source,'candidate':candidate,'mode':mode,'threshold':threshold,'policy':'exclude_and_ignore_removed','confidence':CONF,'iou':IOU,'images':len(names),'tp':tp,'fp':fp,'fn':fn,'precision':pr,'recall':rc,'f1':2*pr*rc/(pr+rc) if pr+rc else 0})
 out=ROOT/'outputs'/'controls';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'exclude_and_ignore_removed_headline.csv',index=False)
if __name__=='__main__':main()
