"""Re-evaluate cached raw prediction boxes under the registered vehicle rule grid."""
from __future__ import annotations
import json, sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from label_reliability.inventory import load_ontology, load_uavdt_coco, load_visdrone_split
from label_reliability.aitod_provenance_bridge import load_aitod, valid_vehicle_records

ROOT=Path(__file__).resolve().parents[1]
CONF=[.10,.15,.20,.25,.30,.35,.40,.50]; IOU=[.25,.30,.40,.50]; THR=[16,20,24,28,32,40,48,64]
CANDIDATES={
 'visdrone_baseline640':('visdrone','/root/zjh_UAV_detection/experiments/visdrone/oracle_route/cache_val_baseline640/pred_rows.parquet',{3,4,5,8}),
 'visdrone_sahi640':('visdrone','/root/zjh_UAV_detection/experiments/visdrone/oracle_route/router_runs/sahi_baseline/canonical_sahi_640_ov20/sahi_pred_rows.parquet',{3,4,5,8}),
 'uavdt_baseline640':('uavdt','/root/zjh_UAV_detection/experiments/uavdt/oracle_route/cache_val_baseline640/pred_rows.parquet',{0,1,2}),
 'uavdt_yolo11n640':('uavdt','/root/zjh_UAV_detection/experiments/uavdt/oracle_route/router_runs/yolo11n_val_pred_rows_imgsz640_conf0p001.parquet',{0,1,2}),
 'uavdt_yolo11m704':('uavdt','/root/zjh_UAV_detection/experiments/uavdt/oracle_route/router_runs/probe_cache_val_yolo11m_imgsz704_conf0p001/pred_rows.parquet',{0,1,2}),
 'uavdt_tiling':('uavdt','/root/zjh_UAV_detection/experiments/uavdt/oracle_route/router_runs/tiling_baseline/pred_rows.parquet',{0,1,2}),
 'aitod_baseline640':('aitod','/root/zjh_UAV_detection/experiments/aitod/oracle_route/cache_val_baseline640/pred_rows.parquet',{5}),
}

def gt_map(source):
 ont=load_ontology(ROOT/'label_reliability/config/ontology.json')
 if source=='visdrone': rec=load_visdrone_split(Path('/root/zjh_UAV_detection/experiments/visdrone/data/VisDrone/VisDrone2019-DET-val'),'val',ont)
 elif source=='uavdt': rec=load_uavdt_coco(Path('/root/zjh_UAV_detection/external/zoomdet_yolo_official/data/UAVDT_local/annotations/test.json'),ont)
 else: rec=load_aitod(Path('/root/zjh_UAV_detection/datasets/aitod/annotations/aitod_val.json'),Path('/root/zjh_UAV_detection/datasets/aitod/images/val'))
 return {r['file_name']:{'boxes':[(b.x1,b.y1,b.x2,b.y2,b.max_side) for b in r['boxes'] if not b.is_source_ignore and b.shared_coarse == 'vehicle'], 'ignore':[(b.x1,b.y1,b.x2,b.y2,b.max_side) for b in r['boxes'] if b.is_source_ignore], 'W':r['width'],'H':r['height']} for r in rec}

def iou_matrix(p,g):
 if not len(p) or not len(g): return np.zeros((len(p),len(g)))
 p=np.asarray(p)[:,:4]; g=np.asarray(g)[:,:4]
 xx1=np.maximum(p[:,None,0],g[None,:,0]); yy1=np.maximum(p[:,None,1],g[None,:,1]); xx2=np.minimum(p[:,None,2],g[None,:,2]); yy2=np.minimum(p[:,None,3],g[None,:,3])
 inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1); ap=(p[:,2]-p[:,0])*(p[:,3]-p[:,1]); ag=(g[:,2]-g[:,0])*(g[:,3]-g[:,1]); return inter/(ap[:,None]+ag[None,:]-inter+1e-12)

def match(pred,gt,iou_thr):
 if not len(pred): return 0,0,len(gt),0
 if not len(gt): return 0,len(pred),0,len(pred)
 mat=iou_matrix(pred,gt); unused=np.ones(len(gt),dtype=bool); tp=0
 for pi in np.argsort([-x[4] for x in pred]):
  # Standard confidence-ordered greedy detection matching: a prediction is
  # paired with its best *currently unmatched* GT.  Choosing the global best
  # GT first and then discarding the prediction when it is already used would
  # under-count valid matches that have a second eligible GT.
  if not unused.any(): break
  scores=np.where(unused,mat[pi],-1.0); gi=int(np.argmax(scores))
  if scores[gi]>=iou_thr: unused[gi]=False; tp+=1
 return tp,len(pred)-tp,len(gt)-tp, len(pred)

def main():
 configs={
  'visdrone':('experiments/visdrone/oracle_route/cache_val_baseline640/pred_rows.parquet',{3,4,5,8}),
  'uavdt':('experiments/uavdt/oracle_route/cache_val_baseline640/pred_rows.parquet',{0,1,2}),
  'aitod':('/root/zjh_UAV_detection/experiments/aitod/oracle_route/cache_val_baseline640/pred_rows.parquet',{5}),}
 ap=argparse.ArgumentParser(); ap.add_argument('--source',choices=list(configs),default=None); ap.add_argument('--candidate',choices=list(CANDIDATES),default=None); ap.add_argument('--out',type=Path,default=ROOT/'outputs/rule_grid'); ap.add_argument('--reference-only',action='store_true'); ap.add_argument('--thresholds',type=str,default=None,help='comma-separated absolute support thresholds'); ap.add_argument('--confidence',type=float,default=None); ap.add_argument('--iou',type=float,default=None); args=ap.parse_args()
 if args.reference_only:
  CONF[:] = [.25]; IOU[:] = [.25]
 if args.thresholds:
  THR[:] = [float(x) for x in args.thresholds.split(',') if x.strip()]
 if args.confidence is not None:
  CONF[:] = [float(args.confidence)]
 if args.iou is not None:
  IOU[:] = [float(args.iou)]
 out=args.out; out.mkdir(parents=True,exist_ok=True); allrows=[]; imrows=[]
 if args.candidate:
  cs,cp,cc=CANDIDATES[args.candidate]; selected={cs:(cp,cc)}; candidate_label=args.candidate
 else:
  selected={args.source:configs[args.source]} if args.source else configs; candidate_label='cache_val_baseline640'
 for source,(pred_path,classes) in selected.items():
  # Establish evaluation coverage before semantic class filtering.  An image
  # present in the raw prediction artifact but with no retained vehicle-class
  # predictions is an empty prediction, not a missing evaluation image.
  gt=gt_map(source); raw_pred=pd.read_parquet(pred_path)
  names=sorted(set(gt) & set(raw_pred.img_name.astype(str).unique()))
  pred=raw_pred[raw_pred.cls.isin(classes)]
  grouped={n:g for n,g in pred.groupby('img_name',sort=False)}
  print(source,'images',len(names),'preds',len(pred),'gt',sum(len(gt[n]['boxes']) for n in names),flush=True)
  agg={}
  for conf in CONF:
   for iou in IOU:
    for thr in THR:
     for policy in ['include_all','exclude_without_ignore_protection','exclude_with_ignore_protection']:
      agg[(conf,iou,thr,policy)] = [0,0,0,0.0,0]
  for name in names:
   g=gt[name]['boxes']; pg=grouped.get(name,pd.DataFrame())
   preds_all=[(r.x1,r.y1,r.x2,r.y2,float(r.score)) for r in pg.itertuples()]
   for conf in CONF:
    p=[x for x in preds_all if x[4]>=conf]
    for iou in IOU:
     tp0,fp0,fn0,_=match(p,g,iou)
     for thr in THR:
      for policy in ['include_all','exclude_without_ignore_protection','exclude_with_ignore_protection']:
       if policy=='include_all':
        tp,fp,fn=tp0,fp0,fn0
       else:
        gg=[x for x in g if x[4]>=thr]; ignored=gt[name].get('ignore',[])
        if policy=='exclude_without_ignore_protection':
         tp,fp,fn,_=match(p,gg,iou)
        else:
         rem=[x for x in p if not ignored or float(iou_matrix([x],ignored).max())<0.5]
         tp,fp,fn,_=match(rem,gg,iou)
       z=agg[(conf,iou,thr,policy)]; z[0]+=tp; z[1]+=fp; z[2]+=fn; z[3]+=abs(len(p)-len(g)); z[4]+=1
  for (conf,iou,thr,policy),(TP,FP,FN,count_sum,nimg) in agg.items():
   prec=TP/(TP+FP) if TP+FP else 0; rec=TP/(TP+FN) if TP+FN else 0; f1=2*prec*rec/(prec+rec) if prec+rec else 0
   allrows.append({'source':source,'candidate':candidate_label,'confidence':conf,'iou':iou,'scale_threshold_px':thr,'small_object_policy':policy,'matching':'greedy_iou','tp':TP,'fp':FP,'fn':FN,'precision':prec,'recall':rec,'f1':f1,'count_mae':count_sum/nimg if nimg else np.nan,'images':nimg})
  print(source,'done',flush=True)
 pd.DataFrame(allrows).to_parquet(out/'metrics_long.parquet',index=False); pd.DataFrame(allrows).to_csv(out/'metrics_long.csv',index=False)
 pd.DataFrame(allrows).groupby(['source','candidate','confidence','iou','scale_threshold_px','small_object_policy'],as_index=False).first().to_csv(out/'count_metrics.csv',index=False)
 (out/'traffic_activity_metrics.csv').write_text('status,details\nblocked,traffic tiers require frozen train-derived thresholds and per-image raw metric export\n')
 print(json.dumps({'rows':len(allrows),'out':str(out),'source':args.source},indent=2))
if __name__=='__main__': main()
