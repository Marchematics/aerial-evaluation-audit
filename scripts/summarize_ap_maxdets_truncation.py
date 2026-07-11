"""Audit pre-cap retained prediction counts for AP maxDets sensitivity."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_cached_vehicle_grid import CANDIDATES,gt_map,iou_matrix

PAIRS={'visdrone':('visdrone_sahi640','visdrone_baseline640'),'uavdt':('uavdt_tiling','uavdt_baseline640')}
POL={'visdrone':['include_all','exclude_without_ignore_protection','exclude_with_ignore_protection'],
     'uavdt':['include_all','exclude_without_ignore_protection']}

def main():
    rows=[]
    for source,candidates in PAIRS.items():
        gt=gt_map(source)
        for candidate in candidates:
            _,path,classes=CANDIDATES[candidate];raw_all=pd.read_parquet(path);names=sorted(set(raw_all.img_name.unique())&set(gt));raw=raw_all[raw_all.cls.isin(classes)];groups={n:g for n,g in raw.groupby('img_name',sort=False)}
            for policy in POL[source]:
                counts=[]
                for n in names:
                    pp=[(x.x1,x.y1,x.x2,x.y2,float(x.score)) for x in groups.get(n,pd.DataFrame()).itertuples()]
                    if policy=='exclude_with_ignore_protection' and gt[n]['ignore']:
                        pp=[p for p in pp if float(iou_matrix([p],gt[n]['ignore']).max())<.5]
                    counts.append(len(pp))
                x=np.asarray(counts)
                for cap in [100,300,2000]:
                    rows.append({'source':source,'candidate':candidate,'policy':policy,'max_dets':cap,'images':len(x),'images_truncated':int((x>cap).sum()),'detections_truncated':int(np.maximum(x-cap,0).sum()),'max_pre_cap_detections':int(x.max())})
    out=ROOT/'outputs/statistics';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'ap_maxdets_truncation_audit.csv',index=False)
if __name__=='__main__':main()
