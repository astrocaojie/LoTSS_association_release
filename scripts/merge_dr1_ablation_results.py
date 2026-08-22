#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

STATUS_COLUMNS=['variant','shard_id','status','real_association_run','path']

def _read_meta(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root',default='outputs/dr1_validation'); args=ap.parse_args()
    root=Path(args.output_root); rows=[]; outputs=[]; rejected=[]
    for success in root.glob('runs/*/shards/*/_SUCCESS'):
        sd=success.parent; variant=sd.parents[1].name; shard=sd.name; meta=_read_meta(sd/'run_metadata.json')
        real=bool(meta.get("real_association_run", False)) and bool(meta.get("association_outputs_verified", False))
        status='success' if real else 'rejected_non_scientific'
        rows.append({'variant':variant,'shard_id':shard,'status':status,'real_association_run':real,'path':str(sd)})
        if not real:
            rejected.append(str(sd)); continue
        f=sd/'field_status.csv'
        if f.exists():
            df=pd.read_csv(f); df['variant']=variant; df['shard_id']=shard; outputs.append(df)
    for failed in root.glob('runs/*/shards/*/_FAILED'):
        sd=failed.parent; rows.append({'variant':sd.parents[1].name,'shard_id':sd.name,'status':'failed','real_association_run':False,'path':str(sd)})
    out=root/'merged'; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows, columns=STATUS_COLUMNS).to_csv(out/'ablation_shard_status.csv',index=False)
    merged=pd.concat(outputs,ignore_index=True) if outputs else pd.DataFrame()
    merged.to_csv(out/'ablation_field_status.csv',index=False)
    (out/'merge_summary.json').write_text(json.dumps({'n_status_rows':len(rows),'n_real_output_rows':len(merged),'rejected_non_scientific':rejected},indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main()
