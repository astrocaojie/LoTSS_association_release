#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def _count_partial_rows(shard_dir: Path) -> dict[str, int]:
    out = {}
    assoc = shard_dir / 'association_outputs'
    for stem in ['local_components','local_groups','merged_components','parent_edges_debug','parent_candidates','diagnostics']:
        total = 0
        for path in assoc.glob(f'*_{stem}.csv'):
            try:
                total += len(pd.read_csv(path))
            except Exception:
                pass
        out[f'n_{stem}'] = int(total)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root',default='outputs/dr1_validation'); args=ap.parse_args()
    root=Path(args.output_root); out=root/'evaluation'; out.mkdir(parents=True,exist_ok=True)
    status_path=root/'merged/ablation_shard_status.csv'
    field_status_path=root/'merged/ablation_field_status.csv'
    mapping=root/'selection/dr1_truth_gaussian_mapping.parquet'
    summary={'mapping_exists':mapping.exists(),'shard_status_exists':status_path.exists(),'field_status_exists':field_status_path.exists()}
    rows=[]
    if status_path.exists():
        
        try:
            shard_status=pd.read_csv(status_path)
        except pd.errors.EmptyDataError:
            shard_status=pd.DataFrame(columns=['variant','shard_id','status','real_association_run','path'])
        summary['shard_status_counts']=shard_status['status'].value_counts(dropna=False).to_dict() if 'status' in shard_status else {}
        for _, r in shard_status.iterrows():
            if str(r.get('status'))!='success' or str(r.get('real_association_run')).lower()!='true':
                continue
            sd=Path(str(r.get('path')))
            rows.append({'variant':r.get('variant'),'shard_id':r.get('shard_id'), **_count_partial_rows(sd)})
    partial_summary=pd.DataFrame(rows)
    partial_summary.to_csv(out/'ablation_real_output_row_counts.csv',index=False)
    if field_status_path.exists():
        
        try:
            fs=pd.read_csv(field_status_path)
        except pd.errors.EmptyDataError:
            fs=pd.DataFrame()
        summary['n_real_field_status_rows']=int(len(fs))
        if 'status' in fs:
            summary['field_status_counts']=fs['status'].value_counts(dropna=False).to_dict()
        for col in ['n_cutouts','n_local_groups','n_parent_candidates','n_merged_rows','n_host_candidates']:
            if col in fs:
                summary[f'total_{col}']=int(pd.to_numeric(fs[col], errors='coerce').fillna(0).sum())
    if mapping.exists():
        m=pd.read_parquet(mapping)
        summary['truth_gaussian_match_status_counts']=m.match_status.value_counts(dropna=False).to_dict() if 'match_status' in m else {}
    summary['evaluation_status']='real_outputs_summarized' if rows else 'pending_real_association_outputs'
    pd.DataFrame([summary]).to_csv(out/'ablation_evaluation_summary.csv',index=False)
    (out/'evaluation_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,default=str)+'\n')
if __name__=='__main__': main()
