#!/usr/bin/env python
"""Prepare DR1 production parent-linking ablation configs, manifest, and shard manifest."""
from __future__ import annotations
import argparse, copy, hashlib, json, subprocess
from pathlib import Path
import pandas as pd, yaml

PROJECT_ROOT=Path(__file__).resolve().parents[1]
DEFAULT_BASE=PROJECT_ROOT/'configs/real_lotss_conservative.yaml'
DEFAULT_VARIANTS=PROJECT_ROOT/'configs/dr1_validation/ablation_variants.yaml'
DEFAULT_ROOT=PROJECT_ROOT/'outputs/dr1_validation'

def git_commit():
    try: return subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip()
    except Exception: return 'unknown'

def sha(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()[:16]

def set_deep(d,path,val):
    cur=d
    parts=path.split('.')
    for p in parts[:-1]: cur=cur.setdefault(p,{})
    cur[parts[-1]]=val

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-config',type=Path,default=DEFAULT_BASE)
    ap.add_argument('--variants',type=Path,default=DEFAULT_VARIANTS)
    ap.add_argument('--output-root',type=Path,default=DEFAULT_ROOT)
    ap.add_argument('--selected-cutouts',type=Path,default=DEFAULT_ROOT/'selection/selected_cutouts.csv')
    ap.add_argument('--h5',type=Path,default=DEFAULT_ROOT/'data/dr1_validation.h5')
    ap.add_argument('--shard-size',type=int,default=16)
    args=ap.parse_args()
    base=yaml.safe_load(args.base_config.read_text())
    variants_doc=yaml.safe_load(args.variants.read_text())
    variants=variants_doc['variants']
    if isinstance(variants,dict):
        iterable=[(k,v) for k,v in variants.items()]
    else:
        iterable=[(f'A{i}',v) for i,v in enumerate(variants)]
    cfg_dir=args.output_root/'configs'; cfg_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for vid,v in iterable:
        cfg=copy.deepcopy(base)
        cfg.setdefault('dr1_validation',{})['input_h5']=str(args.h5)
        cfg['dr1_validation']['variant_id']=vid
        cfg['dr1_validation']['variant_name']=v['name']
        cfg['dr1_validation']['pybdsf_reused']=True
        cfg.setdefault('ablation',{}).update(v.get('ablation',{}) or {})
        run_name='A0_full' if vid == 'A0' else f'{vid}_{v["name"]}'
        outdir=args.output_root/'runs'/run_name
        cfg['output_dir']=str(outdir)
        path=cfg_dir/f'{run_name}.yaml'
        if vid == 'A0':
            (cfg_dir/'A0_full.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        changes=v.get('ablation',{}) or {}
        rows.append({'variant_id':vid,'variant_name':v['name'],'parent_config':str(args.base_config),'config_path':str(path),'disabled_module':v['name'].replace('no_','') if v['name']!='full_method' else '', 'changed_parameters':json.dumps(sorted(changes.keys())),'original_values':json.dumps({k:base.get('ablation',{}).get(k) for k in changes},sort_keys=True),'ablation_values':json.dumps(changes,sort_keys=True),'affected_stage':'layer1_or_layer2','run_name':run_name,'output_directory':str(outdir),'random_seed':base.get('random_seed',42),'config_hash':sha(cfg),'git_commit':git_commit()})
    pd.DataFrame(rows).to_csv(cfg_dir/'ablation_manifest.csv',index=False)
    selected=pd.read_csv(args.selected_cutouts,dtype=str).fillna('') if args.selected_cutouts.exists() else pd.DataFrame()
    shard_dir=args.output_root/'shards'; shard_dir.mkdir(parents=True,exist_ok=True)
    srows=[]
    if selected.empty:
        srows.append({'shard_id':'shard_0000','cutout_id':'','tile_id':'','h5_group':'','spatial_partition':'all','overlap_group':'all','estimated_component_count':0,'estimated_memory':'unknown'})
    else:
        for i,row in selected.iterrows():
            sid=f'shard_{i//max(1,args.shard_size):04d}'
            srows.append({'shard_id':sid,'cutout_id':row.get('cutout_id',''),'tile_id':row.get('tile_id',''),'h5_group':row.get('source_h5_group','/'),'spatial_partition':sid,'overlap_group':sid,'estimated_component_count':row.get('matched_dr3_pybdsf_count',0),'estimated_memory':'unknown'})
    shards=pd.DataFrame(srows)
    shards.to_csv(shard_dir/'shard_manifest.csv',index=False)
    summary={'n_shards':int(shards.shard_id.nunique()),'n_cutout_rows':int(len(shards)),'shard_size':int(args.shard_size),'same_manifest_for_all_variants':True}
    (shard_dir/'shard_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'variants':len(rows),**summary},indent=2))
if __name__=='__main__': main()
