#!/usr/bin/env python
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
ROOT=Path('outputs/dr1_validation')
manifest=pd.read_csv(ROOT/'configs/ablation_manifest.csv')
rows=[]; triggers=[]
for _,m in manifest.iterrows():
    run_name=m.get('run_name') or m.variant_name
    sd=ROOT/'runs'/run_name/'shards'/'shard_0000'
    meta=json.loads((sd/'run_metadata.json').read_text())
    fs=pd.read_csv(sd/'field_status.csv') if (sd/'field_status.csv').exists() else pd.DataFrame()
    assoc=sd/'association_outputs'
    counts={}
    for stem in ['local_components','local_groups','merged_components','parent_edges_debug','parent_candidates','diagnostics']:
        total=0
        frames=[]
        for p in assoc.glob(f'*_{stem}.csv'):
            df=pd.read_csv(p); frames.append(df); total+=len(df)
        counts[stem]=total
        if stem=='parent_edges_debug': pedges=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
        if stem=='local_groups': groups=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    abvals=json.loads(m.ablation_values) if isinstance(m.ablation_values,str) and m.ablation_values else {}
    rows.append({
        'variant_id':m.variant_id,'variant_name':m.variant_name,'run_name':run_name,'shard_id':'shard_0000',
        'real_association_run':meta.get('real_association_run'),'association_outputs_verified':meta.get('association_outputs_verified'),
        'pybdsf_reused':meta.get('pybdsf_reused'),'pybdsf_rerun':meta.get('pybdsf_rerun'),
        'disabled_features':';'.join(meta.get('disabled_features',[])),
        'input_manifest_hash':meta.get('input_manifest_hash'),'shard_manifest_hash':meta.get('shard_manifest_hash'),
        'scientific_config_hash':meta.get('scientific_config_hash'),'runtime_seconds':meta.get('runtime_seconds'),
        'input_cutouts':int(pd.to_numeric(fs.get('n_cutouts',pd.Series(dtype=float)),errors='coerce').fillna(0).sum()) if not fs.empty else 0,
        'input_components':counts['local_components'],'local_components':counts['local_components'],'local_groups':counts['local_groups'],
        'candidate_pairs':counts['parent_edges_debug'],'parent_edges_debug':counts['parent_edges_debug'],'parent_candidates':counts['parent_candidates'],
        'final_associations':counts['merged_components'],'failed_cutouts':int((fs.get('status',pd.Series(dtype=str)).astype(str)!='done').sum()) if not fs.empty else 0,
        'maximum_group_size':int(pd.to_numeric(groups.get('n_gaussians',pd.Series(dtype=float)),errors='coerce').fillna(0).max()) if not groups.empty and 'n_gaussians' in groups else 0,
        'multi_component_groups':int((pd.to_numeric(groups.get('n_gaussians',pd.Series(dtype=float)),errors='coerce').fillna(0)>1).sum()) if not groups.empty else 0,
        'singletons':int((pd.to_numeric(groups.get('n_gaussians',pd.Series(dtype=float)),errors='coerce').fillna(0)<=1).sum()) if not groups.empty else 0,
    })
    triggers.append({
        'variant_id':m.variant_id,'variant_name':m.variant_name,'target_keys':';'.join(abvals.keys()),
        'ridge_positive_pairs':int((pd.to_numeric(pedges.get('ridge_continuity_score_pair',pd.Series(dtype=float)),errors='coerce').fillna(0)>0).sum()) if 'pedges' in locals() and not pedges.empty else 0,
        'artifact_penalty_applications':int((pd.to_numeric(pedges.get('artifact_environment_score_pair',pd.Series(dtype=float)),errors='coerce').fillna(0)>0).sum()) if 'pedges' in locals() and not pedges.empty else 0,
        'parent_edges_debug':counts['parent_edges_debug'],
        'accepted_parent_links':counts['parent_candidates'],
        'host_support_applications':int((pedges.get('host_evidence',pd.Series(dtype=str)).astype(str)=='supports_double_lobe').sum()) if 'pedges' in locals() and not pedges.empty else 0,
        'lobe_peak_contradiction_applications':int((pedges.get('rejection_reason',pd.Series(dtype=str)).astype(str)=='lobe_peak_host_contradiction').sum()) if 'pedges' in locals() and not pedges.empty else 0,
    })
out=ROOT/'validation'; out.mkdir(parents=True,exist_ok=True)
smoke=pd.DataFrame(rows); smoke.to_csv(out/'all_variants_smoke_summary.csv',index=False)
pd.DataFrame(triggers).to_csv(out/'ablation_trigger_counts.csv',index=False)
report={'n_variants':len(smoke),'all_real_runner':bool(smoke.real_association_run.eq(True).all()),'all_outputs_verified':bool(smoke.association_outputs_verified.eq(True).all()),'all_pybdsf_reused':bool(smoke.pybdsf_reused.eq(True).all()),'all_pybdsf_not_rerun':bool(smoke.pybdsf_rerun.eq(False).all()),'same_input_manifest_hash':int(smoke.input_manifest_hash.nunique())==1,'same_shard_manifest_hash':int(smoke.shard_manifest_hash.nunique())==1,'same_input_cutouts':int(smoke.input_cutouts.nunique())==1,'same_input_components':int(smoke.input_components.nunique())==1,'rows':smoke.to_dict(orient='records')}
(out/'all_variants_smoke_report.json').write_text(json.dumps(report,indent=2,sort_keys=True,default=str)+'\n')
print(json.dumps(report,indent=2,sort_keys=True,default=str))
