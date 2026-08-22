#!/usr/bin/env python
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
from typing import Any
import pandas as pd, yaml

PROJECT_ROOT=Path(__file__).resolve().parents[1]
RUN_KEYS={('output_dir',),('dr1_validation','input_h5'),('dr1_validation','variant_id'),('dr1_validation','variant_name'),('dr1_validation','pybdsf_reused')}
RUN_OR_EMPTY_A0_KEYS=RUN_KEYS|{('ablation',)}
CODEPATHS={
 'use_ridge_continuity':('lotss_association/association.py','compute_pair_association_features/run_component_association','ablation_enabled(config, "use_ridge_continuity")'),
 'use_weak_edge_anti_chaining':('lotss_association/association.py','cluster_association_groups','ablation_enabled(config, "use_weak_edge_anti_chaining")'),
 'use_artifact_penalties_layer1':('lotss_association/association.py','compute_pair_association_features/build_association_graph','ablation_enabled(config, "use_artifact_penalties_layer1")'),
 'use_artifact_penalties_layer2':('lotss_association/parent_links.py','run_parent_links/_make_pair_edges','ablation_enabled(config, "use_artifact_penalties_layer2")'),
 'use_midpoint_host_support':('lotss_association/parent_links.py','run_parent_links/classify_parent_acceptance','ablation_enabled(config, "use_midpoint_host_support")'),
 'use_lobe_peak_host_contradiction':('lotss_association/parent_links.py','run_parent_links','ablation_enabled(config, "use_lobe_peak_host_contradiction")'),
 'use_pa_alignment':('lotss_association/association.py','compute_pair_association_features','ablation_enabled(config, "use_pa_alignment")'),
 'use_ellipse_overlap':('lotss_association/association.py','compute_pair_association_features','ablation_enabled(config, "use_ellipse_overlap")'),
 'use_multithreshold_contour':('lotss_association/association.py','compute_pair_association_features/build_association_graph','ablation_enabled(config, "use_multithreshold_contour")'),
 'use_stage2_relative_scale_constraints':('lotss_association/parent_links.py','_make_pair_edges/run_parent_links','ablation_enabled(config, "use_stage2_relative_scale_constraints")'),
 'use_stage2_endpoint_filtering':('lotss_association/parent_links.py','_make_pair_edges/run_parent_links','ablation_enabled(config, "use_stage2_endpoint_filtering")'),
}

def sha_file(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def git_commit():
    try: return subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip()
    except Exception: return 'unknown'
def strip_keys(obj:Any, drop:set[tuple[str,...]], path=()):
    if path in drop: return None
    if isinstance(obj,dict):
        out={}
        for k,v in obj.items():
            vv=strip_keys(v, drop, path+(k,))
            if vv is not None and vv != {}:
                out[k]=vv
        return out
    return obj
def flatten(d,prefix=()):
    if isinstance(d,dict):
        out={}
        for k,v in d.items(): out.update(flatten(v,prefix+(k,)))
        return out
    return {'.'.join(prefix):d}
def diff(a,b,drop):
    fa,fb=flatten(strip_keys(a,drop)),flatten(strip_keys(b,drop)); keys=sorted(set(fa)|set(fb)); return {k:(fa.get(k),fb.get(k)) for k in keys if fa.get(k)!=fb.get(k)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-root',default='outputs/dr1_validation'); ap.add_argument('--official-config',default='configs/real_lotss_conservative.yaml'); args=ap.parse_args()
    root=Path(args.output_root); cfgdir=root/'configs'; cfgdir.mkdir(parents=True,exist_ok=True)
    official=Path(args.official_config); official_cfg=yaml.safe_load(official.read_text()) or {}
    manifest=pd.read_csv(cfgdir/'ablation_manifest.csv')
    a0_row=manifest.iloc[0]; a0_path=Path(a0_row.config_path); a0_cfg=yaml.safe_load(a0_path.read_text()) or {}
    a0_diff=diff(official_cfg,a0_cfg,RUN_OR_EMPTY_A0_KEYS)
    audit={'source_config_path':str(official),'source_config_sha256':sha_file(official),'generated_config_path':str(a0_path),'generated_config_sha256':sha_file(a0_path),'git_commit':git_commit(),'config_equal_to_official':len(a0_diff)==0,'unexpected_differences':list(a0_diff.keys())}
    (cfgdir/'A0_full_config_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    diff_rows=[]; code_rows=[]
    for _,row in manifest.iterrows():
        cfg=yaml.safe_load(Path(row.config_path).read_text()) or {}
        changes=json.loads(row.changed_parameters) if isinstance(row.changed_parameters,str) and row.changed_parameters else []
        d=diff(a0_cfg,cfg,RUN_KEYS)
        sci_changed=sorted(d.keys())
        expected=sorted('ablation.'+k for k in changes)
        diff_rows.append({'variant_id':row.variant_id,'variant_name':row.variant_name,'changed_scientific_keys':json.dumps(sci_changed),'expected_changed_keys':json.dumps(expected),'unexpected_changed_keys':json.dumps(sorted(set(sci_changed)-set(expected))),'missing_expected_changes':json.dumps(sorted(set(expected)-set(sci_changed))),'valid':set(sci_changed)==set(expected)})
        ablation_values=json.loads(row.ablation_values) if isinstance(row.ablation_values,str) and row.ablation_values else {}
        for key,val in ablation_values.items():
            file,func,symbol=CODEPATHS.get(key,('','',''))
            text=(PROJECT_ROOT/file).read_text(errors='ignore') if file else ''
            verified=bool(symbol and symbol in text)
            code_rows.append({'variant_id':row.variant_id,'variant_name':row.variant_name,'disabled_module':row.disabled_module,'config_key':key,'a0_value':True,'ablation_value':val,'config_loader_file':'lotss_association/ablation_config.py','runner_file':'scripts/run_dr1_ablation_shard.py -> scripts/run_lotss_dr3_full.py::_process_field','core_code_file':file,'core_function':func,'code_line_or_symbol':symbol,'runtime_probe':'run_metadata.active_features/disabled_features','verified':verified,'notes':''})
    pd.DataFrame(diff_rows).to_csv(cfgdir/'ablation_config_diff.csv',index=False)
    (cfgdir/'ablation_config_diff.json').write_text(json.dumps(diff_rows,indent=2,sort_keys=True)+'\n')
    pd.DataFrame(code_rows).to_csv(cfgdir/'ablation_codepath_audit.csv',index=False)
    print(json.dumps({'a0_equal':len(a0_diff)==0,'n_variants':len(manifest),'diff_valid':all(r['valid'] for r in diff_rows),'codepaths_verified':all(r['verified'] for r in code_rows)},indent=2))
if __name__=='__main__': main()
