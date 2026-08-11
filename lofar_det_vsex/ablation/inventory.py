"""Module inventory for association ablations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from lofar_det_vsex.utils import json_dumps_safe


def module_inventory_records() -> list[dict[str, Any]]:
    """Return the current implementation inventory for requested modules."""

    return [
        {
            "module_name": "multi-threshold contour connectivity",
            "python_file": "lofar_det_vsex/segmentation.py; lofar_det_vsex/association.py",
            "function_or_class": "segment_snr_map; _shared_label; compute_pair_association_features; compute_association_score",
            "input_fields": "segmentation.labels_by_threshold, segmentation.thresholds, component x/y, snr_thresholds",
            "output_fields": "connected_at_3sigma, connected_at_2p5sigma, connected_at_2sigma, same_label_*",
            "thresholds": "3.0, 2.5, 2.0 sigma; weights conn_3sigma/conn_2p5sigma/conn_2sigma",
            "role": "score/bonus and diagnostic; only_2sigma can also trigger a penalty/cap",
            "decision_stage": "final pair scoring, not candidate generation",
            "exists": True,
        },
        {
            "module_name": "ridge continuity / ridge score",
            "python_file": "lofar_det_vsex/association.py",
            "function_or_class": "compute_ridge_continuity; compute_association_score; assign_association_quality; classify_association_group",
            "input_fields": "segmentation.snr_map, component x/y, association.threshold_weak",
            "output_fields": "ridge_mean_snr, ridge_gap_fraction, ridge_continuity_score, ridge_gradient_smoothness",
            "thresholds": "threshold_weak default 2.0; support cutoffs 0.45/0.55 in group typing/quality",
            "role": "score/bonus and quality/type support",
            "decision_stage": "final pair scoring and group labels, not candidate generation",
            "exists": True,
        },
        {
            "module_name": "ellipse overlap",
            "python_file": "lofar_det_vsex/association.py",
            "function_or_class": "compute_pair_association_features; compute_association_score",
            "input_fields": "_dc_maj/_dc_min or _maj/_min, component separation, beam",
            "output_fields": "ellipse_overlap_score, ellipse_gap_beam",
            "thresholds": "score is clipped 0..1 from gap in beam units; independent-support cutoff 0.75",
            "role": "score/bonus",
            "decision_stage": "final pair scoring, not candidate generation",
            "exists": True,
        },
        {
            "module_name": "position-angle alignment",
            "python_file": "lofar_det_vsex/association.py",
            "function_or_class": "_angle_delta_deg; _alignment_score; compute_pair_association_features; compute_association_score",
            "input_fields": "_dc_pa/_pa and center-to-center line angle",
            "output_fields": "pa_alignment_score, line_to_pa_alignment_score, flow_alignment_score",
            "thresholds": "alignment scale 45 deg; weight pa_alignment 0.8, flow_alignment 0.6 by default",
            "role": "score/bonus",
            "decision_stage": "final pair scoring, not candidate generation",
            "exists": True,
        },
        {
            "module_name": "strong-edge classification",
            "python_file": "lofar_det_vsex/association.py",
            "function_or_class": "build_association_graph",
            "input_fields": "association_score, distance_beam, association.threshold_strong",
            "output_fields": "edge_type, association_decision, rejection_reason",
            "thresholds": "threshold_strong default 3.0; max_pair_distance_beam default 15.0",
            "role": "classification rule",
            "decision_stage": "final decision after candidate feature computation",
            "exists": True,
        },
        {
            "module_name": "weak-edge classification",
            "python_file": "lofar_det_vsex/association.py",
            "function_or_class": "build_association_graph",
            "input_fields": "association_score, distance_beam, association.threshold_weak",
            "output_fields": "edge_type=weak, association_decision initially False, rejection_reason=pending_weak_attachment",
            "thresholds": "threshold_weak default 2.0; max_pair_distance_beam default 15.0",
            "role": "classification rule",
            "decision_stage": "final decision after candidate feature computation",
            "exists": True,
        },
        {
            "module_name": "weak-edge singleton attachment",
            "python_file": "lofar_det_vsex/association.py",
            "function_or_class": "cluster_association_groups",
            "input_fields": "edge_type, association_score, strong connected components",
            "output_fields": "association_decision=True for accepted weak attachments; rejection_reason",
            "thresholds": "weak edges sorted by association_score; strong core size >=2, singleton size ==1",
            "role": "clustering rule",
            "decision_stage": "clustering only",
            "exists": True,
        },
        {
            "module_name": "weak-edge anti-chaining",
            "python_file": "lofar_det_vsex/association.py",
            "function_or_class": "cluster_association_groups",
            "input_fields": "strong cores, weak edges, original group sizes",
            "output_fields": "rejection_reason=weak_edge_would_merge_core_groups/no_core_group/would_form_chain",
            "thresholds": "structural rule; no numeric score threshold beyond weak classification",
            "role": "clustering rule",
            "decision_stage": "clustering only; does not affect edge classification",
            "exists": True,
        },
        {
            "module_name": "artifact flags / artifact penalties / artifact veto",
            "python_file": "lofar_det_vsex/association.py; lofar_det_vsex/parent_links_endpoint_guarded.py",
            "function_or_class": "compute_artifact_penalties; assign_association_quality; classify_association_group; _artifact_environment; _source_morph; _classify_pair",
            "input_fields": "2sigma labels, bridge/ridge/PA support, line SNR, flux ratio, group environment",
            "output_fields": "deep_valley_penalty, only_2sigma_penalty, negative_bowl_penalty, sidelobe_risk_penalty, too_far_penalty, large_mask_swallow_penalty, artifact_risk_flags, is_artifact_risk, noise_artifact_veto",
            "thresholds": "Layer-1 penalty weights from weights_association; Layer-2 artifact_veto_score 1.2, suspicious_score 0.8 by default",
            "role": "penalty and veto",
            "decision_stage": "Layer-1 final score/quality/type and Layer-2 endpoint/pair quality",
            "exists": True,
        },
        {
            "module_name": "Layer-2 endpoint filtering",
            "python_file": "lofar_det_vsex/parent_links_endpoint_guarded.py",
            "function_or_class": "build_source_morph_table; _normal_endpoint_allowed; _rescue_endpoint_allowed; _classify_pair",
            "input_fields": "Layer-1 groups, 3sigma area, LAS, axis_ratio, quality/type, compact/artifact flags",
            "output_fields": "source_morph_class, is_lobe_candidate, is_parent_endpoint_allowed, lobe_like_reject_reason, double_lobe_geometry_pass",
            "thresholds": "peak_snr, area_3sigma, LAS, axis ratio, hard compact and artifact veto thresholds",
            "role": "veto/filter",
            "decision_stage": "Layer-2 candidate generation and final decision",
            "exists": True,
        },
        {
            "module_name": "pairwise lobe geometry",
            "python_file": "lofar_det_vsex/parent_seed.py; lofar_det_vsex/parent_links_endpoint_guarded.py",
            "function_or_class": "_compute_pair; _symmetry_scores; _classify_pair",
            "input_fields": "local group bboxes, centroids, PA, flux, LAS/mask sizes",
            "output_fields": "axis_alignment_score, facing_score, flux_ratio, size_ratio, symmetry_score, lobe_pair_score",
            "thresholds": "max_box_gap_beam 12, max_center_distance_beam 40, min_axis_alignment 0.7, min_facing_score 0.6, min_symmetry_score 0.6 by default",
            "role": "score and veto",
            "decision_stage": "Layer-2 final decision after pair generation",
            "exists": True,
        },
        {
            "module_name": "midpoint host support",
            "python_file": "lofar_det_vsex/host_support.py; lofar_det_vsex/parent_links_endpoint_guarded.py",
            "function_or_class": "_midpoint_ra_dec; _query_hosts_for_pair; score_host_candidates; run_parent_links_endpoint_guarded",
            "input_fields": "parent pair midpoint, WISE host query results, host geometry and WISE photometry",
            "output_fields": "best_host_score, host_quality, host_evidence=supports_double_lobe, parent_score_final, parent_candidate_quality",
            "thresholds": "host_quality medium/high thresholds from host_support; host_score_weight default 1.0",
            "role": "score/bonus and quality upgrade",
            "decision_stage": "Layer-2 final decision; host query does not change radio candidate pairs",
            "exists": True,
        },
        {
            "module_name": "lobe-peak host contradiction",
            "python_file": "lofar_det_vsex/parent_links_endpoint_guarded.py",
            "function_or_class": "_peak_coord_for_group; _query_position_hosts; _score_lobe_peak_hosts; run_parent_links_endpoint_guarded",
            "input_fields": "lobe peak coordinates, WISE host query results near each lobe peak",
            "output_fields": "lobe_peak_host_found, lobe1/2_peak_host_score, host_evidence=contradicts_double_lobe/likely_independent_sources, rejection_reason",
            "thresholds": "lobe peak radius min 5 arcsec, max 10 arcsec; score high >=2.0, medium >=1.2",
            "role": "veto/quality downgrade",
            "decision_stage": "Layer-2 final decision; host query does not change radio candidate pairs",
            "exists": True,
        },
        {
            "module_name": "parent quality/confidence assignment",
            "python_file": "lofar_det_vsex/parent_links_endpoint_guarded.py",
            "function_or_class": "run_parent_links_endpoint_guarded; _apply_limits",
            "input_fields": "geometry pass, symmetry_score, midpoint host quality, lobe-peak host flags, artifact score",
            "output_fields": "parent_candidate_quality, host_evidence, needs_visual_check, parent_score_final",
            "thresholds": "high requires host_quality high and symmetry >=0.70 in current endpoint-guarded; needs_host_check symmetry >=0.68 when no host support",
            "role": "quality/confidence assignment",
            "decision_stage": "Layer-2 final decision",
            "exists": True,
        },
    ]


def write_module_inventory(output_dir: str | Path) -> tuple[Path, Path]:
    """Write module_inventory.md and module_inventory.json."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = module_inventory_records()
    json_path = out / "module_inventory.json"
    md_path = out / "module_inventory.md"
    json_path.write_text(json_dumps_safe({"modules": records}), encoding="utf-8")

    lines = [
        "# Ablation Module Inventory",
        "",
        "Current implementation findings:",
        "",
        "- Ridge score participates in strong/weak edge scoring and group quality/type support.",
        "- Ridge is also present as diagnostic fields and a Layer-2 pair-support diagnostic.",
        "- Artifact handling acts in Layer 1 scoring/quality/type and Layer 2 endpoint/pair quality.",
        "- Midpoint host support and lobe-peak contradiction use separate host-query roles and scores.",
        "- In the current endpoint-guarded path, high parent quality directly depends on midpoint host quality.",
        "- Weak-edge anti-chaining is a clustering rule; it does not change pair feature computation or edge classification.",
        "",
        "| Module | File | Function/Class | Role | Stage | Exists |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                str(record[key]).replace("|", "\\|")
                for key in ["module_name", "python_file", "function_or_class", "role", "decision_stage", "exists"]
            )
            + " |"
        )
    lines.extend(["", "## Detailed Records", ""])
    for record in records:
        lines.extend(
            [
                f"### {record['module_name']}",
                "",
                f"- Python file: `{record['python_file']}`",
                f"- Function/class: `{record['function_or_class']}`",
                f"- Input fields: {record['input_fields']}",
                f"- Output fields: {record['output_fields']}",
                f"- Thresholds: {record['thresholds']}",
                f"- Role: {record['role']}",
                f"- Stage: {record['decision_stage']}",
                f"- Exists: `{record['exists']}`",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(records).to_csv(out / "module_inventory.csv", index=False)
    return md_path, json_path
