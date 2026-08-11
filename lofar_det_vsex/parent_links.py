"""Physics-aware parent-link candidates for large radio sources.

The local Gaussian association catalog remains the primary output. This module
adds a supplementary parent layer only after source morphology, double-lobe
geometry, artifact veto, and host support or contradiction checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from .association import compute_beam_size_arcsec
from .ablation_config import ablation_enabled
from .host_query import HOST_QUERY_LOG_COLUMNS, HostQueryClient
from .parent_seed import (
    _association_quality,
    _association_type,
    _bbox_tuple,
    _candidate_search_pairs,
    _component_ids,
    _compute_pair,
    _mask_area_beam,
    _peak_snr_for_group,
    _robust_bbox_for_group,
    build_parent_seed_table,
    parent_seed_config,
)
from .host_support import (
    _host_search_radius,
    _midpoint_ra_dec,
    _quality_rank,
    _query_hosts_for_pair,
    score_host_candidates,
    host_support_config,
)
from .utils import json_dumps_safe, safe_float


SOURCE_MORPH_TABLE_COLUMNS = [
    "cutout_id",
    "group_id",
    "association_group_id",
    "n_gaussians",
    "LAS_beam",
    "major_axis_beam",
    "minor_axis_beam",
    "area_3sigma_beam",
    "mask_area_beam",
    "axis_ratio",
    "peak_snr",
    "association_quality",
    "association_type",
    "source_morph_class",
    "is_point_like",
    "hard_point_source_veto",
    "point_source_veto_reason",
    "is_beam_like_single_gaussian",
    "hard_compact_veto",
    "compact_veto_reason",
    "noise_artifact_veto",
    "noise_artifact_reason",
    "isolated_compact_veto",
    "isolated_compact_reason",
    "endpoint_veto_final",
    "endpoint_veto_reason",
    "is_lobe_candidate",
    "near_extended_lobe_candidate",
    "near_lobe_rescue_reason",
    "is_parent_endpoint_allowed",
    "lobe_like_reject_reason",
    "same_3sigma_region_as_any_extended_neighbor",
    "same_2p5sigma_region_as_any_extended_neighbor",
    "bridge_snr_support_to_any_extended_neighbor",
    "is_artifact_risk",
    "morph_reject_reason",
    "bright_source_distance_beam",
    "radial_to_bright_source_score",
    "local_fragment_density",
    "artifact_environment_score",
    "artifact_veto_reason",
    "is_parent_seed",
    "robust_bbox",
    "robust_bbox_source",
    "missing_fields",
]

# 第二层字段保留几何双瓣证据、host 支持/反证、冲突裁决和可视检查标记；
# 这样人工复核时可以追溯 parent candidate 为什么被接受或拒绝。
PARENT_CANDIDATE_COLUMNS = [
    "cutout_id",
    "parent_candidate_id",
    "parent_candidate_type",
    "local_group_id_1",
    "local_group_id_2",
    "box_gap_beam_robust",
    "center_distance_arcsec",
    "center_distance_beam",
    "axis_alignment_score",
    "facing_score",
    "flux_ratio",
    "size_ratio",
    "symmetry_score",
    "lobe_pair_score",
    "parent_score_geometry",
    "best_host_score",
    "parent_score_final",
    "host_evidence",
    "host_support",
    "host_ambiguous",
    "host_at_lobe_peak",
    "host_off_axis",
    "host_far_from_midpoint",
    "multiple_host_candidates",
    "no_host_detected",
    "independent_parent_evidence",
    "parent_acceptance_class",
    "parent_acceptance_reason",
    "conflict_resolution_status",
    "conflict_rejection_reason",
    "parent_candidate_quality",
    "host_status",
    "best_host_catalog",
    "best_host_id",
    "best_host_ra",
    "best_host_dec",
    "best_host_sep_midpoint_arcsec",
    "best_host_perp_offset_beam",
    "best_host_fractional_position",
    "best_host_W1",
    "best_host_W2",
    "best_host_W1_W2",
    "host_quality",
    "lobe_peak_host_found",
    "lobe1_peak_host_found",
    "lobe2_peak_host_found",
    "lobe1_peak_host_score",
    "lobe2_peak_host_score",
    "artifact_environment_score_pair",
    "near_boundary_rescue_applied",
    "near_boundary_rescue_reason",
    "gap_to_mean_box_ratio",
    "gap_to_min_box_ratio",
    "suspicious_reason",
    "parent_bbox_xmin",
    "parent_bbox_xmax",
    "parent_bbox_ymin",
    "parent_bbox_ymax",
    "parent_LAS_beam",
    "parent_LAS_arcsec",
    "parent_union_area_beam",
    "rejection_reason",
    "needs_visual_check",
]

PARENT_EDGE_DEBUG_COLUMNS = [
    *PARENT_CANDIDATE_COLUMNS,
    "box_gap_pix",
    "box_gap_arcsec",
    "box_gap_beam_raw",
    "box_gap_source",
    "n_gaussians_1",
    "n_gaussians_2",
    "LAS_beam_1",
    "LAS_beam_2",
    "mask_area_beam_1",
    "mask_area_beam_2",
    "area_3sigma_beam_1",
    "area_3sigma_beam_2",
    "peak_snr_1",
    "peak_snr_2",
    "parent_axis_angle",
    "group1_PA",
    "group2_PA",
    "midpoint_symmetry_score",
    "flux_ratio_score",
    "size_ratio_score",
    "double_lobe_geometry_pass",
    "near_boundary_pair",
    "box_size_1_beam",
    "box_size_2_beam",
    "low_threshold_connected_to_neighbor",
    "same_3sigma_region_as_neighbor",
    "same_2p5sigma_region_as_neighbor",
    "ridge_continuity_score_pair",
    "bridge_snr_support",
    "source_morph_class_1",
    "source_morph_class_2",
    "is_point_like_1",
    "is_point_like_2",
    "endpoint1_veto_final",
    "endpoint2_veto_final",
    "endpoint1_veto_reason",
    "endpoint2_veto_reason",
    "endpoint1_source_morph_class",
    "endpoint2_source_morph_class",
    "endpoint1_hard_point_source_veto",
    "endpoint2_hard_point_source_veto",
    "endpoint1_hard_compact_veto",
    "endpoint2_hard_compact_veto",
    "endpoint1_noise_artifact_veto",
    "endpoint2_noise_artifact_veto",
    "endpoint1_isolated_compact_veto",
    "endpoint2_isolated_compact_veto",
    "endpoint1_near_extended_lobe_candidate",
    "endpoint2_near_extended_lobe_candidate",
    "endpoint1_is_parent_endpoint_allowed",
    "endpoint2_is_parent_endpoint_allowed",
    "is_artifact_risk_1",
    "is_artifact_risk_2",
    "is_lobe_candidate_1",
    "is_lobe_candidate_2",
    "core_candidate_near_midpoint",
    "host_search_radius_arcsec",
    "midpoint_ra",
    "midpoint_dec",
    "query_status",
    "host_query_failed",
    "lobe1_peak_ra",
    "lobe1_peak_dec",
    "lobe2_peak_ra",
    "lobe2_peak_dec",
    "debug_info",
]

PARENT_HOST_CANDIDATE_COLUMNS = [
    "cutout_id",
    "parent_candidate_id",
    "host_role",
    "host_catalog",
    "host_id",
    "host_ra",
    "host_dec",
    "host_sep_midpoint_arcsec",
    "host_perp_offset_beam",
    "host_fractional_position",
    "W1",
    "W2",
    "W1_W2",
    "W1_snr",
    "W2_snr",
    "host_score",
    "host_quality",
    "host_flags",
    "raw_column_map_json",
]

PARENT_DIAGNOSTIC_COLUMNS = [
    "cutout_id",
    "n_total_local_groups",
    "n_hard_point_source_veto",
    "n_hard_compact_veto",
    "n_noise_artifact_veto",
    "n_isolated_compact_veto",
    "n_point_like_hidden",
    "n_resolved_single",
    "n_lobe_candidate",
    "n_near_extended_lobe_candidate",
    "n_artifact_risk",
    "n_parent_pairs_considered",
    "n_near_boundary_pair",
    "n_near_boundary_rescue_applied",
    "n_rejected_near_boundary_no_support",
    "n_double_lobe_geometry_pass",
    "n_lobe_peak_host_contradiction",
    "n_midpoint_host_supports",
    "n_host_queries",
    "n_catwise_queries",
    "n_allwise_fallback_queries",
    "n_final_candidates",
    "n_accepted_high_confidence_parent",
    "n_geometry_only_visual_candidate",
    "n_rejected_parent_candidate",
    "n_parent_conflict_removed",
    "n_parent_high",
    "n_parent_medium",
    "n_parent_needs_host_check",
    "n_parent_suspicious",
    "n_parent_union_boxes",
    "n_point_source_endpoint_rejected",
    "n_compact_endpoint_rejected",
    "n_noise_artifact_endpoint_rejected",
    "n_endpoint_veto_final_rejected",
    "n_geometry_pass_point_source_endpoint",
    "n_geometry_pass_compact_endpoint",
    "n_geometry_pass_noise_artifact_endpoint",
    "n_geometry_pass_endpoint_veto_final",
    "n_parent_candidate_point_source_endpoint",
    "n_parent_candidate_compact_endpoint",
    "n_parent_candidate_noise_artifact_endpoint",
    "n_parent_candidate_endpoint_veto_final",
    "n_rejected_point_like_endpoint",
    "n_rejected_artifact_veto",
    "n_rejected_not_symmetric_lobe_pair",
    "n_rejected_lobe_peak_host_contradiction",
]


@dataclass
class ParentLinkResult:
    candidates: pd.DataFrame
    edges_debug: pd.DataFrame
    host_candidates: pd.DataFrame
    host_query_log: pd.DataFrame
    diagnostics: pd.DataFrame
    needs_visual_check: pd.DataFrame
    source_morph_table: pd.DataFrame


def parent_link_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = (config.get("parent_linking", {}) or {}).copy()
    defaults = {
        "enabled": True,
        "max_box_gap_beam": 12.0,
        "max_center_distance_beam": 40.0,
        "min_axis_alignment": 0.7,
        "min_facing_score": 0.6,
        "max_flux_ratio": 20.0,
        "max_size_ratio": 8.0,
        "min_symmetry_score": 0.6,
        "artifact_veto_score": 1.2,
        "artifact_suspicious_score": 0.8,
        "bright_source_near_beam": 8.0,
        "bright_source_radial_min": 0.72,
        "bright_source_flux_ratio_min": 5.0,
        "fragment_density_radius_beam": 8.0,
        "fragment_count_artifact": 12,
        "lobe_peak_host_radius_arcsec_min": 5.0,
        "lobe_peak_host_radius_arcsec_max": 10.0,
        "max_parent_candidates_per_group": 2,
        "max_parent_candidates_per_cutout": 20,
        "use_artifact_penalties_layer2": True,
    }
    out = dict(defaults)
    out.update(raw)
    out["parent_seed"] = parent_seed_config(config)
    out["host_support"] = host_support_config(config)
    return out


def _with_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out:
            out[col] = pd.Series(dtype=object)
    if out.empty:
        return pd.DataFrame(columns=columns)
    return out[columns]


def _major_axis_beam(row: pd.Series) -> float:
    value = safe_float(row.get("major_axis_beam"), float("nan"))
    if np.isfinite(value):
        return value
    return safe_float(row.get("LAS_beam"), float("nan"))


def _beam_area_pix(pixel_scale: float, beam_arcsec: float) -> float:
    return float(np.pi * (beam_arcsec / max(pixel_scale, 1e-6)) ** 2 / (4.0 * np.log(2.0)))


def _angle_delta_deg(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return 90.0
    return float(abs((a - b + 90.0) % 180.0 - 90.0))


def _alignment_score(delta_deg: float, scale_deg: float = 45.0) -> float:
    return float(np.clip(1.0 - delta_deg / max(scale_deg, 1e-6), 0.0, 1.0))


def _ratio_score(ratio: float, max_ratio: float) -> float:
    if not np.isfinite(ratio):
        return 0.0
    if ratio >= max_ratio:
        return 0.0
    return float(np.clip(1.0 - np.log(max(ratio, 1.0)) / max(np.log(max_ratio), 1e-6), 0.0, 1.0))


def _flux_value(row: pd.Series) -> float:
    return safe_float(row.get("total_flux_gaussian"), safe_float(row.get("peak_flux"), 0.0))


def _point_like(row: pd.Series, area3: float, axis_ratio: float, major_axis: float) -> bool:
    values = [
        safe_float(row.get("n_gaussians"), float("nan")),
        safe_float(row.get("LAS_beam"), float("nan")),
        major_axis,
        area3,
        axis_ratio,
    ]
    if not np.all(np.isfinite(values)):
        return False
    return bool(int(values[0]) == 1 and values[1] < 3.0 and values[2] < 1.5 and values[3] < 2.0 and values[4] < 1.5)


def _minor_axis_beam(row: pd.Series, major_axis: float, axis_ratio: float) -> float:
    value = safe_float(row.get("minor_axis_beam"), float("nan"))
    if np.isfinite(value):
        return value
    if np.isfinite(major_axis) and np.isfinite(axis_ratio) and axis_ratio > 0:
        return float(major_axis / max(axis_ratio, 1e-6))
    return float("nan")


def _hard_point_source_veto(row: pd.Series, area3: float, axis_ratio: float, major_axis: float) -> tuple[bool, str, bool]:
    n_gauss = safe_float(row.get("n_gaussians"), float("nan"))
    las = safe_float(row.get("LAS_beam"), float("nan"))
    connected3 = bool(row.get("same_3sigma_region_as_any_extended_neighbor", False))
    connected25 = bool(row.get("same_2p5sigma_region_as_any_extended_neighbor", False))
    bridge = bool(row.get("bridge_snr_support_to_any_extended_neighbor", False))
    beam_like = bool(
        np.all(np.isfinite([n_gauss, area3, axis_ratio]))
        and int(n_gauss) == 1
        and (np.isfinite(major_axis) and major_axis < 1.8 or np.isfinite(las) and las < 3.0)
        and area3 < 3.0
        and axis_ratio < 1.7
        and not connected3
        and not connected25
        and not bridge
    )
    if beam_like:
        return True, "single_gaussian_beam_like_not_low_threshold_connected", True
    return False, "", False


def _hard_compact_veto(row: pd.Series, area3: float, axis_ratio: float, major_axis: float, minor_axis: float) -> tuple[bool, str]:
    n_gauss = safe_float(row.get("n_gaussians"), float("nan"))
    las = safe_float(row.get("LAS_beam"), float("nan"))
    reasons: list[str] = []
    if (
        np.all(np.isfinite([n_gauss, las, area3, major_axis, axis_ratio]))
        and n_gauss <= 2
        and las < 3.0
        and area3 < 2.0
        and major_axis < 1.5
        and axis_ratio < 1.5
    ):
        reasons.append("beam_like_compact_source")
    if (
        np.all(np.isfinite([n_gauss, las, area3, axis_ratio]))
        and int(n_gauss) == 1
        and las < 3.5
        and area3 < 2.5
        and axis_ratio < 1.6
    ):
        reasons.append("beam_like_compact_source")
    if (
        np.all(np.isfinite([major_axis, minor_axis, area3]))
        and major_axis < 1.4
        and minor_axis < 1.4
        and area3 < 2.0
    ):
        reasons.append("beam_like_compact_source")
    return bool(reasons), ";".join(dict.fromkeys(reasons))


def _noise_artifact_veto(row: pd.Series, area3: float, mask_area: float, peak_snr: float, artifact: bool) -> tuple[bool, str]:
    reasons: list[str] = []
    if np.isfinite(peak_snr) and peak_snr < 6.0:
        reasons.append("peak_snr_lt_6")
    if np.isfinite(area3) and area3 < 1.5:
        reasons.append("area_3sigma_beam_lt_1p5")
    if bool(row.get("artifact_risk", False)) or artifact or _artifact_flags(row):
        reasons.append("artifact_risk")
    if safe_float(row.get("n_only_2sigma_edges"), 0.0) > 0 and safe_float(row.get("n_strong_edges"), 0.0) <= 0:
        reasons.append("only_2sigma_evidence_dominant")
    flags = f"{row.get('artifact_risk_flags', '')} {row.get('debug_info', '')}".lower()
    for token in ["negative_bowl", "sidelobe", "bright residual", "bright_residual"]:
        if token in flags:
            reasons.append(token)
    return bool(reasons), ";".join(dict.fromkeys(reasons))


def _extended_evidence_count(row: pd.Series, area3: float, axis_ratio: float, major_axis: float) -> int:
    atype = _association_type(row)
    allowed_types = {
        "continuous_extended",
        "diffuse_extended",
        "linear_or_tail_like",
        "complex_association",
        "weak_association",
    }
    signals = [
        safe_float(row.get("LAS_beam"), 0.0) >= 3.0,
        np.isfinite(area3) and area3 >= 2.5,
        np.isfinite(major_axis) and major_axis >= 2.0,
        axis_ratio >= 1.5,
        safe_float(row.get("n_gaussians"), 1.0) >= 2,
        atype in allowed_types,
    ]
    return int(sum(bool(value) for value in signals))


def _near_extended_lobe_candidate(row: pd.Series, area3: float, axis_ratio: float, major_axis: float, blocked: bool) -> tuple[bool, str]:
    if blocked:
        return False, ""
    n_gauss = safe_float(row.get("n_gaussians"), float("nan"))
    las = safe_float(row.get("LAS_beam"), float("nan"))
    evidence = _extended_evidence_count(row, area3, axis_ratio, major_axis)
    strong_multi = bool(np.all(np.isfinite([n_gauss, area3, las])) and n_gauss >= 3 and area3 >= 2.0 and las >= 2.5)
    if evidence >= 2 or strong_multi:
        return True, "near_boundary_extended_lobe_fragment"
    return False, ""


def _artifact_flags(row: pd.Series) -> bool:
    quality = _association_quality(row)
    atype = _association_type(row)
    flags = str(row.get("artifact_risk_flags", "")).lower()
    debug = str(row.get("debug_info", "")).lower()
    tokens = ["artifact", "sidelobe", "negative_bowl", "bowl", "bright residual", "bright_residual"]
    explicit = any(bool(row.get(col, False)) for col in ["artifact_risk", "negative_bowl", "sidelobe", "bright_residual"])
    return bool(explicit or quality == "artifact_risk" or atype == "artifact_risk" or any(token in flags or token in debug for token in tokens))


def _artifact_environment(row: pd.Series, groups: pd.DataFrame, cfg: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    beam_arcsec = compute_beam_size_arcsec(config)
    pixel_scale = safe_float(row.get("pixel_scale_arcsec"), 1.5)
    cx = safe_float(row.get("centroid_x"))
    cy = safe_float(row.get("centroid_y"))
    flux = max(_flux_value(row), 1e-6)
    if not np.all(np.isfinite([cx, cy])):
        return {
            "bright_source_distance_beam": np.nan,
            "radial_to_bright_source_score": 0.0,
            "local_fragment_density": 0.0,
            "artifact_environment_score": 0.0,
            "artifact_veto_reason": "",
        }
    work = groups.copy()
    work["_flux_for_artifact"] = work.apply(_flux_value, axis=1)
    bright = work.sort_values("_flux_for_artifact", ascending=False).head(5)
    best_dist = float("inf")
    best_radial = 0.0
    best_ratio = 0.0
    for _, bright_row in bright.iterrows():
        if str(bright_row.get("association_group_id")) == str(row.get("association_group_id")):
            continue
        bx = safe_float(bright_row.get("centroid_x"))
        by = safe_float(bright_row.get("centroid_y"))
        if not np.all(np.isfinite([bx, by])):
            continue
        dist_beam = float(np.hypot(cx - bx, cy - by) * pixel_scale / max(beam_arcsec, 1e-6))
        if dist_beam < best_dist:
            vector_pa = float((np.rad2deg(np.arctan2(cy - by, cx - bx)) + 180.0) % 180.0)
            radial = _alignment_score(_angle_delta_deg(safe_float(row.get("group_PA"), float("nan")), vector_pa))
            best_dist = dist_beam
            best_radial = radial
            best_ratio = safe_float(bright_row.get("_flux_for_artifact"), 0.0) / flux
    radius_pix = float(cfg.get("fragment_density_radius_beam", 8.0)) * beam_arcsec / max(pixel_scale, 1e-6)
    dx = pd.to_numeric(groups.get("centroid_x", pd.Series(dtype=float)), errors="coerce") - cx
    dy = pd.to_numeric(groups.get("centroid_y", pd.Series(dtype=float)), errors="coerce") - cy
    local_count = int(((dx * dx + dy * dy) <= radius_pix * radius_pix).sum()) - 1
    density = float(max(local_count, 0) / (np.pi * float(cfg.get("fragment_density_radius_beam", 8.0)) ** 2))
    score = 0.0
    reasons: list[str] = []
    if (
        best_dist <= float(cfg.get("bright_source_near_beam", 8.0))
        and best_ratio >= float(cfg.get("bright_source_flux_ratio_min", 5.0))
        and best_radial >= float(cfg.get("bright_source_radial_min", 0.72))
    ):
        score += 1.1 + 0.5 * best_radial
        reasons.append("near_radial_to_bright_source")
    if best_dist <= 4.0 and best_ratio >= 10.0:
        score += 0.7
        reasons.append("very_close_to_bright_source")
    if local_count >= int(cfg.get("fragment_count_artifact", 12)):
        score += 0.8
        reasons.append("crowded_fragment_region")
    if _artifact_flags(row):
        score += 1.4
        reasons.append("artifact_flag")
    return {
        "bright_source_distance_beam": best_dist if np.isfinite(best_dist) else np.nan,
        "radial_to_bright_source_score": best_radial,
        "local_fragment_density": density,
        "artifact_environment_score": float(score),
        "artifact_veto_reason": ";".join(reasons),
    }


def _source_morph(
    row: pd.Series,
    area3: float,
    mask_area: float,
    axis_ratio: float,
    major_axis: float,
    minor_axis: float,
    artifact_env: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    # 先把第一层 local group 分成可作为 parent 端点的扩展瓣、普通 resolved single、
    # 点源/紧致源和噪声伪影；第二层只让足够可靠的端点进入双瓣候选。
    artifact = safe_float(artifact_env.get("artifact_environment_score"), 0.0) >= float(cfg.get("artifact_veto_score", 1.2))
    peak_snr = safe_float(row.get("peak_snr"), safe_float(row.get("peak_flux"), 0.0))
    area_for_veto = area3 if np.isfinite(area3) else mask_area
    hard_point, point_reason, beam_like_single = _hard_point_source_veto(row, area_for_veto, axis_ratio, major_axis)
    hard_compact, compact_reason = _hard_compact_veto(row, area_for_veto, axis_ratio, major_axis, minor_axis)
    noise_veto, noise_reason = _noise_artifact_veto(row, area_for_veto, mask_area, peak_snr, artifact)
    point = _point_like(row, area_for_veto, axis_ratio, major_axis) or hard_point or hard_compact
    atype = _association_type(row)
    quality = _association_quality(row)
    signals = [
        safe_float(row.get("LAS_beam"), 0.0) >= 3.0,
        safe_float(row.get("n_gaussians"), 1.0) >= 2,
        np.isfinite(area3) and area3 >= 2.0,
        axis_ratio >= 1.5,
        np.isfinite(major_axis) and major_axis >= 2.0,
        safe_float(row.get("peak_snr"), safe_float(row.get("peak_flux"), 0.0)) >= 6.0,
    ]
    allowed_type = atype not in {"weak_association", "artifact_risk"}
    allowed_quality = quality not in {"low", "artifact_risk"}
    reject: list[str] = []
    lobe_reject: list[str] = []
    if hard_point:
        reject.append("hard_point_source_veto")
        lobe_reject.append("hard_point_source_veto")
    if hard_compact:
        reject.append("hard_compact_veto")
        lobe_reject.append("hard_compact_veto")
    if noise_veto:
        reject.append("noise_artifact_veto")
        lobe_reject.append("noise_artifact_veto")
    if point:
        reject.append("point_like")
    if artifact:
        reject.append("artifact_veto")
        lobe_reject.append("noise_artifact_veto")
    if not allowed_type:
        reject.append(f"bad_type:{atype}")
        lobe_reject.append("not_extended_enough_for_lobe")
    if not allowed_quality:
        reject.append(f"bad_quality:{quality}")
        lobe_reject.append("not_extended_enough_for_lobe")
    if sum(bool(value) for value in signals) < 2:
        reject.append("insufficient_lobe_signals")
        lobe_reject.append("not_extended_enough_for_lobe")
    n_gauss = safe_float(row.get("n_gaussians"), float("nan"))
    enough_self_extended = bool(
        safe_float(row.get("LAS_beam"), 0.0) >= 3.5
        or (np.isfinite(area3) and area3 >= 3.0)
        or (np.isfinite(major_axis) and major_axis >= 2.2)
        or axis_ratio >= 1.8
        or (np.isfinite(n_gauss) and n_gauss >= 3)
    )
    isolated_compact = bool(
        np.isfinite(n_gauss)
        and n_gauss <= 2
        and not enough_self_extended
        and not hard_point
        and not hard_compact
        and not noise_veto
        and not artifact
    )
    isolated_reason = "one_or_two_gaussian_isolated_not_low_threshold_connected" if isolated_compact else ""
    if isolated_compact:
        reject.append("isolated_compact_veto")
        lobe_reject.append("isolated_compact_veto")
    endpoint_veto = bool(hard_point or hard_compact or noise_veto or isolated_compact)
    endpoint_reasons: list[str] = []
    if hard_point:
        endpoint_reasons.append(point_reason)
    if hard_compact:
        endpoint_reasons.append(compact_reason or "beam_like_compact_source")
    if noise_veto:
        endpoint_reasons.append(noise_reason or "noise_or_artifact")
    if isolated_compact:
        endpoint_reasons.append(isolated_reason)
    blocked = bool(endpoint_veto or artifact)
    near_candidate, near_reason = _near_extended_lobe_candidate(row, area3, axis_ratio, major_axis, blocked)
    is_lobe = bool(not blocked and allowed_type and allowed_quality and sum(bool(value) for value in signals) >= 2)
    if hard_point or hard_compact or isolated_compact:
        morph = "point_like_or_compact"
    elif noise_veto:
        morph = "noise_or_artifact"
    elif artifact:
        morph = "noise_or_artifact"
    elif is_lobe:
        morph = "lobe_candidate"
    else:
        morph = "resolved_single"
    return {
        "source_morph_class": morph,
        "is_point_like": bool(point),
        "hard_point_source_veto": bool(hard_point),
        "point_source_veto_reason": point_reason if hard_point else "",
        "is_beam_like_single_gaussian": bool(beam_like_single),
        "hard_compact_veto": bool(hard_compact),
        "compact_veto_reason": compact_reason if hard_compact else "",
        "noise_artifact_veto": bool(noise_veto),
        "noise_artifact_reason": noise_reason if noise_veto else "",
        "isolated_compact_veto": bool(isolated_compact),
        "isolated_compact_reason": isolated_reason,
        "endpoint_veto_final": bool(endpoint_veto),
        "endpoint_veto_reason": ";".join(dict.fromkeys([reason for reason in endpoint_reasons if reason])),
        "is_lobe_candidate": bool(is_lobe),
        "near_extended_lobe_candidate": bool(near_candidate),
        "near_lobe_rescue_reason": near_reason,
        "is_parent_endpoint_allowed": bool(is_lobe or near_candidate),
        "lobe_like_reject_reason": "" if is_lobe else (";".join(dict.fromkeys(lobe_reject)) or "not_extended_enough_for_lobe"),
        "is_artifact_risk": bool(artifact or noise_veto),
        "morph_reject_reason": ";".join(dict.fromkeys(reject)),
    }


def _threshold_index(segmentation: Any, threshold: float) -> int | None:
    if segmentation is None or not hasattr(segmentation, "thresholds"):
        return None
    thresholds = np.asarray(segmentation.thresholds, dtype=float)
    if thresholds.size == 0:
        return None
    idx = int(np.argmin(np.abs(thresholds - float(threshold))))
    if abs(float(thresholds[idx]) - float(threshold)) <= 0.26:
        return idx
    return None


def _label_at_group(segmentation: Any, group: pd.Series, threshold: float) -> int:
    idx = _threshold_index(segmentation, threshold)
    if idx is None or not hasattr(segmentation, "labels_by_threshold"):
        return 0
    labels = segmentation.labels_by_threshold[idx]
    height, width = labels.shape
    x = int(round(np.clip(safe_float(group.get("centroid_x"), -1), 0, width - 1)))
    y = int(round(np.clip(safe_float(group.get("centroid_y"), -1), 0, height - 1)))
    if x < 0 or y < 0:
        return 0
    return int(labels[y, x])


def _line_snr_values(segmentation: Any, group_a: pd.Series, group_b: pd.Series, n: int = 64) -> np.ndarray:
    if segmentation is None or not hasattr(segmentation, "snr_map"):
        return np.asarray([], dtype=float)
    snr = np.asarray(segmentation.snr_map, dtype=float)
    height, width = snr.shape
    x1 = safe_float(group_a.get("centroid_x"), float("nan"))
    y1 = safe_float(group_a.get("centroid_y"), float("nan"))
    x2 = safe_float(group_b.get("centroid_x"), float("nan"))
    y2 = safe_float(group_b.get("centroid_y"), float("nan"))
    if not np.all(np.isfinite([x1, y1, x2, y2])):
        return np.asarray([], dtype=float)
    xs = np.rint(np.linspace(x1, x2, n)).astype(int)
    ys = np.rint(np.linspace(y1, y2, n)).astype(int)
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if not valid.any():
        return np.asarray([], dtype=float)
    values = snr[ys[valid], xs[valid]]
    return values[np.isfinite(values)]


def _neighbor_support_features(segmentation: Any, group_a: pd.Series, group_b: pd.Series) -> dict[str, Any]:
    label3_a = _label_at_group(segmentation, group_a, 3.0)
    label3_b = _label_at_group(segmentation, group_b, 3.0)
    label25_a = _label_at_group(segmentation, group_a, 2.5)
    label25_b = _label_at_group(segmentation, group_b, 2.5)
    same3 = bool(label3_a > 0 and label3_a == label3_b)
    same25 = bool(label25_a > 0 and label25_a == label25_b)
    values = _line_snr_values(segmentation, group_a, group_b)
    if values.size:
        bridge_support = bool(np.nanmean(values >= 2.0) >= 0.45 or np.nanmean(values >= 2.5) >= 0.25)
        ridge = float(np.clip(0.6 * np.nanmean(values >= 2.0) + 0.4 * np.nanmean(values >= 2.5), 0.0, 1.0))
    else:
        bridge_support = False
        ridge = 0.0
    return {
        "same_3sigma_region_as_neighbor": same3,
        "same_2p5sigma_region_as_neighbor": same25,
        "low_threshold_connected_to_neighbor": bool(same3 or same25),
        "bridge_snr_support": bridge_support,
        "ridge_continuity_score_pair": ridge,
    }


def _bbox_gap_pix(box_a: tuple[float, float, float, float] | None, box_b: tuple[float, float, float, float] | None) -> float:
    if box_a is None or box_b is None:
        return float("nan")
    dx = max(box_a[0] - box_b[2], box_b[0] - box_a[2], 0.0)
    dy = max(box_a[1] - box_b[3], box_b[1] - box_a[3], 0.0)
    return float(np.hypot(dx, dy))


def _box_size_beam(seed: pd.Series, group: pd.Series, config: dict[str, Any]) -> float:
    box = _bbox_tuple(seed.get("robust_bbox", "")) or _bbox_tuple(group.get("bounding_box", ""))
    if box is None:
        return float("nan")
    pixel_scale = safe_float(group.get("pixel_scale_arcsec"), 1.5)
    beam_arcsec = compute_beam_size_arcsec(config)
    width = max(box[2] - box[0] + 1, 0.0) * pixel_scale / max(beam_arcsec, 1e-6)
    height = max(box[3] - box[1] + 1, 0.0) * pixel_scale / max(beam_arcsec, 1e-6)
    return float(max(width, height))


def _pair_boundary_features(record: dict[str, Any], seed_a: pd.Series, seed_b: pd.Series, group_a: pd.Series, group_b: pd.Series, segmentation: Any, config: dict[str, Any]) -> dict[str, Any]:
    size1 = _box_size_beam(seed_a, group_a, config)
    size2 = _box_size_beam(seed_b, group_b, config)
    finite_sizes = [value for value in [size1, size2] if np.isfinite(value)]
    mean_size = float(np.mean(finite_sizes)) if finite_sizes else float("nan")
    min_size = float(np.min(finite_sizes)) if finite_sizes else float("nan")
    gap = safe_float(record.get("box_gap_beam_robust"), float("nan"))
    gap_mean = float(gap / max(mean_size, 1e-6)) if np.isfinite(gap) and np.isfinite(mean_size) else float("nan")
    gap_min = float(gap / max(min_size, 1e-6)) if np.isfinite(gap) and np.isfinite(min_size) else float("nan")
    near_boundary = bool(
        np.isfinite(gap)
        and gap >= 0
        and (
            gap <= 6.0
            or (np.isfinite(gap_mean) and gap_mean <= 1.0)
            or (np.isfinite(gap_min) and gap_min <= 1.5)
        )
    )
    support = _neighbor_support_features(segmentation, group_a, group_b)
    support.update(
        {
            "box_size_1_beam": size1,
            "box_size_2_beam": size2,
            "gap_to_mean_box_ratio": gap_mean,
            "gap_to_min_box_ratio": gap_min,
            "near_boundary_pair": near_boundary,
        }
    )
    return support


def _precompute_neighbor_support(frame: pd.DataFrame, groups: pd.DataFrame, segmentation: Any, config: dict[str, Any]) -> pd.DataFrame:
    if frame.empty or groups.empty:
        return frame
    out = frame.copy()
    by_group = {str(row.get("association_group_id")): row for _, row in groups.iterrows()}
    by_seed = {str(row.get("association_group_id")): row for _, row in out.iterrows()}
    ids = list(by_group)
    for gid in ids:
        out.loc[out["association_group_id"].astype(str) == gid, "same_3sigma_region_as_any_extended_neighbor"] = False
        out.loc[out["association_group_id"].astype(str) == gid, "same_2p5sigma_region_as_any_extended_neighbor"] = False
        out.loc[out["association_group_id"].astype(str) == gid, "bridge_snr_support_to_any_extended_neighbor"] = False
    max_gap_pix = 8.0 * compute_beam_size_arcsec(config) / max(safe_float(groups.get("pixel_scale_arcsec", pd.Series([1.5])).iloc[0], 1.5), 1e-6)
    work = groups.copy().reset_index(drop=True)
    work = work.merge(out[["association_group_id", "robust_bbox", "robust_bbox_source"]], on="association_group_id", how="left")
    for idx_i, idx_j in _candidate_search_pairs(work, max_gap_pix):
        group_i = work.iloc[idx_i]
        group_j = work.iloc[idx_j]
        gid_i = str(group_i.get("association_group_id"))
        gid_j = str(group_j.get("association_group_id"))
        seed_i = by_seed.get(gid_i, pd.Series(dtype=object))
        seed_j = by_seed.get(gid_j, pd.Series(dtype=object))
        support = _neighbor_support_features(segmentation, group_i, group_j)
        connected = bool(support["same_3sigma_region_as_neighbor"] or support["same_2p5sigma_region_as_neighbor"] or support["bridge_snr_support"] or safe_float(support["ridge_continuity_score_pair"], 0.0) >= 0.45)
        if not connected:
            continue
        extended_i = _extended_evidence_count(seed_i, safe_float(seed_i.get("area_3sigma_beam"), float("nan")), safe_float(seed_i.get("axis_ratio"), 1.0), safe_float(seed_i.get("major_axis_beam"), float("nan"))) >= 2
        extended_j = _extended_evidence_count(seed_j, safe_float(seed_j.get("area_3sigma_beam"), float("nan")), safe_float(seed_j.get("axis_ratio"), 1.0), safe_float(seed_j.get("major_axis_beam"), float("nan"))) >= 2
        for gid, other_extended in [(gid_i, extended_j), (gid_j, extended_i)]:
            if not other_extended:
                continue
            mask = out["association_group_id"].astype(str) == gid
            if bool(support["same_3sigma_region_as_neighbor"]):
                out.loc[mask, "same_3sigma_region_as_any_extended_neighbor"] = True
            if bool(support["same_2p5sigma_region_as_neighbor"]):
                out.loc[mask, "same_2p5sigma_region_as_any_extended_neighbor"] = True
            if bool(support["bridge_snr_support"]):
                out.loc[mask, "bridge_snr_support_to_any_extended_neighbor"] = True
    return out


def build_source_morph_table(
    cutout_id: str,
    groups: pd.DataFrame,
    components: pd.DataFrame,
    segmentation: Any,
    config: dict[str, Any],
) -> pd.DataFrame:
    cfg = parent_link_config(config)
    parent_seed = build_parent_seed_table(cutout_id, groups, components, segmentation, config)
    parent_by_id = {str(row["association_group_id"]): row for _, row in parent_seed.iterrows()}
    records: list[dict[str, Any]] = []
    for _, row in groups.iterrows():
        work = row.copy()
        if "pixel_scale_arcsec" not in work:
            work["pixel_scale_arcsec"] = safe_float(groups.get("pixel_scale_arcsec", pd.Series([1.5])).iloc[0], 1.5) if not groups.empty else 1.5
        gid = str(work.get("association_group_id", work.get("local_group_id", "")))
        parent = parent_by_id.get(gid, pd.Series(dtype=object))
        robust = _robust_bbox_for_group(work, components, segmentation, config)
        peak_snr, missing_peak = _peak_snr_for_group(work, components, segmentation)
        area3 = safe_float(robust.get("area_3sigma_beam"), safe_float(parent.get("area_3sigma_beam"), float("nan")))
        mask_area = _mask_area_beam(work, config)
        axis_ratio = safe_float(work.get("axis_ratio"), 1.0)
        major_axis = _major_axis_beam(work)
        minor_axis = _minor_axis_beam(work, major_axis, axis_ratio)
        work["peak_snr"] = peak_snr
        work["same_3sigma_region_as_any_extended_neighbor"] = False
        work["same_2p5sigma_region_as_any_extended_neighbor"] = False
        work["bridge_snr_support_to_any_extended_neighbor"] = False
        artifact_env = _artifact_environment(work, groups, cfg, config)
        morph = _source_morph(work, area3, mask_area, axis_ratio, major_axis, minor_axis, artifact_env, cfg)
        missing = []
        if missing_peak:
            missing.append("peak_snr")
        if bool(robust.get("missing_area_3sigma_beam", False)):
            missing.append("area_3sigma_beam")
        if bool(robust.get("missing_robust_bbox", False)):
            missing.append("robust_bbox")
        records.append(
            {
                "cutout_id": cutout_id,
                "group_id": gid,
                "association_group_id": gid,
                "n_gaussians": int(safe_float(work.get("n_gaussians"), 1.0)),
                "LAS_beam": safe_float(work.get("LAS_beam"), float("nan")),
                "major_axis_beam": major_axis,
                "minor_axis_beam": minor_axis,
                "area_3sigma_beam": area3,
                "mask_area_beam": mask_area,
                "axis_ratio": axis_ratio,
                "peak_snr": peak_snr,
                "association_quality": _association_quality(work),
                "association_type": _association_type(work),
                "source_morph_class": morph["source_morph_class"],
                "is_point_like": morph["is_point_like"],
                "hard_point_source_veto": morph["hard_point_source_veto"],
                "point_source_veto_reason": morph["point_source_veto_reason"],
                "is_beam_like_single_gaussian": morph["is_beam_like_single_gaussian"],
                "hard_compact_veto": morph["hard_compact_veto"],
                "compact_veto_reason": morph["compact_veto_reason"],
                "noise_artifact_veto": morph["noise_artifact_veto"],
                "noise_artifact_reason": morph["noise_artifact_reason"],
                "isolated_compact_veto": morph["isolated_compact_veto"],
                "isolated_compact_reason": morph["isolated_compact_reason"],
                "endpoint_veto_final": morph["endpoint_veto_final"],
                "endpoint_veto_reason": morph["endpoint_veto_reason"],
                "is_lobe_candidate": morph["is_lobe_candidate"],
                "near_extended_lobe_candidate": morph["near_extended_lobe_candidate"],
                "near_lobe_rescue_reason": morph["near_lobe_rescue_reason"],
                "is_parent_endpoint_allowed": morph["is_parent_endpoint_allowed"],
                "lobe_like_reject_reason": morph["lobe_like_reject_reason"],
                "same_3sigma_region_as_any_extended_neighbor": False,
                "same_2p5sigma_region_as_any_extended_neighbor": False,
                "bridge_snr_support_to_any_extended_neighbor": False,
                "is_artifact_risk": morph["is_artifact_risk"],
                "morph_reject_reason": morph["morph_reject_reason"],
                "bright_source_distance_beam": artifact_env["bright_source_distance_beam"],
                "radial_to_bright_source_score": artifact_env["radial_to_bright_source_score"],
                "local_fragment_density": artifact_env["local_fragment_density"],
                "artifact_environment_score": artifact_env["artifact_environment_score"],
                "artifact_veto_reason": artifact_env["artifact_veto_reason"],
                "is_parent_seed": bool(parent.get("is_parent_seed", False)),
                "robust_bbox": parent.get("robust_bbox", ""),
                "robust_bbox_source": parent.get("robust_bbox_source", robust.get("robust_bbox_source", "")),
                "missing_fields": ",".join(missing),
            }
        )
    frame = _with_columns(pd.DataFrame(records), SOURCE_MORPH_TABLE_COLUMNS)
    frame = _precompute_neighbor_support(frame, groups, segmentation, config)
    if frame.empty:
        return frame
    recomputed: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        gid = str(row.get("association_group_id"))
        original = groups[groups["association_group_id"].astype(str) == gid]
        work = original.iloc[0].copy() if not original.empty else row.copy()
        for col in [
            "peak_snr",
            "same_3sigma_region_as_any_extended_neighbor",
            "same_2p5sigma_region_as_any_extended_neighbor",
            "bridge_snr_support_to_any_extended_neighbor",
        ]:
            work[col] = row.get(col)
        artifact_env = {
            "bright_source_distance_beam": row.get("bright_source_distance_beam"),
            "radial_to_bright_source_score": row.get("radial_to_bright_source_score"),
            "local_fragment_density": row.get("local_fragment_density"),
            "artifact_environment_score": row.get("artifact_environment_score"),
            "artifact_veto_reason": row.get("artifact_veto_reason"),
        }
        morph = _source_morph(
            work,
            safe_float(row.get("area_3sigma_beam"), float("nan")),
            safe_float(row.get("mask_area_beam"), float("nan")),
            safe_float(row.get("axis_ratio"), 1.0),
            safe_float(row.get("major_axis_beam"), float("nan")),
            safe_float(row.get("minor_axis_beam"), float("nan")),
            artifact_env,
            cfg,
        )
        rec = row.to_dict()
        rec.update(morph)
        recomputed.append(rec)
    return _with_columns(pd.DataFrame(recomputed), SOURCE_MORPH_TABLE_COLUMNS)


def _group_by_id(groups: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(row.get("association_group_id")): row for _, row in groups.iterrows()}


def _symmetry_scores(record: dict[str, Any], cfg: dict[str, Any]) -> tuple[float, float, float]:
    flux_score = _ratio_score(safe_float(record.get("flux_ratio"), float("inf")), float(cfg.get("max_flux_ratio", 20.0)))
    size_score = _ratio_score(safe_float(record.get("size_ratio"), float("inf")), float(cfg.get("max_size_ratio", 8.0)))
    midpoint_symmetry = float(np.clip(0.5 * (flux_score + size_score), 0.0, 1.0))
    symmetry = float(
        0.30 * safe_float(record.get("axis_alignment_score"), 0.0)
        + 0.25 * safe_float(record.get("facing_score"), 0.0)
        + 0.15 * midpoint_symmetry
        + 0.15 * flux_score
        + 0.15 * size_score
    )
    return symmetry, midpoint_symmetry, flux_score, size_score


def _parent_union(row: dict[str, Any], seed_a: pd.Series, seed_b: pd.Series, group_a: pd.Series, group_b: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    box_a = _bbox_tuple(seed_a.get("robust_bbox", "")) or _bbox_tuple(group_a.get("bounding_box", ""))
    box_b = _bbox_tuple(seed_b.get("robust_bbox", "")) or _bbox_tuple(group_b.get("bounding_box", ""))
    if box_a is None or box_b is None:
        return {
            "parent_bbox_xmin": np.nan,
            "parent_bbox_xmax": np.nan,
            "parent_bbox_ymin": np.nan,
            "parent_bbox_ymax": np.nan,
            "parent_LAS_beam": np.nan,
            "parent_LAS_arcsec": np.nan,
            "parent_union_area_beam": np.nan,
        }
    x0 = min(box_a[0], box_b[0])
    y0 = min(box_a[1], box_b[1])
    x1 = max(box_a[2], box_b[2])
    y1 = max(box_a[3], box_b[3])
    pixel_scale = safe_float(group_a.get("pixel_scale_arcsec"), safe_float(group_b.get("pixel_scale_arcsec"), 1.5))
    beam_arcsec = compute_beam_size_arcsec(config)
    las_arcsec = float(np.hypot(x1 - x0, y1 - y0) * pixel_scale)
    area_beam = float(max((x1 - x0 + 1) * (y1 - y0 + 1), 0.0) / max(_beam_area_pix(pixel_scale, beam_arcsec), 1e-6))
    return {
        "parent_bbox_xmin": x0,
        "parent_bbox_xmax": x1,
        "parent_bbox_ymin": y0,
        "parent_bbox_ymax": y1,
        "parent_LAS_beam": las_arcsec / max(beam_arcsec, 1e-6),
        "parent_LAS_arcsec": las_arcsec,
        "parent_union_area_beam": area_beam,
    }


def _classify_pair(record: dict[str, Any], seed_a: pd.Series, seed_b: pd.Series, cfg: dict[str, Any]) -> tuple[bool, str]:
    class_a = str(seed_a.get("source_morph_class", ""))
    class_b = str(seed_b.get("source_morph_class", ""))
    use_endpoint_filtering = bool(cfg.get("use_stage2_endpoint_filtering", True))
    use_relative_scale = bool(cfg.get("use_stage2_relative_scale_constraints", True))
    use_artifact_layer2 = bool(cfg.get("use_artifact_penalties_layer2", True))
    if use_endpoint_filtering:
        if bool(seed_a.get("endpoint_veto_final", False)) or bool(seed_b.get("endpoint_veto_final", False)):
            return False, "endpoint_vetoed_or_not_extended_for_near_boundary_rescue"
        if bool(seed_a.get("hard_point_source_veto", False)) or bool(seed_b.get("hard_point_source_veto", False)):
            return False, "point_like_endpoint"
        if bool(seed_a.get("hard_compact_veto", False)) or bool(seed_b.get("hard_compact_veto", False)):
            return False, "point_like_endpoint"
        if use_artifact_layer2 and (bool(seed_a.get("noise_artifact_veto", False)) or bool(seed_b.get("noise_artifact_veto", False))):
            return False, "artifact_veto"
        if bool(seed_a.get("isolated_compact_veto", False)) or bool(seed_b.get("isolated_compact_veto", False)):
            return False, "endpoint_vetoed_or_not_extended_for_near_boundary_rescue"
        if use_artifact_layer2 and (class_a in {"artifact_risk", "noise_or_artifact"} or class_b in {"artifact_risk", "noise_or_artifact"}):
            return False, "artifact_veto"
        if class_a in {"point_like", "point_like_or_compact"} or class_b in {"point_like", "point_like_or_compact"}:
            return False, "point_like_endpoint"
        if class_a != "lobe_candidate" or class_b != "lobe_candidate":
            return False, "asymmetric_or_point_lobe_pair"
    if safe_float(record.get("box_gap_beam_robust"), float("inf")) > float(cfg.get("max_box_gap_beam", 12.0)):
        return False, "box_gap_too_large"
    if safe_float(record.get("center_distance_beam"), float("inf")) > float(cfg.get("max_center_distance_beam", 40.0)):
        return False, "center_distance_too_large"
    if safe_float(record.get("axis_alignment_score"), 0.0) < float(cfg.get("min_axis_alignment", 0.7)):
        return False, "geometry_axis_alignment_too_low"
    if safe_float(record.get("facing_score"), 0.0) < float(cfg.get("min_facing_score", 0.6)):
        return False, "geometry_facing_too_low"
    if use_relative_scale:
        if safe_float(record.get("flux_ratio"), float("inf")) > float(cfg.get("max_flux_ratio", 20.0)):
            return False, "asymmetric_or_point_lobe_pair"
        if safe_float(record.get("size_ratio"), float("inf")) > float(cfg.get("max_size_ratio", 8.0)):
            return False, "asymmetric_or_point_lobe_pair"
        if safe_float(record.get("symmetry_score"), 0.0) < float(cfg.get("min_symmetry_score", 0.6)):
            return False, "not_symmetric_lobe_pair"
    return True, ""


def _normal_endpoint_allowed(seed_a: pd.Series, seed_b: pd.Series, use_artifact_layer2: bool = True) -> bool:
    return bool(
        str(seed_a.get("source_morph_class", "")) == "lobe_candidate"
        and str(seed_b.get("source_morph_class", "")) == "lobe_candidate"
        and bool(seed_a.get("is_lobe_candidate", False))
        and bool(seed_b.get("is_lobe_candidate", False))
        and not bool(seed_a.get("endpoint_veto_final", False))
        and not bool(seed_b.get("endpoint_veto_final", False))
        and not bool(seed_a.get("hard_point_source_veto", False))
        and not bool(seed_b.get("hard_point_source_veto", False))
        and not bool(seed_a.get("hard_compact_veto", False))
        and not bool(seed_b.get("hard_compact_veto", False))
        and (not use_artifact_layer2 or (not bool(seed_a.get("noise_artifact_veto", False))
        and not bool(seed_b.get("noise_artifact_veto", False))))
        and not bool(seed_a.get("isolated_compact_veto", False))
        and not bool(seed_b.get("isolated_compact_veto", False))
    )


def _rescue_endpoint_allowed(record: dict[str, Any], seed_a: pd.Series, seed_b: pd.Series, use_artifact_layer2: bool = True) -> bool:
    if not bool(record.get("near_boundary_pair", False)):
        return False
    veto_flags = [
        "endpoint_veto_final",
        "hard_point_source_veto",
        "hard_compact_veto",
        "isolated_compact_veto",
    ]
    if use_artifact_layer2:
        veto_flags.append("noise_artifact_veto")
    for flag in veto_flags:
        if bool(seed_a.get(flag, False)) or bool(seed_b.get(flag, False)):
            return False
    if use_artifact_layer2 and (bool(seed_a.get("is_artifact_risk", False)) or bool(seed_b.get("is_artifact_risk", False))):
        return False
    left = bool(seed_a.get("is_lobe_candidate", False) or seed_a.get("near_extended_lobe_candidate", False))
    right = bool(seed_b.get("is_lobe_candidate", False) or seed_b.get("near_extended_lobe_candidate", False))
    return bool(left and right)


def _near_boundary_strong_support(record: dict[str, Any], seed_a: pd.Series, seed_b: pd.Series) -> bool:
    n1 = safe_float(seed_a.get("n_gaussians"), safe_float(record.get("n_gaussians_1"), 0.0))
    n2 = safe_float(seed_b.get("n_gaussians"), safe_float(record.get("n_gaussians_2"), 0.0))
    area1 = safe_float(seed_a.get("area_3sigma_beam"), safe_float(record.get("area_3sigma_beam_1"), 0.0))
    area2 = safe_float(seed_b.get("area_3sigma_beam"), safe_float(record.get("area_3sigma_beam_2"), 0.0))
    ext1 = bool(seed_a.get("near_extended_lobe_candidate", False) or seed_a.get("is_lobe_candidate", False))
    ext2 = bool(seed_b.get("near_extended_lobe_candidate", False) or seed_b.get("is_lobe_candidate", False))
    return bool(
        bool(record.get("same_3sigma_region_as_neighbor", False))
        or bool(record.get("same_2p5sigma_region_as_neighbor", False))
        or safe_float(record.get("ridge_continuity_score_pair"), 0.0) >= 0.45
        or bool(record.get("bridge_snr_support", False))
        or (n1 >= 2 and n2 >= 2)
        or (n1 >= 3 and ext2)
        or (n2 >= 3 and ext1)
        or (area1 >= 2.5 and area2 >= 2.5)
    )


def _classify_near_boundary_rescue(record: dict[str, Any], seed_a: pd.Series, seed_b: pd.Series, cfg: dict[str, Any]) -> tuple[bool, str]:
    use_endpoint_filtering = bool(cfg.get("use_stage2_endpoint_filtering", True))
    use_relative_scale = bool(cfg.get("use_stage2_relative_scale_constraints", True))
    if use_endpoint_filtering and not _rescue_endpoint_allowed(record, seed_a, seed_b):
        return False, "endpoint_vetoed_or_not_extended_for_near_boundary_rescue"
    if safe_float(record.get("axis_alignment_score"), 0.0) < 0.50:
        return False, "near_boundary_axis_alignment_too_low"
    if safe_float(record.get("facing_score"), 0.0) < 0.40:
        return False, "near_boundary_facing_too_low"
    if use_relative_scale:
        if safe_float(record.get("flux_ratio"), float("inf")) > 30.0:
            return False, "near_boundary_flux_ratio_too_high"
        if safe_float(record.get("size_ratio"), float("inf")) > 10.0:
            return False, "near_boundary_size_ratio_too_high"
    if not _near_boundary_strong_support(record, seed_a, seed_b):
        return False, "near_boundary_but_no_extended_or_bridge_support"
    return True, ""


def _record_endpoint_fields(seed_i: pd.Series, seed_j: pd.Series) -> dict[str, Any]:
    return {
        "source_morph_class_1": seed_i.get("source_morph_class", ""),
        "source_morph_class_2": seed_j.get("source_morph_class", ""),
        "is_point_like_1": bool(seed_i.get("is_point_like", False)),
        "is_point_like_2": bool(seed_j.get("is_point_like", False)),
        "endpoint1_veto_final": bool(seed_i.get("endpoint_veto_final", False)),
        "endpoint2_veto_final": bool(seed_j.get("endpoint_veto_final", False)),
        "endpoint1_veto_reason": seed_i.get("endpoint_veto_reason", ""),
        "endpoint2_veto_reason": seed_j.get("endpoint_veto_reason", ""),
        "endpoint1_source_morph_class": seed_i.get("source_morph_class", ""),
        "endpoint2_source_morph_class": seed_j.get("source_morph_class", ""),
        "endpoint1_hard_point_source_veto": bool(seed_i.get("hard_point_source_veto", False)),
        "endpoint2_hard_point_source_veto": bool(seed_j.get("hard_point_source_veto", False)),
        "endpoint1_hard_compact_veto": bool(seed_i.get("hard_compact_veto", False)),
        "endpoint2_hard_compact_veto": bool(seed_j.get("hard_compact_veto", False)),
        "endpoint1_noise_artifact_veto": bool(seed_i.get("noise_artifact_veto", False)),
        "endpoint2_noise_artifact_veto": bool(seed_j.get("noise_artifact_veto", False)),
        "endpoint1_isolated_compact_veto": bool(seed_i.get("isolated_compact_veto", False)),
        "endpoint2_isolated_compact_veto": bool(seed_j.get("isolated_compact_veto", False)),
        "endpoint1_near_extended_lobe_candidate": bool(seed_i.get("near_extended_lobe_candidate", False)),
        "endpoint2_near_extended_lobe_candidate": bool(seed_j.get("near_extended_lobe_candidate", False)),
        "endpoint1_is_parent_endpoint_allowed": bool(seed_i.get("is_parent_endpoint_allowed", False)),
        "endpoint2_is_parent_endpoint_allowed": bool(seed_j.get("is_parent_endpoint_allowed", False)),
        "is_artifact_risk_1": bool(seed_i.get("is_artifact_risk", False)),
        "is_artifact_risk_2": bool(seed_j.get("is_artifact_risk", False)),
        "is_lobe_candidate_1": bool(seed_i.get("is_lobe_candidate", False)),
        "is_lobe_candidate_2": bool(seed_j.get("is_lobe_candidate", False)),
    }


def _make_pair_edges(cutout_id: str, groups: pd.DataFrame, morph: pd.DataFrame, segmentation: Any, config: dict[str, Any]) -> pd.DataFrame:
    # 第二层不再合并所有近邻，而是寻找“两个扩展端点 + 合理间隙/对称性”的 parent 候选；
    # near-boundary rescue 只处理被切图边界截断的疑似双瓣结构。
    cfg = parent_link_config(config)
    cfg["use_stage2_relative_scale_constraints"] = ablation_enabled(config, "use_stage2_relative_scale_constraints")
    cfg["use_stage2_endpoint_filtering"] = ablation_enabled(config, "use_stage2_endpoint_filtering")
    cfg["use_artifact_penalties_layer2"] = ablation_enabled(config, "use_artifact_penalties_layer2")
    if groups.empty or len(groups) < 2:
        return pd.DataFrame(columns=PARENT_EDGE_DEBUG_COLUMNS)
    seed_by_id = {str(row["association_group_id"]): row for _, row in morph.iterrows()}
    work = groups.copy().reset_index(drop=True)
    if "pixel_scale_arcsec" not in work:
        work["pixel_scale_arcsec"] = 1.5
    work = work.merge(morph[["association_group_id", "robust_bbox", "robust_bbox_source", "is_parent_seed"]], on="association_group_id", how="left")
    beam_arcsec = compute_beam_size_arcsec(config)
    pixel_scale = safe_float(work["pixel_scale_arcsec"].iloc[0], 1.5)
    normal_gap_pix = float(cfg.get("max_box_gap_beam", 12.0)) * beam_arcsec / max(pixel_scale, 1e-6)
    max_box_size_beam = 0.0
    for _, row in work.iterrows():
        box = _bbox_tuple(row.get("robust_bbox", "")) or _bbox_tuple(row.get("bounding_box", ""))
        if box is None:
            continue
        max_box_size_beam = max(
            max_box_size_beam,
            max(box[2] - box[0] + 1, box[3] - box[1] + 1) * pixel_scale / max(beam_arcsec, 1e-6),
        )
    debug_gap_beam = max(float(cfg.get("max_box_gap_beam", 12.0)), min(max_box_size_beam, 18.0), 6.0)
    max_gap_pix = debug_gap_beam * beam_arcsec / max(pixel_scale, 1e-6)
    records: list[dict[str, Any]] = []
    for pair_index, (idx_i, idx_j) in enumerate(_candidate_search_pairs(work, max_gap_pix)):
        group_i = work.iloc[idx_i]
        group_j = work.iloc[idx_j]
        seed_i = seed_by_id.get(str(group_i.get("association_group_id")), pd.Series(dtype=object))
        seed_j = seed_by_id.get(str(group_j.get("association_group_id")), pd.Series(dtype=object))
        record = _compute_pair(cutout_id, pair_index, group_i, group_j, seed_i, seed_j, work, seed_by_id, config)
        record.update(_pair_boundary_features(record, seed_i, seed_j, group_i, group_j, segmentation, config))
        symmetry, midpoint_symmetry, flux_score, size_score = _symmetry_scores(record, cfg)
        record["symmetry_score"] = symmetry
        record["midpoint_symmetry_score"] = midpoint_symmetry
        record["flux_ratio_score"] = flux_score
        record["size_ratio_score"] = size_score
        geom_pass, reason = _classify_pair(record, seed_i, seed_j, cfg)
        rescue_pass = False
        rescue_reason = ""
        use_artifact_layer2 = bool(cfg.get("use_artifact_penalties_layer2", True))
        normal_endpoint = _normal_endpoint_allowed(seed_i, seed_j, use_artifact_layer2)
        rescue_endpoint = _rescue_endpoint_allowed(record, seed_i, seed_j, use_artifact_layer2)
        if not geom_pass and bool(record.get("near_boundary_pair", False)):
            if rescue_endpoint:
                rescue_pass, rescue_reason = _classify_near_boundary_rescue(record, seed_i, seed_j, cfg)
            else:
                rescue_reason = "endpoint_vetoed_or_not_extended_for_near_boundary_rescue"
        rescue_applied = bool(rescue_pass)
        geom_pass = bool(geom_pass or rescue_pass)
        if geom_pass:
            reason = ""
        elif rescue_reason:
            reason = rescue_reason
        elif bool(record.get("near_boundary_pair", False)) and not normal_endpoint:
            reason = "endpoint_vetoed_or_not_extended_for_near_boundary_rescue"
        artifact_pair = max(safe_float(seed_i.get("artifact_environment_score"), 0.0), safe_float(seed_j.get("artifact_environment_score"), 0.0)) if bool(cfg.get("use_artifact_penalties_layer2", True)) else 0.0
        lobe_score = float(
            1.4 * symmetry
            + 0.7 * np.clip(1.0 - safe_float(record.get("box_gap_beam_robust"), 999.0) / max(float(cfg.get("max_box_gap_beam", 12.0)), 1e-6), 0.0, 1.0)
            + 0.3 * (1.0 if bool(record.get("core_candidate_near_midpoint", False)) else 0.0)
        )
        union = _parent_union(record, seed_i, seed_j, group_i, group_j, config)
        record.update(union)
        record.update(
            {
                "parent_candidate_id": f"{cutout_id}_parent_pc{pair_index:04d}",
                "parent_candidate_type": "physics_aware_near_boundary_rescue" if rescue_applied else "physics_aware_double_lobe",
                "double_lobe_geometry_pass": bool(geom_pass),
                "near_boundary_rescue_applied": bool(rescue_applied),
                "near_boundary_rescue_reason": "near_boundary_extended_lobe_fragment" if rescue_applied else "",
                "lobe_pair_score": lobe_score,
                "parent_score_geometry": lobe_score,
                "best_host_score": 0.0,
                "parent_score_final": lobe_score,
                "host_evidence": "not_checked",
                "parent_candidate_quality": "rejected",
                "host_status": "not_queried",
                "host_quality": "none",
                "lobe_peak_host_found": False,
                "lobe1_peak_host_found": False,
                "lobe2_peak_host_found": False,
                "lobe1_peak_host_score": 0.0,
                "lobe2_peak_host_score": 0.0,
                "artifact_environment_score_pair": artifact_pair,
                "suspicious_reason": "",
                "rejection_reason": reason,
                "needs_visual_check": False,
                **_record_endpoint_fields(seed_i, seed_j),
                "debug_info": json_dumps_safe(
                    {
                        "morph_reject_reason_1": seed_i.get("morph_reject_reason", ""),
                        "morph_reject_reason_2": seed_j.get("morph_reject_reason", ""),
                        "point_source_veto_reason_1": seed_i.get("point_source_veto_reason", ""),
                        "point_source_veto_reason_2": seed_j.get("point_source_veto_reason", ""),
                        "compact_veto_reason_1": seed_i.get("compact_veto_reason", ""),
                        "compact_veto_reason_2": seed_j.get("compact_veto_reason", ""),
                        "noise_artifact_reason_1": seed_i.get("noise_artifact_reason", ""),
                        "noise_artifact_reason_2": seed_j.get("noise_artifact_reason", ""),
                        "isolated_compact_reason_1": seed_i.get("isolated_compact_reason", ""),
                        "isolated_compact_reason_2": seed_j.get("isolated_compact_reason", ""),
                        "near_lobe_rescue_reason_1": seed_i.get("near_lobe_rescue_reason", ""),
                        "near_lobe_rescue_reason_2": seed_j.get("near_lobe_rescue_reason", ""),
                        "artifact_veto_reason_1": seed_i.get("artifact_veto_reason", ""),
                        "artifact_veto_reason_2": seed_j.get("artifact_veto_reason", ""),
                    }
                ),
            }
        )
        records.append(record)
    return _with_columns(pd.DataFrame(records), PARENT_EDGE_DEBUG_COLUMNS)


def _peak_coord_for_group(group: pd.Series, components: pd.DataFrame) -> tuple[float, float]:
    ids = set(_component_ids(group))
    if ids and components is not None and not components.empty:
        subset = components[components["component_index"].astype(int).isin(ids)].copy()
        if not subset.empty:
            flux_col = "Peak_flux" if "Peak_flux" in subset else "_peak_flux"
            subset["_rank_flux"] = pd.to_numeric(subset.get(flux_col, pd.Series(0.0, index=subset.index)), errors="coerce").fillna(0.0)
            best = subset.sort_values("_rank_flux", ascending=False).iloc[0]
            return safe_float(best.get("RA", best.get("_ra"))), safe_float(best.get("DEC", best.get("_dec")))
    return safe_float(group.get("ra")), safe_float(group.get("dec"))


def _query_position_hosts(
    ra: float,
    dec: float,
    radius_arcsec: float,
    host_client: HostQueryClient,
    host_cfg: dict[str, Any],
    max_host_queries_state: dict[str, int],
    max_host_queries: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    logs: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    any_failed = False
    status = "not_queried"
    for catalogue in host_cfg.get("catalog_priority", ["catwise2020", "allwise"]):
        if max_host_queries is not None and max_host_queries_state["count"] >= max_host_queries:
            status = "max_host_queries_reached"
            break
        result = host_client.query_catalogue(float(ra), float(dec), float(radius_arcsec), str(catalogue))
        max_host_queries_state["count"] += 1
        logs.append(result.log)
        qstatus = str(result.log["status"].iloc[0]) if not result.log.empty else ""
        if qstatus == "failed":
            any_failed = True
        if not result.results.empty:
            raw_frames.append(result.results)
            status = f"{catalogue}_results"
            break
        status = qstatus or f"{catalogue}_empty"
    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=HOST_QUERY_LOG_COLUMNS)
    return raw, log, status, any_failed


def _score_lobe_peak_hosts(
    raw_hosts: pd.DataFrame,
    ra: float,
    dec: float,
    radius_arcsec: float,
    cutout_id: str,
    parent_candidate_id: str,
    role: str,
) -> pd.DataFrame:
    if raw_hosts is None or raw_hosts.empty or not np.all(np.isfinite([ra, dec])):
        return pd.DataFrame(columns=PARENT_HOST_CANDIDATE_COLUMNS)
    center = SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame="icrs")
    records: list[dict[str, Any]] = []
    for _, host in raw_hosts.iterrows():
        hra = safe_float(host.get("host_ra"))
        hdec = safe_float(host.get("host_dec"))
        if not np.all(np.isfinite([hra, hdec])):
            continue
        hc = SkyCoord(hra * u.deg, hdec * u.deg, frame="icrs")
        sep = float(center.separation(hc).arcsec)
        closeness = float(np.clip(1.0 - sep / max(radius_arcsec, 1e-6), 0.0, 1.0))
        w1snr = safe_float(host.get("W1_snr"), float("nan"))
        w2snr = safe_float(host.get("W2_snr"), float("nan"))
        det = 0.0
        if np.isfinite(w1snr):
            det += min(w1snr / 10.0, 1.0)
        if np.isfinite(w2snr):
            det += 0.5 * min(w2snr / 10.0, 1.0)
        if det == 0.0 and np.isfinite(safe_float(host.get("W1"), float("nan"))):
            det = 0.5
        score = float(1.8 * closeness + 0.8 * det)
        quality = "high" if score >= 2.0 and sep <= 0.6 * radius_arcsec else "medium" if score >= 1.2 else "low"
        records.append(
            {
                "cutout_id": cutout_id,
                "parent_candidate_id": parent_candidate_id,
                "host_role": role,
                "host_catalog": host.get("catalogue", ""),
                "host_id": host.get("host_id", ""),
                "host_ra": hra,
                "host_dec": hdec,
                "host_sep_midpoint_arcsec": sep,
                "host_perp_offset_beam": np.nan,
                "host_fractional_position": np.nan,
                "W1": safe_float(host.get("W1"), np.nan),
                "W2": safe_float(host.get("W2"), np.nan),
                "W1_W2": safe_float(host.get("W1_W2"), np.nan),
                "W1_snr": w1snr,
                "W2_snr": w2snr,
                "host_score": score,
                "host_quality": quality,
                "host_flags": "",
                "raw_column_map_json": host.get("raw_column_map_json", ""),
            }
        )
    frame = _with_columns(pd.DataFrame(records), PARENT_HOST_CANDIDATE_COLUMNS)
    if frame.empty:
        return frame
    return frame.sort_values(["host_score", "host_sep_midpoint_arcsec"], ascending=[False, True])


def _apply_limits(candidates: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    rank = {"high": 4, "medium": 3, "needs_host_check": 2, "suspicious": 1}
    work = candidates.copy()
    work["_rank"] = work["parent_candidate_quality"].astype(str).map(rank).fillna(0)
    work = work.sort_values(["_rank", "parent_score_final", "symmetry_score", "box_gap_beam_robust"], ascending=[False, False, False, True])
    max_group = int(cfg.get("max_parent_candidates_per_group", 2))
    max_cutout = int(cfg.get("max_parent_candidates_per_cutout", 20))
    kept: list[int] = []
    counts: dict[str, int] = {}
    for idx, row in work.iterrows():
        if len(kept) >= max_cutout:
            break
        left = str(row.get("local_group_id_1"))
        right = str(row.get("local_group_id_2"))
        if counts.get(left, 0) >= max_group or counts.get(right, 0) >= max_group:
            continue
        kept.append(idx)
        counts[left] = counts.get(left, 0) + 1
        counts[right] = counts.get(right, 0) + 1
    return work.loc[kept].drop(columns=["_rank"], errors="ignore")


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _parent_radio_evidence(row: pd.Series | dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    evidence: list[str] = []
    if _truthy(row.get("same_3sigma_region_as_neighbor")) or _truthy(row.get("same_2p5sigma_region_as_neighbor")):
        evidence.append("common_diffuse_envelope")
    if _truthy(row.get("bridge_snr_support")) or safe_float(row.get("ridge_continuity_score_pair"), 0.0) >= float(
        cfg.get("min_parent_ridge_support", cfg.get("meerklass_min_parent_ridge_support", 0.65))
    ):
        evidence.append("residual_radio_bridge")
    if bool(cfg.get("use_midpoint_host_support", True)) and str(row.get("host_evidence", "")) == "supports_double_lobe" and _quality_rank(str(row.get("host_quality", "none"))) >= _quality_rank("medium"):
        evidence.append("midpoint_host")
    return sorted(set(evidence))


def classify_parent_acceptance(row: pd.Series | dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Classify a parent pair into final accepted, visual-only, or rejected."""

    # host 证据用于加分或降级，但不单独替代射电几何证据；避免把偶然红外源误判为 parent。
    cfg = parent_link_config(config)
    cfg["use_midpoint_host_support"] = ablation_enabled(config, "use_midpoint_host_support")
    cfg["use_lobe_peak_host_contradiction"] = ablation_enabled(config, "use_lobe_peak_host_contradiction")
    cfg["use_artifact_penalties_layer2"] = ablation_enabled(config, "use_artifact_penalties_layer2")
    host_evidence = str(row.get("host_evidence", ""))
    host_quality = str(row.get("host_quality", "none"))
    lobe_peak = _truthy(row.get("lobe_peak_host_found")) or _truthy(row.get("lobe1_peak_host_found")) or _truthy(row.get("lobe2_peak_host_found"))
    midpoint_host = host_evidence == "supports_double_lobe" and _quality_rank(host_quality) >= _quality_rank("medium")
    off_axis = safe_float(row.get("best_host_perp_offset_beam"), 0.0) > float(cfg["host_support"].get("geometry", {}).get("medium_max_perp_offset_beam", 2.0))
    far_midpoint = safe_float(row.get("best_host_sep_midpoint_arcsec"), 0.0) > safe_float(row.get("host_search_radius_arcsec"), np.inf)
    evidence = _parent_radio_evidence(row, cfg)
    geometry_ok = _truthy(row.get("double_lobe_geometry_pass"))
    endpoints_ok = bool(
        _truthy(row.get("endpoint1_is_parent_endpoint_allowed"))
        and _truthy(row.get("endpoint2_is_parent_endpoint_allowed"))
        and not _truthy(row.get("endpoint1_hard_point_source_veto"))
        and not _truthy(row.get("endpoint2_hard_point_source_veto"))
        and not _truthy(row.get("endpoint1_hard_compact_veto"))
        and not _truthy(row.get("endpoint2_hard_compact_veto"))
        and not _truthy(row.get("endpoint1_noise_artifact_veto"))
        and not _truthy(row.get("endpoint2_noise_artifact_veto"))
        and not _truthy(row.get("endpoint1_veto_final"))
        and not _truthy(row.get("endpoint2_veto_final"))
    )
    artifact = safe_float(row.get("artifact_environment_score_pair"), 0.0) >= float(cfg.get("artifact_veto_score", 1.2))

    host_support = "midpoint_host" in evidence
    host_at_lobe_peak = bool(cfg.get("use_lobe_peak_host_contradiction", True) and lobe_peak and host_evidence != "supports_double_lobe")
    no_host = host_evidence in {"needs_host_check", "not_checked"} or str(row.get("host_status", "")) in {"no_plausible_host", "not_queried"}
    host_ambiguous = host_evidence in {"mixed_midpoint_and_lobe_peak_hosts"} or str(row.get("parent_candidate_quality", "")) in {"needs_host_check", "suspicious"}

    acceptance = "rejected_parent_candidate"
    reason = str(row.get("rejection_reason", "")) or "not_evaluated"
    if not geometry_ok:
        reason = reason or "parent_geometry_failed"
    elif not endpoints_ok:
        reason = reason or "parent_endpoint_not_eligible"
    elif artifact:
        reason = "artifact_contradiction"
    elif host_at_lobe_peak:
        reason = "host_at_lobe_peak"
    elif evidence:
        acceptance = "accepted_high_confidence_parent"
        reason = "+".join(evidence)
    else:
        acceptance = "geometry_only_visual_candidate"
        reason = "geometry_only_no_independent_parent_evidence"

    return {
        "host_support": bool(host_support),
        "host_ambiguous": bool(host_ambiguous),
        "host_at_lobe_peak": bool(host_at_lobe_peak),
        "host_off_axis": bool(off_axis),
        "host_far_from_midpoint": bool(far_midpoint),
        "multiple_host_candidates": False,
        "no_host_detected": bool(no_host),
        "independent_parent_evidence": ",".join(evidence),
        "parent_acceptance_class": acceptance,
        "parent_acceptance_reason": reason,
        "conflict_resolution_status": "unresolved",
        "conflict_rejection_reason": "",
    }


def resolve_parent_conflicts(edges: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Ensure each local group appears in at most one accepted parent."""

    # 同一个 local group 可能参与多个 parent 候选；冲突时保留质量和分数最高的一组组合。
    if edges.empty or "parent_acceptance_class" not in edges:
        return edges
    work = edges.copy()
    accepted = work["parent_acceptance_class"].astype(str) == "accepted_high_confidence_parent"
    if not accepted.any():
        work["conflict_resolution_status"] = np.where(accepted, "kept", work.get("conflict_resolution_status", "not_applicable"))
        return work

    def rank(row: pd.Series) -> tuple[int, float, float, float]:
        evidence = set(str(row.get("independent_parent_evidence", "")).split(","))
        evidence.discard("")
        evidence_rank = 0
        if {"midpoint_host", "residual_radio_bridge"}.issubset(evidence):
            evidence_rank = 4
        elif "midpoint_host" in evidence:
            evidence_rank = 3
        elif "residual_radio_bridge" in evidence:
            evidence_rank = 2
        elif "common_diffuse_envelope" in evidence:
            evidence_rank = 1
        return (
            evidence_rank,
            safe_float(row.get("parent_score_final"), safe_float(row.get("lobe_pair_score"), 0.0)),
            safe_float(row.get("symmetry_score"), 0.0),
            -safe_float(row.get("box_gap_beam_robust"), 999.0),
        )

    accepted_rows = work.loc[accepted].copy()
    accepted_rows["_rank_tuple"] = accepted_rows.apply(rank, axis=1)
    accepted_rows = accepted_rows.sort_values("_rank_tuple", ascending=False)
    used: set[str] = set()
    keep: set[int] = set()
    for idx, row in accepted_rows.iterrows():
        left = str(row.get("local_group_id_1"))
        right = str(row.get("local_group_id_2"))
        if left in used or right in used:
            continue
        keep.add(idx)
        used.add(left)
        used.add(right)
    for idx in accepted_rows.index:
        if idx in keep:
            work.at[idx, "conflict_resolution_status"] = "kept"
        else:
            work.at[idx, "parent_acceptance_class"] = "rejected_parent_candidate"
            work.at[idx, "parent_candidate_quality"] = "rejected"
            work.at[idx, "needs_visual_check"] = False
            work.at[idx, "conflict_resolution_status"] = "removed"
            work.at[idx, "conflict_rejection_reason"] = "parent_conflict_lower_independent_evidence"
            current = str(work.at[idx, "rejection_reason"] or "")
            work.at[idx, "rejection_reason"] = (
                "parent_conflict_lower_independent_evidence"
                if not current or current == "nan"
                else f"{current};parent_conflict_lower_independent_evidence"
            )
    not_accepted = work["parent_acceptance_class"].astype(str) != "accepted_high_confidence_parent"
    work.loc[not_accepted & work["conflict_resolution_status"].astype(str).eq("unresolved"), "conflict_resolution_status"] = "not_applicable"
    return work


def run_parent_links(
    cutout_id: str,
    local_groups: pd.DataFrame,
    local_components: pd.DataFrame,
    segmentation: Any,
    config: dict[str, Any],
    host_client: HostQueryClient,
    max_host_queries_state: dict[str, int],
    max_host_queries: int | None = None,
) -> ParentLinkResult:
    # parent-link 主流程：端点筛选 -> 双瓣候选生成 -> host 查询/评分 -> 接受规则和冲突裁决。
    cfg = parent_link_config(config)
    cfg["use_midpoint_host_support"] = ablation_enabled(config, "use_midpoint_host_support")
    cfg["use_lobe_peak_host_contradiction"] = ablation_enabled(config, "use_lobe_peak_host_contradiction")
    cfg["use_artifact_penalties_layer2"] = ablation_enabled(config, "use_artifact_penalties_layer2")
    host_cfg = cfg["host_support"]
    groups = local_groups.copy().reset_index(drop=True)
    if "pixel_scale_arcsec" not in groups:
        if local_components is not None and not local_components.empty and "pixel_scale_arcsec" in local_components:
            groups["pixel_scale_arcsec"] = safe_float(local_components["pixel_scale_arcsec"].iloc[0], 1.5)
        else:
            groups["pixel_scale_arcsec"] = 1.5
    morph = build_source_morph_table(cutout_id, groups, local_components, segmentation, config)
    edges = _make_pair_edges(cutout_id, groups, morph, segmentation, config)
    group_lookup = _group_by_id(groups)
    beam_arcsec = compute_beam_size_arcsec(config)
    query_logs: list[pd.DataFrame] = []
    host_frames: list[pd.DataFrame] = []
    needs_records: list[dict[str, Any]] = []
    if not edges.empty:
        pass_edges = edges[edges["double_lobe_geometry_pass"].astype(bool)].copy()
        for idx, edge in pass_edges.iterrows():
            parent = edge.copy()
            midpoint_ra, midpoint_dec = _midpoint_ra_dec(parent, group_lookup)
            parent["midpoint_ra"] = midpoint_ra
            parent["midpoint_dec"] = midpoint_dec
            parent["host_search_radius_arcsec"] = _host_search_radius(parent, host_cfg)
            raw_mid, log_mid, query_status, query_failed = _query_hosts_for_pair(parent, host_client, host_cfg, max_host_queries_state, max_host_queries)
            if not log_mid.empty:
                query_logs.append(log_mid)
            scored_mid = score_host_candidates(parent, raw_mid, group_lookup, beam_arcsec, host_cfg)
            if not scored_mid.empty:
                scored_mid["host_role"] = "midpoint"
                scored_mid = scored_mid.rename(columns={"host_catalog": "host_catalog"})
                host_frames.append(_with_columns(scored_mid, PARENT_HOST_CANDIDATE_COLUMNS))
                best = scored_mid.iloc[0]
                host_quality = str(best.get("host_quality", "none"))
            else:
                best = pd.Series(dtype=object)
                host_quality = "none"
            left_id = str(parent.get("local_group_id_1"))
            right_id = str(parent.get("local_group_id_2"))
            g1 = group_lookup.get(left_id, pd.Series(dtype=object))
            g2 = group_lookup.get(right_id, pd.Series(dtype=object))
            peak_radius = min(max(float(cfg.get("lobe_peak_host_radius_arcsec_min", 5.0)), beam_arcsec), float(cfg.get("lobe_peak_host_radius_arcsec_max", 10.0)))
            lobe_peak_results: dict[str, Any] = {}
            for role, group in [("lobe1_peak", g1), ("lobe2_peak", g2)]:
                pra, pdec = _peak_coord_for_group(group, local_components)
                raw_peak, log_peak, _peak_status, peak_failed = _query_position_hosts(pra, pdec, peak_radius, host_client, host_cfg, max_host_queries_state, max_host_queries)
                if not log_peak.empty:
                    query_logs.append(log_peak)
                scored_peak = _score_lobe_peak_hosts(raw_peak, pra, pdec, peak_radius, cutout_id, str(parent.get("parent_candidate_id")), role)
                if not scored_peak.empty:
                    host_frames.append(scored_peak)
                    found = bool(scored_peak["host_quality"].astype(str).isin(["high", "medium"]).any())
                    score = float(pd.to_numeric(scored_peak["host_score"], errors="coerce").max())
                else:
                    found = False
                    score = 0.0
                lobe_peak_results[role] = {"found": found, "score": score, "ra": pra, "dec": pdec, "failed": peak_failed}
            lobe1_found = bool(lobe_peak_results["lobe1_peak"]["found"])
            lobe2_found = bool(lobe_peak_results["lobe2_peak"]["found"])
            lobe_peak_found = lobe1_found or lobe2_found
            midpoint_support = bool(cfg.get("use_midpoint_host_support", True)) and _quality_rank(host_quality) >= _quality_rank("medium")
            lobe_peak_contradiction_enabled = bool(cfg.get("use_lobe_peak_host_contradiction", True))
            rejection = ""
            needs = True
            if lobe_peak_contradiction_enabled and lobe1_found and lobe2_found:
                host_evidence = "likely_independent_sources"
                quality = "rejected"
                rejection = "likely_independent_sources"
            elif lobe_peak_contradiction_enabled and lobe_peak_found and not midpoint_support:
                host_evidence = "contradicts_double_lobe"
                quality = "rejected"
                rejection = "lobe_peak_host_contradiction"
            elif midpoint_support and not lobe_peak_found:
                host_evidence = "supports_double_lobe"
                artifact_pair = safe_float(parent.get("artifact_environment_score_pair"), 0.0)
                if artifact_pair >= float(cfg.get("artifact_suspicious_score", 0.8)):
                    quality = "suspicious"
                else:
                    quality = "high" if host_quality == "high" and safe_float(parent.get("symmetry_score"), 0.0) >= 0.70 else "medium"
            elif lobe_peak_contradiction_enabled and midpoint_support and lobe_peak_found:
                host_evidence = "mixed_midpoint_and_lobe_peak_hosts"
                quality = "suspicious"
            else:
                host_evidence = "needs_host_check"
                quality = "needs_host_check" if safe_float(parent.get("symmetry_score"), 0.0) >= 0.68 else "suspicious"
                rejection = "no_midpoint_host" if not query_failed else "host_query_failed"
            suspicious_reason = ""
            if bool(parent.get("near_boundary_rescue_applied", False)):
                if quality == "high":
                    quality = "medium"
                if quality == "suspicious":
                    suspicious_reason = "near_boundary_rescue_low_confidence"
            host_score = safe_float(best.get("host_score"), 0.0)
            final_score = safe_float(parent.get("lobe_pair_score"), 0.0) + float(host_cfg.get("host_score_weight", 1.0)) * host_score
            updates = {
                "best_host_score": host_score,
                "parent_score_final": final_score,
                "host_evidence": host_evidence,
                "parent_candidate_quality": quality,
                "host_status": "host_found" if midpoint_support else "host_query_failed" if query_failed else "no_plausible_host",
                "best_host_catalog": best.get("host_catalog", ""),
                "best_host_id": best.get("host_id", ""),
                "best_host_ra": best.get("host_ra", np.nan),
                "best_host_dec": best.get("host_dec", np.nan),
                "best_host_sep_midpoint_arcsec": best.get("host_sep_midpoint_arcsec", np.nan),
                "best_host_perp_offset_beam": best.get("host_perp_offset_beam", np.nan),
                "best_host_fractional_position": best.get("host_fractional_position", np.nan),
                "best_host_W1": best.get("W1", np.nan),
                "best_host_W2": best.get("W2", np.nan),
                "best_host_W1_W2": best.get("W1_W2", np.nan),
                "host_quality": host_quality,
                "lobe_peak_host_found": lobe_peak_found,
                "lobe1_peak_host_found": lobe1_found,
                "lobe2_peak_host_found": lobe2_found,
                "lobe1_peak_host_score": lobe_peak_results["lobe1_peak"]["score"],
                "lobe2_peak_host_score": lobe_peak_results["lobe2_peak"]["score"],
                "suspicious_reason": suspicious_reason,
                "rejection_reason": rejection,
                "needs_visual_check": bool(needs and quality != "rejected"),
                "host_search_radius_arcsec": parent.get("host_search_radius_arcsec"),
                "midpoint_ra": midpoint_ra,
                "midpoint_dec": midpoint_dec,
                "query_status": query_status,
                "host_query_failed": bool(query_failed),
                "lobe1_peak_ra": lobe_peak_results["lobe1_peak"]["ra"],
                "lobe1_peak_dec": lobe_peak_results["lobe1_peak"]["dec"],
                "lobe2_peak_ra": lobe_peak_results["lobe2_peak"]["ra"],
                "lobe2_peak_dec": lobe_peak_results["lobe2_peak"]["dec"],
            }
            for key, value in updates.items():
                edges.loc[idx, key] = value
            if quality != "rejected":
                needs_records.append(
                    {
                        "cutout_id": cutout_id,
                        "record_type": "parent_link",
                        "object_id": parent.get("parent_candidate_id"),
                        "reason": f"{quality}:{host_evidence}",
                        "priority": quality,
                        "details": json_dumps_safe({**parent.to_dict(), **updates}),
                    }
                )
    if not edges.empty:
        for idx, row in edges.iterrows():
            updates = classify_parent_acceptance(row, config)
            for key, value in updates.items():
                edges.loc[idx, key] = value
            if updates["parent_acceptance_class"] == "accepted_high_confidence_parent":
                if str(edges.loc[idx, "parent_candidate_quality"]) not in {"high", "medium"}:
                    edges.loc[idx, "parent_candidate_quality"] = "medium"
                edges.loc[idx, "rejection_reason"] = ""
                edges.loc[idx, "needs_visual_check"] = False
            elif updates["parent_acceptance_class"] == "geometry_only_visual_candidate":
                edges.loc[idx, "needs_visual_check"] = True
                if str(edges.loc[idx, "parent_candidate_quality"]) in {"high", "medium"}:
                    edges.loc[idx, "parent_candidate_quality"] = "geometry_only"
            elif updates["parent_acceptance_class"] == "rejected_parent_candidate":
                edges.loc[idx, "parent_candidate_quality"] = "rejected"
                edges.loc[idx, "needs_visual_check"] = False
                if not str(edges.loc[idx, "rejection_reason"]).strip():
                    edges.loc[idx, "rejection_reason"] = updates["parent_acceptance_reason"]
        edges = resolve_parent_conflicts(edges, config)
    edges = _with_columns(edges, PARENT_EDGE_DEBUG_COLUMNS)
    candidate_mask = edges["parent_acceptance_class"].astype(str).eq("accepted_high_confidence_parent") if not edges.empty else pd.Series(dtype=bool)
    candidates = _apply_limits(edges[candidate_mask].copy(), cfg) if not edges.empty else pd.DataFrame(columns=PARENT_EDGE_DEBUG_COLUMNS)
    candidates = _with_columns(candidates, PARENT_CANDIDATE_COLUMNS)
    hosts = _with_columns(pd.concat(host_frames, ignore_index=True) if host_frames else pd.DataFrame(), PARENT_HOST_CANDIDATE_COLUMNS)
    query_log = _with_columns(pd.concat(query_logs, ignore_index=True) if query_logs else pd.DataFrame(), HOST_QUERY_LOG_COLUMNS)
    needs = pd.DataFrame(needs_records, columns=["cutout_id", "record_type", "object_id", "reason", "priority", "details"])
    diagnostics = _diagnostics(cutout_id, morph, edges, query_log)
    return ParentLinkResult(
        candidates=candidates,
        edges_debug=edges,
        host_candidates=hosts,
        host_query_log=query_log,
        diagnostics=diagnostics,
        needs_visual_check=needs,
        source_morph_table=morph,
    )


def _diagnostics(cutout_id: str, morph: pd.DataFrame, edges: pd.DataFrame, query_log: pd.DataFrame) -> pd.DataFrame:
    mclass = morph.get("source_morph_class", pd.Series(dtype=str)).astype(str) if not morph.empty else pd.Series(dtype=str)
    quality = edges.get("parent_candidate_quality", pd.Series(dtype=str)).astype(str) if not edges.empty else pd.Series(dtype=str)
    reason = edges.get("rejection_reason", pd.Series(dtype=str)).astype(str) if not edges.empty else pd.Series(dtype=str)
    host_evidence = edges.get("host_evidence", pd.Series(dtype=str)).astype(str) if not edges.empty else pd.Series(dtype=str)
    acceptance = edges.get("parent_acceptance_class", pd.Series(dtype=str)).astype(str) if not edges.empty else pd.Series(dtype=str)
    conflict = edges.get("conflict_resolution_status", pd.Series(dtype=str)).astype(str) if not edges.empty else pd.Series(dtype=str)
    qcat = query_log.get("catalogue", pd.Series(dtype=str)).astype(str) if not query_log.empty else pd.Series(dtype=str)
    hard_point = morph.get("hard_point_source_veto", pd.Series(dtype=bool)).astype(bool) if not morph.empty else pd.Series(dtype=bool)
    hard_compact = morph.get("hard_compact_veto", pd.Series(dtype=bool)).astype(bool) if not morph.empty else pd.Series(dtype=bool)
    noise = morph.get("noise_artifact_veto", pd.Series(dtype=bool)).astype(bool) if not morph.empty else pd.Series(dtype=bool)
    isolated = morph.get("isolated_compact_veto", pd.Series(dtype=bool)).astype(bool) if not morph.empty else pd.Series(dtype=bool)
    near_lobe = morph.get("near_extended_lobe_candidate", pd.Series(dtype=bool)).astype(bool) if not morph.empty else pd.Series(dtype=bool)
    if not edges.empty:
        geometry_pass = edges.get("double_lobe_geometry_pass", pd.Series(False, index=edges.index)).astype(bool)
        final_mask = quality.isin(["high", "medium", "needs_host_check", "suspicious"])
        edge_point = edges.get("endpoint1_hard_point_source_veto", pd.Series(False, index=edges.index)).astype(bool) | edges.get("endpoint2_hard_point_source_veto", pd.Series(False, index=edges.index)).astype(bool)
        edge_compact = edges.get("endpoint1_hard_compact_veto", pd.Series(False, index=edges.index)).astype(bool) | edges.get("endpoint2_hard_compact_veto", pd.Series(False, index=edges.index)).astype(bool)
        edge_noise = edges.get("endpoint1_noise_artifact_veto", pd.Series(False, index=edges.index)).astype(bool) | edges.get("endpoint2_noise_artifact_veto", pd.Series(False, index=edges.index)).astype(bool)
        edge_veto = edges.get("endpoint1_veto_final", pd.Series(False, index=edges.index)).astype(bool) | edges.get("endpoint2_veto_final", pd.Series(False, index=edges.index)).astype(bool)
        near_boundary = edges.get("near_boundary_pair", pd.Series(False, index=edges.index)).astype(bool)
        rescue_applied = edges.get("near_boundary_rescue_applied", pd.Series(False, index=edges.index)).astype(bool)
    else:
        geometry_pass = pd.Series(dtype=bool)
        final_mask = pd.Series(dtype=bool)
        edge_point = pd.Series(dtype=bool)
        edge_compact = pd.Series(dtype=bool)
        edge_noise = pd.Series(dtype=bool)
        edge_veto = pd.Series(dtype=bool)
        near_boundary = pd.Series(dtype=bool)
        rescue_applied = pd.Series(dtype=bool)
    return _with_columns(
        pd.DataFrame(
            [
                {
                    "cutout_id": cutout_id,
                    "n_total_local_groups": int(len(morph)),
                    "n_hard_point_source_veto": int(hard_point.sum()) if not morph.empty else 0,
                    "n_hard_compact_veto": int(hard_compact.sum()) if not morph.empty else 0,
                    "n_noise_artifact_veto": int(noise.sum()) if not morph.empty else 0,
                    "n_isolated_compact_veto": int(isolated.sum()) if not morph.empty else 0,
                    "n_point_like_hidden": int(mclass.isin(["point_like", "point_like_or_compact", "noise_or_artifact"]).sum()),
                    "n_resolved_single": int((mclass == "resolved_single").sum()),
                    "n_lobe_candidate": int((mclass == "lobe_candidate").sum()),
                    "n_near_extended_lobe_candidate": int(near_lobe.sum()) if not morph.empty else 0,
                    "n_artifact_risk": int(mclass.isin(["artifact_risk", "noise_or_artifact"]).sum()),
                    "n_parent_pairs_considered": int(len(edges)),
                    "n_near_boundary_pair": int(near_boundary.sum()) if not edges.empty else 0,
                    "n_near_boundary_rescue_applied": int(rescue_applied.sum()) if not edges.empty else 0,
                    "n_rejected_near_boundary_no_support": int((reason == "near_boundary_but_no_extended_or_bridge_support").sum()),
                    "n_double_lobe_geometry_pass": int(geometry_pass.sum()) if not edges.empty else 0,
                    "n_lobe_peak_host_contradiction": int((reason == "lobe_peak_host_contradiction").sum()),
                    "n_midpoint_host_supports": int((host_evidence == "supports_double_lobe").sum()),
                    "n_host_queries": int(len(query_log)),
                    "n_catwise_queries": int((qcat == "catwise2020").sum()),
                    "n_allwise_fallback_queries": int((qcat == "allwise").sum()),
                    "n_final_candidates": int((acceptance == "accepted_high_confidence_parent").sum()),
                    "n_accepted_high_confidence_parent": int((acceptance == "accepted_high_confidence_parent").sum()),
                    "n_geometry_only_visual_candidate": int((acceptance == "geometry_only_visual_candidate").sum()),
                    "n_rejected_parent_candidate": int((acceptance == "rejected_parent_candidate").sum()),
                    "n_parent_conflict_removed": int((conflict == "removed").sum()),
                    "n_parent_high": int((quality == "high").sum()),
                    "n_parent_medium": int((quality == "medium").sum()),
                    "n_parent_needs_host_check": int((quality == "needs_host_check").sum()),
                    "n_parent_suspicious": int((quality == "suspicious").sum()),
                    "n_parent_union_boxes": int((quality.isin(["high", "medium", "needs_host_check", "suspicious"]) & pd.to_numeric(edges.get("parent_LAS_beam", pd.Series(dtype=float)), errors="coerce").notna()).sum()) if not edges.empty else 0,
                    "n_point_source_endpoint_rejected": int(((reason == "endpoint_vetoed_or_not_extended_for_near_boundary_rescue") & edge_point).sum()) if not edges.empty else 0,
                    "n_compact_endpoint_rejected": int(((reason == "endpoint_vetoed_or_not_extended_for_near_boundary_rescue") & edge_compact).sum()) if not edges.empty else 0,
                    "n_noise_artifact_endpoint_rejected": int(((reason == "endpoint_vetoed_or_not_extended_for_near_boundary_rescue") & edge_noise).sum()) if not edges.empty else 0,
                    "n_endpoint_veto_final_rejected": int(((reason == "endpoint_vetoed_or_not_extended_for_near_boundary_rescue") & edge_veto).sum()) if not edges.empty else 0,
                    "n_geometry_pass_point_source_endpoint": int((geometry_pass & edge_point).sum()) if not edges.empty else 0,
                    "n_geometry_pass_compact_endpoint": int((geometry_pass & edge_compact).sum()) if not edges.empty else 0,
                    "n_geometry_pass_noise_artifact_endpoint": int((geometry_pass & edge_noise).sum()) if not edges.empty else 0,
                    "n_geometry_pass_endpoint_veto_final": int((geometry_pass & edge_veto).sum()) if not edges.empty else 0,
                    "n_parent_candidate_point_source_endpoint": int((final_mask & edge_point).sum()) if not edges.empty else 0,
                    "n_parent_candidate_compact_endpoint": int((final_mask & edge_compact).sum()) if not edges.empty else 0,
                    "n_parent_candidate_noise_artifact_endpoint": int((final_mask & edge_noise).sum()) if not edges.empty else 0,
                    "n_parent_candidate_endpoint_veto_final": int((final_mask & edge_veto).sum()) if not edges.empty else 0,
                    "n_rejected_point_like_endpoint": int((reason == "point_like_endpoint").sum()),
                    "n_rejected_artifact_veto": int((reason == "artifact_veto").sum()),
                    "n_rejected_not_symmetric_lobe_pair": int((reason == "not_symmetric_lobe_pair").sum()),
                    "n_rejected_lobe_peak_host_contradiction": int((reason == "lobe_peak_host_contradiction").sum()),
                }
            ]
        ),
        PARENT_DIAGNOSTIC_COLUMNS,
    )
