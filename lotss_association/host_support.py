"""Host-catalogue support for parent-link candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u

from .host_query import HOST_QUERY_LOG_COLUMNS, HOST_RAW_COLUMNS, HostQueryClient
from .parent_seed import (
    PARENT_SEED_COLUMNS,
    PARENT_SEED_EDGE_DEBUG_COLUMNS,
    build_parent_seed_table,
    run_parent_seed,
)
from .utils import json_dumps_safe, safe_float


# host 支持阶段只给 parent 候选补充宿主星系诊断信息；
# 候选本身仍来自上一层的射电几何和 parent-seed 结果。
HOST_SUPPORTED_CANDIDATE_COLUMNS = [
    "cutout_id",
    "parent_candidate_id",
    "local_group_id_1",
    "local_group_id_2",
    "box_gap_beam_robust",
    "center_distance_arcsec",
    "center_distance_beam",
    "parent_score_geometry",
    "best_host_score",
    "parent_score_final",
    "parent_candidate_quality",
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
    "rejection_reason",
    "needs_visual_check",
]

HOST_SUPPORTED_EDGE_DEBUG_COLUMNS = [
    *HOST_SUPPORTED_CANDIDATE_COLUMNS,
    "parent_score_geometry_quality",
    "host_search_radius_arcsec",
    "midpoint_ra",
    "midpoint_dec",
    "query_status",
    "host_query_failed",
    "geometry_rejection_reason",
    "debug_info",
]

HOST_CANDIDATE_COLUMNS = [
    "cutout_id",
    "parent_candidate_id",
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
]

HOST_SUPPORT_DIAGNOSTIC_COLUMNS = [
    "cutout_id",
    "n_local_groups",
    "n_parent_seed_groups",
    "n_initial_parent_pairs",
    "n_geometry_pass_pairs",
    "n_host_queries",
    "n_catwise_queries",
    "n_allwise_fallback_queries",
    "n_host_found_pairs",
    "n_parent_candidates",
    "n_parent_high",
    "n_parent_medium",
    "n_rejected_no_plausible_midpoint_host",
    "n_host_query_failed",
    "n_rejected_geometry",
]


@dataclass
class HostSupportResult:
    candidates: pd.DataFrame
    edges_debug: pd.DataFrame
    host_candidates: pd.DataFrame
    host_query_log: pd.DataFrame
    diagnostics: pd.DataFrame
    needs_visual_check: pd.DataFrame
    parent_seed_table: pd.DataFrame


def host_support_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "enabled": True,
        "require_host_for_parent_link": True,
        "catalog_priority": ["catwise2020", "allwise"],
        "min_search_radius_arcsec": 10.0,
        "max_search_radius_arcsec": 30.0,
        "search_radius_fraction_of_sep": 0.12,
        "max_host_results_per_query": 20,
        "host_score_weight": 1.0,
        "min_host_quality_for_default_candidate": "medium",
        "host_quality_thresholds": {"high": 3.0, "medium": 2.0},
        "geometry": {
            "high_max_perp_offset_beam": 1.5,
            "medium_max_perp_offset_beam": 2.0,
            "high_fractional_position_min": 0.35,
            "high_fractional_position_max": 0.65,
            "medium_fractional_position_min": 0.25,
            "medium_fractional_position_max": 0.75,
        },
        "wise_color": {
            "agn_bonus_w1_w2_min": 0.8,
            "agn_bonus": 0.5,
            "require_agn_color": False,
        },
    }
    raw = (config.get("host_support", {}) or {}).copy()
    out = dict(defaults)
    out.update({key: value for key, value in raw.items() if key not in {"host_quality_thresholds", "geometry", "wise_color"}})
    out["host_quality_thresholds"] = dict(defaults["host_quality_thresholds"])
    out["host_quality_thresholds"].update(raw.get("host_quality_thresholds", {}) or {})
    out["geometry"] = dict(defaults["geometry"])
    out["geometry"].update(raw.get("geometry", {}) or {})
    out["wise_color"] = dict(defaults["wise_color"])
    out["wise_color"].update(raw.get("wise_color", {}) or {})
    return out


def _with_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out:
            out[col] = pd.Series(dtype=object)
    if out.empty:
        return pd.DataFrame(columns=columns)
    return out[columns]


def _quality_rank(value: str) -> int:
    return {"none": 0, "low": 1, "medium": 2, "high": 3}.get(str(value), 0)


def _midpoint_ra_dec(row: pd.Series, group_by_id: dict[str, pd.Series]) -> tuple[float, float]:
    g1 = group_by_id.get(str(row.get("local_group_id_1")))
    g2 = group_by_id.get(str(row.get("local_group_id_2")))
    if g1 is None or g2 is None:
        return float("nan"), float("nan")
    ra1, dec1 = safe_float(g1.get("ra")), safe_float(g1.get("dec"))
    ra2, dec2 = safe_float(g2.get("ra")), safe_float(g2.get("dec"))
    if not np.all(np.isfinite([ra1, dec1, ra2, dec2])):
        return float("nan"), float("nan")
    c1 = SkyCoord(ra1 * u.deg, dec1 * u.deg, frame="icrs")
    c2 = SkyCoord(ra2 * u.deg, dec2 * u.deg, frame="icrs")
    sep = c1.separation(c2)
    pa = c1.position_angle(c2)
    mid = c1.directional_offset_by(pa, sep / 2.0)
    return float(mid.ra.deg), float(mid.dec.deg)


def _host_search_radius(row: pd.Series, cfg: dict[str, Any]) -> float:
    sep = safe_float(row.get("center_distance_arcsec"), 0.0)
    radius = float(cfg.get("search_radius_fraction_of_sep", 0.12)) * sep
    radius = max(float(cfg.get("min_search_radius_arcsec", 10.0)), radius)
    radius = min(float(cfg.get("max_search_radius_arcsec", 30.0)), radius)
    return float(radius)


def _axis_geometry(
    host_ra: float,
    host_dec: float,
    row: pd.Series,
    group_by_id: dict[str, pd.Series],
    beam_arcsec: float,
) -> tuple[float, float, float]:
    g1 = group_by_id.get(str(row.get("local_group_id_1")))
    g2 = group_by_id.get(str(row.get("local_group_id_2")))
    if g1 is None or g2 is None:
        return float("nan"), float("nan"), float("nan")
    c1 = SkyCoord(safe_float(g1.get("ra")) * u.deg, safe_float(g1.get("dec")) * u.deg, frame="icrs")
    c2 = SkyCoord(safe_float(g2.get("ra")) * u.deg, safe_float(g2.get("dec")) * u.deg, frame="icrs")
    ch = SkyCoord(float(host_ra) * u.deg, float(host_dec) * u.deg, frame="icrs")
    sep12 = c1.separation(c2).arcsec
    if sep12 <= 0:
        return float("nan"), float("nan"), float("nan")
    sep1h = c1.separation(ch).arcsec
    pa12 = c1.position_angle(c2).rad
    pa1h = c1.position_angle(ch).rad
    along = sep1h * np.cos(pa1h - pa12)
    perp = abs(sep1h * np.sin(pa1h - pa12))
    frac = along / sep12
    return float(perp), float(perp / max(beam_arcsec, 1e-6)), float(frac)


def score_host_candidates(
    parent_row: pd.Series,
    raw_hosts: pd.DataFrame,
    group_by_id: dict[str, pd.Series],
    beam_arcsec: float,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    if raw_hosts is None or raw_hosts.empty:
        return pd.DataFrame(columns=HOST_CANDIDATE_COLUMNS)
    midpoint_ra = safe_float(parent_row.get("midpoint_ra"))
    midpoint_dec = safe_float(parent_row.get("midpoint_dec"))
    if not np.all(np.isfinite([midpoint_ra, midpoint_dec])):
        return pd.DataFrame(columns=HOST_CANDIDATE_COLUMNS)
    midpoint = SkyCoord(midpoint_ra * u.deg, midpoint_dec * u.deg, frame="icrs")
    radius = safe_float(parent_row.get("host_search_radius_arcsec"), 10.0)
    geom = cfg["geometry"]
    wise_cfg = cfg["wise_color"]
    thresholds = cfg["host_quality_thresholds"]
    records: list[dict[str, Any]] = []
    for _, host in raw_hosts.iterrows():
        # 对每个候选宿主同时记录“离中点多近”和“是否沿双瓣轴线”；
        # 后续人工复核可据此区分真正的中点宿主和偶然落在搜索半径内的红外源。
        host_ra = safe_float(host.get("host_ra"), float("nan"))
        host_dec = safe_float(host.get("host_dec"), float("nan"))
        if not np.all(np.isfinite([host_ra, host_dec])):
            continue
        hc = SkyCoord(host_ra * u.deg, host_dec * u.deg, frame="icrs")
        sep_mid = float(midpoint.separation(hc).arcsec)
        perp_arcsec, perp_beam, frac = _axis_geometry(host_ra, host_dec, parent_row, group_by_id, beam_arcsec)
        midpoint_closeness = float(np.clip(1.0 - sep_mid / max(radius, 1e-6), 0.0, 1.0))
        axis_consistency = float(np.clip(1.0 - perp_beam / max(float(geom.get("medium_max_perp_offset_beam", 2.0)), 1e-6), 0.0, 1.0))
        w1snr = safe_float(host.get("W1_snr"), float("nan"))
        w2snr = safe_float(host.get("W2_snr"), float("nan"))
        det_scores = []
        for value in [w1snr, w2snr]:
            if np.isfinite(value):
                det_scores.append(float(np.clip(value / 10.0, 0.0, 1.0)))
        wise_detection = float(np.mean(det_scores)) if det_scores else (0.5 if np.isfinite(safe_float(host.get("W1"), float("nan"))) else 0.0)
        w1_w2 = safe_float(host.get("W1_W2"), float("nan"))
        color_bonus = float(wise_cfg.get("agn_bonus", 0.5)) if np.isfinite(w1_w2) and w1_w2 >= float(wise_cfg.get("agn_bonus_w1_w2_min", 0.8)) else 0.0
        flags: list[str] = []
        cc_flags = str(host.get("cc_flags", "")).strip()
        artifact_penalty = 0.0
        if cc_flags and cc_flags.lower() not in {"0000", "0", "nan", "none"}:
            artifact_penalty = 1.0
            flags.append(f"cc_flags={cc_flags}")
        score = 2.0 * midpoint_closeness + 1.5 * axis_consistency + 0.8 * wise_detection + color_bonus - artifact_penalty
        if (
            score >= float(thresholds.get("high", 3.0))
            and perp_beam <= float(geom.get("high_max_perp_offset_beam", 1.5))
            and float(geom.get("high_fractional_position_min", 0.35)) <= frac <= float(geom.get("high_fractional_position_max", 0.65))
        ):
            quality = "high"
        elif (
            score >= float(thresholds.get("medium", 2.0))
            and perp_beam <= float(geom.get("medium_max_perp_offset_beam", 2.0))
            and float(geom.get("medium_fractional_position_min", 0.25)) <= frac <= float(geom.get("medium_fractional_position_max", 0.75))
        ):
            quality = "medium"
        elif sep_mid <= radius:
            quality = "low"
        else:
            quality = "none"
        records.append(
            {
                "cutout_id": parent_row.get("cutout_id"),
                "parent_candidate_id": parent_row.get("parent_candidate_id"),
                "host_catalog": host.get("catalogue", ""),
                "host_id": host.get("host_id", ""),
                "host_ra": host_ra,
                "host_dec": host_dec,
                "host_sep_midpoint_arcsec": sep_mid,
                "host_perp_offset_beam": perp_beam,
                "host_fractional_position": frac,
                "W1": safe_float(host.get("W1"), float("nan")),
                "W2": safe_float(host.get("W2"), float("nan")),
                "W1_W2": w1_w2,
                "W1_snr": w1snr,
                "W2_snr": w2snr,
                "host_score": float(score),
                "host_quality": quality,
                "host_flags": ";".join(flags),
            }
        )
    frame = _with_columns(pd.DataFrame(records), HOST_CANDIDATE_COLUMNS)
    if frame.empty:
        return frame
    return frame.sort_values(["host_score", "host_sep_midpoint_arcsec"], ascending=[False, True])


def _query_hosts_for_pair(
    parent_row: pd.Series,
    host_client: HostQueryClient,
    cfg: dict[str, Any],
    max_host_queries_state: dict[str, int],
    max_host_queries: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, str, bool]:
    logs: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    any_failed = False
    query_status = "not_queried"
    for catalogue in cfg.get("catalog_priority", ["catwise2020", "allwise"]):
        if max_host_queries is not None and max_host_queries_state["count"] >= max_host_queries:
            query_status = "max_host_queries_reached"
            break
        result = host_client.query_catalogue(
            safe_float(parent_row.get("midpoint_ra")),
            safe_float(parent_row.get("midpoint_dec")),
            safe_float(parent_row.get("host_search_radius_arcsec")),
            str(catalogue),
        )
        max_host_queries_state["count"] += 1
        logs.append(result.log)
        status = str(result.log["status"].iloc[0]) if not result.log.empty else ""
        if status == "failed":
            any_failed = True
        if not result.results.empty:
            raw_frames.append(result.results)
            query_status = f"{catalogue}_results"
            break
        query_status = status or f"{catalogue}_empty"
    raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame(columns=HOST_RAW_COLUMNS)
    log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=HOST_QUERY_LOG_COLUMNS)
    return raw, log, query_status, any_failed


def run_host_support(
    cutout_id: str,
    local_groups: pd.DataFrame,
    local_components: pd.DataFrame,
    segmentation: Any,
    config: dict[str, Any],
    host_client: HostQueryClient,
    max_host_queries_state: dict[str, int],
    max_host_queries: int | None = None,
) -> HostSupportResult:
    cfg = host_support_config(config)
    refined = run_parent_seed(cutout_id, local_groups, local_components, config, segmentation=segmentation)
    groups = local_groups.copy()
    group_by_id = {str(row.get("association_group_id")): row for _, row in groups.iterrows()}
    beam_arcsec = 6.0
    try:
        from .association import compute_beam_size_arcsec

        beam_arcsec = compute_beam_size_arcsec(config)
    except Exception:
        pass
    edges = refined.edges_debug.copy()
    host_candidate_frames: list[pd.DataFrame] = []
    query_logs: list[pd.DataFrame] = []
    out_records: list[dict[str, Any]] = []
    needs_records: list[dict[str, Any]] = []
    if edges.empty:
        diagnostics = _diagnostics(cutout_id, local_groups, refined, pd.DataFrame(), pd.DataFrame(), [])
        return HostSupportResult(
            candidates=pd.DataFrame(columns=HOST_SUPPORTED_CANDIDATE_COLUMNS),
            edges_debug=pd.DataFrame(columns=HOST_SUPPORTED_EDGE_DEBUG_COLUMNS),
            host_candidates=pd.DataFrame(columns=HOST_CANDIDATE_COLUMNS),
            host_query_log=pd.DataFrame(columns=HOST_QUERY_LOG_COLUMNS),
            diagnostics=diagnostics,
            needs_visual_check=pd.DataFrame(columns=["cutout_id", "record_type", "object_id", "reason", "priority", "details"]),
            parent_seed_table=refined.parent_seed_table,
        )
    geometry_mask = edges["rejection_reason"].fillna("").astype(str).eq("") | edges["parent_candidate_quality"].astype(str).isin(["high", "medium", "low"])
    for _, edge in edges[geometry_mask].iterrows():
        # 只对已通过几何筛选或仍有诊断价值的 parent pair 查询 host；
        # 这样可以限制外部 catalogue 查询量，并保留 query log 供复现。
        parent = edge.copy()
        midpoint_ra, midpoint_dec = _midpoint_ra_dec(parent, group_by_id)
        parent["midpoint_ra"] = midpoint_ra
        parent["midpoint_dec"] = midpoint_dec
        parent["host_search_radius_arcsec"] = _host_search_radius(parent, cfg)
        raw_hosts, log, query_status, query_failed = _query_hosts_for_pair(parent, host_client, cfg, max_host_queries_state, max_host_queries)
        if not log.empty:
            query_logs.append(log)
        scored_hosts = score_host_candidates(parent, raw_hosts, group_by_id, beam_arcsec, cfg)
        if not scored_hosts.empty:
            host_candidate_frames.append(scored_hosts)
        if not scored_hosts.empty:
            best = scored_hosts.iloc[0]
            host_quality = str(best.get("host_quality", "none"))
        else:
            best = pd.Series(dtype=object)
            host_quality = "none"
        geom_score = safe_float(parent.get("parent_score"), 0.0)
        host_score = safe_float(best.get("host_score"), 0.0)
        final_score = geom_score + float(cfg.get("host_score_weight", 1.0)) * host_score
        rejection = ""
        quality = "rejected"
        default = False
        needs = False
        if query_failed and host_quality == "none":
            rejection = "host_query_failed"
            needs = True
        elif _quality_rank(host_quality) < _quality_rank(str(cfg.get("min_host_quality_for_default_candidate", "medium"))):
            rejection = "no_plausible_midpoint_host"
            quality = "suspicious" if host_quality == "low" else "rejected"
            needs = host_quality == "low"
        else:
            quality = "high" if host_quality == "high" and final_score >= 6.0 else "medium"
            default = quality in {"high", "medium"}
            needs = default
        record = {
            "cutout_id": cutout_id,
            "parent_candidate_id": parent.get("parent_candidate_id"),
            "local_group_id_1": parent.get("local_group_id_1"),
            "local_group_id_2": parent.get("local_group_id_2"),
            "box_gap_beam_robust": parent.get("box_gap_beam_robust"),
            "center_distance_arcsec": parent.get("center_distance_arcsec"),
            "center_distance_beam": parent.get("center_distance_beam"),
            "parent_score_geometry": geom_score,
            "best_host_score": host_score,
            "parent_score_final": final_score,
            "parent_candidate_quality": quality,
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
            "rejection_reason": rejection,
            "needs_visual_check": bool(needs),
            "parent_score_geometry_quality": parent.get("parent_candidate_quality", ""),
            "host_search_radius_arcsec": parent.get("host_search_radius_arcsec"),
            "midpoint_ra": midpoint_ra,
            "midpoint_dec": midpoint_dec,
            "query_status": query_status,
            "host_query_failed": bool(query_failed),
            "geometry_rejection_reason": parent.get("rejection_reason", ""),
            "debug_info": json_dumps_safe({"require_host_for_parent_link": cfg.get("require_host_for_parent_link", True)}),
        }
        out_records.append(record)
        if needs:
            needs_records.append(
                {
                    "cutout_id": cutout_id,
                    "record_type": "host_supported_parent_link",
                    "object_id": parent.get("parent_candidate_id"),
                    "reason": "host-gated parent candidate" if default else rejection,
                    "priority": quality if quality in {"high", "medium"} else "low",
                    "details": json_dumps_safe(record),
                }
            )
    edges_out = _with_columns(pd.DataFrame(out_records), HOST_SUPPORTED_EDGE_DEBUG_COLUMNS)
    candidates = edges_out[edges_out["parent_candidate_quality"].astype(str).isin(["high", "medium"])].copy() if not edges_out.empty else pd.DataFrame(columns=HOST_SUPPORTED_CANDIDATE_COLUMNS)
    candidates = _with_columns(candidates, HOST_SUPPORTED_CANDIDATE_COLUMNS)
    host_candidates = _with_columns(pd.concat(host_candidate_frames, ignore_index=True) if host_candidate_frames else pd.DataFrame(), HOST_CANDIDATE_COLUMNS)
    query_log = _with_columns(pd.concat(query_logs, ignore_index=True) if query_logs else pd.DataFrame(), HOST_QUERY_LOG_COLUMNS)
    needs = pd.DataFrame(needs_records, columns=["cutout_id", "record_type", "object_id", "reason", "priority", "details"])
    diagnostics = _diagnostics(cutout_id, local_groups, refined, edges_out, query_log, out_records)
    return HostSupportResult(
        candidates=candidates,
        edges_debug=edges_out,
        host_candidates=host_candidates,
        host_query_log=query_log,
        diagnostics=diagnostics,
        needs_visual_check=needs,
        parent_seed_table=refined.parent_seed_table,
    )


def _diagnostics(
    cutout_id: str,
    local_groups: pd.DataFrame,
    refined: Any,
    edges_out: pd.DataFrame,
    query_log: pd.DataFrame,
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    quality = edges_out.get("parent_candidate_quality", pd.Series(dtype=str)).astype(str) if not edges_out.empty else pd.Series(dtype=str)
    reason = edges_out.get("rejection_reason", pd.Series(dtype=str)).astype(str) if not edges_out.empty else pd.Series(dtype=str)
    parent_seed = 0
    if refined.parent_seed_table is not None and not refined.parent_seed_table.empty:
        parent_seed = int(refined.parent_seed_table["is_parent_seed"].astype(bool).sum())
    qcat = query_log.get("catalogue", pd.Series(dtype=str)).astype(str) if not query_log.empty else pd.Series(dtype=str)
    qstatus = query_log.get("status", pd.Series(dtype=str)).astype(str) if not query_log.empty else pd.Series(dtype=str)
    return _with_columns(
        pd.DataFrame(
            [
                {
                    "cutout_id": cutout_id,
                    "n_local_groups": int(len(local_groups)),
                    "n_parent_seed_groups": parent_seed,
                    "n_initial_parent_pairs": int(len(refined.edges_debug)),
                    "n_geometry_pass_pairs": int(len(edges_out)),
                    "n_host_queries": int(len(query_log)),
                    "n_catwise_queries": int((qcat == "catwise2020").sum()),
                    "n_allwise_fallback_queries": int((qcat == "allwise").sum()),
                    "n_host_found_pairs": int((edges_out.get("host_quality", pd.Series(dtype=str)).astype(str).isin(["high", "medium", "low"])).sum()) if not edges_out.empty else 0,
                    "n_parent_candidates": int((quality.isin(["high", "medium"])).sum()),
                    "n_parent_high": int((quality == "high").sum()),
                    "n_parent_medium": int((quality == "medium").sum()),
                    "n_rejected_no_plausible_midpoint_host": int((reason == "no_plausible_midpoint_host").sum()),
                    "n_host_query_failed": int((qstatus == "failed").sum() + (reason == "host_query_failed").sum()),
                    "n_rejected_geometry": int(len(refined.edges_debug) - len(edges_out)),
                }
            ]
        ),
        HOST_SUPPORT_DIAGNOSTIC_COLUMNS,
    )
