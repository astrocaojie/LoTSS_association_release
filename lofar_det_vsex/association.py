"""Beam-aware Gaussian component association for radio structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .ablation_config import ablation_enabled
from .beam import (
    angle_delta_180,
    beam_area_arcsec2,
    beam_axis_components,
    beam_axes_from_config,
    beam_covariance_from_config,
    direction_angle_from_delta,
    elliptical_beam_distance,
    projected_beam_distance_from_delta,
    projected_beam_fwhm,
)
from .morphology import (
    add_morphology_columns,
    classify_gaussian_component,
    effective_component_pa_pixel,
    effective_pa_weight,
    intrinsic_axes_from_component,
    is_unresolved_or_beam_like,
)
from .segmentation import component_support_mask, threshold_key
from .utils import json_dumps_safe, safe_float


@dataclass
class AssociationResult:
    """Container returned by the association pipeline."""

    graph: nx.Graph
    edges: pd.DataFrame
    components: pd.DataFrame
    groups: pd.DataFrame
    clusters: list[list[int]]


# 边表字段按“候选生成 -> 形态证据 -> 连通/桥接证据 -> 惩罚项 -> 决策”排列，
# 便于诊断每一对 Gaussian 是否应归入同一射电结构。
EDGE_COLUMNS = [
    "cutout_id",
    "gaussian_id_1",
    "gaussian_id_2",
    "component_index_1",
    "component_index_2",
    "pair_separation_arcsec",
    "distance_arcsec",
    "pair_angle_deg",
    "distance_beam",
    "elliptical_beam_distance",
    "directional_beam_arcsec",
    "beam_parallel_separation_arcsec",
    "beam_perpendicular_separation_arcsec",
    "beam_parallel_arcsec",
    "beam_perpendicular_arcsec",
    "pair_angle_relative_to_bpa_deg",
    "pair_angle_relative_bpa_deg",
    "candidate_generation_reason",
    "morphology_class_1",
    "morphology_class_2",
    "resolved_probability_1",
    "resolved_probability_2",
    "beam_like_score_1",
    "beam_like_score_2",
    "classification_reason_1",
    "classification_reason_2",
    "observed_ellipse_overlap_score",
    "intrinsic_ellipse_overlap_score",
    "ellipse_overlap_score",
    "ellipse_gap_beam",
    "pa_weight_1",
    "pa_weight_2",
    "raw_pa_alignment_score",
    "effective_pa_alignment_score",
    "raw_line_to_pa_alignment_score",
    "effective_line_to_pa_alignment_score",
    "pa_alignment_score",
    "line_to_pa_alignment_score",
    "size_similarity_score",
    "flux_ratio",
    "flux_continuity_score",
    "connected_at_3sigma",
    "connected_at_2p5sigma",
    "connected_at_2sigma",
    "only_2sigma_connected",
    "same_label_3sigma",
    "same_label_2p5sigma",
    "same_label_2sigma",
    "common_envelope_area_beam",
    "bridge_mean_snr",
    "bridge_min_snr",
    "bridge_max_snr",
    "bridge_width_pix",
    "bridge_width_beam",
    "bridge_length_pix",
    "bridge_length_beam",
    "bridge_area_pix",
    "bridge_area_beam",
    "bridge_score",
    "residual_bridge_peak_snr",
    "residual_bridge_mean_snr",
    "residual_bridge_integrated_snr",
    "residual_bridge_area_beams",
    "residual_bridge_length_fraction",
    "residual_bridge_width_beams",
    "residual_bridge_contiguous_fraction",
    "multi_threshold_bridge_persistence",
    "residual_bridge_score",
    "ridge_mean_snr",
    "ridge_gap_fraction",
    "ridge_continuity_score",
    "ridge_gradient_smoothness",
    "closeness_score",
    "flow_alignment_score",
    "deep_valley_penalty",
    "only_2sigma_penalty",
    "negative_bowl_penalty",
    "sidelobe_risk_penalty",
    "too_far_penalty",
    "large_mask_swallow_penalty",
    "unresolved_pair_veto",
    "unresolved_pair_veto_reason",
    "association_score",
    "edge_type",
    "association_decision",
    "rejection_reason",
    "artifact_risk_flags",
    "penalties",
    "debug_info",
]

GROUP_COLUMNS = [
    "cutout_id",
    "association_group_id",
    "association_group_index",
    "component_ids",
    "n_gaussians",
    "gaussian_ids",
    "ra",
    "dec",
    "centroid_x",
    "centroid_y",
    "bounding_box",
    "LAS_arcsec",
    "LAS_beam",
    "total_flux_gaussian",
    "peak_flux",
    "group_PA",
    "axis_ratio",
    "association_score_mean",
    "association_score_min",
    "association_score_max",
    "n_strong_edges",
    "n_weak_edges",
    "n_only_2sigma_edges",
    "association_quality",
    "association_type",
    "morphology_class",
    "resolved_probability",
    "beam_like_score",
    "classification_reason",
    "artifact_risk_flags",
    "debug_info",
]


def _association_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "enabled": True,
        "max_pair_distance_beam": 15.0,
        "max_pair_distance_arcsec": None,
        "threshold_strong": 3.0,
        "threshold_weak": 2.0,
        "use_connected_components_for_strong_edges": True,
        "weak_edges_attach_only": True,
        "min_bridge_width_beam": 0.8,
        "min_bridge_length_beam": 1.0,
        "max_only_2sigma_score": 0.5,
        "enable_beam_aware_morphology": True,
        "enable_unresolved_pair_veto": True,
        "enable_residual_bridge": True,
        "residual_bridge_threshold_snr": 2.5,
        "residual_bridge_min_length_fraction": 0.45,
        "residual_bridge_min_area_beams": 0.25,
        "residual_bridge_min_score": 0.55,
        "residual_bridge_endpoint_exclusion_beam": 0.6,
        "unresolved_veto_beam_like_score": 0.70,
        "unresolved_veto_min_bridge_score": 0.70,
        "unresolved_veto_min_residual_bridge_score": 0.55,
        "unresolved_veto_min_common_envelope_area_beam": 1.0,
        "veto_score_cap": -1.0,
        "quality_thresholds": {"high": 3.5, "medium": 2.5, "low": 1.5},
    }
    out = dict(defaults)
    out.update(config.get("association", {}) or {})
    if out.get("max_pair_distance_arcsec") is None and config.get("max_pair_distance_arcsec") is not None:
        out["max_pair_distance_arcsec"] = config.get("max_pair_distance_arcsec")
    out["quality_thresholds"] = {
        **defaults["quality_thresholds"],
        **(config.get("association", {}).get("quality_thresholds", {}) if config.get("association") else {}),
    }
    return out


def _max_pair_distance_arcsec(config: dict[str, Any]) -> float | None:
    """Return the configured absolute pair-search cap in arcsec, if present."""

    value = _association_config(config).get("max_pair_distance_arcsec")
    if value is None:
        return None
    cap = safe_float(value, float("nan"))
    return float(cap) if np.isfinite(cap) and cap > 0 else None


def _association_weights(config: dict[str, Any]) -> dict[str, float]:
    defaults = {
        "closeness": 1.0,
        "overlap": 1.0,
        "pa_alignment": 0.8,
        "conn_3sigma": 1.5,
        "conn_2p5sigma": 0.6,
        "conn_2sigma": 0.0,
        "bridge": 1.2,
        "ridge": 1.2,
        "flux_continuity": 0.5,
        "flow_alignment": 0.6,
        "valley": 1.5,
        "only_2sigma": 1.0,
        "negative_bowl": 1.2,
        "sidelobe": 1.5,
        "too_far": 2.0,
        "large_mask_swallow": 1.5,
    }
    out = dict(defaults)
    out.update(config.get("weights_association", {}) or {})
    return {key: float(value) for key, value in out.items()}


def compute_beam_size_arcsec(config: dict[str, Any]) -> float:
    """Return the effective beam size used for distance normalization."""

    beam = config.get("beam", {}) or {}
    major = float(beam.get("major_arcsec", 6.0) or 6.0)
    minor = float(beam.get("minor_arcsec", major) or major)
    if major <= 0 or minor <= 0:
        return 6.0
    return float(np.sqrt(major * minor))


def _beam_area_arcsec2(config: dict[str, Any]) -> float:
    return beam_area_arcsec2(config=config)


def _thresholds_from_labels(labels_by_threshold: Any, config: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray]:
    if hasattr(labels_by_threshold, "labels_by_threshold"):
        labels = np.asarray(labels_by_threshold.labels_by_threshold)
        thresholds = np.asarray(labels_by_threshold.thresholds, dtype=float)
        return labels, thresholds
    if labels_by_threshold is None:
        return None, np.asarray(config.get("snr_thresholds", [5.0, 4.0, 3.0, 2.5, 2.0]), dtype=float)
    return np.asarray(labels_by_threshold), np.asarray(config.get("snr_thresholds", [5.0, 4.0, 3.0, 2.5, 2.0]), dtype=float)


def _label_from_row(row: pd.Series, threshold: float) -> int:
    col = f"label_at_{threshold_key(threshold)}"
    if col not in row:
        return 0
    try:
        return int(row[col])
    except Exception:
        return 0


def _label_from_map(row: pd.Series, labels: np.ndarray | None, thresholds: np.ndarray, threshold: float) -> int:
    if labels is None:
        return 0
    idx = int(np.argmin(np.abs(thresholds - float(threshold))))
    label_map = labels[idx]
    height, width = label_map.shape
    x = safe_float(row.get("x"))
    y = safe_float(row.get("y"))
    if not np.isfinite(x) or not np.isfinite(y):
        return 0
    xi = int(round(x))
    yi = int(round(y))
    if xi < 0 or xi >= width or yi < 0 or yi >= height:
        return 0
    return int(label_map[yi, xi])


def _shared_label(
    row_i: pd.Series,
    row_j: pd.Series,
    labels: np.ndarray | None,
    thresholds: np.ndarray,
    threshold: float,
) -> int:
    left = _label_from_row(row_i, threshold) or _label_from_map(row_i, labels, thresholds, threshold)
    right = _label_from_row(row_j, threshold) or _label_from_map(row_j, labels, thresholds, threshold)
    return int(left) if left > 0 and left == right else 0


def _label_count_cache(labels: np.ndarray | None, thresholds: np.ndarray, config: dict[str, Any]) -> dict[int, np.ndarray]:
    """Cache per-threshold label-pixel counts for repeated edge scoring."""

    if labels is None:
        return {}
    cache = config.setdefault("_label_count_cache", {})
    key = id(labels)
    cached = cache.get(key)
    if cached is not None:
        return cached
    out: dict[int, np.ndarray] = {}
    for idx in range(len(thresholds)):
        label_map = np.asarray(labels[idx])
        out[int(idx)] = np.bincount(label_map.ravel())
    cache.clear()
    cache[key] = out
    return out


def _label_area_fraction(
    labels: np.ndarray | None,
    thresholds: np.ndarray,
    threshold: float,
    label_id: int,
    config: dict[str, Any] | None = None,
) -> float:
    if labels is None or label_id <= 0:
        return 0.0
    idx = int(np.argmin(np.abs(thresholds - float(threshold))))
    label_map = labels[idx]
    if config is not None:
        counts = _label_count_cache(labels, thresholds, config).get(idx)
        if counts is not None and label_id < len(counts):
            return float(counts[int(label_id)] / max(label_map.size, 1))
    return float(np.count_nonzero(label_map == label_id) / max(label_map.size, 1))


def _label_area_beam(
    labels: np.ndarray | None,
    thresholds: np.ndarray,
    threshold: float,
    label_id: int,
    pixel_scale_arcsec: float,
    config: dict[str, Any],
) -> float:
    if labels is None or label_id <= 0:
        return 0.0
    idx = int(np.argmin(np.abs(thresholds - float(threshold))))
    label_map = labels[idx]
    counts = _label_count_cache(labels, thresholds, config).get(idx)
    if counts is not None and label_id < len(counts):
        n_pixels = int(counts[int(label_id)])
    else:
        n_pixels = int(np.count_nonzero(label_map == label_id))
    area_arcsec2 = float(n_pixels) * pixel_scale_arcsec * pixel_scale_arcsec
    return area_arcsec2 / max(_beam_area_arcsec2(config), 1e-6)


def _line_samples(x1: float, y1: float, x2: float, y2: float, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    distance = float(np.hypot(x2 - x1, y2 - y1))
    n_samples = n or max(3, int(np.ceil(distance)) + 1)
    return np.linspace(x1, x2, n_samples), np.linspace(y1, y2, n_samples)


def _sample_image_nearest(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    height, width = image.shape
    xi = np.rint(xs).astype(int)
    yi = np.rint(ys).astype(int)
    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    values = np.full(len(xs), np.nan, dtype=float)
    values[valid] = np.asarray(image, dtype=float)[yi[valid], xi[valid]]
    return values


def _corridor_values(
    snr_map: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    half_width_pix: float,
) -> np.ndarray:
    height, width = snr_map.shape
    pad = int(np.ceil(half_width_pix + 2))
    xmin = max(0, int(np.floor(min(x1, x2))) - pad)
    xmax = min(width - 1, int(np.ceil(max(x1, x2))) + pad)
    ymin = max(0, int(np.floor(min(y1, y2))) - pad)
    ymax = min(height - 1, int(np.ceil(max(y1, y2))) + pad)
    if xmax < xmin or ymax < ymin:
        return np.asarray([], dtype=float)
    yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
    dx = x2 - x1
    dy = y2 - y1
    length2 = dx * dx + dy * dy
    if length2 <= 0:
        dist = np.hypot(xx - x1, yy - y1)
        mask = dist <= half_width_pix
    else:
        t = ((xx - x1) * dx + (yy - y1) * dy) / length2
        dist = np.abs((xx - x1) * dy - (yy - y1) * dx) / max(np.sqrt(length2), 1e-6)
        mask = (t >= 0.0) & (t <= 1.0) & (dist <= half_width_pix)
    return np.asarray(snr_map[ymin : ymax + 1, xmin : xmax + 1], dtype=float)[mask]


def _angle_delta_deg(a: float, b: float) -> float:
    return float(abs((a - b + 90.0) % 180.0 - 90.0))


def _alignment_score(delta_deg: float, scale_deg: float = 45.0) -> float:
    return float(np.clip(1.0 - delta_deg / max(scale_deg, 1e-6), 0.0, 1.0))


def _component_major_minor(row: pd.Series, config: dict[str, Any]) -> tuple[float, float]:
    beam = compute_beam_size_arcsec(config)
    major = safe_float(row.get("_dc_maj"), safe_float(row.get("_maj"), beam))
    minor = safe_float(row.get("_dc_min"), safe_float(row.get("_min"), beam))
    if not np.isfinite(major) or major <= 0:
        major = safe_float(row.get("_maj"), beam)
    if not np.isfinite(minor) or minor <= 0:
        minor = safe_float(row.get("_min"), beam)
    if not np.isfinite(major) or major <= 0:
        major = beam
    if not np.isfinite(minor) or minor <= 0:
        minor = beam
    return float(major), float(minor)


def _component_classification(row: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    if "morphology_class" in row and str(row.get("morphology_class", "")).strip():
        return {
            "morphology_class": row.get("morphology_class"),
            "resolved_probability": safe_float(row.get("resolved_probability"), 0.0),
            "resolved_significance": safe_float(row.get("resolved_significance"), 0.0),
            "beam_like_score": safe_float(row.get("beam_like_score"), 0.0),
            "classification_reason": row.get("classification_reason", ""),
        }
    return classify_gaussian_component(row, config)


def _ellipse_overlap_from_axes(
    distance_arcsec: float,
    major_i: float,
    major_j: float,
    projected_beam_arcsec: float,
) -> tuple[float, float]:
    if not np.isfinite(distance_arcsec) or distance_arcsec < 0:
        return 0.0, float("nan")
    if not np.isfinite(major_i) or not np.isfinite(major_j) or major_i <= 0 or major_j <= 0:
        return 0.0, float("nan")
    support_radius = 0.5 * (major_i + major_j)
    gap_arcsec = float(distance_arcsec - support_radius)
    gap_beam = gap_arcsec / max(projected_beam_arcsec, 1e-6)
    return (1.0 if gap_arcsec <= 0 else float(np.clip(1.0 - gap_beam, 0.0, 1.0))), float(gap_beam)


def _intrinsic_overlap_score(
    component_i: pd.Series,
    component_j: pd.Series,
    distance_arcsec: float,
    projected_beam_arcsec: float,
    config: dict[str, Any],
) -> float:
    class_i = str(component_i.get("morphology_class", "")).strip() or str(_component_classification(component_i, config).get("morphology_class", ""))
    class_j = str(component_j.get("morphology_class", "")).strip() or str(_component_classification(component_j, config).get("morphology_class", ""))
    allowed = {"resolved", "marginally_resolved"}
    if class_i not in allowed or class_j not in allowed:
        return 0.0
    maj_i, _min_i, _pa_i = intrinsic_axes_from_component(component_i, config)
    maj_j, _min_j, _pa_j = intrinsic_axes_from_component(component_j, config)
    score, _gap = _ellipse_overlap_from_axes(distance_arcsec, maj_i, maj_j, projected_beam_arcsec)
    return float(score)


def _flux_ratio(row_i: pd.Series, row_j: pd.Series) -> float:
    f1 = safe_float(row_i.get("_total_flux"), safe_float(row_i.get("_peak_flux")))
    f2 = safe_float(row_j.get("_total_flux"), safe_float(row_j.get("_peak_flux")))
    if not np.isfinite(f1) or not np.isfinite(f2) or f1 <= 0 or f2 <= 0:
        return 0.5
    return float(np.clip(min(f1, f2) / max(f1, f2), 0.0, 1.0))


def compute_bridge_features(
    component_i: pd.Series,
    component_j: pd.Series,
    snr_map: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    """Estimate beam-width bridge evidence between two component centers."""

    # 沿两个 Gaussian 中心连线取样，并扩展成窄走廊；这样可以区分真实连续射电桥
    # 和仅因几何距离较近产生的偶然候选对。
    assoc = _association_config(config)
    pixel_scale = safe_float(component_i.get("pixel_scale_arcsec"), 1.5)
    x1 = safe_float(component_i.get("x"))
    y1 = safe_float(component_i.get("y"))
    x2 = safe_float(component_j.get("x"))
    y2 = safe_float(component_j.get("y"))
    dx = x2 - x1
    dy = y2 - y1
    cov = beam_covariance_from_config(config)
    axis_angle = direction_angle_from_delta(dx, dy)
    axis_beam = projected_beam_fwhm(axis_angle, cov)
    perp_beam = projected_beam_fwhm(axis_angle + 90.0, cov) if np.isfinite(axis_angle) else compute_beam_size_arcsec(config)
    length_pix = float(np.hypot(x2 - x1, y2 - y1))
    length_beam = float(length_pix * pixel_scale / max(axis_beam, 1e-6))
    min_width_beam = float(assoc.get("min_bridge_width_beam", 0.8))
    half_width_pix = max(1.0, 0.5 * min_width_beam * perp_beam / max(pixel_scale, 1e-6))
    weak = float(assoc.get("threshold_weak", 2.0))
    strong_mid = 2.5

    xs, ys = _line_samples(x1, y1, x2, y2)
    line_values = _sample_image_nearest(snr_map, xs, ys)
    corridor = _corridor_values(snr_map, x1, y1, x2, y2, half_width_pix=half_width_pix)
    finite_line = line_values[np.isfinite(line_values)]
    finite_corridor = corridor[np.isfinite(corridor)]
    finite = finite_corridor if finite_corridor.size else finite_line
    if finite.size == 0:
        return {
            "bridge_mean_snr": 0.0,
            "bridge_min_snr": 0.0,
            "bridge_max_snr": 0.0,
            "bridge_width_pix": 0.0,
            "bridge_width_beam": 0.0,
            "bridge_length_pix": length_pix,
            "bridge_length_beam": length_beam,
            "bridge_area_pix": 0.0,
            "bridge_area_beam": 0.0,
            "bridge_score": 0.0,
        }

    bridge_area_pix = float(np.count_nonzero(finite_corridor >= weak)) if finite_corridor.size else float(np.count_nonzero(finite_line >= weak))
    bridge_width_pix = bridge_area_pix / max(length_pix, 1.0)
    bridge_width_beam = bridge_width_pix * pixel_scale / max(perp_beam, 1e-6)
    beam_area_pix = _beam_area_arcsec2(config) / max(pixel_scale * pixel_scale, 1e-6)
    bridge_area_beam = bridge_area_pix / max(beam_area_pix, 1e-6)
    line_support = float(np.mean(finite_line >= weak)) if finite_line.size else 0.0
    mid_support = float(np.mean(finite_line >= strong_mid)) if finite_line.size else 0.0
    width_score = float(np.clip(bridge_width_beam / max(min_width_beam, 1e-6), 0.0, 1.0))
    mean_snr = float(np.nanmean(finite))
    min_snr = float(np.nanmin(finite_line)) if finite_line.size else float(np.nanmin(finite))
    max_snr = float(np.nanmax(finite))
    mean_score = float(np.clip((mean_snr - weak) / max(3.0 - weak, 1e-6), 0.0, 1.0))
    length_score = 1.0 if length_beam >= float(assoc.get("min_bridge_length_beam", 1.0)) else 0.7
    bridge_score = length_score * (0.35 * line_support + 0.25 * mid_support + 0.25 * width_score + 0.15 * mean_score)
    return {
        "bridge_mean_snr": mean_snr,
        "bridge_min_snr": min_snr,
        "bridge_max_snr": max_snr,
        "bridge_width_pix": float(bridge_width_pix),
        "bridge_width_beam": float(bridge_width_beam),
        "bridge_length_pix": length_pix,
        "bridge_length_beam": length_beam,
        "bridge_area_pix": bridge_area_pix,
        "bridge_area_beam": float(bridge_area_beam),
        "bridge_score": float(np.clip(bridge_score, 0.0, 1.5)),
    }


def _beam_model_on_grid(
    xx: np.ndarray,
    yy: np.ndarray,
    x0: float,
    y0: float,
    amplitude: float,
    pixel_scale_arcsec: float,
    config: dict[str, Any],
) -> np.ndarray:
    cov_arcsec = beam_covariance_from_config(config)
    cov_pix = cov_arcsec / max(pixel_scale_arcsec * pixel_scale_arcsec, 1e-12)
    inv = np.linalg.pinv(cov_pix)
    dx = xx - float(x0)
    dy = yy - float(y0)
    q = inv[0, 0] * dx * dx + 2.0 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
    return float(amplitude) * np.exp(-0.5 * q)


def _max_contiguous_fraction(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    best = 0
    current = 0
    for value in mask.astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return float(best / max(mask.size, 1))


def compute_residual_bridge_features(
    component_i: pd.Series,
    component_j: pd.Series,
    snr_map: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    """Measure bridge residuals after subtracting two elliptical beam models."""

    assoc = _association_config(config)
    # 先扣除两端的近似波束模型，再检查中间残差是否仍有连续正信号；
    # 这能降低端点强峰对 bridge 统计量的抬高影响。
    if not bool(assoc.get("enable_residual_bridge", True)):
        return {
            "residual_bridge_peak_snr": 0.0,
            "residual_bridge_mean_snr": 0.0,
            "residual_bridge_integrated_snr": 0.0,
            "residual_bridge_area_beams": 0.0,
            "residual_bridge_length_fraction": 0.0,
            "residual_bridge_width_beams": 0.0,
            "residual_bridge_contiguous_fraction": 0.0,
            "multi_threshold_bridge_persistence": 0.0,
            "residual_bridge_score": 0.0,
        }

    x1 = safe_float(component_i.get("x"))
    y1 = safe_float(component_i.get("y"))
    x2 = safe_float(component_j.get("x"))
    y2 = safe_float(component_j.get("y"))
    pixel_scale = safe_float(component_i.get("pixel_scale_arcsec"), 1.5)
    if not np.all(np.isfinite([x1, y1, x2, y2, pixel_scale])):
        return {
            "residual_bridge_peak_snr": 0.0,
            "residual_bridge_mean_snr": 0.0,
            "residual_bridge_integrated_snr": 0.0,
            "residual_bridge_area_beams": 0.0,
            "residual_bridge_length_fraction": 0.0,
            "residual_bridge_width_beams": 0.0,
            "residual_bridge_contiguous_fraction": 0.0,
            "multi_threshold_bridge_persistence": 0.0,
            "residual_bridge_score": 0.0,
        }

    dx = x2 - x1
    dy = y2 - y1
    length_pix = float(np.hypot(dx, dy))
    if length_pix <= 0:
        return {
            "residual_bridge_peak_snr": 0.0,
            "residual_bridge_mean_snr": 0.0,
            "residual_bridge_integrated_snr": 0.0,
            "residual_bridge_area_beams": 0.0,
            "residual_bridge_length_fraction": 0.0,
            "residual_bridge_width_beams": 0.0,
            "residual_bridge_contiguous_fraction": 0.0,
            "multi_threshold_bridge_persistence": 0.0,
            "residual_bridge_score": 0.0,
        }

    cov = beam_covariance_from_config(config)
    axis_angle = direction_angle_from_delta(dx, dy)
    axis_beam = projected_beam_fwhm(axis_angle, cov)
    perp_beam = projected_beam_fwhm(axis_angle + 90.0, cov) if np.isfinite(axis_angle) else compute_beam_size_arcsec(config)
    width_beam = max(float(assoc.get("min_bridge_width_beam", 0.8)), 0.6)
    half_width_pix = max(1.0, 0.5 * width_beam * perp_beam / max(pixel_scale, 1e-6))
    endpoint_exclusion_beam = float(assoc.get("residual_bridge_endpoint_exclusion_beam", 0.6))
    endpoint_exclusion_pix = endpoint_exclusion_beam * axis_beam / max(pixel_scale, 1e-6)

    height, width = snr_map.shape
    pad = int(np.ceil(half_width_pix + 3.0 * max(axis_beam, perp_beam) / max(pixel_scale, 1e-6)))
    xmin = max(0, int(np.floor(min(x1, x2))) - pad)
    xmax = min(width - 1, int(np.ceil(max(x1, x2))) + pad)
    ymin = max(0, int(np.floor(min(y1, y2))) - pad)
    ymax = min(height - 1, int(np.ceil(max(y1, y2))) + pad)
    if xmax < xmin or ymax < ymin:
        return {
            "residual_bridge_peak_snr": 0.0,
            "residual_bridge_mean_snr": 0.0,
            "residual_bridge_integrated_snr": 0.0,
            "residual_bridge_area_beams": 0.0,
            "residual_bridge_length_fraction": 0.0,
            "residual_bridge_width_beams": 0.0,
            "residual_bridge_contiguous_fraction": 0.0,
            "multi_threshold_bridge_persistence": 0.0,
            "residual_bridge_score": 0.0,
        }

    yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
    t = ((xx - x1) * dx + (yy - y1) * dy) / max(length_pix * length_pix, 1e-6)
    dist = np.abs((xx - x1) * dy - (yy - y1) * dx) / max(length_pix, 1e-6)
    along_pix = t * length_pix
    corridor_mask = (
        (t >= 0.0)
        & (t <= 1.0)
        & (along_pix >= endpoint_exclusion_pix)
        & (along_pix <= max(length_pix - endpoint_exclusion_pix, endpoint_exclusion_pix))
        & (dist <= half_width_pix)
    )
    if not corridor_mask.any():
        return {
            "residual_bridge_peak_snr": 0.0,
            "residual_bridge_mean_snr": 0.0,
            "residual_bridge_integrated_snr": 0.0,
            "residual_bridge_area_beams": 0.0,
            "residual_bridge_length_fraction": 0.0,
            "residual_bridge_width_beams": 0.0,
            "residual_bridge_contiguous_fraction": 0.0,
            "multi_threshold_bridge_persistence": 0.0,
            "residual_bridge_score": 0.0,
        }

    sub = np.asarray(snr_map[ymin : ymax + 1, xmin : xmax + 1], dtype=float)
    amp1 = safe_float(_sample_image_nearest(snr_map, np.asarray([x1]), np.asarray([y1]))[0], safe_float(component_i.get("_peak_snr"), 0.0))
    amp2 = safe_float(_sample_image_nearest(snr_map, np.asarray([x2]), np.asarray([y2]))[0], safe_float(component_j.get("_peak_snr"), 0.0))
    amp1 = max(amp1, 0.0)
    amp2 = max(amp2, 0.0)
    null = _beam_model_on_grid(xx, yy, x1, y1, amp1, pixel_scale, config) + _beam_model_on_grid(xx, yy, x2, y2, amp2, pixel_scale, config)
    residual = sub - null
    values = residual[corridor_mask]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        peak = mean = integrated = area_beams = length_fraction = width_beams = contiguous = persistence = score = 0.0
    else:
        threshold = float(assoc.get("residual_bridge_threshold_snr", 2.5))
        positive = finite >= threshold
        peak = float(np.nanmax(finite))
        mean = float(np.nanmean(finite))
        area_pix = float(np.count_nonzero(positive))
        area_beams = area_pix / max(_beam_area_arcsec2(config) / max(pixel_scale * pixel_scale, 1e-6), 1e-6)
        width_beams = (area_pix / max(length_pix, 1.0)) * pixel_scale / max(perp_beam, 1e-6)
        integrated = float(np.nansum(np.clip(finite, 0.0, None)) / np.sqrt(max(area_beams, 1.0)))
        xs, ys = _line_samples(x1, y1, x2, y2)
        line_resid = _sample_image_nearest(
            residual,
            xs - xmin,
            ys - ymin,
        )
        finite_line = line_resid[np.isfinite(line_resid)]
        if finite_line.size:
            trim = int(round(endpoint_exclusion_pix))
            if trim > 0 and finite_line.size > 2 * trim:
                finite_line = finite_line[trim:-trim]
            line_positive = finite_line >= threshold
            length_fraction = float(np.mean(line_positive)) if line_positive.size else 0.0
            contiguous = _max_contiguous_fraction(line_positive)
            persistence = float(np.mean([np.mean(finite_line >= t) for t in [2.0, 2.5, 3.0]]))
        else:
            length_fraction = contiguous = persistence = 0.0
        peak_score = np.clip((peak - threshold) / 2.0, 0.0, 1.0)
        mean_score = np.clip((mean - 0.5 * threshold) / max(threshold, 1e-6), 0.0, 1.0)
        area_score = np.clip(area_beams / max(float(assoc.get("residual_bridge_min_area_beams", 0.25)), 1e-6), 0.0, 1.0)
        length_score = np.clip(length_fraction / max(float(assoc.get("residual_bridge_min_length_fraction", 0.45)), 1e-6), 0.0, 1.0)
        score = float(np.clip(0.30 * peak_score + 0.20 * mean_score + 0.20 * area_score + 0.20 * length_score + 0.10 * contiguous, 0.0, 1.5))

    return {
        "residual_bridge_peak_snr": float(peak),
        "residual_bridge_mean_snr": float(mean),
        "residual_bridge_integrated_snr": float(integrated),
        "residual_bridge_area_beams": float(area_beams),
        "residual_bridge_length_fraction": float(length_fraction),
        "residual_bridge_width_beams": float(width_beams),
        "residual_bridge_contiguous_fraction": float(contiguous),
        "multi_threshold_bridge_persistence": float(persistence),
        "residual_bridge_score": float(score),
    }


def compute_ridge_continuity(
    component_i: pd.Series,
    component_j: pd.Series,
    snr_map: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    """Estimate ridge continuity along the center-to-center path."""

    assoc = _association_config(config)
    weak = float(assoc.get("threshold_weak", 2.0))
    x1 = safe_float(component_i.get("x"))
    y1 = safe_float(component_i.get("y"))
    x2 = safe_float(component_j.get("x"))
    y2 = safe_float(component_j.get("y"))
    xs, ys = _line_samples(x1, y1, x2, y2)
    values = _sample_image_nearest(snr_map, xs, ys)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "ridge_mean_snr": 0.0,
            "ridge_gap_fraction": 1.0,
            "ridge_continuity_score": 0.0,
            "ridge_gradient_smoothness": 0.0,
        }
    ridge_mean = float(np.nanmean(finite))
    gap_fraction = float(np.mean(finite < weak))
    weak_support = float(np.mean(finite >= weak))
    mid_support = float(np.mean(finite >= 2.5))
    if finite.size >= 3:
        diffs = np.abs(np.diff(finite))
        smoothness = float(1.0 / (1.0 + np.nanmedian(diffs) / (abs(ridge_mean) + 1.0)))
    else:
        smoothness = 0.5
    continuity = (0.55 * weak_support + 0.30 * mid_support + 0.15 * smoothness) * (1.0 - 0.5 * gap_fraction)
    return {
        "ridge_mean_snr": ridge_mean,
        "ridge_gap_fraction": gap_fraction,
        "ridge_continuity_score": float(np.clip(continuity, 0.0, 1.5)),
        "ridge_gradient_smoothness": float(np.clip(smoothness, 0.0, 1.0)),
    }


def compute_artifact_penalties(
    component_i: pd.Series,
    component_j: pd.Series,
    image: np.ndarray,
    snr_map: np.ndarray,
    labels_by_threshold: Any,
    config: dict[str, Any],
    features: dict[str, Any] | None = None,
) -> dict[str, float | str]:
    """Compute penalties for weak-only bridges and likely radio artifacts."""

    del image
    features = dict(features or {})
    labels, thresholds = _thresholds_from_labels(labels_by_threshold, config)
    assoc = _association_config(config)
    pixel_scale = safe_float(component_i.get("pixel_scale_arcsec"), 1.5)
    x1 = safe_float(component_i.get("x"))
    y1 = safe_float(component_i.get("y"))
    x2 = safe_float(component_j.get("x"))
    y2 = safe_float(component_j.get("y"))
    distance_beam, _projected_beam, _direction = projected_beam_distance_from_delta(x2 - x1, y2 - y1, pixel_scale, config)
    weak = float(assoc.get("threshold_weak", 2.0))
    max_pair_distance_beam = float(assoc.get("max_pair_distance_beam", 15.0))
    xs, ys = _line_samples(x1, y1, x2, y2)
    line_values = _sample_image_nearest(snr_map, xs, ys)
    finite_line = line_values[np.isfinite(line_values)]
    min_line = float(np.nanmin(finite_line)) if finite_line.size else 0.0
    negative_fraction = float(np.mean(finite_line < -1.0)) if finite_line.size else 0.0
    shared_label_2 = int(features.get("same_label_2sigma", 0) or _shared_label(component_i, component_j, labels, thresholds, 2.0))
    label_fraction = _label_area_fraction(labels, thresholds, 2.0, shared_label_2, config)
    label_area_beam = _label_area_beam(labels, thresholds, 2.0, shared_label_2, pixel_scale, config)
    flux_ratio = float(features.get("flux_ratio", _flux_ratio(component_i, component_j)))
    only_2sigma_connected = bool(features.get("only_2sigma_connected", False))
    bridge_score = float(features.get("bridge_score", 0.0))
    ridge_score = float(features.get("ridge_continuity_score", 0.0))
    pa_score = float(features.get("pa_alignment_score", 0.0))
    line_pa_score = float(features.get("line_to_pa_alignment_score", 0.0))

    deep_valley = float(np.clip((weak - min_line) / max(weak + 1.0, 1e-6), 0.0, 1.5))
    only_2sigma = 1.0 if only_2sigma_connected else 0.0
    negative_bowl = float(np.clip((-min_line - 0.3) / 2.2, 0.0, 2.0) + np.clip(negative_fraction, 0.0, 1.0))
    too_far = float(np.clip((distance_beam / max(max_pair_distance_beam, 1e-6) - 0.7) / 0.3, 0.0, 2.0))
    if distance_beam > 0.55 * max_pair_distance_beam:
        support_count = int(bridge_score >= 0.45) + int(ridge_score >= 0.45) + int(pa_score >= 0.55) + int(line_pa_score >= 0.55)
        if support_count < 2:
            too_far = max(too_far, 1.0)
    large_mask = 0.0
    if label_fraction > 0.05:
        large_mask = max(large_mask, float(np.clip((label_fraction - 0.05) / 0.10, 0.0, 1.5)))
    if label_area_beam > max(30.0, 4.0 * max(distance_beam, 1.0) ** 2) and only_2sigma_connected:
        large_mask = max(large_mask, 1.0)
    sidelobe = 0.0
    if flux_ratio < 0.08 and (negative_bowl > 0.2 or deep_valley > 0.8):
        sidelobe = max(sidelobe, 0.7)
    if distance_beam < 6.0 and bridge_score < 0.2 and ridge_score < 0.2 and max(pa_score, line_pa_score) < 0.25:
        sidelobe = max(sidelobe, 0.4)

    flags = []
    if only_2sigma_connected:
        flags.append("only_2sigma")
    if negative_bowl >= 0.5:
        flags.append("negative_bowl")
    if sidelobe >= 0.5:
        flags.append("sidelobe_risk")
    if large_mask >= 0.5:
        flags.append("large_mask_swallow")
    if too_far >= 1.0:
        flags.append("too_far")

    return {
        "deep_valley_penalty": deep_valley,
        "only_2sigma_penalty": only_2sigma,
        "negative_bowl_penalty": float(np.clip(negative_bowl, 0.0, 2.0)),
        "sidelobe_risk_penalty": float(np.clip(sidelobe, 0.0, 1.5)),
        "too_far_penalty": too_far,
        "large_mask_swallow_penalty": float(np.clip(large_mask, 0.0, 1.5)),
        "artifact_risk_flags": ",".join(flags),
    }


def has_independent_radio_evidence(features: dict[str, Any], config: dict[str, Any]) -> bool:
    """Return True when evidence is not just beam-induced similarity."""

    assoc = _association_config(config)
    if bool(features.get("connected_at_3sigma", False)):
        return True
    common_area = safe_float(features.get("common_envelope_area_beam"), 0.0)
    if bool(features.get("connected_at_2p5sigma", False)) and common_area >= float(
        assoc.get("unresolved_veto_min_common_envelope_area_beam", 1.0)
    ):
        return True
    if safe_float(features.get("bridge_score"), 0.0) >= float(assoc.get("unresolved_veto_min_bridge_score", 0.70)):
        width_ok = safe_float(features.get("bridge_width_beam"), 0.0) >= float(assoc.get("min_bridge_width_beam", 0.8)) * 0.7
        length_ok = safe_float(features.get("bridge_length_beam"), 0.0) >= float(assoc.get("min_bridge_length_beam", 1.0)) * 0.7
        if width_ok and length_ok:
            return True
    if safe_float(features.get("residual_bridge_score"), 0.0) >= float(
        assoc.get("unresolved_veto_min_residual_bridge_score", 0.55)
    ):
        return True
    return False


def unresolved_pair_veto(features: dict[str, Any], config: dict[str, Any]) -> tuple[bool, str]:
    """Apply hard veto for two unresolved or strongly beam-like components."""

    assoc = _association_config(config)
    if not bool(assoc.get("enable_unresolved_pair_veto", True)):
        return False, ""
    class_i = str(features.get("morphology_class_1", ""))
    class_j = str(features.get("morphology_class_2", ""))
    beam_like_threshold = float(assoc.get("unresolved_veto_beam_like_score", 0.70))
    beam_like_i = safe_float(features.get("beam_like_score_1"), 0.0) >= beam_like_threshold
    beam_like_j = safe_float(features.get("beam_like_score_2"), 0.0) >= beam_like_threshold
    both_unresolved = (class_i == "unresolved" or beam_like_i) and (class_j == "unresolved" or beam_like_j)
    if both_unresolved and not has_independent_radio_evidence(features, config):
        return True, "veto_unresolved_pair_no_independent_radio_evidence"
    return False, ""


def compute_pair_association_features(
    component_i: pd.Series,
    component_j: pd.Series,
    image: np.ndarray,
    snr_map: np.ndarray,
    labels_by_threshold: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute beam-aware pair features for two PyBDSF Gaussian components."""

    # 第一层关联只比较候选近邻对；距离、形态方向、等值线连通、桥接和伪影惩罚
    # 都记录成可解释字段，最终分数不作为黑箱输出。
    labels, thresholds = _thresholds_from_labels(labels_by_threshold, config)
    pixel_scale = safe_float(component_i.get("pixel_scale_arcsec"), 1.5)
    x1 = safe_float(component_i.get("x"))
    y1 = safe_float(component_i.get("y"))
    x2 = safe_float(component_j.get("x"))
    y2 = safe_float(component_j.get("y"))
    dx = x2 - x1
    dy = y2 - y1
    distance_pix = float(np.hypot(x2 - x1, y2 - y1))
    distance_arcsec = distance_pix * pixel_scale
    distance_beam, projected_beam, line_angle = projected_beam_distance_from_delta(dx, dy, pixel_scale, config)
    parallel_arcsec, perpendicular_arcsec, angle_to_bpa = beam_axis_components(dx, dy, pixel_scale, config)
    assoc = _association_config(config)
    max_pair_distance_beam = float(assoc.get("max_pair_distance_beam", 15.0))
    max_pair_distance_arcsec = _max_pair_distance_arcsec(config)
    if distance_beam <= max_pair_distance_beam and (
        max_pair_distance_arcsec is None or distance_arcsec <= max_pair_distance_arcsec
    ):
        candidate_generation_reason = (
            "within_elliptical_beam_and_absolute_radius"
            if max_pair_distance_arcsec is not None
            else "within_elliptical_beam_radius_no_absolute_cap"
        )
    elif max_pair_distance_arcsec is not None and distance_arcsec > max_pair_distance_arcsec:
        candidate_generation_reason = "rejected_by_absolute_radius"
    else:
        candidate_generation_reason = "rejected_by_elliptical_beam_distance"
    closeness_score = float(np.clip(1.0 - distance_beam / max(max_pair_distance_beam, 1e-6), 0.0, 1.0))

    obs_major_i = safe_float(component_i.get("_maj"), safe_float(component_i.get("observed_major_arcsec"), compute_beam_size_arcsec(config)))
    obs_major_j = safe_float(component_j.get("_maj"), safe_float(component_j.get("observed_major_arcsec"), compute_beam_size_arcsec(config)))
    observed_overlap, observed_gap_beam = _ellipse_overlap_from_axes(distance_arcsec, obs_major_i, obs_major_j, projected_beam)
    ellipse_gap_beam = observed_gap_beam

    major_i, _minor_i = _component_major_minor(component_i, config)
    major_j, _minor_j = _component_major_minor(component_j, config)
    size_similarity_score = float(np.clip(min(major_i, major_j) / max(major_i, major_j, 1e-6), 0.0, 1.0))

    beam_aware_morphology = bool(assoc.get("enable_beam_aware_morphology", True))
    if beam_aware_morphology:
        class_i = _component_classification(component_i, config)
        class_j = _component_classification(component_j, config)
        intrinsic_overlap = _intrinsic_overlap_score(component_i, component_j, distance_arcsec, projected_beam, config)
        ellipse_overlap_score = intrinsic_overlap
        pa_weight_i = effective_pa_weight({**component_i.to_dict(), **class_i}, config)
        pa_weight_j = effective_pa_weight({**component_j.to_dict(), **class_j}, config)
    else:
        class_i = {
            "morphology_class": str(component_i.get("morphology_class", "unknown") or "unknown"),
            "resolved_probability": safe_float(component_i.get("resolved_probability"), 0.0),
            "beam_like_score": safe_float(component_i.get("beam_like_score"), 0.0),
            "classification_reason": "beam_aware_morphology_disabled",
        }
        class_j = {
            "morphology_class": str(component_j.get("morphology_class", "unknown") or "unknown"),
            "resolved_probability": safe_float(component_j.get("resolved_probability"), 0.0),
            "beam_like_score": safe_float(component_j.get("beam_like_score"), 0.0),
            "classification_reason": "beam_aware_morphology_disabled",
        }
        intrinsic_overlap = observed_overlap
        ellipse_overlap_score = observed_overlap
        pa_weight_i = 1.0
        pa_weight_j = 1.0
    pa_i = effective_component_pa_pixel({**component_i.to_dict(), **class_i}, config)
    pa_j = effective_component_pa_pixel({**component_j.to_dict(), **class_j}, config)
    if np.isfinite(pa_i) and np.isfinite(pa_j):
        raw_pa_alignment_score = _alignment_score(angle_delta_180(pa_i, pa_j))
    else:
        raw_pa_alignment_score = 0.0
    pa_alignment_score = float(pa_weight_i * pa_weight_j * raw_pa_alignment_score)
    line_scores = []
    weighted_line_scores = []
    if np.isfinite(pa_i):
        score_i = _alignment_score(angle_delta_180(pa_i, line_angle))
        line_scores.append(score_i)
        weighted_line_scores.append(pa_weight_i * score_i)
    if np.isfinite(pa_j):
        score_j = _alignment_score(angle_delta_180(pa_j, line_angle))
        line_scores.append(score_j)
        weighted_line_scores.append(pa_weight_j * score_j)
    raw_line_to_pa_alignment_score = float(np.mean(line_scores)) if line_scores else 0.0
    line_to_pa_alignment_score = float(np.mean(weighted_line_scores)) if weighted_line_scores else 0.0
    flow_alignment_score = float(np.mean([pa_alignment_score, line_to_pa_alignment_score]))
    flux_ratio = _flux_ratio(component_i, component_j)
    flux_continuity_score = float(np.sqrt(flux_ratio))

    same_3 = _shared_label(component_i, component_j, labels, thresholds, 3.0)
    same_25 = _shared_label(component_i, component_j, labels, thresholds, 2.5)
    same_2 = _shared_label(component_i, component_j, labels, thresholds, 2.0)
    conn_3 = same_3 > 0
    conn_25 = same_25 > 0
    conn_2 = same_2 > 0
    only_2 = bool(conn_2 and not conn_25 and not conn_3)
    common_envelope_area_beam = 0.0
    for threshold, label_id in [(3.0, same_3), (2.5, same_25), (2.0, same_2)]:
        if int(label_id) > 0:
            common_envelope_area_beam = max(
                common_envelope_area_beam,
                _label_area_beam(labels, thresholds, threshold, int(label_id), pixel_scale, config),
            )

    features: dict[str, Any] = {
        "cutout_id": component_i.get("cutout_id"),
        "gaussian_id_1": component_i.get("_gaussian_id"),
        "gaussian_id_2": component_j.get("_gaussian_id"),
        "component_index_1": int(component_i.get("component_index")),
        "component_index_2": int(component_j.get("component_index")),
        "pair_separation_arcsec": distance_arcsec,
        "distance_pix": distance_pix,
        "distance_arcsec": distance_arcsec,
        "pair_angle_deg": line_angle,
        "distance_beam": distance_beam,
        "elliptical_beam_distance": distance_beam,
        "directional_beam_arcsec": projected_beam,
        "beam_parallel_separation_arcsec": parallel_arcsec,
        "beam_perpendicular_separation_arcsec": perpendicular_arcsec,
        "beam_parallel_arcsec": parallel_arcsec,
        "beam_perpendicular_arcsec": perpendicular_arcsec,
        "pair_angle_relative_to_bpa_deg": angle_to_bpa,
        "pair_angle_relative_bpa_deg": angle_to_bpa,
        "candidate_generation_reason": candidate_generation_reason,
        "morphology_class_1": class_i.get("morphology_class", "unknown"),
        "morphology_class_2": class_j.get("morphology_class", "unknown"),
        "resolved_probability_1": class_i.get("resolved_probability", 0.0),
        "resolved_probability_2": class_j.get("resolved_probability", 0.0),
        "beam_like_score_1": class_i.get("beam_like_score", 0.0),
        "beam_like_score_2": class_j.get("beam_like_score", 0.0),
        "classification_reason_1": class_i.get("classification_reason", ""),
        "classification_reason_2": class_j.get("classification_reason", ""),
        "observed_ellipse_overlap_score": observed_overlap,
        "intrinsic_ellipse_overlap_score": intrinsic_overlap,
        "ellipse_overlap_score": ellipse_overlap_score,
        "ellipse_gap_beam": float(ellipse_gap_beam),
        "pa_weight_1": pa_weight_i,
        "pa_weight_2": pa_weight_j,
        "raw_pa_alignment_score": raw_pa_alignment_score,
        "effective_pa_alignment_score": pa_alignment_score,
        "raw_line_to_pa_alignment_score": raw_line_to_pa_alignment_score,
        "effective_line_to_pa_alignment_score": line_to_pa_alignment_score,
        "pa_alignment_score": pa_alignment_score,
        "line_to_pa_alignment_score": line_to_pa_alignment_score,
        "size_similarity_score": size_similarity_score,
        "flux_ratio": flux_ratio,
        "flux_continuity_score": flux_continuity_score,
        "closeness_score": closeness_score,
        "flow_alignment_score": flow_alignment_score,
        "connected_at_3sigma": bool(conn_3),
        "connected_at_2p5sigma": bool(conn_25),
        "connected_at_2sigma": bool(conn_2),
        "only_2sigma_connected": only_2,
        "same_label_3sigma": int(same_3),
        "same_label_2p5sigma": int(same_25),
        "same_label_2sigma": int(same_2),
        "common_envelope_area_beam": common_envelope_area_beam,
    }
    features.update(compute_bridge_features(component_i, component_j, snr_map, config))
    features.update(compute_residual_bridge_features(component_i, component_j, snr_map, config))
    features.update(compute_ridge_continuity(component_i, component_j, snr_map, config))
    features.update(compute_artifact_penalties(component_i, component_j, image, snr_map, labels_by_threshold, config, features))
    veto, veto_reason = unresolved_pair_veto(features, config)
    features["unresolved_pair_veto"] = bool(veto)
    features["unresolved_pair_veto_reason"] = veto_reason
    features["association_score"] = compute_association_score(features, config)
    features["penalties"] = json_dumps_safe(
        {
            "deep_valley_penalty": features["deep_valley_penalty"],
            "only_2sigma_penalty": features["only_2sigma_penalty"],
            "negative_bowl_penalty": features["negative_bowl_penalty"],
            "sidelobe_risk_penalty": features["sidelobe_risk_penalty"],
            "too_far_penalty": features["too_far_penalty"],
            "large_mask_swallow_penalty": features["large_mask_swallow_penalty"],
        }
    )
    return features


def compute_association_score(features: dict[str, Any], config: dict[str, Any]) -> float:
    """Compute the interpretable association score from feature evidence."""

    # 正证据提高合并分数，伪影/低可信连通会扣分；消融开关用于验证各类证据贡献。
    weights = _association_weights(config)
    assoc = _association_config(config)
    use_contour = ablation_enabled(config, "use_multithreshold_contour")
    use_ridge = ablation_enabled(config, "use_ridge_continuity")
    use_overlap = ablation_enabled(config, "use_ellipse_overlap")
    use_pa = ablation_enabled(config, "use_pa_alignment")
    use_artifacts = ablation_enabled(config, "use_artifact_penalties_layer1")
    score = 0.0
    score += weights["closeness"] * float(features.get("closeness_score", 0.0))
    if use_overlap:
        score += weights["overlap"] * float(features.get("ellipse_overlap_score", 0.0))
    if use_pa:
        score += weights["pa_alignment"] * float(features.get("pa_alignment_score", 0.0))
    if use_contour:
        score += weights["conn_3sigma"] * float(bool(features.get("connected_at_3sigma", False)))
        score += weights["conn_2p5sigma"] * float(bool(features.get("connected_at_2p5sigma", False)))
        score += weights["conn_2sigma"] * float(bool(features.get("connected_at_2sigma", False)))
    score += weights["bridge"] * float(features.get("bridge_score", 0.0))
    score += 0.8 * weights["bridge"] * float(features.get("residual_bridge_score", 0.0))
    if use_ridge:
        score += weights["ridge"] * float(features.get("ridge_continuity_score", 0.0))
    score += weights["flux_continuity"] * float(features.get("flux_continuity_score", 0.0))
    if use_pa:
        score += weights["flow_alignment"] * float(features.get("flow_alignment_score", 0.0))
    if use_artifacts:
        score -= weights["valley"] * float(features.get("deep_valley_penalty", 0.0))
        score -= weights["only_2sigma"] * float(features.get("only_2sigma_penalty", 0.0))
        score -= weights["negative_bowl"] * float(features.get("negative_bowl_penalty", 0.0))
        score -= weights["sidelobe"] * float(features.get("sidelobe_risk_penalty", 0.0))
        score -= weights["too_far"] * float(features.get("too_far_penalty", 0.0))
        score -= weights.get("large_mask_swallow", weights.get("large_mask", 1.5)) * float(features.get("large_mask_swallow_penalty", 0.0))

    only_2 = bool(features.get("only_2sigma_connected", False))
    no_higher_conn = not bool(features.get("connected_at_2p5sigma", False)) and not bool(features.get("connected_at_3sigma", False))
    has_independent_support = (
        float(features.get("bridge_score", 0.0)) >= 0.55
        or float(features.get("residual_bridge_score", 0.0)) >= 0.55
        or (use_ridge and float(features.get("ridge_continuity_score", 0.0)) >= 0.55)
        or (use_overlap and float(features.get("ellipse_overlap_score", 0.0)) >= 0.75)
    )
    if use_contour and use_artifacts and only_2 and no_higher_conn and not has_independent_support:
        score = min(score, float(assoc.get("max_only_2sigma_score", 0.5)))
    if bool(features.get("unresolved_pair_veto", False)):
        score = min(score, float(assoc.get("veto_score_cap", -1.0)))
    return float(score)


def _candidate_pairs(components: pd.DataFrame, config: dict[str, Any]) -> list[tuple[int, int]]:
    if len(components) < 2:
        return []
    # 使用 KD-tree 先做半径筛选，再用椭圆波束距离复核，避免 crowded field 中 O(N^2) 全配对。
    coords = components[["x", "y"]].to_numpy(float)
    finite = np.isfinite(coords).all(axis=1)
    if finite.sum() < 2:
        return []
    valid_positions = np.where(finite)[0]
    valid_coords = coords[finite]
    pixel_scale = safe_float(components["pixel_scale_arcsec"].iloc[0], 1.5)
    beam_major, _beam_minor, _beam_pa = beam_axes_from_config(config)
    assoc = _association_config(config)
    max_pair_distance_beam = float(assoc.get("max_pair_distance_beam", 15.0))
    radius_arcsec = max_pair_distance_beam * beam_major
    max_pair_distance_arcsec = _max_pair_distance_arcsec(config)
    if max_pair_distance_arcsec is not None:
        radius_arcsec = min(radius_arcsec, max_pair_distance_arcsec)
    radius_pix = radius_arcsec / max(pixel_scale, 1e-6)
    tree = cKDTree(valid_coords)
    pairs_local = tree.query_pairs(radius_pix)
    beam_cov = beam_covariance_from_config(config)
    pairs = []
    for i, j in pairs_local:
        dx_pix = float(valid_coords[j, 0] - valid_coords[i, 0])
        dy_pix = float(valid_coords[j, 1] - valid_coords[i, 1])
        dx_arcsec = dx_pix * pixel_scale
        dy_arcsec = dy_pix * pixel_scale
        distance_arcsec = float(np.hypot(dx_arcsec, dy_arcsec))
        if max_pair_distance_arcsec is not None and distance_arcsec > max_pair_distance_arcsec:
            continue
        distance_beam = elliptical_beam_distance((dx_arcsec, dy_arcsec), beam_cov)
        if not np.isfinite(distance_beam) or distance_beam > max_pair_distance_beam:
            continue
        pairs.append((int(valid_positions[i]), int(valid_positions[j])))
    pairs.sort()
    return pairs


def build_association_graph(
    components: pd.DataFrame,
    features: pd.DataFrame | list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[nx.Graph, pd.DataFrame]:
    """Create a graph whose core edges are strong association decisions."""

    # 图节点是单个 Gaussian；strong edge 直接连通成组，weak edge 只在后续聚类规则中辅助吸附。
    assoc = _association_config(config)
    threshold_strong = float(assoc.get("threshold_strong", 3.0))
    threshold_weak = float(assoc.get("threshold_weak", 2.0))
    graph = nx.Graph()
    for _, row in components.iterrows():
        graph.add_node(int(row["component_index"]), **row.to_dict())

    edges = pd.DataFrame(features)
    edges = _with_columns(edges, EDGE_COLUMNS)
    if edges.empty:
        graph.graph["components"] = components.copy()
        graph.graph["association_edges"] = edges
        return graph, edges

    edge_types = []
    decisions = []
    reasons = []
    for _, row in edges.iterrows():
        score = safe_float(row.get("association_score"), -np.inf)
        distance_beam = safe_float(row.get("distance_beam"), np.inf)
        distance_arcsec = safe_float(row.get("distance_arcsec"), np.inf)
        max_pair_distance_arcsec = _max_pair_distance_arcsec(config)
        if max_pair_distance_arcsec is not None and distance_arcsec > max_pair_distance_arcsec:
            edge_types.append("rejected")
            decisions.append(False)
            reasons.append("too_far_absolute_radius")
        elif distance_beam > float(assoc.get("max_pair_distance_beam", 15.0)):
            edge_types.append("rejected")
            decisions.append(False)
            reasons.append("too_far")
        elif bool(row.get("unresolved_pair_veto", False)):
            edge_types.append("rejected")
            decisions.append(False)
            reasons.append(str(row.get("unresolved_pair_veto_reason") or "veto_unresolved_pair_no_independent_radio_evidence"))
        elif score >= threshold_strong:
            edge_types.append("strong")
            decisions.append(True)
            reasons.append("")
        elif score >= threshold_weak:
            edge_types.append("weak")
            decisions.append(False)
            reasons.append("pending_weak_attachment")
        else:
            edge_types.append("rejected")
            decisions.append(False)
            reasons.append("score_below_threshold")
    edges["edge_type"] = edge_types
    edges["association_decision"] = decisions
    edges["rejection_reason"] = reasons

    for _, row in edges[edges["edge_type"] == "strong"].iterrows():
        graph.add_edge(
            int(row["component_index_1"]),
            int(row["component_index_2"]),
            association_score=float(row["association_score"]),
            edge_type="strong",
        )
    graph.graph["components"] = components.copy()
    graph.graph["association_edges"] = edges
    return graph, edges


def cluster_association_groups(graph: nx.Graph, config: dict[str, Any]) -> tuple[list[list[int]], pd.DataFrame, nx.Graph]:
    """Cluster by strong edges, then attach weak edges without chain merging."""

    edges = graph.graph.get("association_edges", pd.DataFrame()).copy()
    nodes = sorted(int(node) for node in graph.nodes)
    strong_graph = nx.Graph()
    strong_graph.add_nodes_from(nodes)
    if not edges.empty:
        for _, row in edges[edges["edge_type"] == "strong"].iterrows():
            strong_graph.add_edge(int(row["component_index_1"]), int(row["component_index_2"]))

    strong_components = [sorted(list(values)) for values in nx.connected_components(strong_graph)]
    strong_components.sort(key=lambda values: (values[0] if values else -1))
    group_by_node: dict[int, int] = {}
    original_group_by_node: dict[int, int] = {}
    original_group_size: dict[int, int] = {}
    for group_idx, members in enumerate(strong_components):
        original_group_size[group_idx] = len(members)
        for node in members:
            group_by_node[int(node)] = group_idx
            original_group_by_node[int(node)] = group_idx

    final_graph = nx.Graph()
    final_graph.add_nodes_from(graph.nodes(data=True))
    for _, row in edges[edges["edge_type"] == "strong"].iterrows():
        final_graph.add_edge(int(row["component_index_1"]), int(row["component_index_2"]), edge_type="strong")

    if not edges.empty:
        weak_order = edges[edges["edge_type"] == "weak"].sort_values("association_score", ascending=False).index.tolist()
        attached_singletons: set[int] = set()
        for idx in weak_order:
            row = edges.loc[idx]
            left = int(row["component_index_1"])
            right = int(row["component_index_2"])
            group_left = group_by_node[left]
            group_right = group_by_node[right]
            if group_left == group_right:
                edges.at[idx, "rejection_reason"] = "weak_edge_already_same_group"
                continue

            left_orig = original_group_by_node[left]
            right_orig = original_group_by_node[right]
            left_core = original_group_size[left_orig] >= 2
            right_core = original_group_size[right_orig] >= 2
            left_single = original_group_size[left_orig] == 1 and left not in attached_singletons
            right_single = original_group_size[right_orig] == 1 and right not in attached_singletons

            attach_node: int | None = None
            target_group: int | None = None
            if left_core and right_single:
                attach_node = right
                target_group = group_left
            elif right_core and left_single:
                attach_node = left
                target_group = group_right

            if attach_node is not None and target_group is not None:
                group_by_node[attach_node] = target_group
                attached_singletons.add(attach_node)
                edges.at[idx, "association_decision"] = True
                edges.at[idx, "rejection_reason"] = ""
                final_graph.add_edge(left, right, edge_type="weak")
            elif left_core and right_core:
                edges.at[idx, "rejection_reason"] = "weak_edge_would_merge_core_groups"
            elif (not left_core) and (not right_core):
                edges.at[idx, "rejection_reason"] = "weak_edge_no_core_group"
            else:
                edges.at[idx, "rejection_reason"] = "weak_edge_would_form_chain"

    if ablation_enabled(config, "use_weak_edge_anti_chaining"):
        grouped: dict[int, list[int]] = {}
        for node, group_idx in group_by_node.items():
            grouped.setdefault(int(group_idx), []).append(int(node))
        clusters = [sorted(values) for values in grouped.values()]
        clusters.sort(key=lambda values: (len(values), -values[0] if values else 0), reverse=True)
    else:
        final_graph = nx.Graph()
        final_graph.add_nodes_from(graph.nodes(data=True))
        if not edges.empty:
            # Keep the diagnostic edge table identical to the constrained run,
            # but cluster over all classified strong/weak edges.
            for _, row in edges[edges["edge_type"].astype(str).isin(["strong", "weak"])].iterrows():
                final_graph.add_edge(int(row["component_index_1"]), int(row["component_index_2"]), edge_type=str(row.get("edge_type", "")))
        clusters = [sorted(int(node) for node in values) for values in nx.connected_components(final_graph)]
        clusters.sort(key=lambda values: (values[0] if values else -1))
    final_graph.graph["association_edges"] = edges
    final_graph.graph["components"] = graph.graph.get("components", pd.DataFrame()).copy()
    return clusters, edges, final_graph


def _with_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out:
            out[col] = pd.Series(dtype=object)
    if out.empty:
        return pd.DataFrame(columns=columns)
    return out


def _bbox_from_points(x: np.ndarray, y: np.ndarray, padding: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    return (
        int(max(0, np.floor(np.nanmin(x) - padding))),
        int(max(0, np.floor(np.nanmin(y) - padding))),
        int(min(width - 1, np.ceil(np.nanmax(x) + padding))),
        int(min(height - 1, np.ceil(np.nanmax(y) + padding))),
    )


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _second_moments(image: np.ndarray, mask: np.ndarray, fallback_x: np.ndarray, fallback_y: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) < 3:
        if len(fallback_x) < 2:
            return float("nan"), float("nan"), float("nan"), 1.0
        xs = fallback_x
        ys = fallback_y
        weights = np.ones_like(xs, dtype=float)
    else:
        weights = np.asarray(image[ys, xs], dtype=float)
        weights = weights - np.nanmin(weights)
        weights[~np.isfinite(weights)] = 0.0
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
    x0 = float(np.average(xs, weights=weights))
    y0 = float(np.average(ys, weights=weights))
    dx = xs - x0
    dy = ys - y0
    cov = np.array(
        [
            [float(np.average(dx * dx, weights=weights)), float(np.average(dx * dy, weights=weights))],
            [float(np.average(dx * dy, weights=weights)), float(np.average(dy * dy, weights=weights))],
        ]
    )
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vec = vecs[:, order[0]]
    major = float(np.sqrt(max(vals[0], 0.0)))
    minor = float(np.sqrt(max(vals[1], 0.0)))
    pa = float((np.rad2deg(np.arctan2(vec[1], vec[0])) + 180.0) % 180.0)
    axis_ratio = float(major / max(minor, 1e-6)) if major > 0 else 1.0
    return major, minor, pa, axis_ratio


def _las_from_points(x: np.ndarray, y: np.ndarray, pixel_scale_arcsec: float) -> tuple[float, float]:
    if len(x) < 2:
        return 0.0, 0.0
    coords = np.column_stack([x, y])
    diff = coords[:, None, :] - coords[None, :, :]
    dist_pix = np.sqrt(np.sum(diff * diff, axis=-1))
    las_pix = float(np.nanmax(dist_pix))
    return las_pix, las_pix * pixel_scale_arcsec


def _support_las(mask: np.ndarray, pixel_scale_arcsec: float, current_las_pix: float) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if len(xs) <= 1:
        return current_las_pix, current_las_pix * pixel_scale_arcsec
    sample = np.column_stack([xs, ys])
    if len(sample) > 1000:
        idx = np.linspace(0, len(sample) - 1, 1000).astype(int)
        sample = sample[idx]
    diff = sample[:, None, :] - sample[None, :, :]
    support_las = float(np.sqrt(np.sum(diff * diff, axis=-1)).max())
    las_pix = max(current_las_pix, support_las)
    return las_pix, las_pix * pixel_scale_arcsec


def classify_association_group(
    group: pd.DataFrame,
    features: pd.DataFrame,
    measurements: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """Assign a radio association type without FRII-specific labels."""

    types = config.get("association_types", {}) or {}
    n_gaussians = int(measurements.get("n_gaussians", len(group)))
    las_beam = float(measurements.get("LAS_beam", 0.0) or 0.0)
    axis_ratio = float(measurements.get("axis_ratio", 1.0) or 1.0)
    quality = str(measurements.get("association_quality", "low"))
    flags = str(measurements.get("artifact_risk_flags", ""))
    if ablation_enabled(config, "use_artifact_penalties_layer1") and (
        "negative_bowl" in flags or "sidelobe" in flags or "large_mask_swallow" in flags
    ):
        return "artifact_risk"
    if n_gaussians >= int(types.get("complex_association", {}).get("min_components", 5)):
        return "complex_association"
    if n_gaussians <= 1:
        return "weak_association"
    if las_beam <= float(types.get("compact_multi_gaussian", {}).get("max_las_beam", 4.0)):
        return "compact_multi_gaussian"
    if axis_ratio >= float(types.get("linear_or_tail_like", {}).get("min_axis_ratio", 2.0)):
        return "linear_or_tail_like"
    has_continuity = False
    if not features.empty:
        has_continuity = bool(
            (
                ablation_enabled(config, "use_multithreshold_contour")
                and (
                    features["connected_at_3sigma"].astype(bool).any()
                    or features["connected_at_2p5sigma"].astype(bool).any()
                )
            )
            or (pd.to_numeric(features["bridge_score"], errors="coerce").fillna(0) >= 0.45).any()
            or (
                ablation_enabled(config, "use_ridge_continuity")
                and (pd.to_numeric(features["ridge_continuity_score"], errors="coerce").fillna(0) >= 0.45).any()
            )
        )
    if las_beam >= float(types.get("diffuse_extended", {}).get("min_las_beam", 6.0)) and quality in {"low", "suspicious"}:
        return "diffuse_extended"
    if has_continuity and las_beam >= float(types.get("continuous_extended", {}).get("min_las_beam", 4.0)):
        return "continuous_extended"
    if quality == "low":
        return "weak_association"
    return "continuous_extended"


def assign_association_quality(
    group: pd.DataFrame,
    edge_scores: pd.DataFrame,
    flags: list[str],
    config: dict[str, Any],
) -> str:
    """Assign high/medium/low/suspicious/artifact_risk quality labels."""

    del group
    thresholds = _association_config(config).get("quality_thresholds", {})
    high = float(thresholds.get("high", 3.5))
    medium = float(thresholds.get("medium", 2.5))
    low = float(thresholds.get("low", 1.5))
    severe_flags = {"negative_bowl", "sidelobe_risk", "large_mask_swallow"}
    if ablation_enabled(config, "use_artifact_penalties_layer1") and any(flag in severe_flags for flag in flags):
        return "artifact_risk"
    if edge_scores.empty:
        return "low"
    scores = pd.to_numeric(edge_scores["association_score"], errors="coerce").dropna()
    mean_score = float(scores.mean()) if len(scores) else 0.0
    score_spread = float(scores.max() - scores.min()) if len(scores) else 0.0
    n_edges = int(len(edge_scores))
    n_strong = int((edge_scores["edge_type"].astype(str) == "strong").sum())
    n_only_2 = int(edge_scores["only_2sigma_connected"].astype(bool).sum()) if "only_2sigma_connected" in edge_scores else 0
    has_strong_support = bool(
        (
            ablation_enabled(config, "use_multithreshold_contour")
            and (
                edge_scores["connected_at_3sigma"].astype(bool).any()
                or edge_scores["connected_at_2p5sigma"].astype(bool).any()
            )
        )
        or (pd.to_numeric(edge_scores["bridge_score"], errors="coerce").fillna(0) >= 0.55).any()
        or (
            ablation_enabled(config, "use_ridge_continuity")
            and (pd.to_numeric(edge_scores["ridge_continuity_score"], errors="coerce").fillna(0) >= 0.55).any()
        )
    )
    if (
        ablation_enabled(config, "use_artifact_penalties_layer1")
        and ("only_2sigma" in flags or n_only_2 > max(1, int(0.4 * n_edges)) or score_spread > 2.5)
    ):
        return "suspicious"
    if mean_score >= high and n_strong >= max(1, int(0.6 * n_edges)) and has_strong_support:
        return "high"
    if mean_score >= medium and has_strong_support:
        return "medium"
    if mean_score >= low:
        return "low"
    return "low"


def _artifact_flags_from_edges(edges: pd.DataFrame) -> list[str]:
    flags: set[str] = set()
    if edges.empty:
        return []
    for text in edges.get("artifact_risk_flags", pd.Series(dtype=str)).astype(str):
        for item in text.split(","):
            item = item.strip()
            if item:
                flags.add(item)
    return sorted(flags)


def _accepted_internal_edges(edges: pd.DataFrame, nodes: set[int]) -> pd.DataFrame:
    if edges.empty or len(nodes) < 2:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    mask = (
        edges["association_decision"].astype(bool)
        & edges["component_index_1"].astype(int).isin(nodes)
        & edges["component_index_2"].astype(int).isin(nodes)
    )
    return edges.loc[mask].copy()


def _measure_groups(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    clusters: list[list[int]],
    edges: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 对每个连通组重新测量包围盒、LAS、总流量和质量等级；
    # 这些量会写入最终 catalogue，并供第二层 parent-link 继续使用。
    image = np.asarray(cutout.image, dtype=float)
    beam_arcsec = compute_beam_size_arcsec(config)
    records: list[dict[str, Any]] = []
    group_id_by_node: dict[int, str] = {}
    group_index_by_node: dict[int, int] = {}
    group_size_by_node: dict[int, int] = {}
    quality_by_node: dict[int, str] = {}
    type_by_node: dict[int, str] = {}

    for group_idx, nodes in enumerate(clusters):
        node_set = set(int(node) for node in nodes)
        group_rows = components[components["component_index"].astype(int).isin(node_set)].copy()
        if group_rows.empty:
            continue
        pixel_scale = safe_float(group_rows["pixel_scale_arcsec"].iloc[0], 1.5)
        padding = max(5, int(round(2.5 * beam_arcsec / max(pixel_scale, 1e-6))))
        support_2 = component_support_mask(segmentation.labels_by_threshold, segmentation.thresholds, group_rows, 2.0)
        support_25 = component_support_mask(segmentation.labels_by_threshold, segmentation.thresholds, group_rows, 2.5)
        support_mask = support_25 if support_25.any() else support_2
        x = group_rows["x"].to_numpy(float)
        y = group_rows["y"].to_numpy(float)
        bbox = _bbox_from_mask(support_mask) if support_mask.any() else _bbox_from_points(x, y, padding, image.shape)
        if bbox is None:
            bbox = _bbox_from_points(x, y, padding, image.shape)
        x0, y0, x1, y1 = bbox
        weights = pd.to_numeric(group_rows["_peak_flux"], errors="coerce").to_numpy(float)
        weights[~np.isfinite(weights) | (weights <= 0)] = 1.0
        centroid_x = float(np.average(x, weights=weights))
        centroid_y = float(np.average(y, weights=weights))
        ra = float("nan")
        dec = float("nan")
        if getattr(cutout, "wcs", None) is not None:
            try:
                ra, dec = cutout.wcs.celestial.pixel_to_world_values(centroid_x, centroid_y)
                ra = float(ra)
                dec = float(dec)
            except Exception:
                pass
        las_pix, las_arcsec = _las_from_points(x, y, pixel_scale)
        if support_mask.any():
            las_pix, las_arcsec = _support_las(support_mask, pixel_scale, las_pix)
        _major, _minor, group_pa, axis_ratio = _second_moments(image, support_mask, x, y)
        total_flux = float(np.nansum(pd.to_numeric(group_rows["_total_flux"], errors="coerce")))
        peak_flux = float(np.nanmax(pd.to_numeric(group_rows["_peak_flux"], errors="coerce")))
        internal_edges = _accepted_internal_edges(edges, node_set)
        scores = pd.to_numeric(internal_edges.get("association_score", pd.Series(dtype=float)), errors="coerce").dropna()
        score_mean = float(scores.mean()) if len(scores) else 0.0
        score_min = float(scores.min()) if len(scores) else 0.0
        score_max = float(scores.max()) if len(scores) else 0.0
        n_strong = int((internal_edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "strong").sum()) if not internal_edges.empty else 0
        n_weak = int((internal_edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "weak").sum()) if not internal_edges.empty else 0
        n_only_2 = int(internal_edges.get("only_2sigma_connected", pd.Series(dtype=bool)).astype(bool).sum()) if not internal_edges.empty else 0
        flags = _artifact_flags_from_edges(internal_edges)
        if n_only_2 > max(1, len(internal_edges) // 2):
            flags.append("only_2sigma")
        flags = sorted(set(flags))
        quality = assign_association_quality(group_rows, internal_edges, flags, config)
        measurement_for_type = {
            "n_gaussians": int(len(group_rows)),
            "LAS_beam": float(las_arcsec / max(beam_arcsec, 1e-6)),
            "axis_ratio": axis_ratio,
            "association_quality": quality,
            "artifact_risk_flags": ",".join(flags),
        }
        association_type = classify_association_group(group_rows, internal_edges, measurement_for_type, config)
        component_classes = group_rows.get("morphology_class", pd.Series(dtype=str)).astype(str).tolist()
        component_probs = pd.to_numeric(group_rows.get("resolved_probability", pd.Series(dtype=float)), errors="coerce")
        component_beam_like = pd.to_numeric(group_rows.get("beam_like_score", pd.Series(dtype=float)), errors="coerce")
        if len(group_rows) == 1 and component_classes:
            group_morphology = component_classes[0] or "unknown"
            group_reason = str(group_rows.iloc[0].get("classification_reason", ""))
        elif association_type == "artifact_risk" or quality == "artifact_risk":
            group_morphology = "artifact_like"
            group_reason = "artifact_risk_group"
        elif len(group_rows) >= 2 and (
            association_type in {"continuous_extended", "diffuse_extended", "linear_or_tail_like", "complex_association"}
            or n_strong > 0
            or las_arcsec / max(beam_arcsec, 1e-6) >= 3.0
        ):
            group_morphology = "multi_gaussian_extended"
            group_reason = "multi_gaussian_extended"
        elif component_classes and all(value == "unresolved" for value in component_classes):
            group_morphology = "unresolved"
            group_reason = "all_components_unresolved"
        else:
            group_morphology = "unknown"
            group_reason = "mixed_or_low_support_group"
        group_resolved_probability = float(component_probs.max()) if len(component_probs.dropna()) else 0.0
        group_beam_like_score = float(component_beam_like.max()) if len(component_beam_like.dropna()) else 0.0
        group_id = f"{cutout.cutout_id}_a{group_idx:03d}"
        record = {
            "cutout_id": cutout.cutout_id,
            "association_group_id": group_id,
            "association_group_index": int(group_idx),
            "component_ids": ",".join(map(str, sorted(node_set))),
            "n_gaussians": int(len(group_rows)),
            "gaussian_ids": ",".join(map(str, group_rows["_gaussian_id"].tolist())),
            "ra": ra,
            "dec": dec,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "bounding_box": f"{x0},{y0},{x1},{y1}",
            "LAS_arcsec": las_arcsec,
            "LAS_beam": float(las_arcsec / max(beam_arcsec, 1e-6)),
            "total_flux_gaussian": total_flux,
            "peak_flux": peak_flux,
            "group_PA": group_pa,
            "axis_ratio": axis_ratio,
            "association_score_mean": score_mean,
            "association_score_min": score_min,
            "association_score_max": score_max,
            "n_strong_edges": n_strong,
            "n_weak_edges": n_weak,
            "n_only_2sigma_edges": n_only_2,
            "association_quality": quality,
            "association_type": association_type,
            "morphology_class": group_morphology,
            "resolved_probability": group_resolved_probability,
            "beam_like_score": group_beam_like_score,
            "classification_reason": group_reason,
            "artifact_risk_flags": ",".join(flags),
            "debug_info": json_dumps_safe(
                {
                    "cluster_nodes": sorted(node_set),
                    "bbox": [x0, y0, x1, y1],
                    "support_pixels_2sigma": int(support_2.sum()),
                    "support_pixels_2p5sigma": int(support_25.sum()),
                }
            ),
        }
        records.append(record)
        for node in node_set:
            group_id_by_node[node] = group_id
            group_index_by_node[node] = group_idx
            group_size_by_node[node] = int(len(group_rows))
            quality_by_node[node] = quality
            type_by_node[node] = association_type

    groups = _with_columns(pd.DataFrame(records), GROUP_COLUMNS)
    components = components.copy()
    components["association_group_id"] = components["component_index"].astype(int).map(group_id_by_node).fillna("")
    components["association_group_index"] = components["component_index"].astype(int).map(group_index_by_node).fillna(-1).astype(int)
    components["association_group_size"] = components["component_index"].astype(int).map(group_size_by_node).fillna(1).astype(int)
    components["association_quality"] = components["component_index"].astype(int).map(quality_by_node).fillna("low")
    components["association_type"] = components["component_index"].astype(int).map(type_by_node).fillna("weak_association")
    return groups, components


def run_component_association(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    config: dict[str, Any],
) -> AssociationResult:
    """Run the full beam-aware association pipeline for one cutout."""

    # 单个 cutout 的主流程：形态预处理 -> 候选对打分 -> 建图聚类 -> 组级测量。
    components = add_morphology_columns(components, config) if bool(_association_config(config).get("enable_beam_aware_morphology", True)) else components.copy()
    if components.empty:
        graph = nx.Graph()
        return AssociationResult(
            graph=graph,
            edges=pd.DataFrame(columns=EDGE_COLUMNS),
            components=components.copy(),
            groups=pd.DataFrame(columns=GROUP_COLUMNS),
            clusters=[],
        )

    feature_records: list[dict[str, Any]] = []
    for idx_i, idx_j in _candidate_pairs(components, config):
        row_i = components.iloc[idx_i]
        row_j = components.iloc[idx_j]
        feature_records.append(
            compute_pair_association_features(
                row_i,
                row_j,
                np.asarray(cutout.image, dtype=float),
                segmentation.snr_map,
                segmentation,
                config,
            )
        )

    graph, edges = build_association_graph(components, feature_records, config)
    clusters, edges, graph = cluster_association_groups(graph, config)
    if not clusters:
        clusters = [[int(value)] for value in components["component_index"].tolist()]
    groups, assoc_components = _measure_groups(cutout, segmentation, components, clusters, edges, config)
    if not edges.empty:
        edges = _with_columns(edges, EDGE_COLUMNS)
        edges["debug_info"] = edges.apply(lambda row: json_dumps_safe(row.to_dict()), axis=1)
    return AssociationResult(graph=graph, edges=edges, components=assoc_components, groups=groups, clusters=clusters)
