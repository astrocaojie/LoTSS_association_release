"""parent-seed refined conservative parent-link candidates.

The refined pass is a post-processing layer on top of local association
groups. It first decides which local groups are reliable parent-link
endpoints, then proposes a small number of nearby parent-link candidates using
robust 3 sigma or 5 sigma bounding boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .association import compute_beam_size_arcsec
from .utils import json_dumps_safe, safe_float


PARENT_SEED_CANDIDATE_COLUMNS = [
    "cutout_id",
    "parent_candidate_id",
    "local_group_id_1",
    "local_group_id_2",
    "box_gap_pix",
    "box_gap_arcsec",
    "box_gap_beam",
    "box_gap_beam_raw",
    "box_gap_beam_robust",
    "box_gap_source",
    "center_distance_arcsec",
    "center_distance_beam",
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
    "axis_alignment_score",
    "facing_score",
    "flux_ratio",
    "size_ratio",
    "core_candidate_near_midpoint",
    "parent_score",
    "parent_candidate_quality",
    "rejection_reason",
    "needs_visual_check",
]

PARENT_SEED_EDGE_DEBUG_COLUMNS = [
    *PARENT_SEED_CANDIDATE_COLUMNS,
    "parent_axis_angle",
    "group1_PA",
    "group2_PA",
    "multi_evidence_pass",
    "support_evidence_count",
    "endpoint1_pass",
    "endpoint2_pass",
    "both_parent_seed",
    "both_groups_extended_or_lobe_like",
    "compact_singleton_pair",
    "is_default_candidate",
    "debug_info",
]

PARENT_SEED_DIAGNOSTIC_COLUMNS = [
    "cutout_id",
    "n_local_groups",
    "n_parent_seed_groups",
    "n_parent_pairs_considered",
    "n_parent_candidates",
    "n_parent_high",
    "n_parent_medium",
    "n_parent_low_debug",
    "n_rejected",
    "n_rejected_compact_singleton_pair",
    "n_rejected_endpoint_not_parent_seed",
    "n_rejected_robust_box_gap_too_large",
    "n_rejected_insufficient_evidence_for_parent_link",
    "n_missing_peak_snr",
    "n_missing_area_3sigma_beam",
    "n_missing_robust_bbox",
    "parent_candidate_overflow_flag",
    "n_candidates_before_cutout_limit",
    "max_parent_candidates_per_cutout",
    "max_parent_candidates_per_group",
    "max_candidate_box_gap_beam_robust",
]

PARENT_SEED_COLUMNS = [
    "cutout_id",
    "association_group_id",
    "n_gaussians",
    "LAS_beam",
    "mask_area_beam",
    "area_3sigma_beam",
    "peak_snr",
    "association_quality",
    "association_type",
    "is_compact_singleton",
    "is_parent_seed",
    "parent_seed_reason",
    "parent_seed_reject_reason",
    "robust_bbox",
    "robust_bbox_source",
    "missing_fields",
]


@dataclass
class ParentSeedResult:
    candidates: pd.DataFrame
    edges_debug: pd.DataFrame
    diagnostics: pd.DataFrame
    needs_visual_check: pd.DataFrame
    parent_seed_table: pd.DataFrame


def parent_seed_config(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "enabled": True,
        "refined": True,
        "max_box_gap_beam": 10.0,
        "strong_box_gap_beam": 5.0,
        "min_endpoint_las_beam": 3.0,
        "min_endpoint_mask_area_beam": 3.0,
        "min_endpoint_n_gaussians": 2,
        "min_parent_seed_peak_snr": 8.0,
        "min_parent_seed_area_3sigma_beam": 2.0,
        "max_parent_candidates_per_group": 1,
        "max_parent_candidates_per_cutout": 10,
        "thresholds": {
            "high_score": 4.0,
            "medium_score": 3.2,
        },
        "evidence": {
            "axis_alignment_min": 0.7,
            "facing_min": 0.6,
            "max_flux_ratio": 10.0,
            "max_size_ratio": 5.0,
        },
    }
    raw = (config.get("parent_seed_selection", {}) or {}).copy()
    out = dict(defaults)
    out.update({key: value for key, value in raw.items() if key not in {"thresholds", "evidence"}})
    out["thresholds"] = dict(defaults["thresholds"])
    out["thresholds"].update(raw.get("thresholds", {}) or {})
    out["evidence"] = dict(defaults["evidence"])
    out["evidence"].update(raw.get("evidence", {}) or {})
    # 这里强制使用保守的候选数量上限；即使配置文件里残留更宽松的调试参数，
    # release 路径也不会让 parent seed 阶段产生过多候选。
    out["max_parent_candidates_per_group"] = int(raw.get("max_parent_candidates_per_group", 1))
    out["max_parent_candidates_per_cutout"] = int(raw.get("max_parent_candidates_per_cutout", 10))
    return out


def _with_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out:
            out[col] = pd.Series(dtype=object)
    if out.empty:
        return pd.DataFrame(columns=columns)
    return out[columns]


def _bbox_tuple(value: Any) -> tuple[float, float, float, float] | None:
    try:
        values = [float(item) for item in str(value).split(",")]
    except Exception:
        return None
    if len(values) != 4 or not np.all(np.isfinite(values)):
        return None
    x0, y0, x1, y1 = values
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def _bbox_string(box: tuple[float, float, float, float] | None) -> str:
    if box is None:
        return ""
    return ",".join(str(int(round(value))) for value in box)


def _box_gap_pix(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return float(np.hypot(dx, dy))


def _angle_delta_deg(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return 90.0
    return float(abs((a - b + 90.0) % 180.0 - 90.0))


def _alignment_score(delta_deg: float, scale_deg: float = 45.0) -> float:
    return float(np.clip(1.0 - delta_deg / max(scale_deg, 1e-6), 0.0, 1.0))


def _ratio_value(a: float, b: float) -> float:
    a = safe_float(a, 0.0)
    b = safe_float(b, 0.0)
    if a <= 0 or b <= 0:
        return float("nan")
    return float(max(a, b) / max(min(a, b), 1e-6))


def _ratio_score(ratio: float, max_ratio: float) -> float:
    if not np.isfinite(ratio):
        return 0.35
    if ratio >= max_ratio:
        return 0.0
    return float(np.clip(1.0 - np.log(ratio) / max(np.log(max_ratio), 1e-6), 0.0, 1.0))


def _association_type(row: pd.Series) -> str:
    return str(row.get("association_type", row.get("local_association_type", ""))).strip().lower()


def _association_quality(row: pd.Series) -> str:
    return str(row.get("association_quality", row.get("local_quality", ""))).strip().lower()


def _parse_debug_support(row: pd.Series, token: str) -> float:
    debug = str(row.get("debug_info", ""))
    marker = f'"{token}":'
    if marker not in debug:
        return float("nan")
    try:
        return float(debug.split(marker, 1)[1].split(",", 1)[0].split("}", 1)[0].strip())
    except Exception:
        return float("nan")


def _beam_area_pix(pixel_scale: float, config: dict[str, Any]) -> float:
    beam_arcsec = compute_beam_size_arcsec(config)
    return float(np.pi * (beam_arcsec / max(pixel_scale, 1e-6)) ** 2 / (4.0 * np.log(2.0)))


def _mask_area_beam(row: pd.Series, config: dict[str, Any]) -> float:
    value = safe_float(row.get("mask_area_beam"), float("nan"))
    if np.isfinite(value):
        return value
    support_pix = _parse_debug_support(row, "support_pixels_2p5sigma")
    if not np.isfinite(support_pix):
        support_pix = _parse_debug_support(row, "support_pixels_2sigma")
    pixel_scale = safe_float(row.get("pixel_scale_arcsec"), 1.5)
    if np.isfinite(support_pix) and support_pix > 0:
        return float(support_pix / max(_beam_area_pix(pixel_scale, config), 1e-6))
    las_beam = safe_float(row.get("LAS_beam"), 0.0)
    n_gauss = safe_float(row.get("n_gaussians"), 1.0)
    return float(max(0.0, 0.45 * las_beam + 0.35 * n_gauss))


def _component_ids(row: pd.Series) -> list[int]:
    ids: list[int] = []
    for item in str(row.get("component_ids", "")).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ids.append(int(float(item)))
        except Exception:
            pass
    return ids


def _threshold_index(segmentation: Any, threshold: float) -> int | None:
    if segmentation is None or not hasattr(segmentation, "thresholds"):
        return None
    thresholds = np.asarray(segmentation.thresholds, dtype=float)
    if thresholds.size == 0:
        return None
    idx = int(np.argmin(np.abs(thresholds - threshold)))
    if abs(float(thresholds[idx]) - threshold) <= 0.26:
        return idx
    return None


def _bbox_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _group_threshold_bbox(
    row: pd.Series,
    components: pd.DataFrame,
    segmentation: Any,
    threshold: float,
) -> tuple[tuple[float, float, float, float] | None, int]:
    idx = _threshold_index(segmentation, threshold)
    if idx is None or segmentation is None:
        return None, 0
    component_ids = set(_component_ids(row))
    if not component_ids or components.empty:
        return None, 0
    group_components = components[components["component_index"].astype(int).isin(component_ids)]
    if group_components.empty:
        return None, 0
    labels = segmentation.labels_by_threshold[idx]
    chosen_labels: set[int] = set()
    for _, comp in group_components.iterrows():
        x = int(round(safe_float(comp.get("x"), -1)))
        y = int(round(safe_float(comp.get("y"), -1)))
        if y < 0 or y >= labels.shape[0] or x < 0 or x >= labels.shape[1]:
            continue
        label = int(labels[y, x])
        if label > 0:
            chosen_labels.add(label)
    if not chosen_labels:
        return None, 0
    mask = np.isin(labels, list(chosen_labels))
    bbox = _bbox_from_mask(mask)
    return bbox, int(np.count_nonzero(mask))


def _peak_snr_for_group(row: pd.Series, components: pd.DataFrame, segmentation: Any) -> tuple[float, bool]:
    value = safe_float(row.get("peak_snr"), float("nan"))
    if np.isfinite(value):
        return value, False
    if segmentation is None or not hasattr(segmentation, "snr_map"):
        return float("nan"), True
    ids = set(_component_ids(row))
    if not ids or components.empty:
        return float("nan"), True
    group_components = components[components["component_index"].astype(int).isin(ids)]
    values: list[float] = []
    for _, comp in group_components.iterrows():
        x = int(round(safe_float(comp.get("x"), -1)))
        y = int(round(safe_float(comp.get("y"), -1)))
        if y < 0 or y >= segmentation.snr_map.shape[0] or x < 0 or x >= segmentation.snr_map.shape[1]:
            continue
        values.append(float(segmentation.snr_map[y, x]))
    if not values:
        return float("nan"), True
    return float(np.nanmax(values)), False


def _robust_bbox_for_group(
    row: pd.Series,
    components: pd.DataFrame,
    segmentation: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    pixel_scale = safe_float(row.get("pixel_scale_arcsec"), 1.5)
    bbox3, area3_pix = _group_threshold_bbox(row, components, segmentation, 3.0)
    if bbox3 is not None:
        return {
            "robust_bbox": bbox3,
            "robust_bbox_source": "3sigma_bbox",
            "area_3sigma_beam": float(area3_pix / max(_beam_area_pix(pixel_scale, config), 1e-6)),
            "missing_area_3sigma_beam": False,
            "missing_robust_bbox": False,
        }
    bbox5, area5_pix = _group_threshold_bbox(row, components, segmentation, 5.0)
    if bbox5 is not None:
        return {
            "robust_bbox": bbox5,
            "robust_bbox_source": "5sigma_core",
            "area_3sigma_beam": float("nan"),
            "missing_area_3sigma_beam": True,
            "missing_robust_bbox": False,
        }
    fallback = _bbox_tuple(row.get("bounding_box", ""))
    return {
        "robust_bbox": fallback,
        "robust_bbox_source": "fallback_bbox",
        "area_3sigma_beam": float("nan"),
        "missing_area_3sigma_beam": True,
        "missing_robust_bbox": fallback is None,
    }


def _compact_singleton(row: pd.Series, area_3sigma_beam: float, mask_area_beam: float) -> bool:
    area3 = area_3sigma_beam if np.isfinite(area_3sigma_beam) else 0.0
    return bool(
        int(safe_float(row.get("n_gaussians"), 1.0)) == 1
        and safe_float(row.get("LAS_beam"), 0.0) < 3.0
        and area3 < 2.0
        and mask_area_beam < 2.0
    )


def _extended_or_lobe_like(row: pd.Series, mask_area_beam: float, cfg: dict[str, Any]) -> bool:
    allowed_types = {
        "continuous_extended",
        "diffuse_extended",
        "linear_or_tail_like",
        "complex_association",
    }
    return bool(
        safe_float(row.get("n_gaussians"), 1.0) >= float(cfg.get("min_endpoint_n_gaussians", 2))
        or safe_float(row.get("LAS_beam"), 0.0) >= float(cfg.get("min_endpoint_las_beam", 3.0))
        or mask_area_beam >= float(cfg.get("min_endpoint_mask_area_beam", 3.0))
        or _association_type(row) in allowed_types
    )


def build_parent_seed_table(
    cutout_id: str,
    groups: pd.DataFrame,
    components: pd.DataFrame,
    segmentation: Any,
    config: dict[str, Any],
) -> pd.DataFrame:
    cfg = parent_seed_config(config)
    records: list[dict[str, Any]] = []
    bad_quality = {"low", "suspicious", "artifact_risk"}
    bad_type = {"weak_association", "artifact_risk"}
    for _, row in groups.iterrows():
        # parent seed 表先判断每个 local group 能否作为大尺度双瓣端点；
        # 点源、低信噪比或伪影风险组会保留诊断原因，但不会默认进入 parent 候选。
        work = row.copy()
        if "pixel_scale_arcsec" not in work:
            work["pixel_scale_arcsec"] = safe_float(groups.get("pixel_scale_arcsec", pd.Series([1.5])).iloc[0], 1.5) if not groups.empty else 1.5
        robust = _robust_bbox_for_group(work, components, segmentation, config)
        peak_snr, missing_peak = _peak_snr_for_group(work, components, segmentation)
        mask_area = _mask_area_beam(work, config)
        area3 = safe_float(robust["area_3sigma_beam"], float("nan"))
        compact = _compact_singleton(work, area3, mask_area)
        extended = _extended_or_lobe_like(work, mask_area, cfg)
        quality = _association_quality(work)
        atype = _association_type(work)
        missing_fields: list[str] = []
        if missing_peak:
            missing_fields.append("peak_snr")
        if bool(robust["missing_area_3sigma_beam"]):
            missing_fields.append("area_3sigma_beam")
        if bool(robust["missing_robust_bbox"]):
            missing_fields.append("robust_bbox")

        reject_reasons: list[str] = []
        if compact:
            reject_reasons.append("compact_singleton")
        if not np.isfinite(peak_snr) or peak_snr < float(cfg.get("min_parent_seed_peak_snr", 8.0)):
            reject_reasons.append("peak_snr_below_8")
        if not np.isfinite(area3) or area3 < float(cfg.get("min_parent_seed_area_3sigma_beam", 2.0)):
            reject_reasons.append("area_3sigma_beam_below_2")
        if not extended:
            reject_reasons.append("not_extended_or_lobe_like")
        if quality in bad_quality:
            reject_reasons.append(f"bad_quality:{quality}")
        if atype in bad_type:
            reject_reasons.append(f"bad_type:{atype}")
        is_seed = not reject_reasons
        reason_parts = [
            f"peak_snr={peak_snr:.2f}" if np.isfinite(peak_snr) else "peak_snr=missing",
            f"area3_beam={area3:.2f}" if np.isfinite(area3) else "area3_beam=missing",
            f"extended={extended}",
            f"quality={quality}",
            f"type={atype}",
            f"bbox={robust['robust_bbox_source']}",
        ]
        records.append(
            {
                "cutout_id": cutout_id,
                "association_group_id": work.get("association_group_id", work.get("local_group_id", "")),
                "n_gaussians": int(safe_float(work.get("n_gaussians"), 1.0)),
                "LAS_beam": safe_float(work.get("LAS_beam"), 0.0),
                "mask_area_beam": mask_area,
                "area_3sigma_beam": area3,
                "peak_snr": peak_snr,
                "association_quality": quality,
                "association_type": atype,
                "is_compact_singleton": bool(compact),
                "is_parent_seed": bool(is_seed),
                "parent_seed_reason": ";".join(reason_parts) if is_seed else "",
                "parent_seed_reject_reason": ";".join(reject_reasons),
                "robust_bbox": _bbox_string(robust["robust_bbox"]),
                "robust_bbox_source": robust["robust_bbox_source"],
                "missing_fields": ",".join(missing_fields),
            }
        )
    return _with_columns(pd.DataFrame(records), PARENT_SEED_COLUMNS)


def _candidate_search_pairs(groups: pd.DataFrame, max_gap_pix: float) -> list[tuple[int, int]]:
    if len(groups) < 2:
        return []
    boxes = [_bbox_tuple(row.get("robust_bbox", "")) for _, row in groups.iterrows()]
    centers: list[list[float]] = []
    valid_indices: list[int] = []
    half_diags: list[float] = []
    for idx, box in enumerate(boxes):
        if box is None:
            continue
        x0, y0, x1, y1 = box
        centers.append([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
        valid_indices.append(idx)
        half_diags.append(0.5 * float(np.hypot(x1 - x0, y1 - y0)))
    if len(centers) < 2:
        return []
    centers_arr = np.asarray(centers, dtype=float)
    half_diags_arr = np.asarray(half_diags, dtype=float)
    max_half_diag = float(np.nanmax(half_diags_arr)) if half_diags_arr.size else 0.0
    tree = cKDTree(centers_arr)
    debug_gap_pix = 1.5 * max_gap_pix
    seen: set[tuple[int, int]] = set()
    for local_i, original_i in enumerate(valid_indices):
        # KD-tree 先用中心距快速召回候选，再用 robust bbox gap 精筛；
        # 这样 crowded cutout 中不会因为全配对而产生大量无意义 parent pair。
        radius = debug_gap_pix + half_diags_arr[local_i] + max_half_diag + 1.0
        for local_j in tree.query_ball_point(centers_arr[local_i], radius):
            if local_j <= local_i:
                continue
            original_j = int(valid_indices[local_j])
            box_i = boxes[original_i]
            box_j = boxes[original_j]
            if box_i is None or box_j is None:
                continue
            if _box_gap_pix(box_i, box_j) <= debug_gap_pix:
                seen.add((int(original_i), original_j))
    return sorted(seen)


def _find_core_candidate_between(
    group_a: pd.Series,
    group_b: pd.Series,
    all_groups: pd.DataFrame,
    seed_by_id: dict[str, pd.Series],
    config: dict[str, Any],
) -> bool:
    beam_arcsec = compute_beam_size_arcsec(config)
    pixel_scale = safe_float(group_a.get("pixel_scale_arcsec"), 1.5)
    radius_pix = 3.0 * beam_arcsec / max(pixel_scale, 1e-6)
    x1, y1 = safe_float(group_a.get("centroid_x")), safe_float(group_a.get("centroid_y"))
    x2, y2 = safe_float(group_b.get("centroid_x")), safe_float(group_b.get("centroid_y"))
    if not np.all(np.isfinite([x1, y1, x2, y2])):
        return False
    midpoint_x = 0.5 * (x1 + x2)
    midpoint_y = 0.5 * (y1 + y2)
    pair_ids = {str(group_a.get("association_group_id")), str(group_b.get("association_group_id"))}
    axis_dx = x2 - x1
    axis_dy = y2 - y1
    axis_len2 = axis_dx * axis_dx + axis_dy * axis_dy
    for _, row in all_groups.iterrows():
        gid = str(row.get("association_group_id", ""))
        if gid in pair_ids:
            continue
        seed_row = seed_by_id.get(gid)
        if seed_row is None or not bool(seed_row.get("is_compact_singleton", False)):
            continue
        cx = safe_float(row.get("centroid_x"))
        cy = safe_float(row.get("centroid_y"))
        if not np.all(np.isfinite([cx, cy])):
            continue
        if float(np.hypot(cx - midpoint_x, cy - midpoint_y)) > radius_pix or axis_len2 <= 0:
            continue
        t = ((cx - x1) * axis_dx + (cy - y1) * axis_dy) / axis_len2
        axis_dist = abs((cx - x1) * axis_dy - (cy - y1) * axis_dx) / max(np.sqrt(axis_len2), 1e-6)
        if 0.15 <= t <= 0.85 and axis_dist <= radius_pix:
            return True
    return False


def _score_pair(features: dict[str, Any], cfg: dict[str, Any]) -> float:
    evidence = cfg["evidence"]
    max_gap = float(cfg.get("max_box_gap_beam", 10.0))
    gap_score = float(np.clip(1.0 - float(features["box_gap_beam_robust"]) / max(max_gap, 1e-6), 0.0, 1.0))
    flux_score = _ratio_score(float(features["flux_ratio"]), float(evidence.get("max_flux_ratio", 10.0)))
    size_score = _ratio_score(float(features["size_ratio"]), float(evidence.get("max_size_ratio", 5.0)))
    core_score = 1.0 if bool(features["core_candidate_near_midpoint"]) else 0.0
    return float(
        1.5 * gap_score
        + 1.4 * float(features["axis_alignment_score"])
        + 1.0 * float(features["facing_score"])
        + 0.7 * flux_score
        + 0.6 * size_score
        + 0.8 * core_score
    )


def _compute_pair(
    cutout_id: str,
    pair_index: int,
    group_a: pd.Series,
    group_b: pd.Series,
    seed_a: pd.Series,
    seed_b: pd.Series,
    all_groups: pd.DataFrame,
    seed_by_id: dict[str, pd.Series],
    config: dict[str, Any],
) -> dict[str, Any]:
    cfg = parent_seed_config(config)
    beam_arcsec = compute_beam_size_arcsec(config)
    pixel_scale = safe_float(group_a.get("pixel_scale_arcsec"), safe_float(group_b.get("pixel_scale_arcsec"), 1.5))
    raw_box_a = _bbox_tuple(group_a.get("bounding_box", ""))
    raw_box_b = _bbox_tuple(group_b.get("bounding_box", ""))
    robust_box_a = _bbox_tuple(seed_a.get("robust_bbox", ""))
    robust_box_b = _bbox_tuple(seed_b.get("robust_bbox", ""))
    raw_gap_pix = _box_gap_pix(raw_box_a, raw_box_b) if raw_box_a is not None and raw_box_b is not None else float("inf")
    robust_gap_pix = _box_gap_pix(robust_box_a, robust_box_b) if robust_box_a is not None and robust_box_b is not None else float("inf")
    raw_gap_beam = raw_gap_pix * pixel_scale / max(beam_arcsec, 1e-6)
    robust_gap_arcsec = robust_gap_pix * pixel_scale
    robust_gap_beam = robust_gap_arcsec / max(beam_arcsec, 1e-6)
    x1, y1 = safe_float(group_a.get("centroid_x")), safe_float(group_a.get("centroid_y"))
    x2, y2 = safe_float(group_b.get("centroid_x")), safe_float(group_b.get("centroid_y"))
    center_pix = float(np.hypot(x2 - x1, y2 - y1))
    center_arcsec = center_pix * pixel_scale
    center_beam = center_arcsec / max(beam_arcsec, 1e-6)
    axis_angle = float((np.rad2deg(np.arctan2(y2 - y1, x2 - x1)) + 180.0) % 180.0) if np.all(np.isfinite([x1, y1, x2, y2])) else float("nan")
    pa1 = safe_float(group_a.get("group_PA"), float("nan"))
    pa2 = safe_float(group_b.get("group_PA"), float("nan"))
    axis_scores = [_alignment_score(_angle_delta_deg(pa, axis_angle)) for pa in [pa1, pa2] if np.isfinite(pa)]
    axis_alignment = float(np.mean(axis_scores)) if axis_scores else 0.0
    facing = axis_alignment
    flux_ratio = _ratio_value(
        safe_float(group_a.get("total_flux_gaussian"), safe_float(group_a.get("peak_flux"), 0.0)),
        safe_float(group_b.get("total_flux_gaussian"), safe_float(group_b.get("peak_flux"), 0.0)),
    )
    size_ratio = _ratio_value(max(safe_float(group_a.get("LAS_beam"), 0.0), 0.5), max(safe_float(group_b.get("LAS_beam"), 0.0), 0.5))
    core_between = _find_core_candidate_between(group_a, group_b, all_groups, seed_by_id, config)
    both_seed = bool(seed_a.get("is_parent_seed", False)) and bool(seed_b.get("is_parent_seed", False))
    compact_pair = bool(seed_a.get("is_compact_singleton", False)) and bool(seed_b.get("is_compact_singleton", False))
    both_extended = _extended_or_lobe_like(group_a, safe_float(seed_a.get("mask_area_beam"), 0.0), cfg) and _extended_or_lobe_like(group_b, safe_float(seed_b.get("mask_area_beam"), 0.0), cfg)
    support_count = int(axis_alignment >= float(cfg["evidence"].get("axis_alignment_min", 0.7)))
    support_count += int(facing >= float(cfg["evidence"].get("facing_min", 0.6)))
    support_count += int(np.isfinite(flux_ratio) and flux_ratio <= float(cfg["evidence"].get("max_flux_ratio", 10.0)))
    support_count += int(np.isfinite(size_ratio) and size_ratio <= float(cfg["evidence"].get("max_size_ratio", 5.0)))
    support_count += int(core_between)
    if robust_gap_beam <= float(cfg.get("strong_box_gap_beam", 5.0)):
        support_count += int(both_seed)
    else:
        support_count += int(both_extended)
    source_a = str(seed_a.get("robust_bbox_source", "fallback_bbox"))
    source_b = str(seed_b.get("robust_bbox_source", "fallback_bbox"))
    box_gap_source = source_a if source_a == source_b else f"{source_a}+{source_b}"
    record = {
        "cutout_id": cutout_id,
        "parent_candidate_id": f"{cutout_id}_seed_pc{pair_index:03d}",
        "local_group_id_1": group_a.get("association_group_id"),
        "local_group_id_2": group_b.get("association_group_id"),
        "box_gap_pix": robust_gap_pix,
        "box_gap_arcsec": robust_gap_arcsec,
        "box_gap_beam": robust_gap_beam,
        "box_gap_beam_raw": raw_gap_beam,
        "box_gap_beam_robust": robust_gap_beam,
        "box_gap_source": box_gap_source,
        "center_distance_arcsec": center_arcsec,
        "center_distance_beam": center_beam,
        "n_gaussians_1": int(safe_float(group_a.get("n_gaussians"), 1.0)),
        "n_gaussians_2": int(safe_float(group_b.get("n_gaussians"), 1.0)),
        "LAS_beam_1": safe_float(group_a.get("LAS_beam"), 0.0),
        "LAS_beam_2": safe_float(group_b.get("LAS_beam"), 0.0),
        "mask_area_beam_1": safe_float(seed_a.get("mask_area_beam"), 0.0),
        "mask_area_beam_2": safe_float(seed_b.get("mask_area_beam"), 0.0),
        "area_3sigma_beam_1": safe_float(seed_a.get("area_3sigma_beam"), float("nan")),
        "area_3sigma_beam_2": safe_float(seed_b.get("area_3sigma_beam"), float("nan")),
        "peak_snr_1": safe_float(seed_a.get("peak_snr"), float("nan")),
        "peak_snr_2": safe_float(seed_b.get("peak_snr"), float("nan")),
        "axis_alignment_score": axis_alignment,
        "facing_score": facing,
        "flux_ratio": flux_ratio,
        "size_ratio": size_ratio,
        "core_candidate_near_midpoint": bool(core_between),
        "parent_axis_angle": axis_angle,
        "group1_PA": pa1,
        "group2_PA": pa2,
        "endpoint1_pass": bool(seed_a.get("is_parent_seed", False)),
        "endpoint2_pass": bool(seed_b.get("is_parent_seed", False)),
        "both_parent_seed": bool(both_seed),
        "both_groups_extended_or_lobe_like": bool(both_extended),
        "compact_singleton_pair": bool(compact_pair),
        "support_evidence_count": int(support_count),
    }
    record["multi_evidence_pass"] = bool(support_count >= (2 if robust_gap_beam <= float(cfg.get("strong_box_gap_beam", 5.0)) else 3))
    record["parent_score"] = _score_pair(record, cfg)

    rejection = ""
    if compact_pair:
        rejection = "compact_singleton_pair"
    elif not both_seed:
        rejection = "endpoint_not_parent_seed"
    elif robust_gap_beam > float(cfg.get("max_box_gap_beam", 10.0)):
        rejection = "robust_box_gap_too_large"
    else:
        required = 2 if robust_gap_beam <= float(cfg.get("strong_box_gap_beam", 5.0)) else 3
        if support_count < required:
            rejection = "insufficient_evidence_for_parent_link"

    thresholds = cfg["thresholds"]
    score = float(record["parent_score"])
    quality = "rejected"
    default = False
    if not rejection:
        if robust_gap_beam <= float(cfg.get("strong_box_gap_beam", 5.0)) and score >= float(thresholds.get("high_score", 4.0)) and bool(record["multi_evidence_pass"]):
            quality = "high"
            default = True
        elif robust_gap_beam <= float(cfg.get("max_box_gap_beam", 10.0)) and score >= float(thresholds.get("medium_score", 3.2)) and bool(record["multi_evidence_pass"]):
            quality = "medium"
            default = True
        elif score >= 2.5:
            quality = "low"
        else:
            quality = "rejected"
            rejection = "score_below_parent_candidate_threshold"
    record["parent_candidate_quality"] = quality
    record["rejection_reason"] = rejection
    record["is_default_candidate"] = bool(default)
    record["needs_visual_check"] = bool(default)
    record["debug_info"] = json_dumps_safe(
        {
            "parent_seed_reject_reason_1": seed_a.get("parent_seed_reject_reason", ""),
            "parent_seed_reject_reason_2": seed_b.get("parent_seed_reject_reason", ""),
            "support_evidence_count": support_count,
            "required_support_count": 2 if robust_gap_beam <= float(cfg.get("strong_box_gap_beam", 5.0)) else 3,
        }
    )
    return record


def _apply_group_limit(edges: pd.DataFrame, max_per_group: int) -> pd.Series:
    if edges.empty:
        return pd.Series(False, index=edges.index)
    selected = pd.Series(False, index=edges.index)
    counts: dict[str, int] = {}
    default_edges = edges[edges["is_default_candidate"].astype(bool)].sort_values(
        ["parent_score", "box_gap_beam_robust"], ascending=[False, True]
    )
    for idx, row in default_edges.iterrows():
        left = str(row.get("local_group_id_1"))
        right = str(row.get("local_group_id_2"))
        if counts.get(left, 0) >= max_per_group or counts.get(right, 0) >= max_per_group:
            continue
        selected.loc[idx] = True
        counts[left] = counts.get(left, 0) + 1
        counts[right] = counts.get(right, 0) + 1
    return selected


def _empty_result(cutout_id: str, n_groups: int, cfg: dict[str, Any]) -> ParentSeedResult:
    diagnostics = pd.DataFrame(
        [
            {
                "cutout_id": cutout_id,
                "n_local_groups": int(n_groups),
                "n_parent_seed_groups": 0,
                "n_parent_pairs_considered": 0,
                "n_parent_candidates": 0,
                "n_parent_high": 0,
                "n_parent_medium": 0,
                "n_parent_low_debug": 0,
                "n_rejected": 0,
                "n_rejected_compact_singleton_pair": 0,
                "n_rejected_endpoint_not_parent_seed": 0,
                "n_rejected_robust_box_gap_too_large": 0,
                "n_rejected_insufficient_evidence_for_parent_link": 0,
                "n_missing_peak_snr": 0,
                "n_missing_area_3sigma_beam": 0,
                "n_missing_robust_bbox": 0,
                "parent_candidate_overflow_flag": False,
                "n_candidates_before_cutout_limit": 0,
                "max_parent_candidates_per_cutout": int(cfg.get("max_parent_candidates_per_cutout", 10)),
                "max_parent_candidates_per_group": int(cfg.get("max_parent_candidates_per_group", 1)),
                "max_candidate_box_gap_beam_robust": 0.0,
            }
        ]
    )
    return ParentSeedResult(
        candidates=pd.DataFrame(columns=PARENT_SEED_CANDIDATE_COLUMNS),
        edges_debug=pd.DataFrame(columns=PARENT_SEED_EDGE_DEBUG_COLUMNS),
        diagnostics=_with_columns(diagnostics, PARENT_SEED_DIAGNOSTIC_COLUMNS),
        needs_visual_check=pd.DataFrame(columns=["cutout_id", "record_type", "object_id", "reason", "priority", "details"]),
        parent_seed_table=pd.DataFrame(columns=PARENT_SEED_COLUMNS),
    )


def run_parent_seed(
    cutout_id: str,
    local_groups: pd.DataFrame,
    local_components: pd.DataFrame,
    config: dict[str, Any],
    segmentation: Any | None = None,
) -> ParentSeedResult:
    cfg = parent_seed_config(config)
    if not bool(cfg.get("enabled", True)):
        return _empty_result(cutout_id, len(local_groups), cfg)

    groups = local_groups.copy().reset_index(drop=True)
    if "pixel_scale_arcsec" not in groups:
        if local_components is not None and not local_components.empty and "pixel_scale_arcsec" in local_components:
            groups["pixel_scale_arcsec"] = safe_float(local_components["pixel_scale_arcsec"].iloc[0], 1.5)
        else:
            groups["pixel_scale_arcsec"] = 1.5
    seed_table = build_parent_seed_table(cutout_id, groups, local_components, segmentation, config)
    if groups.empty:
        result = _empty_result(cutout_id, 0, cfg)
        result.parent_seed_table = seed_table
        return result
    seed_by_id = {str(row["association_group_id"]): row for _, row in seed_table.iterrows()}
    groups = groups.merge(seed_table[["association_group_id", "robust_bbox", "robust_bbox_source", "is_parent_seed"]], on="association_group_id", how="left")
    beam_arcsec = compute_beam_size_arcsec(config)
    pixel_scale = safe_float(groups["pixel_scale_arcsec"].iloc[0], 1.5)
    max_gap_pix = float(cfg.get("max_box_gap_beam", 10.0)) * beam_arcsec / max(pixel_scale, 1e-6)
    records: list[dict[str, Any]] = []
    for pair_index, (idx_i, idx_j) in enumerate(_candidate_search_pairs(groups, max_gap_pix)):
        # 每个 pair 只计算一次几何、尺度和端点证据；后续 host 支持阶段不会重新定义这些射电证据。
        group_i = groups.iloc[idx_i]
        group_j = groups.iloc[idx_j]
        seed_i = seed_by_id.get(str(group_i.get("association_group_id")), pd.Series(dtype=object))
        seed_j = seed_by_id.get(str(group_j.get("association_group_id")), pd.Series(dtype=object))
        records.append(_compute_pair(cutout_id, pair_index, group_i, group_j, seed_i, seed_j, groups, seed_by_id, config))
    edges = _with_columns(pd.DataFrame(records), PARENT_SEED_EDGE_DEBUG_COLUMNS)

    if not edges.empty:
        keep_group = _apply_group_limit(edges, int(cfg.get("max_parent_candidates_per_group", 1)))
        demote_mask = edges["is_default_candidate"].astype(bool) & ~keep_group
        edges.loc[demote_mask, "is_default_candidate"] = False
        edges.loc[demote_mask, "parent_candidate_quality"] = "low"

    candidates_all = edges[edges["is_default_candidate"].astype(bool)].copy() if not edges.empty else pd.DataFrame(columns=PARENT_SEED_EDGE_DEBUG_COLUMNS)
    candidates_before_cutout_limit = int(len(candidates_all))
    max_per_cutout = int(cfg.get("max_parent_candidates_per_cutout", 10))
    overflow = bool(len(candidates_all) > max_per_cutout)
    if overflow:
        kept_idx = candidates_all.sort_values(["parent_score", "box_gap_beam_robust"], ascending=[False, True]).head(max_per_cutout).index
        drop_mask = edges["is_default_candidate"].astype(bool) & ~edges.index.isin(kept_idx)
        edges.loc[drop_mask, "is_default_candidate"] = False
        edges.loc[drop_mask, "parent_candidate_quality"] = "low"
        candidates_all = edges.loc[kept_idx].copy()
    candidates = _with_columns(candidates_all, PARENT_SEED_CANDIDATE_COLUMNS)

    needs_records: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        needs_records.append(
            {
                "cutout_id": cutout_id,
                "record_type": "parent_seed_candidate",
                "object_id": row.get("parent_candidate_id"),
                "reason": "refined parent-seed endpoints with robust bbox gap",
                "priority": row.get("parent_candidate_quality", "medium"),
                "details": json_dumps_safe(
                    {
                        "local_group_id_1": row.get("local_group_id_1"),
                        "local_group_id_2": row.get("local_group_id_2"),
                        "box_gap_beam_robust": row.get("box_gap_beam_robust"),
                        "box_gap_source": row.get("box_gap_source"),
                        "parent_score": row.get("parent_score"),
                    }
                ),
            }
        )
    needs = pd.DataFrame(needs_records, columns=["cutout_id", "record_type", "object_id", "reason", "priority", "details"])
    reason = edges.get("rejection_reason", pd.Series(dtype=str)).astype(str) if not edges.empty else pd.Series(dtype=str)
    quality = edges.get("parent_candidate_quality", pd.Series(dtype=str)).astype(str) if not edges.empty else pd.Series(dtype=str)
    missing = seed_table.get("missing_fields", pd.Series(dtype=str)).astype(str) if not seed_table.empty else pd.Series(dtype=str)
    diagnostics = pd.DataFrame(
        [
            {
                "cutout_id": cutout_id,
                "n_local_groups": int(len(groups)),
                "n_parent_seed_groups": int(seed_table.get("is_parent_seed", pd.Series(dtype=bool)).astype(bool).sum()) if not seed_table.empty else 0,
                "n_parent_pairs_considered": int(len(edges)),
                "n_parent_candidates": int(len(candidates)),
                "n_parent_high": int((candidates.get("parent_candidate_quality", pd.Series(dtype=str)).astype(str) == "high").sum()) if not candidates.empty else 0,
                "n_parent_medium": int((candidates.get("parent_candidate_quality", pd.Series(dtype=str)).astype(str) == "medium").sum()) if not candidates.empty else 0,
                "n_parent_low_debug": int((quality == "low").sum()),
                "n_rejected": int((quality == "rejected").sum()),
                "n_rejected_compact_singleton_pair": int((reason == "compact_singleton_pair").sum()),
                "n_rejected_endpoint_not_parent_seed": int((reason == "endpoint_not_parent_seed").sum()),
                "n_rejected_robust_box_gap_too_large": int((reason == "robust_box_gap_too_large").sum()),
                "n_rejected_insufficient_evidence_for_parent_link": int((reason == "insufficient_evidence_for_parent_link").sum()),
                "n_missing_peak_snr": int(missing.str.contains("peak_snr", na=False).sum()),
                "n_missing_area_3sigma_beam": int(missing.str.contains("area_3sigma_beam", na=False).sum()),
                "n_missing_robust_bbox": int(missing.str.contains("robust_bbox", na=False).sum()),
                "parent_candidate_overflow_flag": overflow,
                "n_candidates_before_cutout_limit": candidates_before_cutout_limit,
                "max_parent_candidates_per_cutout": max_per_cutout,
                "max_parent_candidates_per_group": int(cfg.get("max_parent_candidates_per_group", 1)),
                "max_candidate_box_gap_beam_robust": float(pd.to_numeric(candidates.get("box_gap_beam_robust", pd.Series(dtype=float)), errors="coerce").max()) if not candidates.empty else 0.0,
            }
        ]
    )
    return ParentSeedResult(
        candidates=candidates,
        edges_debug=_with_columns(edges, PARENT_SEED_EDGE_DEBUG_COLUMNS),
        diagnostics=_with_columns(diagnostics, PARENT_SEED_DIAGNOSTIC_COLUMNS),
        needs_visual_check=needs,
        parent_seed_table=_with_columns(seed_table, PARENT_SEED_COLUMNS),
    )
