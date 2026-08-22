"""Merged-source measurement helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .graph_merge import strongest_evidence_for_cluster
from .segmentation import component_support_mask
from .utils import json_dumps_safe, safe_float


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _bbox_from_points(x: np.ndarray, y: np.ndarray, padding: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    return (
        int(max(0, np.floor(np.nanmin(x) - padding))),
        int(max(0, np.floor(np.nanmin(y) - padding))),
        int(min(width - 1, np.ceil(np.nanmax(x) + padding))),
        int(min(height - 1, np.ceil(np.nanmax(y) + padding))),
    )


def _second_moments(image: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return float("nan"), float("nan"), float("nan")
    weights = np.asarray(image[ys, xs], dtype=float)
    weights = weights - np.nanmin(weights)
    weights[~np.isfinite(weights)] = 0.0
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    x0 = float(np.average(xs, weights=weights))
    y0 = float(np.average(ys, weights=weights))
    dx = xs - x0
    dy = ys - y0
    cov_xx = float(np.average(dx * dx, weights=weights))
    cov_yy = float(np.average(dy * dy, weights=weights))
    cov_xy = float(np.average(dx * dy, weights=weights))
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vec = vecs[:, order[0]]
    major = float(np.sqrt(max(vals[0], 0.0)))
    minor = float(np.sqrt(max(vals[1], 0.0)))
    pa = float((np.rad2deg(np.arctan2(vec[1], vec[0])) + 180.0) % 180.0)
    return major, minor, pa


def _las_from_points(x: np.ndarray, y: np.ndarray, pixel_scale_arcsec: float) -> tuple[float, float]:
    if len(x) < 2:
        return 0.0, 0.0
    coords = np.column_stack([x, y])
    diff = coords[:, None, :] - coords[None, :, :]
    dist_pix = np.sqrt(np.sum(diff * diff, axis=-1))
    las_pix = float(np.nanmax(dist_pix))
    return las_pix, las_pix * pixel_scale_arcsec


def _flags(cluster_rows: pd.DataFrame, las_pix: float) -> dict[str, bool]:
    n = len(cluster_rows)
    flux = pd.to_numeric(cluster_rows["_total_flux"], errors="coerce").to_numpy(float)
    finite_flux = flux[np.isfinite(flux) & (flux > 0)]
    flux_ratio_ok = True
    if finite_flux.size >= 2:
        flux_ratio_ok = bool(finite_flux.min() / finite_flux.max() > 0.1)
    pa = pd.to_numeric(cluster_rows["_pa"], errors="coerce").to_numpy(float)
    pa_finite = pa[np.isfinite(pa)]
    aligned = True
    if pa_finite.size >= 2:
        spread = np.nanmax(pa_finite) - np.nanmin(pa_finite)
        spread = min(spread, 180.0 - spread)
        aligned = bool(spread < 35.0)
    return {
        "multi_peak_flag": bool(n >= 2),
        "bent_flag": bool(n >= 3 and not aligned),
        "double_lobe_candidate_flag": bool(n >= 2 and las_pix > 5.0 and flux_ratio_ok),
    }


def measure_merged_sources(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    clusters: list[list[int]],
    edges: pd.DataFrame,
) -> pd.DataFrame:
    """Measure merged source candidates from graph clusters."""

    records: list[dict[str, Any]] = []
    image = np.asarray(cutout.image, dtype=float)
    for merged_idx, cluster_nodes in enumerate(clusters):
        cluster_rows = components[components["component_index"].isin(cluster_nodes)].copy()
        if cluster_rows.empty:
            continue

        mask_2 = component_support_mask(segmentation.labels_by_threshold, segmentation.thresholds, cluster_rows, 2.0)
        mask_25 = component_support_mask(segmentation.labels_by_threshold, segmentation.thresholds, cluster_rows, 2.5)
        support_mask = mask_2 if mask_2.any() else mask_25
        x = cluster_rows["x"].to_numpy(float)
        y = cluster_rows["y"].to_numpy(float)
        pixel_scale = safe_float(cluster_rows["pixel_scale_arcsec"].iloc[0], 1.5)
        padding = max(5, int(round(10.0 / max(pixel_scale, 1e-6))))
        bbox = _bbox_from_mask(support_mask) if support_mask.any() else _bbox_from_points(x, y, padding, image.shape)
        if bbox is None:
            bbox = _bbox_from_points(x, y, padding, image.shape)
        x0, y0, x1, y1 = bbox

        total_flux_gaussian = float(np.nansum(pd.to_numeric(cluster_rows["_total_flux"], errors="coerce")))
        total_flux_pixel_2 = float(np.nansum(image[mask_2])) if mask_2.any() else float("nan")
        total_flux_pixel_25 = float(np.nansum(image[mask_25])) if mask_25.any() else float("nan")
        peak_flux = float(np.nanmax(image[support_mask])) if support_mask.any() else float(
            np.nanmax(pd.to_numeric(cluster_rows["_peak_flux"], errors="coerce"))
        )

        weights = pd.to_numeric(cluster_rows["_peak_flux"], errors="coerce").to_numpy(float)
        weights[~np.isfinite(weights) | (weights <= 0)] = 1.0
        centroid_x = float(np.average(x, weights=weights))
        centroid_y = float(np.average(y, weights=weights))

        if support_mask.any():
            major, minor, pa = _second_moments(image, support_mask)
        else:
            major, minor, pa = float("nan"), float("nan"), float("nan")

        las_pix, las_arcsec = _las_from_points(x, y, pixel_scale)
        if support_mask.any():
            ys, xs = np.where(support_mask)
            if len(xs) > 1:
                sample = np.column_stack([xs, ys])
                if len(sample) > 1000:
                    idx = np.linspace(0, len(sample) - 1, 1000).astype(int)
                    sample = sample[idx]
                diff = sample[:, None, :] - sample[None, :, :]
                support_las = float(np.sqrt(np.sum(diff * diff, axis=-1)).max())
                las_pix = max(las_pix, support_las)
                las_arcsec = las_pix * pixel_scale

        ra = float("nan")
        dec = float("nan")
        if cutout.wcs is not None:
            try:
                ra, dec = cutout.wcs.celestial.pixel_to_world_values(centroid_x, centroid_y)
                ra = float(ra)
                dec = float(dec)
            except Exception:
                pass

        mean_score, max_score, evidence = strongest_evidence_for_cluster(edges, cluster_nodes)
        flags = _flags(cluster_rows, las_pix)
        confidence = float(np.clip(0.2 + 0.15 * len(cluster_rows) + 0.15 * mean_score, 0.0, 1.0))
        if flags["double_lobe_candidate_flag"]:
            confidence = min(1.0, confidence + 0.1)

        record = {
            "cutout_id": cutout.cutout_id,
            "merged_source_id": f"{cutout.cutout_id}_m{merged_idx:03d}",
            "component_ids": ",".join(map(str, cluster_nodes)),
            "gaussian_ids": ",".join(map(str, cluster_rows["_gaussian_id"].tolist())),
            "island_ids": ",".join(map(str, sorted(set(cluster_rows["_island_id"].astype(str))))),
            "pybdsf_island_ids": ",".join(map(str, sorted(set(cluster_rows["_island_id"].astype(str))))),
            "n_components": int(len(cluster_rows)),
            "total_gaussian_flux": total_flux_gaussian,
            "total_flux_gaussian_sum": total_flux_gaussian,
            "total_flux_pixel_2sigma": total_flux_pixel_2,
            "total_flux_pixel_2p5sigma": total_flux_pixel_25,
            "peak_flux": peak_flux,
            "bounding_box": f"{x0},{y0},{x1},{y1}",
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "ra": ra,
            "dec": dec,
            "centroid_ra_dec": f"{ra},{dec}" if np.isfinite(ra) and np.isfinite(dec) else "",
            "LAS_pixel": las_pix,
            "LAS_arcsec": las_arcsec,
            "second_moment_major": major,
            "second_moment_minor": minor,
            "second_moment_PA": pa,
            "PA": pa,
            "merge_score_mean": mean_score,
            "merge_score_max": max_score,
            "strongest_merge_evidence": evidence,
            "merge_confidence": confidence,
            "quality_flags": json_dumps_safe(flags),
            "flags": json_dumps_safe(flags),
            "host_ra": float("nan"),
            "host_dec": float("nan"),
            "confidence_score": confidence,
            "debug_info": json_dumps_safe(
                {
                    "cluster_nodes": cluster_nodes,
                    "bbox": [x0, y0, x1, y1],
                    "support_pixels_2sigma": int(mask_2.sum()),
                    "support_pixels_2p5sigma": int(mask_25.sum()),
                }
            ),
        }
        record.update(flags)
        records.append(record)

    return pd.DataFrame(records)
