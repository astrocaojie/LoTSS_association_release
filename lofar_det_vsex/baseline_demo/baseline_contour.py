"""3 sigma contour-connectivity-only baseline."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy import ndimage as ndi

from .common import group_summary_from_membership, membership_from_clusters, write_method_outputs


def build_contour_labels(
    snr_map: np.ndarray,
    sigma_threshold: float = 3.0,
    pixel_connectivity: int = 8,
    min_mask_area_pixels: int = 1,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build a threshold mask and connected-component labels."""

    snr = np.asarray(snr_map, dtype=float)
    mask = np.isfinite(snr) & (snr >= float(sigma_threshold))
    structure = ndi.generate_binary_structure(2, 2 if int(pixel_connectivity) == 8 else 1)
    if int(min_mask_area_pixels) > 1:
        labels0, n0 = ndi.label(mask, structure=structure)
        if n0 > 0:
            counts = np.bincount(labels0.ravel())
            keep = counts >= int(min_mask_area_pixels)
            keep[0] = False
            mask = keep[labels0]
    labels, n_labels = ndi.label(mask, structure=structure)
    return mask.astype(np.uint8), labels.astype(np.int32), int(n_labels)


def assign_labels_at_components(
    components: pd.DataFrame,
    labels: np.ndarray,
    tolerance_pixels: int = 0,
) -> np.ndarray:
    """Assign each component center to a connected mask label."""

    height, width = labels.shape
    x = np.rint(pd.to_numeric(components["x"], errors="coerce")).astype("Int64")
    y = np.rint(pd.to_numeric(components["y"], errors="coerce")).astype("Int64")
    assigned = np.zeros(len(components), dtype=np.int32)
    tol = int(tolerance_pixels)
    for idx, (xi_val, yi_val) in enumerate(zip(x, y)):
        if pd.isna(xi_val) or pd.isna(yi_val):
            continue
        xi = int(xi_val)
        yi = int(yi_val)
        if xi < 0 or xi >= width or yi < 0 or yi >= height:
            continue
        label = int(labels[yi, xi])
        if label == 0 and tol > 0:
            x0 = max(0, xi - tol)
            x1 = min(width, xi + tol + 1)
            y0 = max(0, yi - tol)
            y1 = min(height, yi + tol + 1)
            window = labels[y0:y1, x0:x1]
            vals, counts = np.unique(window[window > 0], return_counts=True)
            if len(vals):
                label = int(vals[np.argmax(counts)])
        assigned[idx] = label
    return assigned


def run_contour_baseline(
    components: pd.DataFrame,
    snr_map: np.ndarray,
    config: dict[str, Any],
    image_header: Any | None = None,
    image_shape: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray]:
    """Run the 3 sigma connectivity-only grouping baseline."""

    t0 = perf_counter()
    cfg = config.get("contour_baseline", {}) or {}
    threshold = float(cfg.get("sigma_threshold", 3.0))
    connectivity = int(cfg.get("pixel_connectivity", 8))
    min_area = int(cfg.get("min_mask_area_pixels", 1))
    tolerance = int(cfg.get("component_assignment_tolerance_pixels", 0))
    mask, labels, n_labels = build_contour_labels(snr_map, threshold, connectivity, min_area)
    assigned = assign_labels_at_components(components, labels, tolerance)

    clusters: list[list[int]] = []
    for label in sorted(set(int(value) for value in assigned if int(value) > 0)):
        indices = components.loc[assigned == label, "component_index"].astype(int).tolist()
        if indices:
            clusters.append(sorted(indices))
    assigned_nodes = {node for cluster in clusters for node in cluster}
    for node in components["component_index"].astype(int).tolist():
        if int(node) not in assigned_nodes:
            clusters.append([int(node)])
    clusters.sort(key=lambda values: (values[0] if values else -1))

    method = "contour_3sigma"
    parameter_id = f"{threshold:g}sigma_conn{connectivity}"
    membership = membership_from_clusters(clusters, components, method, parameter_id, "contour_3sigma")
    groups = group_summary_from_membership(
        membership,
        components,
        method,
        parameter_id,
        image_shape=image_shape,
        boundary_padding_pixels=int((config.get("demo_region", {}) or {}).get("padding_pixels", 0) or 0),
    )

    label_component_counts = pd.Series(assigned[assigned > 0]).value_counts()
    stats = {
        "method": method,
        "parameter_id": parameter_id,
        "sigma_threshold": threshold,
        "pixel_connectivity": connectivity,
        "min_mask_area_pixels": min_area,
        "component_assignment_tolerance_pixels": tolerance,
        "n_mask_regions": int(n_labels),
        "n_mask_regions_with_0_gaussians": int(max(n_labels - len(label_component_counts), 0)),
        "n_mask_regions_with_1_gaussian": int((label_component_counts == 1).sum()),
        "n_mask_regions_with_ge2_gaussians": int((label_component_counts >= 2).sum()),
        "n_groups": int(len(groups)),
        "n_multi_groups": int((groups["n_components"] >= 2).sum()) if not groups.empty else 0,
        "runtime_seconds": perf_counter() - t0,
    }

    out_dir = Path(config.get("output_dir", "baseline_demo/outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    header = image_header.copy() if image_header is not None else fits.Header()
    fits.writeto(out_dir / "mask_3sigma.fits", mask.astype(np.uint8), header=header, overwrite=True)
    fits.writeto(out_dir / "mask_3sigma_labels.fits", labels.astype(np.int32), header=header, overwrite=True)
    write_method_outputs(out_dir, "groups_contour_3sigma", groups, membership)
    return groups, membership, stats, mask, labels
