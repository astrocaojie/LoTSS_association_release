"""S/N-map construction and SExtractor-style segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .utils import robust_mad_rms


@dataclass
class SegmentationResult:
    """Multi-threshold S/N segmentation products for one cutout."""

    # labels_by_threshold[k] 存储第 k 个 S/N 阈值下的连通域编号；
    # association 阶段只需查两个 Gaussian 是否落在同一个非零 label 中。
    snr_map: np.ndarray
    thresholds: np.ndarray
    masks: np.ndarray
    labels_by_threshold: np.ndarray
    n_labels: np.ndarray
    n_small_objects_removed: np.ndarray | None = None


def estimate_mean(image: np.ndarray, mean: np.ndarray | float | None, mode: str = "median") -> np.ndarray | float:
    """Estimate the image mean/background for S/N construction."""

    if mean is not None:
        return mean
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0
    if mode == "zero":
        return 0.0
    if mode == "mean":
        return float(np.nanmean(finite))
    return float(np.nanmedian(finite))


def estimate_rms(image: np.ndarray, rms: np.ndarray | float | None, mode: str = "mad") -> np.ndarray | float:
    """Estimate the rms/noise for S/N construction."""

    if rms is not None:
        arr = np.asarray(rms)
        if arr.ndim == 0 and float(arr) > 0:
            return float(arr)
        if arr.ndim > 0:
            clean = arr.astype(float, copy=True)
            fallback = robust_mad_rms(image)
            clean[~np.isfinite(clean) | (clean <= 0)] = fallback
            return clean
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 1.0
    if mode == "std":
        value = float(np.nanstd(finite))
    else:
        value = robust_mad_rms(image)
    if not np.isfinite(value) or value <= 0:
        value = float(np.nanstd(finite))
    if not np.isfinite(value) or value <= 0:
        value = 1.0
    return value


def build_snr_map(
    image: np.ndarray,
    rms: np.ndarray | float | None = None,
    mean: np.ndarray | float | None = None,
    mean_mode: str = "median",
    rms_mode: str = "mad",
    smooth_before_segmentation: bool = True,
    gaussian_smooth_sigma_pix: float = 1.0,
) -> tuple[np.ndarray, np.ndarray | float, np.ndarray | float]:
    """Build an S/N map from an image and optional mean/rms maps."""

    # S/N 图是后续等值线连通、bridge/ridge 支持和伪影惩罚的共同基础。
    image = np.asarray(image, dtype=float)
    mean_value = estimate_mean(image, mean, mode=mean_mode)
    rms_value = estimate_rms(image, rms, mode=rms_mode)

    work = image
    if smooth_before_segmentation and gaussian_smooth_sigma_pix > 0:
        filled = np.array(work, copy=True)
        replacement = np.nanmedian(filled[np.isfinite(filled)]) if np.isfinite(filled).any() else 0.0
        filled[~np.isfinite(filled)] = replacement
        work = ndi.gaussian_filter(filled, sigma=gaussian_smooth_sigma_pix)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = (work - mean_value) / rms_value
    snr = np.asarray(snr, dtype=np.float32)
    snr[~np.isfinite(snr)] = 0.0
    return snr, mean_value, rms_value


def _remove_small(mask: np.ndarray, min_area: int, structure: np.ndarray) -> tuple[np.ndarray, int]:
    if min_area <= 1:
        return mask, 0
    labels, n_labels = ndi.label(mask, structure=structure)
    if n_labels == 0:
        return mask, 0
    counts = np.bincount(labels.ravel())
    keep = counts >= min_area
    keep[0] = False
    removed = int(np.count_nonzero((counts[1:] > 0) & (counts[1:] < min_area)))
    return keep[labels], removed


def segment_snr_map(
    snr_map: np.ndarray,
    thresholds: list[float],
    min_mask_area_pix: int = 20,
    connectivity: int = 2,
    binary_opening: bool = False,
    binary_closing: bool = True,
) -> SegmentationResult:
    """Segment an S/N map at multiple thresholds."""

    # 高阈值追踪亮峰，低阈值追踪扩展结构；多阈值结果共同进入 pairwise 证据表。
    snr_map = np.asarray(snr_map, dtype=float)
    thresholds_arr = np.asarray(thresholds, dtype=float)
    structure = ndi.generate_binary_structure(2, 2 if connectivity == 2 else 1)

    masks = []
    labels_all = []
    n_labels = []
    n_small_objects_removed = []

    for threshold in thresholds_arr:
        mask = snr_map > threshold
        mask &= np.isfinite(snr_map)
        if binary_opening:
            mask = ndi.binary_opening(mask, structure=structure)
        if binary_closing:
            mask = ndi.binary_closing(mask, structure=structure)
        mask, removed = _remove_small(mask, min_mask_area_pix, structure=structure)
        labels, count = ndi.label(mask, structure=structure)
        masks.append(mask.astype(np.uint8))
        labels_all.append(labels.astype(np.int32))
        n_labels.append(count)
        n_small_objects_removed.append(removed)

    return SegmentationResult(
        snr_map=snr_map.astype(np.float32),
        thresholds=thresholds_arr.astype(np.float32),
        masks=np.stack(masks, axis=0),
        labels_by_threshold=np.stack(labels_all, axis=0),
        n_labels=np.asarray(n_labels, dtype=np.int32),
        n_small_objects_removed=np.asarray(n_small_objects_removed, dtype=np.int32),
    )


def save_segmentation(path: str | Path, result: SegmentationResult) -> None:
    """Save segmentation products in compressed NPZ format."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        snr_map=result.snr_map,
        thresholds=result.thresholds,
        masks=result.masks,
        labels_by_threshold=result.labels_by_threshold,
        n_labels=result.n_labels,
        n_small_objects_removed=(
            result.n_small_objects_removed
            if result.n_small_objects_removed is not None
            else np.zeros_like(result.n_labels)
        ),
    )


def load_segmentation(path: str | Path) -> SegmentationResult:
    """Load a saved segmentation NPZ."""

    data = np.load(path, allow_pickle=False)
    return SegmentationResult(
        snr_map=data["snr_map"],
        thresholds=data["thresholds"],
        masks=data["masks"],
        labels_by_threshold=data["labels_by_threshold"],
        n_labels=data["n_labels"],
        n_small_objects_removed=data["n_small_objects_removed"]
        if "n_small_objects_removed" in data.files
        else np.zeros_like(data["n_labels"]),
    )


def segmentation_diagnostics(cutout_id: str, result: SegmentationResult) -> list[dict[str, Any]]:
    """Summarize segmentation masks for diagnostics."""

    records: list[dict[str, Any]] = []
    height, width = result.snr_map.shape
    image_area = float(height * width)
    removed = (
        result.n_small_objects_removed
        if result.n_small_objects_removed is not None
        else np.zeros_like(result.n_labels)
    )
    for idx, threshold in enumerate(result.thresholds):
        labels = result.labels_by_threshold[idx]
        counts = np.bincount(labels.ravel())
        largest = int(counts[1:].max()) if len(counts) > 1 else 0
        total = int(result.masks[idx].sum())
        fraction = float(total / image_area) if image_area > 0 else 0.0
        warning = ""
        if fraction > 0.1:
            warning = "mask_fraction_gt_0.1"
        if largest / image_area > 0.05:
            warning = f"{warning};largest_label_gt_0.05".strip(";")
        records.append(
            {
                "cutout_id": cutout_id,
                "threshold": float(threshold),
                "n_labels": int(result.n_labels[idx]),
                "largest_label_area": largest,
                "total_mask_area": total,
                "mask_fraction": fraction,
                "n_small_objects_removed": int(removed[idx]),
                "warning": warning,
            }
        )
    return records


def labels_at_points(
    labels_by_threshold: np.ndarray,
    thresholds: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return component labels at pixel positions for every threshold."""

    labels_by_threshold = np.asarray(labels_by_threshold)
    height, width = labels_by_threshold.shape[-2:]
    xi = np.rint(x).astype(int)
    yi = np.rint(y).astype(int)
    valid = np.isfinite(x) & np.isfinite(y) & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)

    output: dict[str, np.ndarray] = {}
    for idx, threshold in enumerate(thresholds):
        labels = np.zeros(len(xi), dtype=np.int32)
        labels[valid] = labels_by_threshold[idx, yi[valid], xi[valid]]
        key = threshold_key(float(threshold))
        output[f"label_at_{key}"] = labels
    return output


def threshold_key(threshold: float) -> str:
    """Return a stable column-name suffix for a threshold."""

    text = f"{threshold:g}".replace(".", "p").replace("-", "m")
    return f"{text}sigma"


def connected_at_threshold(
    row_i: Any,
    row_j: Any,
    threshold: float,
) -> bool:
    """Check if two component rows share a non-zero label at a threshold."""

    col = f"label_at_{threshold_key(threshold)}"
    if col not in row_i or col not in row_j:
        return False
    left = int(row_i[col])
    right = int(row_j[col])
    return left > 0 and left == right


def component_support_mask(
    labels_by_threshold: np.ndarray,
    thresholds: np.ndarray,
    component_rows: Any,
    threshold: float = 2.0,
) -> np.ndarray:
    """Build a union mask from the threshold labels touched by components."""

    idx = int(np.argmin(np.abs(np.asarray(thresholds, dtype=float) - threshold)))
    col = f"label_at_{threshold_key(float(thresholds[idx]))}"
    labels = set(int(value) for value in component_rows.get(col, []) if int(value) > 0)
    if not labels:
        return np.zeros(labels_by_threshold.shape[-2:], dtype=bool)
    label_map = labels_by_threshold[idx]
    return np.isin(label_map, list(labels))
