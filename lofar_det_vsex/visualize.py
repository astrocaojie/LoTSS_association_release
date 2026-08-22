"""Readable overview and zoom visualizations for association outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lofar_det_vsex_mplconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Rectangle


DEFAULT_OVERVIEW = {
    "draw_all_labels": False,
    "draw_singletons": False,
    "label_min_components": 2,
    "max_labels": 30,
    "label_top_by": "LAS_arcsec",
    "draw_gaussian_ids": False,
    "draw_component_ids": False,
    "max_gaussians_drawn": 500,
    "gaussian_marker_size": 4,
    "max_edges_drawn": 300,
    "draw_nonmerged_edges": False,
    "edge_min_score": None,
    "contour_thresholds": [2.5, 3.0],
}

DEFAULT_ZOOM = {
    "enabled": True,
    "max_zoom_per_cutout": 20,
    "select_min_components": 2,
    "select_top_las": 10,
    "select_top_confidence": 10,
    "padding_pix": 100,
    "min_size_pix": 256,
    "max_size_pix": 1024,
    "contour_thresholds": [2.0, 2.5, 3.0, 5.0],
    "draw_gaussian_ids": True,
    "draw_edge_scores": True,
}


def _display_image(
    image: np.ndarray,
    stretch: str = "asinh",
    percent_clip: tuple[float, float] = (1, 99.5),
) -> np.ndarray:
    data = np.asarray(image, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data)
    lo, hi = np.nanpercentile(finite, percent_clip)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    clipped = np.clip(data, lo, hi)
    if stretch == "asinh":
        scale = np.nanstd(clipped[np.isfinite(clipped)])
        if not np.isfinite(scale) or scale <= 0:
            scale = max(hi - lo, 1.0)
        clipped = np.arcsinh((clipped - lo) / scale)
    return clipped


def _viz_config(config: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = config or {}
    viz = config.get("visualization", {})
    overview = dict(DEFAULT_OVERVIEW)
    overview.update(viz.get("overview", {}))
    zoom = dict(DEFAULT_ZOOM)
    zoom.update(viz.get("zoom", {}))
    return viz, overview, zoom


def _short_source_id(value: Any, fallback_idx: int = 0) -> str:
    text = str(value) if value is not None and str(value) != "nan" else ""
    match = re.search(r"_a(\d+)$", text)
    if match:
        return f"a{int(match.group(1)):03d}"
    match = re.search(r"_m(\d+)$", text)
    if match:
        return f"m{int(match.group(1)):03d}"
    match = re.search(r"a(\d+)$", text)
    if match:
        return f"a{int(match.group(1)):03d}"
    match = re.search(r"m(\d+)$", text)
    if match:
        return f"m{int(match.group(1)):03d}"
    return f"a{fallback_idx:03d}"


def _as_bool(series_or_value: Any) -> Any:
    if isinstance(series_or_value, pd.Series):
        if series_or_value.dtype == bool:
            return series_or_value
        return series_or_value.astype(str).str.lower().isin(["true", "1", "yes"])
    if isinstance(series_or_value, bool):
        return series_or_value
    return str(series_or_value).lower() in {"true", "1", "yes"}


def _bbox_tuple(bbox: Any) -> tuple[float, float, float, float] | None:
    try:
        values = [float(value) for value in str(bbox).split(",")]
    except Exception:
        return None
    if len(values) != 4 or not np.all(np.isfinite(values)):
        return None
    return values[0], values[1], values[2], values[3]


def _component_ids(row: pd.Series) -> set[int]:
    raw = row.get("component_ids", "")
    ids: set[int] = set()
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ids.add(int(float(item)))
        except Exception:
            pass
    return ids


def _is_association_catalog(sources: pd.DataFrame | None) -> bool:
    return sources is not None and not sources.empty and "association_group_id" in sources


def _n_components_column(sources: pd.DataFrame) -> str:
    if "n_gaussians" in sources:
        return "n_gaussians"
    return "n_components"


def _score_column(sources: pd.DataFrame) -> str:
    if "association_score_mean" in sources:
        return "association_score_mean"
    return "merge_confidence"


def _edge_score_column(edges: pd.DataFrame) -> str:
    if edges is not None and not edges.empty and "association_score" in edges:
        return "association_score"
    return "merge_score"


def _edge_decision_column(edges: pd.DataFrame) -> str:
    if edges is not None and not edges.empty and "association_decision" in edges:
        return "association_decision"
    return "merge_decision"


def _source_id_column(sources: pd.DataFrame) -> str:
    if "association_group_id" in sources:
        return "association_group_id"
    return "merged_source_id"


def _select_gaussians(components: pd.DataFrame, overview: dict[str, Any]) -> pd.DataFrame:
    if components is None or components.empty:
        return pd.DataFrame()
    max_gauss = int(overview.get("max_gaussians_drawn", 500) or 0)
    if max_gauss <= 0 or len(components) <= max_gauss:
        return components
    if "_peak_flux" in components:
        return components.sort_values("_peak_flux", ascending=False).head(max_gauss)
    return components.sample(n=max_gauss, random_state=17)


def _select_edges(edges: pd.DataFrame, overview: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    if edges is None or edges.empty:
        return pd.DataFrame(), 0
    work = edges.copy()
    decision_col = _edge_decision_column(work)
    score_col = _edge_score_column(work)
    if not overview.get("draw_nonmerged_edges", False) and decision_col in work:
        work = work[_as_bool(work[decision_col])]
    min_score = overview.get("edge_min_score")
    if min_score is not None and score_col in work:
        work = work[pd.to_numeric(work[score_col], errors="coerce") >= float(min_score)]
    total = len(work)
    max_edges = int(overview.get("max_edges_drawn", 300) or 0)
    if max_edges > 0 and len(work) > max_edges:
        work = work.sort_values(score_col, ascending=False).head(max_edges)
    return work, total


def select_important_sources(
    merged_sources: pd.DataFrame,
    overview: dict[str, Any],
) -> tuple[pd.DataFrame, int]:
    """Select merged sources that deserve labels or zoom panels."""

    if merged_sources is None or merged_sources.empty:
        return pd.DataFrame(), 0
    work = merged_sources.copy()
    n_col = _n_components_column(work)
    score_col = _score_column(work)
    for col in [n_col, "LAS_arcsec", score_col, "LAS_beam"]:
        if col in work:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    candidates = pd.Series(False, index=work.index)
    if overview.get("draw_all_labels", False):
        candidates[:] = True
    else:
        if not overview.get("draw_singletons", False) and n_col in work:
            candidates |= work[n_col].fillna(0) >= int(overview.get("label_min_components", 2))
        elif overview.get("draw_singletons", False):
            candidates[:] = True
        if not _is_association_catalog(work) and "double_lobe_candidate_flag" in work:
            candidates |= _as_bool(work["double_lobe_candidate_flag"])
        if "quality_flags" in work:
            candidates |= work["quality_flags"].astype(str).str.contains("large_connected_component", na=False)
        if "flags" in work:
            candidates |= work["flags"].astype(str).str.contains("large_connected_component", na=False)
        if "association_quality" in work:
            candidates |= work["association_quality"].astype(str).isin(["high", "medium", "suspicious", "artifact_risk"])
        if "association_type" in work:
            candidates |= ~work["association_type"].astype(str).isin(["weak_association"])

        top_by = str(overview.get("label_top_by", "LAS_arcsec"))
        if top_by == "merge_confidence" and score_col in work:
            top_by = score_col
        if top_by in work:
            top_n = min(int(overview.get("max_labels", 30)), len(work))
            candidates.loc[work.sort_values(top_by, ascending=False).head(top_n).index] = True
        if score_col in work:
            top_n = min(int(overview.get("max_labels", 30)), len(work))
            candidates.loc[work.sort_values(score_col, ascending=False).head(top_n).index] = True

    selected = work.loc[candidates].copy()
    total_candidates = len(selected)
    if not selected.empty:
        sort_cols = [col for col in [overview.get("label_top_by", "LAS_arcsec"), score_col, n_col] if col in selected]
        if sort_cols:
            selected = selected.sort_values(sort_cols, ascending=False)
        max_labels = int(overview.get("max_labels", 30) or 0)
        if max_labels > 0:
            selected = selected.head(max_labels)
    return selected, total_candidates


def _draw_contours(ax: Any, segmentation: Any, thresholds: list[float], xlim: tuple[int, int] | None = None, ylim: tuple[int, int] | None = None) -> None:
    colors = {2.0: "lime", 2.5: "orange", 3.0: "red", 5.0: "white"}
    for threshold in thresholds:
        idx = int(np.argmin(np.abs(np.asarray(segmentation.thresholds, dtype=float) - float(threshold))))
        mask = segmentation.masks[idx].astype(bool)
        if xlim is not None and ylim is not None:
            x0, x1 = xlim
            y0, y1 = ylim
            mask = mask[y0:y1, x0:x1]
        if mask.any():
            ax.contour(mask, levels=[0.5], colors=[colors.get(float(threshold), "yellow")], linewidths=0.55, alpha=0.65)


def _draw_gaussian_ellipse(ax: Any, row: pd.Series, pixel_scale_arcsec: float, offset: tuple[float, float] = (0, 0)) -> None:
    maj = row.get("_dc_maj", row.get("_maj", np.nan))
    min_axis = row.get("_dc_min", row.get("_min", np.nan))
    pa = row.get("_dc_pa", row.get("_pa", np.nan))
    try:
        maj = float(maj)
        min_axis = float(min_axis)
        pa = float(pa)
    except Exception:
        return
    if not np.isfinite(maj) or not np.isfinite(min_axis) or maj <= 0 or min_axis <= 0:
        return
    ellipse = Ellipse(
        (float(row["x"]) - offset[0], float(row["y"]) - offset[1]),
        width=max(maj / max(pixel_scale_arcsec, 1e-6), 1.0),
        height=max(min_axis / max(pixel_scale_arcsec, 1e-6), 1.0),
        angle=pa,
        fill=False,
        lw=0.7,
        edgecolor="cyan",
        alpha=0.75,
    )
    ax.add_patch(ellipse)


def _text_effects() -> list[Any]:
    return [pe.withStroke(linewidth=1.6, foreground="black", alpha=0.8)]


def _draw_edges(ax: Any, edges: pd.DataFrame, components: pd.DataFrame, offset: tuple[float, float] = (0, 0), draw_scores: bool = False) -> None:
    if edges is None or edges.empty or components is None or components.empty:
        return
    by_idx = components.set_index("component_index")
    for _, edge in edges.iterrows():
        try:
            i = int(edge["component_index_1"])
            j = int(edge["component_index_2"])
        except Exception:
            continue
        if i not in by_idx.index or j not in by_idx.index:
            continue
        ri = by_idx.loc[i]
        rj = by_idx.loc[j]
        x = [float(ri["x"]) - offset[0], float(rj["x"]) - offset[0]]
        y = [float(ri["y"]) - offset[1], float(rj["y"]) - offset[1]]
        edge_type = str(edge.get("edge_type", "strong"))
        color = "white"
        alpha = 0.42
        lw = 0.65
        if edge_type == "weak":
            color = "deepskyblue"
            alpha = 0.55
            lw = 0.55
        elif edge_type == "rejected":
            color = "gray"
            alpha = 0.18
            lw = 0.4
        ax.plot(x, y, color=color, lw=lw, alpha=alpha)
        score_col = _edge_score_column(pd.DataFrame([edge]))
        if draw_scores and score_col in edge:
            xm = 0.5 * (x[0] + x[1])
            ym = 0.5 * (y[0] + y[1])
            ax.text(xm, ym, f"{float(edge[score_col]):.1f}", color="white", fontsize=5, alpha=0.75, path_effects=_text_effects())


def _overview_title(cutout_id: str, components: pd.DataFrame, edges: pd.DataFrame, merged_sources: pd.DataFrame, edges_shown: int, edge_total: int, labels_shown: int, total_candidates: int) -> str:
    if _is_association_catalog(merged_sources):
        decision_col = _edge_decision_column(edges)
        n_edges_merged = int(_as_bool(edges[decision_col]).sum()) if edges is not None and not edges.empty and decision_col in edges else 0
        n_values = pd.to_numeric(merged_sources.get("n_gaussians", pd.Series(dtype=float)), errors="coerce")
        n_multicomp = int((n_values >= 2).sum()) if not merged_sources.empty else 0
        max_component_size = int(n_values.max()) if len(n_values) else 0
        n_only_2 = 0
        if edges is not None and not edges.empty:
            n_only_2 = int((_as_bool(edges.get(decision_col, False)) & _as_bool(edges.get("only_2sigma_connected", False))).sum())
        return (
            f"{cutout_id} | gauss={len(components)} assoc_edges={n_edges_merged} "
            f"groups={len(merged_sources)} multi={n_multicomp} max_group={max_component_size} "
            f"only2={n_only_2}\n"
            f"labels shown: {labels_shown} / total_candidates: {total_candidates} | "
            f"edges shown: {edges_shown} / total_edges: {edge_total}"
        )
    n_edges_merged = int(_as_bool(edges["merge_decision"]).sum()) if edges is not None and not edges.empty and "merge_decision" in edges else 0
    n_multicomp = int((pd.to_numeric(merged_sources.get("n_components", pd.Series(dtype=float)), errors="coerce") >= 2).sum()) if merged_sources is not None and not merged_sources.empty else 0
    max_component_size = int(pd.to_numeric(merged_sources.get("n_components", pd.Series(dtype=float)), errors="coerce").max()) if merged_sources is not None and not merged_sources.empty else 0
    n_only_2 = 0
    if edges is not None and not edges.empty:
        n_only_2 = int(
            (
                _as_bool(edges.get("merge_decision", False))
                & _as_bool(edges.get("connected_at_2sigma", False))
                & ~_as_bool(edges.get("connected_at_2p5sigma", False))
                & ~_as_bool(edges.get("connected_at_3sigma", False))
            ).sum()
        )
    return (
        f"{cutout_id} | gauss={len(components)} merged_edges={n_edges_merged} "
        f"sources={len(merged_sources)} multi={n_multicomp} max_comp={max_component_size} "
        f"only2={n_only_2}\n"
        f"labels shown: {labels_shown} / total_candidates: {total_candidates} | "
        f"edges shown: {edges_shown} / total_edges: {edge_total}"
    )


def plot_cutout_overview(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    edges: pd.DataFrame,
    merged_sources: pd.DataFrame,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> Path:
    """Generate a readable full-cutout overview."""

    viz, overview, _zoom = _viz_config(config)
    percent_clip = tuple(viz.get("percent_clip", [1, 99.5]))
    stretch = viz.get("stretch", "asinh")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels, total_label_candidates = select_important_sources(merged_sources, overview)
    edges_to_draw, total_edge_candidates = _select_edges(edges, overview)
    gauss_to_draw = _select_gaussians(components, overview)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.7), constrained_layout=True)
    image_disp = _display_image(cutout.image, stretch=stretch, percent_clip=percent_clip)
    axes[0].imshow(image_disp, origin="lower", cmap="gray")
    axes[1].imshow(
        segmentation.snr_map,
        origin="lower",
        cmap="magma",
        vmin=np.nanpercentile(segmentation.snr_map, 1),
        vmax=np.nanpercentile(segmentation.snr_map, 99.5),
    )

    for ax in axes:
        _draw_contours(ax, segmentation, overview.get("contour_thresholds", [2.5, 3.0]))
        if not gauss_to_draw.empty:
            ax.scatter(
                gauss_to_draw["x"],
                gauss_to_draw["y"],
                s=float(overview.get("gaussian_marker_size", 4)),
                c="cyan",
                alpha=0.28,
                linewidths=0,
            )
        _draw_edges(ax, edges_to_draw, components)
        for idx, (_, row) in enumerate(labels.iterrows()):
            bbox = _bbox_tuple(row.get("bounding_box", ""))
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            rect = Rectangle((x0, y0), x1 - x0 + 1, y1 - y0 + 1, fill=False, lw=0.75, edgecolor="yellow", alpha=0.65)
            ax.add_patch(rect)
            label = _short_source_id(row.get(_source_id_column(labels), row.get("merged_source_id")), idx)
            ax.text(
                x0,
                max(0, y0 - 4),
                label,
                color="yellow",
                fontsize=6,
                alpha=0.8,
                va="top",
                ha="left",
                path_effects=_text_effects(),
            )
        ax.set_xlim(0, cutout.image.shape[1] - 1)
        ax.set_ylim(0, cutout.image.shape[0] - 1)
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")

    axes[0].set_title("radio")
    axes[1].set_title("S/N")
    fig.suptitle(
        _overview_title(
            cutout.cutout_id,
            components,
            edges,
            merged_sources,
            len(edges_to_draw),
            total_edge_candidates,
            len(labels),
            total_label_candidates,
        ),
        fontsize=9,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _zoom_window(bbox: tuple[float, float, float, float], image_shape: tuple[int, int], zoom: dict[str, Any]) -> tuple[int, int, int, int]:
    height, width = image_shape
    x0, y0, x1, y1 = bbox
    pad = int(zoom.get("padding_pix", 100))
    x0 -= pad
    x1 += pad
    y0 -= pad
    y1 += pad
    min_size = int(zoom.get("min_size_pix", 256))
    max_size = int(zoom.get("max_size_pix", 1024))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    size = max(x1 - x0 + 1, y1 - y0 + 1, min_size)
    size = min(size, max_size)
    x0 = int(round(cx - size / 2))
    x1 = int(round(cx + size / 2))
    y0 = int(round(cy - size / 2))
    y1 = int(round(cy + size / 2))
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(width, x1)
    y1 = min(height, y1)
    return x0, y0, x1, y1


def select_zoom_sources(merged_sources: pd.DataFrame, zoom: dict[str, Any]) -> pd.DataFrame:
    if merged_sources is None or merged_sources.empty or not zoom.get("enabled", True):
        return pd.DataFrame()
    work = merged_sources.copy()
    n_col = _n_components_column(work)
    score_col = _score_column(work)
    for col in [n_col, "LAS_arcsec", "LAS_beam", score_col]:
        if col in work:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    selected = pd.Series(False, index=work.index)
    if n_col in work:
        selected |= work[n_col].fillna(0) >= int(zoom.get("select_min_components", 2))
    if not _is_association_catalog(work) and "double_lobe_candidate_flag" in work:
        selected |= _as_bool(work["double_lobe_candidate_flag"])
    if "association_quality" in work:
        selected |= work["association_quality"].astype(str).isin(["high", "medium", "suspicious", "artifact_risk"])
    if "quality_flags" in work:
        selected |= work["quality_flags"].astype(str).str.contains("large_connected_component", na=False)
    if "LAS_arcsec" in work:
        selected.loc[work.sort_values("LAS_arcsec", ascending=False).head(int(zoom.get("select_top_las", 10))).index] = True
    if score_col in work:
        selected.loc[work.sort_values(score_col, ascending=False).head(int(zoom.get("select_top_confidence", 10))).index] = True
    out = work.loc[selected].copy()
    if out.empty:
        return out
    sort_cols = [col for col in [n_col, "LAS_arcsec", score_col] if col in out]
    out = out.sort_values(sort_cols, ascending=False)
    return out.head(int(zoom.get("max_zoom_per_cutout", 20)))


def plot_source_zoom(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    edges: pd.DataFrame,
    source_row: pd.Series,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
    fallback_idx: int = 0,
) -> Path | None:
    """Generate one local zoom figure for a merged source."""

    viz, _overview, zoom = _viz_config(config)
    bbox = _bbox_tuple(source_row.get("bounding_box", ""))
    if bbox is None:
        return None
    x0, y0, x1, y1 = _zoom_window(bbox, cutout.image.shape, zoom)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    component_ids = _component_ids(source_row)
    src_components = components[components["component_index"].astype(int).isin(component_ids)].copy() if component_ids else pd.DataFrame()
    if edges is not None and not edges.empty and component_ids:
        decision_col = _edge_decision_column(edges)
        src_edges = edges[
            _as_bool(edges[decision_col])
            & edges["component_index_1"].astype(int).isin(component_ids)
            & edges["component_index_2"].astype(int).isin(component_ids)
        ].copy()
    else:
        src_edges = pd.DataFrame()

    image_crop = cutout.image[y0:y1, x0:x1]
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
    ax.imshow(
        _display_image(image_crop, stretch=viz.get("stretch", "asinh"), percent_clip=tuple(viz.get("percent_clip", [1, 99.5]))),
        origin="lower",
        cmap="gray",
    )
    _draw_contours(ax, segmentation, zoom.get("contour_thresholds", [2.0, 2.5, 3.0, 5.0]), xlim=(x0, x1), ylim=(y0, y1))
    if not src_components.empty:
        ax.scatter(src_components["x"] - x0, src_components["y"] - y0, s=18, c="cyan", alpha=0.8, linewidths=0.2, edgecolors="black")
        pixel_scale = float(src_components["pixel_scale_arcsec"].iloc[0]) if "pixel_scale_arcsec" in src_components else 1.5
        for _, comp in src_components.iterrows():
            _draw_gaussian_ellipse(ax, comp, pixel_scale, offset=(x0, y0))
            if zoom.get("draw_gaussian_ids", True):
                ax.text(
                    float(comp["x"]) - x0 + 4,
                    float(comp["y"]) - y0 + 4,
                    str(int(float(comp.get("_gaussian_id", comp.get("component_index", 0))))),
                    color="cyan",
                    fontsize=5,
                    alpha=0.75,
                    path_effects=_text_effects(),
                )
    _draw_edges(ax, src_edges, components, offset=(x0, y0), draw_scores=bool(zoom.get("draw_edge_scores", True)))

    if "association_group_id" in source_row:
        short_id = _short_source_id(source_row.get("association_group_id"), fallback_idx)
        info = (
            f"{cutout.cutout_id} {short_id} | n={int(source_row.get('n_gaussians', 0))} "
            f"LAS={float(source_row.get('LAS_arcsec', np.nan)):.1f}\" "
            f"beam={float(source_row.get('LAS_beam', np.nan)):.1f} "
            f"quality={source_row.get('association_quality', 'low')} "
            f"type={source_row.get('association_type', 'weak_association')}\n"
            f"score={float(source_row.get('association_score_mean', np.nan)):.2f} "
            f"strong={int(source_row.get('n_strong_edges', 0))} "
            f"weak={int(source_row.get('n_weak_edges', 0))} "
            f"only2={int(source_row.get('n_only_2sigma_edges', 0))} "
            f"flags={source_row.get('artifact_risk_flags', '')}"
        )
    else:
        short_id = _short_source_id(source_row.get("merged_source_id"), fallback_idx)
        info = (
            f"{cutout.cutout_id} {short_id} | n={int(source_row.get('n_components', 0))} "
            f"LAS={float(source_row.get('LAS_arcsec', np.nan)):.1f}\" "
            f"conf={float(source_row.get('merge_confidence', np.nan)):.2f}"
        )
    ax.set_title(info, fontsize=9)
    ax.set_xlim(0, x1 - x0 - 1)
    ax.set_ylim(0, y1 - y0 - 1)
    ax.set_xlabel(f"x [{x0}:{x1}]")
    ax.set_ylabel(f"y [{y0}:{y1}]")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def plot_cutout_result(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    edges: pd.DataFrame,
    merged_sources: pd.DataFrame,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> Path:
    """Backward-compatible entry point for pipeline overview figures."""

    return plot_cutout_overview(cutout, segmentation, components, edges, merged_sources, output_path, config)


def plot_cutout_all(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    edges: pd.DataFrame,
    merged_sources: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    overview_only: bool = False,
    zoom_only: bool = False,
) -> dict[str, list[Path]]:
    """Generate overview and zoom figures for one cutout."""

    _viz, _overview, zoom = _viz_config(config)
    output_dir = Path(output_dir)
    overview_paths: list[Path] = []
    zoom_paths: list[Path] = []
    if not zoom_only:
        overview_path = output_dir / "overview" / f"{cutout.cutout_id}.png"
        overview_paths.append(plot_cutout_overview(cutout, segmentation, components, edges, merged_sources, overview_path, config))
    if not overview_only and zoom.get("enabled", True):
        zoom_sources = select_zoom_sources(merged_sources, zoom)
        for idx, (_, row) in enumerate(zoom_sources.iterrows()):
            short_id = _short_source_id(row.get(_source_id_column(zoom_sources), row.get("merged_source_id")), idx)
            path = output_dir / "zoom" / f"{cutout.cutout_id}_{short_id}.png"
            written = plot_source_zoom(cutout, segmentation, components, edges, row, path, config, fallback_idx=idx)
            if written is not None:
                zoom_paths.append(written)
    return {"overview": overview_paths, "zoom": zoom_paths}


def _local_group_ids(row: pd.Series) -> set[str]:
    if "local_group_ids" in row:
        raw = row.get("local_group_ids", "")
        return {item.strip() for item in str(raw).split(",") if item.strip()}
    ids = {str(row.get("local_group_id_1", "")).strip(), str(row.get("local_group_id_2", "")).strip()}
    return {item for item in ids if item and item.lower() != "nan"}


def _int_list(value: Any) -> list[int]:
    ids: list[int] = []
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ids
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ids.append(int(float(item)))
        except Exception:
            pass
    return ids


def _parent_edge_score_column(edges: pd.DataFrame) -> str:
    return "parent_score" if edges is not None and not edges.empty and "parent_score" in edges else "association_score"


def _draw_local_boxes(
    ax: Any,
    local_groups: pd.DataFrame,
    offset: tuple[float, float] = (0, 0),
    label: bool = True,
) -> None:
    if local_groups is None or local_groups.empty:
        return
    for idx, (_, row) in enumerate(local_groups.iterrows()):
        bbox = _bbox_tuple(row.get("bounding_box", ""))
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        quality = str(row.get("local_quality", "low"))
        color = "gold"
        if quality == "suspicious":
            color = "magenta"
        elif quality == "artifact_risk":
            color = "tomato"
        elif quality in {"high", "medium"}:
            color = "lime"
        rect = Rectangle((x0 - offset[0], y0 - offset[1]), x1 - x0 + 1, y1 - y0 + 1, fill=False, lw=0.9, edgecolor=color, alpha=0.8)
        ax.add_patch(rect)
        if label:
            text = str(row.get("local_group_id", f"l{idx:03d}")).rsplit("_", 1)[-1]
            ax.text(
                x0 - offset[0],
                max(0, y0 - offset[1] - 4),
                text,
                color=color,
                fontsize=6,
                va="top",
                ha="left",
                path_effects=_text_effects(),
            )


def _draw_parent_links(
    ax: Any,
    parent_edges: pd.DataFrame,
    local_groups: pd.DataFrame,
    offset: tuple[float, float] = (0, 0),
    draw_scores: bool = False,
) -> None:
    if parent_edges is None or parent_edges.empty or local_groups is None or local_groups.empty:
        return
    by_id = local_groups.set_index("local_group_id")
    if "is_default_candidate" in parent_edges:
        draw_mask = parent_edges["is_default_candidate"].astype(bool)
    elif "parent_candidate_quality" in parent_edges:
        draw_mask = parent_edges["parent_candidate_quality"].astype(str).isin(["high", "medium"])
    else:
        draw_mask = _as_bool(parent_edges.get("parent_edge_decision", pd.Series(False, index=parent_edges.index)))
    if "needs_visual_check" in parent_edges:
        draw_mask |= parent_edges["needs_visual_check"].astype(bool)
    for _, edge in parent_edges[draw_mask].iterrows():
        left = str(edge.get("local_group_id_1"))
        right = str(edge.get("local_group_id_2"))
        if left not in by_id.index or right not in by_id.index:
            continue
        ri = by_id.loc[left]
        rj = by_id.loc[right]
        x = [float(ri["centroid_x"]) - offset[0], float(rj["centroid_x"]) - offset[0]]
        y = [float(ri["centroid_y"]) - offset[1], float(rj["centroid_y"]) - offset[1]]
        quality = str(edge.get("parent_candidate_quality", edge.get("parent_edge_type", "medium")))
        color = "yellow" if quality == "high" else "deepskyblue"
        lw = 1.3 if quality == "high" else 0.85
        alpha = 0.84 if quality == "high" else 0.64
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, linestyle="-" if quality == "high" else "--")
        if draw_scores:
            xm = 0.5 * (x[0] + x[1])
            ym = 0.5 * (y[0] + y[1])
            ax.text(xm, ym, f"{float(edge.get('parent_candidate_score', edge.get('parent_score', np.nan))):.1f}", color=color, fontsize=6, path_effects=_text_effects())


def plot_parent_cutout_overview(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    parent_groups: pd.DataFrame,
    parent_edges: pd.DataFrame,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> Path:
    """Generate an overview with local boxes and parent links."""

    viz, overview, _zoom = _viz_config(config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.7), constrained_layout=True)
    axes[0].imshow(_display_image(cutout.image, stretch=viz.get("stretch", "asinh"), percent_clip=tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    axes[1].imshow(
        segmentation.snr_map,
        origin="lower",
        cmap="magma",
        vmin=np.nanpercentile(segmentation.snr_map, 1),
        vmax=np.nanpercentile(segmentation.snr_map, 99.5),
    )
    gauss_to_draw = _select_gaussians(components, overview)
    for ax in axes:
        _draw_contours(ax, segmentation, overview.get("contour_thresholds", [2.5, 3.0]))
        if not gauss_to_draw.empty:
            ax.scatter(gauss_to_draw["x"], gauss_to_draw["y"], s=float(overview.get("gaussian_marker_size", 4)), c="cyan", alpha=0.24, linewidths=0)
        _draw_local_boxes(ax, local_groups)
        _draw_parent_links(ax, parent_groups if parent_groups is not None else parent_edges, local_groups, draw_scores=False)
        if parent_groups is not None and not parent_groups.empty:
            shown = parent_groups.sort_values(["parent_candidate_quality", "group_distance_beam"], ascending=[True, False]).head(int(overview.get("max_labels", 30)))
            for idx, (_, row) in enumerate(shown.iterrows()):
                ids = _local_group_ids(row)
                members = local_groups[local_groups["local_group_id"].astype(str).isin(ids)] if ids else pd.DataFrame()
                if members.empty:
                    continue
                x = float(pd.to_numeric(members["centroid_x"], errors="coerce").mean())
                y = float(pd.to_numeric(members["centroid_y"], errors="coerce").mean())
                label = str(row.get("parent_candidate_id", row.get("parent_group_id", f"pc{idx:03d}"))).rsplit("_", 1)[-1]
                ax.text(x, y, label, color="white", fontsize=7, ha="center", va="center", path_effects=_text_effects())
        ax.set_xlim(0, cutout.image.shape[1] - 1)
        ax.set_ylim(0, cutout.image.shape[0] - 1)
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
    axes[0].set_title("radio")
    axes[1].set_title("S/N")
    n_parent_linked = int(len(parent_groups)) if parent_groups is not None and not parent_groups.empty else 0
    n_suspicious = int((local_groups.get("local_quality", pd.Series(dtype=str)).astype(str) == "suspicious").sum()) if local_groups is not None and not local_groups.empty else 0
    fig.suptitle(
        f"{cutout.cutout_id} | local={len(local_groups)} suspicious_local={n_suspicious} "
        f"large_parent_candidates={n_parent_linked}",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def select_local_zoom_groups(local_groups: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    if local_groups is None or local_groups.empty:
        return pd.DataFrame()
    _viz, _overview, zoom = _viz_config(config)
    work = local_groups.copy()
    selected = pd.Series(False, index=work.index)
    selected |= work.get("needs_visual_check", pd.Series(False, index=work.index)).astype(bool)
    selected |= pd.to_numeric(work.get("n_gaussians", pd.Series(dtype=float)), errors="coerce").fillna(0) >= int(zoom.get("select_min_components", 2))
    if "LAS_beam" in work:
        selected.loc[pd.to_numeric(work["LAS_beam"], errors="coerce").sort_values(ascending=False).head(int(zoom.get("select_top_las", 10))).index] = True
    out = work[selected].copy()
    if out.empty:
        return out
    return out.sort_values(["needs_visual_check", "LAS_beam", "n_gaussians"], ascending=False).head(int(zoom.get("max_zoom_per_cutout", 20)))


def select_parent_zoom_groups(parent_groups: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    if parent_groups is None or parent_groups.empty:
        return pd.DataFrame()
    _viz, _overview, zoom = _viz_config(config)
    work = parent_groups.copy()
    selected = work.get("parent_candidate_quality", pd.Series(dtype=str)).astype(str).isin(["high", "medium"])
    selected |= work.get("needs_visual_check", pd.Series(False, index=work.index)).astype(bool)
    out = work[selected].copy()
    if out.empty:
        return out
    return out.sort_values(["parent_candidate_score", "group_distance_beam"], ascending=False).head(int(zoom.get("max_zoom_per_cutout", 20)))


def plot_local_zoom(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_edges: pd.DataFrame,
    local_group_row: pd.Series,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
    fallback_idx: int = 0,
) -> Path | None:
    """Generate one local-group zoom panel."""

    viz, _overview, zoom = _viz_config(config)
    bbox = _bbox_tuple(local_group_row.get("bounding_box", ""))
    if bbox is None:
        return None
    x0, y0, x1, y1 = _zoom_window(bbox, cutout.image.shape, zoom)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    component_ids = _component_ids(local_group_row)
    src_components = components[components["component_index"].astype(int).isin(component_ids)].copy() if component_ids else pd.DataFrame()
    if local_edges is not None and not local_edges.empty and component_ids:
        decision = _as_bool(local_edges.get("local_edge_decision", local_edges.get("association_decision", pd.Series(False, index=local_edges.index))))
        src_edges = local_edges[
            decision
            & local_edges["component_index_1"].astype(int).isin(component_ids)
            & local_edges["component_index_2"].astype(int).isin(component_ids)
        ].copy()
        if "local_edge_type" in src_edges:
            src_edges["edge_type"] = src_edges["local_edge_type"]
    else:
        src_edges = pd.DataFrame()
    image_crop = cutout.image[y0:y1, x0:x1]
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
    ax.imshow(_display_image(image_crop, stretch=viz.get("stretch", "asinh"), percent_clip=tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    _draw_contours(ax, segmentation, zoom.get("contour_thresholds", [2.0, 2.5, 3.0, 5.0]), xlim=(x0, x1), ylim=(y0, y1))
    if not src_components.empty:
        ax.scatter(src_components["x"] - x0, src_components["y"] - y0, s=18, c="cyan", alpha=0.8, linewidths=0.2, edgecolors="black")
        pixel_scale = float(src_components["pixel_scale_arcsec"].iloc[0]) if "pixel_scale_arcsec" in src_components else 1.5
        for _, comp in src_components.iterrows():
            _draw_gaussian_ellipse(ax, comp, pixel_scale, offset=(x0, y0))
            if zoom.get("draw_gaussian_ids", True):
                ax.text(float(comp["x"]) - x0 + 4, float(comp["y"]) - y0 + 4, str(int(float(comp.get("_gaussian_id", comp.get("component_index", 0))))), color="cyan", fontsize=5, alpha=0.75, path_effects=_text_effects())
    _draw_edges(ax, src_edges, components, offset=(x0, y0), draw_scores=bool(zoom.get("draw_edge_scores", True)))
    short_id = str(local_group_row.get("local_group_id", f"l{fallback_idx:03d}")).rsplit("_", 1)[-1]
    title = (
        f"{cutout.cutout_id} {short_id} | local_quality={local_group_row.get('local_quality', 'low')} "
        f"risk={float(local_group_row.get('local_overmerge_risk_score', np.nan)):.2f} "
        f"n={int(local_group_row.get('n_gaussians', 0))} beam={float(local_group_row.get('LAS_beam', np.nan)):.1f}\n"
        f"saddle={float(local_group_row.get('saddle_to_peak_ratio', np.nan)):.2f} "
        f"ridge_gap={float(local_group_row.get('ridge_gap_fraction', np.nan)):.2f} "
        f"weak={float(local_group_row.get('weak_edge_fraction', np.nan)):.2f} "
        f"split={local_group_row.get('split_reason', '')}"
    )
    ax.set_title(title, fontsize=8.5)
    ax.set_xlim(0, x1 - x0 - 1)
    ax.set_ylim(0, y1 - y0 - 1)
    ax.set_xlabel(f"x [{x0}:{x1}]")
    ax.set_ylabel(f"y [{y0}:{y1}]")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def plot_parent_zoom(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    parent_edges: pd.DataFrame,
    parent_group_row: pd.Series,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
    fallback_idx: int = 0,
) -> Path | None:
    """Generate one parent-group zoom panel."""

    viz, _overview, zoom = _viz_config(config)
    ids = _local_group_ids(parent_group_row)
    member_groups = local_groups[local_groups["local_group_id"].astype(str).isin(ids)].copy() if ids else pd.DataFrame()
    if member_groups.empty:
        return None
    boxes = [_bbox_tuple(row.get("bounding_box", "")) for _, row in member_groups.iterrows()]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    bbox = (min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes))
    x0, y0, x1, y1 = _zoom_window(bbox, cutout.image.shape, zoom)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    member_ids = set(member_groups["local_group_id"].astype(str).tolist())
    edge_mask = pd.Series(False, index=parent_edges.index) if parent_edges is not None else pd.Series(dtype=bool)
    if parent_edges is not None and not parent_edges.empty:
        edge_mask = (
            parent_edges["local_group_id_1"].astype(str).isin(member_ids)
            & parent_edges["local_group_id_2"].astype(str).isin(member_ids)
        )
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 7.4), constrained_layout=True)
    ax.imshow(_display_image(cutout.image[y0:y1, x0:x1], stretch=viz.get("stretch", "asinh"), percent_clip=tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    _draw_contours(ax, segmentation, zoom.get("contour_thresholds", [2.0, 2.5, 3.0, 5.0]), xlim=(x0, x1), ylim=(y0, y1))
    comp_ids: set[int] = set()
    for _, row in member_groups.iterrows():
        comp_ids.update(_int_list(row.get("component_ids", "")))
    src_components = components[components["component_index"].astype(int).isin(comp_ids)].copy() if comp_ids else pd.DataFrame()
    if not src_components.empty:
        ax.scatter(src_components["x"] - x0, src_components["y"] - y0, s=16, c="cyan", alpha=0.74, linewidths=0.2, edgecolors="black")
    _draw_local_boxes(ax, member_groups, offset=(x0, y0), label=True)
    _draw_parent_links(ax, pd.DataFrame([parent_group_row]), local_groups, offset=(x0, y0), draw_scores=True)
    core_ids = set(_int_list(parent_group_row.get("core_candidate_ids", "")))
    if core_ids and not components.empty:
        core = components[components["component_index"].astype(int).isin(core_ids)]
        if not core.empty:
            ax.scatter(core["x"] - x0, core["y"] - y0, s=42, facecolors="none", edgecolors="white", linewidths=1.2)
            for _, comp in core.iterrows():
                ax.text(float(comp["x"]) - x0 + 5, float(comp["y"]) - y0 + 5, "core", color="white", fontsize=6, path_effects=_text_effects())
    short_id = str(parent_group_row.get("parent_candidate_id", f"pc{fallback_idx:03d}")).rsplit("_", 1)[-1]
    title = (
        f"{cutout.cutout_id} {short_id} | large-scale parent candidate "
        f"quality={parent_group_row.get('parent_candidate_quality', 'low')}\n"
        f"score={float(parent_group_row.get('parent_candidate_score', np.nan)):.2f} "
        f"distance_beam={float(parent_group_row.get('group_distance_beam', np.nan)):.1f} "
        f"core={parent_group_row.get('core_candidate_near_midpoint', False)}"
    )
    ax.set_title(title, fontsize=8.5)
    ax.set_xlim(0, x1 - x0 - 1)
    ax.set_ylim(0, y1 - y0 - 1)
    ax.set_xlabel(f"x [{x0}:{x1}]")
    ax.set_ylabel(f"y [{y0}:{y1}]")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def plot_parent_cutout_all(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_edges: pd.DataFrame,
    local_groups: pd.DataFrame,
    parent_edges: pd.DataFrame,
    parent_groups: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, list[Path]]:
    """Generate all overview, local zoom, and parent zoom figures."""

    output_dir = Path(output_dir)
    paths: dict[str, list[Path]] = {"overview": [], "local_zoom": [], "parent_zoom": []}
    overview_path = output_dir / "overview" / f"{cutout.cutout_id}.png"
    paths["overview"].append(plot_parent_cutout_overview(cutout, segmentation, components, local_groups, parent_groups, parent_edges, overview_path, config))
    for idx, (_, row) in enumerate(select_local_zoom_groups(local_groups, config).iterrows()):
        short_id = str(row.get("local_group_id", f"l{idx:03d}")).rsplit("_", 1)[-1]
        written = plot_local_zoom(cutout, segmentation, components, local_edges, row, output_dir / "local_zoom" / f"{cutout.cutout_id}_{short_id}.png", config, fallback_idx=idx)
        if written is not None:
            paths["local_zoom"].append(written)
    for idx, (_, row) in enumerate(select_parent_zoom_groups(parent_groups, config).iterrows()):
        short_id = str(row.get("parent_candidate_id", f"pc{idx:03d}")).rsplit("_", 1)[-1]
        written = plot_parent_zoom(cutout, segmentation, components, local_groups, parent_edges, row, output_dir / "parent_zoom" / f"{cutout.cutout_id}_{short_id}.png", config, fallback_idx=idx)
        if written is not None:
            paths["parent_zoom"].append(written)
    return paths


def _draw_parent_seed_selection(
    ax: Any,
    candidates: pd.DataFrame,
    local_groups: pd.DataFrame,
    offset: tuple[float, float] = (0, 0),
    draw_scores: bool = False,
    max_links: int = 30,
    show_debug: bool = False,
) -> None:
    if candidates is None or candidates.empty or local_groups is None or local_groups.empty:
        return
    by_id = local_groups.set_index("association_group_id")
    if show_debug:
        work = candidates.copy()
    else:
        work = candidates[candidates["parent_candidate_quality"].astype(str).isin(["high", "medium"])].copy()
    if work.empty:
        return
    work = work.sort_values(["parent_score", "box_gap_beam"], ascending=[False, True]).head(max_links)
    for _, row in work.iterrows():
        left = str(row.get("local_group_id_1"))
        right = str(row.get("local_group_id_2"))
        if left not in by_id.index or right not in by_id.index:
            continue
        group_a = by_id.loc[left]
        group_b = by_id.loc[right]
        x = [float(group_a["centroid_x"]) - offset[0], float(group_b["centroid_x"]) - offset[0]]
        y = [float(group_a["centroid_y"]) - offset[1], float(group_b["centroid_y"]) - offset[1]]
        quality = str(row.get("parent_candidate_quality", "medium"))
        color = "yellow" if quality == "high" else "deepskyblue"
        if quality not in {"high", "medium"}:
            color = "tomato"
        lw = 1.4 if quality == "high" else 0.95
        alpha = 0.86 if quality == "high" else (0.68 if quality == "medium" else 0.25)
        ax.plot(x, y, color=color, lw=lw, alpha=alpha, linestyle="-" if quality == "high" else "--")
        if draw_scores:
            xm = 0.5 * (x[0] + x[1])
            ym = 0.5 * (y[0] + y[1])
            ax.text(
                xm,
                ym,
                f"gap={float(row.get('box_gap_beam', np.nan)):.1f} score={float(row.get('parent_score', np.nan)):.1f}",
                color=color,
                fontsize=6,
                path_effects=_text_effects(),
            )


def plot_parent_seed_cutout_overview(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    parent_candidates: pd.DataFrame,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
    show_debug_parent_links: bool = False,
) -> Path:
    """Generate a parent-seed overview with local boxes and capped parent links."""

    viz, overview, _zoom = _viz_config(config)
    seed_cfg = (config or {}).get("parent_seed_selection", {}) or {}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 8.0), constrained_layout=True)
    ax.imshow(_display_image(cutout.image, stretch=viz.get("stretch", "asinh"), percent_clip=tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    _draw_contours(ax, segmentation, overview.get("contour_thresholds", [2.5, 3.0]))
    gauss_to_draw = _select_gaussians(components, overview)
    if not gauss_to_draw.empty:
        ax.scatter(gauss_to_draw["x"], gauss_to_draw["y"], s=float(overview.get("gaussian_marker_size", 4)), c="cyan", alpha=0.22, linewidths=0)
    local_boxes = local_groups.copy()
    if "local_quality" not in local_boxes:
        local_boxes["local_quality"] = local_boxes.get("association_quality", "low")
    if "local_group_id" not in local_boxes:
        local_boxes["local_group_id"] = local_boxes.get("association_group_id", "")
    _draw_local_boxes(ax, local_boxes, label=True)
    _draw_parent_seed_selection(
        ax,
        parent_candidates,
        local_groups,
        draw_scores=False,
        max_links=int(seed_cfg.get("max_parent_candidates_per_cutout", 30)),
        show_debug=show_debug_parent_links,
    )
    ax.set_xlim(0, cutout.image.shape[1] - 1)
    ax.set_ylim(0, cutout.image.shape[0] - 1)
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.set_title(
        f"{cutout.cutout_id} | local={len(local_groups)} | parent-seed parent_candidates={len(parent_candidates)}",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def select_parent_seed_zoom_candidates(parent_candidates: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    if parent_candidates is None or parent_candidates.empty:
        return pd.DataFrame()
    _viz, _overview, zoom = _viz_config(config)
    work = parent_candidates[parent_candidates["parent_candidate_quality"].astype(str).isin(["high", "medium"])].copy()
    if work.empty:
        return work
    return work.sort_values(["parent_score", "box_gap_beam"], ascending=[False, True]).head(int(zoom.get("max_zoom_per_cutout", 20)))


def plot_parent_seed_zoom(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    parent_candidate_row: pd.Series,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
    fallback_idx: int = 0,
) -> Path | None:
    """Generate one parent-seed parent-link candidate zoom panel."""

    viz, _overview, zoom = _viz_config(config)
    ids = _local_group_ids(parent_candidate_row)
    member_groups = local_groups[local_groups["association_group_id"].astype(str).isin(ids)].copy() if ids else pd.DataFrame()
    if member_groups.empty:
        return None
    boxes = [_bbox_tuple(row.get("bounding_box", "")) for _, row in member_groups.iterrows()]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    bbox = (min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes))
    x0, y0, x1, y1 = _zoom_window(bbox, cutout.image.shape, zoom)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    local_boxes = member_groups.copy()
    local_boxes["local_quality"] = local_boxes.get("association_quality", "low")
    local_boxes["local_group_id"] = local_boxes.get("association_group_id", "")
    comp_ids: set[int] = set()
    for _, row in member_groups.iterrows():
        comp_ids.update(_int_list(row.get("component_ids", "")))
    src_components = components[components["component_index"].astype(int).isin(comp_ids)].copy() if comp_ids else pd.DataFrame()
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 7.4), constrained_layout=True)
    ax.imshow(_display_image(cutout.image[y0:y1, x0:x1], stretch=viz.get("stretch", "asinh"), percent_clip=tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    _draw_contours(ax, segmentation, zoom.get("contour_thresholds", [2.0, 2.5, 3.0, 5.0]), xlim=(x0, x1), ylim=(y0, y1))
    if not src_components.empty:
        ax.scatter(src_components["x"] - x0, src_components["y"] - y0, s=16, c="cyan", alpha=0.75, linewidths=0.2, edgecolors="black")
    _draw_local_boxes(ax, local_boxes, offset=(x0, y0), label=True)
    _draw_parent_seed_selection(ax, pd.DataFrame([parent_candidate_row]), local_groups, offset=(x0, y0), draw_scores=True, max_links=1, show_debug=True)
    short_id = str(parent_candidate_row.get("parent_candidate_id", f"pc{fallback_idx:03d}")).rsplit("_", 1)[-1]
    rejection = str(parent_candidate_row.get("rejection_reason", ""))
    title = (
        f"{cutout.cutout_id} {short_id} | parent-seed parent link candidate "
        f"quality={parent_candidate_row.get('parent_candidate_quality', 'medium')}\n"
        f"groups={parent_candidate_row.get('local_group_id_1')} + {parent_candidate_row.get('local_group_id_2')} "
        f"gap_beam={float(parent_candidate_row.get('box_gap_beam_robust', parent_candidate_row.get('box_gap_beam', np.nan))):.1f} "
        f"score={float(parent_candidate_row.get('parent_score', np.nan)):.2f} "
        f"reason={rejection}"
    )
    ax.set_title(title, fontsize=8.2)
    ax.set_xlim(0, x1 - x0 - 1)
    ax.set_ylim(0, y1 - y0 - 1)
    ax.set_xlabel(f"x [{x0}:{x1}]")
    ax.set_ylabel(f"y [{y0}:{y1}]")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def plot_parent_seed_cutout_all(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    parent_candidates: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
    show_debug_parent_links: bool = False,
) -> dict[str, list[Path]]:
    """Generate parent-seed overview and parent zoom figures."""

    output_dir = Path(output_dir)
    paths: dict[str, list[Path]] = {"overview": [], "parent_zoom": []}
    overview_path = output_dir / "overview" / f"{cutout.cutout_id}.png"
    paths["overview"].append(plot_parent_seed_cutout_overview(cutout, segmentation, components, local_groups, parent_candidates, overview_path, config, show_debug_parent_links=show_debug_parent_links))
    for idx, (_, row) in enumerate(select_parent_seed_zoom_candidates(parent_candidates, config).iterrows()):
        short_id = str(row.get("parent_candidate_id", f"pc{idx:03d}")).rsplit("_", 1)[-1]
        written = plot_parent_seed_zoom(
            cutout,
            segmentation,
            components,
            local_groups,
            row,
            output_dir / "parent_zoom" / f"{cutout.cutout_id}_{short_id}.png",
            config,
            fallback_idx=idx,
        )
        if written is not None:
            paths["parent_zoom"].append(written)
    return paths


def _draw_host_supported_parent_links(
    ax: Any,
    candidates: pd.DataFrame,
    local_groups: pd.DataFrame,
    offset: tuple[float, float] = (0, 0),
    draw_scores: bool = False,
) -> None:
    if candidates is None or candidates.empty or local_groups is None or local_groups.empty:
        return
    by_id = local_groups.set_index("association_group_id")
    work = candidates[candidates["parent_candidate_quality"].astype(str).isin(["high", "medium"])].copy()
    if work.empty:
        return
    for _, row in work.sort_values("parent_score_final", ascending=False).iterrows():
        left = str(row.get("local_group_id_1"))
        right = str(row.get("local_group_id_2"))
        if left not in by_id.index or right not in by_id.index:
            continue
        g1 = by_id.loc[left]
        g2 = by_id.loc[right]
        x = [float(g1["centroid_x"]) - offset[0], float(g2["centroid_x"]) - offset[0]]
        y = [float(g1["centroid_y"]) - offset[1], float(g2["centroid_y"]) - offset[1]]
        color = "yellow" if str(row.get("parent_candidate_quality")) == "high" else "deepskyblue"
        ax.plot(x, y, color=color, lw=1.35, alpha=0.82, linestyle="-" if color == "yellow" else "--")
        if draw_scores:
            ax.text(
                0.5 * (x[0] + x[1]),
                0.5 * (y[0] + y[1]),
                f"host={row.get('best_host_catalog', '')} score={float(row.get('best_host_score', np.nan)):.1f}",
                color=color,
                fontsize=6,
                path_effects=_text_effects(),
            )


def plot_host_supported_cutout_overview(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    parent_candidates: pd.DataFrame,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> Path:
    """Generate host-supported parent-linking overview with only host-gated high/medium links."""

    viz, overview, _zoom = _viz_config(config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 8.0), constrained_layout=True)
    ax.imshow(_display_image(cutout.image, stretch=viz.get("stretch", "asinh"), percent_clip=tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    _draw_contours(ax, segmentation, overview.get("contour_thresholds", [2.5, 3.0]))
    gauss_to_draw = _select_gaussians(components, overview)
    if not gauss_to_draw.empty:
        ax.scatter(gauss_to_draw["x"], gauss_to_draw["y"], s=float(overview.get("gaussian_marker_size", 4)), c="cyan", alpha=0.22, linewidths=0)
    local_boxes = local_groups.copy()
    local_boxes["local_quality"] = local_boxes.get("association_quality", "low")
    local_boxes["local_group_id"] = local_boxes.get("association_group_id", "")
    _draw_local_boxes(ax, local_boxes, label=True)
    _draw_host_supported_parent_links(ax, parent_candidates, local_groups, draw_scores=False)
    ax.set_xlim(0, cutout.image.shape[1] - 1)
    ax.set_ylim(0, cutout.image.shape[0] - 1)
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.set_title(
        f"{cutout.cutout_id} | local={len(local_groups)} | host-supported parent-linking host_gated={len(parent_candidates)}",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_host_supported_parent_zoom(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    parent_candidate_row: pd.Series,
    host_candidates: pd.DataFrame,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
    fallback_idx: int = 0,
) -> Path | None:
    """Generate one host-supported parent-linking parent zoom with midpoint host evidence."""

    viz, _overview, zoom = _viz_config(config)
    ids = _local_group_ids(parent_candidate_row)
    member_groups = local_groups[local_groups["association_group_id"].astype(str).isin(ids)].copy() if ids else pd.DataFrame()
    if member_groups.empty:
        return None
    boxes = [_bbox_tuple(row.get("bounding_box", "")) for _, row in member_groups.iterrows()]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    bbox = (min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes))
    x0, y0, x1, y1 = _zoom_window(bbox, cutout.image.shape, zoom)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    local_boxes = member_groups.copy()
    local_boxes["local_quality"] = local_boxes.get("association_quality", "low")
    local_boxes["local_group_id"] = local_boxes.get("association_group_id", "")
    comp_ids: set[int] = set()
    for _, row in member_groups.iterrows():
        comp_ids.update(_int_list(row.get("component_ids", "")))
    src_components = components[components["component_index"].astype(int).isin(comp_ids)].copy() if comp_ids else pd.DataFrame()
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 7.6), constrained_layout=True)
    ax.imshow(_display_image(cutout.image[y0:y1, x0:x1], stretch=viz.get("stretch", "asinh"), percent_clip=tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    _draw_contours(ax, segmentation, zoom.get("contour_thresholds", [2.0, 2.5, 3.0, 5.0]), xlim=(x0, x1), ylim=(y0, y1))
    if not src_components.empty:
        ax.scatter(src_components["x"] - x0, src_components["y"] - y0, s=16, c="cyan", alpha=0.75, linewidths=0.2, edgecolors="black")
    _draw_local_boxes(ax, local_boxes, offset=(x0, y0), label=True)
    _draw_host_supported_parent_links(ax, pd.DataFrame([parent_candidate_row]), local_groups, offset=(x0, y0), draw_scores=True)
    # Approximate midpoint in pixel space from the two local centroids.
    if len(member_groups) >= 2:
        mx = float(member_groups["centroid_x"].mean()) - x0
        my = float(member_groups["centroid_y"].mean()) - y0
        ax.scatter([mx], [my], s=54, marker="+", c="white", linewidths=1.5)
        radius_arcsec = float(parent_candidate_row.get("host_search_radius_arcsec", 10.0)) if "host_search_radius_arcsec" in parent_candidate_row else 10.0
        pixel_scale = float(components["pixel_scale_arcsec"].iloc[0]) if components is not None and not components.empty and "pixel_scale_arcsec" in components else 1.5
        ax.add_patch(plt.Circle((mx, my), radius_arcsec / max(pixel_scale, 1e-6), fill=False, color="white", alpha=0.55, lw=0.9))
    pid = str(parent_candidate_row.get("parent_candidate_id", ""))
    host_rows = host_candidates[host_candidates["parent_candidate_id"].astype(str) == pid].copy() if host_candidates is not None and not host_candidates.empty else pd.DataFrame()
    if not host_rows.empty and len(member_groups) >= 2:
        # For display, place the best host relative to the midpoint using the
        # catalogued midpoint offset; exact WCS plotting is unnecessary for this
        # review overlay.
        best = host_rows.sort_values("host_score", ascending=False).iloc[0]
        mx = float(member_groups["centroid_x"].mean()) - x0
        my = float(member_groups["centroid_y"].mean()) - y0
        ax.scatter([mx], [my], s=42, facecolors="none", edgecolors="magenta", linewidths=1.4)
        ax.text(
            mx + 6,
            my + 6,
            f"host={best.get('host_catalog', '')}\nhost_score={float(best.get('host_score', np.nan)):.1f}\nhost_sep={float(best.get('host_sep_midpoint_arcsec', np.nan)):.1f} arcsec\nhost_quality={best.get('host_quality', '')}\nW1={float(best.get('W1', np.nan)):.2f} W2={float(best.get('W2', np.nan)):.2f}",
            color="magenta",
            fontsize=6,
            path_effects=_text_effects(),
        )
    short_id = str(parent_candidate_row.get("parent_candidate_id", f"pc{fallback_idx:03d}")).rsplit("_", 1)[-1]
    title = (
        f"{cutout.cutout_id} {short_id} | host-supported parent-linking host-gated parent candidate "
        f"quality={parent_candidate_row.get('parent_candidate_quality', 'medium')}\n"
        f"host={parent_candidate_row.get('best_host_catalog', '')} "
        f"host_score={float(parent_candidate_row.get('best_host_score', np.nan)):.2f} "
        f"host_quality={parent_candidate_row.get('host_quality', '')}"
    )
    ax.set_title(title, fontsize=8.2)
    ax.set_xlim(0, x1 - x0 - 1)
    ax.set_ylim(0, y1 - y0 - 1)
    ax.set_xlabel(f"x [{x0}:{x1}]")
    ax.set_ylabel(f"y [{y0}:{y1}]")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def plot_host_supported_cutout_all(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    parent_candidates: pd.DataFrame,
    host_candidates: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, list[Path]]:
    """Generate host-supported parent-linking overview and host-aware parent zoom figures."""

    output_dir = Path(output_dir)
    paths: dict[str, list[Path]] = {"overview": [], "parent_zoom": []}
    overview_path = output_dir / "overview" / f"{cutout.cutout_id}.png"
    paths["overview"].append(plot_host_supported_cutout_overview(cutout, segmentation, components, local_groups, parent_candidates, overview_path, config))
    if parent_candidates is None or parent_candidates.empty:
        return paths
    for idx, (_, row) in enumerate(parent_candidates.sort_values("parent_score_final", ascending=False).iterrows()):
        short_id = str(row.get("parent_candidate_id", f"pc{idx:03d}")).rsplit("_", 1)[-1]
        written = plot_host_supported_parent_zoom(
            cutout,
            segmentation,
            components,
            local_groups,
            row,
            host_candidates,
            output_dir / "parent_zoom" / f"{cutout.cutout_id}_{short_id}.png",
            config,
            fallback_idx=idx,
        )
        if written is not None:
            paths["parent_zoom"].append(written)
    return paths


def _lobe_recovery_display_groups(
    local_groups: pd.DataFrame,
    lobe_seed_table: pd.DataFrame | None,
    parent_candidates: pd.DataFrame | None,
) -> pd.DataFrame:
    if local_groups is None or local_groups.empty:
        return pd.DataFrame()
    work = local_groups.copy()
    if lobe_seed_table is not None and not lobe_seed_table.empty:
        seed_cols = [
            "association_group_id",
            "is_parent_seed",
            "is_lobe_seed",
            "compact_point_like",
            "is_compact_singleton",
            "needs_visual_check",
        ]
        available = [col for col in seed_cols if col in lobe_seed_table]
        work = work.merge(lobe_seed_table[available].drop_duplicates("association_group_id"), on="association_group_id", how="left")
    for col in ["is_parent_seed", "is_lobe_seed", "compact_point_like", "is_compact_singleton", "needs_visual_check"]:
        if col not in work:
            work[col] = False
    candidate_ids: set[str] = set()
    if parent_candidates is not None and not parent_candidates.empty:
        qualities = {"high", "medium", "needs_host_check", "suspicious"}
        cand = parent_candidates[parent_candidates["parent_candidate_quality"].astype(str).isin(qualities)]
        for _, row in cand.iterrows():
            candidate_ids.update(_local_group_ids(row))
    atype = work.get("association_type", pd.Series("", index=work.index)).astype(str)
    quality = work.get("association_quality", pd.Series("", index=work.index)).astype(str)
    compact_single = (pd.to_numeric(work.get("n_gaussians", pd.Series(1, index=work.index)), errors="coerce").fillna(1) == 1) & (
        pd.to_numeric(work.get("LAS_beam", pd.Series(0, index=work.index)), errors="coerce").fillna(0) < 3.0
    )
    hide = (
        compact_single
        | _as_bool(work["compact_point_like"])
        | (atype == "weak_association")
        | (quality == "low")
    )
    show = (
        _as_bool(work["is_parent_seed"])
        | _as_bool(work["is_lobe_seed"])
        | _as_bool(work["needs_visual_check"])
        | work["association_group_id"].astype(str).isin(candidate_ids)
    )
    out = work[show & ~hide].copy()
    if out.empty and candidate_ids:
        out = work[work["association_group_id"].astype(str).isin(candidate_ids)].copy()
    out["local_quality"] = out.get("association_quality", "low")
    out["local_group_id"] = out.get("association_group_id", "")
    return out


def _draw_lobe_recovery_parent_links(
    ax: Any,
    candidates: pd.DataFrame,
    local_groups: pd.DataFrame,
    offset: tuple[float, float] = (0, 0),
    draw_scores: bool = False,
) -> None:
    if candidates is None or candidates.empty or local_groups is None or local_groups.empty:
        return
    by_id = local_groups.set_index("association_group_id")
    allowed = {"high", "medium", "needs_host_check", "suspicious"}
    work = candidates[candidates["parent_candidate_quality"].astype(str).isin(allowed)].copy()
    if work.empty:
        return
    for _, row in work.sort_values("parent_score_final", ascending=False).iterrows():
        left = str(row.get("local_group_id_1"))
        right = str(row.get("local_group_id_2"))
        if left not in by_id.index or right not in by_id.index:
            continue
        g1 = by_id.loc[left]
        g2 = by_id.loc[right]
        x = [float(g1["centroid_x"]) - offset[0], float(g2["centroid_x"]) - offset[0]]
        y = [float(g1["centroid_y"]) - offset[1], float(g2["centroid_y"]) - offset[1]]
        quality = str(row.get("parent_candidate_quality", "medium"))
        color = {"high": "yellow", "medium": "deepskyblue", "needs_host_check": "orange", "suspicious": "magenta"}.get(quality, "white")
        linestyle = "-" if quality in {"high", "medium"} else "--"
        ax.plot(x, y, color=color, lw=1.35, alpha=0.86, linestyle=linestyle)
        if draw_scores:
            label = (
                f"{row.get('parent_candidate_type', '')}\n"
                f"gap={float(row.get('box_gap_beam_robust', np.nan)):.1f} "
                f"align={float(row.get('axis_alignment_score', np.nan)):.2f}\n"
                f"{quality} host={row.get('host_status', '')}"
            )
            ax.text(0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1]), label, color=color, fontsize=6, path_effects=_text_effects())


def plot_lobe_recovery_cutout_overview(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    lobe_seed_table: pd.DataFrame,
    parent_candidates: pd.DataFrame,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> Path:
    """Generate lobe-recovery parent-linking overview while hiding ordinary compact point-source boxes."""

    viz, overview, _zoom = _viz_config(config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 8.0), constrained_layout=True)
    ax.imshow(_display_image(cutout.image, stretch=viz.get("stretch", "asinh"), percent_clip=tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    _draw_contours(ax, segmentation, overview.get("contour_thresholds", [2.5, 3.0]))
    gauss_to_draw = _select_gaussians(components, overview)
    if not gauss_to_draw.empty:
        ax.scatter(gauss_to_draw["x"], gauss_to_draw["y"], s=float(overview.get("gaussian_marker_size", 4)), c="cyan", alpha=0.16, linewidths=0)
    display_groups = _lobe_recovery_display_groups(local_groups, lobe_seed_table, parent_candidates)
    _draw_local_boxes(ax, display_groups, label=True)
    _draw_lobe_recovery_parent_links(ax, parent_candidates, local_groups, draw_scores=False)
    ax.set_xlim(0, cutout.image.shape[1] - 1)
    ax.set_ylim(0, cutout.image.shape[0] - 1)
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.set_title(
        f"{cutout.cutout_id} | local={len(local_groups)} | lobe-recovery parent-linking candidates={len(parent_candidates)} | displayed_boxes={len(display_groups)}",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_lobe_recovery_parent_zoom(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    parent_candidate_row: pd.Series,
    host_candidates: pd.DataFrame,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
    fallback_idx: int = 0,
) -> Path | None:
    """Generate one lobe-recovery parent-linking parent zoom with lobe recovery and host status."""

    viz, _overview, zoom = _viz_config(config)
    ids = _local_group_ids(parent_candidate_row)
    member_groups = local_groups[local_groups["association_group_id"].astype(str).isin(ids)].copy() if ids else pd.DataFrame()
    if member_groups.empty:
        return None
    boxes = [_bbox_tuple(row.get("bounding_box", "")) for _, row in member_groups.iterrows()]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    bbox = (min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes))
    x0, y0, x1, y1 = _zoom_window(bbox, cutout.image.shape, zoom)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    local_boxes = member_groups.copy()
    local_boxes["local_quality"] = local_boxes.get("association_quality", "low")
    local_boxes["local_group_id"] = local_boxes.get("association_group_id", "")
    comp_ids: set[int] = set()
    for _, row in member_groups.iterrows():
        comp_ids.update(_int_list(row.get("component_ids", "")))
    src_components = components[components["component_index"].astype(int).isin(comp_ids)].copy() if comp_ids else pd.DataFrame()
    fig, ax = plt.subplots(1, 1, figsize=(7.7, 7.7), constrained_layout=True)
    ax.imshow(_display_image(cutout.image[y0:y1, x0:x1], stretch=viz.get("stretch", "asinh"), percent_clip=tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    _draw_contours(ax, segmentation, zoom.get("contour_thresholds", [2.0, 2.5, 3.0, 5.0]), xlim=(x0, x1), ylim=(y0, y1))
    if not src_components.empty:
        ax.scatter(src_components["x"] - x0, src_components["y"] - y0, s=16, c="cyan", alpha=0.72, linewidths=0.2, edgecolors="black")
    _draw_local_boxes(ax, local_boxes, offset=(x0, y0), label=True)
    _draw_lobe_recovery_parent_links(ax, pd.DataFrame([parent_candidate_row]), local_groups, offset=(x0, y0), draw_scores=True)
    if len(member_groups) >= 2:
        mx = float(member_groups["centroid_x"].mean()) - x0
        my = float(member_groups["centroid_y"].mean()) - y0
        ax.scatter([mx], [my], s=54, marker="+", c="white", linewidths=1.5)
        radius_arcsec = float(parent_candidate_row.get("host_search_radius_arcsec", 10.0)) if "host_search_radius_arcsec" in parent_candidate_row else 10.0
        pixel_scale = float(components["pixel_scale_arcsec"].iloc[0]) if components is not None and not components.empty and "pixel_scale_arcsec" in components else 1.5
        ax.add_patch(plt.Circle((mx, my), radius_arcsec / max(pixel_scale, 1e-6), fill=False, color="white", alpha=0.55, lw=0.9))
    pid = str(parent_candidate_row.get("parent_candidate_id", ""))
    host_rows = host_candidates[host_candidates["parent_candidate_id"].astype(str) == pid].copy() if host_candidates is not None and not host_candidates.empty else pd.DataFrame()
    if not host_rows.empty and len(member_groups) >= 2:
        best = host_rows.sort_values("host_score", ascending=False).iloc[0]
        mx = float(member_groups["centroid_x"].mean()) - x0
        my = float(member_groups["centroid_y"].mean()) - y0
        ax.scatter([mx], [my], s=42, facecolors="none", edgecolors="magenta", linewidths=1.4)
        ax.text(
            mx + 6,
            my + 6,
            f"host={best.get('host_catalog', '')}\nhost_score={float(best.get('host_score', np.nan)):.1f}\nhost_sep={float(best.get('host_sep_midpoint_arcsec', np.nan)):.1f} arcsec\nhost_quality={best.get('host_quality', '')}\nW1={float(best.get('W1', np.nan)):.2f} W2={float(best.get('W2', np.nan)):.2f}\nW1-W2={float(best.get('W1_W2', np.nan)):.2f}",
            color="magenta",
            fontsize=6,
            path_effects=_text_effects(),
        )
    short_id = str(parent_candidate_row.get("parent_candidate_id", f"pc{fallback_idx:03d}")).rsplit("_", 1)[-1]
    title = (
        f"{cutout.cutout_id} {short_id} | lobe-recovery parent-linking separated radio-lobe parent-link candidate\n"
        f"type={parent_candidate_row.get('parent_candidate_type', '')} quality={parent_candidate_row.get('parent_candidate_quality', '')} "
        f"host_status={parent_candidate_row.get('host_status', '')} "
        f"gap={float(parent_candidate_row.get('box_gap_beam_robust', np.nan)):.1f} "
        f"align={float(parent_candidate_row.get('axis_alignment_score', np.nan)):.2f}"
    )
    ax.set_title(title, fontsize=8.0)
    ax.set_xlim(0, x1 - x0 - 1)
    ax.set_ylim(0, y1 - y0 - 1)
    ax.set_xlabel(f"x [{x0}:{x1}]")
    ax.set_ylabel(f"y [{y0}:{y1}]")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def plot_lobe_recovery_cutout_all(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    lobe_seed_table: pd.DataFrame,
    parent_candidates: pd.DataFrame,
    host_candidates: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, list[Path]]:
    """Generate lobe-recovery parent-linking overview and parent zoom figures."""

    output_dir = Path(output_dir)
    paths: dict[str, list[Path]] = {"overview": [], "parent_zoom": []}
    overview_path = output_dir / "overview" / f"{cutout.cutout_id}.png"
    paths["overview"].append(plot_lobe_recovery_cutout_overview(cutout, segmentation, components, local_groups, lobe_seed_table, parent_candidates, overview_path, config))
    if parent_candidates is None or parent_candidates.empty:
        return paths
    for idx, (_, row) in enumerate(parent_candidates.sort_values("parent_score_final", ascending=False).iterrows()):
        short_id = str(row.get("parent_candidate_id", f"pc{idx:03d}")).rsplit("_", 1)[-1]
        written = plot_lobe_recovery_parent_zoom(
            cutout,
            segmentation,
            components,
            local_groups,
            row,
            host_candidates,
            output_dir / "parent_zoom" / f"{cutout.cutout_id}_{short_id}.png",
            config,
            fallback_idx=idx,
        )
        if written is not None:
            paths["parent_zoom"].append(written)
    return paths


def _lobe_first_display_groups(
    local_groups: pd.DataFrame,
    lobe_like_table: pd.DataFrame | None,
    parent_candidates: pd.DataFrame | None,
) -> pd.DataFrame:
    if local_groups is None or local_groups.empty:
        return pd.DataFrame()
    work = local_groups.copy()
    if lobe_like_table is not None and not lobe_like_table.empty:
        seed_cols = [
            "association_group_id",
            "is_point_like",
            "extended_candidate",
            "is_parent_seed",
            "is_lobe_seed",
            "is_lobe_like",
            "association_type_display",
        ]
        available = [col for col in seed_cols if col in lobe_like_table]
        work = work.merge(lobe_like_table[available].drop_duplicates("association_group_id"), on="association_group_id", how="left")
    for col in ["is_point_like", "extended_candidate", "is_parent_seed", "is_lobe_seed", "is_lobe_like"]:
        if col not in work:
            work[col] = False
    candidate_ids: set[str] = set()
    if parent_candidates is not None and not parent_candidates.empty:
        qualities = {"high", "medium", "needs_host_check", "suspicious"}
        cand = parent_candidates[parent_candidates["parent_candidate_quality"].astype(str).isin(qualities)]
        for _, row in cand.iterrows():
            candidate_ids.update(_local_group_ids(row))
    quality = work.get("association_quality", pd.Series("", index=work.index)).astype(str)
    needs = work.get("needs_visual_check", pd.Series(False, index=work.index))
    show = (
        _as_bool(work["extended_candidate"])
        | _as_bool(work["is_parent_seed"])
        | _as_bool(work["is_lobe_seed"])
        | _as_bool(work["is_lobe_like"])
        | _as_bool(needs)
        | (quality == "suspicious")
        | work["association_group_id"].astype(str).isin(candidate_ids)
    )
    hide = _as_bool(work["is_point_like"]) & ~work["association_group_id"].astype(str).isin(candidate_ids)
    out = work[show & ~hide].copy()
    out["local_quality"] = out.get("association_quality", "medium")
    out["local_group_id"] = out.get("association_group_id", "")
    return out


def _draw_lobe_first_local_boxes(
    ax: Any,
    local_groups: pd.DataFrame,
    offset: tuple[float, float] = (0, 0),
    label: bool = True,
) -> None:
    if local_groups is None or local_groups.empty:
        return
    for idx, (_, row) in enumerate(local_groups.iterrows()):
        bbox = _bbox_tuple(row.get("bounding_box", ""))
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        color = "gold"
        if _as_bool(row.get("is_parent_seed", False)) or _as_bool(row.get("is_lobe_seed", False)) or _as_bool(row.get("is_lobe_like", False)):
            color = "lime"
        if str(row.get("association_quality", "")).lower() == "suspicious" or _as_bool(row.get("needs_visual_check", False)):
            color = "magenta"
        rect = Rectangle((x0 - offset[0], y0 - offset[1]), x1 - x0 + 1, y1 - y0 + 1, fill=False, lw=0.85, edgecolor=color, alpha=0.82)
        ax.add_patch(rect)
        if label:
            text = str(row.get("local_group_id", f"l{idx:03d}")).rsplit("_", 1)[-1]
            ax.text(
                x0 - offset[0],
                max(0, y0 - offset[1] - 4),
                text,
                color=color,
                fontsize=5.7,
                va="top",
                ha="left",
                path_effects=_text_effects(),
            )


def _draw_lobe_first_parent_links(
    ax: Any,
    candidates: pd.DataFrame,
    local_groups: pd.DataFrame,
    offset: tuple[float, float] = (0, 0),
    draw_scores: bool = False,
) -> None:
    if candidates is None or candidates.empty or local_groups is None or local_groups.empty:
        return
    by_id = local_groups.set_index("association_group_id")
    allowed = {"high", "medium", "needs_host_check", "suspicious"}
    work = candidates[candidates["parent_candidate_quality"].astype(str).isin(allowed)].copy()
    if work.empty:
        return
    for _, row in work.sort_values("parent_score_final", ascending=False).iterrows():
        left = str(row.get("local_group_id_1"))
        right = str(row.get("local_group_id_2"))
        if left not in by_id.index or right not in by_id.index:
            continue
        g1 = by_id.loc[left]
        g2 = by_id.loc[right]
        x = [float(g1["centroid_x"]) - offset[0], float(g2["centroid_x"]) - offset[0]]
        y = [float(g1["centroid_y"]) - offset[1], float(g2["centroid_y"]) - offset[1]]
        quality = str(row.get("parent_candidate_quality", "medium"))
        color = {"high": "cyan", "medium": "deepskyblue", "needs_host_check": "orange", "suspicious": "magenta"}.get(quality, "white")
        linestyle = "-" if quality in {"high", "medium"} else "--"
        ax.plot(x, y, color=color, lw=1.35, alpha=0.86, linestyle=linestyle)
        if draw_scores:
            label = (
                f"lobe_pair_score={float(row.get('lobe_pair_score', np.nan)):.2f}\n"
                f"align={float(row.get('axis_alignment_score', np.nan)):.2f} face={float(row.get('facing_score', np.nan)):.2f}\n"
                f"flux={float(row.get('flux_ratio', np.nan)):.1f} size={float(row.get('size_ratio', np.nan)):.1f}\n"
                f"{quality} host={row.get('host_quality', '')}"
            )
            ax.text(0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1]), label, color=color, fontsize=6, path_effects=_text_effects())


def plot_lobe_first_cutout_overview(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    lobe_like_table: pd.DataFrame,
    parent_candidates: pd.DataFrame,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> Path:
    """Generate lobe-first parent-linking overview: hide only beam-like point sources."""

    viz, overview, _zoom = _viz_config(config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 8.0), constrained_layout=True)
    ax.imshow(_display_image(cutout.image, stretch=viz.get("stretch", "asinh"), percent_clip=tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    _draw_contours(ax, segmentation, overview.get("contour_thresholds", [2.5, 3.0]))
    gauss_to_draw = _select_gaussians(components, overview)
    if not gauss_to_draw.empty:
        ax.scatter(gauss_to_draw["x"], gauss_to_draw["y"], s=float(overview.get("gaussian_marker_size", 4)), c="cyan", alpha=0.15, linewidths=0)
    display_groups = _lobe_first_display_groups(local_groups, lobe_like_table, parent_candidates)
    _draw_lobe_first_local_boxes(ax, display_groups, label=True)
    _draw_lobe_first_parent_links(ax, parent_candidates, local_groups, draw_scores=False)
    ax.set_xlim(0, cutout.image.shape[1] - 1)
    ax.set_ylim(0, cutout.image.shape[0] - 1)
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.set_title(
        f"{cutout.cutout_id} | local={len(local_groups)} | lobe-first parent-linking candidates={len(parent_candidates)} | shown_nonpoint={len(display_groups)}",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_lobe_first_parent_zoom(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    parent_candidate_row: pd.Series,
    host_candidates: pd.DataFrame,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
    fallback_idx: int = 0,
) -> Path | None:
    """Generate one lobe-first parent-linking parent zoom with lobe-first diagnostics."""

    viz, _overview, zoom = _viz_config(config)
    ids = _local_group_ids(parent_candidate_row)
    member_groups = local_groups[local_groups["association_group_id"].astype(str).isin(ids)].copy() if ids else pd.DataFrame()
    if member_groups.empty:
        return None
    boxes = [_bbox_tuple(row.get("bounding_box", "")) for _, row in member_groups.iterrows()]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    bbox = (min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes))
    x0, y0, x1, y1 = _zoom_window(bbox, cutout.image.shape, zoom)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    local_boxes = member_groups.copy()
    local_boxes["local_quality"] = local_boxes.get("association_quality", "low")
    local_boxes["local_group_id"] = local_boxes.get("association_group_id", "")
    comp_ids: set[int] = set()
    for _, row in member_groups.iterrows():
        comp_ids.update(_int_list(row.get("component_ids", "")))
    src_components = components[components["component_index"].astype(int).isin(comp_ids)].copy() if comp_ids else pd.DataFrame()
    fig, ax = plt.subplots(1, 1, figsize=(7.7, 7.7), constrained_layout=True)
    ax.imshow(_display_image(cutout.image[y0:y1, x0:x1], stretch=viz.get("stretch", "asinh"), percent_clip=tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    _draw_contours(ax, segmentation, zoom.get("contour_thresholds", [2.0, 2.5, 3.0, 5.0]), xlim=(x0, x1), ylim=(y0, y1))
    if not src_components.empty:
        ax.scatter(src_components["x"] - x0, src_components["y"] - y0, s=16, c="cyan", alpha=0.72, linewidths=0.2, edgecolors="black")
    _draw_local_boxes(ax, local_boxes, offset=(x0, y0), label=True)
    _draw_lobe_first_parent_links(ax, pd.DataFrame([parent_candidate_row]), local_groups, offset=(x0, y0), draw_scores=True)
    if len(member_groups) >= 2:
        mx = float(member_groups["centroid_x"].mean()) - x0
        my = float(member_groups["centroid_y"].mean()) - y0
        ax.scatter([mx], [my], s=54, marker="+", c="white", linewidths=1.5)
        radius_arcsec = float(parent_candidate_row.get("host_search_radius_arcsec", 10.0)) if "host_search_radius_arcsec" in parent_candidate_row else 10.0
        pixel_scale = float(components["pixel_scale_arcsec"].iloc[0]) if components is not None and not components.empty and "pixel_scale_arcsec" in components else 1.5
        ax.add_patch(plt.Circle((mx, my), radius_arcsec / max(pixel_scale, 1e-6), fill=False, color="white", alpha=0.55, lw=0.9))
    pid = str(parent_candidate_row.get("parent_candidate_id", ""))
    host_rows = host_candidates[host_candidates["parent_candidate_id"].astype(str) == pid].copy() if host_candidates is not None and not host_candidates.empty else pd.DataFrame()
    if not host_rows.empty and len(member_groups) >= 2:
        best = host_rows.sort_values("host_score", ascending=False).iloc[0]
        mx = float(member_groups["centroid_x"].mean()) - x0
        my = float(member_groups["centroid_y"].mean()) - y0
        ax.scatter([mx], [my], s=42, facecolors="none", edgecolors="magenta", linewidths=1.4)
        ax.text(
            mx + 6,
            my + 6,
            f"host={best.get('host_catalog', '')}\nhost_score={float(best.get('host_score', np.nan)):.1f}\nhost_quality={best.get('host_quality', '')}\nW1={float(best.get('W1', np.nan)):.2f} W2={float(best.get('W2', np.nan)):.2f}\nW1-W2={float(best.get('W1_W2', np.nan)):.2f}",
            color="magenta",
            fontsize=6,
            path_effects=_text_effects(),
        )
    short_id = str(parent_candidate_row.get("parent_candidate_id", f"pc{fallback_idx:03d}")).rsplit("_", 1)[-1]
    title = (
        f"{cutout.cutout_id} {short_id} | lobe-first parent-linking lobe-first host-second parent candidate\n"
        f"quality={parent_candidate_row.get('parent_candidate_quality', '')} "
        f"host={parent_candidate_row.get('host_status', '')} "
        f"score={float(parent_candidate_row.get('lobe_pair_score', np.nan)):.2f} "
        f"reason={parent_candidate_row.get('rejection_reason', '')}"
    )
    ax.set_title(title, fontsize=8.0)
    ax.set_xlim(0, x1 - x0 - 1)
    ax.set_ylim(0, y1 - y0 - 1)
    ax.set_xlabel(f"x [{x0}:{x1}]")
    ax.set_ylabel(f"y [{y0}:{y1}]")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def plot_lobe_first_cutout_all(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    lobe_like_table: pd.DataFrame,
    parent_candidates: pd.DataFrame,
    host_candidates: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, list[Path]]:
    """Generate lobe-first parent-linking overview and parent zoom figures."""

    output_dir = Path(output_dir)
    paths: dict[str, list[Path]] = {"overview": [], "parent_zoom": []}
    overview_path = output_dir / "overview" / f"{cutout.cutout_id}.png"
    paths["overview"].append(plot_lobe_first_cutout_overview(cutout, segmentation, components, local_groups, lobe_like_table, parent_candidates, overview_path, config))
    if parent_candidates is None or parent_candidates.empty:
        return paths
    for idx, (_, row) in enumerate(parent_candidates.sort_values("parent_score_final", ascending=False).iterrows()):
        short_id = str(row.get("parent_candidate_id", f"pc{idx:03d}")).rsplit("_", 1)[-1]
        written = plot_lobe_first_parent_zoom(
            cutout,
            segmentation,
            components,
            local_groups,
            row,
            host_candidates,
            output_dir / "parent_zoom" / f"{cutout.cutout_id}_{short_id}.png",
            config,
            fallback_idx=idx,
        )
        if written is not None:
            paths["parent_zoom"].append(written)
    return paths


def _parent_link_display_groups(local_groups: pd.DataFrame, source_morph_table: pd.DataFrame | None, parent_candidates: pd.DataFrame | None) -> pd.DataFrame:
    if local_groups is None or local_groups.empty:
        return pd.DataFrame()
    work = local_groups.copy()
    if source_morph_table is not None and not source_morph_table.empty:
        cols = [
            "association_group_id",
            "source_morph_class",
            "is_point_like",
            "is_lobe_candidate",
            "is_artifact_risk",
            "hard_compact_veto",
            "noise_artifact_veto",
            "isolated_compact_veto",
            "near_extended_lobe_candidate",
        ]
        available = [col for col in cols if col in source_morph_table]
        work = work.merge(source_morph_table[available].drop_duplicates("association_group_id"), on="association_group_id", how="left")
    for col in ["is_point_like", "is_lobe_candidate", "is_artifact_risk", "hard_compact_veto", "noise_artifact_veto", "isolated_compact_veto", "near_extended_lobe_candidate"]:
        if col not in work:
            work[col] = False
    if "source_morph_class" not in work:
        work["source_morph_class"] = "resolved_single"
    candidate_ids: set[str] = set()
    if parent_candidates is not None and not parent_candidates.empty:
        for _, row in parent_candidates.iterrows():
            candidate_ids.update(_local_group_ids(row))
    quality = work.get("association_quality", pd.Series("", index=work.index)).astype(str)
    hidden_classes = {"point_like", "point_like_or_compact", "compact_resolved_single", "noise_or_artifact"}
    hard_hidden = (
        work["source_morph_class"].astype(str).isin(hidden_classes)
        | _as_bool(work["is_point_like"])
        | _as_bool(work["hard_compact_veto"])
        | _as_bool(work["noise_artifact_veto"])
        | _as_bool(work["isolated_compact_veto"])
    )
    candidate_member = work["association_group_id"].astype(str).isin(candidate_ids)
    show = (~hard_hidden | _as_bool(work["is_lobe_candidate"]) | _as_bool(work["near_extended_lobe_candidate"]) | (quality == "suspicious") | candidate_member)
    hide = hard_hidden & ~candidate_member
    out = work[show & ~hide].copy()
    out["local_quality"] = out.get("association_quality", "medium")
    out["local_group_id"] = out.get("association_group_id", "")
    return out


def _draw_parent_link_local_boxes(ax: Any, local_groups: pd.DataFrame, offset: tuple[float, float] = (0, 0), label: bool = True) -> None:
    if local_groups is None or local_groups.empty:
        return
    for idx, (_, row) in enumerate(local_groups.iterrows()):
        bbox = _bbox_tuple(row.get("bounding_box", ""))
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        morph = str(row.get("source_morph_class", "resolved_single"))
        color = "gold"
        if morph == "lobe_candidate":
            color = "lime"
        elif morph == "artifact_risk" or str(row.get("association_quality", "")).lower() == "suspicious":
            color = "magenta"
        rect = Rectangle((x0 - offset[0], y0 - offset[1]), x1 - x0 + 1, y1 - y0 + 1, fill=False, lw=0.85, edgecolor=color, alpha=0.82)
        ax.add_patch(rect)
        if label:
            text = str(row.get("local_group_id", f"l{idx:03d}")).rsplit("_", 1)[-1]
            ax.text(x0 - offset[0], max(0, y0 - offset[1] - 4), text, color=color, fontsize=5.7, va="top", ha="left", path_effects=_text_effects())


def _draw_parent_link_parent_links(
    ax: Any,
    candidates: pd.DataFrame,
    local_groups: pd.DataFrame,
    offset: tuple[float, float] = (0, 0),
    draw_scores: bool = False,
) -> None:
    if candidates is None or candidates.empty or local_groups is None or local_groups.empty:
        return
    by_id = local_groups.set_index("association_group_id")
    allowed = {"high", "medium", "needs_host_check", "suspicious"}
    work = candidates[candidates["parent_candidate_quality"].astype(str).isin(allowed)].copy()
    for _, row in work.sort_values("parent_score_final", ascending=False).iterrows():
        left = str(row.get("local_group_id_1"))
        right = str(row.get("local_group_id_2"))
        if left not in by_id.index or right not in by_id.index:
            continue
        g1 = by_id.loc[left]
        g2 = by_id.loc[right]
        x = [float(g1["centroid_x"]) - offset[0], float(g2["centroid_x"]) - offset[0]]
        y = [float(g1["centroid_y"]) - offset[1], float(g2["centroid_y"]) - offset[1]]
        quality = str(row.get("parent_candidate_quality", "medium"))
        color = {"high": "cyan", "medium": "deepskyblue", "needs_host_check": "orange", "suspicious": "magenta"}.get(quality, "white")
        ax.plot(x, y, color=color, lw=1.35, alpha=0.86, linestyle="-" if quality in {"high", "medium"} else "--")
        px0 = float(row.get("parent_bbox_xmin", np.nan)) - offset[0]
        px1 = float(row.get("parent_bbox_xmax", np.nan)) - offset[0]
        py0 = float(row.get("parent_bbox_ymin", np.nan)) - offset[1]
        py1 = float(row.get("parent_bbox_ymax", np.nan)) - offset[1]
        if np.all(np.isfinite([px0, px1, py0, py1])):
            ax.add_patch(Rectangle((px0, py0), px1 - px0 + 1, py1 - py0 + 1, fill=False, lw=1.1, edgecolor="cyan", alpha=0.72, linestyle=":"))
        if draw_scores:
            label = (
                f"sym={float(row.get('symmetry_score', np.nan)):.2f} "
                f"score={float(row.get('lobe_pair_score', np.nan)):.2f}\n"
                f"host={row.get('host_evidence', '')} peak_host={row.get('lobe_peak_host_found', False)}\n"
                f"{quality} reason={row.get('rejection_reason', '')}"
            )
            if bool(row.get("near_boundary_rescue_applied", False)):
                label += f"\nnear-boundary rescue gap/box={float(row.get('gap_to_mean_box_ratio', np.nan)):.2f} bridge={row.get('bridge_snr_support', False)}"
            ax.text(0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1]), label, color=color, fontsize=6, path_effects=_text_effects())


def plot_parent_link_cutout_overview(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    source_morph_table: pd.DataFrame,
    parent_candidates: pd.DataFrame,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
) -> Path:
    """Generate production parent-linking overview with point-like sources hidden and union boxes."""

    viz, overview, _zoom = _viz_config(config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 8.0), constrained_layout=True)
    ax.imshow(_display_image(cutout.image, stretch=viz.get("stretch", "asinh"), percent_clip=tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    _draw_contours(ax, segmentation, overview.get("contour_thresholds", [2.5, 3.0]))
    gauss_to_draw = _select_gaussians(components, overview)
    if not gauss_to_draw.empty:
        ax.scatter(gauss_to_draw["x"], gauss_to_draw["y"], s=float(overview.get("gaussian_marker_size", 4)), c="cyan", alpha=0.13, linewidths=0)
    display_groups = _parent_link_display_groups(local_groups, source_morph_table, parent_candidates)
    _draw_parent_link_local_boxes(ax, display_groups, label=True)
    _draw_parent_link_parent_links(ax, parent_candidates, local_groups, draw_scores=False)
    ax.set_xlim(0, cutout.image.shape[1] - 1)
    ax.set_ylim(0, cutout.image.shape[0] - 1)
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.set_title(f"{cutout.cutout_id} | local={len(local_groups)} | production parent-linking candidates={len(parent_candidates)} | shown_nonpoint={len(display_groups)}", fontsize=9)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_parent_link_parent_zoom(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    parent_candidate_row: pd.Series,
    host_candidates: pd.DataFrame,
    output_path: str | Path,
    config: dict[str, Any] | None = None,
    fallback_idx: int = 0,
) -> Path | None:
    """Generate one production parent-linking parent zoom with parent union and host contradiction info."""

    viz, _overview, zoom = _viz_config(config)
    ids = _local_group_ids(parent_candidate_row)
    member_groups = local_groups[local_groups["association_group_id"].astype(str).isin(ids)].copy() if ids else pd.DataFrame()
    if member_groups.empty:
        return None
    boxes = [_bbox_tuple(row.get("bounding_box", "")) for _, row in member_groups.iterrows()]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    union_box = (
        float(parent_candidate_row.get("parent_bbox_xmin", min(box[0] for box in boxes))),
        float(parent_candidate_row.get("parent_bbox_ymin", min(box[1] for box in boxes))),
        float(parent_candidate_row.get("parent_bbox_xmax", max(box[2] for box in boxes))),
        float(parent_candidate_row.get("parent_bbox_ymax", max(box[3] for box in boxes))),
    )
    x0, y0, x1, y1 = _zoom_window(union_box, cutout.image.shape, zoom)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    local_boxes = member_groups.copy()
    local_boxes["local_quality"] = local_boxes.get("association_quality", "low")
    local_boxes["local_group_id"] = local_boxes.get("association_group_id", "")
    comp_ids: set[int] = set()
    for _, row in member_groups.iterrows():
        comp_ids.update(_int_list(row.get("component_ids", "")))
    src_components = components[components["component_index"].astype(int).isin(comp_ids)].copy() if comp_ids else pd.DataFrame()
    fig, ax = plt.subplots(1, 1, figsize=(7.7, 7.7), constrained_layout=True)
    ax.imshow(_display_image(cutout.image[y0:y1, x0:x1], stretch=viz.get("stretch", "asinh"), percent_clip=tuple(viz.get("percent_clip", [1, 99.5]))), origin="lower", cmap="gray")
    _draw_contours(ax, segmentation, zoom.get("contour_thresholds", [2.0, 2.5, 3.0, 5.0]), xlim=(x0, x1), ylim=(y0, y1))
    if not src_components.empty:
        ax.scatter(src_components["x"] - x0, src_components["y"] - y0, s=16, c="cyan", alpha=0.72, linewidths=0.2, edgecolors="black")
    _draw_local_boxes(ax, local_boxes, offset=(x0, y0), label=True)
    _draw_parent_link_parent_links(ax, pd.DataFrame([parent_candidate_row]), local_groups, offset=(x0, y0), draw_scores=True)
    if len(member_groups) >= 2:
        mx = float(member_groups["centroid_x"].mean()) - x0
        my = float(member_groups["centroid_y"].mean()) - y0
        ax.scatter([mx], [my], s=54, marker="+", c="white", linewidths=1.5)
    pid = str(parent_candidate_row.get("parent_candidate_id", ""))
    host_rows = host_candidates[host_candidates["parent_candidate_id"].astype(str) == pid].copy() if host_candidates is not None and not host_candidates.empty else pd.DataFrame()
    if not host_rows.empty and len(member_groups) >= 2:
        midpoint_rows = host_rows[host_rows.get("host_role", pd.Series("", index=host_rows.index)).astype(str) == "midpoint"]
        peak_rows = host_rows[host_rows.get("host_role", pd.Series("", index=host_rows.index)).astype(str).str.contains("lobe")]
        mx = float(member_groups["centroid_x"].mean()) - x0
        my = float(member_groups["centroid_y"].mean()) - y0
        if not midpoint_rows.empty:
            best = midpoint_rows.sort_values("host_score", ascending=False).iloc[0]
            ax.scatter([mx], [my], s=42, facecolors="none", edgecolors="magenta", linewidths=1.4)
            ax.text(
                mx + 6,
                my + 6,
                (
                    f"mid host={best.get('host_catalog', '')} {best.get('host_quality', '')}\n"
                    f"score={float(best.get('host_score', np.nan)):.1f} sep={float(best.get('host_sep_midpoint_arcsec', np.nan)):.1f}\" "
                    f"W1={float(best.get('W1', np.nan)):.2f}\n"
                    f"W2={float(best.get('W2', np.nan)):.2f} W1-W2={float(best.get('W1_W2', np.nan)):.2f}"
                ),
                color="magenta",
                fontsize=5.6,
                path_effects=_text_effects(),
            )
        if not peak_rows.empty:
            ax.text(
                8,
                8,
                (
                    f"lobe peak host rows={len(peak_rows)} "
                    f"l1={parent_candidate_row.get('lobe1_peak_host_found', False)} "
                    f"l2={parent_candidate_row.get('lobe2_peak_host_found', False)}\n"
                    f"contradiction={parent_candidate_row.get('rejection_reason', '') == 'lobe_peak_host_contradiction'}"
                ),
                color="tomato",
                fontsize=5.8,
                path_effects=_text_effects(),
            )
    short_id = str(parent_candidate_row.get("parent_candidate_id", f"pc{fallback_idx:03d}")).rsplit("_", 1)[-1]
    title = (
        f"{cutout.cutout_id} {short_id} | production parent-linking physics-aware parent candidate\n"
        f"quality={parent_candidate_row.get('parent_candidate_quality', '')} host_evidence={parent_candidate_row.get('host_evidence', '')} "
        f"sym={float(parent_candidate_row.get('symmetry_score', np.nan)):.2f} "
        f"parent_LAS={float(parent_candidate_row.get('parent_LAS_beam', np.nan)):.1f} beam\n"
        f"gap={float(parent_candidate_row.get('box_gap_beam_robust', np.nan)):.2f} beam"
    )
    if bool(parent_candidate_row.get("near_boundary_rescue_applied", False)):
        title += (
            f"\nnear-boundary rescue: gap/box={float(parent_candidate_row.get('gap_to_mean_box_ratio', np.nan)):.2f} "
            f"bridge={parent_candidate_row.get('bridge_snr_support', False)} "
            f"quality={parent_candidate_row.get('parent_candidate_quality', '')}"
        )
    ax.set_title(title, fontsize=8.0)
    ax.set_xlim(0, x1 - x0 - 1)
    ax.set_ylim(0, y1 - y0 - 1)
    ax.set_xlabel(f"x [{x0}:{x1}]")
    ax.set_ylabel(f"y [{y0}:{y1}]")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def plot_parent_link_cutout_all(
    cutout: Any,
    segmentation: Any,
    components: pd.DataFrame,
    local_groups: pd.DataFrame,
    source_morph_table: pd.DataFrame,
    parent_candidates: pd.DataFrame,
    host_candidates: pd.DataFrame,
    output_dir: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, list[Path]]:
    """Generate production parent-linking overview and parent zoom figures."""

    output_dir = Path(output_dir)
    paths: dict[str, list[Path]] = {"overview": [], "parent_zoom": []}
    overview_path = output_dir / "overview" / f"{cutout.cutout_id}.png"
    paths["overview"].append(plot_parent_link_cutout_overview(cutout, segmentation, components, local_groups, source_morph_table, parent_candidates, overview_path, config))
    if parent_candidates is None or parent_candidates.empty:
        return paths
    for idx, (_, row) in enumerate(parent_candidates.sort_values("parent_score_final", ascending=False).iterrows()):
        short_id = str(row.get("parent_candidate_id", f"pc{idx:03d}")).rsplit("_", 1)[-1]
        written = plot_parent_link_parent_zoom(
            cutout,
            segmentation,
            components,
            local_groups,
            row,
            host_candidates,
            output_dir / "parent_zoom" / f"{cutout.cutout_id}_{short_id}.png",
            config,
            fallback_idx=idx,
        )
        if written is not None:
            paths["parent_zoom"].append(written)
    return paths
