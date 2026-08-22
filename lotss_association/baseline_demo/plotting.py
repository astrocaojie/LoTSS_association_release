"""Diagnostic plotting for tile baseline comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_figure(fig: plt.Figure, stem: Path, dpi: int = 300) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _display_image(image: np.ndarray) -> np.ndarray:
    data = np.asarray(image, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data)
    lo, hi = np.nanpercentile(finite, [1, 99.5])
    scaled = np.clip((data - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.arcsinh(8 * scaled) / np.arcsinh(8)


def plot_group_size_distribution(all_groups: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, groups in sorted(all_groups.items()):
        if groups.empty or "n_components" not in groups:
            continue
        sizes = groups["n_components"].to_numpy(int)
        bins = np.arange(1, max(2, sizes.max()) + 2) - 0.5
        hist, edges = np.histogram(sizes, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.step(centers, hist, where="mid", label=label, linewidth=1.4)
    ax.set_xlabel("group size (Gaussian components)")
    ax.set_ylabel("number of groups")
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    _save_figure(fig, Path(output_dir) / "figures" / "figure1_group_size_distribution")


def plot_distance_sensitivity(distance_stats: pd.DataFrame, output_dir: str | Path) -> None:
    if distance_stats.empty or "threshold_beam" not in distance_stats:
        return
    scan = distance_stats[distance_stats["parameter_id"].astype(str).str.startswith("tau_")].copy()
    if scan.empty:
        return
    scan = scan.sort_values("threshold_beam")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(scan["threshold_beam"], scan["n_multi_groups"], marker="o", label="multi-component groups")
    ax1.plot(scan["threshold_beam"], scan["n_singletons"], marker="s", label="singletons")
    ax1.set_xlabel("distance threshold (beam units)")
    ax1.set_ylabel("group count")
    ax2 = ax1.twinx()
    ax2.plot(scan["threshold_beam"], scan["max_group_size"], color="tab:red", marker="^", label="largest group size")
    if "n_components" in scan:
        frac = 1.0 - scan["n_singletons"] / scan["n_components"].clip(lower=1)
        ax2.plot(scan["threshold_beam"], frac, color="tab:purple", marker="d", label="fraction in multi groups")
    ax2.set_ylabel("largest size / fraction")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, fontsize=8)
    ax1.grid(alpha=0.25)
    _save_figure(fig, Path(output_dir) / "figures" / "figure2_distance_threshold_sensitivity")


def plot_distance_threshold_named_figures(distance_stats: pd.DataFrame, output_dir: str | Path) -> None:
    """Write the four named distance-threshold diagnostic figures."""

    if distance_stats.empty or "threshold_beam" not in distance_stats:
        return
    scan = distance_stats[distance_stats["parameter_id"].astype(str).str.startswith("tau_")].copy()
    if scan.empty:
        return
    scan = scan.sort_values("threshold_beam")
    figures = [
        ("distance_threshold_vs_group_count", "number of groups", "n_groups"),
        ("distance_threshold_vs_singleton_fraction", "singleton fraction", "singleton_fraction"),
        ("distance_threshold_vs_max_group_size", "maximum group size", "max_group_size"),
        (
            "distance_threshold_vs_merge_disagreement",
            "groups containing multiple full Layer-1 groups",
            "number_groups_containing_multiple_full_layer1_groups",
        ),
    ]
    scan["singleton_fraction"] = scan["n_singletons"] / scan["n_components"].clip(lower=1) if "n_components" in scan else np.nan
    for stem, ylabel, column in figures:
        if column not in scan:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(scan["threshold_beam"], scan[column], marker="o", linewidth=1.7)
        ax.set_xlabel("distance threshold (beam units)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        _save_figure(fig, Path(output_dir) / "figures" / stem)


def plot_agreement_matrix(agreement: pd.DataFrame, output_dir: str | Path, metric: str = "adjusted_rand_index") -> None:
    if agreement.empty or metric not in agreement:
        return
    methods = sorted(set(agreement["method_a"].astype(str)).union(agreement["method_b"].astype(str)))
    mat = pd.DataFrame(np.eye(len(methods)), index=methods, columns=methods)
    for _, row in agreement.iterrows():
        a = str(row["method_a"])
        b = str(row["method_b"])
        mat.loc[a, b] = float(row[metric])
        mat.loc[b, a] = float(row[metric])
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(methods)), max(5, 0.45 * len(methods))))
    im = ax.imshow(mat.to_numpy(float), vmin=-0.1, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(methods)), methods, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(methods)), methods, fontsize=7)
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", color="white", fontsize=6)
    fig.colorbar(im, ax=ax, label=metric)
    _save_figure(fig, Path(output_dir) / "figures" / "figure3_method_agreement_matrix")


def _component_group_map(membership: pd.DataFrame) -> dict[str, str]:
    return dict(zip(membership["component_id"].astype(str), membership["predicted_group_id"].astype(str))) if not membership.empty else {}


def _plot_case(
    image: np.ndarray,
    components: pd.DataFrame,
    edges: pd.DataFrame,
    memberships: list[tuple[str, pd.DataFrame]],
    component_ids: set[str],
    title: str,
    output_stem: Path,
) -> None:
    subset = components[components["component_id"].astype(str).isin(component_ids)].copy()
    if subset.empty:
        return
    pad = 40
    x0 = max(0, int(np.floor(subset["x"].min())) - pad)
    x1 = min(image.shape[1], int(np.ceil(subset["x"].max())) + pad)
    y0 = max(0, int(np.floor(subset["y"].min())) - pad)
    y1 = min(image.shape[0], int(np.ceil(subset["y"].max())) + pad)
    if x1 <= x0 or y1 <= y0:
        return
    ncols = max(1, len(memberships))
    fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.2), squeeze=False)
    for ax, (label, membership) in zip(axes[0], memberships):
        ax.imshow(_display_image(image[y0:y1, x0:x1]), origin="lower", cmap="gray")
        group_map = _component_group_map(membership)
        groups = sorted({group_map.get(cid, cid) for cid in component_ids})
        color_map = {group: plt.cm.tab20(i % 20) for i, group in enumerate(groups)}
        if edges is not None and not edges.empty:
            by_node = components.set_index("component_index")
            for _, edge in edges.iterrows():
                c1 = str(edge.get("gaussian_id_1", ""))
                c2 = str(edge.get("gaussian_id_2", ""))
                if c1 not in component_ids or c2 not in component_ids:
                    continue
                try:
                    r1 = by_node.loc[int(edge["component_index_1"])]
                    r2 = by_node.loc[int(edge["component_index_2"])]
                except Exception:
                    continue
                style = "--" if str(edge.get("edge_type")) == "weak" else "-"
                alpha = 0.8 if str(edge.get("edge_type")) in {"strong", "weak"} else 0.25
                ax.plot([r1["x"] - x0, r2["x"] - x0], [r1["y"] - y0, r2["y"] - y0], style, color="white", linewidth=0.9, alpha=alpha)
        for _, row in subset.iterrows():
            cid = str(row["component_id"])
            group = group_map.get(cid, cid)
            ax.scatter(float(row["x"]) - x0, float(row["y"]) - y0, s=28, edgecolor="black", facecolor=color_map[group], linewidth=0.5)
            ax.text(float(row["x"]) - x0 + 2, float(row["y"]) - y0 + 2, cid, color="yellow", fontsize=5)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    _save_figure(fig, output_stem)


def _empty_case_figure(output_stem: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.38, message, ha="center", va="center", fontsize=9, wrap=True)
    _save_figure(fig, output_stem)


def plot_case_figures(
    image: np.ndarray,
    components: pd.DataFrame,
    edges: pd.DataFrame,
    memberships: dict[str, pd.DataFrame],
    case_rankings: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    out = Path(output_dir) / "figures"
    if not case_rankings.empty and "case_type" in case_rankings:
        weak = case_rankings[case_rankings["case_type"] == "weak_chain"]
    else:
        weak = pd.DataFrame()
    if weak.empty:
        _empty_case_figure(out / "figure4_unconstrained_vs_constrained_cases", "Figure 4", "No weak-chain unconstrained/constrained difference case found in this run.")
    else:
        row = weak.iloc[0]
        uncon = memberships.get("unconstrained_graph:strong_plus_accepted_weak", pd.DataFrame())
        full = memberships.get("full_layer1:current_config", pd.DataFrame())
        comp_ids = set(uncon.loc[uncon["predicted_group_id"].astype(str) == str(row["unconstrained_group_id"]), "component_id"].astype(str))
        _plot_case(
            image,
            components,
            edges,
            [("radio + strong/weak edges", uncon), ("unconstrained", uncon), ("full constrained", full)],
            comp_ids,
            "Unconstrained vs constrained weak-chain case",
            out / "figure4_unconstrained_vs_constrained_cases",
        )

    full = memberships.get("full_layer1:current_config", pd.DataFrame())
    pyb = memberships.get("pybdsf_island:native", pd.DataFrame())
    contour = next((value for key, value in memberships.items() if key.startswith("contour_3sigma:")), pd.DataFrame())
    if full.empty or pyb.empty or contour.empty:
        _empty_case_figure(out / "figure5_pybdsf_vs_3sigma_cases", "Figure 5", "PyBDSF/full/3 sigma membership outputs were not all available.")
    else:
        pyb_map = _component_group_map(pyb)
        candidate_ids: set[str] = set()
        for _gid, rows in full.groupby("predicted_group_id"):
            ids = set(rows["component_id"].astype(str))
            if len({pyb_map.get(cid) for cid in ids}) >= 2 and len(ids) >= 2:
                candidate_ids = ids
                break
        if candidate_ids:
            _plot_case(
                image,
                components,
                edges,
                [("PyBDSF island", pyb), ("3 sigma connectivity", contour), ("full Layer-1", full)],
                candidate_ids,
                "PyBDSF island and 3 sigma connectivity difference case",
                out / "figure5_pybdsf_vs_3sigma_cases",
            )
        else:
            _empty_case_figure(out / "figure5_pybdsf_vs_3sigma_cases", "Figure 5", "No full group spanning multiple PyBDSF islands was found in this run.")

    distance_keys = sorted(key for key in memberships if key.startswith("distance_only:tau_"))
    if len(distance_keys) < 3:
        _empty_case_figure(out / "figure6_distance_threshold_cases", "Figure 6", "Fewer than three distance thresholds were available.")
    else:
        low = memberships[distance_keys[0]]
        mid = memberships[distance_keys[len(distance_keys) // 2]]
        high = memberships[distance_keys[-1]]
        high_groups = high.groupby("predicted_group_id").size().sort_values(ascending=False)
        if high_groups.empty:
            _empty_case_figure(out / "figure6_distance_threshold_cases", "Figure 6", "No distance groups found.")
        else:
            gid = str(high_groups.index[0])
            comp_ids = set(high.loc[high["predicted_group_id"].astype(str) == gid, "component_id"].astype(str))
            _plot_case(
                image,
                components,
                edges,
                [("small threshold", low), ("middle threshold", mid), ("large threshold", high)],
                comp_ids,
                "Distance-only threshold sensitivity case",
                out / "figure6_distance_threshold_cases",
            )


def make_all_plots(
    image: np.ndarray,
    components: pd.DataFrame,
    edges: pd.DataFrame,
    groups: dict[str, pd.DataFrame],
    memberships: dict[str, pd.DataFrame],
    distance_stats: pd.DataFrame,
    agreement: pd.DataFrame,
    case_rankings: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Generate all diagnostic figures requested by the tile demo."""

    plot_group_size_distribution(groups, output_dir)
    plot_distance_sensitivity(distance_stats, output_dir)
    plot_distance_threshold_named_figures(distance_stats, output_dir)
    plot_agreement_matrix(agreement, output_dir)
    plot_case_figures(image, components, edges, memberships, case_rankings, output_dir)
