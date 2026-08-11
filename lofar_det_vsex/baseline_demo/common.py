"""Shared helpers for baseline-demo grouping outputs."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from lofar_det_vsex.utils import safe_float, write_dataframe


GROUP_COLUMNS = [
    "method",
    "parameter_id",
    "group_id",
    "n_components",
    "component_ids",
    "total_flux",
    "peak_snr",
    "xmin",
    "xmax",
    "ymin",
    "ymax",
    "ra_min",
    "ra_max",
    "dec_min",
    "dec_max",
    "boundary_flag",
]

MEMBERSHIP_COLUMNS = [
    "method",
    "parameter_id",
    "component_id",
    "predicted_group_id",
]


def parameter_label(value: Any) -> str:
    """Return a filesystem-safe label for a method parameter."""

    if value is None or value == "":
        return "default"
    text = str(value).strip().replace("-", "m").replace(".", "p")
    text = text.replace(" ", "_").replace("/", "_")
    return text


def component_id_series(components: pd.DataFrame) -> pd.Series:
    """Return stable component identifiers as strings."""

    if "component_id" in components:
        return components["component_id"].astype(str)
    if "_gaussian_id" in components:
        return components["_gaussian_id"].astype(str)
    return components["component_index"].astype(str)


def graph_clusters_from_edges(
    component_indices: Iterable[int],
    edges: Iterable[tuple[int, int]],
) -> list[list[int]]:
    """Return connected components for a graph with singleton nodes retained."""

    graph = nx.Graph()
    graph.add_nodes_from(int(idx) for idx in component_indices)
    graph.add_edges_from((int(i), int(j)) for i, j in edges)
    clusters = [sorted(int(node) for node in cluster) for cluster in nx.connected_components(graph)]
    clusters.sort(key=lambda values: (values[0] if values else -1))
    return clusters


def membership_from_clusters(
    clusters: list[list[int]],
    components: pd.DataFrame,
    method: str,
    parameter_id: str,
    group_prefix: str,
) -> pd.DataFrame:
    """Build component membership table from component-index clusters."""

    id_by_index = dict(zip(components["component_index"].astype(int), component_id_series(components)))
    records: list[dict[str, Any]] = []
    for group_idx, cluster in enumerate(clusters):
        group_id = f"{group_prefix}_{group_idx:05d}"
        for node in cluster:
            records.append(
                {
                    "method": method,
                    "parameter_id": parameter_id,
                    "component_id": str(id_by_index.get(int(node), int(node))),
                    "predicted_group_id": group_id,
                }
            )
    return pd.DataFrame(records, columns=MEMBERSHIP_COLUMNS)


def _boundary_flag(rows: pd.DataFrame, image_shape: tuple[int, int] | None, padding_pixels: int) -> str:
    if image_shape is None or rows.empty:
        return "unknown"
    height, width = image_shape
    x = pd.to_numeric(rows.get("x", pd.Series(dtype=float)), errors="coerce")
    y = pd.to_numeric(rows.get("y", pd.Series(dtype=float)), errors="coerce")
    if x.empty or y.empty:
        return "unknown"
    margin = max(0, int(padding_pixels))
    near = (
        (x <= margin)
        | (x >= max(width - 1 - margin, 0))
        | (y <= margin)
        | (y >= max(height - 1 - margin, 0))
    )
    return "boundary_affected" if bool(near.any()) else "interior"


def group_summary_from_membership(
    membership: pd.DataFrame,
    components: pd.DataFrame,
    method: str,
    parameter_id: str,
    image_shape: tuple[int, int] | None = None,
    boundary_padding_pixels: int = 0,
) -> pd.DataFrame:
    """Build the unified group summary table."""

    if membership.empty:
        return pd.DataFrame(columns=GROUP_COLUMNS)

    work = components.copy()
    work["component_id"] = component_id_series(work)
    joined = membership.merge(work, on="component_id", how="left")
    records: list[dict[str, Any]] = []
    for group_id, rows in joined.groupby("predicted_group_id", sort=True):
        x = pd.to_numeric(rows.get("x", pd.Series(dtype=float)), errors="coerce")
        y = pd.to_numeric(rows.get("y", pd.Series(dtype=float)), errors="coerce")
        ra = pd.to_numeric(rows.get("ra", rows.get("_ra", pd.Series(dtype=float))), errors="coerce")
        dec = pd.to_numeric(rows.get("dec", rows.get("_dec", pd.Series(dtype=float))), errors="coerce")
        total_flux = pd.to_numeric(
            rows.get("total_flux", rows.get("_total_flux", pd.Series(dtype=float))),
            errors="coerce",
        )
        peak_snr = pd.to_numeric(rows.get("peak_snr", pd.Series(dtype=float)), errors="coerce")
        component_ids = sorted(str(value) for value in rows["component_id"].dropna().astype(str).tolist())
        records.append(
            {
                "method": method,
                "parameter_id": parameter_id,
                "group_id": str(group_id),
                "n_components": int(len(component_ids)),
                "component_ids": ",".join(component_ids),
                "total_flux": float(np.nansum(total_flux.to_numpy(float))) if len(total_flux) else np.nan,
                "peak_snr": float(np.nanmax(peak_snr.to_numpy(float))) if peak_snr.notna().any() else np.nan,
                "xmin": safe_float(x.min()),
                "xmax": safe_float(x.max()),
                "ymin": safe_float(y.min()),
                "ymax": safe_float(y.max()),
                "ra_min": safe_float(ra.min()),
                "ra_max": safe_float(ra.max()),
                "dec_min": safe_float(dec.min()),
                "dec_max": safe_float(dec.max()),
                "boundary_flag": _boundary_flag(rows, image_shape, boundary_padding_pixels),
            }
        )
    return pd.DataFrame(records, columns=GROUP_COLUMNS)


def write_method_outputs(
    output_dir: str | Path,
    stem: str,
    groups: pd.DataFrame,
    membership: pd.DataFrame,
) -> None:
    """Write per-method group and membership outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    groups = groups.reindex(columns=GROUP_COLUMNS)
    membership = membership.reindex(columns=MEMBERSHIP_COLUMNS)
    write_dataframe(groups, output / f"{stem}.parquet")
    groups.to_csv(output / f"{stem}.csv", index=False)
    write_dataframe(membership, output / f"{stem}_membership.parquet")
    membership.to_csv(output / f"{stem}_membership.csv", index=False)


def labels_from_membership(membership: pd.DataFrame, component_ids: Iterable[str]) -> dict[str, str]:
    """Map every component id to a predicted group id."""

    labels = {str(component_id): str(component_id) for component_id in component_ids}
    if membership.empty:
        return labels
    for _, row in membership.iterrows():
        labels[str(row["component_id"])] = str(row["predicted_group_id"])
    return labels


def parse_component_ids(value: Any) -> list[str]:
    """Parse comma-separated component-id fields."""

    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]
