"""Candidate neighbour generation for fair baseline comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from lotss_association.association import compute_beam_size_arcsec
from lotss_association.utils import safe_float


@dataclass
class NeighbourSearchResult:
    """Candidate-pair table and summary statistics."""

    pairs: pd.DataFrame
    stats: dict[str, Any]


def max_distance_beam(config: dict[str, Any]) -> float:
    """Return configured candidate-search radius in beam units."""

    neighbour = config.get("neighbour_search", {}) or {}
    if "max_distance_beam" in neighbour:
        return float(neighbour["max_distance_beam"])
    association = config.get("association", {}) or {}
    return float(association.get("max_pair_distance_beam", config.get("max_pair_distance_beam", 15.0)))


def find_candidate_pairs(components: pd.DataFrame, config: dict[str, Any]) -> NeighbourSearchResult:
    """Generate nearby component pairs using a cKDTree."""

    t0 = perf_counter()
    columns = [
        "component_index_1",
        "component_index_2",
        "component_id_1",
        "component_id_2",
        "distance_pix",
        "distance_arcsec",
        "distance_beam",
    ]
    if len(components) < 2:
        return NeighbourSearchResult(
            pairs=pd.DataFrame(columns=columns),
            stats={
                "n_components": int(len(components)),
                "n_candidate_pairs": 0,
                "candidate_neighbour_min": 0,
                "candidate_neighbour_median": 0.0,
                "candidate_neighbour_p90": 0.0,
                "candidate_neighbour_max": 0,
                "runtime_seconds": perf_counter() - t0,
                "method": "ckdtree",
            },
        )

    coords = components[["x", "y"]].to_numpy(float)
    finite = np.isfinite(coords).all(axis=1)
    valid_positions = np.where(finite)[0]
    if len(valid_positions) < 2:
        return NeighbourSearchResult(
            pairs=pd.DataFrame(columns=columns),
            stats={
                "n_components": int(len(components)),
                "n_candidate_pairs": 0,
                "candidate_neighbour_min": 0,
                "candidate_neighbour_median": 0.0,
                "candidate_neighbour_p90": 0.0,
                "candidate_neighbour_max": 0,
                "runtime_seconds": perf_counter() - t0,
                "method": "ckdtree",
            },
        )

    pixel_scale = safe_float(components["pixel_scale_arcsec"].iloc[0], safe_float(config.get("pixel_scale_arcsec"), 1.5))
    beam_arcsec = compute_beam_size_arcsec(config)
    radius_beam = max_distance_beam(config)
    radius_pix = radius_beam * beam_arcsec / max(pixel_scale, 1e-6)
    tree = cKDTree(coords[finite])
    local_pairs = sorted(tree.query_pairs(radius_pix))
    id_series = components["component_id"].astype(str) if "component_id" in components else components["_gaussian_id"].astype(str)

    records: list[dict[str, Any]] = []
    degree = np.zeros(len(components), dtype=int)
    for i_local, j_local in local_pairs:
        i = int(valid_positions[i_local])
        j = int(valid_positions[j_local])
        dx = float(coords[i, 0] - coords[j, 0])
        dy = float(coords[i, 1] - coords[j, 1])
        distance_pix = float(np.hypot(dx, dy))
        distance_arcsec = distance_pix * pixel_scale
        records.append(
            {
                "component_index_1": int(components.iloc[i]["component_index"]),
                "component_index_2": int(components.iloc[j]["component_index"]),
                "component_id_1": str(id_series.iloc[i]),
                "component_id_2": str(id_series.iloc[j]),
                "distance_pix": distance_pix,
                "distance_arcsec": distance_arcsec,
                "distance_beam": distance_arcsec / max(beam_arcsec, 1e-6),
            }
        )
        degree[i] += 1
        degree[j] += 1

    pairs = pd.DataFrame(records, columns=columns)
    stats = {
        "n_components": int(len(components)),
        "n_candidate_pairs": int(len(pairs)),
        "candidate_neighbour_min": int(degree.min()) if len(degree) else 0,
        "candidate_neighbour_median": float(np.median(degree)) if len(degree) else 0.0,
        "candidate_neighbour_p90": float(np.percentile(degree, 90)) if len(degree) else 0.0,
        "candidate_neighbour_max": int(degree.max()) if len(degree) else 0,
        "runtime_seconds": perf_counter() - t0,
        "method": "ckdtree",
        "max_distance_beam": float(radius_beam),
        "radius_pixels": float(radius_pix),
    }
    return NeighbourSearchResult(pairs=pairs, stats=stats)
