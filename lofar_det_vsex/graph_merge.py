"""Gaussian component graph construction and rule-based merging."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .segmentation import connected_at_threshold
from .utils import json_dumps_safe, safe_float


@dataclass
class GraphMergeResult:
    graph: nx.Graph
    edges: pd.DataFrame
    components: pd.DataFrame
    clusters: list[list[int]]


def _line_samples(x1: float, y1: float, x2: float, y2: float, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    distance = float(np.hypot(x2 - x1, y2 - y1))
    n_samples = n or max(3, int(distance) + 1)
    xs = np.linspace(x1, x2, n_samples)
    ys = np.linspace(y1, y2, n_samples)
    return xs, ys


def _sample_image_nearest(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    height, width = image.shape
    xi = np.rint(xs).astype(int)
    yi = np.rint(ys).astype(int)
    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    values = np.full(len(xs), np.nan, dtype=float)
    values[valid] = image[yi[valid], xi[valid]]
    return values


def _bridge_features(snr_map: np.ndarray, row_i: pd.Series, row_j: pd.Series) -> dict[str, float]:
    xs, ys = _line_samples(row_i["x"], row_i["y"], row_j["x"], row_j["y"])
    values = _sample_image_nearest(snr_map, xs, ys)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "bridge_snr_mean": 0.0,
            "bridge_snr_min": 0.0,
            "bridge_snr_max": 0.0,
            "bridge_snr_score": 0.0,
            "valley_penalty": 1.0,
            "negative_bowl_penalty": 0.0,
            "pixel_support_score": 0.0,
        }
    mean = float(np.nanmean(finite))
    min_value = float(np.nanmin(finite))
    max_value = float(np.nanmax(finite))
    bridge_score = float(np.clip((mean - 1.0) / 2.0, 0.0, 1.5))
    support_score = float(np.mean(finite > 2.0))
    valley_penalty = float(np.clip((2.0 - min_value) / 4.0, 0.0, 1.5))
    negative_bowl_penalty = float(np.clip(-min_value / 3.0, 0.0, 2.0))
    return {
        "bridge_snr_mean": mean,
        "bridge_snr_min": min_value,
        "bridge_snr_max": max_value,
        "bridge_snr_score": bridge_score,
        "valley_penalty": valley_penalty,
        "negative_bowl_penalty": negative_bowl_penalty,
        "pixel_support_score": support_score,
    }


def _pa_alignment(row_i: pd.Series, row_j: pd.Series) -> tuple[float, float]:
    pa_i = safe_float(row_i.get("_pa"))
    pa_j = safe_float(row_j.get("_pa"))
    if not np.isfinite(pa_i) or not np.isfinite(pa_j):
        return 0.0, 0.0
    delta = abs((pa_i - pa_j + 90.0) % 180.0 - 90.0)
    score = float(np.clip(1.0 - delta / 45.0, 0.0, 1.0))
    penalty = float(np.clip((delta - 45.0) / 45.0, 0.0, 1.0))
    return score, penalty


def _flux_ratio_score(row_i: pd.Series, row_j: pd.Series) -> float:
    f1 = safe_float(row_i.get("_total_flux"))
    f2 = safe_float(row_j.get("_total_flux"))
    if not np.isfinite(f1) or not np.isfinite(f2) or f1 <= 0 or f2 <= 0:
        return 0.5
    ratio = min(f1, f2) / max(f1, f2)
    return float(np.clip(ratio, 0.0, 1.0))


def _ellipse_overlap_approx(row_i: pd.Series, row_j: pd.Series, distance_pix: float) -> float:
    maj_i = safe_float(row_i.get("_dc_maj"), safe_float(row_i.get("_maj")))
    maj_j = safe_float(row_j.get("_dc_maj"), safe_float(row_j.get("_maj")))
    scale = safe_float(row_i.get("pixel_scale_arcsec"), 1.5)
    if not np.isfinite(maj_i) or not np.isfinite(maj_j) or scale <= 0:
        return 0.0
    radius_pix = 0.5 * (maj_i + maj_j) / scale
    if radius_pix <= 0:
        return 0.0
    return float(np.clip(1.0 - distance_pix / max(radius_pix, 1e-6), 0.0, 1.0))


def _evidence_strings(features: dict[str, Any]) -> tuple[str, str]:
    positive = []
    negative = []
    for key in [
        "same_pybdsf_island",
        "connected_at_3sigma",
        "connected_at_2p5sigma",
        "connected_at_2sigma",
    ]:
        if features.get(key):
            positive.append(key)
    for key in ["bridge_snr_score", "gaussian_ellipse_overlap_approx", "PA_alignment_score"]:
        if features.get(key, 0) > 0.5:
            positive.append(key)
    for key in ["valley_penalty", "negative_bowl_penalty", "too_far_penalty", "pa_misalignment_penalty"]:
        if features.get(key, 0) > 0.5:
            negative.append(key)
    return ",".join(positive), ",".join(negative)


def compute_pair_features(
    row_i: pd.Series,
    row_j: pd.Series,
    segmentation: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute graph edge features for one Gaussian pair."""

    dx = safe_float(row_i["x"]) - safe_float(row_j["x"])
    dy = safe_float(row_i["y"]) - safe_float(row_j["y"])
    distance_pix = float(np.hypot(dx, dy))
    pixel_scale = safe_float(row_i.get("pixel_scale_arcsec"), 1.5)
    distance_arcsec = distance_pix * pixel_scale
    beam = config.get("beam", {})
    beam_major = float(beam.get("major_arcsec", 6.0) or 6.0)
    beam_norm = distance_arcsec / max(beam_major, 1e-6)

    same_island = (
        str(row_i.get("_island_id")) == str(row_j.get("_island_id"))
        and str(row_i.get("_island_id")) not in {"-1", "nan", "None"}
    )
    conn_3 = connected_at_threshold(row_i, row_j, 3.0)
    conn_25 = connected_at_threshold(row_i, row_j, 2.5)
    conn_2 = connected_at_threshold(row_i, row_j, 2.0)
    bridge = _bridge_features(segmentation.snr_map, row_i, row_j)
    pa_score, pa_penalty = _pa_alignment(row_i, row_j)
    flux_ratio = _flux_ratio_score(row_i, row_j)
    overlap = _ellipse_overlap_approx(row_i, row_j, distance_pix)

    max_pair_distance_arcsec = float(config.get("max_pair_distance_arcsec", 180.0))
    max_pair_distance_beam = float(config.get("max_pair_distance_beam", 20.0))
    too_far_penalty = float(
        np.clip(
            max(
                distance_arcsec / max(max_pair_distance_arcsec, 1e-6),
                beam_norm / max(max_pair_distance_beam, 1e-6),
            )
            - 0.75,
            0.0,
            1.5,
        )
    )
    closeness_score = float(np.clip(1.0 - beam_norm / max(max_pair_distance_beam, 1e-6), 0.0, 1.0))
    compact_pair_penalty = 0.0
    if beam_norm < 1.5 and flux_ratio < 0.2:
        compact_pair_penalty = 0.5

    features: dict[str, Any] = {
        "cutout_id": row_i.get("cutout_id"),
        "gaussian_id_1": row_i.get("_gaussian_id"),
        "gaussian_id_2": row_j.get("_gaussian_id"),
        "component_index_1": int(row_i.get("component_index")),
        "component_index_2": int(row_j.get("component_index")),
        "distance_pix": distance_pix,
        "distance_arcsec": distance_arcsec,
        "beam_normalized_distance": beam_norm,
        "same_pybdsf_island": bool(same_island),
        "connected_at_3sigma": bool(conn_3),
        "connected_at_2p5sigma": bool(conn_25),
        "connected_at_2sigma": bool(conn_2),
        "gaussian_ellipse_overlap_approx": overlap,
        "PA_alignment_score": pa_score,
        "flux_ratio_score": flux_ratio,
        "closeness_score": closeness_score,
        "compact_pair_penalty": compact_pair_penalty,
        "too_far_penalty": too_far_penalty,
        "pa_misalignment_penalty": pa_penalty,
    }
    features.update(bridge)
    return features


def score_pair(features: dict[str, Any], config: dict[str, Any]) -> float:
    """Compute the rule-based merge score."""

    weights = config.get("weights", {})
    score = 0.0
    score += float(weights.get("same_island", 1.0)) * float(features["same_pybdsf_island"])
    score += float(weights.get("conn_3sigma", 2.0)) * float(features["connected_at_3sigma"])
    score += float(weights.get("conn_2p5sigma", 1.5)) * float(features["connected_at_2p5sigma"])
    score += float(weights.get("conn_2sigma", 1.0)) * float(features["connected_at_2sigma"])
    score += float(weights.get("bridge", 1.0)) * float(features["bridge_snr_score"])
    score += float(weights.get("overlap", 0.8)) * float(features["gaussian_ellipse_overlap_approx"])
    score += float(weights.get("closeness", 0.8)) * float(features["closeness_score"])
    score += float(weights.get("pa_alignment", 0.7)) * float(features["PA_alignment_score"])
    score += float(weights.get("pixel_support", 0.8)) * float(features["pixel_support_score"])
    score -= float(weights.get("valley", 1.2)) * float(features["valley_penalty"])
    score -= float(weights.get("compact_pair", 1.0)) * float(features["compact_pair_penalty"])
    score -= float(weights.get("too_far", 2.0)) * float(features["too_far_penalty"])
    score -= 0.5 * float(features.get("negative_bowl_penalty", 0.0))
    score -= 0.5 * float(features.get("pa_misalignment_penalty", 0.0))
    return float(score)


def candidate_pairs(components: pd.DataFrame, config: dict[str, Any]) -> list[tuple[int, int]]:
    """Generate nearby component pairs using cKDTree."""

    if len(components) < 2:
        return []
    coords = components[["x", "y"]].to_numpy(float)
    finite = np.isfinite(coords).all(axis=1)
    if finite.sum() < 2:
        return []
    valid_positions = np.where(finite)[0]
    valid_coords = coords[finite]
    pixel_scale = safe_float(components["pixel_scale_arcsec"].iloc[0], 1.5)
    beam_major = float(config.get("beam", {}).get("major_arcsec", 6.0) or 6.0)
    max_arcsec = float(config.get("max_pair_distance_arcsec", 180.0))
    max_beam = float(config.get("max_pair_distance_beam", 20.0)) * beam_major
    max_distance_arcsec = min(max_arcsec, max_beam)
    radius_pix = max_distance_arcsec / max(pixel_scale, 1e-6)
    tree = cKDTree(valid_coords)
    pairs_local = tree.query_pairs(radius_pix)
    pairs = [(int(valid_positions[i]), int(valid_positions[j])) for i, j in pairs_local]
    pairs.sort()
    return pairs


def build_component_graph(
    components: pd.DataFrame,
    segmentation: Any,
    config: dict[str, Any],
) -> GraphMergeResult:
    """Build a graph and merge components by connected components."""

    graph = nx.Graph()
    for _, row in components.iterrows():
        node_id = int(row["component_index"])
        graph.add_node(node_id, **row.to_dict())

    edge_records = []
    threshold = float(config.get("merge_threshold", 2.5))
    for idx_i, idx_j in candidate_pairs(components, config):
        row_i = components.iloc[idx_i]
        row_j = components.iloc[idx_j]
        features = compute_pair_features(row_i, row_j, segmentation, config)
        score = score_pair(features, config)
        decision = score > threshold
        positive, negative = _evidence_strings(features)
        features["merge_score"] = score
        features["merge_decision"] = bool(decision)
        features["positive_evidence"] = positive
        features["negative_evidence"] = negative
        edge_records.append(features)
        if decision:
            graph.add_edge(
                int(row_i["component_index"]),
                int(row_j["component_index"]),
                merge_score=score,
                features=features,
            )

    edges = pd.DataFrame(edge_records)
    clusters = [sorted(list(cluster)) for cluster in nx.connected_components(graph)]
    clusters.sort(key=lambda values: (len(values), values[0] if values else -1), reverse=True)
    components = components.copy()
    cluster_id_by_node = {}
    for cluster_idx, cluster in enumerate(clusters):
        for node in cluster:
            cluster_id_by_node[node] = cluster_idx
    components["merged_component_group"] = components["component_index"].map(cluster_id_by_node)
    components["debug_graph_degree"] = components["component_index"].map(dict(graph.degree())).fillna(0).astype(int)
    if not edges.empty:
        edges["debug_info"] = edges.apply(lambda row: json_dumps_safe(row.to_dict()), axis=1)
    return GraphMergeResult(graph=graph, edges=edges, components=components, clusters=clusters)


def strongest_evidence_for_cluster(edges: pd.DataFrame, nodes: list[int]) -> tuple[float, float, str]:
    """Return mean score, max score, and evidence string for edges inside a cluster."""

    if edges.empty or len(nodes) < 2:
        return 0.0, 0.0, ""
    node_set = set(nodes)
    mask = edges["component_index_1"].isin(node_set) & edges["component_index_2"].isin(node_set) & edges[
        "merge_decision"
    ].astype(bool)
    subset = edges.loc[mask]
    if subset.empty:
        return 0.0, 0.0, ""
    mean_score = float(subset["merge_score"].mean())
    max_idx = subset["merge_score"].idxmax()
    max_score = float(subset.loc[max_idx, "merge_score"])
    evidence = str(subset.loc[max_idx, "positive_evidence"])
    return mean_score, max_score, evidence


def complete_graph_edges_for_singletons(components: pd.DataFrame) -> pd.DataFrame:
    """Build a placeholder edge table for diagnostics when needed."""

    records = []
    for left, right in combinations(range(len(components)), 2):
        row_i = components.iloc[left]
        row_j = components.iloc[right]
        records.append(
            {
                "cutout_id": row_i.get("cutout_id"),
                "gaussian_id_1": row_i.get("_gaussian_id"),
                "gaussian_id_2": row_j.get("_gaussian_id"),
                "merge_score": np.nan,
                "merge_decision": False,
            }
        )
    return pd.DataFrame(records)
