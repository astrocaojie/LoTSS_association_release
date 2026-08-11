"""Analysis helpers for ablation comparisons."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from lofar_det_vsex.baseline_demo.common import component_id_series, group_summary_from_membership, membership_from_clusters, parse_component_ids
from lofar_det_vsex.baseline_demo.comparison_metrics import (
    bcubed_metrics,
    manual_label_metrics,
    overmerge_split_rates,
    pairwise_prf,
    split_merge_against_reference,
    summarize_groups,
)
from lofar_det_vsex.baseline_demo.plotting import _display_image, _save_figure
from lofar_det_vsex.baseline_demo.reporting import edge_table_hash
from lofar_det_vsex.utils import json_dumps_safe, write_dataframe


FULL_VARIANT_KEYS = ("full_method", "full")
RIDGE_VARIANT_KEYS = ("no_ridge_continuity", "no_ridge")
ANTI_CHAINING_VARIANT_KEYS = ("no_weak_edge_anti_chaining", "no_anti_chaining")
ARTIFACT_VARIANT_KEYS = ("no_artifact_penalties", "no_artifact_layer1")
LOBE_PEAK_VARIANT_KEYS = ("no_lobe_peak_host_contradiction", "no_lobe_peak_contradiction")


def _first_frame(mapping: dict[str, pd.DataFrame], keys: tuple[str, ...]) -> pd.DataFrame:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return pd.DataFrame()


def _first_layer1_frame(memberships: dict[str, pd.DataFrame], keys: tuple[str, ...]) -> pd.DataFrame:
    return _first_frame(memberships, tuple(f"layer1_ablation:{key}" for key in keys))


def _full_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table[table["ablation_id"].astype(str).isin(FULL_VARIANT_KEYS)]


def table_hash(frame: pd.DataFrame, drop_columns: set[str] | None = None) -> str:
    """Hash a deterministic CSV representation of a DataFrame."""

    if frame is None or frame.empty:
        return hashlib.sha256(b"empty").hexdigest()
    drop = drop_columns or set()
    cols = [col for col in frame.columns if col not in drop]
    work = frame.loc[:, cols].copy()
    sort_cols = [col for col in ["component_id", "component_id_1", "component_id_2", "component_index_1", "component_index_2", "parent_candidate_id"] if col in work]
    if sort_cols:
        work = work.sort_values(sort_cols).reset_index(drop=True)
    return hashlib.sha256(work.to_csv(index=False, na_rep="NaN").encode("utf-8")).hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    """Hash config values relevant to a run."""

    return hashlib.sha256(json_dumps_safe(config).encode("utf-8")).hexdigest()


def membership_key(method: str, ablation_id: str) -> str:
    return f"{method}:{ablation_id}"


def summarize_layer1_ablation(
    ablation_id: str,
    groups: pd.DataFrame,
    membership: pd.DataFrame,
    full_membership: pd.DataFrame,
    full_edges: pd.DataFrame,
    edges: pd.DataFrame,
    n_components: int,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Return one no-truth Layer-1 structural summary row."""

    row = summarize_groups(groups, "layer1_ablation", ablation_id, n_components, runtime_seconds)
    if full_membership is not None and not full_membership.empty and not membership.empty:
        split, merge = split_merge_against_reference(full_membership, membership)
        row["membership_agreement_with_full"] = _same_group_pair_agreement(full_membership, membership)
        row["number_of_full_groups_split"] = int((split.get("n_other_groups", pd.Series(dtype=int)) >= 2).sum()) if not split.empty else 0
        row["number_of_ablation_groups_merging_multiple_full_groups"] = int((merge.get("n_reference_groups", pd.Series(dtype=int)) >= 2).sum()) if not merge.empty else 0
    else:
        row["membership_agreement_with_full"] = np.nan
        row["number_of_full_groups_split"] = 0
        row["number_of_ablation_groups_merging_multiple_full_groups"] = 0
    row["edge_input_hash"] = edge_table_hash(edges)
    row["edge_table_identical_to_full"] = bool(edge_table_hash(edges) == edge_table_hash(full_edges)) if full_edges is not None else False
    row["n_strong_edges"] = int((edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "strong").sum()) if edges is not None and not edges.empty else 0
    row["n_weak_edges"] = int((edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "weak").sum()) if edges is not None and not edges.empty else 0
    row["n_rejected_edges"] = int((edges.get("edge_type", pd.Series(dtype=str)).astype(str) == "rejected").sum()) if edges is not None and not edges.empty else 0
    anti = anti_chaining_stats(full_membership, membership, edges)
    row.update(
        {
            "number_of_strong_cores_merged": anti.get("n_groups_containing_multiple_full_groups", 0),
            "number_of_weak_only_chain_mergers": anti.get("n_weak_only_chains", 0),
            "largest_merged_group": anti.get("largest_merged_group", 0),
        }
    )
    return row


def _same_group_pair_agreement(reference: pd.DataFrame, other: pd.DataFrame) -> float:
    ref = dict(zip(reference["component_id"].astype(str), reference["predicted_group_id"].astype(str)))
    oth = dict(zip(other["component_id"].astype(str), other["predicted_group_id"].astype(str)))
    ids = sorted(set(ref).intersection(oth))
    if len(ids) < 2:
        return 1.0
    agree = 0
    total = 0
    for a, b in combinations(ids, 2):
        agree += int((ref[a] == ref[b]) == (oth[a] == oth[b]))
        total += 1
    return float(agree / max(total, 1))


def anti_chaining_stats(full_membership: pd.DataFrame, other_membership: pd.DataFrame, edges: pd.DataFrame) -> dict[str, Any]:
    """Quantify groups where weak paths merge multiple full groups."""

    if full_membership.empty or other_membership.empty:
        return {
            "n_groups_containing_multiple_full_groups": 0,
            "n_weak_only_chains": 0,
            "weak_path_length_distribution": "",
            "largest_merged_group": 0,
            "fraction_groups_affected": 0.0,
        }
    full_by_component = dict(zip(full_membership["component_id"].astype(str), full_membership["predicted_group_id"].astype(str)))
    rows = []
    for gid, frame in other_membership.groupby("predicted_group_id"):
        cids = set(frame["component_id"].astype(str))
        full_ids = sorted({full_by_component[cid] for cid in cids if cid in full_by_component})
        if len(full_ids) < 2:
            continue
        rows.append((str(gid), cids, full_ids))
    path_lengths: list[int] = []
    weak_only = 0
    for _gid, cids, full_ids in rows:
        core_graph = nx.Graph()
        core_graph.add_nodes_from(full_ids)
        if edges is not None and not edges.empty:
            for _, edge in edges.iterrows():
                if str(edge.get("edge_type")) != "weak":
                    continue
                c1 = str(edge.get("gaussian_id_1", edge.get("component_id_1", "")))
                c2 = str(edge.get("gaussian_id_2", edge.get("component_id_2", "")))
                if c1 in cids and c2 in cids:
                    g1 = full_by_component.get(c1)
                    g2 = full_by_component.get(c2)
                    if g1 and g2 and g1 != g2:
                        core_graph.add_edge(g1, g2)
        if len(core_graph.edges):
            weak_only += 1
            for source, lengths in nx.all_pairs_shortest_path_length(core_graph):
                for target, length in lengths.items():
                    if str(source) < str(target):
                        path_lengths.append(int(length))
    hist = pd.Series(path_lengths).value_counts().sort_index()
    return {
        "n_groups_containing_multiple_full_groups": int(len(rows)),
        "n_weak_only_chains": int(weak_only),
        "weak_path_length_distribution": json.dumps({str(k): int(v) for k, v in hist.items()}, sort_keys=True),
        "largest_merged_group": int(max((len(cids) for _gid, cids, _full_ids in rows), default=0)),
        "fraction_groups_affected": float(len(rows) / max(other_membership["predicted_group_id"].nunique(), 1)),
    }


def layer1_delta_table(table: pd.DataFrame) -> pd.DataFrame:
    """Return numeric deltas relative to full for structural columns."""

    if table.empty:
        return table
    full_rows = _full_rows(table)
    if full_rows.empty:
        return pd.DataFrame()
    full = full_rows.iloc[0]
    rows = []
    for _, row in table.iterrows():
        out = {"ablation_id": row["ablation_id"]}
        for col in [
            "n_singletons",
            "n_multi_component_groups",
            "fraction_components_in_multi_groups",
            "median_group_size",
            "p90_group_size",
            "max_group_size",
            "number_of_full_groups_split",
            "number_of_ablation_groups_merging_multiple_full_groups",
            "number_of_strong_cores_merged",
            "number_of_weak_only_chain_mergers",
        ]:
            if col in table:
                out[f"delta_{col}"] = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([full[col]]), errors="coerce").iloc[0]
        rows.append(out)
    return pd.DataFrame(rows)


def layer2_summary_row(ablation_id: str, edges: pd.DataFrame, candidates: pd.DataFrame, full_edges: pd.DataFrame | None, runtime_seconds: float) -> dict[str, Any]:
    """Return one no-truth Layer-2 structural summary row."""

    quality = edges.get("parent_candidate_quality", pd.Series(dtype=str)).astype(str) if edges is not None and not edges.empty else pd.Series(dtype=str)
    full_quality: dict[str, str] = {}
    if full_edges is not None and not full_edges.empty and "parent_candidate_id" in full_edges:
        full_quality = dict(zip(full_edges["parent_candidate_id"].astype(str), full_edges.get("parent_candidate_quality", pd.Series("", index=full_edges.index)).astype(str)))
    accepted_values = {"high", "medium", "needs_host_check", "suspicious"}
    ids = edges.get("parent_candidate_id", pd.Series(dtype=str)).astype(str) if edges is not None and not edges.empty else pd.Series(dtype=str)
    accepted = quality.isin(accepted_values)
    full_accept = ids.map(lambda value: full_quality.get(str(value), "rejected")).isin(accepted_values) if len(ids) else pd.Series(dtype=bool)
    rank = {"rejected": 0, "": 0, "suspicious": 1, "needs_host_check": 2, "medium": 3, "high": 4}
    q_rank = quality.map(rank).fillna(0)
    f_rank = ids.map(lambda value: rank.get(full_quality.get(str(value), "rejected"), 0)) if len(ids) else pd.Series(dtype=int)
    return {
        "ablation_id": ablation_id,
        "method": "layer2_ablation",
        "n_candidate_pairs": int(len(edges)) if edges is not None else 0,
        "n_accepted": int(accepted.sum()) if len(quality) else 0,
        "n_high": int((quality == "high").sum()),
        "n_medium": int((quality == "medium").sum()),
        "n_suspicious": int((quality == "suspicious").sum()),
        "n_needs_host_check": int((quality == "needs_host_check").sum()),
        "accepted_only_in_ablation": int((accepted & ~full_accept).sum()) if len(quality) and full_quality else 0,
        "rejected_only_in_ablation": int((~accepted & full_accept).sum()) if len(quality) and full_quality else 0,
        "upgraded": int((q_rank > f_rank).sum()) if len(quality) and full_quality else 0,
        "downgraded": int((q_rank < f_rank).sum()) if len(quality) and full_quality else 0,
        "number_with_midpoint_host": int((edges.get("host_quality", pd.Series(dtype=str)).astype(str).isin(["high", "medium"])).sum()) if edges is not None and not edges.empty else 0,
        "number_with_lobe_peak_host_contradiction": int((edges.get("rejection_reason", pd.Series(dtype=str)).astype(str) == "lobe_peak_host_contradiction").sum()) if edges is not None and not edges.empty else 0,
        "runtime_seconds": float(runtime_seconds),
        "candidate_table_hash": table_hash(edges, drop_columns={"debug_info"}) if edges is not None else table_hash(pd.DataFrame()),
    }


def layer2_delta_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    full_rows = _full_rows(table)
    if full_rows.empty:
        return pd.DataFrame()
    full = full_rows.iloc[0]
    rows = []
    for _, row in table.iterrows():
        out = {"ablation_id": row["ablation_id"]}
        for col in [
            "n_candidate_pairs",
            "n_accepted",
            "n_high",
            "n_medium",
            "n_suspicious",
            "n_needs_host_check",
            "accepted_only_in_ablation",
            "rejected_only_in_ablation",
            "upgraded",
            "downgraded",
        ]:
            if col in table:
                out[f"delta_{col}"] = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([full[col]]), errors="coerce").iloc[0]
        rows.append(out)
    return pd.DataFrame(rows)


def write_layer1_manual_metrics(
    manual_labels_path: str | Path | None,
    memberships: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Compute optional Layer-1 manual metrics in existing label format."""

    if not manual_labels_path:
        return pd.DataFrame()
    metrics = manual_label_metrics(manual_labels_path, memberships)
    if not metrics.empty:
        metrics.to_csv(Path(output_dir) / "layer1_manual_metrics.csv", index=False)
    return metrics


def layer2_manual_metrics(
    manual_labels_path: str | Path | None,
    edges_by_ablation: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Compute simple parent-pair metrics if manual Layer-2 labels are supplied."""

    if not manual_labels_path:
        return pd.DataFrame()
    path = Path(manual_labels_path)
    if not path.exists():
        return pd.DataFrame()
    labels = pd.read_csv(path)
    required = {"candidate_pair_id", "true_same_parent", "label_quality"}
    if not required.issubset(labels.columns):
        return pd.DataFrame()
    labels = labels[labels["label_quality"].astype(str) != "uncertain"].copy()
    if labels.empty:
        return pd.DataFrame()
    truth = labels.set_index("candidate_pair_id")["true_same_parent"].map(lambda value: str(value).lower() in {"1", "true", "yes", "same"}).to_dict()
    rows = []
    accepted_values = {"high", "medium", "needs_host_check", "suspicious"}
    for ablation_id, edges in edges_by_ablation.items():
        if edges.empty:
            continue
        work = edges[edges["parent_candidate_id"].astype(str).isin(truth)].copy()
        if work.empty:
            continue
        pred = work["parent_candidate_quality"].astype(str).isin(accepted_values).to_numpy(bool)
        true = work["parent_candidate_id"].astype(str).map(truth).to_numpy(bool)
        tp = int(np.count_nonzero(pred & true))
        fp = int(np.count_nonzero(pred & ~true))
        fn = int(np.count_nonzero(~pred & true))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "ablation_id": ablation_id,
                "parent_precision": precision,
                "parent_recall": recall,
                "parent_f1": f1,
                "false_parent_rate": fp / max(tp + fp, 1),
                "n_labelled_pairs": int(len(work)),
                "n_accepted": int(pred.sum()),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(Path(output_dir) / "layer2_manual_metrics.csv", index=False)
    return out


def case_tables(
    components: pd.DataFrame,
    full_membership: pd.DataFrame,
    memberships: dict[str, pd.DataFrame],
    edges_by_ablation: dict[str, pd.DataFrame],
    layer2_edges_by_ablation: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build automatically ranked ablation-sensitive case tables."""

    tables: dict[str, pd.DataFrame] = {}
    full_by_component = dict(zip(full_membership["component_id"].astype(str), full_membership["predicted_group_id"].astype(str))) if not full_membership.empty else {}

    def bbox(cids: set[str]) -> dict[str, int]:
        rows = components[components["component_id"].astype(str).isin(cids)]
        if rows.empty:
            return {"xmin": 0, "xmax": 0, "ymin": 0, "ymax": 0, "boundary_flag": "unknown"}
        x0 = int(max(0, np.floor(rows["x"].min()) - 32))
        y0 = int(max(0, np.floor(rows["y"].min()) - 32))
        return {
            "xmin": x0,
            "xmax": int(np.ceil(rows["x"].max()) + 32),
            "ymin": y0,
            "ymax": int(np.ceil(rows["y"].max()) + 32),
            "boundary_flag": "boundary_affected" if x0 <= 0 or y0 <= 0 else "interior",
        }

    no_ridge = _first_layer1_frame(memberships, RIDGE_VARIANT_KEYS)
    full_edges = _first_frame(edges_by_ablation, FULL_VARIANT_KEYS)
    if not full_membership.empty and not no_ridge.empty:
        rows = []
        no_ridge_map = dict(zip(no_ridge["component_id"].astype(str), no_ridge["predicted_group_id"].astype(str)))
        for gid, frame in full_membership.groupby("predicted_group_id"):
            cids = set(frame["component_id"].astype(str))
            if len(cids) < 2 or len({no_ridge_map.get(cid, cid) for cid in cids}) <= 1:
                continue
            internal = _internal_edges(full_edges, cids)
            ridge = pd.to_numeric(internal.get("ridge_continuity_score", pd.Series(dtype=float)), errors="coerce")
            rows.append(
                {
                    "case_id": f"ridge_sensitive_{len(rows):03d}",
                    "case_type": "ridge_sensitive",
                    "full_group_id": gid,
                    "no_ridge_group_ids": ",".join(sorted({no_ridge_map.get(cid, cid) for cid in cids})),
                    "n_components": len(cids),
                    "mean_ridge_score": float(ridge.mean()) if len(ridge) else np.nan,
                    "max_ridge_score": float(ridge.max()) if len(ridge) else np.nan,
                    "component_ids": ",".join(sorted(cids)),
                    **bbox(cids),
                }
            )
        tables["ridge_sensitive"] = pd.DataFrame(rows).sort_values(["boundary_flag", "max_ridge_score", "n_components"], ascending=[True, False, True]).head(20) if rows else pd.DataFrame()

    no_anti = _first_layer1_frame(memberships, ANTI_CHAINING_VARIANT_KEYS)
    if not full_membership.empty and not no_anti.empty:
        rows = []
        for gid, frame in no_anti.groupby("predicted_group_id"):
            cids = set(frame["component_id"].astype(str))
            full_ids = sorted({full_by_component.get(cid, "") for cid in cids if full_by_component.get(cid, "")})
            if len(full_ids) < 2:
                continue
            internal = _internal_edges(full_edges, cids)
            weak = int((internal.get("edge_type", pd.Series(dtype=str)).astype(str) == "weak").sum()) if not internal.empty else 0
            strong = int((internal.get("edge_type", pd.Series(dtype=str)).astype(str) == "strong").sum()) if not internal.empty else 0
            rows.append(
                {
                    "case_id": f"anti_chaining_{len(rows):03d}",
                    "case_type": "anti_chaining",
                    "no_anti_chaining_group_id": gid,
                    "full_group_ids": ",".join(full_ids),
                    "n_components": len(cids),
                    "n_strong_cores": len(full_ids),
                    "n_strong_edges": strong,
                    "n_weak_edges": weak,
                    "component_ids": ",".join(sorted(cids)),
                    **bbox(cids),
                }
            )
        tables["anti_chaining"] = pd.DataFrame(rows).sort_values(["boundary_flag", "n_strong_cores", "n_weak_edges", "n_components"], ascending=[True, False, False, True]).head(30) if rows else pd.DataFrame()

    no_art = _first_layer1_frame(memberships, ARTIFACT_VARIANT_KEYS)
    if not full_membership.empty and not no_art.empty:
        rows = []
        for gid, frame in no_art.groupby("predicted_group_id"):
            cids = set(frame["component_id"].astype(str))
            full_ids = sorted({full_by_component.get(cid, "") for cid in cids if full_by_component.get(cid, "")})
            if len(full_ids) < 2:
                continue
            internal = _internal_edges(_first_frame(edges_by_ablation, ARTIFACT_VARIANT_KEYS), cids)
            art = _artifact_signal(internal)
            if art <= 0:
                continue
            rows.append(
                {
                    "case_id": f"artifact_sensitive_{len(rows):03d}",
                    "case_type": "artifact_sensitive",
                    "no_artifact_group_id": gid,
                    "full_group_ids": ",".join(full_ids),
                    "n_components": len(cids),
                    "artifact_signal": art,
                    "component_ids": ",".join(sorted(cids)),
                    **bbox(cids),
                }
            )
        tables["artifact_sensitive"] = pd.DataFrame(rows).sort_values(["boundary_flag", "artifact_signal", "n_components"], ascending=[True, False, True]).head(20) if rows else pd.DataFrame()

    if layer2_edges_by_ablation:
        tables.update(layer2_case_tables(layer2_edges_by_ablation))
    return tables


def _internal_edges(edges: pd.DataFrame, cids: set[str]) -> pd.DataFrame:
    if edges is None or edges.empty:
        return pd.DataFrame()
    c1 = edges.get("gaussian_id_1", pd.Series(dtype=str)).astype(str)
    c2 = edges.get("gaussian_id_2", pd.Series(dtype=str)).astype(str)
    return edges[c1.isin(cids) & c2.isin(cids)].copy()


def _artifact_signal(edges: pd.DataFrame) -> float:
    if edges is None or edges.empty:
        return 0.0
    cols = [
        "deep_valley_penalty",
        "only_2sigma_penalty",
        "negative_bowl_penalty",
        "sidelobe_risk_penalty",
        "too_far_penalty",
        "large_mask_swallow_penalty",
    ]
    total = 0.0
    for col in cols:
        if col in edges:
            total += float(pd.to_numeric(edges[col], errors="coerce").fillna(0).sum())
    return total


def layer2_case_tables(edges_by_ablation: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    full = _first_frame(edges_by_ablation, FULL_VARIANT_KEYS)
    out: dict[str, pd.DataFrame] = {}
    if full.empty:
        return out
    full_by_id = full.set_index("parent_candidate_id") if "parent_candidate_id" in full else pd.DataFrame()
    accepted = {"high", "medium", "needs_host_check", "suspicious"}

    def compare(ablation_id: str, case_type: str, limit: int) -> pd.DataFrame:
        other = edges_by_ablation.get(ablation_id, pd.DataFrame())
        if other.empty or full_by_id.empty:
            return pd.DataFrame()
        rows = []
        for _, row in other.iterrows():
            pid = str(row.get("parent_candidate_id", ""))
            if pid not in full_by_id.index:
                continue
            full_row = full_by_id.loc[pid]
            fq = str(full_row.get("parent_candidate_quality", "rejected"))
            oq = str(row.get("parent_candidate_quality", "rejected"))
            if fq == oq:
                continue
            if case_type == "midpoint_host_support" and not (fq in accepted and oq not in {"high", "medium"}):
                continue
            if case_type == "lobe_peak_contradiction" and not (oq in accepted and fq not in accepted):
                continue
            rows.append(
                {
                    "case_id": f"{case_type}_{len(rows):03d}",
                    "case_type": case_type,
                    "parent_candidate_id": pid,
                    "full_quality": fq,
                    "ablation_quality": oq,
                    "local_group_id_1": row.get("local_group_id_1", ""),
                    "local_group_id_2": row.get("local_group_id_2", ""),
                    "best_host_score": row.get("best_host_score", np.nan),
                    "host_quality": row.get("host_quality", ""),
                    "lobe_peak_host_found": row.get("lobe_peak_host_found", False),
                    "lobe1_peak_host_score": row.get("lobe1_peak_host_score", np.nan),
                    "lobe2_peak_host_score": row.get("lobe2_peak_host_score", np.nan),
                    "rejection_reason_full": full_row.get("rejection_reason", ""),
                    "rejection_reason_ablation": row.get("rejection_reason", ""),
                }
            )
        return pd.DataFrame(rows).head(limit) if rows else pd.DataFrame()

    out["midpoint_host_support"] = compare("no_host_support", "midpoint_host_support", 20)
    lobe_peak_key = LOBE_PEAK_VARIANT_KEYS[0] if LOBE_PEAK_VARIANT_KEYS[0] in edges_by_ablation else LOBE_PEAK_VARIANT_KEYS[1]
    out["lobe_peak_contradiction"] = compare(lobe_peak_key, "lobe_peak_contradiction", 30)
    return out


def write_case_outputs(
    image: np.ndarray,
    components: pd.DataFrame,
    edges: pd.DataFrame,
    memberships: dict[str, pd.DataFrame],
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Write case CSVs and a compact contact sheet."""

    cases_dir = Path(output_dir) / "cases"
    (cases_dir / "individual_case_png").mkdir(parents=True, exist_ok=True)
    frames = []
    for name, table in tables.items():
        table = table.copy() if table is not None else pd.DataFrame()
        table.to_csv(cases_dir / f"{name}_candidate_cases.csv", index=False)
        if not table.empty:
            frames.append(table)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(Path(output_dir) / "case_rankings.csv", index=False)
    if not combined.empty:
        _write_individual_case_images(image, components, edges, memberships, combined.head(40), cases_dir / "individual_case_png")
    _contact_sheet(image, components, combined.head(24), cases_dir / "contact_sheet")
    return combined


def _write_individual_case_images(
    image: np.ndarray,
    components: pd.DataFrame,
    edges: pd.DataFrame,
    memberships: dict[str, pd.DataFrame],
    cases: pd.DataFrame,
    output_dir: Path,
) -> None:
    full = _first_layer1_frame(memberships, FULL_VARIANT_KEYS)
    no_anti = _first_layer1_frame(memberships, ANTI_CHAINING_VARIANT_KEYS)
    no_ridge = _first_layer1_frame(memberships, RIDGE_VARIANT_KEYS)
    for _, row in cases.iterrows():
        cids = set(parse_component_ids(row.get("component_ids", "")))
        if not cids:
            continue
        panels = [("full", full)]
        if row.get("case_type") == "anti_chaining" and not no_anti.empty:
            panels.append(("no anti-chaining", no_anti))
        elif row.get("case_type") == "ridge_sensitive" and not no_ridge.empty:
            panels.append(("no ridge", no_ridge))
        _plot_membership_case(image, components, edges, panels, cids, str(row.get("case_id")), output_dir / str(row.get("case_id", "case")))


def _plot_membership_case(
    image: np.ndarray,
    components: pd.DataFrame,
    edges: pd.DataFrame,
    panels: list[tuple[str, pd.DataFrame]],
    component_ids: set[str],
    title: str,
    output_stem: Path,
) -> None:
    subset = components[components["component_id"].astype(str).isin(component_ids)]
    if subset.empty:
        return
    x0 = max(0, int(np.floor(subset["x"].min()) - 40))
    x1 = min(image.shape[1], int(np.ceil(subset["x"].max()) + 40))
    y0 = max(0, int(np.floor(subset["y"].min()) - 40))
    y1 = min(image.shape[0], int(np.ceil(subset["y"].max()) + 40))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.2), squeeze=False)
    for ax, (label, membership) in zip(axes[0], panels):
        ax.imshow(_display_image(image[y0:y1, x0:x1]), origin="lower", cmap="gray")
        group_map = dict(zip(membership.get("component_id", pd.Series(dtype=str)).astype(str), membership.get("predicted_group_id", pd.Series(dtype=str)).astype(str))) if not membership.empty else {}
        group_ids = sorted({group_map.get(cid, cid) for cid in component_ids})
        colors = {gid: plt.cm.tab20(i % 20) for i, gid in enumerate(group_ids)}
        if edges is not None and not edges.empty:
            for _, edge in _internal_edges(edges, component_ids).iterrows():
                c1 = str(edge.get("gaussian_id_1", ""))
                c2 = str(edge.get("gaussian_id_2", ""))
                r1 = subset[subset["component_id"].astype(str) == c1]
                r2 = subset[subset["component_id"].astype(str) == c2]
                if r1.empty or r2.empty:
                    continue
                style = "--" if str(edge.get("edge_type")) == "weak" else "-"
                ax.plot([float(r1["x"].iloc[0]) - x0, float(r2["x"].iloc[0]) - x0], [float(r1["y"].iloc[0]) - y0, float(r2["y"].iloc[0]) - y0], style, color="white", lw=0.8, alpha=0.8)
        for _, comp in subset.iterrows():
            cid = str(comp["component_id"])
            ax.scatter(float(comp["x"]) - x0, float(comp["y"]) - y0, s=28, facecolor=colors[group_map.get(cid, cid)], edgecolor="black", lw=0.4)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=11)
    _save_figure(fig, output_stem)


def _contact_sheet(image: np.ndarray, components: pd.DataFrame, cases: pd.DataFrame, output_stem: Path) -> None:
    if cases.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No ablation-sensitive cases found", ha="center", va="center")
        _save_figure(fig, output_stem)
        return
    n = min(12, len(cases))
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, row) in zip(axes, cases.head(n).iterrows()):
        cids = set(parse_component_ids(row.get("component_ids", "")))
        sub = components[components["component_id"].astype(str).isin(cids)]
        if sub.empty:
            continue
        x0 = max(0, int(np.floor(sub["x"].min()) - 32))
        x1 = min(image.shape[1], int(np.ceil(sub["x"].max()) + 32))
        y0 = max(0, int(np.floor(sub["y"].min()) - 32))
        y1 = min(image.shape[0], int(np.ceil(sub["y"].max()) + 32))
        ax.imshow(_display_image(image[y0:y1, x0:x1]), origin="lower", cmap="gray")
        ax.scatter(sub["x"] - x0, sub["y"] - y0, s=16, facecolor="none", edgecolor="tab:cyan", lw=0.6)
        ax.set_title(f"{row.get('case_id')} {row.get('case_type')}", fontsize=7)
    _save_figure(fig, output_stem)


def write_ablation_report(
    output_dir: str | Path,
    layer1_table: pd.DataFrame,
    layer2_table: pd.DataFrame,
    validation: pd.DataFrame,
    manual_layer1: pd.DataFrame,
    manual_layer2: pd.DataFrame,
    case_rankings: pd.DataFrame,
) -> None:
    """Write a compact Markdown ablation report."""

    out = Path(output_dir)

    def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
        if df is None or df.empty:
            return "_No rows available._"
        work = df.head(max_rows).fillna("").copy()
        lines = ["| " + " | ".join(map(str, work.columns)) + " |", "| " + " | ".join("---" for _ in work.columns) + " |"]
        for _, row in work.iterrows():
            lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in work.columns) + " |")
        return "\n".join(lines)

    lines = [
        "# Ablation Study Report",
        "",
        "No full-result-as-truth accuracy claims are made. Precision, recall, F1, and false-association terms appear only if manual labels are supplied.",
        "",
        "## Layer-1 Structural Table",
        md_table(layer1_table),
        "",
        "## Layer-2 Structural Table",
        md_table(layer2_table),
        "",
        "## Fairness Checks",
        md_table(validation, max_rows=50),
        "",
        "## Manual Metrics",
        "Layer-1 manual metrics:",
        md_table(manual_layer1),
        "",
        "Layer-2 manual metrics:",
        md_table(manual_layer2),
        "",
        "## Cases",
        f"- Case ranking table: `{out / 'case_rankings.csv'}`",
        f"- Contact sheet: `{out / 'cases' / 'contact_sheet.pdf'}` and `{out / 'cases' / 'contact_sheet.png'}`",
        md_table(case_rankings.head(20) if case_rankings is not None else pd.DataFrame()),
    ]
    (out / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")
