"""Structural agreement and optional manual-label metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import labels_from_membership


def _labels_array(membership: pd.DataFrame, component_ids: list[str]) -> np.ndarray:
    mapping = labels_from_membership(membership, component_ids)
    return np.asarray([mapping[str(component_id)] for component_id in component_ids], dtype=object)


def pairwise_counts(labels_a: np.ndarray, labels_b: np.ndarray, pairs: list[tuple[int, int]] | None = None) -> dict[str, int]:
    """Count pairwise co-assignment agreement categories."""

    if pairs is None:
        pairs = list(combinations(range(len(labels_a)), 2))
    same_same = a_only = b_only = diff_diff = 0
    for i, j in pairs:
        same_a = labels_a[i] == labels_a[j]
        same_b = labels_b[i] == labels_b[j]
        if same_a and same_b:
            same_same += 1
        elif same_a and not same_b:
            a_only += 1
        elif same_b and not same_a:
            b_only += 1
        else:
            diff_diff += 1
    return {
        "pairwise_same_same": int(same_same),
        "pairwise_a_only": int(a_only),
        "pairwise_b_only": int(b_only),
        "pairwise_different_different": int(diff_diff),
    }


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Compute adjusted Rand index without requiring scikit-learn at runtime."""

    try:
        from sklearn.metrics import adjusted_rand_score

        return float(adjusted_rand_score(labels_a, labels_b))
    except Exception:
        pass

    n = len(labels_a)
    if n < 2:
        return 1.0
    contingency: defaultdict[tuple[Any, Any], int] = defaultdict(int)
    count_a: Counter[Any] = Counter()
    count_b: Counter[Any] = Counter()
    for a, b in zip(labels_a, labels_b):
        contingency[(a, b)] += 1
        count_a[a] += 1
        count_b[b] += 1

    def comb2(value: int) -> float:
        return value * (value - 1) / 2.0

    sum_nij = sum(comb2(v) for v in contingency.values())
    sum_ai = sum(comb2(v) for v in count_a.values())
    sum_bj = sum(comb2(v) for v in count_b.values())
    total = comb2(n)
    expected = sum_ai * sum_bj / total if total else 0.0
    max_index = 0.5 * (sum_ai + sum_bj)
    denom = max_index - expected
    return float((sum_nij - expected) / denom) if denom else 1.0


def variation_of_information(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Compute variation of information in nats."""

    n = len(labels_a)
    if n == 0:
        return 0.0
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    count_ab = Counter(zip(labels_a, labels_b))

    def entropy(counter: Counter[Any]) -> float:
        return float(-sum((c / n) * np.log(c / n) for c in counter.values() if c > 0))

    mutual = 0.0
    for (a, b), c in count_ab.items():
        p_ab = c / n
        p_a = count_a[a] / n
        p_b = count_b[b] / n
        mutual += p_ab * np.log(p_ab / (p_a * p_b))
    return float(entropy(count_a) + entropy(count_b) - 2.0 * mutual)


def method_agreement_table(
    memberships: dict[str, pd.DataFrame],
    component_ids: list[str],
    candidate_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute agreement between every pair of methods."""

    method_names = sorted(memberships)
    arrays = {name: _labels_array(memberships[name], component_ids) for name in method_names}
    index_by_id = {cid: idx for idx, cid in enumerate(component_ids)}
    pairs = None
    if candidate_pairs is not None and not candidate_pairs.empty:
        candidate_pairs = candidate_pairs.copy()
        c1 = candidate_pairs.get("component_id_1", pd.Series(dtype=str)).astype(str)
        c2 = candidate_pairs.get("component_id_2", pd.Series(dtype=str)).astype(str)
        pairs = [(index_by_id[a], index_by_id[b]) for a, b in zip(c1, c2) if a in index_by_id and b in index_by_id]
    records: list[dict[str, Any]] = []
    for i, method_a in enumerate(method_names):
        for method_b in method_names[i + 1 :]:
            labels_a = arrays[method_a]
            labels_b = arrays[method_b]
            counts = pairwise_counts(labels_a, labels_b, pairs=pairs)
            records.append(
                {
                    "method_a": method_a,
                    "method_b": method_b,
                    "adjusted_rand_index": adjusted_rand_index(labels_a, labels_b),
                    "variation_of_information": variation_of_information(labels_a, labels_b),
                    **counts,
                }
            )
    return pd.DataFrame(records)


def summarize_groups(
    groups: pd.DataFrame,
    method: str,
    parameter_id: str,
    n_components: int,
    runtime_seconds: float = np.nan,
    peak_memory_mb: float = np.nan,
) -> dict[str, Any]:
    """Return one row for baseline_summary.csv."""

    sizes = groups["n_components"].to_numpy(int) if not groups.empty and "n_components" in groups else np.asarray([], dtype=int)
    multi = sizes[sizes >= 2]
    return {
        "method": method,
        "parameter_id": parameter_id,
        "n_components": int(n_components),
        "n_singletons": int(np.count_nonzero(sizes == 1)),
        "n_multi_groups": int(np.count_nonzero(sizes >= 2)),
        "n_multi_component_groups": int(np.count_nonzero(sizes >= 2)),
        "fraction_components_in_multi_groups": float(multi.sum() / max(n_components, 1)) if sizes.size else 0.0,
        "median_group_size": float(np.median(sizes)) if sizes.size else 0.0,
        "p90_group_size": float(np.percentile(sizes, 90)) if sizes.size else 0.0,
        "max_group_size": int(sizes.max()) if sizes.size else 0,
        "n_groups_gt_10": int(np.count_nonzero(sizes > 10)),
        "n_groups_gt_20": int(np.count_nonzero(sizes > 20)),
        "n_groups_gt_50": int(np.count_nonzero(sizes > 50)),
        "runtime_seconds": float(runtime_seconds) if np.isfinite(runtime_seconds) else np.nan,
        "peak_memory_mb": float(peak_memory_mb) if np.isfinite(peak_memory_mb) else np.nan,
    }


def split_merge_against_reference(
    reference_membership: pd.DataFrame,
    other_membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute split and merge comparison with a reference method."""

    if reference_membership.empty or other_membership.empty:
        return pd.DataFrame(), pd.DataFrame()
    ref = reference_membership.rename(columns={"predicted_group_id": "reference_group_id"})
    oth = other_membership.rename(columns={"predicted_group_id": "other_group_id"})
    joined = ref[["component_id", "reference_group_id"]].merge(oth[["component_id", "other_group_id"]], on="component_id")
    split_records = []
    for ref_gid, rows in joined.groupby("reference_group_id"):
        split_records.append(
            {
                "reference_group_id": str(ref_gid),
                "n_components": int(len(rows)),
                "n_other_groups": int(rows["other_group_id"].nunique()),
                "other_group_ids": ",".join(sorted(rows["other_group_id"].astype(str).unique())),
            }
        )
    merge_records = []
    for other_gid, rows in joined.groupby("other_group_id"):
        merge_records.append(
            {
                "other_group_id": str(other_gid),
                "n_components": int(len(rows)),
                "n_reference_groups": int(rows["reference_group_id"].nunique()),
                "reference_group_ids": ",".join(sorted(rows["reference_group_id"].astype(str).unique())),
            }
        )
    return pd.DataFrame(split_records), pd.DataFrame(merge_records)


def pairwise_prf(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, float]:
    """Compute pairwise precision, recall, and F1."""

    tp = fp = fn = 0
    for i, j in combinations(range(len(true_labels)), 2):
        true_same = true_labels[i] == true_labels[j]
        pred_same = pred_labels[i] == pred_labels[j]
        if true_same and pred_same:
            tp += 1
        elif not true_same and pred_same:
            fp += 1
        elif true_same and not pred_same:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"pairwise_precision": precision, "pairwise_recall": recall, "pairwise_f1": f1}


def bcubed_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, float]:
    """Compute B-cubed precision, recall, and F1."""

    n = len(true_labels)
    if n == 0:
        return {"bcubed_precision": 0.0, "bcubed_recall": 0.0, "bcubed_f1": 0.0}
    precision_values = []
    recall_values = []
    for idx in range(n):
        pred_cluster = pred_labels == pred_labels[idx]
        true_cluster = true_labels == true_labels[idx]
        intersection = np.count_nonzero(pred_cluster & true_cluster)
        precision_values.append(intersection / max(np.count_nonzero(pred_cluster), 1))
        recall_values.append(intersection / max(np.count_nonzero(true_cluster), 1))
    precision = float(np.mean(precision_values))
    recall = float(np.mean(recall_values))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"bcubed_precision": precision, "bcubed_recall": recall, "bcubed_f1": f1}


def overmerge_split_rates(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, float]:
    """Compute over-merge and fragmentation rates."""

    pred_groups = defaultdict(set)
    true_groups = defaultdict(set)
    for true, pred in zip(true_labels, pred_labels):
        pred_groups[pred].add(true)
        true_groups[true].add(pred)
    overmerge = sum(len(values) >= 2 for values in pred_groups.values()) / max(len(pred_groups), 1)
    split = sum(len(values) >= 2 for values in true_groups.values()) / max(len(true_groups), 1)
    exact = sum(len(values) == 1 for values in true_groups.values()) / max(len(true_groups), 1)
    return {"exact_group_recovery": float(exact), "overmerge_rate": float(overmerge), "split_rate": float(split)}


def manual_label_metrics(
    manual_labels_path: str | Path,
    memberships: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute gold-standard metrics when manual_labels.csv is present."""

    path = Path(manual_labels_path)
    if not path.exists():
        return pd.DataFrame()
    labels = pd.read_csv(path)
    required = {"component_id", "true_local_group_id", "label_quality"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"manual_labels.csv missing columns: {sorted(missing)}")
    if "artifact_flag" not in labels.columns:
        labels["artifact_flag"] = "real"
    records: list[dict[str, Any]] = []
    confident_real = (labels["label_quality"].astype(str) != "uncertain") & (
        labels["artifact_flag"].astype(str) != "likely_artifact"
    )
    labelled = labels[labels["label_quality"].astype(str) != "uncertain"].copy()
    modes = {
        "exclude_uncertain_artifact": labels[confident_real].copy(),
        "include_artifact_independent": labelled,
    }
    for mode, frame in modes.items():
        if frame.empty:
            continue
        frame["component_id"] = frame["component_id"].astype(str)
        if mode == "include_artifact_independent":
            artifact = frame["artifact_flag"].astype(str) == "likely_artifact"
            frame.loc[artifact, "true_local_group_id"] = "artifact_" + frame.loc[artifact, "component_id"].astype(str)
        component_ids = frame["component_id"].tolist()
        true = frame["true_local_group_id"].astype(str).to_numpy()
        for method_key, membership in memberships.items():
            pred_map = labels_from_membership(membership, component_ids)
            pred = np.asarray([pred_map[cid] for cid in component_ids], dtype=object)
            method, _, parameter_id = method_key.partition(":")
            records.append(
                {
                    "method": method,
                    "parameter_id": parameter_id or "default",
                    "label_mode": mode,
                    **pairwise_prf(true, pred),
                    **bcubed_metrics(true, pred),
                    **overmerge_split_rates(true, pred),
                }
            )
    return pd.DataFrame(records)
