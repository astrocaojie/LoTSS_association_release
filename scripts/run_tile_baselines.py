#!/usr/bin/env python
"""Run tile-level Gaussian local-association baseline comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml
import numpy as np

from lotss_association.baseline_demo.baseline_contour import run_contour_baseline
from lotss_association.baseline_demo.baseline_distance import run_distance_baseline
from lotss_association.baseline_demo.baseline_pybdsf import run_pybdsf_island_baseline
from lotss_association.baseline_demo.baseline_unconstrained import run_unconstrained_graph_baseline
from lotss_association.baseline_demo.common import component_id_series
from lotss_association.baseline_demo.comparison_metrics import (
    manual_label_metrics,
    method_agreement_table,
    split_merge_against_reference,
    summarize_groups,
)
from lotss_association.baseline_demo.data_loading import inspect_tile_inputs, load_tile_demo
from lotss_association.baseline_demo.neighbour_search import find_candidate_pairs
from lotss_association.baseline_demo.plotting import make_all_plots
from lotss_association.baseline_demo.reporting import (
    beam_distance_sample,
    build_case_exports,
    catalogue_preflight,
    component_sample_check,
    edge_table_hash,
    expected_output_check,
    fits_preflight_inventory,
    mask_stats,
    plot_gaussian_overlay,
    plot_mask_overlay,
    structural_tables,
    write_json,
    write_preflight_report,
    write_tile_report,
)
from lotss_association.baseline_demo.run_full_layer1 import run_full_layer1_method
from lotss_association.utils import ensure_dir, load_yaml, setup_logging, write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/tile_demo.yaml"))
    parser.add_argument("--manual-labels", default=None, help="Optional manual_labels.csv path for gold-standard metrics.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def peak_memory_mb() -> float:
    """Return process peak resident set size in MB on Linux/macOS."""

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage / (1024 * 1024))
    return float(usage / 1024)


def _method_key(groups: pd.DataFrame) -> str:
    if groups.empty:
        return "unknown:default"
    return f"{groups['method'].iloc[0]}:{groups['parameter_id'].iloc[0]}"


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = ensure_dir(config.get("output_dir", PROJECT_ROOT / "baseline_demo" / "outputs"))
    logs_dir = ensure_dir(output_dir / "logs")
    logger = setup_logging(debug=args.debug, log_path=logs_dir / "run_tile_baselines.log")
    t_start = perf_counter()

    logger.info("Running tile baselines with config %s", args.config)
    logger.info("git_commit=%s", git_commit())
    logger.info("random_seed=%s", config.get("random_seed", 42))
    if args.manual_labels:
        config["manual_labels"] = args.manual_labels
    inventory = inspect_tile_inputs(config)
    pd.DataFrame([inventory]).to_csv(output_dir / "input_inventory.csv", index=False)
    logger.info("Input inventory: %s", inventory)

    data = load_tile_demo(config)
    config = data.config
    _write_yaml(output_dir / "full_layer1_config_resolved.yaml", config)
    preflight_dir = ensure_dir(output_dir / "preflight")
    preflight = fits_preflight_inventory(config)
    catalogue = catalogue_preflight(config, data.components)
    write_json(preflight_dir / "input_inventory.json", {**inventory, **preflight, "catalogue": catalogue})
    write_json(preflight_dir / "data_quality_stats.json", data.quality_stats)
    plot_gaussian_overlay(data.image, data.components, preflight_dir / "gaussian_overlay.png", seed=int(config.get("random_seed", 42)))
    image_shape = data.image.shape
    components = data.components
    n_components = int(len(components))
    if n_components == 0:
        raise SystemExit("No Gaussian components retained after cleaning; cannot run baselines.")

    logger.info("Generating fair candidate neighbours")
    neighbour = find_candidate_pairs(components, config)
    neighbour.pairs.to_csv(output_dir / "candidate_pairs.csv", index=False)
    pd.DataFrame([neighbour.stats]).to_csv(output_dir / "neighbour_search_stats.csv", index=False)
    beam_sample = beam_distance_sample(components, neighbour.pairs, config, n=20)
    beam_sample.to_csv(preflight_dir / "beam_distance_sample.csv", index=False)
    logger.info("Candidate pairs: %d", len(neighbour.pairs))

    groups_by_key: dict[str, pd.DataFrame] = {}
    memberships_by_key: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    case_tables: list[pd.DataFrame] = []

    logger.info("Running full constrained Layer-1 method")
    full_groups, full_membership, full_edges, full_stats, _full_result = run_full_layer1_method(
        data.cutout,
        data.segmentation,
        components,
        config,
        image_shape=image_shape,
    )
    key = _method_key(full_groups)
    groups_by_key[key] = full_groups
    memberships_by_key[key] = full_membership
    summary_rows.append(summarize_groups(full_groups, "full_layer1", "current_config", n_components, full_stats.get("runtime_seconds")))

    logger.info("Running PyBDSF island baseline")
    pyb_groups, pyb_membership, pyb_stats = run_pybdsf_island_baseline(components, config, image_shape=image_shape)
    if pyb_stats.get("available", False):
        key = _method_key(pyb_groups)
        groups_by_key[key] = pyb_groups
        memberships_by_key[key] = pyb_membership
        summary_rows.append(summarize_groups(pyb_groups, "pybdsf_island", "native", n_components, pyb_stats.get("runtime_seconds", float("nan"))))
    else:
        logger.warning("PyBDSF island baseline unavailable: %s", pyb_stats.get("reason"))

    logger.info("Running distance-only threshold scan")
    distance_groups, distance_memberships, distance_stats = run_distance_baseline(
        components,
        neighbour.pairs,
        config,
        image_shape=image_shape,
        target_multi_groups=int((full_groups["n_components"] >= 2).sum()) if not full_groups.empty else None,
    )
    distance_stats["n_components"] = n_components
    distance_stats.to_csv(output_dir / "distance_baseline_stats.csv", index=False)
    for param, groups in distance_groups.items():
        key = f"distance_only:{param}"
        groups_by_key[key] = groups
        memberships_by_key[key] = distance_memberships[param]
        stat = distance_stats.loc[distance_stats["parameter_id"].astype(str) == str(param)]
        runtime = float(stat["runtime_seconds"].iloc[0]) if not stat.empty else float("nan")
        summary_rows.append(summarize_groups(groups, "distance_only", param, n_components, runtime))

    logger.info("Running 3 sigma contour-connectivity baseline")
    contour_groups, contour_membership, contour_stats, _mask, _labels = run_contour_baseline(
        components,
        data.segmentation.snr_map,
        config,
        image_header=data.cutout.header,
        image_shape=image_shape,
    )
    assigned_labels = pd.to_numeric(components.get("label_at_3sigma", pd.Series(np.zeros(len(components)))), errors="coerce").fillna(0).to_numpy(int)
    mask_check = mask_stats(_mask, _labels, assigned_labels)
    pd.DataFrame([mask_check]).to_csv(preflight_dir / "mask_3sigma_stats.csv", index=False)
    plot_mask_overlay(data.image, _mask, components, preflight_dir / "mask_3sigma_overlay.png")
    key = _method_key(contour_groups)
    groups_by_key[key] = contour_groups
    memberships_by_key[key] = contour_membership
    summary_rows.append(summarize_groups(contour_groups, "contour_3sigma", contour_stats["parameter_id"], n_components, contour_stats.get("runtime_seconds")))

    logger.info("Running unconstrained graph baseline")
    uncon_groups, uncon_membership, uncon_edges, weak_cases, uncon_stats = run_unconstrained_graph_baseline(
        components,
        full_edges,
        full_membership,
        config,
        image_shape=image_shape,
    )
    key = _method_key(uncon_groups)
    groups_by_key[key] = uncon_groups
    memberships_by_key[key] = uncon_membership
    summary_rows.append(summarize_groups(uncon_groups, "unconstrained_graph", "strong_plus_accepted_weak", n_components, uncon_stats.get("runtime_seconds")))

    component_ids = component_id_series(components).astype(str).tolist()
    agreement = method_agreement_table(memberships_by_key, component_ids, candidate_pairs=neighbour.pairs)
    agreement.to_csv(output_dir / "method_agreement.csv", index=False)

    split_rows: list[pd.DataFrame] = []
    merge_rows: list[pd.DataFrame] = []
    for method_key, membership in memberships_by_key.items():
        if method_key == "full_layer1:current_config":
            continue
        split, merge = split_merge_against_reference(full_membership, membership)
        if not split.empty:
            split.insert(0, "method_key", method_key)
            split_rows.append(split)
        if not merge.empty:
            merge.insert(0, "method_key", method_key)
            merge_rows.append(merge)
    if split_rows:
        pd.concat(split_rows, ignore_index=True).to_csv(output_dir / "split_comparison_with_full_reference.csv", index=False)
    if merge_rows:
        pd.concat(merge_rows, ignore_index=True).to_csv(output_dir / "merge_comparison_with_full_reference.csv", index=False)

    if not weak_cases.empty:
        case_tables.append(weak_cases)
    case_rankings = pd.concat(case_tables, ignore_index=True) if case_tables else pd.DataFrame()
    if not case_rankings.empty:
        case_rankings = case_rankings.sort_values("ranking_score", ascending=False)
    case_rankings.to_csv(output_dir / "case_rankings.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary["peak_memory_mb"] = peak_memory_mb()
    summary.to_csv(output_dir / "baseline_summary.csv", index=False)

    component_check = component_sample_check(memberships_by_key)
    component_check.to_csv(preflight_dir / "component_sample_consistency.csv", index=False)
    component_sets_identical = bool(component_check["component_id_set_identical"].all()) if not component_check.empty else False
    full_hash = edge_table_hash(full_edges)
    uncon_hash = edge_table_hash(full_edges)
    edge_validation = {
        "edge_input_hash_full": full_hash,
        "edge_input_hash_unconstrained": uncon_hash,
        "edge_tables_identical_before_clustering": bool(full_hash == uncon_hash),
        "component_sets_identical": component_sets_identical,
    }
    write_json(preflight_dir / "edge_input_validation.json", edge_validation)
    write_preflight_report(output_dir, config, preflight, catalogue, data.quality_stats, edge_validation, mask_check)

    structural = structural_tables(components, memberships_by_key, groups_by_key, full_edges, distance_stats, output_dir)
    case_rankings = build_case_exports(data.image, components, full_edges, memberships_by_key, output_dir)
    distance_stats_for_plots = distance_stats
    distance_structural_path = output_dir / "distance_threshold_structural_stats.csv"
    if distance_structural_path.exists():
        distance_stats_for_plots = pd.read_csv(distance_structural_path)

    manual_path = config.get("manual_labels", output_dir / "manual_labels.csv")
    gold = manual_label_metrics(manual_path, memberships_by_key)
    if not gold.empty:
        gold.to_csv(output_dir / "gold_standard_metrics.csv", index=False)
    else:
        logger.info("No manual labels found; gold-standard accuracy/precision/recall metrics were not computed.")

    make_all_plots(
        data.image,
        components,
        uncon_edges if not uncon_edges.empty else full_edges,
        groups_by_key,
        memberships_by_key,
        distance_stats_for_plots,
        agreement,
        case_rankings,
        output_dir,
    )

    write_tile_report(
        output_dir,
        config,
        preflight,
        catalogue,
        data.quality_stats,
        summary,
        agreement,
        edge_validation,
        structural,
        case_rankings,
    )
    expected = expected_output_check(output_dir)
    expected.to_csv(preflight_dir / "expected_outputs_check.csv", index=False)
    missing = expected[~expected["exists"].astype(bool)]
    if not missing.empty:
        raise RuntimeError(f"Missing expected outputs: {missing['path'].tolist()}")

    run_log = {
        "runtime_seconds": perf_counter() - t_start,
        "git_commit": git_commit(),
        "n_components": n_components,
        "n_candidate_pairs": int(len(neighbour.pairs)),
        "warnings": "manual truth absent; no accuracy claims" if gold.empty else "",
    }
    pd.DataFrame([run_log]).to_csv(output_dir / "run_summary.csv", index=False)
    logger.info("Completed tile baseline comparison in %.2f s", run_log["runtime_seconds"])
    logger.info("Outputs written under %s", output_dir)


if __name__ == "__main__":
    main()
