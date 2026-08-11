#!/usr/bin/env python
"""Run configurable ablation studies on the tile baseline inputs."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx
import pandas as pd
import yaml

from lofar_det_vsex.ablation.analysis import (
    case_tables,
    config_hash,
    layer1_delta_table,
    layer2_delta_table,
    layer2_manual_metrics,
    layer2_summary_row,
    summarize_layer1_ablation,
    table_hash,
    write_ablation_report,
    write_case_outputs,
    write_layer1_manual_metrics,
)
from lofar_det_vsex.ablation.inventory import write_module_inventory
from lofar_det_vsex.ablation_config import ABLATION_DEFAULTS
from lofar_det_vsex.association import run_component_association
from lofar_det_vsex.baseline_demo.common import group_summary_from_membership, membership_from_clusters, write_method_outputs
from lofar_det_vsex.baseline_demo.data_loading import load_tile_demo
from lofar_det_vsex.baseline_demo.neighbour_search import find_candidate_pairs
from lofar_det_vsex.baseline_demo.reporting import edge_table_hash, write_json
from lofar_det_vsex.host_query import HostQueryClient
from lofar_det_vsex.parent_links_endpoint_guarded import (
    SOURCE_MORPH_TABLE_COLUMNS,
    PARENT_CANDIDATE_COLUMNS,
    PARENT_DIAGNOSTIC_COLUMNS,
    PARENT_EDGE_DEBUG_COLUMNS,
    PARENT_HOST_CANDIDATE_COLUMNS,
    run_parent_links_endpoint_guarded,
)
from lofar_det_vsex.utils import ensure_dir, load_yaml, setup_logging, write_dataframe


FULL_VARIANT = "full_method"


LAYER1_VARIANTS: dict[str, dict[str, bool]] = {
    FULL_VARIANT: {},
    "no_ridge_continuity": {"use_ridge_continuity": False},
    "no_weak_edge_anti_chaining": {"use_weak_edge_anti_chaining": False},
    "no_artifact_penalties": {"use_artifact_penalties_layer1": False, "use_artifact_penalties_layer2": False},
    "no_pa_alignment": {"use_pa_alignment": False},
    "no_ellipse_overlap": {"use_ellipse_overlap": False},
    "no_multithreshold_contour_connectivity": {"use_multithreshold_contour": False},
}

LAYER2_VARIANTS: dict[str, dict[str, bool]] = {
    FULL_VARIANT: {},
    "no_artifact_penalties": {"use_artifact_penalties_layer1": False, "use_artifact_penalties_layer2": False},
    "no_host_support": {"use_midpoint_host_support": False, "use_lobe_peak_host_contradiction": True},
    "no_lobe_peak_host_contradiction": {"use_midpoint_host_support": True, "use_lobe_peak_host_contradiction": False},
    "no_stage2_relative_scale_constraints": {"use_stage2_relative_scale_constraints": False},
    "no_stage2_endpoint_filtering": {"use_stage2_endpoint_filtering": False},
}

LEGACY_VARIANT_ALIASES: dict[str, str] = {
    "full": "full_method",
    "no_ridge": "no_ridge_continuity",
    "no_anti_chaining": "no_weak_edge_anti_chaining",
    "no_artifact_layer1": "no_artifact_penalties",
    "no_artifact_layer2": "no_artifact_penalties",
    "no_lobe_peak_contradiction": "no_lobe_peak_host_contradiction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/tile_ablation.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--manual-labels", default=None, help="Optional Layer-1 manual labels in the existing manual_labels.csv format.")
    parser.add_argument("--manual-layer2-labels", default=None, help="Optional Layer-2 parent-pair labels.")
    parser.add_argument("--run-layer2", action="store_true", help="Also run endpoint-guarded Layer-2 ablations using the full Layer-1 output.")
    parser.add_argument("--offline-host-cache-only", action="store_true")
    parser.add_argument("--skip-host-query", action="store_true")
    parser.add_argument("--host-cache", default=None)
    parser.add_argument("--max-host-queries", type=int, default=1000)
    parser.add_argument("--variants", nargs="+", default=None, help="Optional ablation variant names/IDs to run.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage / (1024 * 1024))
    return float(usage / 1024)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=True)


def variant_config(base: dict[str, Any], switches: dict[str, bool], output_dir: Path | None = None) -> dict[str, Any]:
    cfg = deepcopy(base)
    ablation = dict(ABLATION_DEFAULTS)
    ablation.update(cfg.get("ablation", {}) or {})
    ablation.update(switches)
    cfg["ablation"] = ablation
    if output_dir is not None:
        cfg["output_dir"] = str(output_dir)
    return cfg


def _normalise_variant_name(name: str) -> str:
    return LEGACY_VARIANT_ALIASES.get(str(name), str(name))


def selected_variant_maps(names: list[str] | None, run_layer2: bool) -> tuple[dict[str, dict[str, bool]], dict[str, dict[str, bool]]]:
    """Return Layer-1/Layer-2 variant maps filtered by requested names."""

    if not names:
        return dict(LAYER1_VARIANTS), dict(LAYER2_VARIANTS)
    requested = {_normalise_variant_name(name) for name in names}
    unknown = sorted(name for name in requested if name not in LAYER1_VARIANTS and name not in LAYER2_VARIANTS)
    if unknown:
        raise SystemExit(f"Unknown ablation variants: {unknown}")
    layer1 = {name: switches for name, switches in LAYER1_VARIANTS.items() if name in requested}
    layer2 = {name: switches for name, switches in LAYER2_VARIANTS.items() if name in requested}
    if run_layer2 and layer2 and FULL_VARIANT not in layer1:
        layer1 = {FULL_VARIANT: LAYER1_VARIANTS[FULL_VARIANT], **layer1}
    if run_layer2 and layer2 and FULL_VARIANT not in layer2:
        layer2 = {FULL_VARIANT: LAYER2_VARIANTS[FULL_VARIANT], **layer2}
    return layer1, layer2


def _run_layer1_variant(
    ablation_id: str,
    data: Any,
    base_config: dict[str, Any],
    switches: dict[str, bool],
    output_dir: Path,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    cfg = variant_config(base_config, switches, output_dir)
    t0 = perf_counter()
    result = run_component_association(data.cutout, data.segmentation, data.components, cfg)
    runtime = perf_counter() - t0
    method = "layer1_ablation"
    membership = membership_from_clusters(result.clusters, data.components, method, ablation_id, f"layer1_{ablation_id}")
    groups = group_summary_from_membership(
        membership,
        data.components,
        method,
        ablation_id,
        image_shape=image_shape,
        boundary_padding_pixels=int((base_config.get("demo_region", {}) or {}).get("padding_pixels", 0) or 0),
    )
    write_method_outputs(output_dir / "layer1" / ablation_id, f"groups_{ablation_id}", groups, membership)
    write_dataframe(result.edges, output_dir / "layer1" / ablation_id / f"edges_{ablation_id}.parquet")
    result.edges.to_csv(output_dir / "layer1" / ablation_id / f"edges_{ablation_id}.csv", index=False)
    result.groups.to_csv(output_dir / "layer1" / ablation_id / f"association_groups_{ablation_id}.csv", index=False)
    return {
        "ablation_id": ablation_id,
        "config": cfg,
        "result": result,
        "groups": groups,
        "membership": membership,
        "edges": result.edges.copy(),
        "runtime_seconds": runtime,
    }


def _strong_edge_feature_hash(edges: pd.DataFrame) -> str:
    if edges.empty:
        return edge_table_hash(edges)
    keep = [col for col in edges.columns if col not in {"association_decision", "rejection_reason", "debug_info"}]
    return table_hash(edges.loc[:, keep])


def _validation_rows(
    base_config: dict[str, Any],
    components: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    layer1_results: dict[str, dict[str, Any]],
    layer2_edges: dict[str, pd.DataFrame],
    host_cache_hash: str,
) -> pd.DataFrame:
    rows = []
    comp_hash = table_hash(components[["component_id", "component_index"]].copy() if {"component_id", "component_index"}.issubset(components.columns) else components)
    pair_hash = table_hash(candidate_pairs)
    full_edge_hash = edge_table_hash(layer1_results[FULL_VARIANT]["edges"])
    full_feature_hash = _strong_edge_feature_hash(layer1_results[FULL_VARIANT]["edges"])
    for ablation_id, payload in sorted(layer1_results.items()):
        rows.append(
            {
                "stage": "layer1",
                "ablation_id": ablation_id,
                "component_set_hash": comp_hash,
                "candidate_pair_hash": pair_hash,
                "host_cache_hash": host_cache_hash,
                "edge_input_hash": edge_table_hash(payload["edges"]),
                "edge_feature_hash": _strong_edge_feature_hash(payload["edges"]),
                "edge_table_identical_to_full": bool(edge_table_hash(payload["edges"]) == full_edge_hash),
                "edge_features_identical_to_full": bool(_strong_edge_feature_hash(payload["edges"]) == full_feature_hash),
                "configuration_hash": config_hash(payload["config"]),
                "git_commit": git_commit(),
            }
        )
    full_layer2_hash = table_hash(layer2_edges.get(FULL_VARIANT, pd.DataFrame()), drop_columns={"debug_info"}) if layer2_edges else table_hash(pd.DataFrame())
    for ablation_id, edges in sorted(layer2_edges.items()):
        rows.append(
            {
                "stage": "layer2",
                "ablation_id": ablation_id,
                "component_set_hash": comp_hash,
                "candidate_pair_hash": "",
                "host_cache_hash": host_cache_hash,
                "edge_input_hash": table_hash(edges, drop_columns={"debug_info"}),
                "edge_feature_hash": table_hash(edges[[c for c in edges.columns if c not in {"parent_candidate_quality", "rejection_reason", "host_evidence", "parent_score_final", "debug_info"}]], drop_columns=set()) if not edges.empty else table_hash(edges),
                "edge_table_identical_to_full": bool(table_hash(edges, drop_columns={"debug_info"}) == full_layer2_hash),
                "edge_features_identical_to_full": "",
                "configuration_hash": config_hash(variant_config(base_config, LAYER2_VARIANTS.get(ablation_id, {}))),
                "git_commit": git_commit(),
            }
        )
    return pd.DataFrame(rows)


def _host_cache_hash(host_client: HostQueryClient) -> str:
    try:
        cache = getattr(host_client, "_cache", pd.DataFrame())
        return table_hash(cache)
    except Exception:
        return table_hash(pd.DataFrame())


def _run_layer2_variants(
    data: Any,
    base_config: dict[str, Any],
    full_layer1_result: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    layer2_variants: dict[str, dict[str, bool]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], str]:
    cache_dir = ensure_dir(output_dir / "cache")
    host_client = HostQueryClient(
        cache_dir=cache_dir,
        cache_path=args.host_cache,
        offline_cache_only=bool(args.offline_host_cache_only),
        skip_query=bool(args.skip_host_query),
    )
    groups = full_layer1_result["result"].groups.copy()
    components = full_layer1_result["result"].components.copy()
    rows = []
    edges_by_id: dict[str, pd.DataFrame] = {}
    all_candidates = []
    all_edges = []
    all_hosts = []
    all_logs = []
    all_diags = []
    all_morph = []
    for ablation_id, switches in layer2_variants.items():
        if ablation_id == "no_artifact_penalties" and "no_artifact_penalties" not in LAYER1_VARIANTS:
            continue
        cfg = variant_config(base_config, switches, output_dir / "layer2" / ablation_id)
        t0 = perf_counter()
        result = run_parent_links_endpoint_guarded(
            str(data.cutout.cutout_id),
            groups,
            components,
            data.segmentation,
            cfg,
            host_client,
            {"count": 0},
            max_host_queries=int(args.max_host_queries) if args.max_host_queries is not None else None,
        )
        runtime = perf_counter() - t0
        subdir = ensure_dir(output_dir / "layer2" / ablation_id)
        result.candidates.to_csv(subdir / f"parent_candidates_{ablation_id}.csv", index=False)
        write_dataframe(result.edges_debug, subdir / f"parent_edges_debug_{ablation_id}.parquet")
        result.edges_debug.to_csv(subdir / f"parent_edges_debug_{ablation_id}.csv", index=False)
        result.host_candidates.to_csv(subdir / f"host_candidates_{ablation_id}.csv", index=False)
        result.host_query_log.to_csv(subdir / f"host_query_log_{ablation_id}.csv", index=False)
        result.diagnostics.to_csv(subdir / f"diagnostics_{ablation_id}.csv", index=False)
        result.source_morph_table.to_csv(subdir / f"source_morph_table_{ablation_id}.csv", index=False)
        edges_by_id[ablation_id] = result.edges_debug.copy()
        rows.append(layer2_summary_row(ablation_id, result.edges_debug, result.candidates, edges_by_id.get(FULL_VARIANT), runtime))
        all_candidates.append(result.candidates.assign(ablation_id=ablation_id))
        all_edges.append(result.edges_debug.assign(ablation_id=ablation_id))
        all_hosts.append(result.host_candidates.assign(ablation_id=ablation_id))
        all_logs.append(result.host_query_log.assign(ablation_id=ablation_id))
        all_diags.append(result.diagnostics.assign(ablation_id=ablation_id))
        all_morph.append(result.source_morph_table.assign(ablation_id=ablation_id))
    host_client.save_cache()
    host_cache_hash = _host_cache_hash(host_client)
    cache = getattr(host_client, "_cache", pd.DataFrame())
    if cache is not None and not cache.empty:
        write_dataframe(cache, cache_dir / "midpoint_host_matches.parquet")
        write_dataframe(cache, cache_dir / "lobe_peak_host_matches.parquet")
    if all_candidates:
        pd.concat(all_candidates, ignore_index=True).to_csv(output_dir / "layer2_parent_candidates_all.csv", index=False)
    if all_edges:
        pd.concat(all_edges, ignore_index=True).to_csv(output_dir / "layer2_parent_edges_debug_all.csv", index=False)
    if all_hosts:
        pd.concat(all_hosts, ignore_index=True).to_csv(output_dir / "layer2_host_candidates_all.csv", index=False)
    if all_logs:
        pd.concat(all_logs, ignore_index=True).to_csv(output_dir / "layer2_host_query_log_all.csv", index=False)
    if all_diags:
        pd.concat(all_diags, ignore_index=True).to_csv(output_dir / "layer2_diagnostics_all.csv", index=False)
    if all_morph:
        pd.concat(all_morph, ignore_index=True).to_csv(output_dir / "layer2_source_morph_table_all.csv", index=False)
    table = pd.DataFrame(rows)
    return table, layer2_delta_table(table), edges_by_id, host_cache_hash


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = ensure_dir(args.output_dir or config.get("output_dir", PROJECT_ROOT / "outputs" / "ablation"))
    config["output_dir"] = str(output_dir)
    logger = setup_logging(debug=args.debug, log_path=ensure_dir(output_dir / "logs") / "run_tile_ablation.log")
    t_start = perf_counter()
    logger.info("Running tile ablation with config %s", args.config)

    layer1_variants, layer2_variants = selected_variant_maps(args.variants, bool(args.run_layer2))
    write_module_inventory(output_dir)
    data = load_tile_demo(config)
    _write_yaml(output_dir / "tile_ablation_config_resolved.yaml", data.config)
    candidate_pairs = find_candidate_pairs(data.components, data.config).pairs
    candidate_pairs.to_csv(output_dir / "candidate_pairs.csv", index=False)
    write_json(
        output_dir / "run_context.json",
        {
            "git_commit": git_commit(),
            "n_components": int(len(data.components)),
            "n_candidate_pairs": int(len(candidate_pairs)),
            "manual_labels": args.manual_labels,
            "manual_layer2_labels": args.manual_layer2_labels,
            "run_layer2": bool(args.run_layer2),
            "requested_variants": args.variants,
            "layer1_variants": sorted(layer1_variants),
            "layer2_variants": sorted(layer2_variants),
        },
    )

    layer1_results: dict[str, dict[str, Any]] = {}
    for ablation_id, switches in layer1_variants.items():
        logger.info("Running Layer-1 ablation %s", ablation_id)
        layer1_results[ablation_id] = _run_layer1_variant(ablation_id, data, data.config, switches, output_dir, image_shape=data.image.shape)

    full = layer1_results[FULL_VARIANT]
    layer1_rows = []
    for ablation_id, payload in layer1_results.items():
        row = summarize_layer1_ablation(
            ablation_id,
            payload["groups"],
            payload["membership"],
            full["membership"],
            full["edges"],
            payload["edges"],
            len(data.components),
            payload["runtime_seconds"],
        )
        row["ablation_id"] = ablation_id
        row["peak_memory_mb"] = peak_memory_mb()
        row["component_set_hash"] = table_hash(data.components[["component_id", "component_index"]])
        row["candidate_pair_hash"] = table_hash(candidate_pairs)
        row["configuration_hash"] = config_hash(payload["config"])
        row["git_commit"] = git_commit()
        layer1_rows.append(row)
    layer1_table = pd.DataFrame(layer1_rows)
    layer1_table.to_csv(output_dir / "layer1_ablation_table.csv", index=False)
    layer1_delta = layer1_delta_table(layer1_table)
    layer1_delta.to_csv(output_dir / "layer1_ablation_delta.csv", index=False)

    memberships = {f"layer1_ablation:{ablation_id}": payload["membership"] for ablation_id, payload in layer1_results.items()}
    edges_by_id = {ablation_id: payload["edges"] for ablation_id, payload in layer1_results.items()}
    manual_layer1 = write_layer1_manual_metrics(args.manual_labels or data.config.get("manual_labels"), memberships, output_dir)

    layer2_table = pd.DataFrame()
    layer2_delta = pd.DataFrame()
    layer2_edges: dict[str, pd.DataFrame] = {}
    host_cache_hash = table_hash(pd.DataFrame())
    manual_layer2 = pd.DataFrame()
    if args.run_layer2:
        logger.info("Running Layer-2 ablations")
        layer2_table, layer2_delta, layer2_edges, host_cache_hash = _run_layer2_variants(data, data.config, full, output_dir, args, layer2_variants)
        layer2_table.to_csv(output_dir / "layer2_ablation_table.csv", index=False)
        layer2_delta.to_csv(output_dir / "layer2_ablation_delta.csv", index=False)
        manual_layer2 = layer2_manual_metrics(args.manual_layer2_labels, layer2_edges, output_dir)
    else:
        pd.DataFrame(columns=["ablation_id", "method", "n_candidate_pairs", "n_accepted"]).to_csv(output_dir / "layer2_ablation_table.csv", index=False)
        pd.DataFrame(columns=["ablation_id"]).to_csv(output_dir / "layer2_ablation_delta.csv", index=False)

    validation = _validation_rows(data.config, data.components, candidate_pairs, layer1_results, layer2_edges, host_cache_hash)
    validation.to_csv(output_dir / "fairness_hashes.csv", index=False)

    cases = case_tables(data.components, full["membership"], memberships, edges_by_id, layer2_edges)
    case_rankings = write_case_outputs(data.image, data.components, full["edges"], memberships, cases, output_dir)
    write_ablation_report(output_dir, layer1_table, layer2_table, validation, manual_layer1, manual_layer2, case_rankings)

    run_summary = pd.DataFrame(
        [
            {
                "runtime_seconds": perf_counter() - t_start,
                "git_commit": git_commit(),
                "n_components": int(len(data.components)),
                "n_candidate_pairs": int(len(candidate_pairs)),
                "n_layer1_variants": int(len(layer1_results)),
                "n_layer2_variants": int(len(layer2_edges)),
                "peak_memory_mb": peak_memory_mb(),
            }
        ]
    )
    run_summary.to_csv(output_dir / "run_summary.csv", index=False)
    logger.info("Ablation outputs written under %s", output_dir)


if __name__ == "__main__":
    main()
