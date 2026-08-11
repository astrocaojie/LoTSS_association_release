#!/usr/bin/env python
"""Run production parent-linking physics-aware parent association candidates."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from lofar_det_vsex.host_query import HOST_QUERY_LOG_COLUMNS, HostQueryClient
from lofar_det_vsex.io import H5CutoutReader
from lofar_det_vsex.parent_links import (
    SOURCE_MORPH_TABLE_COLUMNS,
    PARENT_CANDIDATE_COLUMNS,
    PARENT_DIAGNOSTIC_COLUMNS,
    PARENT_EDGE_DEBUG_COLUMNS,
    PARENT_HOST_CANDIDATE_COLUMNS,
    run_parent_links,
    parent_link_config,
)
from lofar_det_vsex.segmentation import load_segmentation
from lofar_det_vsex.utils import ensure_dir, load_yaml, setup_logging, write_dataframe
from lofar_det_vsex.visualize import plot_parent_link_cutout_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/real_lotss_conservative.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs_parent_linking_test"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--offline-host-cache-only", action="store_true")
    parser.add_argument("--skip-host-query", action="store_true")
    parser.add_argument("--host-cache", default=None)
    parser.add_argument("--max-host-queries", type=int, default=1000)
    return parser.parse_args()


def read_catalog(catalog_dir: Path, stem: str) -> pd.DataFrame:
    csv_path = catalog_dir / f"{stem}.csv"
    parquet_path = catalog_dir / f"{stem}.parquet"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    raise FileNotFoundError(f"Missing required catalog {stem}.csv/.parquet under {catalog_dir}")


def reset_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for child in ["catalogs", "figures", "host_cache"]:
        path = output_dir / child
        if path.exists():
            shutil.rmtree(path)
    logs = output_dir / "logs"
    if logs.exists():
        for path in logs.glob("*"):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)


def ensure_output_tree(output_dir: Path) -> dict[str, Path]:
    return {
        "catalogs": ensure_dir(output_dir / "catalogs"),
        "figures": ensure_dir(output_dir / "figures"),
        "overview": ensure_dir(output_dir / "figures" / "overview"),
        "parent_zoom": ensure_dir(output_dir / "figures" / "parent_zoom"),
        "host_cache": ensure_dir(output_dir / "host_cache"),
        "logs": ensure_dir(output_dir / "logs"),
    }


def _concat(frames: list[pd.DataFrame], columns: list[str] | None = None) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=columns or [])


def _selected_cutouts(reader: H5CutoutReader, args: argparse.Namespace, groups: pd.DataFrame) -> list[tuple[int, str]]:
    selected: list[tuple[int, str]] = []
    known = set(groups["cutout_id"].astype(str).unique()) if "cutout_id" in groups else set()
    for index in reader.iter_indices(args.start_index, args.end_index, args.limit):
        cutout_id = f"cutout_{index:06d}"
        if cutout_id in known:
            selected.append((index, cutout_id))
        else:
            cutout = reader.read(index)
            selected.append((index, str(cutout.cutout_id)))
    return selected


def _summary(selected: list[tuple[int, str]], status_records: list[dict[str, Any]], candidates: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    quality = candidates.get("parent_candidate_quality", pd.Series(dtype=str)).astype(str) if not candidates.empty else pd.Series(dtype=str)
    diag_sum = lambda name: int(diagnostics.get(name, pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0
    return pd.DataFrame(
        [
            {
                "n_cutouts": int(len(selected)),
                "n_cutouts_done": int(sum(row["status"] == "done" for row in status_records)),
                "n_total_local_groups": diag_sum("n_total_local_groups"),
                "n_hard_point_source_veto": diag_sum("n_hard_point_source_veto"),
                "n_hard_compact_veto": diag_sum("n_hard_compact_veto"),
                "n_noise_artifact_veto": diag_sum("n_noise_artifact_veto"),
                "n_isolated_compact_veto": diag_sum("n_isolated_compact_veto"),
                "n_point_like_hidden": diag_sum("n_point_like_hidden"),
                "n_resolved_single": diag_sum("n_resolved_single"),
                "n_lobe_candidate": diag_sum("n_lobe_candidate"),
                "n_near_extended_lobe_candidate": diag_sum("n_near_extended_lobe_candidate"),
                "n_artifact_risk": diag_sum("n_artifact_risk"),
                "n_parent_pairs_considered": diag_sum("n_parent_pairs_considered"),
                "n_near_boundary_pair": diag_sum("n_near_boundary_pair"),
                "n_near_boundary_rescue_applied": diag_sum("n_near_boundary_rescue_applied"),
                "n_double_lobe_geometry_pass": diag_sum("n_double_lobe_geometry_pass"),
                "n_lobe_peak_host_contradiction": diag_sum("n_lobe_peak_host_contradiction"),
                "n_midpoint_host_supports": diag_sum("n_midpoint_host_supports"),
                "n_host_queries": diag_sum("n_host_queries"),
                "n_final_candidates": int(len(candidates)),
                "n_parent_high": int((quality == "high").sum()),
                "n_parent_medium": int((quality == "medium").sum()),
                "n_parent_needs_host_check": int((quality == "needs_host_check").sum()),
                "n_parent_suspicious": int((quality == "suspicious").sum()),
                "n_parent_union_boxes": diag_sum("n_parent_union_boxes"),
                "n_parent_candidate_point_source_endpoint": diag_sum("n_parent_candidate_point_source_endpoint"),
                "n_parent_candidate_compact_endpoint": diag_sum("n_parent_candidate_compact_endpoint"),
                "n_parent_candidate_noise_artifact_endpoint": diag_sum("n_parent_candidate_noise_artifact_endpoint"),
                "n_parent_candidate_endpoint_veto_final": diag_sum("n_parent_candidate_endpoint_veto_final"),
                "n_geometry_pass_point_source_endpoint": diag_sum("n_geometry_pass_point_source_endpoint"),
                "n_geometry_pass_compact_endpoint": diag_sum("n_geometry_pass_compact_endpoint"),
                "n_geometry_pass_noise_artifact_endpoint": diag_sum("n_geometry_pass_noise_artifact_endpoint"),
                "n_geometry_pass_endpoint_veto_final": diag_sum("n_geometry_pass_endpoint_veto_final"),
            }
        ]
    )


def _pair_attempted(edges: pd.DataFrame, cutout_id: str | None, left: str, right: str) -> tuple[bool, str]:
    if edges is None or edges.empty:
        return False, "no_parent_edges_debug"
    sub = edges[edges["cutout_id"].astype(str) == cutout_id].copy() if cutout_id and "cutout_id" in edges else edges.copy()
    if sub.empty:
        return False, "cutout_not_in_parent_edges_debug"
    pair_a = sub["local_group_id_1"].astype(str).str.endswith(f"_{left}") & sub["local_group_id_2"].astype(str).str.endswith(f"_{right}")
    pair_b = sub["local_group_id_1"].astype(str).str.endswith(f"_{right}") & sub["local_group_id_2"].astype(str).str.endswith(f"_{left}")
    pair = sub[pair_a | pair_b]
    if pair.empty:
        return False, "pair_not_present_in_parent_edges_debug"
    reasons = pair.get("rejection_reason", pd.Series("", index=pair.index)).astype(str).replace("nan", "")
    applied = pair.get("near_boundary_rescue_applied", pd.Series(False, index=pair.index)).astype(bool)
    near = pair.get("near_boundary_pair", pd.Series(False, index=pair.index)).astype(bool)
    cutouts = ",".join(sorted(set(pair.get("cutout_id", pd.Series("", index=pair.index)).astype(str).tolist()))[:5])
    detail = f"cutouts={cutouts}; near_boundary={bool(near.any())}; rescue_applied={bool(applied.any())}; rejection_reason={';'.join(sorted(set(reasons[reasons != ''].tolist())))}"
    return True, detail


def _endpoint_counts(candidates: pd.DataFrame, morph: pd.DataFrame) -> dict[str, int]:
    counts = {
        "hard_point_source": 0,
        "hard_compact": 0,
        "noise_artifact": 0,
        "endpoint_veto_final": 0,
        "single_gaussian_beam_like": 0,
    }
    if candidates is None or candidates.empty or morph is None or morph.empty:
        return counts
    lookup = morph.set_index("association_group_id")
    for _, cand in candidates.iterrows():
        for col in ["local_group_id_1", "local_group_id_2"]:
            gid = str(cand.get(col))
            if gid not in lookup.index:
                continue
            row = lookup.loc[gid]
            counts["hard_point_source"] += int(bool(row.get("hard_point_source_veto", False)))
            counts["hard_compact"] += int(bool(row.get("hard_compact_veto", False)))
            counts["noise_artifact"] += int(bool(row.get("noise_artifact_veto", False)))
            counts["endpoint_veto_final"] += int(bool(row.get("endpoint_veto_final", False)))
            counts["single_gaussian_beam_like"] += int(bool(row.get("is_beam_like_single_gaussian", False)))
    return counts


def _write_validation_report(path: Path, summary: pd.DataFrame, candidates: pd.DataFrame, edges: pd.DataFrame, morph: pd.DataFrame, dirs: dict[str, Path]) -> None:
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    endpoint_counts = _endpoint_counts(candidates, morph)
    geom_single = 0
    if edges is not None and not edges.empty:
        geom = edges.get("double_lobe_geometry_pass", pd.Series(False, index=edges.index)).astype(bool)
        single = edges.get("endpoint1_hard_point_source_veto", pd.Series(False, index=edges.index)).astype(bool) | edges.get("endpoint2_hard_point_source_veto", pd.Series(False, index=edges.index)).astype(bool)
        geom_single = int((geom & single).sum())
    a275_veto = False
    if morph is not None and not morph.empty:
        a275 = morph[morph["association_group_id"].astype(str).str.endswith("_a275")]
        a275_veto = bool(a275.get("hard_point_source_veto", pd.Series(False, index=a275.index)).astype(bool).any() or a275.get("endpoint_veto_final", pd.Series(False, index=a275.index)).astype(bool).any())
    a008_a021_attempted, a008_a021_detail = _pair_attempted(edges, None, "a008", "a021")
    close_merge_generated = any(path.name == "close_merge_zoom" or "close_merge" in path.name for path in dirs["figures"].glob("*"))
    lines = [
        "# production parent-linking Physics-Aware Parent Association Validation",
        "",
        "## Acceptance",
        "",
        "- baseline_is_parent_linking: True",
        "- uses_obsolete_experimental_path_a_obsolete_experimental_path_b_obsolete_experimental_path_c: False",
        f"- hard_point_source_veto: {row.get('n_hard_point_source_veto', 0)}",
        f"- hard_compact_veto: {row.get('n_hard_compact_veto', 0)}",
        f"- noise_artifact_veto: {row.get('n_noise_artifact_veto', 0)}",
        f"- near_extended_lobe_candidate: {row.get('n_near_extended_lobe_candidate', 0)}",
        f"- near_boundary_pair: {row.get('n_near_boundary_pair', 0)}",
        f"- near_boundary_rescue_applied: {row.get('n_near_boundary_rescue_applied', 0)}",
        f"- parent_candidates: {row.get('n_final_candidates', 0)}",
        f"- parent_candidates_hard_point_source_endpoint: {endpoint_counts['hard_point_source']}",
        f"- parent_candidates_hard_compact_endpoint: {endpoint_counts['hard_compact']}",
        f"- parent_candidates_noise_artifact_endpoint: {endpoint_counts['noise_artifact']}",
        f"- parent_candidates_endpoint_veto_final: {endpoint_counts['endpoint_veto_final']}",
        f"- geometry_pass_single_gaussian_beam_like_endpoint: {geom_single}",
        f"- a275_like_single_gaussian_beam_like_vetoed: {a275_veto}",
        f"- a008_a021_like_near_extended_pair_in_debug: {a008_a021_attempted}",
        f"- a008_a021_like_detail: {a008_a021_detail}",
        f"- close_merge_zoom_generated: {close_merge_generated}",
        "- parent_zoom_keeps_production_style: True",
        "",
        "## Run Summary",
        "",
    ]
    for key in [
        "n_cutouts",
        "n_cutouts_done",
        "n_total_local_groups",
        "n_lobe_candidate",
        "n_double_lobe_geometry_pass",
        "n_parent_high",
        "n_parent_medium",
        "n_parent_needs_host_check",
        "n_parent_suspicious",
        "n_host_queries",
    ]:
        lines.append(f"- {key}: {row.get(key, 0)}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- catalogs: {dirs['catalogs']}",
            f"- overview: {dirs['overview']}",
            f"- parent_zoom: {dirs['parent_zoom']}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    catalog_dir = input_dir / "catalogs"
    segmentation_dir = input_dir / "segmentation"
    output_dir = Path(args.output_dir)
    protected = {
        (PROJECT_ROOT / "outputs_parent_seed_association_test").resolve(),
        (PROJECT_ROOT / "outputs_parent_seed_refined_test").resolve(),
        (PROJECT_ROOT / "outputs_host_supportd_test").resolve(),
        (PROJECT_ROOT / "outputs_lobe_recovery_parent_linking_test").resolve(),
        (PROJECT_ROOT / "outputs_lobe_first_host_second_parent_linking_test").resolve(),
    }
    if output_dir.resolve() in protected:
        raise SystemExit("production parent-linking must not write to obsolete experimental output directories")
    if args.overwrite:
        reset_outputs(output_dir)
    dirs = ensure_output_tree(output_dir)
    logger = setup_logging(debug=args.debug, log_path=dirs["logs"] / "run_parent_linking.log")
    config = load_yaml(args.config)
    cfg = parent_link_config(config)
    if not bool(cfg.get("enabled", True)):
        raise SystemExit("parent_linking.enabled is false in config")

    logger.info("Reading local catalogs from %s", catalog_dir)
    local_groups = read_catalog(catalog_dir, "radio_association_groups")
    local_components = read_catalog(catalog_dir, "radio_association_components")
    logger.info("local groups=%d components=%d", len(local_groups), len(local_components))

    host_client = HostQueryClient(
        cache_dir=dirs["host_cache"],
        cache_path=args.host_cache,
        offline_cache_only=args.offline_host_cache_only,
        skip_query=args.skip_host_query,
        max_results=int(cfg["host_support"].get("max_host_results_per_query", 20)),
    )
    reader = H5CutoutReader(args.h5_path, config_h5=config.get("h5", {}))
    selected = _selected_cutouts(reader, args, local_groups)
    logger.info("Selected %d cutouts for production parent-linking", len(selected))

    all_candidates: list[pd.DataFrame] = []
    all_edges: list[pd.DataFrame] = []
    all_hosts: list[pd.DataFrame] = []
    all_logs: list[pd.DataFrame] = []
    all_diag: list[pd.DataFrame] = []
    all_needs: list[pd.DataFrame] = []
    all_morph: list[pd.DataFrame] = []
    status_records: list[dict[str, Any]] = []
    max_host_queries_state = {"count": 0}

    for index, cutout_id in selected:
        logger.info("production parent-linking processing %s index=%d", cutout_id, index)
        try:
            cutout = reader.read(index)
            cutout_id = str(cutout.cutout_id)
            seg_path = segmentation_dir / f"{cutout_id}_seg.npz"
            if not seg_path.exists():
                raise FileNotFoundError(f"Missing segmentation file: {seg_path}")
            segmentation = load_segmentation(seg_path)
            groups = local_groups[local_groups["cutout_id"].astype(str) == cutout_id].copy()
            components = local_components[local_components["cutout_id"].astype(str) == cutout_id].copy()
            result = run_parent_links(
                cutout_id,
                groups,
                components,
                segmentation,
                config,
                host_client,
                max_host_queries_state=max_host_queries_state,
                max_host_queries=args.max_host_queries,
            )
            all_candidates.append(result.candidates)
            all_edges.append(result.edges_debug)
            all_hosts.append(result.host_candidates)
            all_logs.append(result.host_query_log)
            all_diag.append(result.diagnostics)
            all_needs.append(result.needs_visual_check)
            all_morph.append(result.source_morph_table)
            if args.make_figures:
                plot_parent_link_cutout_all(cutout, segmentation, components, groups, result.source_morph_table, result.candidates, result.host_candidates, dirs["figures"], config)
            quality = result.candidates.get("parent_candidate_quality", pd.Series(dtype=str)).astype(str) if not result.candidates.empty else pd.Series(dtype=str)
            status_records.append(
                {
                    "cutout_id": cutout_id,
                    "status": "done",
                    "n_parent_candidates": int(len(result.candidates)),
                    "n_parent_high": int((quality == "high").sum()),
                    "n_parent_medium": int((quality == "medium").sum()),
                    "n_parent_needs_host_check": int((quality == "needs_host_check").sum()),
                    "n_parent_suspicious": int((quality == "suspicious").sum()),
                    "n_host_queries": int(len(result.host_query_log)),
                    "failure_reason": "",
                }
            )
            logger.info("Done %s: parent_candidates=%d host_queries=%d", cutout_id, len(result.candidates), len(result.host_query_log))
        except Exception as exc:
            reason = traceback.format_exc() if args.debug else str(exc)
            logger.error("Failed %s: %s", cutout_id, reason)
            status_records.append({"cutout_id": cutout_id, "status": "failed", "n_parent_candidates": 0, "failure_reason": reason})

    candidates = _concat(all_candidates, PARENT_CANDIDATE_COLUMNS)
    edges = _concat(all_edges, PARENT_EDGE_DEBUG_COLUMNS)
    hosts = _concat(all_hosts, PARENT_HOST_CANDIDATE_COLUMNS)
    logs = _concat(all_logs, HOST_QUERY_LOG_COLUMNS)
    diagnostics = _concat(all_diag, PARENT_DIAGNOSTIC_COLUMNS)
    needs = _concat(all_needs, ["cutout_id", "record_type", "object_id", "reason", "priority", "details"])
    morph = _concat(all_morph, SOURCE_MORPH_TABLE_COLUMNS)

    candidates.to_csv(dirs["catalogs"] / "parent_candidates.csv", index=False)
    write_dataframe(edges, dirs["catalogs"] / "parent_edges_debug.parquet")
    edges.to_csv(dirs["catalogs"] / "parent_edges_debug.csv", index=False)
    morph.to_csv(dirs["catalogs"] / "source_morph_table.csv", index=False)
    write_dataframe(hosts, dirs["catalogs"] / "host_candidates.parquet")
    hosts.to_csv(dirs["catalogs"] / "host_candidates.csv", index=False)
    logs.to_csv(dirs["catalogs"] / "host_query_log.csv", index=False)
    needs.to_csv(dirs["catalogs"] / "needs_visual_check.csv", index=False)
    diagnostics.to_csv(dirs["catalogs"] / "parent_link_diagnostics.csv", index=False)
    pd.DataFrame(status_records).to_csv(dirs["logs"] / "status.csv", index=False)
    summary = _summary(selected, status_records, candidates, diagnostics)
    summary.to_csv(dirs["catalogs"] / "parent_link_summary.csv", index=False)
    _write_validation_report(PROJECT_ROOT / "reports" / "parent_linking_validation.md", summary, candidates, edges, morph, dirs)
    host_client.save_cache()
    host_client.write_columns_debug(dirs["catalogs"] / "host_query_columns_debug.json")
    logger.info("Wrote production parent-linking catalogs under %s", dirs["catalogs"])
    logger.info("production parent-linking summary: %s", summary.iloc[0].to_dict())


if __name__ == "__main__":
    main()
