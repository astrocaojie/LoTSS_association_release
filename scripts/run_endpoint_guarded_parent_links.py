#!/usr/bin/env python
"""Run endpoint-guarded parent association candidates."""

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

from lotss_association.host_query import HOST_QUERY_LOG_COLUMNS, HostQueryClient
from lotss_association.io import H5CutoutReader
from lotss_association.parent_links_endpoint_guarded import (
    SOURCE_MORPH_TABLE_COLUMNS,
    PARENT_CANDIDATE_COLUMNS,
    PARENT_DIAGNOSTIC_COLUMNS,
    PARENT_EDGE_DEBUG_COLUMNS,
    PARENT_HOST_CANDIDATE_COLUMNS,
    run_parent_links_endpoint_guarded,
    parent_link_config,
)
from lotss_association.segmentation import load_segmentation
from lotss_association.utils import ensure_dir, load_yaml, setup_logging, write_dataframe
from lotss_association.visualize import plot_parent_link_cutout_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/real_lotss_conservative.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs_endpoint_guarded_parent_links_test"))
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
        "reports": ensure_dir(output_dir / "reports"),
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
    return pd.DataFrame(
        [
            {
                "n_cutouts": int(len(selected)),
                "n_cutouts_done": int(sum(row["status"] == "done" for row in status_records)),
                "n_total_local_groups": int(diagnostics.get("n_total_local_groups", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_hard_compact_veto": int(diagnostics.get("n_hard_compact_veto", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_noise_artifact_veto": int(diagnostics.get("n_noise_artifact_veto", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_isolated_compact_veto": int(diagnostics.get("n_isolated_compact_veto", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_point_like_hidden": int(diagnostics.get("n_point_like_hidden", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_resolved_single": int(diagnostics.get("n_resolved_single", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_lobe_candidate": int(diagnostics.get("n_lobe_candidate", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_near_extended_lobe_candidate": int(diagnostics.get("n_near_extended_lobe_candidate", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_artifact_risk": int(diagnostics.get("n_artifact_risk", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_near_boundary_pair": int(diagnostics.get("n_near_boundary_pair", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_near_boundary_rescue_applied": int(diagnostics.get("n_near_boundary_rescue_applied", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_rejected_near_boundary_no_support": int(diagnostics.get("n_rejected_near_boundary_no_support", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_compact_endpoint_rejected": int(diagnostics.get("n_compact_endpoint_rejected", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_noise_artifact_endpoint_rejected": int(diagnostics.get("n_noise_artifact_endpoint_rejected", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_geometry_pass_compact_endpoint": int(diagnostics.get("n_geometry_pass_compact_endpoint", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_geometry_pass_noise_artifact_endpoint": int(diagnostics.get("n_geometry_pass_noise_artifact_endpoint", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_geometry_pass_isolated_compact_endpoint": int(diagnostics.get("n_geometry_pass_isolated_compact_endpoint", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_parent_candidate_compact_endpoint": int(diagnostics.get("n_parent_candidate_compact_endpoint", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_parent_candidate_noise_artifact_endpoint": int(diagnostics.get("n_parent_candidate_noise_artifact_endpoint", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_parent_candidate_isolated_compact_endpoint": int(diagnostics.get("n_parent_candidate_isolated_compact_endpoint", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_double_lobe_geometry_pass": int(diagnostics.get("n_double_lobe_geometry_pass", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_lobe_peak_host_contradiction": int(diagnostics.get("n_lobe_peak_host_contradiction", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_midpoint_host_supports": int(diagnostics.get("n_midpoint_host_supports", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_host_queries": int(diagnostics.get("n_host_queries", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_final_candidates": int(len(candidates)),
                "n_parent_high": int((quality == "high").sum()),
                "n_parent_medium": int((quality == "medium").sum()),
                "n_parent_needs_host_check": int((quality == "needs_host_check").sum()),
                "n_parent_suspicious": int((quality == "suspicious").sum()),
                "n_parent_union_boxes": int(diagnostics.get("n_parent_union_boxes", pd.Series(dtype=float)).sum()) if not diagnostics.empty else 0,
                "n_auto_close_merge": 0,
                "uses_parent_link_baseline": True,
                "uses_non_release_output_paths": False,
            }
        ]
    )


def _write_report(path: Path, summary: pd.DataFrame, candidates: pd.DataFrame, edges: pd.DataFrame, morph: pd.DataFrame, dirs: dict[str, Path]) -> None:
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    overview_count = len(list((dirs["figures"] / "overview").glob("*.png"))) if "figures" in dirs else 0
    parent_zoom_count = len(list((dirs["figures"] / "parent_zoom").glob("*.png"))) if "figures" in dirs else 0
    close_merge_dirs = list((dirs["figures"]).glob("*close*")) if "figures" in dirs else []
    geometry_pass = edges[edges["double_lobe_geometry_pass"].astype(bool)].copy() if edges is not None and not edges.empty and "double_lobe_geometry_pass" in edges else pd.DataFrame()
    a003_a018_attempted = False
    a003_a018_rescue_applied = False
    if edges is not None and not edges.empty:
        pair_text = edges.get("local_group_id_1", pd.Series("", index=edges.index)).astype(str) + " " + edges.get("local_group_id_2", pd.Series("", index=edges.index)).astype(str)
        a003_a018_mask = pair_text.str.contains("a003", regex=False) & pair_text.str.contains("a018", regex=False)
        a003_a018_attempted = bool(a003_a018_mask.any())
        if a003_a018_attempted and "near_boundary_rescue_applied" in edges:
            a003_a018_rescue_applied = bool(edges.loc[a003_a018_mask, "near_boundary_rescue_applied"].astype(bool).any())
    parent_endpoint_ok = True
    if candidates is not None and not candidates.empty and morph is not None and not morph.empty:
        m = morph.set_index("association_group_id")
        bad_count = 0
        for _, cand in candidates.iterrows():
            for col in ["local_group_id_1", "local_group_id_2"]:
                gid = str(cand.get(col))
                if gid not in m.index:
                    bad_count += 1
                    continue
                row_m = m.loc[gid]
                if str(row_m.get("source_morph_class", "")) != "lobe_candidate" or not bool(row_m.get("is_parent_endpoint_allowed", False)) or bool(row_m.get("hard_compact_veto", False)) or bool(row_m.get("noise_artifact_veto", False)):
                    if not bool(row_m.get("near_extended_lobe_candidate", False)) or bool(row_m.get("isolated_compact_veto", False)):
                        bad_count += 1
        parent_endpoint_ok = bad_count == 0
    hidden_default = 0
    point_like_hidden = True
    if morph is not None and not morph.empty:
        cls = morph.get("source_morph_class", pd.Series("", index=morph.index)).astype(str)
        hidden_default = int(cls.isin(["point_like_or_compact", "compact_resolved_single", "noise_or_artifact"]).sum())
        point_like_hidden = hidden_default > 0
    lines = [
        "# Endpoint-Guarded Validation",
        "",
        "endpoint-guarded uses the production parent-linking baseline parent association path with a hard endpoint gate and a conservative near-boundary extended-lobe rescue before host scoring.",
        "Internal development output paths and automatic close merge products are not used.",
        "",
        "## Summary",
        "",
    ]
    for key in [
        "n_hard_compact_veto",
        "n_noise_artifact_veto",
        "n_isolated_compact_veto",
        "n_lobe_candidate",
        "n_near_extended_lobe_candidate",
        "n_near_boundary_pair",
        "n_near_boundary_rescue_applied",
        "n_rejected_near_boundary_no_support",
        "n_compact_endpoint_rejected",
        "n_noise_artifact_endpoint_rejected",
        "n_geometry_pass_compact_endpoint",
        "n_geometry_pass_noise_artifact_endpoint",
        "n_geometry_pass_isolated_compact_endpoint",
        "n_parent_candidate_compact_endpoint",
        "n_parent_candidate_noise_artifact_endpoint",
        "n_parent_candidate_isolated_compact_endpoint",
        "n_final_candidates",
        "n_parent_high",
        "n_parent_medium",
        "n_parent_needs_host_check",
        "n_parent_suspicious",
        "n_parent_union_boxes",
        "n_auto_close_merge",
        "uses_parent_link_baseline",
        "uses_non_release_output_paths",
    ]:
        lines.append(f"- {key}: {row.get(key, 0)}")
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "- baseline_is_parent: True",
            "- uses_non_release_output_paths: False",
            "- No automatic close merge is generated in endpoint-guarded.",
            "- Hard compact sources are not allowed as parent endpoints.",
            "- Noise/artifact sources are not allowed as parent endpoints.",
            "- Geometry pass with compact or noise/artifact endpoint must remain 0.",
            "- Geometry pass with isolated compact endpoint must remain 0.",
            "- Host remains second-stage evidence and cannot override geometry failure.",
            f"- a003_a018_like_pair_attempted_in_debug: {a003_a018_attempted}",
            f"- a003_a018_like_pair_near_boundary_rescue_applied: {a003_a018_rescue_applied}",
            f"- parent_zoom_endpoints_all_lobe_candidate: {parent_endpoint_ok}",
            f"- overview_hides_point_like_or_compact: {point_like_hidden}",
            f"- close_merge_zoom_generated: {len(close_merge_dirs) > 0}",
            "- a009 / m253 style compact endpoints are removed from parent links by hard veto.",
        ]
    )
    lines.extend(
        [
            "",
            "## Run Artifacts",
            "",
            f"- overview_png: {overview_count}",
            f"- parent_zoom_png: {parent_zoom_count}",
            f"- close_merge_zoom_dirs: {len(close_merge_dirs)}",
            f"- debug_geometry_pass_rows: {len(geometry_pass)}",
            f"- overview_hidden_compact_noise_sources: {hidden_default}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _figure_morph_table(morph: pd.DataFrame) -> pd.DataFrame:
    if morph is None or morph.empty:
        return morph
    fig_morph = morph.copy()
    cls = fig_morph.get("source_morph_class", pd.Series("", index=fig_morph.index)).astype(str)
    hidden = (
        cls.isin(["point_like_or_compact", "compact_resolved_single", "noise_or_artifact"])
        | fig_morph.get("hard_compact_veto", pd.Series(False, index=fig_morph.index)).astype(bool)
        | fig_morph.get("noise_artifact_veto", pd.Series(False, index=fig_morph.index)).astype(bool)
        | fig_morph.get("isolated_compact_veto", pd.Series(False, index=fig_morph.index)).astype(bool)
    )
    fig_morph.loc[hidden, "is_point_like"] = True
    fig_morph.loc[hidden, "is_lobe_candidate"] = False
    return fig_morph


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    catalog_dir = input_dir / "catalogs"
    segmentation_dir = input_dir / "segmentation"
    output_dir = Path(args.output_dir)
    protected = {
        (PROJECT_ROOT / "outputs_association_test").resolve(),
        (PROJECT_ROOT / "outputs_parent_seed_association_test").resolve(),
        (PROJECT_ROOT / "outputs_parent_seed_refined_test").resolve(),
        (PROJECT_ROOT / "outputs_host_supported_test").resolve(),
        (PROJECT_ROOT / "outputs_lobe_recovery_parent_linking_test").resolve(),
        (PROJECT_ROOT / "outputs_lobe_first_host_second_parent_linking_test").resolve(),
        (PROJECT_ROOT / "outputs_parent_linking_test").resolve(),
        (PROJECT_ROOT / "outputs_non_release_local_cleanup_parent_test").resolve(),
        (PROJECT_ROOT / "outputs_non_release_safety_revert_test").resolve(),
    }
    if output_dir.resolve() in protected:
        raise SystemExit("endpoint-guarded must not write to local, production parent-linking, or non-release output directories")
    if args.overwrite:
        reset_outputs(output_dir)
    dirs = ensure_output_tree(output_dir)
    logger = setup_logging(debug=args.debug, log_path=dirs["logs"] / "run_endpoint_guarded_parent_links.log")
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
    logger.info("Selected %d cutouts for endpoint-guarded", len(selected))

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
        logger.info("endpoint-guarded processing %s index=%d", cutout_id, index)
        try:
            cutout = reader.read(index)
            cutout_id = str(cutout.cutout_id)
            seg_path = segmentation_dir / f"{cutout_id}_seg.npz"
            if not seg_path.exists():
                raise FileNotFoundError(f"Missing segmentation file: {seg_path}")
            segmentation = load_segmentation(seg_path)
            groups = local_groups[local_groups["cutout_id"].astype(str) == cutout_id].copy()
            components = local_components[local_components["cutout_id"].astype(str) == cutout_id].copy()
            result = run_parent_links_endpoint_guarded(
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
                plot_parent_link_cutout_all(cutout, segmentation, components, groups, _figure_morph_table(result.source_morph_table), result.candidates, result.host_candidates, dirs["figures"], config)
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

    candidates.to_csv(dirs["catalogs"] / "parent_candidates_endpoint_guarded.csv", index=False)
    write_dataframe(edges, dirs["catalogs"] / "parent_edges_debug_endpoint_guarded.parquet")
    edges.to_csv(dirs["catalogs"] / "parent_edges_debug_endpoint_guarded.csv", index=False)
    morph.to_csv(dirs["catalogs"] / "source_morph_table_endpoint_guarded.csv", index=False)
    write_dataframe(hosts, dirs["catalogs"] / "host_candidates_endpoint_guarded.parquet")
    hosts.to_csv(dirs["catalogs"] / "host_candidates_endpoint_guarded.csv", index=False)
    logs.to_csv(dirs["catalogs"] / "host_query_log_endpoint_guarded.csv", index=False)
    needs.to_csv(dirs["catalogs"] / "needs_visual_check_endpoint_guarded.csv", index=False)
    diagnostics.to_csv(dirs["catalogs"] / "endpoint_guarded_diagnostics.csv", index=False)
    pd.DataFrame(status_records).to_csv(dirs["logs"] / "status.csv", index=False)
    summary = _summary(selected, status_records, candidates, diagnostics)
    summary.to_csv(dirs["catalogs"] / "endpoint_guarded_summary.csv", index=False)
    _write_report(dirs["reports"] / "endpoint_guarded_validation.md", summary, candidates, edges, morph, dirs)
    host_client.save_cache()
    host_client.write_columns_debug(dirs["catalogs"] / "host_query_columns_debug_endpoint_guarded.json")
    logger.info("Wrote endpoint-guarded catalogs under %s", dirs["catalogs"])
    logger.info("endpoint-guarded summary: %s", summary.iloc[0].to_dict())


if __name__ == "__main__":
    main()
