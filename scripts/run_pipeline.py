#!/usr/bin/env python
"""Run the rule-based LoTSS Vsex extended-source pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from lofar_det_vsex.association import EDGE_COLUMNS as ASSOCIATION_EDGE_COLUMNS
from lofar_det_vsex.association import GROUP_COLUMNS as ASSOCIATION_GROUP_COLUMNS
from lofar_det_vsex.association import run_component_association
from lofar_det_vsex.catalog import normalized_gaussian_dataframe, read_gaussian_catalog
from lofar_det_vsex.graph_merge import build_component_graph
from lofar_det_vsex.io import H5CutoutReader
from lofar_det_vsex.matching import match_gaussians_to_cutout
from lofar_det_vsex.measurements import measure_merged_sources
from lofar_det_vsex.segmentation import (
    build_snr_map,
    save_segmentation,
    segment_snr_map,
    segmentation_diagnostics,
)
from lofar_det_vsex.utils import ensure_dir, infer_pixel_scale_arcsec, load_yaml, setup_logging, write_dataframe
from lofar_det_vsex.visualize import plot_cutout_all


EDGE_COLUMNS = [
    "cutout_id",
    "gaussian_id_1",
    "gaussian_id_2",
    "distance_arcsec",
    "same_pybdsf_island",
    "connected_at_3sigma",
    "connected_at_2p5sigma",
    "connected_at_2sigma",
    "bridge_snr_mean",
    "merge_score",
    "merge_decision",
    "positive_evidence",
    "negative_evidence",
]

MERGED_COLUMNS = [
    "cutout_id",
    "merged_source_id",
    "n_components",
    "gaussian_ids",
    "island_ids",
    "ra",
    "dec",
    "centroid_x",
    "centroid_y",
    "total_flux_gaussian",
    "total_flux_pixel_2sigma",
    "total_flux_pixel_2p5sigma",
    "peak_flux",
    "LAS_arcsec",
    "PA",
    "merge_confidence",
    "flags",
    "debug_info",
]

ASSOCIATION_COMPONENT_COLUMNS = [
    "cutout_id",
    "cutout_index",
    "component_index",
    "_source_id",
    "_island_id",
    "_gaussian_id",
    "_ra",
    "_dec",
    "_total_flux",
    "_peak_flux",
    "_maj",
    "_min",
    "_pa",
    "_dc_maj",
    "_dc_min",
    "_dc_pa",
    "x",
    "y",
    "pixel_scale_arcsec",
    "association_group_id",
    "association_group_index",
    "association_group_size",
    "association_quality",
    "association_type",
    "morphology_class",
    "resolved_probability",
    "resolved_significance",
    "beam_like_score",
    "classification_reason",
    "observed_major_arcsec",
    "observed_minor_arcsec",
    "observed_pa_pixel_deg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-path", required=True, help="H5 file containing LoTSS cutouts")
    parser.add_argument(
        "--gaus-catalog",
        required=True,
        help="PyBDSF Gaussian FITS catalog",
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--n-workers", type=int, default=1, help="Accepted for CLI compatibility; first version runs sequentially")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip cutouts marked done in status.csv")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess selected cutouts even if outputs exist")
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--association-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write beam-aware radio association catalogs and use them for figures.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_status(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(
        columns=[
            "cutout_id",
            "status",
            "time_start",
            "time_end",
            "n_gaussians",
            "n_merged_sources",
            "n_association_groups",
            "failure_reason",
        ]
    )


def update_status(path: Path, record: dict) -> None:
    status = load_status(path)
    status = status[status["cutout_id"].astype(str) != str(record["cutout_id"])]
    status = pd.concat([status, pd.DataFrame([record])], ignore_index=True)
    status.to_csv(path, index=False)


def done_cutouts(path: Path) -> set[str]:
    status = load_status(path)
    if status.empty or "status" not in status:
        return set()
    done = status[status["status"].astype(str) == "done"]
    return set(done["cutout_id"].astype(str).tolist())


def ensure_output_tree(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "segmentation": ensure_dir(output_dir / "segmentation"),
        "graphs": ensure_dir(output_dir / "graphs"),
        "catalogs": ensure_dir(output_dir / "catalogs"),
        "figures": ensure_dir(output_dir / "figures"),
        "logs": ensure_dir(output_dir / "logs"),
        "partials": ensure_dir(output_dir / "catalogs" / "partials"),
    }
    return dirs


def reset_overwrite_outputs(dirs: dict[str, Path]) -> None:
    """Clear append-style outputs so --overwrite starts a clean validation run."""

    for name in [
        "matching_diagnostics.csv",
        "segmentation_diagnostics.csv",
        "edge_diagnostics.csv",
        "association_diagnostics.csv",
    ]:
        path = dirs["catalogs"] / name
        if path.exists():
            path.unlink()
    for path in dirs["partials"].glob("*.csv"):
        path.unlink()
    for path in dirs["figures"].glob("*_vsex.png"):
        path.unlink()
    for folder_name in ["overview", "zoom"]:
        folder = dirs["figures"] / folder_name
        if folder.exists():
            for path in folder.glob("*.png"):
                path.unlink()
    for path in dirs["segmentation"].glob("*_seg.npz"):
        path.unlink()
    status_path = dirs["logs"] / "status.csv"
    if status_path.exists():
        status_path.unlink()


def append_diagnostics(path: Path, records: list[dict]) -> None:
    if not records:
        return
    df = pd.DataFrame(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def _with_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out:
            out[col] = pd.Series(dtype=object)
    if out.empty:
        return pd.DataFrame(columns=columns)
    return out


def write_cutout_partials(
    partial_dir: Path,
    cutout_id: str,
    merged: pd.DataFrame,
    edges: pd.DataFrame,
    components: pd.DataFrame,
    association_groups: pd.DataFrame | None = None,
    association_edges: pd.DataFrame | None = None,
    association_components: pd.DataFrame | None = None,
) -> None:
    merged = _with_columns(merged, MERGED_COLUMNS)
    edges = _with_columns(edges, EDGE_COLUMNS)
    association_groups = _with_columns(association_groups if association_groups is not None else pd.DataFrame(), ASSOCIATION_GROUP_COLUMNS)
    association_edges = _with_columns(association_edges if association_edges is not None else pd.DataFrame(), ASSOCIATION_EDGE_COLUMNS)
    association_components = _with_columns(
        association_components if association_components is not None else pd.DataFrame(),
        ASSOCIATION_COMPONENT_COLUMNS,
    )
    merged.to_csv(partial_dir / f"{cutout_id}_merged_sources.csv", index=False)
    edges.to_csv(partial_dir / f"{cutout_id}_edges.csv", index=False)
    components.to_csv(partial_dir / f"{cutout_id}_components.csv", index=False)
    association_groups.to_csv(partial_dir / f"{cutout_id}_radio_association_groups.csv", index=False)
    association_edges.to_csv(partial_dir / f"{cutout_id}_radio_association_edges.csv", index=False)
    association_components.to_csv(partial_dir / f"{cutout_id}_radio_association_components.csv", index=False)


def matching_diagnostic_record(cutout, components: pd.DataFrame, match_mode: str, n_outside: int = 0) -> dict:
    has_ra_dec = cutout.ra is not None and cutout.dec is not None
    warning = ""
    if len(components) == 0:
        warning = "no_gaussians_matched"
    if match_mode == "fallback_no_wcs":
        warning = f"{warning};no_wcs_or_pixel_coords".strip(";")
    return {
        "cutout_id": cutout.cutout_id,
        "n_gaussians_matched": int(len(components)),
        "image_shape": "x".join(map(str, cutout.image.shape)),
        "has_wcs": bool(cutout.wcs is not None),
        "has_ra_dec": bool(has_ra_dec),
        "match_mode": match_mode,
        "min_x": float(components["x"].min()) if len(components) else float("nan"),
        "max_x": float(components["x"].max()) if len(components) else float("nan"),
        "min_y": float(components["y"].min()) if len(components) else float("nan"),
        "max_y": float(components["y"].max()) if len(components) else float("nan"),
        "n_outside_after_projection": int(n_outside),
        "warning": warning,
    }


def edge_diagnostic_record(cutout_id: str, components: pd.DataFrame, edges: pd.DataFrame) -> dict:
    n_nodes = int(len(components))
    if edges.empty:
        return {
            "cutout_id": cutout_id,
            "n_nodes": n_nodes,
            "n_candidate_pairs": 0,
            "n_edges_merged": 0,
            "merge_score_min": float("nan"),
            "merge_score_median": float("nan"),
            "merge_score_max": float("nan"),
            "connected_at_3sigma_count": 0,
            "connected_at_2p5sigma_count": 0,
            "connected_at_2sigma_count": 0,
            "only_2sigma_connected_count": 0,
            "median_distance_arcsec": float("nan"),
            "max_distance_arcsec": float("nan"),
            "max_merged_component_size": 1 if n_nodes else 0,
            "warning": "no_candidate_pairs",
        }
    merged_edges = edges[edges["merge_decision"].astype(bool)]
    only_2 = edges[
        edges["connected_at_2sigma"].astype(bool)
        & ~edges["connected_at_2p5sigma"].astype(bool)
        & ~edges["connected_at_3sigma"].astype(bool)
    ]
    max_component_size = 0
    if "merged_component_group" in components and len(components):
        max_component_size = int(components.groupby("merged_component_group").size().max())
    warning = ""
    if len(merged_edges) and len(only_2) / max(len(edges), 1) > 0.5:
        warning = "many_only_2sigma_pairs"
    if max_component_size > 20:
        warning = f"{warning};large_connected_component".strip(";")
    return {
        "cutout_id": cutout_id,
        "n_nodes": n_nodes,
        "n_candidate_pairs": int(len(edges)),
        "n_edges_merged": int(len(merged_edges)),
        "merge_score_min": float(edges["merge_score"].min()),
        "merge_score_median": float(edges["merge_score"].median()),
        "merge_score_max": float(edges["merge_score"].max()),
        "connected_at_3sigma_count": int(edges["connected_at_3sigma"].astype(bool).sum()),
        "connected_at_2p5sigma_count": int(edges["connected_at_2p5sigma"].astype(bool).sum()),
        "connected_at_2sigma_count": int(edges["connected_at_2sigma"].astype(bool).sum()),
        "only_2sigma_connected_count": int(len(only_2)),
        "median_distance_arcsec": float(edges["distance_arcsec"].median()),
        "max_distance_arcsec": float(edges["distance_arcsec"].max()),
        "max_merged_component_size": max_component_size,
        "warning": warning,
    }


def association_diagnostic_record(
    cutout_id: str,
    components: pd.DataFrame,
    groups: pd.DataFrame,
    edges: pd.DataFrame,
) -> dict:
    n_nodes = int(len(components))
    if edges.empty:
        return {
            "cutout_id": cutout_id,
            "n_nodes": n_nodes,
            "n_candidate_pairs": 0,
            "n_association_groups": int(len(groups)),
            "n_strong_edges": 0,
            "n_weak_edges": 0,
            "n_rejected_edges": 0,
            "n_decision_edges": 0,
            "n_only_2sigma_edges": 0,
            "n_unresolved": 0,
            "n_marginally_resolved": 0,
            "n_resolved": 0,
            "n_artifact_like": 0,
            "n_unresolved_pair_veto": 0,
            "association_score_min": float("nan"),
            "association_score_median": float("nan"),
            "association_score_max": float("nan"),
            "max_group_size": int(groups["n_gaussians"].max()) if not groups.empty and "n_gaussians" in groups else (1 if n_nodes else 0),
            "warning": "no_candidate_pairs",
        }
    only_2 = edges[
        edges["only_2sigma_connected"].astype(bool)
        & edges["association_decision"].astype(bool)
    ]
    morph = components.get("morphology_class", pd.Series(dtype=str)).astype(str) if not components.empty else pd.Series(dtype=str)
    max_group_size = int(groups["n_gaussians"].max()) if not groups.empty and "n_gaussians" in groups else 0
    warning = ""
    if len(only_2) > max(1, int(0.5 * max(len(edges[edges["association_decision"].astype(bool)]), 1))):
        warning = "many_only_2sigma_associations"
    if max_group_size > 20:
        warning = f"{warning};large_association_group".strip(";")
    return {
        "cutout_id": cutout_id,
        "n_nodes": n_nodes,
        "n_candidate_pairs": int(len(edges)),
        "n_association_groups": int(len(groups)),
        "n_strong_edges": int((edges["edge_type"].astype(str) == "strong").sum()),
        "n_weak_edges": int((edges["edge_type"].astype(str) == "weak").sum()),
        "n_rejected_edges": int((edges["edge_type"].astype(str) == "rejected").sum()),
        "n_decision_edges": int(edges["association_decision"].astype(bool).sum()),
        "n_only_2sigma_edges": int(len(only_2)),
        "n_unresolved": int((morph == "unresolved").sum()),
        "n_marginally_resolved": int((morph == "marginally_resolved").sum()),
        "n_resolved": int((morph == "resolved").sum()),
        "n_artifact_like": int((morph == "artifact_like").sum()),
        "n_unresolved_pair_veto": int(edges.get("unresolved_pair_veto", pd.Series(dtype=bool)).astype(bool).sum()),
        "association_score_min": float(pd.to_numeric(edges["association_score"], errors="coerce").min()),
        "association_score_median": float(pd.to_numeric(edges["association_score"], errors="coerce").median()),
        "association_score_max": float(pd.to_numeric(edges["association_score"], errors="coerce").max()),
        "max_group_size": max_group_size,
        "warning": warning,
    }


def combine_partials(output_dir: Path) -> None:
    partial_dir = output_dir / "catalogs" / "partials"
    catalogs_dir = output_dir / "catalogs"

    def combine(pattern: str, exclude: tuple[str, ...] = ()) -> pd.DataFrame:
        frames = []
        for path in sorted(partial_dir.glob(pattern)):
            if any(token in path.name for token in exclude):
                continue
            try:
                frames.append(pd.read_csv(path))
            except pd.errors.EmptyDataError:
                continue
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    merged = combine("*_merged_sources.csv")
    edges = combine("*_edges.csv", exclude=("radio_association",))
    components = combine("*_components.csv", exclude=("radio_association",))
    association_groups = combine("*_radio_association_groups.csv")
    association_edges = combine("*_radio_association_edges.csv")
    association_components = combine("*_radio_association_components.csv")

    merged = _with_columns(merged, MERGED_COLUMNS)
    edges = _with_columns(edges, EDGE_COLUMNS)
    association_groups = _with_columns(association_groups, ASSOCIATION_GROUP_COLUMNS)
    association_edges = _with_columns(association_edges, ASSOCIATION_EDGE_COLUMNS)
    association_components = _with_columns(association_components, ASSOCIATION_COMPONENT_COLUMNS)

    merged.to_csv(catalogs_dir / "lofar_det_vsex_merged_sources.csv", index=False)
    write_dataframe(merged, catalogs_dir / "lofar_det_vsex_merged_sources.parquet")
    write_dataframe(edges, catalogs_dir / "lofar_det_vsex_edges.parquet")
    write_dataframe(components, catalogs_dir / "lofar_det_vsex_components.parquet")
    edges.to_csv(catalogs_dir / "lofar_det_vsex_edges.csv", index=False)
    components.to_csv(catalogs_dir / "lofar_det_vsex_components.csv", index=False)
    association_groups.to_csv(catalogs_dir / "radio_association_groups.csv", index=False)
    write_dataframe(association_groups, catalogs_dir / "radio_association_groups.parquet")
    write_dataframe(association_edges, catalogs_dir / "radio_association_edges.parquet")
    write_dataframe(association_components, catalogs_dir / "radio_association_components.parquet")
    association_edges.to_csv(catalogs_dir / "radio_association_edges.csv", index=False)
    association_components.to_csv(catalogs_dir / "radio_association_components.csv", index=False)


def process_cutout(
    cutout,
    gaussians: pd.DataFrame,
    config: dict,
    dirs: dict[str, Path],
    make_figures: bool,
    association_mode: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 单 cutout 的流水线顺序固定：S/N 图 -> 多阈值分割 -> Gaussian 匹配 -> 图关联 -> 输出/绘图。
    # 每一步都会追加诊断表，方便在大批量运行后定位失败或异常 cutout。
    snr, mean_value, rms_value = build_snr_map(
        cutout.image,
        rms=cutout.rms,
        mean=cutout.mean,
        mean_mode=config.get("mean_mode", "median"),
        rms_mode=config.get("rms_mode", "mad"),
        smooth_before_segmentation=bool(config.get("smooth_before_segmentation", True)),
        gaussian_smooth_sigma_pix=float(config.get("gaussian_smooth_sigma_pix", 1.0)),
    )
    segmentation = segment_snr_map(
        snr,
        thresholds=config.get("snr_thresholds", [5.0, 4.0, 3.0, 2.5, 2.0]),
        min_mask_area_pix=int(config.get("min_mask_area_pix", 20)),
        connectivity=int(config.get("connectivity", 2)),
        binary_opening=bool(config.get("binary_opening", False)),
        binary_closing=bool(config.get("binary_closing", True)),
    )
    save_segmentation(dirs["segmentation"] / f"{cutout.cutout_id}_seg.npz", segmentation)
    append_diagnostics(
        dirs["catalogs"] / "segmentation_diagnostics.csv",
        segmentation_diagnostics(cutout.cutout_id, segmentation),
    )

    components, matching_mode = match_gaussians_to_cutout(gaussians, cutout, segmentation)
    components["matching_mode"] = matching_mode
    components["snr_mean_used"] = str(mean_value if not hasattr(mean_value, "shape") else "map")
    components["snr_rms_used"] = str(rms_value if not hasattr(rms_value, "shape") else "map")
    append_diagnostics(
        dirs["catalogs"] / "matching_diagnostics.csv",
        [
            matching_diagnostic_record(
                cutout,
                components,
                matching_mode,
                int(components.attrs.get("n_outside_after_projection", 0)),
            )
        ],
    )

    if len(components) == 0:
        edges = pd.DataFrame(columns=EDGE_COLUMNS)
        merged = pd.DataFrame(columns=MERGED_COLUMNS)
        association_groups = pd.DataFrame(columns=ASSOCIATION_GROUP_COLUMNS)
        association_edges = pd.DataFrame(columns=ASSOCIATION_EDGE_COLUMNS)
        association_components = pd.DataFrame(columns=ASSOCIATION_COMPONENT_COLUMNS)
    else:
        graph_result = build_component_graph(components, segmentation, config)
        edges = graph_result.edges
        components = graph_result.components
        merged = measure_merged_sources(
            cutout,
            segmentation,
            graph_result.components,
            graph_result.clusters,
            graph_result.edges,
        )
        if association_mode and bool(config.get("association", {}).get("enabled", True)):
            association_result = run_component_association(cutout, segmentation, components, config)
            association_groups = association_result.groups
            association_edges = association_result.edges
            association_components = association_result.components
            components = association_components
        else:
            association_groups = pd.DataFrame(columns=ASSOCIATION_GROUP_COLUMNS)
            association_edges = pd.DataFrame(columns=ASSOCIATION_EDGE_COLUMNS)
            association_components = pd.DataFrame(columns=ASSOCIATION_COMPONENT_COLUMNS)
    append_diagnostics(
        dirs["catalogs"] / "edge_diagnostics.csv",
        [edge_diagnostic_record(cutout.cutout_id, components, edges)],
    )
    if association_mode:
        append_diagnostics(
            dirs["catalogs"] / "association_diagnostics.csv",
            [association_diagnostic_record(cutout.cutout_id, components, association_groups, association_edges)],
        )

    write_cutout_partials(
        dirs["partials"],
        cutout.cutout_id,
        merged,
        edges,
        components,
        association_groups=association_groups,
        association_edges=association_edges,
        association_components=association_components,
    )

    if make_figures:
        figure_edges = association_edges if association_mode else edges
        figure_groups = association_groups if association_mode else merged
        plot_cutout_all(
            cutout,
            segmentation,
            components,
            figure_edges,
            figure_groups,
            dirs["figures"],
            config,
        )
    return merged, edges, components, association_groups


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    dirs = ensure_output_tree(output_dir)
    logger = setup_logging(debug=args.debug, log_path=dirs["logs"] / "run_pipeline.log")
    config = load_yaml(args.config)
    if args.overwrite and not args.resume:
        reset_overwrite_outputs(dirs)

    if args.n_workers != 1:
        logger.info("n-workers=%d requested; first rule-based version processes cutouts sequentially.", args.n_workers)

    logger.info("Reading Gaussian catalog: %s", args.gaus_catalog)
    gaussians, columns = normalized_gaussian_dataframe(read_gaussian_catalog(args.gaus_catalog))
    logger.info("Gaussian catalog rows: %d", len(gaussians))

    reader = H5CutoutReader(args.h5_path, config_h5=config.get("h5", {}))
    logger.info("H5 image key: %s", reader.keys.image_key)
    logger.info("Number of cutouts available: %d", len(reader))

    status_path = dirs["logs"] / "status.csv"
    completed = done_cutouts(status_path) if args.resume else set()

    indices = reader.iter_indices(args.start_index, args.end_index, args.limit)
    logger.info("Selected %d cutouts", len(indices))
    for index in indices:
        # resume 模式按 status.csv 跳过已完成 cutout，支持长任务中断后继续跑。
        cutout = reader.read(index)
        if args.resume and not args.overwrite and str(cutout.cutout_id) in completed:
            logger.info("Skipping %s because status is done", cutout.cutout_id)
            continue

        start = now_iso()
        logger.info("Processing %s index=%s", cutout.cutout_id, index)
        try:
            merged, _edges, components, association_groups = process_cutout(
                cutout,
                gaussians,
                config,
                dirs,
                args.make_figures,
                association_mode=args.association_mode,
            )
            update_status(
                status_path,
                {
                    "cutout_id": cutout.cutout_id,
                    "status": "done",
                    "time_start": start,
                    "time_end": now_iso(),
                    "n_gaussians": int(len(components)),
                    "n_merged_sources": int(len(merged)),
                    "n_association_groups": int(len(association_groups)),
                    "failure_reason": "",
                },
            )
            logger.info(
                "Done %s: n_gaussians=%d n_merged_sources=%d n_association_groups=%d",
                cutout.cutout_id,
                len(components),
                len(merged),
                len(association_groups),
            )
        except Exception as exc:
            reason = str(exc)
            if args.debug:
                reason = traceback.format_exc()
                logger.error("Failed %s\n%s", cutout.cutout_id, reason)
            else:
                logger.error("Failed %s: %s", cutout.cutout_id, reason)
            update_status(
                status_path,
                {
                    "cutout_id": cutout.cutout_id,
                    "status": "failed",
                    "time_start": start,
                    "time_end": now_iso(),
                    "n_gaussians": 0,
                    "n_merged_sources": 0,
                    "n_association_groups": 0,
                    "failure_reason": reason,
                },
            )

    combine_partials(output_dir)
    logger.info("Wrote merged catalogs under %s", output_dir / "catalogs")


if __name__ == "__main__":
    main()
