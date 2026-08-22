#!/usr/bin/env python
"""Redraw association figures from existing catalogs and segmentation products."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from lotss_association.io import H5CutoutReader
from lotss_association.segmentation import load_segmentation
from lotss_association.utils import load_yaml, setup_logging
from lotss_association.visualize import plot_cutout_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--catalog-dir", required=True, help="Directory containing pipeline CSV/parquet catalogs")
    parser.add_argument("--segmentation-dir", required=True, help="Directory containing {cutout_id}_seg.npz files")
    parser.add_argument("--output-dir", required=True, help="Figure output directory")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/real_lotss_conservative.yaml"))
    parser.add_argument("--cutout-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overview-only", action="store_true")
    parser.add_argument("--zoom-only", action="store_true")
    parser.add_argument("--max-labels", type=int, default=None)
    parser.add_argument("--max-zoom-per-cutout", type=int, default=None)
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _read_table(path_base: Path) -> pd.DataFrame:
    csv_path = path_base.with_suffix(".csv")
    parquet_path = path_base.with_suffix(".parquet")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


def load_outputs(catalog_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    association_components = _read_table(catalog_dir / "radio_association_components")
    association_edges = _read_table(catalog_dir / "radio_association_edges")
    association_groups = _read_table(catalog_dir / "radio_association_groups")
    if not association_groups.empty:
        components = association_components if not association_components.empty else _read_table(catalog_dir / "lotss_association_components")
        edges = association_edges
        merged = association_groups
        return components, edges, merged
    components = _read_table(catalog_dir / "lotss_association_components")
    edges = _read_table(catalog_dir / "lotss_association_edges")
    merged = _read_table(catalog_dir / "lotss_association_merged_sources")
    return components, edges, merged


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = dict(config)
    viz = dict(config.get("visualization", {}))
    overview = dict(viz.get("overview", {}))
    zoom = dict(viz.get("zoom", {}))
    if args.max_labels is not None:
        overview["max_labels"] = args.max_labels
    if args.no_labels:
        overview["max_labels"] = 0
        overview["draw_all_labels"] = False
    if args.max_zoom_per_cutout is not None:
        zoom["max_zoom_per_cutout"] = args.max_zoom_per_cutout
    viz["overview"] = overview
    viz["zoom"] = zoom
    config["visualization"] = viz
    return config


def clean_figures(output_dir: Path, cutout_id: str | None = None) -> None:
    overview_dir = output_dir / "overview"
    zoom_dir = output_dir / "zoom"
    if cutout_id:
        for path in overview_dir.glob(f"{cutout_id}.*"):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
                path.unlink()
        for path in zoom_dir.glob(f"{cutout_id}_*.*"):
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
                path.unlink()
        return
    for folder in [output_dir, overview_dir, zoom_dir, output_dir / ".ipynb_checkpoints"]:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}:
                path.unlink()


def main() -> None:
    args = parse_args()
    logger = setup_logging(debug=args.debug)
    config = apply_cli_overrides(load_yaml(args.config), args)

    catalog_dir = Path(args.catalog_dir)
    segmentation_dir = Path(args.segmentation_dir)
    output_dir = Path(args.output_dir)
    (output_dir / "overview").mkdir(parents=True, exist_ok=True)
    (output_dir / "zoom").mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        clean_figures(output_dir, cutout_id=args.cutout_id)

    components, edges, merged = load_outputs(catalog_dir)
    if components.empty and merged.empty:
        raise SystemExit(f"No pipeline outputs found in {catalog_dir}")

    reader = H5CutoutReader(args.h5_path, config_h5=config.get("h5", {}))
    cutout_ids = [reader.read(index).cutout_id for index in reader.iter_indices()]
    id_to_index = {cutout_id: idx for idx, cutout_id in enumerate(cutout_ids)}
    if args.cutout_id:
        selected_ids = [args.cutout_id]
    else:
        if not merged.empty:
            selected_ids = [cutout_id for cutout_id in cutout_ids if cutout_id in set(merged["cutout_id"].astype(str))]
        else:
            selected_ids = [cutout_id for cutout_id in cutout_ids if cutout_id in set(components["cutout_id"].astype(str))]
        if args.limit is not None:
            selected_ids = selected_ids[: args.limit]

    overview_count = 0
    zoom_count = 0
    for cutout_id in selected_ids:
        if cutout_id not in id_to_index:
            logger.warning("Cutout id %s is not in H5 reader", cutout_id)
            continue
        seg_path = segmentation_dir / f"{cutout_id}_seg.npz"
        if not seg_path.exists():
            logger.warning("Missing segmentation for %s: %s", cutout_id, seg_path)
            continue
        cutout = reader.read(id_to_index[cutout_id])
        segmentation = load_segmentation(seg_path)
        cutout_components = components[components["cutout_id"].astype(str) == cutout_id].copy()
        cutout_edges = edges[edges["cutout_id"].astype(str) == cutout_id].copy() if not edges.empty else pd.DataFrame()
        cutout_merged = merged[merged["cutout_id"].astype(str) == cutout_id].copy() if not merged.empty else pd.DataFrame()
        paths = plot_cutout_all(
            cutout,
            segmentation,
            cutout_components,
            cutout_edges,
            cutout_merged,
            output_dir,
            config=config,
            overview_only=args.overview_only,
            zoom_only=args.zoom_only,
        )
        overview_count += len(paths["overview"])
        zoom_count += len(paths["zoom"])
        logger.info("Wrote %s overview and %s zoom figures for %s", len(paths["overview"]), len(paths["zoom"]), cutout_id)

    logger.info("Done. overview=%d zoom=%d output_dir=%s", overview_count, zoom_count, output_dir)


if __name__ == "__main__":
    main()
