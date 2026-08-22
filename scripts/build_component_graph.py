#!/usr/bin/env python
"""Build component graphs for a small cutout sample."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from lotss_association.catalog import normalized_gaussian_dataframe, read_gaussian_catalog
from lotss_association.graph_merge import build_component_graph
from lotss_association.io import H5CutoutReader
from lotss_association.matching import match_gaussians_to_cutout
from lotss_association.segmentation import build_snr_map, segment_snr_map
from lotss_association.utils import ensure_dir, load_yaml, setup_logging, write_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--gaus-catalog", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(debug=args.debug)
    config = load_yaml(args.config)
    gaussians, _ = normalized_gaussian_dataframe(read_gaussian_catalog(args.gaus_catalog))
    reader = H5CutoutReader(args.h5_path, config_h5=config.get("h5", {}))
    all_edges = []
    for index in reader.iter_indices(limit=args.limit):
        cutout = reader.read(index)
        snr, _, _ = build_snr_map(cutout.image, cutout.rms, cutout.mean)
        seg = segment_snr_map(snr, config.get("snr_thresholds", [5, 4, 3, 2.5, 2]))
        comps, mode = match_gaussians_to_cutout(gaussians, cutout, seg)
        if len(comps) == 0:
            logger.info("%s: no components, mode=%s", cutout.cutout_id, mode)
            continue
        result = build_component_graph(comps, seg, config)
        all_edges.append(result.edges)
        logger.info("%s: nodes=%d edges=%d clusters=%d", cutout.cutout_id, len(comps), len(result.edges), len(result.clusters))
    edges = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
    out_dir = ensure_dir(Path(args.output_dir) / "graphs")
    write_dataframe(edges, out_dir / "lotss_association_edges.parquet")
    edges.to_csv(out_dir / "lotss_association_edges.csv", index=False)


if __name__ == "__main__":
    main()
