#!/usr/bin/env python
"""Build segmentation maps for cutouts in an H5 file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lotss_association.io import H5CutoutReader
from lotss_association.segmentation import build_snr_map, save_segmentation, segment_snr_map
from lotss_association.utils import ensure_dir, load_yaml, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(debug=args.debug)
    config = load_yaml(args.config)
    reader = H5CutoutReader(args.h5_path, config_h5=config.get("h5", {}))
    out_dir = ensure_dir(Path(args.output_dir) / "segmentation")
    for index in reader.iter_indices(args.start_index, args.end_index, args.limit):
        cutout = reader.read(index)
        snr, _, _ = build_snr_map(
            cutout.image,
            rms=cutout.rms,
            mean=cutout.mean,
            mean_mode=config.get("mean_mode", "median"),
            rms_mode=config.get("rms_mode", "mad"),
            smooth_before_segmentation=config.get("smooth_before_segmentation", True),
            gaussian_smooth_sigma_pix=float(config.get("gaussian_smooth_sigma_pix", 1.0)),
        )
        seg = segment_snr_map(
            snr,
            thresholds=config.get("snr_thresholds", [5.0, 4.0, 3.0, 2.5, 2.0]),
            min_mask_area_pix=int(config.get("min_mask_area_pix", 20)),
            connectivity=int(config.get("connectivity", 2)),
            binary_opening=bool(config.get("binary_opening", False)),
            binary_closing=bool(config.get("binary_closing", True)),
        )
        path = out_dir / f"{cutout.cutout_id}_seg.npz"
        save_segmentation(path, seg)
        logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
