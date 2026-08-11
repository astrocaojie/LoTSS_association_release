#!/usr/bin/env python
"""Print PyBDSF Gaussian catalog columns and detected aliases."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lofar_det_vsex.catalog import print_catalog_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gaus-catalog",
        required=True,
        help="PyBDSF Gaussian FITS catalog",
    )
    parser.add_argument("--n-rows", type=int, default=5, help="Number of preview rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_catalog_summary(args.gaus_catalog, n_rows=args.n_rows)


if __name__ == "__main__":
    main()
