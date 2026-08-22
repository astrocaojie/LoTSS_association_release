#!/usr/bin/env python
"""Inspect the DR1 component reference catalogue columns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lotss_association.validation.dr1_reference import DEFAULT_DR1_COMPONENT_CSV, load_dr1_component_catalogue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dr1-catalogue", default=str(DEFAULT_DR1_COMPONENT_CSV))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs/dr1_validation/ml/dr1_catalog_columns_report.txt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _table, summary = load_dr1_component_catalogue(args.dr1_catalogue, report_path=args.output)
    print("DR1 component catalogue inspection written to", args.output)
    print(summary)


if __name__ == "__main__":
    main()
