#!/usr/bin/env python
"""Inspect an H5 cutout file and auto-detect likely datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lofar_det_vsex.io import detect_h5_keys, print_h5_structure, summarize_h5_file
from lofar_det_vsex.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-path", required=True, help="Path to H5 cutout file")
    parser.add_argument("--config", default=None, help="Optional YAML config with h5 key overrides")
    parser.add_argument("--max-attrs", type=int, default=20, help="Maximum attributes to print per object")
    parser.add_argument("--max-objects", type=int, default=None, help="Maximum H5 objects to print")
    parser.add_argument("--summary", action="store_true", help="Print concise detected summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_h5 = {}
    if args.config:
        config_h5 = load_yaml(args.config).get("h5", {})
    print_h5_structure(args.h5_path, max_attrs=args.max_attrs, max_objects=args.max_objects)
    print("\nAuto-detected keys:")
    keys = detect_h5_keys(args.h5_path, config_h5=config_h5)
    for field_name in keys.__dataclass_fields__:
        print(f"  {field_name}: {getattr(keys, field_name)}")
    if args.summary:
        print("\nConcise summary:")
        for key, value in summarize_h5_file(args.h5_path, config_h5=config_h5).items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
