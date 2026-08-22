#!/usr/bin/env python3
"""Build a manifest for radio association group annotation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lotss_association.annotation import build_manifest


def parse_multi_value(values):
    if not values:
        return None
    output = []
    for value in values:
        output.extend(part.strip() for part in value.split(",") if part.strip())
    return output or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path, help="radio_association_groups.csv")
    parser.add_argument("--zoom-dir", required=True, type=Path, help="Directory containing zoom PNG files")
    parser.add_argument("--overview-dir", type=Path, help="Directory containing overview PNG files")
    parser.add_argument("--output", required=True, type=Path, help="Output manifest.csv")
    parser.add_argument("--limit", type=int, help="Maximum number of rows to write")
    parser.add_argument("--random-sample", action="store_true", help="Shuffle before applying --limit")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --random-sample or random queue mode")
    parser.add_argument(
        "--filter-quality",
        action="append",
        help="Keep association_quality values. Repeat or comma-separate values.",
    )
    parser.add_argument(
        "--filter-type",
        action="append",
        help="Keep association_type values. Repeat or comma-separate values.",
    )
    parser.add_argument("--min-n-gaussians", type=int, help="Keep groups with n_gaussians >= N")
    parser.add_argument("--min-las-beam", type=float, help="Keep groups with LAS_beam >= N")
    parser.add_argument(
        "--queue-mode",
        choices=["all", "priority", "suspicious", "high", "random"],
        default="all",
        help="Ordering for manifest rows before --limit",
    )
    args = parser.parse_args()

    rows, warnings = build_manifest(
        catalog_path=(REPO_ROOT / args.catalog).resolve() if not args.catalog.is_absolute() else args.catalog,
        zoom_dir=(REPO_ROOT / args.zoom_dir).resolve() if not args.zoom_dir.is_absolute() else args.zoom_dir,
        overview_dir=(REPO_ROOT / args.overview_dir).resolve()
        if args.overview_dir and not args.overview_dir.is_absolute()
        else args.overview_dir,
        output_path=(REPO_ROOT / args.output).resolve() if not args.output.is_absolute() else args.output,
        base_dir=REPO_ROOT,
        qualities=parse_multi_value(args.filter_quality),
        types=parse_multi_value(args.filter_type),
        min_n_gaussians=args.min_n_gaussians,
        min_las_beam=args.min_las_beam,
        limit=args.limit,
        random_sample=args.random_sample or args.queue_mode == "random",
        seed=args.seed,
        queue_mode=args.queue_mode,
    )
    for warning in warnings[:50]:
        print(warning, file=sys.stderr)
    if len(warnings) > 50:
        print(f"warning: {len(warnings) - 50} additional unmatched zoom files", file=sys.stderr)
    print(f"Wrote {len(rows)} annotation items to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

