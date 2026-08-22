#!/usr/bin/env python3
"""Export append-only JSONL annotations to CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lotss_association.annotation import export_annotations


def resolve(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-file", required=True, type=Path, help="Append-only annotations.jsonl")
    parser.add_argument("--manifest", required=True, type=Path, help="Annotation manifest.csv")
    parser.add_argument("--output", required=True, type=Path, help="Output annotations.csv")
    parser.add_argument(
        "--deduplicate",
        choices=["first", "last", "none"],
        default="last",
        help="How to handle repeated item_id labels",
    )
    args = parser.parse_args()
    rows = export_annotations(
        manifest_path_arg=resolve(args.manifest),
        annotation_file=resolve(args.annotation_file),
        output_path=resolve(args.output),
        deduplicate=args.deduplicate,
    )
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

