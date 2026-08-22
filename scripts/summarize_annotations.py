#!/usr/bin/env python3
"""Summarize exported group-level annotations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lotss_association.annotation import summarize_annotations_csv


def resolve(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path, help="Exported annotations.csv")
    parser.add_argument("--output", required=True, type=Path, help="Output annotation_summary.csv")
    args = parser.parse_args()
    rows = summarize_annotations_csv(resolve(args.annotations), resolve(args.output))
    print(f"Wrote {len(rows)} summary rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

