#!/usr/bin/env python
"""Evaluate full A0-A10 DR1 ablation catalogues with component-only support."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validation.dr1_ablation_core6_evaluate import main as core_main


def main() -> None:
    core_main()
    output_dir = Path("outputs/dr1_validation/ablation_full_A0_A10")
    args = sys.argv[1:]
    if "--output-dir" in args:
        idx = args.index("--output-dir")
        if idx + 1 < len(args):
            output_dir = Path(args[idx + 1])
    failed = output_dir / "ablation_failed_variants.csv"
    missing = output_dir / "ablation_missing_inputs.csv"
    if missing.exists() and not failed.exists():
        shutil.copy2(missing, failed)
    elif not failed.exists():
        failed.write_text("variant,missing_prediction_catalogue,status\n", encoding="utf-8")


if __name__ == "__main__":
    main()
