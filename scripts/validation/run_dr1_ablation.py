#!/usr/bin/env python
"""DR1 component-reference ablation runner/preflight.

The full production run is expected to execute the association pipeline per
variant, then call the unified DR1 support evaluator on each produced source
catalogue.  This script provides the reproducible variant inventory and a
smoke/preflight mode that writes the expected table skeletons without launching
large-scale jobs.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from lotss_association.utils import load_yaml


SUPPORT_COLUMNS = [
    "variant",
    "sample",
    "flux_cut_jy",
    "n_in_dr1_footprint",
    "n_supported_by_dr1_component",
    "support_rate",
    "binomial_or_wilson_95ci_low",
    "binomial_or_wilson_95ci_high",
]
DIAG_COLUMNS = [
    "variant",
    "n_components",
    "n_candidate_pairs",
    "n_strong_edges",
    "n_weak_edges",
    "n_rejected_edges",
    "n_singletons",
    "n_multi_component_groups",
    "max_group_size",
    "multi_component_fraction",
    "adjusted_rand_index_against_full_method",
    "n_candidate_endpoint_groups",
    "n_candidate_parent_pairs",
    "n_pass_radio_geometry",
    "n_pass_host_support",
    "n_peak_host_contradiction",
    "n_parent_high",
    "n_parent_medium",
    "n_parent_suspicious",
    "n_rejected",
    "n_conflict_resolved",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/dr1_validation/ablation_variants.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--variant", default=None, help="Optional variant name or ID to run/preflight.")
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _variant_rows(config: dict[str, Any], selected: str | None) -> list[dict[str, Any]]:
    rows = []
    raw_variants = config.get("variants") or {}
    if isinstance(raw_variants, dict):
        iterator = raw_variants.items()
    else:
        iterator = ((str(payload.get("name", f"A{idx}")), payload) for idx, payload in enumerate(raw_variants))
    for variant_id, payload in iterator:
        name = str(payload.get("name", variant_id))
        if selected and selected not in {variant_id, name}:
            continue
        rows.append({"variant_id": variant_id, "variant": name, "ablation": payload.get("ablation", {}) or {}})
    return rows


def _write_latex(csv_path: Path, tex_path: Path) -> None:
    frame = pd.read_csv(csv_path)
    frame.to_latex(tex_path, index=False, float_format="%.4f")


def main() -> None:
    args = parse_args()
    t0 = perf_counter()
    config = load_yaml(args.config)
    output_dir = Path(args.output_dir or config.get("outputs", {}).get("root", PROJECT_ROOT / "outputs/dr1_validation/ablation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _variant_rows(config, args.variant)
    if not rows:
        raise SystemExit(f"No ablation variants selected by {args.variant!r}")

    pd.DataFrame(rows).to_csv(output_dir / "ablation_variant_inventory.csv", index=False)
    support = pd.DataFrame(columns=SUPPORT_COLUMNS)
    diagnostics = pd.DataFrame(columns=DIAG_COLUMNS)
    diff = pd.DataFrame(
        columns=[
            "variant",
            "sample",
            "flux_cut_jy",
            "delta_n_in_footprint",
            "delta_n_supported",
            "delta_support_rate",
            "number_of_full_sources_lost",
            "number_of_new_sources",
            "number_of_sources_with_changed_bbox",
            "number_of_possible_overmerged_cases",
            "number_of_possible_split_cases",
        ]
    )
    support.to_csv(output_dir / "ablation_support_summary.csv", index=False)
    diagnostics.to_csv(output_dir / "ablation_diagnostics_summary.csv", index=False)
    diff.to_csv(output_dir / "ablation_difference_vs_full_method.csv", index=False)
    _write_latex(output_dir / "ablation_support_summary.csv", output_dir / "ablation_support_summary.tex")
    _write_latex(output_dir / "ablation_diagnostics_summary.csv", output_dir / "ablation_diagnostics_summary.tex")
    metadata = {
        "mode": args.mode,
        "note": "Smoke/preflight writes inventories and expected table schemas. Full mode should run production association jobs before evaluation.",
        "config": str(args.config),
        "git_commit": git_commit(),
        "hostname": platform.node(),
        "runtime_seconds": round(perf_counter() - t0, 3),
        "reference": config.get("reference", {}),
        "n_variants": len(rows),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote ablation DR1 validation preflight outputs to {output_dir}")


if __name__ == "__main__":
    main()
