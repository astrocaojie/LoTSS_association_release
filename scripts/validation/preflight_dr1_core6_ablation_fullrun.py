#!/usr/bin/env python
"""Preflight checks for the DR1 core-6 ablation full run."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from lofar_det_vsex.utils import load_yaml


CORE_VARIANTS = [
    "full_method",
    "no_ridge_continuity",
    "no_weak_edge_anti_chaining",
    "no_artifact_penalties",
    "no_host_support",
    "no_lobe_peak_host_contradiction",
]

FORBIDDEN_VARIANT_TOKENS = {
    "A6",
    "A7",
    "A8",
    "A9",
    "A10",
    "no_pa_alignment",
    "no_ellipse_overlap",
    "no_multithreshold_contour_connectivity",
    "no_stage2_relative_scale_constraints",
    "no_stage2_endpoint_filtering",
}

DEFAULT_BASE_OUTPUT_ROOT = Path(os.environ.get("LOTSS_ASSOC_CORE6_OUTPUT_ROOT", PROJECT_ROOT / "outputs" / "dr1_core6_ablation"))
DEFAULT_SOURCE_OUTPUT_ROOT = Path(os.environ.get("LOTSS_ASSOC_OUTPUT_ROOT", PROJECT_ROOT / "outputs" / "lotss_dr3_full"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit-script", default=str(PROJECT_ROOT / "scripts/hpc/submit_dr1_core6_ablation_full_hpc1_4.sh"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/dr1_validation/core_ablation_fullrun_catalogues.yaml"))
    parser.add_argument("--base-output-root", default=os.environ.get("BASE_OUTPUT_ROOT", str(DEFAULT_BASE_OUTPUT_ROOT)))
    parser.add_argument("--source-output-root", default=os.environ.get("SOURCE_OUTPUT_ROOT", str(DEFAULT_SOURCE_OUTPUT_ROOT)))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs/dr1_validation/ablation_core6/preflight"))
    parser.add_argument("--sample-rows", type=int, default=5)
    parser.add_argument("--skip-write-check", action="store_true")
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _script_variants(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"VARIANTS=\(([^)]*)\)", text)
    if not match:
        return []
    return [item.strip().strip("\"'") for item in match.group(1).split() if item.strip()]


def _config_variants(config: dict[str, Any]) -> list[str]:
    return [str(row.get("name")) for row in config.get("variants", []) if row.get("name")]


def _catalogue_path(config: dict[str, Any], variant: str) -> Path | None:
    entry = (config.get("catalogues") or {}).get(variant, {}) or {}
    value = entry.get("predictions")
    return _resolve(value) if value else None


def _write_check(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        marker = path / ".dr1_core6_preflight_write_test"
        marker.write_text("ok\n", encoding="utf-8")
        marker.unlink()
        return True, ""
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def _path_exists(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and Path(text).exists()


def _manifest_checks(manifest_path: Path, sample_rows: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not manifest_path.exists():
        return {"manifest_exists": False, "manifest_path": str(manifest_path)}, []
    manifest = pd.read_csv(manifest_path)
    ready = manifest.copy()
    if "needs_pybdsf" in ready:
        needs = ready["needs_pybdsf"].astype(str).str.lower().isin({"1", "true", "yes"})
        ready = ready.loc[~needs].copy()
    pybdsf_col = "pybdsf_catalog_path" if "pybdsf_catalog_path" in ready else None
    scratch_col = "scratch_fits_path" if "scratch_fits_path" in ready else None
    fits_col = "fits_path" if "fits_path" in ready else None
    h5_col = "h5_path" if "h5_path" in ready else None
    image_exists = pd.Series(False, index=ready.index)
    h5_exists = ready[h5_col].map(_path_exists) if h5_col else pd.Series(False, index=ready.index)
    if scratch_col:
        image_exists |= ready[scratch_col].map(_path_exists)
    if fits_col:
        image_exists |= ready[fits_col].map(_path_exists)
    image_exists |= h5_exists
    pybdsf_exists = ready[pybdsf_col].map(_path_exists) if pybdsf_col else pd.Series(False, index=ready.index)
    samples: list[dict[str, Any]] = []
    keep_cols = [
        col
        for col in ["field_name", "pybdsf_catalog_path", "scratch_fits_path", "fits_path", "h5_path"]
        if col in ready.columns
    ]
    for _, row in ready.head(sample_rows).iterrows():
        samples.append({col: str(row.get(col, "")) for col in keep_cols})
    summary = {
        "manifest_exists": True,
        "manifest_path": str(manifest_path),
        "manifest_rows": int(len(manifest)),
        "ready_rows": int(len(ready)),
        "pybdsf_catalogues_present": int(pybdsf_exists.sum()),
        "pybdsf_catalogues_missing": int((~pybdsf_exists).sum()) if pybdsf_col else int(len(ready)),
        "image_or_cutout_paths_present": int(image_exists.sum()),
        "image_or_cutout_paths_missing": int((~image_exists).sum()),
        "h5_cutout_paths_present": int(h5_exists.sum()) if h5_col else 0,
        "h5_cutout_paths_missing": int((~h5_exists).sum()) if h5_col else int(len(ready)),
    }
    return summary, samples


def main() -> None:
    args = parse_args()
    submit_script = _resolve(args.submit_script)
    config_path = _resolve(args.config)
    base_output_root = Path(args.base_output_root)
    source_output_root = Path(args.source_output_root)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(config_path)
    script_variants = _script_variants(submit_script)
    config_variants = _config_variants(config)
    script_text = submit_script.read_text(encoding="utf-8") if submit_script.exists() else ""

    manifest_path = source_output_root / "manifests" / "lotss_dr3_fits_manifest.csv"
    manifest_summary, manifest_samples = _manifest_checks(manifest_path, args.sample_rows)

    variant_rows: list[dict[str, Any]] = []
    all_ok = True
    for variant in CORE_VARIANTS:
        variant_root = base_output_root / variant
        config_dir = variant_root / "config"
        log_dir = variant_root / "logs"
        prediction_dir = variant_root / "dr1_validation"
        expected_prediction = _catalogue_path(config, variant)
        derived_prediction = prediction_dir / f"{variant}_predictions.csv.gz"
        write_targets = [base_output_root / "logs", variant_root, config_dir, log_dir, prediction_dir]
        write_ok = True
        write_errors = []
        if not args.skip_write_check:
            for target in write_targets:
                ok, err = _write_check(target)
                write_ok &= ok
                if err:
                    write_errors.append(f"{target}: {err}")
        evaluator_identifies = expected_prediction is not None
        prediction_path_matches_submit = bool(expected_prediction and expected_prediction == derived_prediction)
        row = {
            "variant": variant,
            "in_submit_script": variant in script_variants,
            "in_config": variant in config_variants,
            "output_root": str(variant_root),
            "config_copy_path": str(config_dir / f"real_lotss_conservative_{variant}.yaml"),
            "log_dir": str(log_dir),
            "input_manifest_path": str(manifest_path),
            "input_pybdsf_root": str(source_output_root / "pybdsf" / "raw"),
            "input_processed_pybdsf_root": str(source_output_root / "pybdsf" / "processed"),
            "image_cutout_manifest_path": str(manifest_path),
            "expected_prediction_catalogue": str(expected_prediction) if expected_prediction else "",
            "derived_prediction_catalogue": str(derived_prediction),
            "prediction_path_matches_submit": prediction_path_matches_submit,
            "expected_prediction_exists_now": bool(expected_prediction and expected_prediction.exists()),
            "evaluator_identifies_expected_catalogue": evaluator_identifies,
            "output_dirs_writable": write_ok,
            "write_check_errors": "; ".join(write_errors),
            "will_overwrite_full_method_formal": "full_method_formal" in str(variant_root),
            "metadata_path": str(variant_root / f"run_metadata_{variant}.json"),
        }
        variant_ok = (
            row["in_submit_script"]
            and row["in_config"]
            and prediction_path_matches_submit
            and evaluator_identifies
            and write_ok
            and not row["will_overwrite_full_method_formal"]
        )
        row["preflight_ok"] = bool(variant_ok)
        all_ok &= bool(variant_ok)
        variant_rows.append(row)

    forbidden_in_script = sorted(token for token in FORBIDDEN_VARIANT_TOKENS if token in script_text or token in script_variants or token in config_variants)
    no_slurm_array = "#SBATCH --array" not in script_text and "SLURM_ARRAY_TASK_ID" not in script_text
    variant_set_ok = script_variants == CORE_VARIANTS and config_variants == CORE_VARIANTS and not forbidden_in_script and no_slurm_array
    manifest_ok = bool(
        manifest_summary.get("manifest_exists")
        and manifest_summary.get("pybdsf_catalogues_missing", 1) == 0
        and manifest_summary.get("image_or_cutout_paths_missing", 1) == 0
    )
    all_ok &= variant_set_ok and manifest_ok

    variant_table = pd.DataFrame(variant_rows)
    variant_table.to_csv(output_dir / "core6_ablation_fullrun_preflight.csv", index=False)
    pd.DataFrame(manifest_samples).to_csv(output_dir / "core6_ablation_fullrun_manifest_samples.csv", index=False)

    metadata = {
        "preflight_ok": bool(all_ok),
        "variant_set_ok": bool(variant_set_ok),
        "manifest_ok": bool(manifest_ok),
        "script_variants": script_variants,
        "config_variants": config_variants,
        "core_variants": CORE_VARIANTS,
        "forbidden_variant_tokens_found": forbidden_in_script,
        "no_slurm_array": bool(no_slurm_array),
        "submit_script": str(submit_script),
        "config": str(config_path),
        "base_output_root": str(base_output_root),
        "source_output_root": str(source_output_root),
        "manifest_summary": manifest_summary,
        "git_commit": git_commit(),
        "hostname": socket.gethostname(),
        "skip_write_check": bool(args.skip_write_check),
    }
    (output_dir / "core6_ablation_fullrun_preflight.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    report_lines = [
        "# DR1 Core6 Ablation Full-Run Preflight",
        "",
        f"preflight_ok: {all_ok}",
        f"variant_set_ok: {variant_set_ok}",
        f"manifest_ok: {manifest_ok}",
        f"submit_script_variants: {script_variants}",
        f"config_variants: {config_variants}",
        f"forbidden_variant_tokens_found: {forbidden_in_script}",
        f"no_slurm_array: {no_slurm_array}",
        "",
        "## Manifest",
        "",
        json.dumps(manifest_summary, indent=2, sort_keys=True),
        "",
        "## Variant Paths",
        "",
        variant_table.to_string(index=False),
        "",
        "## Sample Input Paths",
        "",
        pd.DataFrame(manifest_samples).to_string(index=False) if manifest_samples else "No manifest samples available.",
        "",
    ]
    (output_dir / "core6_ablation_fullrun_preflight.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
