#!/usr/bin/env python
"""Unified preflight for formal DR1 ablation and baseline jobs."""

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

from lotss_association.utils import load_yaml
from lotss_association.validation.dr1_reference import DEFAULT_DR1_COMPONENT_CSV, load_dr1_component_catalogue


SOURCE_OUTPUT_ROOT = Path(os.environ.get("LOTSS_ASSOC_OUTPUT_ROOT", PROJECT_ROOT / "outputs" / "lotss_dr3_full"))
DEFAULT_MANIFEST = SOURCE_OUTPUT_ROOT / "manifests/lotss_dr3_fits_manifest.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/dr1_validation/hpc_submission"

HPC_SCRIPTS = {
    "ablation": PROJECT_ROOT / "scripts/hpc/submit_dr1_full_ablation_onejob.sh",
    "baseline": PROJECT_ROOT / "scripts/hpc/submit_dr1_full_baseline_onejob.sh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dr1-catalogue", default=str(DEFAULT_DR1_COMPONENT_CSV))
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--expected-ready-fields", type=int, default=1580, help="Expected number of ready fields; use 0 to skip the exact-count check.")
    parser.add_argument("--skip-write-check", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30, check=False)
        return proc.returncode, proc.stdout
    except Exception as exc:
        return 99, f"{exc.__class__.__name__}: {exc}"


def write_check(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        marker = path / ".preflight_write_test"
        marker.write_text("ok\n", encoding="utf-8")
        marker.unlink()
        return True, ""
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def path_exists(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and Path(text).exists()


def script_has_array(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return "#SBATCH --array" in text or "SLURM_ARRAY_TASK_ID" in text


def sbatch_count_per_script(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return len(re.findall(r"(^|\s)sbatch(\s|$)", text))


def expected_ablation_predictions(config_path: Path) -> list[Path]:
    config = load_yaml(config_path)
    paths = []
    for entry in (config.get("catalogues") or {}).values():
        value = entry.get("predictions")
        if value:
            path = Path(value)
            paths.append(path if path.is_absolute() else PROJECT_ROOT / path)
    return paths


def check_formal_implementation() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Ablation runner is production-capable if the onejob script calls run_lotss and evaluator.
    ablation_text = HPC_SCRIPTS["ablation"].read_text(encoding="utf-8")
    rows.append(
        {
            "check": "ablation_onejob_production_entry",
            "ok": "run_lotss_dr3_full.py" in ablation_text and "dr1_ablation_full_evaluate.py" in ablation_text,
            "detail": "A0-A10 one-job wrapper runs production association per variant and unified evaluator.",
        }
    )
    baseline_text = (PROJECT_ROOT / "scripts/validation/run_dr1_baselines.py").read_text(encoding="utf-8")
    baseline_ok = all(token in baseline_text for token in ["CORE_BASELINES", "_build_catalogues", "baseline_skipped_methods.csv", "chance_correction"])
    rows.append(
        {
            "check": "baseline_full_implementation",
            "ok": bool(baseline_ok),
            "detail": "run_dr1_baselines.py generates real B0-B4 prediction catalogues, writes B5-B7 skipped metadata, and runs unified DR1 bbox evaluation.",
        }
    )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    sinfo_rc, sinfo_out = run_command(["sinfo"])
    squeue_rc, squeue_out = run_command(["squeue"])
    idle_report = [
        "# HPC Idle Resource Report",
        "",
        "## sinfo",
        f"return_code={sinfo_rc}",
        "```",
        sinfo_out.strip(),
        "```",
        "",
        "## squeue",
        f"return_code={squeue_rc}",
        "```",
        squeue_out.strip(),
        "```",
        "",
    ]
    (output_dir / "hpc_idle_resource_report.txt").write_text("\n".join(idle_report), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    manifest = Path(args.manifest)
    if manifest.exists():
        frame = pd.read_csv(manifest)
        ready = frame.loc[~frame.get("needs_pybdsf", pd.Series(False, index=frame.index)).astype(str).str.lower().isin({"1", "true", "yes"})].copy()
        pybdsf_missing = int((~ready.get("pybdsf_catalog_path", pd.Series("", index=ready.index)).map(path_exists)).sum())
        image_exists = pd.Series(False, index=ready.index)
        for col in ["scratch_fits_path", "fits_path", "h5_path"]:
            if col in ready:
                image_exists |= ready[col].map(path_exists)
        image_missing = int((~image_exists).sum())
        expected_ok = len(ready) == int(args.expected_ready_fields) if int(args.expected_ready_fields) > 0 else True
        manifest_ok = expected_ok and pybdsf_missing == 0 and image_missing == 0
    else:
        ready = pd.DataFrame()
        pybdsf_missing = image_missing = int(args.expected_ready_fields)
        manifest_ok = False
    rows.append({"check": "manifest_expected_pybdsf_and_images", "ok": manifest_ok, "detail": f"manifest={manifest} ready={len(ready)} expected_ready_fields={args.expected_ready_fields} pybdsf_missing={pybdsf_missing} image_or_cutout_missing={image_missing}"})

    try:
        dr1, summary = load_dr1_component_catalogue(args.dr1_catalogue, print_summary=False, report_path=output_dir / "dr1_component_catalogue_report.txt")
        dr1_ok = bool(summary.get("n_valid_position", 0) > 0)
        source_ok = "dr1_source_id" in dr1 and dr1["dr1_source_id"].nunique(dropna=True) < len(dr1)
        rows.append({"check": "dr1_component_catalogue_readable", "ok": dr1_ok, "detail": json.dumps(summary, sort_keys=True)})
        rows.append({"check": "dr1_source_id_available", "ok": source_ok, "detail": f"source_id_column={summary.get('identified_source_id_column')} component_id_column={summary.get('identified_component_id_column')}"})
    except Exception as exc:
        rows.append({"check": "dr1_component_catalogue_readable", "ok": False, "detail": f"{exc.__class__.__name__}: {exc}"})
        rows.append({"check": "dr1_source_id_available", "ok": False, "detail": "DR1 catalogue could not be loaded."})

    output_paths = [
        PROJECT_ROOT / "outputs/dr1_validation/ablation_full_A0_A10",
        PROJECT_ROOT / "outputs/dr1_validation/baselines_full_B0_B7",
        output_dir,
    ]
    for path in output_paths:
        ok, detail = (True, "skipped") if args.skip_write_check else write_check(path)
        rows.append({"check": f"output_writable:{path}", "ok": ok, "detail": detail})

    prediction_paths = expected_ablation_predictions(PROJECT_ROOT / "configs/dr1_validation/ablation_full_A0_A10_catalogues.yaml")
    unique_ok = len(prediction_paths) == len(set(map(str, prediction_paths)))
    rows.append({"check": "ablation_expected_prediction_paths_unique", "ok": unique_ok and len(prediction_paths) == 11, "detail": f"n_paths={len(prediction_paths)} n_unique={len(set(map(str, prediction_paths)))}"})

    for name, path in HPC_SCRIPTS.items():
        exists = path.exists()
        rows.append({"check": f"{name}_script_exists", "ok": exists, "detail": str(path)})
        if exists:
            rows.append({"check": f"{name}_script_no_slurm_array", "ok": not script_has_array(path), "detail": str(path)})
            rows.append({"check": f"{name}_script_single_job_id", "ok": sbatch_count_per_script(path) == 0, "detail": "script contains no nested sbatch calls"})

    rows.extend(check_formal_implementation())
    hpc_ok = sinfo_rc == 0 and squeue_rc == 0 and bool(sinfo_out.strip())
    rows.append({"check": "hpc_resource_visibility", "ok": hpc_ok, "detail": "sinfo/squeue completed" if hpc_ok else "Could not reliably inspect HPC resources."})

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "full_experiment_preflight.csv", index=False)
    preflight_ok = bool(table["ok"].astype(bool).all())
    lines = [
        "# Full DR1 Experiment Preflight",
        "",
        f"preflight_ok: {preflight_ok}",
        f"hostname: {socket.gethostname()}",
        "",
        "## Checks",
        "",
        table.to_string(index=False),
        "",
        "## Submission Policy",
        "",
        "No jobs are submitted unless every check passes. Source-level validation uses DR1 component-only bbox-containment support; no DR1 optical IDs.",
        "",
    ]
    (output_dir / "full_experiment_preflight.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if not preflight_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
