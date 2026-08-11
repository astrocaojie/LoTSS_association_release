#!/usr/bin/env python
"""Run one real DR1 production parent-linking ablation shard using the production field runner."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import run_lotss_dr3_full as full  # noqa: E402

DEFAULT_DR3_OUTPUT_ROOT = Path(os.environ.get("LOTSS_ASSOC_OFFICIAL_GAUSSIAN_ROOT", PROJECT_ROOT / "outputs" / "lotss_dr3_official_gaussian_catalogs"))

FEATURE_KEYS = [
    "use_multithreshold_contour",
    "use_ridge_continuity",
    "use_ellipse_overlap",
    "use_pa_alignment",
    "use_weak_edge_anti_chaining",
    "use_artifact_penalties_layer1",
    "use_artifact_penalties_layer2",
    "use_midpoint_host_support",
    "use_lobe_peak_host_contradiction",
    "use_stage2_relative_scale_constraints",
    "use_stage2_endpoint_filtering",
]

def file_sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"

def scientific_config_hash(cfg: dict[str, Any]) -> str:
    import hashlib
    science = {k:v for k,v in cfg.items() if k not in {"output_dir", "dr1_validation"}}
    return hashlib.sha256(json.dumps(science, sort_keys=True, default=str).encode()).hexdigest()

def active_features(cfg: dict[str, Any]) -> dict[str, bool]:
    ab = cfg.get("ablation", {}) or {}
    return {key: bool(ab.get(key, True)) for key in FEATURE_KEYS}

def count_output_rows(output_dir: Path) -> dict[str, int]:
    assoc = output_dir / "association_outputs"
    out: dict[str, int] = {}
    for stem in ["local_components", "local_groups", "merged_components", "parent_edges_debug", "parent_candidates", "diagnostics"]:
        total = 0
        for path in assoc.glob(f"*_{stem}.csv"):
            try:
                total += len(pd.read_csv(path))
            except Exception:
                pass
        out[f"n_{stem}"] = int(total)
    return out


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _read_official_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Official production parent-linking manifest missing: {path}")
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"file_id", "pybdsf_catalog_path", "needs_pybdsf"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Official manifest missing columns {missing}: {path}")
    return frame


def _selected_fields(shard_manifest: Path, shard_id: str, max_fields: int | None) -> list[str]:
    shards = pd.read_csv(shard_manifest, dtype=str).fillna("")
    mine = shards[shards["shard_id"].astype(str) == str(shard_id)].copy()
    if mine.empty:
        raise ValueError(f"Shard {shard_id} not found in {shard_manifest}")
    ids = []
    for value in mine.get("cutout_id", pd.Series(dtype=str)).astype(str):
        if value and value not in ids:
            ids.append(value)
    if max_fields is not None:
        ids = ids[: int(max_fields)]
    return ids


def _copy_partials(association_root: Path, output_dir: Path) -> dict[str, str]:
    partials = association_root / "association" / "catalogs" / "partials"
    copied: dict[str, str] = {}
    if not partials.exists():
        return copied
    dest = output_dir / "association_outputs"
    dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(partials.glob("*")):
        if path.is_file():
            target = dest / path.name
            shutil.copy2(path, target)
            copied[path.name] = str(target)
    return copied


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--variant-name", required=True)
    ap.add_argument("--shard-id", required=True)
    ap.add_argument("--shard-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--official-output-root", type=Path, default=DEFAULT_DR3_OUTPUT_ROOT)
    ap.add_argument("--input-format", choices=["fits", "h5"], default="fits")
    ap.add_argument("--cutout-limit", type=int, default=None, help="Smoke limit passed to the production field runner.")
    ap.add_argument("--max-fields", type=int, default=None, help="Limit number of fields in this shard for smoke tests.")
    ap.add_argument("--query-wise-host", action="store_true", help="Allow live WISE host queries. Default is offline cache only.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    official_manifest_path = args.official_output_root / "manifests" / "lotss_dr3_fits_manifest.csv"
    official = _read_official_manifest(official_manifest_path)
    shard_manifest_path = Path(args.shard_manifest)
    selected = _selected_fields(shard_manifest_path, args.shard_id, args.max_fields)
    rows = official[official["file_id"].astype(str).isin(selected)].copy()
    if rows.empty:
        raise SystemExit(f"No official manifest rows found for selected fields: {selected}")
    ready = rows[rows["needs_pybdsf"].astype(str).str.lower().isin(["false", "0", "no"])]
    if len(ready) != len(rows):
        missing = rows.loc[~rows.index.isin(ready.index), ["file_id", "needs_pybdsf", "pybdsf_catalog_path"]].to_dict(orient="records")
        raise SystemExit(f"Selected fields include PyBDSF-not-ready rows: {missing}")
    rows.to_csv(out / "resolved_field_manifest.csv", index=False)

    association_root = out / "association_output"
    proc_args = SimpleNamespace(
        output_root=association_root,
        config=Path(args.config),
        query_wise_host=bool(args.query_wise_host),
        limit=args.cutout_limit,
        max_host_queries_per_field=1000,
        debug=bool(args.debug),
        debug_sample_figures=0,
        save_all_parent_zoom=False,
    )
    figure_budget = {"remaining": 0}
    statuses: list[dict[str, Any]] = []
    for _, row in rows.iterrows():
        field_id = str(row["file_id"])
        try:
            rec = full._process_field(row, proc_args, cfg, args.input_format, figure_budget)
            rec["real_association_run"] = True
            rec["variant_name"] = args.variant_name
            rec["shard_id"] = args.shard_id
            statuses.append(rec)
        except Exception as exc:
            tb = traceback.format_exc()
            (out / f"traceback_{field_id}.txt").write_text(tb, encoding="utf-8")
            statuses.append({"file_id": field_id, "status": "failed", "message": str(exc), "real_association_run": False, "variant_name": args.variant_name, "shard_id": args.shard_id})
            raise
    status = pd.DataFrame(statuses)
    status.to_csv(out / "field_status.csv", index=False)
    copied = _copy_partials(association_root, out)
    features = active_features(cfg)
    disabled = [k for k, v in features.items() if not v]
    output_counts = count_output_rows(out)
    meta = {
        "real_association_run": True,
        "placeholder_adapter_output": False,
        "variant_name": args.variant_name,
        "shard_id": args.shard_id,
        "input_format": args.input_format,
        "cutout_limit": args.cutout_limit,
        "max_fields": args.max_fields,
        "variant_id": cfg.get("dr1_validation", {}).get("variant_id", ""),
        "active_features": features,
        "disabled_features": disabled,
        "scientific_config_hash": scientific_config_hash(cfg),
        "config_hash": file_sha256(cfg_path),
        "input_manifest_hash": file_sha256(official_manifest_path),
        "shard_manifest_hash": file_sha256(shard_manifest_path),
        "official_manifest": str(official_manifest_path),
        "n_fields": int(len(rows)),
        "n_success_fields": int((status["status"].astype(str) == "done").sum()) if not status.empty else 0,
        "runtime_seconds": float(time.time() - started),
        "copied_partial_outputs": copied,
        "pybdsf_reused": True,
        "pybdsf_rerun": False,
        **output_counts,
    }
    write_json(out / "run_metadata.json", meta)
    pd.DataFrame(columns=["merged_source_id", "member_gaussian_ids", "variant", "shard_id", "real_association_run"]).to_csv(out / "association_outputs.csv", index=False)
    write_json(out / "resource_usage.json", {"runtime_seconds": meta["runtime_seconds"], "peak_memory_mb": None})


if __name__ == "__main__":
    main()
