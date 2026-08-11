#!/usr/bin/env python
"""Run production parent-linking association for fields whose PyBDSF catalogs are already ready.

This is a companion drain for the full LoTSS DR3 job: it does not run PyBDSF
and it does not alter production parent-linking scoring. It simply starts association work early for
fields that already have sane PyBDSF Gaussian catalogs.
"""

from __future__ import annotations

import argparse
import fcntl
import traceback
from pathlib import Path

import pandas as pd

import run_lotss_dr3_full as full
from lotss_dr3_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_H5_ROOT,
    DEFAULT_ORIGINAL_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    default_pybdsf_catalog_path,
    ensure_output_dirs,
    read_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--original-data-root", type=Path, default=DEFAULT_ORIGINAL_DATA_ROOT)
    parser.add_argument("--h5-root", type=Path, default=DEFAULT_H5_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=full.PROJECT_ROOT / "configs" / "real_lotss_conservative.yaml")
    parser.add_argument("--input-format", choices=["fits", "h5"], default="fits")
    parser.add_argument("--query-wise-host", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-fields", type=int, default=300)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--stop-pybdsf-remaining-lte", type=int, default=128)
    parser.add_argument("--max-host-queries-per-field", type=int, default=1000)
    parser.add_argument("--debug-sample-figures", type=int, default=0)
    parser.add_argument("--save-all-parent-zoom", action="store_true")
    parser.add_argument("--merge-final", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _replace_record(records: list[dict], rec: dict) -> list[dict]:
    field_id = str(rec.get("file_id", ""))
    return [row for row in records if str(row.get("file_id", "")) != field_id] + [rec]


def _write_status_locked(status_path: Path, rec: dict) -> None:
    lock_path = status_path.with_suffix(status_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            status = full._read_status(status_path)
            records = status.to_dict(orient="records") if not status.empty else []
            full._write_field_status(status_path, _replace_record(records, rec))
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _ready_manifest(args: argparse.Namespace, dirs: object) -> tuple[pd.DataFrame, int]:
    manifest_path = dirs.manifests / "lotss_dr3_fits_manifest.csv"
    manifest = read_manifest(manifest_path)
    if manifest.empty:
        raise SystemExit(f"Manifest is empty or missing: {manifest_path}")

    pybdsf_status_path = dirs.checkpoints / "pybdsf_status.csv"
    if pybdsf_status_path.exists():
        status = pd.read_csv(pybdsf_status_path, dtype=str).fillna("")
        done = status[status.get("status", "").astype(str) == "done"].copy()
        if not done.empty:
            done = done.drop_duplicates("file_id", keep="last").set_index("file_id")
            for idx, row in manifest.iterrows():
                field_id = str(row["file_id"])
                if field_id not in done.index:
                    continue
                catalog = str(done.loc[field_id].get("pybdsf_catalog_path", ""))
                if not catalog:
                    catalog = str(default_pybdsf_catalog_path(dirs.pybdsf_raw, field_id))
                manifest.loc[idx, "pybdsf_catalog_path"] = catalog
                manifest.loc[idx, "has_existing_pybdsf_catalog"] = "True"
                manifest.loc[idx, "needs_pybdsf"] = "False"
                manifest.loc[idx, "status"] = "ready"

    if args.input_format == "h5":
        ready = manifest[(manifest["needs_pybdsf"].astype(str) == "False") & (manifest["h5_path"].astype(str) != "")].copy()
    else:
        ready = manifest[manifest["needs_pybdsf"].astype(str) == "False"].copy()
    remaining = max(0, len(manifest) - len(ready))
    return ready, remaining


def main() -> None:
    args = parse_args()
    dirs = ensure_output_dirs(args.output_root)
    logger = full.setup_logging(args.debug, dirs.association_logs / "ready_association.log")

    ready, remaining_pybdsf = _ready_manifest(args, dirs)
    if remaining_pybdsf <= args.stop_pybdsf_remaining_lte:
        logger.info("Not starting ready association: remaining_pybdsf=%d", remaining_pybdsf)
        return

    status_path = dirs.checkpoints / "association_field_status.csv"
    status = full._read_status(status_path)
    done = set(status.loc[status["status"].astype(str) == "done", "file_id"].astype(str)) if args.resume and not status.empty else set()
    todo = ready[~ready["file_id"].astype(str).isin(done)].copy()
    num_shards = max(1, int(args.num_shards))
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit(f"Invalid shard index {shard_index} for {num_shards} shards")
    if num_shards > 1:
        todo = todo.iloc[shard_index::num_shards].copy()
    todo = todo.head(max(0, int(args.max_fields)))
    logger.info(
        "Ready association drain fields=%d ready=%d done=%d remaining_pybdsf=%d shard=%d/%d",
        len(todo),
        len(ready),
        len(done),
        remaining_pybdsf,
        shard_index,
        num_shards,
    )
    if todo.empty:
        return

    config = full.load_yaml(args.config)
    figure_budget = {"remaining": int(args.debug_sample_figures)}

    # Namespace passed into the original field processor. Keep limit=None so a
    # selected field is processed completely, not as a smoke-test cutout subset.
    proc_args = argparse.Namespace(
        output_root=args.output_root,
        config=args.config,
        query_wise_host=args.query_wise_host,
        limit=None,
        max_host_queries_per_field=args.max_host_queries_per_field,
        debug=args.debug,
        debug_sample_figures=args.debug_sample_figures,
        save_all_parent_zoom=args.save_all_parent_zoom,
    )

    for _, row in todo.iterrows():
        _, remaining_now = _ready_manifest(args, dirs)
        if remaining_now <= args.stop_pybdsf_remaining_lte:
            logger.info("Stopping ready association drain: remaining_pybdsf=%d", remaining_now)
            break

        field_id = str(row["file_id"])
        if field_id in done:
            continue
        started = full._now()
        try:
            rec = full._process_field(row, proc_args, config, args.input_format, figure_budget)
            rec["started_at"] = started
        except Exception as exc:
            message = traceback.format_exc() if args.debug else str(exc)
            logger.error("Field failed %s: %s", field_id, message)
            rec = {
                "file_id": field_id,
                "status": "failed",
                "n_cutouts": 0,
                "n_local_groups": 0,
                "n_parent_candidates": 0,
                "n_merged_rows": 0,
                "n_host_candidates": 0,
                "message": message[:4000],
                "started_at": started,
                "ended_at": full._now(),
            }
        _write_status_locked(status_path, rec)
        done.add(field_id)

    if args.merge_final:
        final = full._merge_final_outputs(args.output_root)
        logger.info("Ready association drain done: %s", final)
    else:
        logger.info("Ready association drain done")


if __name__ == "__main__":
    main()
