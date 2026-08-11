#!/usr/bin/env python
"""Run PyBDSF only for LoTSS DR3 fields missing sane Gaussian catalogs."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from lotss_dr3_common import (
    DEFAULT_OUTPUT_ROOT,
    bool_text,
    default_pybdsf_catalog_path,
    ensure_output_dirs,
    pybdsf_catalog_sane,
    read_manifest,
    write_manifest,
)

PYBDSF_FREQUENCY_HZ = 144000000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--thresh-pix", type=float, default=5.0)
    parser.add_argument("--thresh-isl", type=float, default=3.0)
    parser.add_argument("--rms-box", type=int, nargs=2, default=(160, 40))
    parser.add_argument("--pybdsf-frequency-hz", type=float, default=PYBDSF_FREQUENCY_HZ)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _write_status(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def _read_status(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def _latest_status(path: Path) -> pd.DataFrame:
    status = _read_status(path)
    if status.empty or "file_id" not in status:
        return pd.DataFrame()
    return status.drop_duplicates("file_id", keep="last")


def _replace_record(records: list[dict[str, Any]], rec: dict[str, Any]) -> list[dict[str, Any]]:
    field_id = str(rec.get("file_id", ""))
    return [row for row in records if str(row.get("file_id", "")) != field_id] + [rec]


def _status_lock(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


def _write_status_locked(path: Path, rec: dict[str, Any]) -> None:
    lock_path = _status_lock(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            current = _read_status(path)
            records = current.to_dict(orient="records") if not current.empty else []
            _write_status(path, _replace_record(records, rec))
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _sync_manifest_and_failed_report(manifest_path: Path, status_path: Path, dirs: Any) -> None:
    lock_path = _status_lock(status_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            manifest = read_manifest(manifest_path)
            latest = _latest_status(status_path)
            if not latest.empty:
                done = latest[latest["status"].astype(str) == "done"].copy()
                success_map = done.set_index("file_id") if not done.empty else pd.DataFrame()
                for idx, row in manifest.iterrows():
                    file_id = str(row["file_id"])
                    if file_id not in success_map.index:
                        continue
                    manifest.loc[idx, "pybdsf_catalog_path"] = str(success_map.loc[file_id, "pybdsf_catalog_path"])
                    manifest.loc[idx, "has_existing_pybdsf_catalog"] = "True"
                    manifest.loc[idx, "needs_pybdsf"] = "False"
                    manifest.loc[idx, "status"] = "ready"
                failed = latest[latest["status"].astype(str) != "done"].copy()
            else:
                failed = pd.DataFrame()
            write_manifest(manifest, manifest_path)
            failed_report = failed.copy()
            for col in ["fits_path", "error_type", "error_message", "frequency_hz", "status"]:
                if col not in failed_report:
                    failed_report[col] = ""
            failed_report[["fits_path", "error_type", "error_message", "frequency_hz", "status"]].to_csv(
                dirs.reports / "pybdsf_failed_files.csv",
                index=False,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _run_one(row: dict[str, Any], raw_root: str, log_root: str, opts: dict[str, Any]) -> dict[str, Any]:
    field_id = str(row["file_id"])
    scratch = Path(str(row.get("scratch_fits_path", "")))
    original = Path(str(row.get("fits_path", "")))
    fits_path = scratch if scratch.exists() else original
    out_dir = Path(raw_root) / field_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_root) / f"{field_id}.pybdsf.log"
    gaul_path = default_pybdsf_catalog_path(raw_root, field_id)
    srl_path = out_dir / f"{field_id}.pybdsf.srl.fits"
    started = datetime.now(timezone.utc).isoformat()
    frequency_hz = float(opts.get("pybdsf_frequency_hz", PYBDSF_FREQUENCY_HZ))
    if not opts.get("force", False):
        ok, n_rows, reason = pybdsf_catalog_sane(gaul_path)
        if ok:
            return {
                "file_id": field_id,
                "status": "done",
                "attempt": 0,
                "fits_path_used": str(fits_path),
                "fits_path": str(fits_path),
                "pybdsf_catalog_path": str(gaul_path),
                "n_rows": n_rows,
                "frequency_hz": frequency_hz,
                "error_type": "",
                "error_message": "",
                "started_at": started,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "message": f"existing:{reason}",
            }
    if not fits_path.exists():
        return {
            "file_id": field_id,
            "status": "failed",
            "attempt": 0,
            "fits_path_used": str(fits_path),
            "fits_path": str(fits_path),
            "pybdsf_catalog_path": str(gaul_path),
            "n_rows": 0,
            "frequency_hz": frequency_hz,
            "error_type": "FileNotFoundError",
            "error_message": "fits_missing",
            "started_at": started,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "message": "fits_missing",
        }

    last_message = ""
    last_error_type = ""
    last_error_message = ""
    max_retries = int(opts.get("max_retries", 1))
    for attempt in range(max_retries + 1):
        try:
            import bdsf

            cwd = os.getcwd()
            os.chdir(out_dir)
            try:
                with log_path.open("a", encoding="utf-8") as log_handle:
                    log_handle.write(f"\n# {datetime.now(timezone.utc).isoformat()} field={field_id} attempt={attempt}\n")
                    log_handle.write(f"PyBDSF frequency_hz = {frequency_hz}\n")
                    with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                        image = bdsf.process_image(
                            str(fits_path),
                            thresh_pix=float(opts.get("thresh_pix", 5.0)),
                            thresh_isl=float(opts.get("thresh_isl", 3.0)),
                            rms_box=tuple(opts.get("rms_box", (160, 40))),
                            adaptive_rms_box=True,
                            frequency=frequency_hz,
                            quiet=True,
                        )
                        image.write_catalog(outfile=str(gaul_path), catalog_type="gaul", format="fits", clobber=True)
                        try:
                            image.write_catalog(outfile=str(srl_path), catalog_type="srl", format="fits", clobber=True)
                        except Exception as exc:
                            log_handle.write(f"srl_write_failed: {type(exc).__name__}: {exc}\n")
            finally:
                os.chdir(cwd)
            ok, n_rows, reason = pybdsf_catalog_sane(gaul_path)
            if ok:
                return {
                    "file_id": field_id,
                    "status": "done",
                    "attempt": attempt,
                    "fits_path_used": str(fits_path),
                    "fits_path": str(fits_path),
                    "pybdsf_catalog_path": str(gaul_path),
                    "n_rows": n_rows,
                    "frequency_hz": frequency_hz,
                    "error_type": "",
                    "error_message": "",
                    "started_at": started,
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                    "message": reason,
                }
            last_message = reason
            last_error_type = "CatalogSanityError"
            last_error_message = reason
        except Exception as exc:
            last_error_type = type(exc).__name__
            last_error_message = str(exc)
            last_message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            try:
                with log_path.open("a", encoding="utf-8") as log_handle:
                    log_handle.write(last_message + "\n")
            except Exception:
                pass
    return {
        "file_id": field_id,
        "status": "failed",
        "attempt": max_retries,
        "fits_path_used": str(fits_path),
        "fits_path": str(fits_path),
        "pybdsf_catalog_path": str(gaul_path),
        "n_rows": 0,
        "frequency_hz": frequency_hz,
        "error_type": last_error_type,
        "error_message": last_error_message,
        "started_at": started,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "message": last_message[:4000],
    }


def main() -> None:
    args = parse_args()
    dirs = ensure_output_dirs(args.output_root)
    manifest_path = args.manifest or dirs.manifests / "lotss_dr3_fits_manifest.csv"
    manifest = read_manifest(manifest_path)
    if manifest.empty:
        raise SystemExit(f"Manifest is empty or missing: {manifest_path}")
    status_path = dirs.checkpoints / "pybdsf_status.csv"
    latest = _latest_status(status_path)
    done_existing = set()
    if not latest.empty and "status" in latest:
        done_existing = set(latest.loc[latest["status"].astype(str) == "done", "file_id"].astype(str))
    work = manifest[manifest["needs_pybdsf"].map(bool_text) | args.force].copy()
    if done_existing and not args.force:
        work = work[~work["file_id"].astype(str).isin(done_existing)].copy()
    num_shards = max(1, int(args.num_shards))
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= num_shards:
        raise SystemExit(f"Invalid shard index {shard_index} for {num_shards} shards")
    if num_shards > 1:
        work = work.iloc[shard_index::num_shards].copy()
    if args.limit is not None:
        work = work.head(args.limit)
    records: list[dict[str, Any]] = []
    opts = {
        "force": args.force,
        "max_retries": args.max_retries,
        "thresh_pix": args.thresh_pix,
        "thresh_isl": args.thresh_isl,
        "rms_box": tuple(args.rms_box),
        "pybdsf_frequency_hz": float(args.pybdsf_frequency_hz),
    }
    print(f"PyBDSF frequency_hz = {float(args.pybdsf_frequency_hz)}")
    print(f"shard={shard_index}/{num_shards} selected_fields={len(work)}")
    if work.empty:
        print("No missing PyBDSF catalogs.")
        return
    if args.dry_run:
        print("Dry run: first selected fields")
        for _, row in work.head(20).iterrows():
            field_id = str(row["file_id"])
            print(f"{field_id}\tfits={row.get('scratch_fits_path') or row.get('fits_path')}\tout={default_pybdsf_catalog_path(dirs.pybdsf_raw, field_id)}")
        return
    max_workers = max(1, int(args.num_workers))
    if max_workers == 1:
        for _, row in work.iterrows():
            rec = _run_one(row.to_dict(), str(dirs.pybdsf_raw), str(dirs.pybdsf_logs), opts)
            records.append(rec)
            _write_status_locked(status_path, rec)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_run_one, row.to_dict(), str(dirs.pybdsf_raw), str(dirs.pybdsf_logs), opts)
                for _, row in work.iterrows()
            ]
            for future in as_completed(futures):
                rec = future.result()
                records.append(rec)
                _write_status_locked(status_path, rec)

    status = pd.DataFrame(records)
    _sync_manifest_and_failed_report(manifest_path, status_path, dirs)
    failed = status[status["status"] != "done"].copy()
    print(f"attempted={len(status)}")
    print(f"done={int((status['status'] == 'done').sum()) if len(status) else 0}")
    print(f"failed={len(failed)}")
    print(f"status={status_path}")


if __name__ == "__main__":
    main()
