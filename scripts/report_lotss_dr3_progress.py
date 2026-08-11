#!/usr/bin/env python
"""Write a compact current-progress report for the LoTSS DR3 production parent-linking run."""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lotss_dr3_common import DEFAULT_OUTPUT_ROOT, ensure_output_dirs, read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def _latest_status(path: Path) -> pd.DataFrame:
    frame = _read_csv(path)
    if frame.empty or "file_id" not in frame:
        return pd.DataFrame()
    return frame.drop_duplicates("file_id", keep="last")


def _latest_file(root: Path) -> tuple[str, str]:
    if not root.exists():
        return "", ""
    latest: Path | None = None
    latest_mtime = -1.0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > latest_mtime:
            latest = path
            latest_mtime = mtime
    if latest is None:
        return "", ""
    stamp = datetime.fromtimestamp(latest_mtime, timezone.utc).isoformat()
    return stamp, str(latest)


def _slurm_jobs() -> str:
    try:
        user = os.environ.get("USER", "")
        cmd = ["squeue", "-u", user] if user else ["squeue"]
        proc = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return proc.stdout.strip()
    except Exception as exc:
        return f"squeue_failed: {type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()
    dirs = ensure_output_dirs(args.output_root)
    manifest_path = dirs.manifests / "lotss_dr3_fits_manifest.csv"
    manifest = read_manifest(manifest_path)
    total = len(manifest)

    py_status = _latest_status(dirs.checkpoints / "pybdsf_status.csv")
    assoc_status = _latest_status(dirs.checkpoints / "association_field_status.csv")

    manifest_done = set(manifest.loc[manifest["needs_pybdsf"].astype(str) == "False", "file_id"].astype(str)) if total else set()
    py_done = set(py_status.loc[py_status["status"].astype(str) == "done", "file_id"].astype(str)) if not py_status.empty else set()
    py_failed = set(py_status.loc[py_status["status"].astype(str) == "failed", "file_id"].astype(str)) if not py_status.empty else set()
    py_ready = manifest_done | py_done
    py_pending = max(0, total - len(py_ready))

    assoc_done = set(assoc_status.loc[assoc_status["status"].astype(str) == "done", "file_id"].astype(str)) if not assoc_status.empty else set()
    assoc_failed = set(assoc_status.loc[assoc_status["status"].astype(str) == "failed", "file_id"].astype(str)) if not assoc_status.empty else set()
    assoc_skipped = set(assoc_status.loc[assoc_status["status"].astype(str) == "skipped", "file_id"].astype(str)) if not assoc_status.empty else set()
    assoc_pending_ready = max(0, len(py_ready - assoc_done - assoc_failed - assoc_skipped))

    latest_assoc_stamp, latest_assoc_path = _latest_file(dirs.association_catalogs / "partials")
    latest_pybdsf_stamp, latest_pybdsf_path = _latest_file(dirs.pybdsf_raw)
    latest_stamp, latest_path = (latest_assoc_stamp, latest_assoc_path)
    if latest_pybdsf_stamp and (not latest_stamp or latest_pybdsf_stamp > latest_stamp):
        latest_stamp, latest_path = latest_pybdsf_stamp, latest_pybdsf_path

    now = datetime.now(timezone.utc)
    out_path = dirs.reports / f"current_progress_{now.strftime('%Y%m%dT%H%M%SZ')}.md"
    lines = [
        "# LoTSS DR3 production parent-linking Current Progress",
        "",
        f"- generated_utc: {now.isoformat()}",
        f"- output_root: {dirs.root}",
        f"- total_fields: {total}",
        f"- pybdsf_done: {len(py_ready)}",
        f"- pybdsf_pending: {py_pending}",
        f"- pybdsf_failed: {len(py_failed)}",
        f"- association_done: {len(assoc_done)}",
        f"- association_pending_among_pybdsf_done: {assoc_pending_ready}",
        f"- association_failed: {len(assoc_failed)}",
        f"- association_skipped: {len(assoc_skipped)}",
        f"- latest_output_timestamp_utc: {latest_stamp}",
        f"- latest_output_file: {latest_path}",
        "",
        "## Active Slurm Jobs",
        "",
        "```text",
        _slurm_jobs(),
        "```",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(out_path)
    print(f"total_fields={total}")
    print(f"pybdsf_done={len(py_ready)}")
    print(f"pybdsf_pending={py_pending}")
    print(f"pybdsf_failed={len(py_failed)}")
    print(f"association_done={len(assoc_done)}")
    print(f"association_pending_among_pybdsf_done={assoc_pending_ready}")
    print(f"association_failed={len(assoc_failed)}")
    print(f"association_skipped={len(assoc_skipped)}")
    print(f"latest_output_file={latest_path}")


if __name__ == "__main__":
    main()
