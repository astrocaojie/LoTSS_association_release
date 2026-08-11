#!/usr/bin/env python
"""Check PyBDSF Gaussian-catalog coverage for the LoTSS DR3 manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lotss_dr3_common import (
    DEFAULT_OUTPUT_ROOT,
    ensure_output_dirs,
    find_existing_pybdsf_catalog,
    pybdsf_catalog_sane,
    read_manifest,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--catalog-search-root", type=Path, action="append", default=[])
    parser.add_argument("--write", action="store_true", help="Update the manifest in place.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = ensure_output_dirs(args.output_root)
    manifest_path = args.manifest or dirs.manifests / "lotss_dr3_fits_manifest.csv"
    frame = read_manifest(manifest_path)
    search_roots = [dirs.pybdsf_raw, dirs.pybdsf_processed, *args.catalog_search_root]
    rows = []
    failed_rows = []
    for _, row in frame.iterrows():
        field_id = str(row["file_id"])
        catalog = str(row.get("pybdsf_catalog_path", "") or "")
        ok, n_rows, reason = pybdsf_catalog_sane(catalog) if catalog else (False, 0, "missing")
        if not ok:
            found = find_existing_pybdsf_catalog(field_id, search_roots, raw_root=dirs.pybdsf_raw)
            if found is not None:
                catalog = str(found)
                ok, n_rows, reason = pybdsf_catalog_sane(catalog)
        out = row.to_dict()
        out["pybdsf_catalog_path"] = catalog
        out["has_existing_pybdsf_catalog"] = str(bool(ok))
        out["needs_pybdsf"] = str(not ok)
        out["status"] = "ready" if ok else "needs_pybdsf"
        rows.append(out)
        if not ok:
            failed_rows.append(
                {
                    "file_id": field_id,
                    "fits_path": row.get("fits_path", ""),
                    "scratch_fits_path": row.get("scratch_fits_path", ""),
                    "pybdsf_catalog_path": catalog,
                    "reason": reason,
                    "n_rows": n_rows,
                }
            )
    updated = pd.DataFrame(rows)
    failed = pd.DataFrame(failed_rows)
    failed_path = dirs.reports / "pybdsf_failed_files.csv"
    failed.to_csv(failed_path, index=False)
    if args.write:
        write_manifest(updated, manifest_path)
    print(f"manifest={manifest_path}")
    print(f"fields={len(updated)}")
    print(f"pybdsf_ready={int((updated['needs_pybdsf'].astype(str) == 'False').sum()) if len(updated) else 0}")
    print(f"pybdsf_needed={int((updated['needs_pybdsf'].astype(str) == 'True').sum()) if len(updated) else 0}")
    print(f"failed_report={failed_path}")


if __name__ == "__main__":
    main()
