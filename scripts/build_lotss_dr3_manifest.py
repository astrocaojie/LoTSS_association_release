#!/usr/bin/env python
"""Build the LoTSS DR3 full-run FITS/H5/PyBDSF manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from lotss_dr3_common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_H5_ROOT,
    DEFAULT_ORIGINAL_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    MANIFEST_COLUMNS,
    default_pybdsf_catalog_path,
    ensure_output_dirs,
    field_id_from_path,
    find_existing_pybdsf_catalog,
    h5_map,
    list_fits,
    scratch_path_for,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--original-data-root", type=Path, default=DEFAULT_ORIGINAL_DATA_ROOT)
    parser.add_argument("--h5-root", type=Path, action="append", default=[DEFAULT_H5_ROOT])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--catalog-search-root", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dirs = ensure_output_dirs(args.output_root)
    manifest_path = args.manifest or dirs.manifests / "lotss_dr3_fits_manifest.csv"
    search_roots = [dirs.pybdsf_raw, dirs.pybdsf_processed, *args.catalog_search_root]
    h5_by_field = h5_map(args.h5_root)
    rows = []
    for fits_path in list_fits(args.original_data_root):
        field_id = field_id_from_path(fits_path)
        scratch_path = scratch_path_for(fits_path, args.original_data_root, args.data_root)
        h5_path = h5_by_field.get(field_id)
        catalog = find_existing_pybdsf_catalog(field_id, search_roots, raw_root=dirs.pybdsf_raw)
        if catalog is None:
            expected = default_pybdsf_catalog_path(dirs.pybdsf_raw, field_id)
            catalog_text = str(expected)
            has_catalog = False
            needs_pybdsf = True
            status = "needs_pybdsf"
        else:
            catalog_text = str(catalog)
            has_catalog = True
            needs_pybdsf = False
            status = "ready"
        rows.append(
            {
                "file_id": field_id,
                "fits_path": str(fits_path),
                "scratch_fits_path": str(scratch_path),
                "h5_path": str(h5_path) if h5_path else "",
                "field_name": field_id,
                "has_existing_pybdsf_catalog": str(has_catalog),
                "pybdsf_catalog_path": catalog_text,
                "needs_pybdsf": str(needs_pybdsf),
                "status": status,
            }
        )
    import pandas as pd

    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    write_manifest(frame, manifest_path)
    print(f"manifest={manifest_path}")
    print(f"fits={len(frame)}")
    print(f"h5={int((frame['h5_path'].astype(str) != '').sum()) if len(frame) else 0}")
    print(f"pybdsf_existing={int((frame['has_existing_pybdsf_catalog'].astype(str) == 'True').sum()) if len(frame) else 0}")
    print(f"pybdsf_needed={int((frame['needs_pybdsf'].astype(str) == 'True').sum()) if len(frame) else 0}")


if __name__ == "__main__":
    main()
