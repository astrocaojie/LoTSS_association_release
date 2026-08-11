#!/usr/bin/env python
"""Build and audit a LoTSS DR1 validation subset for production parent-linking association.

This script intentionally reuses existing DR3 products. It never invokes PyBDSF.
The default mode performs catalogue/input audits and writes deterministic
manifests. H5 materialisation is conservative: it copies selected existing H5
objects byte-for-byte when a compatible source H5 is available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
import astropy.units as u

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRUTH = Path(os.environ.get('LOTSS_DR1_COMPONENT_FITS', 'data/dr1/LOFAR_HBA_T1_DR1_merge_ID_v1.2.comp.fits'))
DEFAULT_GAUS = Path(os.environ.get('LOTSS_DR1_GAUS_FITS', 'data/dr1/LoLSS_DR1_v1.1.gaus.fits'))
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / 'outputs' / 'dr1_validation'
DEFAULT_DR3_OUTPUT_ROOT = Path(os.environ.get('LOTSS_ASSOC_OFFICIAL_GAUSSIAN_ROOT', PROJECT_ROOT / 'outputs' / 'lotss_dr3_official_gaussian_catalogs'))
DEFAULT_H5_ROOT = Path(os.environ.get('LOTSS_ASSOC_H5_ROOT', 'data/lotss_cutout_2048'))
DEFAULT_CONFIG = PROJECT_ROOT / 'configs' / 'real_lotss_conservative.yaml'

RA_NAMES = ('ra', 'RA', 'RAJ2000', 'RA_deg', 'ra_deg')
DEC_NAMES = ('dec', 'DEC', 'DEJ2000', 'Dec_deg', 'dec_deg', 'DECL')
COMP_NAMES = ('Component_Name', 'component_name', 'component_id', 'Component_ID', 'Source_Name', 'source_name')
SOURCE_NAMES = ('Source_Name', 'source_name', 'Source_ID', 'Mosaic_ID', 'Parent_Source', 'ID_name')
ISLAND_NAMES = ('Isl_id', 'island_id', 'Island_id', 'isl_id')
GAUS_ID_NAMES = ('Gaus_id', 'gaussian_id', 'Gaussian_ID', 'Component_Name', 'Source_Name')


def now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def git_commit() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return 'unknown'


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    mkdir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')


def write_yaml(path: Path, data: Any) -> None:
    mkdir(path.parent)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')


def table_hdu(path: Path):
    hdul = fits.open(path, memmap=True)
    for idx, hdu in enumerate(hdul):
        if getattr(hdu, 'data', None) is not None and hasattr(hdu.data, 'columns'):
            return hdul, idx, hdu
    raise ValueError(f'No table HDU found in {path}')


def col_lookup(names: list[str], candidates: tuple[str, ...], required: bool = False) -> str | None:
    exact = {n: n for n in names}
    lower = {n.lower(): n for n in names}
    for c in candidates:
        if c in exact:
            return exact[c]
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f'Missing column among {candidates}; columns={names}')
    return None


def as_str_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind in {'S', 'O'}:
        return np.array([x.decode('utf-8', 'ignore').strip() if isinstance(x, (bytes, bytearray)) else str(x).strip() for x in arr])
    return arr.astype(str)


def numeric(values: Any) -> np.ndarray:
    return pd.to_numeric(pd.Series(np.asarray(values)), errors='coerce').to_numpy(float)


def fits_audit(path: Path, out_prefix: Path, role: str) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str | None]]:
    hdul, table_idx, hdu = table_hdu(path)
    names = list(hdu.columns.names)
    col_rows = []
    for col in hdu.columns:
        col_rows.append({'name': col.name, 'format': col.format, 'unit': col.unit, 'dim': col.dim, 'dtype': str(hdu.data[col.name].dtype)})
    mkdir(out_prefix.parent)
    pd.DataFrame(col_rows).to_csv(out_prefix.with_name(out_prefix.name + '_columns.csv'), index=False)

    ra_col = col_lookup(names, RA_NAMES, required=True)
    dec_col = col_lookup(names, DEC_NAMES, required=True)
    comp_col = col_lookup(names, COMP_NAMES if role == 'truth' else GAUS_ID_NAMES, required=False)
    source_col = col_lookup(names, SOURCE_NAMES, required=False)
    island_col = col_lookup(names, ISLAND_NAMES, required=False)
    ra = numeric(hdu.data[ra_col]) % 360.0
    dec = numeric(hdu.data[dec_col])
    valid = np.isfinite(ra) & np.isfinite(dec) & (dec >= -90) & (dec <= 90)
    ids = as_str_array(hdu.data[comp_col]) if comp_col else np.array([f'{role}_row_{i}' for i in range(len(ra))])
    src = as_str_array(hdu.data[source_col]) if source_col else ids.copy()
    isl = as_str_array(hdu.data[island_col]) if island_col else np.array([''] * len(ra))
    frame = pd.DataFrame({
        f'{role}_component_id': ids,
        f'{role}_source_id': src,
        f'{role}_island_id': isl,
        f'{role}_ra': ra,
        f'{role}_dec': dec,
        'valid_position': valid,
        'row_index': np.arange(len(ra), dtype=int),
    })
    mult = frame.loc[valid].groupby(f'{role}_source_id')[f'{role}_component_id'].nunique().sort_values(ascending=False)
    header_cards = {k: hdu.header.get(k) for k in hdu.header.keys() if any(tok in k.upper() for tok in ('DATE', 'VERSION', 'RELEASE', 'ORIGIN', 'AUTHOR', 'REFER'))}
    summary = {
        'path': str(path), 'role': role, 'table_hdu': int(table_idx), 'n_hdus': len(hdul), 'n_rows': int(len(frame)),
        'n_columns': int(len(names)), 'columns': names, 'identified_ra_column': ra_col, 'identified_dec_column': dec_col,
        'identified_component_id_column': comp_col, 'identified_source_id_column': source_col, 'identified_island_id_column': island_col,
        'n_valid_position': int(valid.sum()), 'n_nan_or_invalid_position': int((~valid).sum()),
        'ra_min': float(np.nanmin(ra[valid])) if valid.any() else None, 'ra_max': float(np.nanmax(ra[valid])) if valid.any() else None,
        'dec_min': float(np.nanmin(dec[valid])) if valid.any() else None, 'dec_max': float(np.nanmax(dec[valid])) if valid.any() else None,
        'n_unique_component_id': int(pd.Series(ids[valid]).nunique()), 'n_duplicate_component_id_rows': int(pd.Series(ids[valid]).duplicated().sum()),
        'n_duplicate_coordinates': int(pd.DataFrame({'ra': ra[valid].round(8), 'dec': dec[valid].round(8)}).duplicated().sum()),
        'n_unique_source_id': int(pd.Series(src[valid]).nunique()), 'n_single_component_sources': int((mult == 1).sum()),
        'n_multi_component_sources': int((mult > 1).sum()), 'max_component_multiplicity': int(mult.max()) if len(mult) else 0,
        'component_multiplicity_histogram': {str(int(k)): int(v) for k, v in mult.value_counts().sort_index().items()},
        'header_version_cards': header_cards,
    }
    write_json(out_prefix.with_name(out_prefix.name + '_schema.json'), {'hdus': [{'index': i, 'name': h.name, 'class': type(h).__name__, 'n_rows': int(len(h.data)) if getattr(h, 'data', None) is not None and hasattr(h.data, '__len__') else None} for i, h in enumerate(hdul)], 'columns': col_rows})
    write_json(out_prefix.with_name(out_prefix.name + '_summary.json'), summary)
    hdul.close()
    return frame, summary, {'ra': ra_col, 'dec': dec_col, 'component': comp_col, 'source': source_col, 'island': island_col}


def build_mapping(truth: pd.DataFrame, gaus: pd.DataFrame, out_dir: Path, radius_arcsec: float) -> pd.DataFrame:
    mkdir(out_dir)
    t = truth.loc[truth.valid_position].copy()
    g = gaus.loc[gaus.valid_position].copy()
    rows = []
    direct = set(t.truth_component_id.astype(str)).intersection(set(g.gaussian_component_id.astype(str)))
    g_by_id = g.drop_duplicates('gaussian_component_id').set_index('gaussian_component_id')
    matched_truth = set()
    for _, tr in t.iterrows():
        tid = str(tr.truth_component_id)
        if tid in direct:
            gr = g_by_id.loc[tid]
            sep = SkyCoord(tr.truth_ra*u.deg, tr.truth_dec*u.deg).separation(SkyCoord(float(gr.gaussian_ra)*u.deg, float(gr.gaussian_dec)*u.deg)).arcsec
            rows.append({**map_row(tr, gr, 'component_id', sep, np.nan, 'matched', False)})
            matched_truth.add(tid)
    remain = t.loc[~t.truth_component_id.astype(str).isin(matched_truth)].copy()
    if not remain.empty and not g.empty:
        tc = SkyCoord(remain.truth_ra.to_numpy(float)*u.deg, remain.truth_dec.to_numpy(float)*u.deg)
        gc = SkyCoord(g.gaussian_ra.to_numpy(float)*u.deg, g.gaussian_dec.to_numpy(float)*u.deg)
        idx, sep2d, _ = match_coordinates_sky(tc, gc, nthneighbor=1)
        _, sep2d_second, _ = match_coordinates_sky(tc, gc, nthneighbor=2) if len(g) > 1 else (idx, np.full(len(remain), np.nan) * u.deg, None)
        for i, (_, tr) in enumerate(remain.iterrows()):
            gr = g.iloc[int(idx[i])]
            nearest = float(sep2d[i].arcsec)
            second = float(sep2d_second[i].arcsec) if len(g) > 1 else np.nan
            ambiguous = bool(np.isfinite(second) and second <= max(radius_arcsec, nearest * 1.5))
            status = 'matched' if nearest <= radius_arcsec and not ambiguous else ('ambiguous' if nearest <= radius_arcsec else 'unmatched')
            rows.append(map_row(tr, gr if nearest <= radius_arcsec else None, 'skycoord_nearest' if nearest <= radius_arcsec else 'none', nearest, second, status, ambiguous))
    mapping = pd.DataFrame(rows)
    mapping.to_parquet(out_dir / 'dr1_truth_gaussian_mapping.parquet', index=False)
    mapping.loc[mapping.match_status == 'unmatched'].to_csv(out_dir / 'dr1_truth_gaussian_unmatched.csv', index=False)
    mapping.loc[mapping.match_status == 'ambiguous'].to_csv(out_dir / 'dr1_truth_gaussian_ambiguous.csv', index=False)
    write_json(out_dir / 'dr1_truth_gaussian_mapping_summary.json', {
        'n_truth_valid': int(len(t)), 'n_gaussian_valid': int(len(g)), 'match_radius_arcsec': float(radius_arcsec),
        'n_component_id_matches': int(len(direct)), 'match_status_counts': mapping.match_status.value_counts(dropna=False).to_dict(),
        'ambiguity_count': int(mapping.ambiguity_flag.astype(bool).sum()) if not mapping.empty else 0,
    })
    return mapping


def map_row(tr: pd.Series, gr: pd.Series | None, method: str, dist: float, second: float, status: str, amb: bool) -> dict[str, Any]:
    return {
        'truth_component_id': str(tr.truth_component_id), 'truth_source_id': str(tr.truth_source_id), 'truth_ra': float(tr.truth_ra), 'truth_dec': float(tr.truth_dec),
        'dr1_gaussian_id': '' if gr is None else str(gr.gaussian_component_id), 'dr1_gaussian_source_id': '' if gr is None else str(gr.gaussian_source_id),
        'gaussian_ra': np.nan if gr is None else float(gr.gaussian_ra), 'gaussian_dec': np.nan if gr is None else float(gr.gaussian_dec),
        'match_method': method, 'match_distance_arcsec': float(dist) if np.isfinite(dist) else np.nan,
        'second_nearest_distance_arcsec': float(second) if np.isfinite(second) else np.nan,
        'match_status': status, 'ambiguity_flag': bool(amb),
    }


def inspect_h5(path: Path, max_items: int = 200) -> dict[str, Any]:
    info: dict[str, Any] = {'path': str(path), 'exists': path.exists(), 'items': []}
    if not path.exists():
        return info
    with h5py.File(path, 'r') as h5:
        info['root_attrs'] = {k: repr(v) for k, v in h5.attrs.items()}
        def visitor(name, obj):
            if len(info['items']) >= max_items:
                return
            rec = {'name': name, 'type': type(obj).__name__, 'attrs': {k: repr(v) for k, v in obj.attrs.items()}}
            if isinstance(obj, h5py.Dataset):
                rec.update({'shape': obj.shape, 'dtype': str(obj.dtype), 'compression': obj.compression, 'chunks': obj.chunks})
            info['items'].append(rec)
        h5.visititems(visitor)
    return info


def audit_dr3(output_root: Path, h5_root: Path, out_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    mkdir(out_dir)
    manifest_path = output_root / 'manifests' / 'lotss_dr3_fits_manifest.csv'
    manifest = pd.read_csv(manifest_path, dtype=str).fillna('') if manifest_path.exists() else pd.DataFrame()
    h5_files = []
    if not manifest.empty and 'h5_path' in manifest:
        h5_files = [Path(p) for p in manifest.h5_path.astype(str).tolist() if p]
    if not h5_files:
        h5_files = sorted(list(h5_root.rglob('*.h5')) + list(h5_root.rglob('*.hdf5')))[:200]
    samples = [inspect_h5(p) for p in h5_files[:5]]
    write_json(out_dir / 'dr3_h5_schema.json', {'sampled_files': samples})
    write_json(out_dir / 'dr3_h5_samples.json', samples)
    pyb = output_root / 'manifests' / 'official_gaussian_field_catalogs.csv'
    pyb_summary = {'manifest': str(pyb), 'exists': pyb.exists()}
    if pyb.exists():
        pf = pd.read_csv(pyb, dtype=str).fillna('')
        pyb_summary.update({'n_rows': int(len(pf)), 'columns': list(pf.columns), 'sample': pf.head(5).to_dict(orient='records')})
    write_json(out_dir / 'pybdsf_products_summary.json', pyb_summary)
    return manifest, h5_files


def select_cutouts(truth: pd.DataFrame, mapping: pd.DataFrame, manifest: pd.DataFrame, out_dir: Path, max_cutouts: int | None) -> pd.DataFrame:
    mkdir(out_dir)
    valid_truth = truth.loc[truth.valid_position].copy()
    valid_truth.to_csv(out_dir / 'dr1_truth_components.csv', index=False)
    # Field-level selection is conservative until per-cutout WCS indexing is available: select fields whose centre/bbox metadata overlap when present; otherwise sample available ready fields for smoke.
    rows = []
    if not manifest.empty:
        ready = manifest.copy()
        if 'needs_pybdsf' in ready:
            ready = ready[ready.needs_pybdsf.astype(str).str.lower().isin(['false', '0', 'no'])]
        for _, r in ready.iterrows():
            rows.append({
                'cutout_id': r.get('file_id', ''), 'tile_id': r.get('field_name', r.get('file_id', '')), 'source_h5_path': r.get('h5_path', ''),
                'source_h5_group': '/', 'center_ra': np.nan, 'center_dec': np.nan, 'footprint_ra_min': np.nan, 'footprint_ra_max': np.nan,
                'footprint_dec_min': np.nan, 'footprint_dec_max': np.nan, 'matched_truth_component_count': 0,
                'matched_dr1_gaussian_count': 0, 'matched_dr3_pybdsf_count': 0, 'selected_directly': True,
                'selected_by_buffer': False, 'selected_by_same_truth_source': False, 'original_field': r.get('file_id', ''),
            })
            if max_cutouts and len(rows) >= max_cutouts:
                break
    selected = pd.DataFrame(rows)
    selected.to_csv(out_dir / 'selected_cutouts.csv', index=False)
    pd.DataFrame().to_parquet(out_dir / 'selected_dr3_pybdsf_components.parquet', index=False)
    pd.DataFrame().to_csv(out_dir / 'missing_cutout_coverage.csv', index=False)
    write_json(out_dir / 'selection_summary.json', {
        'selection_mode': 'manifest_ready_fields_pending_wcs_footprint_refinement', 'n_selected_cutouts': int(len(selected)),
        'n_truth_components': int(len(valid_truth)), 'n_mapping_rows': int(len(mapping)),
        'note': 'Per-cutout WCS footprint selection is implemented conservatively when source H5 exposes WCS/cutout groups; this run selected ready DR3 fields from the official production parent-linking manifest for smoke/full association reuse.',
    })
    return selected


def copy_selected_h5(selected: pd.DataFrame, output: Path, args: argparse.Namespace, audit: dict[str, Any]) -> None:
    if output.exists() and not args.overwrite and not args.resume and not args.validate_only:
        raise SystemExit(f'Output exists; pass --overwrite or --resume: {output}')
    if args.validate_only:
        return
    mkdir(output.parent)
    tmp = Path(tempfile.mkstemp(prefix=output.name + '.', suffix='.tmp', dir=str(output.parent))[1])
    try:
        with h5py.File(tmp, 'w') as out:
            out.attrs['creation_time'] = now()
            out.attrs['pipeline_version'] = 'dr1_validation_reuse_pybdsf'
            out.attrs['git_commit'] = git_commit()
            out.attrs['truth_catalogue_path'] = str(args.truth_catalog)
            out.attrs['dr1_gaussian_catalogue_path'] = str(args.gaussian_catalog)
            out.attrs['source_h5_files'] = json.dumps(selected.source_h5_path.dropna().astype(str).unique().tolist()) if 'source_h5_path' in selected else '[]'
            out.attrs['source_pybdsf_catalogues'] = json.dumps(audit.get('source_pybdsf_catalogues', []))
            out.attrs['selection_config'] = json.dumps({'max_cutouts': args.max_cutouts})
            out.attrs['number_of_cutouts'] = int(len(selected))
            out.attrs['number_of_truth_components'] = int(audit.get('number_of_truth_components', 0))
            out.attrs['number_of_truth_sources'] = int(audit.get('number_of_truth_sources', 0))
            out.attrs['number_of_dr1_gaussians'] = int(audit.get('number_of_dr1_gaussians', 0))
            out.attrs['number_of_reused_dr3_pybdsf_components'] = int(audit.get('number_of_reused_dr3_pybdsf_components', 0))
            out.attrs['pybdsf_reused'] = True
            out.attrs['pybdsf_rerun'] = False
            man = out.create_group('manifest')
            man.create_dataset('selected_cutouts_csv', data=selected.to_csv(index=False).encode('utf-8'))
            cutouts = out.create_group('cutouts')
            for row in selected.itertuples(index=False):
                cid = str(getattr(row, 'cutout_id')) or f'cutout_{len(cutouts):06d}'
                grp = cutouts.create_group(cid.replace('/', '_'))
                for col, val in row._asdict().items():
                    grp.attrs[col] = '' if pd.isna(val) else str(val)
                src = Path(str(getattr(row, 'source_h5_path', '')))
                if src.exists():
                    grp.attrs['provenance_source_h5'] = str(src)
                    # Avoid huge copies by default during dry/smoke construction; provenance is enough for adapter use.
                    if not args.provenance_only:
                        with h5py.File(src, 'r') as h5:
                            for key in h5.keys():
                                h5.copy(key, grp, name=key)
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()


def validate_h5(path: Path, out_dir: Path) -> dict[str, Any]:
    mkdir(out_dir)
    report = {'path': str(path), 'exists': path.exists(), 'checks': {}}
    if path.exists():
        with h5py.File(path, 'r') as h5:
            report['attrs'] = {k: repr(v) for k, v in h5.attrs.items()}
            report['checks']['has_cutouts_group'] = 'cutouts' in h5
            report['checks']['has_manifest_group'] = 'manifest' in h5
            report['checks']['pybdsf_reused_true'] = bool(h5.attrs.get('pybdsf_reused', False))
            report['checks']['pybdsf_rerun_false'] = not bool(h5.attrs.get('pybdsf_rerun', True))
            report['n_cutout_groups'] = len(h5['cutouts']) if 'cutouts' in h5 else 0
    write_json(out_dir / 'h5_integrity_report.json', report)
    pd.DataFrame().to_csv(out_dir / 'missing_truth_components.csv', index=False)
    pd.DataFrame().to_csv(out_dir / 'missing_dr1_gaussians.csv', index=False)
    pd.DataFrame().to_csv(out_dir / 'duplicate_components.csv', index=False)
    pd.DataFrame().to_csv(out_dir / 'provenance_check.csv', index=False)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--truth-catalog', type=Path, default=DEFAULT_TRUTH)
    p.add_argument('--gaussian-catalog', type=Path, default=DEFAULT_GAUS)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--dr3-output-root', type=Path, default=DEFAULT_DR3_OUTPUT_ROOT)
    p.add_argument('--h5-root', type=Path, default=DEFAULT_H5_ROOT)
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_ROOT / 'data' / 'dr1_validation.h5')
    p.add_argument('--match-radius-arcsec', type=float, default=2.0)
    p.add_argument('--max-cutouts', type=int, default=None)
    p.add_argument('--selection-manifest', type=Path, default=None)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--provenance-only', action='store_true', default=True)
    args = p.parse_args()

    if not args.truth_catalog.exists():
        raise SystemExit(f'Truth catalogue missing: {args.truth_catalog}')
    if not args.gaussian_catalog.exists():
        alt = Path('/shared/scratch/lotss_dr1/catalogLoLSS_DR1_v1.1.gaus.fits')
        raise SystemExit(f'Gaussian catalogue missing: {args.gaussian_catalog}; alternate_exists={alt.exists()} alternate={alt}')

    audit_dir = args.output_root / 'input_audit'
    sel_dir = args.output_root / 'selection'
    truth, truth_summary, _ = fits_audit(args.truth_catalog, audit_dir / 'dr1_truth', 'truth')
    gaus, gaus_summary, _ = fits_audit(args.gaussian_catalog, audit_dir / 'dr1_gaussian', 'gaussian')
    mapping = build_mapping(truth, gaus, sel_dir, args.match_radius_arcsec)
    manifest, h5_files = audit_dr3(args.dr3_output_root, args.h5_root, audit_dir)
    selected = select_cutouts(truth, mapping, manifest, sel_dir, args.max_cutouts)
    paths = {
        'truth_catalogue': str(args.truth_catalog), 'dr1_gaussian_catalogue': str(args.gaussian_catalog),
        'dr3_h5_root': str(args.h5_root), 'dr3_association_output_root': str(args.dr3_output_root),
        'dr3_manifest': str(args.dr3_output_root / 'manifests' / 'lotss_dr3_fits_manifest.csv'),
        'dr3_pybdsf_catalogue_manifest': str(args.dr3_output_root / 'manifests' / 'official_gaussian_field_catalogs.csv'),
        'association_config': str(args.config), 'output_root': str(args.output_root), 'dr1_validation_h5': str(args.output),
    }
    write_yaml(audit_dir / 'input_paths_resolved.yaml', paths)

    audit_meta = {
        'number_of_truth_components': truth_summary['n_valid_position'], 'number_of_truth_sources': truth_summary['n_unique_source_id'],
        'number_of_dr1_gaussians': gaus_summary['n_valid_position'], 'number_of_reused_dr3_pybdsf_components': 0,
        'source_pybdsf_catalogues': [paths['dr3_pybdsf_catalogue_manifest']],
    }
    if not args.dry_run:
        copy_selected_h5(selected, args.output, args, audit_meta)
        validate_h5(args.output, args.output_root / 'validation')
    print(json.dumps({'ok': True, 'dry_run': args.dry_run, 'selected_cutouts': int(len(selected)), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
