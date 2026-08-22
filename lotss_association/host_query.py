"""Cached CatWISE2020/AllWISE host-candidate queries for host-gated passes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import time

import numpy as np
import pandas as pd


CATALOGS = {
    "catwise2020": {"vizier_id": "II/365", "id_columns": ["Name", "objID", "source_id"]},
    "allwise": {"vizier_id": "II/328/allwise", "id_columns": ["AllWISE", "Source", "cntr"]},
}

HOST_QUERY_LOG_COLUMNS = [
    "query_id",
    "ra",
    "dec",
    "radius_arcsec",
    "catalogue",
    "status",
    "n_results",
    "query_time",
    "error_message",
]

HOST_RAW_COLUMNS = [
    "query_id",
    "catalogue",
    "host_id",
    "host_ra",
    "host_dec",
    "W1",
    "W2",
    "W1_W2",
    "W1_snr",
    "W2_snr",
    "cc_flags",
    "ext_flg",
    "raw_column_map_json",
    "missing_wise_columns",
    "raw_record",
]

HOST_COLUMN_ALIASES = {
    "ra": ["RA_ICRS", "RAJ2000", "ra", "RA", "RAPMdeg"],
    "dec": ["DE_ICRS", "DEJ2000", "dec", "DEC", "DEPMdeg"],
    "host_id": ["Name", "objID", "AllWISE", "CatWISE", "source_id", "designation", "WISEA", "Source", "cntr"],
    "W1": ["W1mpro", "W1mag", "W1", "w1mpro", "W1MPRO", "W1mproPM", "w1mpropm"],
    "W2": ["W2mpro", "W2mag", "W2", "w2mpro", "W2MPRO", "W2mproPM", "w2mpropm"],
    "W1_snr": ["snr1", "w1snr", "W1snr", "snrW1pm", "snrw1pm", "W1SNR"],
    "W2_snr": ["snr2", "w2snr", "W2snr", "snrW2pm", "snrw2pm", "W2SNR"],
    "cc_flags": ["cc_flags", "cc_flags_1", "cc_flags_2", "ccf", "CCflags"],
    "ext_flg": ["ext_flg", "extflg", "ExtFlag", "ab_flags", "abf"],
}


@dataclass
class HostQueryResult:
    results: pd.DataFrame
    log: pd.DataFrame


class HostQueryClient:
    """Query WISE-family catalogues with persistent cache and graceful failure."""

    def __init__(
        self,
        cache_dir: str | Path,
        cache_path: str | Path | None = None,
        offline_cache_only: bool = False,
        skip_query: bool = False,
        max_results: int = 20,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = Path(cache_path) if cache_path else self.cache_dir / "host_query_cache.parquet"
        self.jsonl_path = self.cache_dir / "host_query_cache.jsonl"
        self.offline_cache_only = bool(offline_cache_only)
        self.skip_query = bool(skip_query)
        self.max_results = int(max_results)
        self.columns_debug_records: list[dict[str, Any]] = []
        self._cache = self._load_cache()

    def _load_cache(self) -> pd.DataFrame:
        if self.cache_path.exists():
            try:
                return self._repair_cache(pd.read_parquet(self.cache_path))
            except Exception:
                pass
        if self.jsonl_path.exists():
            rows = []
            with self.jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
            if rows:
                return self._repair_cache(pd.DataFrame(rows))
        return pd.DataFrame(columns=HOST_RAW_COLUMNS)

    def _repair_cache(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Backfill normalized WISE fields from raw_record when an older cache lacks them."""

        if frame.empty or "raw_record" not in frame:
            return self._with_columns(frame, HOST_RAW_COLUMNS)
        repaired_rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            record = row.to_dict()
            needs_repair = any(col not in record or pd.isna(record.get(col)) for col in ["host_ra", "host_dec", "W1", "W2"])
            raw_text = record.get("raw_record", "")
            if needs_repair and isinstance(raw_text, str) and raw_text:
                try:
                    raw = pd.Series(json.loads(raw_text))
                    normalized = self._normalize_row(raw, str(record.get("catalogue", "")), str(record.get("query_id", "")))
                    for key, value in normalized.items():
                        if key == "raw_record":
                            continue
                        if key not in record or pd.isna(record.get(key)):
                            record[key] = value
                except Exception:
                    pass
            repaired_rows.append(record)
        return self._with_columns(pd.DataFrame(repaired_rows), HOST_RAW_COLUMNS)

    def save_cache(self) -> None:
        cache = self._with_columns(self._cache, HOST_RAW_COLUMNS)
        try:
            cache.to_parquet(self.cache_path, index=False)
        except Exception:
            pass

    def write_columns_debug(self, path: str | Path) -> None:
        """Persist raw Vizier column names and normalized mapping decisions."""

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            json.dump(self.columns_debug_records, handle, ensure_ascii=True, indent=2, default=str)
        try:
            with self.jsonl_path.open("w", encoding="utf-8") as handle:
                for record in cache.to_dict(orient="records"):
                    handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
        except Exception:
            pass

    @staticmethod
    def _with_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        out = df.copy()
        for col in columns:
            if col not in out:
                out[col] = pd.Series(dtype=object)
        if out.empty:
            return pd.DataFrame(columns=columns)
        return out[columns]

    @staticmethod
    def query_id(ra: float, dec: float, radius_arcsec: float, catalogue: str) -> str:
        return f"{catalogue}_{ra:.7f}_{dec:.7f}_{radius_arcsec:.2f}"

    def _cache_lookup(self, query_id: str, catalogue: str) -> pd.DataFrame:
        if self._cache.empty:
            return pd.DataFrame(columns=HOST_RAW_COLUMNS)
        mask = (self._cache["query_id"].astype(str) == str(query_id)) & (self._cache["catalogue"].astype(str) == catalogue)
        return self._with_columns(self._cache[mask].copy(), HOST_RAW_COLUMNS)

    def query_catalogue(self, ra: float, dec: float, radius_arcsec: float, catalogue: str) -> HostQueryResult:
        query_id = self.query_id(ra, dec, radius_arcsec, catalogue)
        cached = self._cache_lookup(query_id, catalogue)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not cached.empty:
            return HostQueryResult(
                results=cached,
                log=pd.DataFrame(
                    [
                        {
                            "query_id": query_id,
                            "ra": ra,
                            "dec": dec,
                            "radius_arcsec": radius_arcsec,
                            "catalogue": catalogue,
                            "status": "cache_hit",
                            "n_results": int(len(cached)),
                            "query_time": now,
                            "error_message": "",
                        }
                    ],
                    columns=HOST_QUERY_LOG_COLUMNS,
                ),
            )
        if self.skip_query:
            return self._empty_log(query_id, ra, dec, radius_arcsec, catalogue, "skipped", "")
        if self.offline_cache_only:
            return self._empty_log(query_id, ra, dec, radius_arcsec, catalogue, "cache_miss_offline", "")
        try:
            rows = self._query_vizier(ra, dec, radius_arcsec, catalogue, query_id)
            result = self._with_columns(pd.DataFrame(rows), HOST_RAW_COLUMNS)
            if not result.empty:
                self._cache = pd.concat([self._cache, result], ignore_index=True)
            return HostQueryResult(
                results=result,
                log=pd.DataFrame(
                    [
                        {
                            "query_id": query_id,
                            "ra": ra,
                            "dec": dec,
                            "radius_arcsec": radius_arcsec,
                            "catalogue": catalogue,
                            "status": "ok",
                            "n_results": int(len(result)),
                            "query_time": now,
                            "error_message": "",
                        }
                    ],
                    columns=HOST_QUERY_LOG_COLUMNS,
                ),
            )
        except Exception as exc:
            return self._empty_log(query_id, ra, dec, radius_arcsec, catalogue, "failed", str(exc))

    def _empty_log(
        self,
        query_id: str,
        ra: float,
        dec: float,
        radius_arcsec: float,
        catalogue: str,
        status: str,
        error: str,
    ) -> HostQueryResult:
        return HostQueryResult(
            results=pd.DataFrame(columns=HOST_RAW_COLUMNS),
            log=pd.DataFrame(
                [
                    {
                        "query_id": query_id,
                        "ra": ra,
                        "dec": dec,
                        "radius_arcsec": radius_arcsec,
                        "catalogue": catalogue,
                        "status": status,
                        "n_results": 0,
                        "query_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "error_message": error,
                    }
                ],
                columns=HOST_QUERY_LOG_COLUMNS,
            ),
        )

    def _query_vizier(self, ra: float, dec: float, radius_arcsec: float, catalogue: str, query_id: str) -> list[dict[str, Any]]:
        try:
            from astropy import units as u
            from astropy.coordinates import SkyCoord
            from astroquery.vizier import Vizier
        except Exception as exc:
            raise RuntimeError(f"astroquery.vizier unavailable: {exc}") from exc
        if catalogue not in CATALOGS:
            raise ValueError(f"Unknown host catalogue: {catalogue}")
        vizier = Vizier(columns=["**"], row_limit=self.max_results)
        coord = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame="icrs")
        tables = vizier.query_region(coord, radius=float(radius_arcsec) * u.arcsec, catalog=CATALOGS[catalogue]["vizier_id"])
        if not tables:
            return []
        table = tables[0]
        frame = table.to_pandas()
        self.columns_debug_records.append(
            {
                "query_id": query_id,
                "catalogue": catalogue,
                "raw_columns": [str(col) for col in frame.columns],
                "normalized_column_map": self._column_map_for_columns(frame.columns),
                "missing_wise_columns": self._missing_wise_columns_for_columns(frame.columns),
            }
        )
        rows: list[dict[str, Any]] = []
        for _, row in frame.head(self.max_results).iterrows():
            rows.append(self._normalize_row(row, catalogue, query_id))
        return rows

    def _normalize_row(self, row: pd.Series, catalogue: str, query_id: str) -> dict[str, Any]:
        def first(key: str, default: Any = np.nan) -> tuple[Any, str]:
            aliases = HOST_COLUMN_ALIASES.get(key, [])
            lower_to_name = {str(name).lower(): name for name in row.index}
            for name in aliases:
                selected = name if name in row.index else lower_to_name.get(name.lower())
                if selected is not None and selected in row and pd.notna(row[selected]):
                    return row[selected], str(selected)
            return default, ""

        ra, ra_col = first("ra")
        dec, dec_col = first("dec")
        w1, w1_col = first("W1")
        w2, w2_col = first("W2")
        try:
            w1_w2 = float(w1) - float(w2)
        except Exception:
            w1_w2 = np.nan
        host_id, host_id_col = first("host_id", "")
        w1_snr, w1_snr_col = first("W1_snr")
        w2_snr, w2_snr_col = first("W2_snr")
        cc_flags, cc_flags_col = first("cc_flags", "")
        ext_flg, ext_flg_col = first("ext_flg")
        column_map = {
            "ra": ra_col,
            "dec": dec_col,
            "host_id": host_id_col,
            "W1": w1_col,
            "W2": w2_col,
            "W1_snr": w1_snr_col,
            "W2_snr": w2_snr_col,
            "cc_flags": cc_flags_col,
            "ext_flg": ext_flg_col,
        }
        missing_wise = []
        if not w1_col:
            missing_wise.append("W1")
        if not w2_col:
            missing_wise.append("W2")
        return {
            "query_id": query_id,
            "catalogue": catalogue,
            "host_id": str(host_id),
            "host_ra": float(ra) if pd.notna(ra) else np.nan,
            "host_dec": float(dec) if pd.notna(dec) else np.nan,
            "W1": float(w1) if pd.notna(w1) else np.nan,
            "W2": float(w2) if pd.notna(w2) else np.nan,
            "W1_W2": float(w1_w2) if np.isfinite(w1_w2) else np.nan,
            "W1_snr": w1_snr,
            "W2_snr": w2_snr,
            "cc_flags": str(cc_flags),
            "ext_flg": ext_flg,
            "raw_column_map_json": json.dumps(column_map, ensure_ascii=True),
            "missing_wise_columns": ",".join(missing_wise),
            "raw_record": json.dumps(row.to_dict(), ensure_ascii=True, default=str),
        }

    @staticmethod
    def _column_map_for_columns(columns: Any) -> dict[str, str]:
        names = [str(col) for col in columns]
        lower_to_name = {name.lower(): name for name in names}
        mapping: dict[str, str] = {}
        for key, aliases in HOST_COLUMN_ALIASES.items():
            selected = ""
            for alias in aliases:
                if alias in names:
                    selected = alias
                    break
                selected = lower_to_name.get(alias.lower(), "")
                if selected:
                    break
            mapping[key] = selected
        return mapping

    @staticmethod
    def _missing_wise_columns_for_columns(columns: Any) -> list[str]:
        mapping = HostQueryClient._column_map_for_columns(columns)
        missing = []
        if not mapping.get("W1"):
            missing.append("W1")
        if not mapping.get("W2"):
            missing.append("W2")
        return missing
