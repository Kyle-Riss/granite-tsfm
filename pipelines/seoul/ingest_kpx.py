# Copyright contributors to the TSFM project
#
"""KPX 지역별·시간대별 전력거래량 CSV → long-format Parquet (ts, region, power_mwh).

  uv run python -m pipelines.seoul.ingest_kpx --input Data/kpx/kpx_hourly.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from pipelines.seoul.config import DATA_DIR

DEFAULT_MAP = DATA_DIR / "kpx_region_map.json"
DEFAULT_OUT = DATA_DIR / "kpx_hourly_long.parquet"

# column name aliases (lowercase match)
_ALIASES: dict[str, list[str]] = {
    "region": ["region", "지역", "권역", "지역명", "시장권역", "지역구분", "area", "권역명"],
    "power": [
        "power",
        "전력",
        "전력거래량",
        "거래량",
        "mwh",
        "전력량",
        "time_mwh",
        "value",
    ],
    "datetime": ["datetime", "일시", "거래일시", "date_time", "timestamp"],
    "date": ["date", "일자", "거래일자", "dt", "날짜"],
    "time": ["time", "시간", "거래시간", "hour_label", "시각"],
}


def _norm_col(c: str) -> str:
    return re.sub(r"\s+", "", str(c).strip().lower())


def _pick_column(columns: list[str], kind: str) -> str | None:
    norms = {_norm_col(c): c for c in columns}
    for alias in _ALIASES[kind]:
        if alias in norms:
            return norms[alias]
    for alias in _ALIASES[kind]:
        for nc, orig in norms.items():
            if alias in nc:
                return orig
    return None


def _load_region_map(path: Path) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not str(k).startswith("_")}


def _read_kpx_table(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode CSV (tried utf-8, cp949): {path}")


def _parse_datetime_series(df: pd.DataFrame) -> pd.Series:
    dt_col = _pick_column(list(df.columns), "datetime")
    if dt_col:
        return pd.to_datetime(df[dt_col], errors="coerce")

    d_col = _pick_column(list(df.columns), "date")
    t_col = _pick_column(list(df.columns), "time")
    if d_col and t_col:
        dates = pd.to_datetime(df[d_col], errors="coerce")
        hours = pd.to_numeric(df[t_col], errors="coerce")
        # KPX 공공데이터: 거래시간 1–24 → 00:00 시작 시각 (hour 1 = 00:00)
        if hours.notna().all() and hours.max() <= 24 and hours.min() >= 1:
            return dates + pd.to_timedelta(hours - 1, unit="h")
        combined = df[d_col].astype(str).str.strip() + " " + df[t_col].astype(str).str.strip()
        return pd.to_datetime(combined, errors="coerce")
    if d_col:
        return pd.to_datetime(df[d_col], errors="coerce")
    raise ValueError(f"Cannot find date/time columns in: {list(df.columns)}")


def _long_from_wide(df: pd.DataFrame, region_map: dict[str, str]) -> pd.DataFrame:
    """Wide: one date column + region columns with power values."""
    dt = _parse_datetime_series(df)
    skip = {
        c
        for c in (
            _pick_column(list(df.columns), "datetime"),
            _pick_column(list(df.columns), "date"),
            _pick_column(list(df.columns), "time"),
        )
        if c
    }
    value_cols = [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        value_cols = [c for c in df.columns if c not in skip]
    base = pd.DataFrame({"ts": dt})
    long_rows = []
    for col in value_cols:
        part = base.copy()
        part["region_raw"] = str(col)
        part["power_mwh"] = pd.to_numeric(df[col], errors="coerce")
        long_rows.append(part)
    out = pd.concat(long_rows, ignore_index=True)
    return _apply_region_map(out, region_map)


def _apply_region_map(df: pd.DataFrame, region_map: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["region_raw"] = df["region_raw"].astype(str).str.strip()
    df["region"] = df["region_raw"].map(region_map)
    unknown = df.loc[df["region"].isna(), "region_raw"].unique().tolist()
    if unknown:
        print(f"Warning: unmapped KPX regions (dropped): {unknown[:20]}{'...' if len(unknown) > 20 else ''}")
    df = df.dropna(subset=["ts", "region", "power_mwh"])
    df = df[df["power_mwh"] > 0]
    return df[["ts", "region", "power_mwh", "region_raw"]].sort_values(["region", "ts"])


def _ingest_single_csv(raw: pd.DataFrame, region_map: dict[str, str]) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("Empty CSV frame")

    reg_col = _pick_column(list(raw.columns), "region")
    p_col = _pick_column(list(raw.columns), "power")

    if reg_col and p_col:
        out = raw.rename(columns={reg_col: "region_raw", p_col: "power_mwh"}).copy()
        out["ts"] = _parse_datetime_series(raw)
        out["power_mwh"] = pd.to_numeric(out["power_mwh"], errors="coerce")
        return _apply_region_map(out, region_map)
    return _long_from_wide(raw, region_map)


def ingest_kpx_csv(
    input_path: Path,
    region_map_path: Path,
    output_path: Path,
    years: list[int] | None = None,
) -> pd.DataFrame:
    region_map = _load_region_map(region_map_path)
    raw = _read_kpx_table(input_path)
    out = _ingest_single_csv(raw, region_map)

    if years:
        out = out[out["ts"].dt.year.isin(years)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return out


def ingest_kpx_paths(
    input_paths: list[Path],
    region_map_path: Path,
    output_path: Path,
    years: list[int] | None = None,
) -> pd.DataFrame:
    region_map = _load_region_map(region_map_path)
    parts: list[pd.DataFrame] = []
    for path in input_paths:
        raw = _read_kpx_table(path)
        part = _ingest_single_csv(raw, region_map)
        if len(part):
            parts.append(part)
            print(f"  {path.name}: {len(part):,} rows")
    if not parts:
        raise ValueError("No rows ingested from any CSV")
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["region", "ts"], keep="last").sort_values(["region", "ts"])
    if years:
        out = out[out["ts"].dt.year.isin(years)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return out


def _resolve_input(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        candidates = sorted(path.glob("*.csv"))
        if candidates:
            print(f"Found {len(candidates)} CSV file(s) in {path}")
            return candidates
    return [path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest KPX hourly power CSV to long Parquet.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "kpx" / "kpx_hourly.csv",
        help="KPX CSV file, or directory Data/kpx/ (uses newest .csv)",
    )
    parser.add_argument("--region-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--years", type=str, default=None, help="Comma years to keep e.g. 2023,2024")
    args = parser.parse_args(argv)

    inputs = _resolve_input(args.input)
    if not inputs or not inputs[0].is_file():
        raise SystemExit(
            f"Missing {args.input}.\n"
            "  1) 공공데이터 KPX CSV 폴더/파일 → --input 경로\n"
            "  2) 또는 시연용: uv run python -m pipelines.seoul.bootstrap_kpx_demo\n"
            "     → ingest_kpx --input Data/kpx/kpx_hourly_demo.csv"
        )
    if not args.region_map.is_file():
        raise SystemExit(f"Missing {args.region_map}")

    years = [int(y) for y in args.years.split(",")] if args.years else None
    if len(inputs) == 1:
        out = ingest_kpx_csv(inputs[0], args.region_map, args.output, years=years)
    else:
        out = ingest_kpx_paths(inputs, args.region_map, args.output, years=years)
    print(f"Wrote {args.output}  rows={len(out)}  years={sorted(out['ts'].dt.year.unique().tolist())}")
    print(out.groupby("region")["power_mwh"].agg(["count", "mean"]).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
