# Copyright contributors to the TSFM project
#
"""KPX long Parquet + Temp-pre → hourly_merged.csv (Temp-pre 스키마).

  uv run python -m pipelines.seoul.ingest_kpx --input Data/kpx/kpx_hourly.csv
  uv run python -m pipelines.seoul.merge_hourly
  uv run python -m pipelines.seoul.materialize --hourly-csv Data/hourly_merged.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.seoul.config import DATA_DIR, HOURLY_CSV, HOURLY_MERGED_CSV, KPX_LONG_PARQUET

MERGE_META_JSON = DATA_DIR / "hourly_merge_meta.json"

REGIONS = ("서울시", "부산시", "대전시", "강원도")
SEASON_MAP = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}


def _load_temp_pre(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["datetime"])
    return df


def _temp_climatology(temp_pre: pd.DataFrame) -> pd.DataFrame:
    """Per region × month × hour median temp from existing Temp-pre (proxy for missing years)."""
    g = temp_pre.groupby(["region", "month", "hour"], observed=True)["temp"].median().reset_index()
    g = g.rename(columns={"temp": "temp_clim"})
    return g


def _calibrate_kpx(kpx: pd.DataFrame, temp_pre: pd.DataFrame) -> pd.DataFrame:
    """Scale KPX MWh to Temp-pre power scale using overlapping timestamps in 2024."""
    tp = temp_pre[["ts", "region", "power"]].copy()
    kpx = kpx.copy()
    overlap = kpx.merge(tp, on=["ts", "region"], how="inner", suffixes=("_kpx", "_ref"))
    scales: dict[str, float] = {}
    for reg in REGIONS:
        sub = overlap[overlap["region"] == reg]
        if len(sub) < 24:
            scales[reg] = 1.0
            continue
        ratio = sub["power"] / sub["power_mwh"].replace(0, np.nan)
        scales[reg] = float(np.nanmedian(ratio))
        if not np.isfinite(scales[reg]) or scales[reg] <= 0:
            scales[reg] = 1.0
    kpx["power"] = kpx.apply(lambda r: r["power_mwh"] * scales.get(r["region"], 1.0), axis=1)
    return kpx, scales


def _attach_features(df: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["ts"].dt.hour.astype(int)
    df["weekday"] = df["ts"].dt.weekday.astype(int)
    df["month"] = df["ts"].dt.month.astype(int)
    df["season"] = df["month"].map(SEASON_MAP).fillna("mid")
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df = df.merge(clim, on=["region", "month", "hour"], how="left")
    # Prefer measured temp when present
    if "temp" not in df.columns:
        df["temp"] = df["temp_clim"]
    else:
        df["temp"] = df["temp"].fillna(df["temp_clim"])
    df = df.drop(columns=["temp_clim"], errors="ignore")
    return df


def _format_datetime_series(ts: pd.Series) -> pd.Series:
    def fmt(t: pd.Timestamp) -> str:
        return f"{t.month}/{t.day}/{str(t.year)[-2:]} {t.hour}:{t.minute:02d}"

    return ts.apply(fmt)


def merge_hourly(
    temp_pre_path: Path,
    kpx_long_path: Path,
    output_csv: Path,
    years_kpx: list[int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    temp_pre = _load_temp_pre(temp_pre_path)
    clim = _temp_climatology(temp_pre)

    if not kpx_long_path.is_file():
        raise FileNotFoundError(f"Run ingest_kpx first: {kpx_long_path}")

    kpx = pd.read_parquet(kpx_long_path)
    if years_kpx:
        kpx = kpx[kpx["ts"].dt.year.isin(years_kpx)]

    kpx, scales = _calibrate_kpx(kpx, temp_pre)
    kpx_feat = _attach_features(kpx, clim)
    kpx_feat["datetime"] = _format_datetime_series(kpx_feat["ts"])

    tp_out = temp_pre.copy()
    tp_out["datetime"] = _format_datetime_series(tp_out["ts"])

    cols = ["datetime", "region", "power", "temp", "hour", "weekday", "month", "season", "is_weekend"]
    merged = pd.concat(
        [
            kpx_feat[cols],
            tp_out[cols],
        ],
        ignore_index=True,
    )
    merged = merged.drop_duplicates(subset=["region", "datetime"], keep="last")
    merged = merged.sort_values(["region", "datetime"]).reset_index(drop=True)

    meta = {
        "kpx_rows": len(kpx_feat),
        "temp_pre_rows": len(tp_out),
        "merged_rows": len(merged),
        "calibration_scales": scales,
        "years": sorted(
            pd.to_datetime(merged["datetime"], format="mixed", errors="coerce").dt.year.dropna().astype(int).unique().tolist()
        ),
        "temp_note": "KPX 구간 기온은 Temp-pre 2024 월×시 중앙값(기후학적 proxy) 사용",
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    MERGE_META_JSON.write_text(
        __import__("json").dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return merged, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge KPX + Temp-pre into hourly_merged.csv")
    parser.add_argument("--temp-pre", type=Path, default=HOURLY_CSV)
    parser.add_argument("--kpx-long", type=Path, default=KPX_LONG_PARQUET)
    parser.add_argument("--output", type=Path, default=HOURLY_MERGED_CSV)
    parser.add_argument(
        "--years",
        type=str,
        default="2020,2021,2022,2023,2024",
        help="KPX years (2019 excluded by default — level anomaly)",
    )
    args = parser.parse_args(argv)

    years = [int(y) for y in args.years.split(",")]
    merged, meta = merge_hourly(args.temp_pre, args.kpx_long, args.output, years_kpx=years)
    print(f"Wrote {args.output}  rows={len(merged)}")
    print(f"  years: {meta['years']}")
    print(f"  calibration: {meta['calibration_scales']}")
    print(f"  meta: {MERGE_META_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
