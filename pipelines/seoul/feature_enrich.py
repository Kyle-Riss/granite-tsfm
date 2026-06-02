# Copyright contributors to the TSFM project
#
"""Merge meteo/ASOS temp, calendar, peak/DR proxy flags; district volatility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.seoul.calendar_features import attach_calendar_flags
from pipelines.seoul.config import (
    ASOS_DIR,
    DISTRICT_CSV,
    METEO_DIR,
    REGIONS,
    DEFAULT_MIN_TRAIN_YEAR,
)


def load_meteo_long(out_dir: Path | None = None) -> pd.DataFrame | None:
    out_dir = out_dir or METEO_DIR
    if not out_dir.is_dir():
        return None
    parts = list(out_dir.glob("hourly_temp_*.parquet"))
    if not parts:
        return None
    frames = [pd.read_parquet(p) for p in parts]
    return pd.concat(frames, ignore_index=True)


def load_asos_long(asos_dir: Path | None = None) -> pd.DataFrame | None:
    asos_dir = asos_dir or ASOS_DIR
    if not asos_dir.is_dir():
        return None
    parts = list(asos_dir.glob("asos_hourly_*.parquet"))
    if not parts:
        return None
    frames = []
    for p in parts:
        df = pd.read_parquet(p)
        if "region" not in df.columns and "stnId" in df.columns:
            continue
        col = "temp_c" if "temp_c" in df.columns else "ta"
        frames.append(df.rename(columns={col: "temp_meteo"}))
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"])
    return out[["ts", "region", "temp_meteo"]]


def merge_hourly_temperature(
    df: pd.DataFrame,
    *,
    meteo: pd.DataFrame | None = None,
    asos: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"])
    out["temp_clim"] = out["temp"]
    out["temp_source"] = "climatology_proxy"

    meteo = meteo if meteo is not None else load_meteo_long()
    asos = asos if asos is not None else load_asos_long()

    if meteo is not None:
        meteo = meteo.rename(columns={"temp_c": "temp_meteo"})
        out = out.drop(columns=["temp_meteo"], errors="ignore")
        out = out.merge(meteo[["ts", "region", "temp_meteo"]], on=["ts", "region"], how="left")
        mask = out["temp_meteo"].notna()
        out.loc[mask, "temp"] = out.loc[mask, "temp_meteo"]
        out.loc[mask, "temp_source"] = "open_meteo"

    if asos is not None:
        out = out.drop(columns=["temp_asos"], errors="ignore")
        out = out.merge(asos.rename(columns={"temp_meteo": "temp_asos"}), on=["ts", "region"], how="left")
        mask = out["temp_asos"].notna()
        out.loc[mask, "temp"] = out.loc[mask, "temp_asos"]
        out.loc[mask, "temp_source"] = "asos"

    return out


def attach_event_proxy_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Peak / DR proxy flags (documented heuristics; not dispatch records)."""
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"])
    h = out["hour"]
    m = out["month"]
    wd = out["weekday"]

    # KPX summer peak hours (typical afternoon peak, Jun–Aug weekdays)
    out["is_kpx_peak_hour"] = (
        (m.isin([6, 7, 8])) & (wd < 5) & (h.isin([14, 15, 16, 17, 18, 19, 20]))
    ).astype(int)

    # Economic DR notification window proxy (weekday 14–18, May–Sep) — see KPX DR materials
    out["is_dr_window_proxy"] = (
        (m.isin([5, 6, 7, 8, 9])) & (wd < 5) & (h.between(14, 18))
    ).astype(int)

    # Heat / cold from merged temp (ASOS or meteo)
    out["is_extreme_heat"] = (out["temp"] >= 33.0).astype(int)
    out["is_extreme_cold"] = (out["temp"] <= -10.0).astype(int)

    daily = out.groupby([out["ts"].dt.date, "region"], observed=True)["temp"].agg(["max", "min"]).reset_index()
    daily.columns = ["date", "region", "temp_max_d", "temp_min_d"]
    out["_date"] = out["ts"].dt.date
    out = out.merge(daily, left_on=["_date", "region"], right_on=["date", "region"], how="left")
    out["is_heatwave_day"] = (out["temp_max_d"] >= 33.0).astype(int)
    out["is_coldwave_day"] = (out["temp_min_d"] <= -10.0).astype(int)
    out = out.drop(columns=["_date", "date", "temp_max_d", "temp_min_d"], errors="ignore")
    return out


def seoul_district_monthly_cv(path: Path | None = None) -> pd.DataFrame:
    """Cross-district usage CV by month (higher = more heterogeneous / volatile)."""
    path = path or DISTRICT_CSV
    d = pd.read_csv(path)
    g = d.groupby("month")["usage"].agg(["mean", "std"]).reset_index()
    g["seoul_district_cv"] = (g["std"] / g["mean"].replace(0, np.nan)).fillna(0.0)
    return g[["month", "seoul_district_cv"]]


def attach_district_volatility(df: pd.DataFrame, district_csv: Path | None = None) -> pd.DataFrame:
    out = df.copy()
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype("Int64")
    cv = seoul_district_monthly_cv(district_csv)
    out = out.drop(columns=["seoul_district_cv"], errors="ignore")
    out = out.merge(cv, on="month", how="left")
    out["seoul_district_cv"] = out["seoul_district_cv"].fillna(0.0).astype(float)
    return out


def enrich_hourly_frame(
    df: pd.DataFrame,
    *,
    min_year: int | None = DEFAULT_MIN_TRAIN_YEAR,
    meteo: pd.DataFrame | None = None,
    asos: pd.DataFrame | None = None,
    district_csv: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    if "ts" not in out.columns and "datetime" in out.columns:
        out["ts"] = pd.to_datetime(out["datetime"], format="mixed")

    if min_year is not None:
        out = out[out["ts"].dt.year >= min_year].copy()

    out = merge_hourly_temperature(out, meteo=meteo, asos=asos)
    out = attach_calendar_flags(out, "ts")
    out = attach_event_proxy_flags(out)
    out = attach_district_volatility(out, district_csv)

    meta = {
        "min_year": min_year,
        "temp_sources": out["temp_source"].value_counts().to_dict(),
        "rows": len(out),
        "regions": sorted(out["region"].unique().tolist()) if "region" in out.columns else [],
        "feature_note": (
            "is_dr_window_proxy: weekday 14–18 May–Sep heuristic (not KPX dispatch log). "
            "is_kpx_peak_hour: Jun–Aug weekday afternoon peak heuristic. "
            "seoul_district_cv: cross-gu monthly usage dispersion."
        ),
    }
    return out.reset_index(drop=True), meta
