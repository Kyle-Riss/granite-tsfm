# Copyright contributors to the TSFM project
#
"""Enrich hourly_merged: meteo/ASOS temp, holidays, peak/DR proxies, drop 2019.

  uv run python -m pipelines.seoul.fetch_meteo_temp
  uv run python -m pipelines.seoul.enrich_hourly
  uv run python -m pipelines.seoul.materialize --hourly-csv Data/hourly_merged.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from pipelines.seoul.config import DATA_DIR, DEFAULT_MIN_TRAIN_YEAR, HOURLY_MERGED_CSV
from pipelines.seoul.feature_enrich import enrich_hourly_frame


def _read_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ts" not in df.columns:
        df["ts"] = pd.to_datetime(df["datetime"], format="mixed")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enrich hourly CSV for TSFM pipeline.")
    parser.add_argument("--input", type=Path, default=HOURLY_MERGED_CSV)
    parser.add_argument("--output", type=Path, default=None, help="Default: overwrite input")
    parser.add_argument("--min-year", type=int, default=DEFAULT_MIN_TRAIN_YEAR)
    parser.add_argument("--meta-json", type=Path, default=DATA_DIR / "hourly_enrich_meta.json")
    args = parser.parse_args(argv)

    if not args.input.is_file():
        raise SystemExit(f"Missing {args.input}. Run merge_hourly first.")

    df = _read_hourly(args.input)
    enriched, meta = enrich_hourly_frame(df, min_year=args.min_year)
    out = args.output or args.input
    out.parent.mkdir(parents=True, exist_ok=True)

    export_cols = [
        "datetime",
        "region",
        "power",
        "temp",
        "hour",
        "weekday",
        "month",
        "season",
        "is_weekend",
        "temp_source",
        "temp_clim",
        "is_holiday",
        "is_holiday_eve",
        "is_kpx_peak_hour",
        "is_dr_window_proxy",
        "is_extreme_heat",
        "is_extreme_cold",
        "is_heatwave_day",
        "is_coldwave_day",
        "seoul_district_cv",
    ]
    optional = [c for c in export_cols if c in enriched.columns]
    enriched[optional].to_csv(out, index=False)

    args.meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}  rows={len(enriched)}")
    print(f"  temp_sources: {meta['temp_sources']}")
    print(f"  meta: {args.meta_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
