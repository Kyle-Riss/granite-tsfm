# Copyright contributors to the TSFM project
#
"""KMA ASOS hourly temperature (공공데이터포털). Requires env KMA_ASOS_SERVICE_KEY.

  export KMA_ASOS_SERVICE_KEY='...'   # URL-encoded key from data.go.kr
  uv run python -m pipelines.seoul.download_asos --start 20200101 --end 20241231
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from pipelines.seoul.config import ASOS_DIR, REGIONS, REGION_ASOS_STATION

ASOS_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"


def _fetch_chunk(stn: int, start_dt: str, end_dt: str, service_key: str) -> pd.DataFrame:
    """start_dt/end_dt: YYYYMMDD (inclusive days; API uses startHh/endHh)."""
    params = urlencode(
        {
            "serviceKey": service_key,
            "pageNo": "1",
            "numOfRows": "9999",
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "HR",
            "startDt": start_dt,
            "startHh": "00",
            "endDt": end_dt,
            "endHh": "23",
            "stnIds": str(stn),
        }
    )
    with urlopen(f"{ASOS_URL}?{params}", timeout=120) as resp:
        data = json.load(resp)
    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    if not items:
        return pd.DataFrame(columns=["ts", "temp_c", "stnId"])
    if isinstance(items, dict):
        items = [items]
    rows = []
    for it in items:
        tm = it.get("tm")
        ta = it.get("ta")
        if tm is None or ta is None:
            continue
        rows.append({"ts": pd.to_datetime(tm), "temp_c": float(ta), "stnId": int(stn)})
    return pd.DataFrame(rows)


def download_station(
    region: str,
    stn: int,
    start: str,
    end: str,
    service_key: str,
    *,
    month_step: int = 1,
) -> Path:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    parts: list[pd.DataFrame] = []
    cur = start_ts.replace(day=1)
    while cur <= end_ts:
        chunk_end = min(cur + pd.DateOffset(months=month_step) - pd.DateOffset(days=1), end_ts)
        s = cur.strftime("%Y%m%d")
        e = chunk_end.strftime("%Y%m%d")
        print(f"  {region} stn={stn}  {s}..{e}", flush=True)
        try:
            parts.append(_fetch_chunk(stn, s, e, service_key))
        except Exception as ex:
            print(f"    warn: {ex}", flush=True)
        cur = chunk_end + pd.DateOffset(days=1)
        time.sleep(0.2)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(df):
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    out = ASOS_DIR / f"asos_hourly_stn{stn}_{region}.parquet"
    ASOS_DIR.mkdir(parents=True, exist_ok=True)
    df["region"] = region
    df.to_parquet(out, index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download KMA ASOS hourly temp per region.")
    parser.add_argument("--start", default="20200101", help="YYYYMMDD")
    parser.add_argument("--end", default="20241231", help="YYYYMMDD")
    args = parser.parse_args(argv)

    key = os.environ.get("KMA_ASOS_SERVICE_KEY", "").strip()
    if not key:
        raise SystemExit(
            "Set KMA_ASOS_SERVICE_KEY (공공데이터포털 인증키, URL 인코딩).\n"
            "Until then: uv run python -m pipelines.seoul.fetch_meteo_temp"
        )
    for region in REGIONS:
        stn = REGION_ASOS_STATION[region]
        download_station(region, stn, args.start, args.end, key)
    print(f"Done. Files under {ASOS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
