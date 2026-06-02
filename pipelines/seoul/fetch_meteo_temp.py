# Copyright contributors to the TSFM project
#
"""Hourly temperature via Open-Meteo Archive API (no API key; ASOS substitute).

  uv run python -m pipelines.seoul.fetch_meteo_temp --start 2020-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from pipelines.seoul.config import METEO_DIR, REGIONS, REGION_METEO_COORDS

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def fetch_region_chunk(
    region: str,
    lat: float,
    lon: float,
    start: str,
    end: str,
) -> pd.DataFrame:
    import urllib.parse
    import urllib.request

    params = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": start,
            "end_date": end,
            "hourly": "temperature_2m",
            "timezone": "Asia/Seoul",
        }
    )
    url = f"{OPEN_METEO_ARCHIVE}?{params}"
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=180) as resp:
                body = json.load(resp)
            break
        except Exception as ex:
            last_err = ex
            time.sleep(2.0 * (attempt + 1))
    else:
        raise last_err  # type: ignore[misc]
    hourly = body["hourly"]
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(hourly["time"]),
            "temp_c": hourly["temperature_2m"],
            "region": region,
        }
    )
    return df


def fetch_all(
    start: str,
    end: str,
    *,
    chunk_days: int = 120,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    out_dir = out_dir or METEO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    for region in REGIONS:
        lat, lon = REGION_METEO_COORDS[region]
        chunks: list[pd.DataFrame] = []
        cur = start_ts
        while cur <= end_ts:
            chunk_end = min(cur + pd.Timedelta(days=chunk_days - 1), end_ts)
            print(f"  {region}  {cur.date()} .. {chunk_end.date()}", flush=True)
            chunks.append(
                fetch_region_chunk(
                    region,
                    lat,
                    lon,
                    cur.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                )
            )
            cur = chunk_end + pd.Timedelta(days=1)
            time.sleep(0.3)
        df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["ts"]).sort_values("ts")
        out = out_dir / f"hourly_temp_{region}.parquet"
        df.to_parquet(out, index=False)
        paths[region] = out
        print(f"  wrote {out}  rows={len(df)}", flush=True)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch hourly temp (Open-Meteo) per KPX region.")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--chunk-days", type=int, default=120)
    parser.add_argument("--out-dir", type=Path, default=METEO_DIR)
    args = parser.parse_args(argv)
    meta_path = args.out_dir / "fetch_meta.json"
    paths = fetch_all(args.start, args.end, chunk_days=args.chunk_days, out_dir=args.out_dir)
    meta_path.write_text(
        json.dumps(
            {
                "source": "Open-Meteo Archive API (2m temperature, Asia/Seoul)",
                "start": args.start,
                "end": args.end,
                "regions": {k: str(v) for k, v in paths.items()},
                "note": "Replace with KMA ASOS via download_asos.py when ServiceKey available.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
