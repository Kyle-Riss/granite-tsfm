"""Smoke test KPX ingest + merge without public download."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipelines.seoul.ingest_kpx import ingest_kpx_csv
from pipelines.seoul.merge_hourly import merge_hourly


def test_ingest_and_merge_roundtrip(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    temp_pre = repo / "Data" / "Temp-pre.csv"
    region_map = repo / "Data" / "kpx_region_map.json"

    tp = pd.read_csv(temp_pre, nrows=400)
    tp["ts"] = pd.to_datetime(tp["datetime"])
    # Fake KPX: 2023 copy with MWh = power / 10 (calibration should recover ~10x)
    kpx_rows = tp.copy()
    kpx_rows["ts"] = kpx_rows["ts"] - pd.DateOffset(years=1)
    kpx_csv = tmp_path / "kpx_fake.csv"
    kpx_csv.write_text(
        "일자,시간,권역,전력거래량\n",
        encoding="utf-8",
    )
    extra = kpx_rows.assign(
        일자=kpx_rows["ts"].dt.strftime("%Y-%m-%d"),
        시간=kpx_rows["ts"].dt.strftime("%H:%M:%S"),
        권역=kpx_rows["region"].map(
            {"서울시": "수도권", "부산시": "남부", "대전시": "중부", "강원도": "동부"}
        ),
        전력거래량=kpx_rows["power"] / 10.0,
    )[["일자", "시간", "권역", "전력거래량"]]
    extra.to_csv(kpx_csv, mode="a", index=False, header=False)

    out_pq = tmp_path / "kpx_long.parquet"
    long_df = ingest_kpx_csv(kpx_csv, region_map, out_pq, years=[2023])
    assert len(long_df) > 0
    assert set(long_df["region"].unique()) <= {"서울시", "부산시", "대전시", "강원도"}

    merged_path = tmp_path / "hourly_merged.csv"
    merged, meta = merge_hourly(temp_pre, out_pq, merged_path, years_kpx=[2023])
    assert meta["kpx_rows"] > 0
    assert 2023 in meta["years"] or any("23" in str(d) for d in merged["datetime"].head(20))
    assert len(merged) > len(tp)
