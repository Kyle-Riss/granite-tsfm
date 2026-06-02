# Copyright contributors to the TSFM project
#
"""KPX CSV가 없을 때 파이프라인 시연용 2023 전력 CSV 생성 (Temp-pre 2024 → 2023 시프트).

  실제 제출·보고에는 공공데이터 KPX 원본을 사용하세요.
  https://www.data.go.kr/data/15133498/fileData.do

  uv run python -m pipelines.seoul.bootstrap_kpx_demo
  uv run python -m pipelines.seoul.ingest_kpx --input Data/kpx/kpx_hourly_demo.csv --years 2023
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from pipelines.seoul.config import DATA_DIR, HOURLY_CSV

KPX_DIR = DATA_DIR / "kpx"
DEFAULT_OUT = KPX_DIR / "kpx_hourly_demo.csv"

REGION_TO_KPX = {
    "서울시": "수도권",
    "부산시": "남부",
    "대전시": "중부",
    "강원도": "동부",
}


def build_demo_kpx_csv(
    temp_pre_path: Path,
    output_path: Path,
    year_shift: int = 1,
    power_divisor: float = 10.0,
) -> pd.DataFrame:
    """Build long-format KPX-like CSV from Temp-pre (power only, shifted year)."""
    df = pd.read_csv(temp_pre_path)
    df["ts"] = pd.to_datetime(df["datetime"])
    df["ts"] = df["ts"] - pd.DateOffset(years=year_shift)
    out = pd.DataFrame(
        {
            "일자": df["ts"].dt.strftime("%Y-%m-%d"),
            "시간": df["ts"].dt.strftime("%H:%M:%S"),
            "권역": df["region"].map(REGION_TO_KPX),
            "전력거래량": df["power"] / power_divisor,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create demo KPX CSV from Temp-pre for pipeline testing.")
    parser.add_argument("--temp-pre", type=Path, default=HOURLY_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--year-shift", type=int, default=1, help="Shift years back (default 1 → 2023)")
    args = parser.parse_args(argv)

    if not args.temp_pre.is_file():
        raise SystemExit(f"Missing {args.temp_pre}")

    out = build_demo_kpx_csv(args.temp_pre, args.output, year_shift=args.year_shift)
    print(f"Wrote demo KPX CSV: {args.output}  rows={len(out)}")
    print("  ⚠️  시연용입니다. 보고서에는 data.go.kr KPX 원본을 사용하세요.")
    print(f"\nNext:\n  uv run python -m pipelines.seoul.ingest_kpx --input {args.output} --years 2023")
    print("  uv run python -m pipelines.seoul.merge_hourly --years 2023,2024")
    return 0


if __name__ == "__main__":
    sys.exit(main())
