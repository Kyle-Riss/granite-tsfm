"""
Track 1 — Step 1: KPX 원본 CSV 전체 로드 및 정제
출력: analysis/track1/kpx_all_regions.parquet  (2020-2024, 17개 지역)
      analysis/track1/kpx_4regions_with_temp.parquet (기온 병합, 4개 지역)
"""
from pathlib import Path
import glob
import pandas as pd

BASE = Path(__file__).parent.parent.parent
CSV_DIR = BASE / "notebooks" / "한국전력거래소_지역별 시간대별 전력거래량_20241231"
METEO_DIR = BASE / "Data" / "meteo"
OUT = Path(__file__).parent

# ── 1. KPX CSV 전체 로드 (2020-2024)
files = sorted(glob.glob(str(CSV_DIR / "*.csv")))
# 2020~2024 = index 19~23
target_files = [f for f in files if any(yr in f for yr in ["2020","2021","2022","2023","2024"])]
print(f"로드할 파일: {len(target_files)}개")

dfs = []
for f in target_files:
    df = pd.read_csv(f, encoding="cp949")
    df.columns = df.columns.str.strip()
    # 컬럼명 통일 (연도마다 다를 수 있음)
    col_map = {}
    for c in df.columns:
        if "일자" in c or "날짜" in c:
            col_map[c] = "date"
        elif "시간" in c:
            col_map[c] = "hour"
        elif "지역" in c:
            col_map[c] = "region"
        elif "거래량" in c or "MWh" in c:
            col_map[c] = "power_mwh"
    df = df.rename(columns=col_map)
    df = df[["date","hour","region","power_mwh"]].copy()
    dfs.append(df)

kpx = pd.concat(dfs, ignore_index=True)

# ── 2. 타임스탬프 생성
kpx["date"] = pd.to_datetime(kpx["date"])
kpx["ts"] = kpx["date"] + pd.to_timedelta(kpx["hour"].astype(int) - 1, unit="h")
kpx["power_mwh"] = pd.to_numeric(kpx["power_mwh"], errors="coerce")
kpx = kpx.dropna(subset=["power_mwh"])

# 연도 필터 (2020-2024)
kpx = kpx[kpx["date"].dt.year.between(2020, 2024)]

# 지역명 정리
kpx["region"] = kpx["region"].str.strip()

print(f"\n=== KPX 전체 (2020-2024) ===")
print(f"총 행수: {len(kpx):,}")
print(f"지역 수: {kpx['region'].nunique()}개")
print(f"기간: {kpx['ts'].min()} ~ {kpx['ts'].max()}")
print(f"\n지역별 시간당 평균 거래량 (MWh):")
summary = kpx.groupby("region")["power_mwh"].agg(["mean","sum","count"]).round(1)
summary["sum_gwh"] = (summary["sum"] / 1000).round(0)
print(summary.sort_values("mean", ascending=False).to_string())

# ── 3. 발전기반 / 소비전용 분류
GENERATION_BASED = {"충청남도","경상북도","경기도","전라남도","인천시","경상남도","울산시","강원도","전라북도","제주도"}
CONSUMPTION_BASED = {"서울시","부산시","대전시","광주시","대구시","세종시","충청북도"}

kpx["region_type"] = kpx["region"].apply(
    lambda r: "발전기반" if r in GENERATION_BASED else "소비전용"
)

# ── 4. 계절 추가
def get_season(m):
    if m in (12, 1, 2): return "겨울"
    if m in (3, 4, 5):  return "봄"
    if m in (6, 7, 8):  return "여름"
    return "가을"

kpx["month"] = kpx["ts"].dt.month
kpx["hour_of_day"] = kpx["ts"].dt.hour
kpx["weekday"] = kpx["ts"].dt.weekday
kpx["season"] = kpx["month"].apply(get_season)
kpx["is_weekend"] = (kpx["weekday"] >= 5).astype(int)

# ── 5. 저장
out_all = OUT / "kpx_all_regions.parquet"
kpx.to_parquet(out_all, index=False)
print(f"\n→ 저장: {out_all}")

# ── 6. 기온 있는 4개 지역 + 기온 병합
TEMP_REGIONS = {"서울시":"서울시", "부산시":"부산시", "대전시":"대전시", "강원도":"강원도"}
kpx4 = kpx[kpx["region"].isin(TEMP_REGIONS.keys())].copy()

temp_dfs = []
for region, fname_region in TEMP_REGIONS.items():
    tp = METEO_DIR / f"hourly_temp_{fname_region}.parquet"
    if tp.exists():
        tdf = pd.read_parquet(tp)
        tdf["ts"] = pd.to_datetime(tdf["ts"])
        tdf = tdf.rename(columns={"temp_c":"temp"})[["ts","temp"]]
        tdf["region"] = region
        temp_dfs.append(tdf)

temp_all = pd.concat(temp_dfs, ignore_index=True)
kpx4 = kpx4.merge(temp_all, on=["ts","region"], how="left")

# 기온 구간 분류
kpx4["temp_band"] = pd.cut(
    kpx4["temp"],
    bins=[-999, 10, 18, 26, 999],
    labels=["난방(<10°C)", "적정(10-18°C)", "중간(18-26°C)", "냉방(>26°C)"]
)

out_4 = OUT / "kpx_4regions_with_temp.parquet"
kpx4.to_parquet(out_4, index=False)
print(f"→ 저장: {out_4}")
print(f"\n4개 지역 기온 병합 완료: {len(kpx4):,}행")
print(kpx4.groupby("region")[["power_mwh","temp"]].mean().round(2))
