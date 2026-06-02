# Hourly feature enrichment (ASOS / calendar / events)

## Pipeline order

```bash
# 1) KPX long + merge (2019 excluded by default)
uv run python -m pipelines.seoul.merge_hourly --years 2020,2021,2022,2023,2024

# 2a) Hourly temperature — Open-Meteo (no API key)
uv run python -m pipelines.seoul.fetch_meteo_temp --start 2020-01-01 --end 2024-12-31

# 2b) Optional: KMA ASOS (공공데이터포털 ServiceKey)
export KMA_ASOS_SERVICE_KEY='...'
uv run python -m pipelines.seoul.download_asos --start 20200101 --end 20241231

# 3) Enrich CSV + drop year < 2020
uv run python -m pipelines.seoul.enrich_hourly

# 4) Parquet
uv run python -m pipelines.seoul.materialize --hourly-csv Data/hourly_merged.csv
```

## Sources

| Feature | Source |
|---------|--------|
| `temp` (primary) | ASOS if downloaded, else [Open-Meteo Archive](https://open-meteo.com/en/docs/historical-weather-api), else climatology proxy |
| `is_holiday`, `is_holiday_eve` | [holidays.hyunbin.page](https://holidays.hyunbin.page/basic.json) (공휴일법 기준) |
| `is_kpx_peak_hour` | Heuristic: Jun–Aug weekday 14–20h (KPX summer peak pattern) |
| `is_dr_window_proxy` | Heuristic: May–Sep weekday 14–18h (economic DR **notification window**, not dispatch log) |
| `is_heatwave_day` / `is_coldwave_day` | Daily max/min temp thresholds (33°C / -10°C) |
| `seoul_district_cv` | Cross-gu monthly usage CV from `Data/seoul_energy_use_pre.csv` |

## ASOS stations (KMA)

| Region | stnId | Open-Meteo coords |
|--------|-------|-------------------|
| 서울시 | 108 | 37.57°N, 126.98°E |
| 부산시 | 159 | 35.18°N, 129.08°E |
| 대전시 | 133 | 36.35°N, 127.38°E |
| 강원도 | 105 (춘천) | 37.88°N, 127.73°E |

Official API: [기상청 ASOS 시간자료](https://www.data.go.kr/data/15057210/openapi.do)

## 2019

KPX 2019 mean power is ~half of 2020+ (unit/regime). Training/eval default **`min_year=2020`** via `enrich_hourly` and merge defaults.
