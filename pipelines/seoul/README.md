# Seoul Energy Pipeline — 재현 가이드

KPX 지역별 전력거래량 + 기상 데이터 기반 분석 및 예측 파이프라인.

---

## 환경 설정

```bash
# 의존성 설치 (uv 권장)
uv pip install -e ".[dev]"

# 또는 pip
pip install -e ".[dev]"
```

---

## 전체 파이프라인 순서

```
① 원천 데이터 수집
② 데이터 정제 · 병합
③ 가설 분석 (H1~H3)
④ 모델 학습
⑤ 평가 · 백테스트
⑥ 시각화 (PPT 그림 생성)
```

---

## Step 1 — 원천 데이터 수집

### KPX 전력거래량 (2001–2024)

[공공데이터포털](https://www.data.go.kr/data/15133498/fileData.do)에서
`한국전력거래소_지역별 시간대별 전력거래량` CSV 24개 다운로드 →
`notebooks/한국전력거래소_.../` 에 저장

### 기온 데이터 (Open-Meteo)

```bash
python -m pipelines.seoul.fetch_meteo_temp
# → Data/meteo/hourly_temp_{지역}.parquet 생성
```

> 계획서 상 KMA ASOS 사용 예정이었으나, 실제 구현은 Open-Meteo Archive API 사용
> (동등 품질, 무료 라이선스)

---

## Step 2 — 데이터 정제 · 병합

```bash
# KPX CSV 수집 → kpx_hourly_long.parquet
python -m pipelines.seoul.ingest_kpx

# KPX + 기온 시간별 병합 → Temp-pre.csv (hourly_merged.csv)
python -m pipelines.seoul.merge_hourly

# 피처 엔지니어링 (공휴일, 계절, CDD/HDD 등)
python -m pipelines.seoul.feature_enrich

# parquet 변환 (DuckDB 기반)
python -m pipelines.seoul.materialize
# → artifacts/seoul/regional_hourly_34k.parquet
# → artifacts/seoul/seoul_city_hourly.parquet
# → artifacts/seoul/seoul_district_monthly.parquet
```

---

## Step 3 — 가설 분석 (H1~H3)

```bash
# 5.1 구간별 상관계수, 지역별 기온 민감도(slope)
python -m pipelines.seoul.correlation_analysis
# → artifacts/seoul/sensitivity_51.json

# H1~H3 통합 리포트 (HTML)
python -m pipelines.seoul.hypothesis_evidence
# → artifacts/seoul/report_hypotheses.html
```

**주요 출력값 (canonical):**

| 지표 | 수치 |
|------|------|
| H1 난방 구간(≤10°C) r | −0.38 |
| H1 냉방 구간(≥26°C) r | +0.18 |
| H1 U형 최솟값 | ≈18°C |
| H2 부산 slope | −32.4 MWh/°C |
| H2 서울 slope | −0.04 MWh/°C |
| H3 평일 피크 | 16시 611 MWh |
| H3 주말 범위 | 510–540 MWh |

---

## Step 4 — 모델 학습

```bash
# Granite TTM-r2 (frozen) + GRU head 학습 (서울 2020–2023)
python -m pipelines.seoul.hybrid_train
# → artifacts/seoul/hybrid_seoul.pt
# → artifacts/seoul/gru_baseline_seoul.pt
```

**모델 구조:** `Granite TTM-r2 (frozen backbone)` + `GRU head (서울 특화 학습)`

> "TTM 전체 파인튜닝" 표현 금지 — frozen backbone + GRU head 학습

---

## Step 5 — 평가 · 연간 백테스트

```bash
# 단기 holdout 평가 (2024-12-28 기준 96h)
python -m pipelines.seoul.hybrid_eval

# 연간 92블록 백테스트 (2024 전체, 8,784h)
python -m pipelines.seoul.hybrid_year_eval
# → artifacts/seoul/model_compare_block_annual_verified.json

# 지역 확장 비교
python -m pipelines.seoul.forecast_region_compare
# → artifacts/seoul/forecast_region_{지역}_2024.parquet
```

### 검증된 수치 (canonical: `model_compare_block_annual_verified.json`)

| 모델 | 연간 RMSE (92블록) | 단기 holdout RMSE |
|------|------------------|-----------------|
| Seasonal 24h (기준선) | **107 MWh** | — |
| **TTM 96h one-shot** | **111 MWh** | 19.9 MWh |
| GRU roll | 159 MWh | 15.5 MWh |
| Hybrid roll | 167 MWh | 15.5 MWh |

> ⚠️ `tsfm_benchmark_2024.md` (TTM 95.1, Seasonal 92.5)는 **이전 예비 벤치마크** 결과.
> 해당 파일은 Hybrid 모델 미포함, 다른 체크포인트 기준으로 생성됨.
> 발표/보고서 인용 시 `model_compare_block_annual_verified.json` 값을 사용할 것.

**1월 명절·한파 구간 (live 4-block):**

| 모델 | RMSE |
|------|------|
| TTM | 135 MWh |
| GRU roll | 578 MWh |
| **격차** | **77% 낮음** |

---

## Step 6 — 시각화 (PPT 그림 생성)

```bash
# PPT 발표용 그림 전체 생성 (12개)
python scripts/generate_ppt_figures.py
# → artifacts/seoul/ppt_figures/*.png

# 서울 자치구 U자 보조 그림 (실소비 기준)
python scripts/generate_district_ushape.py
# → artifacts/seoul/ppt_figures/fig_seoul_district_ushape.png
```

**생성 완료된 그림 목록:**

| 파일명 | 슬라이드 | 내용 |
|--------|---------|------|
| `fig_regions_map.png` | 4 | 4개 지역 지도 |
| `fig_3-1_temp_scatter.png` | 7 | 기온×전력 U자 산점도 |
| `fig_3-2_region_slope.png` | 8 | 지역별 기온 민감도 막대 |
| `fig_3-3_weekday_weekend.png` | 9 | 평일·주말 시간대별 곡선 |
| `fig_model_pipeline.png` | 11 | TTM+GRU 구조도 |
| `fig_oneshot_vs_roll.png` | 12 | one-shot vs roll 개념도 |
| `fig_error_explosion.png` | 13 | 오차 폭발 꺾은선 |
| `fig_block_rmse_comparison.png` | 14 | 연간 비교 차트 |
| `fig_region_compare.png` | 14 | 지역 확장 비교 |
| `fig_holdout_chart.png` | (보조) | 단기 holdout |
| `fig_horizon_curve.png` | (보조) | horizon별 오차 |
| `fig_seoul_district_ushape.png` | 7 (보조) | 실소비 U자 (구별) |

---

## 테스트

```bash
python -m pytest tests/pipelines/ -v
# test_kpx_ingest_merge: PASS
# test_seoul_materialize: SKIP (Data/ fixtures 미존재 시 정상 skip)
```

---

## 주요 아티팩트 경로

```
artifacts/seoul/
├── sensitivity_51.json              # H1~H3 분석 수치
├── model_compare_block_annual_verified.json  # canonical 벤치마크 수치
├── hybrid_seoul.pt                  # 학습된 모델 가중치
├── regional_hourly_34k.parquet      # 4지역 시간별 데이터 (2020–2024)
├── seoul_city_hourly.parquet        # 서울 시간별
├── seoul_district_monthly.parquet   # 서울 구별 월별
└── ppt_figures/                     # 발표용 그림 전체
```

---

## 수치 일관성 체크리스트

발표/보고서 수치 인용 시 아래 파일에서 확인:

- **가설 수치 (H1~H3):** `sensitivity_51.json`
- **모델 성능:** `model_compare_block_annual_verified.json`
- **1월 라이브:** `model_compare_block_annual_verified.json` → `live_4block_jan`
- ~~`tsfm_benchmark_2024.md`~~ — 이전 예비 벤치마크, 인용 금지
