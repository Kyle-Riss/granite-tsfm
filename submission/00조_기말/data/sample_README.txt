조3조 기말 프로젝트 — 데이터 샘플 설명
=====================================
과목: 파이썬데이터분석 | 조3조 | 202004124 하유빈 외

■ 이 폴더에 있는 것 (샘플만, 전체 데이터 아님)
────────────────────────────────────────────
sample_01_raw_kpx.csv
  - 출처: 한국전력거래소(KPX) 지역별 시간대별 전력거래량 (공공데이터포털)
  - URL: https://www.data.go.kr/data/15133498/fileData.do
  - 내용: 2024년 1월 1~7일, 4개 지역 (서울·부산·대전·강원)
  - 컬럼: date, hour, region, power_mwh
  - 의미: 원본 CSV 그대로에 가까운 형태 (RAW)

sample_02_merged_hourly.csv
  - 내용: 같은 기간·같은 4지역, 전력 + 기온 + 시간 피처 병합 (정제 후)
  - 컬럼: ts, region, power, temp, hour, weekday, month, season
  - 기온: Open-Meteo Archive API (계획서 KMA ASOS → 실제 Open-Meteo)
  - 의미: 분석·모델에 실제 사용한 형태 (PROCESSED)

sample_03_seoul_district_monthly.csv
  - 출처: 서울시 에너지정보 (자치구별 월별 사용량)
  - URL: https://energyinfo.seoul.go.kr/
  - 내용: 25개 구 × 12개월 (전체 300행, 용량 작아 샘플=전체)
  - 의미: KPX 거래량과 다른 "실소비량" 데이터 (Track 2)

■ 전체 데이터 (제출 zip에는 미포함)
────────────────────────────────────────────
- KPX 원본: 2001~2024 CSV 24개 (2020~2024만 분석 사용)
- 병합 hourly: artifacts/seoul/regional_hourly_34k.parquet (175,392행)
- 전체 재현: GitHub 레포 + pipelines/seoul/README.md 참고

■ KPX vs 서울 구별 에너지
────────────────────────────────────────────
- KPX = 도매시장 "거래량" (발전소 있으면 숫자 큼)
- 서울 구별 = 실제 "사용량" (소비 기준)
- 둘 다 "전력 소비량"이 아님 — 보고서·발표에서 구분 필요
