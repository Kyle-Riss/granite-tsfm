# KPX 지역별·시간대별 전력거래량

## A) 공공데이터 원본 (보고서·제출용)

1. [한국전력거래소 지역별 시간대별 전력거래량](https://www.data.go.kr/data/15133498/fileData.do) CSV 다운로드
2. 연도별 CSV 폴더 그대로 사용 가능 (예: `한국전력거래소_지역별 시간대별 전력거래량_20241231/*.csv`)
3. 또는 이 폴더에 단일 파일 `kpx_hourly.csv` 로 저장

### 원격으로 받을 수 있나?

- 이 데이터는 **fileData(파일)** 이라 OpenAPI 자동 수집 URL이 **없는 경우가 많습니다**.
- `curl`만으로 포털에서 받으면 **403·로그인 페이지**가 나오는 일이 흔합니다.
- 로그인 후 브라우저 «다운로드» **링크 주소를 복사**하면 시도 가능:

```bash
uv run python -m pipelines.seoul.download_kpx --url '복사한_URL'
```

- [서울시 에너지정보 통계](https://energyinfo.seoul.go.kr/energy/energyUsagePattern?menu-id=Z020400) 도 UI·엑셀 다운로드 방식이라 동일합니다.

```bash
# 다운로드 폴더 전체 (연도별 CSV 자동 병합)
uv run python -m pipelines.seoul.ingest_kpx \
  --input "한국전력거래소_지역별 시간대별 전력거래량_20241231" \
  --years 2019,2020,2021,2022,2023,2024

uv run python -m pipelines.seoul.merge_hourly --years 2019,2020,2021,2022,2023,2024
uv run python -m pipelines.seoul.materialize --hourly-csv Data/hourly_merged.csv
```

`Data/kpx_region_map.json` 에서 KPX 권역명 ↔ `서울시` 등 매핑을 확인하세요.

폴더만 지정하면 **가장 최근 `.csv` 하나**를 자동 사용합니다:

```bash
uv run python -m pipelines.seoul.ingest_kpx --input Data/kpx/
```

## B) KPX 파일이 없을 때 (파이프라인 시연만)

`Temp-pre` 2024를 1년 되돌린 **데모 CSV** (기온·전력 스케일은 merge 단계에서 맞춤):

```bash
uv run python -m pipelines.seoul.bootstrap_kpx_demo
uv run python -m pipelines.seoul.ingest_kpx --input Data/kpx/kpx_hourly_demo.csv --years 2023
uv run python -m pipelines.seoul.merge_hourly --years 2023,2024
```

보고서에는 **반드시 (A) 원본 KPX** 를 쓰고, (B)는 “연도 확장 파이프라인 동작 확인”용으로만 적으세요.
