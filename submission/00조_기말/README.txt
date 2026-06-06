조3조 기말 프로젝트 제출물 — 한 폴더 모음
==========================================
과목: 파이썬데이터분석 (서지훈) | 조3조
202004124 하유빈 | 202204173 곽소민 | 202384068 황준연 | 202304226 박선준 | 202301099 공지수

■ 폴더 구조
────────────────────────────────────────────
00조_기말/
├── 00조_기말_프로젝트.ipynb   ← 메인 코드 (샘플 데이터만으로 실행)
├── README.txt                 ← 이 파일
├── data/
│   ├── sample_01_raw_kpx.csv
│   ├── sample_02_merged_hourly.csv
│   ├── sample_03_seoul_district_monthly.csv
│   └── sample_README.txt
├── results/
│   └── backtest_summary_2020_2024.json   ← 2024 모델 검증 수치
└── models/
    └── hybrid_seoul.pt        ← 서울 맞춤 예측기 가중치 (~300KB)

■ 실행 방법
────────────────────────────────────────────
1. 이 폴더(00조_기말)를 작업 디렉터리로 열기
2. pip install pandas matplotlib  (또는 레포 .venv 사용)
3. Jupyter에서 00조_기말_프로젝트.ipynb 전체 실행

■ 전체 코드·재현 (GitHub)
────────────────────────────────────────────
레포: https://github.com/Kyle-Riss/granite-tsfm
- analysis/track1/  — 전국 KPX 분석 + 그림
- analysis/track2/  — 서울 구별 분석
- analysis/track3/  — 모델 성능 요약
- pipelines/seoul/  — 데이터 병합·학습·백테스트 파이프라인
- demo/             — FastAPI 시연 (선택)

■ 분석 주제 (계획서 대응)
────────────────────────────────────────────
지역별·기온별 전력 거래 패턴 분석 + 시계열 AI 4일 예측
- Track 1: 전국 17지역 KPX 거래 구조
- Track 2: 서울 25구 실소비 + 인구
- Track 3: IBM Granite TTM + 서울 맞춤 예측기, 2024 검증

■ 모델 파일 안내
────────────────────────────────────────────
- hybrid_seoul.pt: GRU head 가중치 (제출용 소형)
- TTM 본체(IBM Granite, ~8MB+): zip 미포함 — HuggingFace ibm-granite/granite-timeseries-ttm-r2
- 노트북은 모델 재학습 없이 results/ JSON으로 검증 수치만 표시
