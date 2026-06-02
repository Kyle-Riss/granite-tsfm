# Copyright contributors to the TSFM project
#
"""Paths and constants for the Seoul energy pipeline."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "Data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "seoul"

HOURLY_CSV = DATA_DIR / "Temp-pre.csv"
HOURLY_MERGED_CSV = DATA_DIR / "hourly_merged.csv"
KPX_LONG_PARQUET = DATA_DIR / "kpx_hourly_long.parquet"
ASOS_DIR = DATA_DIR / "asos"
METEO_DIR = DATA_DIR / "meteo"
CALENDAR_JSON = DATA_DIR / "calendar" / "kr_holidays.json"
DEFAULT_MIN_TRAIN_YEAR = 2020

# KPX region → ASOS stnId (기상청 종관). Open-Meteo fallback uses lat/lon below.
REGION_ASOS_STATION: dict[str, int] = {
    "서울시": 108,
    "부산시": 159,
    "대전시": 133,
    "강원도": 105,  # 춘천
}
REGION_METEO_COORDS: dict[str, tuple[float, float]] = {
    "서울시": (37.5665, 126.978),
    "부산시": (35.1796, 129.0756),
    "대전시": (36.3504, 127.3845),
    "강원도": (37.8813, 127.7298),
}
DISTRICT_CSV = DATA_DIR / "seoul_energy_use_pre.csv"
POPULATION_CSV = DATA_DIR / "region_population.csv"

REGIONAL_HOURLY_PARQUET = ARTIFACTS_DIR / "regional_hourly_34k.parquet"
SEOUL_CITY_HOURLY_PARQUET = ARTIFACTS_DIR / "seoul_city_hourly.parquet"
DISTRICT_MONTHLY_PARQUET = ARTIFACTS_DIR / "seoul_district_monthly.parquet"

SEOUL_REGION_LABEL = "서울시"
REGIONS = ("서울시", "부산시", "대전시", "강원도")

# Hybrid model (TSFM frozen backbone + GRU head)
HYBRID_CHECKPOINT_SEOUL = ARTIFACTS_DIR / "hybrid_seoul.pt"
HYBRID_METRICS_SEOUL_JSON = ARTIFACTS_DIR / "hybrid_metrics_seoul.json"
HYBRID_HOLDOUT_FORECAST_PARQUET = ARTIFACTS_DIR / "forecast_hybrid_holdout_seoul.parquet"
FORECAST_SEOUL_FORWARD_PARQUET = ARTIFACTS_DIR / "forecast_seoul_forward.parquet"
FORECAST_SEOUL_FORWARD_JSON = ARTIFACTS_DIR / "forecast_seoul_forward.json"
FORECAST_SEOUL_FORWARD_HTML = ARTIFACTS_DIR / "forecast_seoul_forward.html"
FORECAST_REGION_COMPARE_HTML = ARTIFACTS_DIR / "forecast_region_compare.html"
FORECAST_REGION_COMPARE_JSON = ARTIFACTS_DIR / "forecast_region_compare.json"
HYBRID_REPORT_HTML = ARTIFACTS_DIR / "hybrid_report.html"
HYBRID_EMBEDDINGS_NPZ = ARTIFACTS_DIR / "hybrid_ttm_embeddings.npz"
MODEL_COMPARE_JSON = ARTIFACTS_DIR / "model_compare_seoul.json"
MODEL_COMPARE_PLOTS_HTML = ARTIFACTS_DIR / "model_compare_plots.html"


def model_compare_block_json(target_year: int) -> Path:
    return ARTIFACTS_DIR / f"model_compare_block_{target_year}.json"


def model_compare_block_parquet(target_year: int) -> Path:
    return ARTIFACTS_DIR / f"model_compare_block_{target_year}.parquet"


def model_compare_block_html(target_year: int) -> Path:
    return ARTIFACTS_DIR / f"model_compare_block_{target_year}.html"


ROBUSTNESS_JSON = ARTIFACTS_DIR / "robustness_seoul.json"

# Analysis & exploration
SENSITIVITY_51_JSON = ARTIFACTS_DIR / "sensitivity_51.json"
REPORT_HYPOTHESES_HTML = ARTIFACTS_DIR / "report_hypotheses.html"
EXPLORER_HTML = ARTIFACTS_DIR / "explorer.html"
DASHBOARD_ENERGY_HTML = ARTIFACTS_DIR / "dashboard_energy.html"

BASE_TEMP_C = 18.0
COOLING_BAND_TEMP_C = 26.0
HEATING_BAND_TEMP_C = 10.0
