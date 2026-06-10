"""
전력소비량 예측 데모 서버 — FastAPI
실행: cd /path/to/granite-tsfm && uvicorn demo.app:app --reload --port 8000
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

# granite-tsfm 루트를 sys.path에 추가 (pipelines 모듈 임포트용)
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # 로컬 캐시만 사용

# ── 경로 설정
BASE     = _REPO_ROOT
ARTS     = BASE / "artifacts" / "seoul"
DATA     = BASE / "Data"
TPL_DIR  = Path(__file__).parent / "templates"
_INDEX_HTML = TPL_DIR / "index.html"

TTM_MODEL_PATH   = str(ARTS / "ttm_finetuned_seoul" / "model")
GRU_CHECKPOINT   = ARTS / "hybrid_seoul.pt"
REGIONAL_PARQUET = ARTS / "regional_hourly_34k.parquet"
FORECAST_PARQUET = ARTS / "forecast_region_{region}_2024.parquet"

app = FastAPI(title="전력소비량 예측 데모", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── 데이터 캐시 (서버 시작 시 1회 로드)
_cache: dict = {}


def _safe_float(val, ndigits: int = 1):
    """JSON 직렬화 가능한 float (NaN/Inf → None)."""
    if val is None or pd.isna(val):
        return None
    f = float(val)
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, ndigits)


def _series_to_json_list(series, ndigits: int = 1):
    return [_safe_float(x, ndigits) for x in series]


def _load():
    if _cache:
        return
    _cache["regional"] = pd.read_parquet(REGIONAL_PARQUET)
    _cache["regional"]["ts"] = pd.to_datetime(_cache["regional"]["ts"])

    _cache["forecast_regions"] = {}
    for region in ["서울시", "부산시", "대전시", "강원도"]:
        fp = ARTS / f"forecast_region_{region}_2024.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            df["ts"] = pd.to_datetime(df["ts"])
            _cache["forecast_regions"][region] = df

    _cache["district"] = pd.read_parquet(ARTS / "seoul_district_monthly.parquet")

    with open(ARTS / "sensitivity_51.json") as f:
        _cache["sensitivity"] = json.load(f)

    with open(ARTS / "model_compare_block_annual_verified.json") as f:
        _cache["model_results"] = json.load(f)


# ── 모델 캐시 (첫 예측 요청 시 1회 로드, 약 6초 소요)
_model_cache: dict = {}

def _load_model():
    """TTM encoder + GRU head를 로드 (최초 1회)."""
    if _model_cache:
        return _model_cache

    import warnings
    warnings.filterwarnings("ignore")

    from pipelines.seoul.hybrid_model import FrozenTTMEncoder, HybridTSFMGRU
    from pipelines.seoul.hybrid_common import region_one_hot  # noqa: F401

    ckpt = torch.load(GRU_CHECKPOINT, map_location="cpu", weights_only=False)
    cfg  = ckpt["hybrid_config"]

    # 학습 데이터 (TTM preprocessor 초기화용)
    regional = pd.read_parquet(REGIONAL_PARQUET)
    regional["ts"] = pd.to_datetime(regional["ts"])
    train_df = (
        regional[(regional["region"] == "서울시") & (regional["ts"].dt.year <= 2023)]
        [["ts", "power", "temp"]]
        .rename(columns={"ts": "timestamp"})
        .copy()
    )

    encoder = FrozenTTMEncoder(
        model_path=TTM_MODEL_PATH,
        context_length=512,
        prediction_length=96,
        train_df=train_df,
        device=torch.device("cpu"),
    )

    gru = HybridTSFMGRU(
        input_dim=cfg["base_dim"],
        ttm_emb_dim=cfg["ttm_emb_dim"],
        hidden=cfg["hidden"],
        layers=cfg["layers"],
    )
    gru.load_state_dict(ckpt["state_dict"])
    gru.eval()

    _model_cache["encoder"]  = encoder
    _model_cache["gru"]      = gru
    _model_cache["scaler"]   = ckpt["scaler"]
    _model_cache["cfg"]      = cfg
    return _model_cache


def _make_features(seq_df: pd.DataFrame) -> np.ndarray:
    """168시간 컨텍스트 끝부분 → GRU 입력 피처 행렬 (seq_len × base_dim=8)."""
    from pipelines.seoul.hybrid_common import region_one_hot
    ro = region_one_hot("서울시")
    feats = []
    for _, row in seq_df.iterrows():
        ts = row["timestamp"] if "timestamp" in row.index else row.name
        h  = int(ts.hour)
        wd = int(ts.dayofweek)
        mo = int(ts.month)
        feats.append([
            math.sin(2 * math.pi * h / 24),
            math.cos(2 * math.pi * h / 24),
            float(wd >= 5),
            h / 24,
            mo / 12,
            float(ro[0]), float(ro[1]), float(ro[2]),
        ])
    return np.array(feats, dtype=np.float32)


def _run_prediction(ctx_df: pd.DataFrame) -> dict:
    """
    ctx_df: timestamp, power, temp 컬럼 포함 (최소 512행 권장)
    반환:  pred_mwh (float), emb_norm (list), ctx_summary (dict)
    """
    m = _load_model()
    encoder, gru, scaler = m["encoder"], m["gru"], m["scaler"]

    ctx_for_ttm = ctx_df.rename(
        columns={"ts": "timestamp", "power_actual": "power"}
    ).copy()
    if "timestamp" not in ctx_for_ttm.columns and "ts" in ctx_for_ttm.columns:
        ctx_for_ttm = ctx_for_ttm.rename(columns={"ts": "timestamp"})

    # TTM 임베딩
    emb = encoder.encode_window(ctx_for_ttm).unsqueeze(0)   # [1, 192]

    # GRU 피처 (마지막 168 스텝)
    seq = ctx_for_ttm.tail(168).copy()
    if "timestamp" not in seq.columns:
        seq["timestamp"] = seq.index
    x = torch.tensor(_make_features(seq), dtype=torch.float32).unsqueeze(0)  # [1,168,8]

    with torch.no_grad():
        pred_norm = gru(x, emb).item()

    pred_mwh = pred_norm * scaler["power_std"] + scaler["power_mean"]
    return {
        "pred_mwh": round(pred_mwh, 1),
        "emb_l2norm": round(float(emb.norm().item()), 3),
    }


@app.on_event("startup")
async def startup():
    _load()


# ═══════════════════════════════════════
# UI
# ═══════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(_INDEX_HTML))


# ═══════════════════════════════════════
# API — 지역별 시간대별 전력 거래량
# ═══════════════════════════════════════
@app.get("/api/regions")
async def get_regions():
    return {"regions": ["서울시", "부산시", "대전시", "강원도"]}


@app.get("/api/history")
async def get_history(
    region: str = Query("서울시"),
    start: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end:   Optional[str] = Query(None, description="YYYY-MM-DD"),
    resample: str = Query("1h", description="1h | 1D | 1W | 1ME"),
):
    _load()
    df = _cache["regional"]
    df = df[df["region"] == region].copy()

    if start:
        df = df[df["ts"] >= pd.Timestamp(start)]
    if end:
        df = df[df["ts"] <= pd.Timestamp(end)]

    df = df.set_index("ts")[["power", "temp"]]

    if resample != "1h":
        df = df.resample(resample).mean()

    df = df.dropna(subset=["power"]).reset_index()
    df["ts"] = df["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "region": region,
        "count": len(df),
        "timestamps": df["ts"].tolist(),
        "power": _series_to_json_list(df["power"]),
        "temp":  _series_to_json_list(df["temp"], ndigits=2),
    }


# ═══════════════════════════════════════
# API — TTM 모델 예측 vs 실측 (2024)
# ═══════════════════════════════════════
@app.get("/api/forecast")
async def get_forecast(
    region: str = Query("서울시"),
    start: Optional[str] = Query(None),
    end:   Optional[str] = Query(None),
):
    _load()
    fc = _cache["forecast_regions"].get(region)
    if fc is None:
        raise HTTPException(404, f"{region} 예측 데이터 없음")

    df = fc.copy()
    if start:
        df = df[df["ts"] >= pd.Timestamp(start)]
    if end:
        df = df[df["ts"] <= pd.Timestamp(end)]

    df = df.reset_index(drop=True)

    if df.empty:
        return {
            "region": region,
            "count": 0,
            "timestamps": [],
            "actual": [],
            "predicted": [],
            "rmse": None,
        }

    ts = df["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    result = {"region": region, "count": len(df), "timestamps": ts}

    if "power_actual" in df.columns:
        result["actual"] = _series_to_json_list(df["power_actual"])
    if "power_pred" in df.columns:
        result["predicted"] = _series_to_json_list(df["power_pred"])

    rmse = None
    if "power_actual" in df.columns and "power_pred" in df.columns:
        valid = df.dropna(subset=["power_actual", "power_pred"])
        if len(valid):
            diff = valid["power_actual"] - valid["power_pred"]
            rmse = _safe_float((diff ** 2).mean() ** 0.5)
    result["rmse"] = rmse

    return result


# ═══════════════════════════════════════
# API — 기온×전력 산점도 데이터 (H1)
# ═══════════════════════════════════════
@app.get("/api/temp-scatter")
async def get_temp_scatter(
    region: str = Query("서울시"),
    sample: int = Query(2000, description="최대 샘플 수"),
):
    _load()
    df = _cache["regional"]
    df = df[df["region"] == region][["temp", "power", "month", "season"]].dropna()

    if len(df) > sample:
        df = df.sample(sample, random_state=42)

    def season_label(s):
        m = {"spring": "봄", "summer": "여름", "autumn": "가을", "winter": "겨울"}
        return m.get(s, s)

    return {
        "region": region,
        "temp": df["temp"].round(1).tolist(),
        "power": df["power"].round(1).tolist(),
        "season": [season_label(s) for s in df["season"].tolist()],
    }


# ═══════════════════════════════════════
# API — 지역별 기온 민감도 (H2)
# ═══════════════════════════════════════
@app.get("/api/sensitivity")
async def get_sensitivity():
    _load()
    s = _cache["sensitivity"]
    regions = list(s["regions"].keys())
    slopes = [round(s["regions"][r]["slope_temp_power"], 2) for r in regions]
    means  = [round(s["regions"][r]["mean_power"], 1)  for r in regions]

    return {
        "regions": regions,
        "slope_temp_power": slopes,
        "mean_power": means,
        "bands": s.get("bands", {}),
        "global": s.get("global", {}),
    }


# ═══════════════════════════════════════
# API — 평일 vs 주말 시간대별 패턴 (H3)
# ═══════════════════════════════════════
@app.get("/api/weekday-pattern")
async def get_weekday_pattern(region: str = Query("서울시")):
    _load()
    df = _cache["regional"]
    df = df[df["region"] == region][["ts", "power", "is_weekend"]].copy()
    df["hour"] = df["ts"].dt.hour

    weekday = df[df["is_weekend"] == 0].groupby("hour")["power"].mean().round(1)
    weekend = df[df["is_weekend"] == 1].groupby("hour")["power"].mean().round(1)

    return {
        "region": region,
        "hours": list(range(24)),
        "weekday": weekday.reindex(range(24)).fillna(0).tolist(),
        "weekend": weekend.reindex(range(24)).fillna(0).tolist(),
    }


# ═══════════════════════════════════════
# API — 모델 성능 비교
# ═══════════════════════════════════════
@app.get("/api/model-comparison")
async def get_model_comparison():
    _load()
    r = _cache["model_results"]
    blocks = r.get("block_models", {})

    models, rmses, labels = [], [], []
    for key, info in blocks.items():
        models.append(key)
        rmses.append(info["rmse"])
        labels.append(info.get("label", key))

    seasonal_rmse = None
    for v in blocks.values():
        if "seasonal" in v.get("label", "").lower():
            seasonal_rmse = v["rmse"]
            break

    jan = r.get("live_4block_jan", {})

    return {
        "annual": {
            "model_keys": models,
            "labels": labels,
            "rmse": [round(x, 1) for x in rmses],
        },
        "jan_live": {k: round(v["rmse"], 1) for k, v in jan.items() if isinstance(v, dict) and "rmse" in v},
        "seasonal_baseline": round(seasonal_rmse, 1) if seasonal_rmse else 107.2,
        "ttm_vs_gru_pct": round((1 - 111 / 159) * 100, 0),
    }


# ═══════════════════════════════════════
# API — 서울 자치구별 월별 에너지 사용량
# ═══════════════════════════════════════
@app.get("/api/district-monthly")
async def get_district_monthly():
    _load()
    df = _cache["district"]
    grouped = {}
    for _, row in df.iterrows():
        d = row["district"]
        m = int(row["month"])
        u = round(float(row["usage"]) / 1e6, 3)
        grouped.setdefault(d, {})[m] = u

    months = list(range(1, 13))
    districts = sorted(grouped.keys())
    series = {d: [grouped[d].get(m, 0) for m in months] for d in districts}

    return {
        "months": months,
        "districts": districts,
        "usage_gwh": series,
    }


# ═══════════════════════════════════════════════════════
# API — TTM+GRU 실시간 예측 (핵심 데모 엔드포인트)
# ═══════════════════════════════════════════════════════

# 시나리오 프리셋 (쉬운 시연용)
_SCENARIOS = {
    "summer":       {"label": "여름 폭염 (2025-07)",    "ctx_start": "2025-07-01", "ctx_end": "2025-07-21 23:00:00", "pred_date": "2025-07-22"},
    "winter_cold":  {"label": "겨울 한파 (2025-01)",    "ctx_start": "2025-01-01", "ctx_end": "2025-01-21 23:00:00", "pred_date": "2025-01-22"},
    "holiday":      {"label": "명절 연휴 (2025 설날 1/28~30)", "ctx_start": "2025-01-07", "ctx_end": "2025-01-27 23:00:00", "pred_date": "2025-01-28"},
    "spring":       {"label": "봄 (2025-04)",            "ctx_start": "2025-04-01", "ctx_end": "2025-04-21 23:00:00", "pred_date": "2025-04-22"},
    "year_end":     {"label": "연말 (2025-12)",          "ctx_start": "2025-12-01", "ctx_end": "2025-12-21 23:00:00", "pred_date": "2025-12-22"},
    "summer_2026":  {"label": "여름 폭염 (2026-07)",    "ctx_start": "2026-07-01", "ctx_end": "2026-07-21 23:00:00", "pred_date": "2026-07-22"},
    "winter_2026":  {"label": "겨울 한파 (2026-01)",    "ctx_start": "2026-01-01", "ctx_end": "2026-01-21 23:00:00", "pred_date": "2026-01-22"},
    "holiday_2026": {"label": "명절 연휴 (2026 설날 2/16~18)", "ctx_start": "2026-01-26", "ctx_end": "2026-02-15 23:00:00", "pred_date": "2026-02-16"},
    "spring_2026":  {"label": "봄 (2026-04)",            "ctx_start": "2026-04-01", "ctx_end": "2026-04-21 23:00:00", "pred_date": "2026-04-22"},
    "year_end_2026":{"label": "연말 (2026-12)",          "ctx_start": "2026-12-01", "ctx_end": "2026-12-21 23:00:00", "pred_date": "2026-12-22"},
    "summer_2027":  {"label": "여름 폭염 (2027-07)",    "ctx_start": "2027-07-01", "ctx_end": "2027-07-21 23:00:00", "pred_date": "2027-07-22"},
    "winter_2027":  {"label": "겨울 한파 (2027-01)",    "ctx_start": "2027-01-01", "ctx_end": "2027-01-21 23:00:00", "pred_date": "2027-01-22"},
    "holiday_2027": {"label": "명절 연휴 (2027 설날 2/6)",   "ctx_start": "2027-01-16", "ctx_end": "2027-02-05 23:00:00", "pred_date": "2027-02-06"},
    "spring_2027":  {"label": "봄 (2027-04)",            "ctx_start": "2027-04-01", "ctx_end": "2027-04-21 23:00:00", "pred_date": "2027-04-22"},
    "year_end_2027":{"label": "연말 (2027-12)",          "ctx_start": "2027-12-01", "ctx_end": "2027-12-21 23:00:00", "pred_date": "2027-12-22"},
}

# 시나리오별 고정 인사이트 (정책 행동 · 이점 · 위험)
_SCENARIO_INSIGHT: dict[str, dict] = {
    "summer": {
        "alert_level":      "caution",   # normal / caution / warning
        "demand_direction": "up",         # up / down / stable
        "headline":         "냉방 수요 상승 구간 — DR 프로그램 사전 발동 검토 필요",
        "situation":        "서울 7월 평균 기온 25–27°C, 에어컨 사용이 집중되는 시간대. 오후 2–6시 전력 수요가 평시 대비 20% 이상 높아집니다.",
        "policy_actions": [
            "평일 09–18h 수요반응(DR) 프로그램 사전 발동",
            "냉방 집중 자치구(강남·서초) 우선 절전 유도",
            "예비력 50 MWh 이상 선제 확보",
            "산업용 대형 수요처 자발적 감축 협의",
        ],
        "benefit":          "피크 시간대 긴급 전력 조달 비용을 사전 준비로 최소화할 수 있습니다. DR 1회 발동으로 긴급 계통 조작 없이 수급을 안정시킬 수 있습니다.",
        "risk_if_ignored":  "폭염 피크 구간 전력 예비율 하락 → 수급 불안정 → 지역 단전 위험",
        "real_example":     "2024년 8월 실측 평균 674 MWh. TTM 예측 오차 92 MWh(13%). 여름은 예측 정확도가 가장 높은 구간입니다.",
    },
    "winter_cold": {
        "alert_level":      "warning",
        "demand_direction": "up",
        "headline":         "한파 구간 난방 수요 급증 — 예비력 우선 확보 필요",
        "situation":        "서울 1월 평균 기온 -3°C, 전기 난방·열펌프 사용 급증. 1월 평균 거래량 665 MWh로 연중 최고치 수준입니다.",
        "policy_actions": [
            "발전 예비력 조기 확보 — 피크 대비 여유 용량 점검",
            "노원·도봉 등 주거 밀집 지역 난방 부하 사전 모니터링",
            "한파 경보 연계 자동 예비력 발동 체계 준비",
            "취약 계층 거주지 전기 공급 안정성 우선 확보",
        ],
        "benefit":          "한파 도래 4일 전(96h) 예측으로 발전 출력 계획을 미리 상향 조정할 수 있습니다. 당일 긴급 증발 대비 비용이 50% 이상 절감됩니다.",
        "risk_if_ignored":  "예비력 부족 시 전국 수급 비상 → 긴급 절전령 또는 정전 위험",
        "real_example":     "2024년 1월 TTM RMSE 134 MWh vs GRU 578 MWh. 한파·명절 복합 구간에서 TTM이 GRU 대비 77% 낮은 오차를 기록했습니다.",
    },
    "holiday": {
        "alert_level":      "normal",
        "demand_direction": "down",
        "headline":         "명절 연휴 — 수요 급감 예측, 발전 용량 하향 조정 가능",
        "situation":        "설날·추석 등 명절 연휴는 서울 전력 수요가 평시 대비 30–40% 감소합니다. 2024년 설날 첫날(1/27) 실측 432 MWh — 한 달 평균(665 MWh)의 65% 수준.",
        "policy_actions": [
            "연휴 기간 발전 출력 계획 하향 조정 — 과잉 공급 방지",
            "기저 발전(원자력·석탄) 점검·정비 일정 활용",
            "연휴 후 수요 회복에 대비한 단계적 출력 증가 일정 수립",
            "DR 프로그램 일시 중단 → 인센티브 비용 절감",
        ],
        "benefit":          "수요 급감을 미리 알면 불필요한 발전 출력을 줄여 연료비를 절감할 수 있습니다. 정기 점검 일정을 연휴에 맞추면 공급 차질 없이 유지보수가 가능합니다.",
        "risk_if_ignored":  "과잉 공급 지속 → 계통 주파수 불안정 → 발전기 트립 위험",
        "real_example":     "명절 시나리오는 모델 오차율 3%(14 MWh)로 5개 시나리오 중 가장 정확합니다. 명절 패턴을 모델이 잘 학습한 결과입니다.",
    },
    "spring": {
        "alert_level":      "normal",
        "demand_direction": "stable",
        "headline":         "수요 안정 구간 — 설비 점검 및 효율화 최적 타이밍",
        "situation":        "서울 4월 평균 기온 12°C, 냉·난방 수요 모두 낮은 '완충 구간'. 4월 평균 거래량 500 MWh로 연중 최저 수준입니다.",
        "policy_actions": [
            "송전·배전 설비 정기 점검 및 교체 일정 집중 배치",
            "스마트 미터·AMI 시스템 데이터 품질 점검",
            "여름 피크 대비 수요반응 참여 기업 사전 모집",
            "신재생 에너지 연계 테스트 및 계통 안정성 점검",
        ],
        "benefit":          "수요가 낮은 시기에 설비 점검을 집중하면 피크 시즌 가용률이 높아집니다. 여름·겨울 피크 전 DR 모집을 미리 하면 확보 비용이 절감됩니다.",
        "risk_if_ignored":  "봄철 점검 미실시 → 여름 피크 시 설비 고장 발생 위험",
        "real_example":     "봄·가을은 기온-전력 U자 곡선의 최솟값 구간(11°C 부근). 이 시기 DR 준비가 여름 피크 대응의 핵심입니다.",
    },
    "year_end": {
        "alert_level":      "caution",
        "demand_direction": "up",
        "headline":         "연말 겨울 수요 지속 — 연초 예비력 계획 수립 필요",
        "situation":        "서울 12월 평균 기온 -2°C, 평균 거래량 694 MWh로 1월(665 MWh)과 함께 연중 최고치. 크리스마스·연말 행사로 상업지구 수요도 높습니다.",
        "policy_actions": [
            "내년 1월 한파 대비 예비력 계획 선제 수립",
            "연초 설날 연휴 수요 급감 구간 발전 일정 조정",
            "강남·중구 상업지구 야간 조명·냉난방 모니터링 강화",
            "연간 DR 실적 평가 → 내년 인센티브 구조 개선",
        ],
        "benefit":          "12월 수요 패턴을 4일 앞서 파악하면 연초 예산 및 발전 계획을 정확하게 수립할 수 있습니다. 설날 연휴 수요 급감을 미리 대비해 연료비를 절감할 수 있습니다.",
        "risk_if_ignored":  "연초 한파 예비력 미비 → 1월 수급 불안정 → 긴급 전력 시장 비용 급등",
        "real_example":     "2024년 12월 실측 평균 694 MWh, TTM RMSE 108 MWh. 겨울 수요는 비교적 안정적으로 예측되는 구간입니다.",
    },
}


@app.get("/api/backtest")
async def get_backtest():
    """2020~2024 연간 모델 검증 결과 — 미래 예측 신뢰도 근거."""
    p = ARTS / "backtest_summary_2020_2024.json"
    if not p.exists():
        raise HTTPException(404, "백테스트 결과 파일 없음. scripts/generate_backtest_summary.py 실행 필요.")
    return json.loads(p.read_text())


@app.get("/api/predict/scenarios")
async def list_scenarios():
    """사용 가능한 시나리오 목록."""
    return {
        "scenarios": [
            {"id": k, "label": v["label"], "ctx_start": v["ctx_start"],
             "ctx_end": v["ctx_end"], "pred_date": v["pred_date"]}
            for k, v in _SCENARIOS.items()
        ],
        "note": "/api/predict?scenario=summer 처럼 호출하거나 ctx_start/ctx_end를 직접 지정하세요.",
    }


@app.get("/api/predict")
async def predict(
    scenario:   Optional[str] = Query(None, description="summer | winter_cold | holiday | spring | year_end"),
    ctx_start:  Optional[str] = Query(None, description="컨텍스트 시작 YYYY-MM-DD (최소 21일 전)"),
    ctx_end:    Optional[str] = Query(None, description="컨텍스트 종료 YYYY-MM-DD HH:MM (예측 직전까지)"),
    pred_date:  Optional[str] = Query(None, description="예측 기준일 YYYY-MM-DD (ctx_end 다음 날)"),
):
    """
    [INPUT]  ctx_start ~ ctx_end: 과거 전력·기온 데이터 (최소 512시간 = 21일 권장)
    [MODEL]  TTM(frozen) → 512h 임베딩 → GRU head → 1-step 예측
    [OUTPUT] pred_mwh: 예측 전력거래량 (MWh), actual_mwh: 실측값, error_mwh: 오차
    """
    # ── 시나리오 or 직접 입력
    if scenario:
        if scenario not in _SCENARIOS:
            raise HTTPException(400, f"알 수 없는 시나리오: {scenario}. 사용 가능: {list(_SCENARIOS)}")
        s = _SCENARIOS[scenario]
        ctx_start = ctx_start or s["ctx_start"]
        ctx_end   = ctx_end   or s["ctx_end"]
        pred_date = pred_date or s["pred_date"]
    elif not ctx_start or not ctx_end:
        raise HTTPException(400, "scenario 또는 ctx_start+ctx_end 를 지정하세요.")

    _load()

    # ── 컨텍스트 데이터 로드 (2024년이면 실측값, 2023년 이전이면 학습 데이터)
    fc_2024 = _cache["forecast_regions"].get("서울시")
    reg     = _cache["regional"]
    seoul   = reg[reg["region"] == "서울시"].copy()

    ctx_s = pd.Timestamp(ctx_start)
    ctx_e = pd.Timestamp(ctx_end)

    # 데이터 보유 범위: 2020-01-01 ~ 2024-12-31
    DATA_END = pd.Timestamp("2024-12-31 23:00:00")
    future_mode = ctx_s > DATA_END  # 미래 구간 시뮬레이션 모드

    if future_mode:
        # ── 미래 예측 모드: 미래 연도마다 서로 다른 과거 연도의 같은 계절 패턴을 컨텍스트로 사용
        # 2025→2024, 2026→2023, 2027→2022 — 실제 연도 간 변동성이 예측에 반영되도록
        # 2025→2024, 2026→2023, 2027→2022 (그 이후는 2020년까지 내려감)
        source_year = max(2020, min(2024, 2024 - (ctx_s.year - 2025)))
        years_back = ctx_s.year - source_year
        proxy_s = ctx_s - pd.DateOffset(years=years_back)
        proxy_e = ctx_e - pd.DateOffset(years=years_back)
        # 보유 데이터 범위 클램핑
        proxy_e = min(proxy_e, DATA_END)

        if proxy_s.year >= 2024:
            ctx_df = fc_2024[(fc_2024["ts"] >= proxy_s) & (fc_2024["ts"] <= proxy_e)][["ts", "power_actual", "temp"]].copy()
            ctx_df = ctx_df.rename(columns={"power_actual": "power"})
        else:
            ctx_df = seoul[(seoul["ts"] >= proxy_s) & (seoul["ts"] <= proxy_e)][["ts", "power", "temp"]].copy()
        if len(ctx_df) < 168:
            # fallback: 2024년 마지막 512시간
            ctx_df = fc_2024.tail(512)[["ts", "power_actual", "temp"]].copy()
            ctx_df = ctx_df.rename(columns={"power_actual": "power"})
        ctx_df = ctx_df.rename(columns={"ts": "timestamp"})
        sim_note = (
            f"[미래 예측 시뮬레이션] {ctx_s.date()}~{ctx_e.date()} 구간은 보유 데이터 범위를 초과합니다. "
            f"{proxy_s.date()}~{proxy_e.date()} (같은 계절 {source_year}년 실측 패턴)을 컨텍스트로 사용했습니다."
        )
    elif fc_2024 is not None and ctx_s.year >= 2024:
        ctx_df = fc_2024[(fc_2024["ts"] >= ctx_s) & (fc_2024["ts"] <= ctx_e)][["ts","power_actual","temp"]].copy()
        ctx_df = ctx_df.rename(columns={"ts": "timestamp", "power_actual": "power"})
        sim_note = None
    else:
        ctx_df = seoul[(seoul["ts"] >= ctx_s) & (seoul["ts"] <= ctx_e)][["ts","power","temp"]].copy()
        ctx_df = ctx_df.rename(columns={"ts": "timestamp"})
        sim_note = None

    ctx_hours = len(ctx_df)
    if ctx_hours < 168:
        raise HTTPException(400, f"컨텍스트가 너무 짧습니다 ({ctx_hours}h). 최소 168시간(7일) 이상 필요합니다. 2024년 이전 날짜를 입력하거나 시나리오 버튼을 사용하세요.")

    # ── 모델 실행
    try:
        result = _run_prediction(ctx_df)
    except Exception as e:
        raise HTTPException(500, f"모델 오류: {e}")

    pred_mwh = result["pred_mwh"]

    # ── 실측값 (pred_date 하루 평균, 2024년만 가능)
    actual_avg = None
    actual_h0  = None
    if pred_date and fc_2024 is not None:
        pd_ts = pd.Timestamp(pred_date)
        actual_rows = fc_2024[fc_2024["ts"].dt.date == pd_ts.date()]
        if len(actual_rows) > 0:
            actual_avg = round(actual_rows["power_actual"].mean(), 1)
            actual_h0  = round(actual_rows.iloc[0]["power_actual"], 1)

    error_vs_avg = round(abs(pred_mwh - actual_avg), 1) if actual_avg else None
    error_pct    = round(abs(pred_mwh - actual_avg) / actual_avg * 100, 1) if actual_avg else None

    # ── 컨텍스트 요약
    ctx_power_mean = round(ctx_df["power"].mean(), 1)
    ctx_temp_mean  = round(ctx_df["temp"].mean(), 1)
    ctx_power_max  = round(ctx_df["power"].max(), 1)

    # ── 비교 수치
    SEOUL_ANNUAL_AVG = 567.0
    compare_to_avg = round((pred_mwh - SEOUL_ANNUAL_AVG) / SEOUL_ANNUAL_AVG * 100, 1)
    compare_to_ctx = round((pred_mwh - ctx_power_mean) / ctx_power_mean * 100, 1)
    compare_to_avg_str = (f"+{compare_to_avg}%" if compare_to_avg >= 0 else f"{compare_to_avg}%") + " (서울 연평균 대비)"
    compare_to_ctx_str = (f"+{compare_to_ctx}%" if compare_to_ctx >= 0 else f"{compare_to_ctx}%") + " (직전 컨텍스트 대비)"

    # ── 시나리오 인사이트 (프리셋 or 동적 fallback)
    # winter_2026 → winter_cold 처럼 연도 접미사를 떼고 기본 인사이트에 매핑
    insight_key = None
    if scenario:
        base = scenario.rsplit("_20", 1)[0]
        insight_key = {"winter": "winter_cold"}.get(base, base)
    if insight_key and insight_key in _SCENARIO_INSIGHT:
        raw_insight = _SCENARIO_INSIGHT[insight_key]
    else:
        raw_insight = _dynamic_insight(pred_mwh, ctx_temp_mean, ctx_power_mean)

    insight = {
        **raw_insight,
        "compare_to_avg": compare_to_avg_str,
        "compare_to_ctx": compare_to_ctx_str,
    }

    return {
        # ────────── INPUT 요약 ──────────
        "input": {
            "context_start":    ctx_start,
            "context_end":      ctx_end,
            "context_hours":    ctx_hours,
            "context_days":     round(ctx_hours / 24, 1),
            "avg_power_mwh":    ctx_power_mean,
            "avg_temp_c":       ctx_temp_mean,
            "max_power_mwh":    ctx_power_max,
            "description":      (
                f"{ctx_hours}시간({round(ctx_hours/24,1)}일)의 서울 전력거래량·기온 이력 → "
                "TTM이 512h 패턴 임베딩 → GRU가 다음 시점 예측"
            ),
        },
        # ────────── MODEL 내부 ──────────
        "model": {
            "name":               "Granite TTM-r2 (frozen) + GRU head",
            "context_length":     512,
            "prediction_horizon": 96,
            "ttm_emb_l2norm":     result["emb_l2norm"],
            "note": (
                "TTM은 512h를 한 번에 임베딩(오차 누적 없음). "
                "GRU는 그 임베딩 + 시간·요일 피처로 다음 스텝을 예측."
            ),
        },
        # ────────── OUTPUT ──────────
        "output": {
            "pred_date":          pred_date,
            "pred_mwh":           pred_mwh,
            "actual_day_avg_mwh": actual_avg,
            "actual_h0_mwh":      actual_h0,
            "error_mwh":          error_vs_avg,
            "error_pct":          error_pct,
            "interpretation":     _interpret(pred_mwh, ctx_temp_mean, ctx_power_mean),
        },
        # ────────── INSIGHT (신규) ──────────
        "insight": insight,
        # ────────── 미래 시뮬레이션 안내 ──────────
        "simulation": {
            "future_mode": future_mode,
            "note": sim_note,
        } if future_mode else None,
    }


def _dynamic_insight(pred: float, ctx_temp: float, ctx_power: float) -> dict:
    """시나리오 프리셋이 없을 때 기온·수요 수준으로 동적 인사이트 생성."""
    SEOUL_AVG = 567.0
    if ctx_temp >= 24:
        return _SCENARIO_INSIGHT["summer"]
    elif ctx_temp <= 3:
        return _SCENARIO_INSIGHT["winter_cold"]
    elif pred < SEOUL_AVG * 0.85:
        return _SCENARIO_INSIGHT["holiday"]
    elif ctx_temp <= 14:
        return _SCENARIO_INSIGHT["spring"]
    else:
        return {
            "alert_level":      "normal",
            "demand_direction": "stable",
            "headline":         "평시 수요 구간 — 안정적 운영 가능",
            "situation":        "냉·난방 부하가 크지 않은 구간입니다.",
            "policy_actions":   ["현행 예비력 유지", "모니터링 강화"],
            "benefit":          "안정적인 수급 환경에서 비용 최적화 운영이 가능합니다.",
            "risk_if_ignored":  "갑작스러운 기상 변화에 대응 지연 가능",
            "real_example":     "기온 10–20°C 구간이 전력 수요 최저 구간(U자 곡선 바닥)입니다.",
        }


def _interpret(pred: float, ctx_temp: float, ctx_power: float) -> str:
    """예측값을 쉬운 말로 해석."""
    baseline = 567.0   # 서울 2020-2024 시간 평균
    diff = pred - baseline
    direction = "높은" if diff > 0 else "낮은"
    pct = abs(diff) / baseline * 100

    season = ""
    if ctx_temp <= 5:
        season = "겨울 한파 구간 — 난방 수요가 반영된"
    elif ctx_temp >= 24:
        season = "여름 폭염 구간 — 냉방 수요가 반영된"
    elif ctx_temp <= 12:
        season = "봄·가을 전환기 — 수요가 낮은"
    else:
        season = "온화한 날씨 — 평시"

    return (
        f"{season} 예측입니다. "
        f"서울 평균({baseline:.0f} MWh) 대비 {pct:.1f}% {direction} "
        f"{pred:.0f} MWh로 예측됩니다."
    )
