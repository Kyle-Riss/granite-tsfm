# Copyright contributors to the TSFM project
#
"""Input scenarios for explaining hypotheses with model and sensitivity outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipelines.seoul.config import (
    HYBRID_CHECKPOINT_SEOUL,
    SEOUL_REGION_LABEL,
)
from pipelines.seoul.hybrid_common import FeatureScaler, load_hourly_for_ttm
from pipelines.seoul.hybrid_model import FrozenTTMEncoder, HybridSeqDataset, HybridTSFMGRU


@dataclass(frozen=True)
class ScenarioPreset:
    id: str
    title: str
    hypothesis: str
    month: int
    target_hour: int
    is_weekend: int
    baseline_temp: float
    scenario_temp: float | None = None
    temp_delta: float | None = None
    shock_hours: int = 12
    policy_tag: str = "수요 관리"

    @property
    def scenario_delta(self) -> float:
        if self.temp_delta is not None:
            return float(self.temp_delta)
        if self.scenario_temp is None:
            return 0.0
        return float(self.scenario_temp - self.baseline_temp)


DEFAULT_PRESETS: tuple[ScenarioPreset, ...] = (
    ScenarioPreset(
        id="heat_weekday_afternoon",
        title="폭염 평일 오후",
        hypothesis="3.1 기온 임계치 · 3.3 평일 낮 피크",
        month=8,
        target_hour=14,
        is_weekend=0,
        baseline_temp=26.0,
        scenario_temp=33.0,
        shock_hours=12,
        policy_tag="피크 저감",
    ),
    ScenarioPreset(
        id="cold_weekday_morning",
        title="한파 출근 시간",
        hypothesis="3.1 난방 임계치 · 3.3 평일 출근 시간",
        month=1,
        target_hour=8,
        is_weekend=0,
        baseline_temp=2.0,
        scenario_temp=-8.0,
        shock_hours=12,
        policy_tag="예비력",
    ),
    ScenarioPreset(
        id="heat_weekend_afternoon",
        title="폭염 주말 오후",
        hypothesis="3.1 냉방 수요 · 3.3 주말 완만 패턴",
        month=8,
        target_hour=14,
        is_weekend=1,
        baseline_temp=26.0,
        scenario_temp=33.0,
        shock_hours=12,
        policy_tag="주거 수요",
    ),
    ScenarioPreset(
        id="regional_plus5",
        title="4개 지역 동일 +5도",
        hypothesis="3.2 지역별 민감도 차이",
        month=7,
        target_hour=15,
        is_weekend=0,
        baseline_temp=28.0,
        temp_delta=5.0,
        shock_hours=12,
        policy_tag="지역 비교",
    ),
)


def _select_base_index(df: pd.DataFrame, preset: ScenarioPreset, seq_len: int) -> int:
    cand = df[
        (df["region"] == SEOUL_REGION_LABEL)
        & (df["month"] == preset.month)
        & (df["hour"] == preset.target_hour)
        & (df["is_weekend"] == preset.is_weekend)
    ].copy()
    cand = cand[cand.index >= seq_len]
    if cand.empty:
        raise ValueError(f"No Seoul context rows found for scenario {preset.id}")
    temp_gap = (cand["temp"] - preset.baseline_temp).abs()
    return int(temp_gap.idxmin())


def _scenario_context(df: pd.DataFrame, end_index: int, seq_len: int, preset: ScenarioPreset) -> pd.DataFrame:
    start = end_index - seq_len
    hist = df.iloc[start:end_index].copy()
    if len(hist) != seq_len:
        raise ValueError(f"Need {seq_len} history rows before index {end_index}, got {len(hist)}")

    n = min(preset.shock_hours, len(hist))
    if preset.scenario_temp is not None:
        hist.loc[hist.index[-n:], "temp"] = float(preset.scenario_temp)
    elif preset.temp_delta is not None:
        hist.loc[hist.index[-n:], "temp"] = hist.loc[hist.index[-n:], "temp"] + float(preset.temp_delta)
    return hist


def _predict_from_history(
    *,
    model: HybridTSFMGRU,
    scaler: FeatureScaler,
    encoder: FrozenTTMEncoder | None,
    hist: pd.DataFrame,
    target_row: pd.DataFrame,
    ttm_emb_dim: int,
    device: Any,
) -> float:
    import torch

    win_df = pd.concat([hist, target_row], ignore_index=True)
    tmp_ds = HybridSeqDataset(
        win_df,
        len(hist),
        scaler,
        np.zeros((1, ttm_emb_dim), dtype=np.float32),
        region=SEOUL_REGION_LABEL,
    )
    x, _, _ = tmp_ds[0]
    if encoder is not None:
        emb = encoder.encode_window(load_hourly_for_ttm(hist)).numpy().astype(np.float32)
    else:
        emb = np.zeros(ttm_emb_dim, dtype=np.float32)
    with torch.no_grad():
        pred_norm = model(
            x.unsqueeze(0).to(device),
            torch.from_numpy(emb).unsqueeze(0).to(device),
        )
    return float(pred_norm.item()) * scaler.power_std + scaler.power_mean


def _load_hybrid_stack(
    *,
    checkpoint: Path,
    train_df: pd.DataFrame,
    model_path: str,
    context_length: int,
    prediction_length: int,
) -> tuple[HybridTSFMGRU, FeatureScaler, FrozenTTMEncoder | None, dict[str, Any], Any]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["hybrid_config"]
    scaler = FeatureScaler(**ckpt["scaler"])
    model = HybridTSFMGRU(
        input_dim=int(cfg["base_dim"]),
        ttm_emb_dim=int(cfg["ttm_emb_dim"]),
        hidden=int(cfg["hidden"]),
        layers=int(cfg["layers"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    encoder = None
    if not bool(cfg.get("skip_ttm", False)):
        encoder = FrozenTTMEncoder(
            model_path,
            context_length,
            prediction_length,
            load_hourly_for_ttm(train_df),
            device,
        )
    return model, scaler, encoder, cfg, device


def _slope(x: pd.Series, y: pd.Series) -> float | None:
    if len(x) < 30 or float(x.std()) < 1e-9:
        return None
    return float(np.polyfit(x.to_numpy(), y.to_numpy(), 1)[0])


def _band_for_temp(temp: float) -> str:
    if temp >= 26:
        return "cooling"
    if temp <= 10:
        return "heating"
    return "mid"


def _regional_stats(hourly: pd.DataFrame, preset: ScenarioPreset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_temp = (
        float(preset.scenario_temp)
        if preset.scenario_temp is not None
        else float(preset.baseline_temp + preset.scenario_delta)
    )
    band = _band_for_temp(target_temp)
    for region in sorted(hourly["region"].dropna().unique(), key=str):
        g = hourly[hourly["region"] == region]
        base = g[
            (g["month"] == preset.month)
            & (g["hour"] == preset.target_hour)
            & (g["is_weekend"] == preset.is_weekend)
        ]
        if base.empty:
            base = g[g["hour"] == preset.target_hour]
        band_g = g[g["temp_band"] == band] if "temp_band" in g.columns else g
        slope = _slope(band_g["temp"], band_g["power"])
        baseline = float(base["power"].mean()) if len(base) else float(g["power"].mean())
        delta = (slope or 0.0) * preset.scenario_delta
        rows.append(
            {
                "region": str(region),
                "baseline_mean": baseline,
                "scenario_estimate": baseline + delta,
                "delta": delta,
                "band": band,
                "slope": slope,
            }
        )
    rows.sort(key=lambda r: abs(float(r["delta"])), reverse=True)
    return rows


def _verdict(preset: ScenarioPreset, seoul_delta: float | None, regional: list[dict[str, Any]]) -> str:
    if preset.id == "regional_plus5" and regional:
        top = regional[0]
        return (
            f"동일 +5도 조건에서 통계 민감도 변화폭이 가장 큰 지역은 {top['region']}입니다. "
            "따라서 3.2는 '서울이 항상 최대'로 단정하기보다 지역별 민감도 차이를 비교하는 가설로 해석하는 편이 안전합니다."
        )
    if seoul_delta is None:
        return "모델 예측값을 계산하지 못해 통계 민감도 기준으로만 해석합니다."
    if preset.scenario_delta < 0 and seoul_delta < 0:
        top = regional[0]["region"] if regional else "일부 지역"
        return (
            f"서울 TSFM+GRU 예측은 기준 문맥 대비 {abs(seoul_delta):.1f}만큼 감소했습니다. "
            "따라서 서울의 한파 입력은 단기 모델에서는 난방 수요 증가 가설을 강하게 지지하지 않습니다. "
            f"다만 지역 통계 민감도에서는 {top}의 변화폭이 크게 나타나 3.1·3.2를 함께 해석해야 합니다."
        )
    direction = "증가" if seoul_delta >= 0 else "감소"
    return (
        f"서울 TSFM+GRU 예측은 기준 문맥 대비 {abs(seoul_delta):.1f}만큼 {direction}했습니다. "
        f"이는 {preset.hypothesis} 입력 조건의 점검 결과입니다."
    )


def build_scenario_payload(
    *,
    hourly: pd.DataFrame,
    sensitivity: dict | None = None,
    checkpoint: Path = HYBRID_CHECKPOINT_SEOUL,
    model_path: str = "ibm-granite/granite-timeseries-ttm-r2",
    context_length: int = 512,
    prediction_length: int = 96,
    train_frac: float = 0.85,
    presets: tuple[ScenarioPreset, ...] = DEFAULT_PRESETS,
) -> dict[str, Any]:
    """Return precomputed scenario results for the HTML report."""

    if hourly.empty:
        return {"available": False, "error": "regional hourly data is empty", "scenarios": []}

    seoul = hourly[hourly["region"] == SEOUL_REGION_LABEL].sort_values("ts").reset_index(drop=True)
    if seoul.empty:
        return {"available": False, "error": "Seoul hourly rows not found", "scenarios": []}

    model_bundle = None
    model_error = None
    if checkpoint.is_file():
        try:
            n_train = int(len(seoul) * train_frac)
            model_bundle = _load_hybrid_stack(
                checkpoint=checkpoint,
                train_df=seoul.iloc[:n_train].copy(),
                model_path=model_path,
                context_length=context_length,
                prediction_length=prediction_length,
            )
        except Exception as exc:  # noqa: BLE001 - report should still render with stats.
            model_error = str(exc)
    else:
        model_error = f"Missing checkpoint: {checkpoint}"

    scenarios: list[dict[str, Any]] = []
    for preset in presets:
        regional = _regional_stats(hourly, preset)
        seoul_result = None
        if model_bundle is not None:
            model, scaler, encoder, cfg, device = model_bundle
            seq_len = int(cfg["seq_len"])
            ttm_emb_dim = int(cfg["ttm_emb_dim"])
            end_index = _select_base_index(seoul, preset, seq_len)
            target_row = seoul.iloc[[end_index]].copy()
            baseline_hist = seoul.iloc[end_index - seq_len:end_index].copy()
            scenario_hist = _scenario_context(seoul, end_index, seq_len, preset)
            baseline_pred = _predict_from_history(
                model=model,
                scaler=scaler,
                encoder=encoder,
                hist=baseline_hist,
                target_row=target_row,
                ttm_emb_dim=ttm_emb_dim,
                device=device,
            )
            scenario_pred = _predict_from_history(
                model=model,
                scaler=scaler,
                encoder=encoder,
                hist=scenario_hist,
                target_row=target_row,
                ttm_emb_dim=ttm_emb_dim,
                device=device,
            )
            seoul_result = {
                "base_datetime": str(target_row["ts"].iloc[0]),
                "actual_power": float(target_row["power"].iloc[0]),
                "actual_temp": float(target_row["temp"].iloc[0]),
                "baseline_prediction": baseline_pred,
                "scenario_prediction": scenario_pred,
                "delta": scenario_pred - baseline_pred,
                "shock_hours": preset.shock_hours,
            }

        delta = seoul_result["delta"] if seoul_result else None
        scenarios.append(
            {
                "id": preset.id,
                "title": preset.title,
                "hypothesis": preset.hypothesis,
                "policy_tag": preset.policy_tag,
                "input": {
                    "month": preset.month,
                    "target_hour": preset.target_hour,
                    "is_weekend": bool(preset.is_weekend),
                    "baseline_temp": preset.baseline_temp,
                    "scenario_temp": preset.scenario_temp,
                    "temp_delta": preset.scenario_delta,
                    "shock_hours": preset.shock_hours,
                },
                "seoul_model": seoul_result,
                "regional_stats": regional,
                "verdict": _verdict(preset, delta, regional),
            }
        )

    return {
        "available": True,
        "model_error": model_error,
        "sensitivity_summary": (sensitivity or {}).get("summary"),
        "scenarios": scenarios,
    }
