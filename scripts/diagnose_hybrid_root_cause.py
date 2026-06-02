#!/usr/bin/env python3
"""Root-cause diagnostics: data fragmentation vs model/serve mismatch."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts" / "seoul"
PARQUET = ART / "seoul_city_hourly.parquet"
CKPT_2021 = ART / "hybrid_seoul_through_2021.pt"
COMPARE_2022 = ART / "model_compare_year_2022.parquet"


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def check_data_integrity(df: pd.DataFrame) -> dict:
    df = df.sort_values("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"])
    expected = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
    have = set(df["ts"])
    missing = expected.difference(have)
    dup = df["ts"].duplicated().sum()
    jumps = df["power"].diff().abs()
    by_year = df.groupby(df["ts"].dt.year)["power"].agg(["count", "mean", "std", "min", "max"])
    # level shift: year-over-year mean change
    yr_means = by_year["mean"]
    yoy = yr_means.diff()
    return {
        "rows": len(df),
        "ts_min": str(df["ts"].min()),
        "ts_max": str(df["ts"].max()),
        "missing_hours": len(missing),
        "duplicate_ts": int(dup),
        "max_hourly_jump": float(jumps.max()),
        "p99_jump": float(jumps.quantile(0.99)),
        "by_year": by_year.round(2).to_dict(),
        "yoy_mean_change": yoy.round(2).to_dict(),
    }


def scaler_stats(df: pd.DataFrame, label: str) -> dict:
    from pipelines.seoul.hybrid_common import FeatureScaler

    s = FeatureScaler.from_frame(df)
    return {
        "label": label,
        "n": len(df),
        "power_mean": round(s.power_mean, 2),
        "power_std": round(s.power_std, 2),
        "temp_mean": round(s.temp_mean, 2),
    }


def teacher_forcing_rmse(
    df: pd.DataFrame,
    checkpoint: Path,
    year: int,
    ctx_year: int,
    *,
    use_inf_scaler: bool,
    max_hours: int = 720,
) -> dict:
    """One-step ahead with TRUE power in window (no roll-forward corruption)."""
    from pipelines.seoul.hybrid_forecast import _load_model_and_encoder, _predict_step
    from pipelines.seoul.hybrid_common import FeatureScaler

    ctx_end = pd.Timestamp(f"{ctx_year}-12-31 23:00:00")
    hist = df[df["ts"] <= ctx_end].copy()
    fut = df[(df["ts"] >= f"{year}-01-01") & (df["ts"] <= f"{year}-12-31")].head(max_hours)
    if len(fut) < 10:
        return {"error": "not enough future rows"}

    device = torch.device("cpu")
    inf_scaler = FeatureScaler.from_frame(hist) if use_inf_scaler else None
    model, scaler, encoder, cfg, ttm_dim = _load_model_and_encoder(
        checkpoint,
        hist,
        "ibm-granite/granite-timeseries-ttm-r2",
        512,
        96,
        device,
        inference_scaler=inf_scaler,
    )
    seq_len = int(cfg["seq_len"])
    work = pd.concat([hist, fut], ignore_index=True)
    preds, actuals = [], []
    for ts in fut["ts"]:
        g = int(work.index[work["ts"] == ts][0])
        # keep actual power in work (teacher forcing)
        pred = _predict_step(model, scaler, encoder, work, g, seq_len, ttm_dim, device, region="서울시")
        preds.append(pred)
        actuals.append(float(work.loc[g, "power"]))
    err = np.array(preds) - np.array(actuals)
    return {
        "mode": "teacher_forcing",
        "use_inference_scaler": use_inf_scaler,
        "hours": len(preds),
        "rmse": float(math.sqrt(np.mean(err**2))),
        "bias_mean": float(np.mean(err)),
        "pred_mean": float(np.mean(preds)),
        "actual_mean": float(np.mean(actuals)),
    }


def roll_forward_first_n(
    df: pd.DataFrame,
    checkpoint: Path,
    year: int,
    ctx_year: int,
    n: int,
    *,
    use_inf_scaler: bool,
) -> dict:
    from pipelines.seoul.hybrid_forecast import roll_forward_forecast

    out, meta = roll_forward_forecast(
        hourly=df,
        checkpoint=checkpoint,
        target_year=year,
        context_through_year=ctx_year,
        max_hours=n,
        model_path="ibm-granite/granite-timeseries-ttm-r2",
        context_length=512,
        prediction_length=96,
        log_every=0,
        use_inference_scaler=use_inf_scaler,
    )
    err = out["power_pred"] - out["power_actual"]
    return {
        "mode": "roll_forward",
        "hours": n,
        "use_inference_scaler": use_inf_scaler,
        "rmse": float(meta["rmse_vs_actual"]),
        "bias_mean": float(err.mean()),
        "pred_mean": float(out["power_pred"].mean()),
        "actual_mean": float(out["power_actual"].mean()),
    }


def ablation_from_compare() -> None:
    if not COMPARE_2022.is_file():
        print("(no model_compare_year_2022.parquet)")
        return
    cmp = pd.read_parquet(COMPARE_2022)
    cols = [c for c in cmp.columns if c.startswith("power_pred")]
    actual = cmp["power_actual"].to_numpy()
    print(f"Compare parquet rows: {len(cmp)}")
    for c in cols:
        p = cmp[c].to_numpy()
        err = p - actual
        print(
            f"  {c}: RMSE={math.sqrt(np.nanmean(err**2)):.1f} "
            f"bias={np.nanmean(err):+.1f} corr_with_actual={np.corrcoef(p, actual)[0,1]:.3f}"
        )
    if "power_pred_hybrid" in cmp.columns and "power_pred_gru_only" in cmp.columns:
        h = cmp["power_pred_hybrid"].to_numpy()
        g = cmp["power_pred_gru_only"].to_numpy()
        print(f"  hybrid vs gru corr: {np.corrcoef(h, g)[0,1]:.3f}")
    # error growth by week
    cmp["week"] = pd.to_datetime(cmp["ts"]).dt.isocalendar().week.astype(int)
    for c in cols:
        w = cmp.groupby("week").apply(
            lambda x: math.sqrt(np.mean((x[c] - x["power_actual"]) ** 2)),
            include_groups=False,
        )
        print(f"  {c} RMSE week1={w.iloc[0]:.0f} week26={w.iloc[25]:.0f} week52={w.iloc[-1]:.0f}")


def checkpoint_scaler_vs_hist(hist: pd.DataFrame, checkpoint: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sc = ckpt["scaler"]
    meta = ckpt.get("train_meta", {})
    n_train = meta.get("n_train", "?")
    print(f"Checkpoint train_meta: {meta}")
    print(
        f"  ckpt scaler: power_mean={sc['power_mean']:.2f} std={sc['power_std']:.2f} "
        f"(fit on first {n_train} rows of context)"
    )
    full = scaler_stats(hist, "hist≤2021 full")
    train85 = scaler_stats(hist.iloc[: int(len(hist) * 0.85)], "hist≤2021 first85%")
    print(f"  {full}")
    print(f"  {train85}")
    print(f"  2022 actual mean power: {hist[hist['ts'].dt.year == 2022]['power'].mean():.2f}")


def main() -> None:
    df = pd.read_parquet(PARQUET)
    df["ts"] = pd.to_datetime(df["ts"])
    seoul = df[df["region"] == "서울시"] if "region" in df.columns else df

    section("A. 데이터 연속성·파편화 (서울 hourly parquet)")
    integrity = check_data_integrity(seoul)
    print(json.dumps(integrity, indent=2, ensure_ascii=False, default=str))
    if integrity["missing_hours"] == 0 and integrity["duplicate_ts"] == 0:
        print("→ 시간축 파편화(결측·중복)는 없음.")
    else:
        print("→ 결측/중복 있음 — 파편화 가설 지지.")

    section("B. 연도별 레벨·온도 (파편화 vs 레짐 변화)")
    yr = seoul.groupby(seoul["ts"].dt.year).agg(
        power_mean=("power", "mean"),
        power_std=("power", "std"),
        temp_mean=("temp", "mean"),
        temp_std=("temp", "std"),
        n=("power", "count"),
    )
    print(yr.round(2).to_string())
    print("\nKPX+proxy 온도: 과거 연도는 실측 기온이 아닌 climatology → temp_std가 낮을 수 있음.")

    section("C. 스케일러 불일치 (train 85% vs full hist vs 2022)")
    hist21 = seoul[seoul["ts"] <= "2021-12-31 23:00:00"]
    checkpoint_scaler_vs_hist(hist21, CKPT_2021)
    print(
        "\n만약 ckpt scaler mean이 2022 실측(478)보다 훨씬 크면(예:590), "
        "정규화 공간에서 '낮은 값' 예측이 원단위 과소예측으로 이어질 수 있음."
    )

    section("D. Teacher forcing vs Roll-forward (첫 720h, 2022)")
    if CKPT_2021.is_file():
        for inf in (False, True):
            try:
                tf = teacher_forcing_rmse(
                    seoul, CKPT_2021, 2022, 2021, use_inf_scaler=inf, max_hours=720
                )
                rf = roll_forward_first_n(
                    seoul, CKPT_2021, 2022, 2021, 720, use_inf_scaler=inf
                )
                print(f"\nuse_inference_scaler={inf}:")
                print(f"  teacher_forcing: {tf}")
                print(f"  roll_forward:    {rf}")
            except Exception as e:
                print(f"  failed inf={inf}: {e}")
        print(
            "\n해석: TF RMSE << RF RMSE → autoregressive 누적이 주원인.\n"
            "      TF도 나쁨 → 스케일러/TTM/헤드 자체 문제."
        )

    section("E. 2022 연간 비교 parquet (이미 저장된 결과)")
    ablation_from_compare()

    section("F. hourly_merged vs parquet (소스 일관성)")
    merged = REPO / "Data" / "hourly_merged.csv"
    if merged.is_file():
        m = pd.read_csv(merged)
        m["ts"] = pd.to_datetime(m["datetime"], format="mixed")
        ms = m[m["region"] == "서울시"].sort_values("ts")
        ps = seoul.sort_values("ts")
        join = ms.merge(ps, on="ts", suffixes=("_csv", "_pq"))
        if len(join):
            diff = (join["power_csv"] - join["power_pq"]).abs()
            print(f"overlap rows: {len(join)}, max |power diff|: {diff.max():.6f}")
            if diff.max() < 1e-3:
                print("→ CSV와 parquet 전력 일치.")
        else:
            print("no overlap on ts")


if __name__ == "__main__":
    main()
