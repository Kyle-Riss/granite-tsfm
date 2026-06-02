# Copyright contributors to the TSFM project
#
"""Roll-forward hourly power forecast (e.g. 2025) using trained TSFM+GRU.

  # 2025 full year (slow: ~8760 TTM encodes on CPU)
  uv run python -m pipelines.seoul.hybrid_forecast --target-year 2025

  # Quick smoke (1 week)
  uv run python -m pipelines.seoul.hybrid_forecast --target-year 2025 --max-hours 168

  # Backtest on 2024 (has actuals for MSE)
  uv run python -m pipelines.seoul.hybrid_forecast --target-year 2024 --context-through 2023
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.seoul.config import (
    ARTIFACTS_DIR,
    FORECAST_SEOUL_FORWARD_HTML,
    FORECAST_SEOUL_FORWARD_JSON,
    FORECAST_SEOUL_FORWARD_PARQUET,
    HYBRID_CHECKPOINT_SEOUL,
    REGIONAL_HOURLY_PARQUET,
    SEOUL_CITY_HOURLY_PARQUET,
    SEOUL_REGION_LABEL,
)
from pipelines.seoul.merge_hourly import REGIONS
from pipelines.seoul.hybrid_common import FeatureScaler, load_hourly, load_hourly_for_ttm
from pipelines.seoul.hybrid_model import FrozenTTMEncoder, HybridSeqDataset, HybridTSFMGRU
from pipelines.seoul.merge_hourly import SEASON_MAP

FORECAST_PARQUET = FORECAST_SEOUL_FORWARD_PARQUET
FORECAST_JSON = FORECAST_SEOUL_FORWARD_JSON
FORECAST_HTML = FORECAST_SEOUL_FORWARD_HTML


def forecast_parquet_path(region: str, target_year: int) -> Path:
    if region == SEOUL_REGION_LABEL and target_year == 2025:
        return FORECAST_SEOUL_FORWARD_PARQUET
    return ARTIFACTS_DIR / f"forecast_region_{region}_{target_year}.parquet"


def _temp_climatology(hist: pd.DataFrame) -> pd.DataFrame:
    return (
        hist.groupby(["month", "hour"], observed=True)["temp"]
        .median()
        .reset_index()
        .rename(columns={"temp": "temp_clim"})
    )


def _attach_calendar(df: pd.DataFrame, clim: pd.DataFrame, region: str) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["ts"].dt.hour.astype(int)
    out["weekday"] = out["ts"].dt.weekday.astype(int)
    out["month"] = out["ts"].dt.month.astype(int)
    out["season"] = out["month"].map(SEASON_MAP).fillna("mid")
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["region"] = region
    out = out.merge(clim, on=["month", "hour"], how="left")
    if "temp" not in out.columns:
        out["temp"] = out["temp_clim"]
    else:
        out["temp"] = out["temp"].fillna(out["temp_clim"])
    out = out.drop(columns=["temp_clim"], errors="ignore")
    return out


def _load_model_and_encoder(
    checkpoint: Path,
    hist_for_ttm: pd.DataFrame,
    model_path: str,
    context_length: int,
    prediction_length: int,
    device,
    *,
    inference_scaler: FeatureScaler | None = None,
) -> tuple[HybridTSFMGRU, FeatureScaler, FrozenTTMEncoder | None, dict, int]:
    import torch

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["hybrid_config"]
    scaler = inference_scaler if inference_scaler is not None else FeatureScaler(**ckpt["scaler"])
    ttm_emb_dim = int(cfg["ttm_emb_dim"])
    model = HybridTSFMGRU(
        input_dim=int(cfg["base_dim"]),
        ttm_emb_dim=ttm_emb_dim,
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
            load_hourly_for_ttm(hist_for_ttm),
            device,
        )
    return model, scaler, encoder, cfg, ttm_emb_dim


def _predict_step(
    model: HybridTSFMGRU,
    scaler: FeatureScaler,
    encoder: FrozenTTMEncoder | None,
    df: pd.DataFrame,
    g: int,
    seq_len: int,
    ttm_emb_dim: int,
    device,
    *,
    region: str,
) -> float:
    import torch

    i = g - seq_len
    if i < 0:
        raise ValueError(f"Need {seq_len} rows before index {g}")
    win_df = df.iloc[i : g + 1]
    tmp_ds = HybridSeqDataset(
        win_df,
        seq_len,
        scaler,
        np.zeros((1, ttm_emb_dim), dtype=np.float32),
        region=region,
    )
    x, _, _ = tmp_ds[0]
    if encoder is not None:
        emb = encoder.encode_batch_indices(df, np.array([g], dtype=np.int64))[0]
    else:
        emb = np.zeros(ttm_emb_dim, dtype=np.float32)
    with torch.no_grad():
        pred_norm = model(
            x.unsqueeze(0).to(device),
            torch.from_numpy(emb).unsqueeze(0).to(device),
        )
    return float(pred_norm.item()) * scaler.power_std + scaler.power_mean


def roll_forward_forecast(
    *,
    hourly: pd.DataFrame,
    checkpoint: Path,
    target_year: int,
    context_through_year: int | None,
    max_hours: int | None,
    model_path: str,
    context_length: int,
    prediction_length: int,
    log_every: int = 250,
    region: str = SEOUL_REGION_LABEL,
    use_inference_scaler: bool = False,
) -> tuple[pd.DataFrame, dict]:
    import torch

    hourly = hourly.sort_values("ts").reset_index(drop=True)
    hourly["ts"] = pd.to_datetime(hourly["ts"])

    ctx_year = context_through_year or (target_year - 1)
    ctx_end = pd.Timestamp(f"{ctx_year}-12-31 23:00:00")
    hist = hourly[hourly["ts"] <= ctx_end].copy()
    if len(hist) < 200:
        raise ValueError(f"Not enough history through {ctx_end}: {len(hist)} rows")

    clim = _temp_climatology(hist)
    future_start = pd.Timestamp(f"{target_year}-01-01 00:00:00")
    future_end = pd.Timestamp(f"{target_year}-12-31 23:00:00")
    future_ts = pd.date_range(future_start, future_end, freq="h")
    if max_hours is not None:
        future_ts = future_ts[: max_hours]

    future = pd.DataFrame({"ts": future_ts})
    future = _attach_calendar(future, clim, region)
    future["power"] = np.nan

    work = pd.concat([hist, future], ignore_index=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inf_scaler = FeatureScaler.from_frame(hist) if use_inference_scaler else None
    model, scaler, encoder, cfg, ttm_emb_dim = _load_model_and_encoder(
        checkpoint,
        hist,
        model_path,
        context_length,
        prediction_length,
        device,
        inference_scaler=inf_scaler,
    )
    seq_len = int(cfg["seq_len"])

    preds: list[float] = []
    n_future = len(future_ts)
    for j, ts in enumerate(future_ts):
        g = int(work.index[work["ts"] == ts][0])
        pred = _predict_step(
            model, scaler, encoder, work, g, seq_len, ttm_emb_dim, device, region=region
        )
        work.loc[g, "power"] = pred
        preds.append(pred)
        if log_every and (j + 1) % log_every == 0:
            print(f"  forecast {j + 1}/{n_future} ({100 * (j + 1) / n_future:.0f}%)", flush=True)

    out = work.loc[
        work["ts"].isin(future_ts), ["ts", "power", "temp", "hour", "month", "is_weekend", "region"]
    ].copy()
    out = out.rename(columns={"power": "power_pred"})

    actual = hourly[hourly["ts"].isin(future_ts)][["ts", "power"]].rename(columns={"power": "power_actual"})
    if len(actual):
        out = out.merge(actual, on="ts", how="left")
        err = out["power_pred"] - out["power_actual"]
        mse = float(np.nanmean(err**2))
        rmse = float(math.sqrt(mse))
    else:
        mse = rmse = None

    meta = {
        "region": region,
        "target_year": target_year,
        "context_through_year": ctx_year,
        "use_inference_scaler": use_inference_scaler,
        "context_rows": len(hist),
        "forecast_hours": n_future,
        "temp_source": f"month×hour median from history ≤ {ctx_year}",
        "checkpoint": str(checkpoint),
        "mse_vs_actual": mse,
        "rmse_vs_actual": rmse,
        "pred_mean": float(out["power_pred"].mean()),
        "pred_min": float(out["power_pred"].min()),
        "pred_max": float(out["power_pred"].max()),
    }
    return out, meta


def _write_html(forecast: pd.DataFrame, meta: dict, path: Path, *, region: str) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast["ts"],
            y=forecast["power_pred"],
            name="예측",
            mode="lines",
            line=dict(color="#f97316"),
        )
    )
    if "power_actual" in forecast.columns and forecast["power_actual"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=forecast["ts"],
                y=forecast["power_actual"],
                name="실측",
                mode="lines",
                line=dict(color="#2563eb"),
            )
        )
    title = f"{region} 전력 롤링 예측 · {meta['target_year']}년"
    if meta.get("rmse_vs_actual") is not None:
        title += f" (RMSE={meta['rmse_vs_actual']:.1f})"
    fig.update_layout(title=title, xaxis_title="시간", yaxis_title="전력 (MWh)", height=520)
    note = (
        f"<p>문맥: <b>{meta['context_through_year']}년 말</b>까지 {meta['context_rows']:,}시간 · "
        f"기온: {meta['temp_source']} · 예측 {meta['forecast_hours']:,}시간 "
        f"(평균 {meta['pred_mean']:.1f} MWh, {meta['pred_min']:.0f}–{meta['pred_max']:.0f})</p>"
    )
    if meta.get("mse_vs_actual") is None:
        note += "<p><i>실측 없음 — 2025 KPX·기온 확보 후 MSE 검증 가능</i></p>"
    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"/>
<title>Forward forecast</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head><body style="font-family:system-ui;max-width:1000px;margin:24px auto;padding:0 16px">
<h1>{region} TSFM+GRU 롤링 예측</h1>
{note}
<div id="chart"></div>
<script>
var data = {fig.to_json()};
Plotly.newPlot('chart', data.data, data.layout, {{responsive: true}});
</script></body></html>"""
    path.write_text(html, encoding="utf-8")


def _filter_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    if "region" not in df.columns:
        return df
    out = df[df["region"] == region].copy()
    return out.drop(columns=["region"], errors="ignore").sort_values("ts").reset_index(drop=True)


def _run_one_region(args: argparse.Namespace, region: str) -> int:
    parquet = args.parquet
    if region != SEOUL_REGION_LABEL and parquet == SEOUL_CITY_HOURLY_PARQUET:
        parquet = REGIONAL_HOURLY_PARQUET
    if not parquet.is_file():
        raise SystemExit(f"Missing {parquet}. Run materialize first.")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing {args.checkpoint}. Run hybrid_train first.")

    df = _filter_region(load_hourly(parquet), region)
    if len(df) < 200:
        raise SystemExit(f"Not enough rows for {region} in {parquet}")

    use_inf = args.per_region_scaler or region != SEOUL_REGION_LABEL
    print(
        f"\n[{region}] history: {df['ts'].min()} .. {df['ts'].max()} ({len(df):,} rows)\n"
        f"  target {args.target_year}, context through "
        f"{args.context_through or args.target_year - 1}"
        + (" · inference scaler" if use_inf else "")
    )
    forecast, meta = roll_forward_forecast(
        hourly=df,
        checkpoint=args.checkpoint,
        target_year=args.target_year,
        context_through_year=args.context_through,
        max_hours=args.max_hours,
        model_path=args.model_path,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        region=region,
        use_inference_scaler=use_inf,
    )

    out_pq = args.output_parquet
    out_json = args.output_json
    out_html = args.output_html
    if args.all_regions or region != SEOUL_REGION_LABEL:
        out_pq = forecast_parquet_path(region, args.target_year)
        out_json = out_pq.with_suffix(".json")
        out_html = out_pq.with_suffix(".html")

    out_pq.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_parquet(out_pq, index=False)
    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_html(forecast, meta, out_html, region=region)

    print(f"Wrote {out_pq}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_html}")
    if meta.get("rmse_vs_actual") is not None:
        print(f"vs actual: MSE={meta['mse_vs_actual']:.1f} RMSE={meta['rmse_vs_actual']:.1f}")
    else:
        print("No actuals for target year — prediction only.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Roll-forward regional power forecast.")
    parser.add_argument("--parquet", type=Path, default=SEOUL_CITY_HOURLY_PARQUET)
    parser.add_argument("--checkpoint", type=Path, default=HYBRID_CHECKPOINT_SEOUL)
    parser.add_argument(
        "--region",
        type=str,
        default=SEOUL_REGION_LABEL,
        choices=REGIONS,
        help="Target region label",
    )
    parser.add_argument(
        "--all-regions",
        action="store_true",
        help="Forecast all four KPX regions (uses regional parquet + per-region scaler)",
    )
    parser.add_argument(
        "--per-region-scaler",
        action="store_true",
        help="Normalize power/temp from this region's history (recommended for non-Seoul)",
    )
    parser.add_argument("--target-year", type=int, default=2025)
    parser.add_argument(
        "--context-through",
        type=int,
        default=None,
        help="History ends this year (default: target_year - 1)",
    )
    parser.add_argument("--max-hours", type=int, default=None, help="Cap forecast length (debug)")
    parser.add_argument("--model-path", type=str, default="ibm-granite/granite-timeseries-ttm-r2")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--output-parquet", type=Path, default=FORECAST_PARQUET)
    parser.add_argument("--output-json", type=Path, default=FORECAST_JSON)
    parser.add_argument("--output-html", type=Path, default=FORECAST_HTML)
    args = parser.parse_args(argv)

    if args.all_regions:
        for reg in REGIONS:
            _run_one_region(args, reg)
        print("\nAll regions done. Build map: uv run python -m pipelines.seoul.forecast_region_compare")
        return 0
    return _run_one_region(args, args.region)


if __name__ == "__main__":
    sys.exit(main())
