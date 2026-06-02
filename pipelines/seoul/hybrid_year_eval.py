# Copyright contributors to the TSFM project
#
"""Calendar-year evaluation: context through Y-1, score predictions on year Y.

Use when you need MSE on a full target year (not the last 96h of the dataset).

  # 권장: 풀 학습(최종 모델) 후, 연도별 공정 백테스트
  uv run python -m pipelines.seoul.hybrid_train --seoul-only --epochs 8
  uv run python -m pipelines.seoul.hybrid_year_eval --years 2022,2023,2024 --retrain-hybrid

  # --retrain-hybrid: Hybrid를 «문맥 연도(Y-1)»까지만 학습한 체크포인트로 평가 (누수 방지)
  # --no-retrain-hybrid: hybrid_seoul.pt(풀 학습) 재사용 — 빠르지만 2022·2023 MSE는 과대평가 가능

  # Quick smoke (~1 week per year)
  uv run python -m pipelines.seoul.hybrid_year_eval --years 2022,2023,2024 --retrain-hybrid --max-hours 168
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.seoul.config import ARTIFACTS_DIR, HYBRID_CHECKPOINT_SEOUL, SEOUL_CITY_HOURLY_PARQUET
from pipelines.seoul.hybrid_common import FeatureScaler, load_hourly
from pipelines.seoul.hybrid_eval import _metrics
from pipelines.seoul.hybrid_trainer import checkpoint_path_for_context, train_hybrid_through_year
from pipelines.seoul.hybrid_forecast import (
    _attach_calendar,
    _load_model_and_encoder,
    _predict_step,
    _temp_climatology,
)
from pipelines.seoul.hybrid_model import HybridSeqDataset, HybridTSFMGRU


def _year_paths(target_year: int) -> tuple[Path, Path, Path]:
    stem = f"model_compare_year_{target_year}"
    base = ARTIFACTS_DIR / stem
    return base.with_suffix(".json"), base.with_suffix(".parquet"), base.with_suffix(".html")


YEARS_SUMMARY_JSON = ARTIFACTS_DIR / "model_compare_years_summary.json"
YEARS_SUMMARY_HTML = ARTIFACTS_DIR / "model_compare_years_summary.html"


def _future_frame(
    hourly: pd.DataFrame,
    target_year: int,
    context_through: int,
    max_hours: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Index]:
    hourly = hourly.sort_values("ts").reset_index(drop=True)
    hourly["ts"] = pd.to_datetime(hourly["ts"])
    ctx_end = pd.Timestamp(f"{context_through}-12-31 23:00:00")
    hist = hourly[hourly["ts"] <= ctx_end].copy()
    if len(hist) < 200:
        raise ValueError(f"Not enough history through {ctx_end}: {len(hist)} rows")

    clim = _temp_climatology(hist)
    future_ts = pd.date_range(
        f"{target_year}-01-01 00:00:00",
        f"{target_year}-12-31 23:00:00",
        freq="h",
    )
    if max_hours is not None:
        future_ts = future_ts[:max_hours]

    region = str(hist["region"].iloc[0]) if "region" in hist.columns else "서울시"
    future = _attach_calendar(pd.DataFrame({"ts": future_ts}), clim, region)
    future["power"] = np.nan
    work = pd.concat([hist, future], ignore_index=True)
    return hourly, hist, work, future_ts


def roll_forward_naive(
    hourly: pd.DataFrame,
    target_year: int,
    context_through: int,
    max_hours: int | None,
) -> tuple[pd.DataFrame, dict]:
    hourly, hist, work, future_ts = _future_frame(hourly, target_year, context_through, max_hours)
    preds: list[float] = []
    for ts in future_ts:
        g = int(work.index[work["ts"] == ts][0])
        prev = float(work.loc[g - 1, "power"])
        work.loc[g, "power"] = prev
        preds.append(prev)

    out = work.loc[work["ts"].isin(future_ts), ["ts", "power"]].copy()
    out = out.rename(columns={"power": "power_pred_naive"})
    return _attach_actuals(out, hourly, target_year, context_through, len(hist), "naive")


def roll_forward_gru(
    hourly: pd.DataFrame,
    target_year: int,
    context_through: int,
    max_hours: int | None,
    *,
    seq_len: int,
    hidden: int,
    layers: int,
    epochs: int,
    device,
) -> tuple[pd.DataFrame, dict]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    hourly, hist, work, future_ts = _future_frame(hourly, target_year, context_through, max_hours)
    region = str(hist["region"].iloc[0]) if "region" in hist.columns else "서울시"
    scaler = FeatureScaler.from_frame(hist)
    train_ds = HybridSeqDataset(hist, seq_len, scaler, np.zeros((max(len(hist) - seq_len, 0), 1)), region=region)
    if len(train_ds) == 0:
        raise ValueError("GRU train_ds empty; shorten --seq-len or add history")

    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    model = HybridTSFMGRU(
        input_dim=HybridSeqDataset.BASE_DIM,
        ttm_emb_dim=1,
        hidden=hidden,
        layers=layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for x, emb, y in loader:
            x = x.float().to(device)
            emb = emb.float().to(device)
            y = y.float().to(device)
            opt.zero_grad()
            loss_fn(model(x, emb), y).backward()
            opt.step()
    model.eval()

    preds: list[float] = []
    for ts in future_ts:
        g = int(work.index[work["ts"] == ts][0])
        pred = _predict_step(model, scaler, None, work, g, seq_len, 1, device, region=region)
        work.loc[g, "power"] = pred
        preds.append(pred)

    out = work.loc[work["ts"].isin(future_ts), ["ts", "power"]].copy()
    out = out.rename(columns={"power": "power_pred_gru_only"})
    return _attach_actuals(out, hourly, target_year, context_through, len(hist), "gru_only")


def roll_forward_hybrid(
    hourly: pd.DataFrame,
    checkpoint: Path,
    target_year: int,
    context_through: int,
    max_hours: int | None,
    model_path: str,
    context_length: int,
    prediction_length: int,
) -> tuple[pd.DataFrame, dict]:
    from pipelines.seoul.hybrid_forecast import roll_forward_forecast

    forecast, meta = roll_forward_forecast(
        hourly=hourly,
        checkpoint=checkpoint,
        target_year=target_year,
        context_through_year=context_through,
        max_hours=max_hours,
        model_path=model_path,
        context_length=context_length,
        prediction_length=prediction_length,
    )
    out = forecast[["ts", "power_pred"]].rename(columns={"power_pred": "power_pred_hybrid"})
    if "power_actual" in forecast.columns:
        out["power_actual"] = forecast["power_actual"]
    meta["model"] = "hybrid"
    return out, meta


def _attach_actuals(
    out: pd.DataFrame,
    hourly: pd.DataFrame,
    target_year: int,
    context_through: int,
    context_rows: int,
    model_key: str,
) -> tuple[pd.DataFrame, dict]:
    actual = hourly[hourly["ts"].isin(out["ts"])][["ts", "power"]].rename(columns={"power": "power_actual"})
    if len(actual):
        out = out.merge(actual, on="ts", how="left")
        err = out.filter(regex="^power_pred").iloc[:, 0] - out["power_actual"]
        mse = float(np.nanmean(err**2))
        rmse = float(math.sqrt(mse))
    else:
        mse = rmse = None

    return out, {
        "model": model_key,
        "target_year": target_year,
        "context_through_year": context_through,
        "context_rows": context_rows,
        "forecast_hours": len(out),
        "mse_vs_actual": mse,
        "rmse_vs_actual": rmse,
    }


def _merge_forecasts(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0][["ts"]].copy()
    for f in frames:
        cols = [c for c in f.columns if c != "ts"]
        for c in cols:
            if c not in merged.columns:
                merged = merged.merge(f[["ts", c]], on="ts", how="left")
    if "power_actual" not in merged.columns:
        for f in frames:
            if "power_actual" in f.columns:
                merged = merged.merge(f[["ts", "power_actual"]], on="ts", how="left")
                break
    return merged.sort_values("ts")


def _write_html(merged: pd.DataFrame, meta: dict, path: Path) -> None:
    import plotly.graph_objects as go

    fig = go.Figure()
    if merged["power_actual"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=merged["ts"],
                y=merged["power_actual"],
                name="실측",
                line=dict(color="#2563eb"),
            )
        )
    styles = [
        ("power_pred_naive", "Naive", "#94a3b8"),
        ("power_pred_gru_only", "GRU only", "#22c55e"),
        ("power_pred_hybrid", "TSFM+GRU", "#f97316"),
    ]
    for col, label, color in styles:
        if col in merged.columns:
            fig.add_trace(
                go.Scatter(x=merged["ts"], y=merged[col], name=label, line=dict(color=color))
            )
    ty = meta["target_year"]
    ctx = meta["context_through_year"]
    title = f"서울 · {ty}년 롤링 예측 (문맥 ≤ {ctx}년)"
    fig.update_layout(title=title, height=520, xaxis_title="시간", yaxis_title="전력")
    rows = meta.get("models", {})
    table = "<table border='1' cellpadding='6' style='border-collapse:collapse'>"
    table += "<tr><th>모델</th><th>MSE</th><th>RMSE</th><th>시간</th></tr>"
    for key, r in rows.items():
        mse = r.get("mse_vs_actual")
        rmse = r.get("rmse_vs_actual")
        mse_s = f"{mse:.1f}" if mse is not None else "—"
        rmse_s = f"{rmse:.2f}" if rmse is not None else "—"
        table += (
            f"<tr><td>{r.get('label', key)}</td>"
            f"<td>{mse_s}</td><td>{rmse_s}</td>"
            f"<td>{r.get('forecast_hours', '')}</td></tr>"
        )
    table += "</table>"
    note = meta.get("note", "")
    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"/>
<title>Year eval {ty}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head><body style="font-family:system-ui;max-width:1100px;margin:24px auto;padding:0 16px">
<h1>{title}</h1>
<p>{note}</p>
{table}
<div id="c"></div>
<script>
var _fig = {fig.to_json()};
Plotly.newPlot('c', _fig.data, _fig.layout, {{responsive: true}});
</script>
</body></html>"""
    path.write_text(html, encoding="utf-8")


def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _write_years_summary(results: list[dict]) -> None:
    """Bar chart: RMSE by model × year."""
    import plotly.graph_objects as go

    years = [r["target_year"] for r in results]
    model_keys = ("naive", "gru_only", "hybrid")
    labels = {"naive": "Naive", "gru_only": "GRU only", "hybrid": "TSFM+GRU"}
    colors = {"naive": "#94a3b8", "gru_only": "#22c55e", "hybrid": "#f97316"}

    fig = go.Figure()
    for key in model_keys:
        rmse_vals = []
        for r in results:
            m = r.get("models", {}).get(key, {})
            rmse_vals.append(m.get("rmse_vs_actual"))
        fig.add_trace(
            go.Bar(
                name=labels[key],
                x=[str(y) for y in years],
                y=rmse_vals,
                marker_color=colors[key],
            )
        )
    fig.update_layout(
        title="연도별 홀드아웃 RMSE (서울, 롤링 1-step)",
        barmode="group",
        xaxis_title="평가 연도 (예측 대상)",
        yaxis_title="RMSE",
        height=480,
    )

    rows_html = ""
    for r in results:
        y = r["target_year"]
        ctx = r["context_through_year"]
        rows_html += f"<tr><td>{y}</td><td>≤{ctx}년 말</td>"
        for key in model_keys:
            m = r.get("models", {}).get(key, {})
            mse = m.get("mse_vs_actual")
            rmse = m.get("rmse_vs_actual")
            mse_s = f"{mse:.0f}" if mse is not None else "—"
            rmse_s = f"{rmse:.1f}" if rmse is not None else "—"
            rows_html += f"<td>{mse_s}</td><td>{rmse_s}</td>"
        rows_html += "</tr>"

    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"/>
<title>Years summary</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head><body style="font-family:system-ui;max-width:1000px;margin:24px auto;padding:16px">
<h1>2022–2024 연도별 모델 비교 요약</h1>
<p>각 연도 Y: 문맥은 <b>Y−1년 12/31 23:00</b>까지, <b>Y년 전체</b>(또는 --max-hours 구간)를
1시간 롤링 예측 후 실측 대비 MSE·RMSE.</p>
<table border="1" cellpadding="6" style="border-collapse:collapse;width:100%">
<tr><th>평가년</th><th>문맥</th>
<th>Naive MSE</th><th>Naive RMSE</th>
<th>GRU MSE</th><th>GRU RMSE</th>
<th>Hybrid MSE</th><th>Hybrid RMSE</th></tr>
{rows_html}
</table>
<div id="c"></div>
<script>var _f={fig.to_json()};Plotly.newPlot('c',_f.data,_f.layout,{{responsive:true}});</script>
</body></html>"""

    summary = _jsonify(
        {
            "years": years,
            "design": "context_through = target_year - 1; hybrid --retrain-hybrid per context year",
            "per_year": results,
        }
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    YEARS_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    YEARS_SUMMARY_HTML.write_text(html, encoding="utf-8")
    print(f"\nSummary → {YEARS_SUMMARY_JSON}\n          {YEARS_SUMMARY_HTML}")


def evaluate_one_year(
    df: pd.DataFrame,
    *,
    target_year: int,
    context_through: int | None,
    checkpoint: Path,
    parquet_path: Path,
    max_hours: int | None,
    model_path: str,
    context_length: int,
    prediction_length: int,
    gru_epochs: int,
    hybrid_epochs: int,
    seq_len: int,
    skip_gru: bool,
    skip_hybrid: bool,
    retrain_hybrid: bool,
    reuse_hybrid_checkpoints: bool,
    max_precompute: int | None,
    device,
) -> dict:
    ctx = context_through if context_through is not None else target_year - 1
    print(f"\n{'=' * 60}\neval: predict {target_year} using context through {ctx}\n{'=' * 60}")

    frames: list[pd.DataFrame] = []
    models_meta: dict[str, dict] = {}

    print("Naive roll-forward...")
    n_df, n_meta = roll_forward_naive(df, target_year, ctx, max_hours)
    frames.append(n_df)
    models_meta["naive"] = {**n_meta, "label": "Naive (직전값)"}

    if not skip_gru:
        print("GRU-only roll-forward (retrain on context history)...")
        g_df, g_meta = roll_forward_gru(
            df,
            target_year,
            ctx,
            max_hours,
            seq_len=seq_len,
            hidden=64,
            layers=2,
            epochs=gru_epochs,
            device=device,
        )
        frames.append(g_df)
        models_meta["gru_only"] = {**g_meta, "label": "GRU only"}

    if not skip_hybrid:
        hybrid_ckpt = checkpoint
        train_info: dict | None = None
        if retrain_hybrid:
            hybrid_ckpt = checkpoint_path_for_context(ctx, ARTIFACTS_DIR)
            if reuse_hybrid_checkpoints and hybrid_ckpt.is_file():
                print(f"Hybrid: reuse checkpoint {hybrid_ckpt}")
            else:
                print(f"Hybrid: train through {ctx} → {hybrid_ckpt}")
                train_info = train_hybrid_through_year(
                    df,
                    ctx,
                    hybrid_ckpt,
                    parquet_for_ttm=parquet_path,
                    model_path=model_path,
                    context_length=context_length,
                    prediction_length=prediction_length,
                    seq_len=seq_len,
                    epochs=hybrid_epochs,
                    max_precompute=max_precompute,
                    device=device,
                )
        else:
            print(f"Hybrid: global checkpoint {hybrid_ckpt} (no per-year retrain)")

        print("Hybrid roll-forward (TTM+GRU)...")
        h_df, h_meta = roll_forward_hybrid(
            df,
            hybrid_ckpt,
            target_year,
            ctx,
            max_hours,
            model_path,
            context_length,
            prediction_length,
        )
        h_meta = {**h_meta, "label": "TSFM+GRU hybrid", "checkpoint": str(hybrid_ckpt)}
        if train_info:
            h_meta["train"] = train_info
        if retrain_hybrid:
            h_meta["fair_backtest"] = True
        frames.append(h_df)
        models_meta["hybrid"] = h_meta

    merged = _merge_forecasts(frames)
    has_actual = bool(
        merged["power_actual"].notna().any() if "power_actual" in merged.columns else False
    )

    note = (
        f"문맥: {ctx}년 12/31 23:00까지. 평가: {target_year}년 {len(merged):,}시간."
        + (" 실측 대비 MSE·RMSE." if has_actual else " 실측 없음.")
    )

    payload = _jsonify(
        {
            "target_year": target_year,
            "context_through_year": ctx,
            "forecast_hours": len(merged),
            "has_actuals": has_actual,
            "hybrid_retrained": retrain_hybrid and not skip_hybrid,
            "note": note,
            "models": models_meta,
        }
    )

    json_path, pq_path, html_path = _year_paths(target_year)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    merged.to_parquet(pq_path, index=False)
    _write_html(merged, payload, html_path)

    print(f"Wrote {json_path}\nWrote {pq_path}\nWrote {html_path}")
    print(f"{'Model':28s}  {'MSE':>10s}  {'RMSE':>8s}")
    for key in ("naive", "gru_only", "hybrid"):
        if key not in models_meta:
            continue
        r = models_meta[key]
        mse, rmse = r.get("mse_vs_actual"), r.get("rmse_vs_actual")
        print(f"  {r['label']:28s}  {mse if mse is not None else 'n/a':>10}  {rmse if rmse is not None else 'n/a':>8}")

    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Year-aligned rolling forecast + MSE vs actuals.")
    parser.add_argument("--parquet", type=Path, default=SEOUL_CITY_HOURLY_PARQUET)
    parser.add_argument("--checkpoint", type=Path, default=HYBRID_CHECKPOINT_SEOUL)
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated target years, e.g. 2022,2023,2024 (context = year-1 each)",
    )
    parser.add_argument("--target-year", type=int, default=2024, help="Single year if --years omitted")
    parser.add_argument("--context-through", type=int, default=None, help="History ends this year (default: target-1)")
    parser.add_argument("--max-hours", type=int, default=None, help="Cap hours per year (debug; omit for full year)")
    parser.add_argument("--model-path", type=str, default="ibm-granite/granite-timeseries-ttm-r2")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--gru-epochs", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=168)
    parser.add_argument("--skip-gru", action="store_true")
    parser.add_argument("--skip-hybrid", action="store_true")
    parser.add_argument(
        "--retrain-hybrid",
        action="store_true",
        help="Train Hybrid on data ≤ context year only (fair backtest; default with --years)",
    )
    parser.add_argument(
        "--no-retrain-hybrid",
        action="store_true",
        help="Use --checkpoint (full hybrid_train) for all years",
    )
    parser.add_argument(
        "--reuse-hybrid-checkpoints",
        action="store_true",
        help="Skip retrain if artifacts/seoul/hybrid_seoul_through_{Y}.pt exists",
    )
    parser.add_argument("--hybrid-epochs", type=int, default=8, help="Epochs per year when --retrain-hybrid")
    parser.add_argument(
        "--max-precompute",
        type=int,
        default=None,
        help="Cap TTM embedding windows during per-year train (debug)",
    )
    args = parser.parse_args(argv)

    if not args.parquet.is_file():
        raise SystemExit(f"Missing {args.parquet}")

    retrain_hybrid = args.retrain_hybrid and not args.no_retrain_hybrid
    if args.years and not args.no_retrain_hybrid and not args.retrain_hybrid:
        retrain_hybrid = True
        print("Note: --years defaults to --retrain-hybrid (fair). Use --no-retrain-hybrid to skip.")

    if not args.skip_hybrid and not retrain_hybrid and not args.checkpoint.is_file():
        raise SystemExit(f"Missing {args.checkpoint}. Run hybrid_train or pass --retrain-hybrid.")

    df = load_hourly(args.parquet)
    print(f"data: {df['ts'].min()} .. {df['ts'].max()} ({len(df):,} rows)")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.years:
        years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    else:
        years = [args.target_year]

    results: list[dict] = []
    for year in years:
        ctx_override = args.context_through if len(years) == 1 else None
        payload = evaluate_one_year(
            df,
            target_year=year,
            context_through=ctx_override,
            checkpoint=args.checkpoint,
            parquet_path=args.parquet,
            max_hours=args.max_hours,
            model_path=args.model_path,
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            gru_epochs=args.gru_epochs,
            hybrid_epochs=args.hybrid_epochs,
            seq_len=args.seq_len,
            skip_gru=args.skip_gru,
            skip_hybrid=args.skip_hybrid,
            retrain_hybrid=retrain_hybrid,
            reuse_hybrid_checkpoints=args.reuse_hybrid_checkpoints,
            max_precompute=args.max_precompute,
            device=device,
        )
        results.append(payload)

    if len(results) > 1:
        _write_years_summary(results)
        print(
            "\n→ 발표: 2022·2023·2024 각각 «직전 연도까지 학습·입력 후 해당 연도 예측»."
            + (" Hybrid 연도별 재학습 적용." if retrain_hybrid else " (글로벌 체크포인트 — 참고용)")
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
