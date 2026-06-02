# Copyright contributors to the TSFM project
#
"""Long-horizon evaluation: 96h blocks, seasonal baselines, horizon curve.

  uv run python -m pipelines.seoul.hybrid_block_eval --target-year 2024
  uv run python -m pipelines.seoul.hybrid_block_eval --target-year 2024 --max-blocks 4  # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.seoul.config import (
    HYBRID_CHECKPOINT_SEOUL,
    SEOUL_CITY_HOURLY_PARQUET,
    model_compare_block_html,
    model_compare_block_json,
    model_compare_block_parquet,
)
from pipelines.seoul.hybrid_common import load_hourly, load_hourly_for_ttm
from pipelines.seoul.hybrid_eval import (
    HoldoutResult,
    TTMBlockForecaster,
    _metrics,
    eval_gru_segment,
    eval_hybrid_segment,
    eval_seasonal,
    train_gru_only_model,
)
from pipelines.seoul.hybrid_model import FrozenTTMEncoder


def _year_indices(df: pd.DataFrame, target_year: int, context_through: int) -> tuple[int, int, int]:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    ctx_end = pd.Timestamp(f"{context_through}-12-31 23:00:00")
    y0 = pd.Timestamp(f"{target_year}-01-01 00:00:00")
    y1 = pd.Timestamp(f"{target_year}-12-31 23:00:00")
    ctx_mask = df["ts"] <= ctx_end
    year_mask = (df["ts"] >= y0) & (df["ts"] <= y1)
    if not ctx_mask.any():
        raise ValueError(f"No rows through {ctx_end}")
    if not year_mask.any():
        raise ValueError(f"No rows in {target_year}")
    year_idx = np.where(year_mask.to_numpy())[0]
    return int(year_idx[0]), int(year_idx[-1]) + 1, int(ctx_mask.to_numpy().nonzero()[0][-1]) + 1


def _filter_min_year(df: pd.DataFrame, min_year: int | None) -> pd.DataFrame:
    if min_year is None:
        return df
    out = df[pd.to_datetime(df["ts"]).dt.year >= min_year].copy()
    return out.reset_index(drop=True)


def _concat_results(parts: list[HoldoutResult]) -> HoldoutResult:
    ts: list = []
    actual: list[float] = []
    pred: list[float] = []
    for p in parts:
        ts.extend(p.ts)
        actual.extend(p.actual.tolist())
        pred.extend(p.pred.tolist())
    a, pr = np.array(actual), np.array(pred)
    mse, rmse = _metrics(pr, a)
    return HoldoutResult(ts=ts, actual=a, pred=pr, mse=mse, rmse=rmse)


def run_block_backtest(
    df: pd.DataFrame,
    *,
    year_start: int,
    year_end: int,
    block_hours: int,
    max_blocks: int | None,
    ttm: TTMBlockForecaster | None,
    checkpoint: Path,
    encoder: FrozenTTMEncoder | None,
    gru_model,
    gru_scaler,
    gru_seq_len: int,
    device,
    chained: bool,
) -> dict[str, dict]:
    parts: dict[str, list[HoldoutResult]] = {
        "ttm_block": [],
        "hybrid_block_roll": [],
        "hybrid_block_tf": [],
        "gru_block_roll": [],
        "seasonal_24h": [],
        "seasonal_168h": [],
    }
    work_roll = df.copy() if chained else None
    work_gru = df.copy() if chained else None

    block_starts = list(range(year_start, year_end, block_hours))
    if max_blocks:
        block_starts = block_starts[:max_blocks]

    for bi, b0 in enumerate(block_starts):
        b1 = min(b0 + block_hours, year_end)
        h = b1 - b0
        if h <= 0:
            continue
        print(f"  block {bi + 1}/{len(block_starts)}  idx {b0}..{b1 - 1}  ({h}h)", flush=True)

        parts["seasonal_24h"].append(eval_seasonal(df, b0, h, 24))
        parts["seasonal_168h"].append(eval_seasonal(df, b0, h, 168))

        if ttm is not None:
            pred_ttm = ttm.predict(df, b0, h)
            actual = df["power"].iloc[b0:b1].to_numpy(dtype=np.float64)
            ts = df["ts"].iloc[b0:b1].tolist()
            mse, rmse = _metrics(pred_ttm[: len(actual)], actual)
            parts["ttm_block"].append(
                HoldoutResult(ts=ts, actual=actual, pred=pred_ttm[: len(actual)], mse=mse, rmse=rmse)
            )

        hr, work_roll = eval_hybrid_segment(
            df,
            checkpoint,
            encoder,
            b0,
            h,
            device,
            roll_forward=True,
            work=work_roll,
        )
        parts["hybrid_block_roll"].append(hr)

        ht, _ = eval_hybrid_segment(
            df,
            checkpoint,
            encoder,
            b0,
            h,
            device,
            roll_forward=False,
        )
        parts["hybrid_block_tf"].append(ht)

        if gru_model is not None:
            gr, work_gru = eval_gru_segment(
                gru_model,
                gru_scaler,
                gru_seq_len,
                df,
                b0,
                h,
                device,
                roll_forward=True,
                work=work_gru,
            )
            parts["gru_block_roll"].append(gr)

    labels = {
        "ttm_block": "TTM 96h block (one-shot / block)",
        "hybrid_block_roll": "Hybrid roll within block (oracle between blocks)",
        "hybrid_block_tf": "Hybrid teacher-forcing within block (upper bound)",
        "gru_block_roll": "GRU-only roll within block",
        "seasonal_24h": "Seasonal naive (24h lag)",
        "seasonal_168h": "Seasonal naive (168h lag)",
    }
    out: dict[str, dict] = {}
    for key, plist in parts.items():
        if not plist:
            continue
        merged = _concat_results(plist)
        out[key] = {
            "label": labels[key],
            "mse": merged.mse,
            "rmse": merged.rmse,
            "n_hours": len(merged.actual),
            "bias_mean": float(np.mean(merged.pred - merged.actual)),
        }
    return out


def run_horizon_curve(
    df: pd.DataFrame,
    *,
    year_start: int,
    year_end: int,
    block_hours: int,
    horizons: list[int],
    ttm: TTMBlockForecaster,
    checkpoint: Path,
    encoder: FrozenTTMEncoder | None,
    device,
    anchor_stride: int | None = None,
) -> dict:
    """RMSE at horizons h (1-indexed) averaged over block anchors in target year."""
    from pipelines.seoul.hybrid_eval import load_hybrid_for_eval, _hybrid_one_step

    stride = anchor_stride or block_hours
    anchors = list(range(year_start, year_end - max(horizons), stride))
    model, scaler, cfg, ttm_emb_dim, skip_ttm = load_hybrid_for_eval(checkpoint, device)
    seq_len = int(cfg["seq_len"])

    errs: dict[str, dict[int, list[float]]] = {
        "ttm_block": {h: [] for h in horizons},
        "hybrid_1step": {h: [] for h in horizons},
        "seasonal_24h": {h: [] for h in horizons},
        "seasonal_168h": {h: [] for h in horizons},
    }

    for b0 in anchors:
        horizon = max(horizons)
        try:
            pred_ttm = ttm.predict(df, b0, horizon)
        except Exception:
            continue
        for h in horizons:
            gi = b0 + h - 1
            if gi >= len(df):
                continue
            actual = float(df["power"].iloc[gi])
            if h <= len(pred_ttm):
                errs["ttm_block"][h].append(pred_ttm[h - 1] - actual)
            if gi - 24 >= 0:
                errs["seasonal_24h"][h].append(float(df["power"].iloc[gi - 24]) - actual)
            if gi - 168 >= 0:
                errs["seasonal_168h"][h].append(float(df["power"].iloc[gi - 168]) - actual)
            if gi >= seq_len:
                try:
                    p = _hybrid_one_step(
                        model, scaler, encoder, df, gi, seq_len, ttm_emb_dim, device, skip_ttm
                    )
                    errs["hybrid_1step"][h].append(p - actual)
                except ValueError:
                    pass

    summary = {}
    for model, by_h in errs.items():
        summary[model] = {}
        for h, e in by_h.items():
            if not e:
                continue
            arr = np.array(e)
            summary[model][str(h)] = {
                "rmse": float(np.sqrt(np.mean(arr**2))),
                "bias": float(np.mean(arr)),
                "n": len(e),
            }
    return summary


def _write_html(summary: dict, horizon: dict, path: Path, target_year: int) -> None:
    import plotly.graph_objects as go

    models = summary.get("block_models", {})
    names = [models[k]["label"] for k in models]
    rmses = [models[k]["rmse"] for k in models]

    fig = go.Figure(
        data=[go.Bar(x=names, y=rmses, marker_color=["#6366f1", "#f97316", "#22c55e", "#a855f7", "#94a3b8", "#64748b"])]
    )
    fig.update_layout(
        title=f"서울 {target_year} — 96h 블록 백테스트 (블록 경계마다 실측 갱신)",
        yaxis_title="RMSE (MWh)",
        template="plotly_white",
        height=480,
    )

    if horizon:
        h_keys = sorted({int(h) for m in horizon.values() for h in m})
        fig2 = go.Figure()
        colors = {"ttm_block": "#a855f7", "hybrid_1step": "#f97316", "seasonal_24h": "#94a3b8", "seasonal_168h": "#64748b"}
        for model, by_h in horizon.items():
            xs, ys = [], []
            for h in h_keys:
                cell = by_h.get(str(h))
                if cell:
                    xs.append(h)
                    ys.append(cell["rmse"])
            if xs:
                fig2.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines+markers",
                        name=model,
                        line=dict(color=colors.get(model, "#333")),
                    )
                )
        fig2.update_layout(
            title=f"Horizon RMSE (앵커 stride, {target_year})",
            xaxis_title="Horizon (hours)",
            yaxis_title="RMSE",
            template="plotly_white",
            height=420,
        )
        horizon_html = fig2.to_html(full_html=False, include_plotlyjs=False)
    else:
        horizon_html = ""

    block_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    note = summary.get("note", "")
    body = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"/><title>Block eval {target_year}</title></head>
<body style="font-family:system-ui;max-width:960px;margin:2rem auto;padding:0 1rem">
<h1>장기·블록 예측 비교 ({target_year})</h1>
<p>{note}</p>
{block_html}
<h2>Horizon curve</h2>
{horizon_html}
<pre style="background:#f1f5f9;padding:1rem;overflow:auto">{json.dumps(summary, indent=2, ensure_ascii=False)}</pre>
</body></html>"""
    path.write_text(body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="96h block + horizon eval (Seoul, TSFM-aligned).")
    parser.add_argument("--parquet", type=Path, default=SEOUL_CITY_HOURLY_PARQUET)
    parser.add_argument("--checkpoint", type=Path, default=HYBRID_CHECKPOINT_SEOUL)
    parser.add_argument("--target-year", type=int, default=2024)
    parser.add_argument("--context-through", type=int, default=None)
    parser.add_argument("--block-hours", type=int, default=96)
    parser.add_argument("--min-year", type=int, default=2020, help="TTM/GRU train rows from this year")
    parser.add_argument("--train-frac", type=float, default=0.85)
    parser.add_argument("--model-path", type=str, default="ibm-granite/granite-timeseries-ttm-r2")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--gru-epochs", type=int, default=8)
    parser.add_argument("--chained-blocks", action="store_true", help="Carry preds across blocks (harder)")
    parser.add_argument("--skip-ttm", action="store_true")
    parser.add_argument("--skip-gru", action="store_true")
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--horizons", type=str, default="1,24,48,96")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-html", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.parquet.is_file():
        raise SystemExit(f"Missing {args.parquet}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing {args.checkpoint}. Run hybrid_train first.")

    import torch

    ctx = args.context_through or (args.target_year - 1)
    df = load_hourly(args.parquet)
    year_start, year_end, _ = _year_indices(df, args.target_year, ctx)

    train_cut = df[pd.to_datetime(df["ts"]).dt.year <= ctx].copy()
    train_for_models = _filter_min_year(train_cut, args.min_year)
    n_train_ttm = int(len(train_for_models) * args.train_frac)
    ttm_train_df = train_for_models.iloc[:n_train_ttm]

    print(
        f"Block eval: {args.target_year}  rows [{year_start}:{year_end})  "
        f"block={args.block_hours}h  context≤{ctx}  train≥{args.min_year}",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ttm = None
    if not args.skip_ttm:
        print("Loading TTM block forecaster…", flush=True)
        ttm = TTMBlockForecaster(
            ttm_train_df,
            args.model_path,
            args.context_length,
            args.prediction_length,
        )

    import torch as _torch

    ckpt_meta = _torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    skip_ttm_ckpt = bool(ckpt_meta["hybrid_config"].get("skip_ttm", False))
    encoder = None
    if not skip_ttm_ckpt:
        print("Loading TTM encoder for hybrid head…", flush=True)
        encoder = FrozenTTMEncoder(
            args.model_path,
            args.context_length,
            args.prediction_length,
            load_hourly_for_ttm(ttm_train_df),
            device,
        )

    gru_model = gru_scaler = gru_seq_len = None
    if not args.skip_gru:
        print("Training GRU-only (min_year train)…", flush=True)
        gru_model, gru_scaler, gru_seq_len = train_gru_only_model(
            train_for_models,
            args.train_frac,
            seq_len=168,
            hidden=64,
            layers=2,
            epochs=args.gru_epochs,
            device=device,
        )

    block_models = run_block_backtest(
        df,
        year_start=year_start,
        year_end=year_end,
        block_hours=args.block_hours,
        max_blocks=args.max_blocks,
        ttm=ttm,
        checkpoint=args.checkpoint,
        encoder=encoder,
        gru_model=gru_model,
        gru_scaler=gru_scaler,
        gru_seq_len=gru_seq_len or 168,
        device=device,
        chained=args.chained_blocks,
    )

    horizon = {}
    if ttm is not None:
        h_list = [int(x) for x in args.horizons.split(",")]
        print("Horizon curve…", flush=True)
        horizon = run_horizon_curve(
            df,
            year_start=year_start,
            year_end=year_end,
            block_hours=args.block_hours,
            horizons=h_list,
            ttm=ttm,
            checkpoint=args.checkpoint,
            encoder=encoder,
            device=device,
        )

    summary = {
        "target_year": args.target_year,
        "context_through_year": ctx,
        "block_hours": args.block_hours,
        "min_train_year": args.min_year,
        "chained_blocks": args.chained_blocks,
        "note": (
            "블록마다 문맥은 실측으로 갱신(oracle between blocks). "
            "블록 안: TTM=96h one-shot, Hybrid/GRU=1h roll. "
            "hybrid_block_tf=블록 내 실측 창(상한). Naive lag-1 제외; seasonal 24h/168h 포함."
        ),
        "block_models": block_models,
        "horizon_rmse": horizon,
    }

    out_json = args.output_json or model_compare_block_json(args.target_year)
    out_html = args.output_html or model_compare_block_html(args.target_year)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_html(summary, horizon, out_html, args.target_year)

    print(f"\nWrote {out_json}\nWrote {out_html}\n")
    print("Block RMSE (lower is better):")
    ranked = sorted(block_models.items(), key=lambda kv: kv[1]["rmse"])
    for k, v in ranked:
        print(f"  {v['label']:42s}  RMSE={v['rmse']:.1f}  bias={v['bias_mean']:+.1f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
