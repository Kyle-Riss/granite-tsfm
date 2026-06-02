# Copyright contributors to the TSFM project
#
"""TSFM-centric benchmark: fair long-horizon comparison (96h direct vs roll baselines).

  uv run python -m pipelines.seoul.tsfm_benchmark --target-year 2024
  uv run python -m pipelines.seoul.tsfm_benchmark --target-year 2024 --max-blocks 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.seoul.config import (
    ARTIFACTS_DIR,
    SEOUL_CITY_HOURLY_PARQUET,
)
from pipelines.seoul.hybrid_block_eval import (
    _filter_min_year,
    _year_indices,
)
from pipelines.seoul.hybrid_common import load_hourly, load_hourly_for_ttm
from pipelines.seoul.hybrid_eval import (
    TTMBlockForecaster,
    _metrics,
    eval_gru_segment,
    train_gru_only_model,
)
from pipelines.seoul.hybrid_model import FrozenTTMEncoder


def _pred_flat(df: pd.DataFrame, b0: int, h: int) -> np.ndarray:
    """Repeat power at b0-1 for h steps."""
    if b0 < 1:
        raise ValueError("b0 must be >= 1")
    v = float(df["power"].iloc[b0 - 1])
    return np.full(h, v, dtype=np.float64)


def _pred_seasonal(df: pd.DataFrame, b0: int, h: int, lag: int) -> np.ndarray:
    out = np.empty(h, dtype=np.float64)
    for k in range(h):
        ref = b0 + k - lag
        out[k] = float(df["power"].iloc[ref]) if ref >= 0 else np.nan
    return out


def _pred_naive_roll(df: pd.DataFrame, b0: int, h: int) -> np.ndarray:
    """Lag-1 roll within block (predictions feed forward)."""
    work = df.copy()
    preds = []
    for k in range(h):
        g = b0 + k
        if g < 1:
            preds.append(np.nan)
            continue
        if k == 0:
            p = float(work["power"].iloc[g - 1])
        else:
            p = float(work["power"].iloc[g - 1])
        work.loc[g, "power"] = p
        preds.append(p)
    return np.array(preds, dtype=np.float64)


def _slice_rmse(pred: np.ndarray, actual: np.ndarray, start: int, end: int) -> float | None:
    p = pred[start:end]
    a = actual[start:end]
    mask = np.isfinite(p) & np.isfinite(a)
    if mask.sum() < 1:
        return None
    return float(np.sqrt(np.mean((p[mask] - a[mask]) ** 2)))


def run_tsfm_benchmark(
    df: pd.DataFrame,
    *,
    year_start: int,
    year_end: int,
    block_hours: int,
    max_blocks: int | None,
    ttm: TTMBlockForecaster,
    gru_model,
    gru_scaler,
    gru_seq_len: int,
    device,
    late_start: int = 48,
) -> dict:
    """Compare models on 96h blocks; emphasize late horizon (TSFM design point)."""
    block_starts = list(range(year_start, year_end, block_hours))
    if max_blocks:
        block_starts = block_starts[:max_blocks]

    models = [
        "ttm_96h_direct",
        "gru_96h_roll",
        "seasonal_24h",
        "seasonal_168h",
        "flat_persistence",
        "naive_1h_roll",
    ]
    buckets = ["full_block", "late_horizon_48_96", "endpoint_h96", "early_h1"]
    errs: dict[str, dict[str, list[float]]] = {
        m: {b: [] for b in buckets} for m in models
    }

    for bi, b0 in enumerate(block_starts):
        b1 = min(b0 + block_hours, year_end)
        h = b1 - b0
        if h <= 0:
            continue
        print(f"  block {bi + 1}/{len(block_starts)}  b0={b0}  h={h}", flush=True)
        actual = df["power"].iloc[b0:b1].to_numpy(dtype=np.float64)

        preds: dict[str, np.ndarray] = {}
        preds["ttm_96h_direct"] = ttm.predict(df, b0, h)[:h]
        preds["seasonal_24h"] = _pred_seasonal(df, b0, h, 24)
        preds["seasonal_168h"] = _pred_seasonal(df, b0, h, 168)
        preds["flat_persistence"] = _pred_flat(df, b0, h)
        preds["naive_1h_roll"] = _pred_naive_roll(df, b0, h)

        if gru_model is not None:
            gr, _ = eval_gru_segment(
                gru_model,
                gru_scaler,
                gru_seq_len,
                df,
                b0,
                h,
                device,
                roll_forward=True,
                work=None,
            )
            preds["gru_96h_roll"] = gr.pred[:h]

        for name, p in preds.items():
            n = min(len(p), len(actual))
            p, a = p[:n], actual[:n]
            mse, rmse = _metrics(p, a)
            errs[name]["full_block"].append(rmse)
            late = _slice_rmse(p, a, late_start - 1, n)
            if late is not None:
                errs[name]["late_horizon_48_96"].append(late)
            if n > 0:
                errs[name]["endpoint_h96"].append(float(abs(p[-1] - a[-1])))
            if n > 0:
                errs[name]["early_h1"].append(float(abs(p[0] - a[0])))

    labels = {
        "ttm_96h_direct": "Granite TTM-r2 (96h one-shot)",
        "gru_96h_roll": "GRU-only (96×1h roll, block start)",
        "seasonal_24h": "Seasonal naive (24h lag)",
        "seasonal_168h": "Seasonal naive (168h lag)",
        "flat_persistence": "Flat persistence (last value)",
        "naive_1h_roll": "Naive lag-1 roll (96 steps)",
    }
    summary: dict[str, dict] = {}
    for m in models:
        if not errs[m]["full_block"]:
            continue
        row: dict = {"label": labels[m]}
        for b in buckets:
            arr = errs[m][b]
            if arr:
                row[b] = {
                    "rmse_mean": float(np.mean(arr)),
                    "rmse_median": float(np.median(arr)),
                    "n_blocks": len(arr),
                }
        summary[m] = row

    # head-to-head vs TTM on late horizon
    if "ttm_96h_direct" in summary:
        ttm_late = summary["ttm_96h_direct"].get("late_horizon_48_96", {}).get("rmse_mean")
        wins = []
        for m, row in summary.items():
            if m == "ttm_96h_direct":
                continue
            other = row.get("late_horizon_48_96", {}).get("rmse_mean")
            if ttm_late is not None and other is not None:
                pct = 100.0 * (other - ttm_late) / other if other > 0 else 0.0
                wins.append(
                    {
                        "baseline": m,
                        "baseline_label": row["label"],
                        "ttm_rmse": ttm_late,
                        "baseline_rmse": other,
                        "ttm_wins": ttm_late < other,
                        "improvement_pct": round(pct, 1) if ttm_late < other else round(-pct, 1),
                    }
                )
        summary["_ttm_vs_baselines_late_horizon"] = sorted(
            wins, key=lambda x: -x.get("improvement_pct", 0)
        )

    return summary


def _write_outputs(summary: dict, meta: dict, out_json: Path, out_html: Path, out_md: Path) -> None:
    import plotly.graph_objects as go

    models = [k for k in summary if not k.startswith("_")]
    metric = "late_horizon_48_96"
    names, rmses = [], []
    for m in models:
        cell = summary[m].get(metric)
        if cell:
            names.append(summary[m]["label"])
            rmses.append(cell["rmse_mean"])

    fig = go.Figure(data=[go.Bar(x=names, y=rmses, marker_color="#a855f7")])
    fig.update_layout(
        title=f"Late horizon RMSE (hours {meta.get('late_start', 48)}–96 within each block)",
        yaxis_title="RMSE (MWh)",
        template="plotly_white",
        height=460,
    )

    wins = summary.get("_ttm_vs_baselines_late_horizon", [])
    md_lines = [
        f"# TSFM benchmark — Seoul {meta['target_year']}",
        "",
        meta.get("method_note", ""),
        "",
        "## Main table (mean RMSE over blocks)",
        "",
        "| Model | Full 96h block | Late h48–96 | Endpoint h96 | Early h1 |",
        "|-------|----------------|-------------|--------------|----------|",
    ]
    for m in models:
        row = summary[m]
        def _c(key: str) -> str:
            c = row.get(key)
            return f"{c['rmse_mean']:.1f}" if c else "—"

        md_lines.append(
            f"| {row['label']} | {_c('full_block')} | {_c('late_horizon_48_96')} | "
            f"{_c('endpoint_h96')} | {_c('early_h1')} |"
        )
    md_lines.extend(["", "## TTM vs baselines (late horizon h48–96)", ""])
    for w in wins:
        flag = "✓ TTM wins" if w["ttm_wins"] else "✗"
        md_lines.append(
            f"- **{w['baseline_label']}**: TTM {w['ttm_rmse']:.1f} vs {w['baseline_rmse']:.1f} "
            f"({w['improvement_pct']:+.1f}%) {flag}"
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "models": summary}
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"/>
<title>TSFM benchmark</title></head><body style="font-family:system-ui;max-width:960px;margin:2rem auto">
<h1>TSFM-centric benchmark ({meta['target_year']})</h1>
<p>{meta.get("method_note","")}</p>
{fig.to_html(full_html=False, include_plotlyjs="cdn")}
<pre>{json.dumps(payload, indent=2, ensure_ascii=False)}</pre>
</body></html>"""
    out_html.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TSFM long-horizon benchmark (Seoul).")
    parser.add_argument("--parquet", type=Path, default=SEOUL_CITY_HOURLY_PARQUET)
    parser.add_argument("--target-year", type=int, default=2024)
    parser.add_argument("--context-through", type=int, default=None)
    parser.add_argument("--block-hours", type=int, default=96)
    parser.add_argument("--late-start", type=int, default=48, help="Late horizon slice start (1-based hour in block)")
    parser.add_argument("--min-year", type=int, default=2020)
    parser.add_argument("--train-frac", type=float, default=0.85)
    parser.add_argument("--model-path", type=str, default="ibm-granite/granite-timeseries-ttm-r2")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--gru-epochs", type=int, default=8)
    parser.add_argument("--max-blocks", type=int, default=None)
    parser.add_argument("--skip-gru", action="store_true")
    args = parser.parse_args(argv)

    if not args.parquet.is_file():
        raise SystemExit(f"Missing {args.parquet}")

    import torch

    ctx = args.context_through or (args.target_year - 1)
    df = load_hourly(args.parquet)
    year_start, year_end, _ = _year_indices(df, args.target_year, ctx)

    train_cut = df[pd.to_datetime(df["ts"]).dt.year <= ctx].copy()
    train_for = _filter_min_year(train_cut, args.min_year)
    n_train = int(len(train_for) * args.train_frac)
    ttm_train = train_for.iloc[:n_train]

    print(f"TSFM benchmark  year={args.target_year}  blocks={args.block_hours}h  late≥h{args.late_start}", flush=True)
    ttm = TTMBlockForecaster(
        ttm_train,
        args.model_path,
        args.context_length,
        args.prediction_length,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gru_model = gru_scaler = None
    gru_seq = 168
    if not args.skip_gru:
        print("Training GRU-only for 96h roll baseline…", flush=True)
        gru_model, gru_scaler, gru_seq = train_gru_only_model(
            train_for,
            args.train_frac,
            seq_len=168,
            hidden=64,
            layers=2,
            epochs=args.gru_epochs,
            device=device,
        )

    summary = run_tsfm_benchmark(
        df,
        year_start=year_start,
        year_end=year_end,
        block_hours=args.block_hours,
        max_blocks=args.max_blocks,
        ttm=ttm,
        gru_model=gru_model,
        gru_scaler=gru_scaler,
        gru_seq_len=gru_seq,
        device=device,
        late_start=args.late_start,
    )

    meta = {
        "target_year": args.target_year,
        "context_through_year": ctx,
        "block_hours": args.block_hours,
        "late_start": args.late_start,
        "n_blocks": args.max_blocks or ((year_end - year_start) // args.block_hours),
        "method_note": (
            "동일 블록 시작점·동일 실측 문맥. TTM은 96h를 한 번에 예측(direct); "
            "GRU/Naive는 96회 1h 롤링. late h48–96은 TSFM 설계 horizon. "
            "블록 경계마다 실측 갱신(oracle between blocks)."
        ),
    }

    stem = ARTIFACTS_DIR / f"tsfm_benchmark_{args.target_year}"
    _write_outputs(summary, meta, stem.with_suffix(".json"), stem.with_suffix(".html"), stem.with_suffix(".md"))

    print(f"\nWrote {stem}.json / .html / .md\n")
    wins = summary.get("_ttm_vs_baselines_late_horizon", [])
    print("Late horizon (h48–96) — TTM vs others:")
    for w in wins:
        mark = "WIN" if w["ttm_wins"] else "lose"
        print(f"  [{mark}] vs {w['baseline_label']}: {w['improvement_pct']:+.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
