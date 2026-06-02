# Copyright contributors to the TSFM project
#
"""Naive / GRU-only / TSFM+GRU 비교 Plot (막대·시계열·오차).

  uv run python -m pipelines.seoul.model_compare_plots
  open artifacts/seoul/model_compare_plots.html

  데이터: hybrid_compare 가 만든 model_compare_seoul.json + holdout parquet
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
    HYBRID_HOLDOUT_FORECAST_PARQUET,
    MODEL_COMPARE_JSON,
    MODEL_COMPARE_PLOTS_HTML,
    SEOUL_CITY_HOURLY_PARQUET,
)
from pipelines.seoul.hybrid_common import load_hourly


# 보고서·리포트와 동일 색상
MODEL_STYLES: dict[str, dict[str, str]] = {
    "hybrid": {"label": "TSFM+GRU", "color": "#f97316", "col": "power_pred_hybrid"},
    "gru_only": {"label": "GRU only", "color": "#22c55e", "col": "power_pred_gru_only"},
    "naive": {"label": "Naive", "color": "#94a3b8", "col": "power_pred_naive"},
    "ttm_zeroshot": {"label": "TTM zeroshot", "color": "#a855f7", "col": "power_pred_ttm"},
}

CORE_KEYS = ("naive", "gru_only", "hybrid")


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as e:
        raise SystemExit("plotly required: uv sync --extra pipeline") from e


def _load_compare(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run: uv run python -m pipelines.seoul.hybrid_compare")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_forecast(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    fc = pd.read_parquet(path)
    fc["ts"] = pd.to_datetime(fc["ts"] if "ts" in fc.columns else fc.get("timestamp"))
    return fc


def _merge_legacy_holdout(primary: Path) -> pd.DataFrame | None:
    """holdout parquet에 naive/GRU 컬럼이 없을 때 기존 산출물·실측으로 보강."""
    fc = _load_forecast(primary)
    if fc is None:
        return None
    if all(MODEL_STYLES[k]["col"] in fc.columns for k in CORE_KEYS):
        return fc

    gru_path = ARTIFACTS_DIR / "forecast_gru_holdout_seoul.parquet"
    if MODEL_STYLES["gru_only"]["col"] not in fc.columns and gru_path.is_file():
        gru = pd.read_parquet(gru_path)
        tcol = "ts" if "ts" in gru.columns else "timestamp"
        gru = gru.rename(columns={tcol: "ts", "power_pred": MODEL_STYLES["gru_only"]["col"]})
        gru["ts"] = pd.to_datetime(gru["ts"])
        fc = fc.merge(gru[["ts", MODEL_STYLES["gru_only"]["col"]]], on="ts", how="left")

    if MODEL_STYLES["naive"]["col"] not in fc.columns and SEOUL_CITY_HOURLY_PARQUET.is_file():
        hourly = load_hourly(SEOUL_CITY_HOURLY_PARQUET)
        hourly["ts"] = pd.to_datetime(hourly["ts"])
        hourly = hourly.sort_values("ts").reset_index(drop=True)
        hourly["power_pred_naive"] = hourly["power"].shift(1)
        naive = hourly[["ts", "power_pred_naive"]]
        fc = fc.merge(naive, on="ts", how="left")

    return fc


def _metrics_bar_figure(compare: dict):
    import plotly.graph_objects as go

    models = compare.get("models", {})
    if not models:
        raise ValueError("compare['models'] is empty")
    keys = [k for k in CORE_KEYS if k in models]
    if compare.get("models", {}).get("ttm_zeroshot"):
        keys = list(keys) + ["ttm_zeroshot"]

    labels = [models[k]["label"] for k in keys]
    mse = [models[k]["mse"] for k in keys]
    rmse = [models[k]["rmse"] for k in keys]
    colors = [MODEL_STYLES[k]["color"] for k in keys]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="MSE",
            x=labels,
            y=mse,
            marker_color=colors,
            text=[f"{v:.1f}" for v in mse],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="96h 홀드아웃 — MSE (낮을수록 좋음)",
        yaxis_title="MSE",
        height=420,
        showlegend=False,
        margin=dict(t=60, b=80),
    )
    return fig


def _metrics_rmse_figure(compare: dict):
    import plotly.graph_objects as go

    models = compare.get("models", {})
    keys = [k for k in CORE_KEYS if k in models]
    if models.get("ttm_zeroshot"):
        keys = list(keys) + ["ttm_zeroshot"]

    labels = [models[k]["label"] for k in keys]
    rmse = [models[k]["rmse"] for k in keys]
    colors = [MODEL_STYLES[k]["color"] for k in keys]

    best_idx = int(np.argmin(rmse))
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=rmse,
            marker_color=colors,
            text=[f"{v:.2f}" for v in rmse],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"96h 홀드아웃 — RMSE (최저: {labels[best_idx]})",
        yaxis_title="RMSE",
        height=420,
        showlegend=False,
        margin=dict(t=60, b=80),
    )
    return fig


def _holdout_series_figure(fc: pd.DataFrame, *, include_ttm: bool = False):
    import plotly.graph_objects as go

    fig = go.Figure()
    if "power_actual" in fc.columns:
        fig.add_trace(
            go.Scatter(
                x=fc["ts"],
                y=fc["power_actual"],
                name="실측",
                mode="lines",
                line=dict(color="#2563eb", width=2.5),
            )
        )

    keys = list(CORE_KEYS)
    if include_ttm and "power_pred_ttm" in fc.columns:
        keys = keys + ["ttm_zeroshot"]

    for key in keys:
        style = MODEL_STYLES[key]
        col = style["col"]
        if col not in fc.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=fc["ts"],
                y=fc[col],
                name=style["label"],
                mode="lines",
                line=dict(color=style["color"], width=1.8),
            )
        )

    fig.update_layout(
        title="서울 96h 홀드아웃 — 예측 곡선",
        xaxis_title="시간",
        yaxis_title="전력 (MWh)",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=70),
    )
    return fig


def _error_figure(fc: pd.DataFrame):
    import plotly.graph_objects as go

    if "power_actual" not in fc.columns:
        raise ValueError("power_actual required for error plot")

    actual = fc["power_actual"].astype(float)
    fig = go.Figure()
    for key in CORE_KEYS:
        col = MODEL_STYLES[key]["col"]
        if col not in fc.columns:
            continue
        err = (fc[col].astype(float) - actual).abs()
        fig.add_trace(
            go.Scatter(
                x=fc["ts"],
                y=err,
                name=f"|{MODEL_STYLES[key]['label']} − 실측|",
                mode="lines",
                line=dict(color=MODEL_STYLES[key]["color"], width=1.5),
            )
        )
    fig.update_layout(
        title="시간별 절대 오차 (|예측 − 실측|)",
        xaxis_title="시간",
        yaxis_title="MWh",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _improvement_table_html(compare: dict) -> str:
    models = compare.get("models", {})
    if "hybrid" not in models:
        return ""
    h_mse = models["hybrid"]["mse"]
    rows = []
    for key in CORE_KEYS:
        if key not in models or key == "hybrid":
            continue
        m = models[key]
        delta = m["mse"] - h_mse
        pct = 100.0 * delta / m["mse"] if m["mse"] > 0 else 0.0
        win = "✓" if h_mse < m["mse"] else "—"
        rows.append(
            f"<tr><td>{m['label']}</td><td>{m['mse']:.1f}</td><td>{h_mse:.1f}</td>"
            f"<td>{delta:+.1f}</td><td>{pct:.1f}%</td><td>{win}</td></tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>모델</th><th>MSE</th><th>TSFM+GRU</th><th>차이</th><th>개선율</th><th>Hybrid 우위</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def write_compare_plots_html(
    compare: dict,
    fc: pd.DataFrame | None,
    path: Path,
    *,
    include_ttm: bool = False,
) -> None:
    _require_plotly()

    fig_mse = _metrics_bar_figure(compare)
    fig_rmse = _metrics_rmse_figure(compare)
    charts: list[tuple[str, object]] = [
        ("mse", fig_mse),
        ("rmse", fig_rmse),
    ]

    series_note = ""
    if fc is not None and any(MODEL_STYLES[k]["col"] in fc.columns for k in CORE_KEYS):
        charts.append(("series", _holdout_series_figure(fc, include_ttm=include_ttm)))
        if "power_actual" in fc.columns:
            charts.append(("error", _error_figure(fc)))
    else:
        series_note = (
            "<p class='warn'><b>시계열·오차 Plot 없음</b> — holdout parquet에 naive/GRU 컬럼이 없습니다. "
            "<code>uv run python -m pipelines.seoul.hybrid_compare --skip-ttm-zeroshot</code> "
            "실행 후 다시 생성하세요.</p>"
        )

    holdout = compare.get("holdout_hours", 96)
    table_html = _improvement_table_html(compare)

    divs = "\n".join(f'<div id="chart_{name}" class="chart"></div>' for name, _ in charts)
    scripts = []
    for name, fig in charts:
        payload = json.loads(fig.to_json())
        scripts.append(
            f"Plotly.newPlot('chart_{name}', "
            f"{json.dumps(payload['data'])}, {json.dumps(payload['layout'])}, "
            f"{{responsive: true}});"
        )

    html = f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Naive · GRU · TSFM+GRU 비교</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
.wrap {{ max-width: 1000px; margin: 0 auto; padding: 20px 16px 48px; }}
h1 {{ margin: 0 0 8px; font-size: 1.35rem; }}
.lead {{ color: #475569; font-size: 0.95rem; line-height: 1.55; margin-bottom: 20px; }}
.chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; margin-bottom: 20px; min-height: 400px; }}
.warn {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px; margin-bottom: 16px; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e2e8f0;
  border-radius: 8px; overflow: hidden; margin-top: 8px; }}
th, td {{ padding: 10px 12px; text-align: right; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
th {{ background: #f1f5f9; text-align: center; }}
td:first-child, th:first-child {{ text-align: left; }}
h2 {{ font-size: 1.05rem; margin: 28px 0 12px; }}
</style>
</head>
<body>
<div class="wrap">
<h1>Naive · GRU-only · TSFM+GRU 비교</h1>
<p class="lead">
동일 서울 {holdout}h 홀드아웃(데이터 끝 구간)에서 세 베이스라인을 맞춰 비교합니다.
Naive는 직전 시점 전력 복사, GRU-only는 TTM 임베딩 없이 학습, TSFM+GRU는 동결 TTM + GRU 헤드입니다.
</p>
{series_note}
{divs}
<h2>TSFM+GRU 대비 개선</h2>
{table_html}
</div>
<script>
{chr(10).join(scripts)}
</script>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Model comparison plots (Naive / GRU / Hybrid).")
    parser.add_argument("--compare", type=Path, default=MODEL_COMPARE_JSON)
    parser.add_argument("--forecast", type=Path, default=HYBRID_HOLDOUT_FORECAST_PARQUET)
    parser.add_argument("--output", type=Path, default=MODEL_COMPARE_PLOTS_HTML)
    parser.add_argument("--include-ttm", action="store_true", help="Include TTM zeroshot in plots")
    args = parser.parse_args(argv)

    compare = _load_compare(args.compare)
    fc = _merge_legacy_holdout(args.forecast)
    write_compare_plots_html(compare, fc, args.output, include_ttm=args.include_ttm)
    print(f"Wrote {args.output}")
    if fc is None or not any(MODEL_STYLES[k]["col"] in fc.columns for k in CORE_KEYS):
        print("  (시계열 Plot: hybrid_compare 로 holdout parquet 재생성 필요)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
