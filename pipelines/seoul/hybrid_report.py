# Copyright contributors to the TSFM project
#
"""HTML report: hybrid holdout + model comparison table.

  uv run python -m pipelines.seoul.hybrid_compare
  uv run python -m pipelines.seoul.hybrid_report
  open artifacts/seoul/hybrid_report.html
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from html import escape
from pathlib import Path

import pandas as pd

from pipelines.seoul.config import (
    HYBRID_HOLDOUT_FORECAST_PARQUET,
    HYBRID_METRICS_SEOUL_JSON,
    HYBRID_REPORT_HTML,
    MODEL_COMPARE_JSON,
    MODEL_COMPARE_PLOTS_HTML,
    REGIONAL_HOURLY_PARQUET,
    ROBUSTNESS_JSON,
    SENSITIVITY_51_JSON,
)
from pipelines.seoul.model_compare_plots import write_compare_plots_html
from pipelines.seoul.hybrid_common import load_hourly
from pipelines.seoul.policy_scenarios import build_policy_payload
from pipelines.seoul.scenario_simulator import build_scenario_payload


def _require_plotly() -> None:
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as e:
        raise SystemExit("plotly required: uv sync --extra pipeline") from e


def _comparison_table_html(compare: dict | None) -> str:
    if not compare or "models" not in compare:
        return "<p><i>모델 비교표 없음 — <code>python -m pipelines.seoul.hybrid_compare</code> 실행</i></p>"
    rows = []
    for key in ("naive", "gru_only", "ttm_zeroshot", "hybrid"):
        if key not in compare["models"]:
            continue
        m = compare["models"][key]
        rows.append(
            f"<tr><td>{m['label']}</td><td>{m['mse']:.1f}</td><td>{m['rmse']:.2f}</td></tr>"
        )
    return (
        "<table style='border-collapse:collapse;width:100%;max-width:520px'>"
        "<tr style='background:#e2e8f0'><th>모델</th><th>MSE</th><th>RMSE</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _robustness_html(rob: dict | None) -> str:
    if not rob:
        return (
            "<p><i>견고성 표 없음 — <code>python -m pipelines.seoul.hybrid_robustness</code> 실행</i></p>"
        )
    parts = [f"<p>{rob.get('summary', '')}</p>"]
    if rob.get("seasonal_same_model"):
        parts.append("<h3>동일 모델 · 계절별 96h</h3>")
        parts.append(
            "<table style='border-collapse:collapse;width:100%'>"
            "<tr style='background:#e2e8f0'><th>구간</th><th>Naive</th><th>GRU</th><th>Hybrid</th><th>Hybrid&lt;GRU</th></tr>"
        )
        for r in rob["seasonal_same_model"]:
            win = "✓" if r.get("hybrid_beats_gru") else "—"
            parts.append(
                f"<tr><td>{r['label']}</td><td>{r['naive_mse']:.1f}</td>"
                f"<td>{r['gru_only_mse']:.1f}</td><td>{r['hybrid_mse']:.1f}</td><td>{win}</td></tr>"
            )
        parts.append("</table>")
    if rob.get("retrain_splits"):
        parts.append("<h3>학습 구간 변경 후 재학습</h3>")
        parts.append(
            "<table style='border-collapse:collapse;width:100%'>"
            "<tr style='background:#e2e8f0'><th>설정</th><th>Naive</th><th>GRU</th><th>Hybrid</th></tr>"
        )
        for r in rob["retrain_splits"]:
            parts.append(
                f"<tr><td>{r['label']}</td><td>{r['naive_mse']:.1f}</td>"
                f"<td>{r['gru_only_mse']:.1f}</td><td>{r['hybrid_mse']:.1f}</td></tr>"
            )
        parts.append("</table>")
    if rob.get("note"):
        parts.append(f"<p style='color:#64748b;font-size:14px'>{rob['note']}</p>")
    return "\n".join(parts)


def _scenario_section_html(payload: dict) -> str:
    if not payload.get("available"):
        err = escape(str(payload.get("error", "시나리오 데이터를 만들 수 없습니다.")))
        return f"<p><i>입력 시나리오 없음 — {err}</i></p>"
    if not payload.get("scenarios"):
        return "<p><i>입력 시나리오 없음</i></p>"

    options = "".join(
        f"<option value='{escape(s['id'])}'>{escape(s['title'])}</option>"
        for s in payload["scenarios"]
    )
    note = ""
    if payload.get("model_error"):
        note = (
            "<p style='color:#b45309'>서울 TSFM+GRU 예측은 계산하지 못해 "
            f"지역 민감도만 표시합니다: {escape(str(payload['model_error']))}</p>"
        )
    return f"""
<div class="scenario-box">
  <p>
    실제 데이터에서 가까운 기준 시점을 고르고, 최근 기온 문맥을 바꿔 넣어
    가설 3.1~3.3이 새 입력 조건에서도 같은 방향으로 설명되는지 확인합니다.
  </p>
  {note}
  <label for="scenarioSelect"><b>시나리오 선택</b></label>
  <select id="scenarioSelect">{options}</select>
  <div id="scenarioCards" class="scenario-grid"></div>
  <h3>4개 지역 민감도 비교</h3>
  <div id="regionalScenario"></div>
  <p id="scenarioVerdict" class="verdict"></p>
</div>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid model HTML report.")
    parser.add_argument("--forecast", type=Path, default=HYBRID_HOLDOUT_FORECAST_PARQUET)
    parser.add_argument("--metrics", type=Path, default=HYBRID_METRICS_SEOUL_JSON)
    parser.add_argument("--compare", type=Path, default=MODEL_COMPARE_JSON)
    parser.add_argument("--robustness", type=Path, default=ROBUSTNESS_JSON)
    parser.add_argument("--output", type=Path, default=HYBRID_REPORT_HTML)
    parser.add_argument("--model-path", type=str, default="ibm-granite/granite-timeseries-ttm-r2")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=96)
    args = parser.parse_args(argv)

    if not args.forecast.is_file():
        raise SystemExit(f"Missing {args.forecast}. Run hybrid_train or hybrid_compare first.")

    _require_plotly()
    import plotly.graph_objects as go

    fc = pd.read_parquet(args.forecast)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8")) if args.metrics.is_file() else {}
    compare = (
        json.loads(args.compare.read_text(encoding="utf-8")) if args.compare.is_file() else None
    )
    robustness = (
        json.loads(args.robustness.read_text(encoding="utf-8")) if args.robustness.is_file() else None
    )

    sensitivity = None
    sens_note = ""
    if SENSITIVITY_51_JSON.is_file():
        sensitivity = json.loads(SENSITIVITY_51_JSON.read_text(encoding="utf-8"))
        sens_note = f"<p>5.1 민감도: {sensitivity.get('summary', '')}</p>"

    hourly = load_hourly(REGIONAL_HOURLY_PARQUET) if REGIONAL_HOURLY_PARQUET.is_file() else pd.DataFrame()
    policy = build_policy_payload(hourly, sensitivity)
    scenario_payload = build_scenario_payload(
        hourly=hourly,
        sensitivity=sensitivity,
        model_path=args.model_path,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["ts"], y=fc["power_actual"], name="실측", mode="lines", line=dict(width=2)))
    for col, name, color in [
        ("power_pred_hybrid", "TSFM+GRU", "#f97316"),
        ("power_pred_gru_only", "GRU only", "#22c55e"),
        ("power_pred_ttm", "TTM zeroshot", "#a855f7"),
        ("power_pred_naive", "Naive", "#94a3b8"),
    ]:
        if col in fc.columns:
            fig.add_trace(
                go.Scatter(x=fc["ts"], y=fc[col], name=name, mode="lines", line=dict(color=color))
            )
    fig.update_layout(
        title="서울 96h 홀드아웃 — 모델 비교",
        xaxis_title="시간",
        yaxis_title="전력",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    if compare and compare.get("models"):
        write_compare_plots_html(
            compare,
            fc,
            MODEL_COMPARE_PLOTS_HTML,
            include_ttm="power_pred_ttm" in fc.columns,
        )

    hold_mse = metrics.get("holdout_mse_raw", float("nan"))
    val_mse = metrics.get("val_mse_final", float("nan"))
    hold_rmse = math.sqrt(hold_mse) if hold_mse == hold_mse else float("nan")
    compare_html = _comparison_table_html(compare)
    robustness_html = _robustness_html(robustness)
    policy_html = "<ul>" + "".join(
        f"<li><b>{b.get('title', '')}</b> — {b.get('body', '')}</li>"
        for b in policy.get("_national", [])[:3]
    ) + "</ul>"
    scenario_html = _scenario_section_html(scenario_payload)
    scenario_json = json.dumps(scenario_payload, ensure_ascii=False)


    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"/>
<title>서울 하이브리드 예측 (2차)</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 980px; margin: 24px auto; padding: 0 16px; }}
table td, table th {{ border: 1px solid #cbd5e1; padding: 8px 12px; text-align: right; }}
table td:first-child, table th:first-child {{ text-align: left; }}
.scenario-box {{ border: 1px solid #cbd5e1; border-radius: 12px; padding: 16px; background: #f8fafc; }}
.scenario-box select {{ margin: 8px 0 16px; padding: 8px 10px; min-width: 260px; }}
.scenario-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }}
.metric-card {{ background: white; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px; }}
.metric-card .label {{ color: #64748b; font-size: 13px; }}
.metric-card .value {{ font-size: 22px; font-weight: 700; margin-top: 6px; }}
.verdict {{ background: #fff7ed; border-left: 4px solid #f97316; padding: 12px; }}
</style>
</head><body>
<h1>TSFM + GRU 하이브리드 (2차)</h1>
<pre style="background:#f4f4f4;padding:12px;border-radius:8px;">
Parquet → TTM [frozen] → embedding → GRU [trainable] ← temp, hour, region → 전력 예측
</pre>
<p>하이브리드 검증 MSE: <b>{val_mse:.4f}</b> · 홀드아웃 MSE: <b>{hold_mse:.1f}</b> (RMSE ≈ {hold_rmse:.1f})</p>
{sens_note}
<h2>5.3 모델 비교 (동일 96h 홀드아웃)</h2>
{compare_html}
<p>막대·오차 Plot 전용 페이지:
<a href="model_compare_plots.html">model_compare_plots.html</a>
(또는 <code>python -m pipelines.seoul.model_compare_plots</code>)</p>
<h2>패턴 변화·계절 견고성</h2>
{robustness_html}
<h2>96시간 홀드아웃 차트</h2>
<div id="chart"></div>
<h2>새 input 시나리오</h2>
{scenario_html}
<h2>6.2 정책</h2>
{policy_html}
<script>
var data = {fig.to_json()};
Plotly.newPlot('chart', data.data, data.layout, {{responsive: true}});
var scenarioPayload = {scenario_json};
function fmt(v, digits) {{
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
  return Number(v).toLocaleString('ko-KR', {{ maximumFractionDigits: digits }});
}}
function signed(v) {{
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
  var n = Number(v);
  return (n >= 0 ? '+' : '') + n.toLocaleString('ko-KR', {{ maximumFractionDigits: 1 }});
}}
function renderScenario() {{
  var select = document.getElementById('scenarioSelect');
  if (!select || !scenarioPayload.scenarios || !scenarioPayload.scenarios.length) return;
  var scenario = scenarioPayload.scenarios.find(function(s) {{ return s.id === select.value; }}) || scenarioPayload.scenarios[0];
  var input = scenario.input || {{}};
  var seoul = scenario.seoul_model || null;
  var cards = [
    ['가설', scenario.hypothesis || '—'],
    ['입력 조건', input.month + '월 ' + input.target_hour + '시 · ' + (input.is_weekend ? '주말' : '평일')],
    ['기온 입력', fmt(input.baseline_temp, 1) + '°C → ' + (input.scenario_temp === null ? signed(input.temp_delta) + '°C' : fmt(input.scenario_temp, 1) + '°C')],
    ['최근 문맥 변경', fmt(input.shock_hours, 0) + '시간']
  ];
  if (seoul) {{
    cards.push(['서울 기준 예측', fmt(seoul.baseline_prediction, 1) + ' → ' + fmt(seoul.scenario_prediction, 1)]);
    cards.push(['서울 변화량', signed(seoul.delta)]);
    cards.push(['기준 시점', seoul.base_datetime]);
    cards.push(['해당 시점 실측', fmt(seoul.actual_power, 1)]);
  }} else {{
    cards.push(['서울 TSFM+GRU', '계산 불가']);
  }}
  document.getElementById('scenarioCards').innerHTML = cards.map(function(c) {{
    return '<div class="metric-card"><div class="label">' + c[0] + '</div><div class="value">' + c[1] + '</div></div>';
  }}).join('');

  var rows = (scenario.regional_stats || []).map(function(r) {{
    return '<tr><td>' + r.region + '</td><td>' + fmt(r.baseline_mean, 1) + '</td><td>' +
      fmt(r.scenario_estimate, 1) + '</td><td>' + signed(r.delta) + '</td><td>' +
      fmt(r.slope, 2) + '</td><td>' + r.band + '</td></tr>';
  }}).join('');
  document.getElementById('regionalScenario').innerHTML =
    '<table style="border-collapse:collapse;width:100%"><tr style="background:#e2e8f0">' +
    '<th>지역</th><th>기준 평균</th><th>시나리오 추정</th><th>변화량</th><th>구간 기울기</th><th>기온 구간</th></tr>' +
    rows + '</table>';
  document.getElementById('scenarioVerdict').textContent = scenario.verdict || '';
}}
var scenarioSelect = document.getElementById('scenarioSelect');
if (scenarioSelect) {{
  scenarioSelect.addEventListener('change', renderScenario);
  renderScenario();
}}
</script>
</body></html>"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
