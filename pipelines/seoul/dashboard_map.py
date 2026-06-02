# Copyright contributors to the TSFM project
#
"""서울 에너지 대시보드: 자치구 지도(월별) + 하단 통계·이점 요약. open artifacts/seoul/dashboard_energy.html"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from pipelines.seoul.config import (
    ARTIFACTS_DIR,
    DISTRICT_MONTHLY_PARQUET,
    HYBRID_METRICS_SEOUL_JSON,
    REGIONAL_HOURLY_PARQUET,
    SENSITIVITY_51_JSON,
    SEOUL_CITY_HOURLY_PARQUET,
)
from pipelines.seoul.district_geo import DISTRICT_CENTROIDS
from pipelines.seoul.policy_scenarios import build_policy_payload


DASHBOARD_HTML = ARTIFACTS_DIR / "dashboard_energy.html"

CSS = """
body { font-family: system-ui, -apple-system, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }
header { padding: 20px 24px 8px; }
header h1 { margin: 0 0 8px; font-size: 1.35rem; }
header p { margin: 0; color: #475569; font-size: 0.95rem; }
#seoul-map { background: #fff; border-top: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; }
.summary-box { max-width: 1100px; margin: 24px auto 32px; padding: 24px; background: #fff;
  border: 1px solid #e2e8f0; border-radius: 8px; }
.summary-box h2 { margin-top: 0; font-size: 1.15rem; }
.summary-box h3 { font-size: 1rem; }
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0 20px; }
.stat-card { border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; background: #f8fafc; }
.stat-card .label { display: block; font-size: 0.75rem; color: #64748b; }
.stat-card .value { display: block; font-size: 1.1rem; font-weight: 600; margin-top: 4px; }
.stat-card .sub { font-size: 0.8rem; color: #475569; }
.summary-box ul { margin: 0; padding-left: 1.2rem; line-height: 1.55; color: #334155; }
.policy-grid { display: grid; gap: 12px; margin-top: 16px; }
.policy-card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; background: #f8fafc; }
.policy-card .tag { font-size: 11px; font-weight: 600; color: #2563eb; }
.policy-card h4 { margin: 8px 0 6px; font-size: 14px; }
.policy-card p { margin: 0; font-size: 13px; line-height: 1.55; color: #334155; }
@media (max-width: 800px) { .stats-grid { grid-template-columns: 1fr 1fr; } }
"""


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as e:
        raise SystemExit("plotly required: uv sync --extra pipeline") from e


def _attach_coords(dm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in dm.iterrows():
        d = str(r["district"])
        if d not in DISTRICT_CENTROIDS:
            continue
        lat, lon = DISTRICT_CENTROIDS[d]
        rows.append({**r.to_dict(), "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


def _compute_insights(dm: pd.DataFrame, hourly_seoul: pd.DataFrame | None, hybrid_metrics: dict | None) -> dict:
    annual = dm.groupby("district", observed=True)["usage"].sum().sort_values(ascending=False)
    summer = dm[dm["month"].isin([7, 8])].groupby("district", observed=True)["usage"].mean().sort_values(ascending=False)
    seasonal_range = dm.groupby("district", observed=True)["usage"].agg(lambda s: float(s.max() - s.min()))

    insights: dict = {
        "top_annual_district": annual.index[0] if len(annual) else "—",
        "top_annual_usage": float(annual.iloc[0]) if len(annual) else 0.0,
        "top_summer_district": summer.index[0] if len(summer) else "—",
        "top_summer_usage": float(summer.iloc[0]) if len(summer) else 0.0,
        "max_seasonal_range_district": seasonal_range.idxmax() if len(seasonal_range) else "—",
        "max_seasonal_range": float(seasonal_range.max()) if len(seasonal_range) else 0.0,
        "district_count": int(dm["district"].nunique()),
    }

    if hourly_seoul is not None and len(hourly_seoul) > 0:
        h = hourly_seoul.copy()
        h["ts"] = pd.to_datetime(h["ts"])
        h["month"] = h["ts"].dt.month
        insights["seoul_peak_month"] = int(h.groupby("month")["power"].mean().idxmax())
        wd = h[h["is_weekend"] == 0].groupby("hour")["power"].mean()
        we = h[h["is_weekend"] == 1].groupby("hour")["power"].mean()
        insights["weekday_peak_hour"] = int(wd.idxmax()) if len(wd) else 0
        insights["weekend_peak_hour"] = int(we.idxmax()) if len(we) else 0
        insights["temp_power_corr"] = float(h["temp"].corr(h["power"])) if h["temp"].std() > 0 else 0.0

    if hybrid_metrics:
        insights["hybrid_val_mse"] = hybrid_metrics.get("val_mse_final")

    return insights




def _policy_html(hourly_path: Path | None, sensitivity_path: Path | None, district_df: pd.DataFrame) -> str:
    if hourly_path is None or not hourly_path.is_file():
        return ""
    hourly = pd.read_parquet(hourly_path)
    sens = None
    if sensitivity_path and sensitivity_path.is_file():
        sens = json.loads(sensitivity_path.read_text(encoding="utf-8"))
    scenarios = build_policy_payload(hourly, sens, district_df).get("_national", [])
    if not scenarios:
        return ""
    cards = "".join(
        f'<article class="policy-card"><span class="tag">{s["tag"]}</span>'
        f'<h4>{s["title"]}</h4><p>{s["body"]}</p></article>'
        for s in scenarios
    )
    return f'<h3>6.2 정책 활용 시나리오</h3><div class="policy-grid">{cards}</div>'


def build_dashboard_html(
    district_parquet: Path,
    hourly_seoul_parquet: Path | None,
    hybrid_metrics_path: Path | None,
    output: Path,
    regional_hourly_parquet: Path | None = None,
    sensitivity_json: Path | None = None,
) -> None:
    import plotly.graph_objects as go

    dm = pd.read_parquet(district_parquet)
    dm = _attach_coords(dm)
    if dm.empty:
        raise ValueError("No districts matched DISTRICT_CENTROIDS; check district names in CSV.")

    hourly_seoul = pd.read_parquet(hourly_seoul_parquet) if hourly_seoul_parquet and hourly_seoul_parquet.is_file() else None
    hybrid_metrics = None
    if hybrid_metrics_path and hybrid_metrics_path.is_file():
        hybrid_metrics = json.loads(hybrid_metrics_path.read_text(encoding="utf-8"))

    ins = _compute_insights(dm, hourly_seoul, hybrid_metrics)

    fig = go.Figure()
    months = sorted(dm["month"].unique())
    for m in months:
        sub = dm[dm["month"] == m]
        mx = float(sub["usage"].max())
        sizes = (sub["usage"] / mx * 40 + 12).tolist() if mx > 0 else [16] * len(sub)
        fig.add_trace(
            go.Scattermapbox(
                lat=sub["lat"],
                lon=sub["lon"],
                mode="markers",
                marker={
                    "size": sizes,
                    "color": sub["usage"],
                    "colorscale": "Blues",
                    "showscale": True,
                    "colorbar": {"title": "사용량"},
                },
                text=sub.apply(lambda r: f"{r['district']}<br>{r['usage']:,.0f}", axis=1),
                hoverinfo="text",
                visible=(m == months[0]),
            )
        )

    buttons = []
    for i, m in enumerate(months):
        visible = [False] * len(months)
        visible[i] = True
        buttons.append(
            {
                "label": f"{int(m)}월",
                "method": "update",
                "args": [{"visible": visible}, {"title": f"서울 자치구 월별 에너지 — {int(m)}월"}],
            }
        )

    fig.update_layout(
        mapbox={"style": "open-street-map", "center": {"lat": 37.5665, "lon": 126.9780}, "zoom": 10.2},
        title=f"서울 자치구 월별 에너지 — {int(months[0])}월",
        height=520,
        margin={"t": 48, "b": 0, "l": 0, "r": 0},
        updatemenus=[
            {"buttons": buttons, "direction": "down", "showactive": True, "x": 0.02, "y": 1.12, "xanchor": "left"}
        ],
    )

    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="seoul-map")

    extra = ""
    if "seoul_peak_month" in ins:
        extra += (
            f"<li><b>서울 시간별 피크 월</b> {ins['seoul_peak_month']}월</li>"
            f"<li><b>평일/주말 피크 시각</b> {ins['weekday_peak_hour']}시 / {ins['weekend_peak_hour']}시</li>"
            f"<li><b>기온–전력 상관</b> {ins['temp_power_corr']:.3f}</li>"
        )
    if "hybrid_val_mse" in ins:
        extra += f"<li><b>하이브리드 검증 MSE</b> {ins['hybrid_val_mse']:,.4f}</li>"

    policy_block = _policy_html(regional_hourly_parquet, sensitivity_json, dm)

    summary = f"""
<section class="summary-box">
  <h2>통계 요약 &amp; 활용 이점</h2>
  <div class="stats-grid">
    <div class="stat-card">
      <span class="label">연간 합계 1위</span>
      <span class="value">{ins['top_annual_district']}</span>
      <span class="sub">{ins['top_annual_usage']:,.0f}</span>
    </div>
    <div class="stat-card">
      <span class="label">여름(7–8월) 평균 1위</span>
      <span class="value">{ins['top_summer_district']}</span>
      <span class="sub">{ins['top_summer_usage']:,.0f}</span>
    </div>
    <div class="stat-card">
      <span class="label">계절 진폭 최대</span>
      <span class="value">{ins['max_seasonal_range_district']}</span>
      <span class="sub">Δ {ins['max_seasonal_range']:,.0f}</span>
    </div>
    <div class="stat-card">
      <span class="label">자치구 수</span>
      <span class="value">{ins['district_count']}</span>
    </div>
  </div>
  <h3>이 대시보드로 볼 수 있는 것</h3>
  <ul>
    <li><b>지도</b>: 월별로 부하가 쏠리는 구를 공간적으로 파악 → 피크 대비·인프라 우선순위</li>
    <li><b>34k 시계열</b>: 기온·평일/주말과 연결해 가설 3.1–3.3 설명</li>
    <li><b>모델</b>: TSFM(비선형 기준) + GRU(빠른 검증) — Parquet 파이프라인으로 확장 가능</li>
    {extra}
  </ul>
  {policy_block}
</section>
"""

    doc = f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>서울 에너지 대시보드</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>서울 에너지 · 지도 + 통계 요약</h1>
  <p>지도 상단 메뉴에서 월을 선택하세요. 아래 박스는 자동 계산된 핵심 지표와 활용 이점입니다.</p>
</header>
{plot_html}
{summary}
</body></html>
"""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output.write_text(doc, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build map dashboard HTML with summary box.")
    parser.add_argument("--district-parquet", type=Path, default=DISTRICT_MONTHLY_PARQUET)
    parser.add_argument("--hourly-seoul", type=Path, default=SEOUL_CITY_HOURLY_PARQUET)
    parser.add_argument("--hybrid-metrics", type=Path, default=HYBRID_METRICS_SEOUL_JSON)
    parser.add_argument("--output", type=Path, default=DASHBOARD_HTML)
    args = parser.parse_args(argv)

    _require_plotly()
    if not args.district_parquet.is_file():
        raise SystemExit(f"Missing {args.district_parquet}. Run materialize first.")

    build_dashboard_html(
        district_parquet=args.district_parquet,
        hourly_seoul_parquet=args.hourly_seoul,
        hybrid_metrics_path=args.hybrid_metrics,
        output=args.output,
        regional_hourly_parquet=REGIONAL_HOURLY_PARQUET,
        sensitivity_json=SENSITIVITY_51_JSON,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
