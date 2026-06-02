# Copyright contributors to the TSFM project
#
"""지역별 2024 실측 vs 2025 예측 비교 지도·막대·월별 차트.

  uv run python -m pipelines.seoul.forecast_region_compare
  open artifacts/seoul/forecast_region_compare.html

  # 다른 광역 2025 예측이 없으면 먼저 (서울만 이미 있으면 생략 가능):
  uv run python -m pipelines.seoul.hybrid_forecast --all-regions --target-year 2025
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
    FORECAST_REGION_COMPARE_HTML,
    FORECAST_REGION_COMPARE_JSON,
    FORECAST_SEOUL_FORWARD_PARQUET,
    REGIONAL_HOURLY_PARQUET,
)
from pipelines.seoul.district_geo import REGION_CENTROIDS
from pipelines.seoul.hybrid_forecast import forecast_parquet_path
from pipelines.seoul.hybrid_common import load_hourly
from pipelines.seoul.merge_hourly import REGIONS


def _year_slice(df: pd.DataFrame, year: int, col: str = "power") -> pd.DataFrame:
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    return d[d["ts"].dt.year == year]


def _region_actual(hourly: pd.DataFrame, region: str, year: int) -> pd.DataFrame | None:
    g = hourly[hourly["region"] == region]
    if len(g) == 0:
        return None
    return _year_slice(g, year)


def _load_forecast(region: str, target_year: int) -> pd.DataFrame | None:
    path = forecast_parquet_path(region, target_year)
    if region == "서울시" and not path.is_file():
        path = FORECAST_SEOUL_FORWARD_PARQUET
    if not path.is_file():
        return None
    f = pd.read_parquet(path)
    f["ts"] = pd.to_datetime(f["ts"])
    pred_col = "power_pred" if "power_pred" in f.columns else "power"
    return f.rename(columns={pred_col: "power_pred"})


def _monthly_mean(df: pd.DataFrame, value_col: str) -> dict[int, float]:
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    d["month"] = d["ts"].dt.month
    s = d.groupby("month", observed=True)[value_col].mean()
    return {int(m): float(v) for m, v in s.items()}


def build_summary(
    hourly: pd.DataFrame,
    *,
    baseline_year: int,
    target_year: int,
) -> dict:
    rows: list[dict] = []
    monthly: dict[str, dict] = {}

    for region in REGIONS:
        lat, lon = REGION_CENTROIDS.get(region, (37.5665, 126.9780))
        act = _region_actual(hourly, region, baseline_year)
        fc = _load_forecast(region, target_year)

        act_mean = float(act["power"].mean()) if act is not None and len(act) else None
        act_sum = float(act["power"].sum()) if act is not None and len(act) else None
        fc_mean = float(fc["power_pred"].mean()) if fc is not None and len(fc) else None
        fc_sum = float(fc["power_pred"].sum()) if fc is not None and len(fc) else None

        yoy_pct = None
        if act_mean is not None and fc_mean is not None and act_mean > 0:
            yoy_pct = 100.0 * (fc_mean - act_mean) / act_mean

        rows.append(
            {
                "region": region,
                "lat": lat,
                "lon": lon,
                "actual_mean_mwh": act_mean,
                "actual_sum_mwh": act_sum,
                "forecast_mean_mwh": fc_mean,
                "forecast_sum_mwh": fc_sum,
                "yoy_mean_pct": yoy_pct,
                "has_forecast": fc is not None,
            }
        )
        monthly[region] = {
            "actual": _monthly_mean(act, "power") if act is not None else {},
            "forecast": _monthly_mean(fc, "power_pred") if fc is not None else {},
        }

    return {
        "baseline_year": baseline_year,
        "target_year": target_year,
        "regions": rows,
        "monthly": monthly,
        "forecast_note": (
            "2025 예측: TSFM+GRU 롤링(서울 학습 가중치). "
            "서울은 학습·검증과 동일; 타 광역은 지역별 스케일·원핫으로 추정(참고용)."
        ),
    }


def _write_html(summary: dict, path: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = summary["regions"]
    by = summary["baseline_year"]
    ty = summary["target_year"]

    lats = [r["lat"] for r in rows]
    lons = [r["lon"] for r in rows]
    labels = [r["region"] for r in rows]

    def _sizes(vals: list[float | None], floor: float = 8.0) -> list[float]:
        present = [v for v in vals if v is not None]
        mx = max(present) if present else 1.0
        out = []
        for v in vals:
            if v is None:
                out.append(floor)
            else:
                out.append(floor + 28.0 * (v / mx))
        return out

    act_vals = [r["actual_mean_mwh"] for r in rows]
    fc_vals = [r["forecast_mean_mwh"] for r in rows]

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scattergeo"}, {"type": "scattergeo"}],
            [{"type": "bar", "colspan": 2}, None],
        ],
        subplot_titles=(
            f"{by}년 실측 (시간평균 MWh)",
            f"{ty}년 예측 (시간평균 MWh)",
            f"{by} vs {ty} 광역 비교",
        ),
        row_heights=[0.55, 0.45],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    fig.add_trace(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            text=[
                f"{lab}<br>{by} 평균 {v:,.0f} MWh" if v is not None else f"{lab}<br>데이터 없음"
                for lab, v in zip(labels, act_vals, strict=True)
            ],
            mode="markers+text",
            textposition="top center",
            marker=dict(size=_sizes(act_vals), color="#2563eb", opacity=0.85, line=dict(width=1, color="#fff")),
            name=f"{by} 실측",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fc_text = []
    fc_sizes = []
    for r in rows:
        v = r["forecast_mean_mwh"]
        if v is not None:
            yoy = r["yoy_mean_pct"]
            yoy_s = f"{yoy:+.1f}%" if yoy is not None else "—"
            fc_text.append(f"{r['region']}<br>{ty} 예측 {v:,.0f} MWh<br>vs {by} {yoy_s}")
            fc_sizes.append(v)
        else:
            fc_text.append(f"{r['region']}<br>{ty} 예측 없음")
            fc_sizes.append(None)

    fig.add_trace(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            text=fc_text,
            mode="markers+text",
            textposition="top center",
            marker=dict(
                size=_sizes(fc_sizes),
                color="#f97316",
                opacity=0.9,
                line=dict(width=1, color="#fff"),
            ),
            name=f"{ty} 예측",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    x = labels
    fig.add_trace(
        go.Bar(name=f"{by} 실측", x=x, y=[v if v is not None else 0 for v in act_vals], marker_color="#2563eb"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            name=f"{ty} 예측",
            x=x,
            y=[v if v is not None else np.nan for v in fc_vals],
            marker_color="#f97316",
        ),
        row=2,
        col=1,
    )

    fig.update_geos(
        scope="asia",
        center=dict(lat=36.4, lon=127.8),
        projection_scale=6.5,
        showland=True,
        landcolor="#f1f5f9",
        showcountries=True,
        countrycolor="#cbd5e1",
        showlakes=False,
        bgcolor="white",
    )
    fig.update_layout(
        title=f"광역 전력 {by} 실측 vs {ty} 예측",
        height=920,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=0.02),
        margin=dict(t=80, b=40),
    )
    fig.update_yaxes(title_text="시간평균 전력 (MWh)", row=2, col=1)

    # Monthly lines (one chart per region in tabs via updatemenus)
    month_figs: list[go.Figure] = []
    months = list(range(1, 13))
    for region in REGIONS:
        m = summary["monthly"][region]
        ya = [m["actual"].get(mon) for mon in months]
        yf = [m["forecast"].get(mon) for mon in months]
        mf = go.Figure()
        if any(v is not None for v in ya):
            mf.add_trace(
                go.Scatter(x=months, y=ya, name=f"{by} 실측", mode="lines+markers", line=dict(color="#2563eb"))
            )
        if any(v is not None for v in yf):
            mf.add_trace(
                go.Scatter(
                    x=months, y=yf, name=f"{ty} 예측", mode="lines+markers", line=dict(color="#f97316")
                )
            )
        mf.update_layout(
            title=f"{region} 월별 시간평균 (MWh)",
            xaxis=dict(dtick=1, title="월"),
            yaxis_title="MWh",
            height=360,
            margin=dict(t=50, b=40),
        )
        month_figs.append((region, mf))

    table_rows = ""
    for r in rows:
        act = r["actual_mean_mwh"]
        fc = r["forecast_mean_mwh"]
        yoy = r["yoy_mean_pct"]
        act_s = f"{act:,.1f}" if act is not None else "—"
        fc_s = f"{fc:,.1f}" if fc is not None else "—"
        yoy_s = f"{yoy:+.2f}%" if yoy is not None else "—"
        badge = "" if r["has_forecast"] else ' <span style="color:#b45309">(예측 없음)</span>'
        table_rows += (
            f"<tr><td>{r['region']}</td><td>{act_s}</td><td>{fc_s}{badge}</td>"
            f"<td>{yoy_s}</td></tr>"
        )

    main_json = fig.to_json()
    monthly_html = ""
    monthly_data = {}
    for region, mf in month_figs:
        monthly_data[region] = json.loads(mf.to_json())
    monthly_data_json = json.dumps(monthly_data, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>광역 {by} vs {ty} 비교</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; background: #f8fafc; color: #0f172a; }}
.wrap {{ max-width: 1100px; margin: 0 auto; padding: 20px 16px 40px; }}
h1 {{ margin: 0 0 8px; font-size: 1.35rem; }}
.note {{ color: #475569; font-size: 0.92rem; line-height: 1.5; margin-bottom: 16px; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }}
.chip {{ padding: 8px 14px; border: 2px solid #e2e8f0; border-radius: 8px; background: #fff;
  cursor: pointer; font-size: 13px; }}
.chip.active {{ border-color: #2563eb; font-weight: 600; }}
#main, #monthly {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #fff;
  border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }}
th, td {{ padding: 10px 12px; text-align: right; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
th {{ background: #f1f5f9; text-align: center; }}
td:first-child, th:first-child {{ text-align: left; }}
</style>
</head>
<body>
<div class="wrap">
<h1>광역 전력 {by} 실측 vs {ty} 예측</h1>
<p class="note">{summary["forecast_note"]}</p>
<div id="main"></div>
<h2 style="margin-top:24px;font-size:1.05rem">월별 패턴 (지역 선택)</h2>
<div id="chips" class="chips"></div>
<div id="monthly"></div>
<table>
<thead><tr><th>지역</th><th>{by} 시간평균 (MWh)</th><th>{ty} 시간평균 (MWh)</th><th>YoY</th></tr></thead>
<tbody>{table_rows}</tbody>
</table>
</div>
<script>
const MAIN = {main_json};
const MONTHLY = {monthly_data_json};
Plotly.newPlot('main', MAIN.data, MAIN.layout, {{responsive: true}});

const regions = {json.dumps(list(REGIONS), ensure_ascii=False)};
let active = regions[0];
function renderMonth() {{
  const d = MONTHLY[active];
  Plotly.newPlot('monthly', d.data, d.layout, {{responsive: true}});
}}
const chips = document.getElementById('chips');
regions.forEach(r => {{
  const b = document.createElement('button');
  b.className = 'chip' + (r === active ? ' active' : '');
  b.textContent = r;
  b.onclick = () => {{
    active = r;
    [...chips.children].forEach(c => c.classList.remove('active'));
    b.classList.add('active');
    renderMonth();
  }};
  chips.appendChild(b);
}});
renderMonth();
</script>
</body></html>"""
    path.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regional map: baseline actuals vs forward forecast.")
    parser.add_argument("--hourly", type=Path, default=REGIONAL_HOURLY_PARQUET)
    parser.add_argument("--baseline-year", type=int, default=2024)
    parser.add_argument("--target-year", type=int, default=2025)
    parser.add_argument("--output-html", type=Path, default=FORECAST_REGION_COMPARE_HTML)
    parser.add_argument("--output-json", type=Path, default=FORECAST_REGION_COMPARE_JSON)
    args = parser.parse_args(argv)

    if not args.hourly.is_file():
        raise SystemExit(f"Missing {args.hourly}. Run materialize first.")

    hourly = load_hourly(args.hourly)
    summary = build_summary(hourly, baseline_year=args.baseline_year, target_year=args.target_year)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_html(summary, args.output_html)

    n_fc = sum(1 for r in summary["regions"] if r["has_forecast"])
    print(f"Wrote {args.output_html}")
    print(f"Wrote {args.output_json}")
    print(f"Regions with {args.target_year} forecast: {n_fc}/{len(REGIONS)}")
    if n_fc < len(REGIONS):
        print(
            "  → Run: uv run python -m pipelines.seoul.hybrid_forecast "
            f"--all-regions --target-year {args.target_year}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
