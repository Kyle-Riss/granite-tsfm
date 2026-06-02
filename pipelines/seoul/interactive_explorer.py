# Copyright contributors to the TSFM project
#
"""광역 4개 → 클릭 드릴다운 · 탭별 단일 차트 탐색형 HTML.

  uv run python -m pipelines.seoul.interactive_explorer
  open artifacts/seoul/explorer.html
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
    DISTRICT_MONTHLY_PARQUET,
    EXPLORER_HTML,
    HYBRID_HOLDOUT_FORECAST_PARQUET,
    MODEL_COMPARE_JSON,
    REGIONAL_HOURLY_PARQUET,
    SENSITIVITY_51_JSON,
)
from pipelines.seoul.district_geo import DISTRICT_CENTROIDS, REGION_CENTROIDS
from pipelines.seoul.policy_scenarios import build_policy_payload

SCATTER_N = 2500


def _require_plotly() -> None:
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


def _downsample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    rng = np.random.default_rng(0)
    return df.iloc[rng.choice(len(df), size=n, replace=False)]


def build_payload(
    hourly: pd.DataFrame,
    district: pd.DataFrame | None,
    sensitivity: dict | None,
) -> dict:
    hourly = hourly.copy()
    hourly["ts"] = pd.to_datetime(hourly["ts"])
    regions = sorted(hourly["region"].unique(), key=str)

    sens_regions = (sensitivity or {}).get("regions", {})
    national: dict = {
        "regions": [str(r) for r in regions],
        "region_meta": {},
        "map": {"lat": [], "lon": [], "text": [], "size": [], "region": []},
        "compare_power": [],
        "compare_corr": [],
    }

    profiles: dict = {}
    scatters: dict = {}
    month_hour: dict = {}
    band_power: dict = {}

    for reg in regions:
        reg_s = str(reg)
        g = hourly[hourly["region"] == reg]
        lat, lon = REGION_CENTROIDS.get(reg_s, (37.5665, 126.9780))
        sr = sens_regions.get(reg_s, {})
        meta = {
            "lat": lat,
            "lon": lon,
            "mean_power": float(g["power"].mean()),
            "corr_temp_power": sr.get("corr_temp_power"),
            "corr_cdd_power": sr.get("corr_cdd_power"),
            "corr_hdd_power": sr.get("corr_hdd_power"),
            "power_per_capita": sr.get("power_per_capita"),
            "has_districts": reg_s == "서울시" and district is not None and len(district) > 0,
        }
        national["region_meta"][reg_s] = meta
        mx = float(g["power"].mean())
        national["map"]["lat"].append(lat)
        national["map"]["lon"].append(lon)
        national["map"]["region"].append(reg_s)
        if meta["corr_temp_power"] is not None:
            hover = f"<b>{reg_s}</b><br>평균 전력 {mx:,.0f}<br>r(기온)={meta['corr_temp_power']:.3f}"
        else:
            hover = f"<b>{reg_s}</b><br>평균 전력 {mx:,.0f}"
        national["map"]["text"].append(hover)
        national["map"]["size"].append(mx)
        national["compare_power"].append({"region": reg_s, "value": mx})
        c = meta["corr_temp_power"]
        if c is not None:
            national["compare_corr"].append({"region": reg_s, "value": c})

        prof = g.groupby(["is_weekend", "hour"], observed=True)["power"].mean().reset_index()
        profiles[reg_s] = {
            "weekday": {int(r["hour"]): float(r["power"]) for _, r in prof[prof["is_weekend"] == 0].iterrows()},
            "weekend": {int(r["hour"]): float(r["power"]) for _, r in prof[prof["is_weekend"] == 1].iterrows()},
        }

        sc = _downsample(g, SCATTER_N)
        scatters[reg_s] = {
            "temp": sc["temp"].astype(float).tolist(),
            "power": sc["power"].astype(float).tolist(),
        }

        g2 = g.copy()
        g2["month"] = g2["ts"].dt.month
        mh = g2.groupby(["month", "hour"], observed=True)["power"].mean().unstack(fill_value=np.nan)
        month_hour[reg_s] = {
            "months": [int(m) for m in mh.index],
            "hours": [int(h) for h in mh.columns],
            "z": [[None if np.isnan(v) else float(v) for v in row] for row in mh.values],
        }

        bp = g.groupby("temp_band", observed=True)["power"].mean()
        band_power[reg_s] = {
            "bands": ["heating", "mid", "cooling"],
            "values": [float(bp.get(b, np.nan)) for b in ("heating", "mid", "cooling")],
        }

    seoul_districts: dict | None = None
    if district is not None and len(district):
        dm = _attach_coords(district)
        by_month: dict[str, list] = {}
        for m in sorted(dm["month"].unique()):
            sub = dm[dm["month"] == m]
            mx = float(sub["usage"].max()) or 1.0
            by_month[str(int(m))] = [
                {
                    "district": str(r["district"]),
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                    "usage": float(r["usage"]),
                    "size": float(r["usage"]) / mx * 36 + 14,
                }
                for _, r in sub.iterrows()
            ]
        annual = dm.groupby("district", observed=True)["usage"].sum().sort_values(ascending=False)
        seoul_districts = {
            "by_month": by_month,
            "months": sorted(int(m) for m in dm["month"].unique()),
            "top_district": str(annual.index[0]) if len(annual) else "",
            "top_usage": float(annual.iloc[0]) if len(annual) else 0.0,
        }

    policy = build_policy_payload(hourly, sensitivity, district)

    forecast = None
    if HYBRID_HOLDOUT_FORECAST_PARQUET.is_file():
        fc = pd.read_parquet(HYBRID_HOLDOUT_FORECAST_PARQUET)
        forecast = {"ts": fc["ts"].astype(str).tolist(), "actual": fc["power_actual"].tolist()}
        for col, key in [
            ("power_pred_hybrid", "hybrid"),
            ("power_pred_gru_only", "gru_only"),
            ("power_pred_ttm", "ttm"),
            ("power_pred_naive", "naive"),
        ]:
            if col in fc.columns:
                forecast[key] = fc[col].tolist()
    compare = None
    if MODEL_COMPARE_JSON.is_file():
        compare = json.loads(MODEL_COMPARE_JSON.read_text(encoding="utf-8"))

    return {
        "national": national,
        "profiles": profiles,
        "scatters": scatters,
        "month_hour": month_hour,
        "band_power": band_power,
        "seoul_districts": seoul_districts,
        "policy": policy,
        "forecast": forecast,
        "model_compare": compare,
    }


def render_html(payload: dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>광역 에너지 탐색</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{ --bg:#f1f5f9; --card:#fff; --line:#e2e8f0; --ink:#0f172a; --muted:#64748b; --accent:#2563eb; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:system-ui,-apple-system,sans-serif; background:var(--bg); color:var(--ink); }}
.wrap {{ max-width:1180px; margin:0 auto; padding:16px 20px 32px; }}
.top {{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:12px; }}
.bc {{ font-size:14px; color:var(--muted); }}
.bc button {{ background:none; border:none; color:var(--accent); cursor:pointer; font:inherit; padding:0; }}
.bc button:hover {{ text-decoration:underline; }}
.bc .sep {{ margin:0 6px; color:#cbd5e1; }}
h1 {{ margin:0; font-size:1.25rem; flex:1; min-width:200px; }}
.hint {{ font-size:13px; color:var(--muted); margin:0 0 14px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px; }}
.chip {{ padding:10px 16px; border:2px solid var(--line); border-radius:10px; background:var(--card);
  cursor:pointer; font-size:14px; transition:border-color .15s, box-shadow .15s; }}
.chip:hover {{ border-color:#93c5fd; }}
.chip.active {{ border-color:var(--accent); box-shadow:0 0 0 2px #dbeafe; font-weight:600; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin-bottom:14px; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }}
.kpi .l {{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.03em; }}
.kpi .v {{ font-size:1.15rem; font-weight:600; margin-top:4px; }}
.tabs {{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }}
.tab {{ padding:8px 14px; border:1px solid var(--line); border-radius:8px; background:var(--card);
  cursor:pointer; font-size:13px; }}
.tab:disabled {{ opacity:.4; cursor:not-allowed; }}
.tab.active {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
#chart {{ background:var(--card); border:1px solid var(--line); border-radius:10px; min-height:480px; }}
.month-bar {{ display:flex; flex-wrap:wrap; gap:4px; margin:0 0 10px; }}
.month-bar button {{ padding:6px 10px; border:1px solid var(--line); border-radius:6px; background:var(--card);
  cursor:pointer; font-size:12px; }}
.month-bar button.active {{ background:#1e40af; color:#fff; border-color:#1e40af; }}
.month-bar.hidden {{ display:none; }}
.policy.hidden {{ display:none; }}
.policy {{ margin-bottom:14px; }}
.policy h3 {{ margin:0 0 8px; font-size:14px; color:var(--muted); }}
.policy-grid {{ display:grid; gap:10px; }}
.policy-card {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }}
.policy-card .tag {{ font-size:11px; color:var(--accent); font-weight:600; }}
.policy-card h4 {{ margin:6px 0 4px; font-size:14px; }}
.policy-card p {{ margin:0; font-size:13px; line-height:1.5; color:#334155; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <nav class="bc" id="breadcrumb"></nav>
    <h1 id="title">광역 에너지 탐색</h1>
  </div>
  <p class="hint" id="hint">지역 카드를 클릭해 들어가세요. 차트는 탭으로 하나씩 봅니다.</p>
  <div id="chips" class="chips"></div>
  <div id="kpis" class="kpis"></div>
  <section id="policy" class="policy hidden"></section>
  <div id="monthBar" class="month-bar hidden"></div>
  <div id="tabs" class="tabs"></div>
  <div id="chart"></div>
</div>
<script>
const DATA = {data_json};
const state = {{ level: "national", region: null, tab: "map", month: 1 }};

const TAB_LABELS = {{
  map: "지도",
  profile: "시간대 프로파일",
  scatter: "기온·전력",
  heatmap: "월×시간 부하",
  bands: "냉·중·난방 구간",
  compare: "광역 비교",
  forecast: "96h 예측 (서울)",
}};

function fmt(n, d=0) {{
  if (n == null || Number.isNaN(n)) return "—";
  return n.toLocaleString("ko-KR", {{ maximumFractionDigits: d }});
}}

function setState(patch) {{
  Object.assign(state, patch);
  render();
}}

function goNational() {{
  setState({{ level: "national", region: null, tab: "map" }});
}}

function selectRegion(reg) {{
  const meta = DATA.national.region_meta[reg];
  const tab = meta && meta.has_districts ? "map" : "profile";
  setState({{ level: "region", region: reg, tab, month: 1 }});
}}

function renderBreadcrumb() {{
  const el = document.getElementById("breadcrumb");
  if (state.level === "national") {{
    el.innerHTML = "<span>전국</span>";
    return;
  }}
  el.innerHTML = `<button type="button" onclick="goNational()">전국</button>
    <span class="sep">›</span><span>${{state.region}}</span>`;
}}

function renderChips() {{
  const el = document.getElementById("chips");
  el.innerHTML = DATA.national.regions.map(r => {{
    const active = state.region === r ? " active" : "";
    return `<button type="button" class="chip${{active}}" data-reg="${{r}}">${{r}}</button>`;
  }}).join("");
  el.querySelectorAll(".chip").forEach(btn => {{
    btn.onclick = () => selectRegion(btn.dataset.reg);
  }});
}}

function renderKpis() {{
  const el = document.getElementById("kpis");
  if (state.level === "national") {{
    el.innerHTML = `
      <div class="kpi"><span class="l">광역 수</span><span class="v">${{DATA.national.regions.length}}</span></div>
      <div class="kpi"><span class="l">안내</span><span class="v" style="font-size:13px">카드 클릭</span></div>`;
    return;
  }}
  const m = DATA.national.region_meta[state.region] || {{}};
  el.innerHTML = `
    <div class="kpi"><span class="l">평균 전력</span><span class="v">${{fmt(m.mean_power)}}</span></div>
    <div class="kpi"><span class="l">r(기온·전력)</span><span class="v">${{m.corr_temp_power != null ? m.corr_temp_power.toFixed(3) : "—"}}</span></div>
    <div class="kpi"><span class="l">r(CDD·전력)</span><span class="v">${{m.corr_cdd_power != null ? m.corr_cdd_power.toFixed(3) : "—"}}</span></div>
    <div class="kpi"><span class="l">인구당(상대)</span><span class="v">${{m.power_per_capita != null ? fmt(m.power_per_capita, 4) : "—"}}</span></div>`;
}}

function renderTabs() {{
  const el = document.getElementById("tabs");
  const meta = state.region ? DATA.national.region_meta[state.region] : null;
  let keys = state.level === "national"
    ? ["map", "compare", ...(DATA.forecast ? ["forecast"] : [])]
    : ["profile", "scatter", "heatmap", "bands"];
  if (meta && meta.has_districts) keys = ["map", ...keys];
  el.innerHTML = keys.map(k => {{
    const act = state.tab === k ? " active" : "";
    return `<button type="button" class="tab${{act}}" data-tab="${{k}}">${{TAB_LABELS[k]}}</button>`;
  }}).join("");
  el.querySelectorAll(".tab").forEach(btn => {{
    btn.onclick = () => setState({{ tab: btn.dataset.tab }});
  }});
}}

function renderPolicy() {{
  const box = document.getElementById("policy");
  const key = state.level === "national" ? "_national" : state.region;
  const items = (DATA.policy && DATA.policy[key]) || [];
  if (!items.length) {{
    box.className = "policy hidden";
    box.innerHTML = "";
    return;
  }}
  box.className = "policy";
  box.innerHTML = `<h3>6.2 정책 활용 시나리오</h3><div class="policy-grid">${{items.map(s => `
    <article class="policy-card"><span class="tag">${{s.tag}}</span>
    <h4>${{s.title}}</h4><p>${{s.body}}</p></article>`).join("")}}</div>`;
}}

function renderMonthBar() {{
  const bar = document.getElementById("monthBar");
  const meta = state.region && DATA.national.region_meta[state.region];
  if (state.level !== "region" || state.tab !== "map" || !meta || !meta.has_districts || !DATA.seoul_districts) {{
    bar.className = "month-bar hidden";
    return;
  }}
  bar.className = "month-bar";
  bar.innerHTML = DATA.seoul_districts.months.map(m => {{
    const act = state.month === m ? " active" : "";
    return `<button type="button" class="${{act}}" data-m="${{m}}">${{m}}월</button>`;
  }}).join("");
  bar.querySelectorAll("button").forEach(btn => {{
    btn.onclick = () => setState({{ month: parseInt(btn.dataset.m, 10) }});
  }});
}}

function plotNationalMap() {{
  const m = DATA.national.map;
  const mx = Math.max(...m.size, 1);
  const sizes = m.size.map(v => v / mx * 42 + 18);
  return [{{
    type: "scattermapbox",
    lat: m.lat, lon: m.lon, mode: "markers+text",
    text: m.region, textposition: "top center",
    marker: {{ size: sizes, color: m.size, colorscale: "Blues", showscale: true,
      colorbar: {{ title: "평균 전력" }} }},
    hovertext: m.text, hoverinfo: "text",
    customdata: m.region,
  }}];
}}

function plotDistrictMap(month) {{
  const pts = (DATA.seoul_districts.by_month[String(month)] || []);
  return [{{
    type: "scattermapbox",
    lat: pts.map(p => p.lat), lon: pts.map(p => p.lon), mode: "markers",
    marker: {{ size: pts.map(p => p.size), color: pts.map(p => p.usage), colorscale: "Blues",
      showscale: true, colorbar: {{ title: "사용량" }} }},
    text: pts.map(p => `${{p.district}}<br>${{p.usage.toLocaleString()}}`),
    hoverinfo: "text",
  }}];
}}

function plotProfile(reg) {{
  const p = DATA.profiles[reg];
  const hours = [...new Set([...Object.keys(p.weekday), ...Object.keys(p.weekend)])].map(Number).sort((a,b)=>a-b);
  const wd = hours.map(h => p.weekday[h] ?? null);
  const we = hours.map(h => p.weekend[h] ?? null);
  return [
    {{ x: hours, y: wd, type: "scatter", mode: "lines", name: "평일", line: {{ color: "#2563eb", width: 2 }} }},
    {{ x: hours, y: we, type: "scatter", mode: "lines", name: "주말", line: {{ color: "#dc2626", width: 2 }} }},
  ];
}}

function buildFigure() {{
  let traces = [], layout = {{ height: 500, margin: {{ t: 40, r: 16, b: 48, l: 56 }},
    paper_bgcolor: "#fff", plot_bgcolor: "#fafafa", font: {{ family: "system-ui,sans-serif" }} }};
  const reg = state.region;

  if (state.level === "national") {{
    if (state.tab === "forecast" && DATA.forecast) {{
      const f = DATA.forecast;
      traces = [{{ x: f.ts, y: f.actual, type: "scatter", mode: "lines", name: "실측", line: {{ width: 2 }} }}];
      const extras = [
        ["hybrid", "TSFM+GRU", "#f97316"],
        ["gru_only", "GRU only", "#22c55e"],
        ["ttm", "TTM zeroshot", "#a855f7"],
        ["naive", "Naive", "#94a3b8"],
      ];
      extras.forEach(([k, name, color]) => {{
        if (f[k]) traces.push({{ x: f.ts, y: f[k], type: "scatter", mode: "lines", name, line: {{ color }} }});
      }});
      layout.title = "서울 96h 홀드아웃 — 하이브리드 vs 베이스라인";
      layout.xaxis = {{ title: "시간" }}; layout.yaxis = {{ title: "전력" }};
      layout.showlegend = true;
      return {{ traces, layout }};
    }}
    if (state.tab === "compare") {{
      const rows = DATA.national.compare_power;
      traces = [{{ x: rows.map(r => r.region), y: rows.map(r => r.value), type: "bar",
        marker: {{ color: "#2563eb" }}, hovertemplate: "%{{y:,.0f}}<extra></extra>" }}];
      layout.title = "광역 평균 전력 비교";
      layout.xaxis = {{ title: "지역" }}; layout.yaxis = {{ title: "전력" }};
      return {{ traces, layout }};
    }}
    traces = plotNationalMap();
    layout.mapbox = {{ style: "open-street-map", center: {{ lat: 36.2, lon: 127.5 }}, zoom: 6.4 }};
    layout.title = "광역 4개 — 마커 클릭 또는 상단 카드로 진입";
    layout.clickmode = "event+select";
    return {{ traces, layout }};
  }}

  if (state.tab === "map" && DATA.national.region_meta[reg].has_districts) {{
    traces = plotDistrictMap(state.month);
    layout.mapbox = {{ style: "open-street-map", center: {{ lat: 37.5665, lon: 126.978 }}, zoom: 10.2 }};
    layout.title = `서울 자치구 — ${{state.month}}월 (연간 1위: ${{DATA.seoul_districts.top_district}})`;
    return {{ traces, layout }};
  }}
  if (state.tab === "profile") {{
    traces = plotProfile(reg);
    layout.title = `${{reg}} 시간대 프로파일`;
    layout.xaxis = {{ title: "시(h)", dtick: 2 }}; layout.yaxis = {{ title: "평균 전력" }};
    layout.showlegend = true;
    return {{ traces, layout }};
  }}
  if (state.tab === "scatter") {{
    const s = DATA.scatters[reg];
    traces = [{{ x: s.temp, y: s.power, mode: "markers", type: "scatter",
      marker: {{ size: 5, opacity: 0.15, color: s.temp, colorscale: "RdYlBu_r" }},
      hovertemplate: "%{{x:.1f}}°C · %{{y:,.0f}}<extra></extra>" }}];
    layout.title = `${{reg}} 기온–전력`;
    layout.xaxis = {{ title: "기온(°C)" }}; layout.yaxis = {{ title: "전력" }};
    return {{ traces, layout }};
  }}
  if (state.tab === "heatmap") {{
    const mh = DATA.month_hour[reg];
    traces = [{{ z: mh.z, x: mh.hours, y: mh.months, type: "heatmap", colorscale: "Blues",
      hovertemplate: "%{{y}}월 %{{x}}시 · %{{z:,.0f}}<extra></extra>" }}];
    layout.title = `${{reg}} 월×시간 평균 부하`;
    layout.xaxis = {{ title: "시" }}; layout.yaxis = {{ title: "월" }};
    return {{ traces, layout }};
  }}
  if (state.tab === "bands") {{
    const b = DATA.band_power[reg];
    traces = [{{ x: ["난방≤10°", "중간", "냉방≥26°"], y: b.values, type: "bar",
      marker: {{ color: ["#3b82f6","#94a3b8","#ef4444"] }},
      hovertemplate: "%{{y:,.0f}}<extra></extra>" }}];
    layout.title = `${{reg}} 기온 구간별 평균 전력`;
    layout.yaxis = {{ title: "전력" }};
    return {{ traces, layout }};
  }}
  traces = plotProfile(reg);
  layout.title = reg;
  return {{ traces, layout }};
}}

function renderChart() {{
  const {{ traces, layout }} = buildFigure();
  const cfg = {{ responsive: true, displayModeBar: true, displaylogo: false }};
  Plotly.react("chart", traces, layout, cfg).then(gd => {{
    if (state.level !== "national" || state.tab !== "map") return;
    gd.removeAllListeners("plotly_click");
    gd.on("plotly_click", ev => {{
      const pt = ev.points && ev.points[0];
      const reg = pt && (pt.customdata || pt.text);
      if (reg && DATA.national.region_meta[reg]) selectRegion(reg);
    }});
  }});
}}

function render() {{
  document.getElementById("title").textContent =
    state.level === "national" ? "광역 에너지 탐색" : `${{state.region}} 상세`;
  document.getElementById("hint").textContent =
    state.level === "national"
      ? "지도·비교·96h 예측 탭을 사용하세요. 예측 탭은 hybrid_compare 실행 후 표시됩니다."
      : "탭으로 차트를 바꿉니다. 서울은 지도 탭에서 월별 자치구로 드릴다운됩니다.";
  renderBreadcrumb();
  renderChips();
  renderKpis();
  renderTabs();
  renderPolicy();
  renderMonthBar();
  renderChart();
}}

render();
</script>
</body></html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive regional energy explorer HTML.")
    parser.add_argument("--hourly", type=Path, default=REGIONAL_HOURLY_PARQUET)
    parser.add_argument("--sensitivity", type=Path, default=SENSITIVITY_51_JSON)
    parser.add_argument("--output", type=Path, default=EXPLORER_HTML)
    args = parser.parse_args(argv)

    _require_plotly()
    if not args.hourly.is_file():
        raise SystemExit("Run materialize first.")

    sens = json.loads(args.sensitivity.read_text(encoding="utf-8")) if args.sensitivity.is_file() else None
    hourly = pd.read_parquet(args.hourly)
    dm = pd.read_parquet(DISTRICT_MONTHLY_PARQUET) if DISTRICT_MONTHLY_PARQUET.is_file() else None

    payload = build_payload(hourly, dm, sens)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(payload), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

