# Copyright contributors to the TSFM project
#
"""
가설 3.1–3.3 지원용 탐색 리포트 (34k 시계열 + 자치구 월별).

- 3.1: 기온–전력 산점도 + 1°C 구간 중앙값 곡선(U형 가시화). TSFM 잔차 분석은
  시계열 전 구간 예측값이 있을 때 확장 가능(현재는 통계적 기준선만).
- 3.2: 광역별 단순 선형 민감도(기온→전력 기울기) 비교, 월별 피크 패턴(강원 겨울 vs 여름 등).
- 3.3: 평일 vs 주말 시간대별 평균 전력 곡선(is_weekend × hour).

자치구 301: 월별 데이터만 있으면 구별 ‘시간대 임계 기온’은 산출 불가 —
구별 계절 진폭(월별 범위) 등 컨텍스트 지표로 대체 가능.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.seoul.config import (
    ARTIFACTS_DIR,
    DISTRICT_MONTHLY_PARQUET,
    REGIONAL_HOURLY_PARQUET,
    REPORT_HYPOTHESES_HTML,
)


def _require_plotly():
    try:
        import plotly.graph_objects as go  # noqa: F401
        from plotly.subplots import make_subplots  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "plotly is required. Install with: uv sync --extra pipeline  " "or pip install 'granite-tsfm[pipeline]'"
        ) from e


def _u_curve_binned(df: pd.DataFrame, temp_bin: float) -> pd.DataFrame:
    """1°C 또는 사용자 지정 폭으로 구간화한 중앙값·건수."""
    tb = (df["temp"] / temp_bin).round() * temp_bin
    return (
        df.assign(temp_bin=tb)
        .groupby("temp_bin", as_index=False)
        .agg(power_median=("power", "median"), n=("power", "count"))
    )


def _lin_slope(group: pd.DataFrame) -> float:
    x = group["temp"].to_numpy()
    y = group["power"].to_numpy()
    if len(x) < 30 or np.nanstd(x) < 1e-9:
        return float("nan")
    coef = np.polyfit(x, y, 1)
    return float(coef[0])


def build_html(
    *,
    hourly_path: Path,
    district_path: Path | None,
    temp_bin_width: float,
    scatter_sample: int,
    output_html: Path,
) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = pd.read_parquet(hourly_path)
    for col in ("temp", "power", "region", "hour", "is_weekend"):
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {hourly_path}")

    rng = np.random.default_rng(42)
    n = min(scatter_sample, len(df))
    sample_idx = rng.choice(len(df), size=n, replace=False)
    sc = df.iloc[sample_idx]

    curve = _u_curve_binned(df, temp_bin_width)

    slopes_list = []
    for reg in sorted(df["region"].unique()):
        g = df.loc[df["region"] == reg]
        slopes_list.append({"region": reg, "slope_temp_power": _lin_slope(g)})
    slopes = pd.DataFrame(slopes_list)

    prof = df.groupby(["is_weekend", "hour"], observed=True)["power"].mean().reset_index()
    wd = prof[prof["is_weekend"] == 0].sort_values("hour")
    we = prof[prof["is_weekend"] == 1].sort_values("hour")

    monthly_region = (
        df.assign(month=pd.to_datetime(df["ts"]).dt.month)
        .groupby(["region", "month"], observed=True)["power"]
        .mean()
        .reset_index()
    )

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "3.1 산점도(표본) + 기온 구간 중앙값",
            "3.1 구간별 건수",
            "3.2 광역별 선형 민감도(기울기)",
            "3.2 월별 평균 전력(광역)",
            "3.3 평일 vs 주말 시간대 프로파일",
            "자치구 월별 계절 진폭(프록시)",
        ),
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(
            x=sc["temp"],
            y=sc["power"],
            mode="markers",
            name="표본",
            marker={"opacity": 0.15, "size": 4},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=curve["temp_bin"],
            y=curve["power_median"],
            mode="lines+markers",
            name=f"{temp_bin_width:g}° 구간 중앙값",
            line={"color": "red", "width": 3},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=curve["temp_bin"], y=curve["n"], name="구간 건수"),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Bar(x=slopes["region"], y=slopes["slope_temp_power"], name="기울기"),
        row=2,
        col=1,
    )

    for reg in sorted(df["region"].unique()):
        sub = monthly_region[monthly_region["region"] == reg]
        fig.add_trace(
            go.Scatter(x=sub["month"], y=sub["power"], mode="lines+markers", name=str(reg)),
            row=2,
            col=2,
        )

    fig.add_trace(
        go.Scatter(x=wd["hour"], y=wd["power"], name="평일", line={"color": "#2563eb"}),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=we["hour"], y=we["power"], name="주말", line={"color": "#dc2626", "dash": "dash"}),
        row=3,
        col=1,
    )

    if district_path and district_path.is_file():
        dm = pd.read_parquet(district_path)
        amp = dm.groupby("district", observed=True)["usage"].agg(lambda s: float(s.max() - s.min())).reset_index()
        amp.columns = ["district", "seasonal_range"]
        amp = amp.sort_values("seasonal_range", ascending=False).head(25)
        fig.add_trace(
            go.Bar(x=amp["district"], y=amp["seasonal_range"], name="월별 최대–최소"),
            row=3,
            col=2,
        )
    else:
        fig.add_annotation(
            text="자치구 Parquet 없음",
            xref="paper",
            yref="paper",
            x=0.82,
            y=0.08,
            showarrow=False,
            row=3,
            col=2,
        )

    fig.update_layout(
        height=1200,
        title_text=(
            "가설 입증용 탐색 리포트 · 선형 기울기는 단순 OLS 프록시 · "
            "TSFM은 비선형·장기 패턴을 학습한 기준 모델로 잔차 분석 시 고급 검정 가능"
        ),
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"t": 100},
    )
    fig.update_xaxes(title_text="기온 (°C)", row=1, col=1)
    fig.update_yaxes(title_text="전력", row=1, col=1)
    fig.update_xaxes(title_text="기온 구간", row=1, col=2)
    fig.update_xaxes(title_text="광역", row=2, col=1)
    fig.update_yaxes(title_text="기울기", row=2, col=1)
    fig.update_xaxes(title_text="월", row=2, col=2)
    fig.update_yaxes(title_text="평균 전력", row=2, col=2)
    fig.update_xaxes(title_text="시간(h)", row=3, col=1)
    fig.update_yaxes(title_text="평균 전력", row=3, col=1)

    preamble = """<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>Hypothesis evidence</title></head>
<body style="font-family:system-ui,sans-serif;max-width:1400px;margin:auto;padding:16px;">
<h2>3.1–3.3 분석 로직과 이 리포트의 한계</h2>
<ul>
<li><b>3.1 U형·임계치</b>: 빨간 곡선은 동일 데이터의 기온 구간별 <b>중앙값</b>입니다.
      급증 구간은 기울기·분위수로 추가 정의할 수 있고, TSFM <b>잔차</b>는 전 구간 예측 저장 후
      기온 구간별로 집계하면 됩니다.</li>
<li><b>3.2 지역</b>: 막대는 단순 선형 기울기(프록시)입니다. 인구·면적 보정은 population 컬럼과 결합해
      per-capita 등으로 확장 가능합니다. 강원 겨울 피크는 우측 월별 곡선에서 확인합니다.</li>
<li><b>3.3 시간</b>: is_weekend×hour 평균 곡선 대조입니다. 패치 가중치 해석은 모델 어텐션 추출이 필요합니다.</li>
<li><b>301 자치구</b>: 월별만 있으면 구별 <b>시간대 임계 기온</b>은 불가 ·
      구별 계절 진폭(하단)은 서술적 근거용 프록시입니다.</li>
</ul>
"""

    html = preamble + fig.to_html(full_html=False, include_plotlyjs="cdn") + "</body></html>"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hypothesis 3.1–3.3 exploratory Plotly report.")
    parser.add_argument("--hourly-parquet", type=Path, default=REGIONAL_HOURLY_PARQUET)
    parser.add_argument("--district-parquet", type=Path, default=DISTRICT_MONTHLY_PARQUET)
    parser.add_argument("--no-district", action="store_true")
    parser.add_argument("--temp-bin", type=float, default=1.0, help="기온 구간 폭 (°C), 기본 1도.")
    parser.add_argument("--scatter-sample", type=int, default=8000, help="산점도 표본 수(브라우저 부하).")
    parser.add_argument("--output", type=Path, default=REPORT_HYPOTHESES_HTML)
    args = parser.parse_args(argv)

    _require_plotly()
    if not args.hourly_parquet.is_file():
        raise SystemExit(f"Missing {args.hourly_parquet}. Run: python -m pipelines.seoul.materialize")

    build_html(
        hourly_path=args.hourly_parquet,
        district_path=None if args.no_district else args.district_parquet,
        temp_bin_width=args.temp_bin,
        scatter_sample=args.scatter_sample,
        output_html=args.output,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
