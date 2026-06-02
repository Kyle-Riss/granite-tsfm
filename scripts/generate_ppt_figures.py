"""
Generate PPT figure PNGs for all IMAGE SLOTs.
Output: artifacts/seoul/ppt_figures/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import linregress

# ── 공통 팔레트 (PPT 디자인 지침) ─────────────────────────────────
NAVY    = "#1e3a5f"
BLUE    = "#0ea5e9"
GREEN   = "#22c55e"
ORANGE  = "#f97316"
GRAY    = "#94a3b8"
RED     = "#ef4444"
BG      = "#f8fafc"
YELLOW  = "#fbbf24"

import matplotlib.font_manager as _fm
_KO_CANDIDATES = ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Malgun Gothic", "sans-serif"]
_KO_FONT = next((f for f in _KO_CANDIDATES if any(f == ff.name for ff in _fm.fontManager.ttflist)), "sans-serif")

plt.rcParams.update({
    "font.family": _KO_FONT,
    "axes.unicode_minus": False,
    "axes.facecolor": BG,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.6,
})

OUT = Path("artifacts/seoul/ppt_figures")
OUT.mkdir(parents=True, exist_ok=True)
ART = Path("artifacts/seoul")


# ══════════════════════════════════════════════════════════════════
# SLOT 4 — 기온×전력 산점도  (서울, 2020-2024)
# ══════════════════════════════════════════════════════════════════
def slot4_temp_scatter() -> None:
    print("[SLOT 4] 기온×전력 산점도 생성 중...")
    df = pd.read_parquet(ART / "seoul_city_hourly.parquet")
    seoul = df[df["region"] == "서울시"].copy()
    seoul = seoul.dropna(subset=["temp", "power"])

    fig, ax = plt.subplots(figsize=(12, 6.75))
    fig.patch.set_facecolor("white")

    # 냉방·난방·중간 band 색 배경
    ax.axvspan(-20, 10, alpha=0.08, color="#3b82f6", zorder=0, label="난방 band (≤10°C)")
    ax.axvspan(26, 45, alpha=0.08, color="#ef4444", zorder=0, label="냉방 band (≥26°C)")

    # 18°C 기준선
    ax.axvline(18, color=NAVY, linewidth=1.6, linestyle="--", zorder=3)
    ax.text(18.4, seoul["power"].quantile(0.97), "기준온 18°C",
            color=NAVY, fontsize=10, va="top", fontweight="bold")

    # 산점도 (alpha 낮게, 밀도 강조)
    sc = ax.scatter(
        seoul["temp"], seoul["power"],
        c=seoul["temp"], cmap="RdYlBu_r",
        s=5, alpha=0.25, linewidths=0, zorder=2,
        vmin=-10, vmax=38,
    )

    # band별 선형 회귀 표시
    for lo, hi, c, lbl in [
        (-20, 10, "#3b82f6", "난방 구간 회귀"),
        (26, 45, "#ef4444", "냉방 구간 회귀"),
    ]:
        sub = seoul[(seoul["temp"] >= lo) & (seoul["temp"] <= hi)]
        if len(sub) > 10:
            slope, intercept, r, *_ = linregress(sub["temp"], sub["power"])
            xs = np.linspace(lo, hi, 100)
            ax.plot(xs, slope * xs + intercept, color=c, linewidth=2.5, zorder=4, label=f"{lbl} (r={r:.2f})")

    # 전체 2차 다항식 U형 적합선
    z = np.polyfit(seoul["temp"], seoul["power"], 2)
    p = np.poly1d(z)
    xs = np.linspace(seoul["temp"].min(), seoul["temp"].max(), 300)
    ax.plot(xs, p(xs), color=NAVY, linewidth=2.5, linestyle="-", zorder=5, label="U형 2차 적합선")

    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("기온 (°C)", fontsize=10)

    ax.set_xlabel("기온 (°C)", fontsize=12)
    ax.set_ylabel("전력거래량 (MWh)", fontsize=12)
    ax.set_title("서울시 기온 × 전력거래량 (2020–2024, hourly)", fontsize=14, fontweight="bold", color=NAVY)
    ax.legend(fontsize=9, loc="upper center", ncol=4, framealpha=0.9)
    ax.set_facecolor(BG)

    # 주석: 냉방>난방 설명
    ax.text(0.02, 0.97,
            "냉방 구간(≥26°C)의 기울기 > 난방 구간(≤10°C)",
            transform=ax.transAxes, fontsize=9,
            color="#64748b", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    fig.savefig(OUT / "fig_3-1_temp_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → saved fig_3-1_temp_scatter.png")


# ══════════════════════════════════════════════════════════════════
# SLOT 5 — 지역별 temp slope 수평 막대
# ══════════════════════════════════════════════════════════════════
def slot5_region_slope() -> None:
    print("[SLOT 5] 지역별 temp slope 막대 생성 중...")
    with open(ART / "sensitivity_51.json") as f:
        s = json.load(f)

    regions_raw = s["regions"]
    data = {
        "강원도": regions_raw["강원도"]["slope_temp_power"],
        "대전시": regions_raw["대전시"]["slope_temp_power"],
        "서울시": regions_raw["서울시"]["slope_temp_power"],
        "부산시": regions_raw["부산시"]["slope_temp_power"],
    }
    # 절댓값 기준 정렬 (오름차순 → 막대 아래에 큰 값)
    labels = sorted(data.keys(), key=lambda k: abs(data[k]))
    values = [data[k] for k in labels]

    colors = []
    for k in labels:
        if k == "서울시":
            colors.append(ORANGE)
        elif data[k] > 0:
            colors.append(BLUE)
        else:
            colors.append(NAVY)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55, zorder=3)

    # 값 레이블
    for bar, v in zip(bars, values):
        offset = -0.8 if v < 0 else 0.2
        ax.text(
            v + offset, bar.get_y() + bar.get_height() / 2,
            f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
            fontsize=12, fontweight="bold", color="white" if abs(v) > 5 else NAVY,
        )

    ax.axvline(0, color=NAVY, linewidth=1.2)
    ax.set_xlabel("temp slope (MWh / °C)", fontsize=12)
    ax.set_title("지역별 기온–전력 민감도 (slope, 2020–2024)", fontsize=14, fontweight="bold", color=NAVY)
    ax.set_facecolor(BG)

    # 서울 강조 주석
    seoul_idx = labels.index("서울시")
    ax.annotate(
        "서울 slope ≈ −0.04\n(4지역 중 절댓값 최소)",
        xy=(data["서울시"], seoul_idx),
        xytext=(data["서울시"] - 6, seoul_idx + 0.55),
        fontsize=9, color=ORANGE, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2),
    )

    legend_patches = [
        mpatches.Patch(color=ORANGE, label="서울시 (강조)"),
        mpatches.Patch(color=NAVY, label="음의 slope (기온↑→전력↓ 구간)"),
        mpatches.Patch(color=BLUE, label="양의 slope (기온↑→전력↑ 구간)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT / "fig_3-2_region_slope.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → saved fig_3-2_region_slope.png")


# ══════════════════════════════════════════════════════════════════
# SLOT 6 — 시각 0–23h 평일/주말 평균 곡선
# ══════════════════════════════════════════════════════════════════
def slot6_weekday_weekend() -> None:
    print("[SLOT 6] 평일/주말 시간대별 곡선 생성 중...")
    df = pd.read_parquet(ART / "seoul_city_hourly.parquet")
    seoul = df[df["region"] == "서울시"].copy()
    seoul = seoul.dropna(subset=["power", "hour", "is_weekend"])

    weekday = seoul[seoul["is_weekend"] == 0].groupby("hour")["power"].mean()
    weekend = seoul[seoul["is_weekend"] == 1].groupby("hour")["power"].mean()

    hours = list(range(24))

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")

    # 09–18h 음영
    ax.axvspan(9, 18, alpha=0.12, color=ORANGE, zorder=0, label="평일 집중 구간 (09–18h)")

    ax.plot(hours, [weekday.get(h, np.nan) for h in hours],
            color=NAVY, linewidth=2.5, marker="o", markersize=5, label="평일 평균", zorder=4)
    ax.plot(hours, [weekend.get(h, np.nan) for h in hours],
            color=BLUE, linewidth=2.5, marker="s", markersize=5, linestyle="--", label="주말 평균", zorder=4)

    # 피크 시각 표시
    pk_h = int(weekday.idxmax())
    pk_v = weekday.max()
    ax.annotate(
        f"평일 피크\nh={pk_h}h  {pk_v:.0f} MWh",
        xy=(pk_h, pk_v),
        xytext=(pk_h + 1.5, pk_v + 8),
        fontsize=9, color=NAVY,
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2),
    )

    ax.set_xticks(hours)
    ax.set_xlabel("시각 (hour)", fontsize=12)
    ax.set_ylabel("전력거래량 평균 (MWh)", fontsize=12)
    ax.set_title("서울시 시간대별 전력 소비 패턴 — 평일 vs 주말 (2020–2024)", fontsize=14, fontweight="bold", color=NAVY)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.set_facecolor(BG)

    fig.tight_layout()
    fig.savefig(OUT / "fig_3-3_weekday_weekend.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → saved fig_3-3_weekday_weekend.png")


# ══════════════════════════════════════════════════════════════════
# SLOT 9 — 96h holdout 실측 vs 4모델
# ══════════════════════════════════════════════════════════════════
def slot9_holdout_chart() -> None:
    print("[SLOT 9] 96h holdout 차트 생성 중...")
    df = pd.read_parquet(ART / "forecast_hybrid_holdout_seoul.parquet")

    x = list(range(len(df)))
    actual = df["power_actual"].values
    naive  = df["power_pred_naive"].values
    gru    = df["power_pred_gru_only"].values
    ttm    = df["power_pred_ttm"].values
    hybrid = df["power_pred_hybrid"].values

    rmse = {
        "Naive (lag-1)":      11.1,
        "GRU-only":           15.5,
        "TSFM+GRU Hybrid":    15.8,
        "TTM zeroshot":       19.9,
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")

    ax.fill_between(x, actual, alpha=0.12, color=GRAY, zorder=1)
    ax.plot(x, actual, color="black", linewidth=2.2, label="실측 (Actual)", zorder=5)
    ax.plot(x, naive,  color=GRAY,   linewidth=1.5, linestyle=":",  label=f"Naive  RMSE={rmse['Naive (lag-1)']}", zorder=3)
    ax.plot(x, gru,    color=GREEN,  linewidth=1.8, linestyle="--", label=f"GRU-only  RMSE={rmse['GRU-only']}", zorder=3)
    ax.plot(x, hybrid, color=BLUE,   linewidth=2.0, linestyle="-",  label=f"TSFM+GRU Hybrid  RMSE={rmse['TSFM+GRU Hybrid']}", zorder=4)
    ax.plot(x, ttm,    color=ORANGE, linewidth=1.8, linestyle="-.", label=f"TTM zeroshot  RMSE={rmse['TTM zeroshot']}", zorder=3)

    ax.set_xlabel("예측 시각 (t+h, h=0…95)", fontsize=12)
    ax.set_ylabel("전력거래량 (MWh)", fontsize=12)
    ax.set_title("서울시 단기 96h holdout 비교 — 실측 vs 4모델 (2024-12-28)", fontsize=13, fontweight="bold", color=NAVY)
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.95, ncol=2)
    ax.set_facecolor(BG)

    ax.text(0.99, 0.97,
            "단기 구간: Naive 우위\n(직전값 복사가 안정적)",
            transform=ax.transAxes, fontsize=8.5,
            color="#64748b", va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    fig.tight_layout()
    fig.savefig(OUT / "fig_holdout_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → saved fig_holdout_chart.png")


# ══════════════════════════════════════════════════════════════════
# SLOT 10 — 연간 블록 RMSE 막대
# ══════════════════════════════════════════════════════════════════
def slot10_block_rmse_bar() -> None:
    """임팩트 강화: TTM vs Roll 방식 격차를 Big Feature로 강조."""
    print("[SLOT 10] 연간 블록 RMSE 막대 생성 중 (임팩트 버전)...")
    with open(ART / "model_compare_block_2024.json") as f:
        d = json.load(f)

    bm = d["block_models"]

    s24     = bm["seasonal_24h"]["rmse"]      # 107.2
    ttm     = bm["ttm_block"]["rmse"]          # 111.5
    gru     = bm["gru_block_roll"]["rmse"]     # 296.1
    hybrid  = bm["hybrid_block_roll"]["rmse"]  # 410.0

    labels = ["Seasonal\n24h\n(기준선)", "TTM 96h\n(one-shot)", "GRU\nroll", "Hybrid\nroll"]
    values = [s24, ttm, gru, hybrid]
    colors = ["#94a3b8", BLUE, ORANGE, RED]

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")

    # 배경 구역: 좌=실용권 / 우=오차 누적 구역
    ax.axvspan(-0.5, 1.5, alpha=0.07, color=GREEN, zorder=0)
    ax.axvspan(1.5, 3.5,  alpha=0.06, color=RED,   zorder=0)

    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6, zorder=3)

    # TTM 강조 테두리
    bars[1].set_edgecolor(NAVY)
    bars[1].set_linewidth(3)

    # 값 레이블 (크고 굵게)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 6,
                f"{v:.0f}\nMWh",
                ha="center", va="bottom", fontsize=13, fontweight="bold",
                color=NAVY)

    # 구역 레이블
    y_top = max(values) * 1.22
    ax.text(0.5, y_top * 0.99, "✓ 실용 가능 구간", ha="center", va="top",
            fontsize=11, color="#166534", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#dcfce7", alpha=0.85, edgecolor="#166534"))
    ax.text(2.5, y_top * 0.99, "✗ 오차 누적 — 운영 부적합", ha="center", va="top",
            fontsize=11, color="#991b1b", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fee2e2", alpha=0.85, edgecolor="#991b1b"))

    # 핵심 수치 화살표 브라켓: TTM → GRU
    gru_rmse   = values[2]
    ttm_rmse   = values[1]
    pct_gru    = (gru_rmse - ttm_rmse) / gru_rmse * 100
    pct_hybrid = (values[3] - ttm_rmse) / values[3] * 100

    brace_y = max(values) * 0.78
    ax.annotate(
        "",
        xy=(1, ttm_rmse), xytext=(2, gru_rmse),
        arrowprops=dict(arrowstyle="<->", color=NAVY, lw=2.0),
    )
    ax.text(1.5, (ttm_rmse + gru_rmse) / 2,
            f"GRU roll 대비\n−{pct_gru:.0f}%",
            ha="center", va="center", fontsize=14, fontweight="bold", color=NAVY,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=NAVY, linewidth=2))

    # TTM → Hybrid 수치
    ax.annotate(
        f"Hybrid roll 대비\n−{pct_hybrid:.0f}%",
        xy=(1, ttm_rmse), xytext=(1.65, ttm_rmse + 100),
        fontsize=10, color="#7c3aed", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.6,
                        connectionstyle="arc3,rad=0.3"),
    )

    ax.set_ylabel("RMSE (MWh)  ← 낮을수록 좋음", fontsize=12, color=NAVY)
    ax.set_title(
        "4일(96h) 앞 예측 — 연간 전체 backtest (서울 2024, 8,784h)\n"
        "TTM one-shot vs 1시간씩 굴리는 Roll 방식 비교",
        fontsize=13, fontweight="bold", color=NAVY, pad=14,
    )
    ax.set_facecolor(BG)
    ax.set_ylim(0, max(values) * 1.26)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=10)

    # 하단 설명
    fig.text(0.5, -0.03,
             "* Roll 방식: 1h씩 96번 반복 예측 → 오차 누적  |  "
             "TTM one-shot: 96h를 한 번에 직접 예측 (누적 없음)",
             ha="center", fontsize=9.5, color="#64748b", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "fig_block_rmse_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → saved fig_block_rmse_bar.png")


# ══════════════════════════════════════════════════════════════════
# SLOT 11 — Horizon RMSE 곡선 (h=1,24,48,96)
# ══════════════════════════════════════════════════════════════════
def slot11_horizon_curve() -> None:
    """Horizon curve: TTM vs Seasonal + 실제 운영 조건(GRU roll) 단일 점 추가."""
    print("[SLOT 11] Horizon curve 생성 중 (운영 조건 명확화)...")
    with open(ART / "model_compare_block_2024.json") as f:
        d = json.load(f)

    hr = d["horizon_rmse"]
    bm = d["block_models"]
    horizons = [1, 24, 48, 96]

    def get_rmse(model_key: str) -> list[float]:
        return [hr[model_key][str(h)]["rmse"] for h in horizons]

    ttm   = get_rmse("ttm_block")
    s24   = get_rmse("seasonal_24h")
    # hybrid_1step: teacher-forcing 조건 (실측창, 운영 상한)
    tf    = get_rmse("hybrid_1step")
    # 실제 운영 조건 GRU roll — 연간 평균 단일 점
    gru_real_rmse = bm["gru_block_roll"]["rmse"]  # 296.1

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("white")

    # h48–96 구간 강조 배경
    ax.axvspan(48, 100, alpha=0.08, color=BLUE, zorder=0)
    ax.text(72, 290, "TSFM\n설계 horizon\n(h48–96)", ha="center", va="top",
            fontsize=10, color=BLUE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85))

    # 주요 선
    ax.plot(horizons, ttm, color=NAVY, linewidth=3, marker="o", markersize=9,
            label="TTM 96h (one-shot, 실제 운영)", zorder=5)
    ax.plot(horizons, s24, color=GRAY, linewidth=2, marker="^", markersize=7,
            linestyle="-.", label="Seasonal 24h (기준선)", zorder=3)
    ax.plot(horizons, tf,  color="#a78bfa", linewidth=1.8, marker="s", markersize=7,
            linestyle="--", label="Hybrid (실측창 조건, 운영 불가 — 상한)", zorder=4)

    # 실제 운영 GRU roll — 수평 점선 + 단일 점 (h=96)
    ax.axhline(gru_real_rmse, color=ORANGE, linewidth=1.6, linestyle=":",
               zorder=2, label=f"GRU roll (실제 운영) — 전구간 평균 {gru_real_rmse:.0f} MWh")
    ax.scatter([96], [gru_real_rmse], color=ORANGE, s=120, zorder=6, marker="X")
    ax.text(90, gru_real_rmse + 12, f"GRU roll\n{gru_real_rmse:.0f}", ha="center",
            fontsize=10, color=ORANGE, fontweight="bold")

    # TTM 각 horizon 값 표시
    for h, v in zip(horizons, ttm):
        offset = 10 if h < 90 else -20
        ax.annotate(f"{v:.0f}", xy=(h, v), xytext=(h, v + offset),
                    fontsize=9.5, color=NAVY, ha="center",
                    arrowprops=dict(arrowstyle="-", color=NAVY, lw=0.8) if offset < 0 else None)

    # TTM h=96 vs GRU roll 브라켓
    pct = (gru_real_rmse - ttm[-1]) / gru_real_rmse * 100
    ax.annotate(
        "",
        xy=(96, ttm[-1]), xytext=(96, gru_real_rmse),
        arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.8),
    )
    ax.text(99, (ttm[-1] + gru_real_rmse) / 2,
            f"−{pct:.0f}%",
            ha="left", va="center", fontsize=13, fontweight="bold", color=NAVY)

    ax.set_xticks(horizons)
    ax.set_xticklabels([f"h={h}\n({'1h' if h==1 else f'{h}h'})" for h in horizons], fontsize=11)
    ax.set_xlabel("예측 horizon (시간)", fontsize=12)
    ax.set_ylabel("RMSE (MWh)  ← 낮을수록 좋음", fontsize=12, color=NAVY)
    ax.set_title(
        "Horizon별 RMSE — 서울 2024\n"
        "TTM은 96h 앞도 한 번에 예측 / GRU roll은 오차가 쌓임",
        fontsize=13, fontweight="bold", color=NAVY, pad=14,
    )
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax.set_facecolor(BG)
    ax.set_xlim(-3, 110)
    ax.set_ylim(0, max(gru_real_rmse, max(s24)) * 1.22)

    fig.text(0.5, -0.03,
             "* TTM: 96h를 context 없이 한 번에 직접 예측  "
             "| GRU roll: 1h 예측을 96번 반복 → 오차 누적  "
             "| Hybrid(실측창): 운영 불가 상한",
             ha="center", fontsize=9.5, color="#64748b", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "fig_horizon_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → saved fig_horizon_curve.png")


# ══════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n=== PPT 그림 생성 → {OUT} ===\n")
    slot4_temp_scatter()
    slot5_region_slope()
    slot6_weekday_weekend()
    slot9_holdout_chart()
    slot10_block_rmse_bar()
    slot11_horizon_curve()
    print(f"\n완료! 생성된 파일:")
    for p in sorted(OUT.glob("*.png")):
        size_kb = p.stat().st_size // 1024
        print(f"  {p.name}  ({size_kb} KB)")
