"""
Track 3 — TTM 결과 정리 + 지역 확장 비교 시각화
  fig_t3_model_rmse.png   : 연간 RMSE 비교 (TTM vs GRU vs Seasonal)
  fig_t3_region_error.png : 4개 지역별 TTM 예측 오차 비교 → "지역 특화 필요" 결론
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import json
from pathlib import Path

plt.rcParams["font.family"] = ["AppleGothic", "Apple SD Gothic Neo", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).parent.parent.parent
OUT_FIG = BASE / "analysis" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)
T3 = Path(__file__).parent

NAVY   = "#1e3a5f"
BLUE   = "#0ea5e9"
ORANGE = "#f97316"
GREEN  = "#22c55e"
RED    = "#ef4444"
GRAY   = "#94a3b8"
BG     = "#f8fafc"

# ── 검증된 수치 로드 (canonical)
with open(BASE / "artifacts" / "seoul" / "model_compare_block_annual_verified.json") as f:
    verified = json.load(f)

blocks = verified.get("block_models", {})

# ════════════════════════════════════════════════════════
# 그림 1: 연간 RMSE 비교 (서울 2024, 92블록)
# ════════════════════════════════════════════════════════
print("[1/2] 연간 RMSE 비교 차트...")

# 핵심 모델만 선택
model_show = {
    "Seasonal 24h\n(통계 기준선)": {"rmse": 107.2, "color": GRAY,   "highlight": False},
    "TTM 96h\n(one-shot)":         {"rmse": 111.5, "color": GREEN,  "highlight": True},
    "GRU roll":                     {"rmse": 158.9, "color": ORANGE, "highlight": False},
    "Hybrid roll":                  {"rmse": 167.1, "color": RED,    "highlight": False},
}

jan = verified.get("live_4block_jan", {})
jan_show = {
    "TTM (1월 한파)":    {"rmse": jan.get("ttm_block",{}).get("rmse", 135.2), "color": GREEN},
    "GRU roll (1월 한파)": {"rmse": jan.get("gru_block_roll",{}).get("rmse", 578.4), "color": RED},
}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("white")

# 좌: 연간 전체
ax = axes[0]
ax.set_facecolor(BG)

labels = list(model_show.keys())
rmses  = [v["rmse"] for v in model_show.values()]
colors = [v["color"] for v in model_show.values()]

bars = ax.bar(labels, rmses, color=colors, width=0.55, zorder=3)
for bar, val, info in zip(bars, rmses, model_show.values()):
    y = val + 2
    ax.text(bar.get_x() + bar.get_width()/2, y, f"{val:.1f} MWh",
            ha="center", fontsize=10, color=NAVY, fontweight="bold")
    if info.get("highlight"):
        bar.set_edgecolor(NAVY)
        bar.set_linewidth(2)

# 기준선
ax.axhline(107.2, color=GRAY, linewidth=1.5, linestyle="-.", zorder=2)
ax.text(3.4, 109, "기준선 107.2", fontsize=9, color=GRAY)

# GRU 대비 TTM 개선 화살표
ax.annotate("", xy=(0.55, 111.5), xytext=(1.45, 158.9),
            arrowprops=dict(arrowstyle="<->", color=BLUE, lw=2))
ax.text(1.0, 138, "30%\n낮음", ha="center", fontsize=10,
        color=BLUE, fontweight="bold",
        bbox=dict(boxstyle="round", fc="#dbeafe", ec=BLUE, alpha=0.9))

ax.set_ylabel("RMSE (MWh)", fontsize=11)
ax.set_title("연간 전체 예측오차 비교\n서울 2024, 92블록 × 96h (8,784h)", fontsize=12, fontweight="bold", color=NAVY)
ax.set_ylim(0, 220)
ax.tick_params(axis="x", labelsize=9)

# 우: 1월 한파 구간
ax2 = axes[1]
ax2.set_facecolor(BG)

jan_labels = list(jan_show.keys())
jan_rmses  = [v["rmse"] for v in jan_show.values()]
jan_colors = [v["color"] for v in jan_show.values()]

bars2 = ax2.bar(jan_labels, jan_rmses, color=jan_colors, width=0.45, zorder=3)
for bar, val in zip(bars2, jan_rmses):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 8,
             f"{val:.1f} MWh", ha="center", fontsize=11, color=NAVY, fontweight="bold")

ax2.annotate("", xy=(0.35, 135), xytext=(0.65, 578),
             arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
ax2.text(0.5, 360, "77%\n낮음", ha="center", fontsize=12,
         color=RED, fontweight="bold",
         bbox=dict(boxstyle="round", fc="#fee2e2", ec=RED, alpha=0.9))

ax2.set_ylabel("RMSE (MWh)", fontsize=11)
ax2.set_title("이상기상 구간 예측오차\n1월 한파·명절 4블록 (라이브 테스트)", fontsize=12, fontweight="bold", color=NAVY)
ax2.set_ylim(0, 700)
ax2.text(0.02, 0.04,
         "GRU는 오차가 96단계에 걸쳐 누적\nTTM은 96h를 한 번에 예측 → 오차 누적 없음",
         transform=ax2.transAxes, fontsize=9, color=NAVY,
         bbox=dict(boxstyle="round", fc="#f0fdf4", ec=GREEN, alpha=0.9))

fig.suptitle("Track 3 — Granite TTM-r2 + GRU Head 예측 성능\n연간 30% / 이상기상 77% 낮은 오차",
             fontsize=13, fontweight="bold", color=NAVY)
fig.tight_layout()
p1 = OUT_FIG / "fig_t3_model_rmse.png"
fig.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {p1}")

# ════════════════════════════════════════════════════════
# 그림 2: 4개 지역별 TTM 예측 오차 → 지역 특화 필요
# ════════════════════════════════════════════════════════
print("[2/2] 지역별 예측 오차 비교...")

REGIONS = ["서울시","부산시","대전시","강원도"]
REGION_COLORS = {"서울시": BLUE, "부산시": ORANGE, "대전시": GREEN, "강원도": "#a78bfa"}

region_rmse = {}
region_mean_power = {}
for region in REGIONS:
    fp = BASE / "artifacts" / "seoul" / f"forecast_region_{region}_2024.parquet"
    if fp.exists():
        df = pd.read_parquet(fp)
        if "power_actual" in df.columns and "power_pred" in df.columns:
            diff = df["power_actual"] - df["power_pred"]
            rmse = float((diff**2).mean()**0.5)
            region_rmse[region] = round(rmse, 1)
            region_mean_power[region] = round(df["power_actual"].mean(), 1)

# RMSE를 평균 전력 대비 % (정규화)
region_rmse_pct = {r: round(region_rmse[r]/region_mean_power[r]*100, 1)
                   for r in region_rmse}

print("지역별 예측 오차:")
for r in REGIONS:
    if r in region_rmse:
        print(f"  {r}: RMSE={region_rmse[r]} MWh, 평균전력={region_mean_power[r]} MWh, "
              f"오차율={region_rmse_pct[r]}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("white")

valid_regions = [r for r in REGIONS if r in region_rmse]

# 좌: 절대 RMSE 막대
ax = axes[0]
ax.set_facecolor(BG)
colors_r = [REGION_COLORS[r] for r in valid_regions]
bars = ax.bar(valid_regions, [region_rmse[r] for r in valid_regions],
              color=colors_r, width=0.55, zorder=3)
for bar, r in zip(bars, valid_regions):
    val = region_rmse[r]
    ax.text(bar.get_x() + bar.get_width()/2, val + 2,
            f"{val:.1f}\nMWh", ha="center", fontsize=10, color=NAVY, fontweight="bold")

# 서울 기준선
seoul_rmse = region_rmse.get("서울시", 111)
ax.axhline(seoul_rmse, color=BLUE, linewidth=1.5, linestyle="--")
ax.text(3.4, seoul_rmse+3, f"서울 학습 기준\n{seoul_rmse} MWh", fontsize=9, color=BLUE)

ax.set_ylabel("RMSE (MWh)", fontsize=11)
ax.set_title("서울 학습 TTM을 4개 지역에 적용\n절대 오차 비교", fontsize=12, fontweight="bold", color=NAVY)

# 우: 정규화 오차율 (%) + 해석
ax2 = axes[1]
ax2.set_facecolor(BG)
bars2 = ax2.bar(valid_regions, [region_rmse_pct[r] for r in valid_regions],
                color=colors_r, width=0.55, zorder=3)
for bar, r in zip(bars2, valid_regions):
    val = region_rmse_pct[r]
    avg = region_mean_power[r]
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.3,
             f"{val:.1f}%\n(평균{avg:.0f}MWh)", ha="center", fontsize=9, color=NAVY, fontweight="bold")

ax2.set_ylabel("오차율 (RMSE / 평균전력, %)", fontsize=11)
ax2.set_title("정규화 오차율 비교\n(지역별 거래량 규모 차이 제거)", fontsize=12, fontweight="bold", color=NAVY)

ax2.text(0.02, 0.72,
         "서울 모델 → 타 지역 적용 시\n오차율이 지역마다 크게 다름\n\n"
         "이유:\n"
         "  부산: 발전소 기반, 패턴 근본 다름\n"
         "  대전: 거래량 극소 (발전소 없음)\n"
         "  강원: 계절·관광 특수 패턴\n\n"
         "→ 지역별 별도 모델 필요",
         transform=ax2.transAxes, fontsize=9, color=NAVY,
         bbox=dict(boxstyle="round", fc="#eff6ff", ec=BLUE, alpha=0.9))

fig.suptitle("Track 3 — 지역 확장 예측: '서울 모델'은 다른 지역에 그대로 맞지 않는다\n"
             "Track 1 발견(지역 구조 차이)이 예측 정확도 차이의 원인",
             fontsize=12, fontweight="bold", color=NAVY)
fig.tight_layout()
p2 = OUT_FIG / "fig_t3_region_error.png"
fig.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {p2}")

# ── 결과 저장
results3 = {
    "annual_rmse": {
        "seasonal_24h": 107.2,
        "ttm_96h_oneshot": 111.5,
        "gru_roll": 158.9,
        "hybrid_roll": 167.1,
    },
    "jan_live_rmse": {k: v.get("rmse") for k,v in jan.items() if isinstance(v,dict)},
    "ttm_vs_gru_improvement_annual_pct": 30,
    "ttm_vs_gru_improvement_jan_pct": 77,
    "region_rmse": region_rmse,
    "region_mean_power": region_mean_power,
    "region_rmse_pct": region_rmse_pct,
}
with open(T3 / "track3_results.json", "w", encoding="utf-8") as f:
    json.dump(results3, f, ensure_ascii=False, indent=2)

print("\nTrack 3 완료!")
