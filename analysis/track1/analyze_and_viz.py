"""
Track 1 — Step 2+3: 분석 및 시각화 3개
  fig_t1_national_bar.png    : 전국 17개 지역 거래량 규모 비교
  fig_t1_hourly_pattern.png  : 발전기반 vs 소비전용 시간대 패턴
  fig_t1_temp_ucurve.png     : 기온×거래량 U자 + 지역별 slope
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
T1 = Path(__file__).parent

NAVY   = "#1e3a5f"
BLUE   = "#0ea5e9"
ORANGE = "#f97316"
GREEN  = "#22c55e"
RED    = "#ef4444"
PURPLE = "#a78bfa"
BG     = "#f8fafc"

# ── 데이터 로드
kpx = pd.read_parquet(T1 / "kpx_all_regions.parquet")
kpx4 = pd.read_parquet(T1 / "kpx_4regions_with_temp.parquet")

# ════════════════════════════════════════════════════════
# 그림 1: 전국 17개 지역 연간 평균 거래량 막대 (발전기반/소비전용 색상)
# ════════════════════════════════════════════════════════
print("[1/3] 전국 거래량 막대 생성 중...")

region_avg = kpx.groupby(["region","region_type"])["power_mwh"].mean().reset_index()
region_avg = region_avg.sort_values("power_mwh", ascending=True)

colors = [ORANGE if t == "발전기반" else BLUE for t in region_avg["region_type"]]

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor(BG)

bars = ax.barh(region_avg["region"], region_avg["power_mwh"], color=colors, height=0.7)

for bar, val, region, rtype in zip(bars, region_avg["power_mwh"],
                                    region_avg["region"], region_avg["region_type"]):
    ax.text(val + 100, bar.get_y() + bar.get_height()/2,
            f"{val:,.0f} MWh", va="center", fontsize=9,
            color=NAVY, fontweight="bold")

# 서울 강조 표시
for i, (r, v) in enumerate(zip(region_avg["region"], region_avg["power_mwh"])):
    if r == "서울시":
        ax.annotate(f"서울 = {v:,.0f} MWh\n충남의 1/{int(region_avg[region_avg['region']=='충청남도']['power_mwh'].values[0]/v)}",
                    xy=(v, i), xytext=(v + 3000, i - 1),
                    fontsize=9, color=RED, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

legend_patches = [
    mpatches.Patch(color=ORANGE, label="발전기반 (발전소 보유 → 거래량 높음)"),
    mpatches.Patch(color=BLUE,   label="소비전용 (발전소 없음 → 거래량 낮음)"),
]
ax.legend(handles=legend_patches, fontsize=10, loc="lower right", framealpha=0.9)

ax.set_xlabel("시간당 평균 전력거래량 (MWh)", fontsize=12)
ax.set_title("전국 17개 지역 시간당 평균 전력거래량 비교 (2020–2024)\n"
             "거래량 규모는 발전소 보유 여부가 결정 — 소비와 무관",
             fontsize=13, fontweight="bold", color=NAVY)
ax.set_xlim(0, region_avg["power_mwh"].max() * 1.22)

ax.text(0.02, 0.02,
        "출처: 한국전력거래소(KPX) 지역별 시간대별 전력거래량 2020–2024",
        transform=ax.transAxes, fontsize=8, color="gray")

fig.tight_layout()
p1 = OUT_FIG / "fig_t1_national_bar.png"
fig.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {p1}")

# ════════════════════════════════════════════════════════
# 그림 2: 발전기반 vs 소비전용 시간대별 패턴 비교
# ════════════════════════════════════════════════════════
print("[2/3] 시간대 패턴 생성 중...")

# 대표 지역 선택: 충청남도(발전기반 최대), 서울시(소비전용 대표), 부산시(중간)
REP_REGIONS = {
    "충청남도": (ORANGE, "발전기반 최대"),
    "경상북도": ("#fb923c", "발전기반(원자력)"),
    "부산시":   (PURPLE, "복합(소비+발전)"),
    "서울시":   (BLUE,   "소비전용"),
    "대전시":   ("#94a3b8","소비전용(최소)"),
}

hourly = kpx[kpx["region"].isin(REP_REGIONS)].groupby(
    ["region","hour_of_day"])["power_mwh"].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("white")

# 좌: 원본 값 비교
ax = axes[0]
ax.set_facecolor(BG)
for region, (color, label) in REP_REGIONS.items():
    sub = hourly[hourly["region"] == region].sort_values("hour_of_day")
    ax.plot(sub["hour_of_day"], sub["power_mwh"],
            color=color, linewidth=2.2, label=f"{region} ({label})", marker="o", markersize=3)

ax.set_xlabel("시간대", fontsize=11)
ax.set_ylabel("평균 전력거래량 (MWh)", fontsize=11)
ax.set_title("시간대별 전력거래량 패턴\n(대표 5개 지역)", fontsize=12, fontweight="bold", color=NAVY)
ax.set_xticks(range(0, 24, 3))
ax.legend(fontsize=9, framealpha=0.9)
ax.set_xlim(0, 23)

# 우: 정규화 (각 지역 최댓값 대비 %) → 패턴 형태 비교
ax2 = axes[1]
ax2.set_facecolor(BG)
for region, (color, label) in REP_REGIONS.items():
    sub = hourly[hourly["region"] == region].sort_values("hour_of_day").copy()
    sub["norm"] = sub["power_mwh"] / sub["power_mwh"].max() * 100
    ax2.plot(sub["hour_of_day"], sub["norm"],
             color=color, linewidth=2.2, label=f"{region}", marker="o", markersize=3)

ax2.axvspan(9, 18, alpha=0.07, color=BLUE)
ax2.text(13.5, 98, "업무시간\n09–18h", fontsize=9, color=BLUE,
         ha="center", va="top", fontweight="bold")
ax2.set_xlabel("시간대", fontsize=11)
ax2.set_ylabel("정규화 거래량 (최댓값 대비 %)", fontsize=11)
ax2.set_title("정규화 패턴 비교\n(규모 차이 제거 → 형태만 비교)", fontsize=12, fontweight="bold", color=NAVY)
ax2.set_xticks(range(0, 24, 3))
ax2.legend(fontsize=9, framealpha=0.9)
ax2.set_xlim(0, 23)
ax2.set_ylim(50, 105)

ax2.text(0.02, 0.04,
         "발전기반: 야간에도 유지 (기저발전 특성)\n소비전용: 업무시간 집중, 야간 급감",
         transform=ax2.transAxes, fontsize=9, color=NAVY,
         bbox=dict(boxstyle="round", fc="#dbeafe", ec=BLUE, alpha=0.8))

fig.suptitle("지역 유형별 시간대 패턴 — 발전기반 vs 소비전용",
             fontsize=13, fontweight="bold", color=NAVY, y=1.01)
fig.tight_layout()
p2 = OUT_FIG / "fig_t1_hourly_pattern.png"
fig.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {p2}")

# ════════════════════════════════════════════════════════
# 그림 3: 기온×거래량 U자 곡선 + 지역별 slope 비교
# ════════════════════════════════════════════════════════
print("[3/3] U자 곡선 + slope 생성 중...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("white")

REGION_COLORS = {"서울시": BLUE, "부산시": ORANGE, "대전시": GREEN, "강원도": PURPLE}

# 좌: U자 산점도 (서울 대표)
ax = axes[0]
ax.set_facecolor(BG)

seoul = kpx4[kpx4["region"]=="서울시"].dropna(subset=["temp","power_mwh"])
# 시간별 데이터 → 온도 구간 평균
temp_bin = pd.cut(seoul["temp"], bins=range(-20, 40, 2))
bin_avg = seoul.groupby(temp_bin, observed=True)["power_mwh"].mean()
bin_centers = [iv.mid for iv in bin_avg.index]

ax.scatter(seoul["temp"].sample(3000, random_state=42),
           seoul["power_mwh"].sample(3000, random_state=42),
           alpha=0.15, s=8, color=BLUE, zorder=2)
ax.plot(bin_centers, bin_avg.values, color=NAVY, linewidth=2.5, zorder=5,
        label="구간 평균")

# 2차 적합
valid_mask = ~np.isnan(bin_centers) & ~np.isnan(bin_avg.values)
bc = np.array(bin_centers)[valid_mask]
ba = np.array(bin_avg.values)[valid_mask]
coeffs = np.polyfit(bc, ba, 2)
t_fit = np.linspace(bc.min(), bc.max(), 200)
u_fit = np.polyval(coeffs, t_fit)
t_min = -coeffs[1] / (2 * coeffs[0])

ax.plot(t_fit, u_fit, color=RED, linewidth=2, linestyle="--", label=f"U형 적합선 (최솟값≈{t_min:.0f}°C)")
ax.axvline(t_min, color=ORANGE, linestyle=":", linewidth=1.5)
ax.axvspan(-25, 10, alpha=0.06, color="#6baed6")
ax.axvspan(26, 40, alpha=0.06, color="#e6550d")
ax.text(-15, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 700,
        "난방 구간\n≤10°C", fontsize=9, color="#2563eb", ha="center")
ax.text(33, 700, "냉방 구간\n>26°C", fontsize=9, color="#dc2626", ha="center")

ax.set_xlabel("기온 (°C)", fontsize=11)
ax.set_ylabel("전력거래량 (MWh)", fontsize=11)
ax.set_title(f"서울시: 기온×전력거래량 U자 관계\n최솟값 ≈ {t_min:.0f}°C (2020–2024)",
             fontsize=12, fontweight="bold", color=NAVY)
ax.legend(fontsize=9)

# 우: 4개 지역 slope 비교
ax2 = axes[1]
ax2.set_facecolor(BG)

slopes = {}
for region in ["서울시","부산시","대전시","강원도"]:
    sub = kpx4[kpx4["region"]==region].dropna(subset=["temp","power_mwh"])
    if len(sub) > 100:
        c = np.polyfit(sub["temp"], sub["power_mwh"], 1)
        slopes[region] = round(c[0], 2)

regions_s = list(slopes.keys())
vals_s = [slopes[r] for r in regions_s]
colors_s = [REGION_COLORS[r] for r in regions_s]
bars2 = ax2.bar(regions_s, vals_s, color=colors_s, width=0.6)

for bar, val in zip(bars2, vals_s):
    ypos = val + 0.5 if val >= 0 else val - 2
    ax2.text(bar.get_x() + bar.get_width()/2, ypos,
             f"{val:+.1f}\nMWh/°C", ha="center", fontsize=10,
             color=NAVY, fontweight="bold")

ax2.axhline(0, color="gray", linewidth=1, linestyle="--")
ax2.set_ylabel("기온 민감도 slope (MWh/°C)", fontsize=11)
ax2.set_title("지역별 기온 민감도 비교\n(기온 1°C 상승 시 거래량 변화)",
              fontsize=12, fontweight="bold", color=NAVY)

# 해석 박스
ax2.text(0.5, 0.12,
         "부산: 고리·신고리 원자력 인근\n→ 기온·계절이 발전·거래 패턴까지 영향\n\n"
         "서울: 발전소 없는 소비 전용\n→ 기온 변화에도 거래량 안정",
         transform=ax2.transAxes, fontsize=9, color=NAVY, ha="center",
         bbox=dict(boxstyle="round", fc="#f0f9ff", ec=BLUE, alpha=0.9))

fig.suptitle("Track 1 — 기온×전력 U자 관계 & 지역별 민감도 차이",
             fontsize=13, fontweight="bold", color=NAVY, y=1.01)
fig.tight_layout()
p3 = OUT_FIG / "fig_t1_temp_ucurve.png"
fig.savefig(p3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {p3}")

# ── 분석 수치 저장
results = {
    "region_avg_power": region_avg.set_index("region")["power_mwh"].to_dict(),
    "region_type": region_avg.set_index("region")["region_type"].to_dict(),
    "slopes": slopes,
    "seoul_u_min_temp": round(float(t_min), 1),
}
import json
with open(T1 / "track1_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nTrack 1 분석 완료!")
print(f"지역별 slope: {slopes}")
print(f"서울 U자 최솟값: {t_min:.1f}°C")
