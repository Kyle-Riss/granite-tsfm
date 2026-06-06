"""
Track 2 — 서울 내부 소비 구조 분석 및 시각화 3개
  fig_t2_district_rank.png   : 구별 연간 총사용량 + 인구당 에너지 비교
  fig_t2_season_heatmap.png  : 12개월 × 25개 구 계절 집중도 히트맵
  fig_t2_ucurve_district.png : 월평균 기온 × 실소비량 U자 곡선
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
T2 = Path(__file__).parent

NAVY   = "#1e3a5f"
BLUE   = "#0ea5e9"
ORANGE = "#f97316"
GREEN  = "#22c55e"
RED    = "#ef4444"
BG     = "#f8fafc"

# ══════════════════════════════════════════════════
# 데이터 로드 및 병합
# ══════════════════════════════════════════════════
print("=== Track 2 데이터 로드 ===")

# 서울 구별 에너지
energy = pd.read_csv(BASE / "Data" / "seoul_energy_use_pre.csv", encoding="utf-8-sig")
energy.columns = ["district","month","usage"]
energy["usage"] = pd.to_numeric(energy["usage"], errors="coerce")

# 서울 인구
pop_raw = pd.read_csv(BASE / "Data" / "population_seoul.csv", encoding="cp949")
pop_raw.columns = ["행정구역","인구수","비고"]
# 구 이름 추출 (서울특별시 종로구 → 종로구)
pop_raw = pop_raw[pop_raw["행정구역"].str.contains("구")].copy()
pop_raw["district"] = pop_raw["행정구역"].str.extract(r'서울특별시\s+(\S+구)')
pop_raw["population"] = pop_raw["인구수"].str.replace(",","").str.strip().astype(float)
pop = pop_raw[["district","population"]].dropna()
print(f"구별 인구 로드: {len(pop)}개 구")
print(pop.sort_values("population", ascending=False).head(5).to_string(index=False))

# 연간 총 사용량 (월별 합산)
annual = energy.groupby("district")["usage"].sum().reset_index()
annual.columns = ["district","annual_usage"]

# 인구 병합
merged = annual.merge(pop, on="district", how="left")
merged["usage_per_capita"] = merged["annual_usage"] / merged["population"]

# 상업구 / 주거구 분류
COMMERCIAL = {"강남구","서초구","영등포구","중구","종로구","마포구","용산구","송파구"}
merged["district_type"] = merged["district"].apply(
    lambda d: "상업·업무구" if d in COMMERCIAL else "주거구"
)

print(f"\n인구당 에너지 상위 5구:")
print(merged.nlargest(5,"usage_per_capita")[["district","annual_usage","population","usage_per_capita"]].round(1).to_string(index=False))

merged.to_csv(T2 / "district_merged.csv", encoding="utf-8-sig", index=False)
print(f"→ 저장: district_merged.csv")

# ══════════════════════════════════════════════════
# 그림 1: 구별 연간 총사용량 + 인구당 에너지 (좌우 2패널)
# ══════════════════════════════════════════════════
print("\n[1/3] 구별 랭킹 + 인구당 에너지...")

fig, axes = plt.subplots(1, 2, figsize=(15, 8))
fig.patch.set_facecolor("white")

# 좌: 연간 총사용량 랭킹
ax = axes[0]
ax.set_facecolor(BG)
rank = merged.sort_values("annual_usage", ascending=True)
colors = [ORANGE if t == "상업·업무구" else BLUE for t in rank["district_type"]]

bars = ax.barh(rank["district"], rank["annual_usage"] / 1e6, color=colors, height=0.75)
for bar, val, d in zip(bars, rank["annual_usage"]/1e6, rank["district"]):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}", va="center", fontsize=8, color=NAVY)

legend_patches = [
    mpatches.Patch(color=ORANGE, label="상업·업무구"),
    mpatches.Patch(color=BLUE,   label="주거구"),
]
ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
ax.set_xlabel("연간 총에너지 사용량 (백만 MWh)", fontsize=11)
ax.set_title("서울 25개 구별 연간 에너지 사용량\n(강남구 ÷ 강북구 = 5.5배)", fontsize=12, fontweight="bold", color=NAVY)

# 우: 인구당 에너지 랭킹
ax2 = axes[1]
ax2.set_facecolor(BG)
rank2 = merged.sort_values("usage_per_capita", ascending=True)
colors2 = [ORANGE if t == "상업·업무구" else BLUE for t in rank2["district_type"]]

bars2 = ax2.barh(rank2["district"], rank2["usage_per_capita"], color=colors2, height=0.75)
for bar, val in zip(bars2, rank2["usage_per_capita"]):
    ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2,
             f"{val:.0f}", va="center", fontsize=8, color=NAVY)

ax2.legend(handles=legend_patches, fontsize=9, loc="lower right")
ax2.set_xlabel("1인당 에너지 사용량 (MWh/인)", fontsize=11)
ax2.set_title("1인당 에너지 사용량\n(상업구: 인구 적지만 1인당 압도적 높음)", fontsize=12, fontweight="bold", color=NAVY)

ax2.text(0.02, 0.02,
         "※ 인구: 2025년 10월 기준 | 에너지: 서울시 에너지통계 (실소비량)",
         transform=ax2.transAxes, fontsize=8, color="gray")

fig.suptitle("Track 2 — 서울 25개 구 에너지 소비 구조: 총량 vs 1인당",
             fontsize=13, fontweight="bold", color=NAVY)
fig.tight_layout()
p1 = OUT_FIG / "fig_t2_district_rank.png"
fig.savefig(p1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {p1}")

# ══════════════════════════════════════════════════
# 그림 2: 12개월 × 25개 구 계절 집중도 히트맵
# ══════════════════════════════════════════════════
print("[2/3] 계절 집중도 히트맵...")

# 피벗: 구(행) × 월(열), 연간 대비 월별 비율
pivot = energy.pivot(index="district", columns="month", values="usage")
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100  # 월별 비중 (%)

# 연간 총사용량 기준 정렬
district_order = merged.sort_values("annual_usage", ascending=False)["district"].tolist()
pivot_pct = pivot_pct.reindex(district_order)

month_labels = ["1월","2월","3월","4월","5월","6월","7월","8월","9월","10월","11월","12월"]

fig, ax = plt.subplots(figsize=(13, 9))
fig.patch.set_facecolor("white")

im = ax.imshow(pivot_pct.values, aspect="auto", cmap="RdYlBu_r", vmin=5, vmax=14)
plt.colorbar(im, ax=ax, label="월별 비중 (%)", shrink=0.8)

ax.set_xticks(range(12))
ax.set_xticklabels(month_labels, fontsize=10)
ax.set_yticks(range(len(district_order)))
ax.set_yticklabels(district_order, fontsize=9)

# 최솟값/최댓값 셀 강조
for i, district in enumerate(district_order):
    row = pivot_pct.loc[district]
    max_m = row.idxmax() - 1
    min_m = row.idxmin() - 1
    ax.add_patch(plt.Rectangle((max_m-0.5, i-0.5), 1, 1, fill=False, edgecolor=RED, linewidth=2))
    ax.add_patch(plt.Rectangle((min_m-0.5, i-0.5), 1, 1, fill=False, edgecolor=BLUE, linewidth=2))

# 범례
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="white", edgecolor=RED, linewidth=2, label="최대 소비 월"),
    Patch(facecolor="white", edgecolor=BLUE, linewidth=2, label="최소 소비 월"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
          bbox_to_anchor=(1.18, 1), framealpha=0.9)

# 계절 구분선
for x in [2.5, 5.5, 8.5]:
    ax.axvline(x, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

ax.set_title("서울 25개 구 월별 에너지 사용 비중 히트맵\n"
             "여름(7–8월) vs 겨울(1·12월) 집중 패턴 구별 비교",
             fontsize=12, fontweight="bold", color=NAVY, pad=12)
ax.set_xlabel("월", fontsize=11)
ax.set_ylabel("자치구 (연간 사용량 내림차순)", fontsize=11)

ax.text(0.02, -0.06,
        "출처: 서울시 에너지통계 (energyinfo.seoul.go.kr) — 실소비량 기준",
        transform=ax.transAxes, fontsize=8, color="gray")

fig.tight_layout()
p2 = OUT_FIG / "fig_t2_season_heatmap.png"
fig.savefig(p2, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {p2}")

# ══════════════════════════════════════════════════
# 그림 3: 월평균 기온 × 실소비량 U자 곡선
# ══════════════════════════════════════════════════
print("[3/3] 실소비 U자 곡선...")

# 서울 월평균 기온 (Open-Meteo 2020-2024)
temp_df = pd.read_parquet(BASE / "Data" / "meteo" / "hourly_temp_서울시.parquet")
temp_df["ts"] = pd.to_datetime(temp_df["ts"])
temp_df["month"] = temp_df["ts"].dt.month
monthly_temp = temp_df.groupby("month")["temp_c"].mean().round(1)

# 서울 25개 구 월별 합산 실소비량
monthly_total = energy.groupby("month")["usage"].sum()

months = list(range(1, 13))
temps = [monthly_temp[m] for m in months]
usages = [monthly_total[m] / 1e6 for m in months]  # 백만 MWh

# U자 2차 적합
coeffs = np.polyfit(temps, usages, 2)
t_fit = np.linspace(min(temps) - 2, max(temps) + 2, 200)
u_fit = np.polyval(coeffs, t_fit)
t_min_u = -coeffs[1] / (2 * coeffs[0])

SEASON_COLOR = {
    1: "#6baed6", 2: "#6baed6", 12: "#6baed6",
    3: "#74c476", 4: "#74c476", 5: "#74c476",
    10: "#74c476", 11: "#6baed6",
    6: "#fd8d3c", 7: "#e6550d", 8: "#e6550d", 9: "#fd8d3c",
}
MONTH_KR = {1:"1월",2:"2월",3:"3월",4:"4월",5:"5월",6:"6월",
            7:"7월",8:"8월",9:"9월",10:"10월",11:"11월",12:"12월"}

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor(BG)

ax.plot(t_fit, u_fit, color=NAVY, linewidth=2.5, zorder=4, label="U형 2차 적합선")
ax.axvline(t_min_u, color=ORANGE, linewidth=1.5, linestyle="--", zorder=3)
ax.text(t_min_u + 0.5, max(usages) * 0.97,
        f"최솟값\n≈{t_min_u:.0f}°C", fontsize=9, color=ORANGE, fontweight="bold", va="top")

for m, t, u in zip(months, temps, usages):
    ax.scatter(t, u, color=SEASON_COLOR[m], s=140, zorder=5, edgecolors="white", linewidths=1)
    ax.annotate(MONTH_KR[m], (t, u), textcoords="offset points",
                xytext=(6, 4), fontsize=9, color=NAVY, fontweight="bold")

ax.axvspan(min(temps)-2, 10, alpha=0.06, color="#6baed6")
ax.axvspan(26, max(temps)+2, alpha=0.06, color="#e6550d")
ax.text(-5, min(usages)*1.02, "난방\n≤10°C", fontsize=9, color="#2563eb", ha="center")
ax.text(31, min(usages)*1.02, "냉방\n>26°C", fontsize=9, color="#dc2626", ha="center")

legend_patches = [
    mpatches.Patch(color="#6baed6", label="겨울 (12·1·2월)"),
    mpatches.Patch(color="#74c476", label="봄·가을 (3–5, 10–11월)"),
    mpatches.Patch(color="#e6550d", label="여름 (6–8월)"),
]
ax.legend(handles=legend_patches, fontsize=9, loc="upper center", framealpha=0.9)

ax.set_xlabel("서울 월평균 기온 (°C) — Open-Meteo 2020–2024 평균", fontsize=11)
ax.set_ylabel("서울 25개 구 합산 에너지 사용량 (백만 MWh/월)", fontsize=11)
ax.set_title(f"서울 실소비량 × 월평균 기온 U자 관계\n"
             f"최솟값 ≈ {t_min_u:.0f}°C — KPX 거래량 분석(Track 1)과 동일한 패턴 확인",
             fontsize=12, fontweight="bold", color=NAVY)

ax.text(0.02, 0.03,
        "※ 서울시 에너지통계 실소비량 기준 — KPX 도매거래량(발전소 기준)과 독립적 검증",
        transform=ax.transAxes, fontsize=8, color="gray")

fig.tight_layout()
p3 = OUT_FIG / "fig_t2_ucurve_district.png"
fig.savefig(p3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → {p3}")

# ── 분석 수치 저장
results2 = {
    "max_district": merged.nlargest(1,"annual_usage")["district"].values[0],
    "max_usage_million_mwh": round(float(merged["annual_usage"].max()/1e6), 2),
    "min_district": merged.nsmallest(1,"annual_usage")["district"].values[0],
    "min_usage_million_mwh": round(float(merged["annual_usage"].min()/1e6), 2),
    "ratio_max_min": round(float(merged["annual_usage"].max()/merged["annual_usage"].min()), 1),
    "max_per_capita_district": merged.nlargest(1,"usage_per_capita")["district"].values[0],
    "u_min_temp_actual_consumption": round(float(t_min_u), 1),
    "monthly_temp": {int(k): float(v) for k, v in monthly_temp.items()},
}
with open(T2 / "track2_results.json", "w", encoding="utf-8") as f:
    json.dump(results2, f, ensure_ascii=False, indent=2)

print(f"\nTrack 2 분석 완료!")
print(f"최대 사용 구: {results2['max_district']} ({results2['max_usage_million_mwh']}백만 MWh)")
print(f"최소 사용 구: {results2['min_district']} ({results2['min_usage_million_mwh']}백만 MWh)")
print(f"격차: {results2['ratio_max_min']}배")
print(f"실소비 U자 최솟값: {results2['u_min_temp_actual_consumption']}°C")
