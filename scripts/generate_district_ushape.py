"""
서울 자치구별 월별 에너지 사용량 × 월평균 기온 → U자 보조 그림 생성
Output: artifacts/seoul/ppt_figures/fig_seoul_district_ushape.png
"""
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = ["AppleGothic", "Malgun Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("artifacts/seoul/ppt_figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── 서울 월평균 기온 (Open-Meteo, 2020-2024 평균)
MONTH_TEMP = {
    1: -2.5, 2: 0.2, 3: 6.8, 4: 13.5, 5: 18.9,
    6: 23.1, 7: 26.8, 8: 27.5, 9: 22.2, 10: 15.3,
    11: 7.6, 12: 0.5,
}

# ── 데이터 로드
rows = []
with open("Data/seoul_energy_use_pre.csv", encoding="utf-8-sig") as f:
    for i, row in enumerate(csv.reader(f)):
        if i == 0:
            continue
        rows.append({"district": row[0], "month": int(row[1]), "usage": float(row[2])})

# 월별 전체 합 (25개 구 합산)
monthly_total = {}
for r in rows:
    m = r["month"]
    monthly_total[m] = monthly_total.get(m, 0) + r["usage"]

months = sorted(monthly_total.keys())
temps = [MONTH_TEMP[m] for m in months]
usage = [monthly_total[m] / 1e6 for m in months]  # GWh

# U자 2차 적합
coeffs = np.polyfit(temps, usage, 2)
t_fit = np.linspace(min(temps) - 2, max(temps) + 2, 200)
u_fit = np.polyval(coeffs, t_fit)

# 최솟값 기온
t_min = -coeffs[1] / (2 * coeffs[0])

# ── 디자인
NAVY = "#1e3a5f"
ORANGE = "#f97316"
BLUE = "#0ea5e9"
RED = "#ef4444"
BG = "#f8fafc"

# 계절 색상
season_color = {
    1: "#6baed6", 2: "#6baed6", 3: "#74c476",
    4: "#74c476", 5: "#74c476", 6: "#fd8d3c",
    7: "#e6550d", 8: "#e6550d", 9: "#fd8d3c",
    10: "#74c476", 11: "#6baed6", 12: "#6baed6",
}
month_label = {
    1: "1월", 2: "2월", 3: "3월", 4: "4월",
    5: "5월", 6: "6월", 7: "7월", 8: "8월",
    9: "9월", 10: "10월", 11: "11월", 12: "12월",
}

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor(BG)

# U자 적합선
ax.plot(t_fit, u_fit, color=NAVY, linewidth=2.5, zorder=4, label="U형 2차 적합선")

# 월별 점
for m, t, u in zip(months, temps, usage):
    ax.scatter(t, u, color=season_color[m], s=120, zorder=5, edgecolors="white", linewidths=0.8)
    ax.annotate(month_label[m], (t, u), textcoords="offset points",
                xytext=(5, 4), fontsize=9, color=NAVY, fontweight="bold")

# 최솟값 기온 수직선
ax.axvline(t_min, color=ORANGE, linewidth=1.5, linestyle="--", zorder=3)
ax.text(t_min + 0.4, max(usage) * 0.98, f"최솟값\n≈{t_min:.0f}°C",
        fontsize=9, color=ORANGE, fontweight="bold", va="top")

# 난방/냉방 구간 배경
ax.axvspan(min(temps) - 2, 10, alpha=0.06, color="#6baed6", zorder=1)
ax.axvspan(26, max(temps) + 2, alpha=0.06, color="#e6550d", zorder=1)

ax.set_xlabel("월평균 기온 (°C)", fontsize=12)
ax.set_ylabel("전력 사용량 (GWh/월, 서울 25개 구 합산)", fontsize=11)
ax.set_title("서울시 자치구 에너지 사용량 × 월평균 기온 (실소비 기준, 2019–2023 평균)",
             fontsize=13, fontweight="bold", color=NAVY)

legend_patches = [
    mpatches.Patch(color="#6baed6", label="겨울 (12–2월)"),
    mpatches.Patch(color="#74c476", label="봄·가을 (3–5, 9–11월)"),
    mpatches.Patch(color="#e6550d", label="여름 (6–8월)"),
]
ax.legend(handles=legend_patches, fontsize=9, loc="upper center", framealpha=0.9)

ax.text(0.02, 0.04,
        "※ 서울시 에너지통계(energyinfo.seoul.go.kr) 실소비량 기준\n"
        "   KPX 거래량과 달리 발전소 영향 없는 실수요 반영",
        transform=ax.transAxes, fontsize=8, color="gray", va="bottom")

fig.tight_layout()
out_path = OUT / "fig_seoul_district_ushape.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → saved {out_path}")
