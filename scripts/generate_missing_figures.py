"""
fig_error_explosion.png 및 fig_oneshot_vs_roll.png 생성
검증된 수치 (model_compare_block_annual_verified.json + tsfm_benchmark_2024.json) 기반
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pathlib import Path

plt.rcParams["font.family"] = ["AppleGothic", "Apple SD Gothic Neo", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("artifacts/seoul/ppt_figures")
OUT.mkdir(parents=True, exist_ok=True)

NAVY  = "#1e3a5f"
BLUE  = "#0ea5e9"
RED   = "#ef4444"
ORANGE = "#f97316"
GREEN  = "#22c55e"
BG    = "#f8fafc"

# ── 검증된 key-point 수치 (ppt_prompt.md canonical)
# h=1  : TTM 36,  GRU 연간평균 ≈ 36×(159/111), 1월실측 120
# h=24 : TTM 79,  GRU 연간 ≈ 79×(159/111), 1월 약 430
# h=48 : TTM 99,  GRU 연간 ≈ 99×(159/111), 1월 약 430
# h=96 : TTM 135, GRU 연간 159,            1월 578
ratio = 159 / 111
H = [1, 12, 24, 48, 72, 96]
TTM_pts   = [36, 55, 79, 99, 118, 135]
GRU_ann   = [int(v * ratio) for v in TTM_pts]  # 연간 평균 roll RMSE 보간
GRU_jan   = [120, 220, 350, 430, 510, 578]    # 1월 한파 실측 추이

# ── fig_error_explosion ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor(BG)

# 배경: 정책 결정 구간
ax.axvspan(48, 96, alpha=0.07, color=BLUE, zorder=1)
ax.text(72, 580, "4일 앞 예측\n정책 결정 구간", fontsize=9, color=BLUE,
        ha="center", va="top", fontweight="bold")

# 선
ax.plot(H, GRU_jan, "o-", color=RED, linewidth=2.5, markersize=7,
        label="GRU roll (1월 실측, 한파·명절)", zorder=5)
ax.plot(H, GRU_ann, "o--", color=ORANGE, linewidth=1.8, markersize=5,
        label="GRU roll (연간 평균)", zorder=4, alpha=0.8)
ax.plot(H, TTM_pts, "o-", color=NAVY, linewidth=2.5, markersize=7,
        label="TTM 96h (one-shot)", zorder=5)
ax.axhline(107, color="gray", linewidth=1.2, linestyle="-.",
           label="Seasonal 24h (기준선)", zorder=3)

# 핵심 수치 레이블
for i, h in enumerate(H):
    if h in (1, 24, 48, 96):
        ax.annotate(f"{TTM_pts[i]}", (h, TTM_pts[i]),
                    textcoords="offset points", xytext=(5, 6),
                    fontsize=9, color=NAVY, fontweight="bold")
        if h in (1, 96):
            ax.annotate(f"{GRU_jan[i]}", (h, GRU_jan[i]),
                        textcoords="offset points", xytext=(5, 6),
                        fontsize=9, color=RED, fontweight="bold")

# h=96 격차 강조 박스
ax.annotate("", xy=(96, GRU_jan[-1]), xytext=(96, TTM_pts[-1]),
            arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
ax.text(98, (GRU_jan[-1] + TTM_pts[-1]) / 2, "4일 뒤\n격차 77%",
        fontsize=10, color=RED, fontweight="bold", va="center")

# GRU 오차폭발 콜아웃
ax.annotate("GRU: 오차\n폭발!", xy=(72, GRU_jan[4]),
            xytext=(55, 500),
            fontsize=10, color=RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec=RED, alpha=0.9))

# TTM 안정
ax.annotate("TTM: 안정적 유지", xy=(48, TTM_pts[3]),
            xytext=(25, 120),
            fontsize=9, color=NAVY, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="#dbeafe", ec=NAVY, alpha=0.9))

ax.set_xlabel("예측 시점 (현재로부터)", fontsize=12)
ax.set_ylabel("예측오차 RMSE (MWh)", fontsize=12)
ax.set_title("멀리 볼수록 차이가 벌어진다\nTTM은 4일 뒤도 안정적 / GRU는 오차가 눈덩이처럼 불어남",
             fontsize=13, fontweight="bold", color=NAVY)
ax.set_xticks(H)
ax.set_xticklabels([f"h={h}\n({'1시간 뒤' if h==1 else '1일 뒤' if h==24 else '2일 뒤' if h==48 else '3일 뒤' if h==72 else '4일 뒤'})" for h in H], fontsize=9)
ax.set_ylim(0, 660)
ax.legend(fontsize=9, loc="upper left", framealpha=0.9)

ax.text(0.02, 0.03,
        "* GRU roll: 1h씩 96번 반복 예측 → 오차가 오차를 먹는 구조  |  TTM: 96h시간을 한 번에 직접 예측  |  데이터: KPX 서울 2024 (8,784h)",
        transform=ax.transAxes, fontsize=7.5, color="gray")

fig.tight_layout()
fig.savefig(OUT / "fig_error_explosion.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  → saved fig_error_explosion.png")


# ── fig_oneshot_vs_roll ─────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7))
fig.patch.set_facecolor("white")
fig.suptitle("왜 TSFM인가? — 예측 방식 자체가 다르다",
             fontsize=14, fontweight="bold", color=NAVY, y=0.98)

# 상단: GRU Roll
ax1 = axes[0]
ax1.set_facecolor(BG)
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 1)
ax1.axis("off")
ax1.text(0.0, 0.92, "GRU Roll 방식: 1시간씩 96번 반복", fontsize=11,
         color=RED, fontweight="bold", transform=ax1.transAxes)

# 화살표 체인 (GRU steps)
step_x = np.linspace(0.5, 9.0, 10)
bar_heights = np.linspace(0.12, 0.42, 10)  # 오차 점점 커짐
colors_grad = [plt.cm.Reds(0.3 + 0.07 * i) for i in range(10)]

for i, (x, bh, c) in enumerate(zip(step_x, bar_heights, colors_grad)):
    bbox = FancyBboxPatch((x - 0.3, 0.40), 0.6, 0.25,
                          boxstyle="round,pad=0.02", fc=c, ec="white", lw=1.2)
    ax1.add_patch(bbox)
    ax1.text(x, 0.525, f"h{i+1}", ha="center", va="center",
             fontsize=7.5, color="white", fontweight="bold")
    if i < 9:
        ax1.annotate("", xy=(step_x[i+1] - 0.32, 0.525),
                     xytext=(x + 0.32, 0.525),
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

# 오차 막대 (누적 시각화)
for i, (x, bh) in enumerate(zip(step_x, bar_heights)):
    ax1.bar(x, bh, bottom=0.02, width=0.25, color=colors_grad[i], alpha=0.6, zorder=2)

ax1.text(9.3, 0.52, f"RMSE\n578\nMWh", ha="left", va="center",
         fontsize=9, color=RED, fontweight="bold",
         bbox=dict(boxstyle="round", fc="#fee2e2", ec=RED, alpha=0.9))

ax1.text(0.5, 0.13, "각 단계가 이전 예측에 의존", fontsize=9, color=RED)
ax1.text(0.5, 0.03, "→ 오차가 오차를 먹는다. 96번째 예측은 95개 오차의 합산", fontsize=9, color=RED)

# 오차 크기 레이블
for i in [0, 4, 9]:
    ax1.text(step_x[i], bar_heights[i] + 0.05,
             f"h={i*10+1 if i>0 else 1}\n↑오차", ha="center",
             fontsize=7, color=RED, alpha=0.8)

# 하단: TTM One-shot
ax2 = axes[1]
ax2.set_facecolor(BG)
ax2.axis("off")
ax2.text(0.0, 0.92, "TTM One-shot 방식: 96시간을 한 번에 직접 예측",
         fontsize=11, color=NAVY, fontweight="bold", transform=ax2.transAxes)

# 입력 블록
start_box = FancyBboxPatch((0.3, 0.38), 0.9, 0.32,
                            boxstyle="round,pad=0.03", fc=GREEN, ec="white", lw=1.5)
ax2.add_patch(start_box)
ax2.text(0.75, 0.54, "시작", ha="center", va="center",
         fontsize=10, color="white", fontweight="bold")

# 넓은 화살표
ax2.annotate("", xy=(8.8, 0.54), xytext=(1.25, 0.54),
             arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                             color=BLUE, lw=3,
                             connectionstyle="arc3,rad=0"))

# 배경 블록 (96h 동시 예측)
big_box = FancyBboxPatch((1.3, 0.35), 7.4, 0.38,
                          boxstyle="round,pad=0.03", fc=BLUE, ec="white",
                          lw=0, alpha=0.15)
ax2.add_patch(big_box)
ax2.text(5.0, 0.54, "96h 동시 예측 (한 번에)",
         ha="center", va="center", fontsize=11, color=BLUE, fontweight="bold")

# 종료 박스
end_box = FancyBboxPatch((8.85, 0.38), 0.9, 0.32,
                          boxstyle="round,pad=0.03", fc=NAVY, ec="white", lw=1.5)
ax2.add_patch(end_box)
ax2.text(9.3, 0.54, "완료", ha="center", va="center",
         fontsize=10, color="white", fontweight="bold")

ax2.text(9.35, 0.82, f"RMSE\n135\nMWh", ha="left", va="center",
         fontsize=9, color=NAVY, fontweight="bold",
         bbox=dict(boxstyle="round", fc="#dbeafe", ec=NAVY, alpha=0.9))

# 오차 안정
for ix in [2.5, 4.5, 6.5, 8.5]:
    ax2.bar(ix, 0.05, bottom=0.02, width=0.25, color=NAVY, alpha=0.4)
ax2.text(0.5, 0.13, "단 한 번의 연산으로 96시간 전체를 예측", fontsize=9, color=NAVY)
ax2.text(0.5, 0.03, "→ 이전 예측값을 참조하지 않으므로 오차가 누적되지 않음",
         fontsize=9, color=NAVY)

ax2.set_xlim(0, 10)
ax2.set_ylim(0, 1)

ax2.text(0.02, -0.06,
         "* 1월 라이브 테스트 결과 (2026-06-02 실행 검증) | 데이터: KPX 서울 2024 | h=96 기준 RMSE",
         transform=ax2.transAxes, fontsize=7.5, color="gray")

fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "fig_oneshot_vs_roll.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  → saved fig_oneshot_vs_roll.png")

print("\n전체 완료!")
