"""
PPT 그림 2차 생성 — Slot 2, 6, 8
artifacts/seoul/ppt_figures/ 에 저장
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as _fm
import matplotlib.patheffects as pe

_KO = next((f for f in ["Apple SD Gothic Neo","AppleGothic","NanumGothic","sans-serif"]
            if any(f == ff.name for ff in _fm.fontManager.ttflist)), "sans-serif")
plt.rcParams.update({
    "font.family": _KO, "axes.unicode_minus": False,
    "axes.facecolor": "#f8fafc", "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e2e8f0", "grid.linewidth": 0.6,
})

NAVY   = "#1e3a5f"
BLUE   = "#0ea5e9"
GREEN  = "#22c55e"
ORANGE = "#f97316"
GRAY   = "#94a3b8"
RED    = "#ef4444"
PURPLE = "#8b5cf6"
BG     = "#f8fafc"

OUT = Path("artifacts/seoul/ppt_figures")
ART = Path("artifacts/seoul")
OUT.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Slot 2 — 4개 광역 위치 지도 (간단 좌표 기반)
# ══════════════════════════════════════════════════════════════
def slot2_regions_map() -> None:
    print("[Slot 2] 4개 광역 위치 지도 생성 중...")

    # 한반도 대략 윤곽 (위경도 근사)
    peninsula_lat = [34.0, 34.5, 35.0, 35.5, 36.0, 36.5, 37.0, 37.5,
                     38.0, 38.5, 39.0, 38.5, 38.0, 37.5, 37.0, 36.5,
                     36.0, 35.5, 35.0, 34.5, 34.0]
    peninsula_lon = [126.5, 126.0, 125.5, 126.0, 126.2, 126.0, 126.2,
                     126.5, 126.8, 127.0, 127.5, 128.5, 129.0, 129.3,
                     129.5, 129.3, 129.0, 128.8, 128.5, 127.5, 126.5]

    regions = {
        "서울시":  (37.57, 126.98, BLUE,   "소비 전용\n(562 MWh)"),
        "부산시":  (35.18, 129.07, RED,    "발전 기반\n(4,094 MWh)\n고리 원전"),
        "대전시":  (36.35, 127.38, GREEN,  "소비 전용\n(27 MWh)"),
        "강원도":  (37.55, 128.20, ORANGE, "발전 기반\n(3,855 MWh)\n동해·삼척"),
    }

    fig, ax = plt.subplots(figsize=(7, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#dbeafe")  # 바다색

    # 한반도 윤곽 (다각형)
    ax.fill(peninsula_lon, peninsula_lat, color="#e8f5e9", alpha=0.9,
            linewidth=1.5, edgecolor="#94a3b8", zorder=1)

    # 각 지역 마커
    for name, (lat, lon, color, note) in regions.items():
        ax.scatter(lon, lat, s=250, color=color, zorder=5,
                   edgecolors="white", linewidths=2)
        ax.text(lon + 0.15, lat + 0.08, name,
                fontsize=11, fontweight="bold", color=color, zorder=6)
        ax.text(lon + 0.15, lat - 0.18, note,
                fontsize=7.5, color="#475569", zorder=6,
                linespacing=1.4)

    ax.set_xlim(124.5, 130.5)
    ax.set_ylim(33.5, 39.5)
    ax.set_xlabel("경도", fontsize=9, color=GRAY)
    ax.set_ylabel("위도", fontsize=9, color=GRAY)
    ax.set_title("분석 지역 — 4개 광역\n(발전 기반 vs 소비 전용 구조 차이)", 
                 fontsize=13, fontweight="bold", color=NAVY, pad=12)
    ax.tick_params(labelsize=8, colors=GRAY)

    legend_patches = [
        mpatches.Patch(color=RED,    label="발전 기반 지역 (거래량 多)"),
        mpatches.Patch(color=ORANGE, label="발전 기반 지역 (거래량 多)"),
        mpatches.Patch(color=BLUE,   label="소비 전용 지역 (거래량 少)"),
        mpatches.Patch(color=GREEN,  label="소비 전용 지역 (거래량 少)"),
    ]
    # 범례 중복 제거
    ax.legend(handles=[
        mpatches.Patch(color="#ef4444", label="발전 기반 (부산·강원)"),
        mpatches.Patch(color="#0ea5e9", label="소비 전용 (서울·대전)"),
    ], fontsize=9, loc="lower left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT / "fig_regions_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → saved fig_regions_map.png")


# ══════════════════════════════════════════════════════════════
# Slot 6 — 모델 파이프라인 다이어그램
# ══════════════════════════════════════════════════════════════
def slot6_model_pipeline() -> None:
    print("[Slot 6] 모델 파이프라인 다이어그램 생성 중...")

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")

    def box(ax, x, y, w, h, label, sublabel, color, text_color="white", style="round,pad=0.1"):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle=style, facecolor=color,
            edgecolor="white", linewidth=1.5, zorder=3
        ))
        ax.text(x, y + 0.06, label,
                ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=text_color, zorder=4)
        if sublabel:
            ax.text(x, y - 0.1, sublabel,
                    ha="center", va="center", fontsize=7.5,
                    color=text_color, alpha=0.85, zorder=4)

    def arrow(ax, x1, x2, y=0.5):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="-|>", color=GRAY,
                                   lw=1.8, mutation_scale=14), zorder=2)

    # 노드 정의 (x, y, w, h, label, sublabel, color)
    nodes = [
        (0.08, 0.50, 0.13, 0.30, "전력·기온\n데이터",   "KPX + Open-Meteo\n2020–2024",  "#22c55e"),
        (0.28, 0.50, 0.15, 0.30, "Granite TTM-r2\n[FROZEN]", "ctx 512h\n사전학습 가중치 고정", NAVY),
        (0.28, 0.15, 0.13, 0.18, "Frozen\n고정",        "",                               "#64748b"),  # 배지
        (0.50, 0.50, 0.14, 0.30, "Embedding\n풀링",     "시계열 압축\n표현",              "#7c3aed"),
        (0.70, 0.50, 0.15, 0.30, "GRU Head\n[TRAINABLE]", "seq 168h / hidden 64\n서울 데이터 학습", BLUE),
        (0.70, 0.15, 0.13, 0.18, "Trainable\n학습",     "",                               "#0284c7"),  # 배지
        (0.90, 0.50, 0.13, 0.30, "전력거래량\n예측",    "96h 앞\nRMSE ±15.8 MWh",        ORANGE),
    ]

    for i, (x, y, w, h, lbl, sub, col) in enumerate(nodes):
        if i in (2, 5):  # 배지
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - w/2, y - h/2), w, h,
                boxstyle="round,pad=0.05", facecolor=col,
                edgecolor="white", linewidth=1, alpha=0.85, zorder=3
            ))
            ax.text(x, y, lbl, ha="center", va="center",
                    fontsize=7, fontweight="bold", color="white", zorder=4)
        else:
            box(ax, x, y, w, h, lbl, sub, col)

    # 화살표
    arrow(ax, 0.145, 0.205)   # 데이터 → TTM
    arrow(ax, 0.355, 0.430)   # TTM → embedding
    arrow(ax, 0.570, 0.625)   # embedding → GRU
    arrow(ax, 0.775, 0.835)   # GRU → 예측

    # 상단 레이블
    ax.text(0.28, 0.88, "① 사전학습된 AI\n(건드리지 않음)", ha="center",
            fontsize=8, color=NAVY, style="italic")
    ax.text(0.70, 0.88, "② 서울 특화 학습\n(추가 학습 부분)", ha="center",
            fontsize=8, color=BLUE, style="italic")

    # 구분선
    ax.axvline(0.59, ymin=0.1, ymax=0.9, color="#e2e8f0",
               linewidth=1.5, linestyle="--", zorder=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Hybrid 모델 구조: Granite TTM (frozen) + GRU Head (학습)\n"
                 "= 이미 아는 AI에 서울 데이터만 추가 학습 — 효율적 파인튜닝",
                 fontsize=12, fontweight="bold", color=NAVY, pad=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig_model_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → saved fig_model_pipeline.png")


# ══════════════════════════════════════════════════════════════
# Slot 8 — 지역별 예측 vs 실측 비교 (1월 샘플 7일)
# ══════════════════════════════════════════════════════════════
def slot8_region_compare() -> None:
    print("[Slot 8] 지역별 예측 vs 실측 생성 중...")

    regions = {
        "서울시":  (BLUE,   "소비 전용 — 예측 정확"),
        "부산시":  (RED,    "발전 기반 — 스케일 큼"),
        "대전시":  (GREEN,  "소비 전용 — 스케일 작음"),
        "강원도":  (ORANGE, "발전 기반 — 변동 큼"),
    }

    # RMSE 계산
    rmse_dict = {}
    dfs = {}
    for region in regions:
        df = pd.read_parquet(ART / f"forecast_region_{region}_2024.parquet")
        df = df.dropna(subset=["power_pred", "power_actual"])
        rmse = np.sqrt(((df["power_pred"] - df["power_actual"])**2).mean())
        rmse_dict[region] = rmse
        dfs[region] = df

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor("white")

    sample_days = 7 * 24  # 1주일

    for ax, (region, (color, note)) in zip(axes.flat, regions.items()):
        df = dfs[region].head(sample_days)
        x = range(len(df))

        ax.fill_between(x, df["power_actual"], alpha=0.10, color=GRAY)
        ax.plot(x, df["power_actual"], color="black", linewidth=1.8,
                label="실측", zorder=4)
        ax.plot(x, df["power_pred"], color=color, linewidth=1.6,
                linestyle="--", label=f"예측 (서울 모델)", zorder=3)

        rmse = rmse_dict[region]
        ax.set_title(f"{region}   연간 RMSE: {rmse:.1f} MWh",
                     fontsize=11, fontweight="bold", color=NAVY)
        ax.text(0.98, 0.96, note, transform=ax.transAxes,
                fontsize=8, ha="right", va="top", color="#64748b",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        ax.set_facecolor(BG)
        ax.set_xlabel("시간 (1월 첫 7일)", fontsize=8)
        ax.set_ylabel("전력거래량 (MWh)", fontsize=8)
        ax.legend(fontsize=8, loc="lower right")
        ax.tick_params(labelsize=8)

    # 하단 메시지
    fig.suptitle("지역별 예측 결과 비교 — 서울로 학습한 모델을 4개 지역에 적용\n"
                 "→ 지역마다 정확도 차이 발생 = 지역별 별도 모델 필요",
                 fontsize=13, fontweight="bold", color=NAVY, y=1.01)

    # RMSE 요약 텍스트
    rmse_txt = "  |  ".join([f"{r}: {v:.0f}" for r, v in rmse_dict.items()])
    fig.text(0.5, -0.01, f"연간 RMSE (MWh): {rmse_txt}",
             ha="center", fontsize=9, color="#64748b")

    fig.tight_layout()
    fig.savefig(OUT / "fig_region_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → saved fig_region_compare.png")


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n=== PPT 그림 2차 생성 → {OUT} ===\n")
    slot2_regions_map()
    slot6_model_pipeline()
    slot8_region_compare()
    print("\n완료! 생성된 파일:")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
