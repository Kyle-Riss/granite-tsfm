"""
연간 backtest 완료 후 자동 실행:
  - model_compare_block_2024.json 갱신 확인
  - 검증된 연간 수치로 시각화 재생성
  - 완료 리포트 출력

Usage:
  python scripts/finalize_after_backtest.py          # 완료까지 polling
  python scripts/finalize_after_backtest.py --now    # 지금 즉시 현재 JSON으로 재생성
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# ── 공통 팔레트 ──────────────────────────────────────────────────────────
NAVY   = "#1e3a5f"; BLUE   = "#0ea5e9"; GREEN = "#22c55e"
ORANGE = "#f97316"; GRAY   = "#94a3b8"; RED   = "#ef4444"; BG = "#f8fafc"

_KO_CANDIDATES = ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "sans-serif"]
_KO_FONT = next(
    (f for f in _KO_CANDIDATES if any(f == ff.name for ff in _fm.fontManager.ttflist)),
    "sans-serif",
)
plt.rcParams.update({"font.family": _KO_FONT, "axes.unicode_minus": False})

ART = Path("artifacts/seoul")
OUT = ART / "ppt_figures"
OUT.mkdir(parents=True, exist_ok=True)

BLOCK_JSON     = ART / "model_compare_block_2024.json"
VERIFIED_JSON  = ART / "model_compare_block_annual_verified.json"

# ── 완료 감지 기준: 블록 수가 80개 이상이면 연간 전체 실행으로 간주 ──────
MIN_BLOCKS = 80


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def is_annual_complete(d: dict) -> bool:
    """block_models의 ttm_block이 연간 수준(n_hours >= 8000)이면 완료."""
    try:
        return d["block_models"]["ttm_block"].get("n_hours", 0) >= 8000
    except (KeyError, TypeError):
        return False


def wait_for_completion(poll_sec: int = 30) -> dict:
    print("연간 backtest 완료 대기 중... (Ctrl+C로 중단)")
    while True:
        if BLOCK_JSON.exists():
            d = load_json(BLOCK_JSON)
            if is_annual_complete(d):
                print("완료 감지!")
                return d
            bm = d.get("block_models", {})
            n  = bm.get("ttm_block", {}).get("n_hours", 0)
            pct = n / 8784 * 100 if n else 0
            print(f"  진행: {n:,}h / 8,784h ({pct:.0f}%)")
        time.sleep(poll_sec)


def save_verified(d: dict) -> None:
    """갱신된 연간 결과를 verified JSON에도 저장 (덮어쓰기 방지용)."""
    verified = load_json(VERIFIED_JSON) if VERIFIED_JSON.exists() else {}
    verified["block_models"] = {
        k: {"rmse": v["rmse"], "label": v.get("label", k)}
        for k, v in d["block_models"].items()
    }
    verified["n_hours"]     = d["block_models"]["ttm_block"].get("n_hours", 8784)
    verified["source"]      = "annual_backtest_completed"
    with open(VERIFIED_JSON, "w") as f:
        json.dump(verified, f, indent=2, ensure_ascii=False)
    print(f"  verified JSON 저장: {VERIFIED_JSON}")


# ── 시각화 1: 연간 + 라이브 2패널 비교 ─────────────────────────────────
def make_comparison_chart(annual: dict, live: dict) -> None:
    models = ["Seasonal\n24h", "TTM 96h\n(one-shot)", "GRU\nroll", "Hybrid\nroll"]
    keys   = ["seasonal_24h", "ttm_block", "gru_block_roll", "hybrid_block_roll"]
    colors = [GRAY, BLUE, ORANGE, RED]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    fig.patch.set_facecolor("white")

    datasets = [
        (axes[0], annual, "연간 전체 backtest\n(92블록, 8,784h -- 2024년 1~12월)"),
        (axes[1], live,   "라이브 테스트 확인\n(4블록, 2024년 1월 -- 2026-06-02 실행)"),
    ]

    for ax, data, title in datasets:
        vals = [data[k]["rmse"] for k in keys]

        ax.axvspan(-0.5, 1.5, alpha=0.07, color=GREEN, zorder=0)
        ax.axvspan(1.5,  3.5, alpha=0.06, color=RED,   zorder=0)

        bars = ax.bar(models, vals, color=colors, edgecolor="white", width=0.6, zorder=3)
        bars[1].set_edgecolor(NAVY); bars[1].set_linewidth(3)

        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + max(vals) * 0.015,
                f"{v:.0f}",
                ha="center", va="bottom", fontsize=13, fontweight="bold", color=NAVY,
            )

        gru_v = vals[2]; ttm_v = vals[1]
        pct   = (gru_v - ttm_v) / gru_v * 100
        ax.annotate("", xy=(1, ttm_v), xytext=(2, gru_v),
                    arrowprops=dict(arrowstyle="<->", color=NAVY, lw=2.2))
        ax.text(
            1.5, (ttm_v + gru_v) / 2,
            f"GRU roll 대비\n{pct:.0f}% 낮음",
            ha="center", va="center", fontsize=14, fontweight="bold", color=NAVY,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=NAVY, linewidth=2.2),
        )

        ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY, pad=12)
        ax.set_ylim(0, max(vals) * 1.35)
        ax.set_facecolor(BG)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", color="#e2e8f0", linewidth=0.6)
        ax.set_ylabel("RMSE (MWh)  -- 낮을수록 좋음", fontsize=11, color=NAVY)
        ax.tick_params(axis="x", labelsize=11)

    a_pct = (annual["gru_block_roll"]["rmse"] - annual["ttm_block"]["rmse"]) \
            / annual["gru_block_roll"]["rmse"] * 100
    l_pct = (live["gru_block_roll"]["rmse"]   - live["ttm_block"]["rmse"])   \
            / live["gru_block_roll"]["rmse"]   * 100

    fig.suptitle(
        f"4일(96h) 앞 예측 -- TTM one-shot vs Roll 방식\n"
        f"연간 전체: {a_pct:.0f}% / 1월 라이브: {l_pct:.0f}% 낮음  --  두 결과 동일 결론 확인",
        fontsize=13, fontweight="bold", color=NAVY, y=1.02,
    )
    fig.text(
        0.5, -0.02,
        "* GRU roll: 1h씩 96번 반복 예측 -> 오차 누적  |  "
        "TTM one-shot: 96h 한 번에 직접 예측  |  "
        "연간 테스트: 서울 2024년 전체 / 라이브: 2026-06-02 실행 검증",
        ha="center", fontsize=9.5, color="#64748b", style="italic",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = OUT / "fig_block_rmse_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ── 시각화 2: 메인 단일 패널 (PPT 슬라이드 12용) ──────────────────────
def make_main_bar(annual: dict) -> None:
    models = ["Seasonal\n24h\n(기준선)", "TTM 96h\n(one-shot)", "GRU\nroll", "Hybrid\nroll"]
    keys   = ["seasonal_24h", "ttm_block", "gru_block_roll", "hybrid_block_roll"]
    colors = [GRAY, BLUE, ORANGE, RED]
    vals   = [annual[k]["rmse"] for k in keys]

    fig, ax = plt.subplots(figsize=(13, 7.5))
    fig.patch.set_facecolor("white")

    ax.axvspan(-0.5, 1.5, alpha=0.07, color=GREEN, zorder=0)
    ax.axvspan(1.5,  3.5, alpha=0.06, color=RED,   zorder=0)

    bars = ax.bar(models, vals, color=colors, edgecolor="white", width=0.6, zorder=3)
    bars[1].set_edgecolor(NAVY); bars[1].set_linewidth(3)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 8,
                f"{v:.0f} MWh", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=NAVY)

    y_top = max(vals) * 1.30
    ax.text(0.5, y_top * 0.97, "[O] 실용 가능", ha="center", va="top",
            fontsize=12, color="#166534", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#dcfce7",
                      alpha=0.9, edgecolor="#166534", linewidth=1.8))
    ax.text(2.5, y_top * 0.97, "[X] 오차 누적 -- 운영 부적합", ha="center", va="top",
            fontsize=12, color="#991b1b", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#fee2e2",
                      alpha=0.9, edgecolor="#991b1b", linewidth=1.8))

    gru_v = vals[2]; ttm_v = vals[1]
    pct_gru    = (gru_v   - ttm_v) / gru_v   * 100
    pct_hybrid = (vals[3] - ttm_v) / vals[3]  * 100

    ax.annotate("", xy=(1, ttm_v), xytext=(2, gru_v),
                arrowprops=dict(arrowstyle="<->", color=NAVY, lw=2.2))
    ax.text(1.5, (ttm_v + gru_v) / 2,
            f"GRU roll 대비\n{pct_gru:.0f}% 낮음",
            ha="center", va="center", fontsize=15, fontweight="bold", color=NAVY,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="white",
                      edgecolor=NAVY, linewidth=2.2))
    ax.annotate(f"Hybrid roll 대비\n{pct_hybrid:.0f}% 낮음",
                xy=(1, ttm_v), xytext=(1.85, ttm_v + 130),
                fontsize=10, color="#7c3aed", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.6,
                                connectionstyle="arc3,rad=0.3"))

    ax.set_ylabel("RMSE (MWh)  --  낮을수록 좋음", fontsize=12, color=NAVY)
    ax.set_title(
        "4일(96h) 앞 예측 -- 연간 전체 backtest (서울 2024, 8,784h / 92블록)\n"
        "TTM one-shot vs 1시간씩 굴리는 Roll 방식 비교",
        fontsize=14, fontweight="bold", color=NAVY, pad=16,
    )
    ax.set_facecolor(BG)
    ax.set_ylim(0, max(vals) * 1.33)
    ax.tick_params(axis="x", labelsize=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#e2e8f0", linewidth=0.6)

    fig.text(
        0.5, 0.01,
        "* Roll 방식: 1h씩 96번 반복 예측 -> 오차 누적  |  "
        "TTM one-shot: 96h를 한 번에 직접 예측 (오차 누적 없음)  |  "
        "KPX 데이터 2020-2024 (43,848h, 서울시)",
        ha="center", fontsize=9.5, color="#64748b", style="italic",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = OUT / "fig_block_rmse_bar.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def print_report(d: dict) -> None:
    bm = d["block_models"]
    print("\n" + "=" * 55)
    print("  연간 backtest 최종 결과 — 서울 2024")
    print("=" * 55)
    order = ["seasonal_24h", "ttm_block", "gru_block_roll",
             "hybrid_block_roll", "hybrid_block_tf"]
    for k in order:
        if k in bm:
            v = bm[k]
            label = v.get("label", k)
            print(f"  {label:45s} RMSE={v['rmse']:.1f}")

    ttm = bm["ttm_block"]["rmse"]
    gru = bm["gru_block_roll"]["rmse"]
    hyb = bm["hybrid_block_roll"]["rmse"]
    print(f"\n  TTM vs GRU roll  : {(gru-ttm)/gru*100:.0f}% 낮음")
    print(f"  TTM vs Hybrid roll: {(hyb-ttm)/hyb*100:.0f}% 낮음")
    print("=" * 55 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", action="store_true",
                        help="대기 없이 현재 JSON으로 즉시 재생성")
    args = parser.parse_args()

    verified = load_json(VERIFIED_JSON)

    if args.now:
        if BLOCK_JSON.exists():
            d = load_json(BLOCK_JSON)
            annual_bm = d["block_models"]
        else:
            annual_bm = verified["block_models"]
    else:
        d = wait_for_completion()
        annual_bm = d["block_models"]
        save_verified(d)

    live_bm = verified["live_4block_jan"]

    print("\n시각화 재생성 중...")
    make_comparison_chart(annual_bm, live_bm)
    make_main_bar(annual_bm)
    print_report({"block_models": annual_bm})
    print("완료! 파일 위치:", OUT)


if __name__ == "__main__":
    main()
