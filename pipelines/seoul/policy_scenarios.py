# Copyright contributors to the TSFM project
#
"""6.2 정책 활용 시나리오 문구 — 데이터·민감도 기반 자동 생성."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pipelines.seoul.config import COOLING_BAND_TEMP_C, HEATING_BAND_TEMP_C


def _fmt_corr(v: float | None) -> str:
    if v is None:
        return "측정 불가"
    return f"{v:.3f}"


def build_policy_payload(
    hourly: pd.DataFrame,
    sensitivity: dict | None,
    district: pd.DataFrame | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Return {region_or '_national': [{title, body, tag}, ...]}."""
    sens_regions = (sensitivity or {}).get("regions", {})
    out: dict[str, list[dict[str, str]]] = {"_national": _national_scenarios(hourly, sensitivity)}

    for reg in sorted(hourly["region"].unique(), key=str):
        reg_s = str(reg)
        g = hourly[hourly["region"] == reg]
        sr = sens_regions.get(reg_s, {})
        out[reg_s] = _region_scenarios(reg_s, g, sr, district if reg_s == "서울시" else None)

    return out


def _national_scenarios(hourly: pd.DataFrame, sensitivity: dict | None) -> list[dict[str, str]]:
    by_reg = hourly.groupby("region", observed=True)["power"].mean().sort_values(ascending=False)
    top = str(by_reg.index[0]) if len(by_reg) else "—"
    low = str(by_reg.index[-1]) if len(by_reg) else "—"
    gap = float(by_reg.iloc[0] - by_reg.iloc[-1]) if len(by_reg) >= 2 else 0.0
    g = (sensitivity or {}).get("global", {})
    return [
        {
            "tag": "수요 관리",
            "title": "광역 수요 격차 완화",
            "body": (
                f"평균 전력 1위는 {top}, 최하위는 {low}로 격차 약 {gap:,.0f} 단위입니다. "
                f"피크 시간대 부하이전·DR(수요반응)은 고부하 광역부터 단계적으로 설계하는 것이 효율적입니다."
            ),
        },
        {
            "tag": "인프라",
            "title": "계절·기온 연동 설비 투자",
            "body": (
                f"전국 r(기온·전력)={_fmt_corr(g.get('corr_temp_power'))}, "
                f"냉방구간 r={_fmt_corr(g.get('corr_temp_power_cooling'))}, "
                f"난방구간 r={_fmt_corr(g.get('corr_temp_power_heating'))}. "
                f"냉·난방 임계({COOLING_BAND_TEMP_C}°C / {HEATING_BAND_TEMP_C}°C) 전후 변압·배전 증설 우선순위를 달리 두는 시나리오가 타당합니다."
            ),
        },
        {
            "tag": "기후·탄소",
            "title": "탄소저감 목표와 연계한 부하 관리",
            "body": (
                "시간대·기온 구간별 부하 프로파일을 기준선으로 두고, "
                "피크 저감·재생에너지 수용 시나리오별 감축량을 광역 단위로 환산해 KPI를 설정할 수 있습니다."
            ),
        },
    ]


def _region_scenarios(
    region: str,
    g: pd.DataFrame,
    sens: dict[str, Any],
    district: pd.DataFrame | None,
) -> list[dict[str, str]]:
    corr = sens.get("corr_temp_power")
    cdd = sens.get("corr_cdd_power")
    mean_p = float(g["power"].mean())
    cooling = g[g["temp_band"] == "cooling"]["power"].mean() if "temp_band" in g.columns else None
    heating = g[g["temp_band"] == "heating"]["power"].mean() if "temp_band" in g.columns else None

    scenarios: list[dict[str, str]] = [
        {
            "tag": "수요 관리",
            "title": f"{region} 피크·평일/주말 수요 조정",
            "body": (
                f"평균 전력 {mean_p:,.0f}, r(기온·전력)={_fmt_corr(corr)}. "
                "평일·주말 시간대 프로파일 차이가 크면 요금제·DR 참여 시간대를 분리 설계합니다."
            ),
        },
        {
            "tag": "인프라",
            "title": f"{region} CDD/HDD 민감도 기반 설비",
            "body": (
                f"r(CDD·전력)={_fmt_corr(cdd)}, r(HDD·전력)={_fmt_corr(sens.get('corr_hdd_power'))}. "
                + (
                    f"냉방구간 평균 {cooling:,.0f} · 난방구간 {heating:,.0f} — "
                    if cooling is not None and heating is not None and not pd.isna(cooling)
                    else ""
                )
                + "해당 구간 피크에 맞춘 변전·냉난방 설비 증설 시나리오를 검토합니다."
            ),
        },
    ]

    if district is not None and len(district):
        annual = district.groupby("district", observed=True)["usage"].sum().sort_values(ascending=False)
        top_d = str(annual.index[0])
        scenarios.append(
            {
                "tag": "공간 계획",
                "title": "서울 자치구별 부하 집중 대응",
                "body": (
                    f"연간 사용량 1위 자치구는 {top_d}({float(annual.iloc[0]):,.0f}). "
                    "월별 지도에서 여름철 쏠림 구를 우선 대상으로 분산전원·마이크로그리드 파일럿을 배치할 수 있습니다."
                ),
            }
        )

    scenarios.append(
        {
            "tag": "기후·탄소",
            "title": f"{region} 감축 로드맵 벤치마크",
            "body": (
                "기온 구간별 평균 부하를 기준연도로 고정한 뒤, "
                "효율개선·재생에너지·피크저감 시나리오별 연간 전력·탄소 감축률을 시뮬레이션합니다."
            ),
        }
    )
    return scenarios
