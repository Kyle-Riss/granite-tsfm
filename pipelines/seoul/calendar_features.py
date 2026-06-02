# Copyright contributors to the TSFM project
#
"""Korean public holidays — python-holidays (offline) with optional hyunbin JSON cache."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from pipelines.seoul.config import CALENDAR_JSON

HOLIDAYS_URL = "https://holidays.hyunbin.page/basic.json"


def _holidays_from_library(years: range) -> set[date]:
    import holidays

    kr = holidays.SouthKorea(years=list(years))
    return set(kr.keys())


def fetch_kr_holidays_json(dest: Path | None = None) -> dict:
    dest = dest or CALENDAR_JSON
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(HOLIDAYS_URL, headers={"User-Agent": "granite-tsfm-pipeline/1.0"})
    try:
        with urlopen(req, timeout=60) as resp:
            data = json.load(resp)
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data
    except Exception:
        years = range(2018, 2027)
        lib = _holidays_from_library(years)
        data = {"_source": "python-holidays", "dates": [str(d) for d in sorted(lib)]}
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data


def load_holiday_dates(path: Path | None = None, years: range | None = None) -> set[date]:
    years = years or range(2018, 2027)
    path = path or CALENDAR_JSON
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "dates" in raw:
            return {pd.Timestamp(d).date() for d in raw["dates"]}
        dates: set[date] = set()
        if isinstance(raw, dict):
            for year_block in raw.values():
                if isinstance(year_block, dict):
                    for day_str in year_block:
                        try:
                            dates.add(pd.Timestamp(day_str).date())
                        except Exception:
                            continue
        if dates:
            return dates
    return _holidays_from_library(years)


def attach_calendar_flags(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col])
    yr = range(int(out[ts_col].dt.year.min()), int(out[ts_col].dt.year.max()) + 1)
    hol = load_holiday_dates(years=yr)
    d = out[ts_col].dt.date
    out["is_holiday"] = d.map(lambda x: int(x in hol)).astype(int)
    eve = {date.fromordinal(x.toordinal() + 1) for x in hol}
    out["is_holiday_eve"] = d.map(lambda x: int(x in eve)).astype(int)
    return out
