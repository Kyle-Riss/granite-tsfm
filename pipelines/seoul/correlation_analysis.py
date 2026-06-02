# Copyright contributors to the TSFM project
#
"""5.1: 구간·지역별 상관, CDD/HDD 민감도 → sensitivity_51.json"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.seoul.config import REGIONAL_HOURLY_PARQUET, SENSITIVITY_51_JSON


def _corr(x: pd.Series, y: pd.Series) -> float | None:
    if len(x) < 30 or x.std() < 1e-9 or y.std() < 1e-9:
        return None
    return float(x.corr(y))


def _slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 30 or np.nanstd(x) < 1e-9:
        return None
    return float(np.polyfit(x, y, 1)[0])


def analyze(df: pd.DataFrame) -> dict:
    out: dict = {"regions": {}, "bands": ["cooling", "mid", "heating"], "global": {}}

    out["global"]["corr_temp_power"] = _corr(df["temp"], df["power"])
    out["global"]["corr_cdd_power"] = _corr(df["cdd"], df["power"])
    out["global"]["corr_hdd_power"] = _corr(df["hdd"], df["power"])

    for band in ("cooling", "mid", "heating"):
        sub = df[df["temp_band"] == band]
        out["global"][f"corr_temp_power_{band}"] = _corr(sub["temp"], sub["power"])

    for reg in sorted(df["region"].unique()):
        g = df[df["region"] == reg]
        pop = float(g["population"].iloc[0]) if "population" in g.columns and g["population"].notna().any() else None
        entry = {
            "corr_temp_power": _corr(g["temp"], g["power"]),
            "corr_cdd_power": _corr(g["cdd"], g["power"]),
            "corr_hdd_power": _corr(g["hdd"], g["power"]),
            "slope_temp_power": _slope(g["temp"].to_numpy(), g["power"].to_numpy()),
            "mean_power": float(g["power"].mean()),
            "mean_cdd": float(g["cdd"].mean()),
            "mean_hdd": float(g["hdd"].mean()),
        }
        if pop and pop > 0:
            entry["power_per_capita"] = entry["mean_power"] / pop
        for band in ("cooling", "mid", "heating"):
            sub = g[g["temp_band"] == band]
            entry[f"corr_temp_power_{band}"] = _corr(sub["temp"], sub["power"])
        out["regions"][str(reg)] = entry

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Section 5.1 correlation & sensitivity export.")
    parser.add_argument("--parquet", type=Path, default=REGIONAL_HOURLY_PARQUET)
    parser.add_argument("--output", type=Path, default=SENSITIVITY_51_JSON)
    args = parser.parse_args(argv)

    if not args.parquet.is_file():
        raise SystemExit(f"Missing {args.parquet}. Run: python -m pipelines.seoul.materialize")

    df = pd.read_parquet(args.parquet)
    for c in ("temp", "power", "region", "cdd", "hdd", "temp_band"):
        if c not in df.columns:
            raise SystemExit(f"Column {c} missing — re-run materialize to add CDD/HDD/temp_band.")

    payload = analyze(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
