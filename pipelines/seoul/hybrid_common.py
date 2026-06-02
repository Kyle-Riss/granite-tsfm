# Copyright contributors to the TSFM project
#
"""Shared data loading and scaling for TSFM (frozen) + GRU (trainable head)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REGION_LABELS = ("서울시", "부산시", "대전시", "강원도")


@dataclass
class FeatureScaler:
    power_mean: float
    power_std: float
    temp_mean: float
    temp_std: float

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> FeatureScaler:
        pstd = float(df["power"].std())
        tstd = float(df["temp"].std())
        return cls(
            power_mean=float(df["power"].mean()),
            power_std=pstd if pstd > 1e-9 else 1.0,
            temp_mean=float(df["temp"].mean()),
            temp_std=tstd if tstd > 1e-9 else 1.0,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        raise ValueError(f"Expected column 'ts' in {path}")
    return df.sort_values("ts").reset_index(drop=True)


def as_ttm_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp column for TTM preprocessor / pipeline."""
    out = df.copy()
    if "timestamp" not in out.columns:
        if "ts" not in out.columns:
            raise ValueError("Expected column 'ts' or 'timestamp'")
        out = out.rename(columns={"ts": "timestamp"})
    return out.sort_values("timestamp").reset_index(drop=True)


def load_hourly_for_ttm(source: Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return as_ttm_frame(source)
    return as_ttm_frame(load_hourly(source))


def region_one_hot(region: str | None) -> np.ndarray:
    vec = np.zeros(len(REGION_LABELS), dtype=np.float32)
    if region is None:
        vec[0] = 1.0
        return vec
    try:
        vec[REGION_LABELS.index(region)] = 1.0
    except ValueError:
        vec[0] = 1.0
    return vec


def hour_features(hour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.sin(2 * math.pi * hour / 24.0).astype(np.float32),
        np.cos(2 * math.pi * hour / 24.0).astype(np.float32),
    )
