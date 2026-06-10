# Copyright contributors to the TSFM project
#
"""Build synthetic future history via frozen TTM 96h one-shot blocks (not hybrid roll).

  # Full 2025 (slow on CPU: ~92 TTM calls)
  uv run python -m pipelines.seoul.ttm_roll_forward --through-year 2025

  # Demo scenarios through 2027-12 (pre-materialize for fast API)
  uv run python -m pipelines.seoul.ttm_roll_forward --through-year 2027

  # Smoke
  uv run python -m pipelines.seoul.ttm_roll_forward --through-year 2025 --max-blocks 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.seoul.config import (
    SEOUL_CITY_HOURLY_PARQUET,
    SYNTHETIC_HISTORY_START,
    TTM_SYNTHETIC_HISTORY_JSON,
    TTM_SYNTHETIC_HISTORY_PARQUET,
)
from pipelines.seoul.hybrid_common import load_hourly
from pipelines.seoul.hybrid_eval import TTMBlockForecaster


def _temp_climatology(hist: pd.DataFrame) -> pd.DataFrame:
    return (
        hist.groupby(["month", "hour"], observed=True)["temp"]
        .median()
        .reset_index()
        .rename(columns={"temp": "temp_clim"})
    )


def _attach_temp_clim(ts: pd.Series, clim: pd.DataFrame) -> pd.Series:
    frame = pd.DataFrame({"ts": pd.to_datetime(ts)})
    frame["hour"] = frame["ts"].dt.hour.astype(int)
    frame["month"] = frame["ts"].dt.month.astype(int)
    frame = frame.merge(clim, on=["month", "hour"], how="left")
    return frame["temp_clim"].astype(float)


def _load_hourly(path: Path) -> pd.DataFrame:
    if path.is_file():
        return load_hourly(path)
    from pipelines.seoul.config import REGIONAL_HOURLY_PARQUET

    df = load_hourly(REGIONAL_HOURLY_PARQUET)
    df = df[df["region"] == "서울시"].copy()
    return df.sort_values("ts").reset_index(drop=True)


def _block_starts(synthetic_start: pd.Timestamp, through: pd.Timestamp, block_hours: int) -> list[pd.Timestamp]:
    starts = pd.date_range(synthetic_start, through, freq=f"{block_hours}h")
    return [pd.Timestamp(t) for t in starts if pd.Timestamp(t) <= through]


def _seed_work_from_synthetic(work: pd.DataFrame, seed_synthetic: pd.DataFrame | None) -> None:
    if seed_synthetic is None or seed_synthetic.empty:
        return
    seed = seed_synthetic.copy()
    seed["ts"] = pd.to_datetime(seed["ts"])
    ts_to_idx = {pd.Timestamp(t): int(i) for i, t in enumerate(work["ts"])}
    for row in seed.itertuples(index=False):
        ts = pd.Timestamp(row.ts)
        if ts not in ts_to_idx or pd.isna(row.power):
            continue
        work.loc[ts_to_idx[ts], "power"] = float(row.power)


def extend_synthetic_ttm(
    hourly: pd.DataFrame,
    *,
    synthetic_start: pd.Timestamp,
    through: pd.Timestamp,
    forecaster: TTMBlockForecaster,
    block_hours: int = 96,
    max_blocks: int | None = None,
    log_every: int = 5,
    seed_synthetic: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Append non-overlapping TTM 96h blocks; return (synthetic_df, work_df, meta)."""
    hourly = hourly.sort_values("ts").reset_index(drop=True)
    hourly["ts"] = pd.to_datetime(hourly["ts"])
    data_end = hourly["ts"].max()
    hist = hourly[hourly["ts"] <= data_end].copy()
    if len(hist) < 512:
        raise ValueError(f"Need at least 512h history, got {len(hist)}")

    clim = _temp_climatology(hist)
    future_ts = pd.date_range(synthetic_start, through, freq="h")
    scaffold = pd.DataFrame({"ts": future_ts})
    scaffold["power"] = np.nan
    scaffold["temp"] = _attach_temp_clim(scaffold["ts"], clim)
    scaffold["is_synthetic"] = 1

    work = pd.concat(
        [
            hist.assign(is_synthetic=0),
            scaffold,
        ],
        ignore_index=True,
    )
    work = work.sort_values("ts").reset_index(drop=True)
    _seed_work_from_synthetic(work, seed_synthetic)

    starts = _block_starts(synthetic_start, through, block_hours)
    if max_blocks is not None:
        starts = starts[:max_blocks]

    n_blocks = 0
    n_hours = 0
    for bi, b0 in enumerate(starts):
        idx = work.index[work["ts"] == b0]
        if len(idx) == 0:
            continue
        test_start = int(idx[0])
        end_ts = min(b0 + pd.Timedelta(hours=block_hours - 1), through)
        end_idx = work.index[work["ts"] == end_ts]
        if len(end_idx) == 0:
            continue
        horizon = int(end_idx[0]) - test_start + 1
        if horizon <= 0:
            continue
        if not pd.isna(work.loc[test_start, "power"]):
            continue

        pred = forecaster.predict(work, test_start, horizon)
        n = min(len(pred), horizon)
        for j in range(n):
            work.loc[test_start + j, "power"] = float(pred[j])
        n_blocks += 1
        n_hours += n

        if log_every and (bi + 1) % log_every == 0:
            print(f"  block {bi + 1}/{len(starts)}  {b0}  ({n}h)", flush=True)

    synthetic = work[work["is_synthetic"] == 1][["ts", "power", "temp", "is_synthetic"]].copy()
    meta = {
        "method": "ttm_96h_oneshot_block_chain",
        "backbone": "ibm-granite/granite-timeseries-ttm-r2",
        "note": "Frozen TTM one-shot blocks (non-overlapping). Not hybrid GRU roll.",
        "actual_through": str(data_end),
        "synthetic_start": str(synthetic_start),
        "synthetic_through": str(through),
        "block_hours": block_hours,
        "blocks_run": n_blocks,
        "hours_filled": n_hours,
    }
    return synthetic, work, meta


def merge_synthetic_artifacts(
    existing: pd.DataFrame | None,
    new_synth: pd.DataFrame,
) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new_synth.sort_values("ts").reset_index(drop=True)
    out = pd.concat([existing, new_synth], ignore_index=True)
    out = out.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    return out.reset_index(drop=True)


def build_extended_frame(
    hourly: pd.DataFrame,
    synthetic: pd.DataFrame | None,
    *,
    through: pd.Timestamp,
) -> pd.DataFrame:
    hourly = hourly.sort_values("ts").reset_index(drop=True)
    hourly["ts"] = pd.to_datetime(hourly["ts"])
    data_end = hourly["ts"].max()
    hist = hourly.assign(is_synthetic=0)
    hist = hist[hist["ts"] <= through].copy()
    if synthetic is None or synthetic.empty:
        return hist.reset_index(drop=True)

    synth = synthetic.copy()
    synth["ts"] = pd.to_datetime(synth["ts"])
    synth = synth[(synth["ts"] > data_end) & (synth["ts"] <= through)]
    if synth.empty:
        return hist.reset_index(drop=True)
    return pd.concat([hist, synth], ignore_index=True).sort_values("ts").reset_index(drop=True)


def ensure_synthetic_through(
    hourly: pd.DataFrame,
    through: pd.Timestamp,
    forecaster: TTMBlockForecaster,
    *,
    existing_synthetic: pd.DataFrame | None = None,
    block_hours: int = 96,
    max_blocks: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Extend synthetic parquet coverage up to `through` if needed."""
    synthetic_start = pd.Timestamp(SYNTHETIC_HISTORY_START)
    through = pd.Timestamp(through)
    data_end = pd.to_datetime(hourly["ts"]).max()

    if through <= data_end:
        return (
            pd.DataFrame(columns=["ts", "power", "temp", "is_synthetic"]),
            hourly,
            {"blocks_run": 0, "hours_filled": 0, "synthetic_through": str(data_end)},
        )

    synth = existing_synthetic.copy() if existing_synthetic is not None else pd.DataFrame()
    if not synth.empty:
        synth["ts"] = pd.to_datetime(synth["ts"])
        covered = synth[synth["ts"] <= through]
        if len(covered) and covered["power"].notna().all() and covered["ts"].max() >= through:
            work = build_extended_frame(hourly, synth, through=through)
            return synth, work, {"blocks_run": 0, "hours_filled": 0, "synthetic_through": str(covered["ts"].max())}

    new_synth, work, meta = extend_synthetic_ttm(
        hourly,
        synthetic_start=synthetic_start,
        through=through,
        forecaster=forecaster,
        block_hours=block_hours,
        max_blocks=max_blocks,
        seed_synthetic=synth if not synth.empty else None,
    )
    merged = merge_synthetic_artifacts(synth if not synth.empty else None, new_synth)
    work = build_extended_frame(hourly, merged, through=through)
    meta["synthetic_through"] = str(merged["ts"].max()) if len(merged) else meta["synthetic_through"]
    return merged, work, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TTM 96h one-shot synthetic history builder.")
    parser.add_argument("--parquet", type=Path, default=SEOUL_CITY_HOURLY_PARQUET)
    parser.add_argument("--model-path", type=str, default="ibm-granite/granite-timeseries-ttm-r2")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--block-hours", type=int, default=96)
    parser.add_argument("--through-year", type=int, default=None, help="e.g. 2025, 2027")
    parser.add_argument("--through-date", type=str, default=None, help="YYYY-MM-DD[ HH:MM:SS]")
    parser.add_argument("--max-blocks", type=int, default=None, help="Smoke: cap number of blocks")
    parser.add_argument("--output-parquet", type=Path, default=TTM_SYNTHETIC_HISTORY_PARQUET)
    parser.add_argument("--output-json", type=Path, default=TTM_SYNTHETIC_HISTORY_JSON)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)

    if args.through_date:
        through = pd.Timestamp(args.through_date)
    elif args.through_year:
        through = pd.Timestamp(f"{args.through_year}-12-31 23:00:00")
    else:
        through = pd.Timestamp("2025-12-31 23:00:00")

    hourly = _load_hourly(args.parquet)
    train_df = hourly[pd.to_datetime(hourly["ts"]).dt.year <= 2023].copy()

    print(f"Loading TTM block forecaster ({args.model_path})...", flush=True)
    forecaster = TTMBlockForecaster(
        train_df,
        args.model_path,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        device=args.device,
    )

    existing = None
    if args.output_parquet.is_file():
        existing = pd.read_parquet(args.output_parquet)

    print(f"Building synthetic history {SYNTHETIC_HISTORY_START} → {through} ...", flush=True)
    synthetic, _work, meta = ensure_synthetic_through(
        hourly,
        through,
        forecaster,
        existing_synthetic=existing,
        block_hours=args.block_hours,
        max_blocks=args.max_blocks,
    )

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_parquet(args.output_parquet, index=False)
    meta["rows"] = len(synthetic)
    meta["output_parquet"] = str(args.output_parquet)
    args.output_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Done: {len(synthetic)} synthetic rows → {args.output_parquet}\n"
        f"      blocks_run={meta.get('blocks_run')} hours_filled={meta.get('hours_filled')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
