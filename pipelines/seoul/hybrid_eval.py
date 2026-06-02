# Copyright contributors to the TSFM project
#
"""Shared 96h holdout evaluation for naive / GRU-only / TTM / hybrid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pipelines.seoul.hybrid_common import FeatureScaler, load_hourly, load_hourly_for_ttm
from pipelines.seoul.hybrid_model import FrozenTTMEncoder, HybridSeqDataset, HybridTSFMGRU


@dataclass
class HoldoutResult:
    ts: list
    actual: np.ndarray
    pred: np.ndarray
    mse: float
    rmse: float


def _metrics(pred: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    err = pred - actual
    mse = float(np.mean(err**2))
    return mse, float(np.sqrt(mse))


def eval_seasonal(
    df: pd.DataFrame,
    test_start: int,
    holdout: int,
    lag_hours: int,
) -> HoldoutResult:
    """Seasonal naive: power[t] = power[t - lag_hours] (from full series)."""
    actual, pred, ts = [], [], []
    for g in range(test_start, min(test_start + holdout, len(df))):
        ref = g - lag_hours
        if ref < 0:
            continue
        actual.append(float(df["power"].iloc[g]))
        pred.append(float(df["power"].iloc[ref]))
        ts.append(df["ts"].iloc[g])
    a, p = np.array(actual), np.array(pred)
    mse, rmse = _metrics(p, a)
    return HoldoutResult(ts=ts, actual=a, pred=p, mse=mse, rmse=rmse)


def eval_naive(df: pd.DataFrame, test_start: int, holdout: int) -> HoldoutResult:
    actual, pred, ts = [], [], []
    for g in range(test_start, min(test_start + holdout, len(df))):
        if g < 1:
            continue
        actual.append(float(df["power"].iloc[g]))
        pred.append(float(df["power"].iloc[g - 1]))
        ts.append(df["ts"].iloc[g])
    a, p = np.array(actual), np.array(pred)
    mse, rmse = _metrics(p, a)
    return HoldoutResult(ts=ts, actual=a, pred=p, mse=mse, rmse=rmse)


def eval_hybrid_checkpoint(
    df: pd.DataFrame,
    checkpoint: Path,
    encoder: FrozenTTMEncoder | None,
    test_start: int,
    holdout: int,
    device: torch.device,
) -> HoldoutResult:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    scaler = FeatureScaler(**ckpt["scaler"])
    cfg = ckpt["hybrid_config"]
    skip_ttm = bool(cfg.get("skip_ttm", False))
    ttm_emb_dim = int(cfg["ttm_emb_dim"])

    model = HybridTSFMGRU(
        input_dim=int(cfg["base_dim"]),
        ttm_emb_dim=ttm_emb_dim,
        hidden=int(cfg["hidden"]),
        layers=int(cfg["layers"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    seq_len = int(cfg["seq_len"])
    actual, pred, ts = [], [], []
    with torch.no_grad():
        for g in range(test_start, min(test_start + holdout, len(df))):
            i = g - seq_len
            if i < 0:
                continue
            win_df = df.iloc[i : g + 1]
            tmp_ds = HybridSeqDataset(
                win_df, seq_len, scaler, np.zeros((1, ttm_emb_dim), dtype=np.float32)
            )
            x, _, _ = tmp_ds[0]
            if not skip_ttm and encoder is not None:
                emb = encoder.encode_batch_indices(df, np.array([g], dtype=np.int64))[0]
            else:
                emb = np.zeros(ttm_emb_dim, dtype=np.float32)
            p_norm = model(
                x.unsqueeze(0).to(device),
                torch.from_numpy(emb).unsqueeze(0).to(device),
            )
            pred.append(float(p_norm.item()) * scaler.power_std + scaler.power_mean)
            actual.append(float(df["power"].iloc[g]))
            ts.append(df["ts"].iloc[g])

    a, p = np.array(actual), np.array(pred)
    mse, rmse = _metrics(p, a)
    return HoldoutResult(ts=ts, actual=a, pred=p, mse=mse, rmse=rmse)


def train_gru_only_holdout(
    df: pd.DataFrame,
    train_frac: float,
    seq_len: int,
    hidden: int,
    layers: int,
    epochs: int,
    test_start: int,
    holdout: int,
    device: torch.device,
) -> HoldoutResult:
    """GRU without TTM embeddings (ablation baseline)."""
    from torch.utils.data import DataLoader

    n_train = int(len(df) * train_frac)
    train_df = df.iloc[:n_train].copy()
    scaler = FeatureScaler.from_frame(train_df)
    train_ds = HybridSeqDataset(train_df, seq_len, scaler, np.zeros((max(len(train_df) - seq_len, 0), 1)))
    if len(train_ds) == 0:
        raise ValueError("train_ds empty")
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    model = HybridTSFMGRU(
        input_dim=HybridSeqDataset.BASE_DIM,
        ttm_emb_dim=1,
        hidden=hidden,
        layers=layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    zero_emb = torch.zeros(1, 1, device=device)

    for _ in range(epochs):
        model.train()
        for x, _, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x, zero_emb.expand(x.size(0), -1)), y).backward()
            opt.step()

    model.eval()
    actual, pred, ts = [], [], []
    with torch.no_grad():
        for g in range(test_start, min(test_start + holdout, len(df))):
            i = g - seq_len
            if i < 0:
                continue
            win_df = df.iloc[i : g + 1]
            tmp_ds = HybridSeqDataset(win_df, seq_len, scaler, np.zeros((1, 1), dtype=np.float32))
            x, _, _ = tmp_ds[0]
            p_norm = model(x.unsqueeze(0).to(device), zero_emb)
            pred.append(float(p_norm.item()) * scaler.power_std + scaler.power_mean)
            actual.append(float(df["power"].iloc[g]))
            ts.append(df["ts"].iloc[g])

    a, p = np.array(actual), np.array(pred)
    mse, rmse = _metrics(p, a)
    return HoldoutResult(ts=ts, actual=a, pred=p, mse=mse, rmse=rmse)


def train_gru_only_model(
    df: pd.DataFrame,
    train_frac: float,
    seq_len: int,
    hidden: int,
    layers: int,
    epochs: int,
    device: torch.device,
) -> tuple[HybridTSFMGRU, FeatureScaler, int]:
    """Train GRU-only (zero TTM emb); returns model ready for eval_gru_only_model."""
    from torch.utils.data import DataLoader

    n_train = int(len(df) * train_frac)
    train_df = df.iloc[:n_train].copy()
    scaler = FeatureScaler.from_frame(train_df)
    ds = HybridSeqDataset(train_df, seq_len, scaler, np.zeros((max(len(train_df) - seq_len, 0), 1)))
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    model = HybridTSFMGRU(
        input_dim=HybridSeqDataset.BASE_DIM,
        ttm_emb_dim=1,
        hidden=hidden,
        layers=layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    zero = torch.zeros(1, 1, device=device)
    for _ in range(epochs):
        model.train()
        for x, _, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            nn.MSELoss()(model(x, zero.expand(x.size(0), -1)), y).backward()
            opt.step()
    model.eval()
    return model, scaler, seq_len


def eval_gru_only_model(
    model: HybridTSFMGRU,
    scaler: FeatureScaler,
    seq_len: int,
    df: pd.DataFrame,
    test_start: int,
    holdout: int,
    device: torch.device,
) -> HoldoutResult:
    zero = torch.zeros(1, 1, device=device)
    actual, pred, ts = [], [], []
    with torch.no_grad():
        for g in range(test_start, min(test_start + holdout, len(df))):
            i = g - seq_len
            if i < 0:
                continue
            win_df = df.iloc[i : g + 1]
            tmp_ds = HybridSeqDataset(win_df, seq_len, scaler, np.zeros((1, 1), dtype=np.float32))
            x, _, _ = tmp_ds[0]
            p_norm = model(x.unsqueeze(0).to(device), zero)
            pred.append(float(p_norm.item()) * scaler.power_std + scaler.power_mean)
            actual.append(float(df["power"].iloc[g]))
            ts.append(df["ts"].iloc[g])
    a, p = np.array(actual), np.array(pred)
    mse, rmse = _metrics(p, a)
    return HoldoutResult(ts=ts, actual=a, pred=p, mse=mse, rmse=rmse)


def eval_ttm_zeroshot(
    df: pd.DataFrame,
    train_frac: float,
    context_length: int,
    prediction_length: int,
    model_path: str,
    test_start: int,
    holdout: int,
) -> HoldoutResult:
    """Single-shot TTM 96h forecast aligned to holdout timestamps."""
    from tsfm_public import TimeSeriesForecastingPipeline, TimeSeriesPreprocessor, get_model

    ttm_df = load_hourly_for_ttm(df.iloc[: int(len(df) * train_frac)].copy())
    hist = load_hourly_for_ttm(df.iloc[max(0, test_start - context_length) : test_start].copy())

    model = get_model(
        model_path,
        context_length=context_length,
        prediction_length=prediction_length,
        freq="h",
    )
    tsp = TimeSeriesPreprocessor(
        timestamp_column="timestamp",
        id_columns=[],
        target_columns=["power"],
        observable_columns=["temp"],
        context_length=context_length,
        prediction_length=prediction_length,
        freq="h",
        scaling=True,
    )
    tsp.train(ttm_df)
    pipe = TimeSeriesForecastingPipeline(
        model=model,
        timestamp_column="timestamp",
        id_columns=[],
        target_columns=["power"],
        freq="h",
        feature_extractor=tsp,
        explode_forecasts=True,
        inverse_scale_outputs=True,
        device="cpu",
    )
    fc = pipe(hist)
    ts_col = "timestamp" if "timestamp" in fc.columns else "ts"
    power_col = "power" if "power" in fc.columns else fc.columns[-1]
    fc = fc.sort_values(ts_col).head(holdout)

    hold = df.iloc[test_start : test_start + holdout]
    actual = hold["power"].to_numpy(dtype=np.float64)
    ts = hold["ts"].tolist()
    n = min(len(actual), len(fc))
    pred = fc[power_col].iloc[:n].to_numpy(dtype=np.float64)
    actual = actual[:n]
    ts = ts[:n]
    mse, rmse = _metrics(pred, actual)
    return HoldoutResult(ts=ts, actual=actual, pred=pred, mse=mse, rmse=rmse)


class TTMBlockForecaster:
    """Reuse TTM pipeline; one-shot forecast from context ending before test_start."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        model_path: str,
        context_length: int,
        prediction_length: int,
        device: str = "cpu",
    ):
        from tsfm_public import TimeSeriesForecastingPipeline, TimeSeriesPreprocessor, get_model

        self.context_length = context_length
        self.prediction_length = prediction_length
        ttm_train = load_hourly_for_ttm(train_df)

        self.model = get_model(
            model_path,
            context_length=context_length,
            prediction_length=prediction_length,
            freq="h",
        )
        self.tsp = TimeSeriesPreprocessor(
            timestamp_column="timestamp",
            id_columns=[],
            target_columns=["power"],
            observable_columns=["temp"],
            context_length=context_length,
            prediction_length=prediction_length,
            freq="h",
            scaling=True,
        )
        self.tsp.train(ttm_train)
        self.pipe = TimeSeriesForecastingPipeline(
            model=self.model,
            timestamp_column="timestamp",
            id_columns=[],
            target_columns=["power"],
            freq="h",
            feature_extractor=self.tsp,
            explode_forecasts=True,
            inverse_scale_outputs=True,
            device=device,
        )

    def predict(self, df: pd.DataFrame, test_start: int, horizon: int) -> np.ndarray:
        start = max(0, test_start - self.context_length)
        hist = load_hourly_for_ttm(df.iloc[start:test_start].copy())
        fc = self.pipe(hist)
        power_col = "power" if "power" in fc.columns else fc.columns[-1]
        return fc.sort_values("timestamp" if "timestamp" in fc.columns else "ts")[power_col].iloc[:horizon].to_numpy(
            dtype=np.float64
        )


def load_hybrid_for_eval(
    checkpoint: Path,
    device: torch.device,
) -> tuple[HybridTSFMGRU, FeatureScaler, dict, int, bool]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["hybrid_config"]
    scaler = FeatureScaler(**ckpt["scaler"])
    ttm_emb_dim = int(cfg["ttm_emb_dim"])
    model = HybridTSFMGRU(
        input_dim=int(cfg["base_dim"]),
        ttm_emb_dim=ttm_emb_dim,
        hidden=int(cfg["hidden"]),
        layers=int(cfg["layers"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, scaler, cfg, ttm_emb_dim, bool(cfg.get("skip_ttm", False))


def _hybrid_one_step(
    model: HybridTSFMGRU,
    scaler: FeatureScaler,
    encoder: FrozenTTMEncoder | None,
    df: pd.DataFrame,
    g: int,
    seq_len: int,
    ttm_emb_dim: int,
    device: torch.device,
    skip_ttm: bool,
) -> float:
    i = g - seq_len
    if i < 0:
        raise ValueError(f"Need {seq_len} rows before index {g}")
    win_df = df.iloc[i : g + 1]
    tmp_ds = HybridSeqDataset(
        win_df, seq_len, scaler, np.zeros((1, ttm_emb_dim), dtype=np.float32)
    )
    x, _, _ = tmp_ds[0]
    if not skip_ttm and encoder is not None:
        emb = encoder.encode_batch_indices(df, np.array([g], dtype=np.int64))[0]
    else:
        emb = np.zeros(ttm_emb_dim, dtype=np.float32)
    with torch.no_grad():
        p_norm = model(
            x.unsqueeze(0).to(device),
            torch.from_numpy(emb).unsqueeze(0).to(device),
        )
    return float(p_norm.item()) * scaler.power_std + scaler.power_mean


def eval_hybrid_segment(
    df: pd.DataFrame,
    checkpoint: Path,
    encoder: FrozenTTMEncoder | None,
    test_start: int,
    holdout: int,
    device: torch.device,
    *,
    roll_forward: bool,
    work: pd.DataFrame | None = None,
) -> tuple[HoldoutResult, pd.DataFrame]:
    """Predict holdout hours; roll_forward writes preds into work['power']."""
    model, scaler, cfg, ttm_emb_dim, skip_ttm = load_hybrid_for_eval(checkpoint, device)
    seq_len = int(cfg["seq_len"])
    if work is None:
        work = df.copy()
    actual, pred, ts = [], [], []
    end = min(test_start + holdout, len(df))
    for g in range(test_start, end):
        if g < seq_len:
            continue
        p = _hybrid_one_step(
            model, scaler, encoder, work if roll_forward else df, g, seq_len, ttm_emb_dim, device, skip_ttm
        )
        if roll_forward:
            work.loc[g, "power"] = p
        actual.append(float(df["power"].iloc[g]))
        pred.append(p)
        ts.append(df["ts"].iloc[g])
    a, p = np.array(actual), np.array(pred)
    mse, rmse = _metrics(p, a)
    return HoldoutResult(ts=ts, actual=a, pred=p, mse=mse, rmse=rmse), work


def eval_gru_segment(
    model: HybridTSFMGRU,
    scaler: FeatureScaler,
    seq_len: int,
    df: pd.DataFrame,
    test_start: int,
    holdout: int,
    device: torch.device,
    *,
    roll_forward: bool,
    work: pd.DataFrame | None = None,
) -> tuple[HoldoutResult, pd.DataFrame]:
    zero = torch.zeros(1, 1, device=device)
    if work is None:
        work = df.copy()
    actual, pred, ts = [], [], []
    end = min(test_start + holdout, len(df))
    with torch.no_grad():
        for g in range(test_start, end):
            i = g - seq_len
            if i < 0:
                continue
            src = work if roll_forward else df
            win_df = src.iloc[i : g + 1]
            tmp_ds = HybridSeqDataset(win_df, seq_len, scaler, np.zeros((1, 1), dtype=np.float32))
            x, _, _ = tmp_ds[0]
            p_norm = model(x.unsqueeze(0).to(device), zero)
            p = float(p_norm.item()) * scaler.power_std + scaler.power_mean
            if roll_forward:
                work.loc[g, "power"] = p
            actual.append(float(df["power"].iloc[g]))
            pred.append(p)
            ts.append(df["ts"].iloc[g])
    a, p = np.array(actual), np.array(pred)
    mse, rmse = _metrics(p, a)
    return HoldoutResult(ts=ts, actual=a, pred=p, mse=mse, rmse=rmse), work
