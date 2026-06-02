# Copyright contributors to the TSFM project
#
"""Train TSFM+GRU hybrid on data through a calendar year (fair backtest checkpoints)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from pipelines.seoul.hybrid_common import FeatureScaler, load_hourly, load_hourly_for_ttm
from pipelines.seoul.hybrid_model import FrozenTTMEncoder, HybridSeqDataset, HybridTSFMGRU


def precompute_embeddings(
    encoder: FrozenTTMEncoder,
    full_df: pd.DataFrame,
    seq_len: int,
    start_index: int,
    n_windows: int,
) -> np.ndarray:
    ends = start_index + seq_len + np.arange(n_windows, dtype=np.int64)
    return encoder.encode_batch_indices(full_df, ends)


def checkpoint_path_for_context(context_through_year: int, artifacts_dir: Path) -> Path:
    return artifacts_dir / f"hybrid_seoul_through_{context_through_year}.pt"


def train_hybrid_through_year(
    df: pd.DataFrame,
    context_through_year: int,
    checkpoint_out: Path,
    *,
    parquet_for_ttm: Path | None = None,
    model_path: str = "ibm-granite/granite-timeseries-ttm-r2",
    context_length: int = 512,
    prediction_length: int = 96,
    seq_len: int = 168,
    hidden: int = 64,
    layers: int = 2,
    epochs: int = 8,
    batch_size: int = 32,
    lr: float = 1e-3,
    train_frac: float = 0.85,
    seed: int = 42,
    skip_ttm: bool = False,
    max_precompute: int | None = None,
    device: torch.device | None = None,
) -> dict:
    """Train GRU head (and use TTM frozen) on rows with ts <= context_through_year-12-31 23:00."""
    torch.manual_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = df.sort_values("ts").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"])
    ctx_end = pd.Timestamp(f"{context_through_year}-12-31 23:00:00")
    df = df[df["ts"] <= ctx_end].copy().reset_index(drop=True)
    if len(df) < seq_len + 200:
        raise ValueError(f"Too few rows through {ctx_end}: {len(df)}")

    n_train = int(len(df) * train_frac)
    train_df, val_df = df.iloc[:n_train].copy(), df.iloc[n_train:].copy()
    scaler = FeatureScaler.from_frame(train_df)

    ttm_emb_dim = 16
    encoder = None

    if not skip_ttm:
        _ = parquet_for_ttm  # reserved; training rows come from filtered df
        ttm_train_df = load_hourly_for_ttm(train_df)
        print(
            f"  [train] TTM encoder · context≤{context_through_year} · "
            f"rows={len(df)} train={len(train_df)} val={len(val_df)}",
            flush=True,
        )
        encoder = FrozenTTMEncoder(
            model_path,
            context_length,
            prediction_length,
            ttm_train_df,
            device,
        )
        ttm_emb_dim = int(encoder.encode_window(ttm_train_df.tail(context_length)).numel())

        n_tr = len(train_df) - seq_len
        n_va = len(val_df) - seq_len
        if max_precompute:
            n_tr = min(n_tr, max_precompute)
            n_va = min(n_va, max(max_precompute // 4, 1))

        print(f"  [train] TTM embeddings train={n_tr} val={n_va} …", flush=True)
        train_embs = precompute_embeddings(encoder, df, seq_len, 0, n_tr)
        val_embs = precompute_embeddings(encoder, df, seq_len, n_train, n_va)
    else:
        train_embs = np.zeros((max(len(train_df) - seq_len, 0), ttm_emb_dim), dtype=np.float32)
        val_embs = np.zeros((max(len(val_df) - seq_len, 0), ttm_emb_dim), dtype=np.float32)

    train_ds = HybridSeqDataset(train_df, seq_len, scaler, train_embs)
    val_ds = HybridSeqDataset(val_df, seq_len, scaler, val_embs)
    if max_precompute:
        train_ds = Subset(train_ds, range(min(len(train_ds), len(train_embs))))
        val_ds = Subset(val_ds, range(min(len(val_ds), len(val_embs))))
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("train/val dataset empty; add history or shorten --seq-len")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = HybridTSFMGRU(
        input_dim=HybridSeqDataset.BASE_DIM,
        ttm_emb_dim=ttm_emb_dim,
        hidden=hidden,
        layers=layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    last_train_mse = last_val_mse = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, emb, y in train_loader:
            x, emb, y = x.float().to(device), emb.float().to(device), y.float().to(device)
            opt.zero_grad()
            loss_fn(model(x, emb), y).backward()
            opt.step()
            train_loss += loss_fn(model(x, emb), y).item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, emb, y in val_loader:
                x, emb, y = x.float().to(device), emb.float().to(device), y.float().to(device)
                val_loss += loss_fn(model(x, emb), y).item() * x.size(0)
        val_loss /= len(val_ds)
        last_train_mse, last_val_mse = train_loss, val_loss
        print(
            f"  [train] epoch {epoch + 1}/{epochs}  train_mse={train_loss:.6f}  val_mse={val_loss:.6f}",
            flush=True,
        )

    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "scaler": asdict(scaler),
            "hybrid_config": {
                "seq_len": seq_len,
                "hidden": hidden,
                "layers": layers,
                "ttm_emb_dim": ttm_emb_dim,
                "base_dim": HybridSeqDataset.BASE_DIM,
                "skip_ttm": skip_ttm,
            },
            "train_meta": {
                "context_through_year": context_through_year,
                "context_end": str(ctx_end),
                "n_rows": len(df),
                "n_train": n_train,
            },
        },
        checkpoint_out,
    )
    print(f"  [train] saved {checkpoint_out}", flush=True)

    return {
        "checkpoint": str(checkpoint_out),
        "context_through_year": context_through_year,
        "train_mse_final": last_train_mse,
        "val_mse_final": last_val_mse,
        "n_rows": len(df),
    }
