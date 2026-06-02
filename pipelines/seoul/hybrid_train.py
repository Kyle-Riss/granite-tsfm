# Copyright contributors to the TSFM project
#
"""Train hybrid forecaster: frozen TTM backbone → GRU head.

  uv run python -m pipelines.seoul.materialize
  uv run python -m pipelines.seoul.hybrid_train --seoul-only --epochs 8
  uv run python -m pipelines.seoul.hybrid_report
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from pipelines.seoul.config import (
    ARTIFACTS_DIR,
    HYBRID_CHECKPOINT_SEOUL,
    HYBRID_EMBEDDINGS_NPZ,
    HYBRID_HOLDOUT_FORECAST_PARQUET,
    HYBRID_METRICS_SEOUL_JSON,
    REGIONAL_HOURLY_PARQUET,
    SEOUL_CITY_HOURLY_PARQUET,
)
from pipelines.seoul.hybrid_common import FeatureScaler, load_hourly, load_hourly_for_ttm
from pipelines.seoul.hybrid_model import FrozenTTMEncoder, HybridSeqDataset, HybridTSFMGRU


def _precompute_embeddings(
    encoder: FrozenTTMEncoder,
    full_df: pd.DataFrame,
    seq_len: int,
    start_index: int,
    n_windows: int,
) -> np.ndarray:
    """Embeddings for dataset windows starting at start_index .. start_index+n_windows-1."""
    ends = start_index + seq_len + np.arange(n_windows, dtype=np.int64)
    return encoder.encode_batch_indices(full_df, ends)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TSFM (frozen) + GRU head training.")
    parser.add_argument("--parquet", type=Path, default=REGIONAL_HOURLY_PARQUET)
    parser.add_argument("--seoul-only", action="store_true")
    parser.add_argument("--model-path", type=str, default="ibm-granite/granite-timeseries-ttm-r2")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--seq-len", type=int, default=168, help="GRU history length (hours)")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-frac", type=float, default=0.85)
    parser.add_argument("--holdout-hours", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-ttm", action="store_true", help="Ablation: GRU only (zero TTM embedding)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--max-precompute", type=int, default=None, help="Cap windows for TTM encode (debug)")
    parser.add_argument(
        "--rebuild-embeddings",
        action="store_true",
        help="Ignore cached TTM embeddings at artifacts/seoul/hybrid_ttm_embeddings.npz",
    )
    parser.add_argument("--embeddings-cache", type=Path, default=HYBRID_EMBEDDINGS_NPZ)
    args = parser.parse_args(argv)

    path = SEOUL_CITY_HOURLY_PARQUET if args.seoul_only else args.parquet
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run: python -m pipelines.seoul.materialize")

    torch.manual_seed(args.seed)
    df = load_hourly(path)
    n_train = int(len(df) * args.train_frac)
    train_df, val_df = df.iloc[:n_train].copy(), df.iloc[n_train:].copy()
    scaler = FeatureScaler.from_frame(train_df)

    print(f"rows={len(df)} train={len(train_df)} val={len(val_df)} seq_len={args.seq_len}")

    if args.dry_run:
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ttm_emb_dim = 16
    encoder = None

    if not args.skip_ttm:
        print("Loading frozen TTM and precomputing backbone embeddings (may take a few minutes)...")
        ttm_train_df = load_hourly_for_ttm(path).iloc[:n_train]
        encoder = FrozenTTMEncoder(
            args.model_path,
            args.context_length,
            args.prediction_length,
            ttm_train_df,
            device,
        )
        ttm_emb_dim = int(encoder.encode_window(ttm_train_df.tail(args.context_length)).numel())

        n_tr = len(train_df) - args.seq_len
        n_va = len(val_df) - args.seq_len
        if args.max_precompute:
            n_tr = min(n_tr, args.max_precompute)
            n_va = min(n_va, args.max_precompute // 4 + 1)

        cache_ok = (
            args.embeddings_cache.is_file()
            and not args.rebuild_embeddings
            and not args.max_precompute
        )
        if cache_ok:
            cached = np.load(args.embeddings_cache)
            train_embs = cached["train"]
            val_embs = cached["val"]
            print(f"Loaded cached embeddings from {args.embeddings_cache}")
        else:
            print(f"Precomputing train embeddings ({n_tr} windows)...")
            train_embs = _precompute_embeddings(encoder, df, args.seq_len, 0, n_tr)
            print(f"Precomputing val embeddings ({n_va} windows)...")
            val_embs = _precompute_embeddings(encoder, df, args.seq_len, n_train, n_va)
            if not args.max_precompute:
                ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.embeddings_cache, train=train_embs, val=val_embs)
                print(f"Cached embeddings → {args.embeddings_cache}")
        print(f"TTM embedding dim={ttm_emb_dim}  train_windows={len(train_embs)} val_windows={len(val_embs)}")
    else:
        train_embs = np.zeros((max(len(train_df) - args.seq_len, 0), ttm_emb_dim), dtype=np.float32)
        val_embs = np.zeros((max(len(val_df) - args.seq_len, 0), ttm_emb_dim), dtype=np.float32)

    train_ds = HybridSeqDataset(train_df, args.seq_len, scaler, train_embs)
    val_ds = HybridSeqDataset(val_df, args.seq_len, scaler, val_embs)
    if args.max_precompute:
        train_ds = Subset(train_ds, range(min(len(train_ds), len(train_embs))))
        val_ds = Subset(val_ds, range(min(len(val_ds), len(val_embs))))
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("Not enough rows; shorten --seq-len.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = HybridTSFMGRU(
        input_dim=HybridSeqDataset.BASE_DIM,
        ttm_emb_dim=ttm_emb_dim,
        hidden=args.hidden,
        layers=args.layers,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    last_train_mse = last_val_mse = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x, emb, y in train_loader:
            x, emb, y = x.to(device), emb.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x, emb)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, emb, y in val_loader:
                x, emb, y = x.to(device), emb.to(device), y.to(device)
                pred = model(x, emb)
                val_loss += loss_fn(pred, y).item() * x.size(0)
        val_loss /= len(val_ds)
        last_train_mse, last_val_mse = train_loss, val_loss
        print(f"epoch {epoch + 1}/{args.epochs}  train_mse={train_loss:.6f}  val_mse={val_loss:.6f}")

    holdout = args.holdout_hours
    test_start = len(df) - holdout
    holdout_preds, holdout_actual, holdout_ts = [], [], []
    model.eval()
    with torch.no_grad():
        for g in range(test_start, min(test_start + holdout, len(df))):
            i = g - args.seq_len
            if i < 0:
                continue
            win_df = df.iloc[i : g + 1]
            tmp_ds = HybridSeqDataset(
                win_df,
                args.seq_len,
                scaler,
                np.zeros((1, ttm_emb_dim), dtype=np.float32),
            )
            x, _, _ = tmp_ds[0]
            if not args.skip_ttm:
                emb = encoder.encode_batch_indices(df, np.array([g], dtype=np.int64))[0]
            else:
                emb = np.zeros(ttm_emb_dim, dtype=np.float32)
            pred_norm = model(
                x.unsqueeze(0).to(device),
                torch.from_numpy(emb).unsqueeze(0).to(device),
            )
            pred_raw = float(pred_norm.item()) * scaler.power_std + scaler.power_mean
            holdout_preds.append(pred_raw)
            holdout_actual.append(float(df["power"].iloc[g]))
            holdout_ts.append(df["ts"].iloc[g])

    holdout_mse = float(np.mean((np.array(holdout_preds) - np.array(holdout_actual)) ** 2))
    print(f"holdout {holdout}h MSE (raw power) = {holdout_mse:.2f}")

    if not args.no_save and args.seoul_only:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        ckpt = HYBRID_CHECKPOINT_SEOUL
        torch.save(
            {
                "state_dict": model.state_dict(),
                "scaler": asdict(scaler),
                "hybrid_config": {
                    "seq_len": args.seq_len,
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "ttm_emb_dim": ttm_emb_dim,
                    "base_dim": HybridSeqDataset.BASE_DIM,
                    "skip_ttm": args.skip_ttm,
                },
            },
            ckpt,
        )
        metrics = {
            "train_mse_final": last_train_mse,
            "val_mse_final": last_val_mse,
            "holdout_mse_raw": holdout_mse,
            "epochs": args.epochs,
            "seq_len": args.seq_len,
            "skip_ttm": args.skip_ttm,
            "parquet": str(path),
        }
        HYBRID_METRICS_SEOUL_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        pd.DataFrame(
            {
                "ts": holdout_ts,
                "power_actual": holdout_actual,
                "power_pred_hybrid": holdout_preds,
            }
        ).to_parquet(HYBRID_HOLDOUT_FORECAST_PARQUET, index=False)
        print(f"Saved {ckpt}, {HYBRID_METRICS_SEOUL_JSON}, {HYBRID_HOLDOUT_FORECAST_PARQUET}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
