# Copyright contributors to the TSFM project
#
"""Frozen TTM backbone features + trainable GRU head for power forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pipelines.seoul.hybrid_common import REGION_LABELS, hour_features, region_one_hot


class FrozenTTMEncoder:
    """Loads Granite TTM, keeps backbone frozen, returns pooled representation per context window."""

    def __init__(
        self,
        model_path: str,
        context_length: int,
        prediction_length: int,
        train_df: pd.DataFrame,
        device: torch.device,
    ):
        from tsfm_public import TimeSeriesPreprocessor, get_model

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.device = device

        self.model = get_model(
            model_path,
            context_length=context_length,
            prediction_length=prediction_length,
            freq="h",
        ).to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.preprocessor = TimeSeriesPreprocessor(
            timestamp_column="timestamp",
            id_columns=[],
            target_columns=["power"],
            observable_columns=["temp"],
            context_length=context_length,
            prediction_length=prediction_length,
            freq="h",
            scaling=True,
        )
        self.preprocessor.train(train_df)
        self._frequency_token = self.preprocessor.get_frequency_token(self.preprocessor.freq)

    def _prepare_hist(self, hist_df: pd.DataFrame) -> pd.DataFrame:
        """Left-pad context, append dummy future rows so ForecastDFDataset can emit one window."""
        if "timestamp" not in hist_df.columns and "ts" in hist_df.columns:
            hist_df = hist_df.rename(columns={"ts": "timestamp"})

        if len(hist_df) < self.context_length:
            pad = self.context_length - len(hist_df)
            first = hist_df.iloc[[0]].copy()
            hist_df = pd.concat([pd.concat([first] * pad, ignore_index=True), hist_df], ignore_index=True)
        hist_df = hist_df.iloc[-self.context_length :].copy()

        need = self.context_length + self.prediction_length
        if len(hist_df) < need:
            last = hist_df.iloc[[-1]].copy()
            hist_df = pd.concat([hist_df] + [last] * (need - len(hist_df)), ignore_index=True)
        return hist_df

    @torch.no_grad()
    def encode_window(self, hist_df: pd.DataFrame) -> torch.Tensor:
        """hist_df: rows with timestamp, power, temp (length <= context_length; left-padded)."""
        from tsfm_public.toolkit.dataset import ForecastDFDataset

        hist_df = self._prepare_hist(hist_df)
        prep = self.preprocessor.preprocess(hist_df)

        ds = ForecastDFDataset(
            prep,
            timestamp_column="timestamp",
            id_columns=[],
            target_columns=["power"],
            observable_columns=["temp"],
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            frequency_token=self._frequency_token,
        )
        if len(ds) == 0:
            raise ValueError(
                f"ForecastDFDataset empty (rows={len(prep)}); need at least "
                f"{self.context_length + self.prediction_length}"
            )
        batch = ds[0]
        past_values = batch["past_values"].unsqueeze(0).to(self.device)
        freq_token = batch.get("freq_token")
        if freq_token is not None:
            freq_token = freq_token.unsqueeze(0).to(self.device)
        out = self.model(
            past_values=past_values,
            freq_token=freq_token,
            return_loss=False,
            return_dict=True,
        )
        hidden = out.backbone_hidden_state  # [1, nvars, patches, d_model]
        pooled = hidden.mean(dim=(1, 2)).squeeze(0)
        return pooled.cpu()

    @torch.no_grad()
    def encode_batch_indices(
        self,
        full_df: pd.DataFrame,
        end_indices: np.ndarray,
        log_every: int = 250,
    ) -> np.ndarray:
        """Precompute embeddings for each window ending at end_indices (exclusive end for next-step)."""
        ttm_df = full_df.rename(columns={"ts": "timestamp"}) if "ts" in full_df.columns else full_df
        embs = []
        n = len(end_indices)
        for j, end in enumerate(end_indices):
            if log_every and n >= log_every and j > 0 and j % log_every == 0:
                print(f"  TTM embeddings {j}/{n} ({100 * j / n:.0f}%)", flush=True)
            start = max(0, int(end) - self.context_length)
            hist = ttm_df.iloc[start:int(end)].copy()
            embs.append(self.encode_window(hist).numpy())
        if n >= log_every:
            print(f"  TTM embeddings {n}/{n} (100%)", flush=True)
        return np.stack(embs, axis=0).astype(np.float32)


class HybridSeqDataset(torch.utils.data.Dataset):
    """Sliding window: GRU inputs + precomputed or on-the-fly TTM embedding."""

    BASE_DIM = 4 + len(REGION_LABELS)  # power, temp, sin_h, cos_h, region one-hot

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        scaler,
        ttm_embeddings: np.ndarray | None,
        region: str | None = None,
    ):
        self.seq_len = seq_len
        self.scaler = scaler
        self.ttm_embeddings = ttm_embeddings
        self.region_oh = region_one_hot(region if region else (df["region"].iloc[0] if "region" in df.columns else None))

        self.power = ((df["power"] - scaler.power_mean) / scaler.power_std).to_numpy(dtype=np.float32)
        self.temp = ((df["temp"] - scaler.temp_mean) / scaler.temp_std).to_numpy(dtype=np.float32)
        h = df["hour"].to_numpy(dtype=np.float32)
        self.sin_h, self.cos_h = hour_features(h)
        self.n = len(df) - seq_len

    def __len__(self) -> int:
        return max(self.n, 0)

    def __getitem__(self, i: int):
        sl = slice(i, i + self.seq_len)
        base = np.stack(
            [self.power[sl], self.temp[sl], self.sin_h[sl], self.cos_h[sl]],
            axis=-1,
        )
        region_block = np.tile(self.region_oh, (self.seq_len, 1))
        x = np.concatenate([base, region_block], axis=-1).astype(np.float32)
        y = self.power[i + self.seq_len]
        if self.ttm_embeddings is not None:
            emb = self.ttm_embeddings[i]
        else:
            emb = np.zeros(1, dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(emb), torch.tensor(y, dtype=torch.float32)


class HybridTSFMGRU(nn.Module):
    def __init__(self, input_dim: int, ttm_emb_dim: int, hidden: int, layers: int):
        super().__init__()
        self.gru = nn.GRU(input_dim + ttm_emb_dim, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, ttm_emb: torch.Tensor) -> torch.Tensor:
        ttm_exp = ttm_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        inp = torch.cat([x, ttm_exp], dim=-1)
        out, _ = self.gru(inp)
        return self.head(out[:, -1, :]).squeeze(-1)
