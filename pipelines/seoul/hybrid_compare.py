# Copyright contributors to the TSFM project
#
"""Compare naive / GRU-only / TTM zeroshot / hybrid on the same 96h holdout.

  uv run python -m pipelines.seoul.hybrid_compare
  uv run python -m pipelines.seoul.hybrid_report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.seoul.config import (
    ARTIFACTS_DIR,
    HYBRID_CHECKPOINT_SEOUL,
    HYBRID_HOLDOUT_FORECAST_PARQUET,
    MODEL_COMPARE_JSON,
    SEOUL_CITY_HOURLY_PARQUET,
)
from pipelines.seoul.hybrid_common import load_hourly, load_hourly_for_ttm
from pipelines.seoul.hybrid_eval import (
    eval_hybrid_checkpoint,
    eval_naive,
    eval_seasonal,
    eval_ttm_zeroshot,
    train_gru_only_holdout,
)
from pipelines.seoul.hybrid_model import FrozenTTMEncoder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="96h holdout model comparison (Seoul).")
    parser.add_argument("--parquet", type=Path, default=SEOUL_CITY_HOURLY_PARQUET)
    parser.add_argument("--checkpoint", type=Path, default=HYBRID_CHECKPOINT_SEOUL)
    parser.add_argument("--train-frac", type=float, default=0.85)
    parser.add_argument("--holdout-hours", type=int, default=96)
    parser.add_argument("--model-path", type=str, default="ibm-granite/granite-timeseries-ttm-r2")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=96)
    parser.add_argument("--gru-only-epochs", type=int, default=8)
    parser.add_argument("--skip-ttm-zeroshot", action="store_true", help="Skip slow TTM pipeline forecast")
    parser.add_argument("--skip-gru-only", action="store_true", help="Skip GRU-only retrain")
    parser.add_argument("--output-json", type=Path, default=MODEL_COMPARE_JSON)
    parser.add_argument("--output-parquet", type=Path, default=HYBRID_HOLDOUT_FORECAST_PARQUET)
    args = parser.parse_args(argv)

    if not args.parquet.is_file():
        raise SystemExit(f"Missing {args.parquet}. Run materialize first.")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Missing {args.checkpoint}. Run hybrid_train first.")

    import torch

    df = load_hourly(args.parquet)
    test_start = len(df) - args.holdout_hours
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"holdout: last {args.holdout_hours}h from index {test_start}")

    rows: dict[str, dict] = {}
    series: dict[str, list] = {"ts": [], "power_actual": []}

    naive = eval_naive(df, test_start, args.holdout_hours)
    rows["naive"] = {"mse": naive.mse, "rmse": naive.rmse, "label": "Naive (직전값, lag-1)"}
    series["power_pred_naive"] = naive.pred.tolist()

    s24 = eval_seasonal(df, test_start, args.holdout_hours, 24)
    rows["seasonal_24h"] = {"mse": s24.mse, "rmse": s24.rmse, "label": "Seasonal naive (24h)"}
    series["power_pred_seasonal_24h"] = s24.pred.tolist()

    s168 = eval_seasonal(df, test_start, args.holdout_hours, 168)
    rows["seasonal_168h"] = {"mse": s168.mse, "rmse": s168.rmse, "label": "Seasonal naive (168h)"}
    series["power_pred_seasonal_168h"] = s168.pred.tolist()

    if not args.skip_gru_only:
        print("Training GRU-only baseline...")
        gru = train_gru_only_holdout(
            df,
            args.train_frac,
            seq_len=168,
            hidden=64,
            layers=2,
            epochs=args.gru_only_epochs,
            test_start=test_start,
            holdout=args.holdout_hours,
            device=device,
        )
        rows["gru_only"] = {"mse": gru.mse, "rmse": gru.rmse, "label": "GRU only"}
        series["power_pred_gru_only"] = gru.pred.tolist()
    else:
        gru = None

    if not args.skip_ttm_zeroshot:
        print("Running TTM zeroshot 96h forecast...")
        ttm = eval_ttm_zeroshot(
            df,
            args.train_frac,
            args.context_length,
            args.prediction_length,
            args.model_path,
            test_start,
            args.holdout_hours,
        )
        rows["ttm_zeroshot"] = {"mse": ttm.mse, "rmse": ttm.rmse, "label": "TTM zeroshot (96h)"}
        series["power_pred_ttm"] = ttm.pred.tolist()
    else:
        ttm = None

    print("Evaluating hybrid checkpoint...")
    n_train = int(len(df) * args.train_frac)
    encoder = FrozenTTMEncoder(
        args.model_path,
        args.context_length,
        args.prediction_length,
        load_hourly_for_ttm(args.parquet).iloc[:n_train],
        device,
    )
    hybrid = eval_hybrid_checkpoint(
        df, args.checkpoint, encoder, test_start, args.holdout_hours, device
    )
    rows["hybrid"] = {"mse": hybrid.mse, "rmse": hybrid.rmse, "label": "TSFM+GRU hybrid"}
    series["power_pred_hybrid"] = hybrid.pred.tolist()

    ref = hybrid
    series["ts"] = [str(t) for t in ref.ts]
    series["power_actual"] = ref.actual.tolist()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "holdout_hours": args.holdout_hours,
                "test_start_index": test_start,
                "models": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out_df = pd.DataFrame(series)
    out_df.to_parquet(args.output_parquet, index=False)
    print(f"Wrote {args.output_json} and {args.output_parquet}")
    print("\nModel comparison (96h holdout, raw power):")
    for key in ["naive", "seasonal_24h", "seasonal_168h", "gru_only", "ttm_zeroshot", "hybrid"]:
        if key in rows:
            r = rows[key]
            print(f"  {r['label']:28s}  MSE={r['mse']:8.1f}  RMSE={r['rmse']:6.2f}")

    from pipelines.seoul.model_compare_plots import write_compare_plots_html

    plots_path = ARTIFACTS_DIR / "model_compare_plots.html"
    write_compare_plots_html(
        {"holdout_hours": args.holdout_hours, "models": rows},
        out_df,
        plots_path,
        include_ttm="ttm_zeroshot" in rows,
    )
    print(f"Wrote {plots_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
