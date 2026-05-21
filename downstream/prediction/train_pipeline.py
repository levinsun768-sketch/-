from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.prediction.data_process_utils import (
    BASIC_FLOW_FIELDS,
    BASIC_PRICE_FIELDS,
    prepare_dataset,
)
from downstream.prediction.model_registry import get_model
from downstream.prediction.predict_config import PredictConfig

PRICE_EPS = 1e-8
FLOW_EPS = 1e-8


def rank_ic_loss(pred, target):
    if pred.size(0) < 2:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    target_rank = torch.argsort(torch.argsort(target)).float()
    pred_centered = pred - pred.mean()
    target_centered = target_rank - target_rank.mean()
    cov = (pred_centered * target_centered).sum()
    pred_std = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
    target_std = torch.sqrt((target_centered ** 2).sum() + 1e-8)
    return -(cov / (pred_std * target_std))


def normalize_basic_ohlcv_window(window: np.ndarray, feature_index: dict) -> Optional[np.ndarray]:
    """对单个 20 日窗口做样本内时序标准化。"""
    normalized = window.astype(np.float32, copy=True)

    for field in BASIC_PRICE_FIELDS:
        idx = feature_index.get(field)
        if idx is None:
            continue
        series = normalized[:, idx]
        if not np.all(np.isfinite(series)) or np.any(series <= PRICE_EPS):
            return None
        latest_value = float(series[-1])
        if not np.isfinite(latest_value) or latest_value <= PRICE_EPS:
            return None
        normalized[:, idx] = np.log(series / latest_value)

    for field in BASIC_FLOW_FIELDS:
        idx = feature_index.get(field)
        if idx is None:
            continue
        series = normalized[:, idx]
        if not np.all(np.isfinite(series)):
            return None
        scale = float(series.mean())
        if not np.isfinite(scale) or scale <= FLOW_EPS:
            return None
        normalized[:, idx] = series / scale

    return normalized


def run_training(config: PredictConfig, exp_save_dir: str) -> pd.DataFrame:
    """3-seed ensemble 训练并输出全量 alpha。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(exp_save_dir, exist_ok=True)

    seeds = [42, 2024, 777]
    all_alphas = []

    def set_seed(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def train_one_seed(seed):
        set_seed(seed)
        print("\n==============================")
        print(f"Training seed = {seed}")
        print("==============================")

        df_merged, feature_cols = prepare_dataset(config)

        print("\n=> 构建每日全截面 Batch...")
        date_to_x = defaultdict(list)
        date_to_y = defaultdict(list)
        date_to_stock = defaultdict(list)
        feature_index = {name: idx for idx, name in enumerate(feature_cols)}

        grouped = df_merged.groupby(level="instrument", sort=False)
        for stock, group_df in grouped:
            group_df = group_df.sort_index(level="datetime")
            vals = group_df[feature_cols].values
            ys = group_df["target_return"].values
            dates = group_df.index.get_level_values("datetime")

            if len(group_df) < config.seq_len:
                continue

            for i in range(len(group_df) - config.seq_len + 1):
                window = vals[i:i + config.seq_len]
                if config.use_basic_ohlcv:
                    window = normalize_basic_ohlcv_window(window, feature_index)
                    if window is None:
                        continue
                d = dates[i + config.seq_len - 1]
                date_to_x[d].append(window)
                date_to_y[d].append(ys[i + config.seq_len - 1])
                date_to_stock[d].append(stock)

        sorted_dates = sorted(date_to_x.keys())
        t_start = pd.Timestamp(config.train_start_date)
        t_end = pd.Timestamp(config.train_end_date)
        v_start = pd.Timestamp(config.valid_start_date)
        v_end = pd.Timestamp(config.valid_end_date)
        train_dates = [d for d in sorted_dates if t_start <= d <= t_end]
        valid_dates = [d for d in sorted_dates if v_start <= d <= v_end]

        if not train_dates:
            raise ValueError(
                "No train dates available after daily cross-sectional batching. "
                f"Configured train window: [{t_start.date()}, {t_end.date()}], "
                f"available batched dates: {len(sorted_dates)}."
            )

        if not valid_dates:
            fallback_valid_count = max(1, min(60, len(train_dates) // 10))
            if fallback_valid_count >= len(train_dates):
                fallback_valid_count = max(1, len(train_dates) - 1)
            if fallback_valid_count <= 0:
                raise ValueError(
                    "No valid dates available, and train dates are insufficient for fallback validation."
                )
            fallback_valid_dates = train_dates[-fallback_valid_count:]
            train_dates = train_dates[:-fallback_valid_count]
            valid_dates = fallback_valid_dates
            print(
                "\n[WARN] Configured validation window has no usable dates after batching. "
                f"Fallback to the last {len(valid_dates)} train dates as validation: "
                f"{valid_dates[0].strftime('%Y-%m-%d')} -> {valid_dates[-1].strftime('%Y-%m-%d')}"
            )

        print(
            f" -> train_dates={len(train_dates)} | valid_dates={len(valid_dates)} | "
            f"train_range={train_dates[0].strftime('%Y-%m-%d')}~{train_dates[-1].strftime('%Y-%m-%d')} | "
            f"valid_range={valid_dates[0].strftime('%Y-%m-%d')}~{valid_dates[-1].strftime('%Y-%m-%d')}"
        )

        model = get_model(config, input_dim=len(feature_cols)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_base)

        def lr_lambda(epoch):
            if epoch < config.warmup_epochs:
                return (
                    config.lr_start + (config.lr_base - config.lr_start) * (epoch / float(config.warmup_epochs))
                ) / config.lr_base
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_loss = float("inf")
        patience_counter = 0
        best_path = os.path.join(exp_save_dir, f"best_model_seed{seed}.pt")

        for epoch in range(config.max_epochs):
            model.train()
            train_loss = 0.0
            np.random.shuffle(train_dates)

            for d in train_dates:
                x_batch = torch.tensor(np.array(date_to_x[d]), dtype=torch.float32).to(device)
                y_batch = torch.tensor(np.array(date_to_y[d]), dtype=torch.float32).to(device)

                optimizer.zero_grad()
                preds = model(x_batch)
                loss = rank_ic_loss(preds, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_dates)
            scheduler.step()

            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for d in valid_dates:
                    x_batch = torch.tensor(np.array(date_to_x[d]), dtype=torch.float32).to(device)
                    y_batch = torch.tensor(np.array(date_to_y[d]), dtype=torch.float32).to(device)
                    preds = model(x_batch)
                    valid_loss += rank_ic_loss(preds, y_batch).item()

            valid_loss /= len(valid_dates)
            print(f"[Seed {seed}] Epoch {epoch+1} | Train {-train_loss:.4f} | Valid {-valid_loss:.4f}")

            if valid_loss < best_loss:
                best_loss = valid_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience_counter += 1

            if patience_counter >= config.patience:
                break

        model.load_state_dict(torch.load(best_path))
        model.eval()

        all_preds = []
        all_keys = []
        with torch.no_grad():
            for d in sorted_dates:
                x_batch = torch.tensor(np.array(date_to_x[d]), dtype=torch.float32).to(device)
                stocks = date_to_stock[d]
                preds = model(x_batch).cpu().numpy()
                all_preds.extend(preds)
                for s in stocks:
                    all_keys.append((s, d))

        mi = pd.MultiIndex.from_arrays(
            [[k[0] for k in all_keys], [k[1] for k in all_keys]],
            names=["instrument", "datetime"],
        )
        return pd.DataFrame(all_preds, index=mi, columns=[f"alpha_seed{seed}"])

    for seed in seeds:
        alpha_df = train_one_seed(seed)
        all_alphas.append(alpha_df)

    print("\n=> Ensemble 3 seeds alpha...")
    merged = pd.concat(all_alphas, axis=1)
    final_alpha = merged.mean(axis=1).to_frame(name="Model_Alpha")

    out_file = os.path.join(exp_save_dir, "ensemble_alpha.parquet")
    final_alpha.to_parquet(out_file)
    print(f"\nEnsemble 完成: {out_file}")
    return final_alpha
