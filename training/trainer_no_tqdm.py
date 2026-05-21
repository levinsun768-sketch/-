"""
训练主入口。

该版本与 `training/trainer.py` 保持相同训练逻辑，
仅移除 tqdm 进度条，适合远端 SSH / tmux 场景下减少终端刷新压力。
"""
import datetime
import hashlib
import json
import logging
import os
import platform
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

# 兼容在 `training/` 目录下直接执行 `python trainer_no_tqdm.py`
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.dataset import build_dataloader
from losses import RegularizationLossSmooth, compute_model_losses
from models.autoencoder import AutoEncoderDecoder, AutoEncoderEncoder
from models.decoder import Decoder
from models.encoder import Encoder
from training.trainer_config import cfg


def _sanitize_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "run"


def _set_seed(seed: int) -> None:
    """
    固定训练中的主要随机源。
    这可以稳定以下随机过程：
    - OrthoProjection 的正交初始化
    - 其他模块参数初始化
    - dropout
    - DataLoader shuffle
    - 训练时的随机 mask
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _snapshot_config() -> dict:
    data = {}
    for name in dir(cfg):
        if name.startswith("_"):
            continue
        value = getattr(cfg, name)
        if callable(value):
            continue
        if isinstance(value, tuple):
            value = list(value)
        elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            value = str(value)
        data[name] = value
    return data


def _prepare_run_dirs():
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_data = _snapshot_config()
    config_hash = hashlib.md5(
        json.dumps(config_data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:8]
    run_name = f"model_{config_hash}_{now}"
    run_dir = Path(cfg.MODEL_SAVE_DIR) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(config_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return run_name, run_dir, checkpoint_dir, log_dir


def _configure_logging(log_path: Path) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger("train.trainer_no_tqdm")


def _collect_hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": cfg.DEVICE,
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        info.update({
            "gpu_name": torch.cuda.get_device_name(device_index),
            "gpu_index": device_index,
            "gpu_count": torch.cuda.device_count(),
            "gpu_total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            "cuda_version": torch.version.cuda,
        })
    return info


def _save_summary(summary_path: Path, summary: dict) -> None:
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _save_checkpoint_pair(checkpoint_dir: Path, encoder, decoder, encoder_name: str, decoder_name: str) -> None:
    torch.save(encoder.state_dict(), checkpoint_dir / encoder_name)
    torch.save(decoder.state_dict(), checkpoint_dir / decoder_name)


def _build_model_pair():
    model_type = getattr(cfg, "MODEL_TYPE", "transformer_context")

    if model_type == "transformer_context":
        encoder = Encoder(
            f_in=cfg.F_DIM,
            d_model=cfg.D_MODEL,
            nhead=cfg.NHEAD,
            num_layers=cfg.NUM_LAYERS,
            trade_idx=cfg.TRADE_IDX,
            trainable_proj=cfg.TRAINABLE_PROJ,
            dim_feedforward=cfg.DIM_FEEDFORWARD,
            dropout=cfg.DROPOUT,
        ).to(cfg.DEVICE)

        decoder = Decoder(
            f_price=len(cfg.PRICE_IDX),
            f_trade=len(cfg.TRADE_IDX),
            d_model=cfg.D_MODEL,
            nhead=cfg.NHEAD,
            num_layers=cfg.NUM_LAYERS,
            proj_weight=encoder.fixed_proj.linear.weight.detach(),
            dim_feedforward=cfg.DIM_FEEDFORWARD,
            dropout=cfg.DROPOUT,
        ).to(cfg.DEVICE)
        return model_type, encoder, decoder

    if model_type == "autoencoder":
        encoder = AutoEncoderEncoder(
            f_in=cfg.F_DIM,
            d_model=cfg.D_MODEL,
            nhead=cfg.NHEAD,
            num_layers=cfg.NUM_LAYERS,
            latent_dim=cfg.D_MODEL,
            dim_feedforward=cfg.DIM_FEEDFORWARD,
            dropout=cfg.DROPOUT,
        ).to(cfg.DEVICE)

        decoder = AutoEncoderDecoder(
            latent_dim=cfg.D_MODEL,
            d_model=cfg.D_MODEL,
            nhead=cfg.NHEAD,
            num_layers=cfg.NUM_LAYERS,
            f_out=cfg.F_DIM,
            dim_feedforward=cfg.DIM_FEEDFORWARD,
            dropout=cfg.DROPOUT,
        ).to(cfg.DEVICE)
        return model_type, encoder, decoder

    raise ValueError(f"Unsupported MODEL_TYPE: {model_type}")


def main():
    _set_seed(cfg.SEED)
    run_name, run_dir, checkpoint_dir, log_dir = _prepare_run_dirs()
    logger = _configure_logging(log_dir / "train.log")
    hardware_info = _collect_hardware_info()

    logger.info("Training run directory: %s", run_dir)
    logger.info("Config saved to: %s", run_dir / "config.json")
    logger.info("Random seed: %s", cfg.SEED)
    logger.info("Hardware info: %s", json.dumps(hardware_info, ensure_ascii=False))

    dataloader = build_dataloader(
        std_tensor_dir=cfg.STD_TENSOR_DIR,
        train_dates=cfg.TRANSFORMER_TRAIN_DATES,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
    )
    dataset_size = len(dataloader.dataset)
    num_batches = len(dataloader)
    logger.info("Dataset size: %s | Batches per epoch: %s", dataset_size, num_batches)

    model_type, encoder, decoder = _build_model_pair()
    logger.info("Model type: %s", model_type)

    reg_loss_fn = RegularizationLossSmooth(
        lambda_d=cfg.LAMBDA_D,
        lambda_o=cfg.LAMBDA_O,
        lambda_u=cfg.LAMBDA_U,
        lambda_f=cfg.LAMBDA_F,
        lambda_b=cfg.LAMBDA_B,
    )

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.LR,
    )

    best_val_loss = float("inf")
    no_improve_count = 0
    completed_epochs = 0

    def lr_lambda(current_epoch):
        if current_epoch < cfg.WARMUP_EPOCHS:
            return (1e-8 + (cfg.LR - 1e-8) / cfg.WARMUP_EPOCHS * current_epoch) / cfg.LR
        return 1.0

    scheduler_warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    logger.info("Starting training for %s epochs without tqdm...", cfg.MAX_EPOCHS)

    for epoch in range(cfg.MAX_EPOCHS):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        epoch_start = time.perf_counter()
        batch_counter = 0
        last_batch_metrics = {}

        for batch in dataloader:
            batch_start = time.perf_counter()
            x = batch[0].to(cfg.DEVICE)

            total_loss, batch_metrics = compute_model_losses(
                model_type=model_type,
                encoder=encoder,
                decoder=decoder,
                x=x,
                cfg=cfg,
                reg_loss_fn=reg_loss_fn,
            )

            optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                cfg.GRAD_CLIP_NORM,
            )

            optimizer.step()
            epoch_loss += total_loss.item()
            batch_counter += 1
            batch_time = time.perf_counter() - batch_start

            last_batch_metrics = dict(batch_metrics)
            last_batch_metrics["batch_time_sec"] = batch_time

        if epoch < cfg.WARMUP_EPOCHS:
            scheduler_warmup.step()

        val_loss = epoch_loss / len(dataloader)
        scheduler_plateau.step(val_loss)
        completed_epochs = epoch + 1

        epoch_seconds = time.perf_counter() - epoch_start
        it_per_sec = batch_counter / epoch_seconds if epoch_seconds > 0 else 0.0
        sample_per_sec = (batch_counter * cfg.BATCH_SIZE) / epoch_seconds if epoch_seconds > 0 else 0.0

        logger.info(
            "Epoch %03d/%03d | avg_loss=%.6f | epoch_time=%.2fs | it/s=%.3f | samples/s=%.1f | last_total=%.6f | last_enc=%.6f | last_dec=%.6f",
            epoch + 1,
            cfg.MAX_EPOCHS,
            val_loss,
            epoch_seconds,
            it_per_sec,
            sample_per_sec,
            last_batch_metrics.get("total_loss", 0.0),
            last_batch_metrics.get("enc_loss", 0.0),
            last_batch_metrics.get("dec_loss", 0.0),
        )

        if val_loss < best_val_loss:
            if best_val_loss != float("inf"):
                logger.info("Loss improved from %.6f to %.6f", best_val_loss, val_loss)
            else:
                logger.info("Initial best loss set to %.6f", val_loss)
            best_val_loss = val_loss
            no_improve_count = 0
            _save_checkpoint_pair(
                checkpoint_dir,
                encoder,
                decoder,
                "encoder_best.pt",
                "decoder_best.pt",
            )
        else:
            no_improve_count += 1
            logger.info("No improvement for %d epoch(s)", no_improve_count)
            if no_improve_count >= cfg.EARLY_STOP_PATIENCE:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        if (epoch + 1) % 20 == 0:
            encoder_path = checkpoint_dir / f"encoder_epoch{epoch+1:03d}.pt"
            decoder_path = checkpoint_dir / f"decoder_epoch{epoch+1:03d}.pt"
            torch.save(encoder.state_dict(), encoder_path)
            torch.save(decoder.state_dict(), decoder_path)
            logger.info("Saved weights to %s after epoch %d", checkpoint_dir, epoch + 1)

    _save_checkpoint_pair(
        checkpoint_dir,
        encoder,
        decoder,
        "encoder_last.pt",
        "decoder_last.pt",
    )

    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "config_path": str(run_dir / "config.json"),
        "model_type": model_type,
        "seed": cfg.SEED,
        "completed_epochs": completed_epochs,
        "best_val_loss": best_val_loss,
        "best_encoder_path": str(checkpoint_dir / "encoder_best.pt"),
        "best_decoder_path": str(checkpoint_dir / "decoder_best.pt"),
        "last_encoder_path": str(checkpoint_dir / "encoder_last.pt"),
        "last_decoder_path": str(checkpoint_dir / "decoder_last.pt"),
        "dataset_size": dataset_size,
        "hardware_info": hardware_info,
    }
    _save_summary(run_dir / "training_summary.json", summary)
    logger.info("Training finished. Summary saved to: %s", run_dir / "training_summary.json")


if __name__ == "__main__":
    main()
