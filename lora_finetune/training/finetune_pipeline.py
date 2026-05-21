from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from losses import RegularizationLossSmooth, compute_model_losses
from lora_finetune.training.build_year_schedule import YearSchedule, build_year_schedules
from lora_finetune.training.finetune_config import LoRAFinetuneConfig
from models.decoder import Decoder
from models.encoder import Encoder
from models.lora_utils import attach_lora_to_encoder, freeze_module
from training.dataset import load_feature_config, load_tensor_by_date_range


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
    return logging.getLogger("lora_finetune.training")


def _load_base_snapshot(base_model_run_dir: str) -> dict:
    config_path = Path(base_model_run_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Base model config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _resolve_checkpoint(run_dir: str, ckpt_name: str) -> Path:
    ckpt_path = Path(run_dir) / "checkpoints" / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def _validate_feature_schema(base_snapshot: dict, finetune_feature_cfg: dict) -> None:
    expected_f_dim = int(base_snapshot["F_DIM"])
    expected_price_idx = list(base_snapshot["PRICE_IDX"])
    expected_trade_idx = list(base_snapshot["TRADE_IDX"])
    actual_f_dim = int(finetune_feature_cfg.get("F_DIM", -1))
    actual_price_idx = list(finetune_feature_cfg.get("PRICE_IDX", []))
    actual_trade_idx = list(finetune_feature_cfg.get("TRADE_IDX", []))

    if actual_f_dim != expected_f_dim:
        raise ValueError(f"Feature schema mismatch: F_DIM {actual_f_dim} != base {expected_f_dim}")
    if actual_price_idx != expected_price_idx:
        raise ValueError(f"Feature schema mismatch: PRICE_IDX {actual_price_idx} != base {expected_price_idx}")
    if actual_trade_idx != expected_trade_idx:
        raise ValueError(f"Feature schema mismatch: TRADE_IDX {actual_trade_idx} != base {expected_trade_idx}")


def _build_dataloader(std_tensor_dir: str, start_date: str, end_date: str, batch_size: int, shuffle: bool) -> DataLoader:
    data, _ = load_tensor_by_date_range(start_date, end_date, std_tensor_dir)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _prepare_year_dir(config: LoRAFinetuneConfig, schedule: YearSchedule) -> tuple[Path, Path, Path]:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    year_dir = config.output_group_dir / f"lora_{schedule.year}_{timestamp}"
    checkpoint_dir = year_dir / "checkpoints"
    log_dir = year_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return year_dir, checkpoint_dir, log_dir


def _build_base_model_pair(base_snapshot: dict, device: str) -> tuple[Encoder, Decoder]:
    encoder = Encoder(
        f_in=base_snapshot["F_DIM"],
        d_model=base_snapshot["D_MODEL"],
        nhead=base_snapshot["NHEAD"],
        num_layers=base_snapshot["NUM_LAYERS"],
        trade_idx=base_snapshot["TRADE_IDX"],
        trainable_proj=base_snapshot.get("TRAINABLE_PROJ", False),
        dim_feedforward=base_snapshot.get("DIM_FEEDFORWARD", 512),
        dropout=base_snapshot.get("DROPOUT", 0.1),
    ).to(device)

    decoder = Decoder(
        f_price=len(base_snapshot["PRICE_IDX"]),
        f_trade=len(base_snapshot["TRADE_IDX"]),
        d_model=base_snapshot["D_MODEL"],
        nhead=base_snapshot["NHEAD"],
        num_layers=base_snapshot["NUM_LAYERS"],
        proj_weight=encoder.fixed_proj.linear.weight.detach(),
        dim_feedforward=base_snapshot.get("DIM_FEEDFORWARD", 512),
        dropout=base_snapshot.get("DROPOUT", 0.1),
    ).to(device)
    return encoder, decoder


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _train_one_year(config: LoRAFinetuneConfig, base_snapshot: dict, schedule: YearSchedule) -> dict:
    feature_cfg = load_feature_config(config.STD_TENSOR_DIR)
    _validate_feature_schema(base_snapshot, feature_cfg)

    year_dir, checkpoint_dir, log_dir = _prepare_year_dir(config, schedule)
    logger = _configure_logging(log_dir / "train.log")

    base_encoder_ckpt = _resolve_checkpoint(config.BASE_MODEL_RUN_DIR, config.BASE_ENCODER_CKPT_NAME)
    base_decoder_ckpt = _resolve_checkpoint(config.BASE_MODEL_RUN_DIR, config.BASE_DECODER_CKPT_NAME)

    logger.info("Starting LoRA finetuning for year %s", schedule.year)
    logger.info("Base model directory: %s", config.BASE_MODEL_RUN_DIR)
    logger.info("Finetune tensor dir: %s", config.STD_TENSOR_DIR)
    logger.info("Schedule: %s", schedule.to_dict())

    encoder, decoder = _build_base_model_pair(base_snapshot, config.DEVICE)
    encoder.load_state_dict(torch.load(base_encoder_ckpt, map_location="cpu"))
    decoder.load_state_dict(torch.load(base_decoder_ckpt, map_location="cpu"))
    encoder.to(config.DEVICE)
    decoder.to(config.DEVICE)
    freeze_module(decoder)
    decoder.eval()

    lora_report = attach_lora_to_encoder(
        encoder=encoder,
        num_last_layers=config.LORA_NUM_LAST_LAYERS,
        rank=config.LORA_RANK,
        alpha=config.LORA_ALPHA,
        dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        freeze_fixed_proj=True,
        freeze_input_norm=True,
        freeze_proj_back=True,
        freeze_transformer_norm=True,
    )
    logger.info("LoRA attached: %s", asdict(lora_report))

    train_loader = _build_dataloader(config.STD_TENSOR_DIR, schedule.train_start, schedule.train_end, config.BATCH_SIZE, True)
    valid_loader = _build_dataloader(config.STD_TENSOR_DIR, schedule.valid_start, schedule.valid_end, config.BATCH_SIZE, False)
    logger.info(
        "Year %s dataloaders ready | train_samples=%s | train_batches=%s | valid_samples=%s | valid_batches=%s",
        schedule.year,
        len(train_loader.dataset),
        len(train_loader),
        len(valid_loader.dataset),
        len(valid_loader),
    )

    reg_loss_fn = RegularizationLossSmooth(
        lambda_d=config.LAMBDA_D,
        lambda_o=config.LAMBDA_O,
        lambda_u=config.LAMBDA_U,
        lambda_f=config.LAMBDA_F,
        lambda_b=config.LAMBDA_B,
    )
    optimizer = torch.optim.Adam([p for p in encoder.parameters() if p.requires_grad], lr=config.LR)

    def lr_lambda(current_epoch):
        if current_epoch < config.WARMUP_EPOCHS:
            return (1e-8 + (config.LR - 1e-8) / config.WARMUP_EPOCHS * current_epoch) / config.LR
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_count = 0

    for epoch in range(config.MAX_EPOCHS):
        encoder.train()
        train_loss_sum = 0.0
        train_steps = 0
        epoch_start = time.perf_counter()
        log_every = max(1, len(train_loader) // 10)

        logger.info(
            "Year %s | Epoch %03d/%03d started | lr=%.6e",
            schedule.year,
            epoch + 1,
            config.MAX_EPOCHS,
            optimizer.param_groups[0]["lr"],
        )

        for batch_idx, batch in enumerate(train_loader, start=1):
            x = batch[0].to(config.DEVICE)
            total_loss, batch_metrics = compute_model_losses(
                model_type="transformer_context",
                encoder=encoder,
                decoder=decoder,
                x=x,
                cfg=config,
                reg_loss_fn=reg_loss_fn,
            )
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            train_loss_sum += total_loss.item()
            train_steps += 1

            if batch_idx % log_every == 0 or batch_idx == len(train_loader):
                elapsed = time.perf_counter() - epoch_start
                avg_so_far = train_loss_sum / max(train_steps, 1)
                logger.info(
                    "Year %s | Epoch %03d/%03d | batch %d/%d | avg_train_so_far=%.6f | last_total=%.6f | elapsed=%.1fs",
                    schedule.year,
                    epoch + 1,
                    config.MAX_EPOCHS,
                    batch_idx,
                    len(train_loader),
                    avg_so_far,
                    batch_metrics["total_loss"],
                    elapsed,
                )

        scheduler.step()

        encoder.eval()
        valid_loss_sum = 0.0
        valid_steps = 0
        logger.info(
            "Year %s | Epoch %03d/%03d validation started",
            schedule.year,
            epoch + 1,
            config.MAX_EPOCHS,
        )
        with torch.no_grad():
            for batch in valid_loader:
                x = batch[0].to(config.DEVICE)
                total_loss, _ = compute_model_losses(
                    model_type="transformer_context",
                    encoder=encoder,
                    decoder=decoder,
                    x=x,
                    cfg=config,
                    reg_loss_fn=reg_loss_fn,
                )
                valid_loss_sum += total_loss.item()
                valid_steps += 1
        avg_train_loss = train_loss_sum / max(train_steps, 1)
        avg_valid_loss = valid_loss_sum / max(valid_steps, 1)
        logger.info(
            "Year %s | Epoch %03d/%03d | train_loss=%.6f | valid_loss=%.6f | epoch_time=%.2fs",
            schedule.year,
            epoch + 1,
            config.MAX_EPOCHS,
            avg_train_loss,
            avg_valid_loss,
            time.perf_counter() - epoch_start,
        )
        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            best_epoch = epoch + 1
            no_improve_count = 0
            torch.save(encoder.state_dict(), checkpoint_dir / "encoder_lora_best.pt")
        else:
            no_improve_count += 1
            if no_improve_count >= config.EARLY_STOP_PATIENCE:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

    torch.save(encoder.state_dict(), checkpoint_dir / "encoder_lora_last.pt")

    summary = {
        "year": schedule.year,
        "base_model_run_dir": config.BASE_MODEL_RUN_DIR,
        "base_encoder_ckpt_path": str(base_encoder_ckpt),
        "base_decoder_ckpt_path": str(base_decoder_ckpt),
        "std_tensor_dir": config.STD_TENSOR_DIR,
        "schedule": schedule.to_dict(),
        "best_epoch": best_epoch,
        "best_valid_loss": best_val_loss,
        "output_dir": str(year_dir),
        "best_encoder_path": str(checkpoint_dir / "encoder_lora_best.pt"),
        "last_encoder_path": str(checkpoint_dir / "encoder_lora_last.pt"),
        "lora_report": asdict(lora_report),
        "finetune_config": asdict(config),
    }
    _save_json(year_dir / "config.json", {
        **base_snapshot,
        "MODEL_TYPE": "transformer_context",
        "BASE_MODEL_RUN_DIR": config.BASE_MODEL_RUN_DIR,
        "STD_TENSOR_DIR": config.STD_TENSOR_DIR,
        "LORA_FINETUNE_YEAR": schedule.year,
        "LORA_FINETUNE_SCHEDULE": schedule.to_dict(),
        "LORA_RANK": config.LORA_RANK,
        "LORA_ALPHA": config.LORA_ALPHA,
        "LORA_DROPOUT": config.LORA_DROPOUT,
        "LORA_NUM_LAST_LAYERS": config.LORA_NUM_LAST_LAYERS,
        "LORA_TARGET_MODULES": list(config.LORA_TARGET_MODULES),
    })
    _save_json(year_dir / "finetune_summary.json", summary)
    return summary


def run_finetune(config: LoRAFinetuneConfig) -> list[dict]:
    if not config.BASE_MODEL_RUN_DIR:
        raise ValueError("Please set BASE_MODEL_RUN_DIR in finetune config.")
    if not config.STD_TENSOR_DIR:
        raise ValueError("Please set STD_TENSOR_DIR in finetune config.")
    if config.TARGET_YEAR is None and not config.AUTO_RUN_ALL_AVAILABLE_YEARS:
        raise ValueError("Set TARGET_YEAR or enable AUTO_RUN_ALL_AVAILABLE_YEARS.")

    base_snapshot = _load_base_snapshot(config.BASE_MODEL_RUN_DIR)
    schedules = build_year_schedules(config.BASE_MODEL_RUN_DIR, config.STD_TENSOR_DIR, config.TARGET_YEAR)
    if not schedules:
        raise ValueError("No available LoRA finetuning years found.")

    config.output_group_dir.mkdir(parents=True, exist_ok=True)
    summaries = [_train_one_year(config, base_snapshot, schedule) for schedule in schedules]
    manifest = {
        "base_model_run_dir": config.BASE_MODEL_RUN_DIR,
        "std_tensor_dir": config.STD_TENSOR_DIR,
        "target_year": config.TARGET_YEAR,
        "auto_run_all_available_years": config.AUTO_RUN_ALL_AVAILABLE_YEARS,
        "years": [item["year"] for item in summaries],
        "runs": summaries,
    }
    _save_json(config.output_group_dir / "finetune_manifest.json", manifest)
    return summaries
