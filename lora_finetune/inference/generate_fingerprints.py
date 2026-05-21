from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lora_finetune.inference.genfp_config import cfg
from models.encoder import Encoder
from models.lora_utils import attach_lora_to_encoder
from training.dataset import load_tensor_by_date_range


def _load_finetune_manifest() -> dict:
    manifest_path = Path(cfg.FINETUNE_GROUP_DIR) / cfg.FINETUNE_MANIFEST_FILE_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Finetune manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_encoder_ckpt(run_dir: Path) -> Path:
    ckpt_path = run_dir / "checkpoints" / cfg.ENCODER_CKPT_NAME
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {ckpt_path}")
    return ckpt_path


def _load_training_snapshot(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Training snapshot not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _build_lora_encoder(train_snapshot: dict, encoder_ckpt_path: Path, device: str) -> Encoder:
    encoder = Encoder(
        f_in=train_snapshot["F_DIM"],
        d_model=train_snapshot["D_MODEL"],
        nhead=train_snapshot["NHEAD"],
        num_layers=train_snapshot["NUM_LAYERS"],
        trade_idx=train_snapshot["TRADE_IDX"],
        trainable_proj=train_snapshot.get("TRAINABLE_PROJ", False),
        dim_feedforward=train_snapshot.get("DIM_FEEDFORWARD", 512),
        dropout=train_snapshot.get("DROPOUT", 0.1),
    )
    attach_lora_to_encoder(
        encoder=encoder,
        num_last_layers=train_snapshot["LORA_NUM_LAST_LAYERS"],
        rank=train_snapshot["LORA_RANK"],
        alpha=train_snapshot["LORA_ALPHA"],
        dropout=train_snapshot["LORA_DROPOUT"],
        target_modules=train_snapshot["LORA_TARGET_MODULES"],
        freeze_fixed_proj=True,
        freeze_input_norm=True,
        freeze_proj_back=True,
        freeze_transformer_norm=True,
    )
    encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location="cpu"))
    encoder.to(device)
    encoder.eval()
    return encoder


def _extract_fingerprints(encoder: Encoder, std_tensor_dir: str, start_date: str, end_date: str, year_label: str) -> pd.DataFrame:
    all_tensors, meta_df = load_tensor_by_date_range(start_date, end_date, std_tensor_dir)
    all_fingerprints = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_tensors), cfg.BATCH_SIZE), desc=f"Generate fp {year_label}"):
            batch = all_tensors[i: i + cfg.BATCH_SIZE].to(cfg.DEVICE)
            enc_out, _, _ = encoder(batch, mask_trade_ratio=0.0)
            all_fingerprints.append(enc_out[:, -1, :].cpu())

    fps_np = torch.cat(all_fingerprints, dim=0).numpy()
    fp_cols = [f"fp_{i:02d}" for i in range(fps_np.shape[1])]
    return pd.concat([meta_df.reset_index(drop=True), pd.DataFrame(fps_np, columns=fp_cols)], axis=1)


def generate_fingerprints() -> dict:
    if not cfg.FINETUNE_GROUP_DIR:
        raise ValueError("Please set FINETUNE_GROUP_DIR in genfp config.")

    group_dir = Path(cfg.FINETUNE_GROUP_DIR)
    manifest = _load_finetune_manifest()
    output_dir = group_dir / cfg.FINGERPRINT_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_deploy_frames: list[pd.DataFrame] = []
    year_entries: list[dict] = []

    for run in manifest["runs"]:
        run_dir = Path(run["output_dir"])
        schedule = run["schedule"]
        year = run["year"]
        train_snapshot = _load_training_snapshot(run_dir)
        encoder_ckpt_path = _resolve_encoder_ckpt(run_dir)
        encoder = _build_lora_encoder(train_snapshot, encoder_ckpt_path, cfg.DEVICE)

        full_start = schedule["train_start"]
        full_end = schedule["deploy_end"]
        full_df = _extract_fingerprints(encoder, manifest["std_tensor_dir"], full_start, full_end, f"{year}_full")
        full_path = output_dir / f"fingerprints_{year}.parquet"
        full_df.to_parquet(full_path, index=False)

        deploy_df = full_df.copy()
        deploy_df["date"] = pd.to_datetime(deploy_df["date"])
        deploy_start = pd.Timestamp(schedule["deploy_start"])
        deploy_end = pd.Timestamp(schedule["deploy_end"])
        deploy_df = deploy_df[(deploy_df["date"] >= deploy_start) & (deploy_df["date"] <= deploy_end)].copy()
        deploy_df["date"] = deploy_df["date"].dt.strftime("%Y-%m-%d")
        deploy_path = output_dir / f"fingerprints_deploy_{year}.parquet"
        deploy_df.to_parquet(deploy_path, index=False)

        merged_deploy_frames.append(deploy_df)
        year_entries.append(
            {
                "year": year,
                "source_run_dir": str(run_dir),
                "train_start": schedule["train_start"],
                "train_end": schedule["train_end"],
                "valid_start": schedule["valid_start"],
                "valid_end": schedule["valid_end"],
                "deploy_start": schedule["deploy_start"],
                "deploy_end": schedule["deploy_end"],
                "fingerprint_path": str(full_path),
                "deploy_fingerprint_path": str(deploy_path),
            }
        )

    merged_df = pd.concat(merged_deploy_frames, axis=0, ignore_index=True).sort_values(["date", "stock"]).reset_index(drop=True)
    merged_path = output_dir / cfg.MERGED_FINGERPRINT_FILE_NAME
    merged_df.to_parquet(merged_path, index=False)

    fingerprint_manifest = {
        "finetune_group_dir": str(group_dir),
        "merged_fingerprint_path": str(merged_path),
        "years": year_entries,
    }
    manifest_path = output_dir / cfg.FINGERPRINT_MANIFEST_FILE_NAME
    manifest_path.write_text(json.dumps(fingerprint_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return fingerprint_manifest


def main():
    manifest = generate_fingerprints()
    print("=========================================================")
    print("Finetuning fingerprints generated")
    print(f"Merged fingerprint path: {manifest['merged_fingerprint_path']}")
    print(f"Years: {[item['year'] for item in manifest['years']]}")
    print("=========================================================")


if __name__ == "__main__":
    main()
