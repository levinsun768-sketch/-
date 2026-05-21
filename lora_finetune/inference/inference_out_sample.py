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

from lora_finetune.inference.out_sample_config import cfg
from models.encoder import Encoder
from models.lora_utils import attach_lora_to_encoder
from training.dataset import load_feature_config, load_tensor_by_date_range


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


def _load_dataset_date_range(std_tensor_dir: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    meta_path = Path(std_tensor_dir) / "tensor_meta.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"tensor_meta.csv not found: {meta_path}")
    meta_df = pd.read_csv(meta_path, usecols=["date"])
    if meta_df.empty:
        raise ValueError(f"tensor_meta.csv is empty: {meta_path}")
    dates = pd.to_datetime(meta_df["date"])
    return dates.min(), dates.max()


def _check_feature_compatibility(train_snapshot: dict, override_std_tensor_dir: str) -> dict:
    feature_cfg = load_feature_config(override_std_tensor_dir)
    expected = {
        "F_DIM": int(train_snapshot["F_DIM"]),
        "PRICE_IDX": list(train_snapshot["PRICE_IDX"]),
        "TRADE_IDX": list(train_snapshot["TRADE_IDX"]),
    }
    actual = {
        "F_DIM": int(feature_cfg.get("F_DIM", -1)),
        "PRICE_IDX": list(feature_cfg.get("PRICE_IDX", [])),
        "TRADE_IDX": list(feature_cfg.get("TRADE_IDX", [])),
    }
    mismatches = {key: {"expected": expected[key], "actual": actual[key]} for key in expected if expected[key] != actual[key]}
    if mismatches and cfg.STRICT_FEATURE_CHECK:
        raise ValueError(f"Feature config mismatch detected: {json.dumps(mismatches, ensure_ascii=False)}")
    return {
        "expected": expected,
        "actual": actual,
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def _clip_date_str(date_str: str, dataset_min: pd.Timestamp, dataset_max: pd.Timestamp) -> str:
    clipped = min(max(pd.Timestamp(date_str), dataset_min), dataset_max)
    return clipped.strftime("%Y-%m-%d")


def _resolve_date_window(schedule: dict, dataset_min: pd.Timestamp, dataset_max: pd.Timestamp) -> dict | None:
    original = {
        "train_start": schedule["train_start"],
        "train_end": schedule["train_end"],
        "valid_start": schedule["valid_start"],
        "valid_end": schedule["valid_end"],
        "deploy_start": schedule["deploy_start"],
        "deploy_end": schedule["deploy_end"],
    }
    actual = {key: _clip_date_str(value, dataset_min, dataset_max) for key, value in original.items()}

    if cfg.DATE_START:
        lower = pd.Timestamp(cfg.DATE_START)
        for key in actual:
            actual[key] = max(pd.Timestamp(actual[key]), lower).strftime("%Y-%m-%d")
    if cfg.DATE_END:
        upper = pd.Timestamp(cfg.DATE_END)
        for key in actual:
            actual[key] = min(pd.Timestamp(actual[key]), upper).strftime("%Y-%m-%d")

    if pd.Timestamp(actual["deploy_start"]) > pd.Timestamp(actual["deploy_end"]):
        return None
    if not cfg.USE_DEPLOY_ONLY and pd.Timestamp(actual["train_start"]) > pd.Timestamp(actual["deploy_end"]):
        return None

    return {
        "original": original,
        "actual": actual,
    }


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        existing_files = list(output_dir.iterdir())
        if existing_files and not cfg.OVERWRITE_OUTPUT:
            raise FileExistsError(f"Output dir is not empty: {output_dir}. Set OVERWRITE_OUTPUT=True to reuse it.")
    output_dir.mkdir(parents=True, exist_ok=True)


def generate_out_sample_fingerprints() -> dict:
    if not cfg.FINETUNE_GROUP_DIR:
        raise ValueError("Please set FINETUNE_GROUP_DIR in out-sample config.")
    if not cfg.OVERRIDE_STD_TENSOR_DIR:
        raise ValueError("Please set OVERRIDE_STD_TENSOR_DIR in out-sample config.")

    group_dir = Path(cfg.FINETUNE_GROUP_DIR)
    manifest = _load_finetune_manifest()
    output_dir = group_dir / cfg.OUTPUT_SUBDIR
    _prepare_output_dir(output_dir)

    dataset_min, dataset_max = _load_dataset_date_range(cfg.OVERRIDE_STD_TENSOR_DIR)
    print("=========================================================")
    print("LoRA out-sample fingerprint inference")
    print(f"Finetune group dir: {group_dir}")
    print(f"Override tensor dir: {cfg.OVERRIDE_STD_TENSOR_DIR}")
    print(f"Output dir: {output_dir}")
    print(f"Dataset coverage: {dataset_min.strftime('%Y-%m-%d')} -> {dataset_max.strftime('%Y-%m-%d')}")
    print("=========================================================")

    merged_deploy_frames: list[pd.DataFrame] = []
    year_entries: list[dict] = []
    feature_check_result: dict | None = None

    for run in manifest["runs"]:
        year = run["year"]
        if cfg.TARGET_YEAR is not None and year != cfg.TARGET_YEAR:
            continue

        run_dir = Path(run["output_dir"])
        schedule = run["schedule"]
        train_snapshot = _load_training_snapshot(run_dir)
        if feature_check_result is None:
            feature_check_result = _check_feature_compatibility(train_snapshot, cfg.OVERRIDE_STD_TENSOR_DIR)

        window = _resolve_date_window(schedule, dataset_min, dataset_max)
        if window is None:
            print(f"[skip] year={year} has no valid overlap with override dataset.")
            continue

        encoder_ckpt_path = _resolve_encoder_ckpt(run_dir)
        encoder = _build_lora_encoder(train_snapshot, encoder_ckpt_path, cfg.DEVICE)

        actual = window["actual"]
        full_start = actual["deploy_start"] if cfg.USE_DEPLOY_ONLY else actual["train_start"]
        full_end = actual["deploy_end"]

        full_df = _extract_fingerprints(encoder, cfg.OVERRIDE_STD_TENSOR_DIR, full_start, full_end, f"{year}_out_sample")
        full_path = output_dir / f"fingerprints_{year}.parquet"
        full_df.to_parquet(full_path, index=False)

        deploy_df = full_df.copy()
        deploy_df["date"] = pd.to_datetime(deploy_df["date"])
        deploy_start = pd.Timestamp(actual["deploy_start"])
        deploy_end = pd.Timestamp(actual["deploy_end"])
        deploy_df = deploy_df[(deploy_df["date"] >= deploy_start) & (deploy_df["date"] <= deploy_end)].copy()
        deploy_df["date"] = deploy_df["date"].dt.strftime("%Y-%m-%d")
        deploy_path = output_dir / f"fingerprints_deploy_{year}.parquet"
        deploy_df.to_parquet(deploy_path, index=False)

        merged_deploy_frames.append(deploy_df)
        year_entries.append(
            {
                "year": year,
                "source_run_dir": str(run_dir),
                "train_start": actual["train_start"],
                "train_end": actual["train_end"],
                "valid_start": actual["valid_start"],
                "valid_end": actual["valid_end"],
                "deploy_start": actual["deploy_start"],
                "deploy_end": actual["deploy_end"],
                "original_train_start": window["original"]["train_start"],
                "original_train_end": window["original"]["train_end"],
                "original_valid_start": window["original"]["valid_start"],
                "original_valid_end": window["original"]["valid_end"],
                "original_deploy_start": window["original"]["deploy_start"],
                "original_deploy_end": window["original"]["deploy_end"],
                "fingerprint_path": str(full_path),
                "deploy_fingerprint_path": str(deploy_path),
            }
        )

    if not year_entries:
        raise ValueError("No yearly out-sample fingerprint entries were generated.")

    merged_df = pd.concat(merged_deploy_frames, axis=0, ignore_index=True).sort_values(["date", "stock"]).reset_index(drop=True)
    merged_path = output_dir / cfg.MERGED_FINGERPRINT_FILE_NAME
    merged_df.to_parquet(merged_path, index=False)

    fingerprint_manifest = {
        "mode": "out_sample",
        "finetune_group_dir": str(group_dir),
        "source_finetune_manifest_path": str(group_dir / cfg.FINETUNE_MANIFEST_FILE_NAME),
        "source_std_tensor_dir": manifest.get("std_tensor_dir", ""),
        "override_std_tensor_dir": cfg.OVERRIDE_STD_TENSOR_DIR,
        "output_subdir": cfg.OUTPUT_SUBDIR,
        "merged_fingerprint_path": str(merged_path),
        "feature_check": feature_check_result or {},
        "years": year_entries,
    }
    manifest_path = output_dir / cfg.FINGERPRINT_MANIFEST_FILE_NAME
    manifest_path.write_text(json.dumps(fingerprint_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return fingerprint_manifest


def main():
    manifest = generate_out_sample_fingerprints()
    print("=========================================================")
    print("Out-sample fingerprints generated")
    print(f"Merged fingerprint path: {manifest['merged_fingerprint_path']}")
    print(f"Years: {[item['year'] for item in manifest['years']]}")
    print("=========================================================")


if __name__ == "__main__":
    main()
