"""
指纹生成主入口。

只依赖 `inference/genfp_config.py`，并从训练目录中的 `config.json`
回溯读取源 tensor 数据集与模型结构参数。
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# 兼容在 `inference/` 目录下直接执行 `python generate_fingerprints.py`。
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.dataset import load_feature_config, load_tensor_by_date_range
from inference.genfp_config import cfg
from models.autoencoder import AutoEncoderEncoder
from models.encoder import Encoder


def resolve_run_dir() -> Path:
    """优先根据 MODEL_RUN_DIR 定位训练目录，兼容旧的 checkpoint 路径入口。"""
    if cfg.MODEL_RUN_DIR:
        run_dir = Path(cfg.MODEL_RUN_DIR)
        if not run_dir.exists():
            raise FileNotFoundError(f"未找到训练目录: {run_dir}")
        return run_dir

    if cfg.ENCODER_CKPT_PATH:
        ckpt_path = Path(cfg.ENCODER_CKPT_PATH)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"未找到 encoder checkpoint: {ckpt_path}")
        if ckpt_path.parent.name == "checkpoints":
            return ckpt_path.parent.parent
        return ckpt_path.parent

    raise ValueError("请在 inference/genfp_config.py 中填写 MODEL_RUN_DIR。")


def load_training_snapshot(run_dir: Path) -> dict:
    """读取训练阶段落盘的 config.json。"""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"未找到训练配置快照: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_encoder_ckpt(run_dir: Path) -> Path:
    """在训练目录下解析本次推理要使用的 encoder checkpoint。"""
    if cfg.ENCODER_CKPT_PATH:
        ckpt_path = Path(cfg.ENCODER_CKPT_PATH)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"未找到 encoder checkpoint: {ckpt_path}")
        return ckpt_path

    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"未找到 checkpoint 目录: {checkpoint_dir}")

    if cfg.ENCODER_CKPT_NAME:
        named_path = checkpoint_dir / cfg.ENCODER_CKPT_NAME
        if not named_path.exists():
            raise FileNotFoundError(f"未找到指定 checkpoint: {named_path}")
        return named_path

    for name in ("encoder_best.pt", "encoder_last.pt"):
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate

    epoch_candidates = sorted(checkpoint_dir.glob("encoder_epoch*.pt"))
    if epoch_candidates:
        return epoch_candidates[-1]

    raise FileNotFoundError(f"在 {checkpoint_dir} 下未找到可用的 encoder checkpoint。")


def build_fingerprint_encoder_from_snapshot(train_snapshot: dict, encoder_ckpt_path: Path, device: str):
    """根据训练快照重建指纹提取模型，并加载权重。"""
    std_tensor_dir = train_snapshot["STD_TENSOR_DIR"]
    feature_cfg = load_feature_config(std_tensor_dir)
    model_type = train_snapshot.get("MODEL_TYPE", "transformer_context")

    if model_type == "transformer_context":
        encoder = Encoder(
            f_in=train_snapshot.get("F_DIM", feature_cfg.get("F_DIM")),
            d_model=train_snapshot["D_MODEL"],
            nhead=train_snapshot["NHEAD"],
            num_layers=train_snapshot["NUM_LAYERS"],
            trade_idx=train_snapshot.get("TRADE_IDX", feature_cfg.get("TRADE_IDX", [])),
            trainable_proj=train_snapshot.get("TRAINABLE_PROJ", False),
            dim_feedforward=train_snapshot.get("DIM_FEEDFORWARD", 512),
            dropout=train_snapshot.get("DROPOUT", 0.1),
        )
    elif model_type == "autoencoder":
        encoder = AutoEncoderEncoder(
            f_in=train_snapshot.get("F_DIM", feature_cfg.get("F_DIM")),
            d_model=train_snapshot["D_MODEL"],
            nhead=train_snapshot["NHEAD"],
            num_layers=train_snapshot["NUM_LAYERS"],
            latent_dim=train_snapshot["D_MODEL"],
            dim_feedforward=train_snapshot.get("DIM_FEEDFORWARD", 512),
            dropout=train_snapshot.get("DROPOUT", 0.1),
        )
    else:
        raise ValueError(f"Unsupported MODEL_TYPE in training snapshot: {model_type}")

    encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location="cpu"))
    encoder.to(device)
    encoder.eval()
    return encoder


def generate_fingerprints():
    """按日期区间批量生成指纹并落盘到训练目录下。"""
    run_dir = resolve_run_dir()
    train_snapshot = load_training_snapshot(run_dir)
    encoder_ckpt_path = resolve_encoder_ckpt(run_dir)
    std_tensor_dir = train_snapshot["STD_TENSOR_DIR"]
    f_start, f_end = cfg.FINGERPRINT_GEN_DATES
    model_type = train_snapshot.get("MODEL_TYPE", "transformer_context")

    print(f"1. 读取训练目录: {run_dir}")
    print(f"2. 使用 checkpoint: {encoder_ckpt_path}")
    print(f"3. 回溯源 tensor 数据集: {std_tensor_dir}")
    all_tensors, meta_df = load_tensor_by_date_range(f_start, f_end, std_tensor_dir)

    print("4. 构建推理态 fingerprint encoder...")
    encoder = build_fingerprint_encoder_from_snapshot(train_snapshot, encoder_ckpt_path, cfg.DEVICE)

    print("5. 批量生成指纹...")
    all_fingerprints = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_tensors), cfg.BATCH_SIZE), desc="提取量价指纹"):
            batch = all_tensors[i: i + cfg.BATCH_SIZE].to(cfg.DEVICE)
            if model_type == "transformer_context":
                enc_out, _, _ = encoder(batch, mask_trade_ratio=0.0)
                fps = enc_out[:, -1, :].cpu()
            elif model_type == "autoencoder":
                z_day, _ = encoder(batch)
                fps = z_day.cpu()
            else:
                raise ValueError(f"Unsupported MODEL_TYPE in training snapshot: {model_type}")
            all_fingerprints.append(fps)

    all_fps_pt = torch.cat(all_fingerprints, dim=0)
    fps_np = all_fps_pt.numpy()

    output_dir = run_dir / cfg.FINGERPRINT_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    range_str = f"{f_start}_to_{f_end}" if (f_start or f_end) else "full_range"
    pq_path = output_dir / f"fingerprints_{range_str}_{now}.parquet"

    fp_cols = [f"fp_{i:02d}" for i in range(fps_np.shape[1])]
    fp_df = pd.DataFrame(fps_np, columns=fp_cols)
    final_df = pd.concat([meta_df.reset_index(drop=True), fp_df], axis=1)
    final_df.to_parquet(pq_path, index=False)

    meta = {
        "run_dir": str(run_dir),
        "model_type": model_type,
        "encoder_ckpt_path": str(encoder_ckpt_path),
        "std_tensor_dir": std_tensor_dir,
        "fingerprint_gen_dates": [f_start, f_end],
        "output_path": str(pq_path),
    }
    meta_path = output_dir / f"fingerprints_{range_str}_{now}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n指纹生成完成:")
    print(f" -> 输出路径: {pq_path}")
    print(f" -> 元信息: {meta_path}")
    print(f" -> 落盘记录数: {len(final_df)}")


def main():
    generate_fingerprints()


if __name__ == "__main__":
    main()
