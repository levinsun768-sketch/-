"""
训练配置。

这里只保留 transformer 训练所需的参数：
- 标准化 tensor 数据集位置
- 训练日期范围
- 模型保存根目录
- 模型结构与训练超参

最小示例：
- STD_TENSOR_DIR="/data/lbsun/std_tensor_dataset/tensor_dataset_v3_xxx"
- TRANSFORMER_TRAIN_DATES=("2021-07-01", "2023-12-31")
- MODEL_SAVE_DIR="/data/lbsun/saved_models"
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# 兼容在 `training/` 目录下直接执行相关脚本时的包导入。
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.dataset import load_feature_config, resolve_tensor_paths


@dataclass
class TrainerConfig:
    # 数据
    STD_TENSOR_DIR: str = r"/data/lbsun/std_tensor_dataset/tensor_dataset_v3_2021-07-01_2026-03-25_20260415_142022"  # 输入 tensor 数据集目录；示例：/data/lbsun/std_tensor_dataset/tensor_dataset_v3_xxx
    TRANSFORMER_TRAIN_DATES: tuple[str, str] = ("2021-07-01", "2023-12-31")  # base 模型训练区间；示例：("2021-07-01", "2023-12-31")

    # 模型保存
    MODEL_SAVE_DIR: str = "/data/lbsun/saved_models"  # 训练产物根目录；示例：/data/lbsun/saved_models

    # 模型结构
    MODEL_TYPE: str = "transformer_context"  # 模型类型；可选："transformer_context" 或 "autoencoder"
    D_MODEL: int = 128  # 隐层维度；示例：128 / 256
    NHEAD: int = 4  # multi-head attention 的头数；示例：4 / 8
    NUM_LAYERS: int = 4  # encoder 层数；示例：4 / 6
    DIM_FEEDFORWARD: int = 512  # FFN 隐层维度；示例：512 / 1024
    DROPOUT: float = 0.1  # dropout 比例；示例：0.1
    TRAINABLE_PROJ: bool = True  # 输入投影层是否可训练；示例：True

    # 训练超参
    SEED: int = 42  # 随机种子；示例：42
    BATCH_SIZE: int = 512  # batch size；示例：512 / 1024
    MAX_EPOCHS: int = 100  # 最大训练轮数；示例：100
    LR: float = 1e-4  # 主学习率；示例：1e-4
    MASK_RATIO: float = 0.15  # 自监督 mask 比例；示例：0.15

    # 调度与停止
    WARMUP_EPOCHS: int = 5  # warmup 轮数；示例：5
    EARLY_STOP_PATIENCE: int = 5  # early stop 容忍轮数；示例：5
    GRAD_CLIP_NORM: float = 5.0  # 梯度裁剪阈值；示例：5.0

    # 正则化
    USE_REG_LOSS: bool = True  # 是否启用正则损失；示例：True
    LAMBDA_D: float = 0.3  # diversity 正则权重；示例：0.3
    LAMBDA_O: float = 0.3  # orthogonality 正则权重；示例：0.3
    LAMBDA_U: float = 0.3  # uniformity 正则权重；示例：0.3
    LAMBDA_F: float = 1.0  # 前向 encoder 损失权重；示例：1.0
    LAMBDA_B: float = 1.0  # 后向 decoder 损失权重；示例：1.0

    # 设备
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"  # 训练设备；示例："cuda"

    @property
    def STD_DATA_PATH(self) -> str:
        data_path, _ = resolve_tensor_paths(self.STD_TENSOR_DIR)
        return str(data_path)

    @property
    def STD_META_PATH(self) -> str:
        _, meta_path = resolve_tensor_paths(self.STD_TENSOR_DIR)
        return str(meta_path)

    @property
    def _feature_config(self) -> dict:
        return load_feature_config(self.STD_TENSOR_DIR)

    @property
    def F_DIM(self) -> int:
        return self._feature_config.get("F_DIM", 15)

    @property
    def PRICE_IDX(self) -> list:
        return self._feature_config.get("PRICE_IDX", [0, 1, 2])

    @property
    def TRADE_IDX(self) -> list:
        return self._feature_config.get("TRADE_IDX", [i for i in range(self.F_DIM) if i not in self.PRICE_IDX])


cfg = TrainerConfig()
os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
