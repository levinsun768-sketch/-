"""
指纹生成配置。

推荐只填写 `MODEL_RUN_DIR`，其余信息由推理脚本从训练目录回溯读取。

最小示例：
- MODEL_RUN_DIR="/data/lbsun/saved_models/model_xxx"
- ENCODER_CKPT_NAME="encoder_best.pt"
- FINGERPRINT_GEN_DATES=("2024-01-01", "2024-12-31")
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GenFingerprintConfig:
    # 推荐直接填写一次训练对应的目录，例如:
    # /data/lbsun/saved_models/model_ab12cd34_20260416_103000
    MODEL_RUN_DIR: str = "/data/lbsun/saved_models/model_4c290c2e_20260416_180500"  # 主线模型目录；示例：/data/lbsun/saved_models/model_xxx

    # 兼容旧用法。若填写，则优先使用这个 checkpoint。
    ENCODER_CKPT_PATH: str = ""  # 手动指定 checkpoint 完整路径；通常留空

    # 若不手动指定 checkpoint，则按下面顺序自动选择:
    # 1. encoder_best.pt
    # 2. encoder_last.pt
    # 3. 最新的 encoder_epoch*.pt
    ENCODER_CKPT_NAME: str = "encoder_best.pt"  # checkpoint 文件名；示例：encoder_best.pt

    FINGERPRINT_GEN_DATES: tuple[str, str] = ("", "")  # fingerprint 生成区间；空字符串表示全量
    BATCH_SIZE: int = 1024  # 推理 batch size；示例：1024
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"  # 推理设备；示例："cuda"
    FINGERPRINT_OUTPUT_SUBDIR: str = "fp_dataset"  # 输出子目录；示例：fp_dataset

    # 仅为少量旧代码保留。
    FINGERPRINT_SAVE_DIR: str = ""  # 旧版保留字段；新逻辑通常不用


cfg = GenFingerprintConfig()
