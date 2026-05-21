from __future__ import annotations

"""
LoRA 样本外 fingerprint 推理配置。

用途：
1. 复用已有 `FINETUNE_GROUP_DIR` 中的年度 LoRA encoder；
2. 不沿用训练时使用的 `std_tensor_dir`，而是改吃 `OVERRIDE_STD_TENSOR_DIR`；
3. 产出与 `fp_dataset` 结构一致、但目录名独立的新 fingerprint 数据集；
4. 便于后续直接复用 downstream prediction / backtest 链路。

最小示例：
- FINETUNE_GROUP_DIR="/data/lbsun/saved_models/lora_finetuning_model_xxx"
- OVERRIDE_STD_TENSOR_DIR="/data/lbsun/std_tensor_dataset/tensor_dataset_v4_st_xxx"
- OUTPUT_SUBDIR="fp_dataset_st_cross_regime"
"""

from dataclasses import dataclass

import torch


@dataclass
class OutSampleFingerprintConfig:
    FINETUNE_GROUP_DIR: str = "/data/lbsun/saved_models/lora_finetuning_model_4c290c2e_20260416_180500"  # LoRA 年度实验目录
    FINETUNE_MANIFEST_FILE_NAME: str = "finetune_manifest.json"  # LoRA 微调阶段产出的 manifest 文件名
    ENCODER_CKPT_NAME: str = "encoder_lora_best.pt"  # 每年默认使用的 LoRA checkpoint

    OVERRIDE_STD_TENSOR_DIR: str = r"/data/lbsun/std_tensor_dataset/tensor_dataset_v4_2021-07-01_2026-03-25_20260518_112526_st_cross_regime"  # 新输入数据集目录
    OUTPUT_SUBDIR: str = "fp_dataset_st_cross_regime"  # 样本外 fingerprint 输出目录名

    BATCH_SIZE: int = 1024  # 推理 batch size；示例：1024
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"  # 推理设备；示例："cuda"

    TARGET_YEAR: int | None = None  # 只推某一年；示例：2026；留空则跑全部年份
    USE_DEPLOY_ONLY: bool = False  # 是否只生成 deploy 区间 fingerprint；示例：False
    DATE_START: str = ""  # 手动限制最早推理日期；空字符串表示不限制
    DATE_END: str = ""  # 手动限制最晚推理日期；空字符串表示不限制
    STRICT_FEATURE_CHECK: bool = True  # 是否严格检查特征维度兼容性；示例：True
    OVERWRITE_OUTPUT: bool = False  # 输出目录非空时是否允许覆盖；示例：False

    MERGED_FINGERPRINT_FILE_NAME: str = "fingerprints_merged.parquet"  # 合并后的总 fingerprint 文件名
    FINGERPRINT_MANIFEST_FILE_NAME: str = "fingerprint_manifest.json"  # 输出 manifest 文件名


cfg = OutSampleFingerprintConfig()
