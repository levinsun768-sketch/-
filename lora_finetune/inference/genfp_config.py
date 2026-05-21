from __future__ import annotations

"""
LoRA 年度 fingerprint 生成配置。

简要流程：
1. 先完成 `lora_finetune/training/run_finetune.py`，得到年度 LoRA 模型和
   `finetune_manifest.json`。
2. 在 `FINETUNE_GROUP_DIR` 中填写这一组 LoRA 年度实验目录。
3. 本脚本会自动读取 `finetune_manifest.json`，按年份找到对应的 LoRA encoder。
4. 推理时会：
   - 先用 base encoder 生成最早一段历史 fingerprint
   - 再用各年份 `lora_年份` encoder 生成对应部署区间的 fingerprint
   - 最终把各段 fingerprint 合并成一份总表，并写出 `fingerprint_manifest.json`

最小示例：
- FINETUNE_GROUP_DIR="/data/lbsun/saved_models/lora_finetuning_model_xxx"
- ENCODER_CKPT_NAME="encoder_lora_best.pt"
- FINGERPRINT_OUTPUT_SUBDIR="fp_dataset"
"""

from dataclasses import dataclass

import torch


@dataclass
class FinetuneFingerprintConfig:
    FINETUNE_GROUP_DIR: str = "/data/lbsun/saved_models/lora_finetuning_model_4c290c2e_20260416_180500_QV_v2_ST"  # 一组 LoRA 年度实验目录
    FINETUNE_MANIFEST_FILE_NAME: str = "finetune_manifest.json"  # 年度 LoRA 训练产出的 manifest 文件名
    ENCODER_CKPT_NAME: str = "encoder_lora_best.pt"  # 每个年份默认使用的 LoRA encoder checkpoint

    BATCH_SIZE: int = 1024  # fingerprint 推理 batch size；示例：1024
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"  # 推理设备；示例："cuda"

    FINGERPRINT_OUTPUT_SUBDIR: str = "fp_dataset"  # fingerprint 输出子目录；示例：fp_dataset
    MERGED_FINGERPRINT_FILE_NAME: str = "fingerprints_merged.parquet"  # 合并后的总 fingerprint 文件名
    FINGERPRINT_MANIFEST_FILE_NAME: str = "fingerprint_manifest.json"  # fingerprint 阶段产出的 manifest 文件名


cfg = FinetuneFingerprintConfig()
