from __future__ import annotations

"""
LoRA 年度微调配置。

简要流程：
1. 先用主线 `training/trainer.py` 训练一个 base transformer 模型。
2. 在 `BASE_MODEL_RUN_DIR` 中填写这次 base 模型的 run 目录。
3. 在 `STD_TENSOR_DIR` 中填写 LoRA 微调所使用的标准化 tensor 数据集目录。
4. 微调流程会自动：
   - 读取 `BASE_MODEL_RUN_DIR/config.json`
   - 读取 `STD_TENSOR_DIR/tensor_meta.csv`
   - 根据 base 模型训练截止日期和数据集真实日期，自动构建各年份的
     `train / valid / deploy` 时间区间
   - 如果没有显式指定 `TARGET_YEAR`，则默认把 base 截止年之后所有可用年份依次跑完

`lora_年份` 的含义：
- `lora_2024_xxx` 表示：从同一个 base 模型出发，
  使用 2024 年部署前的历史数据做一次年度 LoRA 适配，
  最终得到一个用于 2024 年生成 fingerprint 的 encoder。
- `lora_2025_xxx`、`lora_2026_xxx` 等同理。

最小示例：
- BASE_MODEL_RUN_DIR="/data/lbsun/saved_models/model_xxx"
- STD_TENSOR_DIR="/data/lbsun/std_tensor_dataset/tensor_dataset_v4_xxx"
- OUTPUT_GROUP_NAME="lora_finetuning_model_xxx"
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from training.dataset import load_feature_config


@dataclass
class LoRAFinetuneConfig:
    BASE_MODEL_RUN_DIR: str = r"/data/lbsun/saved_models/model_4c290c2e_20260416_180500"  # base 模型 run 目录
    BASE_ENCODER_CKPT_NAME: str = "encoder_best.pt"  # base encoder checkpoint 文件名
    BASE_DECODER_CKPT_NAME: str = "decoder_best.pt"  # base decoder checkpoint 文件名

    STD_TENSOR_DIR: str = r"/data/lbsun/std_tensor_dataset/tensor_dataset_v4_2021-07-01_2026-03-25_20260518_112526_st_cross_regime"  # LoRA 微调使用的标准化 tensor 数据集目录
    OUTPUT_ROOT_DIR: str = "/data/lbsun/saved_models"  # LoRA 结果根目录
    OUTPUT_GROUP_NAME: str = "lora_finetuning_model_4c290c2e_20260416_180500_QV_v2_ST"  # LoRA 输出分组名；留空则基于 base 模型目录名自动生成

    TARGET_YEAR: int | None = None  # 只跑某一个年份；示例：2026；留空则自动跑所有可用年份
    AUTO_RUN_ALL_AVAILABLE_YEARS: bool = True  # 未指定 TARGET_YEAR 时是否自动跑完全部年份；示例：True

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"  # 训练设备；示例："cuda"
    BATCH_SIZE: int = 512  # batch size；示例：512
    MAX_EPOCHS: int = 30  # 最大训练轮数；示例：30
    LR: float = 2e-4  # 学习率；示例：2e-4
    EARLY_STOP_PATIENCE: int = 4  # 验证集 early stop 容忍轮数；示例：4
    GRAD_CLIP_NORM: float = 5.0  # 梯度裁剪阈值；示例：5.0

    MASK_RATIO: float = 0.15  # 自监督训练中的 trade mask 比例；示例：0.15
    WARMUP_EPOCHS: int = 3  # 学习率 warmup 轮数；示例：3

    LORA_NUM_LAST_LAYERS: int = 2  # 仅对最后几层 transformer block 挂 LoRA；示例：2
    LORA_RANK: int = 8  # LoRA rank；示例：8
    LORA_ALPHA: int = 16  # LoRA alpha；示例：16
    LORA_DROPOUT: float = 0.05  # LoRA dropout；示例：0.05
    LORA_TARGET_MODULES: tuple[str, ...] = (  # 挂 LoRA 的目标模块
        # "self_attn.out_proj",
        "linear1",
        "linear2",
        "self_attn.q_proj",
        "self_attn.v_proj",
    )

    USE_REG_LOSS: bool = True  # 是否启用 fingerprint 正则；示例：True
    LAMBDA_D: float = 0.3  # diversity 正则权重；示例：0.3
    LAMBDA_O: float = 0.3  # orthogonality 正则权重；示例：0.3
    LAMBDA_U: float = 0.3  # uniformity 正则权重；示例：0.3
    LAMBDA_F: float = 1.0  # 价格任务损失权重；示例：1.0
    LAMBDA_B: float = 1.0  # 交易任务损失权重；示例：1.0

    @property
    def _feature_config(self) -> dict:
        if not self.STD_TENSOR_DIR:
            raise ValueError("STD_TENSOR_DIR must be set before accessing feature configuration.")
        return load_feature_config(self.STD_TENSOR_DIR)

    @property
    def F_DIM(self) -> int:
        return int(self._feature_config.get("F_DIM", 15))

    @property
    def PRICE_IDX(self) -> list[int]:
        return list(self._feature_config.get("PRICE_IDX", [0, 1, 2]))

    @property
    def TRADE_IDX(self) -> list[int]:
        return list(
            self._feature_config.get(
                "TRADE_IDX",
                [i for i in range(self.F_DIM) if i not in self.PRICE_IDX],
            )
        )

    @property
    def output_group_dir(self) -> Path:
        group_name = self.OUTPUT_GROUP_NAME.strip()
        if not group_name:
            base_name = Path(self.BASE_MODEL_RUN_DIR).name if self.BASE_MODEL_RUN_DIR else "base"
            group_name = f"lora_finetuning_{base_name}"
        return Path(self.OUTPUT_ROOT_DIR) / group_name


cfg = LoRAFinetuneConfig()
