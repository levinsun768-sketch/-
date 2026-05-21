from __future__ import annotations

"""
LoRA 年度 GRU 预测配置。

简要流程：
1. 先完成 `lora_finetune/inference/generate_fingerprints.py`，得到
   `fingerprints_merged.parquet` 和 `fingerprint_manifest.json`。
2. 在 `FINETUNE_GROUP_DIR` 中填写这一组 LoRA 年度实验目录。
3. 预测脚本会自动读取 fingerprint manifest，并按年份滚动训练下游 GRU。
4. 每个年份会：
   - 使用过去 3 年的日度 fingerprint 数据
   - 留上一年最后一段时间做验证集
   - 生成当年部署区间的 alpha
5. 最终写出 yearly alpha 和 `prediction_manifest.json`。

最小示例：
- FINETUNE_GROUP_DIR="/data/lbsun/saved_models/lora_finetuning_model_xxx"
- FINGERPRINT_SUBDIR="fp_dataset"
- prediction_output_subdir="prediction"
"""

from dataclasses import dataclass


@dataclass
class FinetunePredictConfig:
    FINETUNE_GROUP_DIR: str = "/data/lbsun/saved_models/lora_finetuning_model_4c290c2e_20260416_180500_QV_v2_ST"  # 一组 LoRA 年度实验目录
    FINGERPRINT_SUBDIR: str = "fp_dataset"  # fingerprint 阶段输出子目录
    FINGERPRINT_MANIFEST_FILE_NAME: str = "fingerprint_manifest.json"  # fingerprint 阶段产出的 manifest 文件名
    MERGED_FINGERPRINT_FILE_NAME: str = "fingerprints_merged.parquet"  # 合并后的总 fingerprint 文件名

    run_name_prefix: str = "ComplexGRUAlphaLoRA_seq20_hrz5"  # 每年 GRU 预测实验名前缀；示例：ComplexGRUAlphaLoRA_seq20_hrz5
    horizon: int = 5  # 预测收益 horizon
    seq_len: int = 20  # 下游 GRU 输入序列长度
    use_basic_ohlcv: bool = False  # 是否拼接基础 OHLCV 特征

    model_type: str = "ComplexGRUAlpha"  # 下游模型类型
    hidden_dim: int = 128  # GRU hidden dim
    num_layers: int = 2  # GRU 层数
    dropout: float = 0.3  # GRU dropout

    max_epochs: int = 100  # 最大训练轮数；示例：100
    patience: int = 10  # early stop 容忍轮数；示例：10
    lr_base: float = 1e-4  # 主学习率；示例：1e-4
    lr_start: float = 1e-6  # warmup 起始学习率；示例：1e-6
    warmup_epochs: int = 10  # warmup 轮数；示例：10

    target_year: int | None = 2026  # 建议逐年跑时填写单一年份；示例：2026；留空则按 manifest 跑全部年份
    prediction_output_subdir: str = "prediction"  # alpha 输出子目录；示例：prediction 或 prediction_st_cross_regime
    prediction_manifest_file_name: str = "prediction_manifest.json"  # prediction 阶段产出的 manifest 文件名


cfg = FinetunePredictConfig()

