from __future__ import annotations

"""
下游 GRU 预测配置。
负责：
- 从 `MODEL_RUN_DIR/fp_dataset/` 读取指纹
- 构造日频监督样本
- 训练下游 GRU 并生成 alpha

最小示例：
- model_run_dir="/data/lbsun/saved_models/model_xxx"
- fingerprint_subdir="fp_dataset"
- horizon=5
- seq_len=20
"""
from dataclasses import dataclass


@dataclass
class PredictConfig:
    run_name_prefix: str = "ComplexGRUAlpha_ohlcv"  # 运行名前缀；示例：ComplexGRUAlpha_ohlcv

    model_run_dir: str = "/data/lbsun/saved_models/model_4c290c2e_20260416_180500"  # 主线模型目录；会从其中读取 fp_dataset
    fingerprint_subdir: str = "fp_dataset"  # fingerprint 子目录；示例：fp_dataset
    fingerprint_file_name: str = ""  # 手动指定 fingerprint 文件名；留空则走默认文件

    horizon: int = 5  # 预测收益 horizon；示例：5
    seq_len: int = 20  # GRU 输入序列长度；示例：20
    use_basic_ohlcv: bool = True  # 是否拼接基础 OHLCV 特征；示例：True

    train_start_date: str = "2021-07-01"  # 训练起始日期
    train_end_date: str = "2023-09-30"  # 训练结束日期
    valid_start_date: str = "2023-10-01"  # 验证集起始日期
    valid_end_date: str = "2023-12-31"  # 验证集结束日期

    model_type: str = "ComplexGRUAlpha"  # 下游模型类型；示例：ComplexGRUAlpha
    hidden_dim: int = 128  # GRU hidden dim；示例：128
    num_layers: int = 2  # GRU 层数；示例：2
    dropout: float = 0.3  # dropout；示例：0.3

    max_epochs: int = 100  # 最大训练轮数；示例：100
    patience: int = 10  # early stop 容忍轮数；示例：10
    lr_base: float = 1e-4  # 主学习率；示例：1e-4
    lr_start: float = 1e-6  # warmup 起始学习率；示例：1e-6
    warmup_epochs: int = 10  # warmup 轮数；示例：10

    @property
    def ret_name(self) -> str:
        return f"ret_{self.horizon}D"


default_predict_config = PredictConfig()
