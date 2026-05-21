from __future__ import annotations

"""
LoRA 年度 alpha 回测配置。

简要流程：
1. 先完成 `lora_finetune/downstream/prediction/run_prediction.py` 和
   `merge_alpha.py`，得到合并后的 `merged_alpha.parquet`。
2. 在 `FINETUNE_GROUP_DIR` 中填写这一组 LoRA 年度实验目录。
3. 回测脚本会自动找到 prediction 阶段输出的合并 alpha。
4. 最终调用主线回测逻辑，对整段 LoRA alpha 做统一评估。

最小示例：
- FINETUNE_GROUP_DIR="/data/lbsun/saved_models/lora_finetuning_model_xxx"
- PREDICTION_SUBDIR="prediction"
- MERGED_ALPHA_FILE_NAME="merged_alpha.parquet"
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class FinetuneBacktestConfig:
    FINETUNE_GROUP_DIR: str = "/data/lbsun/saved_models/lora_finetuning_model_4c290c2e_20260416_180500_QV_v2_ST"  # 一组 LoRA 年度实验目录
    PREDICTION_SUBDIR: str = "prediction"  # prediction 阶段输出子目录
    MERGED_ALPHA_FILE_NAME: str = "merged_alpha.parquet"  # 合并后的总 alpha 文件名

    backtest_name_prefix: str = "backtest_lora"  # 回测实验名前缀；示例：backtest_lora
    horizon: int = 5  # 收益 horizon
    backtest_start_date: str = ""  # 回测起始日期；留空表示按 alpha 全区间
    backtest_end_date: str = ""  # 回测结束日期；留空表示按 alpha 全区间
    backtest_freq: str = "day"  # 回测频率；示例：day
    backtest_cost: float = 0.0012  # 单边交易成本；示例：0.0012
    backtest_benchmarks: List[str] = field(default_factory=lambda: ["IN000852"])  # 因子分析 benchmark；示例：["IN000852"]
    backtest_ic_benchmarks: List[str] = field(default_factory=lambda: ["is_all"])  # IC 评估范围；示例：["is_all"]
    ret_buy_price: str = "close"  # 收益计算买入价字段
    ret_sell_price: str = "close"  # 收益计算卖出价字段
    ret_open_shift: int = 1  # 收益计算 shift
    ret_open_limit: List[str] = field(default_factory=lambda: ["is_new", "is_suspended", "is_st"])  # 开仓限制条件
    ret_benchmarks: List[str] = field(default_factory=lambda: ["IN000852"])  # ret_config 内部 benchmark 配置
    kw_functions_config: dict = field(default_factory=lambda: {  # factor_analysis 功能开关；一般保持默认
        "ic": True,
        "long_short": True,
        "quantile": 10,
        "quantile_excess": True,
        "factor_ret_join_type": "left",
        "factor_corr": False,
        "crowding": False,
    })

    @property
    def ret_name(self) -> str:
        return f"ret_{self.horizon}D"

    @property
    def prediction_run_dir(self) -> str:
        # 与主线 backtest_pipeline 的接口保持一致，便于复用 run_evaluation
        if not self.FINETUNE_GROUP_DIR:
            return ""
        return str(Path(self.FINETUNE_GROUP_DIR) / self.PREDICTION_SUBDIR)

    @property
    def alpha_file_name(self) -> str:
        # 与主线配置字段命名对齐，便于归档和后续复用
        return self.MERGED_ALPHA_FILE_NAME


cfg = FinetuneBacktestConfig()
