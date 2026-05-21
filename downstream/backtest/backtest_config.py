from __future__ import annotations

"""
下游 alpha 回测配置。
负责：
- 读取已经生成好的 alpha parquet
- 运行 factor_analysis
- 归档回测统计与图表

最小示例：
- prediction_run_dir="/home/lbsun/transformer_pipeline/downstream/run/ComplexGRUAlpha_xxx"
- alpha_file_name="ensemble_alpha.parquet"
- horizon=5
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class BacktestConfig:
    backtest_name_prefix: str = "backtest"  # 回测运行名前缀；示例：backtest
    prediction_run_dir: str = "/home/lbsun/transformer_pipeline/downstream/run/ComplexGRUAlpha_ohlcv_5D_20260420_133632"  # prediction 输出目录
    alpha_file_name: str = "ensemble_alpha.parquet"  # 读取的 alpha 文件名；示例：ensemble_alpha.parquet
    horizon: int = 5  # 收益 horizon；示例：5

    backtest_start_date: str = "2024-01-01"  # 回测起始日期；空字符串表示按 alpha 最早日期
    backtest_end_date: str = "2026-01-01"  # 回测结束日期；空字符串表示按 alpha 最晚日期
    backtest_freq: str = "day"  # 回测频率；示例：day
    backtest_cost: float = 0.0012  # 单边交易成本；示例：0.0012
    backtest_benchmarks: List[str] = field(default_factory=lambda: ["IN000852"])  # 因子分析 benchmark；示例：["IN000852"]
    backtest_ic_benchmarks: List[str] = field(default_factory=lambda: ["is_all"])  # IC 评估范围；示例：["is_all"]
    ret_buy_price: str = "close"  # ret_config 买入价字段；示例：close
    ret_sell_price: str = "close"  # ret_config 卖出价字段；示例：close
    ret_open_shift: int = 1  # ret_config shift；示例：1
    ret_open_limit: List[str] = field(default_factory=lambda: ["is_new", "is_suspended", "is_st"])  # 开仓限制标签；示例：["is_new", "is_suspended"]
    ret_benchmarks: List[str] = field(default_factory=lambda: ["IN000852"])  # ret_config 内部 benchmark；示例：["IN000852"]
    kw_functions_config: dict = field(default_factory=lambda: {
        "ic": True,
        "long_short": True,
        "quantile": 10,
        "quantile_excess": True,
        "factor_ret_join_type": "left",
        "factor_corr": False,
        "crowding": False,
    })  # factor_analysis 功能开关；一般保持默认

    @property
    def ret_name(self) -> str:
        return f"ret_{self.horizon}D"


default_backtest_config = BacktestConfig()
