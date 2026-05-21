from __future__ import annotations

"""
将下游模型输出的 alpha 接入 factor_analysis，并保存统计结果与图表。
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from data_process import factor_analysis, plot_res
except ImportError:
    print("WARNING: data_process module not found. Plotting will be mocked.")

    def factor_analysis(*args, **kwargs):
        return {}

    def plot_res(*args, **kwargs):
        return None

def run_evaluation(
    alpha_df: pd.DataFrame,
    config,
    exp_save_dir: str,
    backtest_run_name: str,
):
    print(f"\n=> 开始回测评估: {config.ret_name}")

    factors_df = alpha_df.reset_index()
    if "date" in factors_df.columns:
        factors_df = factors_df.rename(columns={"stock": "instrument", "date": "datetime"})

    start_date = config.backtest_start_date or factors_df["datetime"].min().strftime("%Y-%m-%d")
    end_date = config.backtest_end_date or factors_df["datetime"].max().strftime("%Y-%m-%d")
    print(f" -> 回测区间: {start_date} 至 {end_date}")

    ret_buy_price = getattr(config, "ret_buy_price", "close")
    ret_sell_price = getattr(config, "ret_sell_price", "close")
    ret_open_shift = getattr(config, "ret_open_shift", 1)
    ret_open_limit = getattr(config, "ret_open_limit", ["is_new", "is_suspended", "is_st"])
    ret_benchmarks = getattr(config, "ret_benchmarks", ["IN000852"])

    ret_config = [{
        "name": config.ret_name,
        "buy_price": ret_buy_price,
        "sell_price": ret_sell_price,
        "period": config.horizon,
        "open_shift": ret_open_shift,
        "cost": config.backtest_cost,
        "freq": config.backtest_freq,
        "open_limit": ret_open_limit,
        "benchmarks": ret_benchmarks,
    }]

    try:
        res = factor_analysis(
            factors_df,
            ret_config,
            start_date=start_date,
            end_date=end_date,
            benchmarks=config.backtest_benchmarks,
            ic_benchmarks=config.backtest_ic_benchmarks,
            freq=config.backtest_freq,
            plot=False,
            **config.kw_functions_config,
        )
        print(" -> 因子分析完成，开始归档结果...")

        exp_dir = exp_save_dir
        prediction_run_name = Path(config.prediction_run_dir).name if config.prediction_run_dir else "prediction"
        run_name = backtest_run_name or Path(exp_dir).name
        base_name = f"{prediction_run_name}_{run_name}_{config.ret_name}"

        if "ic_stats" in res:
            res["ic_stats"].to_csv(os.path.join(exp_dir, f"{base_name}_ic_stats.csv"))
        if "rank_ic_stats" in res:
            res["rank_ic_stats"].to_csv(os.path.join(exp_dir, f"{base_name}_rank_ic_stats.csv"))
        if "long_short_ret_stats" in res:
            res["long_short_ret_stats"].to_csv(os.path.join(exp_dir, f"{base_name}_long_short_stats.csv"))

        with plt.rc_context({"figure.figsize": (16, 16 * 0.6)}):
            plot_res(res)
            plot_path = os.path.join(exp_dir, f"{base_name}_evaluation_charts.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()

        print(f" -> 回测结果已保存到: {exp_dir}")
    except Exception as exc:
        print(f"回测失败: {exc}")
