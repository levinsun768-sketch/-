from __future__ import annotations

"""
读取指纹、拼接日频环境数据，并构造下游监督样本。
"""
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from data_process import load_factors
except ImportError:
    print("WARNING: data_process module not found in parent directory.")

    def load_factors(*args, **kwargs):
        return {}


from downstream.prediction.predict_config import PredictConfig

BASIC_PRICE_FIELDS = ("open", "high", "low", "close")
BASIC_FLOW_FIELDS = ("volume", "turnover")


def _resolve_fingerprint_path(config: PredictConfig) -> Path:
    if not config.model_run_dir:
        raise ValueError("请在 PredictConfig.model_run_dir 中填写训练模型目录。")

    run_dir = Path(config.model_run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"未找到模型目录: {run_dir}")

    fp_dir = run_dir / config.fingerprint_subdir
    if not fp_dir.exists():
        raise FileNotFoundError(f"未找到指纹目录: {fp_dir}")

    if config.fingerprint_file_name:
        pq_path = fp_dir / config.fingerprint_file_name
        if not pq_path.exists():
            raise FileNotFoundError(f"未找到指定指纹文件: {pq_path}")
        return pq_path

    candidates = [p for p in fp_dir.glob("fingerprints_*.parquet") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"在 {fp_dir} 下未找到 fingerprints_*.parquet")

    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def _load_env_data(fields: list, start_date: str, end_date: str) -> pd.DataFrame:
    raw_dict = load_factors(fields, freq="day", limit_start=start_date, limit_end=end_date)
    series_list = []

    for field in fields:
        df_field = raw_dict.get(field)
        if df_field is None:
            continue

        if isinstance(df_field, pd.DataFrame) and "instrument" in df_field.columns and "datetime" in df_field.columns:
            df_field = df_field.set_index(["instrument", "datetime"]).iloc[:, 0]
        elif isinstance(df_field, pd.DataFrame) and "instrument" not in df_field.columns:
            stacked = df_field.stack(dropna=False)
            stacked.index.names = ["datetime", "instrument"]
            df_field = stacked.swaplevel().sort_index()

        if hasattr(df_field.index, "names"):
            df_field.index.names = ["instrument", "datetime"]
        df_field.name = field
        series_list.append(df_field)

    if not series_list:
        return pd.DataFrame()

    df_env = pd.concat(series_list, axis=1)
    df_env.index.names = ["instrument", "datetime"]
    return df_env


def prepare_dataset(config: PredictConfig) -> Tuple[pd.DataFrame, list]:
    pq_path = _resolve_fingerprint_path(config)
    print(f"1. 读取指纹文件: {pq_path}")

    df_x = pd.read_parquet(pq_path)
    if "stock" in df_x.columns:
        df_x = df_x.rename(columns={"stock": "instrument"})
    if "date" in df_x.columns:
        df_x = df_x.rename(columns={"date": "datetime"})

    df_x["datetime"] = pd.to_datetime(df_x["datetime"])
    df_x = df_x.set_index(["instrument", "datetime"]).sort_index()

    start_date = df_x.index.get_level_values("datetime").min().strftime("%Y-%m-%d")
    end_date = df_x.index.get_level_values("datetime").max().strftime("%Y-%m-%d")

    print("2. 加载日频环境数据与收益标签...")
    basic_fields = list(BASIC_PRICE_FIELDS + BASIC_FLOW_FIELDS) if config.use_basic_ohlcv else []
    query_fields = ["return_rate"] + basic_fields
    df_env = _load_env_data(query_fields, start_date, end_date)

    print(f"3. 构建目标收益: 留出 1 个开仓缓冲日后，累计 {config.horizon} 日对数收益...")
    if "return_rate" in df_env.columns:
        log_ret = np.log1p(df_env["return_rate"].fillna(0))
        df_target = log_ret.groupby(level="instrument").transform(
            lambda x: x.rolling(window=config.horizon).sum().shift(-(config.horizon + 1))
        )
        df_target.name = "target_return"
        df_env = df_env.drop(columns=["return_rate"]).join(df_target)

    if config.use_basic_ohlcv:
        print(" -> 保留日频 OHLCV 原始值，在滑窗阶段做样本内时序标准化。")

    df_merged = df_x.join(df_env, how="inner").dropna().sort_index()
    feature_cols = [c for c in df_merged.columns if c.startswith("fp_")] + basic_fields
    print(f" -> 特征维度: {len(feature_cols)}")

    return df_merged, feature_cols
