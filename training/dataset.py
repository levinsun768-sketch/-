"""
dataset.py

张量数据加载与训练集切片工具。
训练与指纹生成都通过这里读取标准化 tensor 数据集。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def resolve_tensor_paths(std_tensor_dir: str) -> tuple[Path, Path]:
    """返回标准化 tensor 数据集中的数据文件与元数据文件路径。"""
    root = Path(std_tensor_dir)
    return root / "clean_tensor.npy", root / "tensor_meta.csv"


def load_feature_config(std_tensor_dir: str) -> dict:
    """读取与 tensor 数据集配套保存的 feature_config.yaml。"""
    import yaml

    config_file = Path(std_tensor_dir) / "feature_config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"未找到 feature_config.yaml: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tensor_by_date_range(start_date: str, end_date: str, std_tensor_dir: str):
    """
    从标准 tensor 数据集目录中按日期切取样本。
    空字符串表示不限制边界。
    """
    data_path, meta_path = resolve_tensor_paths(std_tensor_dir)
    print(f"正在使用 mmap 挂载张量数据: {std_tensor_dir}")

    data_mmap = np.load(data_path, mmap_mode="r")
    meta_df = pd.read_csv(meta_path)

    if data_mmap.shape[0] != len(meta_df):
        raise ValueError("Tensor 样本数与 Meta 行数不一致。")

    if not start_date and not end_date:
        print(" -> [全量加载模式] 未指定时间限制，直接返回全部样本。")
        subset_data = torch.from_numpy(data_mmap[:]).contiguous()
        return subset_data, meta_df

    print(f" -> 正在执行精确时间切片：{start_date} 至 {end_date}")
    dates = pd.to_datetime(meta_df["date"])
    mask = pd.Series([True] * len(meta_df))

    if start_date:
        mask = mask & (dates >= pd.Timestamp(start_date))
    if end_date:
        mask = mask & (dates <= pd.Timestamp(end_date))

    indices = np.where(mask)[0]
    if len(indices) == 0:
        raise ValueError(f"指定时间段内没有有效截面数据: {start_date} - {end_date}")

    subset_data_np = data_mmap[indices]
    subset_data = torch.from_numpy(subset_data_np).contiguous()
    subset_meta = meta_df.iloc[indices].reset_index(drop=True)

    print(f" -> 切片完成：共截取 {len(indices)} 个单日样本（原总量 {data_mmap.shape[0]}）")
    return subset_data, subset_meta


def build_dataloader(std_tensor_dir: str, train_dates: tuple[str, str], batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    根据标准 tensor 数据集目录和训练日期范围构造 DataLoader。
    """
    t_start, t_end = train_dates
    data, _ = load_tensor_by_date_range(t_start, t_end, std_tensor_dir)
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
