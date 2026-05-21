from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from training.dataset import resolve_tensor_paths


@dataclass
class YearSchedule:
    year: int
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    deploy_start: str
    deploy_end: str

    def to_dict(self) -> dict:
        return asdict(self)


def _load_base_snapshot(base_model_run_dir: str) -> dict:
    config_path = Path(base_model_run_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Base model config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_dataset_date_range(std_tensor_dir: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    _, meta_path = resolve_tensor_paths(std_tensor_dir)
    if not meta_path.exists():
        raise FileNotFoundError(f"tensor_meta.csv not found: {meta_path}")

    meta_df = pd.read_csv(meta_path, usecols=["date"])
    dates = pd.to_datetime(meta_df["date"])
    if dates.empty:
        raise ValueError(f"No valid dates found in: {meta_path}")
    return dates.min().normalize(), dates.max().normalize()


def _clip_date(date_value: pd.Timestamp, min_date: pd.Timestamp, max_date: pd.Timestamp) -> pd.Timestamp:
    return min(max(date_value.normalize(), min_date), max_date)


def build_year_schedules(base_model_run_dir: str, std_tensor_dir: str, target_year: int | None = None) -> list[YearSchedule]:
    base_snapshot = _load_base_snapshot(base_model_run_dir)
    train_dates = base_snapshot.get("TRANSFORMER_TRAIN_DATES")
    if not train_dates or len(train_dates) != 2:
        raise ValueError("Base model config must contain TRANSFORMER_TRAIN_DATES with [start, end].")

    _, base_train_end = train_dates
    base_cutoff_year = pd.Timestamp(base_train_end).year
    dataset_min_date, dataset_max_date = _load_dataset_date_range(std_tensor_dir)

    if target_year is not None:
        candidate_years = [target_year]
    else:
        candidate_years = list(range(base_cutoff_year + 1, dataset_max_date.year + 1))

    schedules: list[YearSchedule] = []
    for year in candidate_years:
        deploy_start = _clip_date(pd.Timestamp(f"{year}-01-01"), dataset_min_date, dataset_max_date)
        deploy_end = _clip_date(pd.Timestamp(f"{year}-12-31"), dataset_min_date, dataset_max_date)
        if deploy_start > deploy_end:
            continue
        valid_start = _clip_date(pd.Timestamp(f"{year - 1}-10-01"), dataset_min_date, dataset_max_date)
        valid_end = _clip_date(pd.Timestamp(f"{year - 1}-12-31"), dataset_min_date, dataset_max_date)
        if valid_start > valid_end:
            continue
        train_start = max(pd.Timestamp(f"{year - 3}-01-01"), dataset_min_date)
        train_end = min(valid_start - pd.Timedelta(days=1), dataset_max_date)
        if train_start > train_end:
            continue
        schedules.append(
            YearSchedule(
                year=year,
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                valid_start=valid_start.strftime("%Y-%m-%d"),
                valid_end=valid_end.strftime("%Y-%m-%d"),
                deploy_start=deploy_start.strftime("%Y-%m-%d"),
                deploy_end=deploy_end.strftime("%Y-%m-%d"),
            )
        )

    if target_year is not None and not schedules:
        raise ValueError(f"Target year {target_year} is not available under current base model and dataset range.")
    return schedules
