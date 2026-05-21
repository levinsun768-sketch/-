from __future__ import annotations

"""
下游 alpha 回测统一入口。
"""
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.backtest.backtest_config import BacktestConfig
from downstream.backtest.backtest_pipeline import run_evaluation


def _resolve_alpha_path(config: BacktestConfig) -> Path:
    run_dir = Path(config.prediction_run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"未找到预测目录: {run_dir}")

    alpha_path = run_dir / config.alpha_file_name
    if not alpha_path.exists():
        raise FileNotFoundError(f"未找到 alpha 文件: {alpha_path}")
    return alpha_path


def main():
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = BacktestConfig(
        # backtest_name_prefix="backtest",
        # prediction_run_dir="",
        # horizon=5,
    )

    if not config.prediction_run_dir:
        raise ValueError("请先在 downstream/backtest/run_backtest.py 中填写 config.prediction_run_dir。")

    alpha_path = _resolve_alpha_path(config)
    backtest_run_name = f"{config.backtest_name_prefix}_{run_timestamp}"
    exp_root = Path(config.prediction_run_dir) / "experiments"
    exp_root.mkdir(parents=True, exist_ok=True)
    exp_save_dir = str((exp_root / backtest_run_name).resolve())
    Path(exp_save_dir).mkdir(parents=True, exist_ok=True)

    config_payload = asdict(config)
    config_payload.update({
        "backtest_run_name": backtest_run_name,
        "exp_save_dir": exp_save_dir,
        "resolved_alpha_path": str(alpha_path),
    })
    config_path = Path(exp_save_dir) / "backtest_config.json"
    config_path.write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=========================================================")
    print(f"启动下游回测: [{backtest_run_name}]")
    print(f"预测目录: {config.prediction_run_dir}")
    print(f"Alpha 路径: {alpha_path}")
    print(f"输出目录: {exp_save_dir}")
    print("=========================================================")

    alpha_df = pd.read_parquet(alpha_path)
    run_evaluation(
        alpha_df,
        config,
        exp_save_dir=exp_save_dir,
        backtest_run_name=backtest_run_name,
    )

    print("=========================================================")
    print("下游回测完成")
    print("=========================================================")


if __name__ == "__main__":
    main()
