from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.backtest.backtest_pipeline import run_evaluation
from lora_finetune.downstream.backtest.backtest_config import cfg
from lora_finetune.downstream.backtest.backtest_pipeline import resolve_merged_alpha_path


def main():
    if not cfg.FINETUNE_GROUP_DIR:
        raise ValueError("Please set FINETUNE_GROUP_DIR in backtest config.")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backtest_run_name = f"{cfg.backtest_name_prefix}_{run_timestamp}"
    alpha_path = resolve_merged_alpha_path(cfg.FINETUNE_GROUP_DIR, cfg.PREDICTION_SUBDIR, cfg.MERGED_ALPHA_FILE_NAME)
    exp_root = Path(cfg.FINETUNE_GROUP_DIR) / "backtest"
    exp_root.mkdir(parents=True, exist_ok=True)
    exp_save_dir = exp_root / backtest_run_name
    exp_save_dir.mkdir(parents=True, exist_ok=True)

    config_payload = asdict(cfg)
    config_payload.update({
        "prediction_run_dir": cfg.prediction_run_dir,
        "alpha_file_name": cfg.alpha_file_name,
        "resolved_alpha_path": str(alpha_path),
        "backtest_run_name": backtest_run_name,
        "exp_save_dir": str(exp_save_dir),
    })
    (exp_save_dir / "backtest_config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=========================================================")
    print(f"启动 LoRA downstream backtest: [{backtest_run_name}]")
    print(f"Finetune group dir: {cfg.FINETUNE_GROUP_DIR}")
    print(f"Prediction dir: {cfg.prediction_run_dir}")
    print(f"Alpha path: {alpha_path}")
    print(f"Output dir: {exp_save_dir}")
    print("=========================================================")

    alpha_df = pd.read_parquet(alpha_path)
    run_evaluation(alpha_df, cfg, exp_save_dir=str(exp_save_dir), backtest_run_name=backtest_run_name)

    print("=========================================================")
    print("LoRA finetune backtest completed")
    print(f"Alpha path: {alpha_path}")
    print(f"Output dir: {exp_save_dir}")
    print("=========================================================")


if __name__ == "__main__":
    main()
