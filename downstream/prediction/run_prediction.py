from __future__ import annotations

"""
下游 GRU 预测统一入口。
"""
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.prediction.predict_config import PredictConfig
from downstream.prediction.train_pipeline import run_training


def main():
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config = PredictConfig(
        # run_name_prefix="ComplexGRUAlpha",
        # model_run_dir="",
        # horizon=5,
        # seq_len=20,
        # model_type="ComplexGRUAlpha",
        # hidden_dim=128,
        # max_epochs=100,
        # patience=10,
    )

    if not config.model_run_dir:
        raise ValueError("请先在 downstream/prediction/run_prediction.py 中填写 config.model_run_dir。")

    exp_root = CURRENT_DIR.parent / "run"
    exp_root.mkdir(parents=True, exist_ok=True)
    run_dir_name = f"{config.run_name_prefix}_{config.horizon}D_{run_timestamp}"
    exp_save_dir = str((exp_root / run_dir_name).resolve())
    Path(exp_save_dir).mkdir(parents=True, exist_ok=True)

    config_payload = asdict(config)
    config_payload.update({
        "run_dir_name": run_dir_name,
        "exp_save_dir": exp_save_dir,
    })
    config_path = Path(exp_save_dir) / "predict_config.json"
    config_path.write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=========================================================")
    print(f"启动下游预测: [{run_dir_name}]")
    print(f"模型目录: {config.model_run_dir}")
    print(f"输出目录: {exp_save_dir}")
    print("=========================================================")

    run_training(config, exp_save_dir=exp_save_dir)

    print("=========================================================")
    print("下游预测完成")
    print("=========================================================")


if __name__ == "__main__":
    main()
