from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.prediction.train_pipeline import run_training
from lora_finetune.downstream.prediction.predict_config import cfg


@dataclass
class _YearPredictConfig:
    run_name_prefix: str
    model_run_dir: str
    fingerprint_subdir: str
    fingerprint_file_name: str
    horizon: int
    seq_len: int
    use_basic_ohlcv: bool
    train_start_date: str
    train_end_date: str
    valid_start_date: str
    valid_end_date: str
    model_type: str
    hidden_dim: int
    num_layers: int
    dropout: float
    max_epochs: int
    patience: int
    lr_base: float
    lr_start: float
    warmup_epochs: int


def _load_fingerprint_manifest() -> dict:
    manifest_path = Path(cfg.FINETUNE_GROUP_DIR) / cfg.FINGERPRINT_SUBDIR / cfg.FINGERPRINT_MANIFEST_FILE_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Fingerprint manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def run_prediction() -> dict:
    if not cfg.FINETUNE_GROUP_DIR:
        raise ValueError("Please set FINETUNE_GROUP_DIR in prediction config.")

    group_dir = Path(cfg.FINETUNE_GROUP_DIR)
    output_root = group_dir / cfg.prediction_output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    fingerprint_manifest = _load_fingerprint_manifest()

    yearly_entries = [
        entry
        for entry in fingerprint_manifest["years"]
        if (cfg.target_year is None or entry["year"] == cfg.target_year)
    ]
    if not yearly_entries:
        raise ValueError("No yearly fingerprint entries available for prediction.")

    prediction_runs: list[dict] = []
    for entry in yearly_entries:
        year = entry["year"]
        deploy_start = pd.Timestamp(entry["deploy_start"])
        deploy_end = pd.Timestamp(entry["deploy_end"])
        train_start = pd.Timestamp(entry["train_start"])
        train_end = pd.Timestamp(entry["train_end"])
        valid_start = pd.Timestamp(entry["valid_start"])
        valid_end = pd.Timestamp(entry["valid_end"])

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / f"{cfg.run_name_prefix}_{year}_{run_timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        year_config = _YearPredictConfig(
            run_name_prefix=f"{cfg.run_name_prefix}_{year}",
            model_run_dir=str(group_dir),
            fingerprint_subdir=cfg.FINGERPRINT_SUBDIR,
            fingerprint_file_name=Path(entry["fingerprint_path"]).name,
            horizon=cfg.horizon,
            seq_len=cfg.seq_len,
            use_basic_ohlcv=cfg.use_basic_ohlcv,
            train_start_date=train_start.strftime("%Y-%m-%d"),
            train_end_date=train_end.strftime("%Y-%m-%d"),
            valid_start_date=valid_start.strftime("%Y-%m-%d"),
            valid_end_date=valid_end.strftime("%Y-%m-%d"),
            model_type=cfg.model_type,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            max_epochs=cfg.max_epochs,
            patience=cfg.patience,
            lr_base=cfg.lr_base,
            lr_start=cfg.lr_start,
            warmup_epochs=cfg.warmup_epochs,
        )

        alpha_df = run_training(year_config, exp_save_dir=str(run_dir))
        alpha_df = alpha_df.reset_index()
        alpha_df["datetime"] = pd.to_datetime(alpha_df["datetime"])
        alpha_df = alpha_df[(alpha_df["datetime"] >= deploy_start) & (alpha_df["datetime"] <= deploy_end)]
        alpha_df = alpha_df.set_index(["instrument", "datetime"]).sort_index()

        alpha_path = run_dir / f"alpha_{year}.parquet"
        alpha_df.to_parquet(alpha_path)
        (run_dir / "predict_config.json").write_text(json.dumps(asdict(year_config), indent=2, ensure_ascii=False), encoding="utf-8")

        prediction_runs.append(
            {
                "year": year,
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "valid_start": valid_start.strftime("%Y-%m-%d"),
                "valid_end": valid_end.strftime("%Y-%m-%d"),
                "deploy_start": deploy_start.strftime("%Y-%m-%d"),
                "deploy_end": deploy_end.strftime("%Y-%m-%d"),
                "fingerprint_path": entry["fingerprint_path"],
                "run_dir": str(run_dir),
                "alpha_path": str(alpha_path),
            }
        )

    prediction_manifest = {
        "finetune_group_dir": str(group_dir),
        "fingerprint_manifest_path": str(Path(cfg.FINETUNE_GROUP_DIR) / cfg.FINGERPRINT_SUBDIR / cfg.FINGERPRINT_MANIFEST_FILE_NAME),
        "prediction_runs": prediction_runs,
    }
    manifest_path = output_root / cfg.prediction_manifest_file_name
    manifest_path.write_text(json.dumps(prediction_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return prediction_manifest


def main():
    manifest = run_prediction()
    print("=========================================================")
    print("Finetuning prediction completed")
    print(f"Years: {[item['year'] for item in manifest['prediction_runs']]}")
    print("=========================================================")


if __name__ == "__main__":
    main()
