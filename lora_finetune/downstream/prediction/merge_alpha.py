from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lora_finetune.downstream.prediction.predict_config import cfg


def merge_alpha() -> dict:
    if not cfg.FINETUNE_GROUP_DIR:
        raise ValueError("Please set FINETUNE_GROUP_DIR in prediction config.")

    prediction_root = Path(cfg.FINETUNE_GROUP_DIR) / cfg.prediction_output_subdir
    manifest_path = prediction_root / cfg.prediction_manifest_file_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Prediction manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    frames = []
    years = []
    for entry in manifest["prediction_runs"]:
        alpha_path = Path(entry["alpha_path"])
        if not alpha_path.exists():
            raise FileNotFoundError(f"Yearly alpha file not found: {alpha_path}")
        frames.append(pd.read_parquet(alpha_path))
        years.append(entry["year"])

    merged_df = pd.concat(frames, axis=0).sort_index()
    merged_path = prediction_root / "merged_alpha.parquet"
    merged_df.to_parquet(merged_path)

    merge_manifest = {
        "prediction_manifest_path": str(manifest_path),
        "years": years,
        "merged_alpha_path": str(merged_path),
    }
    (prediction_root / "alpha_merge_manifest.json").write_text(json.dumps(merge_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return merge_manifest


def main():
    manifest = merge_alpha()
    print("=========================================================")
    print("Finetuning alpha merged")
    print(f"Merged alpha path: {manifest['merged_alpha_path']}")
    print("=========================================================")


if __name__ == "__main__":
    main()
