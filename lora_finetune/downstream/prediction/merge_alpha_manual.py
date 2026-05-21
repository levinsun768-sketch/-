from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# 手动填写需要合并的 yearly alpha 文件路径。
ALPHA_PATHS: tuple[str, ...] = (
    "/data/lbsun/saved_models/lora_finetuning_model_4c290c2e_20260416_180500/prediction_backup_seq20_hrz5_ST/ComplexGRUAlphaLoRA_seq20_hrz5_2024_20260519_154559/alpha_2024.parquet",
    "/data/lbsun/saved_models/lora_finetuning_model_4c290c2e_20260416_180500/prediction_backup_seq20_hrz5_ST/ComplexGRUAlphaLoRA_seq20_hrz5_2025_20260519_155903/alpha_2025.parquet",
    "/data/lbsun/saved_models/lora_finetuning_model_4c290c2e_20260416_180500/prediction_backup_seq20_hrz5_ST/ComplexGRUAlphaLoRA_seq20_hrz5_2026_20260519_160456/alpha_2026.parquet",
)

# 合并输出目录。
OUTPUT_DIR = Path(
    "/data/lbsun/saved_models/lora_finetuning_model_4c290c2e_20260416_180500/prediction_backup_seq20_hrz5_ST"
)

MERGED_FILE_NAME = "merged_alpha.parquet"
MANIFEST_FILE_NAME = "alpha_merge_manifest.json"


def merge_alpha_manual() -> dict:
    if not ALPHA_PATHS:
        raise ValueError("ALPHA_PATHS is empty. Please fill in at least one alpha parquet path.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    used_paths: list[str] = []
    years: list[int] = []

    for raw_path in ALPHA_PATHS:
        alpha_path = Path(raw_path)
        if not alpha_path.exists():
            raise FileNotFoundError(f"Yearly alpha file not found: {alpha_path}")

        df = pd.read_parquet(alpha_path)
        if df.empty:
            print(f"[WARN] Empty alpha file skipped: {alpha_path}")
            continue

        frames.append(df)
        used_paths.append(str(alpha_path))

        stem = alpha_path.stem
        try:
            years.append(int(stem.split("_")[-1]))
        except ValueError:
            pass

    if not frames:
        raise ValueError("No non-empty alpha parquet files were available to merge.")

    merged_df = pd.concat(frames, axis=0).sort_index()
    merged_path = OUTPUT_DIR / MERGED_FILE_NAME
    merged_df.to_parquet(merged_path)

    merge_manifest = {
        "mode": "manual",
        "alpha_paths": used_paths,
        "years": years,
        "merged_alpha_path": str(merged_path),
        "row_count": int(len(merged_df)),
    }
    manifest_path = OUTPUT_DIR / MANIFEST_FILE_NAME
    manifest_path.write_text(
        json.dumps(merge_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return merge_manifest


def main() -> None:
    manifest = merge_alpha_manual()
    print("=========================================================")
    print("Manual alpha merge completed")
    print(f"Merged alpha path: {manifest['merged_alpha_path']}")
    print(f"Rows: {manifest['row_count']}")
    print("=========================================================")


if __name__ == "__main__":
    main()
