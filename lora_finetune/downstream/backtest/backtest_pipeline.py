from __future__ import annotations

from pathlib import Path


def resolve_merged_alpha_path(finetune_group_dir: str, prediction_subdir: str, merged_alpha_file_name: str) -> Path:
    alpha_path = Path(finetune_group_dir) / prediction_subdir / merged_alpha_file_name
    if not alpha_path.exists():
        raise FileNotFoundError(f"Merged alpha file not found: {alpha_path}")
    return alpha_path
