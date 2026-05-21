from __future__ import annotations

from lora_finetune.training.finetune_config import cfg
from lora_finetune.training.finetune_pipeline import run_finetune


def main():
    summaries = run_finetune(cfg)
    print("=========================================================")
    print("LoRA finetuning completed")
    print(f"Output group dir: {cfg.output_group_dir}")
    print(f"Finished years: {[item['year'] for item in summaries]}")
    print("=========================================================")


if __name__ == "__main__":
    main()
