# lora_finetune

这是一条和主线 `training / inference / downstream / backtest` 隔离的 LoRA 实验链路。

目录职责：
- `lora_finetune/training/`
  - 基于已有 base transformer 模型做年度 LoRA 微调
- `lora_finetune/inference/`
  - 使用年度 LoRA encoder 批量生成各年份 fingerprint，并合并
- `lora_finetune/downstream/prediction/`
  - 使用合并后的 fingerprint，按年份滚动训练 GRU 并输出 yearly alpha
- `lora_finetune/downstream/backtest/`
  - 合并 yearly alpha，并做统一回测

推荐顺序：
1. 先用主线 `training/trainer.py` 训练 base model
2. 跑 `lora_finetune/training/run_finetune.py`
3. 跑 `lora_finetune/inference/generate_fingerprints.py`
4. 跑 `lora_finetune/downstream/prediction/run_prediction.py`
5. 跑 `lora_finetune/downstream/prediction/merge_alpha.py`
6. 跑 `lora_finetune/downstream/backtest/run_backtest.py`
