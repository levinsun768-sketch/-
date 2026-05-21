from __future__ import annotations

"""
下游预测模型注册表。
"""
import sys
from pathlib import Path

import torch.nn as nn

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream.prediction.predict_config import PredictConfig


class ComplexGRUAlpha(nn.Module):
    """GRU 编码时序窗口，并输出单个截面打分。"""

    def __init__(self, input_dim=70, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        final_state = out[:, -1, :]
        x = self.fc1(final_state)
        x = self.ln(x)
        x = self.relu(x)
        x = self.dropout(x)
        pred = self.fc2(x)
        return pred.squeeze(-1)


def get_model(config: PredictConfig, input_dim: int) -> nn.Module:
    if config.model_type == "ComplexGRUAlpha":
        return ComplexGRUAlpha(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    raise ValueError(f"未知的 model_type: {config.model_type}")
