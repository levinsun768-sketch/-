"""
models/decoder.py — Decoder
逻辑与原始 transformer_pipeline.py 完全一致。
"""
import torch
import torch.nn as nn
from models.positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(
        self,
        f_price,
        f_trade,
        d_model,
        nhead,
        num_layers,
        proj_weight,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(f_price + f_trade, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        # Transformer Decoder Layer，前馈维度512，激活函数GELU
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, f_price)
        self.proj_weight = proj_weight

    def forward(self, x_price, x_trade, memory):
        B, T, _ = x_price.shape

        # -----------------------------
        # 核心修改：价格特征移位（严格因果约束）
        # -----------------------------
        # 价格特征右移一位，开头补0（预测t=0时无历史价格）
        x_price_shifted = torch.cat([
            torch.zeros(B, 1, x_price.shape[-1], device=x_price.device),  # t=0补0
            x_price[:, :-1, :]  # t>=1时用t-1的价格
        ], dim=1)  # shape: [B, T, f_price]

        # 交易特征保持不变（可使用t及之前的交易信息）
        x_trade_unchanged = x_trade  # shape: [B, T, f_trade]

        # 拼接移位后的价格和原始交易特征
        x_in = torch.cat([x_price_shifted, x_trade_unchanged], dim=-1)

        # 投影 + 位置编码
        x_proj = self.input_proj(x_in)
        x_proj = self.pos_enc(x_proj)

        # 因果掩码（阻止关注未来信息）
        causal_mask = torch.triu(torch.ones(T, T, device=x_proj.device) * float('-inf'), diagonal=1)
        out = self.transformer(tgt=x_proj, memory=memory, tgt_mask=causal_mask)
        out = self.output_proj(out)

        return out
