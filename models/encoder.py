"""
models/encoder.py — 固定正交投影层 + Encoder
逻辑与原始 transformer_pipeline.py 完全一致。
"""
import torch
import torch.nn as nn
from models.positional_encoding import PositionalEncoding
import torch.nn.utils.parametrizations as P


class OrthoProjection(nn.Module):
    def __init__(self, f_in, d_model, trainable=False):
        super().__init__()
        self.trainable = trainable
        
        # 内置一个没有偏置的标准线性层作为框架
        self.linear = nn.Linear(f_in, d_model, bias=False)
        nn.init.orthogonal_(self.linear.weight)
        
        if self.trainable:
            # 核心黑科技：对 linear施加正交参数化约束，开启参数化梯度降维流形！基于凯莱映射/矩阵指数流形的绝对正交参数化）
            # 无论后续的反向传播梯度多猛，它的权重永远被在数学上死死地夹在 Stiefel 正交流形空间里。
            P.orthogonal(self.linear, "weight")
        else:
            # 剥夺梯度的求导权，纯纯的静态哈希打散器
            self.linear.weight.requires_grad = False
            
    def forward(self, x):
        return self.linear(x)


class Encoder(nn.Module):
    def __init__(
        self,
        f_in,
        d_model,
        nhead,
        num_layers,
        trade_idx,
        trainable_proj=False,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super().__init__()
        self.f_in = f_in
        self.d_model = d_model
        self.TRADE_IDX = trade_idx

        # 正交投影层（f_in→d_model），具备极强的自进化正交降维能力
        self.fixed_proj = OrthoProjection(f_in, d_model, trainable=trainable_proj)

        # 反投影层（d_model→f_in）
        self.proj_back = nn.Linear(d_model, f_in)
        # fixed_proj 的 weight 是 [64, 16]。我们要喂给 proj_back 的 weight 形状必须是 [16, 64]

        W_for_back = self.fixed_proj.linear.weight.detach()
        self.proj_back.weight.data = W_for_back.t().clone()
        if self.proj_back.bias is not None:
            self.proj_back.bias.data = torch.zeros(f_in, device=self.proj_back.weight.device)

        self.pos_enc = PositionalEncoding(d_model)

        # Transformer Encoder Layer，前馈维度512，激活函数GELU
        # 新增 norm_first=True (Pre-LN)，现代大模型标配，让梯度更加稳定
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        # 针对去除全局标准化后的补偿：在进入大模型前进行统一的特征向量规范化
        self.input_norm = nn.LayerNorm(d_model)
        
        # 在整个 Encoder 尾部再加一个 LayerNorm 兜底
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers, 
            norm=nn.LayerNorm(d_model)
        )

    def forward(self, x, mask_trade_ratio=0.0):
        B, T, _ = x.shape
        mask = None

        # Step 1: 随机mask交易特征
        x_masked = x.clone()
        if mask_trade_ratio > 0.0:
            mask = torch.rand(B, T, len(self.TRADE_IDX), device=x.device) < mask_trade_ratio
            x_masked[:, :, self.TRADE_IDX] = x_masked[:, :, self.TRADE_IDX] * (~mask)

        # Step 2: 投影 + 归一化 + 位置编码
        x_proj = self.fixed_proj(x_masked)
        x_proj = self.input_norm(x_proj)       # <--- 新增：在扩维后立刻进行 LayerNorm 压平数值方差
        x_proj = self.pos_enc(x_proj)

        # Step 3: Transformer编码
        enc_out = self.transformer(x_proj)

        # Step 4: 反投影回原始15维
        enc_out_recon = self.proj_back(enc_out)

        return enc_out, enc_out_recon, mask
